"""Production asynchronous implementation of the public FrameIngress contract."""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from observation_transport.frame_ingress import (
    DirectFrameStreamConfig,
    DirectFrameStreamDescriptor,
    DirectFrameStreamStatus,
    FrameAdmissionDisposition,
    FrameAdmissionReceipt,
    FrameIngressError,
    FrameStreamSnapshot,
    FrameTransportSnapshot,
    PreparedDirectFrame,
    StreamSessionView,
)
from observation_transport.rtp_sender import H264RtpSender, StreamLifecycleState
from observation_transport.video_codec import (
    CodecLifecycleState,
    ResolvedCodecBackend,
    VideoCodecRegistry,
    VideoEncoder,
    VideoFrame,
    create_default_video_codec_registry,
)

_RTP_CLOCK_RATE = 90_000
_RTP_PAYLOAD_TYPE = 96


@dataclass(frozen=True, slots=True)
class _QueuedFrame:
    admission_id: str
    session_generation: int
    frame: VideoFrame
    admission_monotonic_ns: int


@dataclass(slots=True)
class _Stream:
    config: DirectFrameStreamConfig
    encoder: VideoEncoder
    sender: Any
    sender_factory: Callable[..., Any]
    selected_backend: str
    ssrc: int
    condition: threading.Condition
    codec_lock: threading.RLock
    queue: deque[_QueuedFrame]
    worker: threading.Thread | None = None
    stopping: bool = False
    failed: bool = False
    active_session_generation: int = 0
    next_admission_sequence: int = 0
    last_input_capture_timestamp_ns: int = 0
    last_capture_timestamp_ns: int = 0
    last_rtp_timestamp: int = 0
    keyframe_sent: bool = False
    accepted_frames: int = 0
    encoded_frames: int = 0
    sent_frames: int = 0
    dropped_frames: int = 0
    queue_overflow_drops: int = 0
    last_admission_id: str = ""
    last_encoded_admission_id: str = ""
    last_sent_admission_id: str = ""
    last_dropped_admission_id: str = ""
    last_dropped_capture_timestamp_ns: int = 0
    dropped_capture_history: deque[tuple[int, str, str]] | None = None
    admission_by_capture: dict[int, str] | None = None
    admission_monotonic_ns: int = 0
    encode_start_monotonic_ns: int = 0
    encode_end_monotonic_ns: int = 0
    send_start_monotonic_ns: int = 0
    send_end_monotonic_ns: int = 0
    receiver_status: Mapping[str, Any] | None = None
    latest_frame: VideoFrame | None = None
    recovery_requested: bool = False
    recovery_attempts: int = 0
    last_drop_reason: str = ""
    last_error: str = ""
    worker_last_progress_monotonic_ns: int = 0
    worker_last_progress_kind: str = ""


class NativeFrameIngress:
    """Production asynchronous ingress shared by device and Benchmark paths."""

    def __init__(
        self,
        *,
        pipeline_id: str,
        contract_fingerprint: str,
        deployment_fingerprint: str,
        streams: Iterable[DirectFrameStreamConfig],
        codec_registry: VideoCodecRegistry | None = None,
        sender_factory: Callable[..., Any] = H264RtpSender,
        on_control_update: Callable[[], None] | None = None,
        protocol_version: int = 5,
    ) -> None:
        if not pipeline_id or not contract_fingerprint or not deployment_fingerprint:
            raise ValueError("frame ingress requires pipeline and fingerprint identity")
        if protocol_version < 1:
            raise ValueError("frame ingress protocol version must be positive")
        configs = tuple(streams)
        if not configs:
            raise ValueError("frame ingress requires at least one stream")
        if len({item.observation_key for item in configs}) != len(configs):
            raise ValueError("frame ingress observation keys must be unique")
        if len({item.stream_id for item in configs}) != len(configs):
            raise ValueError("frame ingress stream IDs must be unique")
        self.pipeline_id = pipeline_id
        self.contract_fingerprint = contract_fingerprint
        self.deployment_fingerprint = deployment_fingerprint
        self.protocol_version = int(protocol_version)
        self._session = StreamSessionView.inactive(
            pipeline_id=pipeline_id,
            contract_fingerprint=contract_fingerprint,
            deployment_fingerprint=deployment_fingerprint,
        )
        self._lock = threading.RLock()
        self._closed = False
        registry = codec_registry or create_default_video_codec_registry()
        self._codec_registry = registry
        self._sender_factory = sender_factory
        self._on_control_update = on_control_update
        self._streams: dict[str, _Stream] = {}
        resolved_backends = {
            config.observation_key: registry.resolve(config.encoder_backend, "encoder") for config in configs
        }
        encoder_channel_ids = {
            observation_key: channel_id
            for channel_id, observation_key in enumerate(
                sorted(key for key, resolved in resolved_backends.items() if resolved.name == "ascend"),
                start=1,
            )
        }
        if encoder_channel_ids and max(encoder_channel_ids.values()) > 127:
            raise ValueError("Ascend DVPP requires at most 128 VENC channels per device")
        try:
            for config in configs:
                stream = self._create_stream(
                    config,
                    resolved_backends[config.observation_key],
                    sender_factory,
                    encoder_channel_ids.get(config.observation_key, 0),
                )
                self._streams[config.observation_key] = stream
                stream.worker = threading.Thread(
                    target=self._worker_loop,
                    args=(stream,),
                    name=f"frame-ingress-{config.stream_id}",
                    daemon=True,
                )
                stream.worker.start()
        except Exception:
            self.close()
            raise

    @property
    def observation_keys(self) -> frozenset[str]:
        return frozenset(self._streams)

    @property
    def stream_references(self) -> tuple[tuple[str, str], ...]:
        return tuple((key, stream.config.stream_id) for key, stream in sorted(self._streams.items()))

    @property
    def session(self) -> StreamSessionView:
        with self._lock:
            return self._session

    @property
    def codec_registry(self) -> VideoCodecRegistry:
        return self._codec_registry

    @property
    def sender_factory(self) -> Callable[..., Any]:
        return self._sender_factory

    def ready(self) -> bool:
        snapshot = self.snapshot()
        return snapshot.ready

    def bind_session(self, session: StreamSessionView) -> bool:
        self._validate_session_identity(session)
        with self._lock:
            self._require_open()
            if session == self._session:
                return False
            self._session = StreamSessionView.inactive(
                pipeline_id=self.pipeline_id,
                contract_fingerprint=self.contract_fingerprint,
                deployment_fingerprint=self.deployment_fingerprint,
            )
        prepared: list[_Stream] = []
        try:
            for stream in self._streams.values():
                with stream.codec_lock, stream.condition:
                    self._drop_queued(stream, "session_rollover")
                    stream.active_session_generation = 0
                    stream.ssrc = secrets.randbits(32)
                    self._rotate_sender(stream)
                    if stream.encoder.state is CodecLifecycleState.RUNNING:
                        stream.encoder.discard_pending_output()
                    else:
                        stream.encoder.reset()
                    self._reset_stream_epoch(stream)
                    prepared.append(stream)
                    stream.condition.notify_all()
        except Exception as exc:
            for stream in prepared:
                with stream.condition:
                    self._drop_queued(stream, "session_bind_rollback")
                    stream.active_session_generation = 0
                    stream.failed = True
                    stream.last_error = f"session bind rolled back: {exc}"
                    stream.condition.notify_all()
            raise
        for stream in prepared:
            with stream.condition:
                stream.active_session_generation = session.generation
        with self._lock:
            self._session = session
        return True

    def clear_session(self) -> None:
        with self._lock:
            self._require_open()
            self._session = StreamSessionView.inactive(
                pipeline_id=self.pipeline_id,
                contract_fingerprint=self.contract_fingerprint,
                deployment_fingerprint=self.deployment_fingerprint,
            )
        for stream in self._streams.values():
            with stream.condition:
                self._drop_queued(stream, "session_cleared")
                stream.active_session_generation = 0
                stream.condition.notify_all()

    def policy_reset(self, session: StreamSessionView | None = None) -> None:
        with self._lock:
            self._require_open()
            if session is not None and session != self._session:
                raise FrameIngressError(
                    "stale_session", "policy reset does not match active stream session", recoverable=True
                )

    def prepare_frame(
        self,
        observation_key: str,
        frame: np.ndarray,
        *,
        capture_timestamp_ns: int,
        receive_timestamp_ns: int,
        provider_capture_timestamp_ns: int | None = None,
        pixel_format: str = "rgb24",
    ) -> PreparedDirectFrame:
        start = time.monotonic_ns()
        stream, session, array = self._validate_frame(
            observation_key,
            frame,
            capture_timestamp_ns=capture_timestamp_ns,
            pixel_format=pixel_format,
        )
        del stream
        end = time.monotonic_ns()
        return PreparedDirectFrame(
            observation_key=observation_key,
            frame=array,
            capture_timestamp_ns=int(capture_timestamp_ns),
            receive_timestamp_ns=int(receive_timestamp_ns),
            provider_capture_timestamp_ns=int(
                capture_timestamp_ns if provider_capture_timestamp_ns is None else provider_capture_timestamp_ns
            ),
            pixel_format=pixel_format,
            session_generation=session.generation,
            prepare_start_monotonic_ns=start,
            prepare_end_monotonic_ns=end,
        )

    def commit_frame(
        self,
        prepared: PreparedDirectFrame,
        *,
        before_send: Callable[[], None] | None = None,
    ) -> FrameAdmissionReceipt:
        del before_send
        with self._lock:
            self._require_open()
            session = self._session
        if not session.active or prepared.session_generation != session.generation:
            raise FrameIngressError(
                "stale_session",
                "prepared frame does not match the active stream session",
                observation_key=prepared.observation_key,
                recoverable=True,
            )
        return self.submit_frame(
            prepared.observation_key,
            prepared.frame,
            capture_timestamp_ns=prepared.capture_timestamp_ns,
            receive_timestamp_ns=prepared.receive_timestamp_ns,
            pixel_format=prepared.pixel_format,
        )

    def submit_frame(
        self,
        observation_key: str,
        frame: np.ndarray,
        *,
        capture_timestamp_ns: int,
        receive_timestamp_ns: int,
        pixel_format: str = "rgb24",
        provider_capture_timestamp_ns: int | None = None,
    ) -> FrameAdmissionReceipt:
        del provider_capture_timestamp_ns
        stream, session, array = self._validate_frame(
            observation_key,
            frame,
            capture_timestamp_ns=capture_timestamp_ns,
            pixel_format=pixel_format,
        )
        admission_ns = time.monotonic_ns()
        with stream.condition:
            if stream.failed:
                raise FrameIngressError(
                    "worker_failed",
                    stream.last_error or "frame ingress worker failed",
                    observation_key=observation_key,
                    stream_id=stream.config.stream_id,
                )
            if stream.active_session_generation != session.generation:
                raise FrameIngressError(
                    "stale_session",
                    "stream is not bound to the active inference generation",
                    observation_key=observation_key,
                    stream_id=stream.config.stream_id,
                    recoverable=True,
                )
            if capture_timestamp_ns <= stream.last_input_capture_timestamp_ns:
                stream.dropped_frames += 1
                self._record_drop_locked(stream, int(capture_timestamp_ns), "", "non_monotonic_timestamp")
                raise FrameIngressError(
                    "non_monotonic_timestamp",
                    "frame capture timestamps must increase",
                    observation_key=observation_key,
                    stream_id=stream.config.stream_id,
                )
            capacity = stream.config.effective_raw_queue_frames
            if len(stream.queue) >= capacity and stream.config.queue_policy == "strict":
                stream.queue_overflow_drops += 1
                stream.dropped_frames += 1
                self._record_drop_locked(stream, int(capture_timestamp_ns), "", "raw_queue_full")
                return FrameAdmissionReceipt(
                    FrameAdmissionDisposition.REJECTED,
                    "",
                    observation_key,
                    stream.config.stream_id,
                    int(capture_timestamp_ns),
                    session.generation,
                    admission_ns,
                    len(stream.queue),
                    "raw_queue_full",
                )
            if len(stream.queue) >= capacity:
                dropped = stream.queue.popleft()
                stream.queue_overflow_drops += 1
                stream.dropped_frames += 1
                self._record_drop_locked(
                    stream,
                    dropped.frame.capture_timestamp_ns,
                    dropped.admission_id,
                    "raw_queue_latest_replaced",
                )
            stream.next_admission_sequence += 1
            admission_id = f"{session.generation}:{stream.config.stream_id}:{stream.next_admission_sequence}"
            video_frame = VideoFrame(
                np.ascontiguousarray(array).copy(),
                int(capture_timestamp_ns),
                int(receive_timestamp_ns),
                stream.config.width,
                stream.config.height,
                pixel_format,
                color_space=stream.config.color_space,
                color_range=stream.config.color_range,
            )
            stream.queue.append(_QueuedFrame(admission_id, session.generation, video_frame, admission_ns))
            stream.latest_frame = video_frame
            stream.last_input_capture_timestamp_ns = int(capture_timestamp_ns)
            stream.accepted_frames += 1
            stream.last_admission_id = admission_id
            stream.admission_monotonic_ns = admission_ns
            queue_depth = len(stream.queue)
            stream.condition.notify()
        return FrameAdmissionReceipt(
            FrameAdmissionDisposition.ACCEPTED,
            admission_id,
            observation_key,
            stream.config.stream_id,
            int(capture_timestamp_ns),
            session.generation,
            admission_ns,
            queue_depth,
        )

    def snapshot(self) -> FrameTransportSnapshot:
        with self._lock:
            session = self._session
        streams = tuple(self._frame_snapshot(stream) for stream in self._ordered_streams())
        return FrameTransportSnapshot(
            self.pipeline_id,
            session.session_id,
            session.generation,
            session.active and all(item.ready for item in streams),
            streams,
        )

    def descriptors(self) -> tuple[DirectFrameStreamDescriptor, ...]:
        with self._lock:
            session = self._session
        if not session.active:
            return ()
        return tuple(self._descriptor(stream, session) for stream in self._ordered_streams())

    def statuses(self) -> tuple[DirectFrameStreamStatus, ...]:
        with self._lock:
            session = self._session
        if not session.active:
            return ()
        return tuple(self._status(stream, session) for stream in self._ordered_streams())

    def observe_receiver_status(self, observation_key: str, status: Mapping[str, Any]) -> None:
        stream = self._streams.get(observation_key)
        if stream is None:
            return
        with stream.condition:
            stream.receiver_status = dict(status)

    def request_keyframe_recovery(self, observation_key: str, *, session_generation: int) -> bool:
        """Schedule one non-blocking IDR recovery from the latest admitted frame.

        Recovery is transport-owned and never re-enters the observation provider.
        Repeated requests coalesce while one recovery is pending, so receiver
        feedback cannot create an unbounded work queue.
        """

        stream = self._streams.get(observation_key)
        if stream is None:
            raise FrameIngressError("unknown_stream", "unknown frame observation", observation_key=observation_key)
        with self._lock:
            self._require_open()
            session = self._session
        if not session.active or session.generation != session_generation:
            raise FrameIngressError(
                "stale_session",
                "keyframe recovery does not match the active inference generation",
                observation_key=observation_key,
                stream_id=stream.config.stream_id,
                recoverable=True,
            )
        with stream.condition:
            if stream.active_session_generation != session_generation:
                raise FrameIngressError(
                    "stale_session",
                    "stream is not bound to the requested recovery generation",
                    observation_key=observation_key,
                    stream_id=stream.config.stream_id,
                    recoverable=True,
                )
            if stream.failed:
                raise FrameIngressError(
                    "worker_failed",
                    stream.last_error or "frame ingress worker failed",
                    observation_key=observation_key,
                    stream_id=stream.config.stream_id,
                )
            if stream.latest_frame is None or stream.recovery_requested:
                return False
            stream.recovery_requested = True
            stream.condition.notify()
            return True

    def reset(self) -> None:
        for stream in self._streams.values():
            with stream.codec_lock, stream.condition:
                self._drop_queued(stream, "transport_reset")
                self._rotate_sender(stream)
                stream.encoder.reset()
                self._reset_stream_epoch(stream)
                stream.condition.notify_all()

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        with self._lock:
            if self._closed:
                return
            self._closed = True
        deadline = time.monotonic() + timeout_s
        for stream in self._streams.values():
            with stream.condition:
                stream.stopping = True
                self._drop_queued(stream, "shutdown")
                stream.condition.notify_all()
        error: Exception | None = None
        for stream in self._streams.values():
            worker = stream.worker
            if worker is not None:
                worker.join(max(0.0, deadline - time.monotonic()))
            stuck = worker is not None and worker.is_alive()
            if stuck:
                with contextlib.suppress(Exception):
                    stream.encoder.close(max(0.0, deadline - time.monotonic()))
                worker.join(max(0.0, deadline - time.monotonic()))
                stuck = worker.is_alive()
            if stuck:
                # A live worker may still touch the sender once its encoder
                # call returns. Preserve the master safety rule: do not close
                # resources still reachable by the wedged worker.
                continue
            try:
                stream.sender.close(max(0.0, deadline - time.monotonic()))
            except Exception as exc:
                error = error or exc
            try:
                stream.encoder.close(max(0.0, deadline - time.monotonic()))
            except Exception as exc:
                error = error or exc
        with self._lock:
            self._session = StreamSessionView.inactive(
                pipeline_id=self.pipeline_id,
                contract_fingerprint=self.contract_fingerprint,
                deployment_fingerprint=self.deployment_fingerprint,
            )
        if error is not None:
            raise FrameIngressError("shutdown_failed", str(error)) from error

    def _validate_frame(
        self,
        observation_key: str,
        frame: np.ndarray,
        *,
        capture_timestamp_ns: int,
        pixel_format: str,
    ) -> tuple[_Stream, StreamSessionView, np.ndarray]:
        stream = self._streams.get(observation_key)
        if stream is None:
            raise FrameIngressError("unknown_stream", "unknown frame observation", observation_key=observation_key)
        with self._lock:
            self._require_open()
            session = self._session
        if not session.active:
            raise FrameIngressError(
                "session_not_ready",
                "frame ingress has no live inference session",
                observation_key=observation_key,
                stream_id=stream.config.stream_id,
                recoverable=True,
            )
        array = np.asarray(frame)
        expected = (stream.config.height, stream.config.width, 3)
        if array.dtype != np.uint8 or array.shape != expected:
            raise FrameIngressError(
                "invalid_frame",
                f"expected uint8 HWC frame {expected}, got {array.shape} {array.dtype}",
                observation_key=observation_key,
                stream_id=stream.config.stream_id,
            )
        if pixel_format != stream.config.input_pixel_format:
            raise FrameIngressError(
                "invalid_pixel_format",
                f"expected {stream.config.input_pixel_format}, got {pixel_format}",
                observation_key=observation_key,
                stream_id=stream.config.stream_id,
            )
        if (
            not isinstance(capture_timestamp_ns, int)
            or isinstance(capture_timestamp_ns, bool)
            or capture_timestamp_ns <= 0
        ):
            raise FrameIngressError("invalid_timestamp", "capture timestamp must be a positive int")
        return stream, session, array

    def _worker_loop(self, stream: _Stream) -> None:
        while True:
            with stream.condition:
                while not stream.queue and not stream.recovery_requested and not stream.stopping:
                    stream.condition.wait()
                if stream.stopping:
                    return
                recovery = not stream.queue and stream.recovery_requested
                if recovery:
                    stream.recovery_requested = False
                    queued = None
                    frame = stream.latest_frame
                    session_generation = stream.active_session_generation
                else:
                    queued = stream.queue.popleft()
                    frame = queued.frame
                    session_generation = queued.session_generation
            try:
                with stream.codec_lock:
                    with self._lock:
                        session = self._session
                    if not session.active or session_generation != session.generation:
                        if queued is not None:
                            self._mark_dropped(
                                stream, queued.admission_id, "stale_session", queued.frame.capture_timestamp_ns
                            )
                        continue
                    if frame is None:
                        continue
                    if recovery:
                        # Recreate the encoder epoch so every recovery attempt is
                        # an SPS/PPS-bearing IDR, independent of GOP position.
                        stream.encoder.reset()
                    encode_start = time.monotonic_ns()
                    packets = stream.encoder.encode(frame)
                    encode_end = time.monotonic_ns()
                    if recovery and (not packets or not any(packet.keyframe for packet in packets)):
                        raise FrameIngressError(
                            "encode_delay",
                            "encoder produced no recovery-capable H.264 access unit",
                            observation_key=stream.config.observation_key,
                            stream_id=stream.config.stream_id,
                        )
                    with stream.condition:
                        stream.encoded_frames += 1
                        if queued is not None:
                            stream.last_encoded_admission_id = queued.admission_id
                        else:
                            stream.recovery_attempts += 1
                        stream.encode_start_monotonic_ns = encode_start
                        stream.encode_end_monotonic_ns = encode_end
                        stream.worker_last_progress_monotonic_ns = encode_end
                        stream.worker_last_progress_kind = "recovery_encode" if recovery else "encode"
                        if queued is not None:
                            assert stream.admission_by_capture is not None
                            stream.admission_by_capture[frame.capture_timestamp_ns] = queued.admission_id
                        publish_control_after_send = recovery or not stream.keyframe_sent
                    send_start = time.monotonic_ns()
                    for packet in packets:
                        stream.sender.enqueue(packet)
                        send_pending = getattr(stream.sender, "send_pending", None)
                        if callable(send_pending):
                            send_pending()
                    send_end = time.monotonic_ns()
                    with stream.condition:
                        stream.send_start_monotonic_ns = send_start
                        stream.send_end_monotonic_ns = send_end
                        stream.worker_last_progress_monotonic_ns = send_end
                        stream.worker_last_progress_kind = "recovery_send" if recovery else "send"
                        sender_metrics = stream.sender.status.metrics
                        if sender_metrics.sender_queue_overflow_drops:
                            stream.last_drop_reason = "sender_queue_overflow"
                        stream.last_error = ""
                    if publish_control_after_send and self._on_control_update is not None:
                        # Publish the first/recovery RTP-to-capture mapping only
                        # after the access unit was delivered. This keeps sender
                        # freshness and the advertised mapping on the same
                        # post-send event. Periodic control publication handles
                        # all later frames without adding work to the media loop.
                        self._on_control_update()
            except Exception as exc:
                recovery_error: Exception | None = None
                if stream.encoder.state is CodecLifecycleState.FAILED:
                    try:
                        stream.encoder.reset()
                    except Exception as reset_exc:
                        recovery_error = reset_exc
                sender_failed = stream.sender.status.state in {
                    StreamLifecycleState.FAILED,
                    StreamLifecycleState.STOPPED,
                }
                fatal = recovery_error is not None or sender_failed
                with stream.condition:
                    stream.keyframe_sent = False
                    stream.last_error = f"{type(exc).__name__}: {exc}"
                    stream.worker_last_progress_monotonic_ns = time.monotonic_ns()
                    stream.worker_last_progress_kind = "encode_failed"
                    if queued is not None:
                        self._mark_dropped_locked(
                            stream, queued.admission_id, "worker_failed", queued.frame.capture_timestamp_ns
                        )
                    if fatal:
                        stream.failed = True
                        if recovery_error is not None:
                            stream.last_error = f"encoder reset failed: {recovery_error}"
                        stream.worker_last_progress_kind = "failed"
                        self._drop_queued(stream, "worker_failed")
                    stream.condition.notify_all()
                if fatal:
                    return

    def _create_stream(
        self,
        config: DirectFrameStreamConfig,
        resolved: ResolvedCodecBackend,
        sender_factory: Callable[..., Any],
        channel_id: int,
    ) -> _Stream:
        encoder_options: dict[str, object] = {
            "width": config.width,
            "height": config.height,
            "frame_rate_hz": config.frame_rate_hz,
            "bitrate_bps": config.bitrate_bps,
            "gop_frames": config.gop_frames,
            "input_pixel_format": config.input_pixel_format,
            "profile": config.codec_profile,
            "color_space": config.color_space,
            "color_range": config.color_range,
        }
        if resolved.name == "ascend":
            encoder_options["channel_id"] = channel_id
        encoder = resolved.create(**encoder_options)
        stream = _Stream(
            config=config,
            encoder=encoder,
            sender=None,
            sender_factory=sender_factory,
            selected_backend=resolved.name,
            ssrc=secrets.randbits(32),
            condition=threading.Condition(threading.RLock()),
            codec_lock=threading.RLock(),
            queue=deque(),
            dropped_capture_history=deque(maxlen=64),
            admission_by_capture={},
        )
        try:
            stream.sender = self._create_sender(stream)
        except Exception:
            encoder.close()
            raise
        return stream

    def _rotate_sender(self, stream: _Stream) -> None:
        sender = stream.sender
        sender.ssrc = stream.ssrc
        if sender.status.state not in {StreamLifecycleState.FAILED, StreamLifecycleState.STOPPED}:
            sender.reset()
            return
        with contextlib.suppress(Exception):
            sender.close()
        stream.sender = self._create_sender(stream)

    def _create_sender(self, stream: _Stream) -> Any:
        holder: dict[str, Any] = {}

        def mark_sent(packet: Any) -> None:
            if stream.sender is not holder.get("sender"):
                return
            with stream.condition:
                stream.last_capture_timestamp_ns = int(packet.capture_timestamp_ns)
                stream.last_rtp_timestamp = int(packet.rtp_timestamp)
                stream.keyframe_sent = stream.keyframe_sent or bool(packet.keyframe)
                assert stream.admission_by_capture is not None
                admission_id = stream.admission_by_capture.pop(int(packet.capture_timestamp_ns), "")
                if admission_id:
                    stream.sent_frames += 1
                    stream.last_sent_admission_id = admission_id

        sender = stream.sender_factory(
            stream_id=stream.config.stream_id,
            endpoint=(stream.config.endpoint_host, stream.config.endpoint_port),
            ssrc=stream.ssrc,
            queue_capacity=stream.config.sender_queue_frames,
            payload_type=_RTP_PAYLOAD_TYPE,
            selected_backend=stream.selected_backend,
            on_sent=mark_sent,
            background_delivery=False,
        )
        sender.start()
        holder["sender"] = sender
        return sender

    def _frame_snapshot(self, stream: _Stream) -> FrameStreamSnapshot:
        with stream.condition:
            sender_status = stream.sender.status
            sender_metrics = sender_status.metrics
            receiver = stream.receiver_status or {}
            return FrameStreamSnapshot(
                stream.config.observation_key,
                stream.config.stream_id,
                not stream.failed and sender_status.ready and stream.keyframe_sent,
                accepted_frames=stream.accepted_frames,
                encoded_frames=stream.encoded_frames,
                sent_frames=stream.sent_frames,
                sent_packets=int(sender_metrics.sent_packets),
                received_packets=int(receiver.get("received_packets", 0)),
                decoded_frames=int(receiver.get("decoded_frames", 0)),
                dropped_frames=stream.dropped_frames + int(sender_metrics.dropped_frames),
                queue_depth=len(stream.queue),
                queue_overflow_drops=stream.queue_overflow_drops + int(sender_metrics.sender_queue_overflow_drops),
                last_accepted_admission_id=stream.last_admission_id,
                last_encoded_admission_id=stream.last_encoded_admission_id,
                last_sent_admission_id=stream.last_sent_admission_id,
                last_dropped_admission_id=stream.last_dropped_admission_id,
                last_dropped_capture_timestamp_ns=stream.last_dropped_capture_timestamp_ns,
                dropped_capture_history=tuple(stream.dropped_capture_history or ()),
                last_drop_reason=stream.last_drop_reason,
                last_error=stream.last_error or sender_status.last_error,
                worker_alive=bool(stream.worker is not None and stream.worker.is_alive()),
                worker_failed=stream.failed,
                worker_last_progress_monotonic_ns=stream.worker_last_progress_monotonic_ns,
                worker_last_progress_kind=stream.worker_last_progress_kind,
                recovery_requested=stream.recovery_requested,
                recovery_attempts=stream.recovery_attempts,
                encode_latency_ns=max(0, stream.encode_end_monotonic_ns - stream.encode_start_monotonic_ns),
                send_latency_ns=max(0, stream.send_end_monotonic_ns - stream.send_start_monotonic_ns),
            )

    def _descriptor(self, stream: _Stream, session: StreamSessionView) -> DirectFrameStreamDescriptor:
        with stream.condition:
            config = stream.config
            return DirectFrameStreamDescriptor(
                self.protocol_version,
                self.pipeline_id,
                session.session_id,
                session.generation,
                config.observation_key,
                config.stream_id,
                config.endpoint_host,
                config.endpoint_port,
                stream.ssrc,
                _RTP_PAYLOAD_TYPE,
                config.codec,
                config.codec_profile,
                config.width,
                config.height,
                config.frame_rate_hz,
                _RTP_CLOCK_RATE,
                config.pixel_format,
                config.color_space,
                config.color_range,
                stream.selected_backend,
                session.contract_fingerprint,
                session.deployment_fingerprint,
            )

    def _status(self, stream: _Stream, session: StreamSessionView) -> DirectFrameStreamStatus:
        with stream.condition:
            sender_status = stream.sender.status
            sender_metrics = sender_status.metrics
            receiver = stream.receiver_status or {}
            return DirectFrameStreamStatus(
                self.protocol_version,
                self.pipeline_id,
                session.session_id,
                session.generation,
                stream.config.observation_key,
                stream.config.stream_id,
                "failed" if stream.failed else sender_status.state.value,
                not stream.failed and sender_status.ready and stream.keyframe_sent,
                stream.selected_backend,
                "sender",
                timestamp_mapping_valid=stream.last_capture_timestamp_ns > 0,
                mapping_rtp_timestamp=stream.last_rtp_timestamp,
                mapping_capture_timestamp_ns=stream.last_capture_timestamp_ns,
                keyframe_ready=stream.keyframe_sent,
                encoded_frames=stream.encoded_frames,
                decoded_frames=int(receiver.get("decoded_frames", 0)),
                sent_packets=int(sender_metrics.sent_packets),
                received_packets=int(receiver.get("received_packets", 0)),
                dropped_frames=stream.dropped_frames + int(sender_metrics.dropped_frames),
                dropped_packets=int(receiver.get("dropped_packets", 0)),
                lost_packets=int(receiver.get("lost_packets", 0)),
                sender_queue_depth=len(stream.queue),
                receiver_queue_depth=int(receiver.get("receiver_queue_depth", 0)),
                decoded_buffer_depth=int(receiver.get("decoded_buffer_depth", 0)),
                reconnect_count=int(sender_metrics.reconnect_count),
                sender_queue_overflow_drops=stream.queue_overflow_drops
                + int(sender_metrics.sender_queue_overflow_drops),
                receiver_queue_overflow_drops=int(receiver.get("receiver_queue_overflow_drops", 0)),
                sequence_gap_events=int(receiver.get("sequence_gap_events", 0)),
                reordered_packets=int(receiver.get("reordered_packets", 0)),
                recovery_keyframes=int(receiver.get("recovery_keyframes", 0)),
                jitter_ns=int(receiver.get("jitter_ns", 0)),
                encode_start_monotonic_ns=stream.encode_start_monotonic_ns,
                encode_end_monotonic_ns=stream.encode_end_monotonic_ns,
                send_start_monotonic_ns=stream.send_start_monotonic_ns,
                send_end_monotonic_ns=stream.send_end_monotonic_ns,
                receive_monotonic_ns=int(receiver.get("receive_monotonic_ns", 0)),
                decode_start_monotonic_ns=int(receiver.get("decode_start_monotonic_ns", 0)),
                decode_end_monotonic_ns=int(receiver.get("decode_end_monotonic_ns", 0)),
                last_decoded_capture_timestamp_ns=int(receiver.get("last_decoded_capture_timestamp_ns", 0)),
                last_accepted_admission_id=stream.last_admission_id,
                last_encoded_admission_id=stream.last_encoded_admission_id,
                last_sent_admission_id=stream.last_sent_admission_id,
                last_dropped_admission_id=stream.last_dropped_admission_id,
                last_dropped_capture_timestamp_ns=stream.last_dropped_capture_timestamp_ns,
                dropped_capture_history_json=json.dumps(
                    list(stream.dropped_capture_history or ()), separators=(",", ":")
                ),
                last_drop_reason=stream.last_drop_reason or str(receiver.get("last_drop_reason", "")),
                last_error=stream.last_error or sender_status.last_error,
            )

    def _validate_session_identity(self, session: StreamSessionView) -> None:
        if not isinstance(session, StreamSessionView) or not session.active:
            raise ValueError("frame ingress requires an active StreamSessionView")
        expected = (self.pipeline_id, self.contract_fingerprint, self.deployment_fingerprint)
        actual = (session.pipeline_id, session.contract_fingerprint, session.deployment_fingerprint)
        if actual != expected:
            raise ValueError("stream session pipeline or fingerprint identity mismatch")

    @staticmethod
    def _mark_dropped(stream: _Stream, admission_id: str, reason: str, capture_timestamp_ns: int = 0) -> None:
        with stream.condition:
            NativeFrameIngress._mark_dropped_locked(stream, admission_id, reason, capture_timestamp_ns)

    @staticmethod
    def _mark_dropped_locked(stream: _Stream, admission_id: str, reason: str, capture_timestamp_ns: int = 0) -> None:
        stream.dropped_frames += 1
        NativeFrameIngress._record_drop_locked(stream, int(capture_timestamp_ns), admission_id, reason)

    @staticmethod
    def _record_drop_locked(stream: _Stream, capture_timestamp_ns: int, admission_id: str, reason: str) -> None:
        stream.last_dropped_admission_id = admission_id
        stream.last_dropped_capture_timestamp_ns = int(capture_timestamp_ns)
        stream.last_drop_reason = reason
        if capture_timestamp_ns <= 0:
            return
        if stream.dropped_capture_history is None:
            stream.dropped_capture_history = deque(maxlen=64)
        stream.dropped_capture_history.append((int(capture_timestamp_ns), admission_id, reason))

    @staticmethod
    def _drop_queued(stream: _Stream, reason: str) -> None:
        while stream.queue:
            queued = stream.queue.popleft()
            NativeFrameIngress._mark_dropped_locked(
                stream, queued.admission_id, reason, queued.frame.capture_timestamp_ns
            )

    @staticmethod
    def _reset_stream_epoch(stream: _Stream) -> None:
        stream.next_admission_sequence = 0
        stream.last_input_capture_timestamp_ns = 0
        stream.last_capture_timestamp_ns = 0
        stream.last_rtp_timestamp = 0
        stream.keyframe_sent = False
        stream.failed = False
        stream.last_error = ""
        stream.admission_monotonic_ns = 0
        stream.encode_start_monotonic_ns = 0
        stream.encode_end_monotonic_ns = 0
        stream.send_start_monotonic_ns = 0
        stream.send_end_monotonic_ns = 0
        stream.last_admission_id = ""
        stream.last_encoded_admission_id = ""
        stream.last_sent_admission_id = ""
        stream.last_dropped_admission_id = ""
        stream.last_dropped_capture_timestamp_ns = 0
        if stream.dropped_capture_history is not None:
            stream.dropped_capture_history.clear()
        stream.receiver_status = None
        stream.latest_frame = None
        stream.recovery_requested = False
        stream.recovery_attempts = 0
        stream.last_drop_reason = ""
        assert stream.admission_by_capture is not None
        stream.admission_by_capture.clear()

    def _ordered_streams(self) -> tuple[_Stream, ...]:
        return tuple(stream for _, stream in sorted(self._streams.items()))

    def _require_open(self) -> None:
        if self._closed:
            raise FrameIngressError("closed", "frame ingress is closed")


__all__ = ["NativeFrameIngress"]

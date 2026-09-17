"""Compute-side ownership and assembly of streamed video observations."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from inference_service.distributed.video_streams import (
    StreamNegotiationRequirements,
    VideoStreamDescriptor,
    VideoStreamDiagnosticSnapshot,
    VideoStreamNegotiator,
    VideoStreamRequirement,
    VideoStreamRuntimeStatus,
    VideoTransportCapabilities,
)
from inference_service.h264_stream_recorder import H264StreamRecorder
from inference_service.observation_sync import (
    RtpTimestampMapper,
    StreamSelection,
    select_synchronized_streams,
)
from inference_service.video_codec import VideoCodecRegistry, VideoFrame, create_default_video_codec_registry
from inference_service.video_recording_coordinator import VideoRecordingCoordinator
from inference_service.video_rtp import H264RtpReceiver
from robot_config.contract_utils import SpecView, StreamBuffer
from robot_config.observation_transport import ObservationTransportSpec, effective_observation_transport
from tensormsg.converter import decoded_frame_to_chw_float


@dataclass(slots=True)
class _ComputeStream:
    spec: SpecView
    transport: ObservationTransportSpec
    descriptor: VideoStreamDescriptor
    mapper: RtpTimestampMapper
    buffer: StreamBuffer
    receiver: H264RtpReceiver
    sender_status: VideoStreamRuntimeStatus | None = None


class _RecordingOnlyDecoder:
    """Minimal decoder lifecycle used when the receiver only records access units."""

    def reset(self) -> None:
        pass

    def close(self, timeout_s: float = 1.0) -> None:
        pass


class ComputeVideoStreamManager:
    """Negotiate, receive, synchronize, and reconstruct compute video inputs."""

    def __init__(
        self,
        *,
        pipeline_id: str,
        session_id: str,
        session_generation: int,
        contract_fingerprint: str,
        deployment_fingerprint: str,
        observation_specs: Iterable[SpecView],
        rate_hz: float,
        n_obs_steps: int = 1,
        codec_registry: VideoCodecRegistry | None = None,
        receiver_factory=H264RtpReceiver,
        recording_coordinator: VideoRecordingCoordinator | None = None,
        decode: bool = True,
        validate_deployment_fingerprint: bool = True,
    ) -> None:
        if rate_hz <= 0 or n_obs_steps < 1:
            raise ValueError("compute video streams require a positive rate and observation history length")
        self.pipeline_id = pipeline_id
        self.contract_fingerprint = contract_fingerprint
        self.deployment_fingerprint = deployment_fingerprint
        self.rate_hz = float(rate_hz)
        self.n_obs_steps = int(n_obs_steps)
        self._receiver_factory = receiver_factory
        self._decode = decode
        self._recorders: dict[str, H264StreamRecorder] = {}
        self._lock = threading.RLock()
        self._selection_diagnostics: tuple[tuple[str, int, float, float], ...] = ()
        self._last_state_alignment_delta_ns: int | None = None
        self._receiver_start_lock = threading.Lock()
        self._specs: dict[str, tuple[SpecView, ObservationTransportSpec]] = {}
        for spec in observation_specs:
            transport = effective_observation_transport(spec.transport)
            if transport.mode == "rtp":
                self._specs[spec.key] = (spec, transport)
                if recording_coordinator is not None:
                    integrity_mode = transport.recording.integrity_mode if transport.recording is not None else "strict"
                    recorder = H264StreamRecorder(integrity_mode=integrity_mode)
                    recording_coordinator.register_recorder(spec.key, recorder)
                    self._recorders[spec.key] = recorder
        registry = codec_registry or create_default_video_codec_registry()
        self._registry = registry
        self._resolved_decoders = {
            observation_key: registry.resolve(transport.decoder_backend, "decoder")
            for observation_key, (_spec, transport) in self._specs.items()
        }
        self._decoder_channel_ids = {
            observation_key: channel_id for channel_id, observation_key in enumerate(sorted(self._specs), start=1)
        }
        requirements = StreamNegotiationRequirements(
            pipeline_id=pipeline_id,
            session_id=session_id,
            session_generation=session_generation,
            contract_fingerprint=contract_fingerprint,
            deployment_fingerprint=deployment_fingerprint,
            streams=tuple(self._requirement(spec, transport) for spec, transport in self._specs.values()),
            transport_mode="rtp" if self._specs else "dds",
        )
        selected_decoder_backends = tuple(sorted({resolved.name for resolved in self._resolved_decoders.values()}))
        self.negotiator = VideoStreamNegotiator(
            requirements,
            VideoTransportCapabilities(decoder_backends=selected_decoder_backends or ("software",)),
            validate_deployment_fingerprint=validate_deployment_fingerprint,
        )
        self._descriptors: dict[str, VideoStreamDescriptor] = {}
        self._streams: dict[str, _ComputeStream] = {}

    def diagnostic_snapshots(self) -> tuple[VideoStreamDiagnosticSnapshot, ...]:
        snapshots = []
        with self._lock:
            streams = dict(self._streams)
            descriptors = dict(self._descriptors)
            for observation_key, (_spec, transport) in sorted(self._specs.items()):
                assert transport.stream_id is not None
                assert transport.endpoint is not None
                stream = streams.get(observation_key)
                descriptor = descriptors.get(observation_key)
                selected_encoder = descriptor.encoder_backend if descriptor is not None else "pending"
                if stream is None:
                    lifecycle_state = "configured"
                    ready = False
                else:
                    receiver_status = stream.receiver.status
                    lifecycle_state = receiver_status.state.value
                    ready = receiver_status.ready and stream.mapper.ready
                snapshots.append(
                    VideoStreamDiagnosticSnapshot(
                        observation_key=observation_key,
                        stream_id=transport.stream_id,
                        mode=transport.mode,
                        configured_encoder_backend=transport.encoder_backend,
                        selected_encoder_backend=selected_encoder,
                        configured_decoder_backend=transport.decoder_backend,
                        selected_decoder_backend=self._resolved_decoders[observation_key].name,
                        endpoint=(transport.endpoint.host, transport.endpoint.port),
                        contract_fingerprint=self.contract_fingerprint,
                        deployment_fingerprint=self.deployment_fingerprint,
                        security="none/trusted-network-only",
                        lifecycle_state=lifecycle_state,
                        ready=ready,
                    )
                )
        return tuple(snapshots)

    def statuses(self) -> tuple[VideoStreamRuntimeStatus, ...]:
        with self._lock:
            streams = tuple(sorted(self._streams.values(), key=lambda item: item.spec.key))
        statuses = []
        for stream in streams:
            receiver_status = stream.receiver.status
            metrics = receiver_status.metrics
            sender_status = stream.sender_status
            statuses.append(
                VideoStreamRuntimeStatus(
                    protocol_version=stream.descriptor.protocol_version,
                    pipeline_id=stream.descriptor.pipeline_id,
                    session_id=stream.descriptor.session_id,
                    session_generation=stream.descriptor.session_generation,
                    observation_key=stream.spec.key,
                    stream_id=stream.descriptor.stream_id,
                    lifecycle_state=receiver_status.state.value,
                    ready=receiver_status.ready and stream.mapper.ready,
                    selected_backend=receiver_status.selected_backend,
                    status_origin="receiver",
                    timestamp_mapping_valid=bool(sender_status and sender_status.timestamp_mapping_valid),
                    mapping_rtp_timestamp=sender_status.mapping_rtp_timestamp if sender_status else 0,
                    mapping_capture_timestamp_ns=(sender_status.mapping_capture_timestamp_ns if sender_status else 0),
                    keyframe_ready=receiver_status.ready,
                    encoded_frames=sender_status.encoded_frames if sender_status else 0,
                    decoded_frames=metrics.decoded_frames,
                    sent_packets=sender_status.sent_packets if sender_status else 0,
                    received_packets=metrics.received_packets,
                    dropped_frames=sender_status.dropped_frames if sender_status else 0,
                    dropped_packets=metrics.dropped_packets,
                    lost_packets=metrics.lost_packets,
                    sender_queue_depth=sender_status.sender_queue_depth if sender_status else 0,
                    receiver_queue_depth=metrics.queued_packets,
                    decoded_buffer_depth=len(stream.buffer),
                    reconnect_count=metrics.reconnect_count,
                    sender_queue_overflow_drops=(sender_status.sender_queue_overflow_drops if sender_status else 0),
                    receiver_queue_overflow_drops=metrics.receiver_queue_overflow_drops,
                    sequence_gap_events=metrics.sequence_gap_events,
                    reordered_packets=metrics.reordered_packets,
                    recovery_keyframes=metrics.recovery_keyframes,
                    jitter_ns=metrics.jitter_ns,
                    encode_start_monotonic_ns=sender_status.encode_start_monotonic_ns if sender_status else 0,
                    encode_end_monotonic_ns=sender_status.encode_end_monotonic_ns if sender_status else 0,
                    send_start_monotonic_ns=sender_status.send_start_monotonic_ns if sender_status else 0,
                    send_end_monotonic_ns=sender_status.send_end_monotonic_ns if sender_status else 0,
                    receive_monotonic_ns=metrics.receive_monotonic_ns,
                    decode_start_monotonic_ns=metrics.decode_start_monotonic_ns,
                    decode_end_monotonic_ns=metrics.decode_end_monotonic_ns,
                    last_decoded_capture_timestamp_ns=metrics.last_capture_timestamp_ns,
                    last_accepted_admission_id=sender_status.last_accepted_admission_id if sender_status else "",
                    last_encoded_admission_id=sender_status.last_encoded_admission_id if sender_status else "",
                    last_sent_admission_id=sender_status.last_sent_admission_id if sender_status else "",
                    last_dropped_admission_id=sender_status.last_dropped_admission_id if sender_status else "",
                    last_dropped_capture_timestamp_ns=max(
                        sender_status.last_dropped_capture_timestamp_ns if sender_status else 0,
                        metrics.last_dropped_capture_timestamp_ns,
                    ),
                    dropped_capture_history_json=(
                        sender_status.dropped_capture_history_json if sender_status else "[]"
                    ),
                    last_drop_reason=(
                        "receiver_queue_overflow"
                        if metrics.receiver_queue_overflow_drops
                        else "rtp_sequence_gap"
                        if metrics.sequence_gap_events
                        else "rtp_reordered_packet"
                        if metrics.reordered_packets
                        else sender_status.last_drop_reason
                        if sender_status
                        else ""
                    ),
                    last_error=receiver_status.last_error,
                )
            )
        return tuple(statuses)

    def decoder_diagnostics(self) -> tuple[tuple[str, object], ...]:
        with self._lock:
            streams = tuple(sorted(self._streams.values(), key=lambda item: item.spec.key))
        return tuple((stream.spec.key, stream.receiver.decoder.metrics) for stream in streams)

    def selection_anchor_ns(self) -> int:
        """Capture timestamp of the newest common selection, 0 when none."""
        with self._lock:
            captures = [item[1] for item in self._selection_diagnostics]
        return min(captures) if captures else 0

    def state_alignment_tolerance_ns(self) -> int:
        for _spec, transport in self._specs.values():
            if transport.readiness is not None:
                return max(0, int(transport.readiness.state_alignment_tolerance_ms)) * 1_000_000
        return 0

    def record_state_alignment(self, delta_ns: int) -> None:
        with self._lock:
            self._last_state_alignment_delta_ns = delta_ns

    def state_alignment_delta_ns(self) -> int | None:
        with self._lock:
            return self._last_state_alignment_delta_ns

    def selection_diagnostics(self) -> tuple[tuple[str, int, float, float], ...]:
        with self._lock:
            return self._selection_diagnostics

    def reset_session(self, session_id: str, session_generation: int) -> None:
        self.negotiator.reset(session_id, session_generation)
        with self._lock:
            streams = tuple(self._streams.values())
            self._streams.clear()
            self._descriptors.clear()
        for stream in streams:
            stream.receiver.close()

    def observe_descriptor(self, descriptor: VideoStreamDescriptor) -> bool:
        ready = self.negotiator.observe_descriptor(descriptor)
        with self._lock:
            self._descriptors[descriptor.observation_key] = descriptor
            active_descriptors = {
                observation_key: stream.descriptor for observation_key, stream in self._streams.items()
            }
            descriptors_changed = self._descriptors != active_descriptors
        if ready and descriptors_changed:
            self._start_receivers()
        return ready

    def observe_status(self, status: VideoStreamRuntimeStatus, *, receive_time_ns: int | None = None) -> bool:
        with self._lock:
            stream = self._streams.get(status.observation_key)
        if stream is None:
            return False
        expected = stream.descriptor
        if (
            status.pipeline_id != expected.pipeline_id
            or status.session_id != expected.session_id
            or status.session_generation != expected.session_generation
            or status.stream_id != expected.stream_id
        ):
            return False
        if status.status_origin != "sender":
            return False
        if status.timestamp_mapping_valid and status.encoded_frames > 0:
            stream.mapper.update(
                status.mapping_rtp_timestamp,
                status.mapping_capture_timestamp_ns,
                time.time_ns() if receive_time_ns is None else receive_time_ns,
                session_generation=status.session_generation,
            )
            stream.sender_status = status
        return True

    def assemble_inputs(self, target_timestamp_ns: int, *, now_ns: int | None = None) -> dict[str, np.ndarray]:
        if not self._specs:
            return {}
        if target_timestamp_ns <= 0:
            raise ValueError("streamed input assembly requires a positive target timestamp")
        current_time_ns = time.time_ns() if now_ns is None else now_ns
        step_ns = round(1_000_000_000 / self.rate_hz)
        timestamps = [target_timestamp_ns - step_ns * offset for offset in reversed(range(self.n_obs_steps))]
        history: dict[str, list[np.ndarray]] = {key: [] for key in self._specs}
        for timestamp_ns in timestamps:
            selections, max_skew_ns = self._selections()
            selected = select_synchronized_streams(
                selections,
                timestamp_ns,
                now_ns=current_time_ns,
                max_inter_camera_skew_ns=max_skew_ns,
            )
            for observation_key, item in selected.items():
                stream = self._streams[observation_key]
                history[observation_key].append(self._canonical_frame(item.value, stream.spec))
            with self._lock:
                self._selection_diagnostics = tuple(
                    (
                        observation_key,
                        item.capture_timestamp_ns,
                        (timestamp_ns - item.capture_timestamp_ns) / 1e6,
                        (current_time_ns - item.receive_timestamp_ns) / 1e6,
                    )
                    for observation_key, item in selected.items()
                )
        return {
            key: values[0] if self.n_obs_steps == 1 else np.ascontiguousarray(np.stack(values)[None, ...])
            for key, values in history.items()
        }

    def close(self) -> None:
        with self._lock:
            streams = tuple(self._streams.values())
            self._streams.clear()
            self._descriptors.clear()
        for stream in streams:
            stream.receiver.close()

    def _start_receivers(self) -> None:
        with self._receiver_start_lock:
            with self._lock:
                descriptors = dict(self._descriptors)
                if set(descriptors) != set(self._specs):
                    return
                if descriptors == {key: stream.descriptor for key, stream in self._streams.items()}:
                    return
                existing = tuple(self._streams.values())
                self._streams.clear()
            for stream in existing:
                stream.receiver.close()
            created: dict[str, _ComputeStream] = {}
            try:
                for observation_key, (spec, transport) in self._specs.items():
                    descriptor = descriptors[observation_key]
                    created[observation_key] = self._create_stream(spec, transport, descriptor)
            except Exception:
                for stream in created.values():
                    stream.receiver.close()
                raise
            with self._lock:
                self._streams = created

    def _create_stream(
        self, spec: SpecView, transport: ObservationTransportSpec, descriptor: VideoStreamDescriptor
    ) -> _ComputeStream:
        assert transport.buffer is not None
        assert transport.readiness is not None
        resolved = self._resolved_decoders[spec.key]
        decoder_options: dict[str, object] = {
            "width": descriptor.width,
            "height": descriptor.height,
            "frame_rate_hz": descriptor.frame_rate_hz,
            "output_pixel_format": "rgb24",
            "color_space": descriptor.color_space,
            "color_range": descriptor.color_range,
        }
        if resolved.name == "ascend":
            decoder_options["channel_id"] = self._decoder_channel_ids[spec.key]
        decoder = resolved.create(**decoder_options) if self._decode else _RecordingOnlyDecoder()
        step_ns = round(1_000_000_000 / self.rate_hz)
        buffer = StreamBuffer(
            spec.resample_policy,
            step_ns,
            spec.asof_tol_ms * 1_000_000,
            max_age_ns=spec.max_age_ms * 1_000_000,
            retention_ns=transport.buffer.retention_ms * 1_000_000,
        )
        mapper = RtpTimestampMapper(
            transport.readiness.timestamp_mapping_max_age_ms * 1_000_000,
            observation_key=spec.key,
            stream_id=descriptor.stream_id,
        )
        mapper.reset(descriptor.session_generation)
        receiver = self._receiver_factory(
            stream_id=descriptor.stream_id,
            observation_key=spec.key,
            ssrc=descriptor.ssrc,
            decoder=decoder,
            frame_buffer=buffer,
            timestamp_mapper=mapper,
            session_generation=descriptor.session_generation,
            packet_queue_capacity=transport.buffer.receiver_queue_packets,
            payload_type=descriptor.payload_type,
            selected_backend=resolved.name,
            endpoint=(descriptor.endpoint_host, descriptor.endpoint_port),
            recorder=self._recorders.get(spec.key),
            decode=self._decode,
        )
        try:
            receiver.start()
        except Exception:
            decoder.close()
            raise
        return _ComputeStream(spec, transport, descriptor, mapper, buffer, receiver)

    def _selections(self) -> tuple[dict[str, StreamSelection], int]:
        with self._lock:
            if set(self._streams) != set(self._specs):
                self.negotiator.validate_request(())
            streams = dict(self._streams)
        max_skew_ns = min(
            stream.transport.readiness.max_inter_camera_skew_ms * 1_000_000 for stream in streams.values()
        )
        return (
            {
                key: StreamSelection(
                    key,
                    stream.descriptor.stream_id,
                    stream.buffer,
                    timestamp_mapping_ready=stream.mapper.ready,
                    keyframe_ready=stream.receiver.status.ready,
                    pad_before_first=self.n_obs_steps > 1,
                    # A 90 kHz RTP timestamp quantizes capture time to about
                    # 11.1 us. Accept only that tiny future offset so a shared
                    # multi-camera reset frame cannot split across epochs.
                    future_tolerance_ns=(1_000_000_000 + 90_000 - 1) // 90_000,
                    session_generation=stream.descriptor.session_generation,
                    last_dropped_capture_timestamp_ns=max(
                        stream.sender_status.last_dropped_capture_timestamp_ns if stream.sender_status else 0,
                        stream.receiver.status.metrics.last_dropped_capture_timestamp_ns,
                    ),
                    dropped_capture_history=self._drop_history(stream),
                    last_dropped_admission_id=(
                        stream.sender_status.last_dropped_admission_id if stream.sender_status else ""
                    ),
                    last_drop_reason=(
                        stream.receiver.status.last_error
                        if stream.receiver.status.metrics.last_dropped_capture_timestamp_ns
                        >= (stream.sender_status.last_dropped_capture_timestamp_ns if stream.sender_status else 0)
                        else stream.sender_status.last_drop_reason
                        if stream.sender_status
                        else ""
                    ),
                )
                for key, stream in streams.items()
            },
            max_skew_ns,
        )

    @staticmethod
    def _drop_history(stream: _ComputeStream) -> tuple[tuple[int, str, str], ...]:
        status = stream.sender_status
        if status is None:
            return ()
        try:
            value = json.loads(status.dropped_capture_history_json or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()
        if not isinstance(value, list):
            return ()
        history = []
        for item in value:
            if isinstance(item, list) and len(item) == 3:
                try:
                    history.append((int(item[0]), str(item[1]), str(item[2])))
                except (TypeError, ValueError):
                    continue
        return tuple(history)

    @staticmethod
    def _canonical_frame(value: object, spec: SpecView) -> np.ndarray:
        if not isinstance(value, VideoFrame):
            raise TypeError(f"decoded stream {spec.key!r} did not produce a VideoFrame")
        encoding = {"rgb24": "rgb8", "bgr24": "bgr8"}.get(value.pixel_format, value.pixel_format)
        return decoded_frame_to_chw_float(
            value.data,
            encoding=encoding,
            output_encoding="rgb8",
            resize=spec.image_resize,
        )

    @staticmethod
    def _requirement(spec: SpecView, transport: ObservationTransportSpec) -> VideoStreamRequirement:
        assert transport.stream_id is not None
        assert transport.endpoint is not None
        assert transport.h264 is not None
        assert transport.media is not None
        return VideoStreamRequirement(
            observation_key=spec.key,
            stream_id=transport.stream_id,
            codec=transport.codec,
            decoder_backend=transport.decoder_backend,
            encoder_backend=transport.encoder_backend,
            codec_profile=transport.h264.profile,
            endpoint_host=transport.endpoint.host,
            endpoint_port=transport.endpoint.port,
            payload_type=96,
            width=transport.media.width,
            height=transport.media.height,
            frame_rate_hz=transport.media.frame_rate_hz,
            pixel_format=transport.media.pixel_format,
            color_space=transport.media.color_space,
            color_range=transport.media.color_range,
        )

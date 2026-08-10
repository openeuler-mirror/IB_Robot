"""Device-side ROS Image compatibility wrapper for the shared frame producer."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import asdict

from inference_service.distributed.types import StreamReference
from inference_service.distributed.video_streams import (
    VideoStreamDescriptor,
    VideoStreamDiagnosticSnapshot,
    VideoStreamRuntimeStatus,
)
from observation_transport.frame_ingress import (
    DirectFrameStreamConfig,
    FrameAdmissionDisposition,
    FrameIngressError,
    StreamSessionView,
    create_frame_ingress,
)
from observation_transport.rtp_sender import H264RtpSender
from observation_transport.video_codec import VideoCodecRegistry, create_default_video_codec_registry
from robot_config.contract_utils import SpecView
from robot_config.observation_transport import effective_observation_transport
from tensormsg.converter import ros_image_to_hwc_uint8

_DEVICE_RAW_QUEUE_FRAMES = 4


class DeviceVideoStreamManager:
    """Translate ROS Image callbacks into the reusable direct-frame ingress.

    Encoding, queueing, RTP packetization, descriptors, statuses, and bounded
    shutdown belong to the native ``FrameIngress``. This class is deliberately
    only the legacy ROS compatibility wrapper used by real camera topics.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        contract_fingerprint: str,
        deployment_fingerprint: str,
        observation_specs: Iterable[SpecView],
        codec_registry: VideoCodecRegistry | None = None,
        sender_factory: Callable[..., H264RtpSender] = H264RtpSender,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.contract_fingerprint = contract_fingerprint
        self.deployment_fingerprint = deployment_fingerprint
        self._specs = {
            spec.key: spec
            for spec in observation_specs
            if effective_observation_transport(spec.transport).mode == "rtp"
        }
        configs = tuple(self._stream_config(spec) for spec in self._specs.values())
        self._producer = create_frame_ingress(
            pipeline_id=pipeline_id,
            contract_fingerprint=contract_fingerprint,
            deployment_fingerprint=deployment_fingerprint,
            streams=configs,
            codec_registry=codec_registry or create_default_video_codec_registry(),
            sender_factory=sender_factory,
        )
        self._diagnostic_started_monotonic_ns = time.monotonic_ns()
        self._diagnostic_baseline: dict[str, tuple[int, int, int, int, int]] = {}
        self._reset_diagnostic_window()

    @property
    def stream_references(self) -> tuple[StreamReference, ...]:
        return tuple(StreamReference(key, stream_id) for key, stream_id in self._producer.stream_references)

    def latest_sent_capture_ns(self, observation_key: str) -> int:
        """Capture timestamp of the newest access unit actually put on the wire.

        Freshness decisions on the device side must reflect what the compute
        side can actually see, not what the local subscription received:
        frames that failed to encode or never left the sender queue do not
        exist remotely, so they must not count as "new".  The value updates
        only from the sender thread's post-send callback and reads as 0 when
        the stream is unknown or nothing has been sent yet -- including right
        after a session rollover, which clears the record so pre-rollover
        frames are never mistaken for fresh.
        """
        for status in self._producer.statuses():
            if status.observation_key == observation_key:
                return int(status.mapping_capture_timestamp_ns) if status.timestamp_mapping_valid else 0
        return 0

    def latest_sent_mapping(self, observation_key: str) -> tuple[int, int]:
        """Return the newest RTP/capture timestamp pair actually sent."""
        for status in self._producer.statuses():
            if status.observation_key == observation_key and status.timestamp_mapping_valid:
                return int(status.mapping_rtp_timestamp), int(status.mapping_capture_timestamp_ns)
        return 0, 0

    @property
    def observation_keys(self) -> frozenset[str]:
        return self._producer.observation_keys

    @property
    def session(self) -> StreamSessionView:
        return self._producer.session

    def diagnostic_snapshots(self) -> tuple[VideoStreamDiagnosticSnapshot, ...]:
        status_by_key = {status.observation_key: status for status in self._producer.statuses()}
        snapshots = []
        for config in sorted(
            (self._stream_config(spec) for spec in self._specs.values()), key=lambda item: item.observation_key
        ):
            status = status_by_key.get(config.observation_key)
            snapshots.append(
                VideoStreamDiagnosticSnapshot(
                    observation_key=config.observation_key,
                    stream_id=config.stream_id,
                    mode="rtp",
                    configured_encoder_backend=config.encoder_backend,
                    selected_encoder_backend=(
                        status.selected_backend if status is not None else config.encoder_backend
                    ),
                    configured_decoder_backend=config.decoder_backend,
                    selected_decoder_backend="not-local",
                    endpoint=(config.endpoint_host, config.endpoint_port),
                    contract_fingerprint=self.contract_fingerprint,
                    deployment_fingerprint=self.deployment_fingerprint,
                    security="none/trusted-network-only",
                    lifecycle_state=status.lifecycle_state if status is not None else "configured",
                    ready=status.ready if status is not None else False,
                )
            )
        return tuple(snapshots)

    def bind_session(
        self,
        session: StreamSessionView | str,
        session_generation: int | None = None,
    ) -> bool:
        """Bind the shared ingress while preserving the native manager API."""
        if isinstance(session, str):
            if session_generation is None:
                raise ValueError("session_generation is required with a session ID")
            session = StreamSessionView(
                pipeline_id=self.pipeline_id,
                session_id=session,
                generation=int(session_generation),
                contract_fingerprint=self.contract_fingerprint,
                deployment_fingerprint=self.deployment_fingerprint,
            )
        elif session_generation is not None:
            raise ValueError("session_generation must not accompany a StreamSessionView")
        changed = self._producer.bind_session(session)
        if changed:
            self._reset_diagnostic_window()
        return changed

    def clear_session(self) -> None:
        self._producer.clear_session()

    def submit_ros_image(
        self,
        observation_key: str,
        message: object,
        *,
        capture_timestamp_ns: int,
        receive_timestamp_ns: int,
    ) -> bool:
        spec = self._specs.get(observation_key)
        if spec is None:
            return False
        if not self.session.active:
            return False
        transport = effective_observation_transport(spec.transport)
        assert transport.media is not None
        frame = ros_image_to_hwc_uint8(
            message,
            output_encoding="rgb8",
            resize=(transport.media.height, transport.media.width),
        )
        try:
            receipt = self._producer.submit_frame(
                observation_key,
                frame,
                capture_timestamp_ns=capture_timestamp_ns,
                receive_timestamp_ns=receive_timestamp_ns,
                pixel_format="rgb24",
            )
        except FrameIngressError as exc:
            if exc.code in {"non_monotonic_timestamp", "session_not_ready", "stale_session"}:
                return False
            raise
        return receipt.disposition is FrameAdmissionDisposition.ACCEPTED

    def descriptors(self) -> tuple[VideoStreamDescriptor, ...]:
        return tuple(VideoStreamDescriptor(**asdict(item)) for item in self._producer.descriptors())

    def statuses(self) -> tuple[VideoStreamRuntimeStatus, ...]:
        return tuple(VideoStreamRuntimeStatus(**asdict(item)) for item in self._producer.statuses())

    def sender_diagnostics(self) -> tuple[dict[str, object], ...]:
        """Project shared-ingress counters into the native sender diagnostics."""
        elapsed_s = max((time.monotonic_ns() - self._diagnostic_started_monotonic_ns) / 1e9, 1e-9)
        status_by_key = {status.observation_key: status for status in self._producer.statuses()}
        diagnostics = []
        for stream in self._producer.snapshot().streams:
            baseline = self._diagnostic_baseline.get(stream.observation_key, (0, 0, 0, 0, 0))
            status = status_by_key.get(stream.observation_key)
            sent_packets = int(status.sent_packets) if status is not None else 0
            diagnostics.append(
                {
                    "observation": stream.observation_key,
                    "submitted_fps": max(0, stream.accepted_frames - baseline[0]) / elapsed_s,
                    "encoded_fps": max(0, stream.encoded_frames - baseline[1]) / elapsed_s,
                    "sent_fps": max(0, stream.sent_frames - baseline[2]) / elapsed_s,
                    "submitted_frames": max(0, stream.accepted_frames - baseline[0]),
                    "encoded_frames": max(0, stream.encoded_frames - baseline[1]),
                    "sent_frames": max(0, stream.sent_frames - baseline[2]),
                    "sent_packets": max(0, sent_packets - baseline[3]),
                    "encode_queue_depth": stream.queue_depth,
                    "sender_queue_depth": int(status.sender_queue_depth) if status is not None else 0,
                    "dropped_frames": max(0, stream.dropped_frames - baseline[4]),
                }
            )
        return tuple(diagnostics)

    def reset(self) -> None:
        self._producer.reset()
        self._reset_diagnostic_window()

    def _reset_diagnostic_window(self) -> None:
        self._diagnostic_started_monotonic_ns = time.monotonic_ns()
        statuses = {status.observation_key: status for status in self._producer.statuses()}
        self._diagnostic_baseline = {
            stream.observation_key: (
                stream.accepted_frames,
                stream.encoded_frames,
                stream.sent_frames,
                int(statuses[stream.observation_key].sent_packets) if stream.observation_key in statuses else 0,
                stream.dropped_frames,
            )
            for stream in self._producer.snapshot().streams
        }

    def close(self, timeout_s: float = 1.0) -> None:
        self._producer.close(timeout_s)

    @staticmethod
    def _stream_config(spec: SpecView) -> DirectFrameStreamConfig:
        transport = effective_observation_transport(spec.transport)
        if transport.stream_id is None or transport.endpoint is None:
            raise ValueError(f"RTP observation {spec.key!r} is missing stream identity or endpoint")
        if transport.h264 is None or transport.media is None or transport.buffer is None:
            raise ValueError(f"RTP observation {spec.key!r} has unresolved codec, media, or buffer settings")
        return DirectFrameStreamConfig(
            observation_key=spec.key,
            stream_id=transport.stream_id,
            endpoint_host=transport.endpoint.host,
            endpoint_port=transport.endpoint.port,
            width=int(transport.media.width),
            height=int(transport.media.height),
            frame_rate_hz=float(transport.media.frame_rate_hz),
            bitrate_bps=transport.h264.bitrate_bps,
            gop_frames=transport.h264.gop_frames,
            codec=transport.codec,
            codec_profile=transport.h264.profile,
            encoder_backend=transport.encoder_backend,
            decoder_backend=transport.decoder_backend,
            pixel_format=transport.media.pixel_format,
            input_pixel_format="rgb24",
            color_space=transport.media.color_space,
            color_range=transport.media.color_range,
            sender_queue_frames=transport.buffer.sender_queue_frames,
            raw_queue_frames=_DEVICE_RAW_QUEUE_FRAMES,
            queue_policy="latest",
        )

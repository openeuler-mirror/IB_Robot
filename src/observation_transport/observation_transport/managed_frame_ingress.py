"""ROS control-plane host for the production IB-Robot :class:`FrameIngress`.

The provider-facing data plane remains the small ``FrameIngress`` protocol.
This module owns distributed-inference session binding, stream descriptor and
status publication, receiver readiness, and the producer lifecycle.  Benchmark
and device integrations create this host through the public factory instead of
implementing those protocol details themselves.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ibrobot_msgs.msg import InferencePipelineStatus, VideoStreamDescriptor, VideoStreamStatus
from observation_transport.frame_ingress import (
    DirectFrameStreamConfig,
    FrameAdmissionReceipt,
    FrameIngress,
    FrameIngressError,
    FrameTransportSnapshot,
    StreamSessionView,
    create_frame_ingress,
)

FRAME_INGRESS_READY_TAG = "[IBROBOT_TRANSPORT][FRAME_INGRESS_READY]"
_MANAGED_CONTRACT_SCHEMA_VERSION = 1
_KEYFRAME_RECOVERY_INTERVAL_NS = 250_000_000
_KEYFRAME_RECOVERY_MAX_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class ManagedFrameIngressConfig:
    """Materialized control and data-plane contract for one ingress owner."""

    pipeline_id: str
    contract_fingerprint: str
    deployment_fingerprint: str
    heartbeat_topic: str
    descriptor_topic: str
    status_topic: str
    streams: tuple[DirectFrameStreamConfig, ...]
    schema_version: int = _MANAGED_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _MANAGED_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported managed frame ingress schema_version={self.schema_version}; "
                f"expected {_MANAGED_CONTRACT_SCHEMA_VERSION}"
            )
        required = (
            self.pipeline_id,
            self.contract_fingerprint,
            self.deployment_fingerprint,
            self.heartbeat_topic,
            self.descriptor_topic,
            self.status_topic,
        )
        if any(not isinstance(value, str) or not value.strip() for value in required):
            raise ValueError("managed frame ingress identity and control topics must be non-empty strings")
        if not self.streams:
            raise ValueError("managed frame ingress requires at least one stream")

    @classmethod
    def from_value(cls, value: str | Mapping[str, Any] | ManagedFrameIngressConfig) -> ManagedFrameIngressConfig:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"frame_ingress_contract_json is invalid: {exc}") from exc
        elif isinstance(value, Mapping):
            payload = dict(value)
        else:
            raise TypeError("managed frame ingress contract must be JSON, a mapping, or a config object")
        if not isinstance(payload, dict):
            raise ValueError("managed frame ingress contract must decode to an object")
        streams_payload = payload.pop("streams", None)
        if not isinstance(streams_payload, list):
            raise ValueError("managed frame ingress contract streams must be a list")
        try:
            streams = tuple(DirectFrameStreamConfig(**item) for item in streams_payload)
            return cls(streams=streams, **payload)
        except TypeError as exc:
            raise ValueError(f"managed frame ingress contract fields are invalid: {exc}") from exc


class ManagedFrameIngress(FrameIngress):
    """Own one native ingress and its distributed ROS control plane."""

    _owners_lock = threading.Lock()
    _owners: set[tuple[str, str, str, str]] = set()

    def __init__(self, node: Any, config: ManagedFrameIngressConfig) -> None:
        self._node = node
        self._config = config
        self._owner_key = (
            config.pipeline_id,
            config.heartbeat_topic,
            config.descriptor_topic,
            config.status_topic,
        )
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._receiver_statuses: dict[str, VideoStreamStatus] = {}
        self._recovery_attempts: dict[str, int] = {}
        self._last_recovery_monotonic_ns: dict[str, int] = {}
        self._closed = False
        self._last_diagnostic_signatures: dict[str, tuple[object, ...]] = {}
        self._descriptor_pub: Any = None
        self._status_pub: Any = None
        self._heartbeat_sub: Any = None
        self._status_sub: Any = None
        self._timer: Any = None
        self._ingress: Any = None
        self._claim_owner()
        try:
            self._ingress = create_frame_ingress(
                pipeline_id=config.pipeline_id,
                contract_fingerprint=config.contract_fingerprint,
                deployment_fingerprint=config.deployment_fingerprint,
                streams=config.streams,
                on_control_update=self.publish_control,
            )
            control_qos = QoSProfile(
                depth=max(1, len(config.streams)),
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._descriptor_pub = node.create_publisher(
                VideoStreamDescriptor,
                config.descriptor_topic,
                control_qos,
            )
            self._status_pub = node.create_publisher(VideoStreamStatus, config.status_topic, 10)
            self._heartbeat_sub = node.create_subscription(
                InferencePipelineStatus,
                config.heartbeat_topic,
                self._heartbeat_callback,
                control_qos,
                callback_group=ReentrantCallbackGroup(),
            )
            self._status_sub = node.create_subscription(
                VideoStreamStatus,
                config.status_topic,
                self._receiver_status_callback,
                10,
                callback_group=ReentrantCallbackGroup(),
            )
            self._timer = node.create_timer(0.1, self.publish_control)
        except Exception:
            try:
                self._close_resources(timeout_s=1.0)
            finally:
                self._release_owner()
            raise

    @property
    def observation_keys(self) -> frozenset[str]:
        return self._ingress.observation_keys

    def ready(self) -> bool:
        with self._lock:
            session = self._ingress.session
            return session.active and all(self._receiver_is_current(key, session) for key in self.observation_keys)

    def wait_ready(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError("frame ingress startup timeout must be positive")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.ready():
                session = self._ingress.session
                self._node.get_logger().info(
                    f"{FRAME_INGRESS_READY_TAG} pipeline={self._config.pipeline_id} "
                    f"session={session.session_id} generation={session.generation} "
                    f"streams={sorted(self.observation_keys)}"
                )
                return
            import rclpy  # noqa: PLC0415

            rclpy.spin_once(self._node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
        raise TimeoutError("frame ingress receiver did not configure before the startup deadline")

    def submit_frame(
        self,
        observation_key: str,
        frame: np.ndarray,
        *,
        capture_timestamp_ns: int,
        receive_timestamp_ns: int,
        pixel_format: str,
    ) -> FrameAdmissionReceipt:
        if not self.ready():
            raise FrameIngressError(
                "receiver_not_ready",
                "frame ingress receiver is not ready for the active session",
                observation_key=observation_key,
                recoverable=True,
            )
        return self._ingress.submit_frame(
            observation_key,
            frame,
            capture_timestamp_ns=capture_timestamp_ns,
            receive_timestamp_ns=receive_timestamp_ns,
            pixel_format=pixel_format,
        )

    def snapshot(self) -> FrameTransportSnapshot:
        return self._ingress.snapshot()

    def publish_control(self) -> None:
        with self._lock:
            if self._closed:
                return
        self._publish_worker_diagnostics()
        stamp = self._node.get_clock().now().to_msg()
        for descriptor in self._ingress.descriptors():
            message = VideoStreamDescriptor()
            message.header.stamp = stamp
            for key, value in asdict(descriptor).items():
                setattr(message, key, value)
            self._descriptor_pub.publish(message)
        for status in self._ingress.statuses():
            if not status.timestamp_mapping_valid:
                continue
            message = VideoStreamStatus()
            message.header.stamp = stamp
            values = asdict(status)
            capture_ns = int(values.pop("mapping_capture_timestamp_ns"))
            for key, value in values.items():
                setattr(message, key, value)
            message.mapping_capture_time.sec, message.mapping_capture_time.nanosec = divmod(capture_ns, 1_000_000_000)
            self._status_pub.publish(message)

    def _publish_worker_diagnostics(self) -> None:
        """Report transport worker failures/stalls without exposing media control to Benchmark."""
        snapshot = self._ingress.snapshot()
        for stream in snapshot.streams:
            signature = (
                stream.worker_alive,
                stream.worker_failed,
                stream.last_error,
                stream.accepted_frames // 25,
                stream.recovery_requested,
                stream.recovery_attempts,
            )
            if signature == self._last_diagnostic_signatures.get(stream.observation_key):
                continue
            self._last_diagnostic_signatures[stream.observation_key] = signature
            message = (
                "[IBROBOT_TRANSPORT][FRAME_INGRESS_WORKER] "
                f"observation={stream.observation_key} stream={stream.stream_id} "
                f"failed={stream.worker_failed} alive={stream.worker_alive} "
                f"accepted_frames={stream.accepted_frames} encoded_frames={stream.encoded_frames} "
                f"sent_frames={stream.sent_frames} queue_depth={stream.queue_depth} "
                f"recovery_requested={stream.recovery_requested} "
                f"recovery_attempts={stream.recovery_attempts} "
                f"last_progress={stream.worker_last_progress_kind} "
                f"last_error={stream.last_error!r}"
            )
            logger = self._node.get_logger()
            if stream.worker_failed or stream.last_error:
                logger.error(message)
            elif stream.accepted_frames == 0 or stream.accepted_frames % 25 == 0 or stream.recovery_attempts:
                logger.info(message)

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        try:
            self._close_resources(timeout_s=timeout_s)
        finally:
            self._release_owner()

    def _heartbeat_callback(self, message: InferencePipelineStatus) -> None:
        if message.role != InferencePipelineStatus.ROLE_CLOUD:
            return
        if (
            message.pipeline_id != self._config.pipeline_id
            or message.deployment_fingerprint != self._config.deployment_fingerprint
        ):
            return
        if not message.ready or not message.session_id or message.session_generation < 1:
            return
        session = StreamSessionView(
            pipeline_id=message.pipeline_id,
            session_id=message.session_id,
            generation=int(message.session_generation),
            contract_fingerprint=self._config.contract_fingerprint,
            deployment_fingerprint=message.deployment_fingerprint,
        )
        if self._ingress.bind_session(session):
            with self._condition:
                self._receiver_statuses.clear()
                self._recovery_attempts.clear()
                self._last_recovery_monotonic_ns.clear()
                self._condition.notify_all()
            self.publish_control()

    def _receiver_status_callback(self, message: VideoStreamStatus) -> None:
        if message.pipeline_id != self._config.pipeline_id or message.observation_key not in self.observation_keys:
            return
        session = self._ingress.session
        if not self._status_matches_session(message, session):
            return
        if message.lifecycle_state not in {"waiting_for_keyframe", "degraded", "ready"}:
            return
        # The topic is bidirectional, so status direction is part of the wire
        # contract.  Never infer it from counters: sender status intentionally
        # includes the latest receiver diagnostics and would otherwise feed its
        # own lifecycle/readiness back into recovery control.
        if message.status_origin != "receiver":
            return
        should_recover = False
        observe_receiver_status = getattr(self._ingress, "observe_receiver_status", None)
        if callable(observe_receiver_status):
            observe_receiver_status(
                message.observation_key,
                {
                    "received_packets": int(message.received_packets),
                    "decoded_frames": int(message.decoded_frames),
                    "dropped_packets": int(message.dropped_packets),
                    "lost_packets": int(message.lost_packets),
                    "receiver_queue_depth": int(message.receiver_queue_depth),
                    "decoded_buffer_depth": int(message.decoded_buffer_depth),
                    "receiver_queue_overflow_drops": int(message.receiver_queue_overflow_drops),
                    "sequence_gap_events": int(message.sequence_gap_events),
                    "reordered_packets": int(message.reordered_packets),
                    "recovery_keyframes": int(message.recovery_keyframes),
                    "jitter_ns": int(message.jitter_ns),
                    "receive_monotonic_ns": int(message.receive_monotonic_ns),
                    "decode_start_monotonic_ns": int(message.decode_start_monotonic_ns),
                    "decode_end_monotonic_ns": int(message.decode_end_monotonic_ns),
                    "last_decoded_capture_timestamp_ns": int(message.last_decoded_capture_timestamp_ns),
                    "last_dropped_capture_timestamp_ns": int(getattr(message, "last_dropped_capture_timestamp_ns", 0)),
                    "dropped_capture_history_json": str(getattr(message, "dropped_capture_history_json", "[]")),
                    "last_drop_reason": str(message.last_drop_reason),
                },
            )
        with self._condition:
            self._receiver_statuses[message.observation_key] = message
            if message.ready and message.lifecycle_state == "ready":
                self._recovery_attempts.pop(message.observation_key, None)
                self._last_recovery_monotonic_ns.pop(message.observation_key, None)
            elif message.lifecycle_state in {"waiting_for_keyframe", "degraded"}:
                now_ns = time.monotonic_ns()
                attempts = self._recovery_attempts.get(message.observation_key, 0)
                last_ns = self._last_recovery_monotonic_ns.get(message.observation_key, 0)
                should_recover = (
                    attempts < _KEYFRAME_RECOVERY_MAX_ATTEMPTS and now_ns - last_ns >= _KEYFRAME_RECOVERY_INTERVAL_NS
                )
            self._condition.notify_all()
        if should_recover:
            request_recovery = getattr(self._ingress, "request_keyframe_recovery", None)
            if callable(request_recovery) and request_recovery(
                message.observation_key,
                session_generation=session.generation,
            ):
                with self._condition:
                    self._recovery_attempts[message.observation_key] = (
                        self._recovery_attempts.get(message.observation_key, 0) + 1
                    )
                    self._last_recovery_monotonic_ns[message.observation_key] = time.monotonic_ns()

    def _receiver_is_current(self, key: str, session: StreamSessionView) -> bool:
        status = self._receiver_statuses.get(key)
        if status is None or not self._status_matches_session(status, session):
            return False
        if status.lifecycle_state in {"waiting_for_keyframe", "degraded"}:
            # Bootstrap and recovery must accept frames before the first/new
            # keyframe can make the inference-side buffer ready.
            return True
        return bool(
            status.lifecycle_state == "ready"
            and status.ready
            and status.timestamp_mapping_valid
            and status.keyframe_ready
            and int(status.received_packets) > 0
            and int(status.decoded_frames) > 0
            and int(status.last_decoded_capture_timestamp_ns) > 0
        )

    @staticmethod
    def _status_matches_session(message: VideoStreamStatus, session: StreamSessionView) -> bool:
        return (
            session.active
            and message.session_id == session.session_id
            and int(message.session_generation) == session.generation
        )

    def _claim_owner(self) -> None:
        with self._owners_lock:
            if self._owner_key in self._owners:
                raise FrameIngressError(
                    "duplicate_owner",
                    "duplicate managed frame ingress owner for "
                    f"pipeline={self._config.pipeline_id!r} control topics={self._owner_key[1:]!r}",
                )
            self._owners.add(self._owner_key)

    def _release_owner(self) -> None:
        with self._owners_lock:
            self._owners.discard(self._owner_key)

    def _close_resources(self, *, timeout_s: float) -> None:
        error: Exception | None = None
        if self._timer is not None:
            with contextlib.suppress(Exception):
                self._node.destroy_timer(self._timer)
            self._timer = None
        for attribute, destroy in (
            ("_heartbeat_sub", "destroy_subscription"),
            ("_status_sub", "destroy_subscription"),
            ("_descriptor_pub", "destroy_publisher"),
            ("_status_pub", "destroy_publisher"),
        ):
            handle = getattr(self, attribute)
            if handle is not None:
                with contextlib.suppress(Exception):
                    getattr(self._node, destroy)(handle)
                setattr(self, attribute, None)
        if self._ingress is not None:
            try:
                self._ingress.close(timeout_s)
            except Exception as exc:  # preserve cleanup of the owner lease
                error = exc
            self._ingress = None
        if error is not None:
            raise error


def create_managed_frame_ingress(
    node: Any,
    contract: str | Mapping[str, Any] | ManagedFrameIngressConfig,
    *,
    startup_timeout_s: float | None = None,
) -> FrameIngress:
    """Create, own, and optionally await one native ingress control plane."""

    managed = ManagedFrameIngress(node, ManagedFrameIngressConfig.from_value(contract))
    if startup_timeout_s is None:
        return managed
    try:
        managed.wait_ready(startup_timeout_s)
    except Exception:
        managed.close()
        raise
    return managed


__all__ = [
    "FRAME_INGRESS_READY_TAG",
    "ManagedFrameIngress",
    "ManagedFrameIngressConfig",
    "create_managed_frame_ingress",
]

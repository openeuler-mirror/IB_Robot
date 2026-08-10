"""Generic observation publisher helper for the benchmark environment node.

Benchmark runtime scope: a tiny generic helper that builds ROS publishers from a SSOT
Contract mapping and publishes pre-encoded observation payloads (NumPy
arrays) with an environment-owned stamp. It is intentionally:

- rclpy-only (no LIBERO, no robot_config, no benchmark_libero);
- payload-agnostic: the adapter codec produces the final contiguous RGB uint8
  HWC images and finite float32 state vector, the publisher only wraps them
  into ``sensor_msgs/Image`` and ``ibrobot_msgs/StampedFloat32MultiArray`` and
  stamps them with the same non-zero environment-owned stamp;
- deterministic: it publishes in the exact order the SSOT Contract lists, so
  downstream subscribers and tests can rely on a stable publication sequence.

The canonical Contract is consumed by both the robot configuration layer and
the inference pipeline.  The publisher only reads the subset of canonical fields it needs
(``image.resize`` and ``image.encoding``); the remaining canonical fields
(``selector.names``, ``align.*``) are consumed by the inference subscriber
and the canonical feature constructor.

Missing keys, wrong ROS type, wrong encoding, or a topic string that is not
an absolute path (leading ``/``) fail fast.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

from benchmark_runtime.io_descriptor import ObservationBatch
from benchmark_runtime.observation_router import DeliveryContext, ObservationSink, PreparedRoute
from ibrobot_msgs.msg import StampedFloat32MultiArray

SUPPORTED_ROS_TYPES: frozenset[str] = frozenset(
    {
        "sensor_msgs/msg/Image",
        "ibrobot_msgs/msg/StampedFloat32MultiArray",
    }
)


class ObservationContractError(ValueError):
    """Raised when the SSOT benchmark Contract cannot be resolved."""


@dataclass(frozen=True, slots=True)
class ObservationPublisherSpec:
    """Resolved publisher spec for one observation key.

    Canonical observation contract: image dimensions come from the canonical ``image.resize: [H, W]``
    field, never from a publisher-local ``image.height``/``image.width``
    schema. The publisher exposes the resolved ``image_height`` /
    ``image_width`` as runtime fields so message construction can validate
    payload shapes against the SSOT.

    Vector observations declare their dimension through the canonical
    ``selector.names`` list. The publisher derives ``state_dim`` from that
    list and requires the payload shape to match exactly. Generic runtime
    never hard-codes a provider-specific vector dimension; the dimension comes from the SSOT
    Contract. When a vector observation lacks ``selector.names`` and the
    payload type requires a declared shape, the publisher fails fast.
    """

    key: str
    topic: str
    ros_type: str
    qos: QoSProfile
    image_encoding: str  # "rgb8" for Image msgs; "" for state
    image_height: int  # 0 for non-image
    image_width: int  # 0 for non-image
    state_dim: int  # 0 when not a vector observation or names not declared
    optional: bool = False
    transport_mode: str = "dds"


def _require_non_empty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationContractError(f"{label} must be a non-empty string")
    return value


def _qos_from_dict(d: Mapping[str, Any] | None) -> QoSProfile:
    """Parse a small QoS mapping. Defaults to reliable + volatile + depth 10."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

    if d is None:
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
    if not isinstance(d, Mapping):
        raise ObservationContractError("qos must be a mapping when present")
    rel_str = str(d.get("reliability", "reliable")).lower()
    hist_str = str(d.get("history", "keep_last")).lower()
    dur_str = str(d.get("durability", "volatile")).lower()
    depth = int(d.get("depth", 10))
    return QoSProfile(
        reliability=(ReliabilityPolicy.BEST_EFFORT if rel_str == "best_effort" else ReliabilityPolicy.RELIABLE),
        history=(HistoryPolicy.KEEP_ALL if hist_str == "keep_all" else HistoryPolicy.KEEP_LAST),
        depth=depth,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if dur_str == "transient_local" else DurabilityPolicy.VOLATILE),
    )


def _parse_image_resize(resize: Any, key: str) -> tuple[int, int]:
    """Parse the canonical ``image.resize: [H, W]`` field into (height, width).

    Canonical observation contract: this is the canonical schema used by
    ``robot_config.contract_utils.iter_specs`` and the inference StreamBuffer.
    The publisher must NOT invent a parallel ``image.height``/``image.width``
    schema.
    """
    if not isinstance(resize, list | tuple) or len(resize) != 2:
        raise ObservationContractError(f"image.resize for key '{key}' must be a 2-element list [H, W]; got {resize!r}")
    try:
        h = int(resize[0])
        w = int(resize[1])
    except (TypeError, ValueError) as exc:
        raise ObservationContractError(f"image.resize for key '{key}' must contain integers; got {resize!r}") from exc
    if h <= 0 or w <= 0:
        raise ObservationContractError(f"image.resize for key '{key}' must be positive; got [h={h}, w={w}]")
    return (h, w)


def _parse_state_dim(entry: Mapping[str, Any], key: str, ros_type: str) -> int:
    """Derive the declared vector dimension from canonical ``selector.names``.

    The canonical Contract declares vector dimensions through
    ``selector.names`` (a list of feature names). The publisher derives the
    expected payload length from that list and enforces it strictly.

    For image observations, ``selector`` is unused; returns 0.

    For non-image (vector) observations:
    - if ``selector.names`` is present, the length is the declared dimension;
    - if ``selector.names`` is missing or empty, the publisher cannot prove
      the declared shape; fail fast rather than silently accepting any
      1-D float32 payload.

    The generic runtime never hard-codes a provider-specific vector dimension; the dimension comes
    from the SSOT Contract.
    """
    if ros_type == "sensor_msgs/msg/Image":
        return 0
    selector = entry.get("selector")
    if not isinstance(selector, Mapping):
        raise ObservationContractError(
            f"selector.names is required for vector observation '{key}' "
            f"(ros_type={ros_type}); the SSOT must declare the expected shape"
        )
    names = selector.get("names")
    if not isinstance(names, list) or not names:
        raise ObservationContractError(
            f"selector.names must be a non-empty list for vector observation '{key}' "
            f"(ros_type={ros_type}); the SSOT must declare the expected shape"
        )
    return len(names)


def resolve_publisher_specs(contract_observations: Any) -> list[ObservationPublisherSpec]:
    """Resolve the SSOT Contract observation list into publisher specs.

    Canonical observation contract: the input is the value of ``robot.contract.observations`` (the
    same list ``inference_service`` reads through the canonical robot
    loader). It must be a list of mappings with ``key`` / ``topic`` /
    ``type`` and optional ``qos`` / ``image`` sub-mappings. Image
    observations must use the canonical ``image.resize: [H, W]`` and
    ``image.encoding`` fields. Topics must be absolute paths starting with
    ``/``. Duplicate keys or topics are rejected.
    """
    if not isinstance(contract_observations, list) or not contract_observations:
        raise ObservationContractError("robot.contract.observations must be a non-empty list")
    specs: list[ObservationPublisherSpec] = []
    seen_keys: set[str] = set()
    seen_topics: set[str] = set()
    for entry in contract_observations:
        if not isinstance(entry, Mapping):
            raise ObservationContractError("each contract entry must be a mapping")
        key = _require_non_empty_str(entry.get("key"), "key")
        topic = _require_non_empty_str(entry.get("topic"), "topic")
        if not topic.startswith("/"):
            raise ObservationContractError(f"topic for key '{key}' must be an absolute path starting with '/'")
        ros_type = _require_non_empty_str(entry.get("type"), "type")
        if ros_type not in SUPPORTED_ROS_TYPES:
            raise ObservationContractError(
                f"unsupported ros_type '{ros_type}' for key '{key}'; expected one of {sorted(SUPPORTED_ROS_TYPES)}"
            )
        qos = _qos_from_dict(entry.get("qos"))
        image_section = entry.get("image")
        encoding = ""
        height = 0
        width = 0
        if ros_type == "sensor_msgs/msg/Image":
            if not isinstance(image_section, Mapping):
                raise ObservationContractError(f"image sub-mapping is required for key '{key}' (sensor_msgs/msg/Image)")
            encoding = _require_non_empty_str(image_section.get("encoding"), "image.encoding")
            if encoding != "rgb8":
                raise ObservationContractError(f"image.encoding for key '{key}' must be 'rgb8'; got '{encoding}'")
            height, width = _parse_image_resize(image_section.get("resize"), key)
        elif image_section is not None:
            raise ObservationContractError(f"image sub-mapping is not allowed for non-image key '{key}'")
        # Derive vector dimension from canonical ``selector.names``.
        # for non-image observations. Image observations return 0.
        state_dim = _parse_state_dim(entry, key, ros_type)
        if key in seen_keys:
            raise ObservationContractError(f"duplicate observation key '{key}'")
        if topic in seen_topics:
            raise ObservationContractError(f"duplicate observation topic '{topic}'")
        transport = entry.get("transport") or {}
        if not isinstance(transport, Mapping):
            raise ObservationContractError(f"transport for key '{key}' must be a mapping when present")
        transport_mode = str(transport.get("mode", "dds")).lower()
        if transport_mode not in {"dds", "rtp"}:
            raise ObservationContractError(f"unsupported transport mode '{transport_mode}' for key '{key}'")
        if transport_mode == "rtp" and ros_type != "sensor_msgs/msg/Image":
            raise ObservationContractError(f"RTP observation '{key}' must be sensor_msgs/msg/Image")
        seen_keys.add(key)
        seen_topics.add(topic)
        specs.append(
            ObservationPublisherSpec(
                key=key,
                topic=topic,
                ros_type=ros_type,
                qos=qos,
                image_encoding=encoding,
                image_height=height,
                image_width=width,
                state_dim=state_dim,
                optional=bool(entry.get("optional", False)),
                transport_mode=transport_mode,
            )
        )
    return specs


def stamp_to_builtin_time(stamp: rclpy.time.Time) -> tuple[int, int]:
    """Convert an rclpy Time to ``(sec, nanosec)``."""
    total_ns = int(stamp.nanoseconds)
    if total_ns < 0:
        raise ValueError(f"invalid stamp nanoseconds={total_ns}")
    sec = total_ns // 1_000_000_000
    nanosec = total_ns % 1_000_000_000
    return (sec, nanosec)


def _build_image_message(
    spec: ObservationPublisherSpec,
    payload: np.ndarray,
    sec: int,
    nanosec: int,
) -> ImageMsg:
    if payload.dtype != np.uint8:
        raise ValueError(f"image payload for key '{spec.key}' must be uint8, got {payload.dtype}")
    if payload.ndim != 3 or payload.shape[2] != 3:
        raise ValueError(f"image payload for key '{spec.key}' must be HWC [H,W,3], got shape {payload.shape}")
    h, w, _ = payload.shape
    if h != spec.image_height or w != spec.image_width:
        raise ValueError(
            f"image payload for key '{spec.key}' has shape {payload.shape}, "
            f"expected [{spec.image_height},{spec.image_width},3]"
        )
    if not payload.flags["C_CONTIGUOUS"]:
        payload = np.ascontiguousarray(payload)
    msg = ImageMsg()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = spec.key
    msg.height = h
    msg.width = w
    msg.encoding = spec.image_encoding
    msg.step = w * 3
    msg.data = payload.tobytes()
    return msg


def _build_stamped_float32_message(
    spec: ObservationPublisherSpec,
    payload: np.ndarray,
    sec: int,
    nanosec: int,
) -> StampedFloat32MultiArray:
    if payload.dtype != np.float32:
        raise ValueError(f"state payload for key '{spec.key}' must be float32, got {payload.dtype}")
    if payload.ndim != 1:
        raise ValueError(f"state payload for key '{spec.key}' must be 1-D, got ndim={payload.ndim}")
    # Enforce the declared vector dimension from the SSOT Contract.
    # (selector.names length). Generic runtime never hard-codes a provider-specific vector dimension;
    # the dimension comes from the canonical Contract. A payload that does
    # not match the declared shape is rejected before any publish.
    if spec.state_dim > 0 and payload.shape[0] != spec.state_dim:
        raise ValueError(
            f"state payload for key '{spec.key}' has shape {payload.shape}, "
            f"expected ({spec.state_dim},) from SSOT selector.names"
        )
    if not bool(np.all(np.isfinite(payload))):
        raise ValueError(f"state payload for key '{spec.key}' contains NaN or Inf")
    if not payload.flags["C_CONTIGUOUS"]:
        payload = np.ascontiguousarray(payload)
    msg = StampedFloat32MultiArray()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = spec.key
    msg.value.layout.dim = []
    msg.value.data = payload.tolist()
    return msg


class DDSObservationSink(ObservationSink):
    """ROS/DDS route that prepares one message and publishes on commit."""

    def __init__(self, spec: ObservationPublisherSpec, publisher: Any) -> None:
        self._spec = spec
        self._publisher = publisher
        self._closed = False

    @property
    def key(self) -> str:
        return self._spec.key

    @property
    def transport(self) -> str:
        return "dds"

    @property
    def optional(self) -> bool:
        return self._spec.optional

    def prepare(self, batch: ObservationBatch, context: DeliveryContext) -> PreparedRoute:
        if self._closed:
            raise RuntimeError("DDS observation route is closed")
        payload = batch[self.key]
        if not isinstance(payload, np.ndarray):
            raise ValueError(f"payload for key '{self.key}' must be np.ndarray, got {type(payload).__name__}")
        if self._spec.ros_type == "sensor_msgs/msg/Image":
            message = _build_image_message(self._spec, payload, context.timestamp_sec, context.timestamp_nanosec)
        else:
            message = _build_stamped_float32_message(
                self._spec, payload, context.timestamp_sec, context.timestamp_nanosec
            )
        return PreparedRoute(key=self.key, transport=self.transport, payload=message)

    def commit(self, prepared: PreparedRoute) -> None:
        if self._closed:
            raise RuntimeError("DDS observation route is closed")
        if prepared.key != self.key or prepared.transport != self.transport:
            raise ValueError("prepared DDS route identity mismatch")
        self._publisher.publish(prepared.payload)

    def close(self) -> None:
        self._closed = True


def build_dds_sinks(
    publishers: Mapping[str, Any],
    specs: list[ObservationPublisherSpec],
) -> tuple[DDSObservationSink, ...]:
    """Build DDS sinks in canonical Contract order."""
    routes: list[DDSObservationSink] = []
    for spec in specs:
        if spec.transport_mode != "dds":
            continue
        publisher = publishers.get(spec.key)
        if publisher is None:
            raise ObservationContractError(f"no publisher registered for key '{spec.key}'")
        routes.append(DDSObservationSink(spec, publisher))
    return tuple(routes)


# Phase 3 exposed route-oriented names.  They remain exact aliases so existing
# callers keep class/function identity while new composition code uses the
# semantically precise sink terminology.
DDSObservationRoute = DDSObservationSink
build_dds_routes = build_dds_sinks


def build_and_validate_messages(
    publishers: Mapping[str, Any],
    specs: list[ObservationPublisherSpec],
    observations: Mapping[str, np.ndarray],
    stamp: rclpy.time.Time,
) -> list[tuple[Any, Any]]:
    """Build and validate ALL observation messages before any publish.

    Observation transaction: returns a list of ``(publisher, msg)`` tuples in deterministic
    Contract order. All payload validation (missing key, wrong dtype, wrong
    shape, non-finite state) happens here, BEFORE any ``publisher.publish``
    call. This guarantees that a validation failure on the third observation
    does not leave the first two already published.

    Raises ``ValueError`` for any missing key, wrong dtype, wrong shape or
    non-finite state payload. On success, the caller publishes the prepared
    messages via :func:`publish_prepared_messages`.
    """
    sec, nanosec = stamp_to_builtin_time(stamp)
    non_dds = [spec.key for spec in specs if spec.transport_mode != "dds"]
    if non_dds:
        raise ObservationContractError(
            "legacy DDS observation publishing cannot route non-DDS observations; "
            f"use ObservationRouter with ObservationSink values for {non_dds}"
        )
    prepared: list[tuple[Any, Any]] = []
    for spec in specs:
        if spec.key not in observations:
            if spec.optional:
                continue
            raise ValueError(f"missing observation payload for key '{spec.key}'")
        payload = observations[spec.key]
        if not isinstance(payload, np.ndarray):
            raise ValueError(f"payload for key '{spec.key}' must be np.ndarray, got {type(payload).__name__}")
        publisher = publishers.get(spec.key)
        if publisher is None:
            raise ValueError(f"no publisher registered for key '{spec.key}'")
        if spec.ros_type == "sensor_msgs/msg/Image":
            msg = _build_image_message(spec, payload, sec, nanosec)
        else:
            msg = _build_stamped_float32_message(spec, payload, sec, nanosec)
        prepared.append((publisher, msg))
    return prepared


def publish_prepared_messages(prepared: list[tuple[Any, Any]]) -> None:
    """Publish legacy DDS messages prepared by :func:`build_and_validate_messages`.

    Observation transaction: this function only publishes; it does NOT validate. All validation
    must have completed in ``build_and_validate_messages`` so that a partial
    publication cannot occur from a payload validation failure.

    A ROS publisher ``publish()`` call that raises AFTER some messages have
    already been published represents transport uncertainty: the episode is
    poisoned/reset-required and the native step is never retried. This is
    documented in the environment node's transaction handlers.
    """
    for publisher, msg in prepared:
        publisher.publish(msg)


def publish_observations(
    publishers: Mapping[str, Any],
    specs: list[ObservationPublisherSpec],
    observations: Mapping[str, np.ndarray],
    stamp: rclpy.time.Time,
) -> None:
    """Build/validate ALL messages, then publish them.

    Convenience wrapper around :func:`build_and_validate_messages` +
    :func:`publish_prepared_messages`. Kept for backward compatibility with
    callers that do not need to interleave serialization between build and
    publish. The environment node uses the split functions directly so that
    JSON serialization happens AFTER all payload validation but BEFORE any
    publish.
    """
    prepared = build_and_validate_messages(publishers, specs, observations, stamp)
    publish_prepared_messages(prepared)


def build_publishers(
    node: Node,
    specs: list[ObservationPublisherSpec],
) -> dict[str, Any]:
    """Create one ROS publisher per spec on ``node`` and return a key->pub map."""
    publishers: dict[str, Any] = {}
    for spec in specs:
        if spec.transport_mode != "dds":
            continue
        if spec.ros_type == "sensor_msgs/msg/Image":
            publishers[spec.key] = node.create_publisher(
                ImageMsg,
                spec.topic,
                spec.qos,
            )
        elif spec.ros_type == "ibrobot_msgs/msg/StampedFloat32MultiArray":
            publishers[spec.key] = node.create_publisher(
                StampedFloat32MultiArray,
                spec.topic,
                spec.qos,
            )
        else:  # pragma: no cover - resolve_publisher_specs already rejects
            raise ObservationContractError(f"unsupported ros_type '{spec.ros_type}'")
    return publishers


def make_zero_header() -> Header:
    """Return a ``Header`` with zero stamp, used by failure responses.

    The benchmark runtime contract: failure service responses use zero observation
    timestamp; never fabricate a sampleable timestamp for a failed operation.
    """
    header = Header()
    header.stamp.sec = 0
    header.stamp.nanosec = 0
    return header

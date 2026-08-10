# -----------------------------------------------------------------------------
# Contract schema + runtime processing.
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import threading
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

# Bridge to tensormsg for global registry
import tensormsg.registry as tensormsg_registry
from robot_config.observation_transport import (
    ObservationTransportSpec,
    effective_observation_transport,
    observation_transport_to_dict,
    parse_observation_transport,
)

# ---------- Contract datamodel ----------


@dataclass(frozen=True, slots=True)
class AlignSpec:
    """Timestamp alignment/selection behavior for an observation stream.

    strategy:  "hold" | "asof" | "drop"
    tol_ms:    as-of tolerance in ms for 'asof' (ignored for others)
    max_age_ms: maximum live age measured from local receipt time
    stamp:     "receive" | "header"
    """

    strategy: str = "hold"
    tol_ms: int = 0
    max_age_ms: int = 0
    stamp: str = "receive"


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Observation stream description (image/vector), driven by AlignSpec."""

    key: str
    topic: str
    type: str
    selector: dict[str, Any] | None = None  # {names: [...]}
    image: dict[str, Any] | None = None  # {resize:[H,W], encoding:'rgb8'|'bgr8'|'mono8'...}
    align: AlignSpec | None = None
    # {reliability, history, depth, durability}
    qos: dict[str, Any] | None = None
    transport: ObservationTransportSpec | None = None


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Action stream description (publisher settings + mapping)."""

    key: str
    publish_topic: str
    type: str
    selector: dict[str, Any] | None = None
    from_tensor: dict[str, Any] | None = None
    publish_qos: dict[str, Any] | None = None
    publish_strategy: dict[str, Any] | None = None
    safety_behavior: str = "zeros"  # "zeros" | "hold"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Optional 'task' channels (e.g., prompts)."""

    key: str
    topic: str
    type: str
    qos: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Contract:
    """Top-level contract describing a policy's ROS 2 I/O surface."""

    name: str
    version: int
    rate_hz: float
    max_duration_s: float
    observations: list[ObservationSpec]
    actions: list[ActionSpec]
    tasks: list[TaskSpec]
    recording: dict[str, Any]
    robot_type: str | None = None
    timestamp_source: str = "receive"
    process: dict[str, Any] = None


def _as_align(d: dict[str, Any] | None) -> AlignSpec | None:
    if not d:
        return None
    tol_ms = int(d.get("tol_ms", 0))
    max_age_ms = int(d.get("max_age_ms", 0))
    if tol_ms < 0:
        raise ValueError("align.tol_ms must be non-negative")
    if max_age_ms < 0:
        raise ValueError("align.max_age_ms must be non-negative")
    return AlignSpec(
        strategy=str(d.get("strategy", "hold")).lower(),
        tol_ms=tol_ms,
        max_age_ms=max_age_ms,
        stamp=str(d.get("stamp", "receive")).lower(),
    )


# ---------- Unified SpecView (runtime) ----------


@dataclass(frozen=True, slots=True)
class SpecView:
    """Normalized runtime view of a single stream (observation or action)."""

    key: str
    topic: str
    ros_type: str
    is_action: bool
    names: list[str]
    image_resize: tuple[int, int] | None
    image_encoding: str
    image_channels: int  # 1, 3, or 4
    resample_policy: str  # obs: align.strategy; actions: "hold"
    asof_tol_ms: int
    max_age_ms: int
    stamp_src: str
    clamp: tuple[float, float] | None  # actions only
    safety_behavior: str | None  # actions only: "zeros" | "hold"
    transport: ObservationTransportSpec | None


# Driver pixel_format names that are not ROS Image encodings but have a
# well-defined encoding on the wire. usb_cam 0.8.1 publishes RGB8 for its
# ``mjpeg2rgb`` conversion, so mirroring that driver means rgb8, not bgr8.
# This mapping is the SSOT for every consumer that must publish or decode
# what the real driver would have published.
_DRIVER_PIXEL_FORMAT_ENCODINGS: dict[str, str] = {
    "mjpeg2rgb": "rgb8",
}

# ROS Image encodings recognized by this codebase. Anything else fails
# closed: silently republishing an unrecognized encoding as bgr8 hides
# contract drift between the config and the consuming nodes.
_KNOWN_IMAGE_ENCODINGS: frozenset[str] = frozenset(
    {
        "mono8",
        "mono16",
        "bgr8",
        "rgb8",
        "bgr16",
        "rgb16",
        "bgra8",
        "rgba8",
        "bgra16",
        "rgba16",
        "yuv422",
        "32fc1",
    }
)


def resolve_image_encoding(encoding: str) -> str:
    """Resolve a contract image encoding to the ROS Image encoding on the wire.

    Driver pixel-format names (``mjpeg2rgb``) map to the encoding the real
    driver publishes. Unknown values raise ``ValueError`` so misconfigured
    contracts fail at launch instead of silently publishing a different
    pixel layout than every consumer expects.

    Args:
        encoding: Raw encoding or driver pixel format from the contract.

    Returns:
        The lowercase ROS Image encoding.

    Raises:
        ValueError: If the encoding is neither a known ROS Image encoding
            nor a mapped driver pixel format.
    """
    normalized = str(encoding).strip().lower()
    if not normalized:
        raise ValueError("image encoding must not be empty")
    resolved = _DRIVER_PIXEL_FORMAT_ENCODINGS.get(normalized, normalized)
    if resolved not in _KNOWN_IMAGE_ENCODINGS:
        raise ValueError(f"unknown image encoding {encoding!r}")
    return resolved


def _num_channels_from_encoding(encoding: str) -> int:
    """Infer channel count from ROS image encoding."""
    enc = encoding.lower()
    if enc in ("mono8", "mono16"):
        return 1
    if enc in ("bgr8", "rgb8", "bgr16", "rgb16"):
        return 3
    if enc in ("bgra8", "rgba8", "bgra16", "rgba16"):
        return 4
    abstract_prefixes = ["8uc", "8sc", "16uc", "16sc", "32sc", "32fc", "64fc"]
    for prefix in abstract_prefixes:
        if enc.startswith(prefix):
            if len(enc) == len(prefix):
                return 1
            try:
                channel_str = enc[len(prefix) :]
                if channel_str.isdigit():
                    return int(channel_str)
            except (ValueError, IndexError):
                pass
    if enc == "yuv422":
        return 2
    return 3


def iter_specs(contract: Contract) -> Iterable[SpecView]:
    """Yield normalized runtime specs for observations and actions."""
    for o in contract.observations:
        resize = None
        default_enc = "bgr8"
        if "depth" in o.topic.lower() or "depth" in o.key.lower():
            default_enc = "32fc1"
        if o.image:
            r = o.image.get("resize")
            if r and len(r) == 2:
                resize = (int(r[0]), int(r[1]))
            if "encoding" in o.image:
                default_enc = str(o.image.get("encoding")).lower()
        channels = _num_channels_from_encoding(default_enc)
        if "depth" in o.topic.lower() or "depth" in o.key.lower():
            channels = 3
        if o.image and "channels" in o.image:
            channels = int(o.image["channels"])
        names = list((o.selector or {}).get("names", []))
        al = o.align or AlignSpec()
        yield SpecView(
            key=o.key,
            topic=o.topic,
            ros_type=o.type,
            is_action=False,
            names=names,
            image_resize=resize,
            image_encoding=default_enc,
            image_channels=channels,
            resample_policy=al.strategy,
            asof_tol_ms=int(al.tol_ms),
            max_age_ms=int(al.max_age_ms),
            stamp_src=al.stamp,
            clamp=None,
            safety_behavior=None,
            transport=o.transport,
        )
    for a in contract.actions:
        names = list((a.selector or {}).get("names", []))
        clamp: tuple[float, float] | None = None
        if a.from_tensor and "clamp" in a.from_tensor:
            lo, hi = a.from_tensor["clamp"]
            clamp = (float(lo), float(hi))
        yield SpecView(
            key=a.key,
            topic=a.publish_topic,
            ros_type=a.type,
            is_action=True,
            names=names,
            image_resize=None,
            image_encoding="bgr8",
            image_channels=3,
            resample_policy="hold",
            asof_tol_ms=0,
            max_age_ms=0,
            stamp_src=contract.timestamp_source,
            clamp=clamp,
            safety_behavior=(a.safety_behavior or "zeros").lower(),
            transport=None,
        )


def feature_from_spec(spec: SpecView, use_videos: bool) -> tuple[str, dict[str, Any], bool]:
    if spec.image_resize:
        h, w = int(spec.image_resize[0]), int(spec.image_resize[1])
        dtype = "video" if use_videos else "image"
        return (
            spec.key,
            {
                "dtype": dtype,
                "shape": (h, w, spec.image_channels),
                "names": ["height", "width", "channel"],
            },
            True,
        )
    if spec.ros_type == "sensor_msgs/msg/PointCloud2":
        return (
            spec.key,
            {"dtype": "pointcloud", "shape": None},  # sentinel, not into LeRobot features
            False,
        )
    if not spec.names:
        raise ValueError(f"{spec.key}: vector features must specify selector.names")
    return (
        spec.key,
        {"dtype": "float32", "shape": (len(spec.names),), "names": list(spec.names)},
        False,
    )


# ---------- Decoder/Encoder registries (Bridged to tensormsg) ----------

# Maintain compatibility with local DECODERS/ENCODERS dictionaries
DECODERS = tensormsg_registry.DECODER_REGISTRY
ENCODERS = tensormsg_registry.ENCODER_REGISTRY

# register_decoder and register_encoder are already imported from tensormsg


# ---------- Time helpers ----------


def stamp_from_header_ns(msg) -> int | None:
    try:
        st = msg.header.stamp
        ts_ns = int(st.sec) * 1_000_000_000 + int(st.nanosec)
        if ts_ns == 0:
            return None
        return ts_ns
    except (AttributeError, TypeError, ValueError):
        return None


# ---------- Decoders (Bridged) ----------


def decode_value(ros_type: str, msg, spec) -> Any:
    """Decode a ROS message using tensormsg through the rosetta API."""
    from tensormsg.converter import TensorMsgConverter

    return TensorMsgConverter.decode(msg, spec)


# ---------- Resampling ----------


def resample_hold(ts_ns: np.ndarray, vals: list[Any], ticks_ns: np.ndarray) -> list[Any]:
    out: list[Any] = []
    j, last = 0, None
    if len(ts_ns) > 0 and len(ticks_ns) > 0 and ticks_ns[0] < ts_ns[0]:
        last = vals[0]
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        if j < len(vals) and ts_ns[j] <= t:
            last = vals[j]
        out.append(last)
    return out


def resample_asof(ts_ns: np.ndarray, vals: list[Any], ticks_ns: np.ndarray, tol_ns: int) -> list[Any | None]:
    if tol_ns <= 0:
        return resample_hold(ts_ns, vals, ticks_ns)
    out: list[Any | None] = []
    j = 0
    for t in ticks_ns:
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= t:
            j += 1
        ok = j < len(vals) and ts_ns[j] <= t and (t - ts_ns[j]) <= tol_ns
        out.append(vals[j] if ok else None)
    return out


def resample_drop(ts_ns: np.ndarray, vals: list[Any], ticks_ns: np.ndarray, step_ns: int) -> list[Any | None]:
    out: list[Any | None] = []
    j, n = -1, len(ts_ns)
    for t in ticks_ns:
        while j + 1 < n and ts_ns[j + 1] <= t:
            j += 1
        out.append(vals[j] if (j >= 0 and ts_ns[j] > t - step_ns) else None)
    return out


def resample(
    policy: str,
    ts_ns: np.ndarray,
    vals: list[Any],
    ticks_ns: np.ndarray,
    step_ns: int,
    tol_ms: int,
) -> list[Any]:
    if policy == "drop":
        return resample_drop(ts_ns, vals, ticks_ns, step_ns)
    if policy == "asof":
        return resample_asof(ts_ns, vals, ticks_ns, max(0, int(tol_ms)) * 1_000_000)
    return resample_hold(ts_ns, vals, ticks_ns)


class StreamBuffer:
    """Ordered timestamped history shared by DDS and decoded video streams.

    Capture timestamps drive alignment while receive timestamps drive live-age
    checks. Out-of-order samples are inserted in capture-time order, and a new
    sample with an existing capture timestamp replaces the previous sample.
    """

    def __init__(
        self,
        policy: str,
        step_ns: int,
        tol_ns: int = 0,
        *,
        max_age_ns: int = 0,
        retention_ns: int = 0,
    ) -> None:
        self.policy = str(policy).lower()
        self.step_ns = int(step_ns)
        self.tol_ns = max(0, int(tol_ns))
        self.max_age_ns = max(0, int(max_age_ns))
        self.retention_ns = max(0, int(retention_ns))
        self.history: list[tuple[int, int, Any]] = []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.history)

    def reset(self) -> None:
        with self._lock:
            self.history.clear()

    def push(self, ts_ns: int, val: Any, *, receive_time_ns: int | None = None) -> bool:
        timestamp_ns = int(ts_ns)
        received_ns = timestamp_ns if receive_time_ns is None else int(receive_time_ns)
        if self.retention_ns > 0 and timestamp_ns > received_ns + self.retention_ns:
            return False

        with self._lock:
            timestamps = [item[0] for item in self.history]
            index = bisect_right(timestamps, timestamp_ns)
            item = (timestamp_ns, received_ns, val)
            if index > 0 and self.history[index - 1][0] == timestamp_ns:
                self.history[index - 1] = item
            else:
                self.history.insert(index, item)

            if self.retention_ns > 0:
                cutoff_ns = min(self.history[-1][0], received_ns) - self.retention_ns
                timestamps = [item[0] for item in self.history]
                del self.history[: bisect_left(timestamps, cutoff_ns)]
        return True

    def select(
        self,
        tick_ns: int,
        *,
        now_ns: int | None = None,
        check_live_age: bool = True,
    ) -> tuple[Any | None, dict[str, object] | None]:
        item, issue = self.select_entry(tick_ns, now_ns=now_ns, check_live_age=check_live_age)
        return (item[2] if item is not None else None), issue

    def select_entry(
        self,
        tick_ns: int,
        *,
        now_ns: int | None = None,
        check_live_age: bool = True,
    ) -> tuple[tuple[int, int, Any] | None, dict[str, object] | None]:
        with self._lock:
            return self._select_entry_locked(tick_ns, now_ns=now_ns, check_live_age=check_live_age)

    def _select_entry_locked(
        self,
        tick_ns: int,
        *,
        now_ns: int | None,
        check_live_age: bool,
    ) -> tuple[tuple[int, int, Any] | None, dict[str, object] | None]:
        if not self.history:
            return None, {"reason": "missing", "sample_timestamp_ns": tick_ns}

        timestamps = [item[0] for item in self.history]
        index = bisect_right(timestamps, tick_ns) - 1
        if index < 0:
            timestamp_ns = self.history[0][0]
            return None, {
                "reason": "newer_than_request",
                "sample_timestamp_ns": tick_ns,
                "first_timestamp_ns": timestamp_ns,
                "age_ms": (tick_ns - timestamp_ns) / 1_000_000,
            }

        timestamp_ns, receive_time_ns, value = self.history[index]
        age_ns = tick_ns - timestamp_ns
        if self.policy == "asof" and self.tol_ns > 0 and age_ns > self.tol_ns:
            return None, self._stale_issue("asof", tick_ns, timestamp_ns, age_ns, self.tol_ns)
        if self.policy == "drop" and age_ns >= self.step_ns:
            return None, self._stale_issue("drop", tick_ns, timestamp_ns, age_ns, self.step_ns)
        if self.policy not in {"hold", "asof", "drop"}:
            return None, {"reason": "unsupported_alignment_strategy", "strategy": self.policy}

        live_age_ns = max(0, (tick_ns if now_ns is None else now_ns) - receive_time_ns)
        if value is None or (check_live_age and self.max_age_ns > 0 and live_age_ns > self.max_age_ns):
            return None, self._stale_issue("max_age", tick_ns, timestamp_ns, live_age_ns, self.max_age_ns)
        return (timestamp_ns, receive_time_ns, value), None

    def sample(self, tick_ns: int, *, now_ns: int | None = None) -> Any | None:
        value, _ = self.select(tick_ns, now_ns=now_ns)
        return value

    def first_entry(self) -> tuple[int, int, Any] | None:
        with self._lock:
            return self.history[0] if self.history else None

    def entries(self) -> tuple[tuple[int, int, Any], ...]:
        """Return a stable snapshot of the timestamped history."""
        with self._lock:
            return tuple(self.history)

    def first_entry_after(self, timestamp_ns: int) -> tuple[int, int, Any] | None:
        """Return the earliest entry strictly newer than ``timestamp_ns``."""
        with self._lock:
            timestamps = [item[0] for item in self.history]
            index = bisect_right(timestamps, int(timestamp_ns))
            return self.history[index] if index < len(self.history) else None

    @staticmethod
    def _stale_issue(
        constraint: str,
        tick_ns: int,
        timestamp_ns: int,
        age_ns: int,
        tolerance_ns: int,
    ) -> dict[str, object]:
        return {
            "reason": "stale",
            "constraint": constraint,
            "sample_timestamp_ns": tick_ns,
            "last_timestamp_ns": timestamp_ns,
            "age_ms": age_ns / 1_000_000,
            "tolerance_ms": tolerance_ns / 1_000_000,
        }


# ---------- Encoders (Bridged) ----------


def encode_value(
    ros_type: str,
    names: list[str],
    action_vec: Sequence[float],
    clamp: tuple[float, float] | None = None,
):
    """Encode an action vector using tensormsg."""
    from tensormsg.converter import TensorMsgConverter

    return TensorMsgConverter.encode(ros_type, action_vec, names, clamp)


# ---------- QoS utilities ----------


def qos_profile_from_dict(d: dict[str, Any] | None) -> QoSProfile | None:
    if not d:
        return None
    rel = str(d.get("reliability", "reliable")).lower()
    hist = str(d.get("history", "keep_last")).lower()
    dur = str(d.get("durability", "volatile")).lower()
    depth = int(d.get("depth", 10))
    return QoSProfile(
        reliability=(ReliabilityPolicy.BEST_EFFORT if rel == "best_effort" else ReliabilityPolicy.RELIABLE),
        history=(HistoryPolicy.KEEP_ALL if hist == "keep_all" else HistoryPolicy.KEEP_LAST),
        depth=depth,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if dur == "transient_local" else DurabilityPolicy.VOLATILE),
    )


# ---------- Data processing utilities ----------


def zero_pad(feature_meta: dict[str, Any]) -> Any:
    dtype = feature_meta["dtype"]
    shape = tuple(feature_meta.get("shape") or ())
    if dtype in ("video", "image"):
        return np.zeros(shape, dtype=np.float32)
    if dtype == "float32":
        return np.zeros(shape, dtype=np.float32)
    if dtype == "float64":
        return np.zeros(shape, dtype=np.float64)
    if dtype == "string":
        return ""
    return None


def contract_fingerprint(contract) -> str:
    if hasattr(contract, "__dataclass_fields__"):
        contract_dict = asdict(contract)
    else:
        contract_dict = contract
    fingerprint_data = {
        "name": contract_dict.get("name"),
        "observations": [],
        "actions": [],
    }
    for obs in contract_dict.get("observations", []):
        transport = obs.get("transport")
        if transport is None:
            transport = observation_transport_to_dict(effective_observation_transport(None))
        elif transport.get("mode", "dds") == "dds":
            transport = {"mode": "dds"}
        else:
            transport = observation_transport_to_dict(parse_observation_transport(transport))
        fingerprint_data["observations"].append(
            {
                "key": obs.get("key"),
                "topic": obs.get("topic"),
                "type": obs.get("type"),
                "selector": obs.get("selector"),
                "image": obs.get("image"),
                "align": obs.get("align"),
                "transport": transport,
            }
        )
    for act in contract_dict.get("actions", []):
        fingerprint_data["actions"].append(
            {
                "key": act.get("key"),
                "publish_topic": act.get("publish_topic"),
                "type": act.get("type"),
                "selector": act.get("selector"),
            }
        )
    json_str = json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]

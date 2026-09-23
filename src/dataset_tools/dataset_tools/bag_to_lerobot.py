#!/usr/bin/env python3
"""
ROS 2 bag → LeRobot v3.0 exporter.

Overview
--------
This script converts one or more ROS 2 bags into a LeRobot v3 dataset using
the *same* contract-aware processing utilities used for live
inference. That keeps train/serve paths aligned and minimizes skew.

The conversion pipeline:

1) Preflight all selected episodes using their dataset-embedded contract and conversion metadata
2) Scan a bag once; decode each contract topic using shared `decode_value`.
3) Select timestamps per a policy (`contract` / `bag` / `header`).
4) Resample each stream at the contract rate and assemble frames.
5) Coerce/resize images with the shared helpers and write to LeRobot.

Dependencies
------------
Shared modules (keep it unified with live inference):

- `robot_config.contract_utils`:
    `contract_from_dict`, `iter_specs`, `feature_from_spec`

Command-line usage
------------------
Convert a single bag:

    $ python bag_to_lerobot.py \\
        --bag /path/to/bag_dir \\
        --out /path/to/out_root

Convert multiple bags:

    $ python bag_to_lerobot.py \\
        --bags /bag/epi1 /bag/epi2 \\
        --out /path/to/out_root

Options of note:

The removed `--robot-config` option is rejected; legacy datasets without snapshots are unsupported.

- `--timestamp {contract,bag,header}`
    How to pick per-message timestamps before resampling:
    * contract: per-spec `stamp_src` (default)
    * bag:      use the bag receive time
    * header:   prefer `msg.header.stamp` with bag time as fallback

- `--no-videos`
    Store PNG images instead of H.264/MP4 videos.

Outputs
-------
A LeRobot v3 dataset with:

- `videos/<image_key>/chunk-*/file-*.mp4`  (or `images/*/*.png` if `--no-videos`)
- `data/chunk-*/file-*.parquet`
- `meta/info.json`, `meta/tasks.parquet`, `meta/stats.json`
- `meta/episodes/*/*.parquet`

Notes
-----
- Image coercion uses shared helpers to consistently handle grayscale/alpha,
  float ranges, and nearest-neighbor resize.
- Feature dicts are built directly from `feature_from_spec()` so train-time and
  serve-time shapes match exactly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import rosbag2_py
import yaml

# ---- LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Import decoders to register them
import tensormsg.converter  # noqa: F401

# ---- Shared core (ONLY these two)
from robot_config.contract_utils import (
    Contract,
    contract_fingerprint,
    decode_value,
    feature_from_spec,
    iter_specs,
    resample,
    stamp_from_header_ns,
)
from robot_config.contract_utils import (
    zero_pad as make_zero_pad,
)
from robot_config.utils import normalize_lerobot_norm_mode
from robot_runtime.model_metadata import build_joint_conversion_table_from_model, validate_public_conversion_metadata

# ---------------------------------------------------------------------------


@dataclass
class _Stream:
    """Decoded per-topic stream buffers accumulated from a bag scan.

    Attributes
    ----------
    spec : Any
        The `SpecView` for this stream (observation or action).
    ros_type : str
        Fully-qualified ROS message type string for deserialization.
    ts : list[int]
        Per-message timestamps in nanoseconds (selected by policy).
    val : list[Any]
        Decoded values in contract-native form (e.g., HWC arrays for images).
    """

    spec: Any
    ros_type: str
    ts: list[int]
    val: list[Any]


# ---------------------------------------------------------------------------
# Input adapters
# ---------------------------------------------------------------------------


@dataclass
class VideoFrameEntry:
    """One decoded video frame with its capture timestamp and integrity metadata.

    Attributes
    ----------
    timestamp_ns : int
        Capture timestamp in nanoseconds (RTP-mapped for Annex-B sources).
    image : Any
        Decoded frame as an HWC uint8 numpy array (RGB).
    keyframe : bool
        True when this frame is an IDR/keyframe.
    lost_packets : int
        RTP packets lost immediately before this frame (0 when clean).
    """

    timestamp_ns: int
    image: Any
    keyframe: bool = False
    lost_packets: int = 0


@dataclass
class IntegrityReport:
    """Per-episode integrity summary propagated into LeRobot ``info.json``.

    Attributes
    ----------
    clean : bool
        True when no frame gaps or dropped frames were recorded.
    frame_gaps : list[dict[str, Any]]
        One entry per affected frame: ``frame_index``, ``lost_packets``, ``reason``.
    """

    clean: bool = True
    frame_gaps: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render as the ``integrity`` block written into dataset ``info.json``."""
        payload: dict[str, Any] = {"clean": self.clean}
        if self.frame_gaps:
            payload["frame_gaps"] = self.frame_gaps
        return payload


def _merge_integrity_report(
    dataset_info: dict[str, Any],
    episode_index: int,
    obs_key: str,
    report: IntegrityReport,
) -> None:
    """Merge one Annex-B stream report into dataset-level integrity metadata."""
    integrity = dataset_info.setdefault("integrity", {"clean": True})
    if report.clean:
        return

    integrity["clean"] = False
    frame_gaps = integrity.setdefault("frame_gaps", [])
    for gap in report.frame_gaps or []:
        frame_gaps.append(
            {
                "episode_index": episode_index,
                "observation_key": obs_key,
                **gap,
            }
        )


def _persist_custom_info(
    info_path: Path,
    custom_info: dict[str, Any],
) -> None:
    """Merge project-specific metadata into LeRobot's typed info output."""
    with info_path.open("r", encoding="utf-8") as info_file:
        serialized_info = json.load(info_file)
    serialized_info.update(custom_info)
    with info_path.open("w", encoding="utf-8") as info_file:
        json.dump(serialized_info, info_file, indent=4, ensure_ascii=False)


class VideoInputAdapter(Protocol):
    """Read decoded video frames for one observation key from an episode directory.

    Implementations supply visual observations from a specific on-disk format.
    Non-visual observations (action, state, task) always come from the rosbag and
    are not routed through this abstraction.

    ``AnnexBInputAdapter`` implements this for ``.h264`` + ``.h264.json`` pairs
    produced by cross-device RTP recording. DDS recordings keep reading images
    from the rosbag directly, so they need no adapter.
    """

    def list_observations(self) -> list[str]:
        """Return observation keys this adapter can supply."""
        ...

    def read_frames(self, obs_key: str) -> list[VideoFrameEntry]:
        """Return all frames for ``obs_key`` in capture-timestamp order."""
        ...

    def integrity_report(self, obs_key: str) -> IntegrityReport:
        """Return the integrity summary recorded for ``obs_key``."""
        ...

    def close(self) -> None:
        """Release any open file handles or decoders."""
        ...


def discover_video_adapters(episode_dir: Path) -> dict[str, VideoInputAdapter]:
    """Detect recorded video streams in ``episode_dir`` and build their adapters.

    Detection is filesystem-based rather than contract-based, so it reflects what
    was actually recorded even when a run ends early. An episode with no ``.h264``
    files yields an empty mapping, which keeps DDS-only conversions on the
    original rosbag path.

    Parameters
    ----------
    episode_dir : Path
        Episode directory to scan (contains the rosbag plus any video streams).

    Returns
    -------
    dict[str, VideoInputAdapter]
        Mapping from observation key to the adapter serving it.
    """
    streams = sorted(episode_dir.glob("*.h264"))
    if not streams:
        return {}

    from dataset_tools.annex_b_input_adapter import AnnexBInputAdapter

    adapters: dict[str, VideoInputAdapter] = {}
    try:
        for stream_path in streams:
            obs_key = stream_path.stem
            if obs_key in adapters:
                raise ValueError(f"Duplicate external video observation: {obs_key}")
            adapters[obs_key] = AnnexBInputAdapter(stream_path)
    except Exception:
        for adapter in adapters.values():
            adapter.close()
        raise
    return adapters


# ---------------------------------------------------------------------------


def _read_yaml(p: Path) -> dict[str, Any]:
    """Read a YAML file if it exists; return {} on absence/parse failures."""
    if not p.exists():
        print(f"[WARN] {p} does not exist")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_metadata_for_bag(bag_dir: Path) -> dict[str, Any]:
    """Load dataset metadata when the bag lives under <dataset_root>/episodes/."""
    if bag_dir.parent.name != "episodes":
        return {}
    dataset_meta = bag_dir.parent.parent / "dataset.yaml"
    if not dataset_meta.exists():
        return {}
    return _read_yaml(dataset_meta)


def _lerobot_metadata_entry(
    dataset_meta: dict[str, Any],
    bag_info: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve the active LeRobot conversion metadata for one bag."""
    if not isinstance(dataset_meta, dict):
        return "", {}

    lerobot_meta = dataset_meta.get("lerobot")
    if not isinstance(lerobot_meta, dict):
        return "", {}

    conversions = lerobot_meta.get("conversions")
    if not isinstance(conversions, dict):
        return "", {}

    custom_data = bag_info.get("custom_data")
    fingerprint = ""
    if isinstance(custom_data, dict):
        fingerprint = custom_data.get("ibrobot.lerobot_conversion_fingerprint", "")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("recording is missing its per-episode conversion fingerprint")

    conversion_meta = conversions.get(fingerprint)
    if not isinstance(conversion_meta, dict):
        raise ValueError(f"recorded conversion fingerprint is not present: {fingerprint}")
    return fingerprint, conversion_meta


def _build_feature_conversion_table(
    feature_names: list[str],
    conversion_meta: dict[str, Any],
    feature_kind: str = "",
) -> list[tuple[float, float, float, float]]:
    """Build a per-feature conversion table in the feature's declared order.

    Hardware-only contracts use semantic feature names rather than calibration keys.
    The suffix of ``position.<joint>`` / ``velocity.<joint>`` is the joint name itself,
    and ``action.<i>`` indexes the arm joints in declared order, because the ``action``
    feature is the concatenation of the arm, gripper and base specs. The authoritative
    joint list comes from the contract, so no index range is assumed here: LeKiWi merely
    happens to name its joints "1".."6".

    Velocity selectors and declared base action slots keep native values. Missing
    position/action mappings fail validation instead of silently passing through.
    """
    ordered_names = [str(name) for name in feature_names]
    if not ordered_names:
        raise ValueError("state/action conversion requires feature names")

    arm_joints = [str(name) for name in (conversion_meta.get("joint_names") or [])]
    arm_joint_set = set(arm_joints)

    def conversion_names() -> list[str | None]:
        resolved: list[str | None] = []
        for name in ordered_names:
            field, _, suffix = name.partition(".")
            if field in ("position", "velocity"):
                # Only a position maps onto calibration. `range_min`/`range_max` describe
                # travel in ticks, so scaling a rad/s value by them is a dimensional error
                # that still reads as plausible data. Velocity passes through whether it
                # is a base wheel or an arm joint.
                resolved.append(suffix if field == "position" and suffix in arm_joint_set else None)
            elif feature_kind == "action" and field == "action" and suffix.isdigit():
                index = int(suffix)
                resolved.append(arm_joints[index] if index < len(arm_joints) else None)
            else:
                resolved.append(name)
        return resolved

    resolved_names = conversion_names()

    def build_in_feature_order(build: Any) -> list[tuple[float, float, float, float]]:
        """Build from the public snapshot, then restore native velocity slots."""
        mapped = [name for name in resolved_names if name is not None]
        mapped_table = list(build(mapped)) if mapped else []
        if len(mapped_table) != len(mapped):
            raise ValueError(f"Conversion table has {len(mapped_table)} rows for {len(mapped)} calibrated joints")
        rows = iter(mapped_table)
        return [(0.0, 1.0, 1.0, 0.0) if name is None else next(rows) for name in resolved_names]

    if not isinstance(conversion_meta, dict) or "norm_mode" not in conversion_meta:
        raise ValueError("conversion metadata requires explicit norm_mode")
    norm_mode = normalize_lerobot_norm_mode(conversion_meta["norm_mode"])
    if "description" in conversion_meta or norm_mode != "none":
        validate_public_conversion_metadata(conversion_meta)
    if norm_mode == "none":
        return []
    feature_map = conversion_meta["feature_names"]
    if feature_kind == "state":
        recorded = list(feature_map.get("observation.state") or [])
        requested = [name for name in resolved_names if name is not None]
        if requested != recorded:
            raise ValueError("recorded public conversion feature order conflicts with requested feature order")
    base_joints = conversion_meta["description"]["model"].get("joint_groups", {}).get("base", [])
    for name, resolved in zip(ordered_names, resolved_names, strict=True):
        if feature_kind == "action" and name.startswith("action.") and resolved is None:
            suffix = name.partition(".")[2]
            if not suffix.isdigit() or int(suffix) >= len(arm_joints) + len(set(base_joints) - arm_joint_set):
                raise ValueError(f"Feature {name!r} has no calibration mapping")
        if name.startswith("position.") and resolved is None:
            raise ValueError(f"Feature {name!r} has no calibration mapping")
    return build_in_feature_order(
        lambda names: build_joint_conversion_table_from_model(conversion_meta["description"]["model"], names, norm_mode)
    )


def _rad_to_lerobot(values: np.ndarray, table: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Convert a flat radian vector into LeRobot units using a conversion table."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if not table:
        return arr

    out = arr.copy()
    for i, (rad_min, rad_max, span, offset) in enumerate(table):
        if i >= out.shape[0]:
            break
        out[i] = (arr[i] - rad_min) / (rad_max - rad_min) * span + offset
    return out


def _clean_float_array(
    values: Any,
    dtype: Any = np.float32,
    *,
    feature_name: str = "",
) -> np.ndarray:
    """Convert a flat numeric vector and replace non-finite values with zeros.

    When non-finite values are detected, a warning is logged for features
    other than ``observation.current`` (where NaN indicates expected missing
    data from older recordings).
    """
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    mask = ~np.isfinite(arr)
    if mask.any():
        if feature_name and feature_name != "observation.current":
            logging.warning(
                f"{feature_name}: {int(mask.sum())}/{len(arr)} values "
                f"are non-finite (NaN/Inf), replaced with 0.0. "
                f"This usually indicates a joint-name mismatch."
            )
        arr[mask] = 0.0
    return arr


def _image_to_hwc(
    values: Any,
    feature_shape: tuple[int, ...] | list[int],
    *,
    feature_name: str = "",
) -> np.ndarray:
    """Normalize decoded image arrays to the HWC layout expected by the writer."""
    arr = np.asarray(values)
    expected = tuple(int(dim) for dim in feature_shape)
    if arr.shape == expected:
        return np.ascontiguousarray(arr)
    if len(expected) == 3 and arr.shape == (expected[2], expected[0], expected[1]):
        return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))
    label = f" for {feature_name}" if feature_name else ""
    chw_shape = (expected[2], expected[0], expected[1])
    raise ValueError(f"Image shape{label} must be HWC {expected} or CHW {chw_shape}, got {arr.shape}")


def _dataset_feature_names_for_spec(spec: Any) -> list[str]:
    """Return metadata names for a decoded contract spec."""
    return [str(name) for name in getattr(spec, "names", [])]


def _topic_type_map(reader: rosbag2_py.SequentialReader) -> dict[str, str]:
    """Build a `{topic: type}` map from a rosbag2 reader."""
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def _resolve_video_codec(requested_codec: str) -> str:
    """Resolve a playback-friendly video codec supported by the local PyAV build."""
    if requested_codec != "auto":
        return requested_codec

    try:
        import av
    except Exception:
        return "libsvtav1"

    for codec_name in ("h264", "libsvtav1"):
        try:
            if av.codec.Codec(codec_name, "w").is_encoder:
                return codec_name
        except Exception:
            continue
    return "libsvtav1"


def _estimate_stream_rate_hz(ts_ns: list[int]) -> float:
    """Estimate stream frequency from monotonically increasing nanosecond stamps."""
    if len(ts_ns) < 2:
        return 0.0

    arr = np.asarray(ts_ns, dtype=np.int64)
    span_ns = int(arr[-1] - arr[0])
    if span_ns <= 0:
        return 0.0
    return float((len(arr) - 1) * 1e9 / span_ns)


def _selected_indices_for_ticks(
    policy: str,
    ts_ns: np.ndarray,
    ticks_ns: np.ndarray,
    step_ns: int,
    tol_ns: int,
) -> np.ndarray:
    """Return the source-message index selected for each output tick."""
    out = np.full((len(ticks_ns),), -1, dtype=np.int64)
    if len(ts_ns) == 0 or len(ticks_ns) == 0:
        return out

    if policy == "drop":
        j, n = -1, len(ts_ns)
        for i, tick in enumerate(ticks_ns):
            while j + 1 < n and ts_ns[j + 1] <= tick:
                j += 1
            if j >= 0 and ts_ns[j] > tick - step_ns:
                out[i] = j
        return out

    if policy == "asof":
        if tol_ns <= 0:
            policy = "hold"
        else:
            j = 0
            for i, tick in enumerate(ticks_ns):
                while j + 1 < len(ts_ns) and ts_ns[j + 1] <= tick:
                    j += 1
                if ts_ns[j] <= tick and (tick - ts_ns[j]) <= tol_ns:
                    out[i] = j
            return out

    j = 0
    last_idx = -1
    if ticks_ns[0] < ts_ns[0]:
        last_idx = 0
    for i, tick in enumerate(ticks_ns):
        while j + 1 < len(ts_ns) and ts_ns[j + 1] <= tick:
            j += 1
        if ts_ns[j] <= tick:
            last_idx = j
        out[i] = last_idx
    return out


def _log_image_stream_diagnostics(
    streams: dict[str, _Stream],
    ticks_ns: np.ndarray,
    step_ns: int,
    target_fps: int,
    video_sources: dict[str, str] | None = None,
) -> None:
    """Log observed image rates and repeated-frame ratios after resampling."""
    video_sources = video_sources or {}
    for key, st in streams.items():
        if st.spec.image_resize is None or not st.ts:
            continue

        ts = np.asarray(st.ts, dtype=np.int64)
        rate_hz = _estimate_stream_rate_hz(st.ts)
        selected = _selected_indices_for_ticks(
            policy=st.spec.resample_policy,
            ts_ns=ts,
            ticks_ns=ticks_ns,
            step_ns=step_ns,
            tol_ns=max(0, int(st.spec.asof_tol_ms)) * 1_000_000,
        )
        used_frames = int(np.count_nonzero(selected >= 0))
        unique_frames = int(np.unique(selected[selected >= 0]).size) if used_frames else 0
        repeated_ratio = 1.0 - (unique_frames / len(ticks_ns)) if len(ticks_ns) else 0.0
        valid_ticks = np.flatnonzero(selected >= 0)
        max_alignment_error_ms = 0.0
        if len(valid_ticks):
            selected_ts = ts[selected[valid_ticks]]
            max_alignment_error_ms = float(np.max(np.abs(ticks_ns[valid_ticks] - selected_ts)) / 1e6)
        print(
            f"  [diag] {key}: source={video_sources.get(key, 'rosbag')}, "
            f"source_frames={len(st.ts)} (~{rate_hz:.1f} Hz), "
            f"unique_output_frames={unique_frames}/{len(ticks_ns)}, "
            f"repeated_frame_ratio={repeated_ratio:.1%}, max_alignment_error={max_alignment_error_ms:.3f} ms"
        )
        if rate_hz > 0 and rate_hz < target_fps * 0.9:
            print(
                f"  [warn] {key}: source image rate (~{rate_hz:.1f} Hz) is below "
                f"dataset rate ({target_fps} Hz); direct playback will repeat frames."
            )
        elif repeated_ratio > 0.15:
            print(
                f"  [warn] {key}: {repeated_ratio:.1%} of output ticks reuse an older frame. "
                f"If playback still looks choppy, inspect camera timestamp jitter or try a lower contract rate."
            )


def _plan_streams(
    specs: Iterable[Any],
    tmap: dict[str, str],
    external_video_keys: set[str] | None = None,
) -> tuple[dict[str, _Stream], dict[str, list[str]]]:
    """Plan `_Stream` buffers for contract specs and build a topic dispatch index.

    Parameters
    ----------
    specs : Iterable[Any]
        Iterable of `SpecView` objects derived from the contract.
    tmap : dict[str, str]
        Map from topic name to ROS type in the bag.

    Returns
    -------
    streams : dict[str, _Stream]
        Mapping from contract key to `_Stream` state.
    by_topic : dict[str, list[str]]
        Mapping from topic name to a list of contract keys using it.

    Raises
    ------
    RuntimeError
        If none of the contract topics exist in the bag.
    """
    external_video_keys = external_video_keys or set()
    streams: dict[str, _Stream] = {}
    by_topic: dict[str, list[str]] = {}
    for sv in specs:
        uses_external_video = sv.key in external_video_keys
        if uses_external_video and sv.image_resize is None:
            raise ValueError(f"External video input '{sv.key}' does not map to an image observation in the contract")
        if sv.topic not in tmap and not uses_external_video:
            # Derive a human-readable kind for logging without assuming SpecView internals.
            if hasattr(sv, "is_action") and sv.is_action:
                kind = "action"
            elif str(getattr(sv, "key", "")).startswith("task."):
                kind = "task"
            else:
                kind = "observation"
            print(f"[WARN] Missing {kind} '{getattr(sv, 'key', '?')}' topic in bag: {sv.topic}")
            continue
        rt = sv.ros_type or tmap.get(sv.topic, "")

        # Create unique key for multiple observation.state specs and action specs
        if sv.key == "observation.state":
            # Remove leading underscore from topic replacement
            topic_suffix = sv.topic.replace("/", "_").lstrip("_")
            unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
        elif sv.is_action:
            # For action specs, we need to check if there are multiple specs with the same key
            # This will be handled later in the consolidation logic
            topic_suffix = sv.topic.replace("/", "_").lstrip("_")
            unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
        else:
            unique_key = sv.key

        streams[unique_key] = _Stream(spec=sv, ros_type=rt, ts=[], val=[])
        if not uses_external_video:
            by_topic.setdefault(sv.topic, []).append(unique_key)
    if not streams:
        raise RuntimeError("No contract topics found in bag.")
    return streams, by_topic


# ---------------------------------------------------------------------------


def _print_contract_streams(contract: Contract) -> None:
    print(f"[bag_to_lerobot]   Observations: {len(contract.observations)}")
    for obs in contract.observations:
        print(f"[bag_to_lerobot]     - {obs.key} <- {obs.topic}")
    print(f"[bag_to_lerobot]   Actions: {len(contract.actions)}")
    for act in contract.actions:
        print(f"[bag_to_lerobot]     - {act.key} -> {act.publish_topic}")


def _contract_from_dataset_metadata(dataset_meta: dict[str, Any]) -> Contract:
    """Reconstruct only complete recorded contracts; never consult source configuration."""
    from robot_config.contract_utils import contract_from_dict

    if not isinstance(dataset_meta, dict):
        raise ValueError("dataset metadata must be a mapping")
    data = dataset_meta.get("contract")
    if not isinstance(data, dict) or not data:
        raise ValueError("dataset has no valid contract snapshot")
    for field in ("name", "rate_hz", "timestamp_source", "observations", "actions", "tasks"):
        if field not in data:
            raise ValueError(f"contract snapshot requires {field}")
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ValueError("contract snapshot name must be a non-empty string")
    if data["timestamp_source"] not in ("receive", "header"):
        raise ValueError("contract snapshot timestamp_source must be receive or header")
    rate = data["rate_hz"]
    if (
        isinstance(rate, bool)
        or not isinstance(rate, int | float)
        or not math.isfinite(rate)
        or rate < 1
        or not float(rate).is_integer()
    ):
        raise ValueError("contract snapshot rate_hz must be a finite positive integer")
    for field in ("observations", "actions", "tasks"):
        if not isinstance(data[field], list) or any(not isinstance(item, dict) for item in data[field]):
            raise ValueError(f"contract snapshot {field} must be a list of mappings")
    if not data["observations"] and not data["actions"]:
        raise ValueError("contract snapshot requires observations or actions")
    try:
        contract = contract_from_dict(data)
        for spec in iter_specs(contract):
            if any(not isinstance(value, str) or not value.strip() for value in (spec.key, spec.topic, spec.ros_type)):
                raise ValueError("stream key, topic and type must be non-empty strings")
            if spec.stamp_src not in ("header", "receive") or spec.resample_policy not in ("hold", "asof", "drop"):
                raise ValueError("invalid stream timestamp/alignment policy")
            feature_from_spec(spec, False)
        actual = contract_fingerprint(contract)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid contract snapshot: {exc}") from exc
    recorded = dataset_meta.get("contract_fingerprint")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("dataset requires contract_fingerprint")
    if actual != recorded:
        raise ValueError(f"recorded contract snapshot does not match its fingerprint: {recorded} != {actual}")
    return contract


def _preflight_bags(bag_dirs: list[Path]) -> tuple[Contract, list[dict[str, Any]]]:
    """Validate every selected episode and build conversion tables before creating output.

    The existing fingerprint excludes rate. Comparing snapshots detects rate differences
    across dataset roots, but cannot recover historical per-episode rate changes within
    one dataset whose only snapshot has been overwritten.
    """
    from robot_config.contract_utils import contract_to_dict

    if not bag_dirs:
        raise ValueError("No bags selected")
    selected_contract = None
    selected_snapshot = None
    episodes = []
    for bag_dir in bag_dirs:
        try:
            dataset_meta = _dataset_metadata_for_bag(bag_dir)
            contract = _contract_from_dataset_metadata(dataset_meta)
            snapshot = contract_to_dict(contract)
            if selected_snapshot is not None and snapshot != selected_snapshot:
                raise ValueError("selected episodes have incompatible contract snapshots")
            selected_contract, selected_snapshot = contract, snapshot
            meta = _read_yaml(bag_dir / "metadata.yaml")
            info = meta.get("rosbag2_bagfile_information")
            if not isinstance(info, dict):
                raise ValueError("episode requires rosbag2_bagfile_information")
            custom = info.get("custom_data")
            if not isinstance(custom, dict) or custom.get("ibrobot.contract_fingerprint") != contract_fingerprint(
                contract
            ):
                raise ValueError("episode contract_fingerprint does not match dataset contract snapshot")
            feature_names: dict[str, list[str]] = {}
            for spec in iter_specs(contract):
                if spec.is_action or spec.key == "observation.state":
                    feature_names.setdefault(spec.key, []).extend(spec.names)
            tables = {}
            if feature_names:
                fingerprint, conversion = _lerobot_metadata_entry(dataset_meta, info)
                if not conversion:
                    raise ValueError("episode requires conversion metadata")
                if (
                    "norm_mode" not in conversion
                    or not isinstance(conversion["norm_mode"], str)
                    or not conversion["norm_mode"].strip()
                ):
                    raise ValueError("conversion metadata requires explicit norm_mode")
                mode = normalize_lerobot_norm_mode(conversion["norm_mode"])
                if mode != "none" or "description" in conversion:
                    validate_public_conversion_metadata(conversion)
                    if conversion["conversion_fingerprint"] != fingerprint:
                        raise ValueError("conversion fingerprint does not match metadata")
                for key, names in feature_names.items():
                    tables[key] = _build_feature_conversion_table(
                        names, conversion, feature_kind="state" if key == "observation.state" else "action"
                    )
            episodes.append({"dataset_meta": dataset_meta, "info": info, "tables": tables})
        except (KeyError, TypeError, ValueError, AttributeError, OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Preflight failed for {bag_dir}: {exc}") from exc
    return selected_contract, episodes


def export_bags_to_lerobot(
    bag_dirs: list[Path],
    out_root: Path = Path("output"),
    repo_id: str = "rosbag_v30",
    use_videos: bool = True,
    image_writer_threads: int = 4,
    image_writer_processes: int = 0,
    chunk_size: int = 1000,
    data_mb: int = 100,
    video_mb: int = 500,
    timestamp_source: str = "contract",
    video_codec: str = "auto",
) -> None:
    """Convert bag directories into a LeRobot v3 dataset under `out_root`.

    Requires dataset-embedded contract and per-episode conversion metadata.

    Parameters
    ----------
    bag_dirs : list[pathlib.Path]
        One or more bag directories (episodes) to convert.
    out_root : pathlib.Path
        Root directory where the LeRobot dataset will be created/updated.
    repo_id : str, default "rosbag_v30"
        Dataset repo_id metadata stored by LeRobot.
    use_videos : bool, default True
        If True, store videos; otherwise store per-frame PNG images.
    image_writer_threads : int, default 4
        Worker threads per process for image writing.
    image_writer_processes : int, default 0
    chunk_size : int, default 1000
        Max number of frames per Parquet/video chunk.
    data_mb : int, default 100
        Target data file size in MB per chunk.
    video_mb : int, default 500
        Target video file size in MB per chunk.
    timestamp_source : {"contract","receive","header"}, default "contract"
        Timestamp selection policy per decoded message.

    Raises
    ------
    ValueError
        If contract `rate_hz` is invalid (<= 0).
    RuntimeError
        If a bag contains no usable/decodable messages.
    """
    contract, preflight = _preflight_bags(bag_dirs)
    fps = int(contract.rate_hz)
    if fps <= 0:
        raise ValueError("Contract rate_hz must be > 0")
    resolved_video_codec = _resolve_video_codec(video_codec)
    if use_videos:
        os.environ["LEROBOT_VIDEO_VCODEC"] = resolved_video_codec
        print(f"[bag_to_lerobot] Using video codec: {resolved_video_codec}")
    step_ns = int(round(1e9 / fps))
    specs = list(iter_specs(contract))

    # Features (also detect first image key as anchor)
    features: dict[str, dict[str, Any]] = {}
    primary_image_key: str | None = None
    state_specs = []  # Track multiple observation.state specs
    action_specs_by_key: dict[str, list[Any]] = {}  # Track multiple action specs by key
    pc_keys: set = set()  # PointCloud2 keys routed to side-car (not into LeRobot features)

    for sv in specs:
        # Handle multiple observation.state specs
        if sv.key == "observation.state":
            state_specs.append(sv)
            # Don't add to features yet - we'll consolidate them
            continue

        # Handle action specs
        if sv.is_action:
            if sv.key not in action_specs_by_key:
                action_specs_by_key[sv.key] = []
            action_specs_by_key[sv.key].append(sv)
            # Don't add to features yet - we'll consolidate them
            continue

        # Process other specs normally
        k, ft, is_img = feature_from_spec(sv, use_videos)

        # Route PointCloud2 to side-car; keep out of LeRobot features
        if ft.get("dtype") == "pointcloud":
            pc_keys.add(k)
            continue

        # Ensure task.* specs are treated as per-frame strings even if the
        # underlying helper doesn't special-case them yet.
        if str(k).startswith("task."):  # TODO: why is this special-cased? Shouldn't this be handled in constract_utils?
            # Normalize to a simple scalar string field.
            features[k] = {"dtype": "string", "shape": [1]}
        else:
            # Special handling for depth images - they now have 3 channels
            if k.endswith(".depth") and ft["shape"][-1] == 1:
                # Update the shape to reflect 3 channels
                ft["shape"] = list(ft["shape"])
                ft["shape"][-1] = 3
            if k == "observation.current":
                ft["names"] = _dataset_feature_names_for_spec(sv)
            features[k] = ft
        if is_img and primary_image_key is None:
            primary_image_key = sv.key

    # Consolidate multiple observation.state specs into a single feature
    if state_specs:
        all_names = []
        total_shape = 0
        for sv in state_specs:
            all_names.extend(sv.names)
            total_shape += len(sv.names)

        features["observation.state"] = {"dtype": "float32", "shape": (total_shape,), "names": all_names}

    # Consolidate multiple action specs with the same key into a single feature
    for action_key, action_specs in action_specs_by_key.items():
        if len(action_specs) > 1:
            # Multiple specs with same key - consolidate them
            all_names = []
            total_shape = 0
            for sv in action_specs:
                all_names.extend(sv.names)
                total_shape += len(sv.names)

            features[action_key] = {"dtype": "float32", "shape": (total_shape,), "names": all_names}
        else:
            # Single spec - use it as-is
            sv = action_specs[0]
            k, ft, _ = feature_from_spec(sv, use_videos)
            features[k] = ft

    # Mark depth videos in features metadata before dataset creation
    for key, feature in features.items():
        if key.endswith(".depth") and feature.get("dtype") == "video":
            if "info" not in feature:
                feature["info"] = {}
            feature["info"]["video.is_depth_map"] = True

    # Dataset
    out_root_existed = out_root.exists()
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_root,
        robot_type=contract.robot_type,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,  # keep simple & predictable
        image_writer_threads=image_writer_threads,
        batch_encoding_size=1,
    )

    custom_info: dict[str, Any] = {}

    # Persist the contract fingerprint into info.json so training can validate & propagate it.
    # DatasetInfo is typed and intentionally rejects unknown fields, so project-specific
    # metadata is merged into the serialized info.json after LeRobot writes its metadata.
    try:
        fp = contract_fingerprint(contract)
        custom_info["ibrobot_fingerprint"] = fp
    except Exception:
        pass  # non-fatal; downstream will just skip the check
    custom_info["integrity"] = {"clean": True}
    ds.meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_mb,
        video_files_size_in_mb=video_mb,
    )

    # Precompute zero pads + shapes for fast frame assembly.
    zero_pad_map = {k: make_zero_pad(ft) for k, ft in features.items()}
    write_keys = [k for k, ft in features.items() if ft["dtype"] in ("video", "image", "float32", "float64", "string")]
    # Episodes
    converted_episodes = 0
    skipped_bags: list[tuple[Path, str]] = []
    for epi_idx, bag_dir in enumerate(bag_dirs):
        print(f"[Episode {epi_idx}] {bag_dir}")

        # Per-episode point cloud buffer (one entry per key)
        pc_buf: dict[str, dict[str, list]] = {k: {"xyz": [], "rgb": [], "ts": []} for k in pc_keys}

        video_adapters: dict[str, VideoInputAdapter] = {}
        video_sources: dict[str, str] = {}
        integrity_reports: dict[str, IntegrityReport] = {}
        try:
            video_adapters = discover_video_adapters(bag_dir)
            video_sources.update({key: "Annex-B" for key in video_adapters})
            episode = preflight[epi_idx]
            dataset_meta = episode["dataset_meta"]
            info = episode["info"]
            storage = info.get("storage_identifier") or "mcap"
            meta_dur_ns = int((info.get("duration") or {}).get("nanoseconds") or 0)

            # Operator prompt (if present). Accept either old/new keys gracefully.
            prompt = ""
            cd = info.get("custom_data")
            if isinstance(cd, dict):
                prompt = cd.get("lerobot.operator_prompt", prompt) or prompt
            if not prompt and isinstance(dataset_meta, dict):
                prompt = str(dataset_meta.get("default_task") or dataset_meta.get("task") or prompt)

            # Reader
            reader = rosbag2_py.SequentialReader()
            reader.open(
                rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage),
                rosbag2_py.ConverterOptions(
                    input_serialization_format="cdr",
                    output_serialization_format="cdr",
                ),
            )
        except Exception as e:
            for adapter in video_adapters.values():
                adapter.close()
            print(f"⚠️  Skipping bag {bag_dir} due to error: {e}")
            skipped_bags.append((bag_dir, str(e)))
            continue

        tmap = _topic_type_map(reader)
        print(f"tmap: {tmap}")

        # Plan once - handle multiple observation.state specs and action specs
        streams, by_topic = _plan_streams(specs, tmap, set(video_adapters))

        try:
            for obs_key, adapter in video_adapters.items():
                if obs_key not in streams:
                    raise ValueError(f"Annex-B observation '{obs_key}' is not defined by the robot contract")
                frames = adapter.read_frames(obs_key)
                streams[obs_key].ts.extend(frame.timestamp_ns for frame in frames)
                streams[obs_key].val.extend(frame.image for frame in frames)
                integrity_reports[obs_key] = adapter.integrity_report(obs_key)
                print(f"  [video] {obs_key}: source=Annex-B, decoded_frames={len(frames)}")
        finally:
            for adapter in video_adapters.values():
                adapter.close()

        # Create consolidated observation.state stream if we have multiple state specs
        if state_specs:
            # Find all observation.state streams
            state_streams = [k for k in streams if k == "observation.state"]
            if len(state_streams) > 1:
                # Create a consolidated stream that will concatenate the data
                # We'll handle this in the frame processing
                pass
        print(f"streams: {streams}")

        # Counters for light diagnostics
        decoded_msgs = sum(len(st.val) for st in streams.values())

        # Decode single pass
        while reader.has_next():
            topic, data, bag_ns = reader.read_next()
            if topic not in by_topic:
                continue
            for key in by_topic[topic]:
                st = streams[key]
                msg = deserialize_message(data, get_message(st.ros_type))
                sv = st.spec

                # Timestamp selection policy
                if timestamp_source == "receive":
                    ts_sel = int(bag_ns)
                elif timestamp_source == "header":
                    ts_sel = stamp_from_header_ns(msg) or int(bag_ns)
                else:  # 'contract' (per-spec stamp_src)
                    ts_sel = int(bag_ns)
                    if sv.stamp_src == "header":
                        hdr = stamp_from_header_ns(msg)
                        if hdr is not None:
                            ts_sel = int(hdr)

                val = decode_value(st.ros_type, msg, sv)

                if val is not None:
                    st.ts.append(ts_sel)
                    st.val.append(val)
                    decoded_msgs += 1

        if decoded_msgs == 0:
            raise RuntimeError(f"No usable messages in {bag_dir} (none decoded).")
        if video_adapters and (state_specs or action_specs_by_key):
            has_action = any(st.spec.is_action and st.ts for st in streams.values())
            has_state = any(st.spec.key == "observation.state" and st.ts for st in streams.values())
            if (action_specs_by_key and not has_action) or (state_specs and not has_state):
                raise RuntimeError(
                    f"action/state data required for LeRobot dataset: {bag_dir} "
                    f"(action={'present' if has_action else 'missing'}, state={'present' if has_state else 'missing'})"
                )

        # Choose anchor + duration
        valid_ts = [np.asarray(st.ts, dtype=np.int64) for st in streams.values() if st.ts]
        if not valid_ts:
            raise RuntimeError(f"No usable messages in {bag_dir} (no timestamps).")
        if primary_image_key and streams.get(primary_image_key) and streams[primary_image_key].ts:
            start_ns = int(np.asarray(streams[primary_image_key].ts, dtype=np.int64).min())
        else:
            start_ns = int(min(ts.min() for ts in valid_ts))

        ts_max = int(max(ts.max() for ts in valid_ts))
        observed_dur_ns = max(0, ts_max - start_ns)

        # Prefer observed duration unless bag metadata matches within ~2 ticks.
        if meta_dur_ns > 0 and abs(meta_dur_ns - observed_dur_ns) <= 2 * step_ns:
            dur_ns = meta_dur_ns
            print("Using duration from metadata")
        else:
            dur_ns = observed_dur_ns
            print("Metadata duration disagrees with observed duration. Using observed duration")

        # Ticks
        n_ticks = int(dur_ns // step_ns) + 1
        ticks_ns = start_ns + np.arange(n_ticks, dtype=np.int64) * step_ns

        _log_image_stream_diagnostics(
            streams=streams,
            ticks_ns=ticks_ns,
            step_ns=step_ns,
            target_fps=fps,
            video_sources=video_sources,
        )

        # Resample onto ticks
        resampled: dict[str, list[Any]] = {}
        for key, st in streams.items():
            if not st.ts:
                resampled[key] = [None] * n_ticks
                continue
            ts = np.asarray(st.ts, dtype=np.int64)
            pol = st.spec.resample_policy
            resampled[key] = resample(pol, ts, st.val, ticks_ns, step_ns, st.spec.asof_tol_ms)

        state_conversion_table = episode["tables"].get("observation.state", [])
        action_conversion_tables = episode["tables"]

        # Write frames
        try:
            for i in range(n_ticks):
                frame: dict[str, Any] = {}

                # Handle consolidated observation.state by concatenating multiple state streams first
                if "observation.state" in features and state_specs:
                    # Concatenate all observation.state values from different topics
                    state_values = []
                    for sv in state_specs:
                        topic_suffix = sv.topic.replace("/", "_").lstrip("_")
                        unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
                        stream_val = resampled.get(unique_key, [None] * n_ticks)[i]
                        if stream_val is not None:
                            val_array = _clean_float_array(stream_val, np.float32, feature_name="observation.state")
                            state_values.append(val_array)

                    if state_values:
                        # Concatenate all state values
                        concatenated_state = np.concatenate(state_values)
                        exp = int(features["observation.state"]["shape"][0])
                        if concatenated_state.shape[0] != exp:
                            fixed = np.zeros((exp,), dtype=np.float32)
                            fixed[: min(exp, concatenated_state.shape[0])] = concatenated_state[
                                : min(exp, concatenated_state.shape[0])
                            ]
                            concatenated_state = fixed
                        if state_conversion_table:
                            concatenated_state = _rad_to_lerobot(concatenated_state, state_conversion_table)
                        frame["observation.state"] = concatenated_state
                    else:
                        # Use zero padding if no state values available
                        frame["observation.state"] = zero_pad_map["observation.state"]

                # Handle consolidated action specs by concatenating multiple action streams
                for action_key, action_specs in action_specs_by_key.items():
                    if action_key in features:
                        # Concatenate all action values from different topics
                        action_values = []
                        for sv in action_specs:
                            topic_suffix = sv.topic.replace("/", "_").lstrip("_")
                            unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
                            stream_val = resampled.get(unique_key, [None] * n_ticks)[i]
                            if stream_val is not None:
                                val_array = _clean_float_array(stream_val, np.float32, feature_name=action_key)
                                action_values.append(val_array)

                        if action_values:
                            # Concatenate all action values
                            concatenated_action = np.concatenate(action_values)

                            # Pad or truncate to match feature shape if necessary
                            exp = int(features[action_key]["shape"][0])
                            if concatenated_action.shape[0] != exp:
                                fixed = np.zeros((exp,), dtype=np.float32)
                                fixed[: min(exp, concatenated_action.shape[0])] = concatenated_action[
                                    : min(exp, concatenated_action.shape[0])
                                ]
                                concatenated_action = fixed
                            conversion_table = action_conversion_tables.get(action_key, [])
                            if conversion_table:
                                concatenated_action = _rad_to_lerobot(concatenated_action, conversion_table)

                            frame[action_key] = concatenated_action
                        else:
                            # Use zero padding if no action values available
                            frame[action_key] = zero_pad_map[action_key]

                # Process all other features
                for name in write_keys:
                    # Skip observation.state as it's handled above
                    if name == "observation.state":
                        continue

                    # Skip actions as they're handled above
                    if name in action_specs_by_key:
                        continue
                    ft = features[name]
                    dtype = ft["dtype"]
                    val = resampled.get(name, [None] * n_ticks)[i]

                    if val is None:
                        frame[name] = zero_pad_map[name]
                        continue

                    if dtype in ("video", "image"):
                        arr = _image_to_hwc(val, ft["shape"], feature_name=name)
                        # Ensure deterministic storage; lerobot loaders will map back to float [0,1]
                        if arr.dtype != np.uint8:
                            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                        frame[name] = arr

                    elif dtype in ("float32", "float64"):
                        tgt_dt = np.float32 if dtype == "float32" else np.float64
                        arr = _clean_float_array(val, tgt_dt, feature_name=name)
                        exp = int(ft["shape"][0])
                        if arr.shape[0] != exp:
                            fixed = np.zeros((exp,), dtype=tgt_dt)
                            fixed[: min(exp, arr.shape[0])] = arr[: min(exp, arr.shape[0])]
                            arr = fixed
                        frame[name] = arr

                    elif dtype == "string":
                        frame[name] = str(val)

                    else:
                        # Fallback – should not happen with current features
                        frame[name] = val

                # Collect point cloud data into side-car buffers (not passed to ds.add_frame)
                for pc_key in pc_keys:
                    pc_val = resampled.get(pc_key, [None] * n_ticks)[i]
                    buf = pc_buf[pc_key]
                    if pc_val is not None and isinstance(pc_val, dict):
                        buf["xyz"].append(pc_val["xyz"])
                        buf["rgb"].append(pc_val["rgb"])
                    else:
                        buf["xyz"].append(np.zeros((0, 3), dtype=np.float32))
                        buf["rgb"].append(np.zeros((0, 3), dtype=np.uint8))
                    buf["ts"].append(int(ticks_ns[i]))

                # Episode-level operator prompt from bag metadata (kept for policy compatibility).
                # This is`` distinct from any per-frame task.* fields coming from ROS topics.
                # LeRobot requires 'task' field in every frame, so always set it (empty string if no prompt).
                frame["task"] = prompt if prompt else ""
                ds.add_frame(frame)
        except ValueError as exc:
            # Reader-layer failures above already drop a bag and carry on; a decode
            # failure here has to do the same or one bad frame aborts a run that has
            # already written good episodes. The half-filled buffer must go with it,
            # otherwise the next bag inherits this episode's frames.
            print(f"⚠️  Skipping bag {bag_dir} due to error: {exc}")
            skipped_bags.append((bag_dir, str(exc)))
            ds.clear_episode_buffer()
            continue

        output_episode_index = int(ds.meta.info.get("total_episodes", 0))
        for obs_key, report in integrity_reports.items():
            _merge_integrity_report(custom_info, output_episode_index, obs_key, report)
        ds.save_episode()
        converted_episodes += 1
        # save_episode() ends in LeRobot's write_info, so info.json is already on disk;
        # merge the project-specific fields into it by rewriting the file. Mutating
        # ds.meta.info here would not reach disk -- nothing writes it again afterwards.
        _persist_custom_info(ds.meta.root / "meta" / "info.json", custom_info)
        expected_duration_s = n_ticks / fps if fps > 0 else 0
        print(
            f"  → saved {n_ticks} frames @ {int(round(fps))} FPS "
            f"({expected_duration_s:.1f}s)  | decoded_msgs={decoded_msgs}"
        )
        # Warn when the majority of decoded messages are discarded during
        # resampling – usually means the recording was cut short by a crash.
        n_streams = len(streams)
        if n_streams > 0 and decoded_msgs > 0:
            avg_per_stream = decoded_msgs / n_streams
            if avg_per_stream > 0 and n_ticks < avg_per_stream * 0.5:
                print(
                    f"  ⚠️  Warning: only {n_ticks} frames kept from "
                    f"~{int(avg_per_stream)} msgs/stream. Possible causes:\n"
                    f"      - Recording was interrupted or the recorder crashed\n"
                    f"      - Bag metadata duration is inaccurate\n"
                    f"      Check the recording logs for errors."
                )

        # Write point cloud side-car (.npz CSR) for each key
        for pc_key in pc_keys:
            buf = pc_buf[pc_key]
            chunk_idx = epi_idx // chunk_size
            pc_chunk_dir = out_root / "pointclouds" / pc_key / f"chunk-{chunk_idx:03d}"
            pc_chunk_dir.mkdir(parents=True, exist_ok=True)

            xyz_cat = np.concatenate(buf["xyz"], axis=0) if buf["xyz"] else np.zeros((0, 3), dtype=np.float32)
            rgb_cat = np.concatenate(buf["rgb"], axis=0) if buf["rgb"] else np.zeros((0, 3), dtype=np.uint8)
            counts = np.array([len(a) for a in buf["xyz"]], dtype=np.int64)
            offsets = np.concatenate([[0], np.cumsum(counts)])

            np.savez_compressed(
                pc_chunk_dir / f"episode_{epi_idx:06d}.npz",
                xyz=xyz_cat,
                rgb=rgb_cat,
                offsets=offsets,
                timestamps_ns=np.array(buf["ts"], dtype=np.int64),
                episode_index=np.int32(epi_idx),
            )
            print(
                f"  → pointclouds/{pc_key}/chunk-{chunk_idx:03d}/episode_{epi_idx:06d}.npz  (M={len(xyz_cat)}, T={len(buf['ts'])})"
            )

    # Write pointclouds/meta.json (side-car format descriptor)
    if pc_keys:
        import json as _json

        (out_root / "pointclouds" / "meta.json").write_text(
            _json.dumps(
                {
                    "format": "csr",
                    "description": "Unorganized PointCloud2, variable N per frame, CSR format",
                    "pointcloud_keys": sorted(pc_keys),
                    "fields": {
                        "xyz": {"shape": "(M,3)", "dtype": "float32", "unit": "meters"},
                        "rgb": {"shape": "(M,3)", "dtype": "uint8", "range": "0-255"},
                        "offsets": {
                            "shape": "(T+1,)",
                            "dtype": "int64",
                            "note": "frame i -> xyz[offsets[i]:offsets[i+1]]",
                        },
                        "timestamps_ns": {
                            "shape": "(T,)",
                            "dtype": "int64",
                            "note": "align with parquet timestamp column (ns)",
                        },
                    },
                },
                indent=2,
            )
        )

    # A run that converted nothing must fail loudly. Skipped bags are only warned
    # about inside the loop, so without this an all-failed run would still print
    # "[OK]" over an empty dataset and exit 0.
    if converted_episodes == 0:
        details = "; ".join(f"{bag_dir}: {reason}" for bag_dir, reason in skipped_bags) or "no bags were provided"
        # Drop the skeleton LeRobotDataset.create() laid down, but only when this run
        # created it — an out_root the caller already had is never ours to delete.
        if not out_root_existed and out_root.exists():
            shutil.rmtree(out_root, ignore_errors=True)
        raise RuntimeError(
            f"Converted 0 of {len(bag_dirs)} bag(s); no episode was written to {out_root}. Causes: {details}"
        )

    print(f"\n[OK] Dataset root: {ds.root.resolve()}")
    print(f"  - converted {converted_episodes}/{len(bag_dirs)} bags")
    if skipped_bags:
        print(f"⚠️  Skipped {len(skipped_bags)} of {len(bag_dirs)} bag(s):")
        for bag_dir, reason in skipped_bags:
            print(f"    - {bag_dir}: {reason}")
    if use_videos:
        print("  - videos/<image_key>/chunk-*/file-*.mp4")
    else:
        print("  - images/*/*.png")
    print("  - data/chunk-*/file-*.parquet")
    print("  - meta/info.json, meta/tasks.parquet, meta/stats.json, meta/episodes/*/*.parquet")
    if pc_keys:
        print("  - pointclouds/<pc_key>/chunk-*/episode_*.npz  (CSR format)")
        print("  - pointclouds/meta.json")


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line args for bag → LeRobot conversion."""
    ap = argparse.ArgumentParser(
        "ROS2 bag → LeRobot v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python bag_to_lerobot.py --bag /path/to/bag --out /path/to/out
""",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bag", help="Path to a single bag directory (episode)")
    g.add_argument("--bags", nargs="+", help="Paths to multiple bag directories")
    g.add_argument(
        "--bags-dir",
        help="Directory containing multiple bag subdirectories (auto-discovers all valid bags)",
    )

    ap.add_argument("--out", required=True, help="Output dataset root")
    ap.add_argument("--repo-id", default="rosbag_v30", help="repo_id metadata")
    ap.add_argument("--no-videos", action="store_true", help="Store images instead of videos")
    ap.add_argument("--image-threads", type=int, default=4, help="Image writer threads")
    ap.add_argument("--image-processes", type=int, default=0, help="Image writer processes")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--data-mb", type=int, default=100)
    ap.add_argument("--video-mb", type=int, default=500)
    ap.add_argument(
        "--video-codec",
        choices=("auto", "h264", "hevc", "libsvtav1", "h264_nvenc", "hevc_nvenc"),
        default="auto",
        help=(
            "Codec for generated mp4 files. 'auto' prefers h264 for smoother local playback "
            "and falls back to libsvtav1 when h264 is unavailable."
        ),
    )
    ap.add_argument(
        "--timestamp",
        choices=("contract", "bag", "header"),
        default="contract",
        help=(
            "Which time base to use when resampling: "
            "'contract' (per-spec), 'bag' (receive), or 'header' (message header)."
        ),
    )
    return ap.parse_args()


def main() -> None:
    """CLI entry point for batch conversion of ROS 2 bags to LeRobot."""
    args = parse_args()
    if args.bag:
        bag_dirs = [Path(args.bag)]
    elif args.bags:
        bag_dirs = [Path(p) for p in args.bags]
    else:
        session_dir = Path(args.bags_dir)
        if (session_dir / "dataset.yaml").exists() and (session_dir / "episodes").is_dir():
            print(f"[bag_to_lerobot] Using dataset root {session_dir} → scanning {session_dir / 'episodes'}")
            session_dir = session_dir / "episodes"

        # Auto-discover: find all subdirectories that contain metadata.yaml
        bag_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir() and (p / "metadata.yaml").exists())
        if not bag_dirs:
            raise SystemExit(f"No valid bags found in {session_dir}")
        print(f"[bag_to_lerobot] Found {len(bag_dirs)} bags in {session_dir}:")
        for p in bag_dirs:
            print(f"  {p.name}")

    export_bags_to_lerobot(
        bag_dirs=bag_dirs,
        out_root=Path(args.out),
        repo_id=args.repo_id,
        use_videos=not args.no_videos,
        image_writer_threads=args.image_threads,
        image_writer_processes=args.image_processes,
        chunk_size=args.chunk_size,
        data_mb=args.data_mb,
        video_mb=args.video_mb,
        timestamp_source=args.timestamp,
        video_codec=args.video_codec,
    )


if __name__ == "__main__":
    main()

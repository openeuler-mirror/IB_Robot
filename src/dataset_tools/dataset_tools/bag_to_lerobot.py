#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 bag → LeRobot v3.0 exporter.

Overview
--------
This script converts one or more ROS 2 bags into a LeRobot v3 dataset using
the *same* contract-aware processing utilities used for live
inference. That keeps train/serve paths aligned and minimizes skew.

The conversion pipeline:

1) Load contract from robot_config.yaml (Single Source of Truth)
2) Scan a bag once; decode each contract topic using shared `decode_value`.
3) Select timestamps per a policy (`contract` / `bag` / `header`).
4) Resample each stream at the contract rate and assemble frames.
5) Coerce/resize images with the shared helpers and write to LeRobot.

Dependencies
------------
Shared modules (keep it unified with live inference):

- `robot_config.contract_utils`:
    `load_contract`, `iter_specs`, `feature_from_spec`

Command-line usage
------------------
Convert a single bag:

    $ python bag_to_lerobot.py \\
        --bag /path/to/bag_dir \\
        --robot-config /path/to/robot_config.yaml \\
        --out /path/to/out_root

Convert multiple bags:

    $ python bag_to_lerobot.py \\
        --bags /bag/epi1 /bag/epi2 \\
        --robot-config /path/to/robot_config.yaml \\
        --out /path/to/out_root

Options of note:

- `--robot-config`
    Path to robot_config.yaml. The contract section is used as the
    Single Source of Truth for observations and actions.

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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# ---- LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---- Shared core (ONLY these two)
from robot_config.contract_utils import (
    iter_specs,
    feature_from_spec,
    contract_fingerprint,
    Contract,
)
from robot_config.contract_utils import (
    decode_value,
    resample,
    stamp_from_header_ns,
    zero_pad as make_zero_pad,
)
from robot_config.utils import (
    build_joint_conversion_table,
    build_joint_conversion_table_from_calibration,
    normalize_lerobot_norm_mode,
    resolve_calibration_path_from_config,
    resolve_gripper_joints_from_config,
    resolve_lerobot_norm_mode,
)

# Import decoders to register them
import tensormsg.converter  # noqa: F401

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
    ts: List[int]
    val: List[Any]


# ---------------------------------------------------------------------------


def _read_yaml(p: Path) -> Dict[str, Any]:
    """Read a YAML file if it exists; return {} on absence/parse failures."""
    if not p.exists():
        print(f"[WARN] {p} does not exist")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_metadata_for_bag(bag_dir: Path) -> Dict[str, Any]:
    """Load dataset metadata when the bag lives under <dataset_root>/episodes/."""
    if bag_dir.parent.name != "episodes":
        return {}
    dataset_meta = bag_dir.parent.parent / "dataset.yaml"
    if not dataset_meta.exists():
        return {}
    return _read_yaml(dataset_meta)


def _lerobot_metadata_entry(
    dataset_meta: Dict[str, Any],
    bag_info: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
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
        fingerprint = str(custom_data.get("ibrobot.lerobot_conversion_fingerprint", "") or "")
    if not fingerprint:
        fingerprint = str(lerobot_meta.get("default_conversion_fingerprint", "") or "")
    if not fingerprint:
        return "", {}

    conversion_meta = conversions.get(fingerprint)
    if not isinstance(conversion_meta, dict):
        return fingerprint, {}
    return fingerprint, conversion_meta


def _resolve_fallback_conversion_config(robot_config_path: Path) -> Dict[str, Any]:
    """Resolve conversion inputs directly from robot_config for older datasets."""
    robot_config = _read_yaml(robot_config_path)
    if not robot_config:
        return {}
    return {
        "norm_mode": resolve_lerobot_norm_mode(robot_config),
        "gripper_joints": resolve_gripper_joints_from_config(robot_config),
        "calibration_file": resolve_calibration_path_from_config(robot_config),
    }


def _build_feature_conversion_table(
    feature_names: List[str],
    conversion_meta: Dict[str, Any],
    fallback_config: Dict[str, Any],
) -> List[Tuple[float, float, float, float]]:
    """Build a per-feature conversion table in the feature's declared joint order."""
    ordered_names = [str(name) for name in feature_names]
    if not ordered_names:
        return []

    if conversion_meta:
        norm_mode = normalize_lerobot_norm_mode(str(conversion_meta.get("norm_mode", "")))
        if norm_mode == "none":
            return []
        calibration = conversion_meta.get("calibration")
        if isinstance(calibration, dict):
            return build_joint_conversion_table_from_calibration(
                calibration=calibration,
                joint_names=ordered_names,
                gripper_joints=conversion_meta.get("gripper_joints"),
                norm_mode=norm_mode,
            )
        raise ValueError("Dataset conversion metadata is missing calibration snapshot")

    calib_file = str(fallback_config.get("calibration_file", "") or "")
    if not calib_file:
        return []
    return build_joint_conversion_table(
        calib_file=calib_file,
        joint_names=ordered_names,
        gripper_joints=fallback_config.get("gripper_joints"),
        norm_mode=str(fallback_config.get("norm_mode", "")),
    )


def _rad_to_lerobot(values: np.ndarray, table: List[Tuple[float, float, float, float]]) -> np.ndarray:
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


def _topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, str]:
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


def _estimate_stream_rate_hz(ts_ns: List[int]) -> float:
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
    streams: Dict[str, _Stream],
    ticks_ns: np.ndarray,
    step_ns: int,
    target_fps: int,
) -> None:
    """Log observed image rates and repeated-frame ratios after resampling."""
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
        print(
            f"  [diag] {key}: source_frames={len(st.ts)} (~{rate_hz:.1f} Hz), "
            f"unique_output_frames={unique_frames}/{len(ticks_ns)}, "
            f"repeated_frame_ratio={repeated_ratio:.1%}"
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
    tmap: Dict[str, str],
) -> Tuple[Dict[str, _Stream], Dict[str, List[str]]]:
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
    streams: Dict[str, _Stream] = {}
    by_topic: Dict[str, List[str]] = {}
    for sv in specs:
        if sv.topic not in tmap:
            # Derive a human-readable kind for logging without assuming SpecView internals.
            if hasattr(sv, "is_action") and sv.is_action:
                kind = "action"
            elif str(getattr(sv, "key", "")).startswith("task."):
                kind = "task"
            else:
                kind = "observation"
            print(
                f"[WARN] Missing {kind} '{getattr(sv, 'key', '?')}' topic in bag: {sv.topic}"
            )
            continue
        rt = sv.ros_type or tmap[sv.topic]

        # Create unique key for multiple observation.state specs and action specs
        if sv.key == "observation.state":
            # Remove leading underscore from topic replacement
            topic_suffix = sv.topic.replace('/', '_').lstrip('_')
            unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
        elif sv.is_action:
            # For action specs, we need to check if there are multiple specs with the same key
            # This will be handled later in the consolidation logic
            topic_suffix = sv.topic.replace('/', '_').lstrip('_')
            unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
        else:
            unique_key = sv.key

        streams[unique_key] = _Stream(spec=sv, ros_type=rt, ts=[], val=[])
        by_topic.setdefault(sv.topic, []).append(unique_key)
    if not streams:
        raise RuntimeError("No contract topics found in bag.")
    return streams, by_topic


# ---------------------------------------------------------------------------


def _load_contract_from_robot_config(robot_config_path: Path) -> Contract:
    """Load contract from robot_config.yaml (Single Source of Truth)."""
    print(f"[bag_to_lerobot] Loading contract from robot_config: {robot_config_path}")
    from robot_config.loader import load_robot_config
    robot_config = load_robot_config(str(robot_config_path))
    contract = robot_config.to_contract()
    
    print(f"[bag_to_lerobot]   Observations: {len(contract.observations)}")
    for obs in contract.observations:
        print(f"[bag_to_lerobot]     - {obs.key} <- {obs.topic}")
    print(f"[bag_to_lerobot]   Actions: {len(contract.actions)}")
    for act in contract.actions:
        print(f"[bag_to_lerobot]     - {act.key} -> {act.publish_topic}")

    return contract


def export_bags_to_lerobot(
    bag_dirs: List[Path],
    robot_config_path: Path,
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

    Uses robot_config.yaml as the Single Source of Truth for contract.

    Parameters
    ----------
    bag_dirs : list[pathlib.Path]
        One or more bag directories (episodes) to convert.
    robot_config_path : pathlib.Path
        Path to robot_config.yaml. The contract section will be used.
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
    contract = _load_contract_from_robot_config(robot_config_path)
    fallback_conversion_config = _resolve_fallback_conversion_config(robot_config_path)
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
    features: Dict[str, Dict[str, Any]] = {}
    primary_image_key: Optional[str] = None
    state_specs = []  # Track multiple observation.state specs
    action_specs_by_key: Dict[str, List[Any]] = {}  # Track multiple action specs by key
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
        if str(k).startswith(
            "task."
        ):  # TODO: why is this special-cased? Shouldn't this be handled in constract_utils?
            # Normalize to a simple scalar string field.
            features[k] = {"dtype": "string", "shape": [1]}
        else:
            # Special handling for depth images - they now have 3 channels
            if k.endswith(".depth") and ft["shape"][-1] == 1:
                # Update the shape to reflect 3 channels
                ft["shape"] = list(ft["shape"])
                ft["shape"][-1] = 3
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

        features["observation.state"] = {
            "dtype": "float32",
            "shape": (total_shape,),
            "names": all_names
        }

    # Consolidate multiple action specs with the same key into a single feature
    for action_key, action_specs in action_specs_by_key.items():
        if len(action_specs) > 1:
            # Multiple specs with same key - consolidate them
            all_names = []
            total_shape = 0
            for sv in action_specs:
                all_names.extend(sv.names)
                total_shape += len(sv.names)

            features[action_key] = {
                "dtype": "float32",
                "shape": (total_shape,),
                "names": all_names
            }
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

    # Persist the contract fingerprint into info.json so training can validate & propagate it
    try:
        fp = contract_fingerprint(contract)
        ds.meta.info["ibrobot_fingerprint"] = fp
    except Exception:
        pass  # non-fatal; downstream will just skip the check
    ds.meta.update_chunk_settings(
        chunks_size=chunk_size,
        data_files_size_in_mb=data_mb,
        video_files_size_in_mb=video_mb,
    )


    # Precompute zero pads + shapes for fast frame assembly.
    zero_pad_map = {k: make_zero_pad(ft) for k, ft in features.items()}
    write_keys = [
        k
        for k, ft in features.items()
        if ft["dtype"] in ("video", "image", "float32", "float64", "string")
    ]
    shapes = {k: tuple(features[k]["shape"]) for k in write_keys}
    state_feature_names = [
        str(name) for name in features.get("observation.state", {}).get("names", [])
    ]
    action_feature_names = {
        action_key: [str(name) for name in features[action_key].get("names", [])]
        for action_key in action_specs_by_key
        if action_key in features
    }
    conversion_table_cache: Dict[
        Tuple[str, Tuple[str, ...]],
        List[Tuple[float, float, float, float]],
    ] = {}

    # Episodes
    for epi_idx, bag_dir in enumerate(bag_dirs):
        print(f"[Episode {epi_idx}] {bag_dir}")

        # Per-episode point cloud buffer (one entry per key)
        pc_buf: Dict[str, Dict[str, list]] = {
            k: {"xyz": [], "rgb": [], "ts": []} for k in pc_keys
        }

        try:
            meta = _read_yaml(bag_dir / "metadata.yaml")
            dataset_meta = _dataset_metadata_for_bag(bag_dir)
            info = meta.get("rosbag2_bagfile_information") or {}
            storage = info.get("storage_identifier") or "mcap"
            meta_dur_ns = int((info.get("duration") or {}).get("nanoseconds") or 0)
            conversion_fp, conversion_meta = _lerobot_metadata_entry(dataset_meta, info)

            # Operator prompt (if present). Accept either old/new keys gracefully.
            prompt = ""
            cd = info.get("custom_data")
            if isinstance(cd, dict):
                prompt = cd.get("lerobot.operator_prompt", prompt) or prompt
            if not prompt and isinstance(dataset_meta, dict):
                prompt = str(
                    dataset_meta.get("default_task")
                    or dataset_meta.get("task")
                    or prompt
                )

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
            print(f"⚠️  Skipping bag {bag_dir} due to error: {e}")
            continue

        tmap = _topic_type_map(reader)
        print(f"tmap: {tmap}")

        # Plan once - handle multiple observation.state specs and action specs
        streams, by_topic = _plan_streams(specs, tmap)

        # Create consolidated observation.state stream if we have multiple state specs
        if state_specs:
            # Find all observation.state streams
            state_streams = [k for k in streams.keys() if k == "observation.state"]
            if len(state_streams) > 1:
                # Create a consolidated stream that will concatenate the data
                # We'll handle this in the frame processing
                pass
        print(f"streams: {streams}")

        # Counters for light diagnostics
        decoded_msgs = 0

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

        # Choose anchor + duration
        valid_ts = [
            np.asarray(st.ts, dtype=np.int64) for st in streams.values() if st.ts
        ]
        if not valid_ts:
            raise RuntimeError(f"No usable messages in {bag_dir} (no timestamps).")
        if (
            primary_image_key
            and streams.get(primary_image_key)
            and streams[primary_image_key].ts
        ):
            start_ns = int(
                np.asarray(streams[primary_image_key].ts, dtype=np.int64).min()
            )
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
            print(
                "Metadata duration disagrees with observed duration. Using observed duration"
            )

        # Ticks
        n_ticks = int(dur_ns // step_ns) + 1
        ticks_ns = start_ns + np.arange(n_ticks, dtype=np.int64) * step_ns

        _log_image_stream_diagnostics(
            streams=streams,
            ticks_ns=ticks_ns,
            step_ns=step_ns,
            target_fps=fps,
        )


        # Resample onto ticks
        resampled: Dict[str, List[Any]] = {}
        for key, st in streams.items():
            if not st.ts:
                resampled[key] = [None] * n_ticks
                continue
            ts = np.asarray(st.ts, dtype=np.int64)
            pol = st.spec.resample_policy
            resampled[key] = resample(
                pol, ts, st.val, ticks_ns, step_ns, st.spec.asof_tol_ms
            )

        state_conversion_table: List[Tuple[float, float, float, float]] = []
        if state_feature_names:
            cache_key = (conversion_fp or "robot_config", tuple(state_feature_names))
            if cache_key not in conversion_table_cache:
                try:
                    conversion_table_cache[cache_key] = _build_feature_conversion_table(
                        feature_names=state_feature_names,
                        conversion_meta=conversion_meta,
                        fallback_config=fallback_conversion_config,
                    )
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    print(
                        f"[WARN] Failed to build observation.state conversion table for "
                        f"{bag_dir}: {exc}"
                    )
                    conversion_table_cache[cache_key] = []
            state_conversion_table = conversion_table_cache[cache_key]

        action_conversion_tables: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for action_key, feature_names in action_feature_names.items():
            cache_key = (conversion_fp or "robot_config", tuple(feature_names))
            if cache_key not in conversion_table_cache:
                try:
                    conversion_table_cache[cache_key] = _build_feature_conversion_table(
                        feature_names=feature_names,
                        conversion_meta=conversion_meta,
                        fallback_config=fallback_conversion_config,
                    )
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    print(
                        f"[WARN] Failed to build {action_key} conversion table for "
                        f"{bag_dir}: {exc}"
                    )
                    conversion_table_cache[cache_key] = []
            action_conversion_tables[action_key] = conversion_table_cache[cache_key]

        # Write frames
        for i in range(n_ticks):
            frame: Dict[str, Any] = {}

            # Handle consolidated observation.state by concatenating multiple state streams first
            if "observation.state" in features and state_specs:
                # Concatenate all observation.state values from different topics
                state_values = []
                for sv in state_specs:
                    topic_suffix = sv.topic.replace('/', '_').lstrip('_')
                    unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
                    stream_val = resampled.get(unique_key, [None] * n_ticks)[i]
                    if stream_val is not None:
                        val_array = np.asarray(stream_val, dtype=np.float32).reshape(-1)
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
                        concatenated_state = _rad_to_lerobot(
                            concatenated_state, state_conversion_table
                        )
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
                        topic_suffix = sv.topic.replace('/', '_').lstrip('_')
                        unique_key = f"{sv.key}_{topic_suffix}" if topic_suffix else sv.key
                        stream_val = resampled.get(unique_key, [None] * n_ticks)[i]
                        if stream_val is not None:
                            val_array = np.asarray(stream_val, dtype=np.float32).reshape(-1)
                            action_values.append(val_array)

                    if action_values:
                        # Concatenate all action values
                        concatenated_action = np.concatenate(action_values)

                        # Pad or truncate to match feature shape if necessary
                        exp = int(features[action_key]["shape"][0])
                        if concatenated_action.shape[0] != exp:
                            fixed = np.zeros((exp,), dtype=np.float32)
                            fixed[: min(exp, concatenated_action.shape[0])] = (
                                concatenated_action[: min(exp, concatenated_action.shape[0])]
                            )
                            concatenated_action = fixed
                        conversion_table = action_conversion_tables.get(action_key, [])
                        if conversion_table:
                            concatenated_action = _rad_to_lerobot(
                                concatenated_action, conversion_table
                            )

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
                    arr = np.asarray(val)
                    # Ensure deterministic storage; lerobot loaders will map back to float [0,1]
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    frame[name] = arr

                elif dtype in ("float32", "float64"):
                    tgt_dt = np.float32 if dtype == "float32" else np.float64
                    arr = np.asarray(val, dtype=tgt_dt).reshape(-1)
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

        ds.save_episode()
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
            print(f"  → pointclouds/{pc_key}/chunk-{chunk_idx:03d}/episode_{epi_idx:06d}.npz  (M={len(xyz_cat)}, T={len(buf['ts'])})")


    # Write pointclouds/meta.json (side-car format descriptor)
    if pc_keys:
        import json as _json
        (out_root / "pointclouds" / "meta.json").write_text(_json.dumps({
            "format": "csr",
            "description": "Unorganized PointCloud2, variable N per frame, CSR format",
            "pointcloud_keys": sorted(pc_keys),
            "fields": {
                "xyz":           {"shape": "(M,3)",  "dtype": "float32", "unit": "meters"},
                "rgb":           {"shape": "(M,3)",  "dtype": "uint8",   "range": "0-255"},
                "offsets":       {"shape": "(T+1,)", "dtype": "int64",
                                  "note": "frame i -> xyz[offsets[i]:offsets[i+1]]"},
                "timestamps_ns": {"shape": "(T,)",   "dtype": "int64",
                                  "note": "align with parquet timestamp column (ns)"},
            },
        }, indent=2))

    print(f"\n[OK] Dataset root: {ds.root.resolve()}")
    if use_videos:
        print("  - videos/<image_key>/chunk-*/file-*.mp4")
    else:
        print("  - images/*/*.png")
    print("  - data/chunk-*/file-*.parquet")
    print(
        "  - meta/info.json, meta/tasks.parquet, meta/stats.json, meta/episodes/*/*.parquet"
    )
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
    python bag_to_lerobot.py --bag /path/to/bag --robot-config /path/to/robot_config.yaml --out /path/to/out
""",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bag", help="Path to a single bag directory (episode)")
    g.add_argument("--bags", nargs="+", help="Paths to multiple bag directories")
    g.add_argument(
        "--bags-dir",
        help="Directory containing multiple bag subdirectories (auto-discovers all valid bags)",
    )

    ap.add_argument(
        "--robot-config",
        required=True,
        help="Path to robot_config.yaml (Single Source of Truth)",
    )
    ap.add_argument("--out", required=True, help="Output dataset root")
    ap.add_argument("--repo-id", default="rosbag_v30", help="repo_id metadata")
    ap.add_argument(
        "--no-videos", action="store_true", help="Store images instead of videos"
    )
    ap.add_argument("--image-threads", type=int, default=4, help="Image writer threads")
    ap.add_argument(
        "--image-processes", type=int, default=0, help="Image writer processes"
    )
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
            print(
                f"[bag_to_lerobot] Using dataset root {session_dir} "
                f"→ scanning {session_dir / 'episodes'}"
            )
            session_dir = session_dir / "episodes"

        # Auto-discover: find all subdirectories that contain metadata.yaml
        bag_dirs = sorted(
            p for p in session_dir.iterdir()
            if p.is_dir() and (p / "metadata.yaml").exists()
        )
        if not bag_dirs:
            raise SystemExit(f"No valid bags found in {session_dir}")
        print(f"[bag_to_lerobot] Found {len(bag_dirs)} bags in {session_dir}:")
        for p in bag_dirs:
            print(f"  {p.name}")

    export_bags_to_lerobot(
        bag_dirs=bag_dirs,
        robot_config_path=Path(args.robot_config),
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

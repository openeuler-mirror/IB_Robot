"""Public robot model projection and ROS-free, file-free consumer conversions.

Only ``build_model_metadata`` reads robot-private calibration. All other helpers
consume JSON-compatible public metadata, including offline dataset snapshots.
No runtime helper imports consumer configuration or starts hardware.

``joint_conversions`` schema version 1:
  joint_names: ordered, unique strings
  quantities: joint -> position | velocity (rad | rad/s)
  modes: degrees | range_m100_100 -> joint -> {min, max, span, offset}
         none -> {} (explicit identity for both positions and velocities)
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from robot_runtime.joint_conversion import (
    NORM_MODE_DEGREES,
    NORM_MODE_NONE,
    NORM_MODE_RANGE,
    JointConversionEntry,
    build_joint_conversion_table_from_calibration,
    build_joint_conversion_table_from_urdf,
    joint_limits_from_urdf,
    normalize_lerobot_norm_mode,
)

_MODES = (NORM_MODE_DEGREES, NORM_MODE_RANGE, NORM_MODE_NONE)
_ENTRY_FIELDS = ("min", "max", "span", "offset")


def _names(names, label: str, *, empty: bool = False) -> None:
    if (
        not isinstance(names, list)
        or (not names and not empty)
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{label} must be an ordered list of unique nonempty joint names")


def validate_model_metadata(model: dict) -> None:
    """Validate portable model semantics independently of ROS or provider files."""
    required = {
        "schema_version",
        "authority",
        "joint_groups",
        "joint_limits",
        "home_positions",
        "frames",
        "joint_conversions",
        "joint_conversions_fingerprint",
    }
    if not isinstance(model, dict) or set(model) != required:
        raise ValueError(f"robot_model requires exactly {sorted(required)}")
    if type(model["schema_version"]) is not int or model["schema_version"] != 1:
        raise ValueError("unsupported robot_model.schema_version")
    if model["authority"] not in ("calibration", "urdf"):
        raise ValueError("robot_model.authority must be calibration or urdf")
    groups = model["joint_groups"]
    if not isinstance(groups, dict) or set(groups) != {"all", "arm", "gripper", "base"}:
        raise ValueError("robot_model.joint_groups requires all, arm, gripper and base")
    for group, names in groups.items():
        _names(names, f"robot_model.joint_groups.{group}", empty=group != "all")
    members = groups["arm"] + groups["gripper"] + groups["base"]
    if len(set(members)) != len(members) or set(members) != set(groups["all"]):
        raise ValueError("robot_model joint groups must partition all joints")
    conversions = model["joint_conversions"]
    validate_joint_conversions(conversions)
    if conversions["joint_names"] != groups["all"]:
        raise ValueError("robot_model conversion order must match joint_groups.all")
    if model["joint_conversions_fingerprint"] != joint_conversions_fingerprint(conversions):
        raise ValueError("robot_model joint_conversions_fingerprint mismatch")
    limits = model["joint_limits"]
    if not isinstance(limits, dict) or set(limits) != set(groups["all"]):
        raise ValueError("robot_model.joint_limits must cover all joints")
    for name, bounds in limits.items():
        if (
            not isinstance(bounds, dict)
            or set(bounds) != {"min", "max"}
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds.values())
            or bounds["min"] >= bounds["max"]
        ):
            raise ValueError(f"robot_model.joint_limits.{name} requires finite increasing min/max")
        quantity = "velocity" if name in groups["base"] else "position"
        if conversions["quantities"][name] != quantity:
            raise ValueError(f"robot_model joint quantity mismatch: {name}")
        for mode in (NORM_MODE_DEGREES, NORM_MODE_RANGE):
            entry = conversions["modes"][mode][name]
            if bounds["min"] < entry["min"] or bounds["max"] > entry["max"]:
                raise ValueError(f"robot_model limits exceed conversion range: {name}")
            if name in groups["gripper"] and (entry["span"], entry["offset"]) != (100, 0):
                raise ValueError(f"robot_model gripper {name} requires closed=0/open=100 policy semantics")
    home = model["home_positions"]
    if not isinstance(home, dict) or not set(home) <= set(groups["all"]):
        raise ValueError("robot_model.home_positions references unknown joints")
    for name, value in home.items():
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not limits[name]["min"] <= value <= limits[name]["max"]
        ):
            raise ValueError(f"robot_model.home_positions.{name} is outside physical limits")
    frames = model["frames"]
    if (
        not isinstance(frames, dict)
        or set(frames) - {"base_link", "ee_link", "shoulder_link"}
        or any(not isinstance(value, str) or not value.strip() for value in frames.values())
    ):
        raise ValueError("robot_model.frames requires public nonempty frame names")


def validate_joint_conversions(conversions: dict) -> None:
    """Reject unknown versions, incomplete tables and non-invertible mappings."""
    if not isinstance(conversions, dict) or set(conversions) != {
        "schema_version",
        "joint_names",
        "quantities",
        "modes",
    }:
        raise ValueError("joint_conversions requires schema_version, joint_names, quantities and modes")
    if type(conversions["schema_version"]) is not int or conversions["schema_version"] != 1:
        raise ValueError("unsupported joint_conversions.schema_version")
    names = conversions["joint_names"]
    _names(names, "joint_conversions.joint_names", empty=True)
    quantities, modes = conversions["quantities"], conversions["modes"]
    if not isinstance(quantities, dict) or set(quantities) != set(names):
        raise ValueError("joint_conversions.quantities must cover joint_names exactly")
    if any(quantity not in ("position", "velocity") for quantity in quantities.values()):
        raise ValueError("joint_conversions quantities must be position or velocity")
    if not isinstance(modes, dict) or set(modes) != set(_MODES) or modes[NORM_MODE_NONE] != {}:
        raise ValueError("joint_conversions.modes requires degrees, range_m100_100 and none: {}")
    for mode in (NORM_MODE_DEGREES, NORM_MODE_RANGE):
        entries = modes[mode]
        if not isinstance(entries, dict) or set(entries) != set(names):
            raise ValueError(f"joint_conversions.{mode} must cover joint_names exactly")
        for name, entry in entries.items():
            if not isinstance(entry, dict) or set(entry) != set(_ENTRY_FIELDS):
                raise ValueError(f"joint_conversions.{mode}.{name} requires {_ENTRY_FIELDS}")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in entry.values()):
                raise ValueError(f"joint_conversions.{mode}.{name} must contain finite numbers")
            if entry["max"] <= entry["min"] or entry["span"] <= 0:
                raise ValueError(f"joint_conversions.{mode}.{name} has invalid range or span")


def joint_conversions_fingerprint(conversions: dict) -> str:
    validate_joint_conversions(conversions)
    encoded = json.dumps(conversions, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_joint_conversion_table_from_model(
    model: dict, joint_names: list[str], norm_mode: str = NORM_MODE_RANGE
) -> list[JointConversionEntry]:
    """Select in feature order, never in robot order; missing metadata fails closed."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    _names(joint_names, f"contract joints for normalization {mode}")
    conversions = model.get("joint_conversions") if isinstance(model, dict) else None
    if conversions is None:
        raise ValueError(f"robot_model.joint_conversions is required for LeRobot normalization {mode!r}")
    validate_joint_conversions(conversions)
    names = [str(name) for name in joint_names]
    missing = set(names) - set(conversions["joint_names"])
    if missing:
        hint = (
            " (names look like message field selectors, not joint names:"
            " 'position.<joint>'/'current.<joint>' carry the joint name in the suffix)"
            if any("." in name for name in missing)
            else ""
        )
        raise ValueError(
            f"robot_model.joint_conversions is missing contract joints for {mode}: {sorted(missing)}{hint}"
        )
    if mode == NORM_MODE_NONE:
        return []
    return [tuple(float(conversions["modes"][mode][name][field]) for field in _ENTRY_FIELDS) for name in names]


def build_public_conversion_metadata(
    model: dict, joint_names: list[str], norm_mode: str, *, description: dict, feature_names: dict[str, list[str]]
) -> dict:
    """Snapshot conversion authority and selected feature names without private inputs."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    from robot_runtime.interface_description import validate_description

    validate_description(description)
    validate_model_metadata(model)
    if model != description.get("model"):
        raise ValueError("conversion model conflicts with bound public description")
    if not isinstance(feature_names, dict) or not feature_names:
        raise ValueError("public conversion metadata requires named feature order")
    for key, names in feature_names.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("conversion feature keys must be nonempty strings")
        build_joint_conversion_table_from_model(model, names, mode)
    build_joint_conversion_table_from_model(model, joint_names, mode)
    result = {
        "schema_version": 1,
        "norm_mode": mode,
        "joint_names": list(joint_names),
        "gripper_joints": list((model.get("joint_groups") or {}).get("gripper", [])),
        "feature_names": deepcopy(feature_names),
        "description": deepcopy(description),
        "description_digest": description["digest"],
    }
    if "joint_conversions" in model:
        result["joint_conversions"] = deepcopy(model["joint_conversions"])
        result["joint_conversions_fingerprint"] = joint_conversions_fingerprint(result["joint_conversions"])
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["conversion_fingerprint"] = hashlib.sha256(encoded.encode()).hexdigest()
    return result


def validate_public_conversion_metadata(metadata: dict) -> None:
    """Validate a recorded snapshot before trusting it as offline conversion authority."""
    required = {
        "schema_version",
        "norm_mode",
        "joint_names",
        "gripper_joints",
        "feature_names",
        "description",
        "description_digest",
        "joint_conversions",
        "joint_conversions_fingerprint",
        "conversion_fingerprint",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required
        or type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != 1
    ):
        raise ValueError("invalid or incomplete public conversion metadata schema_version 1")
    description = metadata["description"]
    if not isinstance(description, dict) or "model" not in description:
        raise ValueError("public conversion metadata requires description.model")
    expected = build_public_conversion_metadata(
        description["model"],
        metadata["joint_names"],
        metadata["norm_mode"],
        description=description,
        feature_names=metadata["feature_names"],
    )
    for key in required:
        if metadata[key] != expected[key]:
            raise ValueError(f"public conversion metadata {key} mismatch")


def build_model_metadata(profile: dict, *, simulated: bool, robot_description: str | None) -> dict:
    """Project calibrated ranges, grouping, rest pose and public motion frame names.

    Conversion ranges preserve calibration semantics; admissible command limits
    intersect those ranges with the effective URDF. Simulation never reads calibration.
    """
    names = [str(name) for name in profile.get("joints", [])]
    base = profile.get("base") or {}
    wheels = [str(name) for name in base.get("wheel_joints", [])]
    arm = [str(name) for name in profile.get("arm_joints", [])]
    grippers = profile.get("gripper_joints")
    if grippers is None:
        grippers = [
            name
            for channel in profile.get("command_channels", [])
            if channel.get("channel") == "gripper_stream"
            for name in channel.get("joints", [])
        ]
    grippers = [str(name) for name in grippers]
    if len(set(names)) != len(names) or not set(arm + grippers + wheels) <= set(names):
        raise ValueError("model joint groups must reference unique profile joints")
    if set(arm) & set(grippers) or set(arm + grippers) & set(wheels):
        raise ValueError("model arm, gripper and base groups must be disjoint")
    motion, hardware = profile.get("motion") or {}, profile.get("hardware") or {}
    home = deepcopy(profile.get("home_positions") or {})
    if not set(home) <= set(names) or any(not math.isfinite(value) for value in home.values()):
        raise ValueError("home_positions must contain finite values for profile joints")
    model = {
        "schema_version": 1,
        "authority": "urdf" if simulated else "calibration",
        "joint_groups": {"all": names, "arm": arm, "gripper": grippers, "base": wheels},
        "joint_limits": {},
        "home_positions": home,
        "frames": {key: str(motion[key]) for key in ("base_link", "ee_link", "shoulder_link") if motion.get(key)},
    }
    positions = [name for name in names if name not in wheels]
    if not robot_description:
        raise ValueError("public model requires the effective rendered robot_description")
    root = ET.fromstring(robot_description)
    link_names = {link.get("name") for link in root.findall("link")}
    for frame in model["frames"].values():
        if frame not in link_names:
            raise ValueError(f"public model frame {frame!r} is absent from rendered URDF")
    modes = {mode: {} for mode in _MODES}
    calibration = None
    if not simulated and hardware.get("calib_file"):
        from robot_runtime.path_subst import resolve_path

        with Path(resolve_path(hardware["calib_file"])).expanduser().open(encoding="utf-8") as handle:
            calibration = json.load(handle)
        if not isinstance(calibration, dict):
            raise ValueError("runtime calibration must be a JSON object")
    if positions and not simulated and calibration is None:
        raise ValueError("physical model requires effective calibrated conversion authority")
    for mode in (NORM_MODE_DEGREES, NORM_MODE_RANGE):
        table = []
        if positions:
            if calibration is not None:
                table = build_joint_conversion_table_from_calibration(calibration, positions, grippers, mode)
            elif simulated and robot_description:
                table = build_joint_conversion_table_from_urdf(robot_description, positions, grippers, mode)
        for name, entry in zip(positions, table, strict=True):
            modes[mode][name] = dict(zip(_ENTRY_FIELDS, entry, strict=True))
        if wheels:
            maximum = float(base["max_wheel_radps"])
            if not math.isfinite(maximum) or maximum <= 0:
                raise ValueError("base.max_wheel_radps must be finite and positive")
            for name in wheels:
                modes[mode][name] = {"min": -maximum, "max": maximum, "span": 200.0, "offset": -100.0}
    converted_names = [name for name in names if name in modes[NORM_MODE_RANGE]]
    if converted_names:
        conversions = {
            "schema_version": 1,
            "joint_names": converted_names,
            "quantities": {name: "velocity" if name in wheels else "position" for name in converted_names},
            "modes": modes,
        }
        validate_joint_conversions(conversions)
        model["joint_conversions"] = conversions
        model["joint_conversions_fingerprint"] = joint_conversions_fingerprint(conversions)
        model["joint_limits"] = {
            name: {key: modes[NORM_MODE_RANGE][name][key] for key in ("min", "max")} for name in converted_names
        }
        for name in positions:
            if not simulated and name in grippers:
                # The URDF gripper limit (0.0 closed .. 1.0 open) is the
                # normalized model convention consumed by the simulation's
                # urdf authority. On physical hardware the calibration is the
                # range authority, so the URDF convention must not clamp it;
                # existence is still validated.
                joint_limits_from_urdf(root, name)
                continue
            lower, upper = joint_limits_from_urdf(root, name)
            bounds = model["joint_limits"][name]
            bounds["min"] = max(bounds["min"], lower)
            bounds["max"] = min(bounds["max"], upper)
    else:
        model["joint_conversions"] = {
            "schema_version": 1,
            "joint_names": [],
            "quantities": {},
            "modes": {NORM_MODE_DEGREES: {}, NORM_MODE_RANGE: {}, NORM_MODE_NONE: {}},
        }
    model["joint_conversions_fingerprint"] = joint_conversions_fingerprint(model["joint_conversions"])
    validate_model_metadata(model)
    return model

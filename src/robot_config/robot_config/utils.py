"""Utility functions for robot_config package.

This module contains common utility functions used across the robot_config package:
- Path resolution (ROS-style substitutions)
- Boolean parsing
- Type conversion helpers
- Joint configuration validation
"""

import hashlib
import json
import math
import os
import re
import sys
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ament_index_python.packages import get_package_share_directory

logger = logging.getLogger(__name__)


def resolve_ros_path(path):
    """Resolve ROS-style path substitutions like $(find pkg) and $(env VAR).

    Handles ROS path substitution syntax:
    - $(find package_name): Resolves to package share directory
    - $(env VAR_NAME): Resolves to environment variable value

    Args:
        path: Path string that may contain $(find package) or $(env VAR)

    Returns:
        Resolved path string. Returns original path if it's None or empty.

    Example:
        >>> resolve_ros_path("$(find so101_hardware)/config/controllers.yaml")
        "/home/user/workspace/install/share/so101_hardware/config/controllers.yaml"

        >>> resolve_ros_path("$(env HOME)/.config/robot.yaml")
        "/home/user/.config/robot.yaml"
    """
    if not path:
        return path

    # Resolve $(find package)
    find_pattern = re.compile(r'\$\(find\s+(\w+)\)')
    for match in find_pattern.finditer(path):
        pkg_name = match.group(1)
        try:
            pkg_path = get_package_share_directory(pkg_name)
            path = path.replace(f"$(find {pkg_name})", pkg_path)
        except Exception as e:
            logger.warning(f"Could not find package '{pkg_name}': {e}")

    # Resolve $(env VAR)
    env_pattern = re.compile(r'\$\(env\s+(\w+)\)')
    for match in env_pattern.finditer(path):
        var_name = match.group(1)
        var_value = os.environ.get(var_name, "")
        path = path.replace(f"$(env {var_name})", var_value)
        if not var_value:
            logger.info(f"WARNING: Environment variable '{var_name}' is not set or empty")

    return path


def parse_bool(value, default=False):
    """Parse various value types to boolean with robust handling.

    Handles multiple input formats:
    - Strings: "true", "TRUE", "True", "1", "yes", "on" -> True
    - Strings: "false", "FALSE", "False", "0", "no", "off" -> False
    - Booleans: True/False -> as-is
    - Numbers: 1/0 -> True/False
    - None: -> default value

    Args:
        value: Input value to parse (string, bool, int, or None)
        default: Default value if input is None or unparseable

    Returns:
        Boolean value

    Example:
        >>> parse_bool("true")
        True
        >>> parse_bool("FALSE")
        False
        >>> parse_bool(True)
        True
        >>> parse_bool(None, default=False)
        False
    """
    if value is None:
        return default

    # Handle boolean types directly
    if isinstance(value, bool):
        return value

    # Convert to string and normalize
    str_value = str(value).strip().lower()

    # Check for true-like values
    if str_value in ('true', '1', 'yes', 'on'):
        return True

    # Check for false-like values
    if str_value in ('false', '0', 'no', 'off', ''):
        return False

    # Unknown value, return default
    return default


def validate_joint_config(robot_config):
    """Validate joint configuration across controllers and robot config.

    Implements DRY principle by checking that joint definitions are consistent
    between robot_config and controller configuration files.

    Args:
        robot_config: Robot configuration dict with joints and ros2_control sections

    Returns:
        True if validation passes, False otherwise

    Raises:
        Prints warnings/errors but does not raise exceptions to avoid blocking startup
    """
    logger.info("[robot_config] ========== Joint Configuration Validation ==========")

    joints_config = robot_config.get("joints", {})
    if not joints_config:
        logger.info("[robot_config] WARNING: No 'joints' configuration found")
        return True

    expected_arm_joints = set(joints_config.get("arm", []))
    expected_gripper_joints = set(joints_config.get("gripper", []))
    expected_all_joints = set(joints_config.get("all", []))

    logger.info(f"Canonical joints from robot_config:")
    logger.info(f"  arm: {sorted(expected_arm_joints)}")
    logger.info(f"  gripper: {sorted(expected_gripper_joints)}")
    logger.info(f"  all: {sorted(expected_all_joints)}")

    # Load controllers configuration
    ros2_control_config = robot_config.get("ros2_control", {})
    controllers_config_path = ros2_control_config.get("controllers_config", "")

    if not controllers_config_path:
        logger.info("[robot_config] WARNING: No controllers_config path specified")
        return True

    controllers_config_path = resolve_ros_path(controllers_config_path)

    if not Path(controllers_config_path).exists():
        logger.info(f"WARNING: Controllers config not found at {controllers_config_path}")
        return True

    # Load controllers YAML
    try:
        with open(controllers_config_path, 'r') as f:
            controllers_yaml = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load controllers config: {e}")
        return False

    validation_passed = True
    controllers_checked = 0

    # Check arm_position_controller
    arm_pos_ctrl = controllers_yaml.get("arm_position_controller", {}).get("ros__parameters", {})
    if arm_pos_ctrl:
        ctrl_joints = set(arm_pos_ctrl.get("joints", []))
        if ctrl_joints != expected_arm_joints:
            logger.error("[robot_config] ERROR: arm_position_controller joints mismatch!")
            validation_passed = False
        else:
            logger.info(f"✓ arm_position_controller joints match")
        controllers_checked += 1

    # Check gripper_position_controller
    grip_pos_ctrl = controllers_yaml.get("gripper_position_controller", {}).get("ros__parameters", {})
    if grip_pos_ctrl:
        ctrl_joints = set(grip_pos_ctrl.get("joints", []))
        if ctrl_joints != expected_gripper_joints:
            logger.error("[robot_config] ERROR: gripper_position_controller joints mismatch!")
            validation_passed = False
        else:
            logger.info(f"✓ gripper_position_controller joints match")
        controllers_checked += 1

    # Check joint_state_broadcaster
    jsb_ctrl = controllers_yaml.get("joint_state_broadcaster", {}).get("ros__parameters", {})
    if jsb_ctrl:
        ctrl_joints = set(jsb_ctrl.get("joints", []))
        if ctrl_joints != expected_all_joints:
            logger.error("[robot_config] ERROR: joint_state_broadcaster joints mismatch!")
            validation_passed = False
        else:
            logger.info(f"✓ joint_state_broadcaster joints match")
        controllers_checked += 1

    logger.info(f"Validated {controllers_checked} controller configurations")

    if validation_passed:
        logger.info("[robot_config] ✓ All joint configurations are consistent")
    else:
        logger.info("[robot_config] ✗ Joint configuration validation FAILED")

    logger.info("[robot_config] =========================================================")

    return validation_passed


def prepare_lerobot_env():
    """Prepare environment with lerobot PYTHONPATH."""
    env = os.environ.copy()
    workspace_path = os.environ.get("WORKSPACE", os.getcwd())
    lerobot_src = os.path.join(workspace_path, "libs/lerobot/src")

    if os.path.exists(lerobot_src):
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{lerobot_src}:{current_pp}" if current_pp else lerobot_src

    return env


# ---------------------------------------------------------------------------
# Joint unit-conversion helpers  (LeRobot percentage  ↔  ros2_control radians)
# ---------------------------------------------------------------------------

# Each entry: (rad_min, rad_max, pct_span, pct_offset)
JointConversionEntry = Tuple[float, float, float, float]
CalibrationSnapshot = Dict[str, Dict[str, Any]]

_TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)


# Supported LeRobot motor normalization modes.
# Keep in sync with the YAML ``lerobot_norm_mode`` option.
NORM_MODE_RANGE = "range_m100_100"   # arm [-100,+100], gripper [0,100]
NORM_MODE_DEGREES = "degrees"         # arm centred degrees, gripper [0,100]
NORM_MODE_NONE = "none"               # pass-through (no conversion)

_MODEL_RESOLUTION = 4096              # Feetech STS3215 12-bit encoder
_CALIBRATION_SNAPSHOT_FIELDS = (
    "id",
    "model",
    "drive_mode",
    "homing_offset",
    "range_min",
    "range_max",
)


def normalize_lerobot_norm_mode(norm_mode: str) -> str:
    """Normalize and validate LeRobot motor normalization mode."""
    mode = str(norm_mode or NORM_MODE_RANGE).strip().lower()
    if mode not in (NORM_MODE_RANGE, NORM_MODE_DEGREES, NORM_MODE_NONE):
        raise ValueError(
            f"Unsupported LeRobot normalization mode '{norm_mode}'. "
            f"Expected one of: {NORM_MODE_RANGE}, {NORM_MODE_DEGREES}, {NORM_MODE_NONE}"
        )
    return mode


def resolve_joint_names_from_config(robot_config: Dict[str, Any]) -> List[str]:
    """Resolve ordered joint names from raw robot_config YAML content."""
    ros2_control = robot_config.get("ros2_control", {}) or {}
    joints_cfg = robot_config.get("joints", {}) or {}
    joint_names = ros2_control.get("joint_names") or joints_cfg.get("all") or []
    return [str(name) for name in joint_names]


def resolve_gripper_joints_from_config(robot_config: Dict[str, Any]) -> List[str]:
    """Resolve gripper joint names from raw robot_config YAML content."""
    ros2_control = robot_config.get("ros2_control", {}) or {}
    joints_cfg = robot_config.get("joints", {}) or {}
    gripper_joints = ros2_control.get("gripper_joints") or joints_cfg.get("gripper") or []
    return [str(name) for name in gripper_joints]


def resolve_calibration_path_from_config(robot_config: Dict[str, Any]) -> str:
    """Resolve the ros2_control calibration file path from raw robot_config."""
    ros2_control = robot_config.get("ros2_control", {}) or {}
    calib_file = str(ros2_control.get("calib_file", "") or "")
    return resolve_ros_path(calib_file) if calib_file else ""


def resolve_lerobot_norm_mode(
    robot_config: Dict[str, Any],
    preferred_control_mode: Optional[str] = None,
) -> str:
    """Resolve the LeRobot normalization mode from robot_config semantics."""
    recording_cfg = robot_config.get("recording", {}) or {}
    explicit_mode = recording_cfg.get("lerobot_norm_mode")
    if explicit_mode:
        return normalize_lerobot_norm_mode(str(explicit_mode))

    control_modes = robot_config.get("control_modes", {}) or {}
    models = robot_config.get("models", {}) or {}
    mode_candidates: List[str] = []
    for mode_name in (
        preferred_control_mode,
        robot_config.get("default_control_mode"),
        "model_inference",
    ):
        if mode_name and mode_name not in mode_candidates:
            mode_candidates.append(str(mode_name))

    for mode_name in mode_candidates:
        inference_cfg = (control_modes.get(mode_name, {}) or {}).get("inference", {}) or {}
        model_name = inference_cfg.get("model")
        if model_name and isinstance(models.get(model_name), dict):
            model_mode = models[model_name].get("lerobot_norm_mode")
            if model_mode:
                return normalize_lerobot_norm_mode(str(model_mode))

    for model_cfg in models.values():
        if isinstance(model_cfg, dict) and model_cfg.get("lerobot_norm_mode"):
            return normalize_lerobot_norm_mode(str(model_cfg["lerobot_norm_mode"]))

    return NORM_MODE_RANGE


def load_calibration_data(calib_file: str) -> Dict[str, Any]:
    """Load a calibration JSON file from disk."""
    resolved_path = resolve_ros_path(calib_file)
    if not resolved_path:
        raise FileNotFoundError("Calibration file path is empty")

    calib_path = Path(resolved_path).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with calib_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {calib_path}")
    return data


def extract_calibration_snapshot(
    calibration: Dict[str, Any],
    joint_names: List[str],
) -> CalibrationSnapshot:
    """Extract a canonical calibration snapshot for the selected joints."""
    snapshot: CalibrationSnapshot = {}
    for joint_name in [str(name) for name in joint_names]:
        if joint_name not in calibration:
            raise KeyError(f"Joint '{joint_name}' missing from calibration data")

        entry = calibration[joint_name]
        if not isinstance(entry, dict):
            raise ValueError(f"Calibration entry for joint '{joint_name}' must be an object")

        joint_snapshot: Dict[str, Any] = {}
        for field in _CALIBRATION_SNAPSHOT_FIELDS:
            if field not in entry:
                continue
            if field == "model":
                joint_snapshot[field] = str(entry[field])
            else:
                joint_snapshot[field] = int(entry[field])

        if "range_min" not in joint_snapshot or "range_max" not in joint_snapshot:
            raise KeyError(
                f"Calibration entry for joint '{joint_name}' must contain range_min/range_max"
            )
        joint_snapshot.setdefault("drive_mode", 0)
        snapshot[joint_name] = joint_snapshot

    return snapshot


def lerobot_conversion_fingerprint(
    calibration: CalibrationSnapshot,
    joint_names: List[str],
    gripper_joints: Optional[List[str]] = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> str:
    """Compute a stable fingerprint for LeRobot conversion semantics."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    ordered_joints = [str(name) for name in joint_names]
    ordered_gripper_joints = [str(name) for name in (gripper_joints or [])]

    payload: Dict[str, Any] = {
        "norm_mode": mode,
        "joint_names": ordered_joints,
        "gripper_joints": ordered_gripper_joints,
        "joints": {},
    }

    for joint_name in ordered_joints:
        entry = calibration.get(joint_name, {})
        joint_payload: Dict[str, int] = {}
        if mode != NORM_MODE_NONE:
            if "range_min" not in entry or "range_max" not in entry:
                raise KeyError(
                    f"Calibration snapshot for joint '{joint_name}' must contain range_min/range_max"
                )
            joint_payload["range_min"] = int(entry["range_min"])
            joint_payload["range_max"] = int(entry["range_max"])
            joint_payload["drive_mode"] = int(entry.get("drive_mode", 0))
        payload["joints"][joint_name] = joint_payload

    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]


def build_lerobot_conversion_metadata(
    calib_file: str,
    joint_names: List[str],
    gripper_joints: Optional[List[str]] = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> Dict[str, Any]:
    """Build a dataset-storable snapshot of LeRobot conversion semantics."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    ordered_joints = [str(name) for name in joint_names]
    ordered_gripper_joints = [str(name) for name in (gripper_joints or [])]

    metadata: Dict[str, Any] = {
        "norm_mode": mode,
        "joint_names": ordered_joints,
        "gripper_joints": ordered_gripper_joints,
    }

    if mode == NORM_MODE_NONE:
        metadata["conversion_fingerprint"] = lerobot_conversion_fingerprint(
            calibration={},
            joint_names=ordered_joints,
            gripper_joints=ordered_gripper_joints,
            norm_mode=mode,
        )
        return metadata

    calibration_source = resolve_ros_path(calib_file)
    calibration = load_calibration_data(calibration_source)
    snapshot = extract_calibration_snapshot(calibration, ordered_joints)
    metadata["calibration_source"] = str(Path(calibration_source).expanduser().resolve())
    metadata["calibration"] = snapshot
    metadata["conversion_fingerprint"] = lerobot_conversion_fingerprint(
        calibration=snapshot,
        joint_names=ordered_joints,
        gripper_joints=ordered_gripper_joints,
        norm_mode=mode,
    )
    return metadata


def build_joint_conversion_table_from_calibration(
    calibration: Dict[str, Any],
    joint_names: List[str],
    gripper_joints: Optional[List[str]] = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> List[JointConversionEntry]:
    """Build a conversion table from calibration content already in memory."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    if mode == NORM_MODE_NONE:
        return []

    ordered_joints = [str(name) for name in joint_names]
    gripper_joint_set = {str(name) for name in (gripper_joints or [])}
    table: List[JointConversionEntry] = []

    for joint_name in ordered_joints:
        if joint_name not in calibration:
            raise KeyError(f"Joint '{joint_name}' missing from calibration data")

        entry = calibration[joint_name]
        if not isinstance(entry, dict):
            raise ValueError(f"Calibration entry for joint '{joint_name}' must be an object")

        tick_min = int(entry["range_min"])
        tick_max = int(entry["range_max"])

        if mode == NORM_MODE_DEGREES and joint_name not in gripper_joint_set:
            mid = (tick_min + tick_max) / 2.0
            max_res = _MODEL_RESOLUTION - 1  # 4095
            deg_at_tick_min = (tick_min - mid) * 360.0 / max_res
            deg_at_tick_max = (tick_max - mid) * 360.0 / max_res

            rad_min = (tick_min - 2048.0) / _TICKS_PER_RAD
            rad_max = (tick_max - 2048.0) / _TICKS_PER_RAD
            span = deg_at_tick_max - deg_at_tick_min
            offset = deg_at_tick_min
        else:
            rad_min = (tick_min - 2048.0) / _TICKS_PER_RAD
            rad_max = (tick_max - 2048.0) / _TICKS_PER_RAD

            if joint_name in gripper_joint_set:
                span = 100.0
                offset = 0.0
            else:
                span = 200.0
                offset = -100.0

        table.append((rad_min, rad_max, span, offset))

    return table


def build_joint_conversion_table(
    calib_file: str,
    joint_names: List[str],
    gripper_joints: Optional[List[str]] = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> List[JointConversionEntry]:
    """Build per-joint ``(rad_min, rad_max, lerobot_span, lerobot_offset)``.

    The C++ hardware layer converts  ticks ↔ radians  using a fixed formula::

        rad  = (ticks - 2048) / (4096 / 2π)
        ticks = rad * (4096 / 2π) + 2048

    LeRobot normalises ticks differently depending on *norm_mode*:

    **range_m100_100** (default)
        * Arm joints  ``RANGE_M100_100``:  ``pct = (t-tmin)/(tmax-tmin)*200 - 100``
        * Gripper      ``RANGE_0_100``:     ``pct = (t-tmin)/(tmax-tmin)*100``

    **degrees**
        * Arm joints: ``deg = (t - mid) * 360 / 4095``  where ``mid=(tmin+tmax)/2``
        * Gripper joints keep ``RANGE_0_100`` semantics so model action ``0``
          maps to the closed calibration end and ``100`` maps to open.

    **none**
        No conversion – returns an empty table so the caller does a pass-through.

    Parameters
    ----------
    calib_file : str
        Path to the calibration JSON produced by ``calibrate_arm``.
    joint_names : list[str]
        Ordered joint identifiers (e.g. ``["1","2",…,"6"]``).
    gripper_joints : list[str] | None
        Selects RANGE_0_100 semantics for these joints in ``range_m100_100``
        and ``degrees`` modes.
    norm_mode : str
        One of ``"range_m100_100"``, ``"degrees"``, ``"none"``.

    Returns
    -------
    list[JointConversionEntry]
        One ``(rad_min, rad_max, span, offset)`` per joint.  The linear
        mapping is::

            lerobot_val  = (rad - rad_min) / (rad_max - rad_min) * span + offset
            rad          = (lerobot_val - offset) / span * (rad_max - rad_min) + rad_min

        For *degrees* mode ``rad_min / rad_max`` are the rad equivalents of the
        degree endpoints, and ``span / offset`` encode the degree range so that
        the same linear formula works.
    """
    calibration = load_calibration_data(calib_file)
    return build_joint_conversion_table_from_calibration(
        calibration=calibration,
        joint_names=joint_names,
        gripper_joints=gripper_joints,
        norm_mode=norm_mode,
    )

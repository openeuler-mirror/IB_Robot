"""Pure joint conversion math shared by runtime publishers and legacy consumers.

An entry is (minimum, maximum, span, offset), with
``model = (physical - minimum) / (maximum - minimum) * span + offset``.
Positions use radians; wheel velocities use rad/s. No clamping is performed.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any

JointConversionEntry = tuple[float, float, float, float]
NORM_MODE_RANGE = "range_m100_100"
NORM_MODE_DEGREES = "degrees"
NORM_MODE_NONE = "none"
_MODEL_RESOLUTION = 4096
_TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)


def normalize_lerobot_norm_mode(norm_mode: str) -> str:
    mode = str(norm_mode or NORM_MODE_RANGE).strip().lower()
    if mode not in (NORM_MODE_RANGE, NORM_MODE_DEGREES, NORM_MODE_NONE):
        raise ValueError(
            f"Unsupported LeRobot normalization mode '{norm_mode}'. "
            f"Expected one of: {NORM_MODE_RANGE}, {NORM_MODE_DEGREES}, {NORM_MODE_NONE}"
        )
    return mode


def resolve_calibration_key(calibration: dict[str, Any], joint_name: str) -> str:
    if joint_name in calibration:
        return joint_name
    if joint_name.isdigit() and str(int(joint_name)) == joint_name:
        arm_key = f"joint{int(joint_name)}_arm"
        if arm_key in calibration:
            return arm_key
    raise KeyError(f"Joint '{joint_name}' missing from calibration data")


def build_joint_conversion_table_from_calibration(
    calibration: dict[str, Any],
    joint_names: list[str],
    gripper_joints: list[str] | None = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> list[JointConversionEntry]:
    """Preserve the hardware tick origin and LeRobot's centred 4095 degree scale."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    if mode == NORM_MODE_NONE:
        return []
    grippers = {str(name) for name in (gripper_joints or [])}
    table = []
    for joint_name in map(str, joint_names):
        entry = calibration[resolve_calibration_key(calibration, joint_name)]
        if not isinstance(entry, dict):
            raise ValueError(f"Calibration entry for joint '{joint_name}' must be an object")
        tick_min, tick_max = entry["range_min"], entry["range_max"]
        if any(type(value) is not int for value in (tick_min, tick_max)) or not 0 <= tick_min < tick_max <= 4095:
            raise ValueError(f"Calibration joint '{joint_name}' requires increasing integer ticks within [0, 4095]")
        rad_min = (tick_min - 2048.0) / _TICKS_PER_RAD
        rad_max = (tick_max - 2048.0) / _TICKS_PER_RAD
        if mode == NORM_MODE_DEGREES and joint_name not in grippers:
            mid = (tick_min + tick_max) / 2.0
            offset = (tick_min - mid) * 360.0 / (_MODEL_RESOLUTION - 1)
            span = (tick_max - mid) * 360.0 / (_MODEL_RESOLUTION - 1) - offset
        else:
            span, offset = (100.0, 0.0) if joint_name in grippers else (200.0, -100.0)
        table.append((rad_min, rad_max, span, offset))
    return table


def joint_limits_from_urdf(root: ET.Element, joint_name: str) -> tuple[float, float]:
    """Read URDF position limits, including ros2_control-only gripper joints."""
    for joint in root.findall(".//joint"):
        if joint.get("name") == joint_name:
            limit = joint.find("limit")
            if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
                return float(limit.get("lower")), float(limit.get("upper"))
    for joint in root.findall(".//ros2_control/joint"):
        if joint.get("name") != joint_name:
            continue
        command = joint.find("./command_interface[@name='position']")
        if command is not None:
            params = {param.get("name"): param.text for param in command.findall("param")}
            if params.get("min") is not None and params.get("max") is not None:
                return float(params["min"]), float(params["max"])
    raise KeyError(f"Joint '{joint_name}' missing lower/upper limits in URDF")


def build_joint_conversion_table_from_urdf(
    urdf_xml: str,
    joint_names: list[str],
    gripper_joints: list[str] | None = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> list[JointConversionEntry]:
    """Simulated ranges retain the 4096/4095 scale used by the tick-based path."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    if mode == NORM_MODE_NONE:
        return []
    root = ET.fromstring(urdf_xml)
    grippers = {str(name) for name in (gripper_joints or [])}
    table = []
    for joint_name in map(str, joint_names):
        rad_min, rad_max = joint_limits_from_urdf(root, joint_name)
        if not all(math.isfinite(value) for value in (rad_min, rad_max)) or rad_max <= rad_min:
            raise ValueError(f"Joint '{joint_name}' has invalid URDF limits: lower={rad_min}, upper={rad_max}")
        if joint_name in grippers:
            span, offset = 100.0, 0.0
        elif mode == NORM_MODE_DEGREES:
            span = math.degrees(rad_max - rad_min) * _MODEL_RESOLUTION / (_MODEL_RESOLUTION - 1)
            offset = -span / 2.0
        else:
            span, offset = 200.0, -100.0
        table.append((rad_min, rad_max, span, offset))
    return table

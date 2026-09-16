"""Trajectory template expansion for config-driven embodied skills."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ament_index_python.packages import get_package_share_directory


@dataclass(frozen=True)
class JointKinematics:
    name: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    lower: float
    upper: float


@dataclass(frozen=True)
class ArmKinematicModel:
    name: str
    joints: tuple[JointKinematics, ...]
    ee_origin_xyz: tuple[float, float, float]
    ee_origin_rpy: tuple[float, float, float]


MODEL_URDF_PATHS = {
    "so101_arm_v1": "$(find so101_description)/urdf/lerobot/so101/so101_base.xacro",
}


def _resolve_ros_path(path: str) -> str:
    """Resolve the small subset of ROS substitutions used by trajectory templates."""

    find_pattern = re.compile(r"\$\(find\s+(\w+)\)")
    for match in find_pattern.finditer(path):
        pkg_name = match.group(1)
        path = path.replace(f"$(find {pkg_name})", get_package_share_directory(pkg_name))

    env_pattern = re.compile(r"\$\(env\s+(\w+)\)")
    for match in env_pattern.finditer(path):
        var_name = match.group(1)
        path = path.replace(f"$(env {var_name})", os.environ.get(var_name, ""))

    return path


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _zero_waypoint(joint_names: list[str]) -> dict[str, Any]:
    return {
        "primitive_name": "move_to_joint_positions",
        "joint_positions": {joint_name: 0.0 for joint_name in joint_names},
        "duration_sec": 0,
    }


def _clone_waypoint(waypoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "primitive_name": str(waypoint["primitive_name"]),
        "joint_positions": {
            joint_name: float(position) for joint_name, position in waypoint["joint_positions"].items()
        },
        "duration_sec": float(waypoint["duration_sec"]),
    }


def _parse_xyz(value: str | None, field_name: str) -> tuple[float, float, float]:
    if not value:
        return (0.0, 0.0, 0.0)
    parts = value.split()
    if len(parts) != 3:
        raise ValueError(f"{field_name} must contain 3 floats")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4)] for row in range(4)]


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translation_matrix(xyz: tuple[float, float, float]) -> list[list[float]]:
    matrix = _identity()
    matrix[0][3], matrix[1][3], matrix[2][3] = xyz
    return matrix


def _rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0.0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0.0],
        [-sp, cp * sr, cp * cr, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_rotation_matrix(axis_xyz: tuple[float, float, float], angle: float) -> list[list[float]]:
    x, y, z = axis_xyz
    norm = math.sqrt((x * x) + (y * y) + (z * z))
    if norm == 0.0:
        raise ValueError("Joint axis cannot be zero vector")
    x /= norm
    y /= norm
    z /= norm
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _origin_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> list[list[float]]:
    return _matmul(_translation_matrix(xyz), _rotation_matrix_from_rpy(rpy))


def _transform_point(
    transform: list[list[float]], point_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> dict[str, float]:
    x, y, z = point_xyz
    return {
        "x": transform[0][0] * x + transform[0][1] * y + transform[0][2] * z + transform[0][3],
        "y": transform[1][0] * x + transform[1][1] * y + transform[1][2] * z + transform[1][3],
        "z": transform[2][0] * x + transform[2][1] * y + transform[2][2] * z + transform[2][3],
    }


def _midpoint(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {
        "x": (a["x"] + b["x"]) / 2.0,
        "y": (a["y"] + b["y"]) / 2.0,
        "z": (a["z"] + b["z"]) / 2.0,
    }


@cache
def load_kinematic_model(model_name: str) -> ArmKinematicModel:
    if model_name not in MODEL_URDF_PATHS:
        raise ValueError(f"Unsupported workspace model: {model_name}")

    urdf_path = Path(_resolve_ros_path(MODEL_URDF_PATHS[model_name]))
    root = ET.parse(urdf_path).getroot()

    joints: list[JointKinematics] = []
    for joint_name in ("1", "2", "3", "4", "5"):
        joint_element = root.find(f"./joint[@name='{joint_name}']")
        if joint_element is None:
            raise ValueError(f"Joint '{joint_name}' not found in {urdf_path}")
        origin_element = joint_element.find("origin")
        axis_element = joint_element.find("axis")
        limit_element = joint_element.find("limit")
        if origin_element is None or axis_element is None or limit_element is None:
            raise ValueError(f"Joint '{joint_name}' is missing origin/axis/limit in {urdf_path}")

        joints.append(
            JointKinematics(
                name=joint_name,
                origin_xyz=_parse_xyz(origin_element.get("xyz"), f"joint {joint_name} origin xyz"),
                origin_rpy=_parse_xyz(origin_element.get("rpy"), f"joint {joint_name} origin rpy"),
                axis_xyz=_parse_xyz(axis_element.get("xyz"), f"joint {joint_name} axis xyz"),
                lower=_as_float(limit_element.get("lower"), f"joint {joint_name} lower"),
                upper=_as_float(limit_element.get("upper"), f"joint {joint_name} upper"),
            )
        )

    ee_joint = root.find("./joint[@name='6']")
    if ee_joint is None or ee_joint.find("origin") is None:
        raise ValueError(f"Joint '6' not found in {urdf_path}")
    ee_origin = ee_joint.find("origin")
    assert ee_origin is not None

    return ArmKinematicModel(
        name=model_name,
        joints=tuple(joints),
        ee_origin_xyz=_parse_xyz(ee_origin.get("xyz"), "joint 6 origin xyz"),
        ee_origin_rpy=_parse_xyz(ee_origin.get("rpy"), "joint 6 origin rpy"),
    )


def compute_workspace_checkpoints(
    model_name: str, joint_positions: dict[str, float] | dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Compute workspace checkpoints for a joint pose using the configured arm model."""
    model = load_kinematic_model(model_name)
    transform = _identity()
    joint_points: dict[str, dict[str, float]] = {}

    for joint in model.joints:
        transform = _matmul(transform, _origin_transform(joint.origin_xyz, joint.origin_rpy))
        joint_points[joint.name] = _transform_point(transform)
        transform = _matmul(
            transform,
            _axis_rotation_matrix(
                joint.axis_xyz, _as_float(joint_positions[joint.name], f"joint_positions.{joint.name}")
            ),
        )

    ee_transform = _matmul(transform, _origin_transform(model.ee_origin_xyz, model.ee_origin_rpy))
    ee_point = _transform_point(ee_transform)

    return {
        "joint1": joint_points["1"],
        "joint2": joint_points["2"],
        "joint3": joint_points["3"],
        "joint4": joint_points["4"],
        "joint5": joint_points["5"],
        "upper_arm_mid": _midpoint(joint_points["2"], joint_points["3"]),
        "forearm_mid": _midpoint(joint_points["3"], joint_points["4"]),
        "wrist_mid": _midpoint(joint_points["4"], joint_points["5"]),
        "ee": ee_point,
    }


def _pose_within_joint_limits(model: ArmKinematicModel, joint_positions: dict[str, float]) -> bool:
    return all(joint.lower <= joint_positions[joint.name] <= joint.upper for joint in model.joints)


def _point_within_bounds(point: dict[str, float], bounds: dict[str, Any]) -> bool:
    for axis_name in ("x", "y", "z"):
        if axis_name not in bounds:
            continue
        axis_bounds = bounds[axis_name]
        if not isinstance(axis_bounds, list) or len(axis_bounds) != 2:
            raise ValueError(f"workspace bound '{axis_name}' must be a [min, max] list")
        minimum = _as_float(axis_bounds[0], f"workspace_limits.points.{axis_name}[0]")
        maximum = _as_float(axis_bounds[1], f"workspace_limits.points.{axis_name}[1]")
        if point[axis_name] < minimum or point[axis_name] > maximum:
            return False
    return True


def _pose_within_workspace(workspace_limits: dict[str, Any], joint_positions: dict[str, float]) -> bool:
    model_name = workspace_limits.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("trajectory_template.workspace_limits.model must be a non-empty string")
    points = workspace_limits.get("points", {})
    if not isinstance(points, dict) or not points:
        raise ValueError("trajectory_template.workspace_limits.points must be a non-empty mapping")

    checkpoints = compute_workspace_checkpoints(model_name, joint_positions)
    return all(_point_within_bounds(checkpoints[point_name], bounds) for point_name, bounds in points.items())


def _round_pose(joint_positions: dict[str, float]) -> dict[str, float]:
    return {joint_name: round(value, 2) for joint_name, value in joint_positions.items()}


def _pose_is_valid(
    workspace_limits: dict[str, Any] | None, model: ArmKinematicModel | None, joint_positions: dict[str, float]
) -> bool:
    if model is not None and not _pose_within_joint_limits(model, joint_positions):
        return False
    if workspace_limits is None:
        return True
    return _pose_within_workspace(workspace_limits, joint_positions)


def _scale_pose_into_workspace(
    base_pose: dict[str, float],
    candidate_pose: dict[str, float],
    workspace_limits: dict[str, Any] | None,
) -> dict[str, float]:
    if workspace_limits is None:
        return candidate_pose

    model_name = workspace_limits.get("model")
    model = load_kinematic_model(model_name)
    if not _pose_is_valid(workspace_limits, model, base_pose):
        raise ValueError("trajectory_template.base_pose must already satisfy workspace_limits and joint limits")
    if _pose_is_valid(workspace_limits, model, candidate_pose):
        return candidate_pose

    deltas = {joint_name: candidate_pose[joint_name] - base_pose[joint_name] for joint_name in base_pose}
    lower = 0.0
    upper = 1.0
    best_alpha = 0.0

    for _ in range(32):
        alpha = (lower + upper) / 2.0
        scaled_pose = {joint_name: base_pose[joint_name] + (deltas[joint_name] * alpha) for joint_name in base_pose}
        if _pose_is_valid(workspace_limits, model, scaled_pose):
            best_alpha = alpha
            lower = alpha
        else:
            upper = alpha

    for step in range(21):
        alpha = max(best_alpha - (step * 0.01), 0.0)
        rounded_pose = _round_pose(
            {joint_name: base_pose[joint_name] + (deltas[joint_name] * alpha) for joint_name in base_pose}
        )
        if _pose_is_valid(workspace_limits, model, rounded_pose):
            return rounded_pose
    return _round_pose(base_pose)


def generate_wave_dance_v1(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a sinusoidal wave dance template into concrete joint waypoints."""
    active_waypoint_count = int(template.get("active_waypoint_count", 0))
    repeat_count = int(template.get("repeat_count", 1))
    zero_hold_count = int(template.get("zero_hold_count", 0))
    base_pose = template.get("base_pose")
    joints = template.get("joints")
    workspace_limits = template.get("workspace_limits")

    if active_waypoint_count <= 0:
        raise ValueError("trajectory_template.active_waypoint_count must be positive")
    if repeat_count <= 0:
        raise ValueError("trajectory_template.repeat_count must be positive")
    if zero_hold_count < 0:
        raise ValueError("trajectory_template.zero_hold_count cannot be negative")
    if not isinstance(base_pose, dict) or not base_pose:
        raise ValueError("trajectory_template.base_pose must be a non-empty mapping")
    if not isinstance(joints, dict) or not joints:
        raise ValueError("trajectory_template.joints must be a non-empty mapping")
    if workspace_limits is not None and not isinstance(workspace_limits, dict):
        raise ValueError("trajectory_template.workspace_limits must be a mapping")

    joint_names = [str(joint_name) for joint_name in base_pose]
    base_pose_values = {
        joint_name: _as_float(base_pose[joint_name], f"trajectory_template.base_pose.{joint_name}")
        for joint_name in joint_names
    }
    cycle_waypoints: list[dict[str, Any]] = []

    for index in range(active_waypoint_count):
        theta = (2.0 * math.pi * index) / active_waypoint_count
        candidate_pose: dict[str, float] = {}

        for joint_name in joint_names:
            joint_config = joints.get(joint_name, {})
            if not isinstance(joint_config, dict):
                raise ValueError(f"trajectory_template.joints.{joint_name} must be a mapping")

            terms = joint_config.get("terms", [])
            if not isinstance(terms, list):
                raise ValueError(f"trajectory_template.joints.{joint_name}.terms must be a list")

            value = base_pose_values[joint_name]
            for term_index, term in enumerate(terms):
                if not isinstance(term, dict):
                    raise ValueError(f"trajectory_template.joints.{joint_name}.terms[{term_index}] must be a mapping")
                amplitude = _as_float(
                    term.get("amplitude", 0.0),
                    f"trajectory_template.joints.{joint_name}.terms[{term_index}].amplitude",
                )
                harmonic = _as_float(
                    term.get("harmonic", 1.0),
                    f"trajectory_template.joints.{joint_name}.terms[{term_index}].harmonic",
                )
                phase = _as_float(
                    term.get("phase", 0.0),
                    f"trajectory_template.joints.{joint_name}.terms[{term_index}].phase",
                )
                value += amplitude * math.sin((harmonic * theta) + phase)
            candidate_pose[joint_name] = value

        safe_pose = _scale_pose_into_workspace(base_pose_values, candidate_pose, workspace_limits)
        cycle_waypoints.append(
            {
                "primitive_name": "move_to_joint_positions",
                "joint_positions": _round_pose(safe_pose),
                "duration_sec": 0,
            }
        )

    waypoints: list[dict[str, Any]] = []
    for _ in range(repeat_count):
        waypoints.extend(_clone_waypoint(waypoint) for waypoint in cycle_waypoints)

    if zero_hold_count > 0:
        if not waypoints:
            raise ValueError("zero_hold_count requires at least one active waypoint")
        hold_waypoint = _clone_waypoint(waypoints[-1])
        for _ in range(zero_hold_count):
            waypoints.append(_clone_waypoint(hold_waypoint))

    return waypoints


def generate_single_joint_wave_v1(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a single-joint sinusoidal wave template into concrete waypoints."""
    active_waypoint_count = int(template.get("active_waypoint_count", 16))
    repeat_count = int(template.get("repeat_count", 1))
    base_pose = template.get("base_pose")
    joint_name = str(template.get("joint", "")).strip()
    amplitude = _as_float(template.get("amplitude", 0.0), "trajectory_template.amplitude")
    phase = _as_float(template.get("phase", 0.0), "trajectory_template.phase")
    workspace_limits = template.get("workspace_limits")

    if active_waypoint_count <= 0:
        raise ValueError("trajectory_template.active_waypoint_count must be positive")
    if repeat_count <= 0:
        raise ValueError("trajectory_template.repeat_count must be positive")
    if not isinstance(base_pose, dict) or not base_pose:
        raise ValueError("trajectory_template.base_pose must be a non-empty mapping")
    if not joint_name:
        raise ValueError("trajectory_template.joint must be a non-empty string")
    if joint_name not in base_pose:
        raise ValueError("trajectory_template.joint must exist in base_pose")
    if workspace_limits is not None and not isinstance(workspace_limits, dict):
        raise ValueError("trajectory_template.workspace_limits must be a mapping")

    joint_names = [str(name) for name in base_pose]
    base_pose_values = {
        name: _as_float(base_pose[name], f"trajectory_template.base_pose.{name}") for name in joint_names
    }
    cycle_waypoints: list[dict[str, Any]] = []

    for index in range(active_waypoint_count):
        theta = (2.0 * math.pi * index) / active_waypoint_count
        candidate_pose = dict(base_pose_values)
        candidate_pose[joint_name] = base_pose_values[joint_name] + (amplitude * math.sin(theta + phase))
        safe_pose = _scale_pose_into_workspace(base_pose_values, candidate_pose, workspace_limits)
        cycle_waypoints.append(
            {
                "primitive_name": "move_to_joint_positions",
                "joint_positions": _round_pose(safe_pose),
                "duration_sec": 0,
            }
        )

    waypoints: list[dict[str, Any]] = []
    for _ in range(repeat_count):
        waypoints.extend(_clone_waypoint(waypoint) for waypoint in cycle_waypoints)

    return waypoints


def expand_trajectory_template(template: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a supported trajectory template into concrete joint waypoints."""
    template_type = template.get("type")
    if template_type == "wave_dance_v1":
        return generate_wave_dance_v1(template)
    if template_type == "single_joint_wave_v1":
        return generate_single_joint_wave_v1(template)
    raise ValueError(f"Unsupported trajectory_template type: {template_type}")

"""Robot-agnostic plane and quaternion geometry for grasp execution.

Extracted from the former ``so101_geometry`` module (design D8): generic
math stays in ``manipulation_execution``; SO-101-specific gripper mesh
geometry and joint-5 wrist guards live in ``so101_suite``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from manipulation_execution.grasp_geometry import Quaternion, quaternion_matrix

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class TablePlane:
    normal: Vector3
    offset: float
    inlier_ratio: float = 0.0


def transform_point(transform: np.ndarray, point: Iterable[float]) -> Vector3:
    value = np.asarray(transform, dtype=np.float64) @ np.array([*point, 1.0], dtype=np.float64)
    return (float(value[0]), float(value[1]), float(value[2]))


def transform_table_plane(
    transform: np.ndarray,
    normal: Iterable[float],
    offset: float,
    *,
    inlier_ratio: float = 0.0,
) -> TablePlane:
    """Transform ``normal dot point + offset = 0`` into the target frame."""

    matrix = np.asarray(transform, dtype=np.float64)
    source_normal = np.asarray(list(normal), dtype=np.float64)
    source_norm = float(np.linalg.norm(source_normal))
    if source_norm <= 1e-9:
        raise ValueError("table plane normal must be non-zero")
    source_normal /= source_norm
    source_offset = float(offset) / source_norm
    target_normal = matrix[:3, :3] @ source_normal
    target_offset = source_offset - float(np.dot(target_normal, matrix[:3, 3]))
    return TablePlane(
        normal=(float(target_normal[0]), float(target_normal[1]), float(target_normal[2])),
        offset=target_offset,
        inlier_ratio=float(inlier_ratio),
    )


def orient_table_plane_upward(plane: TablePlane) -> TablePlane:
    """Canonicalize a table plane so its safe half-space faces base +Z."""

    normal = np.asarray(plane.normal, dtype=np.float64)
    if float(np.dot(normal, (0.0, 0.0, 1.0))) < 0.0:
        return TablePlane(
            normal=tuple(float(value) for value in -normal),
            offset=-float(plane.offset),
            inlier_ratio=float(plane.inlier_ratio),
        )
    return plane


def axis_error_deg(
    planned_quaternion: Quaternion,
    actual_quaternion: Quaternion,
    axis_ee: Iterable[float],
) -> float:
    """Return directed angular error between planned and actual EE axes."""

    axis = np.asarray(list(axis_ee), dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("axis must be non-zero")
    axis /= norm
    planned = quaternion_matrix(planned_quaternion) @ axis
    actual = quaternion_matrix(actual_quaternion) @ axis
    return math.degrees(math.acos(float(np.clip(np.dot(planned, actual), -1.0, 1.0))))


def quaternion_error_deg(planned_quaternion: Quaternion, actual_quaternion: Quaternion) -> float:
    """Return the shortest full-orientation error between two quaternions."""

    planned = np.asarray(planned_quaternion, dtype=np.float64)
    actual = np.asarray(actual_quaternion, dtype=np.float64)
    planned_norm = float(np.linalg.norm(planned))
    actual_norm = float(np.linalg.norm(actual))
    if planned_norm <= 1e-9 or actual_norm <= 1e-9:
        raise ValueError("quaternion must be non-zero")
    dot = abs(float(np.dot(planned / planned_norm, actual / actual_norm)))
    return math.degrees(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))

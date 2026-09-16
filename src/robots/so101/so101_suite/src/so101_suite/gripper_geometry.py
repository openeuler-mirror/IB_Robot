"""SO-101 gripper mesh geometry (STL loading, convex hull, jaw motion).

Self-contained: does not import from manipulation_execution to avoid a
circular package dependency. The minimal math it needs is local.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError
except ImportError:  # pragma: no cover - full vertices remain geometrically correct.
    ConvexHull = None
    QhullError = ValueError

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _quaternion_matrix(q: Quaternion) -> np.ndarray:
    """Build a 3x3 rotation matrix from a quaternion [x, y, z, w]."""
    x, y, z, w = (float(value) for value in q)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _euler_xyz_matrix(rpy: Iterable[float]) -> np.ndarray:
    """Build a 3x3 rotation matrix from Euler XYZ angles."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class _TablePlane:
    normal: Vector3
    offset: float
    inlier_ratio: float = 0.0


def _matrix_from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _euler_xyz_matrix(rpy)
    matrix[:3, 3] = [float(value) for value in xyz]
    return matrix


@lru_cache(maxsize=4)
def _read_stl_vertices(path: str, max_triangles: int = 2500) -> np.ndarray:
    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    data = mesh_path.read_bytes()
    triangles = np.zeros((0, 3, 3), dtype=np.float64)
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        if triangle_count > 0 and 84 + triangle_count * 50 == len(data):
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            raw = np.frombuffer(data, dtype=dtype, count=triangle_count, offset=84)
            triangles = raw["vertices"].astype(np.float64)
    if len(triangles) == 0:
        vertices = []
        for line in data.decode("ascii", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(vertices) >= 3:
            triangles = np.asarray(vertices[: len(vertices) // 3 * 3], dtype=np.float64).reshape(-1, 3, 3)
    if len(triangles) == 0:
        raise ValueError(f"No STL triangles found in {mesh_path}")
    step = max(1, int(math.ceil(len(triangles) / max_triangles)))
    return triangles[::step].reshape(-1, 3)


@lru_cache(maxsize=4)
def _read_stl_convex_hull_vertices(path: str, max_triangles: int = 2500) -> np.ndarray:
    vertices = np.unique(_read_stl_vertices(path, max_triangles), axis=0)
    if ConvexHull is None or len(vertices) < 4:
        return vertices
    try:
        return vertices[ConvexHull(vertices).vertices]
    except QhullError:
        return vertices


def _width_to_jaw_angle(width_m: float | None) -> float:
    if width_m is None or not math.isfinite(float(width_m)):
        return 0.45
    normalized = (float(width_m) - 0.008) / (0.080 - 0.008)
    return max(0.0, min(1.0, normalized))


@lru_cache(maxsize=4)
def _gripper_geometry_data(mesh_directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    directory = Path(mesh_directory)
    fixed_vertices = _read_stl_convex_hull_vertices(str(directory / "wrist_roll_follower_so101_v1.stl"))
    moving_vertices = _read_stl_convex_hull_vertices(str(directory / "moving_jaw_so101_v1.stl"))
    gripper_visual = _matrix_from_xyz_rpy(
        (5.55112e-17, -0.000218214, 0.000949706),
        (-3.14159, -5.55112e-17, -9.17912e-24),
    )
    gripper_to_jaw = _matrix_from_xyz_rpy((0.0202, 0.0188, -0.0234), (1.5708, 0.209440, 0.000001))
    jaw_visual = _matrix_from_xyz_rpy((-5.55112e-17, -1.94746e-17, 0.0189), (9.53145e-17, -4.66093e-24, 0.0))
    return fixed_vertices, moving_vertices, gripper_visual, gripper_to_jaw, jaw_visual


def gripper_mesh_vertices(
    mesh_directory: Path,
    xyz: Vector3,
    quaternion: Quaternion,
    width_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed and moving SO101 mesh vertices in the base frame."""

    base_to_gripper = np.eye(4, dtype=np.float64)
    base_to_gripper[:3, :3] = _quaternion_matrix(quaternion)
    base_to_gripper[:3, 3] = xyz
    fixed_local, jaw_local, gripper_visual, gripper_to_jaw, jaw_visual = _gripper_geometry_data(str(mesh_directory))
    jaw_motion = _matrix_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, _width_to_jaw_angle(width_m)))

    def apply(transform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        vertices_h = np.hstack([vertices, np.ones((len(vertices), 1), dtype=np.float64)])
        return (transform[:3, :] @ vertices_h.T).T

    return (
        apply(base_to_gripper @ gripper_visual, fixed_local),
        apply(base_to_gripper @ gripper_to_jaw @ jaw_motion @ jaw_visual, jaw_local),
    )


def gripper_mesh_min_z(
    mesh_directory: Path,
    xyz: Vector3,
    quaternion: Quaternion,
    width_m: float | None,
) -> float:
    meshes = gripper_mesh_vertices(mesh_directory, xyz, quaternion, width_m)
    return min(float(vertices[:, 2].min()) for vertices in meshes if len(vertices))


def tabletop_clearance(
    mesh_directory: Path,
    approach: Vector3,
    grasp: Vector3,
    quaternion: Quaternion,
    width_m: float | None,
    plane: _TablePlane,
    *,
    sweep_steps: int,
) -> float:
    del sweep_steps
    normal = np.asarray(plane.normal, dtype=np.float64)
    grasp_clearance = math.inf
    for vertices in gripper_mesh_vertices(mesh_directory, grasp, quaternion, width_m):
        if len(vertices):
            grasp_clearance = min(grasp_clearance, float(np.min(vertices @ normal + plane.offset)))
    if not math.isfinite(grasp_clearance):
        raise ValueError("SO101 gripper meshes contain no vertices")
    translation = np.asarray(approach, dtype=np.float64) - np.asarray(grasp, dtype=np.float64)
    approach_clearance = grasp_clearance + float(translation @ normal)
    return min(grasp_clearance, approach_clearance)


def gripper_geometry_metrics_batch(
    mesh_directory: Path,
    candidates: Sequence[tuple[Vector3, Vector3, Quaternion, float | None]],
    plane: _TablePlane | None,
    *,
    clearance_threshold_m: float = 0.0,
    threshold_fallback_m: float = 1e-5,
) -> list[tuple[float, float | None]]:
    """Vectorize SO101 mesh height and tabletop checks across candidates."""

    if not candidates:
        return []
    fixed_local, moving_local, gripper_visual, gripper_to_jaw, jaw_visual = _gripper_geometry_data(str(mesh_directory))
    fixed_transforms = []
    moving_transforms = []
    for _, grasp, quaternion, width_m in candidates:
        base_to_gripper = np.eye(4, dtype=np.float64)
        base_to_gripper[:3, :3] = _quaternion_matrix(quaternion)
        base_to_gripper[:3, 3] = grasp
        jaw_motion = _matrix_from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, _width_to_jaw_angle(width_m)))
        fixed_transforms.append(base_to_gripper @ gripper_visual)
        moving_transforms.append(base_to_gripper @ gripper_to_jaw @ jaw_motion @ jaw_visual)

    fixed_transform_array = np.stack(fixed_transforms)
    moving_transform_array = np.stack(moving_transforms)
    fixed_world = (
        np.matmul(fixed_local[None, :, :], np.swapaxes(fixed_transform_array[:, :3, :3], 1, 2))
        + fixed_transform_array[:, None, :3, 3]
    )
    moving_world = (
        np.matmul(moving_local[None, :, :], np.swapaxes(moving_transform_array[:, :3, :3], 1, 2))
        + moving_transform_array[:, None, :3, 3]
    )
    minimum_z = np.minimum(np.min(fixed_world[:, :, 2], axis=1), np.min(moving_world[:, :, 2], axis=1))
    if plane is None:
        return [(float(value), None) for value in minimum_z]

    normal = np.asarray(plane.normal, dtype=np.float64)
    fixed_grasp = np.min(fixed_world @ normal + plane.offset, axis=1)
    moving_grasp = np.min(moving_world @ normal + plane.offset, axis=1)
    grasp_clearance = np.minimum(fixed_grasp, moving_grasp)
    approaches = np.asarray([candidate[0] for candidate in candidates], dtype=np.float64)
    grasps = np.asarray([candidate[1] for candidate in candidates], dtype=np.float64)
    approach_clearance = grasp_clearance + (approaches - grasps) @ normal
    clearances = np.minimum(grasp_clearance, approach_clearance)

    near_threshold = np.abs(clearances - float(clearance_threshold_m)) <= max(0.0, float(threshold_fallback_m))
    for index in np.flatnonzero(near_threshold):
        approach, grasp, quaternion, width_m = candidates[int(index)]
        minimum_z[index] = gripper_mesh_min_z(mesh_directory, grasp, quaternion, width_m)
        clearances[index] = tabletop_clearance(
            mesh_directory,
            approach,
            grasp,
            quaternion,
            width_m,
            plane,
            sweep_steps=1,
        )
    return [(float(z_value), float(clearance)) for z_value, clearance in zip(minimum_z, clearances, strict=True)]

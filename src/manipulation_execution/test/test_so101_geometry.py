import math
from pathlib import Path

import numpy as np

from manipulation_execution.so101_geometry import (
    TablePlane,
    axis_error_deg,
    gripper_geometry_metrics_batch,
    gripper_mesh_min_z,
    orient_table_plane_upward,
    quaternion_error_deg,
    tabletop_clearance,
    transform_point,
    transform_table_plane,
)


def test_transform_table_plane_preserves_signed_distance():
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [0.2, -0.1, 0.3]
    plane = transform_table_plane(transform, (0.0, 0.0, 1.0), -0.5)

    source_point = (0.1, 0.2, 0.5)
    target_point = transform_point(transform, source_point)
    signed = np.dot(plane.normal, target_point) + plane.offset
    assert math.isclose(signed, 0.0, abs_tol=1e-9)


def test_orient_table_plane_upward_flips_downward_normal_and_offset():
    plane = TablePlane(normal=(0.1, -0.2, -0.9), offset=-0.4, inlier_ratio=0.75)

    oriented = orient_table_plane_upward(plane)

    assert oriented == TablePlane(normal=(-0.1, 0.2, 0.9), offset=0.4, inlier_ratio=0.75)
    point_on_plane = np.array((4.0, 0.0, 0.0), dtype=np.float64)
    assert math.isclose(np.dot(plane.normal, point_on_plane) + plane.offset, 0.0, abs_tol=1e-9)
    assert math.isclose(np.dot(oriented.normal, point_on_plane) + oriented.offset, 0.0, abs_tol=1e-9)


def test_orient_table_plane_upward_leaves_upward_plane_unchanged():
    plane = TablePlane(normal=(0.1, -0.2, 0.9), offset=-0.4, inlier_ratio=0.75)

    assert orient_table_plane_upward(plane) is plane


def test_axis_error_uses_directed_closing_axis():
    half_turn_z = (0.0, 0.0, 1.0, 0.0)
    assert math.isclose(axis_error_deg((0.0, 0.0, 0.0, 1.0), half_turn_z, (1.0, 0.0, 0.0)), 180.0)


def test_quaternion_error_treats_sign_equivalent_quaternions_as_equal():
    assert math.isclose(quaternion_error_deg((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0)), 0.0)


def test_batch_gripper_geometry_matches_scalar_checks():
    mesh_directory = (
        Path(__file__).parents[2] / "robots" / "so101" / "so101_description" / "meshes" / "lerobot" / "so101"
    )
    plane = TablePlane(normal=(0.0, 0.0, 1.0), offset=0.02)
    candidates = [
        ((0.10, -0.16, 0.18), (0.10, -0.16, 0.08), (0.0, 0.0, 0.0, 1.0), 0.035),
        ((0.14, -0.12, 0.16), (0.14, -0.12, 0.06), (0.0, 0.0, 0.0, 1.0), 0.020),
    ]

    batch = gripper_geometry_metrics_batch(mesh_directory, candidates, plane)

    for candidate, (batch_min_z, batch_clearance) in zip(candidates, batch, strict=True):
        approach, grasp, quaternion, width = candidate
        scalar_min_z = gripper_mesh_min_z(mesh_directory, grasp, quaternion, width)
        scalar_clearance = tabletop_clearance(
            mesh_directory,
            approach,
            grasp,
            quaternion,
            width,
            plane,
            sweep_steps=5,
        )
        assert math.isclose(batch_min_z, scalar_min_z, abs_tol=1e-10)
        assert batch_clearance is not None
        assert math.isclose(batch_clearance, scalar_clearance, abs_tol=1e-10)

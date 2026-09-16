import math

import numpy as np

from manipulation_execution.geometry import (
    TablePlane,
    axis_error_deg,
    orient_table_plane_upward,
    quaternion_error_deg,
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

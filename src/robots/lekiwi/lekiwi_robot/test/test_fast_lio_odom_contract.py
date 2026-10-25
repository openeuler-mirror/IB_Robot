"""Pure contracts for adapting official FAST-LIO odometry to robot frames."""

import math

import pytest

from lekiwi_robot.fast_lio_odom_contract import (
    base_pose_from_body_pose,
    compose_pose,
    invert_transform,
    project_planar_pose,
    transform_twist,
    validate_odometry_sample,
    validate_odometry_timestamp,
)


def test_official_fast_lio_frames_are_accepted():
    assert validate_odometry_sample("camera_init", "body", [0.0, 1.0, -2.0, 1.0]) is None


def test_unexpected_fast_lio_frames_are_rejected():
    assert (
        validate_odometry_sample("odom", "base_link", [0.0, 0.0, 0.0, 1.0])
        == "expected camera_init -> body, got odom -> base_link"
    )


def test_non_finite_fast_lio_sample_is_rejected():
    assert (
        validate_odometry_sample("camera_init", "body", [0.0, float("nan"), 0.0, 1.0])
        == "odometry contains non-finite values"
    )


def test_zero_fast_lio_timestamp_is_rejected():
    assert validate_odometry_timestamp(0, 0, now_sec=10.0) == "odometry timestamp is zero"


def test_fast_lio_timestamp_far_in_the_future_is_rejected():
    assert (
        validate_odometry_timestamp(12, 0, now_sec=10.0, max_future_skew_sec=0.1)
        == "odometry timestamp is too far in the future"
    )


def test_identity_body_to_base_keeps_the_lio_pose():
    position, orientation = base_pose_from_body_pose(
        (1.0, 2.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    assert position == pytest.approx((1.0, 2.0, 0.0))
    assert orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_yawed_mount_keeps_the_odometry_frame_base_aligned():
    """The MID-360 mount regression: a +90 degree sensor mount.

    The base pose is the body pose conjugated by the mount. With the real
    mount (8 cm forward, 20 cm up, +90 deg yaw) a pure base translation or
    rotation must come back exactly as the base motion — a right-composition
    alone would leave the pose parented in the sensor-rotated camera_init
    frame (the translation would come out rotated -90 degrees).
    """
    half = math.sqrt(0.5)
    # body_to_base = the mount's inverse (lekiwi_mid360_mount.yaml)
    body_to_base_translation = (0.0, -0.08, -0.2)
    body_to_base_rotation = (0.0, 0.0, -half, half)

    # the robot translated +1 m along base x: FAST-LIO's body pose is that
    # translation rotated into the sensor frame (P = M^-1 o O o M)
    position, orientation = base_pose_from_body_pose(
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        body_to_base_translation,
        body_to_base_rotation,
    )
    assert position == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert orientation == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-12)

    # the robot rotated 90 deg in place
    position, orientation = base_pose_from_body_pose(
        (-0.08, -0.08, 0.0),
        (0.0, 0.0, half, half),
        body_to_base_translation,
        body_to_base_rotation,
    )
    assert position == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert orientation == pytest.approx((0.0, 0.0, half, half), abs=1e-12)


def test_invert_transform_round_trips_a_rigid_transform():
    translation = (-0.08, 0.0, 0.2)
    rotation = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    inverse_translation, inverse_rotation = invert_transform(translation, rotation)

    position, orientation = compose_pose(
        *compose_pose((1.0, -2.0, 0.5), (0.1, -0.2, 0.3, 0.9), translation, rotation),
        inverse_translation,
        inverse_rotation,
    )
    assert position == pytest.approx((1.0, -2.0, 0.5))
    assert orientation == pytest.approx(_normalize_quaternion((0.1, -0.2, 0.3, 0.9)))


def _normalize_quaternion(quaternion):
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)


def test_planar_projection_removes_height_roll_and_pitch():
    position, orientation = project_planar_pose((1.2, -0.4, 0.35), (0.2, -0.1, 0.3, 0.9))

    assert position == (1.2, -0.4, 0.0)
    assert orientation[0:2] == (0.0, 0.0)
    assert math.sqrt(sum(value * value for value in orientation)) == pytest.approx(1.0)


def test_twist_includes_body_to_base_lever_arm_and_rotation():
    linear, angular = transform_twist(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
    )

    assert linear == pytest.approx((0.0, 1.0, 0.0))
    assert angular == pytest.approx((0.0, 0.0, 2.0))


@pytest.mark.parametrize(
    "rotation",
    [(0.0, 0.0, 0.0, 0.0), (0.0, float("inf"), 0.0, 1.0)],
)
def test_invalid_body_to_base_rotation_is_rejected(rotation):
    with pytest.raises(ValueError, match="quaternion must be finite and non-zero"):
        compose_pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), rotation)

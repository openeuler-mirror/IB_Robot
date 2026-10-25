"""Pure transformation and validation contracts for official FAST-LIO odometry."""

import math
from collections.abc import Iterable


def _normalize(quaternion: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    return tuple(value / norm for value in quaternion)


def _multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate(quaternion: tuple[float, ...], vector: tuple[float, ...]) -> tuple[float, ...]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def transform_twist(
    linear: tuple[float, ...],
    angular: tuple[float, ...],
    child_translation: tuple[float, ...],
    child_rotation: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Transform a body twist to a base frame, including the lever-arm term."""
    rotation = _normalize(child_rotation)
    inverse_rotation = (-rotation[0], -rotation[1], -rotation[2], rotation[3])
    ax, ay, az = angular
    tx, ty, tz = child_translation
    lever_arm = (ay * tz - az * ty, az * tx - ax * tz, ax * ty - ay * tx)
    body_linear = tuple(linear[index] + lever_arm[index] for index in range(3))
    return _rotate(inverse_rotation, body_linear), _rotate(inverse_rotation, angular)


def compose_pose(
    position: tuple[float, ...],
    orientation: tuple[float, ...],
    child_translation: tuple[float, ...],
    child_rotation: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compose camera_init->body with body->base_link."""
    orientation = _normalize(orientation)
    child_rotation = _normalize(child_rotation)
    rotated = _rotate(orientation, child_translation)
    output_position = tuple(position[index] + rotated[index] for index in range(3))
    return output_position, _normalize(_multiply(orientation, child_rotation))


def invert_transform(
    translation: tuple[float, ...], quaternion: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The inverse of a rigid transform given as (translation, xyzw quaternion)."""
    rotation = _normalize(quaternion)
    inverse_rotation = (-rotation[0], -rotation[1], -rotation[2], rotation[3])
    return tuple(-value for value in _rotate(inverse_rotation, translation)), inverse_rotation


def base_pose_from_body_pose(
    position: tuple[float, ...],
    orientation: tuple[float, ...],
    body_to_base_translation: tuple[float, ...],
    body_to_base_rotation: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The ``odom -> base_link`` pose for FAST-LIO's ``camera_init -> body`` pose.

    FAST-LIO reports the body (= lidar/IMU) pose in the ``camera_init`` frame
    the body itself defined at startup, and the body sits at the sensor mount
    relative to the base (``body_to_base`` = the mount's inverse). The base
    pose in the base's own startup frame is the body pose conjugated by the
    mount — a right-composition alone would leave the result parented in the
    sensor-rotated ``camera_init`` frame, so a yawed mount (the MID-360 sits
    at +90 degrees) would yaw the whole odometry frame with it.
    """
    pose_orientation = _normalize(orientation)
    body_to_base_orientation = _normalize(body_to_base_rotation)
    # P o X: the base's pose in the camera_init frame
    camera_init_position, camera_init_orientation = compose_pose(
        position, pose_orientation, body_to_base_translation, body_to_base_orientation
    )
    # M o (P o X) with M the mount (the inverse of body_to_base)
    mount_translation, mount_orientation = invert_transform(body_to_base_translation, body_to_base_orientation)
    base_position = tuple(
        mount_translation[index] + value for index, value in enumerate(_rotate(mount_orientation, camera_init_position))
    )
    return base_position, _normalize(_multiply(mount_orientation, camera_init_orientation))


def project_planar_pose(
    position: tuple[float, ...], orientation: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Project a 3D LIO pose onto the planar navigation odometry contract."""
    qx, qy, qz, qw = _normalize(orientation)
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return (position[0], position[1], 0.0), (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def validate_odometry_sample(
    frame_id: str,
    child_frame_id: str,
    values: Iterable[float],
    odom_frame: str = "camera_init",
    body_frame: str = "body",
) -> str | None:
    """Validate the fixed frame and finite-value contract of official FAST-LIO."""
    if frame_id != odom_frame or child_frame_id != body_frame:
        return f"expected {odom_frame} -> {body_frame}, got {frame_id} -> {child_frame_id}"
    if not all(math.isfinite(value) for value in values):
        return "odometry contains non-finite values"
    return None


def validate_odometry_timestamp(
    stamp_sec: int,
    stamp_nanosec: int,
    *,
    now_sec: float,
    max_future_skew_sec: float = 0.1,
) -> str | None:
    """Reject only missing or clearly future-dated source timestamps."""
    stamp = float(stamp_sec) + float(stamp_nanosec) / 1_000_000_000.0
    if stamp == 0.0:
        return "odometry timestamp is zero"
    if not math.isfinite(stamp):
        return "odometry timestamp is non-finite"
    if stamp > now_sec + max_future_skew_sec:
        return "odometry timestamp is too far in the future"
    return None

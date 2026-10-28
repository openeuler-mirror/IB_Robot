"""Derive all runtime MID-360 transforms from one human-editable mount config."""

import copy
import math
from typing import Any


def _finite_vector(value: object, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _inverse_transform(translation: list[float], quaternion: list[float]) -> tuple[list[float], list[float]]:
    x, y, z, w = quaternion
    inverse_rotation = [-x, -y, -z, w]
    vx, vy, vz = (-translation[0], -translation[1], -translation[2])
    tx = 2.0 * (inverse_rotation[1] * vz - inverse_rotation[2] * vy)
    ty = 2.0 * (inverse_rotation[2] * vx - inverse_rotation[0] * vz)
    tz = 2.0 * (inverse_rotation[0] * vy - inverse_rotation[1] * vx)
    rotated = [
        vx + inverse_rotation[3] * tx + inverse_rotation[1] * tz - inverse_rotation[2] * ty,
        vy + inverse_rotation[3] * ty + inverse_rotation[2] * tx - inverse_rotation[0] * tz,
        vz + inverse_rotation[3] * tz + inverse_rotation[0] * ty - inverse_rotation[1] * tx,
    ]
    return rotated, inverse_rotation


def normalize_mid360_mount(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "1.0":
        raise ValueError("mount schema_version must equal 1.0")
    if value.get("status") != "provisional":
        raise ValueError("mount status must be provisional")
    if value.get("parent_frame") != "base_link":
        raise ValueError("mount parent_frame must be base_link")
    translation = _finite_vector(value.get("translation_m"), 3, "translation_m")
    rpy_deg = _finite_vector(value.get("rpy_deg"), 3, "rpy_deg")
    rpy = [math.radians(item) for item in rpy_deg]
    quaternion = _quaternion_from_rpy(*rpy)
    inverse_translation, inverse_quaternion = _inverse_transform(translation, quaternion)
    return {
        "parent_frame": "base_link",
        "lidar_frame": str(value.get("lidar_frame", "livox_frame")),
        "body_frame": str(value.get("body_frame", "body")),
        "translation": translation,
        "rpy": rpy,
        "rpy_deg": rpy_deg,
        "rotation_xyzw": quaternion,
        "inverse_translation": inverse_translation,
        "inverse_rotation_xyzw": inverse_quaternion,
    }


def apply_mid360_mount(robot_config: dict[str, Any], mount: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(robot_config)
    translation = mount["translation"]
    rpy = mount["rpy"]
    lidar_frame = mount["lidar_frame"]
    body_frame = mount["body_frame"]
    runtime = result.get("runtime") or {}
    provider = bool(isinstance(runtime, dict) and runtime.get("provider"))
    if provider:
        # One MID-360 mount (lekiwi-gazebo-runtime-platform D13): on the
        # provider path the mount also drives the runtime's (lidar, mid360)
        # peripheral entry — the public body -> livox_frame identity
        # transform and the simulator attachment pose relative to base_link —
        # so the fragment the runtime merges carries both pose keys. The
        # entry is created when the application declares none (the migrated
        # lekiwi_lidar) and updated in place when it does (lekiwi_nav_grasp),
        # preserving the application's device settings.
        peripherals = result.setdefault("peripherals", [])
        entry = next(
            (
                item
                for item in peripherals
                if isinstance(item, dict) and item.get("type") == "lidar" and item.get("name") == "mid360"
            ),
            None,
        )
        if entry is None:
            entry = {"type": "lidar", "name": "mid360"}
            peripherals.append(entry)
        entry["transform"] = {
            "parent_frame": body_frame,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
        entry.setdefault("simulation", {}).setdefault("gazebo", {})["transform"] = {
            "parent_frame": mount["parent_frame"],
            "x": translation[0],
            "y": translation[1],
            "z": translation[2],
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
        }
        # lekiwi-mujoco-runtime-platform D1: the same base_link-relative mount
        # drives the MuJoCo attachment pose (one helper, both platforms).
        entry.setdefault("simulation", {}).setdefault("mujoco", {})["transform"] = {
            "parent_frame": mount["parent_frame"],
            "x": translation[0],
            "y": translation[1],
            "z": translation[2],
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
        }
    else:
        # runtime-peripheral-overrides D1 (revision 2): on the require_model
        # provider path the lidar mounting is runtime-owned; peripheral
        # entries are overrides, not device definitions to re-mount. The
        # navigation provisioning below still applies (fast_lio is
        # application-owned).
        for peripheral in result.get("peripherals", []):
            if peripheral.get("frame_id") == lidar_frame:
                peripheral["transform"] = {
                    "parent_frame": body_frame,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                }
    navigation = result.setdefault("navigation", {})
    static_tfs = navigation.setdefault("static_tfs", [])
    static_tfs[:] = [item for item in static_tfs if item.get("child_frame") != body_frame]
    static_tfs.append(
        {
            "name": "static_tf_fast_lio_body",
            "parent_frame": mount["parent_frame"],
            "child_frame": body_frame,
            "translation": translation,
            "rotation": rpy,
        }
    )
    fast_lio = navigation.setdefault("fast_lio", {})
    fast_lio["body_to_base_translation"] = mount["inverse_translation"]
    fast_lio["body_to_base_rotation"] = mount["inverse_rotation_xyzw"]
    result["_mid360_mount"] = mount
    return result

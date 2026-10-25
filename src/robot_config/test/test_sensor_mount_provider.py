"""MID-360 mount forwarding on the provider path (lekiwi-gazebo-runtime-platform D13).

``apply_mid360_mount`` is the one mount source: on the provider path it
creates or updates the partial ``(lidar, mid360)`` peripheral entry — the
public ``body -> livox_frame`` identity transform plus the simulator
attachment pose relative to ``base_link`` — next to the existing
``base_link -> body`` static TF and the FAST-LIO inverse extrinsics, and the
runtime fragment serialises it inside ``peripherals``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.runtime import _write_peripherals_fragment
from robot_config.loader import load_robot_config_dict
from robot_config.sensor_mount import apply_mid360_mount, normalize_mid360_mount

ROOT = Path(__file__).resolve().parents[3]
ROBOTS = ROOT / "src/robot_config/config/robots"
MOUNT = ROOT / "src/robots/lekiwi/lekiwi_description/config/lekiwi_mid360_mount.yaml"


def _mount():
    return normalize_mid360_mount(yaml.safe_load(MOUNT.read_text(encoding="utf-8")))


def _rotated_mount():
    return normalize_mid360_mount(
        {
            "schema_version": "1.0",
            "status": "provisional",
            "parent_frame": "base_link",
            "translation_m": [-0.12, 0.05, 0.28],
            "rpy_deg": [11.5, -8.6, 108.9],
        }
    )


def test_provider_config_without_a_lidar_entry_gets_the_partial_entry():
    """lekiwi_lidar declares no application peripherals: the mount creates
    the partial (lidar, mid360) override the runtime merges over its
    profile."""
    config = load_robot_config_dict(ROBOTS / "lekiwi_lidar.yaml", defer_interface_binding=True)
    assert config["runtime"]["provider"] == "lekiwi_robot"
    entry = next(
        (item for item in config["peripherals"] if item.get("type") == "lidar" and item.get("name") == "mid360"),
        None,
    )
    assert entry is not None, "the provider path must create the (lidar, mid360) entry"
    mount = _mount()
    assert entry["transform"] == {
        "parent_frame": "body",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    assert entry["simulation"]["gazebo"]["transform"] == {
        "parent_frame": "base_link",
        "x": mount["translation"][0],
        "y": mount["translation"][1],
        "z": mount["translation"][2],
        "roll": mount["rpy"][0],
        "pitch": mount["rpy"][1],
        "yaw": mount["rpy"][2],
    }


def test_provider_entry_update_preserves_device_settings():
    config = {
        "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
        "peripherals": [
            {
                "type": "lidar",
                "name": "mid360",
                "frame_id": "livox_frame",
                "lidar_ip": "192.168.1.139",
                "pointcloud_topic": "/livox/lidar",
            }
        ],
        "navigation": {"fast_lio": {}},
    }
    mount = _rotated_mount()
    result = apply_mid360_mount(config, mount)
    entry = result["peripherals"][0]
    # device settings survive the in-place update
    assert entry["lidar_ip"] == "192.168.1.139"
    assert entry["pointcloud_topic"] == "/livox/lidar"
    assert entry["transform"]["parent_frame"] == "body"
    sim = entry["simulation"]["gazebo"]["transform"]
    assert (sim["x"], sim["y"], sim["z"]) == tuple(mount["translation"])
    assert (sim["roll"], sim["pitch"], sim["yaw"]) == tuple(mount["rpy"])


def test_non_provider_path_keeps_the_frame_id_remount():
    config = {
        "peripherals": [{"type": "lidar", "name": "mid360", "frame_id": "livox_frame", "lidar_ip": "1.2.3.4"}],
        "navigation": {"fast_lio": {}},
    }
    result = apply_mid360_mount(config, _mount())
    # no provider: the legacy re-mount by frame_id applies and no (lidar,
    # mid360) entry is created or annotated
    entry = result["peripherals"][0]
    assert entry["transform"]["parent_frame"] == "body"
    assert entry["transform"]["x"] == 0.0 and entry["transform"]["yaw"] == 0.0
    assert "simulation" not in entry


def test_mount_provisioning_still_feeds_static_tf_and_fast_lio():
    config = {"runtime": {"provider": "lekiwi_robot"}, "navigation": {"fast_lio": {}}}
    mount = _rotated_mount()
    result = apply_mid360_mount(config, mount)
    static_tf = next(item for item in result["navigation"]["static_tfs"] if item["child_frame"] == "body")
    assert static_tf["parent_frame"] == "base_link"
    assert static_tf["translation"] == mount["translation"]
    assert static_tf["rotation"] == mount["rpy"]
    fast_lio = result["navigation"]["fast_lio"]
    assert fast_lio["body_to_base_translation"] == mount["inverse_translation"]
    assert fast_lio["body_to_base_rotation"] == mount["inverse_rotation_xyzw"]
    # the inverse really is the mount's inverse (D13): the inverse rotation
    # is the conjugate of the mount's own quaternion
    x, y, z, w = mount["rotation_xyzw"]
    assert fast_lio["body_to_base_rotation"] == pytest.approx([-x, -y, -z, w])


def test_the_runtime_fragment_carries_the_mount_entry():
    """_write_peripherals_fragment serialises the mount's entry inside
    peripherals, so the runtime merge applies it over the profile (D13)."""
    config = load_robot_config_dict(ROBOTS / "lekiwi_lidar.yaml", defer_interface_binding=True)
    fragment_path = _write_peripherals_fragment(config)
    try:
        fragment = yaml.safe_load(Path(fragment_path).read_text(encoding="utf-8"))
    finally:
        Path(fragment_path).unlink(missing_ok=True)
    entry = next(item for item in fragment["peripherals"] if item.get("name") == "mid360")
    assert entry["type"] == "lidar"
    assert entry["transform"]["parent_frame"] == "body"
    assert entry["simulation"]["gazebo"]["transform"]["parent_frame"] == "base_link"
    # the entry deep-merges over the runtime profile's declaration by
    # (type, name): the profile's driver and device settings stay
    from robot_runtime.interface_description import merge_peripherals
    from robot_runtime.profile import load_profile

    profile = load_profile(ROOT / "src/robots/lekiwi/lekiwi_robot/profiles/lekiwi_lidar.yaml")
    merged = merge_peripherals(profile.get("peripherals"), fragment["peripherals"])
    lidar = next(periph for periph in merged if periph.get("name") == "mid360")
    assert lidar["driver"] == "livox_mid360"
    assert lidar["frame_id"] == "livox_frame"
    assert lidar["transform"]["parent_frame"] == "body"
    assert lidar["simulation"]["gazebo"]["transform"]["parent_frame"] == "base_link"
    # the profile's sensor parameters survive the key-wise deep merge
    assert lidar["simulation"]["gazebo"]["horizontal_samples"] == 900

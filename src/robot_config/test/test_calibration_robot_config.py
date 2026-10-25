from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.perception import generate_tf_nodes
from robot_config.loader import load_robot_config_dict

CONFIG_PATH = Path(__file__).parents[1] / "config" / "robots" / "lekiwi_sensor_calib.yaml"


def test_calibration_config_selects_the_lekiwi_runtime_with_a_front_camera_addition():
    config = load_robot_config_dict(CONFIG_PATH)

    # Runtime selection (openspec lekiwi-application-migration task 1.5): the
    # base-only LeKiwi runtime profile owns the chassis, controllers and the
    # MID-360; the fixed front camera is profile-declared (disabled by
    # default) and this deployment opts in through an override entry.
    assert config["runtime"]["provider"] == "lekiwi_robot"
    assert config["runtime"]["profile"] == "lekiwi_lidar"
    assert "require_model" not in config["runtime"]
    assert "ros2_control" not in config

    peripherals = {item["name"]: item for item in config["peripherals"]}
    # The MID-360 mount forwards both pose keys into the runtime fragment
    # (lekiwi-gazebo-runtime-platform D13): the front camera override joins
    # the mount-created partial lidar entry.
    assert set(peripherals) == {"front", "mid360"}
    camera = peripherals["front"]
    # runtime-peripheral-overrides: no driver key on the provider path — the
    # lekiwi_lidar profile declares the realsense driver; the entry only
    # enables the device and carries deployment settings.
    assert camera["enabled"] is True
    assert "driver" not in camera
    assert camera["driver_camera_name"] == "front"
    assert camera["streams"] == ["color", "depth"]
    assert (camera["width"], camera["height"], camera["fps"]) == (640, 480, 30)
    assert camera["align_depth"] is False
    assert camera["enable_sync"] is False
    assert camera["initial_reset"] is False
    assert camera["transform"]["parent_frame"] == "base_link"
    lidar = peripherals["mid360"]
    assert lidar["type"] == "lidar"
    assert lidar["transform"]["parent_frame"] == "body"
    assert lidar["simulation"]["gazebo"]["transform"]["parent_frame"] == "base_link"

    navigation = config["navigation"]
    assert navigation["enabled"] is True
    assert navigation["fast_lio"]["enabled"] is True
    assert navigation["slam_toolbox"]["enabled"] is False
    assert navigation["nav2_bringup"]["enabled"] is False
    # Calibration teleoperation is admitted through the contract channel: the
    # launch-time admission step opens the gate after the runtime is ready.
    assert navigation["open_gate_on_startup"] is True
    assert "cmd_vel_bridge" not in navigation

    assert config["control_modes"]["teleop"]["runtime_mode"] == "base_navigation"
    assert "controllers" not in config["control_modes"]["teleop"]


def test_calibration_config_consumes_approved_camera_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / ".ros" / "ibrobot" / "calib" / "current" / "base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status": "approved",
                "device": {"name": "front_camera", "serial": "camera-1"},
                "transform": {
                    "parent_frame": "base_link",
                    "child_frame": "camera_front_optical_frame",
                    "translation": [0.12, -0.34, 0.56],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_robot_config_dict(CONFIG_PATH)
    transform = next(item for item in config["peripherals"] if item["name"] == "front")["transform"]

    assert (transform["x"], transform["y"], transform["z"]) == (0.12, -0.34, 0.56)
    assert [transform[key] for key in ("qx", "qy", "qz", "qw")] == pytest.approx([0.5, -0.5, 0.5, 0.5])
    main_tf = next(
        node for node in generate_tf_nodes(config) if vars(node).get("_Node__node_name") == "static_tf_front"
    )
    arguments = [str(value) for value in main_tf._Node__arguments]
    assert "--roll" not in arguments
    assert [float(arguments[arguments.index(key) + 1]) for key in ("--qx", "--qy", "--qz", "--qw")] == pytest.approx(
        [0.5, -0.5, 0.5, 0.5]
    )


def test_calibration_config_keeps_placeholder_when_artifact_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_robot_config_dict(CONFIG_PATH)
    transform = next(item for item in config["peripherals"] if item["name"] == "front")["transform"]

    assert (transform["x"], transform["y"], transform["z"]) == (0.0, 0.0, 0.0)
    assert (transform["roll"], transform["pitch"], transform["yaw"]) == (0.0, 0.0, 0.0)


def test_calibration_config_rejects_installed_non_approved_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / ".ros" / "ibrobot" / "calib" / "current" / "base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("status: candidate\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(ValueError, match="approved camera calibration"):
        load_robot_config_dict(CONFIG_PATH)


def test_calibration_launch_delegates_to_the_runtime_for_migrated_configs():
    launch_path = Path(__file__).parents[1] / "launch/sensor_calibration.launch.py"
    content = launch_path.read_text(encoding="utf-8")

    # Provider branch: the runtime owns sensors, FAST-LIO, the chassis and the
    # gate; the admission client fails the launch if mode/gate admission is
    # rejected. The generic bridge builder is gone entirely.
    assert "generate_runtime_provider_actions" in content
    assert "navigation_admission" in content
    assert "RegisterEventHandler" in content
    assert "OnProcessExit" in content
    assert "generate_cmd_vel_nodes" not in content
    assert "generate_navigation_nodes" not in content
    assert "generate_nav2_nodes" not in content
    # Legacy branch for provider-less calibration configurations.
    assert "generate_ros2_control_nodes" in content
    assert "controller_startup_timeout=300.0" in content
    assert "generate_static_tf_nodes" in content


def test_calibration_config_contains_no_realsense_overlay_or_solver_runtime():
    content = CONFIG_PATH.read_text(encoding="utf-8")
    document = yaml.safe_load(content)

    assert "overlay" not in content.lower()
    assert "src/realsense" not in content.lower()
    assert "fast_calib" not in document["robot"]
    assert document["robot"].get("recording", {}).get("enabled", False) is False

"""Unit tests for runtime sensor peripheral composition (D12)."""

from __future__ import annotations

import json
import textwrap

import launch_ros.actions
import pytest
import yaml

from robot_runtime.peripherals import (
    PERCEPTION_CAMERA,
    PERCEPTION_LIDAR,
    fast_lio_nodes,
    load_peripherals_file,
    merge_peripherals,
    perception_capabilities,
    peripheral_nodes,
    scan_converter_nodes,
    synthetic_perception_nodes,
)


def _node_specs(nodes):
    return [(node.node_package, node.node_executable) for node in nodes if isinstance(node, launch_ros.actions.Node)]


def test_merge_peripherals_overrides_by_type_and_name():
    profile = [
        {"type": "camera", "name": "front", "driver": "opencv", "index": 0},
        {"type": "lidar", "name": "mid360", "driver": "livox_mid360"},
    ]
    deployment = [
        {"type": "camera", "name": "front", "driver": "realsense", "serial_number": "X"},
        {"type": "microphone", "name": "respeaker", "driver": "alsa"},
    ]
    merged = merge_peripherals(profile, deployment)
    assert len(merged) == 3
    by_key = {(p["type"], p["name"]): p for p in merged}
    assert by_key[("camera", "front")]["driver"] == "realsense"
    assert by_key[("camera", "front")]["serial_number"] == "X"
    assert by_key[("lidar", "mid360")]["driver"] == "livox_mid360"
    assert by_key[("microphone", "respeaker")]["driver"] == "alsa"


def test_perception_capabilities_from_peripherals():
    caps = perception_capabilities(
        [
            {"type": "camera", "name": "front", "driver": "realsense", "streams": ["color"]},
            {"type": "camera", "name": "virtual_cam", "driver": "virtual"},
            {"type": "lidar", "name": "mid360", "driver": "livox_mid360", "pointcloud_topic": "/livox/lidar"},
            {"type": "microphone", "name": "respeaker", "driver": "alsa"},
        ]
    )
    assert caps[PERCEPTION_CAMERA]["topics"] == ["/camera/front/image_raw", "/camera/front/camera_info"]
    assert caps[PERCEPTION_LIDAR]["topics"] == ["/livox/lidar", "/livox/imu"]
    assert set(caps) == {PERCEPTION_CAMERA, PERCEPTION_LIDAR}


def test_synthetic_perception_covers_declared_surface():
    nodes = synthetic_perception_nodes(
        [
            {"type": "camera", "name": "front", "driver": "opencv"},
            {"type": "lidar", "name": "mid360", "driver": "livox_mid360"},
        ]
    )
    assert _node_specs(nodes) == [("robot_runtime", "synthetic_perception")]


def test_peripheral_nodes_camera_drivers():
    nodes = peripheral_nodes(
        [
            {"type": "camera", "name": "top", "driver": "opencv", "index": 1, "frame_id": "top_cam_frame"},
            {"type": "camera", "name": "wrist", "driver": "virtual", "source_topic": "/camera/top/image_raw"},
        ],
        use_sim=False,
    )
    specs = _node_specs(nodes)
    assert ("usb_cam", "usb_cam_node_exe") in specs
    assert ("robot_runtime", "topic_relay") not in [s for s in specs if s[0] == "robot_runtime"], (
        "virtual camera composes no node"
    )
    usb_cam = next(
        node for node in nodes if (node.node_package, node.node_executable) == ("usb_cam", "usb_cam_node_exe")
    )

    def _text(value):
        return "".join(item.text if hasattr(item, "text") else str(item) for item in value)

    def _decode(value):
        return yaml.safe_load(_text(value)) if isinstance(value, tuple | list) else value

    params = {_text(key): _decode(value) for key, value in usb_cam._Node__parameters[0].items()}
    assert params["frame_id"] == "top_cam_frame", "usb_cam reads upstream frame_id, not camera_frame_id"
    assert "camera_frame_id" not in params


def test_peripheral_nodes_realsense_spawns_relay_chain():
    nodes = peripheral_nodes(
        [
            {
                "type": "camera",
                "name": "front",
                "driver": "realsense",
                "frame_id": "front_cam",
                "optical_frame_id": "front_opt",
            }
        ],
        use_sim=False,
    )
    specs = _node_specs(nodes)
    assert ("realsense2_camera", "realsense2_camera_node") in specs
    assert ("robot_runtime", "topic_relay") in specs, "realsense composition declares its relays"


def test_peripheral_nodes_compose_the_alsa_capture_owner():
    """Microphones used to be data-only; the runtime now owns the ALSA node (runtime-owned-audio-io)."""
    nodes = peripheral_nodes(
        [{"type": "microphone", "name": "respeaker", "driver": "alsa", "params": {"channels": 4}}], use_sim=False
    )
    assert [(node.node_package, node.node_executable, node._Node__node_name) for node in nodes] == [
        ("audio_capture", "audio_capture_node", "audio_capture")
    ]


def test_peripheral_nodes_sim_skips_physical():
    peripherals = [{"type": "camera", "name": "front", "driver": "opencv"}]
    assert peripheral_nodes(peripherals, use_sim=True) == []


def test_peripheral_tf_mounting_composed_on_real_hardware():
    nodes = peripheral_nodes(
        [
            {
                "type": "camera",
                "name": "front",
                "driver": "opencv",
                "frame_id": "front_cam",
                "transform": {
                    "parent_frame": "base_link",
                    "x": 0.1,
                    "y": 0.0,
                    "z": 0.2,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 1.57,
                },
            }
        ],
        use_sim=False,
    )
    tf_nodes = [node for node in nodes if getattr(node, "node_package", "") == "tf2_ros"]
    assert tf_nodes, "camera frame TF mounts with the driver"


def test_fast_lio_nodes_compose_mapping_and_bridge():
    nodes = fast_lio_nodes(
        {"enabled": True, "params_file": "/tmp/fastlio.yaml"},
        bridge_package="lekiwi_robot",
        use_sim=False,
    )
    assert _node_specs(nodes) == [
        ("fast_lio", "fastlio_mapping"),
        ("lekiwi_robot", "fast_lio_odom_bridge"),
    ]


def test_fast_lio_nodes_disabled_or_sim_compose_nothing():
    assert fast_lio_nodes({"enabled": False}, bridge_package="lekiwi_robot", use_sim=False) == []
    assert (
        fast_lio_nodes({"enabled": True, "params_file": "/tmp/x.yaml"}, bridge_package="lekiwi_robot", use_sim=True)
        == []
    )
    # the sdk platform (explicitly) composes nothing either
    assert (
        fast_lio_nodes(
            {"enabled": True, "params_file": "/tmp/x.yaml"},
            bridge_package="lekiwi_robot",
            use_sim=True,
            sim_platform="sdk",
        )
        == []
    )


# ---------------------------------------------------------------------------
# lekiwi-gazebo-runtime-platform D6: FAST-LIO on a physics platform
# ---------------------------------------------------------------------------


def _params_of(node):
    params = {}
    for entry in node._Node__parameters or []:
        if isinstance(entry, dict):
            params.update({_text(key): _value(value) for key, value in entry.items()})
    return params


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _value(value):
    if isinstance(value, tuple | list):
        if value and all(not hasattr(item, "text") and not isinstance(item, tuple | list) for item in value):
            return list(value)
        if any(isinstance(item, tuple | list) for item in value):
            return [_value(item) for item in value]
        return yaml.safe_load(_text(value)) if value else []
    return value


def test_fast_lio_physics_platform_runs_the_simulation_parameters_on_sim_time(tmp_path):
    sim_params = tmp_path / "fast_lio_sim.yaml"
    nodes = fast_lio_nodes(
        {"enabled": True, "params_file": "/tmp/x.yaml", "sim_params_file": str(sim_params)},
        bridge_package="lekiwi_robot",
        use_sim=True,
        sim_platform="gazebo",
    )
    assert _node_specs(nodes) == [
        ("fast_lio", "fastlio_mapping"),
        ("lekiwi_robot", "fast_lio_odom_bridge"),
    ]
    mapping, bridge = nodes
    mapping_params = _params_of(mapping)
    assert mapping_params["use_sim_time"] is True
    # the simulation parameter file replaces the hardware one (D6)
    assert any(
        _text(getattr(entry, "param_file", []) or []) == str(sim_params)
        for entry in mapping._Node__parameters
        if type(entry).__name__ == "ParameterFile"
    ), mapping._Node__parameters
    bridge_params = _params_of(bridge)
    assert bridge_params["use_sim_time"] is True
    # wheel odometry owns odom -> base_link in simulation (D4)
    assert bridge_params["publish_tf"] is False


def test_fast_lio_hardware_keeps_the_wall_time_chain():
    nodes = fast_lio_nodes(
        {"enabled": True, "params_file": "/tmp/x.yaml", "publish_tf": True, "sim_params_file": "/tmp/unused.yaml"},
        bridge_package="lekiwi_robot",
        use_sim=False,
        sim_platform="",
    )
    mapping, bridge = nodes
    assert _params_of(mapping)["use_sim_time"] is False
    assert _params_of(bridge)["publish_tf"] is True
    assert "use_sim_time" not in _params_of(bridge)


def test_fast_lio_physics_platform_without_sim_params_fails_naming_the_key():
    with pytest.raises(ValueError, match="fast_lio.sim_params_file is required"):
        fast_lio_nodes(
            {"enabled": True, "params_file": "/tmp/x.yaml"},
            bridge_package="lekiwi_robot",
            use_sim=True,
            sim_platform="gazebo",
        )


# ---------------------------------------------------------------------------
# lekiwi-gazebo-runtime-platform D5: the split-out scan converter builder
# ---------------------------------------------------------------------------


def test_scan_converter_nodes_match_the_driver_composition(tmp_path):
    user_config = tmp_path / "MID360_config.json"
    user_config.write_text(json.dumps({"MID360": {"host_net_info": {}}, "lidar_configs": [{}]}), encoding="utf-8")
    periph = {
        "type": "lidar",
        "name": "mid360",
        "driver": "livox_mid360",
        "frame_id": "livox_frame",
        "user_config_path": str(user_config),
        "host_ip": "192.168.1.5",
        "lidar_ip": "192.168.1.139",
        "scan_converter": {
            "enabled": True,
            "queue_size": 1,
            "pointcloud_topic": "/cloud_registered_body",
            "scan_topic": "/scan",
            "target_frame": "body",
            "min_height": -0.05,
            "max_height": 0.5,
            "range_max": 8.0,
        },
    }
    nodes = scan_converter_nodes(periph, use_sim=False)
    assert _node_specs(nodes) == [("pointcloud_to_laserscan", "pointcloud_to_laserscan_node")]
    params = _params_of(nodes[0])
    assert params["target_frame"] == "body"
    assert params["min_height"] == -0.05
    assert params["range_max"] == 8.0
    assert params["queue_size"] == 1
    assert params["use_sim_time"] is False
    remappings = {_text(source): _text(target) for source, target in nodes[0]._Node__remappings}
    assert remappings == {"cloud_in": "/cloud_registered_body", "scan": "/scan"}
    # the physical driver composition includes the same converter
    driver_nodes = peripheral_nodes([periph], use_sim=False)
    assert any(node._Node__node_name == "mid360_pointcloud_to_laserscan" for node in driver_nodes)
    # a disabled scan block composes nothing
    assert scan_converter_nodes({**periph, "scan_converter": {"enabled": False}}, use_sim=True) == []
    assert scan_converter_nodes({"name": "l"}, use_sim=True) == []


# ---------------------------------------------------------------------------
# runtime-peripheral-overrides D6: one structural loader for every fragment
# ---------------------------------------------------------------------------


def _fragment(tmp_path, content, name="peripherals.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(path)


def test_load_peripherals_file_empty_path_is_the_empty_fragment():
    assert load_peripherals_file("") == {}
    assert load_peripherals_file("   ") == {}


def test_load_peripherals_file_returns_the_fragment(tmp_path):
    path = _fragment(
        tmp_path,
        """
        peripherals:
        - type: camera
          name: top
          width: 1280
        fast_lio:
          enabled: true
        """,
    )
    assert load_peripherals_file(path) == {
        "peripherals": [{"type": "camera", "name": "top", "width": 1280}],
        "fast_lio": {"enabled": True},
    }


@pytest.mark.parametrize(
    "content,pattern",
    [
        ("- type: camera\n", "must be a mapping"),
        ("peripherals: {a: 1}\n", "peripherals must be a list"),
        ("unknown_key: 1\n", "unknown top-level keys"),
        ("peripherals:\n- type: camera\n", "requires type and name"),
        ("peripherals:\n- type: camera\n  name: top\n- type: camera\n  name: top\n", "duplicate peripheral"),
        ("fast_lio: [1]\n", "fast_lio must be a mapping"),
    ],
)
def test_load_peripherals_file_structural_errors_name_the_file(tmp_path, content, pattern):
    path = _fragment(tmp_path, content)
    with pytest.raises(ValueError, match=pattern) as excinfo:
        load_peripherals_file(path)
    assert path in str(excinfo.value), "every structural error names the file"


def test_load_peripherals_file_missing_file_names_the_file(tmp_path):
    missing = str(tmp_path / "absent.yaml")
    with pytest.raises(ValueError, match=f"peripherals file is unreadable: {missing}"):
        load_peripherals_file(missing)


def test_load_peripherals_file_expands_env_and_find(tmp_path, monkeypatch):
    real = _fragment(tmp_path, "peripherals: []\n", name="env_fragment.yaml")
    monkeypatch.setenv("IBROBOT_TEST_PERIPHERALS_FILE", real)
    assert load_peripherals_file("$(env IBROBOT_TEST_PERIPHERALS_FILE)") == {"peripherals": []}
    # $(find pkg) resolves through the ament index: the error names both the
    # raw argument and the expanded share-dir path, proving the substitution ran
    with pytest.raises(ValueError, match=r"unreadable") as excinfo:
        load_peripherals_file("$(find robot_runtime)/no-such-fragment.yaml")
    assert "no-such-fragment.yaml" in str(excinfo.value)
    assert "/robot_runtime/" in str(excinfo.value)


def test_the_offline_cli_reports_the_same_structural_error(tmp_path, capsys):
    # D6: the CLI stopped reading YAML directly; a malformed file yields the
    # same message the runtime launches and the facade/mock nodes report.
    from robot_runtime.interface_description import main as cli_main

    bad = _fragment(tmp_path, "unknown_key: 1\n")
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"name": "t", "version": "0.1"},
                "modes": {"idle": {"controllers": []}},
                "capabilities": {"joint.state": {}},
                "joints": ["j1"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--profile", str(profile_path), "--peripherals", bad])
    assert excinfo.value.code == 1
    assert f"peripherals file {bad} has unknown top-level keys" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [
        "[]\n",
        "false\n",
        "0\n",
        "- peripherals\n",
    ],
)
def test_load_peripherals_file_rejects_non_mapping_documents(tmp_path, content):
    path = _fragment(tmp_path, content)
    with pytest.raises(ValueError, match=f"peripherals file must be a mapping: {path}"):
        load_peripherals_file(path)


def test_load_peripherals_file_accepts_an_empty_document(tmp_path):
    assert load_peripherals_file(str(_fragment(tmp_path, ""))) == {}
    assert load_peripherals_file(str(_fragment(tmp_path, "# only a comment\n"))) == {}


def test_load_peripherals_file_distinguishes_missing_keys_from_null(tmp_path):
    # a missing key is fine; an explicit null is a malformed fragment
    assert load_peripherals_file(str(_fragment(tmp_path, "peripherals: []\n"))) == {"peripherals": []}
    with pytest.raises(ValueError, match="peripherals file.*must be a list"):
        load_peripherals_file(str(_fragment(tmp_path, "peripherals:\n")))
    assert load_peripherals_file(str(_fragment(tmp_path, "fast_lio: {}\n"))) == {"fast_lio": {}}
    with pytest.raises(ValueError, match="peripherals file.*must be a mapping"):
        load_peripherals_file(str(_fragment(tmp_path, "fast_lio:\n")))

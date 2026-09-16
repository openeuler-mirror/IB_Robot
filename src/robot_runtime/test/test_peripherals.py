"""Unit tests for runtime sensor peripheral composition (D12)."""

from __future__ import annotations

import launch_ros.actions
import yaml

from robot_runtime.peripherals import (
    PERCEPTION_CAMERA,
    PERCEPTION_LIDAR,
    fast_lio_nodes,
    merge_peripherals,
    perception_capabilities,
    peripheral_nodes,
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


def test_peripheral_nodes_microphone_is_data_only():
    assert peripheral_nodes([{"type": "microphone", "name": "respeaker", "driver": "alsa"}], use_sim=False) == []


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

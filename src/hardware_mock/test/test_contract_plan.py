"""Tests for hardware_mock.contract_plan.build_plan.

These tests exercise the architectural validation rules so a regression cannot
silently let a bad contract through.
"""

import pytest

from hardware_mock.contract_plan import build_plan
from hardware_mock.type_registry import UnsupportedMessageTypeError


def _base_robot():
    return {
        "joints": {"all": ["1", "2", "3", "4", "5", "6"]},
        "ros2_control": {"reset_positions": {"2": -1.5854, "3": 1.5708}},
        "peripherals": [
            {
                "type": "camera",
                "name": "top",
                "width": 640,
                "height": 480,
                "fps": 30,
                "frame_id": "camera_top_frame",
                "optical_frame_id": "camera_top_optical_frame",
            },
            {
                "type": "camera",
                "name": "wrist",
                "width": 640,
                "height": 480,
                "fps": 60,
                "frame_id": "camera_wrist_frame",
                "optical_frame_id": "camera_wrist_optical_frame",
            },
        ],
        "contract": {
            "rate_hz": 20,
            "observations": [
                {
                    "key": "observation.images.top",
                    "topic": "/camera/top/image_raw",
                    "type": "sensor_msgs/msg/Image",
                    "peripheral": "top",
                    "image": {"resize": [480, 640]},
                    "align": {"tol_ms": 1500},
                    "qos": {"reliability": "best_effort", "depth": 10},
                },
                {
                    "key": "observation.images.wrist",
                    "topic": "/camera/wrist/image_raw",
                    "type": "sensor_msgs/msg/Image",
                    "peripheral": "wrist",
                    "image": {"resize": [480, 640]},
                    "align": {"tol_ms": 1500},
                    "qos": {"reliability": "best_effort", "depth": 10},
                },
                {
                    "key": "observation.state",
                    "topic": "/joint_states",
                    "type": "sensor_msgs/msg/JointState",
                    "align": {"tol_ms": 1500},
                    "qos": {"reliability": "best_effort", "depth": 50},
                },
            ],
            "actions": [
                {
                    "key": "action",
                    "selector": {"names": ["action.0", "action.1", "action.2", "action.3", "action.4"]},
                    "publish": {
                        "topic": "/arm_position_controller/commands",
                        "type": "std_msgs/msg/Float64MultiArray",
                        "qos": {"reliability": "best_effort", "depth": 10},
                    },
                },
                {
                    "key": "action",
                    "selector": {"names": ["action.5"]},
                    "publish": {
                        "topic": "/gripper_position_controller/commands",
                        "type": "std_msgs/msg/Float64MultiArray",
                        "qos": {"reliability": "best_effort", "depth": 10},
                    },
                },
            ],
        },
    }


def test_build_plan_happy_path_matches_so101_layout():
    plan = build_plan(_base_robot())

    assert plan.joint_ids == ["1", "2", "3", "4", "5", "6"]
    assert plan.initial_positions["2"] == pytest.approx(-1.5854)
    assert plan.initial_positions["1"] == 0.0

    kinds = [o.kind for o in plan.observations]
    assert kinds.count("image") == 2
    assert kinds.count("joint_state") == 1

    top = next(o for o in plan.observations if o.topic == "/camera/top/image_raw")
    assert top.image.width == 640 and top.image.height == 480
    assert top.rate_hz == 30
    assert top.frame_id == "camera_top_optical_frame"

    arm, gripper = plan.actions
    assert arm.index_to_joint_index == [0, 1, 2, 3, 4]
    assert gripper.index_to_joint_index == [5]


def test_unknown_observation_type_rejected():
    robot = _base_robot()
    robot["contract"]["observations"][0]["type"] = "sensor_msgs/msg/CompressedImage"
    with pytest.raises(UnsupportedMessageTypeError):
        build_plan(robot)


def test_unknown_action_type_rejected():
    robot = _base_robot()
    robot["contract"]["actions"][0]["publish"]["type"] = "trajectory_msgs/msg/JointTrajectory"
    with pytest.raises(UnsupportedMessageTypeError):
        build_plan(robot)


def test_action_index_out_of_range_rejected():
    robot = _base_robot()
    robot["contract"]["actions"][1]["selector"]["names"] = ["action.99"]
    with pytest.raises(ValueError):
        build_plan(robot)


def test_image_peripheral_missing_rejected():
    robot = _base_robot()
    robot["contract"]["observations"][0]["peripheral"] = "does_not_exist"
    with pytest.raises(ValueError):
        build_plan(robot)


def test_rate_check_fails_when_fps_below_safe_minimum():
    robot = _base_robot()
    # tol_ms=1500 → min rate = 2000/1500 ≈ 1.33 Hz. Force fps below.
    robot["peripherals"][0]["fps"] = 1
    with pytest.raises(ValueError, match="below the safe minimum"):
        build_plan(robot)


def test_rate_check_skipped_when_flag_set():
    robot = _base_robot()
    robot["peripherals"][0]["fps"] = 1
    robot["hardware_mock"] = {"skip_rate_check": True}
    plan = build_plan(robot)  # must not raise
    assert plan.observations[0].rate_hz == 1


def test_image_resize_overrides_peripheral_dims():
    robot = _base_robot()
    robot["contract"]["observations"][0]["image"]["resize"] = [240, 320]
    plan = build_plan(robot)
    top = next(o for o in plan.observations if o.topic == "/camera/top/image_raw")
    assert (top.image.width, top.image.height) == (320, 240)


def test_image_source_override_applied():
    robot = _base_robot()
    robot["hardware_mock"] = {"image_sources": {"top": {"kind": "solid", "color": "#ff0000"}}}
    plan = build_plan(robot)
    top = next(o for o in plan.observations if o.topic == "/camera/top/image_raw")
    assert top.image.kind == "solid"
    assert top.image.color_rgb == (255, 0, 0)

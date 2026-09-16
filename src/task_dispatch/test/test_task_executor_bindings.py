"""Public motion/feedback binding and provider-less legacy routing."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rclpy.qos import ReliabilityPolicy

from robot_config.interface_binding import InterfaceBindingError
from robot_runtime import contract as RUNTIME
from task_dispatch.task_executor_node import TaskExecutorNode, _load_robot_yaml


@pytest.fixture
def config():
    return {
        "runtime": {
            "provider": "unit_robot",
            "interface_description": {
                "interfaces": {
                    "motion.move_to_pose": {
                        "kind": "service",
                        "direction": "serve",
                        "message_type": "ibrobot_msgs/srv/MoveToPose",
                        "endpoint": "/unit/move",
                    },
                    "gripper.trajectory": {
                        "kind": "action",
                        "direction": "serve",
                        "message_type": "control_msgs/action/FollowJointTrajectory",
                        "endpoint": "/unit/gripper",
                    },
                    "joint.state": {
                        "kind": "topic",
                        "direction": "publish",
                        "message_type": "sensor_msgs/msg/JointState",
                        "endpoint": "/unit/joints",
                        "joint_names": ["arm", "finger"],
                        "qos": {"depth": 7, "reliability": "best_effort"},
                    },
                }
            },
        },
        "robot_model": {
            "joint_groups": {"arm": ["arm"], "gripper": ["finger"]},
            "joint_limits": {"finger": {"min": -0.2, "max": 0.8}},
        },
        "task_dispatch": {"gripper_trajectory_interface": "gripper.trajectory"},
    }


def node_for(config):
    node = object.__new__(TaskExecutorNode)
    node._robot_cfg = config
    return node


def test_public_bindings_select_endpoints_qos_and_model_limits(config):
    node = node_for(config)
    node._configure_interfaces()
    assert node._move_service_name == "/unit/move"
    assert node._gripper_action_name == "/unit/gripper"
    assert node._joint_state_topic == "/unit/joints"
    assert node._joint_state_qos.depth == 7
    assert node._joint_state_qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert node._gripper_joint == "finger"
    assert node._gripper_limits == {"min": -0.2, "max": 0.8}


@pytest.mark.parametrize(
    "field,value",
    [
        ("gripper_trajectory_interface", ""),
        ("gripper_trajectory_interface", "unknown"),
        ("gripper_trajectory_interface", "joint.state"),
        ("move_to_pose_interface", "gripper.trajectory"),
        ("joint_state_interface", "motion.move_to_pose"),
    ],
)
def test_public_binding_failure_never_uses_legacy_fallback(config, field, value):
    config["task_dispatch"][field] = value
    node = node_for(config)
    with pytest.raises(InterfaceBindingError):
        node._configure_interfaces()


@pytest.mark.parametrize("mutation", ["missing_model", "two_grippers", "missing_feedback"])
def test_public_model_and_feedback_required(config, mutation):
    if mutation == "missing_model":
        config.pop("robot_model")
    elif mutation == "two_grippers":
        config["robot_model"]["joint_groups"]["gripper"] = ["finger", "other"]
    else:
        config["runtime"]["interface_description"]["interfaces"]["joint.state"]["joint_names"] = ["arm"]
    with pytest.raises(InterfaceBindingError):
        node_for(config)._configure_interfaces()


@pytest.mark.parametrize("position", [-0.3, 0.9, float("nan"), float("inf")])
def test_gripper_limits_reject_before_noop_or_motion(config, position):
    node = node_for(config)
    node._configure_interfaces()
    node._is_redundant_gripper_open = Mock()
    node._gripper_action_client = Mock()
    success, message = node._exec_gripper(SimpleNamespace(gripper_position=position))
    assert not success and "public joint limits" in message
    node._is_redundant_gripper_open.assert_not_called()
    assert node._gripper_action_client.mock_calls == []


def test_providerless_config_uses_neutral_motion_endpoint():
    node = node_for({"joints": {"gripper": ["6"]}})
    node._configure_interfaces()
    assert node._move_service_name == RUNTIME.MOVE_TO_POSE_SERVICE
    assert node._joint_state_topic == "/joint_states"
    assert node._gripper_joint == "6"
    assert node._gripper_limits is None


def test_loader_requires_public_model_and_preserves_resolved_config_path(monkeypatch, config):
    original = deepcopy(config)
    monkeypatch.setattr(
        "task_dispatch.task_executor_node.load_robot_section", lambda _: ("/resolved/robot.yaml", config)
    )
    resolver = Mock(return_value={"bound": True})
    monkeypatch.setattr("task_dispatch.task_executor_node.resolve_robot_interfaces", resolver)
    assert _load_robot_yaml("robot.yaml") == {"bound": True}
    supplied = resolver.call_args.args[0]
    assert supplied["_config_path"] == "/resolved/robot.yaml"
    assert supplied["runtime"]["require_model"] is True
    assert config == original

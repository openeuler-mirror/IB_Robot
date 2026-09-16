"""Binding errors cannot fall back to an empty legacy dispatcher contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Twist

from action_dispatch import action_dispatcher_node as dispatcher
from action_dispatch.executors.topic import TopicExecutor
from robot_config.interface_binding import InterfaceBindingError
from robot_config.loader import build_contract_from_robot_config_dict


@pytest.fixture
def startup(monkeypatch):
    # Stop at executor construction: no ROS node, timers, clients or publishers.
    monkeypatch.setattr(dispatcher.Node, "__init__", lambda self, *args, **kwargs: None)
    node = dispatcher.ActionDispatcherNode.__new__(dispatcher.ActionDispatcherNode)
    node._parameter_overrides = {}
    parameters = {"robot_config_path": "/test/robot.yaml"}
    node.declare_parameter = lambda name, default: parameters.setdefault(name, default)
    node.get_parameter = lambda name: SimpleNamespace(value=parameters[name])
    node.add_on_set_parameters_callback = Mock()
    logger = Mock()
    node.get_logger = lambda: logger
    loader = Mock()
    monkeypatch.setattr("robot_config.loader.load_robot_config", loader)
    executor = Mock(side_effect=RuntimeError("executor construction reached"))
    monkeypatch.setattr(dispatcher, "create_executor", executor)
    return node, loader, executor, logger


@pytest.mark.parametrize("stage", ["load", "to_contract", "iter_specs"])
def test_named_binding_failures_escape_legacy_contract_loading(startup, monkeypatch, stage):
    node, loader, executor, logger = startup
    error = InterfaceBindingError("unknown_interface", "contract.actions[0].publish", "base.missing")
    if stage == "load":
        loader.side_effect = error
    elif stage == "to_contract":
        loader.return_value.to_contract.side_effect = error
    else:
        monkeypatch.setattr(dispatcher, "iter_specs", Mock(side_effect=error))
    with pytest.raises(InterfaceBindingError) as caught:
        node.__init__()
    assert caught.value is error
    executor.assert_not_called()
    logger.error.assert_not_called()


@pytest.mark.parametrize("error", [FileNotFoundError("missing config"), ValueError("unrelated invalid config")])
def test_unrelated_load_failure_keeps_legacy_fallback(startup, error):
    node, loader, executor, logger = startup
    loader.side_effect = error
    with pytest.raises(RuntimeError, match="executor construction reached"):
        node.__init__()
    assert executor.call_args.args[2]["action_specs"] == []
    logger.error.assert_called_once_with(f"Failed to load contract from /test/robot.yaml: {error}")


@pytest.fixture
def bound_contract():
    from robot_runtime.interface_description import description_digest

    descriptor = {
        "schema_version": 1,
        "robot": {"id": "unit-1", "type": "test", "runtime_name": "test_robot", "runtime_version": "1.0.0"},
        "execution": "simulated",
        "interfaces": {
            "base.cmd_vel": {
                "capability": "base.cmd_vel",
                "kind": "topic",
                "direction": "subscribe",
                "endpoint": "/unit/velocity",
                "message_type": "geometry_msgs/msg/Twist",
                "qos": {"reliability": "reliable", "history": "keep_last", "depth": 5, "durability": "volatile"},
            }
        },
        "states": {},
    }
    descriptor["digest"] = description_digest(descriptor)
    return build_contract_from_robot_config_dict(
        {
            "name": "test",
            "runtime": {"interface_description": descriptor},
            "contract": {
                "actions": [
                    {
                        "key": "action.arm",
                        "publish": {"topic": "/unit/arm", "type": "std_msgs/msg/Float64MultiArray"},
                        "selector": {"names": ["action.0", "action.1"]},
                        "safety_behavior": "hold",
                    },
                    {
                        "key": "action.base",
                        "publish": {"interface": "base.cmd_vel"},
                        "selector": {"names": ["vx", "vy", "wz"]},
                        "safety_behavior": "zeros",
                    },
                ]
            },
        }
    )


def test_legacy_stop_base_zeros_bound_twist_without_publishing_arm(startup, bound_contract):
    node, loader, _executor, _logger = startup
    loader.return_value.to_contract.return_value = bound_contract
    with pytest.raises(RuntimeError, match="executor construction reached"):
        node.__init__()

    assert bound_contract.actions[1]._interface_source["id"] == "base.cmd_vel"
    assert node._base_act_spec is node._action_specs[1]
    arm, base = Mock(), Mock()
    node.create_publisher = Mock(side_effect=[arm, base])
    node._executor = TopicExecutor(node, {"action_specs": node._action_specs})
    assert node._executor.initialize()
    assert node._executor.execute(np.array([1.0, 2.0, 0.25, -0.5, 0.75]))
    moving = base.publish.call_args.args[0]
    assert isinstance(moving, Twist)
    assert [moving.linear.x, moving.linear.y, moving.angular.z] == [0.25, -0.5, 0.75]
    arm.publish.reset_mock()
    base.publish.reset_mock()

    node._stop_base()

    arm.publish.assert_not_called()
    base.publish.assert_called_once()
    stopped = base.publish.call_args.args[0]
    assert isinstance(stopped, Twist)
    assert [stopped.linear.x, stopped.linear.y, stopped.linear.z] == [0.0, 0.0, 0.0]
    assert [stopped.angular.x, stopped.angular.y, stopped.angular.z] == [0.0, 0.0, 0.0]


def test_bound_twist_rejects_hold_before_any_publisher_is_created(startup, bound_contract):
    from dataclasses import replace

    node, loader, _executor, _logger = startup
    bound_contract.actions[1] = replace(bound_contract.actions[1], safety_behavior="hold")
    loader.return_value.to_contract.return_value = bound_contract
    with pytest.raises(RuntimeError, match="executor construction reached"):
        node.__init__()

    node.create_publisher = Mock()
    executor = TopicExecutor(node, {"action_specs": node._action_specs})
    with pytest.raises(ValueError, match="requires safety_behavior='zeros'"):
        executor.initialize()
    node.create_publisher.assert_not_called()
    assert node._base_act_spec.safety_behavior == "hold"

"""Real rclpy startup parameters retain omission and strict SSOT semantics."""

from unittest.mock import Mock

import pytest
import rclpy
from rclpy.parameter import Parameter

from action_dispatch.action_dispatcher_node import ActionDispatcherNode
from robot_config.dispatch_strategies import DispatchStrategyError


@pytest.fixture
def ros_context(monkeypatch):
    rclpy.init()
    # Keep real parameter/CLI resolution, but avoid DDS endpoint discovery in
    # this configuration matrix. Transport behavior has separate tests.
    for method in ("create_service", "create_client", "create_publisher", "create_subscription", "create_timer"):
        monkeypatch.setattr(ActionDispatcherNode, method, Mock())
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.rclpy.action.ActionClient", Mock())
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.ActionServer", Mock())
    yield
    rclpy.shutdown()


@pytest.mark.parametrize(
    "settings,enabled",
    [
        ({}, False),
        ({"blending_strategy": "temporal_ensemble"}, True),
        ({"blending_strategy": "none"}, False),
        ({"chunking_strategy": "", "blending_strategy": ""}, False),
        ({"executor_type": "benchmark", "scheduler_mode": "wait_for_feedback"}, False),
    ],
)
def test_startup_selection_matrix(ros_context, monkeypatch, settings, enabled):
    executor = Mock()
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.create_executor", lambda *args: executor)
    node = ActionDispatcherNode(parameter_overrides=[Parameter(name, value=value) for name, value in settings.items()])
    try:
        assert (node._strategy_selection.blending == "temporal_ensemble") is enabled
        assert node._smoothing_enabled is enabled
        assert (node._smoother is not None) is enabled
        assert node._strategy_selection.chunking == "full_chunk"
    finally:
        node.destroy_node()


@pytest.mark.parametrize("name", ["chunking_strategy", "blending_strategy", "scheduler_mode", "executor_type"])
@pytest.mark.parametrize("value", [False, 0, [], ["full_chunk"]])
def test_startup_rejects_non_string_names(ros_context, name, value):
    with pytest.raises(DispatchStrategyError, match="unknown"):
        ActionDispatcherNode(parameter_overrides=[Parameter(name, value=value)])


@pytest.mark.parametrize("value", [True, False, None, "false", 0, [], [True]])
def test_startup_rejects_removed_smoothing(ros_context, value):
    with pytest.raises(ValueError, match="removed"):
        ActionDispatcherNode(parameter_overrides=[Parameter("temporal_smoothing_enabled", value=value)])


@pytest.mark.parametrize(
    "settings",
    [
        {"blending_strategy": "temporal_ensemble", "temporal_smoothing_enabled": False},
        {"blending_strategy": "none", "temporal_smoothing_enabled": True},
        {"executor_type": "benchmark", "scheduler_mode": "continuous"},
        {"executor_type": "topic", "scheduler_mode": "wait_for_feedback"},
        {"chunking_strategy": "rtc"},
    ],
)
def test_startup_rejects_invalid_combinations(ros_context, settings):
    with pytest.raises(ValueError):
        ActionDispatcherNode(parameter_overrides=[Parameter(name, value=value) for name, value in settings.items()])


def test_cli_blending_without_smoothing_uses_resolver_default(ros_context, monkeypatch):
    monkeypatch.setattr("action_dispatch.action_dispatcher_node.create_executor", lambda *args: Mock())
    node = ActionDispatcherNode(cli_args=["--ros-args", "-p", "blending_strategy:=temporal_ensemble"])
    try:
        assert node._smoothing_enabled
        assert node._smoother is not None
    finally:
        node.destroy_node()

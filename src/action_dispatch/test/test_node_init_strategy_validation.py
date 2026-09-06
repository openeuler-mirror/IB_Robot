"""Node-level init-time defensive strategy validation.

The launch builders validate strategy selections at launch-plan time; these
tests pin the second defensive layer: both dispatcher nodes fail fast in
``__init__`` on unknown chunking names and on the removed smoothing flag.
The strategy resolution happens
before any executor/scheduler construction, so the nodes are exercised with
real parameters and no further fixtures.
"""

from __future__ import annotations

import pytest
import rclpy
from rclpy.parameter import Parameter

from action_dispatch.action_dispatcher_node import ActionDispatcherNode
from action_dispatch.scheduled_action_dispatcher_node import ScheduledActionDispatcherNode
from robot_config.dispatch_strategies import DispatchStrategyError


def _overrides(**kwargs) -> list[Parameter]:
    return [Parameter(name=name, value=value) for name, value in kwargs.items()]


@pytest.fixture
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_legacy_node_init_rejects_unknown_chunking_strategy(ros_context):
    with pytest.raises(DispatchStrategyError, match="unknown chunking strategy"):
        ActionDispatcherNode(parameter_overrides=_overrides(chunking_strategy="adaptive"))


def test_legacy_node_init_rejects_benchmark_auto_horizon_combination(ros_context):
    """Benchmark episodes execute full chunks; the SSOT rejects the pairing."""
    with pytest.raises(DispatchStrategyError, match="requires executor type 'topic'"):
        ActionDispatcherNode(
            parameter_overrides=_overrides(
                executor_type="benchmark",
                scheduler_mode="wait_for_feedback",
                chunking_strategy="auto_horizon",
            )
        )


def test_legacy_node_init_accepts_auto_horizon_for_topic_executor(ros_context):
    node = ActionDispatcherNode(parameter_overrides=_overrides(chunking_strategy="auto_horizon"))
    assert node._chunking_strategy == "auto_horizon"


def test_legacy_node_init_rejects_removed_smoothing_flag(ros_context):
    with pytest.raises(ValueError, match="removed"):
        ActionDispatcherNode(
            parameter_overrides=_overrides(
                blending_strategy="temporal_ensemble",
                temporal_smoothing_enabled=False,
            )
        )


def test_scheduled_node_init_rejects_unknown_chunking_strategy(ros_context):
    with pytest.raises(DispatchStrategyError, match="unknown chunking strategy"):
        ScheduledActionDispatcherNode(parameter_overrides=_overrides(chunking_strategy="adaptive"))


def test_scheduled_node_init_rejects_removed_smoothing_flag(ros_context):
    with pytest.raises(ValueError, match="removed"):
        ScheduledActionDispatcherNode(
            parameter_overrides=_overrides(
                blending_strategy="temporal_ensemble",
                temporal_smoothing_enabled=False,
            )
        )

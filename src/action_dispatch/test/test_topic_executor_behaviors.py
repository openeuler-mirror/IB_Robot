"""Topic behavior coverage using captured ROS messages, without DDS endpoints."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from action_dispatch.executors import topic as topic_executor_module
from action_dispatch.executors.completion import CompletionStatus, ExecutionContext
from action_dispatch.executors.topic import TopicExecutor
from action_dispatch.safe_stop import (
    JointSnapshot,
    SafeStopError,
    build_safe_stop_plan,
    construct_safety_command,
    validate_joint_state,
)
from action_dispatch.schedulers.continuous import should_replenish_plan


class CapturingNode:
    def __init__(self):
        self.messages = {}

    def create_publisher(self, message_type, topic, qos):
        messages = self.messages.setdefault(topic, [])
        return SimpleNamespace(publish=messages.append)

    def get_logger(self):
        return logging.getLogger(__name__)


def spec(topic="/arm", names=("joint1", "joint2"), ros_type="std_msgs/msg/Float64MultiArray", behavior="hold"):
    return SimpleNamespace(topic=topic, names=list(names), ros_type=ros_type, safety_behavior=behavior)


def make_executor(*specs):
    node = CapturingNode()
    executor = TopicExecutor(node, {"action_specs": list(specs)})
    assert executor.initialize()
    return executor, node.messages


@pytest.mark.parametrize("ros_type", ["std_msgs/msg/Float64MultiArray", "trajectory_msgs/msg/JointTrajectory"])
def test_safety_command_holds_joint_target_but_zeros_twist(ros_type):
    specs = [spec(ros_type=ros_type), spec("/base", ("vx", "vy", "wz"), "geometry_msgs/msg/Twist", "zeros")]
    executor, published = make_executor(*specs)
    action = np.array([0.4, -0.6, 0.5, -0.2, 0.3])
    assert executor.execute(action)
    plan = build_safe_stop_plan(action_specs=specs, joint_order=["joint1", "joint2"])
    commands = construct_safety_command(
        plan=plan, last_action=action, joint_snapshot=JointSnapshot(positions=[9.0, 9.0], valid=True)
    )
    assert commands == [[0.4, -0.6], [0.0, 0.0, 0.0]]
    for index in (1, 0):
        assert executor.execute_channel(specs[index].topic, np.asarray(commands[index]))
    assert all(len(messages) == 2 for messages in published.values())
    arm = published["/arm"][-1]
    assert list(arm.data if ros_type.endswith("Float64MultiArray") else arm.points[0].positions) == [0.4, -0.6]
    twist = published["/base"][-1]
    assert (twist.linear.x, twist.linear.y, twist.linear.z) == (0.0, 0.0, 0.0)
    assert (twist.angular.x, twist.angular.y, twist.angular.z) == (0.0, 0.0, 0.0)


def test_twist_hold_is_rejected_before_creating_any_publishers():
    node = CapturingNode()
    executor = TopicExecutor(
        node, {"action_specs": [spec(), spec("/base", ("vx", "vy", "wz"), "geometry_msgs/msg/Twist", "hold")]}
    )
    with pytest.raises(ValueError, match="requires safety_behavior='zeros'"):
        executor.initialize()
    assert node.messages == {}


@pytest.mark.parametrize("has_observation", [False, True])
def test_hold_without_last_action_uses_observation_or_refuses_to_fabricate(has_observation):
    specs = [spec()]
    executor, published = make_executor(*specs)
    plan = build_safe_stop_plan(action_specs=specs, joint_order=["joint1", "joint2"])
    snapshot = (
        validate_joint_state(
            joint_names=["joint2", "joint1"], positions=[0.8, 0.3], expected_joint_order=["joint1", "joint2"]
        )
        if has_observation
        else JointSnapshot()
    )
    if has_observation:
        commands = construct_safety_command(plan=plan, last_action=None, joint_snapshot=snapshot)
        assert executor.execute_channel("/arm", np.asarray(commands[0]))
        assert list(published["/arm"][-1].data) == [0.3, 0.8]
    else:
        with pytest.raises(SafeStopError, match="refusing to fabricate"):
            construct_safety_command(plan=plan, last_action=None, joint_snapshot=snapshot)
        assert published["/arm"] == []


@pytest.mark.parametrize("entry", ["cancel", "cleanup", "get_status"])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="TopicExecutor v2 has no cancel/cleanup/get_status entrypoints"
)
def test_requested_lifecycle_entry_exists(entry):
    executor, _ = make_executor(spec())
    assert callable(getattr(executor, entry, None)), f"Missing TopicExecutor.{entry}"


def test_invalidate_pending_preserves_immediate_completion_without_republishing():
    executor, published = make_executor(spec())
    receipt = executor.submit(np.array([0.2, 0.4]), ExecutionContext(correlation_id="completed"))
    executor.invalidate_pending()
    executor.invalidate_pending()
    assert receipt.accepted
    assert receipt.immediate_completion.status is CompletionStatus.COMPLETED
    assert receipt.immediate_completion.correlation_id == "completed"
    assert executor.drain_completions() == ()
    assert len(published["/arm"]) == 1


def test_queue_watermark_and_publish_trace_report_remaining_actions(monkeypatch):
    trace = Mock(enabled=True)
    monkeypatch.setattr(topic_executor_module, "trace", trace)
    executor, published = make_executor(spec())
    decisions = []
    for index, remaining in enumerate((3, 2, 1, 0)):
        decisions.append(should_replenish_plan(remaining, 2, inference_in_progress=False))
        with topic_executor_module.execution_trace("watermark", index, index, remaining):
            assert executor.execute(
                np.array([0.2, 0.4]), {"request_id": "watermark", "execute_index": index, "queue_size": remaining}
            )
    assert decisions == [False, True, True, True]
    assert not should_replenish_plan(1, 2, inference_in_progress=True)
    assert not should_replenish_plan(1, 2, inference_in_progress=False, policy_reset_in_progress=True)
    assert [call.args for call in trace.event.call_args_list] == [("action_topic_publish",)] * 4
    assert [call.kwargs for call in trace.event.call_args_list] == [
        dict(
            timestamp_ns=None,
            origin="built-in",
            trace_id="watermark",
            consumed_index=index,
            execute_index=index,
            topic="/arm",
            values=2,
            queue_size=remaining,
        )
        for index, remaining in enumerate((3, 2, 1, 0))
    ]
    assert len(published["/arm"]) == 4


@pytest.mark.parametrize("ros_type", ["std_msgs/msg/Float64MultiArray", "trajectory_msgs/msg/JointTrajectory"])
@pytest.mark.parametrize("values", [[1.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_named_channels_reject_short_and_overlong_vectors_before_publishing(ros_type, values):
    executor, published = make_executor(spec(ros_type=ros_type), spec("/gripper", ("g1", "g2"), ros_type))
    with pytest.raises(ValueError, match="expects 4 values"):
        executor.execute(np.asarray(values))
    assert all(not messages for messages in published.values())


@pytest.mark.parametrize("ros_type", ["std_msgs/msg/Float64MultiArray", "trajectory_msgs/msg/JointTrajectory"])
@pytest.mark.parametrize("names", [(), ("joint1", "joint2")])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_legacy_channels_reject_nonfinite_vectors_before_publishing(ros_type, names, value):
    executor, published = make_executor(spec(names=names, ros_type=ros_type))
    with pytest.raises(ValueError, match="finite"):
        executor.execute(np.array([1.0, value]))
    assert published["/arm"] == []


@pytest.mark.parametrize("ros_type", ["std_msgs/msg/Float64MultiArray", "trajectory_msgs/msg/JointTrajectory"])
def test_unnamed_legacy_channel_publishes_whole_vector(ros_type):
    executor, published = make_executor(spec(names=(), ros_type=ros_type))
    assert executor.execute(np.array([1.0, 2.0, 3.0]))
    message = published["/arm"][0]
    values = message.data if ros_type.endswith("Float64MultiArray") else message.points[0].positions
    assert list(values) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("width", [4, 6])
def test_named_arm_and_unnamed_twist_still_validate_total_width(width):
    executor, published = make_executor(spec(), spec("/base", (), "geometry_msgs/msg/Twist", "zeros"))
    with pytest.raises(ValueError, match="expects 5 values"):
        executor.execute(np.zeros(width))
    assert all(not messages for messages in published.values())

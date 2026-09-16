"""Topic execution tests against the current completion-aware executor API."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

from action_dispatch.executors.base import ActionExecutor
from action_dispatch.executors.completion import CompletionStatus, ExecutionContext
from action_dispatch.executors.topic import TopicExecutor


def make_executor(*specs):
    node = Mock()
    node.create_publisher.side_effect = lambda *_: Mock()
    executor = TopicExecutor(node, {"action_specs": list(specs)})
    assert executor.initialize()
    return executor, node


def spec(topic="/unit/commands", names=None, ros_type="std_msgs/msg/Float64MultiArray"):
    return SimpleNamespace(topic=topic, names=names, ros_type=ros_type, qos=None)


def test_empty_contract_creates_no_implicit_controller_publisher():
    executor, node = make_executor()
    assert isinstance(executor, ActionExecutor)
    assert executor.executor_type == "topic"
    node.create_publisher.assert_not_called()
    assert executor.execute(np.array([]))


@pytest.mark.parametrize(
    "ros_type,message_type",
    [
        ("std_msgs/msg/Float64MultiArray", Float64MultiArray),
        ("trajectory_msgs/msg/JointTrajectory", JointTrajectory),
    ],
)
def test_publish_preserves_type_values_and_default_qos(ros_type, message_type):
    executor, node = make_executor(spec(ros_type=ros_type))
    assert executor.execute(np.array([0.5, 1.0, 1.5]))
    publisher = executor._publishers["/unit/commands"]["pub"]
    publisher.publish.assert_called_once()
    message = publisher.publish.call_args.args[0]
    assert isinstance(message, message_type)
    qos = node.create_publisher.call_args.args[2]
    assert qos.reliability == ReliabilityPolicy.RELIABLE
    assert qos.durability == DurabilityPolicy.VOLATILE
    assert qos.depth == 1
    if message_type is JointTrajectory:
        assert len(message.points) == 1
        assert message.points[0].time_from_start.nanosec == 10_000_000
        values = message.points[0].positions
    else:
        values = message.data
    assert list(values) == [0.5, 1.0, 1.5]


def test_action_slicing_uses_channel_width_and_contract_order():
    executor, _ = make_executor(spec("/arm", ["a", "b"]), spec("/gripper", ["g"]))
    assert executor.execute(np.array([1.0, 2.0, 3.0]))
    assert list(executor._publishers["/arm"]["pub"].publish.call_args.args[0].data) == [1.0, 2.0]
    assert list(executor._publishers["/gripper"]["pub"].publish.call_args.args[0].data) == [3.0]


def test_unsupported_type_rejected_before_any_publication():
    node = Mock()
    executor = TopicExecutor(node, {"action_specs": [spec(), spec(ros_type="unsupported/MessageType")]})
    with pytest.raises(ValueError, match="unsupported TopicExecutor ROS type"):
        executor.initialize()
    node.create_publisher.assert_not_called()


def test_submit_has_one_immediate_completion_and_no_deferred_output():
    executor, _ = make_executor(spec())
    context = ExecutionContext(correlation_id="unit-request", metadata={"queue_size": 7})
    receipt = executor.submit(np.array([1.0, 2.0]), context)
    assert receipt.accepted
    assert receipt.correlation_id == "unit-request"
    assert receipt.immediate_completion.status is CompletionStatus.COMPLETED
    assert receipt.immediate_completion.correlation_id == "unit-request"
    assert executor.drain_completions() == ()
    executor.invalidate_pending()
    assert executor.drain_completions() == ()
    executor._publishers["/unit/commands"]["pub"].publish.assert_called_once()


def test_publish_error_propagates_without_claiming_completion():
    executor, _ = make_executor(spec())
    executor._publishers["/unit/commands"]["pub"].publish.side_effect = RuntimeError("publish failed")
    with pytest.raises(RuntimeError, match="publish failed"):
        executor.submit(np.array([1.0]), ExecutionContext(correlation_id="failed-request"))
    assert executor.drain_completions() == ()


@pytest.mark.parametrize("count", [50, 100, 1000])
def test_continuous_stream_preserves_every_action_and_trace_metadata(count):
    executor, _ = make_executor(spec(names=["a", "b"]))
    for index in range(count):
        assert executor.execute(
            np.array([float(index), -float(index)]),
            {
                "request_id": "chunk",
                "execute_index": index,
                "queue_size": count - index,
            },
        )
    calls = executor._publishers["/unit/commands"]["pub"].publish.call_args_list
    assert len(calls) == count
    for index, call in enumerate(calls):
        assert list(call.args[0].data) == [float(index), -float(index)]


def test_channel_execution_does_not_publish_other_channels():
    executor, _ = make_executor(spec("/arm", ["a", "b"]), spec("/gripper", ["g"]))
    assert executor.execute_channel("/gripper", np.array([0.5]))
    executor._publishers["/arm"]["pub"].publish.assert_not_called()
    assert list(executor._publishers["/gripper"]["pub"].publish.call_args.args[0].data) == [0.5]
    with pytest.raises(ValueError, match="expects 2"):
        executor.execute_channel("/arm", np.array([1.0]))
    with pytest.raises(ValueError, match="no TopicExecutor publisher"):
        executor.execute_channel("/missing", np.array([1.0]))

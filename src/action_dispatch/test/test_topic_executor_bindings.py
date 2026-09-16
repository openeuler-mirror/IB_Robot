"""Contract-bound publishers, exercised without creating any DDS endpoints."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Twist
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

from action_dispatch.executors.completion import CompletionStatus, ExecutionContext
from action_dispatch.executors.topic import TopicExecutor
from robot_config.contract_utils import ActionSpec, Contract, iter_specs


class FakeNode:
    def __init__(self):
        self.publishers = {}
        self.logger = Mock()

    def create_publisher(self, message_type, topic, qos):
        publisher = Mock(message_type=message_type, qos=qos)
        self.publishers[topic] = publisher
        return publisher

    def get_logger(self):
        return self.logger


def _spec(topic="/unit/velocity", ros_type="geometry_msgs/msg/Twist", names=("action.0", "action.1", "action.2")):
    return SimpleNamespace(topic=topic, ros_type=ros_type, names=list(names))


def _contract_specs(*actions):
    contract = Contract(
        name="test",
        version=1,
        rate_hz=50,
        max_duration_s=10,
        observations=[],
        actions=list(actions),
        tasks=[],
        recording={},
    )
    return list(iter_specs(contract))


def _assert_twist(message, values):
    assert isinstance(message, Twist)
    assert [message.linear.x, message.linear.y, message.angular.z] == pytest.approx(values)
    assert [message.linear.z, message.angular.x, message.angular.y] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "ros_type,message_type",
    [
        ("std_msgs/msg/Float64MultiArray", Float64MultiArray),
        ("trajectory_msgs/msg/JointTrajectory", JointTrajectory),
        ("geometry_msgs/msg/Twist", Twist),
    ],
)
def test_publisher_uses_action_specview_qos(ros_type, message_type):
    specs = _contract_specs(
        ActionSpec(
            key="action",
            publish_topic="/unit/command",
            type=ros_type,
            selector={"names": ["action.0", "action.1", "action.2"]},
            publish_qos={
                "reliability": "best_effort",
                "history": "keep_last",
                "depth": 7,
                "durability": "transient_local",
            },
        )
    )
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": specs})
    assert executor.initialize()

    publisher = node.publishers["/unit/command"]
    assert publisher.message_type is message_type
    assert publisher.qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert publisher.qos.history == HistoryPolicy.KEEP_LAST
    assert publisher.qos.depth == 7
    assert publisher.qos.durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_configured_qos_is_per_publisher_and_uses_shared_helper_defaults():
    configured = _spec()
    configured.qos = {
        "reliability": "best_effort",
        "history": "keep_all",
        "depth": 7,
        "durability": "transient_local",
    }
    partial = _spec("/arm", "std_msgs/msg/Float64MultiArray", ["action.3"])
    partial.qos = {"depth": 4}
    default = _spec("/gripper", "trajectory_msgs/msg/JointTrajectory", ["action.4"])
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": [configured, partial, default]})
    assert executor.initialize()
    qos = node.publishers["/unit/velocity"].qos
    assert qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert qos.history == HistoryPolicy.KEEP_ALL
    assert qos.depth == 7
    assert qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    for topic, depth in [("/arm", 4), ("/gripper", 1)]:
        qos = node.publishers[topic].qos
        assert qos.reliability == ReliabilityPolicy.RELIABLE
        assert qos.history == HistoryPolicy.KEEP_LAST
        assert qos.durability == DurabilityPolicy.VOLATILE
        assert qos.depth == depth


@pytest.mark.parametrize("qos", [None, {}])
@pytest.mark.parametrize("method", ["execute", "execute_channel"])
def test_legacy_specview_defaults_and_message_contents_are_unchanged(qos, method):
    specs = _contract_specs(
        ActionSpec(
            key="action.arm",
            publish_topic="/arm",
            type="std_msgs/msg/Float64MultiArray",
            selector={"names": ["action.0", "action.1"]},
            publish_qos=qos,
        ),
        ActionSpec(
            key="action.gripper",
            publish_topic="/gripper",
            type="trajectory_msgs/msg/JointTrajectory",
            selector={"names": ["action.2"]},
            publish_qos=qos,
        ),
    )
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": specs})
    assert executor.initialize()
    for publisher in node.publishers.values():
        assert publisher.qos.reliability == ReliabilityPolicy.RELIABLE
        assert publisher.qos.history == HistoryPolicy.KEEP_LAST
        assert publisher.qos.durability == DurabilityPolicy.VOLATILE
        assert publisher.qos.depth == 1

    if method == "execute":
        assert executor.execute(np.array([1, 2, 3], dtype=np.float32))
    else:
        assert executor.execute_channel("/arm", np.array([1, 2], dtype=np.float32))
        assert executor.execute_channel("/gripper", np.array([3], dtype=np.float32))
    array = node.publishers["/arm"].publish.call_args.args[0]
    assert isinstance(array, Float64MultiArray)
    assert list(array.data) == [1.0, 2.0]
    trajectory = node.publishers["/gripper"].publish.call_args.args[0]
    assert isinstance(trajectory, JointTrajectory)
    assert trajectory.joint_names == []
    assert len(trajectory.points) == 1
    point = trajectory.points[0]
    assert list(point.positions) == [3.0]
    assert list(point.velocities) == list(point.accelerations) == list(point.effort) == []
    assert (point.time_from_start.sec, point.time_from_start.nanosec) == (0, 10_000_000)


@pytest.mark.parametrize("ros_type", ["std_msgs/msg/Float64MultiArray", "trajectory_msgs/msg/JointTrajectory"])
def test_legacy_nameless_channel_still_publishes_the_full_vector(ros_type):
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": [_spec("/arm", ros_type, [])]})
    assert executor.initialize()
    assert executor.execute(np.array([1.0, 2.0, 3.0, 4.0]))
    message = node.publishers["/arm"].publish.call_args.args[0]
    values = message.data if isinstance(message, Float64MultiArray) else message.points[0].positions
    assert list(values) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("names", [[], ["action.0", "action.1", "action.2"]])
@pytest.mark.parametrize("method", ["execute", "execute_channel"])
def test_twist_publishes_exactly_vx_vy_wz(names, method):
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": [_spec(names=names)]})
    assert executor.initialize()
    action = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    if method == "execute":
        assert executor.execute(action)
    else:
        assert executor.execute_channel("/unit/velocity", action)

    publisher = node.publishers["/unit/velocity"]
    publisher.publish.assert_called_once()
    _assert_twist(publisher.publish.call_args.args[0], action)


def test_mixed_channels_slice_in_contract_order_and_stop_only_twist_channel():
    node = FakeNode()
    specs = [
        _spec("/arm", "std_msgs/msg/Float64MultiArray", ["action.0", "action.1"]),
        _spec(names=[]),
        _spec("/gripper", "trajectory_msgs/msg/JointTrajectory", ["action.5"]),
    ]
    executor = TopicExecutor(node, {"action_specs": specs})
    assert executor.initialize()
    assert executor.execute(np.array([1.0, 2.0, 0.25, -0.5, 0.75, 3.0]))
    assert list(node.publishers["/arm"].publish.call_args.args[0].data) == [1.0, 2.0]
    _assert_twist(node.publishers["/unit/velocity"].publish.call_args.args[0], [0.25, -0.5, 0.75])
    assert list(node.publishers["/gripper"].publish.call_args.args[0].points[0].positions) == [3.0]

    assert executor.execute_channel("/unit/velocity", np.zeros(3))
    node.publishers["/arm"].publish.assert_called_once()
    node.publishers["/gripper"].publish.assert_called_once()
    assert node.publishers["/unit/velocity"].publish.call_count == 2
    _assert_twist(node.publishers["/unit/velocity"].publish.call_args.args[0], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("names", [[], ["action.0", "action.1", "action.2"]])
@pytest.mark.parametrize("width", [0, 1, 2, 4, 6])
@pytest.mark.parametrize("method", ["execute", "execute_channel"])
def test_twist_rejects_wrong_width_without_publishing(names, width, method):
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": [_spec(names=names)]})
    assert executor.initialize()
    with pytest.raises(ValueError, match="expects 3 values"):
        if method == "execute":
            executor.execute(np.zeros(width))
        else:
            executor.execute_channel("/unit/velocity", np.zeros(width))
    node.publishers["/unit/velocity"].publish.assert_not_called()


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, np.nan]])
def test_invalid_mixed_vector_is_rejected_before_any_channel_publishes(values):
    node = FakeNode()
    specs = [_spec("/arm", "std_msgs/msg/Float64MultiArray", ["action.0", "action.1"]), _spec()]
    executor = TopicExecutor(node, {"action_specs": specs})
    assert executor.initialize()
    with pytest.raises(ValueError):
        executor.execute(np.asarray(values))
    for publisher in node.publishers.values():
        publisher.publish.assert_not_called()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("method", ["execute", "execute_channel"])
def test_twist_rejects_non_finite_values(value, method):
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": [_spec()]})
    assert executor.initialize()
    with pytest.raises(ValueError, match="finite"):
        if method == "execute":
            executor.execute(np.array([0.0, value, 0.0]))
        else:
            executor.execute_channel("/unit/velocity", np.array([0.0, value, 0.0]))
    node.publishers["/unit/velocity"].publish.assert_not_called()


@pytest.mark.parametrize("width", [1, 2, 4, 6])
def test_twist_rejects_invalid_selector_width_before_creating_publishers(width):
    node = FakeNode()
    specs = [_spec("/arm", "std_msgs/msg/Float64MultiArray", ["action.0"]), _spec(names=["value"] * width)]
    executor = TopicExecutor(node, {"action_specs": specs})
    with pytest.raises(ValueError, match="expects exactly 3 components"):
        executor.initialize()
    assert node.publishers == {}
    node.logger.info.assert_not_called()


def test_mixed_twist_and_unknown_channel_width_fail_closed():
    node = FakeNode()
    specs = [_spec("/arm", "std_msgs/msg/Float64MultiArray", []), _spec()]
    executor = TopicExecutor(node, {"action_specs": specs})
    with pytest.raises(ValueError, match="require selector.names"):
        executor.initialize()
    assert node.publishers == {}


def test_duplicate_twist_topics_cannot_silently_drop_a_channel():
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": [_spec(), _spec()]})
    with pytest.raises(ValueError, match="require distinct topics"):
        executor.initialize()
    assert node.publishers == {}


@pytest.mark.parametrize(
    "ros_type",
    [
        "geometry_msgs/msg/TwistStamped",
        "trajectory_msgs/msg/JointTrajectoryPoint",
        "custom_msgs/msg/Float64MultiArray",
        "",
    ],
)
def test_unsupported_types_fail_before_creating_or_logging_publishers(ros_type):
    node = FakeNode()
    specs = [_spec("/arm", "std_msgs/msg/Float64MultiArray", ["action.0"]), _spec(ros_type=ros_type)]
    executor = TopicExecutor(node, {"action_specs": specs})
    with pytest.raises(ValueError, match="unsupported TopicExecutor ROS type"):
        executor.initialize()
    assert node.publishers == {}
    node.logger.info.assert_not_called()
    with pytest.raises(ValueError, match="no TopicExecutor publisher"):
        executor.execute_channel("/unit/velocity", np.zeros(3))


def test_submit_twist_publishes_once_and_completes_synchronously():
    node = FakeNode()
    executor = TopicExecutor(node, {"action_specs": [_spec()]})
    assert executor.initialize()
    receipt = executor.submit(np.array([0.25, -0.5, 0.75]), ExecutionContext(correlation_id="twist-1"))
    assert receipt.accepted
    assert receipt.correlation_id == "twist-1"
    assert receipt.immediate_completion.status == CompletionStatus.COMPLETED
    assert executor.drain_completions() == ()
    node.publishers["/unit/velocity"].publish.assert_called_once()

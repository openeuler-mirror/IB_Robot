"""Topic-based ActionExecutor.

This is the real TopicExecutor implementation moved from ``topic_executor.py``.
Behaviour is preserved exactly: ROS topics, QoS, action spec ordering, value
conversion, JointTrajectory 10ms point and trace metadata. Only the location and
the ``ActionExecutor`` base class are new.

completion-aware executor contract adds ``submit`` and ``drain_completions`` so the topic executor satisfies
the v2 contract. ``submit`` reuses the existing ``execute`` publish path so
there is exactly one publish per submission; it returns an immediate
``COMPLETED`` receipt. ``drain_completions`` always returns an empty tuple
because topic publishing is synchronous.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robot_config.tracing_utils import create_trace_logger

from .base import ActionExecutor
from .completion import (
    CompletionStatus,
    ExecutionCompletion,
    ExecutionContext,
    ExecutionReceipt,
)

_trace = create_trace_logger("ib_trace.execute")


class TopicExecutor(ActionExecutor):
    """Topic-based action executor for high-frequency position control.

    Uses action_specs from contract to route actions to correct topics. Supports
    ``Float64MultiArray`` and ``JointTrajectory`` message types with
    ``RELIABLE + VOLATILE + depth=1`` QoS.
    """

    def __init__(self, node: Node, config: dict[str, Any]):
        self.node = node
        self.action_specs = config.get("action_specs", [])
        self._publishers: dict[str, Any] = {}

        # Use Reliable delivery so ros2_control command subscribers accept live action topics.
        self._qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=1)

    @property
    def executor_type(self) -> str:
        return "topic"

    def initialize(self) -> bool:
        """Initialize publishers based on contract."""
        for spec in self.action_specs:
            topic = spec.topic
            if not topic:
                continue

            if "Float64MultiArray" in spec.ros_type:
                pub = self.node.create_publisher(Float64MultiArray, topic, self._qos)
                self._publishers[topic] = {"pub": pub, "type": "float", "spec": spec}
            elif "JointTrajectory" in spec.ros_type:
                pub = self.node.create_publisher(JointTrajectory, topic, self._qos)
                self._publishers[topic] = {"pub": pub, "type": "trajectory", "spec": spec}

            self.node.get_logger().info(f"Created publisher for {topic}")
        return True

    def execute(self, action: np.ndarray, metadata: Mapping[str, Any] | None = None) -> bool:
        """Route action to publishers."""
        metadata = metadata or {}
        request_id = str(metadata.get("request_id", ""))
        execute_index = int(metadata.get("execute_index", -1))
        queue_size = int(metadata.get("queue_size", -1))

        # Flat tracking of index in the action vector
        current_idx = 0

        for topic, info in self._publishers.items():
            spec = info["spec"]

            # Determine how many joints this topic expects
            num_joints = len(spec.names) if spec.names else 0

            # 1. Slice action based on expected joint count
            if num_joints > 0:
                data = action[current_idx : current_idx + num_joints]
                current_idx += num_joints
            else:
                data = action

            # 2. Convert to list of pure Python floats
            data_list = [float(x) for x in data.ravel()]

            # 3. Publish
            if info["type"] == "float":
                msg = Float64MultiArray(data=data_list)
                info["pub"].publish(msg)
            elif info["type"] == "trajectory":
                traj = JointTrajectory()
                point = JointTrajectoryPoint(positions=data_list)
                point.time_from_start.nanosec = 10000000  # 10ms
                traj.points.append(point)
                info["pub"].publish(traj)
            _trace.info(
                "[action_topic_publish] request_id=%s index=%d topic=%s values=%d queue_size=%d",
                request_id,
                execute_index,
                topic,
                len(data_list),
                queue_size,
            )
        return True

    def execute_channel(self, topic: str, action: np.ndarray) -> bool:
        """Publish one already-sliced contract channel.

        Safe-stop uses this entrypoint so zeros and hold channels can be sent in
        the required order without re-slicing a partial vector as a full action.
        """
        info = self._publishers.get(topic)
        if info is None:
            raise ValueError(f"no TopicExecutor publisher for {topic!r}")
        expected = len(info["spec"].names) if info["spec"].names else 0
        flat = np.asarray(action).reshape(-1)
        if expected and len(flat) != expected:
            raise ValueError(f"channel {topic!r} expects {expected} values, got {len(flat)}")
        data_list = [float(x) for x in flat.ravel()]
        if info["type"] == "float":
            info["pub"].publish(Float64MultiArray(data=data_list))
        elif info["type"] == "trajectory":
            trajectory = JointTrajectory()
            point = JointTrajectoryPoint(positions=data_list)
            point.time_from_start.nanosec = 10000000  # 10ms
            trajectory.points.append(point)
            info["pub"].publish(trajectory)
        _trace.info(
            "[action_topic_publish] request_id=%s index=%d topic=%s values=%d queue_size=%d",
            "safe_stop",
            -1,
            topic,
            len(data_list),
            0,
        )
        return True

    def submit(
        self,
        action: np.ndarray,
        context: ExecutionContext,
    ) -> ExecutionReceipt:
        """Submit one action, publishing exactly once via the legacy path.

        The topic executor is synchronous: it publishes immediately inside
        ``execute`` and returns True. ``submit`` reuses that path so there is
        only one publish per submission, then returns an immediate
        ``COMPLETED`` receipt with the same correlation id. Topic immediate
        completion does not carry an environment observation timestamp,
        episode id or step id because topic publishing is not a benchmark
        environment step.

        ``drain_completions`` returns an empty tuple so the immediate
        completion is processed exactly once via the receipt, never again
        via the drain queue.
        """
        metadata: Mapping[str, Any] = dict(context.metadata) if context.metadata else {}
        # Make the correlation id visible to the trace without changing the
        # existing request_id/execute_index/queue_size keys.
        metadata.setdefault("request_id", "")
        metadata.setdefault("execute_index", -1)
        metadata.setdefault("queue_size", -1)
        # Carry the correlation id into the trace metadata for diagnostics.
        metadata.setdefault("correlation_id", context.correlation_id)

        success = self.execute(action, metadata)
        if not success:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="topic publish returned False",
            )
        completion = ExecutionCompletion(
            correlation_id=context.correlation_id,
            status=CompletionStatus.COMPLETED,
            message="topic publish completed",
        )
        return ExecutionReceipt(
            correlation_id=context.correlation_id,
            accepted=True,
            message="topic publish accepted",
            immediate_completion=completion,
        )

    def drain_completions(self) -> tuple[ExecutionCompletion, ...]:
        """Return an empty tuple; topic publishing is fully synchronous."""
        return ()

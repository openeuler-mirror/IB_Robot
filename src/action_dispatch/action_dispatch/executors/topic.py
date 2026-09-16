"""Topic-based ActionExecutor.

Routes contract action channels with per-spec QoS. Legacy array and trajectory
channels retain their default QoS, ordering, value conversion, JointTrajectory
10ms point and trace metadata. Twist channels carry exactly vx/vy/wz.

completion-aware executor contract adds ``submit`` and ``drain_completions`` so the topic executor satisfies
the v2 contract. ``submit`` reuses the existing ``execute`` publish path so
there is exactly one publish per submission; it returns an immediate
``COMPLETED`` receipt. ``drain_completions`` always returns an empty tuple
because topic publishing is synchronous.
"""

import time
from collections.abc import Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from typing import Any

import numpy as np
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ibrobot_tracing import get_trace_emitter
from robot_config.contract_utils import qos_profile_from_dict

from .base import ActionExecutor
from .completion import (
    CompletionStatus,
    ExecutionCompletion,
    ExecutionContext,
    ExecutionReceipt,
)

trace = get_trace_emitter("ib_trace.execute", component_id="action_dispatcher.execute")
_trace_records = ContextVar("action_trace_records", default=None)
_execution_trace = ContextVar("action_execution_trace", default=None)
_NO_CAPTURE = nullcontext()
_MAX_CAPTURE_RECORDS = 256


class _TraceRecords(list):
    error = None


def _defer_error(exc):
    records = _trace_records.get()
    if records is None:
        raise exc
    if records.error is None:
        records.error = exc


def capture_time(*, monotonic=False):
    """Read only a trace clock; report fatal errors after the business scope."""
    try:
        return time.perf_counter() if monotonic else time.time_ns()
    except (MemoryError, SystemError) as exc:
        _defer_error(exc)
    except Exception:
        pass
    return None


class _TraceScope:
    """A synchronous call-local capture; never handles a business exception."""

    def __init__(self, variable, value, *, flush=False):
        self.variable = variable
        self.value = value
        self.flush = flush
        self.token = None
        self.previous = None

    def __enter__(self):
        try:
            self.previous = self.variable.get()
            self.token = self.variable.set(self.value)
        except (MemoryError, SystemError):
            raise
        except Exception:
            pass

    def __exit__(self, _type, _exception, _traceback):
        if self.token is not None:
            try:
                self.variable.reset(self.token)
            except (MemoryError, SystemError):
                raise
            except Exception:
                try:
                    self.variable.set(self.previous)
                except (MemoryError, SystemError):
                    raise
                except Exception:
                    pass
        if self.flush:
            try:
                for emitter, name, timestamp_ns, fields in self.value:
                    try:
                        emitter.event(name, timestamp_ns=timestamp_ns, **fields)
                    except (MemoryError, SystemError):
                        if _exception is None:
                            raise
                    except Exception:
                        pass
                error = self.value.error
                if error is not None and _exception is None:
                    raise error
            finally:
                self.value.clear()
                self.value.error = None
        return False


def capture_traces(enabled):
    """Flush bounded nested records after the caller's original lock exits."""
    if not enabled and not trace.enabled:
        return _NO_CAPTURE
    try:
        return (
            _NO_CAPTURE
            if _trace_records.get() is not None
            else _TraceScope(_trace_records, _TraceRecords(), flush=True)
        )
    except (MemoryError, SystemError):
        raise
    except Exception:
        return _NO_CAPTURE


def capture_event(emitter, name, *, timestamp_ns=None, **fields):
    if not emitter.enabled:
        return
    try:
        records = _trace_records.get()
        if records is None:
            emitter.event(name, timestamp_ns=timestamp_ns, **fields)
        elif len(records) < _MAX_CAPTURE_RECORDS:
            # Only bounded builtin scalars are retained while the business lock is held.
            captured = {}
            for key in (
                "trace_id",
                "request_id",
                "inference_id",
                "span_id",
                "parent_span_id",
                "flow_id",
                "edge_id",
                "component_id",
                "origin",
                "span_name",
            ):
                if key in fields:
                    value = fields[key]
                    if type(value) is not str or len(value) > 1024:
                        return
                    captured[key] = value
            for key, value in fields.items():
                if key in captured:
                    continue
                if len(captured) >= 32:
                    break
                if type(value) is str:
                    captured[key] = value[:1024]
                elif value is None or type(value) in (bool, int, float):
                    captured[key] = value
            records.append((emitter, name, time.time_ns() if timestamp_ns is None else timestamp_ns, captured))
    except (MemoryError, SystemError) as exc:
        _defer_error(exc)
    except Exception:
        pass


def execution_trace(request_id, consumed_index, execute_index, queue_size):
    return _TraceScope(_execution_trace, (request_id, consumed_index, execute_index, queue_size))


class TopicExecutor(ActionExecutor):
    """Topic-based action executor for position and planar velocity control.

    Uses action_specs from contract to route actions to correct topics. Supports
    ``Float64MultiArray``, ``JointTrajectory`` and ``Twist`` message types.
    Specs without QoS use ``RELIABLE + VOLATILE + depth=1``.
    Twist channels require ``safety_behavior=zeros``; holding velocity is not a stop.
    """

    def __init__(self, node: Node, config: dict[str, Any]):
        self.node = node
        self.action_specs = config.get("action_specs", [])
        self._publishers: dict[str, Any] = {}
        self._action_width: int | None = None

        # Use Reliable delivery so ros2_control command subscribers accept live action topics.
        self._qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=1)

    @property
    def executor_type(self) -> str:
        return "topic"

    def initialize(self) -> bool:
        """Initialize publishers based on contract."""
        message_types = {
            "std_msgs/msg/Float64MultiArray": (Float64MultiArray, "float"),
            "trajectory_msgs/msg/JointTrajectory": (JointTrajectory, "trajectory"),
            "geometry_msgs/msg/Twist": (Twist, "twist"),
        }
        publisher_specs = []
        for spec in self.action_specs:
            if not spec.topic:
                continue
            if spec.ros_type not in message_types:
                raise ValueError(f"unsupported TopicExecutor ROS type {spec.ros_type!r} for topic {spec.topic!r}")
            message_type, kind = message_types[spec.ros_type]
            if kind == "twist" and spec.names and len(spec.names) != 3:
                raise ValueError(f"Twist channel {spec.topic!r} expects exactly 3 components (vx, vy, wz)")
            if kind == "twist" and getattr(spec, "safety_behavior", "zeros") != "zeros":
                raise ValueError(
                    f"Twist channel {spec.topic!r} requires safety_behavior='zeros'; holding velocity is unsafe"
                )
            qos = qos_profile_from_dict(getattr(spec, "qos", None)) or self._qos
            publisher_specs.append((spec, message_type, kind, qos))

        if any(kind == "twist" for _, _, kind, _ in publisher_specs):
            if len({spec.topic for spec, _, _, _ in publisher_specs}) != len(publisher_specs):
                raise ValueError("action channels with Twist require distinct topics")
            if any(kind != "twist" and not spec.names for spec, _, kind, _ in publisher_specs):
                raise ValueError("action channels alongside Twist require selector.names to define their widths")

        if all(kind == "twist" or spec.names for spec, _, kind, _ in publisher_specs):
            self._action_width = sum(3 if kind == "twist" else len(spec.names) for spec, _, kind, _ in publisher_specs)

        for spec, message_type, kind, qos in publisher_specs:
            pub = self.node.create_publisher(message_type, spec.topic, qos)
            self._publishers[spec.topic] = {"pub": pub, "type": kind, "spec": spec}
            self.node.get_logger().info(f"Created publisher for {spec.topic}")
        return True

    def execute(self, action: np.ndarray, metadata: Mapping[str, Any] | None = None) -> bool:
        """Route action to publishers."""
        metadata = metadata or {}
        request_id = str(metadata.get("request_id", ""))
        execute_index = int(metadata.get("execute_index", -1))
        queue_size = int(metadata.get("queue_size", -1))
        trace_step = _execution_trace.get() if trace.enabled else None

        # Validate the complete vector before any publish, including earlier arm channels.
        action = np.asarray(action, dtype=float).reshape(-1)
        if not np.isfinite(action).all():
            raise ValueError("action requires finite values")
        if self._action_width is not None and action.size != self._action_width:
            raise ValueError(f"action expects {self._action_width} values, got {action.size}")

        # Flat tracking of index in the action vector
        current_idx = 0

        for topic, info in self._publishers.items():
            spec = info["spec"]

            # Twist has an intrinsic width even without selector names.
            num_joints = 3 if info["type"] == "twist" else len(spec.names) if spec.names else 0

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
            elif info["type"] == "twist":
                msg = Twist()
                msg.linear.x, msg.linear.y, msg.angular.z = data_list
                info["pub"].publish(msg)
            if trace_step:
                capture_event(
                    trace,
                    "action_topic_publish",
                    origin="built-in",
                    trace_id=trace_step[0] or request_id,
                    consumed_index=trace_step[1],
                    execute_index=trace_step[2] if trace_step[2] is not None else execute_index,
                    topic=topic,
                    values=len(data_list),
                    queue_size=trace_step[3] if trace_step[3] is not None else queue_size,
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
        expected = 3 if info["type"] == "twist" else len(info["spec"].names) if info["spec"].names else 0
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
        elif info["type"] == "twist":
            if not np.isfinite(data_list).all():
                raise ValueError(f"Twist channel {topic!r} requires finite values")
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = data_list
            info["pub"].publish(msg)
        if trace.enabled:
            capture_event(
                trace,
                "safe_stop_topic_publish",
                origin="built-in",
                topic=topic,
                values=len(data_list),
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

        with execution_trace("", None, None, None) if trace.enabled and _execution_trace.get() is None else _NO_CAPTURE:
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

"""Contract-driven mock node entry point.

This node makes the IB-Robot end-to-end inference pipeline runnable without
real hardware. It uses :mod:`robot_config.loader` to read the same YAML the
production launch consumes, compiles a :class:`MockPlan`, and:

* publishes every ``contract.observations`` topic (images + joint_states)
* subscribes to every ``contract.actions`` topic and feeds values back into
  the internal :class:`JointModel`, immediately re-publishing joint_states so
  the closed loop never lags more than one tick.

Only ``control_mode: model_inference`` is supported; the launch layer is
responsible for enforcing that and for not starting ros2_control / Gazebo
in parallel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np  # noqa: F401  # cv_bridge runtime dependency
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from hardware_mock.contract_plan import (
    ActionSpec,
    MockPlan,
    ObservationSpec,
    build_plan,
)
from hardware_mock.image_sources import make_generator
from hardware_mock.joint_model import JointModel
from hardware_mock.type_registry import resolve_msg_class


def _qos_from_dict(d: dict) -> QoSProfile:
    """Mirror robot_config / rosetta semantics so producers and consumers match."""
    rel = str((d or {}).get("reliability", "reliable")).lower()
    hist = str((d or {}).get("history", "keep_last")).lower()
    dur = str((d or {}).get("durability", "volatile")).lower()
    depth = int((d or {}).get("depth", 10))
    return QoSProfile(
        reliability=(ReliabilityPolicy.BEST_EFFORT if rel == "best_effort" else ReliabilityPolicy.RELIABLE),
        history=(HistoryPolicy.KEEP_ALL if hist == "keep_all" else HistoryPolicy.KEEP_LAST),
        depth=depth,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if dur == "transient_local" else DurabilityPolicy.VOLATILE),
    )


class ContractMockNode(Node):
    """rclpy node implementing the contract-driven mock."""

    def __init__(self) -> None:
        super().__init__("contract_mock")

        self.declare_parameter("robot_config_path", "")
        cfg_path = self.get_parameter("robot_config_path").get_parameter_value().string_value
        if not cfg_path:
            raise RuntimeError("hardware_mock contract_mock requires the 'robot_config_path' parameter")
        path = Path(cfg_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"robot_config_path does not exist: {path}")

        # Force wall-clock; no /clock will be running in pure-mock mode.
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=False)])

        # Local import to avoid an ament_python install-time dep on robot_config
        # (it is a runtime dep declared in package.xml).
        from robot_config.loader import load_robot_config_dict  # noqa: WPS433

        robot = load_robot_config_dict(path)
        self._plan: MockPlan = build_plan(robot)
        self._joints = JointModel(self._plan.joint_ids, self._plan.initial_positions)
        self._bridge = CvBridge()

        self._image_publishers: list = []
        self._joint_state_publishers: list = []
        self._image_timers: list = []
        self._joint_state_timer = None
        self._action_subs: list = []

        self._setup_publishers()
        self._setup_subscribers()
        self._log_summary()

    # -- setup ---------------------------------------------------------------

    def _setup_publishers(self) -> None:
        for obs in self._plan.observations:
            msg_cls = resolve_msg_class(obs.msg_type)
            qos = _qos_from_dict(obs.qos)
            pub = self.create_publisher(msg_cls, obs.topic, qos)
            period = 1.0 / max(obs.rate_hz, 1e-6)
            if obs.kind == "image":
                gen = make_generator(obs.image)
                self._image_publishers.append((obs, pub, gen))
                self._image_timers.append(
                    self.create_timer(
                        period,
                        lambda o=obs, p=pub, g=gen: self._publish_image(o, p, g),
                    )
                )
            elif obs.kind == "joint_state":
                self._joint_state_publishers.append((obs, pub))
                # One shared timer per joint_state observation is fine: usually
                # there is only one such topic in the contract.
                self._joint_state_timer = self.create_timer(
                    period, lambda o=obs, p=pub: self._publish_joint_state(o, p)
                )

    def _setup_subscribers(self) -> None:
        for act in self._plan.actions:
            msg_cls = resolve_msg_class(act.msg_type)  # Float64MultiArray only today
            qos = _qos_from_dict(act.qos)
            sub = self.create_subscription(
                msg_cls,
                act.topic,
                lambda msg, a=act: self._on_action(a, msg),
                qos,
            )
            self._action_subs.append(sub)

    def _log_summary(self) -> None:
        self.get_logger().info("=" * 60)
        self.get_logger().info("hardware_mock contract_mock active")
        self.get_logger().info(f"  joints: {self._plan.joint_ids}")
        for obs in self._plan.observations:
            self.get_logger().info(f"  PUB  [{obs.kind:>11}] {obs.topic} ({obs.msg_type}) @ {obs.rate_hz:.1f} Hz")
        for act in self._plan.actions:
            mapped = [self._plan.joint_ids[i] for i in act.index_to_joint_index]
            self.get_logger().info(f"  SUB  [{'action':>11}] {act.topic} ({act.msg_type}) -> joints {mapped}")
        self.get_logger().info("=" * 60)

    # -- callbacks -----------------------------------------------------------

    def _publish_image(self, obs: ObservationSpec, pub, gen) -> None:
        frame = gen()
        msg: Image = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = obs.frame_id
        pub.publish(msg)

    def _publish_joint_state(self, obs: ObservationSpec, pub) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(obs.joint_names)
        msg.position = list(self._joints.positions())
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        pub.publish(msg)

    def _on_action(self, act: ActionSpec, msg: Float64MultiArray) -> None:
        data: list[float] = list(msg.data)
        expected = len(act.index_to_joint_index)
        if len(data) != expected:
            # Don't crash; just throttle-warn and ignore. Inference may send
            # padded arrays in some configurations.
            self.get_logger().warn(
                f"action '{act.key}' on {act.topic} expected {expected} values, got {len(data)}; dropping",
                throttle_duration_sec=2.0,
            )
            return
        try:
            self._joints.set_by_index(act.index_to_joint_index, data)
        except (IndexError, ValueError) as exc:
            self.get_logger().error(f"action '{act.key}' rejected: {exc}")
            return
        # Closed loop: republish joint_state immediately so consumers see the
        # update without waiting for the next periodic tick.
        for obs, pub in self._joint_state_publishers:
            self._publish_joint_state(obs, pub)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ContractMockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

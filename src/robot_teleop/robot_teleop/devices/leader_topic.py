"""Generic named leader-input subscriber, with no hardware dependencies."""

import math
import time

from sensor_msgs.msg import JointState

from ..base_teleop import BaseTeleopDevice


class LeaderTopicDevice(BaseTeleopDevice):
    def __init__(self, config, node=None):
        super().__init__(config, node)
        self.mapping = dict(config["joint_mapping"])
        self.gripper = config["input_gripper_joint"]
        self.timeout = float(config["input_stale_s"])
        self.sample = None
        self.received = 0.0
        self.stamp = 0.0
        self.last_invalid_reason = ""
        self._stale_logged = False
        if self.gripper not in self.mapping or len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("leader joint mapping must be complete and one-to-one")

    def connect(self):
        self.subscription = self._node.create_subscription(JointState, self._config["source_topic"], self._on_sample, 1)
        self._is_connected = True
        return True

    def _on_sample(self, msg):
        now = self._node.get_clock().now().nanoseconds * 1e-9
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        reason = ""
        if msg.header.frame_id != "leader_radians_gripper_ratio_v1":
            reason = f"unexpected frame_id: {msg.header.frame_id!r}"
        elif len(msg.name) != len(msg.position):
            reason = "joint name/position length mismatch"
        elif len(set(msg.name)) != len(msg.name):
            reason = "duplicate joint names"
        elif set(msg.name) != set(self.mapping):
            reason = "joint names do not match mapping"
        elif not 0 <= now - stamp <= self.timeout:
            reason = f"source timestamp outside freshness window (age={now - stamp:.3f}s)"
        elif stamp <= self.stamp:
            reason = "non-increasing source timestamp"
        if reason:
            self.last_invalid_reason = reason
            return
        values = dict(zip(msg.name, msg.position, strict=True))
        if not all(math.isfinite(value) for value in values.values()):
            self.last_invalid_reason = "non-finite joint position"
            return
        if not 0 <= values[self.gripper] <= 1:
            self.last_invalid_reason = "gripper ratio outside [0,1]"
            return
        self.sample = {self.mapping[name]: value for name, value in values.items()}
        self.received = time.monotonic()
        self.stamp = stamp
        self.last_invalid_reason = ""
        self._stale_logged = False

    def get_joint_targets(self):
        source_age = self._node.get_clock().now().nanoseconds * 1e-9 - self.stamp
        if (
            self.sample is None
            or time.monotonic() - self.received > self.timeout
            or not 0 <= source_age <= self.timeout
        ):
            if not self._stale_logged:
                received_age = f"{time.monotonic() - self.received:.3f}s" if self.sample is not None else "unavailable"
                source_age = (
                    f"{self._node.get_clock().now().nanoseconds * 1e-9 - self.stamp:.3f}s"
                    if self.sample is not None
                    else "unavailable"
                )
                self._node.get_logger().warn(
                    f"Leader input unavailable or stale on {self._config.get('source_topic', 'unknown')}: "
                    f"last_valid_received_age={received_age}, last_valid_source_age={source_age}, "
                    f"timeout={self.timeout:.3f}s, last_invalid_reason={self.last_invalid_reason or 'none'}"
                )
                self._stale_logged = True
            return {}
        return dict(self.sample)

    def emergency_stop(self):
        self.sample = None
        self.received = 0.0

    def prepare_rearm(self):
        self.emergency_stop()
        # Exclude queued DDS samples produced before this explicit request.
        self.stamp = max(self.stamp, self._node.get_clock().now().nanoseconds * 1e-9)

    def disconnect(self):
        self.emergency_stop()
        self._is_connected = False

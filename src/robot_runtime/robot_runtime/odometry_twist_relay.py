"""The odometry-twist relay: the wheeled base's own odometry drives the simulator (D3.3).

Robot-agnostic, simulation-only (``use_sim_time``). The base node stays the
single ``odom -> base_link`` publisher; this relay subscribes its wheel
odometry and forwards the body twist it reports as a ``TwistStamped`` on the
simulator-internal command topic the ``BaseVelocityPlugin`` consumes — so the
simulated base moves by exactly the motion the wheels realised and every
gating, limit and staleness rule of the base applies unchanged.

Forwarding is validated: the child frame must equal the configured base
frame (the twist is body-frame — ``odom`` would mislabel it), every value
finite, the stamp strictly newer than the last forwarded one, and
``-max_future <= now - stamp <= max_age`` on the simulation clock. Anything
else is dropped (wrong frames counted and reported once per second).

Reset epoch: the plugin caches the last command with its receipt time and has
no reset hook, so after a simulation reset (``now`` jumps back) a cached
command would read as fresh and keep driving. The reset is detected from the
CLOCK alone — :meth:`OdometryTwistRelayLogic.observe_clock` compares
consecutive ``now`` readings (called from the odometry callback and the
always-running zero-rate timer), never from message stamps (a stamp legally
sits slightly ahead of ``now`` inside the future window, which made the old
stamp comparison misfire), and it fires even when no odometry arrives at all
(a dead publisher must not leave a stale cached command driving the base).
On a jump the relay drops its state and publishes ZERO ``TwistStamped`` at
50 Hz with the current stamp until the first valid fresh odometry of the new
epoch arrives — every zero overwrites the plugin's cached command and its
receipt time. Wrong-frame and non-finite messages are dropped without
affecting the reset detection.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


class OdometryTwistRelayLogic:
    """The validated-forwarding state machine (pure; clock injected).

    ``now``/``stamp`` are seconds on the simulation clock. The caller drives
    :meth:`observe_clock` for every clock reading it has (both from the
    odometry callback and from the zero-rate timer), :meth:`on_odometry` for
    every message and :meth:`zero_message` on its zero-rate timer while
    :attr:`resetting` is set.
    """

    def __init__(self, base_frame: str, max_future_s: float = 0.02, max_age_s: float = 0.1):
        self.base_frame = base_frame
        self.max_future_s = max_future_s
        self.max_age_s = max_age_s
        self.last_stamp: float | None = None
        self.resetting = False
        self._last_observed_now: float | None = None
        self.dropped_frames = 0
        self.dropped_invalid = 0
        self.forwarded = 0
        self.zeros = 0

    def _fresh(self, now: float, stamp: float) -> bool:
        age = now - stamp
        return -self.max_future_s <= age <= self.max_age_s

    def observe_clock(self, now: float) -> bool:
        """Feed one ``now`` reading; True when the clock jumped backwards.

        The only reset detector (review round 1): comparing consecutive clock
        readings detects a simulation reset even with no odometry arriving,
        and never confuses a legally future-stamped message with a jump.
        """
        jumped = self._last_observed_now is not None and now < self._last_observed_now
        self._last_observed_now = now
        if jumped:
            self.resetting = True
            self.last_stamp = None
        return jumped

    def on_odometry(self, message: Odometry, now: float) -> tuple[TwistStamped | None, str]:
        """Validate one odometry message; return the forwarding result.

        The result is ``(twist, reason)``: a non-None twist means forward it
        (header stamp = the odometry stamp, frame = the validated child
        frame); a None twist carries the drop reason.
        """
        self.observe_clock(now)
        child = str(message.child_frame_id or "")
        if child != self.base_frame:
            self.dropped_frames += 1
            return None, f"child_frame_id {child!r} != base frame {self.base_frame!r}"
        twist = message.twist.twist
        values = (
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        )
        if not all(math.isfinite(value) for value in values):
            self.dropped_invalid += 1
            return None, "non-finite twist values"
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if self.resetting:
            # the world was reset: only a message fresh on the NEW clock ends
            # the zero stream (the pre-reset backlog never passes the window)
            if not self._fresh(now, stamp):
                self.dropped_invalid += 1
                return None, "stale across the reset"
            self.resetting = False
            self.last_stamp = stamp
            self.forwarded += 1
            return self._stamped(twist, stamp, child), "forwarded (first fresh after reset)"
        if self.last_stamp is not None and stamp <= self.last_stamp:
            self.dropped_invalid += 1
            return None, "stamp not newer than the last forwarded one"
        if not self._fresh(now, stamp):
            self.dropped_invalid += 1
            age = now - stamp
            return None, f"stamp age {age:+.3f} s outside [-{self.max_future_s}, {self.max_age_s}]"
        self.last_stamp = stamp
        self.forwarded += 1
        return self._stamped(twist, stamp, child), "forwarded"

    @staticmethod
    def _stamped(twist, stamp: float, child: str) -> TwistStamped:
        out = TwistStamped()
        out.header.stamp.sec = int(stamp)
        out.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
        out.header.frame_id = child
        out.twist = twist
        return out

    def zero_message(self, now: float) -> TwistStamped | None:
        """A zero twist with the current stamp while resetting (else None).

        Published at the zero rate, every message overwrites the plugin's
        cached command and its receipt time — the base stays at rest no
        matter what the pre-reset backlog replays.
        """
        if not self.resetting:
            return None
        self.zeros += 1
        return self._zero(now)

    def _zero(self, now: float) -> TwistStamped:
        out = TwistStamped()
        out.header.stamp.sec = int(now)
        out.header.stamp.nanosec = int((now - int(now)) * 1e9)
        out.header.frame_id = self.base_frame
        return out


class OdometryTwistRelay(Node):
    """The ROS wrapper: odometry in, validated TwistStamped out (D3.3)."""

    def __init__(self):
        super().__init__("odometry_twist_relay")
        self.declare_parameter("twist_source_topic", "")
        self.declare_parameter("cmd_vel_topic", "")
        self.declare_parameter("base_frame", "")
        self.declare_parameter("max_future_s", 0.02)
        self.declare_parameter("max_age_s", 0.1)
        self.declare_parameter("zero_rate_hz", 50.0)
        source = str(self.get_parameter("twist_source_topic").value)
        target = str(self.get_parameter("cmd_vel_topic").value)
        base_frame = str(self.get_parameter("base_frame").value)
        if not source or not target or not base_frame:
            raise ValueError(
                "odometry_twist_relay requires the twist_source_topic, cmd_vel_topic and base_frame parameters"
            )
        self._logic = OdometryTwistRelayLogic(
            base_frame,
            max_future_s=float(self.get_parameter("max_future_s").value),
            max_age_s=float(self.get_parameter("max_age_s").value),
        )
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=10)
        self._publisher = self.create_publisher(TwistStamped, target, reliable)
        self._subscription = self.create_subscription(Odometry, source, self._on_odometry, 50)
        rate = max(float(self.get_parameter("zero_rate_hz").value), 0.1)
        self._zero_timer = self.create_timer(1.0 / rate, self._on_zero_timer)

    def _on_odometry(self, message: Odometry) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        twist, reason = self._logic.on_odometry(message, now)
        if twist is not None:
            self._publisher.publish(twist)
            return
        if "child_frame_id" in reason and self._logic.dropped_frames > 0:
            # a wrong-frame stream is a configuration error: report it once
            # per second while it persists (never per message;
            # throttle_duration_sec — rclpy loggers take one preformatted
            # message, printf-style positional args raise TypeError, review
            # round 2)
            self.get_logger().error(
                f"dropping odometry with a foreign child frame: {reason} (dropped {self._logic.dropped_frames} so far)",
                throttle_duration_sec=1.0,
            )

    def _on_zero_timer(self) -> None:
        # The zero-rate timer runs unconditionally, so the clock observation
        # here detects a reset even when odometry stops arriving entirely.
        now = self.get_clock().now().nanoseconds / 1e9
        self._logic.observe_clock(now)
        zero = self._logic.zero_message(now)
        if zero is not None:
            self._publisher.publish(zero)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryTwistRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

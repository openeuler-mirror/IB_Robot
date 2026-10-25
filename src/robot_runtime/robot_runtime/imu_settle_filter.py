"""Hold back the simulator's pre-settle IMU samples (D5, same principle as
the finite-point cloud filter).

Gazebo's IMU sensor publishes physically impossible samples until the physics
engine has settled the spawned entity on the ground: first all-zero readings
(no gravity, no spin — gravity must appear on a real sensor), then a large
settle impulse when the entity lands. FAST-LIO scales its input by the initial
mean acceleration magnitude and orients its gravity estimate from it, so
either artifact poisons the initialization and the odometry diverges from the
first scan. The provider bridges the raw IMU on an internal topic and this
robot-agnostic node republishes the stream on the advertised endpoint, holding
it back until the sensor has provably settled: the gate opens after
``settle_run_length`` consecutive samples carry a plausible gravity magnitude,
and from then on every sample passes through verbatim — in-motion spikes are
real dynamics the consumers must see. Headers are preserved.
"""

from __future__ import annotations

import math
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

#: A gravity-plausible sample magnitude window (0.5 g .. 2 g). Orientation
#: independent: the specific-force magnitude of a resting IMU is g regardless
#: of its tilt, while the pre-settle artifacts read 0 (no gravity yet) or a
#: landing impulse far above 2 g.
GRAVITY_MIN = 0.5 * 9.80665
GRAVITY_MAX = 2.0 * 9.80665

#: Consecutive gravity-plausible samples (50 ms at 200 Hz) that prove the
#: physics has settled; mirrors FAST-LIO's own MAX_INI_COUNT of ten samples.
DEFAULT_SETTLE_RUN_LENGTH = 10


def acceleration_magnitude(message: Imu) -> float:
    return math.hypot(
        float(message.linear_acceleration.x),
        float(message.linear_acceleration.y),
        float(message.linear_acceleration.z),
    )


def is_settle_sample(message: Imu) -> bool:
    """Whether one IMU sample is a pre-settle artifact (no gravity, no spin)."""
    acceleration = acceleration_magnitude(message)
    angular = math.hypot(
        float(message.angular_velocity.x),
        float(message.angular_velocity.y),
        float(message.angular_velocity.z),
    )
    return acceleration < 1e-9 and angular < 1e-9


class SettleGate:
    """Opens after the IMU has read a plausible gravity for a full run.

    Closed: every sample is held back, and only an unbroken run of
    gravity-plausible samples advances the counter (the zero window and the
    landing impulse reset it). Open: every sample is accepted, including
    motion spikes — once the physics has settled they are real dynamics.
    """

    def __init__(self, run_length: int = DEFAULT_SETTLE_RUN_LENGTH):
        if run_length < 1:
            raise ValueError("settle run length must be at least 1")
        self._run_length = run_length
        self._run = 0
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    def accept(self, message: Imu) -> bool:
        """Whether one sample may pass the gate now."""
        if self._settled:
            return True
        if GRAVITY_MIN <= acceleration_magnitude(message) <= GRAVITY_MAX:
            self._run += 1
            if self._run >= self._run_length:
                self._settled = True
            return self._settled
        self._run = 0
        return False


class ImuSettleFilter(Node):
    def __init__(self):
        super().__init__("imu_settle_filter")
        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", "")
        self.declare_parameter("settle_run_length", DEFAULT_SETTLE_RUN_LENGTH)
        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        if not input_topic or not output_topic:
            raise ValueError("imu_settle_filter requires the input_topic and output_topic parameters")
        self._gate = SettleGate(int(self.get_parameter("settle_run_length").value))
        self._publisher = self.create_publisher(Imu, output_topic, 50)
        self._subscription = self.create_subscription(Imu, input_topic, self._on_imu, 50)

    def _on_imu(self, message: Imu) -> None:
        # a stateless relay on the simulation-time stamps: accepted samples
        # pass through with their header untouched
        if self._gate.accept(message):
            self._publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = ImuSettleFilter()
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
    sys.exit(main())

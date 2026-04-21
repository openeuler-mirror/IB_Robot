"""cmd_vel bridge node for LeKiwi omni-wheel robot.

Bridges Nav2 cmd_vel to ros2_control by:
  1. IK: /cmd_vel (Twist) -> /base_velocity_controller/commands (Float64MultiArray, rad/s)
  2. FK + Odometry: /joint_states -> /odom (Odometry) + TF (odom -> base_link)

Note: ros2_control velocity command interface expects rad/s, NOT raw steps/s.
      lekiwi_hardware internally converts rad/s to raw steps/s for the motors.
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

# Wheel mount angles (matching physical robot configuration)
# Angles are relative to robot front (x-axis), counter-clockwise positive
WHEEL_MOUNT_ANGLES_DEG = np.array([240, 0, 120]) - 90  # [150, -90, 30] for [left, back, right]


def _build_kinematics_matrix(base_radius: float) -> np.ndarray:
    """Build the 3x3 kinematics matrix M.

    Each row: [cos(a), sin(a), base_radius] for wheel mount angle a.
    M @ [vx, vy, vtheta_rad] -> [v_wheel1, v_wheel2, v_wheel3] (linear speeds)
    """
    angles = np.radians(WHEEL_MOUNT_ANGLES_DEG)
    return np.array([[math.cos(a), math.sin(a), base_radius] for a in angles])


def _body_to_wheel_radps(
    vx: float,
    vy: float,
    vtheta_rad: float,
    wheel_radius: float,
    base_radius: float,
    max_radps: float,
) -> list[float]:
    """Convert body-frame velocities to wheel angular velocity (rad/s).

    Returns wheel angular velocities in rad/s for ros2_control velocity command interface.
    lekiwi_hardware internally converts rad/s to raw steps/s.
    """
    m = _build_kinematics_matrix(base_radius)
    velocity_vector = np.array([vx, vy, vtheta_rad])
    wheel_linear_speeds = m.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius  # rad/s

    # Scale if any wheel exceeds max_radps
    max_computed = max(abs(s) for s in wheel_angular_speeds) if wheel_angular_speeds.size > 0 else 0
    if max_computed > max_radps:
        scale = max_radps / max_computed
        wheel_angular_speeds = wheel_angular_speeds * scale

    return [float(s) for s in wheel_angular_speeds]


def _wheel_radps_to_body(
    wheel_radps: list[float],
    wheel_radius: float,
    base_radius: float,
) -> list[float]:
    """Convert wheel angular velocity feedback (rad/s) to body-frame velocities.

    Returns [vx, vy, vtheta] in m/s and rad/s.
    """
    if len(wheel_radps) != 3:
        return [0.0, 0.0, 0.0]

    # wheel angular velocity (rad/s) -> linear speed
    wheel_linear_speeds = [w_radps * wheel_radius for w_radps in wheel_radps]

    # Pseudo-inverse of kinematics matrix
    m = _build_kinematics_matrix(base_radius)
    m_pinv = np.linalg.pinv(m)
    body_vel = m_pinv.dot(np.array(wheel_linear_speeds))
    return [float(body_vel[0]), float(body_vel[1]), float(body_vel[2])]


class CmdVelBridgeNode(Node):
    """Bridge node: cmd_vel -> raw wheel commands (via ros2_control) + odometry."""

    def __init__(self):
        super().__init__("cmd_vel_bridge")

        # ==================== Parameters ====================
        self.declare_parameter("wheel_radius", 0.05)
        self.declare_parameter("base_radius", 0.125)
        self.declare_parameter("max_radps", 4.602)  # Max wheel angular velocity in rad/s
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("control_frequency", 50.0)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("odom_topic", "/odom")

        self.wheel_radius = self.get_parameter("wheel_radius").value
        self.base_radius = self.get_parameter("base_radius").value
        self.max_radps = self.get_parameter("max_radps").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.publish_tf = self.get_parameter("publish_tf").value
        control_freq = self.get_parameter("control_frequency").value
        self.cmd_timeout = self.get_parameter("cmd_timeout").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value
        odom_topic = self.get_parameter("odom_topic").value

        # ==================== State ====================
        # Target velocities from /cmd_vel
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vtheta = 0.0
        self.last_cmd_time: float | None = None

        # Odometry state
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0
        self.last_odom_time: float | None = None

        # Cached wheel feedback velocities from /joint_states (rad/s)
        self.wheel_feedback: list[float] | None = None

        # ==================== QoS ====================
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ==================== Subscribers ====================
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.cmd_vel_callback,
            qos_reliable,
        )

        self.joint_states_sub = self.create_subscription(
            JointState,
            joint_states_topic,
            self.joint_states_callback,
            qos_best_effort,
        )

        # ==================== Publishers ====================
        self.wheel_cmd_pub = self.create_publisher(
            Float64MultiArray,
            "/base_velocity_controller/commands",
            qos_reliable,
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            odom_topic,
            qos_reliable,
        )

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # ==================== Control Timer ====================
        timer_period = 1.0 / control_freq
        self.control_timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(
            f"cmd_vel_bridge node started "
            f"(wheel_radius={self.wheel_radius}, base_radius={self.base_radius}, "
            f"max_radps={self.max_radps}, freq={control_freq}Hz)"
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        """Cache latest cmd_vel command."""
        self.target_vx = msg.linear.x
        self.target_vy = msg.linear.y
        self.target_vtheta = msg.angular.z  # rad/s
        self.last_cmd_time = self.get_clock().now().nanoseconds / 1e9

    def joint_states_callback(self, msg: JointState) -> None:
        """Cache wheel velocity feedback from joint_states.

        Reads velocity of joints "7", "8", "9" (left, back, right wheels).
        """
        try:
            left_idx = msg.name.index("7")
            back_idx = msg.name.index("8")
            right_idx = msg.name.index("9")
            if left_idx < len(msg.velocity) and back_idx < len(msg.velocity) and right_idx < len(msg.velocity):
                self.wheel_feedback = [
                    msg.velocity[left_idx],
                    msg.velocity[back_idx],
                    msg.velocity[right_idx],
                ]
        except ValueError:
            # Joint names not found in this message
            pass

    def control_loop(self) -> None:
        """Main control loop: publish wheel commands and update odometry."""
        now = self.get_clock().now().nanoseconds / 1e9

        # ---------- IK: cmd_vel -> wheel commands ----------
        vx = self.target_vx
        vy = self.target_vy
        vtheta = self.target_vtheta

        # Check timeout
        if self.last_cmd_time is not None and (now - self.last_cmd_time) > self.cmd_timeout:
            vx = 0.0
            vy = 0.0
            vtheta = 0.0
            self.target_vx = 0.0
            self.target_vy = 0.0
            self.target_vtheta = 0.0
        elif self.last_cmd_time is None:
            vx = 0.0
            vy = 0.0
            vtheta = 0.0

        wheel_speeds = _body_to_wheel_radps(
            vx,
            vy,
            vtheta,
            self.wheel_radius,
            self.base_radius,
            self.max_radps,
        )

        cmd_msg = Float64MultiArray()
        cmd_msg.data = wheel_speeds
        self.wheel_cmd_pub.publish(cmd_msg)

        # ---------- FK + Odometry: wheel feedback -> odom ----------
        if self.wheel_feedback is not None:
            body_vel = _wheel_radps_to_body(
                self.wheel_feedback,
                self.wheel_radius,
                self.base_radius,
            )
            fk_vx, fk_vy, fk_vtheta = body_vel
            self._update_odometry(fk_vx, fk_vy, fk_vtheta, now)

    def _update_odometry(self, vx: float, vy: float, vtheta: float, now: float) -> None:
        """Integrate body-frame velocities and publish /odom + TF."""
        if self.last_odom_time is not None:
            dt = now - self.last_odom_time
            if dt <= 0 or dt > 1.0:
                # Skip invalid dt
                self.last_odom_time = now
                return

            # Integrate in world frame
            cos_theta = math.cos(self.odom_theta)
            sin_theta = math.sin(self.odom_theta)

            self.odom_x += (vx * cos_theta - vy * sin_theta) * dt
            self.odom_y += (vx * sin_theta + vy * cos_theta) * dt
            self.odom_theta += vtheta * dt

            # Normalize angle to [-pi, pi]
            self.odom_theta = math.atan2(math.sin(self.odom_theta), math.cos(self.odom_theta))

        self.last_odom_time = now

        # Build Odometry message
        ros_now = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = ros_now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.odom_x
        odom.pose.pose.position.y = self.odom_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _yaw_to_quaternion(self.odom_theta)

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = vtheta

        self.odom_pub.publish(odom)

        # Publish TF
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = ros_now
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.odom_x
            t.transform.translation.y = self.odom_y
            t.transform.translation.z = 0.0
            t.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert yaw angle to Quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

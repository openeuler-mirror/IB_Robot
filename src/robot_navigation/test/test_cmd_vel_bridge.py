"""Tests for cmd_vel_bridge_node.

Covers kinematics matrix construction, IK/FK conversions, quaternion
transformation, velocity caching, joint-state feedback, control-loop
topic integration, and odometry integration.
"""

import math

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from robot_navigation.cmd_vel_bridge_node import (
    CmdVelBridgeNode,
    _body_to_wheel_radps,
    _build_kinematics_matrix,
    _wheel_radps_to_body,
    _yaw_to_quaternion,
)

# ── constants ───────────────────────────────────────────────────────────────

WHEEL_RADIUS = 0.05
BASE_RADIUS = 0.125
MAX_RADPS = 4.602


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="module")
def node(rclpy_init):
    """Module-scoped CmdVelBridgeNode with 50Hz timer disabled."""
    n = CmdVelBridgeNode()
    # Cancel 50Hz control timer to avoid interference
    n.destroy_timer(n.control_timer)
    yield n
    n.destroy_node()


# ── helper ──────────────────────────────────────────────────────────────────


def _collect_msg(node, topic, msg_type, trigger_fn, timeout_sec=1.0):
    """Subscribe first, spin to connect, then trigger publish and collect."""
    received = []
    sub = node.create_subscription(msg_type, topic, lambda m: received.append(m), 10)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.02)
    trigger_fn()
    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while not received and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)
    return received[-1] if received else None


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestBuildKinematicsMatrix
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildKinematicsMatrix:
    def test_matrix_shape(self):
        m = _build_kinematics_matrix(BASE_RADIUS)
        assert m.shape == (3, 3)

    def test_third_column_is_base_radius(self):
        m = _build_kinematics_matrix(BASE_RADIUS)
        np.testing.assert_allclose(m[:, 2], BASE_RADIUS)

    def test_first_row_150deg(self):
        m = _build_kinematics_matrix(BASE_RADIUS)
        angle = math.radians(150)
        np.testing.assert_allclose(m[0], [math.cos(angle), math.sin(angle), BASE_RADIUS], atol=1e-10)

    def test_second_row_minus90deg(self):
        m = _build_kinematics_matrix(BASE_RADIUS)
        angle = math.radians(-90)
        np.testing.assert_allclose(m[1], [math.cos(angle), math.sin(angle), BASE_RADIUS], atol=1e-10)

    def test_third_row_30deg(self):
        m = _build_kinematics_matrix(BASE_RADIUS)
        angle = math.radians(30)
        np.testing.assert_allclose(m[2], [math.cos(angle), math.sin(angle), BASE_RADIUS], atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestBodyToWheelRadps — inverse kinematics IK
# ═══════════════════════════════════════════════════════════════════════════


class TestBodyToWheelRadps:
    def test_zero_velocity(self):
        result = _body_to_wheel_radps(0, 0, 0, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        assert result == [0.0, 0.0, 0.0]

    def test_pure_forward(self):
        # Use vx=0.2 so wheel speeds stay below max_radps (no scaling)
        vx = 0.2
        result = _body_to_wheel_radps(vx, 0, 0, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        m = _build_kinematics_matrix(BASE_RADIUS)
        expected = [float(vx * m[i, 0] / WHEEL_RADIUS) for i in range(3)]
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_pure_lateral(self):
        # Use vy=0.2 so wheel speeds stay below max_radps (no scaling)
        vy = 0.2
        result = _body_to_wheel_radps(0, vy, 0, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        m = _build_kinematics_matrix(BASE_RADIUS)
        expected = [float(vy * m[i, 1] / WHEEL_RADIUS) for i in range(3)]
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_pure_rotation(self):
        vtheta = 1.0
        result = _body_to_wheel_radps(0, 0, vtheta, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        # Third column is all base_radius, so all wheels get same contribution
        expected_val = vtheta * BASE_RADIUS / WHEEL_RADIUS
        np.testing.assert_allclose(result, [expected_val, expected_val, expected_val], atol=1e-10)

    def test_negative_inputs(self):
        pos = _body_to_wheel_radps(0.5, 0.3, 0.2, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        neg = _body_to_wheel_radps(-0.5, -0.3, -0.2, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        for p, n in zip(pos, neg, strict=False):
            assert p == pytest.approx(-n, abs=1e-10)

    def test_scaling_when_exceeds_max(self):
        result = _body_to_wheel_radps(10.0, 10.0, 10.0, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        assert max(abs(w) for w in result) == pytest.approx(MAX_RADPS, rel=1e-6)

    def test_no_scaling_below_max(self):
        result = _body_to_wheel_radps(0.01, 0.01, 0.01, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        m = _build_kinematics_matrix(BASE_RADIUS)
        v = np.array([0.01, 0.01, 0.01])
        expected = m.dot(v) / WHEEL_RADIUS
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_scaling_preserves_ratio(self):
        large = _body_to_wheel_radps(5.0, 3.0, 2.0, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        small = _body_to_wheel_radps(0.5, 0.3, 0.2, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        # Ratio between any two wheels should be the same after scaling
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(small[j]) > 1e-12:
                    ratio_large = large[i] / large[j]
                    ratio_small = small[i] / small[j]
                    assert ratio_large == pytest.approx(ratio_small, rel=1e-6)

    def test_wheel_radius_effect(self):
        r2 = _body_to_wheel_radps(1.0, 0, 0, WHEEL_RADIUS / 2, BASE_RADIUS, MAX_RADPS)
        # With half the wheel radius, angular velocity doubles (before scaling)
        # Both should hit max_radps after scaling, so check that r2 max == MAX_RADPS
        assert max(abs(w) for w in r2) == pytest.approx(MAX_RADPS, rel=1e-6)

    def test_known_values(self):
        # vx=0.5, vy=0, vtheta=0
        # wheel_linear = M @ [0.5, 0, 0] = [0.5*cos(150°), 0.5*cos(-90°), 0.5*cos(30°)]
        # = [0.5*(-0.866), 0.5*0.0, 0.5*0.866] = [-0.433, 0, 0.433]
        # wheel_radps = [-0.433/0.05, 0, 0.433/0.05] = [-8.66, 0, 8.66]
        # max_abs = 8.66 > 4.602 => scale = 4.602/8.66
        # result = [-4.602, 0, 4.602]
        result = _body_to_wheel_radps(0.5, 0, 0, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        np.testing.assert_allclose(result, [-MAX_RADPS, 0.0, MAX_RADPS], atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestWheelRadpsToBody — forward kinematics FK
# ═══════════════════════════════════════════════════════════════════════════


class TestWheelRadpsToBody:
    def test_three_zeros(self):
        result = _wheel_radps_to_body([0, 0, 0], WHEEL_RADIUS, BASE_RADIUS)
        assert result == [0.0, 0.0, 0.0]

    def test_wrong_length_returns_zeros(self):
        result = _wheel_radps_to_body([1.0, 2.0], WHEEL_RADIUS, BASE_RADIUS)
        assert result == [0.0, 0.0, 0.0]

    def test_empty_list_returns_zeros(self):
        result = _wheel_radps_to_body([], WHEEL_RADIUS, BASE_RADIUS)
        assert result == [0.0, 0.0, 0.0]

    def test_pure_rotation_feedback(self):
        # All three wheels same angular velocity => pure rotation
        val = 2.0
        result = _wheel_radps_to_body([val, val, val], WHEEL_RADIUS, BASE_RADIUS)
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0, abs=1e-10)
        assert result[2] != 0.0  # vtheta non-zero

    def test_known_values(self):
        # Manual pseudo-inverse computation
        wheel_radps = [1.0, 2.0, 3.0]
        m = _build_kinematics_matrix(BASE_RADIUS)
        m_pinv = np.linalg.pinv(m)
        wheel_linear = np.array([w * WHEEL_RADIUS for w in wheel_radps])
        expected = m_pinv.dot(wheel_linear)
        result = _wheel_radps_to_body(wheel_radps, WHEEL_RADIUS, BASE_RADIUS)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_inverse_of_known_ik(self):
        # Use small velocities to avoid IK scaling, so FK recovers original values
        vx, vy, vtheta = 0.1, 0.1, 0.1
        ik = _body_to_wheel_radps(vx, vy, vtheta, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        fk = _wheel_radps_to_body(ik, WHEEL_RADIUS, BASE_RADIUS)
        np.testing.assert_allclose(fk, [vx, vy, vtheta], atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 4. TestIKFKRoundTrip — IK/FK round-trip consistency
# ═══════════════════════════════════════════════════════════════════════════


class TestIKFKRoundTrip:
    def _roundtrip(self, vx, vy, vtheta):
        ik = _body_to_wheel_radps(vx, vy, vtheta, WHEEL_RADIUS, BASE_RADIUS, MAX_RADPS)
        return _wheel_radps_to_body(ik, WHEEL_RADIUS, BASE_RADIUS)

    def test_roundtrip_forward(self):
        # Use small velocity to avoid IK scaling
        result = self._roundtrip(0.2, 0.0, 0.0)
        np.testing.assert_allclose(result, [0.2, 0.0, 0.0], atol=1e-6)

    def test_roundtrip_lateral(self):
        result = self._roundtrip(0.0, 0.2, 0.0)
        np.testing.assert_allclose(result, [0.0, 0.2, 0.0], atol=1e-6)

    def test_roundtrip_rotation(self):
        result = self._roundtrip(0.0, 0.0, 1.0)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-6)

    def test_roundtrip_combined(self):
        result = self._roundtrip(0.1, 0.1, 0.1)
        np.testing.assert_allclose(result, [0.1, 0.1, 0.1], atol=1e-6)

    def test_roundtrip_scaled_direction_preserved(self):
        # Large velocity triggers scaling, but direction should be preserved
        vx, vy, vtheta = 10.0, 5.0, 3.0
        result = self._roundtrip(vx, vy, vtheta)
        # Normalize both vectors and compare direction
        orig = np.array([vx, vy, vtheta])
        rt = np.array(result)
        orig_norm = np.linalg.norm(orig)
        rt_norm = np.linalg.norm(rt)
        if orig_norm > 0 and rt_norm > 0:
            np.testing.assert_allclose(rt / rt_norm, orig / orig_norm, atol=1e-6)
        # Magnitude should be scaled proportionally
        assert rt_norm == pytest.approx(orig_norm, rel=0.01) or rt_norm < orig_norm

    def test_roundtrip_zero(self):
        result = self._roundtrip(0.0, 0.0, 0.0)
        assert result == [0.0, 0.0, 0.0]


# ═══════════════════════════════════════════════════════════════════════════
# 5. TestYawToQuaternion
# ═══════════════════════════════════════════════════════════════════════════


class TestYawToQuaternion:
    def test_zero_yaw(self):
        q = _yaw_to_quaternion(0.0)
        assert q.z == pytest.approx(0.0, abs=1e-10)
        assert q.w == pytest.approx(1.0, abs=1e-10)

    def test_pi_yaw(self):
        q = _yaw_to_quaternion(math.pi)
        assert q.z == pytest.approx(1.0, abs=1e-10)
        assert q.w == pytest.approx(0.0, abs=1e-10)

    def test_negative_yaw(self):
        yaw = -math.pi / 4
        q = _yaw_to_quaternion(yaw)
        assert q.z == pytest.approx(math.sin(yaw / 2.0), abs=1e-10)
        assert q.w == pytest.approx(math.cos(yaw / 2.0), abs=1e-10)

    def test_normalization(self):
        for yaw in [0, 0.5, 1.0, math.pi, -math.pi / 3, 2.5]:
            q = _yaw_to_quaternion(yaw)
            norm_sq = q.x**2 + q.y**2 + q.z**2 + q.w**2
            assert norm_sq == pytest.approx(1.0, abs=1e-10)

    def test_x_y_always_zero(self):
        for yaw in [0, 0.5, 1.0, math.pi, -math.pi / 3]:
            q = _yaw_to_quaternion(yaw)
            assert q.x == 0.0
            assert q.y == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. TestCmdVelCallback
# ═══════════════════════════════════════════════════════════════════════════


class TestCmdVelCallback:
    def test_cache_velocities(self, node):
        msg = Twist()
        msg.linear.x = 1.0
        msg.linear.y = 0.5
        msg.angular.z = 0.3
        node.cmd_vel_callback(msg)
        assert node.target_vx == 1.0
        assert node.target_vy == 0.5
        assert node.target_vtheta == 0.3

    def test_last_cmd_time_updated(self, node):
        node.last_cmd_time = None
        msg = Twist()
        msg.linear.x = 0.1
        node.cmd_vel_callback(msg)
        assert node.last_cmd_time is not None

    def test_zero_twist(self, node):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = 0.0
        node.cmd_vel_callback(msg)
        assert node.target_vx == 0.0
        assert node.target_vy == 0.0
        assert node.target_vtheta == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7. TestJointStatesCallback
# ═══════════════════════════════════════════════════════════════════════════


class TestJointStatesCallback:
    def _make_joint_state(self, names, velocities):
        msg = JointState()
        msg.name = list(names)
        msg.velocity = list(velocities)
        return msg

    def test_extracts_correct_joints(self, node):
        node.wheel_feedback = None
        msg = self._make_joint_state(["7", "8", "9"], [1.0, 2.0, 3.0])
        node.joint_states_callback(msg)
        assert node.wheel_feedback == [1.0, 2.0, 3.0]

    def test_ignores_missing_joints(self, node):
        node.wheel_feedback = [9.0, 9.0, 9.0]
        # No joint "7" present
        msg = self._make_joint_state(["8", "9"], [2.0, 3.0])
        node.joint_states_callback(msg)
        # Should remain unchanged due to ValueError
        assert node.wheel_feedback == [9.0, 9.0, 9.0]

    def test_out_of_order_joints(self, node):
        node.wheel_feedback = None
        msg = self._make_joint_state(["9", "7", "8"], [3.0, 1.0, 2.0])
        node.joint_states_callback(msg)
        # Should index by name, not position
        assert node.wheel_feedback == [1.0, 2.0, 3.0]

    def test_extra_joints_ignored(self, node):
        node.wheel_feedback = None
        msg = self._make_joint_state(
            ["1", "2", "3", "7", "8", "9"],
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
        )
        node.joint_states_callback(msg)
        assert node.wheel_feedback == [1.0, 2.0, 3.0]


# ═══════════════════════════════════════════════════════════════════════════
# 8. TestControlLoop
# ═══════════════════════════════════════════════════════════════════════════


class TestControlLoop:
    def test_publishes_wheel_commands(self, node):
        # Set a target velocity and call control_loop
        node.target_vx = 1.0
        node.target_vy = 0.0
        node.target_vtheta = 0.0
        node.last_cmd_time = node.get_clock().now().nanoseconds / 1e9

        msg = _collect_msg(
            node,
            "/base_velocity_controller/commands",
            Float64MultiArray,
            node.control_loop,
        )
        assert msg is not None
        assert len(msg.data) == 3

    def test_timeout_zeros_commands(self, node):
        node.target_vx = 1.0
        node.target_vy = 1.0
        node.target_vtheta = 1.0
        # Set last_cmd_time far in the past
        node.last_cmd_time = (node.get_clock().now().nanoseconds / 1e9) - 10.0

        msg = _collect_msg(
            node,
            "/base_velocity_controller/commands",
            Float64MultiArray,
            node.control_loop,
        )
        assert msg is not None
        for val in msg.data:
            assert val == pytest.approx(0.0, abs=1e-10)

    def test_no_cmd_publishes_zero(self, node):
        node.target_vx = 1.0
        node.target_vy = 1.0
        node.target_vtheta = 1.0
        node.last_cmd_time = None

        msg = _collect_msg(
            node,
            "/base_velocity_controller/commands",
            Float64MultiArray,
            node.control_loop,
        )
        assert msg is not None
        for val in msg.data:
            assert val == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# 9. TestOdometry
# ═══════════════════════════════════════════════════════════════════════════


class TestOdometry:
    def test_forward_integration(self, node):
        # Reset odom state
        node.odom_x = 0.0
        node.odom_y = 0.0
        node.odom_theta = 0.0
        now = 100.0
        node.last_odom_time = now - 0.02  # dt = 0.02

        node._update_odometry(1.0, 0.0, 0.0, now)
        assert node.odom_x == pytest.approx(0.02, abs=1e-6)
        assert node.odom_y == pytest.approx(0.0, abs=1e-6)

    def test_lateral_integration(self, node):
        node.odom_x = 0.0
        node.odom_y = 0.0
        node.odom_theta = 0.0
        now = 200.0
        node.last_odom_time = now - 0.02

        node._update_odometry(0.0, 1.0, 0.0, now)
        assert node.odom_y == pytest.approx(0.02, abs=1e-6)
        assert node.odom_x == pytest.approx(0.0, abs=1e-6)

    def test_rotation_integration(self, node):
        node.odom_theta = 0.0
        now = 300.0
        node.last_odom_time = now - 0.02

        node._update_odometry(0.0, 0.0, 1.0, now)
        assert node.odom_theta == pytest.approx(0.02, abs=1e-6)

    def test_angle_normalization(self, node):
        # Set odom_theta just above pi so that after adding a small vtheta*dt
        # it wraps to negative
        node.odom_theta = math.pi - 0.01
        now = 400.0
        node.last_odom_time = now - 0.02

        # Add enough rotation to push past pi
        node._update_odometry(0.0, 0.0, 2.0, now)
        assert -math.pi <= node.odom_theta <= math.pi

    def test_skip_invalid_dt(self, node):
        # dt <= 0
        node.odom_x = 0.0
        node.odom_y = 0.0
        node.odom_theta = 0.0
        now = 500.0
        node.last_odom_time = now  # dt = 0

        node._update_odometry(1.0, 1.0, 1.0, now)
        assert node.odom_x == 0.0
        assert node.odom_y == 0.0
        assert node.odom_theta == 0.0

        # dt > 1.0
        node.last_odom_time = now - 2.0
        node._update_odometry(1.0, 1.0, 1.0, now)
        assert node.odom_x == 0.0
        assert node.odom_y == 0.0
        assert node.odom_theta == 0.0

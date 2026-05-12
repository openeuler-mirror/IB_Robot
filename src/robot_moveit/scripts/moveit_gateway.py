#!/usr/bin/env python3
"""
MoveIt 2 Gateway Node for IB-Robot.

ROS Interfaces:
    Subscriptions:
        /cmd_pose (geometry_msgs/Pose) — fire-and-forget Pose commands
        /joint_states (sensor_msgs/JointState)
    Publishers:
        /robot_status/ee_pose (geometry_msgs/PoseStamped) — 10 Hz
        /moveit_gateway/motion_status (std_msgs/String) — "idle" | "executing" | "succeeded" | "failed"
    Services:
        /moveit_gateway/move_to_pose (ibrobot_msgs/MoveToPose) — synchronous move (blocks until done)
"""

import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import Constraints, OrientationConstraint
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, String

try:
    from ibrobot_msgs.srv import MoveToPose

    _HAS_MOVE_TO_POSE_SRV = True
except ImportError:
    _HAS_MOVE_TO_POSE_SRV = False

# TF2 and MoveIt 2 imports
import tf2_ros

from pymoveit2 import MoveIt2


class MoveItGateway(Node):
    def __init__(self):
        super().__init__("moveit_gateway")

        # 1. Callback Group
        self.callback_group = ReentrantCallbackGroup()

        # 2. Parameters (no defaults - fail-fast if not provided via launch file)
        self.declare_parameter("arm_group_name")
        self.declare_parameter("base_link")
        self.declare_parameter("ee_link")
        self.declare_parameter("joint_names")
        self.declare_parameter("shoulder_link")

        self.group_name = self.get_parameter("arm_group_name").value
        self.base_link = self.get_parameter("base_link").value
        self.ee_link = self.get_parameter("ee_link").value
        self.joint_names = self.get_parameter("joint_names").value
        self.shoulder_link = self.get_parameter("shoulder_link").value

        self.latest_joint_state = None
        self.get_logger().info("Initializing MoveIt Gateway for SO101...")

        # 3. TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 4. Initialize MoveIt 2 with the callback group
        try:
            self.moveit2 = MoveIt2(
                node=self,
                group_name=self.group_name,
                joint_names=self.joint_names,
                base_link_name=self.base_link,
                end_effector_name=self.ee_link,
                use_move_group_action=True,
                callback_group=self.callback_group,
            )
            self.get_logger().info("MoveIt2 interface connected")
        except Exception as e:
            self.get_logger().error(f"MoveIt2 connect failed: {e}")
            self.moveit2 = None

        # 5. Publishers and Subscribers (using the reentrant group)
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        self.ee_pose_pub = self.create_publisher(PoseStamped, "/robot_status/ee_pose", 10)

        self.cmd_pose_sub = self.create_subscription(
            Pose,
            "/cmd_pose",
            self.cmd_pose_callback,
            10,
            callback_group=self.callback_group,
        )

        # 6. Motion status publisher (for task_dispatch and external monitors)
        self.motion_status_pub = self.create_publisher(String, "/moveit_gateway/motion_status", 10)
        self._motion_status = "idle"

        # 7. MoveToPose service (synchronous move, used by task_dispatch)
        if _HAS_MOVE_TO_POSE_SRV:
            self.move_to_pose_srv = self.create_service(
                MoveToPose,
                "/moveit_gateway/move_to_pose",
                self._move_to_pose_service_cb,
                callback_group=self.callback_group,
            )
            self.get_logger().info("MoveToPose service registered")
        else:
            self.get_logger().warn(
                "ibrobot_msgs.srv.MoveToPose not available — service disabled (rebuild ibrobot_msgs to enable)"
            )

        self.timer = self.create_timer(0.1, self.publish_ee_pose, callback_group=self.callback_group)

        self.get_logger().info("MoveIt Gateway fully initialized")

    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        四元数乘法: q = q1 * q2 (使用 scipy)
        四元数格式: [x, y, z, w]
        """
        r1 = R.from_quat([q1[0], q1[1], q1[2], q1[3]])
        r2 = R.from_quat([q2[0], q2[1], q2[2], q2[3]])
        result = r1 * r2
        return tuple(result.as_quat().tolist())

    @staticmethod
    def quaternion_conjugate(q):
        """
        四元数共轭: q* = [ -x, -y, -z, w ] (使用 scipy)
        """
        r = R.from_quat([q[0], q[1], q[2], q[3]])
        return tuple(r.inv().as_quat().tolist())

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        四元数转旋转矩阵 (使用 scipy)
        q: [x, y, z, w]
        返回: 3x3旋转矩阵 (numpy array)
        """
        return R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()

    @staticmethod
    def rotation_matrix_to_quaternion(R_mat):
        """
        旋转矩阵转四元数 (使用 scipy)
        R_mat: 3x3旋转矩阵 (numpy array or nested list)
        返回: [x, y, z, w]
        """
        return tuple(R.from_matrix(R_mat).as_quat().tolist())

    def constrain_to_z_axis_only(self, quat):
        """
        只约束末端执行器的Z轴方向，放松绕Z轴的旋转 (numpy简化版)。
        这适用于5自由度机械臂，因为5个关节无法满足完整的6DOF约束。

        原理：
        - 保持Z轴方向不变（这约束了2个自由度：pitch和yaw）
        - 放松绕Z轴的旋转（释放1个自由度：roll）
        - 使用"最小旋转"原则，保持与原姿态接近

        Args:
            quat: 原始四元数 (x, y, z, w)

        Returns:
            tuple: 约束后的四元数 (x', y', z', w')
        """
        # 1. 转换为旋转矩阵
        R = self.quaternion_to_rotation_matrix(quat)

        # 2. 提取并归一化Z轴（第3列）
        z_axis = R[:, 2]
        z_norm = np.linalg.norm(z_axis)
        if z_norm > 1e-6:
            z_axis = z_axis / z_norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

        # 3. 构造新的X轴（最小旋转原则）
        orig_x_axis = R[:, 0]
        # 将原X轴投影到垂直于Z轴的平面: proj = x - (x·z) * z
        x_axis = orig_x_axis - np.dot(orig_x_axis, z_axis) * z_axis
        x_norm = np.linalg.norm(x_axis)

        if x_norm > 1e-6:
            x_axis = x_axis / x_norm
        else:
            # X轴退化，使用替代策略
            if abs(z_axis[2]) < 0.9:
                # Z轴非垂直，使用水平方向
                z_xy_norm = np.linalg.norm(z_axis[:2])
                x_axis = np.array([-z_axis[1], z_axis[0], 0.0]) / z_xy_norm
            else:
                # Z轴垂直，使用world X方向
                x_axis = np.array([1.0, 0.0, 0.0])

        # 4. Y轴 = Z × X (叉积)
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-6:
            y_axis = y_axis / y_norm

        # 5. 重建旋转矩阵（列存储：X、Y、Z轴）
        R_constrained = np.column_stack([x_axis, y_axis, z_axis])

        # 6. 转换回四元数
        q_constrained = self.rotation_matrix_to_quaternion(R_constrained)

        return q_constrained

    def project_orientation_to_shoulder_xz_plane(self, quat):
        """
        将方向四元数投影到shoulder坐标系的XZ平面 (numpy简化版)。

        流程：
        1. 获取base到shoulder的变换
        2. 将方向从base坐标系转换到shoulder坐标系
        3. 在shoulder坐标系中，将旋转矩阵的Y轴分量约束到XZ平面
        4. 转换回四元数并转回base坐标系

        Args:
            quat: base坐标系中的四元数 (x, y, z, w)

        Returns:
            tuple: 投影后的四元数 (x', y', z', w')
        """
        try:
            # 获取base到shoulder的静态变换
            transform = self.tf_buffer.lookup_transform(
                self.base_link,
                self.shoulder_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )

            # 提取变换的四元数
            base_to_shoulder_q = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )

            # shoulder到base的变换（共轭）
            shoulder_to_base_q = self.quaternion_conjugate(base_to_shoulder_q)

        except Exception as e:
            self.get_logger().warning(f"Failed to get base->shoulder transform: {e}, using identity")
            # 如果获取失败，假设base和shoulder对齐
            base_to_shoulder_q = (0.0, 0.0, 0.0, 1.0)
            shoulder_to_base_q = (0.0, 0.0, 0.0, 1.0)

        # 1. 将方向从base转换到shoulder坐标系: q_shoulder = q_base_to_shoulder * q_base
        q_shoulder = self.quaternion_multiply(base_to_shoulder_q, quat)

        self.get_logger().debug(
            f"  Base to shoulder quaternion: ({base_to_shoulder_q[0]:.3f}, {base_to_shoulder_q[1]:.3f}, {base_to_shoulder_q[2]:.3f}, {base_to_shoulder_q[3]:.3f})"
        )
        self.get_logger().debug(
            f"  Orientation in shoulder frame: ({q_shoulder[0]:.3f}, {q_shoulder[1]:.3f}, {q_shoulder[2]:.3f}, {q_shoulder[3]:.3f})"
        )

        # 2. 转换为旋转矩阵
        R_shoulder = self.quaternion_to_rotation_matrix(q_shoulder)

        # 3. 在shoulder坐标系中，约束到XZ平面（Y分量为0）
        # 提取三个轴向量
        x_axis = R_shoulder[:, 0]
        y_axis = R_shoulder[:, 1]
        z_axis = R_shoulder[:, 2]

        # 约束X轴和Z轴到XZ平面（将Y分量设为0）
        x_axis_constrained = np.array([x_axis[0], 0.0, x_axis[2]])
        z_axis_constrained = np.array([z_axis[0], 0.0, z_axis[2]])

        # 归一化X轴
        x_norm = np.linalg.norm(x_axis_constrained)
        if x_norm > 1e-6:
            x_axis = x_axis_constrained / x_norm
        else:
            x_axis = np.array([1.0, 0.0, 0.0])

        # 归一化Z轴
        z_norm = np.linalg.norm(z_axis_constrained)
        if z_norm > 1e-6:
            z_axis = z_axis_constrained / z_norm
        else:
            z_axis = np.array([0.0, 0.0, 1.0])

        # 重建Y轴 = Z × X (叉积)
        y_axis = np.cross(z_axis, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm > 1e-6:
            y_axis = y_axis / y_norm

        # 4. 重建旋转矩阵（列存储）
        R_constrained = np.column_stack([x_axis, y_axis, z_axis])

        # 5. 转换回四元数
        q_shoulder_constrained = self.rotation_matrix_to_quaternion(R_constrained)

        # 6. 转换回base坐标系: q_base = q_shoulder_to_base * q_shoulder_constrained
        q_base_constrained = self.quaternion_multiply(shoulder_to_base_q, q_shoulder_constrained)

        return q_base_constrained

    def create_orientation_constraint(self, target_quat, link_name, frame_id, tolerances=(0.3, 0.3, 0.05)):
        """
        创建带有容差的姿态约束，用于5DOF机械臂的IK求解。

        Args:
            target_quat: 目标四元数
            link_name: 约束的link（如"gripper"）
            frame_id: 参考坐标系（如"base"）
            tolerances: (x_tol, y_tol, z_tol) 容差元组（弧度）

        Returns:
            OrientationConstraint: 姿态约束对象
        """
        constraint = OrientationConstraint()
        constraint.header = Header()
        constraint.header.frame_id = frame_id
        constraint.link_name = link_name

        # 设置目标姿态
        constraint.orientation.x = target_quat[0]
        constraint.orientation.y = target_quat[1]
        constraint.orientation.z = target_quat[2]
        constraint.orientation.w = target_quat[3]

        # 设置容差（弧度）
        # X/Y轴容差较大（放松绕Z轴旋转），Z轴容差较小（保持方向）
        constraint.absolute_x_axis_tolerance = tolerances[0]
        constraint.absolute_y_axis_tolerance = tolerances[1]
        constraint.absolute_z_axis_tolerance = tolerances[2]

        # 约束权重（1.0表示严格约束）
        constraint.weight = 1.0

        return constraint

    def joint_state_callback(self, msg):
        self.latest_joint_state = msg
        # 调试：打印关节状态
        if msg is not None and hasattr(msg, "name") and hasattr(msg, "position"):
            self.get_logger().debug(f"Joint state updated: {list(msg.name)} = {[f'{p:.3f}' for p in msg.position]}")

    def cmd_pose_callback(self, msg):
        self.get_logger().info(f"Target Pose: x={msg.position.x:.3f}, y={msg.position.y:.3f}, z={msg.position.z:.3f}")
        # 计算并输出目标位置在shoulder坐标系中的Z轴坐标
        try:
            trans = self.tf_buffer.lookup_transform(
                self.shoulder_link,
                self.base_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            # 获取平移偏移（shoulder原点在base坐标系中的位置）
            trans_x = trans.transform.translation.x
            trans_y = trans.transform.translation.y
            trans_z = trans.transform.translation.z

            # 获取旋转变换
            t_x = trans.transform.rotation.x
            t_y = trans.transform.rotation.y
            t_z = trans.transform.rotation.z
            t_w = trans.transform.rotation.w

            # 目标位置在base坐标系中
            p_base = (msg.position.x, msg.position.y, msg.position.z)

            # 计算目标点相对于shoulder原点的位置向量
            # p_relative = p_base - translation
            p_relative = (p_base[0] - trans_x, p_base[1] - trans_y, p_base[2] - trans_z)

            # 应用旋转变换到相对位置向量
            R = self.quaternion_to_rotation_matrix((t_x, t_y, t_z, t_w))

            p_shoulder = (
                R[0][0] * p_relative[0] + R[0][1] * p_relative[1] + R[0][2] * p_relative[2],
                R[1][0] * p_relative[0] + R[1][1] * p_relative[1] + R[1][2] * p_relative[2],
                R[2][0] * p_relative[0] + R[2][1] * p_relative[1] + R[2][2] * p_relative[2],
            )

            # 计算距离shoulder原点的距离
            dist_shoulder = math.sqrt(p_shoulder[0] ** 2 + p_shoulder[1] ** 2 + p_shoulder[2] ** 2)

            # 计算距离base原点的距离
            dist_base = math.sqrt(p_base[0] ** 2 + p_base[1] ** 2 + p_base[2] ** 2)

            self.get_logger().info(
                f"  Target in shoulder frame: x={p_shoulder[0]:.3f}, y={p_shoulder[1]:.3f}, z={p_shoulder[2]:.3f}"
            )
            self.get_logger().info(f"  Distance from base origin: {dist_base:.3f} m")
            self.get_logger().info(f"  Distance from shoulder origin: {dist_shoulder:.3f} m")
        except Exception as e:
            self.get_logger().warning(f"Failed to transform to shoulder frame: {e}")

        self._move_with_strategies(msg.position, msg.orientation)

    def _move_with_strategies(self, position, orientation_msg) -> bool:
        """尝试多种 5-DOF 姿态策略 + 分层容差，直到 IK 成功。

        被 cmd_pose_callback 和 _move_to_pose_service_cb 共同调用，
        保证两条路径的 5-DOF 适配行为完全一致。

        Returns:
            True 表示某个策略成功，False 表示全部策略均失败。
        """
        orig_quat = (
            orientation_msg.x,
            orientation_msg.y,
            orientation_msg.z,
            orientation_msg.w,
        )

        # 零四元数保护
        if (
            abs(orig_quat[0]) < 1e-9
            and abs(orig_quat[1]) < 1e-9
            and abs(orig_quat[2]) < 1e-9
            and abs(orig_quat[3]) < 1e-9
        ):
            self.get_logger().warning("Received zero quaternion, using default orientation (0, 0, 0, 1)")
            orig_quat = (0.0, 0.0, 0.0, 1.0)

        # 5-DOF 姿态约束策略（从严格到宽松）
        strategies = [
            ("Gripper Z-axis constraint", self.constrain_to_z_axis_only(orig_quat)),
            (
                "Shoulder XZ plane projection",
                self.project_orientation_to_shoulder_xz_plane(orig_quat),
            ),
        ]

        # Fallback: 当前末端姿态（只改位置，保持现有姿态）
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            current_quat = (
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            )
            strategies.append(("Current orientation (position only)", current_quat))
        except Exception as e:
            self.get_logger().warning(f"Failed to get current orientation: {e}")
            strategies.append(("Default orientation (no rotation)", (0.0, 0.0, 0.0, 1.0)))

        # 分层容差策略（从严格到宽松）
        tolerance_strategies = [
            ("Strict tolerance", (0.1, 0.1, 0.05)),  # X/Y ~5.7°, Z ~2.8°
            ("Medium tolerance", (0.3, 0.3, 0.1)),  # X/Y ~17.2°, Z ~5.7°
            ("Relaxed tolerance", (0.5, 0.5, 0.15)),  # X/Y ~28.6°, Z ~8.6°
            ("Z-axis only", (1.0, 1.0, 0.2)),  # X/Y ~57.3°, Z ~11.5°
            ("No constraints", None),
        ]

        for strategy_name, quat in strategies:
            adjusted_pose = Pose()
            adjusted_pose.position = position
            adjusted_pose.orientation.x = quat[0]
            adjusted_pose.orientation.y = quat[1]
            adjusted_pose.orientation.z = quat[2]
            adjusted_pose.orientation.w = quat[3]

            self.get_logger().info(
                f"Trying {strategy_name}: "
                f"({orig_quat[0]:.3f}, {orig_quat[1]:.3f}, {orig_quat[2]:.3f}, {orig_quat[3]:.3f}) -> "
                f"({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})"
            )

            for tol_name, tolerances in tolerance_strategies:
                if self.solve_and_move(adjusted_pose, orientation_tolerance=tolerances):
                    self.get_logger().info(f"IK succeeded with {strategy_name} + {tol_name}")
                    return True
                else:
                    self.get_logger().debug(f"  Failed with {tol_name}, trying next...")

        self.get_logger().error("IK failed with all strategies!")
        return False

    def publish_ee_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.base_link,
                self.ee_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01),
            )
            msg = PoseStamped()
            msg.header = trans.header
            msg.pose.position.x = trans.transform.translation.x
            msg.pose.position.y = trans.transform.translation.y
            msg.pose.position.z = trans.transform.translation.z
            msg.pose.orientation = trans.transform.rotation
            self.ee_pose_pub.publish(msg)
        except Exception:
            pass

    def _publish_motion_status(self, status: str):
        """Publish motion status for external observers (task_dispatch, etc.)."""
        self._motion_status = status
        msg = String()
        msg.data = status
        self.motion_status_pub.publish(msg)

    def _move_to_pose_service_cb(self, request, response):
        """Synchronous move-to-pose service handler for task_dispatch integration.

        Performs the full IK + plan + execute pipeline and blocks until
        motion completes (or fails/times out).
        """
        t0 = time.time()
        target = request.target_pose
        self.get_logger().info(
            f"[Service] MoveToPose request: ({target.position.x:.3f}, {target.position.y:.3f}, {target.position.z:.3f})"
        )
        self._publish_motion_status("executing")

        try:
            # Apply the same 5-DOF orientation strategies as cmd_pose_callback
            success = self._move_with_strategies(target.position, target.orientation)
            if success:
                # Poll execution state instead of calling wait_until_executed(),
                # which uses rclpy.spin_once() and deadlocks under MultiThreadedExecutor.
                is_executing_attr = "_MoveIt2__is_executing"
                is_requested_attr = "_MoveIt2__is_motion_requested"

                # Phase 1: Wait for motion to actually start (goal accepted)
                start_timeout = 5.0
                t_start = time.time()
                while time.time() - t_start < start_timeout:
                    if getattr(self.moveit2, is_executing_attr, False):
                        break
                    if not getattr(self.moveit2, is_requested_attr, False):
                        break
                    time.sleep(0.05)

                # Phase 2: Wait for motion to complete
                exec_timeout = 30.0
                t_exec = time.time()
                while time.time() - t_exec < exec_timeout:
                    if not getattr(self.moveit2, is_executing_attr, False):
                        break
                    time.sleep(0.1)

                total_exec = time.time() - t_start
                if total_exec >= exec_timeout + start_timeout:
                    self.get_logger().warn(f"[Service] MoveToPose execution timed out after {total_exec:.1f}s")
                    response.success = False
                    response.message = f"Execution timed out after {total_exec:.1f}s"
                    self._publish_motion_status("failed")
                elif self.moveit2.motion_suceeded:
                    response.success = True
                    response.message = "Motion completed"
                    self._publish_motion_status("succeeded")
                else:
                    response.success = False
                    response.message = "Motion execution failed (MoveIt reported unsuccessful)"
                    self._publish_motion_status("failed")
            else:
                response.success = False
                response.message = "IK/planning failed"
                self._publish_motion_status("failed")
        except Exception as e:
            response.success = False
            response.message = f"Exception: {e}"
            self.get_logger().error(f"[Service] MoveToPose exception: {e}")
            self._publish_motion_status("failed")

        response.execution_time_s = time.time() - t0
        # Brief hold so external observers (e.g., task_dispatch) can read
        # succeeded/failed before the status transitions back to idle.
        time.sleep(0.3)
        self._publish_motion_status("idle")
        self.get_logger().info(
            f"[Service] MoveToPose result: success={response.success}, time={response.execution_time_s:.1f}s"
        )
        return response

    def solve_and_move(self, target_pose, orientation_tolerance=None):
        """
        尝试IK求解并移动到目标位姿。

        Args:
            target_pose: 目标位姿
            orientation_tolerance: 姿态容差 or None（无constraints）

        Returns:
            True表示成功，False表示失败
        """
        if not self.moveit2:
            self.get_logger().error("MoveIt2 engine not ready")
            return False

        # 打印目标位置（用于调试可达性）
        target_pos = target_pose.position
        self.get_logger().info(f"  Target position: ({target_pos.x:.3f}, {target_pos.y:.3f}, {target_pos.z:.3f})")

        # 简单的可达性检查：距离原点的距离
        dist_from_origin = math.sqrt(target_pos.x**2 + target_pos.y**2 + target_pos.z**2)
        self.get_logger().info(f"  Distance from origin: {dist_from_origin:.3f} m")

        # 打印当前关节状态（如果有）
        if self.latest_joint_state is not None and hasattr(self.latest_joint_state, "position"):
            self.get_logger().debug(f"  Current joints: {[f'{p:.2f}' for p in self.latest_joint_state.position]}")

        try:
            # 检查关节状态是否有效
            start_state = None
            if self.latest_joint_state is not None:
                # 验证关节状态是否包含所需的关节数量
                if hasattr(self.latest_joint_state, "position") and len(self.latest_joint_state.position) >= len(
                    self.joint_names
                ):
                    start_state = self.latest_joint_state
                else:
                    self.get_logger().warning(
                        f"Invalid joint state: has {len(self.latest_joint_state.position) if hasattr(self.latest_joint_state, 'position') else 0} joints, "
                        f"need {len(self.joint_names)}. Using solver's internal state."
                    )
            else:
                self.get_logger().warning("No joint state available, using solver's internal state")

            # 创建Constraints（如果指定了容差）
            constraints = None
            if orientation_tolerance is not None:
                constraints = Constraints()
                target_quat = (
                    target_pose.orientation.x,
                    target_pose.orientation.y,
                    target_pose.orientation.z,
                    target_pose.orientation.w,
                )
                constraints.orientation_constraints.append(
                    self.create_orientation_constraint(
                        target_quat=target_quat,
                        link_name=self.ee_link,
                        frame_id=self.base_link,
                        tolerances=orientation_tolerance,
                    )
                )
                self.get_logger().info(f"Using orientation tolerance: {orientation_tolerance}")

            # 1. Use async IK call to avoid internal spin_once calls
            # 只在有有效状态时才传递start_joint_state参数
            if start_state is not None:
                if constraints is not None:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position,
                        quat_xyzw=target_pose.orientation,
                        start_joint_state=start_state,
                        constraints=constraints,
                    )
                else:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position,
                        quat_xyzw=target_pose.orientation,
                        start_joint_state=start_state,
                    )
            else:
                # 不传递start_joint_state，让求解器使用内部状态
                if constraints is not None:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position,
                        quat_xyzw=target_pose.orientation,
                        constraints=constraints,
                    )
                else:
                    future = self.moveit2.compute_ik_async(
                        position=target_pose.position, quat_xyzw=target_pose.orientation
                    )

            # 2. Wait for the future safely in a MultiThreadedExecutor environment
            # Since the executor is running in parallel, it will fulfill the future.
            start_wait = time.time()
            while not future.done():
                time.sleep(0.01)
                if time.time() - start_wait > 5.0:
                    self.get_logger().error("IK Service Timeout")
                    return False

            ik_solution = self.moveit2.get_compute_ik_result(future)

            if ik_solution is not None:
                joint_positions = []
                for name in self.joint_names:
                    if name in ik_solution.name:
                        idx = ik_solution.name.index(name)
                        joint_positions.append(float(ik_solution.position[idx]))

                self.get_logger().info(f"IK Success: {joint_positions}")
                self.move_to_joint(joint_positions)
                return True
            else:
                self.get_logger().warning("IK Solver failed: No valid solution")
                # 检查IK求解器是否支持Constraints
                if orientation_tolerance is not None:
                    self.get_logger().warning(
                        "  Note: Constraints may not be supported by LMA solver. Try position_only_ik: True in kinematics.yaml"
                    )
                return False

        except Exception as e:
            self.get_logger().error(f"IK Workflow failed: {e}")
            return False

    def move_to_joint(self, joint_positions):
        if not self.moveit2:
            return False
        self.get_logger().info("Moving to joints...")
        try:
            self.moveit2.clear_goal_constraints()
            self.moveit2.move_to_configuration(joint_positions)
            return True
        except Exception as e:
            self.get_logger().error(f"Move error: {e}")
            return False


def main(args=None):
    rclpy.init(args=args)
    node = MoveItGateway()

    # Use MultiThreadedExecutor to handle concurrent callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

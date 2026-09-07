#!/usr/bin/env python3
"""Task Executor Node — orchestrates sequential task plans via MoveIt gateway.

Architecture Position:
    task_dispatch sits between task-level planners (visual_grasp, VoxPoser, etc.)
    and the motion execution layer (moveit_gateway + ros2_control).

    Planners produce a sequence of TaskSteps (waypoints, gripper commands, waits).
    This node executes them one by one:
      - MOVE_TO_POSE  → calls /moveit_gateway/move_to_pose service
      - GRIPPER       → sends FollowJointTrajectory goal to gripper controller
      - WAIT          → sleeps for the requested duration

ROS Interfaces:
    Action Server:
        ~/execute_task_plan (ibrobot_msgs/action/ExecuteTaskPlan)
    Service Client:
        /moveit_gateway/move_to_pose (ibrobot_msgs/srv/MoveToPose)
    Action Client:
        /{gripper_controller}/follow_joint_trajectory (control_msgs/action/FollowJointTrajectory)
    Subscribers:
        /joint_states (sensor_msgs/msg/JointState)

Parameters (all loaded from robot_config YAML):
    robot_config_path (str): Path to robot_config YAML
"""

import math
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ibrobot_msgs.action import ExecuteTaskPlan
from ibrobot_msgs.msg import TaskStep
from ibrobot_msgs.srv import MoveToPose
from robot_config.loader import load_robot_section


def _load_robot_yaml(config_path: str) -> dict:
    """Load the resolved robot section through the robot_config SSOT loader."""
    _, robot_config = load_robot_section(config_path)
    return robot_config


class TaskExecutorNode(Node):
    """Executes sequential task plans by delegating to moveit_gateway and gripper controllers."""

    def __init__(self):
        super().__init__("task_executor")

        self._cb_group = ReentrantCallbackGroup()
        self._action_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- Parameters ----
        self.declare_parameter("robot_config_path", "")
        config_path = self.get_parameter("robot_config_path").get_parameter_value().string_value

        if not config_path:
            self.get_logger().fatal("robot_config_path parameter is required")
            raise RuntimeError("robot_config_path parameter is required")

        self._robot_cfg = _load_robot_yaml(config_path)
        robot_name = self._robot_cfg.get("name", "unknown")
        self.get_logger().info(f"Loaded robot config: {robot_name}")

        self.declare_parameter("skip_redundant_gripper_open", False)
        self.declare_parameter("gripper_open_position", 1.0)
        self.declare_parameter("gripper_position_tolerance", 0.05)
        self.declare_parameter("joint_state_max_age_s", 0.25)
        self._skip_redundant_gripper_open = bool(self.get_parameter("skip_redundant_gripper_open").value)
        self._gripper_open_position = float(self.get_parameter("gripper_open_position").value)
        self._gripper_position_tolerance = max(float(self.get_parameter("gripper_position_tolerance").value), 0.0)
        self._joint_state_max_age_s = max(float(self.get_parameter("joint_state_max_age_s").value), 0.0)

        # Extract gripper joint name and controller action from robot_config
        self._gripper_joint = self._resolve_gripper_joint()
        self._gripper_action_name = self._resolve_gripper_action_name()

        # ---- Service Client: MoveIt gateway ----
        self._move_client = self.create_client(
            MoveToPose,
            "/moveit_gateway/move_to_pose",
            callback_group=self._cb_group,
        )

        # ---- Action Client: Gripper trajectory controller ----
        self._gripper_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self._gripper_action_name,
            callback_group=self._cb_group,
        )

        # ---- Subscriber: Joint states (for gripper feedback) ----
        self._latest_joint_state = None
        self._latest_joint_state_received_at = None
        self._joint_state_lock = threading.Lock()
        self._joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            10,
            callback_group=self._cb_group,
        )

        # ---- Action Server: ExecuteTaskPlan ----
        self._action_server = ActionServer(
            self,
            ExecuteTaskPlan,
            "~/execute_task_plan",
            execute_callback=self._execute_plan_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._action_cb_group,
        )

        # ---- Internal state ----
        self._current_goal_handle = None
        self._cancel_requested = False

        self.get_logger().info(
            f"TaskExecutor ready — gripper_joint={self._gripper_joint}, gripper_action={self._gripper_action_name}"
        )

    # ------------------------------------------------------------------ #
    #  Configuration helpers
    # ------------------------------------------------------------------ #

    def _resolve_gripper_joint(self) -> str:
        """Get gripper joint name from robot_config."""
        joints = self._robot_cfg.get("joints", {})
        gripper_joints = joints.get("gripper", [])
        if gripper_joints and len(gripper_joints) > 0:
            return gripper_joints[0]
        # Fallback: look for a joint containing 'gripper'
        all_joints = joints.get("all", [])
        for j in all_joints:
            if "gripper" in j.lower():
                return j
        self.get_logger().warn('Could not find gripper joint in robot_config, using "gripper"')
        return "gripper"

    def _resolve_gripper_action_name(self) -> str:
        """Get gripper FollowJointTrajectory action name from robot_config control_modes."""
        control_modes = self._robot_cfg.get("control_modes", {})
        for mode_name in ["moveit_planning", "visual_grasp"]:
            mode = control_modes.get(mode_name, {})
            controllers = mode.get("controllers", [])
            for ctrl in controllers:
                if "gripper" in ctrl and "trajectory" in ctrl:
                    return f"/{ctrl}/follow_joint_trajectory"
        return "/gripper_trajectory_controller/follow_joint_trajectory"

    # ------------------------------------------------------------------ #
    #  Action server callbacks
    # ------------------------------------------------------------------ #

    def _goal_cb(self, goal_request):
        """Accept or reject incoming goals."""
        if self._current_goal_handle is not None:
            self.get_logger().warn("Rejecting new goal — execution already in progress")
            return GoalResponse.REJECT
        n_steps = len(goal_request.steps)
        self.get_logger().info(f'Accepted task plan: "{goal_request.task_description}" ({n_steps} steps)')
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        """Accept cancel requests."""
        self.get_logger().info("Cancel requested")
        self._cancel_requested = True
        return CancelResponse.ACCEPT

    def _execute_plan_cb(self, goal_handle):
        """Main execution loop — iterate through task steps."""
        self._current_goal_handle = goal_handle
        self._cancel_requested = False

        steps = goal_handle.request.steps
        task_desc = goal_handle.request.task_description or goal_handle.request.task_id
        total = len(steps)
        result = ExecuteTaskPlan.Result()
        feedback = ExecuteTaskPlan.Feedback()

        self.get_logger().info(f"▶ Executing task plan: {task_desc} ({total} steps)")
        t0 = time.monotonic()

        for idx, step in enumerate(steps):
            # Check cancellation
            if self._cancel_requested:
                self.get_logger().info(f"Task cancelled at step {idx}/{total}")
                result.success = False
                result.message = f"Cancelled at step {idx}"
                result.steps_completed = idx
                result.total_duration_s = time.monotonic() - t0
                goal_handle.canceled()
                self._current_goal_handle = None
                return result

            # Publish feedback
            label = step.label or f"step_{idx}"
            feedback.current_step = idx
            feedback.total_steps = total
            feedback.current_label = label
            feedback.status = "executing"
            feedback.progress = float(idx) / max(total, 1)
            goal_handle.publish_feedback(feedback)

            self.get_logger().info(f"  [{idx + 1}/{total}] {self._step_type_name(step.type)}: {label}")

            # Dispatch by step type
            try:
                if step.type == TaskStep.MOVE_TO_POSE:
                    ok, msg = self._exec_move_to_pose(step)
                elif step.type == TaskStep.GRIPPER:
                    ok, msg = self._exec_gripper(step)
                elif step.type == TaskStep.WAIT:
                    ok, msg = self._exec_wait(step)
                else:
                    ok, msg = False, f"Unknown step type: {step.type}"

                if not ok:
                    self.get_logger().error(f"  Step {idx} failed: {msg}")
                    result.success = False
                    result.message = f"Step {idx} ({label}) failed: {msg}"
                    result.steps_completed = idx
                    result.total_duration_s = time.monotonic() - t0
                    goal_handle.abort()
                    self._current_goal_handle = None
                    return result

            except Exception as e:
                self.get_logger().error(f"  Step {idx} exception: {e}")
                result.success = False
                result.message = f"Step {idx} ({label}) exception: {e}"
                result.steps_completed = idx
                result.total_duration_s = time.monotonic() - t0
                goal_handle.abort()
                self._current_goal_handle = None
                return result

        # All steps completed
        elapsed = time.monotonic() - t0
        self.get_logger().info(f"✓ Task plan completed in {elapsed:.1f}s ({total} steps)")

        feedback.status = "completed"
        feedback.progress = 1.0
        goal_handle.publish_feedback(feedback)

        result.success = True
        result.message = "All steps completed successfully"
        result.steps_completed = total
        result.total_duration_s = elapsed
        goal_handle.succeed()
        self._current_goal_handle = None
        return result

    # ------------------------------------------------------------------ #
    #  Step executors
    # ------------------------------------------------------------------ #

    def _exec_move_to_pose(self, step: TaskStep) -> tuple:
        """Execute a MOVE_TO_POSE step via moveit_gateway service."""
        if not self._move_client.service_is_ready():
            self.get_logger().warn("Waiting for /moveit_gateway/move_to_pose service...")
            if not self._move_client.wait_for_service(timeout_sec=10.0):
                return False, "moveit_gateway/move_to_pose service not available"

        req = MoveToPose.Request()
        req.target_pose = step.target_pose
        req.velocity_scaling = step.velocity_scaling

        future = self._move_client.call_async(req)

        motion_timeout = 60.0
        t0 = time.monotonic()
        while not future.done():
            if self._cancel_requested:
                future.cancel()
                return False, "Cancelled during motion"
            if time.monotonic() - t0 > motion_timeout:
                future.cancel()
                return False, f"Motion timed out after {motion_timeout}s"
            time.sleep(0.1)

        resp = future.result()
        if resp is None:
            return False, "Service call returned None"
        if not resp.success:
            return False, resp.message
        self.get_logger().info(f"    Motion completed in {resp.execution_time_s:.1f}s")
        return True, "ok"

    def _exec_gripper(self, step: TaskStep) -> tuple:
        """Execute a GRIPPER step via FollowJointTrajectory action."""
        target_position = step.gripper_position

        if self._is_redundant_gripper_open(target_position):
            self.get_logger().info(f"    Gripper already open at {target_position:.2f}; skipping trajectory")
            return True, "already at open target"

        if not self._gripper_action_client.server_is_ready():
            self.get_logger().info("Waiting for gripper trajectory action server...")
            if not self._gripper_action_client.wait_for_server(timeout_sec=5.0):
                return False, "Gripper trajectory action server not available"

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = [self._gripper_joint]
        point = JointTrajectoryPoint()
        point.positions = [target_position]
        point.time_from_start = Duration(sec=1, nanosec=0)
        goal_msg.trajectory.points = [point]

        future = self._gripper_action_client.send_goal_async(goal_msg, feedback_callback=self._gripper_feedback_cb)

        while not future.done():
            if self._cancel_requested:
                return False, "Cancelled while sending gripper goal"
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False, "Gripper trajectory goal rejected"

        result_future = goal_handle.get_result_async()
        self.get_logger().info(f"    Gripper → {target_position:.2f}")

        while not result_future.done():
            if self._cancel_requested:
                goal_handle.cancel_goal_async()
                return False, "Cancelled during gripper motion"
            time.sleep(0.05)

        result = result_future.result()
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return False, f"Gripper failed: error_code={result.result.error_code}"

        return True, "ok"

    def _gripper_feedback_cb(self, feedback_msg):
        pass

    def _exec_wait(self, step: TaskStep) -> tuple:
        """Execute a WAIT step."""
        duration = step.wait_duration_s
        self.get_logger().info(f"    Waiting {duration:.1f}s")

        elapsed = 0.0
        while elapsed < duration:
            if self._cancel_requested:
                return False, "Cancelled during wait"
            time.sleep(min(0.1, duration - elapsed))
            elapsed += 0.1

        return True, "ok"

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _joint_state_cb(self, msg: JointState):
        with self._joint_state_lock:
            self._latest_joint_state = msg
            self._latest_joint_state_received_at = time.monotonic()

    def _is_redundant_gripper_open(self, target: float, *, now: float | None = None) -> bool:
        """Return true only when fresh feedback confirms an open-target no-op."""
        if not self._skip_redundant_gripper_open or not math.isfinite(float(target)):
            return False
        tolerance = self._gripper_position_tolerance
        if abs(float(target) - self._gripper_open_position) > tolerance:
            return False

        checked_at = time.monotonic() if now is None else float(now)
        with self._joint_state_lock:
            joint_state = self._latest_joint_state
            received_at = self._latest_joint_state_received_at
        if joint_state is None or received_at is None or checked_at - received_at > self._joint_state_max_age_s:
            return False
        if self._gripper_joint not in joint_state.name:
            return False
        index = joint_state.name.index(self._gripper_joint)
        if index >= len(joint_state.position):
            return False
        current_position = float(joint_state.position[index])
        return math.isfinite(current_position) and abs(current_position - float(target)) <= tolerance

    def _wait_for_gripper(self, target: float, timeout_s: float = 3.0, tolerance: float = 0.05):
        """Wait until gripper joint reaches target position (best-effort)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            with self._joint_state_lock:
                js = self._latest_joint_state
            if js is not None and self._gripper_joint in js.name:
                idx = js.name.index(self._gripper_joint)
                if abs(js.position[idx] - target) < tolerance:
                    return
            time.sleep(0.05)
        self.get_logger().debug(f"Gripper wait timed out after {timeout_s}s (best-effort, continuing)")

    @staticmethod
    def _step_type_name(step_type: int) -> str:
        names = {
            TaskStep.MOVE_TO_POSE: "MOVE_TO_POSE",
            TaskStep.GRIPPER: "GRIPPER",
            TaskStep.WAIT: "WAIT",
        }
        return names.get(step_type, f"UNKNOWN({step_type})")


def main(args=None):
    rclpy.init(args=args)
    node = TaskExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

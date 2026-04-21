#!/usr/bin/env python3
"""
Nav2 导航客户端

用于向Nav2发送导航目标并确认抵达状态。
语音识别和关键词匹配由 funasr_client_node 完成。

用法:
    # 独立运行
    ros2 run robot_navigation nav2_goal_client

    # 直接发送目标点
    ros2 run robot_navigation nav2_goal_client --ros-args -p x:=1.0 -p y:=2.0 -p theta:=0.0

    # 到达目标后触发 action_dispatcher 评估
    ros2 run robot_navigation nav2_goal_client --ros-args -p x:=1.0 -p trigger_evaluation:=true
"""

import json
import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class Nav2GoalClient(Node):
    """Nav2导航客户端 - 发送目标并确认抵达"""

    def __init__(self):
        super().__init__("nav2_goal_client")

        # 声明参数
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("theta", 0.0)
        self.declare_parameter("timeout_sec", 60.0)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("enable_feedback", True)
        self.declare_parameter("trigger_evaluation", False)
        self.declare_parameter("subscribe_voice", True)
        self.declare_parameter("topic_keyword_matched", "/voice_asr/keyword_matched")
        self.declare_parameter("topic_nav_stop", "/voice_asr/nav_stop")

        # 获取参数值
        self.goal_x = self.get_parameter("x").value
        self.goal_y = self.get_parameter("y").value
        self.goal_theta = self.get_parameter("theta").value
        self.timeout_sec = self.get_parameter("timeout_sec").value
        self.global_frame = self.get_parameter("global_frame").value
        self.enable_feedback = self.get_parameter("enable_feedback").value
        self.trigger_evaluation = self.get_parameter("trigger_evaluation").value
        self.subscribe_voice = self.get_parameter("subscribe_voice").value
        self.topic_keyword_matched = self.get_parameter("topic_keyword_matched").value
        self.topic_nav_stop = self.get_parameter("topic_nav_stop").value

        # 创建Action客户端
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # 创建评估服务客户端
        self.evaluation_client = self.create_client(Trigger, "/action_dispatcher/start_evaluate")

        # 订阅语音命令话题 (来自 funasr_client_node)
        if self.subscribe_voice:
            self.voice_sub = self.create_subscription(
                String, self.topic_keyword_matched, self.voice_command_callback, 10
            )
            self.get_logger().info(f"Subscribed to {self.topic_keyword_matched}")
            self.nav_stop_sub = self.create_subscription(String, self.topic_nav_stop, self.stop_callback, 10)
            self.get_logger().info(f"Subscribed to {self.topic_nav_stop}")

        # 导航状态
        self.navigation_succeeded = False
        self.navigation_failed = False
        self.feedback_received = False
        self.current_task = ""
        self.current_task_description = ""
        self.is_navigating = False
        self.goal_handle = None
        self._nav_start_time = None
        self._timeout_timer = self.create_timer(0.5, self._check_timeout)

    def voice_command_callback(self, msg: String):
        """处理来自 funasr_client_node 的语音命令

        消息格式 (JSON):
        {
            "keyword": "去.*厨房",
            "type": "destination",
            "info": {"x": 1.0, "y": 2.0, "theta": 0.0}
        }
        或
        {
            "keyword": "捡.*蓝色方块",
            "type": "action",
            "info": {"task_description": "Pick up the blue square"}
        }
        """
        try:
            data = json.loads(msg.data)
            keyword = data.get("keyword", "")
            keyword_type = data.get("type", "")
            info = data.get("info", {})

            self.get_logger().info(f"收到语音命令: keyword='{keyword}', type={keyword_type}, info={info}")

            if keyword_type == "destination":
                # 目的地导航
                x = info.get("x", 0.0)
                y = info.get("y", 0.0)
                theta = info.get("theta", 0.0)
                self.send_goal(x, y, theta, keyword)

            elif keyword_type == "action":
                # 动作关键词 - 保存任务描述
                task_description = info.get("task_description", "")
                if task_description:
                    self.current_task_description = task_description
                    self.get_logger().info(f"任务描述已保存: {task_description}")

            elif keyword_type == "stop":
                # 停止导航
                self._cancel_navigation("收到停止命令")

        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON解析错误: {e}")
        except Exception as e:
            self.get_logger().error(f"处理语音命令错误: {e}")

    def send_goal(self, x: float, y: float, theta: float = 0.0, keyword: str = "") -> None:
        """发送导航目标到 Nav2

        Args:
            x: 目标x坐标 (米)
            y: 目标y坐标 (米)
            theta: 目标朝向 (弧度)
            keyword: 匹配的关键词 (用于日志)
        """
        if self.is_navigating:
            self.get_logger().warn("已有导航任务正在进行中，请先取消")
            return

        self.get_logger().info(f"发送导航目标: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
        if keyword:
            self.get_logger().info(f"触发关键词: '{keyword}'")

        # 创建目标位姿
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.global_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        # 发送目标
        self.is_navigating = True
        self._nav_start_time = time.monotonic()
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _check_timeout(self):
        """ROS timer 回调：在 executor 线程中检查导航超时，避免跨线程操作 goal_handle"""
        if not self.is_navigating or self._nav_start_time is None or self.timeout_sec <= 0:
            return
        elapsed = time.monotonic() - self._nav_start_time
        if elapsed > self.timeout_sec:
            self.get_logger().error(f"导航超时 ({self.timeout_sec}秒)")
            self._cancel_navigation("超时")
            self._nav_start_time = None

    def _goal_response_callback(self, future):
        """目标响应回调"""
        try:
            goal_response = future.result()
            if not goal_response.accepted:
                self.get_logger().error("❌ 导航目标被拒绝")
                self.is_navigating = False
                return

            self.get_logger().info("✓ 导航目标已接受")

            # 保存 goal handle 并请求结果
            self.goal_handle = goal_response
            get_result_future = self.goal_handle.get_result_async()
            get_result_future.add_done_callback(self._get_result_callback)

        except Exception as e:
            self.get_logger().error(f"目标响应错误: {e}")
            self.is_navigating = False

    def _feedback_callback(self, feedback_msg):
        """反馈回调"""
        if not self.enable_feedback:
            return

        feedback = feedback_msg.feedback
        self.get_logger().debug(f"导航反馈: {feedback}")

    def _get_result_callback(self, future):
        """获取结果回调"""
        try:
            status = future.result().status

            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("✅ 导航成功！已到达目标点")
                self.is_navigating = False
                self._on_navigation_succeeded()

            elif status == GoalStatus.STATUS_ABORTED:
                self.get_logger().error("❌ 导航中止 (Aborted)")
                self.is_navigating = False

            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().warn("⚠ 导航被取消 (Canceled)")
                self.is_navigating = False

            else:
                self.get_logger().error(f"❌ 导航失败, 状态码: {status}")
                self.is_navigating = False

        except Exception as e:
            self.get_logger().error(f"获取结果错误: {e}")
            self.is_navigating = False

    def _on_navigation_succeeded(self):
        """导航成功后的回调"""
        self.get_logger().info("-" * 50)
        self.get_logger().info("导航成功，检查是否有待执行的任务...")

        if self.current_task_description:
            self.get_logger().info(f"发现任务描述: {self.current_task_description}")
            self._trigger_evaluation()
            # 清空任务描述
            self.current_task_description = ""
        else:
            self.get_logger().info("没有待执行的任务")

    def _trigger_evaluation(self):
        """触发 action_dispatcher 评估服务"""
        self.get_logger().info("")
        self.get_logger().info("-" * 50)
        self.get_logger().info("正在触发 Robot 评估服务...")

        if not self.evaluation_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/action_dispatcher/start_evaluate 服务不可用！")
            self.navigation_failed = True
            return

        request = Trigger.Request()
        future = self.evaluation_client.call_async(request)

        def evaluation_callback(future):
            try:
                response = future.result()
                if response.success:
                    self.get_logger().info("=" * 50)
                    self.get_logger().info("Robot 评估服务已成功触发！")
                    self.get_logger().info(f"响应: {response.message}")
                    self.get_logger().info("=" * 50)
                    self.navigation_succeeded = True
                    self.current_task_description = ""
                else:
                    self.get_logger().error(f"评估服务返回失败: {response.message}")
                    self.navigation_failed = True
            except Exception as e:
                self.get_logger().error(f"调用评估服务错误: {e}")
                self.navigation_failed = True

        future.add_done_callback(evaluation_callback)

    def _cancel_navigation(self, reason: str = ""):
        """取消当前导航"""
        if not self.is_navigating:
            return

        if self.goal_handle is None:
            self.get_logger().warn("没有有效的导航句柄")
            self.is_navigating = False
            return

        log_msg = f"正在取消导航: {reason}" if reason else "正在取消导航..."
        self.get_logger().info(log_msg)
        cancel_future = self.goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self._cancel_callback)

    def stop_callback(self, msg: String):
        """停止导航回调"""
        if not self.is_navigating:
            self.get_logger().info("当前没有正在进行的导航")
            return

        self._cancel_navigation("收到停止命令")

    def _cancel_callback(self, future):
        """取消导航回调"""
        try:
            cancel_response = future.result()
            if cancel_response.return_code == 0:
                self.get_logger().info("导航已成功取消")
            else:
                self.get_logger().warn(f"取消导航返回码: {cancel_response.return_code}")
        except Exception as e:
            self.get_logger().error(f"取消导航错误: {e}")


def main():
    rclpy.init()

    node = Nav2GoalClient()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

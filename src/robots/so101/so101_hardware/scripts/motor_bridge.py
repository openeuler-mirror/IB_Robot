#!/usr/bin/env python3
"""ROS 2 node that bridges Feetech servos <-> topic_based_ros2_control.

它现在包含一个完整的交互式校准程序。

它:
* 订阅 `/so101_follower/joint_commands` (sensor_msgs/JointState)
  并写入 Goal_Position 到舵机。
* 发布 `/so101_follower/joint_states` (使用校准过的零点)。
"""

from __future__ import annotations

import math
import pathlib
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ibrobot_msgs.msg import JointCurrent

try:
    import deepdiff  # type: ignore  # noqa: F401  # probe: presence decides the stub below
except ImportError:  # create stub
    import types

    deepdiff_stub = types.ModuleType("deepdiff")

    class _DD(dict):
        pass

    deepdiff_stub.DeepDiff = _DD  # type: ignore
    sys.modules["deepdiff"] = deepdiff_stub

try:
    import tqdm  # type: ignore  # noqa: F401  # probe: presence decides the stub below
except ImportError:  # create stub
    import types

    tqdm_stub = types.ModuleType("tqdm")

    def _tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

    tqdm_stub.tqdm = _tqdm  # type: ignore
    sys.modules["tqdm"] = tqdm_stub
# -----------------------------------------------------------------------------

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus, OperatingMode

from so101_hardware.calibration.interactive import load_calibration
from so101_hardware.motor_current import read_motor_currents

PORT_DEFAULT = "/dev/ttyACM0"

JOINTS = {
    "1": {"id": 1, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "2": {"id": 2, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "3": {"id": 3, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "4": {"id": 4, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "5": {"id": 5, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "6": {"id": 6, "model": "sts3215", "mode": MotorNormMode.RANGE_0_100},
}

# INITIAL_RAW_POSITIONS = { ... }

CALIB_PATH = pathlib.Path.home() / ".calibrate" / "so101_follower_calibrate.json"


class MotorBridge(Node):
    def __init__(self):
        super().__init__("so101_motor_bridge")
        # Declare parameters so they can be overridden from launch/CLI
        self.declare_parameter("port", PORT_DEFAULT)
        self.declare_parameter("calib_file", str(CALIB_PATH))

        port = self.get_parameter("port").get_parameter_value().string_value
        if not port:
            port = PORT_DEFAULT
        # Kept on the instance: the calibration error paths below quote the port
        # back to the operator in the command they need to run.
        self.port = port

        # Build motor objects
        motors = {name: Motor(cfg["id"], cfg["model"], cfg["mode"]) for name, cfg in JOINTS.items()}
        self.bus = FeetechMotorsBus(port, motors)
        self.joint_names = list(JOINTS.keys())

        self.get_logger().info(f"Connecting to Feetech bus on {port} …")
        try:
            self.bus.connect()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Could not open motor bus: {exc}")
            raise

        # Publishers / Subscribers
        self.joint_states_pub = self.create_publisher(JointState, "so101_follower/joint_states", 10)
        self.joint_currents_pub = self.create_publisher(JointCurrent, "/so101_follower/joint_currents", 10)
        self.joint_commands_sub = self.create_subscription(
            JointState,
            "so101_follower/joint_commands",
            self._command_cb,
            10,
        )

        # State variables
        self.current_commands: dict[str, float] = {}
        self._home_offsets: dict[str, int] | None = None
        self._limits: dict[str, tuple[int, int]] | None = None
        self._is_read_turn = True  # Flag to alternate between read and write
        self._steps_per_rad = 4096.0 / (2 * math.pi)

        self._reset_rad_offsets: dict[str, float] = {}

        # --- Type hint 现在是 MotorCalibration ---
        self.calibration_data: dict[str, MotorCalibration] | None = None

        # --- 加载或运行校准 ---
        if not self._load_or_run_calibration():
            self.get_logger().error("校准/加载失败。正在关闭...")
            if self.bus:
                self.bus.disconnect()
            rclpy.shutdown()
            return

        # --- 从校准数据设置 home offsets 和 limits ---
        self._home_offsets = {name: cal.homing_offset for name, cal in self.calibration_data.items()}
        self._limits = {name: (cal.range_min, cal.range_max) for name, cal in self.calibration_data.items()}
        self.get_logger().info(f"Follower 臂已加载 home offsets: {self._home_offsets}")
        self.get_logger().info(f"Follower 臂已加载 limits: {self._limits}")

        # --- 现在运行 configure_motors 和 enable_torque ---
        self.get_logger().info("正在配置电机 (P/I/D, Torque)...")
        with self.bus.torque_disabled():
            self.bus.configure_motors()  # 设置 Acceleration 等
            for motor in self.joint_names:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "6":  # 假设 6 号是夹爪
                    self.get_logger().info("为夹爪 '6' 设置特殊扭矩限制。")
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

        self.bus.enable_torque()
        self.get_logger().info("电机总线已连接并配置完毕。")

        self.get_logger().info("正在设置初始复位位置...")
        reset_normalized_goals = {
            "1": -1.172161172161168,
            "2": -98.91304347826087,
            "3": 98.82724402345514,
            "4": -81.3107822410148,
            "5": -47.3015873015873,
            "6": 2.922077922077922,
        }

        try:
            self.current_commands = reset_normalized_goals

            # 写入初始位置
            self.bus.sync_write("Goal_Position", self.current_commands, normalize=True)

            self.get_logger().info("初始复位指令 (归一化值) 已设置。")
            self.get_logger().info(f"（保留两位小数）: { {k: round(v, 2) for k, v in self.current_commands.items()} }")

            # 2. (--- 新增 ---)
            # 现在, 计算这些 归一化值 对应的 真实弧度值, 作为偏移量
            self.get_logger().info("正在计算复位位置的弧度偏移量...")
            self._reset_rad_offsets = {}
            for name, norm_val in reset_normalized_goals.items():
                motor = self.bus.motors[name]
                cal = self.calibration_data[name]
                min_ = cal.range_min
                max_ = cal.range_max
                home = cal.homing_offset

                # 2a. 归一化值 -> 原始步数(raw)
                raw = 0
                if (max_ - min_) == 0:
                    self.get_logger().error(f"电机 {name} 的校准范围无效 (min=max)。")
                    continue

                if motor.norm_mode == MotorNormMode.RANGE_M100_100:
                    raw = int(((norm_val + 100) / 200) * (max_ - min_) + min_)
                elif motor.norm_mode == MotorNormMode.RANGE_0_100:
                    raw = int((norm_val / 100) * (max_ - min_) + min_)

                # 2b. 原始步数(raw) -> 真实弧度(rad)
                rad = (raw - home) * (2 * math.pi) / 4096.0
                self._reset_rad_offsets[name] = rad

            self.get_logger().info(
                f"计算出的弧度偏移量 (保留两位): { {k: round(v, 2) for k, v in self._reset_rad_offsets.items()} }"
            )

        except Exception as e:
            self.get_logger().error(f"设置初始复位位置失败: {e}")
            self.get_logger().warn("将使用 0.0 (归一化值) 作为默认初始位置。")
            self.current_commands = {n: 0.0 for n in self.joint_names}

        # Timer for periodic read/write (50 Hz)
        self.timer = self.create_timer(0.02, self._timer_cb)
        self.get_logger().info("Motor bridge 节点启动成功。")

    # ---------------------------------------------------------------------
    # Calibration helper functions
    # ---------------------------------------------------------------------

    def _load_or_run_calibration(self) -> bool:
        """Load calibration from file."""
        if CALIB_PATH.is_file():
            self.get_logger().info(f"Found calibration file: {CALIB_PATH}. Auto-loading.")
            try:
                self.calibration_data = load_calibration(CALIB_PATH, self.joint_names, self.get_logger())
                self.get_logger().info("Writing loaded calibration to motor firmware...")
                self.bus.write_calibration(self.calibration_data)
                self.get_logger().info("Calibration written to motors.")
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to load/write calibration ({e}).")
                self.get_logger().error("Run calibration tool:")
                self.get_logger().error(f"  ros2 run so101_hardware calibrate_arm --arm follower --port {self.port}")
                return False
        else:
            self.get_logger().error(f"Calibration file not found: {CALIB_PATH}")
            self.get_logger().error("Run calibration tool first:")
            self.get_logger().error(f"  ros2 run so101_hardware calibrate_arm --arm follower --port {self.port}")
            return False

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------
    def _command_cb(self, msg: JointState):
        """Store desired joint positions (converting rad -> normalized)."""
        try:
            # strict=False: a malformed JointState may carry fewer positions
            # than names; dropping the extras is preferable to raising here.
            for name, pos_rad in zip(msg.name, msg.position, strict=False):
                if name in JOINTS:
                    # 修复: 将 ros2_control 的相对指令 (pos_rad) 转换为 绝对弧度目标
                    # 0.0 (来自 controller) + 偏移量 = 我们的复位位置
                    true_goal_rad = pos_rad + self._reset_rad_offsets.get(name, 0.0)

                    # --- 剩下的逻辑使用 true_goal_rad ---
                    motor = self.bus.motors[name]
                    cal = self.calibration_data[name]
                    home = cal.homing_offset
                    min_ = cal.range_min
                    max_ = cal.range_max

                    # 1. 将 绝对弧度(rad) 转换为 目标原始步数(raw)
                    target_raw = int((true_goal_rad * self._steps_per_rad) + home)

                    # 2. 将 原始步数(raw) 限制在电机的物理范围内
                    target_raw = min(max(target_raw, min_), max_)

                    # 3. 将 目标原始步数(raw) 转换为 归一化值(normalized)
                    norm_val = 0.0
                    # 避免除以零
                    if (max_ - min_) == 0:
                        self.get_logger().warn(f"电机 {name} 的校准范围无效 (min=max)。", throttle_duration_sec=5.0)
                        continue

                    if motor.norm_mode == MotorNormMode.RANGE_M100_100:
                        norm_val = ((target_raw - min_) / (max_ - min_)) * 200.0 - 100.0
                    elif motor.norm_mode == MotorNormMode.RANGE_0_100:
                        norm_val = ((target_raw - min_) / (max_ - min_)) * 100.0
                    else:
                        self.get_logger().warn(
                            f"不支持的 norm_mode {motor.norm_mode} (在 _command_cb 中)", throttle_duration_sec=5.0
                        )
                        continue

                    # 4. 存储归一化值
                    self.current_commands[name] = norm_val

        except Exception as e:
            self.get_logger().error(f"转换 (弧度->归一化) 指令失败: {e}", throttle_duration_sec=5.0)

    def _timer_cb(self):
        # On each timer tick, we either do a read or a write, but never both.
        if self._is_read_turn:
            self._do_read()
        else:
            self._do_write()

        # Flip the flag for the next turn
        self._is_read_turn = not self._is_read_turn

    def _do_read(self):
        try:
            raw_positions = self.bus.sync_read("Present_Position", normalize=False)
        except Exception as exc:
            self.get_logger().warn(f"sync_read failed: {exc}")
            return
        if not raw_positions or len(raw_positions) != len(self.joint_names):
            self.get_logger().warn(
                f"sync_read incomplete: {len(raw_positions) if raw_positions else 0}/{len(self.joint_names)}",
                throttle_duration_sec=5.0,
            )
            return
        currents = read_motor_currents(
            self.bus,
            self.joint_names,
            self.get_logger(),
            "current sync_read failed",
        )

        # Publish current joint states
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(JOINTS.keys())

        # 修复: 报告 (真实弧度 - 复位弧度偏移量)
        # 这样当机械臂在复位位置时, ros2_control 会看到 0.0
        js.position = [
            (raw - self._home_offsets.get(n, 0)) * (2 * math.pi) / 4096.0 - self._reset_rad_offsets.get(n, 0.0)
            for n, raw in raw_positions.items()
        ]
        self.joint_states_pub.publish(js)

        current_msg = JointCurrent()
        current_msg.header.stamp = js.header.stamp
        current_msg.name = list(js.name)
        current_msg.current = [currents.get(n, 0.0) for n in current_msg.name]
        self.joint_currents_pub.publish(current_msg)

    def _do_write(self):
        if self._home_offsets is None:
            self.get_logger().warn("Skipping write: home offsets not yet captured (校准未完成?)")
            return

        # On subsequent write cycles, send the latest commands from the topic.
        if not self.current_commands:
            # 第一次写入时, 读取当前位置, 并将其设为初始目标, 以防止突然移动。
            self.get_logger().info(
                "第一次写入: 尚未收到外部指令。读取当前位置作为初始目标。", throttle_duration_sec=5.0
            )
            try:
                # 1. 新增一次读取, 获取当前电机位置 (相对步数)
                raw_positions = self.bus.sync_read("Present_Position")

                self.current_commands = raw_positions
                self.get_logger().info(f"当前位置: {raw_positions}")

            except Exception as exc:
                self.get_logger().warn(f"在 _do_write 中读取初始位置失败: {exc}")
                return

        try:
            raw_goals = self.current_commands
            # self.get_logger().info(f"Trying to write: {raw_goals}")
            self.bus.sync_write("Goal_Position", raw_goals, normalize=True)
        except Exception as exc:
            self.get_logger().warn(f"sync_write failed: {exc}")


def main():
    """Entry-point."""
    rclpy.init()
    node = None
    try:
        node = MotorBridge()
        # 只有在 __init__ 成功 (即校准成功) 且 bus 有效时才 spin
        if node.bus is not None and rclpy.ok() and node._home_offsets is not None:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node and node.bus is not None:
            node.get_logger().info("Disconnecting from motor bus...")
            try:
                node.bus.disable_torque()
                node.get_logger().info("电机扭矩已禁用。")
            except Exception as e:
                node.get_logger().warn(f"禁用扭矩失败: {e}")
            node.bus.disconnect()
        if node and rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

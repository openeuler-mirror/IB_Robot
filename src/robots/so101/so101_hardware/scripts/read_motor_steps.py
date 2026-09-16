#!/usr/bin/env python3
"""
ROS 2 命令行工具, 用于读取 SO-101 机械臂 (Leader 或 Follower) 的舵机位置。

该脚本可以读取原始的编码器步数 (0-4095) 或加载校准文件以读取
归一化的值 (例如 -100 到 100)。

用法
-----
# 1. 读取 "原始" (0-4095) 步数值 (连续)
#    (此时 --arm 参数被忽略)
ros2 run so101_hardware read_motor_steps --raw --rate 1

# 2. 读取 "原始" (0-4095) 步数值 (一次)
ros2 run so101_hardware read_motor_steps --raw --once

# 3. 读取 Follower 臂的 "校准后" (归一化) 值 (连续, 默认)
ros2 run so101_hardware read_motor_steps --arm follower --rate 1

# 4. 读取 Leader 臂的 "校准后" (归一化) 值 (一次)
ros2 run so101_hardware read_motor_steps --arm leader --once
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import yaml
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus

# 定义所有关节的共享配置
JOINTS = {
    "1": {"id": 1, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "2": {"id": 2, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "3": {"id": 3, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "4": {"id": 4, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "5": {"id": 5, "model": "sts3215", "mode": MotorNormMode.RANGE_M100_100},
    "6": {"id": 6, "model": "sts3215", "mode": MotorNormMode.RANGE_0_100},
}

# 定义两个校准文件路径
CALIB_PATH_LEADER = pathlib.Path.home() / ".calibrate" / "so101_leader_calibrate.json"
CALIB_PATH_FOLLOWER = pathlib.Path.home() / ".calibrate" / "so101_follower_calibrate.json"


def main() -> None:
    """so101_read_steps 的主入口点"""
    parser = argparse.ArgumentParser(
        prog="so101_read_steps",
        description="Read Present_Position of all SO-101 servos (Leader or Follower).",
        formatter_class=argparse.RawTextHelpFormatter,  # 保持 usage 格式
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="连接 Feetech 总线的串口 (默认: /dev/ttyACM0)",
    )

    parser.add_argument(
        "--arm",
        choices=["leader", "follower"],
        default="follower",
        help="指定要读取的机械臂 (leader 或 follower)。这将决定加载哪个校准文件。\n(如果指定了 --raw, 此参数将被忽略。)",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="打印原始编码器值 (0-4095) 并跳过加载校准文件。",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="读取一次然后退出。 (默认: 连续读取)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="连续读取时的轮询频率 (Hz)。(默认: 1.0)",
    )
    args = parser.parse_args()

    # 仅在 --raw 未指定时才设置 CALIB_PATH
    CALIB_PATH = None
    if not args.raw:
        if args.arm == "leader":
            CALIB_PATH = CALIB_PATH_LEADER
        else:
            CALIB_PATH = CALIB_PATH_FOLLOWER
        print(f"INFO: 目标机械臂: '{args.arm}' (将加载校准文件)", file=sys.stderr)
    else:
        print("INFO: 已指定 --raw。将读取原始值 (忽略 --arm 参数)。", file=sys.stderr)

    motors = {name: Motor(cfg["id"], cfg["model"], cfg["mode"]) for name, cfg in JOINTS.items()}

    try:
        bus = FeetechMotorsBus(args.port, motors)
        bus.connect()
    except Exception as exc:
        print(f"错误: 无法连接到总线 {args.port}: {exc}", file=sys.stderr)
        sys.exit(1)

    # 仅在 --raw 未指定时才尝试加载校准
    calibration_data: dict[str, MotorCalibration] | None = None
    if not args.raw:
        if CALIB_PATH and CALIB_PATH.is_file():
            print(f"INFO: 正在加载校准文件: {CALIB_PATH}", file=sys.stderr)
            try:
                with open(CALIB_PATH) as f:
                    loaded_data = json.load(f)

                # 将 dict 转换回 MotorCalibration 对象
                calibration_data = {}
                for name, data_dict in loaded_data.items():
                    if name in JOINTS:
                        calibration_data[name] = MotorCalibration(**data_dict)

                # 检查是否所有 joint 都有数据
                if all(j in calibration_data for j in JOINTS):
                    print("INFO: 成功加载校准文件。正在写入电机...", file=sys.stderr)
                    # 这会将校准数据(homing offsets, ranges)注册到总线实例中
                    # 供 sync_read(normalize=True) 使用
                    bus.write_calibration(calibration_data)
                    print("INFO: 校准数据已写入电机。", file=sys.stderr)
                else:
                    print(f"警告: 校准文件 {CALIB_PATH} 不完整或无效。将忽略。", file=sys.stderr)
                    calibration_data = None  # 强制设为 None, 导致回退到 raw

            except Exception as e:
                print(f"警告: 加载/写入校准文件失败 ({e})。将忽略。", file=sys.stderr)
                calibration_data = None  # 强制设为 None, 导致回退到 raw
        else:
            print(f"INFO: 未找到校准文件: {CALIB_PATH}。将读取原始值。", file=sys.stderr)
            # calibration_data 保持为 None

    # (如果 args.raw 为 True, calibration_data 自动为 None)

    try:

        def _read_and_print() -> None:
            # 决定是读取原始值还是归一化值
            if calibration_data is not None:
                # 2. 用户未要求 --raw, 且校准文件加载成功
                positions = bus.sync_read("Present_Position", normalize=True)
                print_title = f"校准后的归一化值 ({args.arm})"
            else:
                # 1. 用户明确要求 --raw, 或校准加载失败/未找到
                positions = bus.sync_read("Present_Position", normalize=False)
                if args.raw:
                    print_title = "原始值 (0-4095)"
                else:
                    print_title = "原始值 (校准文件丢失或无效)"

            # 打印 YAML 块
            if not args.once and args.rate > 0:
                print(f"--- {print_title} (@ {time.strftime('%H:%M:%S')}) ---")
            else:
                print(f"--- {print_title} ---")

            print(yaml.safe_dump(positions, sort_keys=False))

        # 运行循环
        if args.once or args.rate == 0:
            _read_and_print()
        else:
            period = 1.0 / max(args.rate, 0.01)
            while True:
                start = time.time()
                _read_and_print()
                elapsed = time.time() - start
                # 确保 sleep 时间非负
                sleep_duration = max(0.0, period - elapsed)
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\n正在退出...", file=sys.stderr)
    finally:
        bus.disconnect()
        print("INFO: 总线连接已断开。", file=sys.stderr)


if __name__ == "__main__":
    main()

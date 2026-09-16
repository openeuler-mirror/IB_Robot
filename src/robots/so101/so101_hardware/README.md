# SO-101 机械臂硬件驱动包

SO-101 机械臂的硬件驱动包，提供高性能 C++ ros2_control 接口和 Python 工具集。

## 概述

本包为 SO-101 机械臂提供了完整的硬件驱动解决方案，支持两种模式：
- **C++ ros2_control 插件**：直接调用 FTServo SDK，低延迟、高性能，适用于生产和控制环境。
- **Python 工具**：包含校准 (Calibration)、数据采集 (Leader Arm Publisher) 和诊断工具。

## 核心特性

- **直接通信**：通过 FTServo SDK 与 Feetech 舵机直接通信。
- **混合构建**：采用 ament_cmake_python 支持 C++ 插件和 Python 脚本的混合安装。
- **启动位置保护**：支持配置 `reset_positions`，防止机械臂在启动时因回零产生剧烈跳动（对机器狗背负式机械臂尤为重要）。
- **生命周期管理**：支持标准的 `on_configure`, `on_activate`, `on_deactivate` 生命周期。
- **电流反馈**：C++ 插件和 Python 桥接脚本会按 STS3215 `1 LSB = 6.5mA` 把 Feetech `Present_Current` 转为安培，并通过 `/so101_follower/joint_currents` 或 `/so101_leader/joint_currents` 发布 `ibrobot_msgs/msg/JointCurrent`，供数据集转换生成 `observation.current`。
- **安全保障**：节点关闭时自动卸载舵机力矩（Torque Off）；`on_activate` 失败回滚时先对所有舵机
  fail-closed 卸力矩，再尝试 relock 任何处于解锁状态的 EPROM，最后才关闭串口，并对两类失败给出独立诊断。
  该回滚保护覆盖整个激活流程，包括最后的初始同步读（initial sync read）：只要 sync-read 发送或任意一个
  舵机返回包失败，激活立即中止并走同一条回滚路径。

## 架构

```
ros2_control (Controller Manager)
      ↓
SO101SystemHardware (C++ Plugin)  ←──┐
      ↓                              │
FTServo SDK (C++)                    │
      ↓                              │
Feetech Servos (Hardware)            │
      ↑                              │
Python Utilities (Scripts) ──────────┘
```

## 依赖

### Git Submodule
FTServo_Linux SDK 作为子模块引入：
```bash
git submodule update --init --recursive
```

### 系统依赖
- ROS 2 Humble
- nlohmann_json 库
- hardware_interface, pluginlib, rclcpp_lifecycle
- pyserial (Python 驱动)

## 编译

```bash
cd ~/Research/lerobot_ros2/src/ros2/ros2_ws
source /opt/ros/humble/setup.sh
# 建议指定 PYTHONPATH 以确保混合编译成功
colcon build --packages-select so101_hardware
source install/setup.sh
```

## 工具文档

- [arm_calibration_transfer](docs/tools/arm_calibration_transfer.md)：旧版标定数据迁移与生成新 follower 标定文件
- [arm_calibration_checker](docs/tools/arm_calibration_checker.md)：机械臂标定结果的真机检查流程

## 使用方法

### 1. 校准机械臂 (Python)
首次使用前必须校准，生成 `~/.calibrate/` 目录下的 JSON 文件。
```bash
# 校准 Follower 臂
ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

### 2. C++ ros2_control 插件配置
在 URDF 中指定硬件插件：
```xml
<hardware>
  <plugin>so101_hardware/SO101SystemHardware</plugin>
  <param name="port">/dev/ttyACM0</param>
  <param name="calib_file">$(env HOME)/.calibrate/so101_follower_calibrate.json</param>
  <!-- 可选：启动时的安全姿态 (JSON 格式, 单位为弧度) -->
  <param name="reset_positions">{"1": 0.0, "2": 0.0}</param>
</hardware>
```

### 3. Leader 臂数据采集
用于手撸数据或示教：
```bash
ros2 run so101_hardware leader_arm_pub --port /dev/ttyACM0 --publish_rate 50.0
```

## 实现细节

### 同步读取超时

同步读取采用 LeRobot Feetech 的延迟余量：报文传输时间 + 3 个字节时间 + 50 ms，
由 `feetech::Bus::sync_read_timeout` 统一计算。当前 6 电机、1 Mbps、每个电机
15 字节反馈对应 51.29 ms，向上取整为 **52 ms**。硬件读取使用同一轮次预算，
不再按 `0.4 × 控制周期` 缩短为 4 ms。每个控制周期只尝试一次同步读，符合 LeRobot
默认的 `num_retry=0`；收齐回包立即返回，失败在后续控制周期再次读取。

100 Hz 仍是目标控制频率；偶发的慢响应可以使单轮超过 10 ms，而不是被提前判为读取失败。
短暂失败保持上一帧，连续失败达到 200 ms 仍上报 ERROR。写入保留独立的周期预算与重试退避。

### 启动位置 (Reset Positions)
`reset_positions` 参数允许指定初始位置。
- **有配置**：启动时机械臂会先平滑移动到指定姿态。
- **无配置 (默认)**：机械臂会保持当前舵机位置，不发生任何动作。

### 坐标转换公式
插件内部自动处理步数 (Steps) 与弧度 (Radians) 的转换：
- **读取**：`radians = ((steps - range_min) / range - 0.5) * 2.0 * PI`
- **写入**：`steps = (radians / (2.0 * PI) + 0.5) * range + range_min`

### 激活回滚 (Activation Rollback)
`on_activate` 在配置舵机过程中失败时执行 fail-closed 回滚，由 `detail::rollback_activation` 完成：

1. **先卸力矩**：对所有 `motor_ids_` 执行 `EnableTorque(id, 0)`（带重试），这是安全默认动作，独立于
   EPROM 状态。卸力矩优先于 relock，因为部分 Feetech 舵机仅在力矩关闭时才接受 EPROM 锁定指令。
2. **再 relock EPROM**：仅对回滚开始时仍处于解锁状态（`unlocked_motors` 集合）的舵机尝试
   `LockEprom(id)`（带重试），在**关闭串口之前**完成。正常流程中每完成一个舵机的配置就会把它从
   解锁集合移除，因此只有中途失败时仍解锁的舵机才会进入 relock。
3. **最后关闭串口**：`sms_sts_.end()`。

两类结果独立汇报，互不掩盖：
- 力矩卸载失败 → `Failed to disable torque for one or more motors during activation abort`；
- EPROM relock 失败 → `Failed to relock EPROM for N motor(s) during activation abort; persistent parameters may be unprotected`，
  并在 `relock_failures` 中列出具体舵机 ID。

即使力矩卸载失败，relock 仍会被尝试（均为 best-effort、fail-closed 语义），调用方可据此判断是否需要
人工复位持久参数。

### 初始同步读 (Initial Sync Read)

`on_activate` 在使能力矩之后，会做一次 sync-read 把真实反馈写入
`hw_commands_/hw_positions_/hw_velocities_/hw_currents_`。该步骤**同样受上面的激活回滚保护**
（fail-closed），由 `detail::perform_initial_sync_feedback` 实现：

1. **发送/总线应答（syncReadPacketTx）**：返回收到 SDK 缓冲区的字节数；`<= 0` 表示发送失败或无应答
   （超时）。`syncReadBegin` 仅返回 `void`（分配 SDK 接收缓冲区、记录超时），不是 fail-closed 门禁，
   真正的门禁是这里的 Tx 返回值。
2. **每个舵机返回包（syncReadPacketRx）**：返回内存字节数表示成功、`0` 表示失败。**必须全部舵机**
   都返回完整且 CRC 校验通过的包，才会初始化状态并 dismiss 回滚守卫返回 SUCCESS。

任何一个门禁失败，`on_activate` 立即调用 `abort_activation()`（力矩 off / EPROM relock / 关闭串口），
绝不会在 `hw_commands_/positions/velocities/currents` 未完整初始化的情况下解除回滚守卫。失败日志：

- 发送/总线失败 → `Initial sync read transmit failed; aborting activation`；
- 某舵机 Rx 失败 → `Initial sync read for motor ID <id> failed; aborting activation`，`<id>` 为
  第一个返回包失败的舵机 ID（与 EPROM relock 的 `relock_failures` 相互独立）。

该 helper 以回调形式注入 sync-read 操作（Tx/Rx），因此可在不接真机的情况下用 gtest 覆盖 Tx 失败、
某舵机 Rx 失败以及全部成功的路径。

## 对比：C++ 插件 vs Python 工具

| 特性 | C++ 插件 (Production) | Python 工具 (Dev/Calib) |
|------|----------------------|------------------------|
| 延迟 | 极低 (直连) | 较高 (Python 开销) |
| 性能 | 高 (实时性好) | 中等 |
| 模式 | ros2_control 硬件接口 | 话题桥接/脚本 |
| 用途 | 强化学习/轨迹执行 | 标定/示教记录/诊断 |

## 故障排除

- **串口权限**：`sudo chmod 666 /dev/ttyACM0` 或 `sudo usermod -a -G dialout $USER`。
- **标定文件缺失**：若报错 `Calibration file not found`，请先运行 `calibrate_arm`。
- **子模块为空**：确保运行了 `git submodule update`。

## 许可证
TODO: License declaration

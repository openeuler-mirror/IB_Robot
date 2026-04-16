# lekiwi_hardware (硬件接口包)

`lekiwi_hardware` 是 LeKiwi 移动操纵机器人的 ros2_control 硬件接口插件。通过 Feetech SMS_STS 串口通信协议驱动 6 个手臂位置舵机 + 3 个底盘全向轮速度舵机，实现低延迟的同步多电机控制。

## 功能特性

- ros2_control `SystemInterface` 插件，即插即用
- 6 个手臂关节（ID 1-6）：位置控制模式
- 3 个底盘全向轮（ID 7-9）：速度（轮式）控制模式
- 同步读写（SyncRead / SyncWrite）保证多电机时序一致
- JSON 标定文件支持：homing offset、关节限位
- 内置 `scan_motors` 诊断工具
- 控制频率 100 Hz

## 架构

```text
+-------------------------------------------+
| 上层节点 (ACT 策略 / 遥操作 / MoveIt)       |
+-------------------------------------------+
        | 话题 (position/velocity commands)
        v
+---------------------------+
| ros2_control Controllers  |  arm_position_controller
|  (JointGroupPosition/     |  gripper_position_controller
|   ForwardCommand/         |  base_velocity_controller
|   JointGroupVelocity)     |  joint_state_broadcaster
+---------------------------+
        | CommandInterface / StateInterface
        v
+-------------------------------+
| LeKiwiSystemHardware          |  ros2_control 插件
|  (此包)                       |  read() / write() @ 100Hz
+-------------------------------+
        | Feetech SMS_STS 协议 (串口 1Mbps)
        v
+-------------------------------+
| Feetech 舵机 (ID 1-9)         |  6×STS位置舵机 + 3×轮式舵机
+-------------------------------+
```

## 目录结构

```text
lekiwi_hardware/
├── CMakeLists.txt                       # 构建配置，自动拉取 Feetech SDK
├── package.xml                          # ROS2 包清单
├── lekiwi_hardware_plugin.xml           # pluginlib 插件注册
├── config/
│   └── lekiwi_controllers.yaml          # 控制器配置
├── include/lekiwi_hardware/
│   └── lekiwi_system_hardware.hpp       # 硬件接口头文件
├── src/
│   └── lekiwi_system_hardware.cpp       # 硬件接口实现
└── tools/
    └── scan_motors.cpp                  # 电机扫描诊断工具
```

## 快速开始

### 1. 构建

```bash
cd ~/workspace/IB_Robot
colcon build --packages-select lekiwi_hardware
source install/setup.bash
```

> 构建时会自动通过 CMake FetchContent 从 GitHub 拉取 [Feetech SDK](https://github.com/ftservo/FTServo_Linux.git)。

### 2. 串口权限

```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

### 3. 电机诊断（可选）

```bash
# 默认扫描 /dev/ttyACM0
ros2 run lekiwi_hardware scan_motors

# 指定串口设备
ros2 run lekiwi_hardware scan_motors /dev/ttyUSB0
```

该工具会逐个 Ping ID 1-9 的电机并报告响应状态，用于排查连接问题。

### 4. 启动硬件

通常通过 `robot_config` 包的 launch 文件统一启动：

```bash
# 推理
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=model_inference
# 遥操
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=teleop
```

也可通过 `ros2 control` 命令行单独加载控制器管理器（需配合 URDF 中的 ros2_control 配置）。

## 硬件参数（URDF ros2_control 标签）

| 参数 | 说明 | 示例 |
|------|------|------|
| `port` | 串口设备路径 | `/dev/ttyACM0` |
| `calib_file` | 标定 JSON 文件路径 | `/path/to/lekiwi_calib.json` |

每个 joint 需指定 `id` 参数映射到物理电机 ID：

```xml
<joint name="1">
  <param name="id">1</param>
  <command_interface name="position"/>
  <state_interface name="position"/>
  <state_interface name="velocity"/>
</joint>
```

### 标定 JSON 格式

```json
{
  "shoulder_pan": {
    "id": 1,
    "homing_offset": 0,
    "range_min": 0,
    "range_max": 4095
  },
  "shoulder_lift": {
    "id": 2,
    "homing_offset": 50,
    "range_min": 100,
    "range_max": 4000
  }
}
```

- JSON 的 key 可以是电机名或 ID 字符串，内部按 `id` 字段匹配
- 底盘电机（ID 7-9）无需标定条目，自动使用默认值

## 控制器配置

`config/lekiwi_controllers.yaml` 定义了 4 个控制器：

| 控制器 | 类型 | 控制关节 | 模式 |
|--------|------|----------|------|
| `arm_position_controller` | `JointGroupPositionController` | 1-5 | 位置 |
| `gripper_position_controller` | `ForwardCommandController` | 6 | 位置 |
| `base_velocity_controller` | `JointGroupVelocityController` | 7-9 | 速度 |
| `joint_state_broadcaster` | `JointStateBroadcaster` | 1-9 | 状态发布 |

控制器管理器更新频率：**100 Hz**，状态发布频率：**100 Hz**。

## 关节映射

| 关节 ID | 名称约定 | 控制模式 | 说明 |
|---------|----------|----------|------|
| 1 | arm_shoulder_pan | 位置 | 肩部偏航 |
| 2 | arm_shoulder_lift | 位置 | 肩部俯仰 |
| 3 | arm_elbow_flex | 位置 | 肘部弯曲 |
| 4 | arm_wrist_flex | 位置 | 腕部弯曲 |
| 5 | arm_wrist_roll | 位置 | 腕部旋转 |
| 6 | arm_gripper | 位置 | 夹爪 |
| 7 | base_left_wheel | 速度 | 底盘左轮 |
| 8 | base_back_wheel | 速度 | 底盘后轮 |
| 9 | base_right_wheel | 速度 | 底盘右轮 |

## 核心实现细节

### 坐标转换

- **手臂位置**：4096 ticks = 2π rad，中心点 2048 ticks
  - ticks → rad: `(ticks - 2048) / (4096 / 2π)`
  - rad → ticks: `rad × (4096 / 2π) + 2048`
- **底盘速度**：ros2_control 统一使用 rad/s，内部通过 `TICKS_PER_RAD` 转换为 raw steps/s 写入电机
  - rad/s → steps/s: `rad × TICKS_PER_RAD`（钳位到 s16 范围 [-32768, 32767]）
  - steps/s → rad/s: `steps / TICKS_PER_RAD`（read 时反向转换）

### 电机配置

**手臂电机激活流程**（`on_activate`）：
1. 禁用扭矩 → 解锁 EPROM
2. 写入 homing offset（寄存器 31）
3. 写入最小/最大限位（寄存器 9/11）
4. 设置驱动模式 0（伺服模式）
5. 设置加速度 16ms、速度 profile 32
6. 锁定 EPROM → 启用扭矩

**底盘电机激活流程**：
1. 禁用扭矩 → 解锁 EPROM
2. 设置为轮式模式（连续旋转）
3. 锁定 EPROM → 启用扭矩

### 通信

- 串口波特率：**1,000,000 bps**
- 读操作：`syncReadPacketTx/Rx` 同步读取所有 9 个电机
- 写手臂：`SyncWritePosEx`（位置 + 速度 + 加速度）
- 写底盘：`SyncWriteSpe`（速度 + 加速度）
- 激活时会 Ping 每个电机 3 次确认连接

### 安全机制

- 写入手臂时将目标位置钳位到 [0, 4095]
- 写入底盘时将速度钳位到 [-32768, 32767]
- 停用（`on_deactivate`）时先发送零速度到底盘，再禁用所有电机扭矩

## 依赖

### ROS2 依赖

| 包 | 用途 |
|----|------|
| `hardware_interface` | ros2_control 硬件接口基类 |
| `pluginlib` | 插件加载 |
| `rclcpp` | ROS2 C++ 客户端库 |
| `rclcpp_lifecycle` | 生命周期节点支持 |
| `nlohmann_json` | JSON 标定文件解析 |
| `sensor_msgs` | 传感器消息类型 |
| `geometry_msgs` | 几何消息类型 |

### 外部依赖

| 库 | 来源 | 用途 |
|----|------|------|
| Feetech SDK (FTServo_Linux) | GitHub FetchContent | SMS_STS 串口通信协议 |

## 常见问题

### 1. 串口权限拒绝

```
Failed to connect to motors on port /dev/ttyACM0
open:: Permission denied
```

```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

### 2. 电机无响应

使用 `scan_motors` 工具排查：

```bash
ros2 run lekiwi_hardware scan_motors /dev/ttyACM0
```

可能原因：电机未上电、接线松脱、电机 ID 配置与 URDF 不匹配。

### 3. 标定文件未找到

```
Calibration file not found: /path/to/calib.json
```

确保 URDF 中 `calib_file` 参数指向正确的 JSON 文件路径。

### 4. 关节数量不匹配

```
Expected 9 joints, got X
```

URDF 中的 ros2_control 配置必须恰好包含 9 个 joint（6 手臂 + 3 底盘）。

## 许可证

Apache-2.0

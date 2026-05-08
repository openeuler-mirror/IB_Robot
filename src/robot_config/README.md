# robot_config

ros2_control 和外设的统一机器人配置系统。

## 概述

本软件包为机器人硬件提供统一的配置系统，桥接以下组件：

- **ros2_control**：用于关节/电机控制接口
- **外设**：用于相机和其他设备（通过现有 ROS2 驱动）
- **tensormsg**：用于 ML 策略 I/O 契约

目标是建立机器人硬件配置的单一数据源，消除不同配置系统之间的重复。

## 特性

- **单一 YAML 配置**：在一个文件中定义 ros2_control、相机和 ML 契约
- **使用现有 ROS2 相机驱动**：
  - `usb_cam` 用于 USB 相机（基于 OpenCV）
  - `realsense2_camera` 用于 RealSense D400 系列
- **TF 发布**：自动发布相机坐标系变换
- **标定支持**：标准 ROS2 camera_info_manager 集成
- **tensormsg 集成**：契约通过名称引用外设

## 架构

```
robot_config YAML（单一数据源）
        │
        ├───► ros2_control（关节/电机）
        │       └───► so101_hardware 插件
        │
        ├───► 相机驱动（现有 ROS2 包）
        │       ├───► usb_cam（USB 相机）
        │       └───► realsense2_camera（RealSense D400）
        │
        └───► tensormsg 契约（ML I/O）
                └───► PolicyBridge / EpisodeRecorder
```

## 配置示例

```yaml
robot:
  name: so101_single_arm
  type: so101
  robot_type: so_101

  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  peripherals:
    - type: camera
      name: top
      driver: opencv  # 使用 usb_cam 包
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame

    - type: camera
      name: wrist
      driver: realsense  # 使用 realsense2_camera 包
      serial_number: "12345678"
      width: 640
      height: 480
      fps: 30
      depth_width: 640
      depth_height: 480
      frame_id: camera_wrist_frame

  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top  # 引用上面的相机
        image:
          resize: [480, 640]
```

## 控制模式配置

robot_config 包支持双控制模式，以满足不同 AI 模型的需求：

### 可用控制模式

#### 1. teleop 模式（人工遥操作）

**适用于：** 人工遥操作设备（leader arm、游戏手柄、VR设备）

**特点：**
- 实时直接控制
- 零延迟直通（< 5ms）
- 支持多种输入设备
- 内置安全过滤（关节限位）

**配置：**
```yaml
robot:
  default_control_mode: "teleop"

  control_modes:
    teleop:
      description: "人工遥操作模式（直接控制）"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: false
        force_disable: true

  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"
```

**启动命令：**
```bash
# 遥操作模式（episodic 录制）
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=teleop \
  record:=true \
  record_mode:=episodic \
  use_sim:=false

# 遥操作模式（episodic + Rerun）
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=teleop \
  record:=true \
  record_mode:=episodic \
  record_visualizer:=rerun \
  use_sim:=false

# 另一个终端启动录制客户端
ros2 run dataset_tools record_cli

# 录制完成后转换为 LeRobot 数据集
ros2 run dataset_tools bag_to_lerobot \
  --bags-dir ~/rosbag/episodes/so101_single_arm \
  --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
  --out /path/to/output_dataset
```

#### 2. model_inference 模式（高频位置控制）

**适用于：** 端到端模仿学习模型（ACT、pi0、Diffusion Policy）

**特点：**
- 高频控制（50-100Hz）
- 低延迟（1-3ms）
- 直接基于话题的位置命令
- 反应迅速、运动流畅

**配置：**
```yaml
robot:
  default_control_mode: "model_inference"

  control_modes:
    model_inference:
      description: "高频端到端控制模式（ACT/pi0）"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: true
        execution_mode: "distributed" # 或 "monolithic" (单机零拷贝)
        model: so101_act
```

**启动的控制器：**
- `arm_position_controller` (JointGroupPositionController)
- `gripper_position_controller` (ForwardCommandController)

**命令接口：**
```bash
# 机械臂位置命令
ros2 topic pub /arm_position_controller/commands std_msgs/msg/Float64MultiArray "data: [1.0, 2.0, 3.0, 4.0, 5.0]"
```

### Tracing（ros2_tracing / LTTng）

`robot.launch.py` 支持两个 tracing 相关 launch 参数：

- `enable_tracing:=true`：在启动时开启 LTTng tracing session
- `trace_session_name:=...`：覆盖默认 session 名 `ib_robot_trace`

按包内架构，`robot.launch.py` 只保留编排职责；LTTng session 的创建、命名冲突处理、
以及 shutdown 时的 stop/destroy 生命周期都由
`robot_config/launch_builders/tracing.py` 统一管理。

### Voice ASR（语音识别）

`robot_config` 通过 `robot.voice_asr` 作为语音识别节点的机器人级单一配置来源，并由
`robot_config/launch_builders/voice_asr.py` 注入到 `voice_asr_service` 节点。

常用启动覆盖参数：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  voice_asr_auto_start:=true \
  voice_asr_device_name:=Blackwire \
  voice_asr_realtime_pre_roll_seconds:=0.5
```

| Launch 参数 | 作用 |
| --- | --- |
| `voice_asr_auto_start` | 临时覆盖 `robot.voice_asr.enabled`，设为 `true` 时强制启动 ASR 节点 |
| `voice_asr_device_index` | 临时覆盖 `robot.voice_asr.device_index` |
| `voice_asr_device_name` | 临时覆盖 `robot.voice_asr.device_name`，优先按设备名匹配 |
| `voice_asr_realtime_pre_roll_seconds` | 临时覆盖实时识别 pre-roll 时长 |

`voice_asr_service` 的包级默认值与 `robot_config` 中的 `VoiceASRConfig` 默认值保持同步；
具体机器人仍应以 `config/robots/<robot>.yaml` 中的 `robot.voice_asr` 为准。

#### 2. moveit_planning 模式（轨迹规划控制）

**适用于：** 基于规划的模型（VoxPoser、VLM、目标条件化）

**特点：**
- 轨迹插值和时间参数化
- 通过 MoveIt 动作接口执行
- 碰撞检测和避障
- 更平滑的轨迹

**配置：**
```yaml
robot:
  default_control_mode: "moveit_planning"

  control_modes:
    moveit_planning:
      description: "MoveIt 轨迹规划模式"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller
```

**启动的控制器：**
- `arm_trajectory_controller` (JointTrajectoryController)
- `gripper_trajectory_controller` (JointTrajectoryController)

**命令接口：**
```bash
# 通过 MoveIt 动作执行轨迹
ros2 action send_goal /arm_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{...}"
```

### 完整控制模式配置示例

```yaml
robot:
  name: so101_single_arm
  type: so101
  default_control_mode: "model_inference"  # 或 "teleop" 或 "moveit_planning"

  joints:
    arm: ["1", "2", "3", "4", "5"]
    gripper: ["6"]
    all: ["1", "2", "3", "4", "5", "6"]

  # 控制模式配置
  control_modes:
    teleop:
      description: "人工遥操作模式"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller

    model_inference:
      description: "高频端到端控制模式"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller

    moveit_planning:
      description: "MoveIt轨迹规划模式"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller

  # 遥操作配置
  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"

  # 硬件配置
  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  # 外设（相机、传感器）
  peripherals:
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30

  # ML 契约
  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top
        image:
          resize: [480, 640]
    actions:
      - key: action
        topic: /arm_position_controller/commands  # 根据模式变化
        ros_type: std_msgs/msg/Float64MultiArray
        names: ["1", "2", "3", "4", "5", "6"]
```

### 模式切换工作原理

1. **配置阶段：**
   - `robot.launch.py` 从 YAML 读取 `default_control_mode`
   - 可通过 `control_mode:=xxx` 命令行参数覆盖
   - 验证模式是否存在于 `control_modes` 部分

2. **控制器生成：**
   - 仅生成所选模式中列出的控制器
   - 确保无控制器冲突（同一关节不能被多个控制器控制）

3. **动作分发集成：**
   - `action_dispatch` 节点从 `robot_config` 读取当前模式
   - 实例化适当的执行器（TopicExecutor 或 ActionExecutor）
   - 为上游推理服务提供统一 API

### 控制模式故障排除

#### 模式未切换

**问题：** 命令行覆盖未生效

**解决方案：** 确保 `control_mode` 参数拼写正确：
```bash
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning use_sim:=true
```

#### 控制器冲突

**问题：** 同一关节被多个控制器控制

**解决方案：** 检查配置，确保每种模式使用互斥的控制器：
```yaml
control_modes:
  model_inference:
    controllers:
      - arm_position_controller      # 位置控制器
  moveit_planning:
    controllers:
      - arm_trajectory_controller    # 轨迹控制器（不同类型）
```

#### 执行器类型不匹配

**问题：** 推理服务发送位置命令但启动了轨迹控制器

**解决方案：** 确保控制模式与模型类型匹配：
- ACT/pi0 模型 → `model_inference` 模式
- VoxPoser/VLM 模型 → `moveit_planning` 模式
- 人工遥操作 → `teleop` 模式

## 使用方法

### 启动机器人

```bash
# 启动真实硬件
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm

# 启动仿真
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# MoveIt 规划模式（带 RViz）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true

# MoveIt 模式无 RViz（headless）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true moveit_display:=false

# 分布式推理 — 单机调试（Edge + Cloud 同时启动）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference execution_mode:=distributed use_sim:=true cloud_local:=true

# 分布式推理 — 跨机器部署（端侧仅启动 Edge，Cloud 在算力机器上单独启动）
# 端侧：
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference execution_mode:=distributed use_sim:=true
# 算力机器（需设置相同 ROS_DOMAIN_ID）：
# ros2 launch inference_service cloud_inference.launch.py policy_path:=/path/to/model device:=cuda
# 端侧开发板 Ascend NPU：
# ros2 launch inference_service cloud_inference.launch.py policy_path:=/path/to/model device:=npu
```

### 验证配置

```bash
# 直接使用 Python 验证机器人配置文件
python3 src/robot_config/robot_config/scripts/validate_config.py \
    src/robot_config/config/robots/so101_single_arm.yaml
```

## 相机驱动

### USB 相机（通过 `usb_cam`）

```yaml
- type: camera
  name: usb_cam
  driver: opencv
  index: 0
  width: 640
  height: 480
  fps: 30
  pixel_format: mjpeg2rgb
  frame_id: camera_frame
  camera_info_url: file://$(env HOME)/.ros/camera_info/top.yaml
```

**安装：**
```bash
sudo apt install ros-humble-usb-cam
```

### RealSense 相机（通过 `realsense2_camera`）

```yaml
- type: camera
  name: realsense
  driver: realsense
  serial_number: "12345678"
  width: 640
  height: 480
  fps: 30
  depth_width: 640
  depth_height: 480
  depth_fps: 30
  enable_depth: true
  enable_color: true
  align_depth: false
```

**安装：**
```bash
# Ubuntu
sudo apt install ros-humble-librealsense2*
# openEuler
sudo dnf install ros-humble-librealsense2*
```

## 相机标定

相机内参可以存储在标准 ROS2 位置：

```yaml
- type: camera
  name: top
  driver: opencv
  index: 0
  width: 640
  height: 480
  camera_info_url: file://$(env HOME)/.ros/camera_info/top_camera.yaml
```

可以使用标准 ROS2 相机标定工具创建标定文件：

```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.024 \
  image:=/camera/top/image_raw
```

## tensormsg 集成

robot_config 通过允许观察通过名称引用外设来与 tensormsg 契约集成：

```yaml
# 在 robot_config 中
peripherals:
  - type: camera
    name: top
    width: 640
    height: 480

# 在 contract 部分
contract:
  observations:
    - key: observation.images.top
      topic: /camera/top
      peripheral: top  # 自动填充外设的 width、height、fps
```

当契约加载时，将自动包含外设定义中的相机元数据。


旧的 `robot_interface` 包直接使用 LeRobot 的 Robot 类。本包替换为：

1. **ROS2 层无 LeRobot 依赖**：直接使用 ros2_control
2. **ros2_control 原生**：标准 ROS2 硬件接口
3. **现有 ROS2 相机驱动**：使用 `usb_cam` 和 `realsense2_camera` 包
4. **单一 YAML 配置**：所有硬件在一个地方定义

`robot_interface` 的两个示例配置已手动迁移到：
- `config/robots/so101_single_arm.yaml`（来自 `single_arm_banana.yaml`）
- `config/robots/so101_dual_arm.yaml`（来自 `dual_arms_pencil.yaml`）

## 故障排除

### 相机无法打开

检查 USB 权限：
```bash
ls -l /dev/video*
sudo chmod 666 /dev/video0
```

或将用户添加到 `video` 组：
```bash
sudo usermod -a -G video $USER
```

### RealSense 相机未找到

安装 librealsense2：
```bash
# Ubuntu
sudo apt install librealsense2-utils librealsense2-dev
sudo apt install ros-humble-librealsense2*

# openEuler
sudo dnf install librealsense2-utils librealsense2-devel
sudo dnf install ros-humble-librealsense2*
```

检查相机是否连接：
```bash
realsense-viewer
```

### 标定文件未找到

确保路径正确并以 `file://` 开头：
```yaml
camera_info_url: file:///home/user/.ros/camera_info/top.yaml
```

### 控制器加载失败

如果遇到 "Controller already loaded" 错误，运行清理脚本：
```bash
./scripts/cleanup_ros.sh
```

## 仿真依赖（MuJoCo）

`use_sim:=true` 模式依赖 `ros-humble-mujoco-ros2-control`，通过 rosdep 声明为运行时依赖：

```bash
# 安装所有依赖（含 MuJoCo 仿真包）
rosdep install --from-paths src --ignore-src -y
```

该命令会自动安装 `ros-humble-mujoco-ros2-control` 及其传递依赖
（`mujoco_ros2_control_msgs`、`mujoco_ros2_control_plugins`、`mujoco_vendor`）。
不再需要手动初始化 git submodule。

## 参考资料

- [usb_cam GitHub](https://github.com/ros-drivers/usb_cam) - ROS2 USB 相机驱动
- [realsense-ros GitHub](https://github.com/realsenseai/realsense-ros) - Intel RealSense ROS2 封装
- [action_dispatch README](../action_dispatch/README.md) - 详细执行器文档
- [docs/architecture.md](../../docs/architecture.md) - 系统架构概览

## 许可证

Apache-2.0

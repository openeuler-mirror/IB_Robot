# Dataset Tools

ROS 2 数据集采集与转换工具，用于 LeRobot v3 数据集格式。

## 概述

本包提供以下功能：

- **Episode 录制**: 通过 Action Server 控制的分段录制
- **Bag 转 LeRobot**: 将 ROS 2 bag 转换为 LeRobot v3 数据集格式

## 架构设计

### 单一真理来源 (Single Source of Truth)

所有数据集工具使用 `robot_config` 包下的配置文件作为唯一配置来源，例如：

```
src/robot_config/config/robots/so101_single_arm.yaml
├── contract.observations    ← 观测定义（相机、状态等）
├── contract.actions         ← 动作定义（arm、gripper）
├── contract.rate_hz         ← 采样率
└── control_modes            ← 运行时控制模式配置
```

这确保了：
- 训练数据导出与在线推理配置一致
- 无需维护重复的 contract 文件
- 配置变更自动传播到所有组件

## 工具

### 1. record_cli - 交互式录制客户端

用于控制 episode 录制的命令行工具。

**启动录制服务**（Ubuntu 录制服务器）：
```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    use_sim:=false
```

**如需启用 Rerun 可视化**：
```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    record_visualizer:=rerun \
    use_sim:=false
```

**启动录制客户端**（同机或另一台设置了相同 `ROS_DOMAIN_ID` 的机器）：
```bash
ros2 run dataset_tools record_cli
```

episodic 录制目录现在按 dataset 组织：

```text
<bag_base_dir>/
└── <dataset_name>/
    ├── dataset.yaml
    └── episodes/
        ├── episode_000001/
        │   ├── metadata.yaml
        │   └── *.mcap
        └── episode_000002/
            ├── metadata.yaml
            └── *.mcap
```

- `bag_base_dir` 来自 `robot_config.recording.bag_base_dir`
- `dataset_name` 默认取 `recording.dataset_name`，未配置时回退到机器人名
- `dataset.yaml` 保存 dataset 级元信息；可选通过 `recording.default_task`、`recording.task_family` 预填任务语义
- episode 级 prompt 仍写入各自 bag 的 `metadata.yaml`

**使用方式**：
```
========================================================
Dataset Collection CLI
Enter prompt text to start recording. (Press Enter to reuse: 'get')
Type 'q' or 'quit' to exit.
========================================================
Prompt > get        # 输入任务描述开始录制
[INFO] 🔴 RECORDING STARTED. (Press Enter to stop early)
[INFO] ✅ RECORDING SAVED: Wrote 1894 messages to /path/to/episode
Prompt > q          # 退出
```

录制完成后，推荐直接把整个 dataset 根目录转换成 LeRobot v3 数据集：

```bash
ros2 run dataset_tools bag_to_lerobot \
    --bags-dir ~/rosbag/episodes/so101_single_arm \
    --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
    --out /path/to/output_dataset
```

### 2. bag_to_lerobot - Bag 转 LeRobot 数据集

将 ROS 2 episodic dataset 根目录转换为 LeRobot v3 数据集格式。

**基本用法**：
```bash
ros2 run dataset_tools bag_to_lerobot \
    --bags-dir ~/rosbag/episodes/so101_single_arm \
    --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
    --out /path/to/output_dataset
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bags-dir` | dataset 根目录或 episodes 目录，自动发现多个 episode bag | 必需 |
| `--robot-config` | robot_config.yaml 路径 | 必需 |
| `--out` | 输出数据集目录 | 必需 |
| `--repo-id` | 数据集 repo_id | `rosbag_v30` |
| `--no-videos` | 存储 PNG 图像而非视频 | `false` |
| `--timestamp` | 时间戳来源 (contract/bag/header) | `contract` |
| `--image-threads` | 图像写入线程数 | `4` |
| `--chunk-size` | 每个 chunk 的帧数 | `1000` |

**输出结构**：
```
output_dataset/
├── videos/
│   ├── observation.images.front/
│   │   └── chunk-000/file-000.mp4
│   ├── observation.images.top/
│   └── observation.images.wrist/
├── data/
│   └── chunk-000/file-000.parquet
└── meta/
    ├── info.json
    ├── tasks.parquet
    ├── stats.json
    └── episodes/
```

### 3. episode_recorder - 录制服务节点

由 launch 文件自动启动的录制服务，提供 `record_episode` Action Server。

通常不需要直接运行，由 `robot.launch.py` 根据 `record_mode:=episodic` 参数自动加载。
录制结果会写到 `<bag_base_dir>/<dataset_name>/episodes/episode_XXXXXX/`，并在 dataset 根目录生成 `dataset.yaml`。

### 4. camera_alignment - 基于 ArUco 的相机对齐工具

用于在数据采集或复现前，直接读取本机视频设备并对齐摄像头视角。

**基本用法**：
```bash
ros2 run dataset_tools camera_alignment \
    --cameras_index_or_path /dev/video0 \
    --reference-path /tmp/camera_reference_multi.json \
    --reference-image-path /tmp/reference_img.png
```

工具支持：

- 保存当前 ArUco 角点作为参考基准
- 实时显示与参考画面的平均像素误差
- 进入“虚影对齐”界面辅助恢复视角

详细说明见：

- `docs/tools/camera_alignment.md`

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│   src/robot_config/config/robots/so101_single_arm.yaml     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ contract (单一真理来源)                               │   │
│  │ - observations (front, top, wrist, state)           │   │
│  │ - actions (arm, gripper)                            │   │
│  │ - rate_hz: 20                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │  录制服务    │     │  数据转换    │     │  推理服务    │
   │ episode_    │     │ bag_to_     │     │ lerobot_    │
   │ recorder    │     │ lerobot     │     │ policy_node │
   └─────────────┘     └─────────────┘     └─────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ROS 2 Bag          LeRobot Dataset      Model Inference
```

## 配置示例

`src/robot_config/config/robots/so101_single_arm.yaml` 中的 contract 配置：

```yaml
robot:
  name: so101_single_arm
  
  contract:
    rate_hz: 20
    max_duration_s: 90.0
    
    observations:
      - key: observation.images.front
        topic: /camera/front/image_raw
        type: sensor_msgs/msg/Image
        image:
          resize: [480, 640]
          
      - key: observation.images.top
        topic: /camera/top/image_raw
        type: sensor_msgs/msg/Image
        image:
          resize: [480, 640]
          
      - key: observation.images.wrist
        topic: /camera/wrist/image_raw
        type: sensor_msgs/msg/Image
        image:
          resize: [480, 640]
          
      - key: observation.state
        topic: /joint_states
        type: sensor_msgs/msg/JointState
        selector:
          names: [position.1, position.2, position.3, position.4, position.5, position.6]
    
    actions:
      # Arm joints (1-5)
      - key: action
        selector:
          names: [action.0, action.1, action.2, action.3, action.4]
        publish:
          topic: /arm_position_controller/commands
          type: std_msgs/msg/Float64MultiArray
          
      # Gripper joint (6) - same key for consolidation
      - key: action
        selector:
          names: [action.5]
        publish:
          topic: /gripper_position_controller/commands
          type: std_msgs/msg/Float64MultiArray
```

## 注意事项

1. **Action 合并**: 多个 action spec 使用相同的 `key: action` 会被自动合并为一个 6-DOF action
2. **观测过滤**: 推理服务会根据模型的 `config.json` 自动过滤需要的观测
3. **录制模式**: 
   - `record_mode:=continuous` - 持续录制到一个文件
   - `record_mode:=episodic` - 分段录制，需要 `record_cli` 控制

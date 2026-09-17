# IB-Robot

> IB-Robot (Intelligence Boom Robot): 融合 LeRobot 与 ROS 2 生态的智能具身机器人开发框架

## 重磅更新：支持 OpenClaw 社交控制！

IB-Robot 现已全面支持通过 **[OpenClaw](https://github.com/openclaw/openclaw)** AI Agent 进行远程社交控制！无论是在 **Gazebo 仿真环境** 还是 **真实 SO-101 机械臂** 上，你都可以通过 飞书、QQ、Discord 等软件，用最自然的语言与机器人对话并下达指令。

|                            仿真演示 (Simulation)                            |                             真实硬件 (Real Robot)                            |
| :---------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| ![仿真演示](docs/pictures/openclaw_sim.gif) | ![真实硬件](docs/pictures/openclaw_real.gif) |

***

## 项目定位

**用户指导文档**：[IB-Robot 使用指南](https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/features/embodied_ai/index.html)

IB-Robot 是一个**智能融合机器人开发框架**，旨在打通 Hugging Face LeRobot 机器学习生态与 ROS 2 机器人中间件之间的壁垒，为具身智能研发提供从采集、训练到部署的完整工具链。

### 核心融合能力

| 维度       | LeRobot 生态    | ROS 2 生态   | IB-Robot 方案              |
| -------- | ------------- | ---------- | ------------------------ |
| **数据流**  | Episode 回合    | Topic 话题   | 契约驱动的双向实时转换              |
| **时间观**  | 离散时间步 (Steps) | 连续时间流 (RT) | 自动对齐与高频插值平滑              |
| **控制方式** | 端到端神经网络模型     | 分层规划控制架构   | **双模控制 (ACT vs MoveIt)** |
| **部署形态** | Python 脚本     | ROS 2 节点   | 分布式端边协同部署                |

### 平台支持

当前主干工作流已支持三类运行平台：

| 平台 | 角色定位 | 当前支持状态 | 典型场景 |
| --- | --- | --- | --- |
| **Ubuntu 22.04** | 主机 / 开发机 | 完整支持 setup、build、仿真、录制、MoveIt、推理联调 | Gazebo 仿真、数据采集、单机推理、边云联调 |
| **openEuler Embedded 24.03** | 端侧开发板 | 支持 setup、clean build 与板端运行 | NPU 推理、实机控制、录制客户端 |
| **OpenHarmony 5.1** | 端侧开发板 | 支持板端运行工作流、HDC 调试、最小 inference workspace 构建辅助与 LeRobot patch profile | BQ3588HM 板端推理、HDC/SSH 调试 |

## 系统架构

![IB-Robot 架构图](docs/pictures/architecture.png)

### 架构深度解析

IB-Robot 构建了一个从感知、决策到执行的端到端闭环体系，实现了机器学习世界与机器人控制世界的无缝对接：

1. **多模态感知与采集**:
   - **底层感知**: 通过 ROS 2 Driver 统一接入多路相机 (USB/RealSense)、雷达及麦克风。
   - **多样化采集**: 支持 **VR 手柄、Xbox 控制器及手机 IMU** 等遥操作设备，为模仿学习提供专家示范数据。
2. **协议转换枢纽 (tensormsg)**:
   - 作为架构的枢纽，tensormsg 负责 `ros_msg` 与 `tensor` 之间的双向转换，通过合约（Contract）机制保证数据流的类型安全与一致性。
3. **推理与研发服务 (Inference Service)**:
   - 支持各类 VLA（视觉-语言-动作）大模型（如 SmolVLA, Pi0.5）以及端到端策略模型（如 ACT, Diffusion Policy）。系统支持 **自动检测后端** 并根据控制模式按需启动。
4. **统一动作执行器 (Action Dispatch)**:
   - 充当机器人的"小脑"。在 ACT 模式下负责 **Action Chunking** 调度与高频插值；在规划模式下对接 **MoveIt 2** 执行受限轨迹，并提供统一的 `RobotStatus` 汇报。
5. **配置驱动中心 (robot\_config)**:
   - 实现"规格驱动本体行为"。通过单一 YAML 定义关节、控制器模式及传感器外参，支持一键切换仿真与实机环境。

***

## 仓库结构

```text
IB_Robot/                           # 主工作空间 (本仓库)
├── .gitmodules                     # Git 子模块配置
├── README.md                       # 本文件
├── LICENSE                         # Apache 2.0 许可证
├── config.json                     # AI Agent 配置文件 (AtomGit API 令牌等)
│
├── .agents/                        # AI Agent 配置目录
│   └── skills/                     # AI Agent 技能库 (详见 .agents/skills/README.md)
│
├── libs/                           # 外部依赖库
│   └── lerobot/                    # [子模块] LeRobot 训练框架
│
├── src/                            # 核心源码包集合
│   ├── robot_config/               # 系统总控、规格定义与启动入口
│   ├── action_dispatch/            # 统一动作执行器 (双模支持)
│   ├── task_dispatch/              # 任务调度与分发服务
│   ├── tensormsg/                  # LeRobot ↔ ROS 2 协议转换枢纽
│   ├── ibrobot_msgs/               # 系统统一接口定义 (Message/Action/Service)
│   ├── dataset_tools/              # 数据集采集与转换工具 (Episode Recorder)
│   ├── robot_teleop/               # 遥操作控制 (Leader Arm/Xbox 手柄)
│   ├── robot_description/          # 统一机器人 URDF/SRDF/MJCF 模型描述
│   ├── lekiwi_description/         # Lekiwi 底盘 URDF/Mesh 模型描述
│   ├── robot_moveit/               # MoveIt 2 运动规划集成
│   ├── robot_navigation/           # 导航功能包
│   ├── inference_service/          # 多模型推理与部署服务
│   ├── so101_hardware/             # SO-101 电机驱动接口
│   ├── lekiwi_hardware/            # Lekiwi 底盘硬件驱动接口
│   ├── hardware_mock/              # 硬件模拟 (Mock) 接口
│   ├── omni_wheel_controller/      # 全向轮控制器插件
│   ├── pymoveit2/                  # [子模块] MoveIt2 Python 接口
│   ├── rosclaw/                    # [子模块] OpenClaw 社交控制集成
│   ├── sim_models/                 # 仿真场景模型 (Gazebo/MuJoCo)
│   ├── model_utils/                # 模型工具库
│   ├── torch_models/               # 自研 PyTorch 模型源码，每个模型独立目录
│   ├── attention_viz/              # 注意力可视化工具
│   ├── voice_asr_service/          # 语音识别服务
│   ├── workflows/                  # CI/CD 配置
│   │
│   ├── embodied_agent/             # Hermes Agent plan 生命周期与执行编排
│   ├── perception_service/         # 连续场景理解服务 (RGB-D / 多视角感知)
│   ├── skill_library/              # 技能执行层 (skill → primitive → MoveIt)
│   └── safety_guard/               # 显式安全校验层 (白名单 + 工作空间边界)
│
├── docs/                           # 深度架构文档与开发指南
│   ├── pictures/                   # 架构图与演示 GIF
│   └── videos/                     # 演示视频 (源文件)
├── scripts/                        # 环境配置与验证工具脚本
└── build/                          # 编译输出 (自动创建)
```

***

## 环境初始化 (First-time Setup)

**重要：本步骤仅需在初次克隆项目后运行一次。**

### 0. 系统要求

- **操作系统**: 当前支持三平台协同：Ubuntu 主机负责仿真、录制服务或云侧推理；端侧开发板支持 openEuler Embedded 与 OpenHarmony
- **ROS 版本**: ROS 2 Humble
- **Python**: >= 3.10，默认使用系统自带 Python。**严禁在 Conda 激活的环境中执行，否则会导致动态库冲突。**
- **加速器**: 支持 NVIDIA GPU、Ascend 310B、Ascend 310P，若未检测到则按 CPU-only 路径运行。

### 1. 执行一键初始化

运行 `./scripts/setup.sh`。该脚本会自动完成以下重型操作：

1.  **子模块同步**: 执行 `git submodule update --init --recursive`，下载核心源码。
2.  **平台与硬件检测**: 自动识别 Ubuntu / openEuler Embedded / OpenHarmony，以及 NVIDIA GPU / Ascend 310B / 310P / CPU-only 环境。
3.  **ROS 2 安装** (如未安装): 自动检测并安装 ROS 2 Humble 和 colcon 构建工具。
4.  **系统依赖安装**: 通过系统包管理器安装 C++ 编译工具、`nlohmann-json` 等硬件驱动依赖。
5.  **虚拟环境 (venv) 构建**: 在根目录创建 `venv` 文件夹。这能确保 ML 相关依赖与系统 ROS 2 环境隔离，同时通过 `--system-site-packages` 复用系统 `rclpy`。
6.  **ML 栈安装**: 自动在 `venv` 中安装 `lerobot`、硬件依赖以及适配 ROS 2 Humble 的 NumPy 1.26.x。
7.  **环境验证**: 自动验证 `rosdepc`、`colcon`、`rclpy`、`lerobot` 与 NumPy 兼容性。

***

## 开发工作流

### 1. 加载环境

每次开启新终端后，请在 `IB_Robot` 项目根目录下加载环境：

```bash
cd /path/to/IB_Robot
source .shrc_local
```

> **注意**：`.shrc_local` 会自动完成 `venv` 激活、ROS 2 环境加载和工作区 `install/setup.sh` 的 source。每次另起新终端都必须重新执行上述命令，否则 `ros2` 命令和 Python 包将不可用。

完成首次构建后，再额外加载工作区环境：
```bash
source install/setup.sh
```

### 2. 分配 Domain ID

为了避免与局域网内其他 ROS 2 用户冲突，建议设置唯一的 Domain ID。**每次另起新终端都需要重新设置**：

```bash
export ROS_DOMAIN_ID=<0-232之间的唯一数字>
```

> **注意**：跨机器运行时，参与的所有机器必须使用**相同的 `ROS_DOMAIN_ID`**。

### 3. 编译项目

代码修改后，运行统一构建脚本：

```bash
./scripts/build.sh --clean
```

*注：`build.sh` 现在只负责加载环境并执行构建；Python 环境、`lerobot` 可编辑安装与 NumPy 兼容性由 `setup.sh` 统一负责。*

***

## 运行指南

所有运行入口都以 `robot_config` 包的统一入口 `robot.launch.py` 为主。下文中的"端侧开发板"统一指可运行 **openEuler Embedded** 或 **OpenHarmony** 的板端设备。

开始任一场景前，请先完成环境加载并设置唯一的 `ROS_DOMAIN_ID`。跨机器运行时，参与的所有机器必须使用**相同的 `ROS_DOMAIN_ID`**。

```bash
source .shrc_local
export ROS_DOMAIN_ID=<0-232之间的唯一数字>
```

更详细的子模块说明可参考下表：

| 文档 | 简短说明 |
| :--- | :--- |
| [`src/inference_service/README.md`](src/inference_service/README.md) | 推理服务架构、单机/分布式部署与 NPU/GPU Cloud 节点启动方式 |
| [`src/robot_moveit/README.md`](src/robot_moveit/README.md) | MoveIt Planning 控制、`/cmd_pose` 用法与 headless 启动方式 |
| [`src/dataset_tools/README.md`](src/dataset_tools/README.md) | episodic 录制、`record_cli` 用法与 `bag_to_lerobot` 数据集转换流程 |

### 一、Ubuntu 仿真场景

#### 1. Ubuntu 启动仿真环境（仅仿真与控制器）

适合验证 Gazebo、相机、控制器和基础 ROS 2 拓扑，不启动模型推理。

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true \
    with_inference:=false
```

#### 2. Ubuntu 启动仿真并用模型推理控制仿真机械臂

显式切到 `model_inference` 模式，使用本机推理链路控制仿真机械臂。

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true
```

#### 3. Ubuntu 启动 MoveIt Planning 控制（仿真）

该场景默认会启动 MoveIt 与 RViz，并暴露 `/cmd_pose` 接口用于发送目标位姿。

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true
```

如需在板端或无图形界面的环境中运行，可关闭 RViz：

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true \
    moveit_display:=false
```

发送位姿命令控制机械臂移动：

```bash
ros2 topic pub /cmd_pose geometry_msgs/Pose "{
  position: {x: 0.15, y: 0.0, z: 0.25},
  orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}
}" --once
```

查看末端位姿反馈：

```bash
ros2 topic echo /robot_status/ee_pose
```

### 二、真机场景

#### 1. 端侧开发板启动 MoveIt Planning 控制（真机）

与 Ubuntu 上的 MoveIt 用法保持一致，只是 `use_sim` 不再开启，适合真实机械臂控制。

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning
```

控制接口仍然是同一套话题：

```bash
ros2 topic pub /cmd_pose geometry_msgs/Pose "{
  position: {x: 0.15, y: 0.0, z: 0.25},
  orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}
}" --once
```

### 三、分布式推理部署场景

分布式模式在 robot YAML 的命名 pipeline 中声明 `execution_mode: distributed`。机器人侧启动
Edge pipeline，算力侧单独启动 `cloud_inference.launch.py`。两端必须使用相同的 pipeline ID、
deployment name 和 bundle identity。

#### 1. Ubuntu 单机调试分布式推理（Edge + Cloud 同机）

适合开发和联调，在一台 Ubuntu 机器的两个终端运行两侧节点。先准备一个将 `policy` pipeline
配置为 distributed 的 YAML。

```bash
# 终端 1：Edge
ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/so101_single_arm_distributed.yaml \
    control_mode:=model_inference \
    use_sim:=true

# 终端 2：Cloud
ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cpu
```

#### 2. Ubuntu 启动仿真环境，端侧开发板启动 NPU 推理

Ubuntu 主机负责仿真与 Edge 侧预处理/后处理；端侧开发板负责云侧纯推理。两台机器必须位于同一局域网，并设置相同的 `ROS_DOMAIN_ID`。

**Ubuntu 主机（仿真 + Edge）**

```bash
ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/so101_single_arm_distributed.yaml \
    control_mode:=model_inference \
    use_sim:=true
```

**端侧开发板（NPU Cloud 节点）**

```bash
ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=npu
```

这里的 `npu` 是 manifest 中命名的 Torch NPU deployment，而不是 backend alias。GPU server
使用与 edge YAML 相同的命名 deployment，例如 `cuda`。

快速验证分布式链路是否打通：

```bash
ros2 node list | grep -E 'inference_policy|inference_policy_cloud'
ros2 action info /inference/policy/dispatch
ros2 topic info /inference/policy/request
ros2 topic info /inference/policy/result
ros2 topic hz /inference/policy/heartbeat
```

#### 3. OpenHarmony 板端作为算力侧（RK3588）

OpenHarmony 板端的完整指南（构建、部署、RKNN 推理、内核、SSH 配置等）详见 **[README.OpenHarmony.md](README.OpenHarmony.md)**。

### 四、数据集录制场景

episodic 录制始终由两部分组成：

1. `robot.launch.py` 启动 `episode_recorder` 录制服务端
2. `ros2 run dataset_tools record_cli` 启动交互式录制客户端

`record_visualizer:=rerun` 只会额外拉起 Rerun 可视化 sidecar，不会替代 `record_cli`。

#### 1. Ubuntu 启动录制服务器 + Ubuntu 启动录制客户端

**不启用 Rerun**

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    use_sim:=false
```

**启用 Rerun**

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    record_visualizer:=rerun \
    use_sim:=false
```

**客户端（同机另一个终端）**

```bash
ros2 run dataset_tools record_cli
```

启动 `record_cli` 后输入任务描述即可开始录制，按回车可提前结束当前 episode。

#### 2. Ubuntu 启动录制服务器，端侧开发板启动录制客户端

该模式适合把机器人控制与录制操作分离。Ubuntu 主机负责录制服务端，端侧开发板只负责运行 `record_cli`。两端仍需保持相同的 `ROS_DOMAIN_ID`。

**Ubuntu 录制服务器（可选启用 Rerun）**

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    use_sim:=false
```

如需开启可视化，在服务端命令中增加：

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    record_visualizer:=rerun \
    use_sim:=false
```

**端侧开发板录制客户端**

```bash
ros2 run dataset_tools record_cli
```

录制完成后，可将整个 episodic dataset 根目录转换为 LeRobot 数据集格式：

```bash
ros2 run dataset_tools bag_to_lerobot \
    --bags-dir ~/rosbag/episodes/so101_single_arm \
    --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
    --out /path/to/output_dataset
```

bag 目录组织、`dataset.yaml` 元信息和更多转换参数，详见 `src/dataset_tools/README.md`。

***

## 参数说明

| 参数名 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `robot_config` | 机器人配置名称（对应 `config/robots/` 下的 YAML） | `so101_single_arm` |
| `config_path` | 配置文件绝对路径（可选，覆盖 `robot_config`） | 空 |
| `use_sim` | 是否启用仿真模式 | `false` |
| `control_mode` | 覆盖默认控制模式（`model_inference` / `moveit_planning` / `teleop`） | 从 YAML 读取 |
| `with_inference` | 强制启用/禁用推理服务（空则自动检测） | 空 |
| `with_moveit` | 强制启用/禁用 MoveIt 核心（空则自动检测） | 空 |
| `moveit_display` | 是否启动 MoveIt RViz 可视化界面 | `true` |
| `record` | 是否启用录制流水线 | `false` |
| `record_mode` | 录制模式（`continuous` / `episodic`） | `continuous` |
| `record_visualizer` | 录制可视化器（`none` / `rerun`） | `none` |
| `with_embodied` | 基础 `robot_config` launch 中保留的兼容覆盖；完整具身链路请使用 `embodied_bringup` | 空 |
| `auto_start_controllers` | 是否在启动后自动激活控制器 | `true` |

推理执行模式不是 launch 参数；它在 robot YAML 的每个命名 pipeline 中配置。分布式 Cloud 节点需
使用 `inference_service cloud_inference.launch.py` 单独启动。

***

## 四、具身 AI 流水线

IB-Robot 内置了一条 Hermes Agent 具身执行链路，Agent 通过结构化技能计划驱动 MoveIt 2 执行真实动作。该链路在 `moveit_planning` 控制模式下可用。

### 链路结构

```text
Hermes / robot-skill
  → agent_plan_node         # Agent plan / validate / confirm / execute
  → skill_executor_node     # 技能 → primitive 分解
  → safety_guard_node       # 安全校验（白名单 + 工作空间边界）
  → moveit_gateway          # MoveIt 2 运动规划执行
```

### 启动具身流水线

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true \
    moveit_display:=false \
    authorize_motion:=false
```

`authorize_motion` 默认 `false`，因此上述命令只启动可查询、默认拒绝运动的 Gateway。只有操作员完成
现场安全检查后，才能在 launch 时显式设为 `true`；Agent、CLI、YAML 和动态 ROS 参数都不能代替操作员授权。

### Agent 默认控制入口

Hermes 等 Agent 默认通过 `robot-skill` 访问 Capability Gateway：

```text
Hermes -> ibrobot-control Agent Skill -> robot-skill -> ROS Capability Gateway
```

`list-skills`、`describe` 和 `list-poses` 是不初始化 ROS 的 catalog-only 命令；`status`、`validate`、
`execute` 和 `cancel` 是 Gateway runtime 命令。普通命令输出单行 JSON，`execute` 输出 JSONL feedback 和
唯一 terminal result。命令、退出码和授权说明见
[`src/robot_skill_cli/README.md`](src/robot_skill_cli/README.md)。

`robot_mcp` 兼容层已移除，统一通过 `robot-skill` 访问 Capability Gateway。

SO-101 真机从启动、控制器检查、Hermes 自然语言计划展示并立即执行到回原位和关闭的完整操作流程，见
[`docs/hermes_so101_real_robot_manual_validation_zh.md`](docs/hermes_so101_real_robot_manual_validation_zh.md)。

### 发送 Agent 计划

```bash
# Hermes 通过 robot-skill 提交 Agent 计划
robot-skill --config-name so101_single_arm plan-workflow \
  --request-id plan-request-001 --text "打开夹爪" \
  --workflow-json '[{"schema_version":1,"skill_name":"open_gripper_skill"}]'
```

更多配置说明见：
- [`src/embodied_agent/README.md`](src/embodied_agent/README.md) — 任务入口与编排
- [`src/perception_service/README.md`](src/perception_service/README.md) — 场景感知服务
- [`src/skill_library/README.md`](src/skill_library/README.md) — 技能执行层
- [`src/safety_guard/README.md`](src/safety_guard/README.md) — 安全校验层

***

## AI Agent Skills

IB-Robot 内置 AI 编程代理技能，帮助 Claude Code、Gemini CLI、OpenCode 等 AI Agent 更好地理解项目架构和开发流程。可用技能详见 [.agents/skills/README.md](.agents/skills/README.md)。

机器人能力发现和受控执行的默认接口是 `robot-skill`，而不是 MCP、裸 `ros2`、primitive、MoveIt 或
controller 命令。自然语言 Workflow 必须先展示并 flush，随后立即进入 Gateway 校验和执行；运行中的 Agent
不得开启 `authorize_motion`。

### config.json 配置文件

`config.json` 用于存储 AI Agent 所需的配置信息，目前主要用于 AtomGit API 集成：

```json
{
  "atomgit": {
    "token": "$ATOMGIT_TOKEN",
    "owner": "openEuler",
    "repo": "IB_Robot",
    "baseUrl": "https://api.atomgit.com"
  }
}
```

**获取 AtomGit Personal Access Token**：

1. 访问 <https://atomgit.com> 并登录
2. 点击右上角头像 → 个人设置
3. 找到「访问令牌」选项
4. 点击「新建访问令牌」，勾选 `repo` 和 `pull_request` 权限
5. **立即复制保存** Token（只显示一次）

设置环境变量：

将以下内容添加到你本地的 `~/.zshrc` 或 `~/.bashrc` 中：

```bash
export ATOMGIT_TOKEN="your_token_here"
```

### 支持的 Agent

所有符合 Agent Skills 标准的客户端都会自动扫描 `.agents/skills/`：
详见 [agentskills.io](https://agentskills.io)。

***

## 基于 OpenClaw 的社交控制与远程 AI 代理

IB-Robot 深度集成 [OpenClaw](https://github.com/openclaw/openclaw) AI Agent 框架，配合 [RosClaw](https://github.com/PlaiPin/rosclaw) 桥接器，实现通过 飞书、QQ、Discord 或 Slack 以自然语言对话的方式远程控制机器人。

> **兼容性说明**：本节描述可选的旧 RosClaw 社交桥接，不是 Hermes 的默认控制面。Hermes 和本地 Agent
> 必须使用 `robot-skill` 访问 Capability Gateway；不得把 `docs/ib_robot_social_skill.md` 复制或注册为
> `ibrobot-control`，因为该旧文档包含裸 ROS 运动接口。

> **致谢**: 感谢 OpenClaw 团队提供的强大 AI 代理框架，以及 RosClaw 提供的 ROS 2 桥接方案。

### 1. 机器人端配置 (RosClaw & Bridge)

机器人端需要安装 WebSocket 桥接驱动并启动发现服务。

- **拉取子模块**:
  确保已拉取最新的 RosClaw 子模块源码：
  ```bash
  git submodule update --init --recursive
  ```
- **安装系统依赖**:
  ```bash
  # 必须安装 rosbridge_suite 以提供 WebSocket 通信能力
  sudo apt-get update && sudo apt-get install -y ros-humble-rosbridge-suite
  ```
- **启动机器人本体**:
  首先启动机器人本体程序（支持仿真或实机）：
  ```bash
  # use_sim:=true 为仿真模式，false 为真实硬件模式
  ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference use_sim:=true with_inference:=false
  ```
- **启动社交桥梁**:
  本项目已将 RosClaw 作为子模块引入 `src/rosclaw`。执行以下脚本一键启动：
  ```bash
  # 自动编译子模块并启动 rosbridge_websocket, rosapi 和 discovery 节点
  ./scripts/start_rosclaw.sh
  ```
  启动后，系统将在 `9090` 端口开启 WebSocket 服务。

### 2. 控制端配置 (OpenClaw)

- OpenClaw 是机器人的"大脑"和"前端"，负责连接社交软件并调用 LLM 理解指令。

> **重要**：在使用 OpenClaw 控制机器人之前，必须确保 OpenClaw 侧的 `ROS_DOMAIN_ID` 与机器人端一致。否则 OpenClaw 将无法发现 ROS2 话题和服务，表现为"ros2 CLI 不可用"或无法发送控制指令。需要在会话时告知 OpenClaw 对应的 `ROS_DOMAIN_ID`。

- **安装 OpenClaw**:
  推荐使用官方提供的快速安装脚本（需要 Node.js 22+）：
  ```bash
  # 安装 OpenClaw CLI
  npm install -g openclaw

  # 执行初始化向导，配置你的 LLM (如 GLM-4/5 或 GPT-4)
  openclaw onboard
  ```
- **集成 RosClaw 插件**:
  ```bash
  # 在 IB_Robot 根目录下执行，将插件安装到 OpenClaw
  openclaw plugins install ./src/rosclaw/extensions/openclaw-plugin
  ```
- **配置机器人连接**:
  ```bash
  # 设置机器人 WebSocket 地址（替换为实际 IP）
  openclaw config set plugins.entries.rosclaw.config.rosbridge.url "ws://<机器人IP>:9090"
  ```
- **启动 Gateway**:
  ```bash
  openclaw gateway
  ```

### 3. 交互示例

连接成功后，你可以在网页端 (`http://localhost:18789`) 或绑定的飞书、QQ 或 Discord 中输入：

- *"查看机器人当前的能力清单"* —— 获取所有传感器话题。
- *"把机械臂恢复到初始位置"* —— AI 会根据技能文档自动将角度转换为**弧度**。
- *"帮我看看桌子上有什么？"* —— AI 会调用 `/camera/top/image_raw` 抓拍并分析图像。
- *"帮我抓取桌上的瓶子"* —— AI 将触发 IB-Robot 的 `DispatchInfer` AI 任务。

***

## FAQ

### 1. 控制器残留/清理

如果遇到控制器无法启动或端口占用的问题，请运行清理脚本重置 ROS 2 后台进程：

```bash
./scripts/cleanup_ros.sh
```

### 2. 共享内存 (SHM) 报错

若出现 `RTPS_TRANSPORT_SHM Error`，请尝试清理缓存：

```bash
sudo rm -rf /dev/shm/fastrtps_*
export ROS_LOCALHOST_ONLY=1
```

### 3. 仿真窗口无法显示

若启动仿真后没有出现可视化窗口（如 MuJoCo/Gazebo），请检查 `DISPLAY` 环境变量。在 Wayland 或某些远程桌面环境下，可能需要手动设置：

```bash
export DISPLAY=:1
```

---

## 更新日志

### 2025-06-15：感知系统增强 + 未实现技能清理

#### 感知服务（perception_service）重大升级

**新增 2D 物体检测 + 3D 坐标接地能力**，感知服务从"场景描述"升级为"物体级感知"：

- **新增 `object_parser.py`**：解析 VLM 返回的物体级 grounding 结果（label、bbox、confidence），支持 `0-1000` 归一化坐标系到实际像素坐标的自动缩放
- **新增 `grounding_3d.py`**：将 2D bbox 结合 RGB-D 深度图反投影为相机坐标系 3D 位置（中位深度估计 + 越界过滤）
- **新增 ROS 消息类型**：
  - `SceneObject.msg`：单物体检测结果（label、bbox、3D pose、confidence）
  - `SceneObservation.msg`：完整场景观测（多物体列表 + 场景摘要 + 风险评估）
- **新增 `perception_observation` topic**：发布结构化 `SceneObservation` 消息，供下游 planner / executor 消费
- **prompt_builder 升级**：在 VLM prompt 中增加 `objects` 字段请求，要求返回 bbox 坐标；删除重复的 `append_images` 定义
- **并发安全增强**：会话历史加线程锁（`_history_lock`），新增 `max_concurrent_requests` 信号量限制
- **最小置信度过滤**：新增 `min_object_confidence` 参数，自动过滤低置信度物体

**实际 VLM 验证通过**（使用 `glm-4.5v` 模型）：

| 物体 | 2D bbox | 3D 位置 (m) | 置信度 |
|---|---|---|---|
| 草莓 | [355, 15, 548, 254] | (0.039, -0.036, 0.196) | 0.85 |
| 白色长方体 | [204, 0, 481, 273] | (0.004, -0.029, 0.165) | 0.85 |
| 黑色小方块 | [179, 373, 266, 443] | (-0.030, 0.046, 0.172) | 0.60 |
| 电线 | [0, 313, 179, 479] | (-0.064, 0.040, 0.162) | 0.60 |

> ⚠️ `glm-4.7` 不支持图像输入，感知节点请使用 `glm-4.5v` 作为视觉模型。

#### 未实现技能清理（死代码移除）

移除了 8 个依赖物理抓取流水线但尚未实现的技能及其全部关联代码（模板、配置、解析器、命令路由）：

| 移除的技能 | 说明 |
|---|---|
| `pick_named_target` | 未对接实际抓取 pipeline |
| `place_named_pose` | 同上 |
| `observe_target_area` | 无实际感知闭环 |
| `approach_named_target` | 无目标定位能力 |
| `hover_named_target` | 同上 |
| `lift_named_target` | 同上 |
| `retreat_from_target` | 同上 |
| `release_at_named_pose` | 同上 |

**涉及变更的模块**：
- `embodied_agent`：删除旧的目标名文本路由和规则 Planner ROS 节点
- `embodied_common/skill_templates.py`：删除 8 个技能的模板定义
- `robot_config/config/robots/so101_single_arm.yaml`：删除 `entry` 路由配置、`named_targets` 定义、被移除技能的 `allowed_skills` 和 `skill_templates`
- `robot_config/robot_config/loader.py`：精简配置校验逻辑，移除 `target_pose_key` / `place_name_from_request` 等间接引用
- `robot_config/robot_config/launch_builders/embodied.py`：移除 entry 参数注入

#### 测试

- 新增 `test_object_parser.py`、`test_grounding_3d_fixture.py`、`test_perception_node_observation.py`、`test_realsense_rgbd_fixture.py`
- 更新 Agent workflow 与 catalog contract 测试适配 Hermes-only 运行时
- **全部 16 项测试通过**，ruff lint 通过，3 个 ROS 包构建成功

---

**维护者**: IB-Robot Team\
**使用指导**: <https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/features/embodied_ai/index.html>\
**项目地址**: <https://atomgit.com/openEuler/IB_Robot>\
**反馈**: <https://atomgit.com/openEuler/IB_Robot/issues>

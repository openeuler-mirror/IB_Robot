# IB-Robot

> IB-Robot (Intelligence Boom Robot)：面向多形态机器人与异构算力的具身智能开发和运行框架。

IB-Robot 贯通遥操作、数据采集、模型训练、边端协同推理、具身技能与机器人控制。系统通过统一的观测／动作契约、机器人能力接口和模型执行抽象组织业务，并通过 Robot Runtime 适配机械臂、移动底盘、人形机器人及厂商控制系统。

**[用户指导文档](https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/features/embodied_ai/index.html)** · **[机器人配置](src/robot_config/README.md)** · **[推理运行时](src/inference_service/README.md)** · **[Agent 技能入口](src/agent/robot_skill_cli/README.md)** · **[开发与测试](docs/testing.md)**

## 项目能力

| 方向 | 核心能力 |
| --- | --- |
| 具身 Agent 与技能 | 将自然语言请求转为经过计划、安全预检和能力网关约束的机器人技能执行 |
| 统一推理与执行优化 | 统一异构算力，基于 `ModelSession` 抽象构建多模型执行与调度子系统，让同一模型通过对应后端适配在多种硬件上执行；支持边端协同推理，以及 AutoHorizon、RTC、时序集成、投机执行等多种执行优化策略 |
| 数据与训练 | 贯通遥操作、契约驱动的数据采集与转换、模型训练和评估流程；支持 Xbox、同构硬件、手机 IMU、动捕设备等多种设备遥操作；支持蒸馏学习、数据质量权重配置等训练优化方法 |
| 感知、操作与空间能力 | 提供场景理解、抓取操作、语音交互、SLAM、语义地图、目标追踪和导航能力 |
| Robot Runtime | 提供统一本体抽象，通过一致的状态、能力与接口契约将硬件与业务功能解耦，便于接入机械臂、移动机器人、人形机器人等不同形态的硬件本体 |
| 中间件与系统互操作 | 通过独立项目 [IBMW](https://atomgit.com/openeuler/IBMW) 提供模块生命周期、执行调度、传输与 RPC 等基础能力，并通过 DDS 与现有机器人系统互操作 |

## 系统架构

![IB-Robot 当前架构：Agent 技能链路、VLA 增强、机器人运行时与具身模型统一执行框架](docs/pictures/ib-robot-architecture.drawio.svg)

IB-Robot 以观测／动作契约、Robot Runtime 和模型执行框架组织机器人业务。图中的 IBMW 是独立构建和部署的机器人与实时系统中间件，不属于本仓库源码包或子模块；ROS 2 和 LeRobot 分别作为现有通信运行环境及模型／数据能力来源按需接入。

架构分层、模块职责、IBMW 集成边界与关键数据流详见 [架构文档](docs/architecture.md#架构说明)。

## 平台与部署

| 平台 | 用途与入口 |
| --- | --- |
| Ubuntu 22.04 / ROS 2 Humble | 开发、构建、数据处理、仿真与 CPU／GPU 推理，使用 `scripts/setup.sh` 和 `scripts/build.sh` |
| openEuler Embedded | 端侧构建、硬件运行与设备适配；依赖对应平台软件源和加速器运行库 |
| OpenHarmony | 通过主机交叉构建、发布包与 HDC／SSH 部署板端运行时，按板端指南准备环境 |

OpenHarmony 的准备和运行流程见 [板端搭建](docs/OpenHarmony_EmbodiedAI_Board_Setup.md) 与 [RKNN 推理部署](docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md)。不同平台的后端依赖、模型产物和硬件能力需单独匹配。

## 快速开始

### 初始化与构建

在 Ubuntu／openEuler 开发工作区使用项目统一环境入口。Python 要求 3.10+；避免在已激活的 Conda 环境中搭建工作区。

```bash
cd /path/to/IB_Robot
source .shrc_local && ./scripts/setup.sh
```

`setup.sh` 负责平台依赖、子模块和 LeRobot patch 栈、Python 环境及验证。新克隆中尚不存在的 venv／install 会在初始化、构建后补齐。

不带参数时构建工作区全部可用包，默认使用 `dev` 配置（Debug、关闭测试构建、symlink-install）：

```bash
source .shrc_local && ./scripts/build.sh
```

日常开发建议按机器人或功能组缩小构建范围，减少编译时间。以下参数追加在 `./scripts/build.sh` 后：

| 参数 | 构建范围与适用场景 |
| --- | --- |
| `--so101` / `--lekiwi` / `--aimdk` | 分别选择 SO-101、LeKiwi、灵犀 X2 的相关包，并自动包含所需的工作区依赖 |
| `--agent` / `--rosclaw` | 选择具身 Agent 技能栈或 RosClaw 集成及其工作区依赖 |
| `--base` | 选择公共基础功能组，涵盖推理、感知、遥操作、数据等模块 |
| `--list-groups --so101` | 仅列出 SO-101 组直接匹配的包，不执行编译；构建时还会补齐工作区依赖 |
| `--clean` | 清理 CMake 缓存后构建，用于排查缓存问题；日常增量构建无需添加 |

功能组参数可叠加，但不能与 `--this` 或显式 `--packages-*` 选择混用。例如：

```bash
# 仅构建 SO-101 运行时及依赖
source .shrc_local && ./scripts/build.sh --so101

# 构建 SO-101 与具身 Agent 技能栈及依赖
source .shrc_local && ./scripts/build.sh --so101 --agent
```

其他配置与参数见 `source .shrc_local && ./scripts/build.sh --help`。

构建完成后，在**新终端**进入同一工作区并重新加载 `source .shrc_local`。该入口统一加载 ROS 2、venv 和工作区 overlay。

## 运行指南

### 运行前：加载环境与分配 ROS Domain ID

每次开启新终端，都先在项目根目录加载环境，并设置 `ROS_DOMAIN_ID`，避免与局域网内其他独立机器人系统冲突。下面以 `42` 为例，请根据实际环境分配：

```bash
cd /path/to/IB_Robot
source .shrc_local
export ROS_DOMAIN_ID=42
```

**同一机器人系统的所有终端，以及跨机器协作的机器人端、推理端和录制客户端，必须使用相同的 `ROS_DOMAIN_ID`；独立系统使用不同的 ID。每次另起新终端都要重新加载环境并设置。**

按用途选择启动入口：

| 用途 | 启动入口 | 说明 |
| --- | --- | --- |
| 独立启动 SO-101 本体 | `so101_robot runtime.launch.py` | 由 runtime profile 管理串口、标定、控制器和传感器，提供状态与运动服务 |
| Hermes／Agent 技能控制 | `embodied_bringup embodied_pipeline.launch.py` | 编排基础机器人、运行时就绪检查与具身技能链路，运动授权由操作员显式提供 |
| 策略推理、遥操作、数据录制 | `robot_config robot.launch.py` | 加载应用 YAML、接入 Robot Runtime 并启动对应业务模块；不直接启动具身 Agent 技能节点 |
| 边端协同的算力侧 | `inference_service cloud_inference.launch.py` | 加载命名 deployment，并与机器人侧 pipeline 通信 |

源码工作区可通过 `dds-config set`（或 `set --tmp`）配置 Fast DDS / CycloneDDS；
`dds-config unset` 可恢复当前终端的原始环境。`robot.launch.py`、`embodied_pipeline.launch.py`、
`teleop_operator.launch.py` 和 `cloud_inference.launch.py` 会在启动节点前检查本机已受管的
DDS 环境。此检查不证明跨机通信或 QoS 匹配。用法与适用范围见
[DDS 配置工具](scripts/dds/README.md)。

SO-101 默认配置使用 Robot Runtime，应用入口的 `use_sim:=true` 对应 SDK 模拟传输。独立运行时使用 `simulated:=true`；两者都不提供 Gazebo 物理场景。Ubuntu／openEuler 源码工作区使用以下入口；OpenHarmony 的构建、部署和启动方式见 [板端指南](README.OpenHarmony.md)。

更详细的子模块说明可参考下表：

| 文档 | 简短说明 |
| :--- | :--- |
| [`src/inference_service/README.md`](src/inference_service/README.md) | 推理服务架构、单机/分布式部署与 NPU/GPU Cloud 节点启动方式 |
| [`src/robots/so101/so101_motion/README.md`](src/robots/so101/so101_motion/README.md) | SO-101 运动服务（motion_server / Placo servo / IK workers）与 headless 启动方式 |
| [`src/dataset_tools/README.md`](src/dataset_tools/README.md) | episodic 录制、`record_cli` 用法与 `bag_to_lerobot` 数据集转换流程 |

### 一、Ubuntu 模拟运行场景

#### 1. 独立验证 SO-101 模拟运行时

只验证本体运行时和控制栈时，无需启动推理、语音或 Agent 业务模块：

```bash
ros2 launch so101_robot runtime.launch.py \
    profile:=so101_single_arm \
    simulated:=true
```

运行时就绪后可读取 `/runtime_status`。与下面的应用入口二选一，不要在同一 ROS domain 中重复启动同一本体。

#### 2. Ubuntu 启动基础应用（不启用推理）

适合验证 SO-101 SDK 模拟传输、控制器和运行时接口，不需要真实机械臂。传感器是否可用取决于运行时 profile 的配置。

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true \
    with_inference:=false \
    with_embodied:=false
```

#### 3. Ubuntu 用模型推理控制模拟机械臂

先准备机器人 YAML 指向的模型 bundle，并核对 deployment 与观测／动作契约，再使用 `model_inference` 模式启动本机推理链路。

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true
```

#### 4. Ubuntu 启动运动规划控制（模拟）

`moveit_planning` 模式通过机器人运行时的运动服务执行规划。需要 RViz 时显式设置 `moveit_display:=true`；无图形界面时使用 `false`。

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true \
    moveit_display:=true
```

运动服务与接口说明见 [SO-101 运动服务](src/robots/so101/so101_motion/README.md)。Agent 的能力发现与受控执行通过 [robot-skill](src/agent/robot_skill_cli/README.md) 进行。

### 二、真机场景

#### 1. 准备本体配置

先确认串口、标定、相机及控制器配置。SO-101 本体参数由 [runtime profile](src/robots/so101/so101_robot/profiles/so101_single_arm.yaml) 管理；[应用 YAML](src/robot_config/config/robots/so101_single_arm.yaml) 通过 `runtime.provider` 和 `runtime.profile` 引用本体，并声明业务模式和接口绑定。

只接入本体、检查运行时接口时，可以独立启动：

```bash
ros2 launch so101_robot runtime.launch.py \
    profile:=so101_single_arm \
    simulated:=false
```

这只启动机器人运行时，不包含 Hermes／Agent 技能链路。运行下面的完整应用前，应先正常停止独立运行时，避免重复占用串口和控制器。

#### 2. 启动 Hermes／Agent 技能控制

真机 Agent 控制使用具身流水线入口，由它启动本体并等待运行时就绪，再接入技能网关、安全预检和计划执行。以下示例启动非视觉技能链路，默认关闭运动授权：

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false \
    control_mode:=moveit_planning \
    entry_mode:=hermes \
    with_embodied:=true \
    with_perception:=false \
    authorize_motion:=false \
    moveit_display:=true
```

操作员完成现场检查、确认可以执行运动后，在启动命令中显式设置 `authorize_motion:=true`。该参数控制具身技能执行授权，不替代本体启动前的现场检查；Agent 不得自行开启授权。非图形环境可设置 `moveit_display:=false`。

在另一个已加载环境、使用相同 Domain ID 的终端检查技能网关：

```bash
robot-skill --config-name so101_single_arm status
```

后续通过 Hermes／`robot-skill` 的计划、校验、确认和执行流程控制机器人。只运行 `robot_config robot.launch.py control_mode:=moveit_planning` 不会启动这套技能链路；真机策略推理和遥操作采集则仍使用下文的 `robot_config` 入口。

完整流程见 [具身启动说明](src/agent/embodied_bringup/README.md)、[robot-skill CLI](src/agent/robot_skill_cli/README.md) 和 [真机验证指南](docs/hermes_so101_real_robot_manual_validation_zh.md)。

### 三、分布式推理部署场景

分布式模式在 robot YAML 的命名 pipeline 中声明 `execution_mode: distributed`。机器人侧启动
Edge pipeline，算力侧单独启动 `cloud_inference.launch.py`。两端必须使用相同的 pipeline ID、
deployment name 和 bundle identity。跨机器还需统一 `RMW_IMPLEMENTATION`，并保证 ROS 2 通信可达。

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
    deployment:=cpu \
    robot_config_path:=/absolute/path/to/so101_single_arm_distributed.yaml
```

#### 2. Ubuntu 启动模拟运行时，端侧开发板启动 NPU 推理

Ubuntu 主机负责模拟运行时、观测接入与动作执行；端侧开发板负责模型预处理、推理和后处理。两台机器必须位于同一局域网，并设置相同的 `ROS_DOMAIN_ID`。

**Ubuntu 主机（模拟运行时 + Edge）**

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
    deployment:=npu \
    robot_config_path:=/absolute/path/to/so101_single_arm_distributed.yaml
```

`cpu`、`npu` 是命名 deployment 的示例，不是自动选择硬件的别名；请替换为模型 manifest 中实际存在、且与 Edge pipeline 一致的名称。`model_path` 和必填的 `robot_config_path` 都是算力侧本机可访问的路径；两端目录可以不同，但模型身份、契约和 pipeline 配置必须匹配。

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

先按 [遥操作说明](src/robot_teleop/README.md) 配置输入设备并确认准入条件满足，再在 `record_cli` 中输入任务描述开始录制；按回车可提前结束当前 episode。托管 leader 输入会由 CLI 在每个 episode 前执行准入和 rearm；手机／VR 仍需满足设备自身的 deadman 条件。

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

录制完成后，使用 `record_cli` 输出的实际 dataset 根目录（替换下面的占位路径）转换为 LeRobot 数据集格式：

```bash
ros2 run dataset_tools bag_to_lerobot \
    --bags-dir /path/to/recorded_dataset \
    --out /path/to/output_dataset
```

bag 目录组织、`dataset.yaml` 元信息和更多转换参数，详见 [数据集工具](src/dataset_tools/README.md)。

***

## 仓库结构

```text
IB_Robot/
├── src/                         # ROS 2 / colcon 包
│   ├── robot_config/            # 应用配置、Contract、启动编排与接口绑定
│   ├── robot_runtime/           # 公共能力/状态契约、facade、mock 与一致性测试
│   ├── ibrobot_msgs/            # 消息、服务和 action 定义
│   ├── tensormsg/               # ROS 消息与 tensor 编解码
│   ├── inference_manifest/      # 模型 bundle / deployment 契约
│   ├── inference_service/       # 统一推理运行时、策略/模型服务与调度
│   ├── observation_transport/   # 图像帧接入与视频传输
│   ├── torch_models/            # Torch 模型实现与适配
│   ├── model_utils/             # 模型导出与转换工具
│   ├── dataset_tools/           # 数据采集、转换、评估与可视化
│   ├── benchmark/               # Benchmark 运行时与 LIBERO 适配
│   ├── agent/                   # Agent 栈（技能目录、Gateway、规划、CLI、自然语言 Agent）
│   │   ├── ibrobot_agent/       # 自然语言 Agent（孵化）
│   │   ├── robot_skill_cli/     # Agent 受控 CLI 与集成接口
│   │   ├── embodied_agent/      # Agent plan 生命周期
│   │   ├── embodied_bringup/    # 具身节点组合与启动
│   │   ├── skill_catalog/       # 技能目录编译与快照
│   │   ├── skill_library/       # 技能执行网关与 primitives
│   │   └── safety_guard/        # 只读安全预检
│   ├── embodied_common/         # 共享契约与工具
│   ├── action_dispatch/         # 策略动作调度与流式执行
│   ├── task_dispatch/           # 任务序列执行
│   ├── manipulation_service/    # 抓取规划与验证
│   ├── manipulation_execution/  # 抓取/放置/模仿执行
│   ├── perception_service/      # 场景理解与感知模型服务
│   ├── semantic_mapping/        # 持久化 3D 语义地图
│   ├── object_tracker/          # RGB-D 目标追踪
│   ├── robot_navigation/        # 导航命令、Nav2 与底盘桥接
│   ├── voice_asr_service/       # 语音识别与声源方向
│   ├── voice_tts_service/       # 语音合成服务
│   ├── robot_teleop/            # 遥操作输入与机器人公开接口桥接
│   ├── robots/                  # 机器人本体适配与 SDK
│   │   ├── so101/               # SO-101 SDK / hardware / motion / runtime
│   │   ├── lekiwi/              # LeKiwi SDK / hardware / description / calibration / runtime
│   │   ├── feetech/             # Feetech 舵机 SDK
│   │   └── aimdk/               # 灵犀 X2 厂商 MC 运行时适配
│   ├── aero_hand_hardware/      # 灵巧手命令与状态桥接
│   ├── hardware_mock/           # 契约驱动的模拟硬件数据
│   ├── sim_models/              # 仿真场景资源与编译
│   └── ibrobot_tracing/         # 埋点与离线性能分析
├── libs/lerobot/                # LeRobot 子模块
├── third_party/patches/lerobot/  # LeRobot 受管 patch 栈
├── models/                      # 模型 bundle；_work/ 存放转换中间产物
├── scripts/                     # 环境、构建、部署与验证入口
├── tools/                       # 可选开发工具
├── docs/                        # 使用指南、架构图源文件与 SVG 导出
└── .agents/skills/              # 仓库开发/运维 Agent 技能
```

树中列出主要模块；外部驱动和其他子模块以 [`.gitmodules`](.gitmodules) 为准。`libs/lerobot` 的改动通过 patch 栈管理，不直接提交子模块工作区修改。

## 演示与外部 Agent 集成

通用agent接入控制，支持仿真和真机控制。

| 仿真演示 | 真实硬件演示 |
| :---: | :---: |
| ![OpenClaw 仿真演示](docs/pictures/openclaw_sim.gif) | ![OpenClaw 真机演示](docs/pictures/openclaw_real.gif) |

已有集成说明见 [社交控制技能](docs/ib_robot_social_skill.md) 与 [OpenHarmony OpenClaw Gateway](docs/OpenHarmony_EmbodiedAI_NodeJS_OpenClaw_Gateway.md)。

## AI Agent Skills

IB-Robot 内置 AI 编程代理技能，帮助 Claude Code、Gemini CLI、OpenCode 等 AI Agent 更好地理解项目架构和开发流程。可用技能详见 [.agents/skills/README.md](.agents/skills/README.md)。

机器人能力发现和受控执行的默认接口是 `robot-skill`，而不是 MCP、裸 `ros2`、primitive、MoveIt 或
controller 命令。自然语言 Workflow 必须先展示并 flush，随后立即进入 Gateway 校验和执行；运行中的 Agent
不得开启 `authorize_motion`。

### AtomGit 认证与目标仓库

Agent 直接使用官方 `gitcode` 处理 PR、Issue 和评审。
仓库根目录的 `config.json` 及 token/owner/repo/baseUrl 读取逻辑已删除。
认证仍由 CLI 自己的配置管理，不删除 CLI 登录配置。
操作时显式指定 `-R owner/repo`（目标仓库不要从个人 fork 的 `origin` 推断）。
模型配置和 DeepWiki 的 `doc_config.json` 不受影响。

**安装与认证**：

1. 将官方 AtomGit CLI 安装到 `/usr/local/bin/gitcode` 并确保该目录在 PATH 上。
   升级时更新同一个可执行文件；保留系统 Graphviz 的 `/usr/bin/gc`。
   具体步骤见[安装说明](docs/atomgit_cli_migration.md#正式安装linux)。
2. 访问 <https://atomgit.com> → 右上角头像 → 个人设置 → 「访问令牌」→ 新建访问令牌
   （勾选 `repo` 和 `pull_request` 权限），**立即复制保存**（只显示一次）。
3. 在私密终端完成认证（凭证存储由 CLI 管理；stdin 输入不代表不会持久保存）：

```bash
gitcode auth login --hostname atomgit.com --with-token
GC_HOST=atomgit.com gitcode auth status   # 验证 AtomGit 认证状态
```

CLI 缺失、非官方或未认证时，链接校验把受影响的 AtomGit 链接明确报告为 inconclusive
（不会误报断链）；inconclusive 本身默认不导致失败，`--fail-on-inconclusive`
可将其视为失败，详见 [doc-link-validator](.agents/skills/doc-link-validator/SKILL.md)。
同一批中的真实断链——包括 `/bolb/` 路由拼写等离线即可判定的错误——仍照常判
broken 并使校验失败，不因 CLI 不可用而被掩盖。
`intro` 为纯本地检查，不访问平台，也不依赖 CLI 认证。

### 支持的 Agent

支持 Agent Skills 的客户端可加载 `.agents/skills/`；具体发现与加载方式以所用客户端为准。
详见 [agentskills.io](https://agentskills.io)。

***

## 开发与贡献

- 遵循 [AGENTS.md](AGENTS.md) 的代码风格、DCO、AI 辅助贡献与提交范围要求。
- Python 修改只对涉及的文件执行 Ruff；不要全量格式化仓库。
- 正式测试验收使用 `colcon test`；裸 `pytest` 仅用于定向排查。测试依赖、ROS domain 隔离和退出行为见 [测试规范](docs/testing.md)。
- worktree 使用其自身的 `.shrc_local`，环境复用按 [worktree 环境指南](.agents/skills/ibrobot-worktree-env/SKILL.md) 操作。
- README 直接使用 SVG 架构图；修改时将 `.drawio` 和 SVG 两个文件同步提交。

**维护者**：IB-Robot Team · **[项目主页](https://atomgit.com/openEuler/IB_Robot)** · **[问题反馈](https://atomgit.com/openEuler/IB_Robot/issues)**

# RoboFrame for OpenHarmony

> 基于 OpenHarmony 的端侧具身智能机器人框架，支持在 OpenHarmony 上完成 CPU / NPU 推理、数据采集、机器人控制等功能。

RoboFrame 融合视觉-动作策略模型（ACT、Diffusion Policy 等 VLA 模型）与 ROS 2 机器人控制链路，在 OpenHarmony 上提供端到端的具身智能能力——涵盖 **CPU / NPU 模型推理**、**多模态数据采集**与**机械臂运动控制**。

## 效果展示：OpenClaw 社交控制

|                            仿真演示 (Simulation)                            |                             真实硬件 (Real Robot)                            |
| :---------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| ![仿真演示](docs/pictures/openclaw_sim.gif) | ![真实硬件](docs/pictures/openclaw_real.gif) |

## 目录

- [快速开始](#快速开始)
- [支持的硬件板卡](#支持的硬件板卡)
- [系统架构](#系统架构)
- [一、获取 OpenHarmony EmbodiedAI 源码与镜像](#一获取-openharmony-embodiedai-源码与镜像)
- [二、板端调试连接：HDC 与 SSH](#二板端调试连接hdc-与-ssh)
- [三、板端安装 ROS 2 Humble 运行时](#三板端安装-ros-2-humble-运行时)
- [四、交叉编译与发布包](#四交叉编译与发布包)
- [五、启动推理与验证](#五启动推理与验证)
- [六、常见问题（FAQ）](#六常见问题faq)
- [七、相关文档与生态](#七相关文档与生态)

---

## 快速开始

已有编译好的发布包？3 步即可在 RoboOH 1.0.1 板卡上运行推理：

```bash
# ① 推送到板端并安装
scp roboframe-robopi-*.tar.gz root@<board-ip>:/data/local/tmp/
ssh root@<board-ip> 'cd /data/local/tmp && tar xzf roboframe-robopi-*.tar.gz && cd roboframe-ohos && sh install.sh'

# ② 加载环境
ssh root@<board-ip>
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ③ 启动推理
ros2 launch hardware_mock hardware_mock.launch.py robot_config:=so101_single_arm &
ros2 launch inference_service eval_inference.launch.py \
    model_path:=/data/models/502000_rknn/pretrained_model \
    deployment:=rknn \
    pipeline_id:=policy \
    robot_config_path:=/data/roboframe/install/robot_config/share/robot_config/config/robots/so101_single_arm.yaml
```

> 没有发布包？按照[第四节](#四交叉编译与发布包)从源码交叉编译并打包。

## 支持的硬件板卡

本框架当前以 Rockchip RK3588（6 TOPS NPU）系列板卡为目标，已验证支持以下三款（均来自 OpenHarmony EmbodiedAI 1.0.1 Release）：

| 板卡 | 芯片 | NPU | 内存/存储 | 典型用途 |
| --- | --- | --- | --- | --- |
| **贝启 BQ3588HM** | RK3588 | 6 TOPS | 8GB + 64GB | 具身智能机器人主控、端侧推理 |
| **曦胧 RoboPi** | RK3588 | 6 TOPS | 8GB + 64GB | 4×千兆网口（2 路 EtherCAT），工业级机器人控制 |
| **贝启 Robo3588** | RK3588 + RK1828 | 6 TOPS | 8GB + 64GB | 6×CAN、4×GMSL2 车载相机，人形 / AMR |

详细规格与固件下载见 [OpenHarmony EmbodiedAI 1.0.1 Release](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md)。

## 系统架构

![RoboFrame 架构图](docs/pictures/architecture.png)

RoboFrame 构建从感知、决策到执行的端到端闭环：

1. **感知**：ROS 2 Driver 接入多路相机 / 雷达 / 麦克风；支持 VR 手柄、Xbox 控制器遥操作采集
2. **协议转换 (tensormsg)**：`ros_msg` ↔ `tensor` 双向转换，合约机制保证类型安全
3. **推理 (inference_service)**：通过 bundle 内唯一的 `inference_manifest.json` 和命名 deployment 选择 Torch、RKNN 等后端
4. **动作执行 (action_dispatch)**：Action Chunking 调度 / MoveIt 2 轨迹执行，统一 `RobotStatus` 汇报
5. **配置中心 (robot_config)**：单一 YAML 驱动关节、控制器、传感器外参，一键切换仿真 / 实机

### 部署拓扑

```text
┌─────────────────────────────────────────────────────────┐
│  Host (Ubuntu 22.04 x86_64)                             │
│                                                         │
│  Docker: voxelsky/ohos-ros-humble-builder:v0.1.5       │
│    build_roboframe_oh.sh → 13 个 aarch64/musl 包   │
│  pack_roboframe_release.sh → roboframe-robopi-*.tar.gz  │
│                                                         │
│  ONNX ── rknn-toolkit2 ──► *.rknn (float16, NPU)        │
└───────────────────────────┬─────────────────────────────┘
                            │ scp / hdc file send
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Board (RK3588, RoboOH 1.0.1, musl)                     │
│                                                         │
│  /data/roboframe/                                       │
│    ├── install/     ← 13 个 RoboFrame ROS 包 + lerobot  │
│    ├── pysite/      ← Python 依赖 (lerobot_deps 发布)    │
│    └── scripts/robooh_1.0.1.env                         │
│                                                         │
│  /sys_prod/robot/out/   ← 系统 ROS + sysdeps (预装)     │
│  /sys_prod/robot/install/ ← ROS 2 Humble 运行时 (预装)  │
│  /vendor/lib64/librknnrt.so ← NPU 驱动                  │
└─────────────────────────────────────────────────────────┘
```

> **核心约束**：OpenHarmony 使用 musl libc + aarch64，所有 C++ 节点和 Python 扩展必须用 OHOS 交叉工具链编译，不能复用 glibc 构建产物。

## 一、获取 OpenHarmony EmbodiedAI 源码与镜像

源码获取、镜像编译与烧录流程见社区讨论：👉 [获取 OpenHarmony EmbodiedAI 源码与镜像](https://gitcode.com/org/openharmony-robot/discussions/4)

## 二、板端调试连接：HDC 与 SSH

### HDC 工具

```bash
echo 'export PATH=<sdk-root>/toolchains:$PATH' >> ~/.bashrc && source ~/.bashrc
hdc list targets   # 验证
```

### TCP 网络调试（推荐）

```bash
hdc tmode port 8710
hdc shell ifconfig                  # 获取板端 IP
hdc tconn <board-ip>:8710
hdc -t <board-ip>:8710 shell
```

### SSH（推荐）

HDC shell 的 PTY 缓冲区有限，长时间 launch 日志会被截断。RoboFrame 提供一键配置脚本：

```bash
hdc -t <board-ip>:8710 file send scripts/setup_sshd.sh /data/setup_sshd.sh
hdc -t <board-ip>:8710 shell 'sh /data/setup_sshd.sh'
hdc -t <board-ip>:8710 shell 'passwd root'  # 设置密码
ssh root@<board-ip>
```

更详细的开发板连接、HDC/TCP 调试和 SSH 配置流程，请参考
[docs/OpenHarmony_EmbodiedAI_Board_Setup.md](docs/OpenHarmony_EmbodiedAI_Board_Setup.md)。

## 三、板端安装 ROS 2 Humble 运行时

RoboOH 1.0.1 固件已预装 ROS 2 Humble 运行时和系统依赖库（sysdeps），通常**无需额外安装**。

如需手动安装，从 [usage.md](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md) 下载：

| 类别 | 文件 | 用途 |
| --- | --- | --- |
| ROS 系统依赖 | `ohos-*-sysdeps-*.tar.gz` | tinyxml2/openssl/libz/python 等 1200+ 库 |
| ROS 2 Humble | `ohos-humble-build-*.tar.gz` | 板端 ROS 2 Humble 运行时 |

```bash
hdc -t <board-ip>:8710 file send ohos-humble-build-aarch64-*.tar.gz /data/
hdc -t <board-ip>:8710 file send ohos-*-sysdeps-aarch64-*.tar.gz /data/
hdc -t <board-ip>:8710 shell 'cd /data && tar -zxpvf ohos-humble-build-*.tar.gz && tar -zxpvf ohos-*-sysdeps-*.tar.gz'
```

## 四、交叉编译与发布包

### 4.1 前提条件

- Ubuntu 22.04 主机，已安装 Docker
- Docker 镜像：`docker pull voxelsky/ohos-ros-humble-builder:v0.1.5`
- 三类官方包（从 [usage.md](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md) 下载）：
  - `ohos-sdk-18-linux-aarch64-*.tar.gz`（OHOS SDK）
  - `ohos-*-sysdeps-*.tar.gz`（系统依赖）
  - `ohos-humble-build-aarch64-*.tar.gz`（ROS 2 运行时）
- Python 依赖（`pysite`）由 [lerobot_deps](https://atomgit.com/openharmony-robot/lerobot_deps) 仓库以 Release 资产形式发布，无需在本地手动构建，请按下面的步骤进入 Release 页面并获取完整依赖包：

  1. 打开 [lerobot_deps 仓库](https://atomgit.com/openharmony-robot/lerobot_deps)，在页面右侧找到“发行版”（Releases）区域，点击最新版本（例如 `RoboFrame 依赖包 v1.0.5-robopi-20260807`），操作位置如下图所示：

     ![在 lerobot_deps 仓库主页进入最新发行版](docs/pictures/lerobot_deps_repository.png)

  2. 进入发行版详情页后，在“下载”区域找到与 RoboPi/OpenHarmony 版本匹配的压缩包，例如 `roboframe-deps-1.0.5-robopi-20260807.tar.gz`，点击该文件下载。

     ![在发行版详情页下载 roboframe-deps 资产](docs/pictures/lerobot_deps_release_asset.png)


### 4.2 主机目录布局

```text
<OH_ROOT>/
├── downloads/
│   ├── sdk/         ohos-sdk-18-linux-aarch64-*.tar.gz
│   ├── sysdeps/     ohos-*-sysdeps-*.tar.gz
│   ├── runtime/     ohos-humble-build-aarch64-*.tar.gz
│   └── deps/        roboframe-deps-*-robopi-*.tar.gz
└── custom_build_root/
    ├── ibrobot_oh_ws/install/   ← 交叉编译产物
    ├── ohos-robot-toolchain/18/ ← OHOS SDK
    └── ...
```

### 4.3 交叉编译 13 个包

```bash
export OH_ROOT="<your-oh-root>"
export DEPS_ARCHIVE="$OH_ROOT/downloads/deps/roboframe-deps-1.0.5-robopi-20260807.tar.gz"
# 请将上面的文件名替换为实际下载的 roboframe-deps 版本

./scripts/openharmony/build_roboframe_oh.sh \
  --oh-root "$OH_ROOT"
```

默认编译以下 19 个包（覆盖推理 + 控制 + 仿真全链路）：

| 类别 | 包 |
| --- | --- |
| 消息 | `ibrobot_msgs` `tensormsg` |
| 配置/契约 | `robot_config` `robot_runtime` `inference_manifest` |
| 推理 | `inference_service` `dataset_tools` |
| 控制 | `action_dispatch` `task_dispatch` |
| SO-101 套件 | `feetech_sdk` `so101_sdk` `so101_hardware` `so101_description` `so101_motion` `so101_suite` `so101_robot` |
| 硬件 | `hardware_mock` |
| 其他 | `embodied_common` `voice_asr_service` |

> 脚本自动处理：SDK 解压、sysdeps overlay（tinyxml2/openssl/libz 等 1200+ 库整体提取到 sysroot）、lerobot patch 应用、wrapper 脚本生成。如需增减包，用 `--packages pkg1,pkg2,...` 覆盖。

### 4.4 打包发布包

```bash
./scripts/pack_roboframe_release.sh \
    --build-install "$OH_ROOT/custom_build_root/ibrobot_oh_ws/install" \
    --deps-archive "$DEPS_ARCHIVE" \
    --output roboframe-robopi-$(date +%Y%m%d).tar.gz
```

发布包结构：

```text
roboframe-ohos/
├── install/              # 13 个 ROS 包 + lerobot(patched) + wrapper 入口
├── scripts/
│   ├── robooh_1.0.1.env  # 统一环境脚本 (PYTHONPATH / LD_PRELOAD)
│   └── setup_sshd.sh     # SSH 服务配置脚本
└── install.sh            # 一键部署
```

### 4.5 部署到板端

```bash
#下载 RoboFrame 发布包（install + scripts）
scp roboframe-robopi-*.tar.gz root@<board-ip>:/data/local/tmp/
ssh root@<board-ip> 'cd /data/local/tmp && tar xzf roboframe-robopi-*.tar.gz && cd roboframe-ohos && sh install.sh'
```

部署完成后板端目录结构：

```text
/data/roboframe/
├── install/     ← 13 个 RoboFrame ROS 包 + lerobot
├── pysite/      ← Python 依赖 (torch/numpy/transformers/tokenizers/regex ~200MB)
└── scripts/
    ├── robooh_1.0.1.env
    └── setup_sshd.sh
```

### 4.6 验证

```bash
ssh root@<board-ip>
. /data/roboframe/scripts/robooh_1.0.1.env

ros2 pkg list | grep -E 'inference_service|hardware_mock|so101_hardware'
python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import transformers; print('transformers', transformers.__version__)"
python3 -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
python3 -c "from rknnlite.api import RKNNLite; print('RKNN OK')"
```

## 五、启动推理与验证

### 5.1 RKNN NPU 推理

```bash
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 启动硬件 mock（发布模拟相机 + 关节状态）
ros2 launch hardware_mock hardware_mock.launch.py robot_config:=so101_single_arm &

# 启动 RKNN 推理
ros2 launch inference_service eval_inference.launch.py \
    model_path:=/data/models/502000_rknn/pretrained_model \
    deployment:=rknn \
    pipeline_id:=policy \
    robot_config_path:=/data/roboframe/install/robot_config/share/robot_config/config/robots/so101_single_arm.yaml
```

`/data/models/502000_rknn/pretrained_model` 必须是完整 policy bundle，至少包含
`config.json`、processor 文件、`inference_manifest.json` 和 manifest 声明的 RKNN artifact。
`rknn` 是该 manifest 中的 deployment 名称；运行时不会扫描目录寻找 `*.rknn`。

### 5.2 CPU 推理

将 deployment 改为 bundle 中的 Torch CPU deployment 名称，例如 `cpu`：

```bash
ros2 launch inference_service eval_inference.launch.py \
    model_path:=/data/models/502000/pretrained_model \
    deployment:=cpu \
    pipeline_id:=policy \
    robot_config_path:=/data/roboframe/install/robot_config/share/robot_config/config/robots/so101_single_arm.yaml
```

### 5.3 触发推理

默认 `policy` pipeline 暴露 `/inference/policy/dispatch` ROS 2 Action，发送 goal 即可触发一次推理：

```bash
ros2 action send_goal /inference/policy/dispatch \
    ibrobot_msgs/action/DispatchInfer \
    "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'test-001', deadline: {sec: 0, nanosec: 0}}"
```

期望输出：

```text
Goal accepted with ID: ...
success: true
inference_latency_ms: ~570 (RKNN) / ~700 (CPU)
Goal finished with status: SUCCEEDED
```

### 5.4 全链路闭环（真实硬件）

真实 SO-101 机械臂闭环需要额外的 USB 相机驱动和内核配置，详见 [RKNN 推理指南](docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md) §5。

### 5.5 性能数据

| 指标 | RKNN (NPU) | CPU |
| --- | --- | --- |
| 推理延迟 | ~470ms | ~80s |
| 端到端（含预处理） | ~570ms | ~80s |
| Action chunk size | 100 | 100 |
| 输出 shape | `(1, 100, 6)` | `(1, 100, 6)` |

## 六、常见问题（FAQ）

| 问题 | 原因 | 解决方法 |
| --- | --- | --- |
| `Calibration file not found: /data/local/tmp/ros_home/...` | launch 环境的 `HOME` 与 SSH 不同 | `ln -sf /root/.calibrate /data/local/tmp/ros_home/.calibrate` |
| `URLError: download.pytorch.org` | 板端无外网，torchvision 下载 ResNet18 权重 | 在有网主机下载 `resnet18-f37072fd.pth`，推送到板端 `/root/.cache/torch/hub/checkpoints/` 和 `/data/local/tmp/ros_home/.cache/torch/hub/checkpoints/` |
| `Can not find dynamic library on RK3588!` | rknnlite 硬编码搜索 `/usr/lib/librknnrt.so` | `ln -sf /vendor/lib64/librknnrt.so /usr/lib/librknnrt.so` |
| `Assertion failed: cast_or_create_topic` | 使用了 `rmw_fastrtps_cpp`（旧版 OH） | `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| `/dev/ttyACM0` 不存在 | 内核缺少 `CONFIG_USB_ACM` | 重新编译内核，见 `oh-rebuild-kernel` skill |
| 推理节点 SIGSEGV | `LD_PRELOAD` 未设置 | 确认 `source robooh_1.0.1.env` 已执行 |
| `ModuleNotFoundError: 'rknnlite.xxx'` | `.so` 后缀不匹配 | 重命名 `-gnu.so` → `-ohos.so` |
| `Deployment 'rknn' is not present` | manifest 中没有同名 deployment | 查看 `inference_manifest.json` 并使用实际 deployment 名称，或重新运行 exporter |
| `Bundle digest mismatch` / artifact load failure | Manifest identity 被手改，或 artifact 损坏/ABI 不兼容 | 重新运行 RKNN exporter 并发布新 revision |

> 更多问题与高级配置见各专项 skill（`oh-rebuild-kernel`、`oh-cross-build-ros-pkg`）与 [RKNN 推理指南](docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md)。

## 七、相关文档与生态

### 详细文档

| 文档 / Skill | 内容 |
| --- | --- |
| [板端烧录与调试](docs/OpenHarmony_EmbodiedAI_Board_Setup.md) | 开发板烧录、HDC 工具准备、TCP 调试、SSH 配置 |
| [RKNN NPU 推理](docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md) | ONNX→RKNN 转换、推理验证、单板全链路闭环 |
| [Node.js + OpenClaw Gateway](docs/OpenHarmony_EmbodiedAI_NodeJS_OpenClaw_Gateway.md) | 板端 Node.js 部署与 OpenClaw 社交控制 |
| [`oh-cross-build-ros-pkg`](.agents/skills/oh-cross-build-ros-pkg/SKILL.md) | 第三方 ROS 2 包交叉编译（含 ros2_control 补丁说明） |
| [`oh-build-roboframe`](.agents/skills/oh-build-roboframe/SKILL.md) | RoboFrame 发布包构建（`build_roboframe_oh.sh`） |

### 官方资源

- [OpenHarmony EmbodiedAI 1.0.1 Release](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md)
- [Docker 交叉编译官方文档](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/docker-build.md)

### OpenHarmony 机器人生态

| 仓库 | 说明 |
| --- | --- |
| [ros_ros2_base](https://gitcode.com/openharmony-robot/ros_ros2_base) | ROS 2 核心基础功能包 |
| [ros_ros2_control](https://gitcode.com/openharmony-robot/ros_ros2_control) | ros2_control 控制框架 |
| [ros_moveit2](https://gitcode.com/openharmony-robot/ros_moveit2) | MoveIt2 运动规划框架 |
| [ros_navigation2](https://gitcode.com/openharmony-robot/ros_navigation2) | Navigation2 自主导航 |
| [ros_peripheral](https://gitcode.com/openharmony-robot/ros_peripheral) | 传感器及外设驱动 |
| [oh_robot_sim](https://gitcode.com/openharmony-robot/oh_robot_sim) | 具身智能模拟器框架 |
| [tools_ohloha](https://gitcode.com/openharmony-robot/tools_ohloha) | ohloha 系统级包管理工具 |
| [tools_ohloha_pkgs](https://gitcode.com/openharmony-robot/tools_ohloha_pkgs) | 80+ 系统依赖库源码级迁移方案 |
| [thirdparty_pytorch](https://gitcode.com/openharmony-robot/thirdparty_pytorch) | 板端 PyTorch / Python runtime |

### 工具与镜像

| 资源 | 地址 |
| --- | --- |
| 交叉编译 Docker 镜像 | `docker pull voxelsky/ohos-ros-humble-builder:v0.1.5` |
| ROS 2 Humble Desktop 镜像 | `docker pull voxelsky/ros-humble-desktop-classic:v0.0.1` |

---

**许可证**：Apache 2.0

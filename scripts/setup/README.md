# `scripts/setup/` — 工作区初始化模块

本目录承载 `scripts/setup.sh` 加载的模块化组件：

| 文件 | 职责 |
| --- | --- |
| `rosdep.sh` | 包含虚拟环境的路径解析、早期建立以及 `rosdepc` 初始化。 |
| `submodules.sh` | Git submodule 的状态探测和自动同步。 |
| `python_venv.sh` | 完整建立和配置最终的 Python 环境以及包安装，包括 colcon、NumPy 的版本锁定，以及 MuJoCo 仿真版本钉（`mujoco_sim_pin_mode`，见下）。 |
| `benchmark_profile.sh` | `--with-benchmark` 专属的 Ubuntu GPU 前置检查、PyTorch CUDA profile、Torch/TorchVision/TorchCodec 与 provider ABI 约束；要求 NVIDIA driver >=560.28.03，不依赖本机 nvcc，并隔离 GraspGen；生成的 benchmark 约束包含 `mujoco<3.9`（hf-libero 0.1.4 的上限）。 |
| `common.sh` | 通用工具函数，例如日志格式、`run_cmd` 与 `ask_yn`、`sudo` 会话等基础设施提取。 |
| `detect.sh` | OS / 架构 / Python 探测；解析 `IBR_HOST_*` 与 `IBR_LEROBOT_PROFILES`。 |
| `verify_env.sh` | 独立环境验证库；暴露 `verify_ros` / `verify_colcon` / `verify_numpy_compat` / `verify_lerobot` / `verify_mujoco_sim` / `verify_tracing` 及入口 `verify_env`。 |
| `platforms/<id>.sh` | 各平台的包管理器、ROS 源、`platform_lerobot_profiles()` 默认值。 |
| `lerobot_patches.sh` | 驱动 `git am` 将精选的 lerobot 补丁栈应用到 `libs/lerobot`。 |
| `lerobot_resolve_active.py` | 从 `INDEX.yaml` 解析当前生效的 lerobot tag；并交叉校验对应 `manifest.yaml`。 |
| `lerobot_filter_series.py` | 读取 `manifest.yaml` + 主机事实，输出真正适用的补丁序列。 |
| `tests/test_lerobot_filter.sh` | 锁定 3 平台基准矩阵 + tag 绑定用例的回归 fixture。 |
| `tests/test_cann_torch_versions.sh` | 无需安装 CANN，验证 CANN 版本探测与 openEuler Torch ABI 选择。 |
| `tests/test_openeuler_cv_bridge_abi.sh` | 验证 openEuler 安装匹配 OpenCV 4.13 的 `cv_bridge` RPM。 |
| `tests/test_setup_profile_training.sh` | 验证 `--profile training` 不触发 ROS 2 / rosdep / colcon，且 `full` / `inference` 行为不变。 |
| `tests/test_mujoco_sim_pin.sh` | 验证 MuJoCo 仿真钉的三种安装顺序、benchmark `mujoco<3.9` 约束、inference 跳过与未构建工作区的 stdlib 校验（issue #190）。 |

> 英文原版见 [`README.en.md`](./README.en.md)。

## Setup Profile（依赖范围参数化）

`setup.sh` 通过 `--profile <name>`（或环境变量 `IBR_SETUP_PROFILE`）控制依赖安装范围：

| Profile | 用途 |
| --- | --- |
| `full`（默认） | 完整开发工作区：硬件、遥操作、感知、抓取、语音、仿真与开发工具全量安装。 |
| `inference` | 仅运行策略推理的最小运行时，面向端边协同推理验证主机。 |
| `training` | **仅支持 Ubuntu 22.04** 的 LeRobot 训练主机：只建 venv、应用 LeRobot 补丁栈并安装 LeRobot + 训练依赖，不装 ROS 2 / rosdep / colcon。 |

能力矩阵的**单一事实源**是 [`common.sh`](./common.sh) 里的 `_SETUP_PROFILE_TABLE`：一行一个 profile，单元格以 `|` 分隔（列 = `rosdep` 单元格 / `submodules` 范围 / `verify` 校验档 / `features` 能力清单），长单元格可用缩进续行 —— 物理行先粘合再切格、格内空白归一化，所以任一单元格都能在逗号后折行。`rosdep` 单元格写作 `<scope>:<paths>`（如 `full:src`、`narrow:src/a,src/b`、`skip:-`），scope 与 `--from-paths` 同格声明，因此不会出现“skip 却仍在解析某个路径集”的非法组合。`setup_profile_enables <feature>` 查“装什么”，`setup_profile_get rosdep|rosdep_paths|submodules|verify` 查范围、路径与校验档；**新增一个 profile = 新增一行**。

`inference` profile 相对 `full` 的差异：

- **submodules**：只初始化 `libs/lerobot`（跳过 LiDAR / bringup 子模块与 ROS 第三方 patch 栈）。
- **rosdep**：范围收窄到 `ibrobot_msgs`、`tensormsg`、`inference_manifest`、`torch_models`、`inference_service`、`model_utils`、`dataset_tools`；`robot_config` 仍需构建（SSOT YAML）但不参与 rosdep 解析，避免拉入 nav2 / slam_toolbox / gz / 相机 / 雷达等 bringup 依赖。
- **Python venv**：跳过 `hardware.txt`、WebPhone、`dev-tools.txt`、voice-tts、perception（SAM2 / Grounding-DINO / RAM++ / SigLIP2 与审计 wheel）、FullSubNet、GraspGen、GraspGenX、pre-commit / gitlint 钩子；Ubuntu 侧只安装 `requirements/inference.txt`（ONNX 工具链），openEuler 侧仍安装完整的 OS 依赖，与 CANN 匹配的 `torch_npu` 由加速器轴安装；lerobot editable 安装去掉仅服务于遥操作的 `kinematics` extra。
- **CANN Torch ABI**：LeRobot 按自身声明正常安装（含所选 extras），随后由 `accel-cann-8.1.txt`（已真机验证的 2.5.1 三元组）或 `accel-cann-8.3.txt`（官方配套的 2.8.0 三元组）按检测到的 CANN 版本覆盖，再由对应 `constraints-cann-<release>.txt` 经 `PIP_CONSTRAINT` 保护其后每一步 pip。`openeuler-24.03.txt` 只负责 OS 包。
- **OpenCV / cv_bridge ABI**：openEuler aarch64 同时安装 `opencv>=4.13.0` 与 `ros-humble-cv-bridge>=3.2.1-2.oe2403`；后者是针对 OpenCV 4.13 重新构建的 RPM。环境校验在工作区 Python 中执行 BGR/RGB 转换、非连续数组及 mono8/mono16/32FC1 往返，不要求将 NumPy 1.x 兼容的 `cv2` wheel 升到 4.13，也不拒绝后续兼容的 RPM 修订。
- **验证**：跳过 tracing 校验（lttng / tracetools 属于全工作区诊断能力），并跳过 Ubuntu 平台的 tracing 工具安装钩子（`platform_post_install_rosdeps`）。

```bash
# 端边协同推理验证主机：
./scripts/setup.sh --yes --profile inference
./scripts/build.sh --packages-up-to inference_service
```

`training` profile 相对 `full` 的差异（面向只跑 LeRobot 训练的 Ubuntu 22.04 训练机）：

`training` 仅支持 `ubuntu-22.04`。平台识别后，其他平台会在依赖预览、子模块同步和系统安装前报错退出；`--only-patch` / `--skip-verify` 不绕过此限制。`full` / `inference` 的平台支持范围不变。`training` 与 `--with-benchmark` 不兼容，benchmark 环境请使用 `full`。

- **ROS 2**：完全不安装。跳过 `scripts/install_ros.sh`、rosdep、colcon 与 ROS 第三方 patch 栈，也不要求 `install/` overlay。
- **submodules**：只初始化 `libs/lerobot`。
- **系统依赖**：只需要 Python bootstrap（`python3-venv` / `python3-pip`）与 `ffmpeg`（torchcodec 视频解码）。与其他 profile 共用 Ubuntu 的系统依赖安装入口，由 apt 幂等处理已安装的包；FFmpeg 不再依赖 ROS/OpenCV 间接带入。
- **Python venv**：保留 `--system-site-packages`（与 `full` / `inference` 布局一致）；只安装 `base.txt`、`requirements/training.txt`（TensorBoard）与 LeRobot editable（extras：`smolvla,pi,diffusion,dataset,deepdiff-dep`）。跳过 `hardware.txt`、WebPhone、`dev-tools.txt`、voice-tts、perception、FullSubNet、GraspGen、colcon、`empy` / `rosdep`、pre-commit / gitlint 钩子以及仅服务遥操作的 `kinematics` extra。
- **NumPy / OpenCV**：**不做** ROS 2 ABI 钉版（`numpy==1.26.4` / `opencv<4.12`），沿用 LeRobot 解析出的版本；CANN 侧的依赖落位改由**加速器轴**决定（见下文[「加速器轴依赖落位」](#加速器轴依赖落位)）。
- **验证**：跳过 ROS 2 / rosdep / colcon / cv_bridge / pygraphviz / tracing 校验，校验 venv + `import lerobot` + `lerobot.policies.act` API + `lerobot-train` 可执行文件；安装了 torchcodec 时还会加载其解码模块，以发现缺失的 FFmpeg 动态库或 ABI 不匹配。
- **`.shrc_local`**：读取 `venv/.ibrobot_setup_profile`；`training` 时欢迎信息标注 `(profile: training)`，把 ROS 2 / `install/` / CANN 的缺失显示为「按预期缺失」，并不再提示 `cb` / `cbp` / `src` 等 colcon 别名。

> 命名提醒：此处的 `training` 是**安装范围** profile；它与 LeRobot 补丁过滤的 `training` profile（`IBR_LEROBOT_PROFILES` / `--lerobot-profiles`）属于不同命名空间。`--profile training` 不会改动补丁过滤集合，训练相关补丁照常应用。

```bash
# 纯训练机（无 ROS 2）：
./scripts/setup.sh --yes --profile training
source .shrc_local && lerobot-train --dataset.repo_id=<repo_id> --policy.type=act
```

## MuJoCo 仿真版本规则（issue #190）

MuJoCo 仿真平台（SO-101、LeKiwi lidar 阶段一）运行在
`mujoco_ros2_control` 0.1.2 / `mujoco_vendor` 0.1.1（libmujoco 3.12.0）
以及 `mujoco_3d_lidar` 0.1.2（MuJoCo 雷达引擎扩展，LeKiwi 的 MID-360
仿真需要它）上。模型由工作区 Python `mujoco` 编译，且版本必须与仿真器的
libmujoco 完全一致，因此：

- Python 侧钉版移入独立的 `requirements/mujoco-sim.txt`（`mujoco==3.12.0`），
  只在 Ubuntu `full` profile、未启用 benchmark 且 venv 中没有
  `hf-libero`/`robosuite` 时安装（`python_venv.sh::mujoco_sim_pin_mode`）。
- LIBERO benchmark 与 MuJoCo ROS 仿真**互斥于同一个 venv**：hf-libero 0.1.4
  声明 `mujoco<3.9`（robosuite 1.4 在 MuJoCo >= 3.10 上崩溃），因此
  `--with-benchmark` 生成的 pip 约束包含 `mujoco<3.9`；已有 benchmark 安装的
  venv 跳过 3.12.0 钉版并警告，要使用 MuJoCo 仿真需**重建 venv**。
- `verify_env.sh::verify_mujoco_sim` 在钉版会安装的环境里用纯标准库脚本复核
  同一组版本，并检查 `mujoco_3d_lidar` 的 ament `mujoco_plugins` 资源、
  包版本（>= 0.1.2）与其指向的扩展库（ROS prefix 取自
  `SETUP_ROS_SETUP_PATH`）；其余 profile 与 benchmark venv 只输出提示行。
  包缺失时提示用 sudo 重新运行 `./scripts/setup.sh`，由其中的 rosdep 安装；
  包已安装但版本过旧或已损坏时提示 apt 命令，因为 rosdep 不会升级已安装的包。
- 运行时还有一道严格检查（`robot_runtime.mujoco.check_simulator_versions`）：
  版本不一致或仿真器过旧时，`sim_platform:=mujoco` 在模型组装前直接失败并
  给出安装诊断。
- openEuler 无 MuJoCo rpm：rosdep skip 键在 `mujoco_ros2_control` /
  `mujoco_ros2_control_msgs` 之外新增 `mujoco_3d_lidar`（雷达引擎扩展同样
  没有 openEuler rpm）。`imu_sensor_broadcaster` 属于 ros2_controllers 家族
  （与 joint_state_broadcaster 同源），未加入 skip 键。

## 加速器轴依赖落位

`SETUP_ACCEL_ID`（`cuda` / `cann` / `cpu`）由 `detect.sh::detect_accel_id` 探测**设备**得出
（`nvidia-smi -L` / `/dev/davinci[0-9]*`，不看工具包），实现在 `scripts/setup/accels/<id>.sh`。
下表是各加速器**当前实际**的依赖来源，不是理想形态：

| | `cuda` | `cann` | `cpu` |
| --- | --- | --- | --- |
| `accel_requirements` | 无（PointNet2 预检确认可安装审计 wheel 或编译本地 wheel 时，`install_graspgen_torch_abi` 才把 Torch 固定到 `2.7.1+cu126`；预检跳过时保留 LeRobot 解析的 Torch） | `accel-cann-8.1.txt` / `accel-cann-8.3.txt`（按检测到的版本）；其他已探到的版本**不安装**、setup 继续，报官方配套版本并在结束摘要提示手工配置；未探到 toolkit 则同样不装 | 无 |
| `accel_pip_constraints` | 无 | `constraints-cann-8.1.txt` / `constraints-cann-8.3.txt`（按版本） | 无 |
| `accel_post_requirements` | 无 | 8.1 / 8.3 卸载 `torchcodec`（aarch64 wheel 的 ABI 绑定在更新的 Torch 上，但其元数据不声明 torch 依赖，pip 察觉不到） | 无 |
| `accel_supports_torch_grasp` | 是 | 否（`pointnet2_ops` 无昇腾构建） | 否 |
| `accel_supports_torch_graspgenx` | 是 | 是（纯 Python wheel，编码器为 PTV3，不碰 `pointnet2_ops`；310P 部署跑在 `torch_npu` 上，扩散循环为离线 OM 图） | 否（无后端可服务） |
| `accel_verify_runtime` | 无 | `verify_cann_runtime_versions`（8.1 / 8.3，按版本） | 无 |

已知的不对称，列在这里以免被误读成违规：

- **CUDA 侧没有自己的 requirements 文件**。PointNet2 预检会先检查 Python、架构和所有可见
  GPU；只有确认将安装审计 wheel 或编译本地 wheel 时，才由 `install_graspgen_torch_abi`
  提前固定 Torch/TorchVision/TorchCodec 并生成局部约束。Blackwell、非 x86_64、非 Python 3.10
  或缺少本地构建条件时跳过 GraspGen，并保留 LeRobot 解析的 Torch。
- **`requirements/manipulation.txt` 仍含 `spconv-cu120`、`torch-scatter` 等 CUDA 包**。该文件
  只由 `install_graspgen_pip` 安装，而那一步受 `accel_supports_torch_grasp` 把关，因此实际
  只在 `cuda` 上执行；但它位于 OS 无关的文件里，尚未迁入 `accel-cuda*.txt`。
- **`requirements/graspgenx.txt` 与 `manipulation.txt` 重复声明 `h5py`、`trimesh`、`scikit-learn`
  等**。这不是冗余：GraspGenX 的 wheel 以 `--no-deps` 安装，而 `manipulation.txt` 只由
  `install_graspgen_pip` 在 `cuda` 上安装，所以 `cann` 上没有任何东西提供
  `graspgenx.grasp_server` 在模块级 import 的那些包。该文件因此必须对
  `accel_supports_torch_graspgenx` 答"是"的每个加速器自足，由
  `scripts/setup/tests/test_graspgenx_requirements.sh` 比对 wheel 的模块级 import 闭包守住。
- **CANN 8.1 与 8.3 自动安装，其余版本留给操作者**。8.1 的三元组经 Ascend310P 真机验证；
  8.3 的三元组取自昇腾官方配套表（`torch_npu` 2.8.0 恒要求 torch 2.8.0）并核实了 cp310/cp311
  的 aarch64 wheel 可用性，但尚未上真机。其他 CANN 版本一律不装 NPU 后端且 setup 继续
  （保留 LeRobot 已装的普通 Torch，不中止半装的工作区）：一个 pin 不可能通吃多个版本
  （树内曾用的 2.10.0.post4 其实是 9.1.0 的配套）。此时 setup 结束摘要会提示"NPU 运行期待
  手工配置"，指向[昇腾版本配套查询助手](https://www.hiascend.com/developer/download/compatibility)
  核对 CANN / Torch / torch_npu / torchvision 配套版本，再用工作区 `venv/bin/python -m pip`
  安装（不要用系统 pip）。新增自动安装的版本需补齐三元组、约束文件和末尾校验。
- **CANN 约束的启用时机在 LeRobot 之后**。CANN 8.1 的 Torch 2.5.1 低于 LeRobot 声明的 `>=2.7`，
  提前启用会让 LeRobot 自身解析直接失败；因此顺序是"LeRobot 正常解析 → 覆盖 Torch →
  启用约束保护其后所有步骤"。8.3 的 2.8.0 在声明区间内，无需例外，但沿用同一顺序。
  8.1 的覆盖是一个**已验证的兼容例外**，包元数据检查会报冲突。CUDA GraspGen 的约束不同：
  预检允许安装时在 LeRobot 前启用，以防后续解析替换 PointNet2 所需 ABI。

## CANN 依赖文件分工

| 文件（`requirements/` 下） | 使用方式 |
| --- | --- |
| `accel-cann-8.1.txt` | 加速器轴：CANN 8.1 的 Torch 2.5.1/Torch-NPU 2.5.1/TorchVision 0.20.1（真机验证），任何 OS 上检测到昇腾即安装。 |
| `constraints-cann-8.1.txt` | 版本约束，不是功能安装清单；通过函数局部的 `PIP_CONSTRAINT` 保护每次 pip 安装及子进程，安装结束后核验全部受保护版本。 |
| `accel-cann-8.3.txt` | 加速器轴：CANN 8.3 的 Torch 2.8.0/Torch-NPU 2.8.0/TorchVision 0.23.0（官方配套 + aarch64 wheel 核实，未上真机），任何 OS 上检测到昇腾即安装。 |
| `constraints-cann-8.3.txt` | 同 8.1 约束文件的职责，按 8.3 的三元组。 |

两个 profile 都先正常安装 LeRobot（含所选 extras），再由 `accel-cann-<release>.txt` 覆盖为
该 CANN 版本的配套版本，随后约束才生效并保护其后每一次 pip 安装。Torch ABI、Transformers、
tokenizers、safetensors、Pillow 和 OpenCV wheel 的版本约束与 profile 无关。NumPy 沿用仓库
既有的最终 ROS ABI 恢复步骤，安装结束后固定并核验为 1.26.4；full 的 Rerun 元数据要求
NumPy 2，因此不将 NumPy 加入全过程约束。Transformers 5.3.0 是当前冻结精度的兼容选择，
本轮不解决其已知安全公告；模型升级和其他模型兼容性验证另行安排。

## LeRobot 补丁分发

`libs/lerobot` 在 `third_party/patches/lerobot/<tag>/` 下为每一个支持的上游 tag 提供一份补丁序列。决定哪个 tag 当前生效的**单一事实源**是 `third_party/patches/lerobot/INDEX.yaml`（见下文 [多 tag 布局](#多-tag-布局)）。在生效的 tag 内，**并非每个补丁都适用于所有主机**。应用器通过 `lerobot_filter_series.py` 过滤原始 `series.txt`，按 `manifest.yaml` 中独立声明的谓词为每个补丁打门：

- `python_min` / `python_max` — 上下界区间（`>=3.10`、`<3.11` 等）。
- `profiles` — 与当前 profile 集合做交集。

当前 profile 集合按以下优先级解析（高优先在前）：

1. `setup.sh` 的 `--lerobot-profiles core,ascend,...` CLI 参数。
2. 环境变量 `IBR_LEROBOT_PROFILES=core,ascend,...`。
3. 当前平台脚本的 `platform_lerobot_profiles` 回调。
4. 兜底默认 `core,ros,hardware,dev`。

### 硬件上电常用覆盖

```bash
# 在桌面端验证推理一致性时，强制启用 Ascend NPU 补丁：
./scripts/setup.sh --yes --lerobot-profiles core,ros,hardware,ascend

# 把 OpenHarmony 设备当作 vanilla core 目标拉起：
./scripts/setup.sh --yes --lerobot-profiles core

# 显式跳过过滤（兜底逃生口——直接照搬 series.txt，
# 即便补丁与主机事实不匹配也会被应用）：
IBR_LEROBOT_FORCE_UNFILTERED=1 ./scripts/setup.sh --yes
```

### 诊断过滤器

应用器会打印 `[CTX] python=X.Y profiles=...` 审计行，以及每个补丁的 `KEEP` / `SKIP` 决策。要在不动 `libs/lerobot` 的前提下预览：

```bash
IBR_HOST_PYTHON_VERSION=3.11 \
IBR_LEROBOT_PROFILES=core,ros,hardware,openeuler \
  python3 scripts/setup/lerobot_filter_series.py \
    --manifest third_party/patches/lerobot/v0.6.0/manifest.yaml \
    --series   third_party/patches/lerobot/v0.6.0/series.txt
```

### 何时新增一个补丁

1. 把补丁文件放到当前生效 tag 的目录下（`third_party/patches/lerobot/<active_tag>/`；`<active_tag>` 由 `INDEX.yaml.active_tag` 当前指向决定）。
2. 追加到该 tag 的 `series.txt`。
3. 在同目录的 `manifest.yaml` 中注册一条 `patches[]` 条目，**显式**声明 `python_min` / `python_max` / `profiles` 谓词。先按已知最稳的窄集合声明；在更多平台验证后再放宽。
4. 给 `scripts/setup/tests/test_lerobot_filter.sh` 加一条 fixture，让 pre-commit 钩子能挡住意外扩散范围。

## 多 tag 布局

`third_party/patches/lerobot/INDEX.yaml` 是 setup 脚本目标 lerobot tag 的单一事实源。每个 tag 拥有独立的补丁目录 + 一份 `manifest.yaml`，重新声明绑定，让解析器能在漂移时 fail-closed：

```text
third_party/patches/lerobot/
├── INDEX.yaml                  # active_tag + supported_tags + archived_tags
├── v0.6.0/                     # 当前 active tag，每个支持的 tag 一个目录
│   ├── manifest.yaml           # 声明 lerobot_tag + lerobot_commit_range
│   ├── series.txt
│   └── 0001-*.patch ...
└── v0.6.0/                     # （示例：未来 tag 并存添加）
    └── ...
```

解析器（`lerobot_resolve_active.py`）强制以下不变量：

| 不变量 | 失败模式 |
| --- | --- |
| `INDEX.active_tag` 必须命中某个 `supported_tags[]` 条目 | 解析器退出 1，提示 "active_tag not found" |
| 生效条目的 `dir` 必须存在且包含 `manifest.yaml` + `series.txt` | 解析器退出 1，提示 "directory does not exist" |
| `INDEX.supported_tags[i].upstream_commit` 必须等于 `manifest.lerobot_commit_range.min` | 解析器退出 1，提示 "INDEX vs manifest mismatch" |
| `INDEX.active_tag` **不得**出现在 `archived_tags[]` 中 | 解析器退出 1，提示 "archived tag" |
| 应用补丁前 `libs/lerobot` HEAD 必须落在 `manifest.lerobot_commit_range` 内 | 过滤器退出 1，提示 "HEAD not in commit_range" |

### 升级到新 lerobot tag

1. 将 `libs/lerobot` 子模块升到新的上游 tip，记下其 sha 为 `<NEW_COMMIT>`。
2. 创建 `third_party/patches/lerobot/<NEW_TAG>/`，把需要的补丁基于 `<NEW_COMMIT>` rebase 一遍（不再适用的 drop / port）。撰写一份新的 `manifest.yaml`：
   ```yaml
   lerobot_tag: <NEW_TAG>
   lerobot_commit_range:
     min: <NEW_COMMIT>
     max: <NEW_COMMIT>     # 当你接受上游 fast-forward 时再放宽
   ```
3. 在 `INDEX.yaml` 的 `supported_tags[]` 追加一条指向新目录的条目，`upstream_commit` 一致，`branch_name` 唯一（约定：`ibrobot/lerobot-<NEW_TAG>-patched`）。
4. 把 `INDEX.yaml.active_tag` 翻到 `<NEW_TAG>`。可选：若旧 tag 不再维护，把它的条目从 `supported_tags[]` 移到 `archived_tags[]`——归档项保留在树内便于审计，但解析器拒绝使用。
5. 在 `scripts/setup/tests/test_lerobot_filter.sh` 加上新 tag 的 fixture 覆盖，并跑一遍回归套件；解析器与各平台过滤器用例都必须通过后才能提交。

### 失败模式

| 现象 | 原因 | 修复 |
| --- | --- | --- |
| `lerobot_filter_series.py failed with exit 1`，"PyYAML required but missing" | venv 缺 `python3-yaml` | 在工作区 venv 内 `pip install pyyaml`，或设 `IBR_LEROBOT_FORCE_UNFILTERED=1` 跳过过滤。 |
| `failed to parse manifest`，退出 1 | `manifest.yaml` 格式错误 | lint YAML；应用器按设计 fail-closed。 |
| `lerobot_resolve_active.py failed`，"active_tag not found" | `INDEX.yaml.active_tag` 与任何 `supported_tags[]` 条目都不匹配 | 添加条目，或修正 `active_tag` 值。 |
| `lerobot_resolve_active.py failed`，"INDEX vs manifest mismatch" | `INDEX.supported_tags[i].upstream_commit` 与 `manifest.lerobot_commit_range.min` 不一致 | 修改任一方使其同步；二者**必须**保持一致。 |
| `libs/lerobot HEAD ... is not in the manifest commit_range` | 子模块 HEAD 已偏离当前 tag 声明的范围 | 要么 checkout 到期望 base，要么放宽 manifest range，要么新增一个 tag 目录并翻转 `INDEX.yaml.active_tag`。 |
| 工作树脏，应用器中止 | `libs/lerobot` 内有本地编辑 | stash/commit 这些编辑，或设 `IBR_LEROBOT_FORCE_REBUILD=1` 丢弃并从头重建打过补丁的分支。 |

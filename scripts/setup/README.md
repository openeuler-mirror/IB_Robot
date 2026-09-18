# `scripts/setup/` — 工作区初始化模块

本目录承载 `scripts/setup.sh` 加载的模块化组件：

| 文件 | 职责 |
| --- | --- |
| `rosdep.sh` | 包含虚拟环境的路径解析、早期建立以及 `rosdepc` 初始化。 |
| `submodules.sh` | Git submodule 的状态探测和自动同步。 |
| `python_venv.sh` | 完整建立和配置最终的 Python 环境以及包安装，包括 colcon 及 NumPy 的版本锁定。 |
| `benchmark_profile.sh` | `--with-benchmark` 专属的 Ubuntu GPU 前置检查、PyTorch CUDA profile、Torch/TorchVision/TorchCodec 与 provider ABI 约束；要求 NVIDIA driver >=560.28.03，不依赖本机 nvcc，并隔离 GraspGen。 |
| `common.sh` | 通用工具函数，例如日志格式、`run_cmd` 与 `ask_yn`、`sudo` 会话等基础设施提取。 |
| `detect.sh` | OS / 架构 / Python 探测；解析 `IBR_HOST_*` 与 `IBR_LEROBOT_PROFILES`。 |
| `verify_env.sh` | 独立环境验证库；暴露 `verify_ros` / `verify_colcon` / `verify_numpy_compat` / `verify_lerobot` / `verify_tracing` 及入口 `verify_env`。 |
| `platforms/<id>.sh` | 各平台的包管理器、ROS 源、`platform_lerobot_profiles()` 默认值。 |
| `lerobot_patches.sh` | 驱动 `git am` 将精选的 lerobot 补丁栈应用到 `libs/lerobot`。 |
| `lerobot_resolve_active.py` | 从 `INDEX.yaml` 解析当前生效的 lerobot tag；并交叉校验对应 `manifest.yaml`。 |
| `lerobot_filter_series.py` | 读取 `manifest.yaml` + 主机事实，输出真正适用的补丁序列。 |
| `tests/test_lerobot_filter.sh` | 锁定 3 平台基准矩阵 + tag 绑定用例的回归 fixture。 |
| `tests/test_cann_torch_versions.sh` | 无需安装 CANN，验证 CANN 版本探测与 openEuler Torch ABI 选择。 |
| `tests/test_openeuler_cv_bridge_abi.sh` | 验证 openEuler 安装匹配 OpenCV 4.13 的 `cv_bridge` RPM。 |

> 英文原版见 [`README.en.md`](./README.en.md)。

## Setup Profile（依赖范围参数化）

`setup.sh` 通过 `--profile <name>`（或环境变量 `IBR_SETUP_PROFILE`）控制依赖安装范围：

| Profile | 用途 |
| --- | --- |
| `full`（默认） | 完整开发工作区：硬件、遥操作、感知、抓取、语音、仿真与开发工具全量安装。 |
| `inference` | 仅运行策略推理的最小运行时，面向端边协同推理验证主机。 |

`inference` profile 相对 `full` 的差异：

- **submodules**：只初始化 `libs/lerobot`（跳过 LiDAR / bringup 子模块与 ROS 第三方 patch 栈）。
- **rosdep**：范围收窄到 `ibrobot_msgs`、`tensormsg`、`inference_manifest`、`torch_models`、`inference_service`、`model_utils`、`dataset_tools`；`robot_config` 仍需构建（SSOT YAML）但不参与 rosdep 解析，避免拉入 nav2 / slam_toolbox / gz / 相机 / 雷达等 bringup 依赖。
- **Python venv**：跳过 `hardware.txt`、WebPhone、`dev-tools.txt`、voice-tts、perception（SAM2 / Grounding-DINO / RAM++ / SigLIP2 与审计 wheel）、FullSubNet、GraspGen、pre-commit / gitlint 钩子；Ubuntu 侧只安装 `requirements/inference.txt`（ONNX 工具链），openEuler 侧仍安装完整的平台依赖与 CANN 匹配的 `torch_npu`；lerobot editable 安装去掉仅服务于遥操作的 `kinematics` extra。
- **CANN 8.1 ABI**：`full` 和 `inference` 共用 `lerobot-v0.6-cann-8.1-compat.txt`，包含已验证的 Transformers 5.3.0。`full` 仅追加 `lerobot-v0.6-cann-8.1-extras.txt` 的 dataset/kinematics 功能依赖，不切换推理版本。LeRobot 以 `--no-deps` 安装；`openeuler-24.03-cann-8.1.txt` 负责平台包和 Torch ABI。
- **OpenCV / cv_bridge ABI**：openEuler aarch64 同时安装 `opencv>=4.13.0` 与 `ros-humble-cv-bridge>=3.2.1-2.oe2403`；后者是针对 OpenCV 4.13 重新构建的 RPM。环境校验在工作区 Python 中执行 BGR/RGB 转换、非连续数组及 mono8/mono16/32FC1 往返，不要求将 NumPy 1.x 兼容的 `cv2` wheel 升到 4.13，也不拒绝后续兼容的 RPM 修订。
- **验证**：跳过 tracing 校验（lttng / tracetools 属于全工作区诊断能力），并跳过 Ubuntu 平台的 tracing 工具安装钩子（`platform_post_install_rosdeps`）。

```bash
# 端边协同推理验证主机：
./scripts/setup.sh --yes --profile inference
./scripts/build.sh --packages-up-to inference_service
```

## CANN 8.1 依赖文件分工

| 文件（`requirements/` 下） | 使用方式 |
| --- | --- |
| `lerobot-v0.6-cann-8.1-compat.txt` | 两种 profile 都安装，公共 LeRobot/PI05 兼容运行时。 |
| `lerobot-v0.6-cann-8.1-extras.txt` | full 在 compat 之上追加数据集和运动学依赖。 |
| `openeuler-24.03-cann-8.1.txt` | openEuler+CANN 8.1 的平台包及 Torch 2.5.1/Torch-NPU 2.5.1/TorchVision 0.20.1。 |
| `constraints-cann-8.1.txt` | 版本约束，不是功能安装清单；通过函数局部的 `PIP_CONSTRAINT` 保护每次 pip 安装及子进程，安装结束后核验全部受保护版本。 |

`full = compat + extras + 平台依赖`，`inference = compat + 平台依赖`。
Torch ABI、Transformers、tokenizers、safetensors、Pillow 和 OpenCV wheel 的版本约束与 profile 无关。
外部已有的 pip constraints 会保留并共同生效；这些运行库的冲突明确报错，不先升级后降级。
NumPy 沿用仓库既有的最终 ROS ABI 恢复步骤，安装结束后固定并核验为 1.26.4；full 的 Rerun 元数据要求 NumPy 2，因此不将 NumPy 加入全过程约束。
Transformers 5.3.0 是当前冻结精度的兼容选择，本轮不解决其已知安全公告；模型升级和其他模型兼容性验证另行安排。

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

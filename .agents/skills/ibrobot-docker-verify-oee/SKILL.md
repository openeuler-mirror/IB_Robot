---
name: ibrobot-docker-verify-oee
description: "在 openEuler Embedded (aarch64) Docker 容器中实际执行 setup.sh + build.sh。Use when the user explicitly asks for openEuler/OEE Docker verification, or after an author-side PR gate asks WIP vs review and the user confirms review-ready. Skip author-side [WIP] PRs; do not run automatically during PR review."
---

# IB-Robot openEuler Embedded Docker Verification Skill

在全新 openEuler Embedded aarch64 Docker 容器中完整验证 `setup.sh` 和 `build.sh`。

> **⚠️ 与 Ubuntu 验证的两个核心差异：**
> 1. **以 root 用户操作**（openEuler Embedded 开发板默认 root，无需 sudo/testuser）
> 2. **所有 docker exec 命令必须 `chroot /root/openeuler_rootfs`** 进入 arm64 rootfs 环境才能执行

容器镜像通过 `--privileged` + `chroot /root/openeuler_rootfs` 进入 qemu-user 模拟的 arm64 环境，
模拟真实 openEuler Embedded 开发板的首用体验。

## When to Use

- 用户明确要求 "openEuler Docker 验证" / "oee container test" / "实际验证 openEuler setup/build"。
- 作者侧创建/更新 PR 流程触发依赖/setup 门禁，且用户确认 PR 已准备交给 reviewer 正式检视，需要真实结果写入描述。
- 当前任务是验证本地对 `scripts/setup/platforms/openeuler-embedded-24.03.sh`、`scripts/setup.sh`、`scripts/setup/lerobot_patches.sh` 或 dnf/rosdep 相关逻辑的修改。
- 不要仅因为 PR review 触发本 skill。
- 作者侧门禁调用本 skill 前必须通过交互式 ask-user 工具（opencode 中的 `question` 工具）询问 WIP/正式检视阶段，不得只在文本中询问或替用户默认选择；`[WIP]` PR 暂缓两个 Docker skill，直到用户将其转为正式检视。用户单独明确要求实际 Docker 验证时仍正常执行。

## Review Boundary

- PR review 过程中，禁止因为 PR 修改了 `package.xml`、`setup.py`、setup 脚本或 build 文件就自动运行本 skill。
- 用户要求"review 这个 PR / 检查这个 PR 有没有问题"不等于授权运行 openEuler Docker 验证。
- review 默认只检查 PR 描述中开发者声明的 openEuler Embedded Verification。如果缺少或不完整，应作为阻塞性 review 问题要求开发者补充。
- 只有当用户在当前请求中明确要求 agent 实际执行 openEuler / 双平台 Docker setup/build 验证时，才运行本 skill。

## PR 验证的 Tree 绑定

作者侧 PR 门禁触发两个 Docker skill 时，两平台必须验证同一个已提交代码树：

1. 启动第一个平台前解析目标 commit，并记录完整 40 位
   `VERIFIED_TREE="$(git rev-parse "${VERIFIED_COMMIT}^{tree}")"`；用户当前 worktree 无需干净。
2. 在用户 worktree 外生成目标 commit 的独立快照。openEuler 与 Ubuntu 必须验证 tree SHA
   均等于 `VERIFIED_TREE` 的快照，PR 证据模式不得直接复制 dirty 工作区。
3. 本 skill 的结果中写入 `Verified tree: <full SHA>`；PR 工作流会将其组装成结构化
   `## Docker Verification` 块。
4. 创建或更新 PR 时将该字段与远端 head commit 的 tree 比对。只有 tree 改变才要求重跑；
   单纯修改 commit message、作者或 trailer 不会让结果失效。
5. `docker cp` 当前 dirty 工作区仍可用于本地调试，但结果不得写成 PR 验证证据。

## Prerequisites

- 宿主机已安装 Docker CLI（检查 `command -v docker`）。
- PR 证据快照要求宿主机安装 Git LFS；流程会执行 `git lfs pull` 和 `git lfs fsck`，避免相同
  tree SHA 因 LFS smudge 状态不同而复制出不同字节内容。
- 宿主机已安装 openEuler aarch64 验证所需的 `qemu-user-static`（只检查不自动安装；缺失时提醒用户 `sudo apt install -y qemu-user-static`）。
- 当前用户有运行容器的权限。
- **必须检查验证镜像**：本地不存在或创建超过 30 天则重新拉取 `swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env`。
- 网络：可访问 `repo.openeuler.org`、`eur.openeuler.openatom.cn`、`eulermaker.compass-ci.openeuler.openatom.cn`、华为 pip 镜像、`gitcode.com`。
- IB-Robot workspace 中有待验证的修改。

## Container Architecture

```
┌─ Docker container (x86_64) ──────────────────────┐
│  entrypoint: /bin/bash -l                        │
│  .bashrc auto-chroots on interactive login       │
│  ┌─ chroot /root/openeuler_rootfs (aarch64) ────┐│
│  │  openEuler Embedded Reference Distro         ││
│  │  openEuler ROS repos ( Embedded + SIG )      ││
│  │  ROS 2 Humble, python3, dnf, git, pip cache  ││
│  │  workspace at /root/IB_Robot                 ││
│  └───────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

> **关于 ROS 安装：** `:env` 镜像已预装 ROS 2 Humble，因此本流程验证 setup.sh 的
> ROS 检测和复用路径。若要验证从零安装 ROS，必须另用不含 ROS 的基础镜像，不能把
> 本流程的成功结果描述为完整验证了 `install_ros.sh`。
>
> **ROS 源切换预清理（强制）：** 当验证的变更切换了 ROS 包源（`install_ros.sh` /
> 平台脚本修改了 openEuler ROS repo、EUR 源地址或 GPG key）时，`:env` 镜像中预装的
> `ros-humble-*` 包来自旧源，会被 setup.sh 的 ROS 检测直接复用，新源的安装路径
> 根本没有被执行。必须先按 Phase 4.5 在 chroot 中 `dnf remove 'ros-humble-*'`
> 移除全部 ROS 包后再跑 setup.sh；属于环境预清理（只删除、不安装），不违反
> Core Principle。

所有 `docker exec` 命令需要通过 `chroot /root/openeuler_rootfs` 进入 arm64 环境。

> **注意：** 容器内 `which` 命令在 rootfs 中行为异常（即使工具已安装也报 not found），
> 应使用 `type git` 或直接用绝对路径 `/usr/bin/git` 来检测工具。

## Container Naming Convention

| Variable       | Value                                                    |
|----------------|----------------------------------------------------------|
| Container name | `verify-oee`                                             |
| User           | `root`（openEuler Embedded 默认 root 操作）              |
| Image          | `swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env` |
| Rootfs path    | `/root/openeuler_rootfs`（容器内，chroot 前）            |
| Workspace      | `/root/openeuler_rootfs/root/IB_Robot`（chroot 内路径）  |
| Host workspace | 宿主机上 IB_Robot 项目根目录                              |

## Core Principle

> **setup.sh 和 build.sh 是唯一合法的软件安装途径。**

- 禁止手动 `dnf install` / `pip install` / `sed` patch 脚本 / 手动配置 ROS 仓库。
- 允许修复 chroot 基础设施（DNS、`/var/log`、`git safe.directory`）、通过环境变量传递镜像 URL。
- 允许修复 `:env` 镜像自带的状态不一致基础包（RPM DB 与磁盘内容版本不符），
  用 `dnf reinstall -y --nogpgcheck <pkg>` 把磁盘内容重新对齐到 RPM DB 已声明的状态。
  这属于环境修复，不是"安装新软件"。详见 Base Image Integrity Pre-flight 与
  [references/discipline.md](references/discipline.md)。
- 完整的禁止/允许事项清单和论述见 [references/discipline.md](references/discipline.md)。

## Base Image Integrity Pre-flight

> **`:env` 镜像有时会自带状态不一致的基础包。**

`swr.cn-north-4.myhuaweicloud.com/.../openeuler-ibrobot-dev:env` 是预装的
openEuler Embedded rootfs。当镜像构建过程中基础包升级但 RPM DB 未同步刷新时，
会出现"RPM 数据库记录版本 X，磁盘实际二进制 / soname 为版本 Y"的不一致状态。
这种**原始镜像就已损坏**的状态会让 setup.sh 的 dnf 事务在尝试安装新子包（如
`*-devel`）时失败：RPM 会按数据库记录创建指向"应当存在但磁盘上不存在"的 soname
符号链接，导致后续编译 / 运行时找不到库。

**典型示例（lz4）：**

- RPM DB 标称：`lz4-1.9.4-2.oe2403`
- 磁盘实际：`lz4 1.10.0`，`/usr/lib64/liblz4.so.1 -> liblz4.so.1.10.0`
- `/usr/lib64/liblz4.so.1.9.4` 不存在，`/usr/lib64/liblz4.so` 不存在
- `rpm -V lz4` 直接报告 `liblz4.so.1` 链接错误与 `liblz4.so.1.9.4` 缺失
- setup.sh rosdep 解析 PCL/FLANN 依赖链时需安装 `flann-devel` → `lz4-devel`；
  dnf 事务创建指向不存在的 `liblz4.so.1.9.4` 的 `/usr/lib64/liblz4.so`，导致
  后续链接错误

**处理策略（两道门禁，缺一不可）：**

1. **Pre-flight（Phase 3.5，setup 之前）：** 对已知易损坏包用 `rpm -V` 探测，
   若发现不一致就在 chroot 中用 `dnf reinstall -y --nogpgcheck <pkg> <pkg>-devel`
   重装。`reinstall` 不是"安装新软件"，而是把磁盘内容重新对齐到 RPM DB 已声明
   的状态——属于环境修复，不违反 Core Principle。
2. **Post-setup 门禁（Phase 5.5，build 之前，必须执行）：** 2026-09 实测教训：
   一次验证中 pre-flight 探测完全干净，但 setup.sh 的 dnf 事务（rosdep 安装
   `flann-devel` → `lz4-devel`，并从 repo 拉入旧版 `lz4`）**重新引入**不一致，
   build 在 `livox_ros_driver2` 链接阶段失败，浪费约 1.5 小时 qemu 构建时间。
   因此 setup 完成后必须重跑探测并检查 `/usr/lib64/liblz4.so` 存在，失败则
   reinstall 修复后再启动 build。

**扩展规则：** lz4 是首个已知示例，不是封闭列表。当 setup.sh 或 build.sh 因其他包出现
"RPM DB 标称版本 vs 磁盘实际版本不一致"失败时，把该包加入 Phase 3.5 与
Phase 5.5 的易损坏包列表，并追加对应的 `rpm -V` 检测与 `dnf reinstall` 修复步骤。

## Error Classification

验证完成后，必须收集 `setup.log` 和 `build.log` 中的所有 ERROR 行，并分类报告：

| 分类 | 定义 | 处理方式 |
|------|------|----------|
| **Fatal** | setup.sh 或 build.sh 未输出完成消息（`Setup complete` / `Build complete`）；进程异常退出 | 必须报告为阻塞问题 |
| **Non-fatal** | pip dependency resolver 警告；rosdep keys 未解析；updmap 字体映射缺失；其他不影响最终构建结果的警告 | 列出但标注为 non-fatal，不阻塞 |

报告时必须逐条列出所有 ERROR 行内容，并标注分类。

## Procedure Overview

Each phase's complete bash commands and detailed rationale live in
[references/procedure.md](references/procedure.md). The summary below is for
routing and mental model only; execute the actual commands by reading the
corresponding phase section.

| Phase | Purpose | Key Output |
|-------|---------|------------|
| **0** | Check host prerequisites (docker, qemu-user-static) | Pass/fail |
| **1** | Ensure `:env` image is fresh (pull if >30 days old) | Image ready |
| **2** | Start container, verify aarch64 emulation, fix chroot env (DNS, /var/log, git safe.directory wildcard for all submodules) | Container ready |
| **3** | Inspect chroot environment (git, python3, dnf, ROS 2) | Environment confirmed |
| **3.5** | Verify & repair base image integrity (RPM DB vs on-disk binary mismatch) — see [Base Image Integrity Pre-flight](#base-image-integrity-pre-flight) | Base packages consistent |
| **4** | Prepare workspace — **choose one mode** (docker cp or git clone) | `SOURCE_MODE` |
| **4.5** | ROS source-switch pre-clean (conditional) — remove all `ros-humble-*` in chroot when the change switches the ROS source | No ROS packages left |
| **5** | Run `setup.sh --yes --no-sudo`, capture log (20-40 min under qemu) | `/tmp/setup.log` |
| **5.5** | Post-setup integrity gate (MUST run before build): re-probe fragile pairs, repair via `dnf reinstall` | Gate OK before build |
| **6** | Run `build.sh`, capture log | `/tmp/build.log` |
| **7** | Collect ERROR lines, clean up container | Final report |

### Phase 4 Source Modes

- **PR 证据 — 独立 commit 快照**：在当前 worktree 外生成目标 commit 的 standalone clone，
  校验 tree 后 `docker cp`；作者侧 PR 门禁必须使用此模式。
- **本地调试 — 当前工作区 copy**：可包含未提交改动，但不能作为 PR 证据。
- **显式远端 clone**：仅在用户指定远端 commit/branch 时使用，并记录实际 commit 与 tree。

## Variants

- **ROS source switch**（变更切换 ROS repo / 镜像 / GPG key）:
  **必须**先执行 Phase 4.5 的 ROS 源切换预清理（chroot 内移除全部 `ros-humble-*` 包），
  再运行 setup.sh，见 [references/procedure.md](references/procedure.md)。
- **Iterative local testing** (re-run without full Phase 0-3):
  use the [quick-run one-liner](references/quick-run.md) to copy changed scripts and re-run setup.
  若两次运行之间 ROS 源发生变化，同样必须先重跑 Phase 4.5 预清理。

## Key Differences from Ubuntu Verification

| Aspect | Ubuntu 22.04 (`ibrobot-docker-verify`) | openEuler Embedded (this skill) |
|--------|----------------------------------------|--------------------------------|
| Architecture | x86_64 native | aarch64 via qemu-user chroot |
| **User** | `testuser` + NOPASSWD sudo | **`root`（无需 sudo）** |
| **Command prefix** | `docker exec verify-ubuntu2204` | **`docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "..."'`** |
| Package manager | apt | dnf |
| ROS install | setup.sh 自动调用 `install_ros.sh` 安装 | setup.sh 自动调用 `install_ros.sh` 安装 |
| Speed | Native (fast) | Emulated (20-40 min for setup) |
| DNS | Works by default | Must copy resolv.conf |
| git-lfs | Available | Not available; lerobot_patches.sh auto-removes LFS hook |

## Troubleshooting

Common issues (DNS resolution, `/var/log` symlink, `dubious ownership`, GPGMEError, git-lfs hook,
`which` unreliable under qemu, pip ReadTimeout, etc.) are documented in
[references/pitfalls.md](references/pitfalls.md). Consult that table when a verification run fails
with an unfamiliar error.

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch from base commit |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `ROS_OS_OVERRIDE` | `rhel:8` | Set by platform script for rosdep compatibility |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |

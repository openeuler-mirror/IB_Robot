# Full Procedure — Phase 0 to Phase 7

> This document contains the complete bash commands for each phase.
> The main SKILL.md only lists the phase purposes; execute the actual
> commands by reading the corresponding section here.

## Phase 0 — Check Host Prerequisites

> **只允许在宿主机执行本节检查命令。** 本节只检查 `docker` 命令和宿主机 apt 包，
> 不执行 chroot、不挂载 rootfs、不进入 aarch64 环境。

```bash
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is not installed. Install Docker on the host, then rerun verification."
  exit 1
fi

missing_pkgs=()
for pkg in qemu-user-static; do
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
    missing_pkgs+=("$pkg")
  fi
done

if [ "${#missing_pkgs[@]}" -gt 0 ]; then
  echo "Missing host packages: ${missing_pkgs[*]}"
  echo "Install them on the host with: sudo apt update && sudo apt install -y ${missing_pkgs[*]}"
  exit 1
fi
```

## Phase 1 — Ensure Required Image Is Fresh

> **必须执行。** openEuler Embedded 验证不使用本地默认镜像名，也不假设镜像已预装。
> 如果本地没有镜像，或镜像创建时间距本次验证超过 30 天，需要先从 SWR 重新拉取。
> 通过 `date` 获取当前时间，通过 `docker image inspect --format '{{.Created}}'` 获取本地镜像创建时间。

```bash
IMAGE=swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env
CREATED=$(docker image inspect "$IMAGE" --format '{{.Created}}' 2>/dev/null || true)

if [ -z "$CREATED" ]; then
  echo "Image not found locally, pulling: $IMAGE"
  docker pull "$IMAGE"
else
  NOW=$(date +%s)
  CREATED_TS=$(date -d "$CREATED" +%s)
  AGE_DAYS=$(( (NOW - CREATED_TS) / 86400 ))
  if [ "$AGE_DAYS" -gt 30 ]; then
    echo "Image is ${AGE_DAYS} days old, refreshing: $IMAGE"
    docker pull "$IMAGE"
  else
    echo "Using local image created ${AGE_DAYS} days ago: $IMAGE"
  fi
fi
```

## Phase 2 — Start Container and Fix chroot Environment

```bash
# 2.1 Start detached container (--entrypoint override to keep container alive)
#     NOTE: The image's default entrypoint /bin/bash -l triggers auto-chroot
#     via .bashrc on interactive login. For detached mode we override to
#     keep the container running, then manually chroot as needed.
docker run -d --name verify-oee --privileged \
  --entrypoint /bin/bash \
  swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env \
  -c "sleep infinity"

# 2.2 Verify aarch64 emulation
docker exec verify-oee chroot /root/openeuler_rootfs uname -m
# Expected: aarch64

# 2.3 Fix DNS (rootfs has no /etc/resolv.conf)
docker exec verify-oee bash -c \
  'rm -f /root/openeuler_rootfs/etc/resolv.conf && cp /etc/resolv.conf /root/openeuler_rootfs/etc/resolv.conf'

# 2.4 Fix /var/log (symlink target missing in rootfs)
docker exec verify-oee bash -c \
  'mkdir -p /root/openeuler_rootfs/var/volatile/log'

# 2.5 Fix git safe.directory for UID mismatch after docker cp.
#     Use the wildcard: docker cp preserves host UIDs, and EVERY submodule is a
#     separate git repository, so enumerating paths (e.g. only /root/IB_Robot and
#     libs/lerobot) fails later on the first submodule setup.sh touches
#     (observed: libs/Livox-SDK2 aborted a setup run in the submodule stage).
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs git config --global --replace-all safe.directory "*"'
```

**Why --privileged:** chroot 和 qemu-user binfmt 模拟需要 privileged 权限。

**Why --entrypoint override:** 镜像默认 entrypoint `/bin/bash -l` 会在交互模式下
通过 `.bashrc` 自动 chroot。但 detached 模式下需要 override 以保持容器存活，
然后通过 `docker exec ... chroot /root/openeuler_rootfs` 手动进入 arm64 环境。

**Why chroot:** 容器镜像的 `.bashrc` 在交互 login 时自动
`exec chroot /root/openeuler_rootfs /bin/bash`。`docker exec` 命令在容器宿主空间
执行，必须手动 `chroot /root/openeuler_rootfs` 才能进入 arm64 环境。

**Host safety rule:** 任何 chroot 或 qemu/rootfs 相关命令都必须包在
`docker exec verify-oee ...` 里执行。不要在宿主机直接 chroot 到 aarch64 rootfs，
也不要把宿主机 `/proc`、`/sys`、`/dev` bind 到容器 rootfs；setup/build 不需要这些资源。

## Phase 3 — Inspect chroot Environment

```bash
docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "
  uname -a
  cat /etc/os-release | head -3
  type git python3 dnf
  /usr/bin/git --version
  /usr/bin/python3 --version
  ls /opt/ros/humble/setup.bash 2>/dev/null || echo no-ros-yet
  cat /etc/yum.repos.d/openEulerROS.repo
"'
```

容器镜像应包含：git、python3、dnf、ROS 2 Humble、pip 下载缓存和两个 openEuler ROS repo 配置。
setup.sh 仍须完整执行，以验证 ROS 检测、workspace venv 安装和项目依赖配置；无需手动干预。

## Phase 3.5 — Verify & Repair Base Image Integrity

> **必须执行。** `:env` 镜像有时会自带状态不一致的基础包：RPM 数据库记录
> 版本 X，但磁盘上的实际二进制 / soname 是版本 Y。这种"原始镜像就已损坏"
> 的状态会让 setup.sh 的 dnf 事务失败：当 setup.sh 安装新子包（如 `*-devel`）
> 时，RPM 会按数据库记录创建指向"应当存在但磁盘上不存在"的 soname 符号链接，
> 导致运行时找不到库。参见 SKILL.md 的 [Base Image Integrity Pre-flight](../SKILL.md#base-image-integrity-pre-flight)。

**通用处理流程：**

1. 在 chroot 中用 `rpm -V` 探测已知易损坏包的一致性
2. 若 `rpm -V` 报告缺失或链接错误，对受影响包及 `-devel` 子包执行 `dnf reinstall -y --nogpgcheck`
3. 重新 `rpm -V` 确认磁盘内容与数据库一致

**已知易损坏包列表（按发现顺序追加，非封闭列表）：**

| 包名 | 子包 | 发现镜像批次 | 现象 |
|------|------|----------|------|
| `lz4` | `lz4-devel` | `:env` 镜像（2026-07 前后） | RPM DB 标称 `lz4-1.9.4-2.oe2403`，磁盘实为 `lz4 1.10.0`；`/usr/lib64/liblz4.so.1 -> liblz4.so.1.10.0`；`liblz4.so.1.9.4` 与 `liblz4.so` 不存在；`rpm -V lz4` 报 `liblz4.so.1` 链接错误与 `liblz4.so.1.9.4` 缺失。setup.sh 安装 `flann-devel` → `lz4-devel` 时 dnf 创建指向不存在的 `liblz4.so.1.9.4` 的 `/usr/lib64/liblz4.so`，导致后续链接错误 |

**步骤 1 — 探测已知易损坏包：**

```bash
docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "
  for pkg in lz4 lz4-devel; do
    if rpm -q \"\$pkg\" >/dev/null 2>&1; then
      echo \"=== \$pkg ===\"
      rpm -V \"\$pkg\" || true
    else
      echo \"=== \$pkg (not installed) ===\"
    fi
  done
"'
```

预期输出（损坏时）：包含 `missing /usr/lib64/liblz4.so.1.9.4` 和
`..?......  /usr/lib64/liblz4.so.1` 之类的行。如果所有包都无输出，说明基础
层一致，可跳过步骤 2 直接进入 Phase 4。

**步骤 2 — 修复（按需执行，仅对 `rpm -V` 报错的包重装）：**

```bash
docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "
  dnf reinstall -y --nogpgcheck lz4 lz4-devel
"'
```

> **为什么是 `reinstall` 而不是 `install` / `upgrade`：** 包已经在 RPM DB 中
> 声明，问题是磁盘内容与数据库不一致。`reinstall` 强制重新解包 RPM，把磁盘
> 内容对齐到数据库记录；`install` 看到包已声明会跳过，`upgrade` 会改写 RPM
> DB 引入新的版本不一致风险。

**步骤 3 — 重新验证一致性：**

```bash
docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "
  rpm -V lz4 lz4-devel && echo OK || echo STILL_BROKEN
"'
```

输出 `OK` 才能继续。若仍 `STILL_BROKEN`，按 Fatal 错误上报：基础镜像损坏
超出本 Phase 修复范围，需要重新拉取 `:env` 镜像（回到 Phase 1）或联系镜像
维护者。

**何时扩展：** 当 setup.sh 因其他包出现类似"RPM DB 标称版本 vs 磁盘实际版本
不一致"失败时，把该包加入上表，并在步骤 1 的 `for pkg in ...` 列表与步骤 2 的
`dnf reinstall` 命令中追加对应包名。lz4 是已知示例，不是封闭列表。

> **⚠️ 本 Phase 探测通过 ≠ 一劳永逸（2026-09 实测教训）：** 一次真实验证中
> Phase 3.5 探测完全干净，但 setup.sh 的 dnf 事务（rosdep 安装 `flann-devel`
> → `lz4-devel`，同时从 repo 拉入旧版 `lz4`）**重新引入**了不一致，导致
> build 在 `livox_ros_driver2` 链接阶段失败（`No rule to make target
> '/usr/lib64/liblz4.so'`），浪费约 1.5 小时 qemu 构建时间。因此 Phase 5.5
> 的 post-setup 门禁是**必须执行**的，不能因为 Phase 3.5 干净就跳过。

## Phase 5.5 — Post-Setup Integrity Gate (MUST run before build)

> **必须在 setup.sh 完成之后、build.sh 启动之前执行。** 成本为秒级；跳过它
> 的代价是构建在链接阶段失败（实测一次失败浪费约 1.5 小时 qemu 构建时间）。
> setup 的 dnf 事务可能拉入与磁盘内容冲突的基础包版本，即使 Phase 3.5
> 探测干净，此门禁也可能失败——这正是它存在的意义。

```bash
docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "
  set -e
  # Build-blocking condition: a file recorded in the RPM DB is MISSING from
  # disk, or the -devel soname symlink that C++ linking consumes is absent.
  broken=0
  for pkg in lz4 lz4-devel; do
    if rpm -q \"\$pkg\" >/dev/null 2>&1; then
      if rpm -V \"\$pkg\" 2>&1 | grep -q missing; then broken=1; fi
    fi
  done
  if [ -e /usr/lib64/liblz4.so ]; then
    echo \"liblz4.so present\"
  else
    echo \"liblz4.so MISSING\"; broken=1
  fi
  if [ \"\$broken\" -ne 0 ]; then
    echo \"Integrity gate FAILED; realigning disk with RPM DB\"
    dnf reinstall -y --nogpgcheck lz4 lz4-devel
    test -e /usr/lib64/liblz4.so
    rpm -V lz4 lz4-devel 2>&1 | grep missing && exit 2 || true
  fi
  echo \"gate-ok\"
"'
```

> **判定标准（2026-09 复跑实测补充）：** 重装能恢复**缺失的文件与 devel
> soname 符号链接**，但 rpm **不会覆盖磁盘上已存在的运行时符号链接**
> （`liblz4.so.1 -> 1.10.0` 保持不变，DB 记录 1.9.4）。因此重装后
> `rpm -V` 仍会报良性的 `.L`（symlink 内容偏好）差异。这是预期行为，
> 不是失败：链接走 `liblz4.so`，运行走 `.1`，两个版本的实体库文件都已
> 存在。门禁以 **"无 `missing` 条目 + `liblz4.so` 存在"** 为通过条件，
> 不要把纯 `.L` 差异当失败，否则会无限误判。

若重装后仍失败（`rpm -V` 持续报错或缺文件），按 Fatal 错误上报并停止：
基础镜像/仓库状态超出修复范围，不要带着已知损坏状态启动多小时构建。

**何时扩展：** 构建若因其他 `lib<name>.so` 缺失失败，把 `<name> <name>-devel`
加入本门禁与 Phase 3.5 的探测/重装列表（同一根因：dnf 事务引入的版本回退）。

## Phase 4 — Prepare Workspace

根据验证目的选择来源。PR 证据必须使用独立 commit 快照；当前工作区 copy 只用于本地调试。

### PR 证据：从独立 commit 快照 docker cp

> 当前工作区无需 clean。先在宿主机 worktree 外创建 standalone clone，只检出目标 commit，
> 初始化 submodule 并核对 tree SHA，再把快照复制进 rootfs。

```bash
set -e
command -v git-lfs >/dev/null

PROJECT_ROOT=<project_root>
VERIFICATION_REF=<pushed-branch-or-commit>
VERIFICATION_REPO=<pushed-repository-url>
VERIFIED_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse "${VERIFICATION_REF}^{commit}")
VERIFIED_TREE=$(git -C "${PROJECT_ROOT}" rev-parse "${VERIFIED_COMMIT}^{tree}")
SNAPSHOT_ROOT=""
cleanup_snapshot() {
  if [ -n "${SNAPSHOT_ROOT}" ] && [ -d "${SNAPSHOT_ROOT}" ]; then
    rm -rf -- "${SNAPSHOT_ROOT}"
  fi
}
trap cleanup_snapshot EXIT INT TERM
SNAPSHOT_ROOT=$(mktemp -d "/tmp/ibrobot-verify-${VERIFIED_TREE:0:12}.XXXXXX")
SOURCE_MODE="isolated committed snapshot"

git clone --no-hardlinks --no-checkout "${PROJECT_ROOT}" "${SNAPSHOT_ROOT}"
git -C "${SNAPSHOT_ROOT}" remote set-url origin "${VERIFICATION_REPO}"
git -C "${SNAPSHOT_ROOT}" checkout --detach "${VERIFIED_COMMIT}"
git -C "${SNAPSHOT_ROOT}" submodule update --init --recursive
git -C "${SNAPSHOT_ROOT}" lfs pull origin
git -C "${SNAPSHOT_ROOT}" lfs fsck
test "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD^{tree})" = "${VERIFIED_TREE}"
test -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=all)"

docker cp "${SNAPSHOT_ROOT}" verify-oee:/root/openeuler_rootfs/root/IB_Robot
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

结果同时记录 provenance 用的 `VERIFIED_COMMIT` 和门禁使用的 `VERIFIED_TREE`。保留快照直到
两个平台都已复制，或让两个平台分别按同一 commit 生成并核对相同 tree 的快照。

### 本地调试：从当前宿主机工作区 docker cp

> 用于验证本地未提交的改动。`docker cp` 拷入当前项目目录，可直接验证修改效果，但该结果
> 不具备不可变 tree 身份，不能作为 PR Verification。

```bash
# 4.1 Copy current workspace into rootfs
docker cp <project_root> verify-oee:/root/openeuler_rootfs/root/IB_Robot

# 4.2 Remove stale artifacts (host paths are wrong in container)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

### 显式请求：在容器内 git clone 远程 commit 或 branch

> 仅在用户明确指定远端来源时使用。clone 后必须记录实际 HEAD commit 及其 tree，不能只记录
> 可移动的 branch 名。注意 rootfs 中 `which` 不可靠，git 命令请用绝对路径 `/usr/bin/git`。

```bash
# 4.1 Clone the branch inside chroot
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "
    cd /root && /usr/bin/git clone -b <branch> <repo_url> /root/IB_Robot
  "'

# 4.2 Remove stale artifacts (if any)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

## Phase 4.5 — ROS 源切换预清理（条件执行）

> **触发条件：** 验证的变更切换了 ROS 包源——`install_ros.sh`、平台脚本或 repo 配置
> 修改了 openEuler ROS repo（Embedded + SIG）、EUR 源地址、镜像或 GPG key。容器中
> 已有 ROS 包（`:env` 镜像预装或上次验证残留）时**必须执行本 Phase**。

`:env` 镜像预装的 `ros-humble-*` 包来自旧源。不移除的话，setup.sh 的 ROS 检测会
直接复用旧包，`install_ros.sh` 从新源安装的路径完全没有被验证。本 Phase 只删除
包、不安装任何软件，安装仍由 setup.sh 唯一完成，不违反 Core Principle
（与 AGENTS.md「跨发行版 ROS 包版本一致性」中 openEuler 侧 `dnf remove 'ros-humble-*'`
后重装的处置一致）。

```bash
# 以 root 在 chroot 中执行；在 Phase 5 运行 setup.sh 之前完成。
docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "
  set -e
  ros_count=\$(rpm -qa --qf \"%{NAME}\\n\" | grep -c \"^ros-humble-\" || true)
  if [ \"\${ros_count}\" -gt 0 ]; then
    echo \"Removing \${ros_count} ros-humble-* packages before source-switch verification\"
    dnf remove -y \"ros-humble-*\"
    rm -rf /opt/ros/humble
  else
    echo \"No ros-humble-* packages installed; nothing to pre-clean.\"
  fi
  # Hard check: no ROS package and no /opt/ros residue may remain.
  test -z \"\$(rpm -qa --qf \"%{NAME}\\n\" | grep \"^ros-humble-\" || true)\"
  test ! -e /opt/ros/humble
  echo \"ROS source-switch pre-clean complete.\"
"'
```

验证报告中必须写明本步骤（移除的包数量），让 reviewer 知道 setup.sh 验证的是
从新源完整安装。同时注意：本 Phase 只覆盖"新源可完整安装"；若变更还涉及从零
配置 repo/GPG（无 ROS 基础镜像场景），需另行使用不含 ROS 的基础镜像验证。

## Phase 5 — Run setup.sh

```bash
docker exec -d verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "cd /root/IB_Robot && IBR_LEROBOT_FORCE_REBUILD=1 bash scripts/setup.sh --yes --no-sudo > /tmp/setup.log 2>&1"'
```

Monitor progress (qemu-user 模拟下 pip 安装很慢，全流程约 20-40 min):

```bash
# Poll (note: always chroot to read log)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "tail -15 /tmp/setup.log"'
```

Wait until the log ends with:

```
Setup complete! Run ./scripts/build.sh to build the workspace.
```

**Phase 5 完成后必须执行 Phase 5.5 的 post-setup integrity gate，再进入 Phase 6。**

## Phase 6 — Run build.sh

```bash
docker exec -d verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "cd /root/IB_Robot && source /opt/ros/humble/setup.bash && source venv/bin/activate && bash scripts/build.sh > /tmp/build.log 2>&1"'
```

Monitor:

```bash
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "tail -15 /tmp/build.log"'
```

Wait until:

```
Build complete. Source with: source install/setup.sh
```

## Phase 7 — Inspect and Clean Up

```bash
# Collect all ERROR lines with context
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "git -C /root/IB_Robot rev-parse HEAD HEAD^{tree}"'
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "grep ERROR /tmp/setup.log || echo \"(none)\""'
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "grep ERROR /tmp/build.log || echo \"(none)\""'

# Clean up
docker stop verify-oee && docker rm verify-oee
if [ "${SOURCE_MODE:-}" = "isolated committed snapshot" ]; then
  rm -rf -- "${SNAPSHOT_ROOT}"
  SNAPSHOT_ROOT=""
fi
```

> **错误报告要求**：必须逐条列出所有 ERROR 行，并按 Verification Discipline 中的分类标准标注 Fatal / Non-fatal。不能只给 ERROR 行数不给内容。

PR 证据的最终报告必须写出 `Source: isolated committed snapshot`、完整 commit 和完整 tree，
并将 tree SHA 交给 PR 工作流组装成结构化 `## Docker Verification` 块。当前工作区 copy 的结果必须
标为 local-only，不得伪装成 tree-bound PR 证据。

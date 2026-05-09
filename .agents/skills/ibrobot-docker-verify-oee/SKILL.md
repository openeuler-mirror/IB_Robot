---
name: ibrobot-docker-verify-oee
description: "在 openEuler Embedded (aarch64) Docker 容器中端到端验证 setup.sh + build.sh。容器通过 chroot /root/rootfs 进入 qemu-user 模拟的 arm64 环境，以 root 用户操作。Use when user wants to 'openEuler Docker 验证', 'oee container test', 'openEuler 容器测试', '验证 openEuler setup', 'aarch64 验证', or after modifying openEuler platform scripts to ensure changes work on emulated arm64 environment."
---

# IB-Robot openEuler Embedded Docker Verification Skill

在全新 openEuler Embedded aarch64 Docker 容器中完整验证 `setup.sh` 和 `build.sh`。

> **⚠️ 与 Ubuntu 验证的两个核心差异：**
> 1. **以 root 用户操作**（openEuler Embedded 开发板默认 root，无需 sudo/testuser）
> 2. **所有 docker exec 命令必须 `chroot /root/rootfs`** 进入 arm64 rootfs 环境才能执行
容器镜像通过 `--privileged` + `chroot /root/rootfs` 进入 qemu-user 模拟的 arm64 环境，
模拟真实 openEuler Embedded 开发板的首用体验。

## When to Use

- 修改了 `scripts/setup/platforms/openeuler-embedded-24.03.sh` 后
- 修改了 `scripts/setup.sh` 中影响 dnf/rosdep 的逻辑后
- 修改了 `scripts/setup/lerobot_patches.sh` 后
- 用户要求 "openEuler Docker 验证" / "oee container test"
- PR 合入前需要双平台验证时

## Prerequisites

- Docker 已安装且当前用户有运行容器的权限
- openEuler Embedded IB_Robot Docker 镜像已加载（`docker load < xxx.tar.gz`）
  *(注: 获取该测试镜像需联系维护者或从内部 release 渠道下载)*
- 网络：可访问 `repo.openeuler.org`、`eur.openeuler.openatom.cn`、
  `eulermaker.compass-ci.openeuler.openatom.cn`、华为 pip 镜像
- IB-Robot workspace 中有待验证的修改

## Container Architecture

```
┌─ Docker container (x86_64) ──────────────────────┐
│  entrypoint.sh                                    │
│    mount --bind /dev /root/rootfs/dev             │
│    exec chroot /root/rootfs /bin/bash             │
│  ┌─ chroot /root/rootfs (aarch64 via qemu-user) ─┐│
│  │  openEuler Embedded Reference Distro          ││
│  │  openEuler ROS repos ( Embedded + SIG )       ││
│  │  python3, dnf, git                            ││
│  │  workspace at /root/IB_Robot                  ││
│  └────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

> **关于 ROS 安装：** setup.sh 会自动检测 ROS 是否已安装。若未安装，会调用
> `scripts/install_ros.sh` 完成安装（配置 openEuler ROS repo + dnf 安装）。
> **不需要也不应该手动预装 ROS**，让 setup.sh 完整跑一遍才能验证安装流程。

所有 `docker exec` 命令需要通过 `chroot /root/rootfs` 进入 arm64 环境。

## Container Naming Convention

| Variable       | Value                                       |
|----------------|---------------------------------------------|
| Container name | `verify-oee`                                |
| User           | `root`（openEuler Embedded 默认 root 操作） |
| Workspace      | `/root/rootfs/root/IB_Robot`（chroot 内路径）|
| Host workspace | 宿主机上 IB_Robot 项目根目录                 |

## Procedure

### Phase 1 — Start Container and Fix chroot Environment

```bash
# 1.1 Start detached container (entrypoint chroots into arm64 rootfs)
docker run -d --name verify-oee --privileged \
  openeuler-embedded-ibrobot "sleep infinity"

# 1.2 Verify aarch64 emulation
docker exec verify-oee chroot /root/rootfs uname -m
# Expected: aarch64

# 1.3 Fix DNS (rootfs has no /etc/resolv.conf)
docker exec verify-oee bash -c \
  'rm -f /root/rootfs/etc/resolv.conf && cp /etc/resolv.conf /root/rootfs/etc/resolv.conf'

# 1.4 Fix /var/log (symlink target missing in rootfs)
docker exec verify-oee bash -c \
  'mkdir -p /root/rootfs/var/volatile/log'

# 1.5 Mount /proc and /sys for chroot compatibility
docker exec verify-oee bash -c \
  'mount -t proc proc /root/rootfs/proc 2>/dev/null; mount --bind /sys /root/rootfs/sys 2>/dev/null'

# 1.6 Fix git safe.directory for UID mismatch after docker cp
docker exec verify-oee bash -c \
  'chroot /root/rootfs git config --global --add safe.directory /root/IB_Robot
   chroot /root/rootfs git config --global --add safe.directory /root/IB_Robot/libs/lerobot'

# 1.7 Fix dnf GPGME emulation bug (qemu-aarch64 only)
docker exec verify-oee bash -c \
  'chroot /root/rootfs sed -i "s/gpgcheck=1/gpgcheck=0/" /etc/dnf/dnf.conf'
```

**Why --privileged:** entrypoint 执行 `mount --bind /dev` 和 `chroot`，需要
privileged 权限。qemu-user binfmt 模拟也需要。

**Why chroot:** 容器镜像的 entrypoint 会 `chroot /root/rootfs` 进入 arm64
rootfs。`docker exec` 命令在容器宿主空间执行，必须手动 `chroot /root/rootfs`
才能进入 arm64 环境。

### Phase 2 — Inspect chroot Environment

```bash
docker exec verify-oee bash -c 'chroot /root/rootfs /bin/bash -c "
  uname -a
  cat /etc/os-release | head -3
  which dnf python3 git
  ls /opt/ros/humble/setup.bash
  cat /etc/yum.repos.d/openEulerROS.repo
"'
```

容器镜像应包含：git、python3、dnf + 两个 openEuler ROS repo 配置。
ROS 2 安装由 setup.sh 通过 `install_ros.sh` 自动完成，无需手动干预。

### Phase 3 — Prepare a Standalone Workspace Copy

> **重要：不要直接把 linked worktree 拷进容器。**
> 如果后续要运行 `setup.sh` 且不使用 `--skip-submodules`，请先在宿主机准备一个
> 普通 clone（非 linked worktree），并完成 `git submodule update --init --recursive`。
> 这样容器内的 `.git` / submodule 状态才是自洽的。

```bash
# 3.1 Create a standalone clone for verification
git clone --branch <branch> --single-branch <project_root> /tmp/oee-verify
git -C /tmp/oee-verify submodule update --init --recursive

# 3.2 Copy the standalone workspace into rootfs
docker cp /tmp/oee-verify verify-oee:/root/rootfs/root/IB_Robot

# 3.3 Remove stale artifacts (host paths are wrong in container)
docker exec verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

### Phase 4 — Run setup.sh

```bash
docker exec -d verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "cd /root/IB_Robot && IBR_LEROBOT_FORCE_REBUILD=1 bash scripts/setup.sh --yes --no-sudo > /tmp/setup.log 2>&1"'
```

Monitor progress (qemu-user 模拟下 pip 安装很慢，全流程约 20-40 min):

```bash
# Poll (note: always chroot to read log)
docker exec verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "tail -15 /tmp/setup.log"'
```

Wait until the log ends with:

```
Setup complete! Run ./scripts/build.sh to build the workspace.
```

### Phase 5 — Run build.sh

```bash
docker exec -d verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "cd /root/IB_Robot && source /opt/ros/humble/setup.bash && source venv/bin/activate && bash scripts/build.sh > /tmp/build.log 2>&1"'
```

Monitor:

```bash
docker exec verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "tail -15 /tmp/build.log"'
```

Wait until:

```
Build complete. Source with: source install/setup.sh
```

### Phase 6 — Inspect and Clean Up

```bash
# Check results
docker exec verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "grep -c ERROR /tmp/setup.log"'
docker exec verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "grep -c ERROR /tmp/build.log"'

# Clean up
docker stop verify-oee && docker rm verify-oee
```

## Quick-Run (Iterative Testing)

When only scripts changed (ROS 2 + system deps already installed), copy
updated files and re-run without recreating the container:

```bash
# Copy changed files into rootfs
docker cp scripts/setup.sh verify-oee:/root/rootfs/root/IB_Robot/scripts/
docker cp scripts/setup/platforms/openeuler-embedded-24.03.sh \
  verify-oee:/root/rootfs/root/IB_Robot/scripts/setup/platforms/
docker cp scripts/setup/lerobot_patches.sh \
  verify-oee:/root/rootfs/root/IB_Robot/scripts/setup/

# Clean and re-run
docker exec verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
docker exec -d verify-oee bash -c \
  'chroot /root/rootfs /bin/bash -c "cd /root/IB_Robot && IBR_LEROBOT_FORCE_REBUILD=1 bash scripts/setup.sh --yes --no-sudo > /tmp/setup.log 2>&1"'
```

## Key Differences from Ubuntu Verification

| Aspect | Ubuntu 22.04 (`ibrobot-docker-verify`) | openEuler Embedded (this skill) |
|--------|----------------------------------------|--------------------------------|
| Architecture | x86_64 native | aarch64 via qemu-user chroot |
| **User** | `testuser` + NOPASSWD sudo | **`root`（无需 sudo）** |
| **Command prefix** | `docker exec verify-ubuntu2204` | **`docker exec verify-oee bash -c 'chroot /root/rootfs /bin/bash -c "..."'`** |
| Package manager | apt | dnf |
| ROS install | setup.sh 自动调用 `install_ros.sh` 安装 | setup.sh 自动调用 `install_ros.sh` 安装 |
| Speed | Native (fast) | Emulated (20-40 min for setup) |
| DNS | Works by default | Must copy resolv.conf |
| git-lfs | Available | Not available; lerobot_patches.sh auto-removes LFS hook |

## Known Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Couldn't resolve host name` | rootfs missing `/etc/resolv.conf` | Phase 1.3 copies from host container |
| `Config error: File exists: /var/log` | `/var/log` symlink target missing | Phase 1.4 creates `/var/volatile/log` |
| `/dev/stdout: No such file or directory` | `/proc` not mounted in chroot | Phase 1.5 mounts proc/sys |
| `dubious ownership in repository` | UID mismatch after `docker cp` | Phase 1.6 adds `safe.directory` |
| `gpg.errors.GPGMEError` during `rosdep install` | qemu-aarch64 emulation bug with Python `gpg` | Phase 1.7 disables `gpgcheck=1` globally in `/etc/dnf/dnf.conf` |
| `git-lfs was not found` post-checkout hook | No git-lfs in rootfs | `lerobot_patches.sh` auto-removes hook when git-lfs missing |
| `ERROR: file:///root/IB_Robot/libs/lerobot does not appear to be a Python project` | Copied a linked worktree or an uninitialized submodule tree into the container, then ran setup without `--skip-submodules` | Use a standalone clone and run `git submodule update --init --recursive` before `docker cp` |
| `pip3 not found, cannot install colcon` | `platform_install_python_bootstrap` not called before `ensure_colcon` | `install_system_deps` calls bootstrap first |
| `python%{python3_pkgversion}-scipy` not found | `ROS_OS_OVERRIDE=rhel:8` uses RHEL naming; openEuler dnf can't match macro | Platform script skips `python3-scipy` in rosdep, installs via explicit `dnf install` |
| `rosdep install failed` for missing packages | Some ROS packages not in openEuler repos (e.g. `robot_localization`) | Platform script uses non-fatal rosdep + skip-keys |
| dnf outputs config dump instead of installing | Running dnf without `--nogpgcheck --setopt=strict=0` in chroot | Always use `dnf install -y --nogpgcheck` |

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch from base commit |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `ROS_OS_OVERRIDE` | `rhel:8` | Set by platform script for rosdep compatibility |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |

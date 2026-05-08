---
name: ibrobot-docker-verify
description: "Validate setup.sh + build.sh in a clean Ubuntu 22.04 Docker container. Use when user wants to 'docker verify', 'container test', 'validate setup', 'Docker 验证', '容器测试', '验证 setup', or after modifying setup.sh / platform scripts / verify_env.sh to ensure changes work on a fresh system."
---

# IB-Robot Docker Verification Skill

Full end-to-end validation of `setup.sh` and `build.sh` inside a pristine
Ubuntu 22.04 container — the closest approximation to a user's first-run
experience without requiring real hardware.

## When to Use

- After modifying `scripts/setup.sh`, `scripts/setup/platforms/*.sh`,
  `scripts/setup/verify_env.sh`, or `scripts/install_ros.sh`.
- After changes that affect pip/apt dependency resolution.
- Before merging PRs that touch the install/build pipeline.
- User explicitly requests "Docker 验证" / "container test".

## Prerequisites

- Docker installed and the current user has permission to run containers.
- The IB-Robot workspace has uncommitted or committed changes to validate.
- Network access to Aliyun apt mirror, TUNA ROS 2 repo, Huawei pip mirror,
  and `gitcode.com` / `atomgit.com` for lerobot submodule fetch.

## Container Naming Convention

| Variable      | Value                        |
|---------------|------------------------------|
| Container name | `verify-ubuntu2204`          |
| User           | `testuser`                   |
| Workspace      | `/home/testuser/IB_Robot`    |

## Procedure

### Phase 1 — Create and Provision Container

```bash
# 1.1 Start detached Ubuntu 22.04 container
docker run -d --name verify-ubuntu2204 \
  -e TZ=Asia/Shanghai \
  -e DEBIAN_FRONTEND=noninteractive \
  ubuntu:22.04 tail -f /dev/null

# 1.2 Install core prerequisites as root
docker exec verify-ubuntu2204 bash -c '
  apt-get update -qq &&
  apt-get install -y -qq \
    sudo git git-lfs locales python3 curl \
    gnupg2 lsb-release software-properties-common \
  > /dev/null 2>&1 &&
  useradd -m -s /bin/bash testuser &&
  echo "testuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/testuser &&
  locale-gen en_US.UTF-8 &&
  echo "container ready"
'
```

**Why NOPASSWD is required:** Docker `exec -d` (detached mode) allocates no
tty.  Ubuntu 22.04's default sudoers enables `use_pty`, which makes
`sudo -v` (validate credential cache) require a terminal even when the user
has `NOPASSWD:ALL`.  The `ensure_sudo_session` function in `setup.sh`
already works around this by calling `sudo -n true` first (non-interactive,
no tty needed), but the container still needs `NOPASSWD:ALL` so that
`sudo -n true` succeeds without a password prompt.

If you skip `NOPASSWD:ALL`, setup.sh will fail at `ensure_sudo_session` because `sudo -n true` returns non-zero
without either NOPASSWD or cached credentials — and there is no tty to
cache credentials through in detached mode.

### Phase 2 — Configure Mirrors

> **ROS 安装由 setup.sh 自动完成：** `setup.sh` 检测到 ROS 未安装时会调用
> `install_ros.sh` 自动安装（配置 ROS repo + apt 安装）。
> 不需要也不应该手动预装 ROS，让 setup.sh 完整跑一遍才能验证安装流程。

```bash
# 2.1 Aliyun apt mirror
docker exec verify-ubuntu2204 bash -c '
  sed -i "s|http://archive.ubuntu.com|http://mirrors.aliyun.com|g;
          s|http://security.ubuntu.com|http://mirrors.aliyun.com|g" \
    /etc/apt/sources.list
'
```

### Phase 3 — Copy Workspace

```bash
# 3.1 Copy the workspace into the container
docker cp <project_root> verify-ubuntu2204:/home/testuser/IB_Robot
docker exec verify-ubuntu2204 chown -R testuser:testuser /home/testuser/IB_Robot

# 3.2 Remove stale venv/build/install/log (copied from host, paths are wrong)
docker exec verify-ubuntu2204 bash -c '
  rm -rf /home/testuser/IB_Robot/{venv,build,install,log}
'
```

### Phase 4 — Run setup.sh

```bash
docker exec -d \
  -u testuser \
  -e HOME=/home/testuser \
  -e IBR_LEROBOT_FORCE_REBUILD=1 \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/setup.sh --yes --skip-submodules \
    > /tmp/setup.log 2>&1'
```

Monitor progress (pip downloads of torch/CUDA libraries take 10-20 min):

```bash
# Poll every 60s
docker exec verify-ubuntu2204 bash -c 'tail -5 /tmp/setup.log'
```

Wait until the log ends with:

```
Setup complete! Run ./scripts/build.sh to build the workspace.
```

### Phase 5 — Run build.sh

```bash
docker exec -d \
  -u testuser \
  -e HOME=/home/testuser \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/build.sh > /tmp/build.log 2>&1'
```

Monitor:

```bash
docker exec verify-ubuntu2204 bash -c 'tail -5 /tmp/build.log'
```

Wait until:

```
Build complete. Source with: source install/setup.sh
```

### Phase 6 — Inspect and Clean Up

```bash
# Check results
docker exec verify-ubuntu2204 bash -c 'grep -c "ERROR" /tmp/setup.log'
docker exec verify-ubuntu2204 bash -c 'grep -c "ERROR" /tmp/build.log'

# Clean up
docker stop verify-ubuntu2204 && docker rm verify-ubuntu2204
```

## Quick-Run One-Liner (for Iterative Testing)

When only the setup/platform scripts changed (ROS 2 already installed in a
running container), copy updated files and re-run without recreating the
container:

```bash
# Copy changed files
docker cp scripts/setup.sh verify-ubuntu2204:/home/testuser/IB_Robot/scripts/
docker cp scripts/setup/platforms/ubuntu-22.04.sh \
  verify-ubuntu2204:/home/testuser/IB_Robot/scripts/setup/platforms/

# Remove stale venv so setup recreates it with new code
docker exec verify-ubuntu2204 bash -c \
  'rm -rf /home/testuser/IB_Robot/{venv,build,install,log}'

# Re-run
docker exec -d -u testuser -e HOME=/home/testuser \
  -e IBR_LEROBOT_FORCE_REBUILD=1 \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/setup.sh --yes --skip-submodules \
    > /tmp/setup.log 2>&1'
```

## Known Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sudo: a terminal is required to read the password` | Docker exec has no tty; `use_pty` in sudoers blocks `sudo -v` | Phase 1 **must** set `NOPASSWD:ALL`; `setup.sh` code uses `sudo -n true` first |
| `sh: 1: rosdep: not found` | Was caused by rosdepc calling `os.system('rosdep ...')` | Replaced rosdepc with direct `pip install rosdep` |
| `error loading sources list: Permission denied` | `write_rosdep_sources_list` wrote file as 600 root | Now does `chmod 644` after writing |
| `rosdep update` times out | `ROSDISTRO_INDEX_URL` not passed to platform script | Platform scripts now pass `env ROSDISTRO_INDEX_URL=...` |
| pip downloads from pypi.org at ~10 KB/s | No pip mirror configured in container | `ensure_workspace_venv` writes `${VENV_PATH}/pip.conf` |
| `The build time path ... doesn't exist` | Copied host venv has hardcoded `/home/xqw/...` paths | Always `rm -rf venv build install log` before re-running setup |
| `git: command not found` mid-setup | `install_ros.sh` apt install may remove git | Phase 1 already installed git; re-run `apt-get install -y git git-lfs` if needed |
| lerobot patch stack fetch fails | Submodule base commit not in local checkout | Rebase branch onto `upstream/master` before copying |

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch in container |
| `DEBIAN_FRONTEND` | `noninteractive` | Prevent tzdata etc. from blocking |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |

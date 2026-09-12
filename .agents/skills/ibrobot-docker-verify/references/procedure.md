# Full Procedure — Phase 0 to Phase 6

> This document contains the complete bash commands for each phase.
> The main SKILL.md only lists the phase purposes; execute the actual
> commands by reading the corresponding section here.

## Phase 0 — Define Inputs and Check Host Prerequisites

> Run this on the host before any Docker operation. Do not continue if the
> `docker` command is unavailable.

```bash
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker CLI is not installed. Install Docker on the host, then rerun verification."
    exit 1
fi

IMAGE=osrf/ros:humble-desktop-full-jammy
CONTAINER=verify-ubuntu2204
PROJECT_ROOT=$(git rev-parse --show-toplevel)
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_PIP_CACHE=${PIP_CACHE_DIR:-"${HOME}/.cache/pip"}

if [ "${HOST_UID}" -eq 0 ]; then
    echo "Run Ubuntu Docker verification as the non-root workspace owner." >&2
    exit 1
fi

if [ ! -d "${HOST_PIP_CACHE}" ]; then
    echo "Host pip cache does not exist: ${HOST_PIP_CACHE}"
    exit 1
fi

# GraspGen's pointnet2_ops CUDA extension is now installed by default (no
# --with-grasp flag). The container needs nvcc to compile it, but does NOT
# need GPU passthrough (TORCH_CUDA_ARCH_LIST is fixed, no GPU detection).
# If the host has nvcc, mount the CUDA toolkit read-only so grasp deps can
# be compiled; if not, setup.sh will gracefully skip grasp install and the
# verification continues without the grasp smoke test.
HOST_CUDA_HOME=""
if command -v nvcc >/dev/null 2>&1; then
    HOST_CUDA_HOME="$(readlink -f "$(command -v nvcc)")"
    HOST_CUDA_HOME="$(dirname "$(dirname "${HOST_CUDA_HOME}")")"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    HOST_CUDA_HOME="$(readlink -f /usr/local/cuda)"
fi

# Reject system directories that would break the container if bind-mounted.
if [[ "${HOST_CUDA_HOME}" == "/" || "${HOST_CUDA_HOME}" == "/usr" || "${HOST_CUDA_HOME}" == "/usr/local" ]]; then
    echo "Host CUDA toolkit path is a system directory (${HOST_CUDA_HOME}); refusing to bind-mount." >&2
    echo "Install CUDA toolkit under /usr/local/cuda-<version> instead of apt nvidia-cuda-toolkit." >&2
    HOST_CUDA_HOME=""
fi

if [ -z "${HOST_CUDA_HOME}" ] || [ ! -x "${HOST_CUDA_HOME}/bin/nvcc" ]; then
    echo "Host CUDA toolkit (nvcc) not found; grasp deps will be skipped in verification." >&2
    echo "setup.sh will warn and skip GraspGen install on this container." >&2
    HOST_CUDA_HOME=""
else
    echo "Host CUDA toolkit: ${HOST_CUDA_HOME}"
    nvcc --version | head -4
fi

TOTAL_START=$(date +%s)
```

The source mode is selected in Phase 3. Author-side PR verification must use an
isolated committed snapshot. Use the current workspace copy only for local
diagnosis of uncommitted changes.

The CUDA toolkit is mounted read-only into the container at the same path
(`/usr/local/cuda-<version>`) so `install_graspgen_pip.sh`'s
`/usr/local/cuda-${torch_cuda_version}` detection also works when the PyTorch
CUDA version matches the host toolkit version.

## Phase 1 — Prepare ROS Desktop-Full Image

> The desktop-full image matches the `ros-humble-desktop-full` level installed
> by `install_ros.sh`. Do not use `ros:humble-ros-base-jammy`; rosdep would need
> to install omitted desktop, RViz, rqt, xacro, and robot-state-publisher
> dependencies during every verification.

```bash
IMAGE_START=$(date +%s)
docker pull "${IMAGE}"
IMAGE_SECONDS=$(( $(date +%s) - IMAGE_START ))
```

The first pull is network-dependent and should be reported separately. Later
runs normally reuse the local image layers.

## Phase 2 — Create and Provision Container

```bash
# 2.1 Remove a stale verification container, then start the ROS-ready image.
# CUDA toolkit (if available on the host) is mounted read-only so GraspGen's
# pointnet2_ops can be compiled. --gpus is NOT required: the CUDA arch is
# fixed via TORCH_CUDA_ARCH_LIST and the smoke test only does find_spec.
# If no host CUDA toolkit is available, setup.sh will skip grasp install
# gracefully and the verification continues without the grasp smoke test.
CONTAINER_START=$(date +%s)
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

DOCKER_RUN_ARGS=(
  -d --name "${CONTAINER}"
  --entrypoint /bin/bash
  --mount "type=bind,src=${HOST_PIP_CACHE},dst=/var/cache/ibrobot-pip"
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip
  -e TZ=Asia/Shanghai
  -e DEBIAN_FRONTEND=noninteractive
)

if [ -n "${HOST_CUDA_HOME}" ]; then
    DOCKER_RUN_ARGS+=(
        --mount "type=bind,src=${HOST_CUDA_HOME},dst=${HOST_CUDA_HOME},readonly"
        -e CUDA_HOME="${HOST_CUDA_HOME}"
        -e PATH="${HOST_CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        -e LD_LIBRARY_PATH="${HOST_CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu"
    )
fi

docker run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}" -c 'sleep infinity'

# 2.2 Configure domestic Ubuntu and ROS 2 mirrors before the first apt update,
# then install container bootstrap tools only. The ROS desktop-full image uses
# a deb822 ros2.sources file, while older images may use a traditional .list.
docker exec "${CONTAINER}" bash -c '
  sed -i "s|http://archive.ubuntu.com|http://mirrors.aliyun.com|g;
          s|http://security.ubuntu.com|http://mirrors.aliyun.com|g" \
    /etc/apt/sources.list &&
  if [ -f /etc/apt/sources.list.d/ros2.sources ]; then
    sed -i \
      "s|http://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g;
       s|https://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g;
       s|^Types: deb deb-src$|Types: deb|" \
      /etc/apt/sources.list.d/ros2.sources
  fi &&
  if [ -f /etc/apt/sources.list.d/ros2.list ]; then
    sed -i \
      "s|http://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g;
       s|https://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|g;
       /^deb-src /d" \
      /etc/apt/sources.list.d/ros2.list
  fi &&
  if grep -R -E "(archive|security).ubuntu.com|packages.ros.org/ros2/ubuntu" \
      /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then
    echo "Official Ubuntu or ROS 2 apt source remains after mirror configuration" >&2
    exit 1
  fi &&
  apt-get update -qq &&
  apt-get install -y -qq \
    sudo git git-lfs locales python3 curl \
    gnupg2 lsb-release software-properties-common \
    > /dev/null 2>&1
'

# 2.3 Match the container user to the host cache owner. HOST_UID/HOST_GID must
# come from Phase 0; never replace them with fixed values such as 1000:1000.
docker exec \
  -e HOST_UID="${HOST_UID}" \
  -e HOST_GID="${HOST_GID}" \
  "${CONTAINER}" bash -c '
    if ! getent group "${HOST_GID}" >/dev/null; then
      groupadd -g "${HOST_GID}" testuser
    fi
    group_name=$(getent group "${HOST_GID}" | cut -d: -f1)
    useradd -m -u "${HOST_UID}" -g "${group_name}" -s /bin/bash testuser
    echo "testuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/testuser
    locale-gen en_US.UTF-8 >/dev/null
  '

# 2.3b Pre-register the workspace as a git safe.directory for the container user.
#      docker cp preserves host UIDs; without this, every later git probe (e.g.
#      the Phase 6 `git rev-parse HEAD HEAD^{tree}` provenance line) fails with
#      "dubious ownership in repository" instead of producing evidence.
docker exec -u testuser -e HOME=/home/testuser "${CONTAINER}" \
  git config --global --add safe.directory /home/testuser/IB_Robot

# 2.4 Fail fast unless the actual setup user can use the mounted cache. A root
# check would produce a false positive for host directories owned by another
# UID. Creating a file verifies both traversal and write permissions.
docker exec \
  -u testuser \
  -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  "${CONTAINER}" bash -c '
    set -e
    test -r "${PIP_CACHE_DIR}"
    test -w "${PIP_CACHE_DIR}"
    probe="${PIP_CACHE_DIR}/.ibrobot-cache-write-test-$$"
    : > "${probe}"
    rm -f "${probe}"
    printf "pip cache writable: %s (uid=%s gid=%s)\n" \
      "${PIP_CACHE_DIR}" "$(id -u)" "$(id -g)"
  '

# 2.5 Verify the mounted CUDA toolkit is accessible from the container (only
# when a host CUDA toolkit was detected in Phase 0). If skipped, setup.sh will
# warn and skip GraspGen install — the verification continues.
if [ -n "${HOST_CUDA_HOME}" ]; then
    docker exec \
      -u testuser \
      -e HOME=/home/testuser \
      "${CONTAINER}" bash -c '
        set -e
        test -x "${CUDA_HOME}/bin/nvcc"
        "${CUDA_HOME}/bin/nvcc" --version | head -1
        printf "CUDA_HOME=%s\n" "${CUDA_HOME}"
      '
else
    echo "No host CUDA toolkit; grasp deps will be skipped in this verification run."
fi

CONTAINER_SECONDS=$(( $(date +%s) - CONTAINER_START ))
```

Mount pip at `/var/cache/ibrobot-pip`, not below `/home/testuser/.cache`.
Docker may create missing parent directories as root before `useradd`, which
can later break tools such as pre-commit even when the pip directory itself is
writable.

The cache probe is a hard prerequisite. Do not continue with setup when it
fails, and do not work around it by making the host cache world-writable. Fix
the UID/GID mapping or select a cache directory owned by the current host user.

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

## Phase 3 — Prepare the Source Workspace

Choose exactly one source mode. Record the selected mode and source details in
the verification result and in the PR description's Verification section.

### PR Evidence: Copy an Isolated Committed Snapshot

Use this mode whenever the result will be placed in a PR description. It does
not inspect or copy the user's working tree, so unrelated tracked or untracked
changes do not block verification or leak into the evidence.

Resolve `VERIFICATION_REF` to the exact commit intended for the PR. Create a
standalone clone rather than a linked worktree: the standalone `.git` directory
and initialized submodule remain usable after `docker cp`.

```bash
set -e
command -v git-lfs >/dev/null

VERIFICATION_REF="<pushed-branch-or-commit>"
VERIFICATION_REPO="<pushed-repository-url>"
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

git clone --no-hardlinks --no-checkout "${PROJECT_ROOT}" "${SNAPSHOT_ROOT}"
git -C "${SNAPSHOT_ROOT}" remote set-url origin "${VERIFICATION_REPO}"
git -C "${SNAPSHOT_ROOT}" checkout --detach "${VERIFIED_COMMIT}"
git -C "${SNAPSHOT_ROOT}" submodule update --init --recursive
git -C "${SNAPSHOT_ROOT}" lfs pull origin
git -C "${SNAPSHOT_ROOT}" lfs fsck
test "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD^{tree})" = "${VERIFIED_TREE}"
test -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=all)"

SOURCE_MODE="isolated committed snapshot"
SOURCE_DETAILS="commit ${VERIFIED_COMMIT}, tree ${VERIFIED_TREE}"

docker cp "${SNAPSHOT_ROOT}" "${CONTAINER}":/home/testuser/IB_Robot
docker exec "${CONTAINER}" \
  chown -R "${HOST_UID}:${HOST_GID}" /home/testuser/IB_Robot
docker exec "${CONTAINER}" bash -c '
  rm -rf /home/testuser/IB_Robot/{venv,build,install,log}
'
```

Keep `SNAPSHOT_ROOT` until both platforms have copied or independently
materialized the same commit. Remove it during final cleanup. Report both
`VERIFIED_COMMIT` for provenance and `VERIFIED_TREE` as the binding identity.

### Local Diagnosis: Copy the Current Host Workspace

Use this mode to test local, uncommitted work. It preserves committed,
uncommitted, and untracked changes from the current IB-Robot workspace. The
result is not valid PR evidence because it has no immutable tree identity.

```bash
SOURCE_MODE="local workspace copy"
SOURCE_DETAILS="${PROJECT_ROOT}"

# Record exactly what local state is being tested before setup mutates
# submodules or installs hooks.
git -C "${PROJECT_ROOT}" rev-parse HEAD
git -C "${PROJECT_ROOT}" status --short --branch

docker cp "${PROJECT_ROOT}" "${CONTAINER}":/home/testuser/IB_Robot
docker exec "${CONTAINER}" \
  chown -R "${HOST_UID}:${HOST_GID}" /home/testuser/IB_Robot

# Never reuse host-specific virtual environments or build products.
docker exec "${CONTAINER}" bash -c '
  rm -rf /home/testuser/IB_Robot/{venv,build,install,log}
'

docker exec -u testuser -w /home/testuser/IB_Robot "${CONTAINER}" \
  git status --short --branch
```

If the host workspace contains large unrelated untracked files, report that
fact before copying. Do not silently switch to a remote clone because doing so
would omit the uncommitted changes the user may intend to verify.

### Explicit Request: Clone a Clean Remote Commit or Branch

Use this mode only when the user explicitly asks to validate a pushed commit,
fork branch, or clean upstream state. Resolve and report the exact commit and
tree after cloning; branch names alone are not stable verification identities.

```bash
SOURCE_MODE="remote clean clone"
SOURCE_DETAILS="<repo_url> branch <branch>"

docker exec \
  -u testuser \
  -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -w /home/testuser \
  "${CONTAINER}" \
  git clone --branch "<branch>" --single-branch \
    "<repo_url>" /home/testuser/IB_Robot

docker exec -u testuser -w /home/testuser/IB_Robot "${CONTAINER}" \
  git status --short --branch
```

The remote-clone status must contain only the branch/tracking line before
setup. For either mode, record `git rev-parse HEAD`; for a local copy, also
preserve and report the pre-setup dirty status.

## Phase 3.5 — ROS Source-Switch Pre-Clean (conditional)

> **Trigger:** the verified change switches the ROS package source — repo URL,
> mirror, or GPG key in `scripts/install_ros.sh` or setup scripts. Execute this
> phase whenever the container already has `ros-*` packages installed
> (desktop-full image, or a quick-run container from a previous verification).

The desktop-full image ships ROS packages from the OLD source. If they remain
installed, `setup.sh`'s ROS detection reuses them and the new-source install
path in `install_ros.sh` is never exercised — the verification result would be
invalid. Removing packages is environment pre-clean, not software installation;
everything is still installed solely by `setup.sh`.

```bash
# Run as root inside the container before Phase 4.
docker exec "${CONTAINER}" bash -c '
  set -e
  ros_pkgs="$(dpkg-query -W -f="\${Package}\n" "ros-*" 2>/dev/null || true)"
  if [ -n "${ros_pkgs}" ]; then
    printf "Removing %s ros-* packages before source-switch verification\n" \
      "$(printf "%s\n" "${ros_pkgs}" | wc -l)"
    apt-get purge -y "ros-*"
    apt-get autoremove -y --purge
    rm -rf /opt/ros
  else
    echo "No ros-* packages installed; nothing to pre-clean."
  fi
  # Hard check: no ROS package and no /opt/ros residue may remain.
  test -z "$(dpkg-query -W -f="\${Package}\n" "ros-*" 2>/dev/null || true)"
  test ! -e /opt/ros/humble
  echo "ROS source-switch pre-clean complete."
'
```

Report this pre-clean in the verification result (package count removed) so
reviewers know the setup validated a full reinstall from the new source. For
complete first-install coverage (apt source setup, GPG key handling), the
[bootstrap variant](bootstrap-variant.md) with `ubuntu:22.04` is still
required in addition.

## Phase 4 — Run setup.sh

```bash
docker exec \
  -u testuser \
  -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -e IBR_LEROBOT_FORCE_REBUILD=1 \
  -e DEBIAN_FRONTEND=noninteractive \
  -w /home/testuser/IB_Robot \
  "${CONTAINER}" \
  bash -c '
    SECONDS=0
    bash scripts/setup.sh --yes > /tmp/setup.log 2>&1
    status=$?
    printf "%s\n" "${status}" > /tmp/setup.status
    printf "SETUP_ELAPSED_SECONDS=%s\n" "${SECONDS}" > /tmp/setup.time
    exit "${status}"
  '
```

Verify the status is zero and the log contains:

```
Setup complete! Run ./scripts/build.sh to build the workspace.
```

Also fail the verification as an infrastructure error if setup reports that
pip disabled the mounted cache:

```bash
if docker exec "${CONTAINER}" \
    grep -Fq "cache has been disabled" /tmp/setup.log; then
  echo "Mounted pip cache was disabled; verification timing is invalid." >&2
  exit 1
fi
```

## Phase 5 — Run build.sh

```bash
docker exec \
  -u testuser \
  -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -e DEBIAN_FRONTEND=noninteractive \
  -w /home/testuser/IB_Robot \
  "${CONTAINER}" \
  bash -c '
    SECONDS=0
    bash scripts/build.sh > /tmp/build.log 2>&1
    status=$?
    printf "%s\n" "${status}" > /tmp/build.status
    printf "BUILD_ELAPSED_SECONDS=%s\n" "${SECONDS}" > /tmp/build.time
    exit "${status}"
  '
```

Verify the status is zero and the log contains:

```
Build complete. Source with: source install/setup.sh
```

## Phase 6 — Inspect, Report Timing, and Clean Up

```bash
# Collect source, status, timing, commit/tree provenance, and all ERROR lines with context.
printf 'source_mode=%s\n' "${SOURCE_MODE}"
printf 'source_details=%s\n' "${SOURCE_DETAILS}"
docker exec "${CONTAINER}" bash -c '
  cat /tmp/setup.status /tmp/setup.time
  cat /tmp/build.status /tmp/build.time
  git -C /home/testuser/IB_Robot rev-parse HEAD HEAD^{tree}
  grep -n -E "ERROR|Error|error" /tmp/setup.log || true
  grep -n -E "ERROR|Error|error|Failed|failed" /tmp/build.log || true
'

TOTAL_SECONDS=$(( $(date +%s) - TOTAL_START ))
printf 'image_prepare=%ss\n' "${IMAGE_SECONDS}"
printf 'container_start=%ss\n' "${CONTAINER_SECONDS}"
docker exec "${CONTAINER}" cat /tmp/setup.time /tmp/build.time
printf 'total=%ss\n' "${TOTAL_SECONDS}"

# Clean up
docker stop "${CONTAINER}" && docker rm "${CONTAINER}"
if [ "${SOURCE_MODE}" = "isolated committed snapshot" ]; then
  rm -rf -- "${SNAPSHOT_ROOT}"
  SNAPSHOT_ROOT=""
fi
```

> **错误报告要求**：必须逐条列出所有 ERROR 行，并按 Verification Discipline 中的分类标准标注 Fatal / Non-fatal。不能只给 ERROR 行数不给内容。

Report both totals when relevant:

- Cold total including the first image pull.
- Steady-state total using an already-present desktop-full image.

Also record the pip cache size before and after the run. A warm cache result
must not be presented as a cold-cache baseline.

The final report must explicitly state one source mode:

- `Source: isolated committed snapshot`, with the exact commit and tree. This
  is the normal PR-evidence mode; the PR's canonical identity is
  `## Docker Verification` block (mode, Verified inputs, Tested source tree, Docker environment).
- `Source: local workspace copy`, with the host commit and whether the copied
  workspace contained uncommitted or untracked changes. Mark it local-only and
  do not put it in a PR as tree-bound evidence.
- `Source: remote clean clone`, with the exact repository URL, branch, and
  tested commit and tree.

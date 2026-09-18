#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IB_ROBOT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

ensure_dir() {
    mkdir -p "$1"
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        log_error "Missing required command: $1"
        exit 1
    fi
}

OH_ROOT="${OH_ROOT:-}"
OH_DOWNLOAD_ROOT="${OH_DOWNLOAD_ROOT:-}"
OH_CUSTOM_ROOT="${OH_CUSTOM_ROOT:-}"
OH_CUSTOM_WS="${OH_CUSTOM_WS:-}"
OH_CUSTOM_SRC=""
OH_CUSTOM_TOOLCHAIN_ROOT="${OH_CUSTOM_TOOLCHAIN_ROOT:-}"
OH_CUSTOM_SDK_TAR_GLOB="${OH_CUSTOM_SDK_TAR_GLOB:-}"
OH_CUSTOM_IMAGE="${OH_CUSTOM_IMAGE:-voxelsky/ohos-ros-humble-builder:v0.1.5}"
OH_CUSTOM_CONTAINER_NAME="${OH_CUSTOM_CONTAINER_NAME:-ibrobot-oh-build}"
OH_CUSTOM_PREFIX="${OH_CUSTOM_PREFIX:-/data/roboframe/install}"
OH_BOARD_ROS_PREFIX="${OH_BOARD_ROS_PREFIX:-/data/install}"
OH_CUSTOM_CPU="${OH_CUSTOM_CPU:-aarch64}"
OH_CUSTOM_SYSDEPS_TAR_GLOB="${OH_CUSTOM_SYSDEPS_TAR_GLOB:-}"
OH_CUSTOM_ROS2_BASE_REPO="${OH_CUSTOM_ROS2_BASE_REPO:-}"
OH_CUSTOM_VERSION_REPO="${OH_CUSTOM_VERSION_REPO:-}"
OH_CUSTOM_HUMBLE_TAR_GLOB="${OH_CUSTOM_HUMBLE_TAR_GLOB:-}"

USE_SUDO=0
DRY_RUN=0
PULL_IMAGE=1
declare -a PACKAGES=(
    "ibrobot_msgs"
    "tensormsg"
    "embodied_common"
    "voice_asr_service"
    "inference_manifest"
    "ibrobot_tracing"
    "robot_config"
    "robot_description"
    "inference_service"
    "hardware_mock"
    "action_dispatch"
    "so101_hardware"
    "task_dispatch"
    "robot_moveit"
    "dataset_tools"
)
declare -a COLCON_ARGS=()
declare -a CMAKE_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/openharmony/build_roboframe_oh.sh [options]

Prepare and run the official OpenHarmony ROS custom-package cross-build for the
RoboFrame (IB_Robot) inference workspace.

Options:
  --oh-root <dir>          Unified external OpenHarmony root (recommended)
                           Defaults to deriving downloads/ and custom_build_root/ from it
  --root <dir>             Custom build root (default: <OH_ROOT>/custom_build_root)
  --workspace <dir>        Custom ROS workspace root (default: <root>/ibrobot_oh_ws)
  --toolchain-root <dir>   OHOS SDK root containing 18/native
  --sdk-tar <path>         Official OH ROS SDK tarball providing the 18/ tree
                           (default: <OH_ROOT>/downloads/sdk/ohos-sdk-18-linux-aarch64-*.tar.gz)
  --sysdeps-tar <path>     OH sysdeps tarball used to augment the SDK sysroot
                           (default: <OH_ROOT>/downloads/sysdeps/ohos-*-sysdeps-*.tar.gz)
  --humble-tar <path>      OH ROS 2 Humble runtime tarball
                           (default: <OH_ROOT>/downloads/runtime/ohos-humble-build-aarch64-*.tar.gz)
  --custom-prefix <path>   Final on-device install prefix (default: /data/roboframe/install)
  --cpu <arch>             Target OHOS CPU (default: aarch64)
  --image <image>          Builder image (default: voxelsky/ohos-ros-humble-builder:v0.1.5)
  --container-name <name>  Container name (default: ibrobot-oh-build)
  --packages <csv>         Comma-separated package list
  --colcon-args <...>      Extra arguments passed through to build-ros-humble
  --cmk-args <...>         Extra CMake arguments passed through to build-ros-humble
  --sudo                   Run docker via sudo
  --no-pull                Skip docker pull if image is missing locally
  --dry-run                Print the docker command without running it
  -h, --help               Show this help
EOF
}

docker_cmd() {
    if [[ "${USE_SUDO}" -eq 1 ]]; then
        sudo docker "$@"
    else
        docker "$@"
    fi
}

abspath() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${PWD}" "${path}"
    fi
}

apply_layout_defaults() {
    if [[ -n "${OH_ROOT}" ]]; then
        [[ -z "${OH_DOWNLOAD_ROOT}" ]] && OH_DOWNLOAD_ROOT="${OH_ROOT}/downloads"
        [[ -z "${OH_CUSTOM_ROOT}" ]] && OH_CUSTOM_ROOT="${OH_ROOT}/custom_build_root"
    fi

    if [[ -z "${OH_CUSTOM_ROOT}" ]]; then
        log_error "Missing OpenHarmony build root."
        log_error "Pass --oh-root <dir> (recommended), or set OH_CUSTOM_ROOT / --root explicitly."
        exit 1
    fi

    [[ -z "${OH_CUSTOM_WS}" ]] && OH_CUSTOM_WS="${OH_CUSTOM_ROOT}/ibrobot_oh_ws"
    [[ -z "${OH_CUSTOM_TOOLCHAIN_ROOT}" ]] && OH_CUSTOM_TOOLCHAIN_ROOT="${OH_CUSTOM_ROOT}/ohos-robot-toolchain"
    [[ -z "${OH_CUSTOM_ROS2_BASE_REPO}" ]] && OH_CUSTOM_ROS2_BASE_REPO="${OH_CUSTOM_ROOT}/ros_ros2_base"
    [[ -z "${OH_CUSTOM_VERSION_REPO}" ]] && OH_CUSTOM_VERSION_REPO="${OH_CUSTOM_ROOT}/version"

    if [[ -n "${OH_DOWNLOAD_ROOT}" ]]; then
        [[ -z "${OH_CUSTOM_SDK_TAR_GLOB}" ]] && OH_CUSTOM_SDK_TAR_GLOB="${OH_DOWNLOAD_ROOT}/sdk/ohos-sdk-18-linux-aarch64-*.tar.gz"
        [[ -z "${OH_CUSTOM_SYSDEPS_TAR_GLOB}" ]] && OH_CUSTOM_SYSDEPS_TAR_GLOB="${OH_DOWNLOAD_ROOT}/sysdeps/ohos-*-sysdeps-*.tar.gz"
        [[ -z "${OH_CUSTOM_HUMBLE_TAR_GLOB}" ]] && OH_CUSTOM_HUMBLE_TAR_GLOB="${OH_DOWNLOAD_ROOT}/runtime/ohos-humble-build-aarch64-*.tar.gz"
    fi

    OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
}

normalize_paths() {
    [[ -n "${OH_ROOT}" ]] && OH_ROOT="$(abspath "${OH_ROOT}")"
    [[ -n "${OH_DOWNLOAD_ROOT}" ]] && OH_DOWNLOAD_ROOT="$(abspath "${OH_DOWNLOAD_ROOT}")"
    OH_CUSTOM_ROOT="$(abspath "${OH_CUSTOM_ROOT}")"
    OH_CUSTOM_WS="$(abspath "${OH_CUSTOM_WS}")"
    OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
    OH_CUSTOM_TOOLCHAIN_ROOT="$(abspath "${OH_CUSTOM_TOOLCHAIN_ROOT}")"
    OH_CUSTOM_ROS2_BASE_REPO="$(abspath "${OH_CUSTOM_ROS2_BASE_REPO}")"
    OH_CUSTOM_VERSION_REPO="$(abspath "${OH_CUSTOM_VERSION_REPO}")"
}

ensure_repo_checkout() {
    local repo_url="$1"
    local dest_dir="$2"

    if [[ -d "${dest_dir}/.git" ]]; then
        return
    fi

    log_info "Cloning $(basename "${dest_dir}") into ${dest_dir}..."
    git clone --depth 1 "${repo_url}" "${dest_dir}"
}

ensure_humble_install() {
    local tar_path

    if [[ -d "${OH_CUSTOM_ROOT}/install" ]]; then
        return
    fi

    tar_path="$(compgen -G "${OH_CUSTOM_HUMBLE_TAR_GLOB}" | sort | tail -n 1 || true)"
    if [[ -z "${tar_path}" ]]; then
        log_error "Cannot find a host ohos-humble-build tarball matching:"
        log_error "  ${OH_CUSTOM_HUMBLE_TAR_GLOB}"
        exit 1
    fi

    log_info "Extracting $(basename "${tar_path}") into ${OH_CUSTOM_ROOT}..."
    tar -zxpf "${tar_path}" -C "${OH_CUSTOM_ROOT}"
}

ensure_workspace_links() {
    ensure_dir "${OH_CUSTOM_SRC}"

    for pkg in "${PACKAGES[@]}"; do
        local src_pkg="${IB_ROBOT_ROOT}/src/${pkg}"
        local dst_pkg="${OH_CUSTOM_SRC}/${pkg}"
        if [[ ! -e "${src_pkg}" ]]; then
            log_error "Source package not found: ${src_pkg}"
            exit 1
        fi
        rm -rf "${dst_pkg}"
        cp -a "${src_pkg}" "${dst_pkg}"
    done
}

ensure_lerobot_submodule() {
    local lerobot_dir="${IB_ROBOT_ROOT}/libs/lerobot"

    if git -C "${lerobot_dir}" rev-parse --git-dir >/dev/null 2>&1 && [[ -d "${lerobot_dir}/src" ]]; then
        return
    fi

    if ! git -C "${IB_ROBOT_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
        log_warn "${IB_ROBOT_ROOT} is not a git checkout; skipping local libs/lerobot submodule init." >&2
        return
    fi

    log_info "Initializing libs/lerobot submodule for OpenHarmony runtime staging..."
    git -C "${IB_ROBOT_ROOT}" submodule update --init --recursive libs/lerobot

    if ! git -C "${lerobot_dir}" rev-parse --git-dir >/dev/null 2>&1 || [[ ! -d "${lerobot_dir}/src" ]]; then
        log_error "LeRobot source tree still missing after submodule init: ${lerobot_dir}/src"
        exit 1
    fi
}

resolve_openharmony_lerobot_patch_stack() {
    local index_file="${IB_ROBOT_ROOT}/third_party/patches/lerobot/INDEX.yaml"
    local active_tag=""

    if [[ ! -f "${index_file}" ]]; then
        log_error "LeRobot patch index not found: ${index_file}"
        exit 1
    fi

    active_tag="$(awk -F': *' '/^active_tag:/ { print $2; exit }' "${index_file}")"
    if [[ -z "${active_tag}" ]]; then
        log_error "Could not resolve active_tag from ${index_file}"
        exit 1
    fi

    LEROBOT_OH_PATCH_DIR="${IB_ROBOT_ROOT}/third_party/patches/lerobot/${active_tag}"
    LEROBOT_OH_PATCH_SERIES="${LEROBOT_OH_PATCH_DIR}/series.openharmony-5.1.0-musl.txt"
    LEROBOT_OH_PATCH_MANIFEST="${LEROBOT_OH_PATCH_DIR}/manifest.yaml"
    LEROBOT_OH_UPSTREAM_REPO="$(awk '
        /^upstream:/ { in_upstream=1; next }
        in_upstream && /^[^[:space:]]/ { in_upstream=0 }
        in_upstream && $1 == "repo:" { print $2; exit }
    ' "${LEROBOT_OH_PATCH_MANIFEST}" 2>/dev/null || true)"
    LEROBOT_OH_BASE_COMMIT="$(awk '
        /^lerobot_commit_range:/ { in_range=1; next }
        in_range && /^[^[:space:]]/ { in_range=0 }
        in_range && $1 == "min:" { print $2; exit }
    ' "${LEROBOT_OH_PATCH_MANIFEST}")"

    if [[ ! -d "${LEROBOT_OH_PATCH_DIR}" || ! -f "${LEROBOT_OH_PATCH_SERIES}" || ! -f "${LEROBOT_OH_PATCH_MANIFEST}" ]]; then
        log_error "OpenHarmony lerobot patch stack is incomplete under ${LEROBOT_OH_PATCH_DIR}"
        exit 1
    fi
    if [[ -z "${LEROBOT_OH_BASE_COMMIT}" ]]; then
        log_error "Could not resolve lerobot base commit from ${LEROBOT_OH_PATCH_MANIFEST}"
        exit 1
    fi
    if [[ -z "${LEROBOT_OH_UPSTREAM_REPO}" && -f "${IB_ROBOT_ROOT}/.gitmodules" ]]; then
        LEROBOT_OH_UPSTREAM_REPO="$(git config -f "${IB_ROBOT_ROOT}/.gitmodules" --get submodule.libs/lerobot.url || true)"
    fi
    if [[ -z "${LEROBOT_OH_UPSTREAM_REPO}" ]]; then
        log_error "Could not resolve lerobot upstream repo from ${LEROBOT_OH_PATCH_MANIFEST} or .gitmodules"
        exit 1
    fi
}

resolve_openharmony_lerobot_repo_source() {
    local lerobot_dir="${IB_ROBOT_ROOT}/libs/lerobot"

    ensure_lerobot_submodule

    if git -C "${lerobot_dir}" rev-parse --git-dir >/dev/null 2>&1 && [[ -d "${lerobot_dir}/src" ]]; then
        printf '%s\n' "${lerobot_dir}"
        return
    fi

    log_info "Using upstream lerobot repo for OpenHarmony runtime staging: ${LEROBOT_OH_UPSTREAM_REPO}" >&2
    printf '%s\n' "${LEROBOT_OH_UPSTREAM_REPO}"
}

prepare_openharmony_lerobot_runtime_src() {
    local stage_root="${OH_CUSTOM_ROOT}/.lerobot_openharmony_runtime"
    local repo_dir="${stage_root}/repo"
    local lerobot_repo_source=""
    local git_user_name="${IBR_LEROBOT_GIT_USER_NAME:-IB Robot Setup}"
    local git_user_email="${IBR_LEROBOT_GIT_USER_EMAIL:-ibrobot@example.invalid}"
    local patch_file=""

    resolve_openharmony_lerobot_patch_stack
    lerobot_repo_source="$(resolve_openharmony_lerobot_repo_source)"

    rm -rf "${stage_root}"
    ensure_dir "${stage_root}"

    log_info "Preparing OpenHarmony-patched LeRobot runtime staging tree..." >&2
    if [[ -d "${lerobot_repo_source}" ]]; then
        git clone --local --no-checkout "${lerobot_repo_source}" "${repo_dir}" >/dev/null
    else
        git clone --no-checkout "${lerobot_repo_source}" "${repo_dir}" >/dev/null
    fi
    git -C "${repo_dir}" checkout --detach "${LEROBOT_OH_BASE_COMMIT}" >/dev/null

    while IFS= read -r patch_file; do
        [[ -z "${patch_file}" || "${patch_file}" == \#* ]] && continue
        log_info "Applying OpenHarmony lerobot runtime patch ${patch_file}..." >&2
        git -C "${repo_dir}" \
            -c "user.name=${git_user_name}" \
            -c "user.email=${git_user_email}" \
            am "${LEROBOT_OH_PATCH_DIR}/${patch_file}" >/dev/null
    done < "${LEROBOT_OH_PATCH_SERIES}"

    printf '%s\n' "${repo_dir}/src"
}

stage_lerobot_runtime() {
    local install_root="$1"
    local lerobot_dst="${install_root}/lerobot/src"
    local lerobot_src=""

    lerobot_src="$(prepare_openharmony_lerobot_runtime_src)"

    rm -rf "${install_root}/lerobot"
    ensure_dir "${lerobot_dst}"
    cp -a "${lerobot_src}/." "${lerobot_dst}/"
}

rewrite_runtime_prefix_chain() {
    local install_root="$1"
    local file
    local package_setup

    for file in \
        "${install_root}/setup.sh" \
        "${install_root}/setup.bash" \
        "${install_root}/setup.zsh" \
        "${install_root}/setup.ps1"; do
        [[ -f "${file}" ]] || continue
        sed -i "s|/mnt/ohos/tmp/install|${OH_BOARD_ROS_PREFIX}|g" "${file}"
    done

    while IFS= read -r -d '' package_setup; do
        sed -i "s|/mnt/ohos/tmp/install|${OH_BOARD_ROS_PREFIX}|g" "${package_setup}"
    done < <(find "${install_root}" \( -path '*/local_setup.sh' -o -path '*/local_setup.bash' -o -path '*/local_setup.zsh' -o -path '*/package.sh' \) -type f -print0)

    while IFS= read -r -d '' file; do
        sed -i "s|/mnt/ohos/tmp/install|${OH_BOARD_ROS_PREFIX}|g" "${file}"
    done < <(find "${install_root}" -path '*/share/ament_index/resource_index/parent_prefix_path/*' -type f -print0)
}

append_lerobot_runtime_hook() {
    local install_root="$1"
    local file
    local marker="# ibrobot openharmony lerobot runtime path"

    for file in \
        "${install_root}/setup.sh" \
        "${install_root}/setup.bash" \
        "${install_root}/setup.zsh"; do
        [[ -f "${file}" ]] || continue
        if grep -qF "${marker}" "${file}"; then
            continue
        fi
        cat <<EOF >> "${file}"

${marker}
_ibrobot_lerobot_src="${OH_CUSTOM_PREFIX}/lerobot/src"
if [ -d "\$_ibrobot_lerobot_src" ]; then
  case ":\${PYTHONPATH:-}:" in
    *":\$_ibrobot_lerobot_src:"*) ;;
    *) export PYTHONPATH="\$_ibrobot_lerobot_src\${PYTHONPATH:+:\$PYTHONPATH}" ;;
  esac
fi
unset _ibrobot_lerobot_src
EOF
    done

    while IFS= read -r -d '' file; do
        [[ -f "${file}" ]] || continue
        if grep -qF "${marker}" "${file}"; then
            continue
        fi
        cat <<EOF >> "${file}"

${marker}
_ibrobot_lerobot_src="${OH_CUSTOM_PREFIX}/lerobot/src"
if [ -d "\$_ibrobot_lerobot_src" ]; then
  case ":\${PYTHONPATH:-}:" in
    *":\$_ibrobot_lerobot_src:"*) ;;
    *) export PYTHONPATH="\$_ibrobot_lerobot_src\${PYTHONPATH:+:\$PYTHONPATH}" ;;
  esac
fi
unset _ibrobot_lerobot_src
EOF
    done < <(find "${install_root}" -path '*/local_setup.sh' -type f -print0)
}

rewrite_inference_entrypoints_for_board_runtime() {
    local install_root="$1"
    local rel_path=""
    local module_name=""
    local script_path=""

    while IFS='|' read -r rel_path module_name; do
        for script_path in "${install_root}/${rel_path}" "${install_root}"/*/${rel_path}; do
            [[ -f "${script_path}" ]] || continue

            cat <<EOF > "${script_path}"
#!/system/bin/sh
# Auto-generated by build_roboframe_oh.sh — do not edit.
# Python deps (pysite) are deployed separately from lerobot_deps releases:
#   https://atomgit.com/openharmony-robot/lerobot_deps/releases
ROBOFRAME_ROOT="${OH_CUSTOM_PREFIX%/install}"
ROS_HOME_ROOT="/data/local/tmp/ros_home"
ROS_LOG_ROOT="/data/local/tmp/ros_logs"

mkdir -p "\${ROS_HOME_ROOT}" "\${ROS_LOG_ROOT}" >/dev/null 2>&1 || true

export HOME="\${ROS_HOME_ROOT}"
export ROS_LOG_DIR="\${ROS_LOG_ROOT}"
export PYTHONPATH="\${ROBOFRAME_ROOT}/pysite:${OH_CUSTOM_PREFIX}/lerobot/src:${OH_CUSTOM_PREFIX}/dataset_tools/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/inference_manifest/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/inference_service/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/robot_config/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/ibrobot_tracing/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/tensormsg/lib/python3.12/site-packages:${OH_CUSTOM_PREFIX}/ibrobot_msgs/lib/python3.12/site-packages:${OH_BOARD_ROS_PREFIX}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:/sys_prod/robot/install/lib/python3.12/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="${OH_CUSTOM_PREFIX}/dataset_tools/lib:${OH_CUSTOM_PREFIX}/inference_service/lib:${OH_CUSTOM_PREFIX}/robot_config/lib:${OH_CUSTOM_PREFIX}/tensormsg/lib:${OH_CUSTOM_PREFIX}/ibrobot_msgs/lib:${OH_BOARD_ROS_PREFIX}/lib:/sys_prod/robot/out/lib:/sys_prod/robot/install/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LD_PRELOAD="/sys_prod/robot/out/lib/libpython3.12.so.1.0\${LD_PRELOAD:+:\$LD_PRELOAD}"

exec python3 -m ${module_name} "\$@"
EOF
            chmod +x "${script_path}"
        done
    done <<'EOF'
lib/inference_service/pipeline_policy_node|inference_service.pipeline_policy_node
lib/inference_service/pure_inference_node|inference_service.pure_inference_node
lib/dataset_tools/policy_eval|dataset_tools.policy_eval
EOF
}

stage_board_scripts() {
    local install_root="${OH_CUSTOM_WS}/install"

    log_info "Staging board scripts (robooh env, Houmo env, setup_sshd) ..."

    local scripts_dest="${install_root}/../scripts"
    mkdir -p "${scripts_dest}"

    local src
    for src in "${IB_ROBOT_ROOT}/scripts/robooh_1.0.1.env" \
               "${IB_ROBOT_ROOT}/scripts/setup_sshd.sh"; do
        if [[ -f "${src}" ]]; then
            cp -f "${src}" "${scripts_dest}/"
            log_info "  staged $(basename "${src}")"
        else
            log_warn "  missing: ${src}"
        fi
    done

    mkdir -p "${scripts_dest}/setup"
    cp -f "${IB_ROBOT_ROOT}/scripts/setup/houmo_hmm_env.sh" "${scripts_dest}/setup/"
    log_info "  staged setup/houmo_hmm_env.sh"
}

postprocess_runtime_bundle() {
    local install_root="${OH_CUSTOM_WS}/install"

    if [[ ! -d "${install_root}" ]]; then
        log_error "Missing install tree after build: ${install_root}"
        exit 1
    fi

    log_info "Post-processing OpenHarmony runtime bundle..."
    stage_lerobot_runtime "${install_root}"
    rewrite_runtime_prefix_chain "${install_root}"
    append_lerobot_runtime_hook "${install_root}"
    rewrite_inference_entrypoints_for_board_runtime "${install_root}"
    stage_board_scripts
}

normalize_runtime_bundle_ownership() {
    local host_uid
    local host_gid

    host_uid="$(id -u)"
    host_gid="$(id -g)"

    docker_cmd run --rm \
        -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
        "${OH_CUSTOM_IMAGE}" \
        sh -c "chown -R ${host_uid}:${host_gid} /mnt/ohos/ibrobot_oh_ws/install /mnt/ohos/ibrobot_oh_ws/build /mnt/ohos/ibrobot_oh_ws/log || true"
}

ensure_toolchain_root() {
    local sdk_tar=""

    ensure_dir "${OH_CUSTOM_TOOLCHAIN_ROOT}"

    if [[ ! -d "${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native" ]]; then
        sdk_tar="$(compgen -G "${OH_CUSTOM_SDK_TAR_GLOB}" | sort | tail -n 1 || true)"
        if [[ -n "${sdk_tar}" ]]; then
            log_info "Extracting official OH ROS SDK $(basename "${sdk_tar}") into ${OH_CUSTOM_TOOLCHAIN_ROOT}..."
            tar -zxpf "${sdk_tar}" -C "${OH_CUSTOM_TOOLCHAIN_ROOT}"
        fi
    fi

    if [[ ! -d "${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native" ]]; then
        log_error "Missing OHOS SDK under ${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native"
        log_error "Tried SDK archive glob:"
        log_error "  ${OH_CUSTOM_SDK_TAR_GLOB}"
        log_error "Place the downloaded OHOS ROS SDK there, set OH_CUSTOM_SDK_TAR_GLOB, or pass --sdk-tar."
        exit 1
    fi
}

ensure_sysdeps_overlay() {
    local sysroot_usr="${OH_CUSTOM_TOOLCHAIN_ROOT}/18/native/sysroot/usr"
    local sysdeps_tar=""
    local stage_dir="${OH_CUSTOM_ROOT}/.sysdeps_overlay"

    # Check if already overlayed (tinyxml2 is a good canary — it's in sysdeps but not in the base SDK)
    if [[ -f "${sysroot_usr}/include/tinyxml2.h" && \
          -f "${sysroot_usr}/lib/libtinyxml2.so" && \
          -f "${sysroot_usr}/lib/libssl.so" ]]; then
        return
    fi

    sysdeps_tar="$(compgen -G "${OH_CUSTOM_SYSDEPS_TAR_GLOB}" | sort | tail -n 1 || true)"
    if [[ -z "${sysdeps_tar}" ]]; then
        log_error "Cannot find an OH sysdeps tarball matching:"
        log_error "  ${OH_CUSTOM_SYSDEPS_TAR_GLOB}"
        log_error "Set OH_CUSTOM_SYSDEPS_TAR_GLOB or pass --sysdeps-tar to point at ohos-*-sysdeps-*.tar.gz."
        exit 1
    fi

    log_info "Overlaying full sysdeps from $(basename "${sysdeps_tar}") into the SDK sysroot..."
    rm -rf "${stage_dir}"
    mkdir -p "${stage_dir}" "${sysroot_usr}/include" "${sysroot_usr}/lib"

    # The sysdeps tarball has two layouts:
    #   Full layout:  out/include/...  out/lib/...
    #   Extract layout: include/...  lib/...  (already stripped)
    # Detect layout and extract everything
    if tar tzf "${sysdeps_tar}" 2>/dev/null | grep -q "^out/"; then
        tar -xzf "${sysdeps_tar}" -C "${stage_dir}" out/include out/lib
        cp -a "${stage_dir}/out/include/"* "${sysroot_usr}/include/"
        cp -a "${stage_dir}/out/lib/"* "${sysroot_usr}/lib/"
    else
        tar -xzf "${sysdeps_tar}" -C "${stage_dir}" include lib
        cp -a "${stage_dir}/include/"* "${sysroot_usr}/include/"
        cp -a "${stage_dir}/lib/"* "${sysroot_usr}/lib/"
    fi
    rm -rf "${stage_dir}"
    log_info "  Done. Key libraries: $(ls "${sysroot_usr}/lib/"lib{tinyxml2,ssl,crypto,z,python3.12}*.so 2>/dev/null | wc -l) files"
}

prepare_root_layout() {
    ensure_dir "${OH_CUSTOM_ROOT}"
    ensure_dir "${OH_CUSTOM_TOOLCHAIN_ROOT}"
    ensure_repo_checkout "https://gitcode.com/openharmony-robot/ros_ros2_base.git" "${OH_CUSTOM_ROS2_BASE_REPO}"
    ensure_repo_checkout "https://gitcode.com/openharmony-robot/version.git" "${OH_CUSTOM_VERSION_REPO}"
    ensure_humble_install
    ensure_workspace_links
    ensure_toolchain_root
    ensure_sysdeps_overlay
}

ensure_builder_image() {
    if docker_cmd image inspect "${OH_CUSTOM_IMAGE}" >/dev/null 2>&1; then
        return
    fi

    if [[ "${PULL_IMAGE}" -eq 0 ]]; then
        log_error "Docker image not found locally: ${OH_CUSTOM_IMAGE}"
        exit 1
    fi

    log_info "Pulling builder image ${OH_CUSTOM_IMAGE}..."
    docker_cmd pull "${OH_CUSTOM_IMAGE}"
}

build_command_string() {
    local package_args
    local colcon_str=""
    local cmake_str=""

    package_args="$(IFS=,; echo "${PACKAGES[*]}")"

    if [[ "${#COLCON_ARGS[@]}" -gt 0 ]]; then
        # shellcheck disable=SC2206
        colcon_str=" --colcon-args ${COLCON_ARGS[*]}"
    fi
    if [[ "${#CMAKE_ARGS[@]}" -gt 0 ]]; then
        # shellcheck disable=SC2206
        cmake_str=" --cmk-args ${CMAKE_ARGS[*]}"
    fi

    cat <<EOF
set -euo pipefail
export OHOS_CPU=${OH_CUSTOM_CPU}
export OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
build-ros-humble --custom \
  --wd /mnt/ohos/tmp/ibrobot_oh_ws \
  --custom-prefix ${OH_CUSTOM_PREFIX} \
  --colcon-args --packages-select ${package_args//,/ }${colcon_str}${cmake_str}
EOF
}

run_builder() {
    local inner_cmd
    inner_cmd="$(build_command_string)"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "docker run --rm -it -e WS_ROOT=/mnt/ohos/tmp -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 --name ${OH_CUSTOM_CONTAINER_NAME} -v ${OH_CUSTOM_ROOT}:/mnt/ohos -v ${OH_CUSTOM_ROOT}:/mnt/ohos/tmp ${OH_CUSTOM_IMAGE} bash -lc '<build command>'"
        echo ""
        echo "${inner_cmd}"
        return
    fi

    docker_cmd run --rm -i \
        -e WS_ROOT=/mnt/ohos/tmp \
        -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 \
        --name "${OH_CUSTOM_CONTAINER_NAME}" \
        -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
        -v "${OH_CUSTOM_ROOT}:/mnt/ohos/tmp" \
        "${OH_CUSTOM_IMAGE}" \
        bash -lc "${inner_cmd}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --oh-root)
            shift
            OH_ROOT="$1"
            ;;
        --root)
            shift
            OH_CUSTOM_ROOT="$1"
            OH_CUSTOM_WS="${OH_CUSTOM_ROOT}/ibrobot_oh_ws"
            OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
            OH_CUSTOM_TOOLCHAIN_ROOT="${OH_CUSTOM_ROOT}/ohos-robot-toolchain"
            OH_CUSTOM_ROS2_BASE_REPO="${OH_CUSTOM_ROOT}/ros_ros2_base"
            OH_CUSTOM_VERSION_REPO="${OH_CUSTOM_ROOT}/version"
            ;;
        --workspace)
            shift
            OH_CUSTOM_WS="$1"
            OH_CUSTOM_SRC="${OH_CUSTOM_WS}/src"
            ;;
        --toolchain-root)
            shift
            OH_CUSTOM_TOOLCHAIN_ROOT="$1"
            ;;
        --sdk-tar)
            shift
            OH_CUSTOM_SDK_TAR_GLOB="$1"
            ;;
        --sysdeps-tar)
            shift
            OH_CUSTOM_SYSDEPS_TAR_GLOB="$1"
            ;;
        --humble-tar)
            shift
            OH_CUSTOM_HUMBLE_TAR_GLOB="$1"
            ;;
        --custom-prefix)
            shift
            OH_CUSTOM_PREFIX="$1"
            ;;
        --cpu)
            shift
            OH_CUSTOM_CPU="$1"
            ;;
        --image)
            shift
            OH_CUSTOM_IMAGE="$1"
            ;;
        --container-name)
            shift
            OH_CUSTOM_CONTAINER_NAME="$1"
            ;;
        --packages)
            shift
            IFS=',' read -r -a PACKAGES <<<"$1"
            ;;
        --colcon-args)
            shift
            while [[ $# -gt 0 && "$1" != --cmk-args && "$1" != --oh-root && "$1" != --root && "$1" != --workspace && "$1" != --toolchain-root && "$1" != --sdk-tar && "$1" != --sysdeps-tar && "$1" != --humble-tar && "$1" != --custom-prefix && "$1" != --cpu && "$1" != --image && "$1" != --container-name && "$1" != --packages && "$1" != --sudo && "$1" != --no-pull && "$1" != --dry-run && "$1" != -h && "$1" != --help ]]; do
                COLCON_ARGS+=("$1")
                shift
            done
            continue
            ;;
        --cmk-args)
            shift
            while [[ $# -gt 0 && "$1" != --colcon-args && "$1" != --oh-root && "$1" != --root && "$1" != --workspace && "$1" != --toolchain-root && "$1" != --sdk-tar && "$1" != --sysdeps-tar && "$1" != --humble-tar && "$1" != --custom-prefix && "$1" != --cpu && "$1" != --image && "$1" != --container-name && "$1" != --packages && "$1" != --sudo && "$1" != --no-pull && "$1" != --dry-run && "$1" != -h && "$1" != --help ]]; do
                CMAKE_ARGS+=("$1")
                shift
            done
            continue
            ;;
        --sudo)
            USE_SUDO=1
            ;;
        --no-pull)
            PULL_IMAGE=0
            ;;
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

require_cmd git
require_cmd awk
require_cmd tar
require_cmd docker

apply_layout_defaults
normalize_paths
prepare_root_layout
ensure_builder_image
run_builder
normalize_runtime_bundle_ownership
postprocess_runtime_bundle

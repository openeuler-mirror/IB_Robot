#!/bin/bash
# setup.sh - Workspace setup script for ROS 2 Humble
# Handles repository import, dependency installation, and environment setup
#
# Usage:
#   ./scripts/setup.sh                               # Interactive mode
#   ./scripts/setup.sh --yes                         # Auto-yes mode
#   ./scripts/setup.sh --git-http                    # Use HTTP instead of SSH for git remotes
#   ./scripts/setup.sh --skip-submodules             # Keep current submodule state
#   ./scripts/setup.sh --skip-system-deps            # Skip ROS/system dependency installation
#   ./scripts/setup.sh --skip-python                 # Skip Python venv/dependency setup
#   ./scripts/setup.sh --skip-verify                 # Skip final ROS/Python verification
#   ./scripts/setup.sh --platform <id>               # Override detected platform
#   ./scripts/setup.sh --help                        # Show help
#
# Auto-yes defaults:
#   - Submodule init:  initialize all submodules (option 1)
#   - Fork setup:      skipped
#   - Other prompts:   confirmed automatically
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-$(pwd)}"
PARALLEL_WORKERS=$(($(nproc) / 2))
AUTO_YES=false
VERBOSE=${VERBOSE:-false}
DRY_RUN=${DRY_RUN:-false}
GIT_HTTP=false
USE_SUDO=true
SUMMARY=()
SETUP_PLATFORM_ID="unknown"
SETUP_OS_ID="unknown"
SETUP_OS_VERSION="unknown"
SETUP_OS_PRETTY_NAME="unknown"
SETUP_ARCH="unknown"
SETUP_KERNEL="unknown"
SETUP_PACKAGE_MANAGER="unknown"
SETUP_ACTIVE_VENV=""
SETUP_PYTHONPATH=""
SETUP_SHELL_PYTHON_BIN=""
SETUP_SHELL_PYTHON_VERSION=""
SETUP_BOOTSTRAP_PYTHON_BIN=""
SETUP_BOOTSTRAP_PYTHON_VERSION=""
SETUP_ROS_SETUP_PATH=""
SETUP_GPU_SUMMARY="unknown"
SETUP_RAM_SUMMARY="unknown"
SETUP_DISK_FREE_SUMMARY="unknown"
SETUP_ROS_SUMMARY="unknown"
GUM_BIN=""
GUM_TMPDIR=""
GUM_VERSION="0.17.0"
USE_GUM=false
SUDO_AUTH_READY=false
PLATFORM_OVERRIDE=""
SKIP_SUBMODULES=false
SKIP_SYSTEM_DEPS=false
SKIP_PYTHON=false
SKIP_VERIFY=false
CURRENT_STAGE="initializing"
SYSTEM_DEPS_STATUS="pending"
PYTHON_ENV_STATUS="pending"
SETUP_ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/rosdistro/index-v4.yaml}"
SETUP_ROSDEP_DEFAULT_SOURCES_URL="${SETUP_ROSDEP_DEFAULT_SOURCES_URL:-https://mirrors.tuna.tsinghua.edu.cn/github-raw/ros/rosdistro/master/rosdep/sources.list.d/20-default.list}"
SETUP_ROSDEP_DEFAULT_SOURCES_FILE="${SETUP_ROSDEP_DEFAULT_SOURCES_FILE:-/etc/ros/rosdep/sources.list.d/20-default.list}"

# Mirror build.sh / .shrc_local: ignore ~/.local site-packages so that the
# install-time view of Python packages matches what the build/runtime sees.
# Otherwise tools installed via `pip install --user` (e.g. legacy colcon in
# ~/.local) can make setup verification pass while build.sh fails with
# "No module named colcon" because PYTHONNOUSERSITE=1 is set there.
export PYTHONNOUSERSITE=1

# NOTE on NumPy / OpenCV pinning strategy
# ---------------------------------------
# We deliberately do NOT pass `-c numpy==1.26.4` to the lerobot install.
# lerobot's transitive dependency graph (rerun-sdk, opencv, datasets, ...)
# is solved against numpy>=2.x, and forcing numpy==1.26.4 as a constraint
# during the lerobot resolution explodes pip backtracking into
# `resolution-too-deep`. Instead, we let lerobot install whatever NumPy/
# OpenCV it wants, and AFTERWARDS force-reinstall numpy==1.26.4 +
# opencv-python-headless<4.12 to restore ROS 2 Humble ABI compatibility.
# This produces a few cosmetic dependency-resolver warnings during install,
# which are harmless because we never call the numpy-2-only APIs in the
# affected packages from the ROS pipeline.

# Detect if running as root
if [[ $EUID -eq 0 ]]; then
    USE_SUDO=false
fi

# Helper to run commands with or without sudo
run_sudo() {
    if [[ "${USE_SUDO}" == true ]]; then
        sudo "$@"
    else
        "$@"
    fi
}

print_cmd() {
    printf '%q ' "$@"
    printf '
'
}

run_cmd() {
    if [[ "${DRY_RUN}" == true ]]; then
        echo -n "[DRY-RUN] "
        print_cmd "$@"
        return 0
    fi
    [[ "${VERBOSE}" == true ]] && { echo -n "[CMD] "; print_cmd "$@"; }
    "$@"
}

rosdep_sources_list_needs_refresh() {
    local target_file="${SETUP_ROSDEP_DEFAULT_SOURCES_FILE}"

    if [[ ! -f "${target_file}" ]]; then
        return 0
    fi

    grep -q "https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/" "${target_file}"
}

write_rosdep_sources_list() {
    local target_file="${SETUP_ROSDEP_DEFAULT_SOURCES_FILE}"
    local target_dir
    local tmpfile
    target_dir="$(dirname "${target_file}")"
    tmpfile="$(mktemp /tmp/ibrobot-rosdep.XXXXXX)"

    if ! "${VENV_PYTHON}" -c 'from pathlib import Path; import sys; from rosdep2.sources_list import download_default_sources_list; Path(sys.argv[2]).write_text(download_default_sources_list(sys.argv[1]), encoding="utf-8")' "${SETUP_ROSDEP_DEFAULT_SOURCES_URL}" "${tmpfile}"; then
        rm -f "${tmpfile}"
        return 1
    fi

    if ! run_sudo mkdir -p "${target_dir}"; then
        rm -f "${tmpfile}"
        return 1
    fi

    if [[ -f "${target_file}" ]] && ! run_sudo cp "${target_file}" "${target_file}.bak"; then
        rm -f "${tmpfile}"
        return 1
    fi

    if ! run_sudo cp "${tmpfile}" "${target_file}"; then
        rm -f "${tmpfile}"
        return 1
    fi

    rm -f "${tmpfile}"
}

resolve_venv_python() {
    local venv_path="$1"
    local candidate=""

    for candidate in "${venv_path}/bin/python3" "${venv_path}/bin/python"; do
        if [[ -e "${candidate}" ]] && "${candidate}" -c "import sys" >/dev/null 2>&1; then
            printf '%s
' "${candidate}"
            return 0
        fi
    done

    return 1
}


ensure_sudo_session() {
    if [[ "${USE_SUDO}" != true || "${SUDO_AUTH_READY}" == true ]]; then
        return 0
    fi

    log_info "Sudo authentication required for upcoming system package operations."
    sudo -v
    SUDO_AUTH_READY=true
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_done()    { SUMMARY+=("${GREEN}✓${NC} $*"); }
log_skipped() { SUMMARY+=("${YELLOW}⊘${NC} $* (skipped by --yes)"); }

cleanup_setup_artifacts() {
    if [[ -n "${GUM_TMPDIR}" && -d "${GUM_TMPDIR}" ]]; then
        rm -rf "${GUM_TMPDIR}"
        GUM_TMPDIR=""
    fi
}

install_gum_bootstrap() {
    [[ -t 1 ]] || return 1
    command -v curl >/dev/null 2>&1 || return 1
    command -v tar >/dev/null 2>&1 || return 1

    local arch os gum_os gum_arch gum_url gum_candidate
    arch="$(uname -m)"
    os="$(uname -s)"

    case "${os}" in
        Linux) gum_os="Linux" ;;
        Darwin) gum_os="Darwin" ;;
        *) return 1 ;;
    esac

    case "${arch}" in
        x86_64|amd64) gum_arch="x86_64" ;;
        aarch64|arm64) gum_arch="arm64" ;;
        armv7*|armhf) gum_arch="armv7" ;;
        *) return 1 ;;
    esac

    GUM_TMPDIR="$(mktemp -d /tmp/ibrobot-gum.XXXXXX)"
    gum_url="https://github.com/charmbracelet/gum/releases/download/v${GUM_VERSION}/gum_${GUM_VERSION}_${gum_os}_${gum_arch}.tar.gz"

    if curl -fsSL "${gum_url}" | tar xz -C "${GUM_TMPDIR}" >/dev/null 2>&1; then
        gum_candidate="$(find "${GUM_TMPDIR}" -name gum -type f 2>/dev/null | head -n1)"
        if [[ -n "${gum_candidate}" ]] && chmod +x "${gum_candidate}" && [[ -x "${gum_candidate}" ]]; then
            GUM_BIN="${gum_candidate}"
            USE_GUM=true
            return 0
        fi
    fi

    cleanup_setup_artifacts
    return 1
}

detect_ui_backend() {
    if command -v gum >/dev/null 2>&1; then
        GUM_BIN="$(command -v gum)"
        USE_GUM=true
        return 0
    fi

    install_gum_bootstrap || true
}

ui_render_block() {
    local content="$1"

    if [[ "${USE_GUM}" == true ]]; then
        "${GUM_BIN}" style \
            --border rounded \
            --border-foreground 212 \
            --padding "0 1" \
            --margin "0 0 1 0" \
            "${content}"
    else
        echo "${content}"
        echo ""
    fi
}

ui_stage_progress() {
    local title="$1"

    if [[ "${USE_GUM}" == true ]]; then
        "${GUM_BIN}" spin --spinner dot --title "${title}" -- sleep 0.15
    fi
}

run_with_progress() {
    local title="$1"
    shift
    local status
    local heartbeat_pid=""

    if [[ -t 1 ]]; then
        (
            sleep 8
            while true; do
                echo "[INFO] ${title} still running..."
                sleep 8
            done
        ) &
        heartbeat_pid=$!
    fi

    if [[ "${USE_GUM}" == true ]]; then
        "${GUM_BIN}" spin --title "${title}" --spinner dot --show-output -- "$@"
        status=$?
    else
        log_info "${title}"
        "$@"
        status=$?
    fi

    if [[ -n "${heartbeat_pid}" ]]; then
        kill "${heartbeat_pid}" >/dev/null 2>&1 || true
        wait "${heartbeat_pid}" 2>/dev/null || true
    fi

    return "${status}"
}

run_privileged_with_progress() {
    local title="$1"
    shift

    if [[ "${USE_SUDO}" == true ]]; then
        run_with_progress "${title}" sudo "$@"
    else
        run_with_progress "${title}" "$@"
    fi
}

run_with_live_output() {
    local title="$1"
    shift

    log_info "${title}"
    # DimOS approach: run commands directly on the terminal so
    # interactive tools (apt, dnf) show full progress output.
    "$@"
    return $?
}

run_privileged_with_live_output() {
    local title="$1"
    shift

    if [[ "${USE_SUDO}" == true ]]; then
        log_info "${title}"
        sudo "$@"
        return $?
    else
        run_with_live_output "${title}" "$@"
    fi
}

ui_prompt_input() {
    local prompt="$1"
    local placeholder="${2:-}"

    if [[ "${USE_GUM}" == true ]]; then
        "${GUM_BIN}" input --prompt "${prompt}: " --placeholder "${placeholder}"
        return 0
    fi

    read -r -p "${prompt}: " REPLY
    printf '%s\n' "${REPLY}"
}

submodule_is_initialized() {
    local path="$1"
    [[ -e "${path}/.git" ]] || return 1
    git -C "${path}" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

install_gitlint_hook() {
    local hook_path
    hook_path="$(git rev-parse --git-path hooks/commit-msg)"

    if [[ -f "${hook_path}" ]]; then
        if grep -qi "gitlint" "${hook_path}"; then
            log_info "gitlint commit-msg hook already installed."
            return 0
        fi

        log_warn "Existing commit-msg hook detected at ${hook_path}."
        log_warn "Skipping automatic gitlint hook installation to avoid overwriting a shared hook."
        return 0
    fi

    log_info "Installing gitlint commit-msg hook..."
    yes | gitlint install-hook
}

print_banner() {
    local cols banner
    cols=$(stty size </dev/tty 2>/dev/null | awk '{print $2}') \
        || cols=$(tput cols 2>/dev/null) \
        || cols=80

    if [[ ${cols} -ge 90 ]]; then
        banner="$(cat <<'EOF'
   ██╗██████╗     ██████╗  ██████╗ ██████╗  ██████╗ ████████╗
   ██║██╔══██╗    ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝
   ██║██████╔╝    ██████╔╝██║   ██║██████╔╝██║   ██║   ██║
   ██║██╔══██╗    ██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║
   ██║██████╔╝    ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║
   ╚═╝╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝
EOF
)"
    elif [[ ${cols} -ge 56 ]]; then
        banner="$(cat <<'EOF'
   ██╗██████╗   ██████╗  ██████╗
   ██║██╔══██╗  ██╔══██╗██╔═══██╗
   ██║██████╔╝  ██████╔╝██║   ██║
   ██║██╔══██╗  ██╔══██╗██║   ██║
   ██║██████╔╝  ██║  ██║╚██████╔╝
   ╚═╝╚═════╝   ╚═╝  ╚═╝ ╚═════╝
EOF
)"
    else
        banner="IB Robot"
    fi

    if [[ "${USE_GUM}" == true ]]; then
        "${GUM_BIN}" style --foreground 212 --bold "${banner}"
        echo ""
        "${GUM_BIN}" style --foreground 245 "   Intelligent BooM Robot · workspace installer"
        "${GUM_BIN}" style --foreground 241 "   using gum for interactive prompts"
        echo ""
        return 0
    fi

    echo "${banner}"
    echo ""
    echo -e "\033[38;5;245m   Intelligent BooM Robot · workspace installer${NC}"
    echo -e "\033[38;5;241m   using gum for interactive prompts${NC}"
    echo ""
}

set_stage() {
    CURRENT_STAGE="$1"
    echo ""
    if [[ "${USE_GUM}" == true ]]; then
        "${GUM_BIN}" style --foreground 212 --bold "▸ ${CURRENT_STAGE}..."
    else
        echo -e "${YELLOW}▸ ${CURRENT_STAGE}...${NC}"
    fi
}

on_setup_failure() {
    local exit_code="$1"
    local line_no="$2"
    trap - ERR
    echo ""
    log_error "Setup failed during stage: ${CURRENT_STAGE} (line ${line_no})"
    if [[ ${#SUMMARY[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Partial summary:${NC}"
        for entry in "${SUMMARY[@]}"; do
            echo -e "  ${entry}"
        done
    fi
    exit "${exit_code}"
}

on_setup_interrupt() {
    trap - INT
    echo ""
    log_error "Setup interrupted during stage: ${CURRENT_STAGE}"
    exit 130
}

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/detect.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/lerobot_patches.sh"

show_help() {
    cat <<'EOF'
Workspace setup script for IB_Robot

Usage:
  ./scripts/setup.sh [OPTIONS]

Options:
  -y, --yes              Auto-confirm prompts using defaults
      --git-http         Use HTTPS instead of SSH for git remotes
      --sudo             Force sudo for privileged operations
      --no-sudo          Never use sudo
      --skip-submodules  Skip submodule initialization/update
      --skip-system-deps Skip ROS/system dependency installation
      --skip-python      Skip Python virtual environment setup
      --skip-verify      Skip final ROS/Python verification
      --platform ID      Override platform detection
      --lerobot-profiles CSV
                         Override lerobot patch profile selection
                         (e.g. core,ros,hardware,ascend). Highest
                         precedence; overrides IBR_LEROBOT_PROFILES env.
  -h, --help             Show this help

Known platform IDs:
  ubuntu-22.04
  openeuler-embedded-24.03
  openharmony-5.1.0-musl
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y) AUTO_YES=true ;;
            --git-http) GIT_HTTP=true ;;
            --no-sudo) USE_SUDO=false ;;
            --sudo) USE_SUDO=true ;;
            --skip-submodules) SKIP_SUBMODULES=true ;;
            --skip-system-deps) SKIP_SYSTEM_DEPS=true ;;
            --skip-python) SKIP_PYTHON=true ;;
            --skip-verify) SKIP_VERIFY=true ;;
            --platform)
                shift
                if [[ $# -eq 0 ]]; then
                    log_error "--platform requires a platform ID."
                    exit 1
                fi
                PLATFORM_OVERRIDE="$1"
                ;;
            --lerobot-profiles)
                shift
                if [[ $# -eq 0 ]]; then
                    log_error "--lerobot-profiles requires a comma-separated profile list."
                    exit 1
                fi
                # Exported here (not just assigned) so detect.sh's
                # resolve_lerobot_profiles can consume it later.
                export IBR_LEROBOT_PROFILES_CLI="$1"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

# ask_yn <prompt> <default>
# default: "y" = yes by default (Y/n), "n" = no by default (y/N)
# Returns 0 if confirmed, 1 if declined.
ask_yn() {
    local prompt="$1"
    local default="${2:-n}"
    if [[ "${AUTO_YES}" == true ]]; then
        echo -e "${prompt} [auto-yes: YES]"
        return 0
    fi

    if [[ "${USE_GUM}" == true ]]; then
        if [[ "${default}" == "y" ]]; then
            "${GUM_BIN}" confirm --default=true "${prompt}"
        else
            "${GUM_BIN}" confirm --default=false "${prompt}"
        fi
        return $?
    fi

    local hint
    if [[ "${default}" == "y" ]]; then hint="Y/n"; else hint="y/N"; fi
    read -r -p "${prompt} [${hint}]: " REPLY
    REPLY="${REPLY:-${default}}"
    [[ "${REPLY}" == "y" || "${REPLY}" == "Y" ]]
}

system_package_installed() {
    local package_name="$1"

    if command -v dpkg-query >/dev/null 2>&1; then
        dpkg-query -W -f='${Status}' "${package_name}" 2>/dev/null | grep -q "install ok installed"
        return $?
    fi

    if command -v rpm >/dev/null 2>&1; then
        rpm -q "${package_name}" >/dev/null 2>&1
        return $?
    fi

    return 1
}

preview_system_packages() {
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            echo "python3-colcon-common-extensions python3-venv python3-pip"
            ;;
        openeuler-embedded-24.03)
            echo "gcc-c++ vim-enhanced ffmpeg-devel libvpx libvpx-devel nlohmann-json-devel python3-virtualenv python3-pip python3-devel"
            ;;
        *)
            echo ""
            ;;
    esac
}

print_dependency_preview() {
    local preview_message=""

    if [[ "${SKIP_SYSTEM_DEPS}" == true ]]; then
        preview_message="system dependency installation disabled (--skip-system-deps)"
        ui_render_block "${preview_message}"
        return 0
    fi

    if [[ "${SETUP_PLATFORM_ID}" == "openharmony-5.1.0-musl" || "${SETUP_PACKAGE_MANAGER}" == "unknown" ]]; then
        preview_message="manual system dependency provisioning required on this platform"
        ui_render_block "${preview_message}"
        return 0
    fi

    local package_name
    local missing_packages=()
    for package_name in $(preview_system_packages); do
        if ! system_package_installed "${package_name}"; then
            missing_packages+=("${package_name}")
        fi
    done

    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        preview_message="need to install: ${missing_packages[*]}"
    else
        preview_message="system bootstrap packages already installed"
    fi

    ui_render_block "${preview_message}"
}

# ============================================================================
# Environment Checks
# ============================================================================
check_conda() {
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        log_error "Active Conda environment detected at: ${CONDA_PREFIX}"
        log_warn "Conda environments are known to conflict with ROS 2 dependencies (especially Python libraries)."
        log_warn "Please deactivate the Conda environment before running this script:"
        echo -e "    ${YELLOW}conda deactivate${NC}"
        exit 1
    fi
}

check_lerobot_python_compat() {
    local pyproject="${WORKSPACE}/libs/lerobot/pyproject.toml"
    [[ ! -f "${pyproject}" ]] && return 0

    local min_python
    min_python="$(grep -E '^requires-python = ">=[0-9]+\.[0-9]+"' "${pyproject}" | sed -E 's/.*">=([0-9]+\.[0-9]+)".*/\1/' | head -n1)"
    [[ -z "${min_python}" ]] && return 0

    if ! python3 - "${min_python}" <<'PY'
import re
import sys

required = sys.argv[1]
match = re.fullmatch(r"(\d+)\.(\d+)", required)
if not match:
    raise SystemExit(0)

major, minor = map(int, match.groups())
raise SystemExit(0 if sys.version_info >= (major, minor) else 1)
PY
    then
            local current_py
            current_py="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
            log_error "libs/lerobot requires Python >= ${min_python}, but the active interpreter is ${current_py}."
            log_error "The current ROS 2 Humble workspace venv is not new enough for the active LeRobot patch tag."
            exit 1
    fi
}

check_lerobot_ros_numpy_compat() {
    local pyproject="${WORKSPACE}/libs/lerobot/pyproject.toml"
    [[ ! -f "${pyproject}" ]] && return 0

    grep -q '"numpy>=2\.0\.0,<2\.3\.0"' "${pyproject}"
}

install_lerobot_editable() {
    local pip_runner=("$@")

    check_lerobot_python_compat

    if check_lerobot_ros_numpy_compat; then
        log_warn "libs/lerobot declares NumPy >= 2.0, but ROS 2 requires NumPy 1.26.x."
        log_warn "Installing lerobot with full dependencies; NumPy will be pinned to 1.26.4 afterward."
    fi

    "${pip_runner[@]}" install -e "${WORKSPACE}/libs/lerobot"
}

platform_ros_setup_path() {
    if [[ -n "${ROS_HUMBLE_SETUP_PATH:-}" ]]; then
        echo "${ROS_HUMBLE_SETUP_PATH}"
    elif [[ -f /opt/ros/humble/setup.sh ]]; then
        echo "/opt/ros/humble/setup.sh"
    elif [[ -f /opt/ros/humble/setup.bash ]]; then
        echo "/opt/ros/humble/setup.bash"
    else
        echo "/opt/ros/humble/setup.bash"
    fi
}

platform_handle_missing_ros() {
    if [[ "${SETUP_PLATFORM_ID}" == "openharmony-5.1.0-musl" ]]; then
        log_error "ROS 2 Humble setup script is not configured for OpenHarmony."
        log_error "Set ROS_HUMBLE_SETUP_PATH to your Python 3.11 ROS Humble environment before running setup."
        exit 1
    fi

    log_info "Running ROS 2 installation script..."

    local install_args=()
    if [[ "${AUTO_YES}" == true ]]; then
        install_args+=("--yes")
    fi
    if [[ "${USE_SUDO}" == false ]]; then
        install_args+=("--no-sudo")
    fi

    if "${WORKSPACE}/scripts/install_ros.sh" "${install_args[@]}"; then
        log_done "ROS 2 Humble installed"
    else
        log_error "ROS 2 installation failed"
        log_error "Please run ${WORKSPACE}/scripts/install_ros.sh manually to diagnose the issue"
        exit 1
    fi
}

platform_prepare_host() {
    if [[ "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_warn "openEuler detected. Setting ROS_OS_OVERRIDE=rhel:8 for rosdep compatibility."
        export ROS_OS_OVERRIDE=rhel:8

        if ! dnf repolist | grep -qi "openEuler-24.03-LTS"; then
            local arch
            arch=$(uname -m)
            run_privileged_with_live_output "Adding openEuler repo for ${arch}..." dnf config-manager --add-repo "https://repo.openeuler.org/openEuler-24.03-LTS/OS/${arch}"
            run_privileged_with_live_output "Refreshing dnf metadata..." dnf clean all
            run_privileged_with_live_output "Building dnf cache..." dnf makecache
        else
            log_info "openEuler repo already configured, skipping add-repo."
        fi

        run_privileged_with_live_output "Installing openEuler host packages required by the workspace..." dnf install -y --nogpgcheck gcc-c++ vim-enhanced ffmpeg-devel libvpx libvpx-devel nlohmann-json-devel
        return 0
    fi

    if [[ "${SETUP_PLATFORM_ID}" == "openharmony-5.1.0-musl" ]]; then
        log_warn "OpenHarmony musl platform detected."
        log_warn "System dependency bootstrap is not yet fully automated for this platform."
        log_warn "Proceeding with shared setup steps only; platform-specific package installation must be supplied separately."
    fi
}

platform_install_colcon() {
    if command -v apt-get &>/dev/null; then
        run_privileged_with_live_output "Installing colcon build tool..." apt-get install -y python3-colcon-common-extensions
        return 0
    fi

    if command -v dnf &>/dev/null; then
        if command -v pip3 &>/dev/null; then
            run_with_progress "Installing colcon build tool..." pip3 install colcon-common-extensions --quiet
            return 0
        fi
        log_error "pip3 not found, cannot install colcon."
        exit 1
    fi

    if command -v pip3 &>/dev/null; then
        run_with_progress "Installing colcon build tool..." pip3 install colcon-common-extensions --quiet
        return 0
    fi

    log_error "Unable to install colcon automatically on this platform."
    exit 1
}

platform_install_python_bootstrap() {
    if command -v apt-get &>/dev/null; then
        run_privileged_with_live_output "Updating apt package lists..." apt-get update
        run_privileged_with_live_output "Installing Python venv tooling..." apt-get install -y python3-venv python3-pip
        return 0
    fi

    if command -v dnf &>/dev/null; then
        run_privileged_with_live_output "Installing Python venv tooling..." dnf install -y --nogpgcheck python3-virtualenv python3-pip python3-devel
        return 0
    fi

    if [[ "${SETUP_PLATFORM_ID}" == "openharmony-5.1.0-musl" ]]; then
        if ! "${SETUP_BOOTSTRAP_PYTHON_BIN:-python3}" -m venv --help >/dev/null 2>&1; then
            log_error "python3 venv module is unavailable on this OpenHarmony environment."
            log_error "Install the Python venv package or provide a pre-created virtual environment."
            exit 1
        fi
        return 0
    fi
}

platform_install_rosdeps() {
    local rosdepc_cmd="${ROSDEPC_BIN:-rosdepc}"
    local rosdep_install_extra_args=()

    if [[ "${USE_SUDO}" != true ]]; then
        rosdep_install_extra_args+=(--as-root apt:false --as-root dnf:false --as-root pip:false)
    fi

    if command -v apt-get &>/dev/null; then
        run_privileged_with_live_output "Updating apt package lists..." apt-get update

        if ! run_with_progress "Updating rosdepc database..." env ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL}" "${rosdepc_cmd}" update --rosdistro=humble; then
            log_error "rosdepc update failed. This is usually due to network issues."
            log_error "Please check your network connection and re-run ./scripts/setup.sh"
            exit 1
        fi

        if ! run_with_progress "Installing ROS dependencies via apt..." env ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL}" "${rosdepc_cmd}" install \
            "${rosdep_install_extra_args[@]}" \
            --from-paths src \
            --ignore-src \
            --rosdistro=humble \
            -y -r \
            --skip-keys=catkin \
            --skip-keys=roscpp \
            --skip-keys=lerobot \
            --skip-keys=trimesh\[easy\] \
            --skip-keys=simple-parsing \
            --skip-keys=cupy-cuda12x \
            --skip-keys=ctl_system_interface \
            --skip-keys=numpy_lessthan_2 \
            --skip-keys=ament_python \
            --skip-keys=feetech-servo-sdk \
            --skip-keys=pyserial; then
            log_error "rosdepc install failed."
            log_error "Please check your network connection or dependency lists and re-run ./scripts/setup.sh"
            exit 1
        fi
        return 0
    fi

    if command -v dnf &>/dev/null; then
        if ! run_with_progress "Updating rosdepc database..." env ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL}" "${rosdepc_cmd}" update --rosdistro=humble; then
            log_error "rosdepc update failed. This is usually due to network issues."
            log_error "Please check your network connection and re-run ./scripts/setup.sh"
            exit 1
        fi

        if ! run_with_progress "Installing ROS dependencies via dnf..." env ROSDISTRO_INDEX_URL="${SETUP_ROSDISTRO_INDEX_URL}" "${rosdepc_cmd}" install \
            "${rosdep_install_extra_args[@]}" \
            --from-paths src \
            --ignore-src \
            --rosdistro=humble \
            -y -r \
            --skip-keys=catkin \
            --skip-keys=roscpp \
            --skip-keys=lerobot \
            --skip-keys=trimesh \
            --skip-keys=simple-parsing \
            --skip-keys=cupy-cuda12x \
            --skip-keys=ctl_system_interface \
            --skip-keys=numpy_lessthan_2 \
            --skip-keys=ament_python \
            --skip-keys=feetech-servo-sdk \
            --skip-keys=pyserial; then
            log_error "rosdepc install failed."
            log_error "Please check your network connection or dependency lists and re-run ./scripts/setup.sh"
            exit 1
        fi
        return 0
    fi

    if [[ "${SETUP_PLATFORM_ID}" == "openharmony-5.1.0-musl" ]]; then
        log_warn "Skipping automated rosdepc installation on OpenHarmony musl."
        log_warn "Provide ROS/system dependencies externally or rerun with --skip-system-deps if this is intentional."
        return 0
    fi

    log_warn "Unknown package manager. Please ensure ROS 2 Humble dependencies are installed manually."
}

platform_verify_ros_python_bridge() {
    local ros_setup="$(platform_ros_setup_path)"
    local python_bin="$(command -v python3 || true)"
    if [ -z "${ros_setup}" ]; then
        return 1
    fi
    if [ ! -f "${ros_setup}" ]; then
        return 1
    fi
    if [ -z "${python_bin}" ]; then
        return 1
    fi
    (
        set +u
        set +e
        source "${ros_setup}" >/dev/null 2>&1
        "${python_bin}" -c 'import rclpy; print("ROS 2 Humble connection successful")'
    ) 2>/dev/null
}

# ============================================================================
# Repository Management
# ============================================================================
apply_git_http_config() {
    if [[ "${GIT_HTTP}" == true ]]; then
        log_info "Configuring git to use HTTPS instead of SSH globally for this session..."
        # Use git config insteadOf for all major domains
        # We use 'git config' without --global to keep it local to this repo
        git config url."https://gitcode.com/".insteadOf "git@gitcode.com:" || true
        git config url."https://github.com/".insteadOf "git@github.com:" || true
        git config url."https://atomgit.com/".insteadOf "git@atomgit.com:" || true

        # Sync submodule URLs to apply the insteadOf mapping to .git/config
        log_info "Syncing submodule URLs..."
        git submodule sync --recursive
    fi
}
update_submodules() {
    if [[ "${SKIP_SUBMODULES}" == true ]]; then
        log_info "Skipping submodule initialization/update (--skip-submodules)."
        log_skipped "Submodule sync/update"
        return 0
    fi

    # Define submodules
    local submodules=(
        "libs/lerobot:LeRobot"
        "src/pymoveit2:PyMoveIt2"
    )

    # Check which submodules need initialization
    local need_init=()
    for entry in "${submodules[@]}"; do
        local path="${entry%%:*}"
        local name="${entry##*:}"
        if ! submodule_is_initialized "${path}"; then
            need_init+=("${path}:${name}")
        fi
    done

    # If all submodules exist, ask if user wants to update
    if [[ ${#need_init[@]} -eq 0 ]]; then
        log_info "All submodules are already initialized:"
        for entry in "${submodules[@]}"; do
            local path="${entry%%:*}"
            local name="${entry##*:}"
            echo "  ✓ ${name} (${path})"
        done
        echo ""
        if ! ask_yn "Do you want to sync/update all submodules?" "n"; then
            log_info "Skipping submodule update."
            log_skipped "Submodule sync/update"
            return 0
        fi
        log_info "Updating all submodules..."
        export GIT_LFS_SKIP_SMUDGE=1
        run_with_progress "Updating all submodules..." git submodule update --init --recursive
        log_done "Submodules synced/updated"
        return 0
    fi

    # Some submodules need initialization
    log_warn "The following submodules are not initialized:"
    for entry in "${need_init[@]}"; do
        local path="${entry%%:*}"
        local name="${entry##*:}"
        echo "  ✗ ${name} (${path})"
    done
    echo ""

    if [[ "${AUTO_YES}" == true ]] || ask_yn "Initialize all missing submodules?" "y"; then
        export GIT_LFS_SKIP_SMUDGE=1
        run_with_progress "Initializing submodules..." git submodule update --init --recursive
        log_done "Submodules initialized"
    else
        log_warn "Submodule initialization skipped."
        log_skipped "Submodule initialization"
    fi
}

setup_developer_forks() {
    echo "If you have forked the repository on GitCode, enter your username"
    echo "to automatically set up your personal fork as 'origin' and the"
    echo "original repository as 'upstream'."
    echo ""
    if [[ "${AUTO_YES}" == true ]]; then
        log_info "Auto-yes: skipping fork setup."
        log_skipped "Developer fork setup"
        return 0
    fi
    USERNAME="$(ui_prompt_input "Enter your GitCode username (leave empty to skip)")"

    if [[ -n "${USERNAME}" ]]; then
        local MAIN_FORK LEROBOT_FORK UPSTREAM_URL

        # We define as SSH format, which will be auto-translated to HTTPS 
        # if GIT_HTTP is true due to 'insteadOf' config applied in main()
        MAIN_FORK="git@gitcode.com:${USERNAME}/IB_Robot.git"
        LEROBOT_FORK="git@gitcode.com:${USERNAME}/lerobot.git"
        UPSTREAM_URL="git@atomgit.com:openeuler/IB_Robot.git"

        echo -e "\nProposed Fork URLs:"
        echo -e "  Main Repo:    ${MAIN_FORK}"
        echo -e "  libs/lerobot: ${LEROBOT_FORK}"
        echo ""
        if ask_yn "Confirm setting these as 'origin'?" "n"; then
            log_info "Configuring personal forks..."
            
            # 1. Update main repo remotes
            git remote set-url origin "${MAIN_FORK}"
            git remote add upstream "${UPSTREAM_URL}" 2>/dev/null || git remote set-url upstream "${UPSTREAM_URL}"

            # 2. Update submodule fork
            if [[ -d "libs/lerobot/.git" ]]; then
                (cd libs/lerobot && git remote set-url origin "${LEROBOT_FORK}")
                local LEROBOT_UPSTREAM=$(git config -f .gitmodules submodule.libs/lerobot.url)
                (cd libs/lerobot && git remote add upstream "${LEROBOT_UPSTREAM}" 2>/dev/null || git remote set-url upstream "${LEROBOT_UPSTREAM}")
            fi

            log_info "Forks configured successfully!"
            log_done "Developer forks configured (origin=${MAIN_FORK})"
        else
            log_info "Fork setup cancelled."
            log_skipped "Developer fork setup (cancelled)"
        fi
    else
        log_info "Skipping fork setup."
    fi
}

# ============================================================================
# Dependency Management
# ============================================================================
check_ros_installation() {
    local ros_setup_path
    ros_setup_path="$(platform_ros_setup_path)"

    # Check if ROS 2 Humble is installed
    if [[ -n "${ros_setup_path}" && -f "${ros_setup_path}" ]]; then
        log_info "ROS 2 Humble is already installed"
    else
        log_warn "ROS 2 Humble setup script not found${ros_setup_path:+ at ${ros_setup_path}}"
        platform_handle_missing_ros
    fi
}

ensure_colcon() {
    # Note: this only checks whether a colcon CLI exists somewhere on PATH.
    # The authoritative installation that build.sh consumes happens later in
    # setup_python_venv via `${venv_python} -m pip install colcon-common-extensions`,
    # which guarantees the module is importable from the workspace venv even
    # when PYTHONNOUSERSITE=1 is set (which is the case in build.sh and
    # .shrc_local). Installing colcon system-wide here is only a convenience
    # for users invoking `colcon` directly outside of build.sh.
    if command -v colcon &>/dev/null; then
        log_info "colcon CLI is already available on PATH (will also be installed into venv later)"
        return 0
    fi

    log_info "Installing colcon build tool..."
    platform_install_colcon
    log_done "colcon installed"
}

check_openeuler() {
    if [[ "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_warn "openEuler detected. Setting ROS_OS_OVERRIDE=rhel:8 for rosdep compatibility."
        export ROS_OS_OVERRIDE=rhel:8

        if ! dnf repolist | grep -qi "openEuler-24.03-LTS"; then
            local arch
            arch=$(uname -m)
            log_info "Adding openEuler repo for ${arch}..."
            run_sudo dnf install -y dnf-plugins-core
            run_sudo dnf config-manager --add-repo "https://repo.openeuler.org/openEuler-24.03-LTS/OS/${arch}"
            run_sudo dnf clean all
            run_sudo dnf makecache
        else
            log_info "openEuler repo already configured, skipping add-repo."
        fi

        log_info "Installing gcc-c++, vim-enhanced, ffmpeg-devel, libvpx, libvpx-devel, and nlohmann-json-devel..."
        run_sudo dnf install -y --nogpgcheck gcc-c++ vim-enhanced ffmpeg-devel libvpx libvpx-devel nlohmann-json-devel
    fi
}

ensure_workspace_venv() {
    # Idempotent: create ${WORKSPACE}/venv if missing and resolve its
    # python/rosdepc paths into script-global vars. The full venv setup
    # (lerobot, colcon, numpy pin, etc.) still happens later in
    # setup_python_venv; this helper only guarantees the venv exists
    # early enough that ensure_rosdepc can install rosdepc into it.
    VENV_PATH="${WORKSPACE}/venv"

    if [[ ! -d "${VENV_PATH}" ]]; then
        if [[ "${SKIP_PYTHON}" == true ]]; then
            log_error "rosdepc requires a workspace venv at ${VENV_PATH}, but --skip-python was passed."
            log_error "Re-run without --skip-python, or create the venv manually:"
            log_error "  python3 -m venv --system-site-packages ${VENV_PATH}"
            exit 1
        fi
        log_info "Creating workspace venv at ${VENV_PATH} (early, for rosdepc)..."
        run_cmd python3 -m venv --system-site-packages "${VENV_PATH}"
    fi

    VENV_PYTHON="$(resolve_venv_python "${VENV_PATH}" || true)"
    if [[ -z "${VENV_PYTHON}" ]]; then
        log_error "No working Python interpreter found under ${VENV_PATH}/bin."
        exit 1
    fi
    ROSDEPC_BIN="${VENV_PATH}/bin/rosdepc"
}

ensure_rosdepc() {
    # Single source of truth: rosdepc lives in the workspace venv.
    #
    # Why not rely on a system-wide rosdepc on PATH?
    # - A stale shim from a previous python install (e.g. an orphan
    #   /usr/local/bin/rosdepc whose shebang points at a python that no
    #   longer has the module) survives `command -v` checks and only
    #   blows up at runtime with `ModuleNotFoundError: No module named
    #   'rosdepc'`.
    # - `pip3 install --force-reinstall rosdepc` does NOT heal that shim
    #   when pip3's python differs from the shim's shebang python: pip
    #   writes a new shim into its own bin dir while the broken one
    #   keeps winning on PATH.
    #
    # Both failure modes vanish if we install rosdepc into the workspace
    # venv (using the venv's own python) and always invoke it via the
    # explicit venv path. There is no shim to go stale, and pip always
    # installs into the same interpreter we then use.
    ensure_workspace_venv

    if ! "${ROSDEPC_BIN}" --version &>/dev/null; then
        log_info "Installing rosdepc into the workspace venv..."
        run_cmd "${VENV_PYTHON}" -m pip install --upgrade pip --quiet
        run_cmd "${VENV_PYTHON}" -m pip install rosdepc --quiet
        if ! "${ROSDEPC_BIN}" --version &>/dev/null; then
            log_error "rosdepc install did not produce a working CLI at ${ROSDEPC_BIN}."
            log_error "Re-run with VERBOSE=1 to see the full pip output."
            exit 1
        fi
    fi
    log_info "Using venv rosdepc: ${ROSDEPC_BIN}"

    if rosdep_sources_list_needs_refresh; then
        log_info "Configuring rosdep sources list..."
        # Pre-authenticate sudo so password prompt is visible
        if [[ "${USE_SUDO}" == true ]]; then
            sudo -v
        fi

        local init_output=""
        local init_exit=0
        init_output=$(write_rosdep_sources_list 2>&1) || init_exit=$?

        # Check both exit code and output for SSL/network errors
        if [[ ${init_exit} -ne 0 ]] || echo "${init_output}" | grep -qi "error\|failed\|certificate\|urlopen"; then
            if echo "${init_output}" | grep -qi "certificate\|ssl\|urlopen"; then
                log_warn "SSL certificate error detected while preparing rosdep sources:"
                echo "${init_output}"
                log_warn "Attempting to fix SSL certificates..."
            else
                log_warn "rosdep sources setup failed, attempting SSL certificate fix..."
                echo "${init_output}"
            fi

            # Get the .pem path used by Python's ssl module
            local ssl_pem
            ssl_pem=$("${VENV_PYTHON}" -c "import ssl; print(ssl.get_default_verify_paths().openssl_cafile)" 2>/dev/null)

            if [[ -z "${ssl_pem}" ]]; then
                log_error "Could not determine Python SSL certificate path."
                exit 1
            fi

            # Find the first available system CA bundle
            local ca_bundle=""
            for candidate in \
                /etc/pki/tls/certs/ca-bundle.crt \
                /etc/ssl/certs/ca-bundle.crt \
                /etc/ssl/certs/ca-certificates.crt; do
                if [[ -f "${candidate}" ]]; then
                    ca_bundle="${candidate}"
                    break
                fi
            done

            if [[ -z "${ca_bundle}" ]]; then
                log_error "No system CA bundle found. Cannot fix SSL certificates."
                exit 1
            fi

            log_info "Python SSL cert path: ${ssl_pem}"
            log_info "System CA bundle: ${ca_bundle}"

            # Check if source and destination are the same file (e.g. symlinks)
            local real_ssl_pem real_ca_bundle
            real_ssl_pem=$(realpath "${ssl_pem}" 2>/dev/null || echo "${ssl_pem}")
            real_ca_bundle=$(realpath "${ca_bundle}" 2>/dev/null || echo "${ca_bundle}")

            if [[ "${real_ssl_pem}" == "${real_ca_bundle}" ]]; then
                log_info "Python SSL cert path and system CA bundle are already the same file."
                log_info "  ${ssl_pem} -> ${real_ssl_pem}"
                log_info "  ${ca_bundle} -> ${real_ca_bundle}"
                log_info "No copy needed. SSL certificates are already correctly configured."
                log_done "SSL certificates verified (already linked)"
            elif ask_yn "Apply SSL certificate fix (copy system CA bundle to Python SSL path)?" "n"; then
                log_info "Creating directory: $(dirname "${ssl_pem}")"
                run_sudo mkdir -p "$(dirname "${ssl_pem}")"

                if [[ -f "${ssl_pem}" ]]; then
                    log_info "Backing up existing cert: ${ssl_pem} -> ${ssl_pem}.bak"
                    run_sudo cp "${ssl_pem}" "${ssl_pem}.bak"
                fi

                log_info "Copying ${ca_bundle} -> ${ssl_pem}"
                run_sudo cp "${ca_bundle}" "${ssl_pem}"
                log_done "SSL certificate fix applied"

                # Retry writing sources list, capture output again
                local retry_output
                if ! retry_output=$(write_rosdep_sources_list 2>&1) || echo "${retry_output}" | grep -qi "error\|failed\|certificate\|urlopen"; then
                    log_error "rosdep sources setup failed even after SSL fix."
                    echo "${retry_output}"
                    log_error "Please check network and certificate configuration, then re-run ./scripts/setup.sh."
                    exit 1
                fi
            else
                log_warn "Skipped SSL fix. Please manually check your network or certificate configuration."
                log_warn "Then re-run ./scripts/setup.sh with root or sudo access."
                exit 1
            fi
        fi

        log_done "rosdep sources list configured"
    fi
}

install_system_deps() {
    if [[ "${SKIP_SYSTEM_DEPS}" == true ]]; then
        log_info "Skipping ROS/system dependency installation (--skip-system-deps)."
        log_skipped "System ROS dependencies"
        SYSTEM_DEPS_STATUS="skipped"
        return 0
    fi

    # Check for ROS 2 installation first
    ensure_sudo_session
    check_ros_installation
    ensure_colcon

    platform_prepare_host
    ensure_rosdepc
    platform_install_rosdeps
    SYSTEM_DEPS_STATUS="done"
    log_done "System ROS dependencies installed"
}

setup_python_venv() {
    if [[ "${SKIP_PYTHON}" == true ]]; then
        log_info "Skipping Python virtual environment setup (--skip-python)."
        log_skipped "Python virtual environment"
        PYTHON_ENV_STATUS="skipped"
        return 0
    fi

    local venv_path="${WORKSPACE}/venv"
    local lerobot_dir="${WORKSPACE}/libs/lerobot"
    local venv_python=""
    local host_python_path host_python_version host_py_major host_py_minor

    # 0. Python interpreter preflight
    # lerobot + ROS 2 Humble require Python >= 3.10. We print the resolved
    # interpreter so users can immediately see whether a stale shell prompt
    # misled them about which python3 actually drives setup.sh.
    host_python_path="$(command -v python3 || true)"
    if [[ -z "${host_python_path}" ]]; then
        log_error "python3 not found on PATH. Install python3 (>=3.10) before running setup.sh."
        exit 1
    fi
    host_python_version="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown")"
    log_info "Using host python3: ${host_python_path} (version ${host_python_version})"
    host_py_major="$(python3 -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
    host_py_minor="$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
    if (( host_py_major < 3 )) || { (( host_py_major == 3 )) && (( host_py_minor < 10 )); }; then
        log_error "Python ${host_python_version} is too old. setup.sh requires Python >= 3.10."
        log_error "On openEuler: 'sudo dnf install -y python3.10 python3.10-devel' and re-run."
        exit 1
    fi

    # 1. Ensure system-level venv tools are installed
    log_info "Checking for Python venv and pip..."
    ensure_sudo_session
    platform_install_python_bootstrap

    # 2. 创建虚拟环境 (必须包含 --system-site-packages 以使用系统的 rclpy)
    if [[ ! -d "${venv_path}" ]]; then
        run_with_progress "Creating virtual environment at ${venv_path} with --system-site-packages..." "${SETUP_BOOTSTRAP_PYTHON_BIN:-python3}" -m venv --system-site-packages "${venv_path}"
    else
        log_info "Virtual environment already exists at ${venv_path}."
    fi

    # 3. 激活虚拟环境并安装依赖
    log_info "Configuring Python environment and dependencies..."
    source "${venv_path}/bin/activate"

    if [[ -n "${PYTHONPATH:-}" ]]; then
        log_warn "Clearing inherited PYTHONPATH for isolated package installation inside ${venv_path}."
        unset PYTHONPATH
    fi

    if [[ -n "${PYTHONHOME:-}" ]]; then
        log_warn "Clearing inherited PYTHONHOME for isolated package installation inside ${venv_path}."
        unset PYTHONHOME
    fi

    venv_python="$(resolve_venv_python "${venv_path}" || true)"
    if [[ -z "${venv_python}" ]]; then
        log_error "No working virtual environment python was found under ${venv_path}/bin."
        exit 1
    fi

    local pip_install=("${venv_python}" -m pip install)

    # 升级 pip
    run_cmd "${venv_python}" -m pip install --upgrade pip --quiet

    # 解决 setuptools 版本冲突 (兼容 LeRobot 和 colcon)
    run_cmd "${venv_python}" -m pip install "setuptools<80" "setuptools>=71" --quiet

    # ------------------------------------------------------------------
    # LeRobot patch stack — MUST run before install_lerobot_editable.
    # The platform-aware filter (lerobot_filter_series.py) requires
    # PyYAML; install it explicitly first because --system-site-packages
    # cannot be relied on across stripped openEuler/OpenHarmony images.
    #
    # Why here (not after setup_python_venv): install_lerobot_editable
    # invokes check_lerobot_python_compat which reads
    # libs/lerobot/pyproject.toml. On a fresh clone with py3.10/py3.11
    # hosts, pyproject.toml still carries upstream's
    # `requires-python>=3.12` until patches 0001/0002 lower it. Running
    # the patch stack first ensures the compat gate reads the patched
    # pyproject.toml. (See PR #98 review comment 169589247.)
    # ------------------------------------------------------------------
    log_info "Installing PyYAML for the lerobot patch dispatcher..."
    run_cmd "${venv_python}" -m pip install pyyaml --quiet
    # Export VENV_PYTHON so lerobot_patches.sh::_lerobot_filter_python
    # picks up *this* venv (ensure_workspace_venv may have set a stale
    # value if the venv was recreated since then).
    export VENV_PYTHON="${venv_python}"
    if [[ -d "${lerobot_dir}" ]]; then
        ensure_lerobot_patch_stack_applied
    fi

    # 以可编辑模式安装 LeRobot
    # 注意: 不传 -c numpy==1.26.4 约束。lerobot 依赖图(rerun-sdk/opencv/
    # datasets/...)在 numpy>=2 下解析，硬约束会让 pip 进入 resolution-too-deep。
    # 这里放任安装 numpy 2.x，最后再 force-reinstall 回 1.26.4 + opencv<4.12。
    if [[ -d "${lerobot_dir}" ]]; then
        log_info "Installing LeRobot in editable mode..."
        install_lerobot_editable "${venv_python}" -m pip
    fi

    # 安装原有的硬件依赖
    log_info "Installing hardware dependencies (pyserial, feetech)..."
    run_cmd "${pip_install[@]}" pyserial feetech-servo-sdk --quiet

    if [[ "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_info "Installing openEuler fallback dependency (aiortc) into the workspace venv..."
        run_cmd "${pip_install[@]}" aiortc --quiet
    fi

    # 安装 scipy 用于数学计算 (四元数/旋转矩阵转换)
    log_info "Installing scipy for mathematical computations..."
    run_cmd "${pip_install[@]}" scipy --quiet

    # 安装手机遥操可选依赖（iOS: hebi, Android: teleop）
    log_info "Installing optional phone teleoperation dependencies..."
    if [[ "${AUTO_YES}" == true ]]; then
        log_info "Auto-yes mode: installing both phone backends (hebi + teleop)..."
        run_cmd "${pip_install[@]}" hebi teleop --quiet
        log_done "Phone teleoperation dependencies installed (hebi + teleop)"
    else
        echo ""
        echo "  Phone teleoperation backends (optional):"
        echo "    1) iOS only  — hebi  (HEBI Mobile I/O + ARKit)"
        echo "    2) Android only — teleop  (WebXR WebSocket)"
        echo "    3) Both (iOS + Android)"
        echo "    0) Skip phone backends"
        echo ""
        while true; do
            read -r -p "  Enter your choice [0-3]: " PHONE_CHOICE
            case "${PHONE_CHOICE}" in
                1)
                    run_cmd "${pip_install[@]}" hebi --quiet
                    log_done "Phone dependencies installed: hebi (iOS)"
                    break
                    ;;
                2)
                    run_cmd "${pip_install[@]}" teleop --quiet
                    log_done "Phone dependencies installed: teleop (Android)"
                    break
                    ;;
                3)
                    run_cmd "${pip_install[@]}" hebi teleop --quiet
                    log_done "Phone dependencies installed: hebi + teleop (iOS + Android)"
                    break
                    ;;
                0)
                    log_info "Skipping phone teleoperation dependencies."
                    break
                    ;;
                *)
                    echo "  Invalid choice. Please enter 0-3."
                    ;;
            esac
        done
    fi

    # 安装训练可视化依赖
    log_info "Installing training visualization dependencies (tensorboard)..."
    run_cmd "${pip_install[@]}" tensorboard --quiet

    # 安装录制可视化依赖
    log_info "Installing recording visualization dependency (rerun-sdk)..."
    run_cmd "${pip_install[@]}" "rerun-sdk>=0.24,<0.26" --quiet
    log_info "Installing rerun compatibility dependency (typing-extensions>=4.12)..."
    run_cmd "${pip_install[@]}" "typing-extensions>=4.12" --quiet

    # 安装 ONNX 导出相关依赖
    if [[ "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_info "Installing ONNX export dependencies (onnx, onnxruntime); skipping onnxsim on openEuler..."
        run_cmd "${pip_install[@]}" onnx onnxruntime --quiet
    else
        log_info "Installing ONNX export dependencies (onnx, onnxsim, onnxruntime)..."
        run_cmd "${pip_install[@]}" onnx onnxsim onnxruntime --quiet
    fi

    # 安装 gitlint 并设置 git hook
    log_info "Installing gitlint..."
    run_cmd "${venv_python}" -m pip install gitlint --quiet

    # 安装 ruff (代码规范) 和 pre-commit hook
    log_info "Installing ruff and pre-commit..."
    run_cmd "${venv_python}" -m pip install ruff pre-commit --quiet
    if [[ -f "${WORKSPACE}/.pre-commit-config.yaml" ]]; then
        "${venv_python}" -m pre_commit install
    fi

    # ------------------------------------------------------------------
    # Authoritative colcon install: build.sh runs `python3 -m colcon` from
    # this venv with PYTHONNOUSERSITE=1, so colcon MUST live in the venv's
    # site-packages (not in ~/.local). Without this, build.sh fails with
    # "No module named colcon" even though `command -v colcon` succeeds.
    # ------------------------------------------------------------------
    log_info "Installing colcon-common-extensions + colcon-mixin into the workspace venv..."
    run_cmd "${pip_install[@]}" colcon-common-extensions colcon-mixin --quiet

    # rosdepc was already installed into this same venv by the early
    # ensure_workspace_venv + ensure_rosdepc step. Re-running pip install
    # here is a no-op when the package is current, and acts as a safety net
    # in case the venv was recreated between the two steps.
    log_info "Ensuring rosdepc is present in the workspace venv..."
    run_cmd "${pip_install[@]}" rosdepc --quiet

    # 强制把 NumPy/OpenCV 拉回 ROS 2 Humble ABI 兼容版本。
    # lerobot 安装会带入 numpy 2.x，这里无条件覆盖，确保 cv_bridge / image_transport
    # 等 ROS 包在 runtime 不会触发 numpy.core.multiarray 二进制不兼容错误。
    log_info "Pinning NumPy 1.26.4 + opencv-python-headless<4.12 (ROS 2 Humble ABI)..."
    run_cmd "${pip_install[@]}" --force-reinstall "numpy==1.26.4" "opencv-python-headless<4.12" --quiet
    local commit_msg_hook
    commit_msg_hook="$(git rev-parse --git-path hooks/commit-msg)"
    if [[ -e "${commit_msg_hook}" ]]; then
        log_warn "gitlint commit-msg hook already exists at ${commit_msg_hook}; keeping it."
        log_done "gitlint commit-msg hook already exists"
    else
        log_info "Installing gitlint commit-msg hook..."
        printf 'y\n' | gitlint install-hook
        log_done "gitlint commit-msg hook installed"
    fi

    # Venv summary: print the key facts users need to debug "wrong python /
    # wrong colcon" issues without having to source the venv themselves.
    local venv_numpy_ver="unknown" venv_colcon_path="missing" venv_py_ver="unknown"
    venv_py_ver="$(PYTHONNOUSERSITE=1 "${venv_python}" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)"
    venv_numpy_ver="$(PYTHONNOUSERSITE=1 "${venv_python}" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo unknown)"
    venv_colcon_path="$(PYTHONNOUSERSITE=1 "${venv_python}" -c 'import colcon, os; print(os.path.dirname(colcon.__file__))' 2>/dev/null || echo missing)"

    # User-site inspection: even though build.sh sets PYTHONNOUSERSITE=1 and
    # we install colcon into the venv, a stale ~/.local/lib/.../colcon left
    # over from a prior `pip install` (without an active venv) is a known
    # foot-gun: pip's "Requirement already satisfied" short-circuit will
    # silently skip re-installing colcon at the system level when it sees
    # the user-site copy, so any future setup change that relies on
    # system-level colcon would silently no-op. We surface this state in
    # the summary and warn explicitly when colcon shadows are detected.
    local user_site user_site_status="not-present" user_site_colcon=""
    user_site="$("${venv_python}" -m site --user-site 2>/dev/null || true)"
    if [[ -n "${user_site}" && -d "${user_site}" ]]; then
        local user_pkg_count
        user_pkg_count="$(find "${user_site}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
        user_site_status="${user_pkg_count} packages"
        if compgen -G "${user_site}/colcon*" >/dev/null 2>&1; then
            user_site_colcon="$(compgen -G "${user_site}/colcon*" | head -n1)"
            user_site_status="${user_pkg_count} packages, has colcon (shadowed)"
        fi
    fi

    log_info "Venv summary:"
    log_info "  python      : ${venv_python} (${venv_py_ver})"
    log_info "  numpy       : ${venv_numpy_ver}"
    log_info "  colcon path : ${venv_colcon_path}"
    log_info "  user-site   : ${user_site:-<unknown>} [${user_site_status}]"

    if [[ -n "${user_site_colcon}" ]]; then
        log_warn "Stale colcon detected in user-site: ${user_site_colcon}"
        log_warn "  build.sh sets PYTHONNOUSERSITE=1 and will ignore it,"
        log_warn "  but pip's 'Requirement already satisfied' short-circuit"
        log_warn "  may have prevented system-level colcon updates from"
        log_warn "  taking effect during earlier setup runs."
        log_warn "  Recommended cleanup:"
        log_warn "    rm -rf ${user_site}/colcon* ${user_site%/lib/*}/bin/colcon*"
    fi

    log_done "Python environment configured"
}

verify_setup() {
    local ros_setup=""
    local python_bin=""
    local venv_python=""

    if [[ "${SKIP_VERIFY}" == true ]]; then
        PYTHON_ENV_STATUS="done"
        log_info "Skipping ROS/Python verification (--skip-verify)."
        log_skipped "ROS/Python verification"
        return 0
    fi

    log_info "Verifying ROS 2 connection..."
    ros_setup="$(platform_ros_setup_path)"
    python_bin="$(command -v python3 || true)"
    venv_python="$(resolve_venv_python "${WORKSPACE}/venv" || true)"
    if [[ -z "${venv_python}" ]]; then
        log_error "Verification failed: no working virtual environment python was found under ${WORKSPACE}/venv/bin."
        exit 1
    fi

    if [ -n "${ros_setup}" ] && [ -f "${ros_setup}" ] && [ -n "${python_bin}" ] && \
        ( set +u; set +e; source "${ros_setup}" >/dev/null 2>&1; "${python_bin}" -c 'import rclpy; print("ROS 2 Humble connection successful")' >/dev/null 2>&1 ); then
        log_info "Verification complete: venv can access ROS 2 packages."
        PYTHON_ENV_STATUS="done"
    else
        log_error "Verification failed: rclpy not found. Ensure ROS 2 is installed and --system-site-packages was used."
        log_error "If ${WORKSPACE}/venv was created without --system-site-packages, remove it and rerun ./scripts/setup.sh."
        PYTHON_ENV_STATUS="failed"
        exit 1
    fi

    # Note: do NOT gate on `[[ -x "${ROSDEPC_BIN}" ]]`. The venv/bin
    # directory is created with mode 0700 by `python3 -m venv`, and bash's
    # `-x` test consults the access(2)-style permission view, which can
    # disagree with the actual exec(2) capability under chroot/sudo
    # combinations (we have hit this on openEuler dev boards). Trying to
    # run the binary is both more accurate and more useful here.
    if [[ -z "${ROSDEPC_BIN:-}" ]]; then
        log_error "Verification failed: ROSDEPC_BIN is unset; ensure_workspace_venv was not called."
        exit 1
    fi
    if ! "${ROSDEPC_BIN}" --help >/dev/null 2>&1; then
        log_error "Verification failed: ${ROSDEPC_BIN} did not respond to --help."
        log_error "  - Check that rosdepc was installed into the workspace venv:"
        log_error "      ${VENV_PYTHON:-${WORKSPACE}/venv/bin/python3} -m pip show rosdepc"
        log_error "  - Re-run setup with VERBOSE=1 to see the install transcript."
        exit 1
    fi

    # Check colcon the same way build.sh does: import via the venv python with
    # PYTHONNOUSERSITE=1, NOT via `command -v colcon`. This catches the legacy
    # case where ~/.local/bin/colcon shadows a missing venv install.
    if ! PYTHONNOUSERSITE=1 "${venv_python}" -m colcon --help >/dev/null 2>&1; then
        log_error "Verification failed: 'python3 -m colcon --help' does not work inside the venv."
        log_error "build.sh runs colcon this exact way; please reinstall colcon into the venv:"
        log_error "  source venv/bin/activate"
        log_error "  PYTHONNOUSERSITE=1 python3 -m pip install --upgrade colcon-common-extensions colcon-mixin"
        exit 1
    fi

    # Keep the legacy CLI-on-PATH check as a soft signal for users who run
    # colcon directly outside of build.sh.
    if ! command -v colcon &>/dev/null; then
        log_warn "colcon is importable from the venv but no 'colcon' CLI is on PATH."
        log_warn "Activate the venv (source venv/bin/activate) before running colcon directly."
    fi

    # ROS 2's setup.sh reads AMENT_TRACE_SETUP_FILES directly and is not nounset-safe.
    if ! (set +u; source /opt/ros/humble/setup.sh && set -u && "${venv_python}" -c "import rclpy" >/dev/null 2>&1); then
        log_error "Verification failed: rclpy is not accessible from the virtual environment."
        exit 1
    fi

    if ! "${venv_python}" -c "import lerobot" >/dev/null 2>&1; then
        log_error "Verification failed: lerobot import failed."
        exit 1
    fi

    if ! "${venv_python}" -c "import numpy; assert numpy.__version__.startswith('1.26.')" >/dev/null 2>&1; then
        log_error "Verification failed: NumPy is not pinned to the expected ROS-compatible 1.26.x series."
        exit 1
    fi

    log_done "Verified ROS, rosdepc, colcon, lerobot, and NumPy compatibility"
}

# ============================================================================
# Main
# ============================================================================
main() {
    trap 'on_setup_failure $? $LINENO' ERR
    trap 'on_setup_interrupt' INT
    trap 'cleanup_setup_artifacts' EXIT
    parse_args "$@"

    cd "${WORKSPACE}"
    print_banner
    detect_ui_backend
    
    # Check for conflicting environments
    set_stage "detecting system"
    ui_stage_progress "detecting system"
    check_conda
    initialize_platform
    print_platform_summary
    set_stage "checking system dependencies"
    ui_stage_progress "checking system dependencies"
    print_dependency_preview
    
    # Apply git HTTP config if requested
    apply_git_http_config

    log_info "Setting up workspace at ${WORKSPACE}"
    
    # Update submodules
    set_stage "syncing submodules"
    update_submodules

    # Optional: Setup developer forks
    set_stage "configuring developer forks"
    setup_developer_forks
    
    # Install dependencies
    set_stage "installing system dependencies"
    install_system_deps
    if [[ "${SYSTEM_DEPS_STATUS}" == "done" ]]; then
        log_done "System ROS dependencies installed"
    fi
    set_stage "configuring python environment"
    setup_python_venv
    if [[ "${PYTHON_ENV_STATUS}" == "done" ]]; then
        log_done "Python virtual environment configured"
    fi

    # LeRobot patch stack is normally applied inline by setup_python_venv
    # (before install_lerobot_editable so check_lerobot_python_compat sees
    # the patched pyproject.toml). The call below is an idempotent safety
    # net for code paths that bypass setup_python_venv (e.g. --skip-python
    # on a workspace whose patched branch was lost). It is a no-op when
    # the patch stack is already applied.
    set_stage "verifying lerobot patch stack"
    ensure_lerobot_patch_stack_applied

    echo ""
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW} Setup Summary${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    for entry in "${SUMMARY[@]}"; do
        echo -e "  ${entry}"
    done
    echo ""
    if [[ "${USE_GUM}" == true ]]; then
        "${GUM_BIN}" style --foreground 42 --bold "[INFO] Setup complete! Run ./scripts/build.sh to build the workspace."
    else
        echo -e "\033[1;32m[INFO] Setup complete! Run ./scripts/build.sh to build the workspace.${NC}"
    fi
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

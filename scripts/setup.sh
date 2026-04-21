#!/bin/bash
# setup.sh - Workspace setup script for ROS 2 Humble
# Handles repository import, dependency installation, and environment setup
#
# Usage:
#   ./scripts/setup.sh                              # Interactive mode
#   ./scripts/setup.sh --yes                        # Auto-yes mode
#   ./scripts/setup.sh --skip-submodules            # Keep existing submodule state
#   ./scripts/setup.sh --skip-system-deps           # Skip ROS/system dependency installation
#   ./scripts/setup.sh --skip-python                # Skip venv/Python dependency setup
#   ./scripts/setup.sh --skip-verify                # Skip final verification
#   ./scripts/setup.sh --help                       # Show help
#
# Auto-yes defaults:
#   - Submodule init:  initialize all submodules (option 1)
#   - Fork setup:      skipped
#   - Other prompts:   confirmed automatically
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
WORKSPACE="${WORKSPACE:-$(pwd)}"
PARALLEL_WORKERS=$(($(nproc) / 2))
AUTO_YES=false
USE_SUDO=true
VERBOSE=false
DRY_RUN=false
SKIP_SUBMODULES=false
SKIP_SYSTEM_DEPS=false
SKIP_PYTHON=false
SKIP_VERIFY=false
SUMMARY=()
DETECTED_OS="unknown"
DETECTED_ACCELERATOR="cpu-only"

# Detect if running as root
if [[ $EUID -eq 0 ]]; then
    USE_SUDO=false
fi

# Helper to run commands with or without sudo
run_sudo() {
    if [[ "${USE_SUDO}" == true ]]; then
        run_cmd sudo "$@"
    else
        run_cmd "$@"
    fi
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }
log_debug()   { [[ "${VERBOSE}" == true ]] && echo -e "[DEBUG] $*"; }
log_done()    { SUMMARY+=("${GREEN}✓${NC} $*"); }
log_skipped() { SUMMARY+=("${YELLOW}⊘${NC} $* (skipped)"); }

print_cmd() {
    printf '%q ' "$@"
    printf '\n'
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

is_openeuler() {
    uname -r | grep -qi "openeuler" || grep -qi "openeuler" /etc/os-release 2>/dev/null
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
    local hint
    if [[ "${default}" == "y" ]]; then hint="Y/n"; else hint="y/N"; fi
    read -r -p "${prompt} [${hint}]: " REPLY
    REPLY="${REPLY:-${default}}"
    [[ "${REPLY}" == "y" || "${REPLY}" == "Y" ]]
}

show_help() {
    cat <<'EOF'
Workspace setup script for ROS 2 Humble

Usage:
  ./scripts/setup.sh [OPTIONS]

Options:
  -y, --yes              Auto-confirm prompts using default actions
      --sudo             Force sudo for privileged operations
      --no-sudo          Never use sudo
      --verbose          Show extra detection and execution details
      --dry-run          Print mutating commands without executing them
      --skip-submodules  Skip submodule initialization/update
      --skip-system-deps Skip ROS/system dependency installation
      --skip-python      Skip venv and Python dependency setup
      --skip-verify      Skip final verification stage
  -h, --help             Show this help

Supported platforms:
  - Ubuntu with ROS 2 Humble
  - openEuler Embedded with ROS 2 Humble

Supported accelerator profiles:
  - NVIDIA GPU
  - Ascend 310B / 310P
  - CPU-only fallback
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y) AUTO_YES=true ;;
            --no-sudo) USE_SUDO=false ;;
            --sudo) USE_SUDO=true ;;
            --verbose) VERBOSE=true ;;
            --dry-run) DRY_RUN=true ;;
            --skip-submodules) SKIP_SUBMODULES=true ;;
            --skip-system-deps) SKIP_SYSTEM_DEPS=true ;;
            --skip-python) SKIP_PYTHON=true ;;
            --skip-verify) SKIP_VERIFY=true ;;
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

detect_os() {
    local os_id="" os_name="" os_release_file="/etc/os-release"

    # Some embedded environments expose /etc/os-release as a symlink whose
    # readability check is unreliable via [[ -r ]]. Resolve it first when
    # possible, then fall back to direct reads.
    if command -v readlink &>/dev/null; then
        local resolved_os_release
        resolved_os_release="$(readlink -f /etc/os-release 2>/dev/null || true)"
        if [[ -n "${resolved_os_release}" ]]; then
            os_release_file="${resolved_os_release}"
        fi
    fi

    if [[ -r "${os_release_file}" ]]; then
        os_id="$(awk -F= '$1=="ID"{gsub(/"/,"",$2); print tolower($2)}' "${os_release_file}")"
        os_name="$(awk -F= '$1=="NAME"{gsub(/"/,"",$2); print tolower($2)}' "${os_release_file}")"
    else
        os_id="$(awk -F= '$1=="ID"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release 2>/dev/null || true)"
        os_name="$(awk -F= '$1=="NAME"{gsub(/"/,"",$2); print tolower($2)}' /etc/os-release 2>/dev/null || true)"
    fi

    if [[ "${os_id}" == "ubuntu" ]]; then
        DETECTED_OS="ubuntu"
    elif [[ "${os_id}" == "openeuler" ]] || [[ "${os_name}" == *"openeuler"* ]] || uname -r | grep -qi "openeuler"; then
        DETECTED_OS="openeuler-embedded"
    else
        log_error "Unsupported OS detected. This setup script supports Ubuntu and openEuler Embedded."
        exit 1
    fi
}

detect_accelerator() {
    DETECTED_ACCELERATOR="cpu-only"

    if command -v nvidia-smi &>/dev/null; then
        DETECTED_ACCELERATOR="nvidia-gpu"
        return
    fi

    if command -v npu-smi &>/dev/null; then
        local npu_info=""
        npu_info="$(npu-smi info 2>/dev/null || true)"
        if echo "${npu_info}" | grep -qi "310b"; then
            DETECTED_ACCELERATOR="ascend-310b"
            return
        fi
        if echo "${npu_info}" | grep -qi "310p"; then
            DETECTED_ACCELERATOR="ascend-310p"
            return
        fi
        DETECTED_ACCELERATOR="ascend-unknown"
        return
    fi

    if [[ -d /usr/local/Ascend ]]; then
        DETECTED_ACCELERATOR="ascend-unknown"
    fi
}

print_environment_summary() {
    echo ""
    echo -e "${YELLOW}--- Environment Detection ---${NC}"
    log_info "Detected OS: ${DETECTED_OS}"
    log_info "Detected accelerator: ${DETECTED_ACCELERATOR}"

    if [[ "${DETECTED_ACCELERATOR}" == "ascend-unknown" ]]; then
        log_warn "Ascend runtime detected, but the device model could not be identified as 310B or 310P."
    fi
}

# ============================================================================
# Repository Management
# ============================================================================
update_submodules() {
    if [[ "${SKIP_SUBMODULES}" == true ]]; then
        log_info "Skipping submodule initialization/update (--skip-submodules)."
        log_skipped "Submodule sync/update"
        return 0
    fi

    echo ""
    echo -e "${YELLOW}--- Git Submodule Management ---${NC}"

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
        if [[ ! -d "${path}/.git" ]]; then
            need_init+=("${path}:${name}")
        fi
    done

    # If all submodules exist, ask if user wants to update
    if [[ ${#need_init} -eq 0 ]]; then
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
        run_cmd git submodule update --init --recursive
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

    # Ask which submodules to initialize
    log_info "Select which submodules to initialize:"
    echo "  1) All submodules"
    echo "  2) LeRobot only (libs/lerobot)"
    echo "  3) PyMoveIt2 only (src/pymoveit2)"
    echo "  4) Select individually"
    echo "  0) Skip"
    echo ""
    if [[ "${AUTO_YES}" == true ]]; then
        CHOICE="1"
        log_info "Auto-yes: selecting option 1 (all submodules)"
    else
        read -r -p "Enter your choice [1-4, 0]: " CHOICE
    fi

    case "${CHOICE}" in
        1)
            log_info "Initializing all submodules..."
            export GIT_LFS_SKIP_SMUDGE=1
            run_cmd git submodule update --init --recursive
            log_done "Submodules initialized: all"
            ;;
        2)
            log_info "Initializing LeRobot (libs/lerobot)..."
            export GIT_LFS_SKIP_SMUDGE=1
            run_cmd git submodule update --init --recursive libs/lerobot
            log_done "Submodules initialized: LeRobot"
            ;;
        3)
            log_info "Initializing PyMoveIt2 (src/pymoveit2)..."
            export GIT_LFS_SKIP_SMUDGE=1
            run_cmd git submodule update --init --recursive src/pymoveit2
            log_done "Submodules initialized: PyMoveIt2"
            ;;
        4)
            echo ""
            for entry in "${need_init[@]}"; do
                local path="${entry%%:*}"
                local name="${entry##*:}"
                if ask_yn "Initialize ${name} (${path})?" "y"; then
                    log_info "Initializing ${name}..."
                    export GIT_LFS_SKIP_SMUDGE=1
                    run_cmd git submodule update --init --recursive "${path}"
                    log_done "Submodule initialized: ${name}"
                else
                    log_warn "Skipped ${name}"
                    log_skipped "Submodule: ${name}"
                fi
            done
            ;;
        0)
            log_warn "Submodule initialization skipped."
            log_skipped "Submodule initialization"
            ;;
        *)
            log_error "Invalid choice. Skipping submodule initialization."
            log_skipped "Submodule initialization (invalid choice)"
            ;;
    esac
}

setup_developer_forks() {
    echo ""
    echo -e "${YELLOW}--- Developer Fork Setup ---${NC}"
    echo "If you have forked the repository on GitCode, enter your username"
    echo "to automatically set up your personal fork as 'origin' and the"
    echo "original repository as 'upstream'."
    echo ""
    if [[ "${AUTO_YES}" == true ]]; then
        log_info "Auto-yes: skipping fork setup."
        log_skipped "Developer fork setup"
        return 0
    fi
    read -r -p "Enter your GitCode username (leave empty to skip): " USERNAME

    if [[ -n "${USERNAME}" ]]; then
        local MAIN_FORK LEROBOT_FORK UPSTREAM_URL

        MAIN_FORK="git@gitcode.com:${USERNAME}/IB_Robot.git"
        LEROBOT_FORK="git@gitcode.com:${USERNAME}/lerobot_ros2.git"
        UPSTREAM_URL="git@atomgit.com:openeuler/IB_Robot.git"

        echo -e "\nProposed Fork URLs:"
        echo -e "  Main Repo:    ${MAIN_FORK}"
        echo -e "  libs/lerobot: ${LEROBOT_FORK}"
        echo ""
        if ask_yn "Confirm setting these as 'origin'?" "n"; then
            log_info "Configuring personal forks..."
            
            # 1. Update main repo remotes
            run_cmd git remote set-url origin "${MAIN_FORK}"
            if git remote get-url upstream &>/dev/null; then
                run_cmd git remote set-url upstream "${UPSTREAM_URL}"
            else
                run_cmd git remote add upstream "${UPSTREAM_URL}"
            fi

            # 2. Update submodule fork
            if [[ -d "libs/lerobot/.git" ]]; then
                (cd libs/lerobot && run_cmd git remote set-url origin "${LEROBOT_FORK}")
                local LEROBOT_UPSTREAM=$(git config -f .gitmodules submodule.libs/lerobot.url)
                if (cd libs/lerobot && git remote get-url upstream &>/dev/null); then
                    (cd libs/lerobot && run_cmd git remote set-url upstream "${LEROBOT_UPSTREAM}")
                else
                    (cd libs/lerobot && run_cmd git remote add upstream "${LEROBOT_UPSTREAM}")
                fi
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
    # Check if ROS 2 Humble is installed
    if [[ ! -f /opt/ros/humble/setup.bash ]]; then
        log_warn "ROS 2 Humble not found at /opt/ros/humble/setup.bash"
        log_info "Running ROS 2 installation script..."

        local install_args=()
        if [[ "${AUTO_YES}" == true ]]; then
            install_args+=("--yes")
        fi
        if [[ "${USE_SUDO}" == false ]]; then
            install_args+=("--no-sudo")
        fi

        if run_cmd "${WORKSPACE}/scripts/install_ros.sh" "${install_args[@]}"; then
            log_done "ROS 2 Humble installed"
        else
            log_error "ROS 2 installation failed"
            log_error "Please run ${WORKSPACE}/scripts/install_ros.sh manually to diagnose the issue"
            exit 1
        fi
    else
        log_info "ROS 2 Humble is already installed"
    fi
}

ensure_colcon() {
    if command -v colcon &>/dev/null; then
        log_info "colcon is already installed"
        return 0
    fi

    log_info "Installing colcon build tool..."
    if command -v apt-get &> /dev/null; then
        run_sudo apt-get install -y python3-colcon-common-extensions
    elif command -v dnf &> /dev/null; then
        # On openEuler, we usually install colcon via pip to get the latest extensions
        if command -v pip3 &> /dev/null; then
            run_cmd pip3 install colcon-common-extensions --quiet
        else
            log_error "pip3 not found, cannot install colcon."
            exit 1
        fi
    fi
    log_done "colcon installed"
}

check_openeuler() {
    if [[ "${DETECTED_OS}" == "openeuler-embedded" ]]; then
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

ensure_rosdepc() {
    # 1. Ensure user's local bin is in PATH for pip installed tools
    if [[ -d "${HOME}/.local/bin" && ":$PATH:" != *":${HOME}/.local/bin:"* ]]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    fi

    if ! command -v rosdepc &> /dev/null; then
        log_warn "rosdepc not found. Installing rosdepc (rosdep with Chinese mirror support)..."
        if command -v pip3 &> /dev/null; then
            run_cmd pip3 install rosdepc
        elif command -v pip &> /dev/null; then
            run_cmd pip install rosdepc
        else
            log_error "pip/pip3 not found. Cannot install rosdepc automatically."
            exit 1
        fi
        
        # Refresh bash hash so command -v finds the newly installed binary
        hash -r
    fi

    if ! command -v rosdepc &> /dev/null; then
        log_error "rosdepc was installed but cannot be found in PATH. Please check your python environment."
        exit 1
    fi

    # Init if sources list doesn't exist yet
    if [[ ! -d /etc/ros/rosdep/sources.list.d ]]; then
        log_info "Initializing rosdepc..."

        if [[ "${DRY_RUN}" == true ]]; then
            local rosdepc_bin="rosdepc"
            if command -v rosdepc &>/dev/null; then
                rosdepc_bin="$(command -v rosdepc)"
            fi
            run_sudo -E env PATH="${PATH}" "${rosdepc_bin}" init
            return 0
        fi
        
        # Pre-authenticate sudo so password prompt is visible
        if [[ "${USE_SUDO}" == true ]]; then
            sudo -v
        fi

        local init_output=""
        local init_exit=0
        
        init_output=$(run_sudo env PATH="${PATH}" "$(command -v rosdepc)" init 2>&1) || init_exit=$?

        # Check both exit code and output for SSL/network errors
        if [[ ${init_exit} -ne 0 ]] || echo "${init_output}" | grep -qi "error\|failed\|certificate\|urlopen"; then
            if echo "${init_output}" | grep -qi "certificate\|ssl\|urlopen"; then
                log_warn "SSL certificate error detected during rosdepc init:"
                echo "${init_output}"
                log_warn "Attempting to fix SSL certificates..."
            else
                log_warn "rosdepc init failed, attempting SSL certificate fix..."
                echo "${init_output}"
            fi

            # Get the .pem path used by Python's ssl module
            local ssl_pem
            ssl_pem=$(python3 -c "import ssl; print(ssl.get_default_verify_paths().openssl_cafile)" 2>/dev/null)

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

                # Retry init, capture output again
                local retry_output
                if ! retry_output=$(run_sudo rosdepc init 2>&1) || echo "${retry_output}" | grep -qi "error\|failed\|certificate\|urlopen"; then
                    log_error "rosdepc init failed even after SSL fix."
                    echo "${retry_output}"
                    log_error "Try running manually: run_sudo rosdepc init"
                    exit 1
                fi
            else
                log_warn "Skipped SSL fix. Please manually check your network or certificate configuration."
                log_warn "You can also try running: run_sudo rosdepc init"
                exit 1
            fi
        fi
    fi
}

install_system_deps() {
    if [[ "${SKIP_SYSTEM_DEPS}" == true ]]; then
        log_info "Skipping ROS/system dependency installation (--skip-system-deps)."
        log_skipped "System ROS dependencies"
        return 0
    fi

    # Check for ROS 2 installation first
    check_ros_installation
    ensure_colcon

    check_openeuler
    ensure_rosdepc

    if [[ "${DETECTED_OS}" == "ubuntu" ]]; then
        if ! command -v apt-get &> /dev/null; then
            log_error "apt-get not found on Ubuntu system."
            exit 1
        fi
        log_info "Updating apt package lists..."
        run_sudo apt-get update -qq

        log_info "Updating rosdepc database..."
        if ! run_cmd rosdepc update --rosdistro=humble; then
            log_error "rosdepc update failed. This is usually due to network issues."
            log_error "Please check your network connection and re-run ./scripts/setup.sh"
            exit 1
        fi

        log_info "Installing ROS dependencies via apt..."
        if ! run_cmd rosdepc install \
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
    elif [[ "${DETECTED_OS}" == "openeuler-embedded" ]]; then
        if ! command -v dnf &> /dev/null; then
            log_error "dnf not found on openEuler Embedded system."
            exit 1
        fi
        log_info "Updating dnf package repositories..."

        log_info "Updating rosdepc database..."
        if ! run_cmd rosdepc update --rosdistro=humble; then
            log_error "rosdepc update failed. This is usually due to network issues."
            log_error "Please check your network connection and re-run ./scripts/setup.sh"
            exit 1
        fi

        log_info "Installing ROS dependencies via dnf..."
        if ! run_cmd rosdepc install \
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
            # openEuler relies on manual/system installs for these keys or does
            # not currently provide matching rosdep package mappings.
            --skip-keys=nlohmann-json-dev \
            --skip-keys=python3-opencv \
            --skip-keys=python3-aiortc \
            --skip-keys=gz_ros2_control \
            --skip-keys=ros_gz_sim \
            --skip-keys=ros_gz_bridge \
            --skip-keys=mujoco_ros2_control \
            --skip-keys=pyserial; then
            log_error "rosdepc install failed."
            log_error "Please check your network connection or dependency lists and re-run ./scripts/setup.sh"
            exit 1
        fi
    else
        log_warn "Unknown package manager. Please ensure ROS 2 Humble dependencies are installed manually."
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_done "System ROS dependency steps planned"
    else
        log_done "System ROS dependencies installed"
    fi
}

setup_python_venv() {
    if [[ "${SKIP_PYTHON}" == true ]]; then
        log_info "Skipping Python environment setup (--skip-python)."
        log_skipped "Python virtual environment"
        return 0
    fi

    local venv_path="${WORKSPACE}/venv"
    local lerobot_dir="${WORKSPACE}/libs/lerobot"
    local lerobot_submodule_status=""
    
    # 1. Ensure system-level venv tools are installed
    log_info "Checking for Python venv and pip..."
    if [[ "${DETECTED_OS}" == "ubuntu" ]]; then
        run_sudo apt-get update -qq
        run_sudo apt-get install -y python3-venv python3-pip -qq
    elif [[ "${DETECTED_OS}" == "openeuler-embedded" ]]; then
        run_sudo dnf install -y --nogpgcheck python3-virtualenv python3-pip python3-devel -q
    fi

    # 2. 创建虚拟环境 (必须包含 --system-site-packages 以使用系统的 rclpy)
    if [[ ! -d "${venv_path}" ]]; then
        log_info "Creating virtual environment at ${venv_path} with --system-site-packages..."
        run_cmd python3 -m venv --system-site-packages "${venv_path}"
    else
        log_info "Virtual environment already exists at ${venv_path}."
    fi

    lerobot_submodule_status="$(git submodule status -- libs/lerobot 2>/dev/null || true)"
    if [[ "${lerobot_submodule_status}" == -* ]]; then
        log_error "LeRobot submodule is not initialized at ${lerobot_dir}."
        log_error "This usually happens when submodule initialization was skipped."
        log_error "Run: git submodule update --init --recursive libs/lerobot"
        exit 1
    fi

    if [[ ! -d "${lerobot_dir}" ]]; then
        log_error "LeRobot dependency directory not found at ${lerobot_dir}."
        log_error "Initialize the submodule with: git submodule update --init --recursive libs/lerobot"
        exit 1
    fi

    if [[ ! -f "${lerobot_dir}/pyproject.toml" || ! -d "${lerobot_dir}/src/lerobot" ]]; then
        log_error "LeRobot submodule at ${lerobot_dir} appears empty or uninitialized."
        log_error "This usually happens when submodule initialization was skipped."
        log_error "Run: git submodule update --init --recursive libs/lerobot"
        exit 1
    fi

    # 3. 激活虚拟环境并安装依赖
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "Skipping venv activation in dry-run mode."
        run_cmd python3 -m pip install --upgrade pip --quiet
        run_cmd python3 -m pip install "setuptools<80" "setuptools>=71" --quiet
        run_cmd python3 -m pip install -e "${lerobot_dir}"
        run_cmd python3 -m pip install pyserial feetech-servo-sdk --quiet
        if [[ "${DETECTED_OS}" == "openeuler-embedded" ]]; then
            run_cmd python3 -m pip install aiortc --quiet
        fi
        run_cmd python3 -m pip install scipy --quiet
        run_cmd python3 -m pip install gitlint --quiet
        run_cmd python3 -m pip install "numpy==1.26.4" --quiet
        run_cmd gitlint install-hook
        log_done "Python dependencies planned for venv"
        return 0
    fi

    log_info "Configuring Python environment and dependencies..."
    source "${venv_path}/bin/activate"
    
    # 升级 pip
    run_cmd python3 -m pip install --upgrade pip --quiet
    
    # 解决 setuptools 版本冲突 (兼容 LeRobot 和 colcon)
    run_cmd python3 -m pip install "setuptools<80" "setuptools>=71" --quiet

    # 以可编辑模式安装 LeRobot
    log_info "Installing LeRobot in editable mode..."
    run_cmd python3 -m pip install -e "${lerobot_dir}"

    # 安装原有的硬件依赖
    log_info "Installing hardware dependencies (pyserial, feetech)..."
    run_cmd python3 -m pip install pyserial feetech-servo-sdk --quiet

    if [[ "${DETECTED_OS}" == "openeuler-embedded" ]]; then
        log_info "Installing openEuler fallback dependency (aiortc) into the workspace venv..."
        run_cmd python3 -m pip install aiortc --quiet
    fi

    # 安装 scipy 用于数学计算 (四元数/旋转矩阵转换)
    log_info "Installing scipy for mathematical computations..."
    run_cmd python3 -m pip install scipy --quiet

    # 安装训练可视化依赖
    log_info "Installing training visualization dependencies (tensorboard)..."
    python3 -m pip install tensorboard --quiet

    # 安装录制可视化依赖
    log_info "Installing recording visualization dependency (rerun-sdk)..."
    python3 -m pip install "rerun-sdk>=0.24,<0.27" --quiet
    log_info "Installing rerun compatibility dependency (typing-extensions>=4.12)..."
    python3 -m pip install "typing-extensions>=4.12" --quiet

    # 安装 ONNX 导出相关依赖
    if is_openeuler; then
        log_info "Installing ONNX export dependencies (onnx, onnxruntime); skipping onnxsim on openEuler..."
        python3 -m pip install onnx onnxruntime --quiet
    else
        log_info "Installing ONNX export dependencies (onnx, onnxsim, onnxruntime)..."
        python3 -m pip install onnx onnxsim onnxruntime --quiet
    fi

    # 安装 gitlint 并设置 git hook
    log_info "Installing gitlint..."
    run_cmd python3 -m pip install gitlint --quiet

    # 安装 ruff (代码规范) 和 pre-commit hook
    log_info "Installing ruff and pre-commit..."
    python3 -m pip install ruff pre-commit --quiet
    if [[ -f "${WORKSPACE}/.pre-commit-config.yaml" ]]; then
        pre-commit install
    fi

    # 核心修复：所有依赖安装完毕后，强制固定 NumPy 1.26.4 以兼容 ROS 2 系统组件
    # 必须放在最后，防止 lerobot/scipy 等依赖将 numpy 升级到 2.x
    log_info "Pinning NumPy to 1.26.4 for ROS 2 compatibility..."
    run_cmd python3 -m pip install "numpy==1.26.4" --quiet
    local commit_msg_hook
    commit_msg_hook="$(git rev-parse --git-path hooks/commit-msg)"
    if [[ -e "${commit_msg_hook}" ]]; then
        log_warn "gitlint commit-msg hook already exists at ${commit_msg_hook}; keeping it."
        log_done "gitlint commit-msg hook already exists"
    else
        log_info "Installing gitlint commit-msg hook..."
        if [[ "${DRY_RUN}" == true ]]; then
            run_cmd gitlint install-hook
        else
            printf 'y\n' | gitlint install-hook
        fi
        log_done "gitlint commit-msg hook installed"
    fi

    log_done "Python environment configured"
}

verify_setup() {
    if [[ "${SKIP_VERIFY}" == true ]]; then
        log_info "Skipping final verification (--skip-verify)."
        log_skipped "Setup verification"
        return 0
    fi

    local venv_python="${WORKSPACE}/venv/bin/python3"

    log_info "Running final verification..."

    if [[ ! -x "${venv_python}" ]]; then
        log_error "Verification failed: virtual environment python not found at ${venv_python}"
        exit 1
    fi

    if ! command -v rosdepc &>/dev/null; then
        log_error "Verification failed: rosdepc is not available in PATH."
        exit 1
    fi

    if ! command -v colcon &>/dev/null; then
        log_error "Verification failed: colcon is not available in PATH."
        exit 1
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

print_summary() {
    echo ""
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW} Setup Summary${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    for entry in "${SUMMARY[@]}"; do
        echo -e "  ${entry}"
    done
    echo ""
}

print_next_steps() {
    log_info "Setup complete! Recommended next steps:"
    echo "  source venv/bin/activate"
    echo "  source /opt/ros/humble/setup.sh"
    echo "  ./scripts/build.sh"
    echo "  # After the first build, also execute: source install/setup.sh"
}

# ============================================================================
# Main
# ============================================================================
main() {
    parse_args "$@"

    cd "${WORKSPACE}"
    
    # Check for conflicting environments
    check_conda
    detect_os
    detect_accelerator
    print_environment_summary
    
    log_info "Setting up workspace at ${WORKSPACE}"
    
    # Update submodules
    update_submodules
    
    # Optional: Setup developer forks
    setup_developer_forks
    
    # Install dependencies
    install_system_deps
    setup_python_venv
    verify_setup
    print_summary
    print_next_steps
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

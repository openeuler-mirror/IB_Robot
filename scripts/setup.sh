#!/bin/bash
# setup.sh - Workspace setup script for ROS 2 Humble
# Handles repository import, dependency installation, and environment setup
#
# Usage:
#   ./scripts/setup.sh               # Interactive mode (prompts for each step)
#   ./scripts/setup.sh --yes         # Auto-yes mode (skips all prompts with defaults)
#   ./scripts/setup.sh -y            # Same as --yes
#   ./scripts/setup.sh --git-http    # Use HTTP instead of SSH for git remotes
#   ./scripts/setup.sh -y --git-http # Combine options
#
# Auto-yes defaults:
#   - Submodule init:  initialize all submodules (option 1)
#   - Fork setup:      skipped
#   - Other prompts:   confirmed automatically
set -e

# ============================================================================
# Configuration
# ============================================================================
WORKSPACE="${WORKSPACE:-$(pwd)}"
PARALLEL_WORKERS=$(($(nproc) / 2))
AUTO_YES=false
GIT_HTTP=false
USE_SUDO=true
SUMMARY=()

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

# ============================================================================
# Environment Checks
# ============================================================================
check_conda() {
    if [[ -n "${CONDA_PREFIX}" ]]; then
        log_error "Active Conda environment detected at: ${CONDA_PREFIX}"
        log_warn "Conda environments are known to conflict with ROS 2 dependencies (especially Python libraries)."
        log_warn "Please deactivate the Conda environment before running this script:"
        echo -e "    ${YELLOW}conda deactivate${NC}"
        exit 1
    fi
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
        git submodule update --init --recursive
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
            git submodule update --init --recursive
            log_done "Submodules initialized: all"
            ;;
        2)
            log_info "Initializing LeRobot (libs/lerobot)..."
            export GIT_LFS_SKIP_SMUDGE=1
            git submodule update --init --recursive libs/lerobot
            log_done "Submodules initialized: LeRobot"
            ;;
        3)
            log_info "Initializing PyMoveIt2 (src/pymoveit2)..."
            export GIT_LFS_SKIP_SMUDGE=1
            git submodule update --init --recursive src/pymoveit2
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
                    git submodule update --init --recursive "${path}"
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

        # We define as SSH format, which will be auto-translated to HTTPS 
        # if GIT_HTTP is true due to 'insteadOf' config applied in main()
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

        if "${WORKSPACE}/scripts/install_ros.sh" "${install_args[@]}"; then
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
            pip3 install colcon-common-extensions --quiet
        else
            log_error "pip3 not found, cannot install colcon."
            exit 1
        fi
    fi
    log_done "colcon installed"
}

check_openeuler() {
    if uname -r | grep -qi "openeuler"; then
        log_warn "openEuler detected. Setting ROS_OS_OVERRIDE=rhel:8 for rosdep compatibility."
        export ROS_OS_OVERRIDE=rhel:8

        if ! dnf repolist | grep -qi "openEuler-24.03-LTS"; then
            local arch
            arch=$(uname -m)
            log_info "Adding openEuler repo for ${arch}..."
            run_sudo dnf config-manager --add-repo "https://repo.openeuler.org/openEuler-24.03-LTS/OS/${arch}"
            run_sudo dnf clean all && run_sudo dnf makecache
        else
            log_info "openEuler repo already configured, skipping add-repo."
        fi

        log_info "Installing gcc-c++, vim-enhanced, ffmpeg-devel, libvpx, and libvpx-devel..."
        run_sudo dnf install -y --nogpgcheck gcc-c++ vim-enhanced ffmpeg-devel libvpx libvpx-devel

        # usb_cam is not available as a system package on openEuler,
        # so we initialize the submodule to build from source.
        if [[ ! -d "${WORKSPACE}/src/usb_cam/.git" ]]; then
            log_info "usb_cam package not available on openEuler, initializing submodule for source build..."
            export GIT_LFS_SKIP_SMUDGE=1
            git submodule update --init --recursive src/usb_cam
            log_done "usb_cam submodule initialized (source build for openEuler)"
        else
            log_info "usb_cam submodule already initialized."
        fi
    fi
}

ensure_rosdepc() {
    if ! command -v rosdepc &> /dev/null; then
        log_warn "rosdepc not found. Installing rosdepc (rosdep with Chinese mirror support)..."
        if command -v pip3 &> /dev/null; then
            pip3 install rosdepc
        elif command -v pip &> /dev/null; then
            pip install rosdepc
        else
            log_error "pip/pip3 not found. Cannot install rosdepc automatically."
            exit 1
        fi
    fi

    # Init if sources list doesn't exist yet
    if [[ ! -d /etc/ros/rosdep/sources.list.d ]]; then
        log_info "Initializing rosdepc..."
        local init_output
        init_output=$(run_sudo rosdepc init 2>&1)
        local init_exit=$?

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

            log_warn "This fix will copy the system CA bundle to Python's SSL cert path."
            log_warn "This is a global change that may affect other Python applications."
            if ask_yn "Apply SSL certificate fix (copy system CA bundle to Python SSL path)?" "n"; then
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
    # Check for ROS 2 installation first
    check_ros_installation
    ensure_colcon

    check_openeuler
    ensure_rosdepc

    if command -v apt-get &> /dev/null; then
        log_info "Updating apt package lists..."
        run_sudo apt-get update -qq

        log_info "Updating rosdepc database..."
        rosdepc update --rosdistro=humble

        log_info "Installing ROS dependencies via apt..."
        rosdepc install \
            --from-paths src \
            --ignore-src \
            --rosdistro=humble \
            -y -r \
            --skip-keys "catkin roscpp lerobot trimesh[easy] simple-parsing cupy-cuda12x ctl_system_interface numpy_lessthan_2 ament_python feetech-servo-sdk pyserial"
    elif command -v dnf &> /dev/null; then
        log_info "Updating dnf package repositories..."

        log_info "Updating rosdepc database..."
        rosdepc update --rosdistro=humble

        log_info "Installing ROS dependencies via dnf..."
        rosdepc install \
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
            --skip-keys=pyserial \
            --skip-keys=usb_cam
    else
        log_warn "Unknown package manager. Please ensure ROS 2 Humble dependencies are installed manually."
    fi
}

setup_python_venv() {
    local venv_path="${WORKSPACE}/venv"
    
    # 1. Ensure system-level venv tools are installed
    log_info "Checking for Python venv and pip..."
    if command -v apt-get &> /dev/null; then
        run_sudo apt-get update -qq
        run_sudo apt-get install -y python3-venv python3-pip -qq
    elif command -v dnf &> /dev/null; then
        run_sudo dnf install -y --nogpgcheck python3-virtualenv python3-pip python3-devel -q
    fi

    # 2. 创建虚拟环境 (必须包含 --system-site-packages 以使用系统的 rclpy)
    if [[ ! -d "${venv_path}" ]]; then
        log_info "Creating virtual environment at ${venv_path} with --system-site-packages..."
        python3 -m venv --system-site-packages "${venv_path}"
    else
        log_info "Virtual environment already exists at ${venv_path}."
    fi

    # 3. 激活虚拟环境并安装依赖
    log_info "Configuring Python environment and dependencies..."
    source "${venv_path}/bin/activate"
    
    # 升级 pip
    python3 -m pip install --upgrade pip --quiet
    
    # 解决 setuptools 版本冲突 (兼容 LeRobot 和 colcon)
    python3 -m pip install "setuptools<80" "setuptools>=71" --quiet

    # 以可编辑模式安装 LeRobot
    if [[ -d "${WORKSPACE}/libs/lerobot" ]]; then
        log_info "Installing LeRobot in editable mode..."
        python3 -m pip install -e "${WORKSPACE}/libs/lerobot"
    fi

    # 安装原有的硬件依赖
    log_info "Installing hardware dependencies (pyserial, feetech)..."
    python3 -m pip install pyserial feetech-servo-sdk --quiet

    # 安装 scipy 用于数学计算 (四元数/旋转矩阵转换)
    log_info "Installing scipy for mathematical computations..."
    python3 -m pip install scipy --quiet

    # 安装 gitlint 并设置 git hook
    log_info "Installing gitlint..."
    python3 -m pip install gitlint --quiet

    # 核心修复：所有依赖安装完毕后，强制固定 NumPy 1.26.4 以兼容 ROS 2 系统组件
    # 必须放在最后，防止 lerobot/scipy 等依赖将 numpy 升级到 2.x
    log_info "Pinning NumPy to 1.26.4 for ROS 2 compatibility..."
    python3 -m pip install "numpy==1.26.4" --quiet
    log_info "Installing gitlint pre-commit hook..."
    yes | gitlint install-hook

    # 4. 环境验证
    log_info "Verifying ROS 2 connection..."
    # 显式加载 ROS 2 环境后再验证
    if (source /opt/ros/humble/setup.sh && python3 -c "import rclpy; print('ROS 2 Humble connection successful')") 2>/dev/null; then
        log_info "Verification complete: venv can access ROS 2 packages."
    else
        log_error "Verification failed: rclpy not found. Ensure ROS 2 is installed and --system-site-packages was used."
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    # Parse arguments
    for arg in "$@"; do
        case "${arg}" in
            --yes|-y) AUTO_YES=true ;;
            --git-http) GIT_HTTP=true ;;
            --no-sudo) USE_SUDO=false ;;
            --sudo) USE_SUDO=true ;;
            *) log_warn "Unknown argument: ${arg}" ;;
        esac
    done

    cd "${WORKSPACE}"
    
    # Check for conflicting environments
    check_conda
    
    # Apply git HTTP config if requested
    apply_git_http_config

    log_info "Setting up workspace at ${WORKSPACE}"
    
    # Update submodules
    update_submodules
    
    # Optional: Setup developer forks
    setup_developer_forks
    
    # Install dependencies
    install_system_deps
    log_done "System ROS dependencies installed"
    setup_python_venv
    log_done "Python virtual environment configured"

    echo ""
    echo -e "${YELLOW}============================================================${NC}"
    echo -e "${YELLOW} Setup Summary${NC}"
    echo -e "${YELLOW}============================================================${NC}"
    for entry in "${SUMMARY[@]}"; do
        echo -e "  ${entry}"
    done
    echo ""
    log_info "Setup complete! Run ./scripts/build.sh to build the workspace."
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
#!/bin/bash
# install_ros.sh - Automated ROS 2 Humble installation script
#
# Description:
#   This script automates the installation of ROS 2 Humble
#   on supported Linux distributions. It detects the operating system,
#   configures the appropriate package repositories, and installs all required
#   components.
#
# Supported Operating Systems:
#   - Ubuntu 22.04 (Jammy Jellyfish) - using official ROS 2 apt repository
#   - openEuler 24.03-LTS - using ROS-SIG community repository
#
# What gets installed:
#   - ROS 2 Humble (desktop on Ubuntu, base + selected packages on openEuler)
#   - Required dependencies for the IB_Robot project (MoveIt, ros2_control, etc.)
#
# Usage:
#   ./scripts/install_ros.sh           # Interactive mode with prompts
#   ./scripts/install_ros.sh --yes     # Automated mode (skip prompts)
#   ./scripts/install_ros.sh --help    # Show help message
#
# Requirements:
#   - sudo privileges (or root) for package installation
#   - internet connection for downloading packages
#   - supported operating system (Ubuntu or openEuler)
#
# Known Limitations:
#   - Only supports ROS 2 Humble (not other distributions)
#   - Only supports Ubuntu and openEuler (not other Linux distributions)
#   - openEuler support is primarily tested on aarch64 architecture
#   - Requires approximately 2-3 GB of disk space for ROS 2 installation
#
# Exit Codes:
#   0 - Success
#   1 - Error (see error message for details)
#
# For more information, see: https://docs.ros.org/en/humble/Installation.html
#
set -e

# ============================================================================
# Configuration
# ============================================================================
AUTO_YES=false
USE_SUDO=true

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

# Logging functions
log_info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================================
# Argument Parsing
# ============================================================================
parse_args() {
    for arg in "$@"; do
        case "${arg}" in
            --yes|-y) AUTO_YES=true ;;
            --no-sudo) USE_SUDO=false ;;
            --sudo) USE_SUDO=true ;;
            --help|-h) show_usage; exit 0 ;;
            *)
                log_error "Unknown argument: ${arg}"
                show_usage
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# Pre-flight Checks
# ============================================================================
check_sudo() {
    if [[ "${USE_SUDO}" == false ]]; then
        log_info "Running without sudo (root or --no-sudo requested)"
        return 0
    fi
    if ! sudo -v &>/dev/null; then
        log_error "This script requires sudo privileges to install packages."
        log_error "Please ensure you have sudo access and try again."
        exit 1
    fi
}

show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Automated ROS 2 Humble installation script.

OPTIONS:
    --yes, -y       Auto-yes mode (skip all confirmation prompts)
    --no-sudo       Run without sudo (for root users or containers)
    --sudo          Force use of sudo (default if not root)
    --help, -h      Show this help message and exit

SUPPORTED OPERATING SYSTEMS:
    - Ubuntu (22.04 Jammy recommended)
    - openEuler (24.03-LTS)

WHAT THIS SCRIPT INSTALLS:
    - ROS 2 Humble (desktop or base package depending on OS)
    - Required dependencies for IB_Robot project

EXAMPLES:
    $(basename "$0")           # Interactive mode with prompts
    $(basename "$0") --yes     # Automated installation (no prompts)

EOF
}

# ============================================================================
# OS Detection
# ============================================================================
detect_os() {
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot detect operating system: /etc/os-release not found."
        log_error "This script supports Ubuntu and openEuler only."
        exit 1
    fi

    . /etc/os-release

    OS_ID="$ID"
    OS_VERSION="$VERSION_ID"
    log_info "Detected OS: $PRETTY_NAME"

    if [[ "$OS_ID" != "ubuntu" && "$OS_ID" != "openeuler" ]]; then
        log_error "Unsupported operating system: $OS_ID"
        log_error "This script supports Ubuntu and openEuler only."
        exit 1
    fi
}

detect_architecture() {
    ARCH=$(uname -m)
    log_info "Detected architecture: $ARCH"

    # Validate architecture for openEuler (only aarch64 is supported by the repo)
    if [[ "$OS_ID" == "openeuler" && "$ARCH" != "aarch64" && "$ARCH" != "x86_64" ]]; then
        log_warn "Detected architecture '$ARCH' may not be supported by openEuler ROS packages."
        log_warn "The ROS-SIG repository primarily supports aarch64."
    fi
}

setup_os_variables() {
    if [[ "$OS_ID" == "ubuntu" ]]; then
        PACKAGE_MANAGER="apt"
        ROS_DISTRO="humble"
        # Ubuntu uses official ROS 2 repository
        ROS_REPO_URL="http://packages.ros.org/ros2/ubuntu"
        ROS_GPG_KEY="https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc"
    elif [[ "$OS_ID" == "openeuler" ]]; then
        PACKAGE_MANAGER="dnf"
        ROS_DISTRO="humble"
        # openEuler uses ROS-SIG community repository
        # URL is dynamically constructed with architecture
        ROS_REPO_URL="https://eulermaker.compass-ci.openeuler.openatom.cn/api/ems1/repositories/ROS-SIG-Multi-Version_ros-${ROS_DISTRO}_openEuler-24.03-LTS-TEST4/openEuler%3A24.03-LTS/${ARCH}/"
    fi
}

# ============================================================================
# Installation Detection
# ============================================================================
check_ros_installation() {
    ROS_SETUP_PATH="/opt/ros/${ROS_DISTRO}/setup.bash"

    if [[ -f "$ROS_SETUP_PATH" ]]; then
        log_info "ROS 2 ${ROS_DISTRO} is already installed at $ROS_SETUP_PATH"
        ROS_INSTALLED=true
    else
        log_info "ROS 2 ${ROS_DISTRO} is not installed"
        ROS_INSTALLED=false
    fi
}

# Check if everything is already installed
check_if_complete() {
    if [[ "$ROS_INSTALLED" == true ]]; then
        log_info "============================================================"
        log_info "ROS 2 ${ROS_DISTRO} is already installed!"
        log_info "============================================================"
        log_info "No installation needed. You're all set!"
        log_info "To use ROS 2, source the setup script:"
        log_info "  source /opt/ros/${ROS_DISTRO}/setup.bash"
        exit 0
    fi
}

confirm_installation() {
    local components=()
    if [[ "$ROS_INSTALLED" == false ]]; then
        components+=("ROS 2 ${ROS_DISTRO}")
    fi

    local list
    list=$(IFS=", "; echo "${components[*]}")

    echo ""
    log_info "The following components will be installed: $list"
    echo ""

    if [[ "$AUTO_YES" == true ]]; then
        log_info "Auto-yes mode: proceeding with installation"
        return 0
    fi

    read -r -p "Continue? [y/N]: " REPLY
    if [[ "$REPLY" != "y" && "$REPLY" != "Y" ]]; then
        log_info "Installation cancelled by user"
        exit 0
    fi
}

# ============================================================================
# Ubuntu Installation
# ============================================================================
install_ubuntu_ros() {
    if [[ "$ROS_INSTALLED" == true ]]; then
        log_info "Skipping ROS 2 installation (already installed)"
        return 0
    fi

    log_info "Installing ROS 2 ${ROS_DISTRO} on Ubuntu..."

    # Add ROS 2 GPG key
    log_info "Adding ROS 2 GPG key..."
    if ! run_sudo apt update &>/dev/null; then
        log_error "Failed to update package lists"
        return 1
    fi

    if ! run_sudo apt install -y ca-certificates &>/dev/null; then
        log_error "Failed to install ca-certificates"
        return 1
    fi

    if ! run_sudo install -m 0755 -d /etc/apt/keyrings &>/dev/null; then
        log_error "Failed to create keyrings directory"
        return 1
    fi

    if ! curl -fsSL "$ROS_GPG_KEY" -o /tmp/ros.asc 2>/dev/null; then
        log_error "Failed to download ROS 2 GPG key from $ROS_GPG_KEY"
        log_error "Please check your internet connection"
        return 1
    fi

    if ! run_sudo mv /tmp/ros.asc /etc/apt/keyrings/ros.asc &>/dev/null; then
        log_error "Failed to move GPG key to keyrings directory"
        return 1
    fi

    if ! run_sudo chmod a+r /etc/apt/keyrings/ros.asc &>/dev/null; then
        log_error "Failed to set GPG key permissions"
        return 1
    fi

    log_info "ROS 2 GPG key added successfully"

    # Add ROS 2 repository to sources list
    log_info "Adding ROS 2 repository to apt sources..."
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros.asc] $ROS_REPO_URL $(lsb_release -cs) main" | \
        run_sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    log_info "ROS 2 repository added successfully"

    # Update package lists with retry logic
    log_info "Updating package lists..."
    local max_retries=3
    local retry=0
    local success=false

    while [[ $retry -lt $max_retries && $success == false ]]; do
        if run_sudo apt-get update -qq; then
            success=true
            log_info "Package lists updated successfully"
        else
            retry=$((retry + 1))
            if [[ $retry -lt $max_retries ]]; then
                log_warn "apt-get update failed (attempt $retry/$max_retries), retrying..."
                sleep 2
            else
                log_error "apt-get update failed after $max_retries attempts"
                log_error "Please check your internet connection and try again"
                return 1
            fi
        fi
    done

    # Install ROS 2 Humble desktop-full
    log_info "Installing ROS 2 ${ROS_DISTRO} desktop-full..."
    if ! run_sudo apt-get install -y ros-${ROS_DISTRO}-desktop-full; then
        log_error "Failed to install ROS 2 ${ROS_DISTRO} desktop-full"
        return 1
    fi
    log_info "ROS 2 ${ROS_DISTRO} desktop-full installed successfully"
}

# ============================================================================
# openEuler Installation
# ============================================================================
install_openeuler_ros() {
    if [[ "$ROS_INSTALLED" == true ]]; then
        log_info "Skipping ROS 2 installation (already installed)"
        return 0
    fi

    log_info "Installing ROS 2 ${ROS_DISTRO} on openEuler..."

    # Create ROS.repo with dynamic architecture
    log_info "Creating ROS repository configuration..."
    run_sudo bash -c "cat << 'EOF' > /etc/yum.repos.d/openEulerROS.repo
[openEuler-Embedded-ROS-humble]
name=openEuler-Embedded-ROS-humble
baseurl=https://eur.openeuler.openatom.cn/results/openEuler_Embedded/IB_Robot-ROS_humble-release_1/openeuler-24.03_LTS-\$basearch/
skip_if_unavailable=True
enabled=1
gpgcheck=0
priority=1

[openEulerROS-humble]
name=openEulerROS-humble
baseurl=https://eulermaker.compass-ci.openeuler.openatom.cn/api/ems1/repositories/ROS-SIG-Multi-Version_ros-humble_openEuler-24.03-LTS-TEST4/openEuler%3A24.03-LTS/\$basearch/
enabled=1
gpgcheck=0
priority=2
EOF"

    if [[ ! -f /etc/yum.repos.d/openEulerROS.repo ]]; then
        log_error "Failed to create /etc/yum.repos.d/openEulerROS.repo"
        return 1
    fi

    log_info "ROS repository configuration created successfully"

    # Update package cache
    log_info "Updating dnf package cache..."
    if ! run_sudo dnf clean all &>/dev/null; then
        log_warn "dnf clean all failed, continuing..."
    fi

    if ! run_sudo dnf makecache; then
        log_error "Failed to update dnf package cache"
        log_error "Please check your internet connection and repository configuration"
        return 1
    fi

    log_info "Package cache updated successfully"

    # Install ROS 2 packages
    install_openeuler_ros_packages
}

install_openeuler_ros_packages() {
    log_info "Installing ROS 2 packages for openEuler..."

    # List of ROS 2 packages to install
    local ros_packages=(
        "ros-${ROS_DISTRO}-ros-base"
    )

    for pkg in "${ros_packages[@]}"; do
        log_info "Installing $pkg..."
        if ! run_sudo dnf install -y --nogpgcheck "$pkg"; then
            log_warn "Failed to install $pkg (may not be available in repo)"
        else
            log_info "Successfully installed $pkg"
        fi
    done

    log_info "ROS 2 packages installation completed"
}

# ============================================================================
# Error Handling
# ============================================================================
handle_installation_error() {
    local component="$1"
    log_error "Failed to install $component"
    log_error "Please check:"
    log_error "  1. Your internet connection"
    log_error "  2. Repository configuration"
    log_error "  3. Available disk space"
    log_error ""
    log_error "For Ubuntu, ensure you can access: $ROS_REPO_URL"
    log_error "For openEuler, ensure you can access: $ROS_REPO_URL"
}

# ============================================================================
# Verification
# ============================================================================
verify_installation() {
    local errors=0

    echo ""
    log_info "============================================================"
    log_info "Verifying installation..."
    log_info "============================================================"

    # Check ROS 2 installation
    if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
        log_info "✓ ROS 2 ${ROS_DISTRO} installed at /opt/ros/${ROS_DISTRO}"
    else
        log_error "✗ ROS 2 ${ROS_DISTRO} installation not found"
        errors=$((errors + 1))
    fi

    log_info "============================================================"

    if [[ $errors -eq 0 ]]; then
        log_info "Installation verification PASSED"
        return 0
    else
        log_error "Installation verification FAILED with $errors error(s)"
        return 1
    fi
}

show_success_message() {
    echo ""
    log_info "============================================================"
    log_info "Installation Complete!"
    log_info "============================================================"
    echo ""
    log_info "ROS 2 ${ROS_DISTRO} has been installed successfully."
    echo ""
    log_info "To start using ROS 2, source the setup script:"
    log_info "  source /opt/ros/${ROS_DISTRO}/setup.bash"
    echo ""
    log_info "You may want to add this to your ~/.bashrc:"
    log_info "  echo \"source /opt/ros/${ROS_DISTRO}/setup.bash\" >> ~/.bashrc"
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    # Parse arguments
    parse_args "$@"

    # Pre-flight checks
    check_sudo

    # Detect environment
    detect_os
    detect_architecture
    setup_os_variables

    # Check existing installations
    check_ros_installation

    # Check if everything is already installed
    check_if_complete

    # Confirm installation
    confirm_installation

    # Perform OS-specific installation
    if [[ "$OS_ID" == "ubuntu" ]]; then
        if ! install_ubuntu_ros; then
            handle_installation_error "ROS 2 on Ubuntu"
            exit 1
        fi
    elif [[ "$OS_ID" == "openeuler" ]]; then
        if ! install_openeuler_ros; then
            handle_installation_error "ROS 2 on openEuler"
            exit 1
        fi
    fi

    # Verify installation
    if ! verify_installation; then
        log_error "Installation verification failed"
        exit 1
    fi

    # Show success message
    show_success_message
}

# Run if executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

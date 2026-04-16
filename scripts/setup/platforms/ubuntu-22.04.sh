#!/bin/bash

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
    :
}

platform_install_colcon() {
    run_sudo apt-get install -y python3-colcon-common-extensions
}

platform_install_python_bootstrap() {
    run_sudo apt-get update -qq
    run_sudo apt-get install -y python3-venv python3-pip -qq
}

platform_install_rosdeps() {
    log_info "Updating apt package lists..."
    run_sudo apt-get update -qq

    log_info "Updating rosdepc database..."
    if ! rosdepc update --rosdistro=humble; then
        log_error "rosdepc update failed. This is usually due to network issues."
        log_error "Please check your network connection and re-run ./scripts/setup.sh"
        exit 1
    fi

    log_info "Installing ROS dependencies via apt..."
    if ! rosdepc install \
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
}

platform_verify_ros_python_bridge() {
    local ros_setup
    ros_setup="$(platform_ros_setup_path)"
    [[ -z "${ros_setup}" || ! -f "${ros_setup}" ]] && return 1
    (source "${ros_setup}" && python3 -c "import rclpy; print('ROS 2 Humble connection successful')") 2>/dev/null
}

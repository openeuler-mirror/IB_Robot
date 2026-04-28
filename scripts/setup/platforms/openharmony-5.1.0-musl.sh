#!/bin/bash

platform_ros_setup_path() {
    if [[ -n "${ROS_HUMBLE_SETUP_PATH:-}" ]]; then
        echo "${ROS_HUMBLE_SETUP_PATH}"
    elif [[ -f /data/ros2ohos.env ]]; then
        echo "/data/ros2ohos.env"
    else
        echo ""
    fi
}

# Default lerobot patch-series profile selection for this platform.
# Consumed by detect.sh::resolve_lerobot_profiles when neither
# IBR_LEROBOT_PROFILES_CLI nor IBR_LEROBOT_PROFILES is set.
# 0001-0003 are excluded by python_max on Python 3.12 board runtimes,
# while OpenHarmony-specific runtime patches opt into the openharmony profile.
platform_lerobot_profiles() {
    echo "core,openharmony"
}

platform_handle_missing_ros() {
    log_error "ROS 2 Humble setup script is not configured for OpenHarmony."
    log_error "If you are on the board, source /data/ros2ohos.env in the current shell before running setup."
    log_error "Otherwise set ROS_HUMBLE_SETUP_PATH to an OpenHarmony ROS environment that exposes Python 3.12."
    log_error "For host-side x86_64_virt emulation, use scripts/openharmony/ and docs/openharmony_qemu.md instead."
    exit 1
}

platform_prepare_host() {
    log_warn "OpenHarmony musl platform detected."
    log_warn "This setup path validates the board runtime only; it does not bootstrap a local build workspace."
    log_warn "Local colcon/venv/rosdepc setup is skipped because OpenHarmony artifacts must be cross-built on the host."
    log_warn "Use scripts/openharmony/build_ibrobot_oh_custom.sh on the host for deployable builds."
    log_warn "If you want OpenHarmony support through QEMU on the Ubuntu host, see docs/openharmony_qemu.md."
}

platform_supports_local_workspace_build() {
    return 1
}

platform_install_colcon() {
    log_warn "Skipping colcon installation on OpenHarmony musl."
    return 0
}

platform_install_python_bootstrap() {
    log_warn "Skipping Python venv bootstrap on OpenHarmony musl."
    return 0
}

platform_install_rosdeps() {
    log_warn "Skipping automated rosdepc installation on OpenHarmony musl."
    log_warn "Board runtime uses the prebuilt ROS environment instead of local rosdep resolution."
    log_warn "Host-side QEMU helpers are available under scripts/openharmony/."
}

platform_verify_ros_python_bridge() {
    local ros_setup
    ros_setup="$(platform_ros_setup_path)"
    [[ -z "${ros_setup}" || ! -f "${ros_setup}" ]] && return 1
    (source "${ros_setup}" && python3 -c "import rclpy; print('ROS 2 Humble connection successful')") 2>/dev/null
}

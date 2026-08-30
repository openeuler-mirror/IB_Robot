#!/bin/bash

# Default lerobot patch-series profile selection for this platform.
# Consumed by detect.sh::resolve_lerobot_profiles when neither
# IBR_LEROBOT_PROFILES_CLI nor IBR_LEROBOT_PROFILES is set.
platform_lerobot_profiles() {
    echo "core,ros,hardware,dev,training,distillation"
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

platform_install_benchmark_system_deps() {
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" != true ]]; then
        return 0
    fi

    log_info "Installing benchmark headless rendering dependency..."
    run_sudo apt-get install -y --no-install-recommends libosmesa6-dev ffmpeg -qq
}

platform_pre_install_rosdeps() {
    log_info "Updating apt package lists..."
    run_sudo apt-get update -qq
}

platform_post_install_rosdeps() {
    # Explicitly install tracing tools whose rosdep keys exist in base.yaml but
    # may be skipped when the rosdep database update fails (network issues).
    # ROS packages (ros2trace, tracetools-analysis) are resolved by rosdep
    # from package.xml <exec_depend> entries and do not need explicit install.
    log_info "Installing remaining tracing tools without rosdep rules..."
    run_sudo apt-get install -y --no-install-recommends lttng-tools python3-lttngust babeltrace2 -qq

    if ! groups | grep -q '\btracing\b'; then
        run_sudo usermod -aG tracing "$(whoami)" 2>/dev/null || true
        log_warn "Added $(whoami) to the 'tracing' group; re-login may be required before using LTTng."
    fi
}

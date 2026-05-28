#!/bin/bash

# Default lerobot patch-series profile selection for this platform.
# Consumed by detect.sh::resolve_lerobot_profiles when neither
# IBR_LEROBOT_PROFILES_CLI nor IBR_LEROBOT_PROFILES is set.
platform_lerobot_profiles() {
    echo "core,ros,hardware,openeuler"
}

platform_prepare_host() {
    log_warn "openEuler detected. Setting ROS_OS_OVERRIDE=rhel:8 for rosdep compatibility."
    export ROS_OS_OVERRIDE=rhel:8

    ensure_openeuler_volatile_dirs
    ensure_openeuler_builtin_repos
    ensure_openeuler_ca_certificates
    ensure_openeuler_extras_repo
    ensure_openeuler_gpg_key

    log_info "Installing openEuler host packages required by the workspace..."
    run_sudo dnf install -y --nogpgcheck \
        gcc-c++ \
        vim-enhanced \
        ffmpeg-devel \
        libvpx \
        libvpx-devel \
        nlohmann-json-devel
}

ensure_openeuler_ca_certificates() {
    log_info "Installing openEuler CA certificates..."
    run_sudo dnf install -y --nogpgcheck ca-certificates
    if command -v update-ca-trust >/dev/null 2>&1; then
        log_info "Refreshing system CA trust store..."
        run_sudo update-ca-trust extract
    fi
}

ensure_openeuler_volatile_dirs() {
    # openEuler Embedded uses /var/tmp -> volatile/tmp; some validation
    # rootfs images miss the volatile target, which breaks rpm scriptlets.
    run_sudo mkdir -p /var/volatile/tmp /var/volatile/log
    run_sudo chmod 1777 /var/volatile/tmp
}

platform_install_colcon() {
    if command -v pip3 &> /dev/null; then
        pip3 install colcon-common-extensions --quiet
    else
        log_error "pip3 not found, cannot install colcon."
        exit 1
    fi
}

openeuler_builtin_repos_configured() {
    dnf repolist --enabled | awk '
        $1 == "everything" { everything = 1 }
        $1 == "update" { update = 1 }
        $1 == "EPOL" { epol = 1 }
        END { exit (everything && update && epol) ? 0 : 1 }
    '
}

ensure_openeuler_builtin_repos() {
    if openeuler_builtin_repos_configured; then
        log_info "Built-in openEuler 24.03 repos already configured."
        return 0
    fi

    log_error "Required built-in openEuler repos are missing (expected: everything, update, EPOL)."
    log_error "Please restore /etc/yum.repos.d/openEuler.repo before running setup.sh."
    exit 1
}

ensure_openeuler_extras_repo() {
    local extras_repo_url
    extras_repo_url="https://repo.oepkgs.net/openeuler/rpm/openEuler-24.03-LTS/extras/$(uname -m)"

    if dnf repolist --enabled | awk '$1 == "extras" { found = 1 } END { exit found ? 0 : 1 }'; then
        log_info "openEuler extras repo already configured."
        return 0
    fi

    log_info "Adding openEuler extras repo required for python3-lttngust..."
    run_sudo dnf config-manager --add-repo "${extras_repo_url}"
}

ensure_openeuler_gpg_key() {
    if rpm -qi gpg-pubkey'*' 2>/dev/null | grep -q 'openEuler'; then
        log_info "openEuler RPM GPG key already imported."
        return 0
    fi

    local arch tmp key_url
    arch=$(uname -m)
    tmp=$(mktemp)
    for key_url in \
        "https://mirrors.tuna.tsinghua.edu.cn/openeuler/openEuler-24.03-LTS/OS/${arch}/RPM-GPG-KEY-openEuler" \
        "https://repo.openeuler.org/openEuler-24.03-LTS/OS/${arch}/RPM-GPG-KEY-openEuler"
    do
        if curl -fsSL "${key_url}" -o "${tmp}"; then
            log_info "Importing openEuler RPM GPG key from ${key_url}..."
            run_sudo rpm --import "${tmp}"
            rm -f "${tmp}"
            return 0
        fi
    done

    rm -f "${tmp}"
    log_error "Failed to download the openEuler RPM GPG key from all known mirrors."
    exit 1
}

platform_install_python_bootstrap() {
    run_sudo dnf install -y --nogpgcheck python3-virtualenv python3-pip python3-devel -q
}

platform_pre_install_rosdeps() {
    log_info "Updating dnf package repositories..."
}

platform_get_extra_skip_keys() {
    echo "lttng-tools nlohmann-json-dev python3-opencv python3-aiortc gz_ros2_control ros_gz_sim ros_gz_bridge mujoco_ros2_control python3-scipy robot_localization"
}

platform_post_install_rosdeps() {
    # rosdep resolves python3-scipy to python%{python3_pkgversion}-scipy (RHEL
    # convention) via ROS_OS_OVERRIDE=rhel:8, but openEuler dnf cannot match
    # that macro name.  Install with the native package name instead.
    log_info "Installing python3-scipy (rosdep uses RHEL macro naming on openEuler)..."
    run_sudo dnf install -y --nogpgcheck python3-scipy

    log_info "Installing graphviz packages for ros2 control topology visualization..."
    run_sudo dnf install -y --nogpgcheck graphviz graphviz-devel -q

    # On openEuler, tracing packages are not reliably provisioned by rosdep in
    # the container/chroot validation environment. Keep explicit dnf fallback
    # installs here so setup can converge even when rosdep fails the batch.
    log_info "Installing remaining tracing tools without rosdep rules..."
    run_sudo dnf install -y --nogpgcheck \
        ros-humble-ros2trace \
        ros-humble-tracetools-analysis \
        babeltrace \
        python3-babeltrace \
        python3-lttngust
}

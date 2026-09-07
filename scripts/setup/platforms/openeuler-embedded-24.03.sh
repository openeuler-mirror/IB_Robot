#!/bin/bash

# Default lerobot patch-series profile selection for this platform.
# Consumed by detect.sh::resolve_lerobot_profiles when neither
# IBR_LEROBOT_PROFILES_CLI nor IBR_LEROBOT_PROFILES is set.
platform_lerobot_profiles() {
    echo "core,ros,hardware,openeuler"
}

# Source shared openEuler ROS repo management (SSOT for openEulerROS.repo).
# Also sourced by scripts/install_ros.sh, so repo content lives in one place.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/setup/openeuler_ros_repo.sh"

platform_prepare_host() {
    log_warn "openEuler detected. Setting ROS_OS_OVERRIDE=rhel:8 for rosdep compatibility."
    export ROS_OS_OVERRIDE=rhel:8

    ensure_openeuler_volatile_dirs
    ensure_openeuler_builtin_repos
    ensure_openeuler_ros_repo
    ensure_openeuler_ca_certificates
    ensure_openeuler_extras_repo
    warn_openeuler_duplicate_ros_repos
    ensure_openeuler_gpg_key

    log_info "Installing openEuler host packages required by the workspace..."
    run_sudo dnf install -y --nogpgcheck \
        gcc-c++ \
        vim-enhanced \
        ffmpeg-devel \
        libvpx \
        libvpx-devel \
        nlohmann-json-devel \
        yaml-cpp \
        yaml-cpp-devel

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
        local pip_args=(--quiet)
        if [[ -n "${SETUP_PIP_INDEX_URL:-}" ]]; then
            pip_args+=(--index-url "${SETUP_PIP_INDEX_URL}" --trusted-host "${SETUP_PIP_TRUSTED_HOST}")
        fi
        pip3 install colcon-common-extensions "${pip_args[@]}"
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
    local extras_repo_file extras_repo_url
    extras_repo_url="https://repo.oepkgs.net/openeuler/rpm/openEuler-24.03-LTS/extras/$(uname -m)"
    extras_repo_file=$(openeuler_extras_repo_file "${extras_repo_url}")

    if [[ -n "${extras_repo_file}" ]]; then
        log_info "openEuler extras repo already configured."
    else
        log_info "Adding openEuler extras repo required for python3-lttngust..."
        run_sudo dnf config-manager --add-repo "${extras_repo_url}"
        extras_repo_file=$(openeuler_extras_repo_file "${extras_repo_url}")
    fi

    if [[ -z "${extras_repo_file}" ]]; then
        log_error "Failed to locate the generated openEuler extras repo file."
        exit 1
    fi

    ensure_openeuler_extras_repo_excludes "${extras_repo_file}"
}

openeuler_extras_repo_file() {
    local extras_repo_url="$1"

    grep -rlF "baseurl=${extras_repo_url}" /etc/yum.repos.d 2>/dev/null | sed -n '1p' || true
}

ensure_openeuler_extras_repo_excludes() {
    local repo_file="$1" exclude_line="" missing_excludes=""

    grep -Eq '^exclude=([[:space:]]|[^[:space:]]+[[:space:]])*yaml-cpp\*([[:space:]]|$)' "${repo_file}" \
        || missing_excludes="yaml-cpp*"
    if ! grep -Eq '^exclude=([[:space:]]|[^[:space:]]+[[:space:]])*qhull\*([[:space:]]|$)' "${repo_file}"; then
        missing_excludes="${missing_excludes:+${missing_excludes} }qhull*"
    fi
    if [[ -z "${missing_excludes}" ]]; then
        return 0
    fi

    log_info "Excluding yaml-cpp and qhull from openEuler extras repo to preserve the ROS 2 ABI packages."
    if grep -q '^exclude=' "${repo_file}"; then
        exclude_line="$(grep -m1 '^exclude=' "${repo_file}")"
        run_sudo sed -i "0,/^exclude=/ s|^exclude=.*|${exclude_line} ${missing_excludes}|" "${repo_file}"
    else
        printf '\nexclude=yaml-cpp* qhull*\n' | run_sudo tee -a "${repo_file}" >/dev/null
    fi
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

    log_info "Disabling per-repo GPG checks for rosdep-managed dnf transactions..."
    if [[ -f /etc/dnf/dnf.conf ]]; then
        run_sudo sed -i 's/^\s*gpgcheck\s*=\s*1/gpgcheck=0/' /etc/dnf/dnf.conf
        if ! grep -q '^\s*gpgcheck\s*=\s*0' /etc/dnf/dnf.conf; then
            run_sudo sed -i '/^\[main\]/a gpgcheck=0' /etc/dnf/dnf.conf
        fi
    fi

    # Refresh metadata so that newly-restored ROS repos are visible to
    # rosdep before it resolves ros-humble-* keys into dnf transactions.
    log_info "Refreshing dnf metadata (ensures ROS repos are visible to rosdep)..."
    run_sudo dnf clean all || log_warn "dnf clean all failed, continuing..."
    if ! run_sudo dnf makecache; then
        log_warn "dnf makecache encountered errors — some repos may be unreachable."
        log_warn "Continuing; rosdep will report unresolved keys if a repo is truly broken."
    fi
}

platform_get_extra_skip_keys() {
    # Source-model CUDA extensions stay optional. The manifest-driven Ascend OM
    # perception/manipulation paths only require the core NumPy/OpenCV runtime.
    echo "lttng-tools nlohmann-json-dev python3-opencv python3-aiortc gz_ros2_control ros_gz_sim ros_gz_bridge mujoco_ros2_control mujoco_ros2_control_msgs python3-scipy robot_localization sam2 groundingdino grounding-dino graspgen spconv-cu120 torch-scatter pointnet2_ops"
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

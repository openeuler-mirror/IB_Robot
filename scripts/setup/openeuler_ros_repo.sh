#!/bin/bash
# openeuler_ros_repo.sh - Shared openEuler ROS Humble repo management
#
# Single source of truth for /etc/yum.repos.d/openEulerROS.repo.
# Sourced by both:
#   - scripts/install_ros.sh::install_openeuler_ros()
#   - scripts/setup/platforms/openeuler-embedded-24.03.sh::platform_prepare_host()
#
# This avoids maintaining two copies of the repo file content.  When a
# URL or section changes, it only needs to be updated here.

OPENEULER_ROS_REPO_FILE="/etc/yum.repos.d/openEulerROS.repo"

# Emit the authoritative repo file content to stdout.
# The heredoc delimiter is single-quoted so $basearch stays literal —
# dnf expands it at resolve time.
_openeuler_ros_repo_content() {
    cat << 'REPOEOF'
[openEuler-Embedded-ROS2-humble]
name=openEuler-Embedded-ROS2-humble
baseurl=https://eulermaker.openeuler.openatom.cn/api/ems1/repositories/openEuler-Embedded-ROS2-humble-2609-release-1/openEuler%3A24.03-LTS/$basearch/
enabled=1
gpgcheck=0
priority=1
REPOEOF
}

# Verify the openEuler ROS Humble repo file; create or update it to match
# the authoritative content if missing, stale, or incomplete.
#
# Uses full-content comparison (cmp) rather than just checking section
# headers, so that stale baseurl values (e.g. changed release/TEST
# identifiers) are detected and refreshed.
ensure_openeuler_ros_repo() {
    local tmp rc=0
    tmp=$(mktemp)
    _openeuler_ros_repo_content > "${tmp}"

    if [[ -f "${OPENEULER_ROS_REPO_FILE}" ]] && cmp -s "${tmp}" "${OPENEULER_ROS_REPO_FILE}"; then
        log_info "openEuler ROS Humble repo already configured and up to date."
        rm -f "${tmp}"
        return 0
    fi

    if [[ -f "${OPENEULER_ROS_REPO_FILE}" ]]; then
        log_warn "openEuler ROS Humble repo file is stale or incomplete, updating..."
    else
        log_warn "openEuler ROS Humble repo file is missing, creating..."
    fi

    # Capture exit code explicitly so that temp cleanup always runs,
    # even when the caller has errexit (set -e) enabled.
    run_sudo install -m 0644 "${tmp}" "${OPENEULER_ROS_REPO_FILE}" || rc=$?
    rm -f "${tmp}"

    if [[ ${rc} -ne 0 ]]; then
        log_error "Failed to write ${OPENEULER_ROS_REPO_FILE}."
        return 1
    fi

    log_info "openEuler ROS Humble repo file configured successfully."
}

# Detect and warn about duplicate repo IDs across /etc/yum.repos.d/*.repo.
# dnf emits "Repository X is listed more than once in the configuration"
# when the same section header appears in multiple files.
warn_openeuler_duplicate_ros_repos() {
    local repos_dir="/etc/yum.repos.d"
    [[ -d "${repos_dir}" ]] || return 0

    local dupes
    dupes=$(grep -rh '^\[' "${repos_dir}"/*.repo 2>/dev/null \
        | sed 's/^\[//; s/\].*//' \
        | sort \
        | uniq -d || true)
    [[ -n "${dupes}" ]] || return 0

    log_warn "Duplicate dnf repo IDs detected across ${repos_dir}/*.repo:"
    local dupe
    while IFS= read -r dupe; do
        local hit_files
        hit_files=$(grep -rl "^\[${dupe}\]" "${repos_dir}"/*.repo 2>/dev/null | tr '\n' ' ' || true)
        log_warn "  [${dupe}] found in: ${hit_files}"
    done <<< "${dupes}"
    log_warn "Remove stale duplicate entries to silence 'listed more than once' warnings."
}

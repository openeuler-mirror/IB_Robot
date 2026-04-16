#!/bin/bash

lerobot_patch_dir() {
    echo "${WORKSPACE}/third_party/patches/lerobot/v0.5.1"
}

lerobot_patch_base_commit() {
    local manifest
    manifest="$(lerobot_patch_dir)/manifest.yaml"
    grep -E '^  commit:' "${manifest}" | head -n1 | awk '{print $2}'
}

ensure_lerobot_patch_stack_applied() {
    local submodule_dir="${WORKSPACE}/libs/lerobot"
    local patch_dir
    local series_file
    local base_commit
    local branch_name="ibrobot/lerobot-v0.5.1-patched"

    [[ ! -d "${submodule_dir}" ]] && return 0
    [[ ! -d "${submodule_dir}/.git" && ! -f "${submodule_dir}/.git" ]] && return 0

    patch_dir="$(lerobot_patch_dir)"
    series_file="${patch_dir}/series.txt"
    base_commit="$(lerobot_patch_base_commit)"

    if [[ ! -f "${series_file}" || -z "${base_commit}" ]]; then
        log_warn "LeRobot patch stack metadata is missing. Skipping automatic patch application."
        return 0
    fi

    if git -C "${submodule_dir}" show-ref --verify --quiet "refs/heads/${branch_name}"; then
        if [[ "$(git -C "${submodule_dir}" branch --show-current)" != "${branch_name}" ]]; then
            log_info "Switching libs/lerobot to existing patched branch ${branch_name}..."
            git -C "${submodule_dir}" checkout "${branch_name}" >/dev/null
        fi
        return 0
    fi

    if [[ "$(git -C "${submodule_dir}" rev-parse HEAD)" != "${base_commit}" ]]; then
        log_warn "libs/lerobot is not at the expected upstream base commit ${base_commit}; skipping automatic patch application."
        return 0
    fi

    if ! git -C "${submodule_dir}" diff --quiet || ! git -C "${submodule_dir}" diff --cached --quiet; then
        log_error "libs/lerobot has local changes; refusing to apply the IB_Robot patch stack automatically."
        exit 1
    fi

    log_info "Applying IB_Robot lerobot patch stack on top of upstream v0.5.1..."
    git -C "${submodule_dir}" checkout -b "${branch_name}" >/dev/null

    while IFS= read -r patch_file; do
        [[ -z "${patch_file}" ]] && continue
        git -C "${submodule_dir}" am "${patch_dir}/${patch_file}" >/dev/null
    done < "${series_file}"

    log_done "LeRobot patch stack applied"
}

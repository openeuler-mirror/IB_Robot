#!/bin/bash
# submodules.sh - Submodule initialization and update routines

submodule_is_initialized() {
    local path="$1"
    [[ -e "${path}/.git" ]] || return 1
    git -C "${path}" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

submodule_update_block_reason() {
    local path="$1"
    
    if ! submodule_is_initialized "${path}"; then
        echo "not initialized"
        return 0
    fi

    if [[ -n "$(git -C "${path}" status --porcelain)" ]]; then
        echo "has uncommitted changes"
        return 0
    fi

    local head_commit
    head_commit="$(git -C "${path}" rev-parse HEAD 2>/dev/null || echo "")"
    local expected_commit
    expected_commit="$(git ls-tree HEAD "${path}" | awk '{print $3}')"

    if [[ "${head_commit}" != "${expected_commit}" ]]; then
        echo "currently checked out to ${head_commit:0:7}, expected ${expected_commit:0:7}"
        return 0
    fi

    return 1
}

update_submodules() {
    if [[ "${SKIP_SUBMODULES}" == true ]]; then
        log_info "Skipping submodule initialization/update (--skip-submodules)."
        log_skipped "Git submodules"
        return 0
    fi

    log_info "Synchronizing submodule URLs..."
    git submodule sync --recursive >/dev/null

    local submodules=()
    local sm_path
    while IFS= read -r sm_path; do
        [[ -n "${sm_path}" ]] && submodules+=("${sm_path}")
    done <<< "$(git config --file .gitmodules --get-regexp path | awk '{ print $2 }')"
    
    local to_update=()
    local skipped=()

    for sm in "${submodules[@]}"; do
        local reason
        if reason="$(submodule_update_block_reason "${sm}")"; then
            if [[ "${reason}" == "not initialized" ]]; then
                to_update+=("${sm}")
            else
                skipped+=("${sm} (${reason})")
            fi
        else
            to_update+=("${sm}")
        fi
    done

    if [[ ${#skipped[@]} -gt 0 ]]; then
        log_warn "The following submodules will NOT be updated automatically:"
        for sm in "${skipped[@]}"; do
            echo "  - ${sm}"
        done
        echo ""
        if ! ask_yn "Continue anyway?" "y"; then
            log_error "Submodule update aborted by user."
            exit 1
        fi
    fi

    if [[ ${#to_update[@]} -gt 0 ]]; then
        log_info "Updating submodules..."
        for sm in "${to_update[@]}"; do
            log_info "  -> ${sm}"
            run_cmd git submodule update --init --recursive "${sm}"
        done
        log_done "Submodules updated"
    else
        log_info "All submodules are up to date."
        log_done "Submodules checked"
    fi
}

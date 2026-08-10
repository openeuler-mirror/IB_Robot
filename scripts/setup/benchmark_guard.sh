#!/bin/bash
# benchmark_guard.sh — Setup-owned guards for benchmark dependency installation.
#
# Sourced by python_venv.sh before editable install of libs/libero.
# Can be sourced independently for behavioral testing.

# verify_libero_gitlink: verify libs/libero submodule gitlink matches HEAD
# and working tree is clean before any editable install.
#
# Returns:
#   0 = passed (gitlink matches HEAD, working tree clean)
#   1 = failed (mismatch, dirty, or git command failure)
verify_libero_gitlink() {
    local workspace="${1:-${WORKSPACE:-$(pwd)}}"
    local libero_dir="${workspace}/libs/libero"

    if [[ ! -d "${libero_dir}" ]]; then
        echo "ERROR: libs/libero is not initialized" >&2
        return 1
    fi

    # Read the gitlink from the parent repo.
    # `|| true` suppresses non-zero exit under set -e/pipefail; the
    # subsequent length check handles the failure case.
    local parent_gitlink=""
    parent_gitlink="$(git -C "${workspace}" ls-tree HEAD libs/libero 2>/dev/null | awk '{print $3}')" || true
    if [[ -z "${parent_gitlink}" || ${#parent_gitlink} -ne 40 ]]; then
        echo "ERROR: Failed to read libs/libero gitlink from parent repo" >&2
        echo "Output: '${parent_gitlink}'" >&2
        return 1
    fi

    # Read the actual HEAD of the submodule.
    local submodule_head=""
    submodule_head="$(git -C "${libero_dir}" rev-parse HEAD 2>/dev/null)" || true
    if [[ -z "${submodule_head}" || ${#submodule_head} -ne 40 ]]; then
        echo "ERROR: Failed to read libs/libero submodule HEAD" >&2
        echo "Output: '${submodule_head}'" >&2
        return 1
    fi

    if [[ "${parent_gitlink}" != "${submodule_head}" ]]; then
        echo "ERROR: libs/libero submodule HEAD mismatch" >&2
        echo "  Parent gitlink: ${parent_gitlink}" >&2
        echo "  Submodule HEAD: ${submodule_head}" >&2
        return 1
    fi

    # Check working tree is clean.
    local submodule_dirty=""
    submodule_dirty="$(git -C "${libero_dir}" status --porcelain 2>/dev/null)" || true
    if [[ -n "${submodule_dirty}" ]]; then
        echo "ERROR: libs/libero has uncommitted changes" >&2
        echo "${submodule_dirty}" >&2
        return 1
    fi

    echo "OK: gitlink=${parent_gitlink:0:12} HEAD matches, working tree clean"
    return 0
}

#!/usr/bin/env bash
# Regression harness for scripts/setup/lerobot_filter_series.py.
#
# Mocks the host facts via environment variables and asserts the filtered
# series output matches canonical fixtures. The harness purposefully does NOT
# invoke `git am`; it exercises the filter logic in isolation so it can run
# inside pre-commit and lightweight CI containers without a lerobot checkout.
#
# Usage:
#   scripts/setup/tests/test_lerobot_filter.sh
#
# Exit status:
#   0  — all fixtures pass.
#   1  — at least one fixture failed; details are printed to stderr.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
FILTER="${REPO_ROOT}/scripts/setup/lerobot_filter_series.py"
RESOLVER="${REPO_ROOT}/scripts/setup/lerobot_resolve_active.py"
INDEX="${REPO_ROOT}/third_party/patches/lerobot/INDEX.yaml"
MANIFEST="${REPO_ROOT}/third_party/patches/lerobot/v0.5.1/manifest.yaml"
DEFAULT_SERIES="${REPO_ROOT}/third_party/patches/lerobot/v0.5.1/series.txt"
EXPECTED_BASE_COMMIT="1396b9fab7aecddd10006c33c47a487ffdcb54b4"

PASS=0
FAIL=0

PYTHON_BIN="${PYTHON_BIN:-python3}"

# run_fixture <name> <expected-series-csv> -- env=val [env=val...]
# Captures stdout from the filter and compares against the expected CSV.
run_fixture() {
    local name="$1"
    local expected="$2"
    shift 2

    local env_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        env_args+=("$1")
        shift
    done
    [[ "${1:-}" == "--" ]] && shift

    local series_file="${1:-${DEFAULT_SERIES}}"

    local actual
    actual="$(env -i PATH="${PATH}" HOME="${HOME}" \
        "${env_args[@]}" \
        "${PYTHON_BIN}" "${FILTER}" \
            --manifest "${MANIFEST}" \
            --series "${series_file}" 2>/dev/null \
        | paste -sd, -)"

    if [[ "${actual}" == "${expected}" ]]; then
        printf '  PASS  %s\n' "${name}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected: %s\n' "${expected}" >&2
        printf '    actual:   %s\n' "${actual}" >&2
        FAIL=$((FAIL + 1))
    fi
}

# run_negative <name> <expected-exit> -- env=val [env=val...] -- args...
# Asserts the filter exits with the expected non-zero status.
run_negative() {
    local name="$1"
    local expected_exit="$2"
    shift 2

    local env_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        env_args+=("$1")
        shift
    done
    [[ "${1:-}" == "--" ]] && shift

    local extra_args=("$@")

    local actual_exit=0
    env -i PATH="${PATH}" HOME="${HOME}" \
        "${env_args[@]}" \
        "${PYTHON_BIN}" "${FILTER}" "${extra_args[@]}" >/dev/null 2>&1 \
        || actual_exit=$?

    if [[ "${actual_exit}" -eq "${expected_exit}" ]]; then
        printf '  PASS  %s (exit %d)\n' "${name}" "${actual_exit}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected exit: %d\n' "${expected_exit}" >&2
        printf '    actual exit:   %d\n' "${actual_exit}" >&2
        FAIL=$((FAIL + 1))
    fi
}

echo "== lerobot_filter_series regression harness =="

# Ubuntu 22.04 / Python 3.10 — must keep the legacy 5-patch series intact so
# PR #96's baseline output does not regress.
run_fixture "ubuntu-22.04 / py3.10 / desktop profile" \
    "0001-python-compat-syntax-and-metadata.patch,0002-python-compat-min-version-3.10.patch,0003-python-compat-typing-unpack.patch,0005-compat-add-npu-device-detection.patch,0006-compat-add-ascend-om-config-fields.patch" \
    IBR_HOST_PYTHON_VERSION=3.10 \
    IBR_LEROBOT_PROFILES=core,ros,hardware,dev \
    --

# openEuler Embedded 24.03 / Python 3.11 — must drop 0001-0003 (down-grade
# patches) but keep the Ascend-side compat patches.
run_fixture "openeuler-embedded-24.03 / py3.11" \
    "0005-compat-add-npu-device-detection.patch,0006-compat-add-ascend-om-config-fields.patch" \
    IBR_HOST_PYTHON_VERSION=3.11 \
    IBR_LEROBOT_PROFILES=core,ros,hardware,openeuler \
    --

# OpenHarmony 5.1.0 / Python 3.12 — must drop both the down-grade and
# the Ascend compat patches (no profile overlap).
run_fixture "openharmony-5.1.0 / py3.12" \
    "" \
    IBR_HOST_PYTHON_VERSION=3.12 \
    IBR_LEROBOT_PROFILES=core,openharmony \
    --

# Force-Ascend bring-up scenario where the operator wants the master-parity
# Ascend patches as well. We point the
# filter at the master-parity series file because the default series only
# contains 0001-0006.
MASTER_PARITY_SERIES="${REPO_ROOT}/third_party/patches/lerobot/v0.5.1/series.master-parity-candidates.txt"
if [[ -f "${MASTER_PARITY_SERIES}" ]]; then
    run_fixture "ascend-forced / py3.10 / master-parity series" \
        "0007-ascend-om-act-runtime.patch,0008-ascend-3403-actwrapper.patch,0009-weighted-training.patch,0010-knowledge-distillation.patch,0011-mt-act-model.patch,0012-attention-visualization-tools.patch" \
        IBR_HOST_PYTHON_VERSION=3.10 \
        IBR_LEROBOT_PROFILES=core,ascend,master-parity-candidates,om,training,distillation,models,mt-act,visualization,tooling \
        -- \
        "${MASTER_PARITY_SERIES}"
else
    printf '  SKIP  ascend-forced fixture (master-parity series file missing)\n'
fi

# Negative: malformed manifest must exit non-zero with parser error.
TMP_BAD_MANIFEST="$(mktemp)"
trap 'rm -rf "${TMP_BAD_MANIFEST}" "${TMP_INDEX_DIR:-/nonexistent}"' EXIT
printf 'patches:\n  - file: 0001\n      bad-indent: true\n' > "${TMP_BAD_MANIFEST}"
run_negative "malformed manifest exits 1" 1 \
    IBR_HOST_PYTHON_VERSION=3.10 \
    IBR_LEROBOT_PROFILES=core \
    -- \
    --manifest "${TMP_BAD_MANIFEST}" --series "${DEFAULT_SERIES}"

# ---------------------------------------------------------------------------
# Tag binding fixtures (--lerobot-head-commit + INDEX.yaml resolution)
# ---------------------------------------------------------------------------

# HEAD matching commit_range.min must succeed and produce the same default
# 5-patch series as the baseline ubuntu fixture above.
run_fixture "tag-binding / head_commit==range.min keeps default series" \
    "0001-python-compat-syntax-and-metadata.patch,0002-python-compat-min-version-3.10.patch,0003-python-compat-typing-unpack.patch,0005-compat-add-npu-device-detection.patch,0006-compat-add-ascend-om-config-fields.patch" \
    IBR_HOST_PYTHON_VERSION=3.10 \
    IBR_LEROBOT_PROFILES=core,ros,hardware,dev \
    -- \
    "${DEFAULT_SERIES}" \
    --lerobot-head-commit "${EXPECTED_BASE_COMMIT}"

# HEAD that falls outside commit_range must abort with exit 1 even when
# every per-patch predicate would otherwise match.
run_negative "tag-binding / head_commit out of range fails closed" 1 \
    IBR_HOST_PYTHON_VERSION=3.10 \
    IBR_LEROBOT_PROFILES=core,ros,hardware,dev \
    -- \
    --manifest "${MANIFEST}" \
    --series "${DEFAULT_SERIES}" \
    --lerobot-head-commit "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"

# ---------------------------------------------------------------------------
# Resolver fixtures (lerobot_resolve_active.py + INDEX.yaml)
# ---------------------------------------------------------------------------

run_resolver_pass() {
    local name="$1"
    local index_path="$2"
    local expected_tag="$3"

    local actual_tag
    if ! actual_tag="$("${PYTHON_BIN}" "${RESOLVER}" --index "${index_path}" 2>/dev/null \
            | grep -E '^LEROBOT_TAG=' | head -n1 | cut -d= -f2)"; then
        printf '  FAIL  %s (resolver exited non-zero)\n' "${name}" >&2
        FAIL=$((FAIL + 1))
        return
    fi
    if [[ "${actual_tag}" == "${expected_tag}" ]]; then
        printf '  PASS  %s (LEROBOT_TAG=%s)\n' "${name}" "${actual_tag}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected LEROBOT_TAG=%s\n' "${expected_tag}" >&2
        printf '    actual   LEROBOT_TAG=%s\n' "${actual_tag}" >&2
        FAIL=$((FAIL + 1))
    fi
}

run_resolver_fail() {
    local name="$1"
    local index_path="$2"

    local actual_exit=0
    "${PYTHON_BIN}" "${RESOLVER}" --index "${index_path}" >/dev/null 2>&1 \
        || actual_exit=$?
    if [[ "${actual_exit}" -ne 0 ]]; then
        printf '  PASS  %s (exit %d)\n' "${name}" "${actual_exit}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected non-zero exit, got 0\n' >&2
        FAIL=$((FAIL + 1))
    fi
}

# Baseline: in-tree INDEX.yaml resolves cleanly to v0.5.1.
run_resolver_pass "resolver / in-tree INDEX picks v0.5.1" \
    "${INDEX}" "v0.5.1"

# Build a synthetic multi-tag INDEX in a temp dir to verify selection
# semantics without touching the in-tree layout.
TMP_INDEX_DIR="$(mktemp -d)"
mkdir -p "${TMP_INDEX_DIR}/v0.5.1" "${TMP_INDEX_DIR}/v0.6.0"
cp "${MANIFEST}" "${TMP_INDEX_DIR}/v0.5.1/manifest.yaml"
cp "${DEFAULT_SERIES}" "${TMP_INDEX_DIR}/v0.5.1/series.txt"
# v0.6.0 fixture: same manifest contents but with lerobot_tag retagged so
# the resolver can validate cross-checks. We rewrite the two relevant
# fields with sed so the rest of the manifest stays in sync.
sed -e 's/^lerobot_tag: v0.5.1/lerobot_tag: v0.6.0/' \
    -e 's/^  min: 1396b9fab7aecddd10006c33c47a487ffdcb54b4/  min: feedfacefeedfacefeedfacefeedfacefeedface/' \
    -e 's/^  max: 1396b9fab7aecddd10006c33c47a487ffdcb54b4/  max: feedfacefeedfacefeedfacefeedfacefeedface/' \
    "${MANIFEST}" > "${TMP_INDEX_DIR}/v0.6.0/manifest.yaml"
cp "${DEFAULT_SERIES}" "${TMP_INDEX_DIR}/v0.6.0/series.txt"

# Multi-tag INDEX with active_tag=v0.6.0 must resolve to v0.6.0 even
# though v0.5.1 also exists in supported_tags.
cat > "${TMP_INDEX_DIR}/INDEX.yaml" <<EOF
schema_version: 1
active_tag: v0.6.0
supported_tags:
  - tag: v0.5.1
    dir: v0.5.1
    upstream_commit: 1396b9fab7aecddd10006c33c47a487ffdcb54b4
    branch_name: ibrobot/lerobot-v0.5.1-patched
  - tag: v0.6.0
    dir: v0.6.0
    upstream_commit: feedfacefeedfacefeedfacefeedfacefeedface
    branch_name: ibrobot/lerobot-v0.6.0-patched
archived_tags: []
EOF
run_resolver_pass "resolver / multi-tag selects active_tag=v0.6.0" \
    "${TMP_INDEX_DIR}/INDEX.yaml" "v0.6.0"

# Archived tag selected as active_tag must be rejected.
cat > "${TMP_INDEX_DIR}/INDEX.yaml" <<EOF
schema_version: 1
active_tag: v0.5.1
supported_tags: []
archived_tags:
  - tag: v0.5.1
    dir: v0.5.1
    upstream_commit: 1396b9fab7aecddd10006c33c47a487ffdcb54b4
    branch_name: ibrobot/lerobot-v0.5.1-patched
EOF
run_resolver_fail "resolver / archived tag rejected" \
    "${TMP_INDEX_DIR}/INDEX.yaml"

# INDEX upstream_commit not matching manifest.lerobot_commit_range.min must fail.
cat > "${TMP_INDEX_DIR}/INDEX.yaml" <<EOF
schema_version: 1
active_tag: v0.5.1
supported_tags:
  - tag: v0.5.1
    dir: v0.5.1
    upstream_commit: 0000000000000000000000000000000000000000
    branch_name: ibrobot/lerobot-v0.5.1-patched
archived_tags: []
EOF
run_resolver_fail "resolver / INDEX vs manifest commit mismatch rejected" \
    "${TMP_INDEX_DIR}/INDEX.yaml"

# ---------------------------------------------------------------------------
# Shell applier fixtures (_lerobot_validate_head_commit)
# ---------------------------------------------------------------------------
#
# The Python filter's --lerobot-head-commit covers the in-process binding,
# but the real applier (scripts/setup/lerobot_patches.sh) calls a shell-level
# helper that the Python filter never reaches when IBR_LEROBOT_FORCE_UNFILTERED=1
# bypasses the predicate. We exercise that helper directly here so a
# regression in the shell binding is caught even when the Python path is
# muted.

LEROBOT_PATCHES_LIB="${REPO_ROOT}/scripts/setup/lerobot_patches.sh"

# Minimal logging stubs — lerobot_patches.sh is normally sourced into
# scripts/setup.sh which defines log_info / log_warn / log_error / log_done.
# In the harness we inline single-line stubs so the helper can be exercised
# in isolation. Single-line form avoids newline/`;` interleaving issues
# when the stubs are interpolated into `bash -c "..."`.
LOG_STUBS='log_info() { printf "[INFO] %s\n" "$*" >&2; } ; log_warn() { printf "[WARN] %s\n" "$*" >&2; } ; log_error() { printf "[ERROR] %s\n" "$*" >&2; } ; log_done() { printf "[DONE] %s\n" "$*" >&2; }'

run_validator() {
    # run_validator <name> <expected-exit> <head-sha>
    local name="$1"
    local expected_exit="$2"
    local head_sha="$3"

    local actual_exit=0
    env -i PATH="${PATH}" HOME="${HOME}" \
        LEROBOT_TAG="v0.5.1" \
        LEROBOT_MANIFEST="${MANIFEST}" \
        LEROBOT_BASE_COMMIT_MIN="${EXPECTED_BASE_COMMIT}" \
        LEROBOT_BASE_COMMIT_MAX="${EXPECTED_BASE_COMMIT}" \
        bash -c "set -euo pipefail; ${LOG_STUBS}; source '${LEROBOT_PATCHES_LIB}'; _lerobot_validate_head_commit '${head_sha}'" \
        >/dev/null 2>&1 || actual_exit=$?

    if [[ "${actual_exit}" -eq "${expected_exit}" ]]; then
        printf '  PASS  %s (exit %d)\n' "${name}" "${actual_exit}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s\n' "${name}" >&2
        printf '    expected exit: %d\n' "${expected_exit}" >&2
        printf '    actual exit:   %d\n' "${actual_exit}" >&2
        FAIL=$((FAIL + 1))
    fi
}

# HEAD equal to commit_range.min must pass.
run_validator "shell-validator / HEAD == range.min passes" 0 "${EXPECTED_BASE_COMMIT}"

# HEAD equal to commit_range.max must pass (covers future widening).
run_validator "shell-validator / HEAD == range.max passes" 0 "${EXPECTED_BASE_COMMIT}"

# HEAD outside commit_range must fail-closed (exit 1) — this is the path
# the reviewer flagged: previously the applier silently warned + return 0
# when patched branch was missing AND HEAD drifted.
run_validator "shell-validator / HEAD out of range fails closed" 1 \
    "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"

# Empty HEAD (rev-parse failed) is intentionally a soft-skip — branch
# existence checks elsewhere gate the rebuild path.
run_validator "shell-validator / empty HEAD soft-skips" 0 ""

# Missing commit_range env vars must fail-closed (catches malformed
# manifest / resolver pairs).
run_validator_missing_range() {
    local name="$1"
    local actual_exit=0
    env -i PATH="${PATH}" HOME="${HOME}" \
        LEROBOT_TAG="v0.5.1" \
        LEROBOT_MANIFEST="${MANIFEST}" \
        bash -c "set -euo pipefail; ${LOG_STUBS}; source '${LEROBOT_PATCHES_LIB}'; _lerobot_validate_head_commit '${EXPECTED_BASE_COMMIT}'" \
        >/dev/null 2>&1 || actual_exit=$?
    if [[ "${actual_exit}" -eq 1 ]]; then
        printf '  PASS  %s (exit 1)\n' "${name}"
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %s (exit %d, expected 1)\n' "${name}" "${actual_exit}" >&2
        FAIL=$((FAIL + 1))
    fi
}
run_validator_missing_range "shell-validator / missing commit_range env fails closed"

echo
echo "== summary: ${PASS} passed, ${FAIL} failed =="
[[ ${FAIL} -eq 0 ]]

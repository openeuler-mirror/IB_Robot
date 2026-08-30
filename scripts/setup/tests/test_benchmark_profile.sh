#!/usr/bin/env bash
# Lightweight regression tests for the --with-benchmark installation boundary.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROFILE="${REPO_ROOT}/scripts/setup/benchmark_profile.sh"
PYTHON_VENV="${REPO_ROOT}/scripts/setup/python_venv.sh"
ROSDEP="${REPO_ROOT}/scripts/setup/rosdep.sh"

PASS=0
FAIL=0
pass() { printf '  PASS  %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '  FAIL  %s\n' "$1" >&2; FAIL=$((FAIL + 1)); }

# shellcheck disable=SC1090
source "${PROFILE}"

if benchmark_version_at_least 570.169 560.28.03 && ! benchmark_version_at_least 550.54.15 560.28.03; then
    pass "driver version comparison"
else
    fail "driver version comparison"
fi

if [[ "${BENCHMARK_TORCH_VERSION}" == "2.7.1+cu126" \
   && "${BENCHMARK_TORCHVISION_VERSION}" == "0.22.1+cu126" \
   && "${BENCHMARK_TORCHCODEC_VERSION}" == "0.5" \
   && "${BENCHMARK_NUMBA_VERSION}" == "0.59.1" \
   && "${BENCHMARK_LLVM_LITE_VERSION}" == "0.42.0" ]]; then
    pass "validated profile is fail-closed"
else
    fail "validated profile is fail-closed"
fi

if grep -Fq 'if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]' "${ROSDEP}" \
   && grep -Fq 'run_cmd "${bootstrap_python}" -m venv --system-site-packages' "${ROSDEP}"; then
    pass "early venv uses selected bootstrap Python"
else
    fail "early venv uses selected bootstrap Python"
fi

if grep -Fq 'Skipping GraspGen/manipulation dependencies for Benchmark-only setup.' "${PYTHON_VENV}" \
   && grep -Fq 'if [[ "${INSTALL_BENCHMARK_DEPS:-false}" != true' "${PYTHON_VENV}"; then
    pass "Benchmark setup excludes GraspGen install and smoke test"
else
    fail "Benchmark setup excludes GraspGen install and smoke test"
fi

if grep -Fq -- '--constraint "${BENCHMARK_PIP_CONSTRAINTS}"' "${PYTHON_VENV}" \
   && grep -Fq 'pip_install+=(--constraint "${BENCHMARK_PIP_CONSTRAINTS}")' "${PYTHON_VENV}" \
   && grep -Fq 'cat "${BENCHMARK_PIP_CONSTRAINTS}" >> "${ros_abi_constraints}"' "${PYTHON_VENV}" \
   && grep -Fq 'torchcodec==${BENCHMARK_TORCHCODEC_VERSION}' "${PROFILE}"; then
    pass "Torch profile is protected during later dependency installs"
else
    fail "Torch profile is protected during later dependency installs"
fi

printf '\nBenchmark profile tests: %d passed, %d failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]

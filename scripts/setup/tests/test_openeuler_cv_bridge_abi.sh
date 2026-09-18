#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts"
source "${REPO_ROOT}/scripts/setup/platforms/openeuler-embedded-24.03.sh"

PASS=0
FAIL=0
CAPTURED=""

log_info() { :; }
log_warn() { :; }
TEST_ARCH=aarch64
uname() { printf '%s\n' "${TEST_ARCH}"; }
run_sudo() { CAPTURED="$(printf '%q ' "$@")"$'\n'; }

ensure_openeuler_ros_cv_bridge_abi

expected="dnf install --refresh -y --nogpgcheck opencv\ \>=\ 4.13.0 ros-humble-cv-bridge\ \>=\ 3.2.1-2.oe2403 "
if [[ "${CAPTURED%$'\n'}" == "${expected}" ]]; then
    PASS=$((PASS + 1))
    printf '  PASS  openEuler installs the OpenCV 4.13/cv_bridge ABI pair\n'
else
    FAIL=$((FAIL + 1))
    printf '  FAIL  openEuler installs the OpenCV 4.13/cv_bridge ABI pair\n' >&2
    printf '    expected: %s\n    actual:   %s\n' "${expected}" "${CAPTURED%$'\n'}" >&2
fi

CAPTURED=""
TEST_ARCH=x86_64
ensure_openeuler_ros_cv_bridge_abi
if [[ -z "${CAPTURED}" ]]; then
    PASS=$((PASS + 1))
    printf '  PASS  architecture-specific RPM pair is not installed on another architecture\n'
else
    FAIL=$((FAIL + 1))
fi

TEST_ARCH=aarch64
run_sudo() { return 17; }
if ensure_openeuler_ros_cv_bridge_abi; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  installation failure was swallowed\n' >&2
else
    PASS=$((PASS + 1))
    printf '  PASS  unavailable RPM pair propagates the package manager failure\n'
fi

printf '\n== summary: %d passed, %d failed ==\n' "${PASS}" "${FAIL}"
[[ ${FAIL} -eq 0 ]]

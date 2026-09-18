#!/usr/bin/env bash
# Regression harness for CANN detection and openEuler Torch ABI selection.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SETUP_LIB="${REPO_ROOT}/scripts/setup/python_venv.sh"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT
WORKSPACE="${REPO_ROOT}"

# shellcheck disable=SC1090
source "${SETUP_LIB}"

PASS=0
FAIL=0

assert_eq() {
    local name="$1"
    local expected="$2"
    local actual="$3"

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

detect_or_empty() {
    IBR_CANN_TOOLKIT_ROOT="$1" detect_cann_version 2>/dev/null || true
}

LATEST_ROOT="${TMP_ROOT}/ascend-toolkit/latest"
mkdir -p "${LATEST_ROOT}"
cat > "${LATEST_ROOT}/version.cfg" <<'EOF'
# CANN 8.1.RC1 layout observed on the Ascend310P1 openEuler host.
runtime_running_version=[7.7.0.1.238:8.1.RC1]
compiler_running_version=[7.7.0.1.238:8.1.RC1]
runtime_installed_version=[7.7.0.1.238:8.1.RC1]
EOF
assert_eq "merged latest/version.cfg reports the public CANN version" \
    "8.1.RC1" "$(detect_or_empty "${LATEST_ROOT}")"

RELEASE_ROOT="${TMP_ROOT}/ascend-toolkit/8.1.RC1"
mkdir -p "${RELEASE_ROOT}/aarch64-linux"
cat > "${RELEASE_ROOT}/aarch64-linux/ascend_toolkit_install.info" <<'EOF'
package_name=Ascend-cann-toolkit
version=8.1.RC1
innerversion=V100R001C21SPC001B238
arch=aarch64
os=linux
path=/usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux
EOF
assert_eq "versioned aarch64 install metadata reports CANN 8.1" \
    "8.1.RC1" "$(detect_or_empty "${RELEASE_ROOT}")"

COMPONENT_ROOT="${TMP_ROOT}/component-layout"
mkdir -p "${COMPONENT_ROOT}/runtime"
cat > "${COMPONENT_ROOT}/runtime/version.info" <<'EOF'
Version=7.7.0.1.238
version_dir=8.1.RC1
timestamp=20250428_203017761
EOF
assert_eq "component metadata ignores the internal runtime version" \
    "8.1.RC1" "$(detect_or_empty "${COMPONENT_ROOT}")"

assert_eq "CANN 8.1 selects its requirements variant" \
    "openeuler-24.03-cann-8.1.txt" \
    "$(openeuler_requirements_for_cann "8.1.RC1")"
assert_eq "CANN 8.10 does not match the CANN 8.1 family" \
    "openeuler-24.03.txt" \
    "$(openeuler_requirements_for_cann "8.10.RC1")"
assert_eq "CANN 8.3 preserves the current openEuler requirements" \
    "openeuler-24.03.txt" \
    "$(openeuler_requirements_for_cann "8.3.RC1")"
assert_eq "missing CANN preserves the current openEuler requirements" \
    "openeuler-24.03.txt" \
    "$(openeuler_requirements_for_cann "")"

requirements_packages() {
    local line
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" || "${line}" == \#* ]] || printf '%s\n' "${line}"
    done < "$1"
}
assert_eq "CANN 8.1 requirements pin the complete Torch ABI" \
    "aiortc,av>=15,<16,onnx,onnxruntime,pygraphviz,atomgit_sdk,decorator,torch==2.5.1,torch_npu==2.5.1,torchvision==0.20.1" \
    "$(requirements_packages "${REPO_ROOT}/requirements/openeuler-24.03-cann-8.1.txt" | paste -sd, -)"
assert_eq "CANN 8.1 LeRobot runtime pins the PI05-compatible Transformers version" \
    "transformers==5.3.0" \
    "$(grep -E '^transformers==' "${REPO_ROOT}/requirements/lerobot-v0.6-cann-8.1-inference.txt")"

# Exercise the setup integration without invoking pip. The fake CANN root is
# selected through the same environment variable honored by a sourced CANN
# environment, and run_cmd captures the exact package arguments.
CAPTURED_COMMAND=""
log_info() { :; }
run_cmd() { CAPTURED_COMMAND="$(printf '%q ' "$@")"; }
ASCEND_TOOLKIT_HOME="${LATEST_ROOT}" \
    install_openeuler_python_dependencies fake-python -m pip install
assert_eq "openEuler setup passes all CANN 8.1 pins to pip" \
    "fake-python -m pip install -r ${REPO_ROOT}/requirements/openeuler-24.03-cann-8.1.txt --quiet " \
    "${CAPTURED_COMMAND}"

EMPTY_ROOT="${TMP_ROOT}/no-cann"
mkdir -p "${EMPTY_ROOT}"
ASCEND_TOOLKIT_HOME="${EMPTY_ROOT}" \
    install_openeuler_python_dependencies fake-python -m pip install
assert_eq "openEuler setup retains the master default without CANN 8.1" \
    "fake-python -m pip install -r ${REPO_ROOT}/requirements/openeuler-24.03.txt --quiet " \
    "${CAPTURED_COMMAND}"

ABI_STUBS="${TMP_ROOT}/abi-stubs"
mkdir -p "${ABI_STUBS}/torch" "${ABI_STUBS}/torch_npu" "${ABI_STUBS}/torchvision"
printf '__version__ = "2.5.1"\n' > "${ABI_STUBS}/torch/__init__.py"
printf '__version__ = "2.5.1"\n' > "${ABI_STUBS}/torch_npu/__init__.py"
printf '__version__ = "0.20.1"\n' > "${ABI_STUBS}/torchvision/__init__.py"
if IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONPATH="${ABI_STUBS}" verify_cann_torch_abi python3 >/dev/null; then
    PASS=$((PASS + 1))
    printf '  PASS  CANN 8.1 accepts the exact Torch ABI\n'
else
    FAIL=$((FAIL + 1))
    printf '  FAIL  CANN 8.1 accepts the exact Torch ABI\n' >&2
fi

printf '__version__ = "2.10.0"\n' > "${ABI_STUBS}/torch/__init__.py"
if IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" PYTHONPATH="${ABI_STUBS}" verify_cann_torch_abi python3 >/dev/null 2>&1; then
    FAIL=$((FAIL + 1))
    printf '  FAIL  CANN 8.1 rejects a mixed Torch ABI\n' >&2
else
    PASS=$((PASS + 1))
    printf '  PASS  CANN 8.1 rejects a mixed Torch ABI\n'
fi

check_lerobot_python_compat() { return 0; }
check_lerobot_ros_numpy_compat() { return 0; }
CAPTURED_COMMANDS=""
run_cmd() { CAPTURED_COMMANDS+="$(printf '%q ' "$@")"$'\n'; }
IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" SETUP_PROFILE=inference \
    install_lerobot_editable fake-python -m pip
expected_lerobot_commands="fake-python -m pip install -r ${REPO_ROOT}/requirements/lerobot-v0.6-cann-8.1.txt --quiet "$'\n'
expected_lerobot_commands+="fake-python -m pip install -r ${REPO_ROOT}/requirements/lerobot-v0.6-cann-8.1-inference.txt --quiet "$'\n'
expected_lerobot_commands+="fake-python -m pip install --no-deps -e ${REPO_ROOT}/libs/lerobot "
assert_eq "CANN 8.1 editable LeRobot install bypasses incompatible Torch constraints" \
    "${expected_lerobot_commands}" "${CAPTURED_COMMANDS%$'\n'}"

CAPTURED_COMMANDS=""
IBR_CANN_TOOLKIT_ROOT="${LATEST_ROOT}" SETUP_PROFILE=full \
    install_lerobot_editable fake-python -m pip
expected_lerobot_commands="fake-python -m pip install -r ${REPO_ROOT}/requirements/lerobot-v0.6-cann-8.1.txt --quiet "$'\n'
expected_lerobot_commands+="fake-python -m pip install -r ${REPO_ROOT}/requirements/lerobot-v0.6-cann-8.1-full.txt --quiet "$'\n'
expected_lerobot_commands+="fake-python -m pip install --no-deps -e ${REPO_ROOT}/libs/lerobot "
assert_eq "CANN 8.1 full setup retains dataset and kinematics dependencies" \
    "${expected_lerobot_commands}" "${CAPTURED_COMMANDS%$'\n'}"

echo
echo "== summary: ${PASS} passed, ${FAIL} failed =="
[[ ${FAIL} -eq 0 ]]

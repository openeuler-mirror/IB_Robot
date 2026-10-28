#!/usr/bin/env bash
# Regression harness for the MuJoCo simulation pin (issue #190,
# upgrade-mujoco-ros2-control-0-1 D5).
#
# Covers the three install orders through mujoco_sim_pin_mode, the mujoco<3.9
# benchmark constraint generated before LeRobot resolves the libero extra,
# the inference-profile verification skip, and the stdlib-only
# verify_mujoco_sim on an unbuilt workspace (a fake ROS prefix plus a fake
# mujoco dist-info on PYTHONPATH; no workspace import, no network, no pip).
#
# Everything here is hermetic: the only touched paths are a temp directory.
# The gates are exercised by sourcing setup.sh (its main() only runs when
# executed directly) and stubbing the side-effecting helpers.
#
# Usage:
#   scripts/setup/tests/test_mujoco_sim_pin.sh
#
# Exit status:
#   0  — all checks pass.
#   1  — at least one check failed.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

PASS=0
FAIL=0

ok() { PASS=$((PASS + 1)); printf '  PASS  %s\n' "$1"; }

ko() {
    FAIL=$((FAIL + 1))
    printf '  FAIL  %s\n' "$1" >&2
    if [[ $# -gt 1 ]]; then
        printf '        %s\n' "$2" >&2
    fi
}

assert_contains() {
    if grep -qF -- "$2" <<<"$3"; then
        ok "$1"
    else
        ko "$1" "missing: $2"
        printf '        ---- actual ----\n%s\n' "$3" >&2
    fi
}

assert_not_contains() {
    if grep -qF -- "$2" <<<"$3"; then
        ko "$1" "unexpected: $2"
        printf '        ---- actual ----\n%s\n' "$3" >&2
    else
        ok "$1"
    fi
}

assert_eq() {
    if [[ "$2" == "$3" ]]; then ok "$1"; else ko "$1" "expected '$2', got '$3'"; fi
}

line_for() { grep -F -- "$1" <<<"$2" | head -1; }

probe() { ( cd "${REPO_ROOT}" && bash -c "$1" 2>&1 ); }

# ---------------------------------------------------------------------------
# 1. requirements files
# ---------------------------------------------------------------------------
assert_contains "mujoco-sim.txt pins the simulator's libmujoco" \
    "mujoco==3.12.0" "$(cat "${REPO_ROOT}/requirements/mujoco-sim.txt")"
assert_contains "mujoco-sim.txt documents the LIBERO exclusion" \
    "LIBERO benchmark venvs must NOT install this pin" "$(cat "${REPO_ROOT}/requirements/mujoco-sim.txt")"
assert_not_contains "ubuntu-22.04.txt no longer pins mujoco" \
    "mujoco==" "$(cat "${REPO_ROOT}/requirements/ubuntu-22.04.txt")"

# ---------------------------------------------------------------------------
# 2. mujoco_sim_pin_mode: the three install orders + the none modes
# ---------------------------------------------------------------------------
PROBE_MODES="$(probe '
set +u
source scripts/setup.sh
log_info() { :; }
mujoco_sim_venv_has_libero() { return 1; }
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_PROFILE=full
INSTALL_BENCHMARK_DEPS=false
echo "MODE fresh-full $(mujoco_sim_pin_mode)"
INSTALL_BENCHMARK_DEPS=true
echo "MODE sim-then-benchmark $(mujoco_sim_pin_mode)"
INSTALL_BENCHMARK_DEPS=false
mujoco_sim_venv_has_libero() { return 0; }
echo "MODE benchmark-then-sim $(mujoco_sim_pin_mode)"
SETUP_PROFILE=inference
mujoco_sim_venv_has_libero() { return 1; }
echo "MODE inference $(mujoco_sim_pin_mode)"
SETUP_PROFILE=training
echo "MODE training $(mujoco_sim_pin_mode)"
SETUP_PROFILE=full
SETUP_PLATFORM_ID=openeuler-embedded-24.03
echo "MODE openeuler $(mujoco_sim_pin_mode)"
')"
assert_eq "fresh full setup installs the pin" "MODE fresh-full install" "$(line_for "MODE fresh-full" "${PROBE_MODES}")"
assert_eq "simulation -> benchmark skips the pin" \
    "MODE sim-then-benchmark skip-benchmark" "$(line_for "MODE sim-then-benchmark" "${PROBE_MODES}")"
assert_eq "benchmark -> simulation skips the pin (existing install intact)" \
    "MODE benchmark-then-sim skip-libero-present" "$(line_for "MODE benchmark-then-sim" "${PROBE_MODES}")"
assert_eq "inference profile has no pin" "MODE inference none" "$(line_for "MODE inference" "${PROBE_MODES}")"
assert_eq "training profile has no pin" "MODE training none" "$(line_for "MODE training" "${PROBE_MODES}")"
assert_eq "openEuler has no pin" "MODE openeuler none" "$(line_for "MODE openeuler" "${PROBE_MODES}")"

# The install/skip behaviour itself: run_cmd sees mujoco-sim.txt only in
# install mode; the skip modes warn and install nothing.
PROBE_INSTALL="$(probe '
set +u
source scripts/setup.sh
CAPTURED=""
run_cmd() { CAPTURED+="$(printf "%q " "$@") ; "; }
log_info() { :; }
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_PROFILE=full
mujoco_sim_venv_has_libero() { return 1; }
CAPTURED=""
install_mujoco_sim_pin fake-pip >/dev/null 2>&1
echo "INSTALL mode=install [${CAPTURED}]"
mujoco_sim_venv_has_libero() { return 0; }
CAPTURED=""
SKIP_LIBERO_OUT="$(install_mujoco_sim_pin fake-pip 2>&1)"
echo "INSTALL mode=skip-libero [${CAPTURED}]"
echo "SKIP_LIBERO_OUT ${SKIP_LIBERO_OUT}"
INSTALL_BENCHMARK_DEPS=true
CAPTURED=""
SKIP_BENCH_OUT="$(install_mujoco_sim_pin fake-pip 2>&1)"
echo "INSTALL mode=skip-benchmark [${CAPTURED}]"
echo "SKIP_BENCH_OUT ${SKIP_BENCH_OUT}"
')"
assert_contains "install mode installs requirements/mujoco-sim.txt" \
    "requirements/mujoco-sim.txt" "$(line_for "INSTALL mode=install" "${PROBE_INSTALL}")"
assert_not_contains "skip-libero-present installs nothing" "mujoco-sim.txt" \
    "$(line_for "INSTALL mode=skip-libero" "${PROBE_INSTALL}")"
# the warnings are multi-line log_warn output, so match the whole probe
assert_contains "skip-libero-present warns about recreating the venv" \
    "Recreate the venv" "${PROBE_INSTALL}"
assert_not_contains "skip-benchmark installs nothing" "mujoco-sim.txt" \
    "$(line_for "INSTALL mode=skip-benchmark" "${PROBE_INSTALL}")"
assert_contains "skip-benchmark warns the MuJoCo sim is unavailable" \
    "MuJoCo ROS simulation (sim_platform:=mujoco) is unavailable" "${PROBE_INSTALL}"

# ---------------------------------------------------------------------------
# 3. benchmark constraints carry mujoco<3.9 before LeRobot resolves
# ---------------------------------------------------------------------------
CONSTRAINTS_FILE="${TMP_ROOT}/benchmark_constraints.txt"
PROBE_CONSTRAINTS="$(probe "
set +u
source scripts/setup.sh
INSTALL_BENCHMARK_DEPS=true
run_cmd() { :; }
benchmark_prepare_torch_profile /usr/bin/python3 '${CONSTRAINTS_FILE}' >/dev/null 2>&1 || true
cat '${CONSTRAINTS_FILE}' 2>/dev/null
echo \"EXPORTED \${BENCHMARK_PIP_CONSTRAINTS:-<none>}\"
")"
assert_contains "the generated benchmark constraints pin mujoco<3.9" \
    "mujoco<3.9" "${PROBE_CONSTRAINTS}"
assert_contains "the constraint file is exported as BENCHMARK_PIP_CONSTRAINTS" \
    "EXPORTED ${CONSTRAINTS_FILE}" "${PROBE_CONSTRAINTS}"

# ---------------------------------------------------------------------------
# 4. verify_mujoco_sim: scope, unbuilt workspace, diagnostics
# ---------------------------------------------------------------------------
# A fake ROS prefix and a fake mujoco dist-info: the check is stdlib-only and
# must pass with the system interpreter on an unbuilt workspace.
FAKE_ROS="${TMP_ROOT}/ros"
mkdir -p "${FAKE_ROS}/share/mujoco_ros2_control" "${FAKE_ROS}/opt/mujoco_vendor/lib" "${TMP_ROOT}/empty_ws"
cat >"${FAKE_ROS}/share/mujoco_ros2_control/package.xml" <<'EOF'
<?xml version="1.0"?>
<package><name>mujoco_ros2_control</name><version>0.1.2</version></package>
EOF
: >"${FAKE_ROS}/opt/mujoco_vendor/lib/libmujoco.so.3.12.0"
: >"${FAKE_ROS}/setup.bash"
# lekiwi-mujoco-runtime-platform D11: the fake prefix also carries the MuJoCo
# lidar engine extension's ament resource, package version and library
mkdir -p "${FAKE_ROS}/share/ament_index/resource_index/mujoco_plugins" \
    "${FAKE_ROS}/share/mujoco_3d_lidar" \
    "${FAKE_ROS}/lib/mujoco_3d_lidar/mujoco_plugin"
printf 'lib/mujoco_3d_lidar/mujoco_plugin' \
    >"${FAKE_ROS}/share/ament_index/resource_index/mujoco_plugins/mujoco_3d_lidar"
cat >"${FAKE_ROS}/share/mujoco_3d_lidar/package.xml" <<'EOF'
<?xml version="1.0"?>
<package><name>mujoco_3d_lidar</name><version>0.1.2</version></package>
EOF
: >"${FAKE_ROS}/lib/mujoco_3d_lidar/mujoco_plugin/libmujoco_3d_lidar.so"

make_dist_info() {
    local version="$1"
    local dir="${TMP_ROOT}/site-${version}/mujoco-${version}.dist-info"
    mkdir -p "${dir}"
    printf 'Metadata-Version: 2.1\nName: mujoco\nVersion: %s\n' "${version}" >"${dir}/METADATA"
}
make_dist_info 3.12.0

PROBE_VERIFY="$(probe '
set +u
source scripts/setup.sh
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_PROFILE=full
VENV_PYTHON=/usr/bin/python3
WORKSPACE="'"${TMP_ROOT}"'/empty_ws"
USE_SUDO=false
ROSDEP_BIN=/bin/true
SETUP_ROS_SETUP_PATH="'"${FAKE_ROS}"'/setup.bash"
# verify_env.sh is sourced by setup.sh main(); source it here for the checks
source scripts/setup/verify_env.sh
mujoco_sim_venv_has_libero() { return 1; }

# inference profile with the vendor installed on the host: skipped, no failure
SETUP_PROFILE=inference
INFERENCE_OUT="$(verify_mujoco_sim 2>&1)"
echo "VERIFY inference rc=$?"
echo "${INFERENCE_OUT}"

# full profile, unbuilt workspace, matching versions
SETUP_PROFILE=full
export PYTHONPATH="'"${TMP_ROOT}"'/site-3.12.0"
verify_mujoco_sim 2>&1
')"
assert_contains "inference profile skips the MuJoCo verification" \
    "Skipping MuJoCo simulation pin verification" "${PROBE_VERIFY}"
assert_contains "the inference skip is a pass, not a failure" \
    "VERIFY inference rc=0" "${PROBE_VERIFY}"
assert_contains "unbuilt workspace verifies through the fake prefix" \
    "MuJoCo simulation pin verified: mujoco_ros2_control 0.1.2, libmujoco 3.12.0, python mujoco 3.12.0, mujoco_3d_lidar 0.1.2" "${PROBE_VERIFY}"

# Mismatch: a 3.4.0 dist-info next to the 3.12.0 vendor library fails and
# names both versions.
make_dist_info 3.4.0
PROBE_MISMATCH="$(probe '
set +u
source scripts/setup.sh
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_PROFILE=full
VENV_PYTHON=/usr/bin/python3
WORKSPACE="'"${TMP_ROOT}"'/empty_ws"
USE_SUDO=false
ROSDEP_BIN=/bin/true
SETUP_ROS_SETUP_PATH="'"${FAKE_ROS}"'/setup.bash"
source scripts/setup/verify_env.sh
export PYTHONPATH="'"${TMP_ROOT}"'/site-3.4.0"
mujoco_sim_venv_has_libero() { return 1; }
if verify_mujoco_sim 2>&1; then rc=0; else rc=$?; fi
echo "VERIFY_MISMATCH rc=${rc}"
')"
assert_contains "a version mismatch fails verification" "VERIFY_MISMATCH rc=1" "${PROBE_MISMATCH}"
assert_contains "the mismatch names both versions and the remedy" \
    "Python mujoco 3.4.0 != mujoco_vendor libmujoco 3.12.0" "${PROBE_MISMATCH}"

# A pre-CameraPlugin simulator fails with the upgrade diagnostic even when
# the MuJoCo versions would match.
make_dist_info 3.4.0
OLD_ROS="${TMP_ROOT}/ros-old"
mkdir -p "${OLD_ROS}/share/mujoco_ros2_control" "${OLD_ROS}/opt/mujoco_vendor/lib"
cat >"${OLD_ROS}/share/mujoco_ros2_control/package.xml" <<'EOF'
<?xml version="1.0"?>
<package><name>mujoco_ros2_control</name><version>0.0.3</version></package>
EOF
: >"${OLD_ROS}/opt/mujoco_vendor/lib/libmujoco.so.3.4.0"
PROBE_TOO_OLD="$(probe '
set +u
source scripts/setup.sh
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_PROFILE=full
VENV_PYTHON=/usr/bin/python3
WORKSPACE="'"${TMP_ROOT}"'/empty_ws"
USE_SUDO=false
ROSDEP_BIN=/bin/true
SETUP_ROS_SETUP_PATH="'"${OLD_ROS}"'/setup.bash"
source scripts/setup/verify_env.sh
export PYTHONPATH="'"${TMP_ROOT}"'/site-3.4.0"
mujoco_sim_venv_has_libero() { return 1; }
if verify_mujoco_sim 2>&1; then rc=0; else rc=$?; fi
echo "VERIFY_TOO_OLD rc=${rc}"
')"
assert_contains "a too-old control package fails verification" "VERIFY_TOO_OLD rc=1" "${PROBE_TOO_OLD}"
assert_contains "the too-old diagnostic names the apt upgrade" \
    "is older than the CameraPlugin release (0.1.0)" "${PROBE_TOO_OLD}"

# lekiwi-mujoco-runtime-platform D11: a missing lidar extension is rosdep's
# job (re-run setup); an installed-but-old one names the apt upgrade, because
# rosdep never upgrades an installed package.
verify_lidar_probe() {
    probe '
set +u
source scripts/setup.sh
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_PROFILE=full
VENV_PYTHON=/usr/bin/python3
WORKSPACE="'"${TMP_ROOT}"'/empty_ws"
USE_SUDO=false
ROSDEP_BIN=/bin/true
SETUP_ROS_SETUP_PATH="'"$1"'/setup.bash"
source scripts/setup/verify_env.sh
export PYTHONPATH="'"${TMP_ROOT}"'/site-3.12.0"
mujoco_sim_venv_has_libero() { return 1; }
if verify_mujoco_sim 2>&1; then rc=0; else rc=$?; fi
echo "VERIFY_LIDAR rc=${rc}"
'
}
make_dist_info 3.12.0
NO_LIDAR_ROS="${TMP_ROOT}/ros-no-lidar"
cp -r "${FAKE_ROS}" "${NO_LIDAR_ROS}"
rm -rf "${NO_LIDAR_ROS}/share/ament_index/resource_index/mujoco_plugins" \
    "${NO_LIDAR_ROS}/share/mujoco_3d_lidar" "${NO_LIDAR_ROS}/lib/mujoco_3d_lidar"
PROBE_NO_LIDAR="$(verify_lidar_probe "${NO_LIDAR_ROS}")"
assert_contains "a missing lidar extension fails verification" "VERIFY_LIDAR rc=1" "${PROBE_NO_LIDAR}"
assert_contains "the missing-package diagnostic points at setup's rosdep step" \
    "re-run ./scripts/setup.sh with sudo, its rosdep step installs it" "${PROBE_NO_LIDAR}"
assert_not_contains "the missing-package diagnostic does not bypass rosdep" \
    "apt install" "${PROBE_NO_LIDAR}"

OLD_LIDAR_ROS="${TMP_ROOT}/ros-old-lidar"
cp -r "${FAKE_ROS}" "${OLD_LIDAR_ROS}"
cat >"${OLD_LIDAR_ROS}/share/mujoco_3d_lidar/package.xml" <<'EOF2'
<?xml version="1.0"?>
<package><name>mujoco_3d_lidar</name><version>0.1.1</version></package>
EOF2
PROBE_OLD_LIDAR="$(verify_lidar_probe "${OLD_LIDAR_ROS}")"
assert_contains "a too-old lidar extension fails verification" "VERIFY_LIDAR rc=1" "${PROBE_OLD_LIDAR}"
assert_contains "the too-old lidar diagnostic names the apt upgrade" \
    "sudo apt install --only-upgrade ros-humble-mujoco-3d-lidar" "${PROBE_OLD_LIDAR}"

# ---------------------------------------------------------------------------
# §8 (D5 addendum): the pin also constrains every later venv install.
# ---------------------------------------------------------------------------
PROBE_CONSTRAINT="$(probe '
set +u
source scripts/setup.sh
log_info() { :; }
SETUP_PLATFORM_ID=ubuntu-22.04
SETUP_PROFILE=full
mujoco_sim_venv_has_libero() { return 1; }
INSTALL_BENCHMARK_DEPS=false
echo "CONSTRAINT install $(mujoco_sim_pin_constraint)"
INSTALL_BENCHMARK_DEPS=true
echo "CONSTRAINT skip-benchmark [$(mujoco_sim_pin_constraint)]"
INSTALL_BENCHMARK_DEPS=false
mujoco_sim_venv_has_libero() { return 0; }
echo "CONSTRAINT skip-libero-present [$(mujoco_sim_pin_constraint)]"
SETUP_PROFILE=inference
mujoco_sim_venv_has_libero() { return 1; }
echo "CONSTRAINT none-inference [$(mujoco_sim_pin_constraint)]"
SETUP_PLATFORM_ID=openeuler-embedded-24.03
SETUP_PROFILE=full
echo "CONSTRAINT none-openeuler [$(mujoco_sim_pin_constraint)]"
')"
install_line="$(line_for "CONSTRAINT install " "${PROBE_CONSTRAINT}")"
case "${install_line}" in
  *requirements/mujoco-sim.txt) ok "install mode constrains later installs to the pin file (${install_line#CONSTRAINT install })" ;;
  *) ko "install mode constrains later installs to the pin file" "${install_line:-<no line>}" ;;
esac
assert_eq "skip-benchmark adds no constraint" "CONSTRAINT skip-benchmark []" "$(line_for "CONSTRAINT skip-benchmark" "${PROBE_CONSTRAINT}")"
assert_eq "skip-libero-present adds no constraint" "CONSTRAINT skip-libero-present []" "$(line_for "CONSTRAINT skip-libero-present" "${PROBE_CONSTRAINT}")"
assert_eq "inference profile adds no constraint" "CONSTRAINT none-inference []" "$(line_for "CONSTRAINT none-inference" "${PROBE_CONSTRAINT}")"
assert_eq "openEuler adds no constraint" "CONSTRAINT none-openeuler []" "$(line_for "CONSTRAINT none-openeuler" "${PROBE_CONSTRAINT}")"

# The wiring must actually use the constraint: the exported PIP_CONSTRAINT is
# prepended after the pin install, and the pin file is appended to
# ros_abi_constraints.txt (explicit --constraint users).
PIN_VENV_SH="$(cat "${REPO_ROOT}/scripts/setup/python_venv.sh")"
assert_contains "PIP_CONSTRAINT gets the pin file prepended after the pin install" \
    'PIP_CONSTRAINT="${mujoco_pin_constraint}${PIP_CONSTRAINT:+ ${PIP_CONSTRAINT}}"' "${PIN_VENV_SH}"
assert_contains "ros_abi_constraints.txt gets the pin file appended" \
    'cat "${mujoco_pin_constraint}" >> "${ros_abi_constraints}"' "${PIN_VENV_SH}"
grep -q "install_graspgen_pip" "${REPO_ROOT}/scripts/setup/python_venv.sh" \
  && grep -q 'constraint_args=(--constraint "\${ROS_ABI_CONSTRAINTS}")' "${REPO_ROOT}/scripts/setup/install_graspgen_pip.sh" \
  && ok "graspgen explicit --constraint path carries the pin via ROS_ABI_CONSTRAINTS" \
  || ko "graspgen explicit --constraint path carries the pin via ROS_ABI_CONSTRAINTS" "wiring not found"

# ---------------------------------------------------------------------------
printf '\n%s passed, %s failed\n' "${PASS}" "${FAIL}"
[[ "${FAIL}" -eq 0 ]]

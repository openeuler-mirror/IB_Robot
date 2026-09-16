#!/usr/bin/env bash
# verify_runtime_independence.sh — the two-direction independence gate
# (robot-runtime-packaging spec, design D9; so101-runtime-migration task 6.4).
#
# Usage:
#   scripts/verify_runtime_independence.sh runtime so101_robot [--profile so101_single_arm]
#   scripts/verify_runtime_independence.sh core
#
# runtime mode: build so101_robot in an ISOLATED workspace containing only its
#   declared dependency closure. The closure is audited against a FIXED
#   allow-list (runtime members + the contract layer + vendored third-party);
#   membership is NOT derived from the closure itself. The isolated build uses
#   the workspace venv and the project's colcon flags (merge-install,
#   symlink-install, the repo's mixin file) so it matches ./scripts/build.sh.
#   The runtime is then launched through its own entry in simulated transport
#   and the capability-scoped conformance suite runs against it.
#
# core mode: build the generic IB-Robot packages with every robot runtime,
#   SDK, driver adapter, motion package, robot suite, robot description and
#   planning-framework package excluded, then actually launch
#   robot_config/robot.launch.py with the mock runtime provider and verify the
#   contract chain: RuntimeStatus reconciliation, a mock motion service call,
#   and the stop service.
#
# Both modes run in a temporary workspace (package dirs are symlinked; C++
# requires real copies, so colcon copies via symlink-install only for python).
set -eo pipefail

MODE="${1:-}"
shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_WORKSPACE="${MAIN_WORKSPACE:-$(dirname "$SCRIPT_DIR")}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Fixed allow-lists (so101-runtime-migration): membership is declared here,
# never derived from the closure under audit.
# ---------------------------------------------------------------------------
CONTRACT_PACKAGES=(ibrobot_msgs robot_runtime)
SO101_RUNTIME_MEMBERS=(
  so101_robot so101_sdk so101_hardware so101_description so101_motion so101_suite
  feetech_sdk
)
# Third-party/vendored packages that may appear in a runtime closure without
# being IB-Robot generic packages (they are not listed by `colcon list` in the
# main workspace src/ tree anyway; this list documents intent).
THIRD_PARTY_ALLOWED=(pymoveit2)

# Generic packages that must build WITHOUT any robot package (core gate).
CORE_EXCLUDED_SUFFIXES=("_robot" "_sdk" "_hardware" "_motion" "_suite")
CORE_EXCLUDED_PACKAGES=(
  so101_description moveit lekiwi_description robot_calibration
  livox_ros_driver2 fast_lio fast_calib omni_wheel_controller rosclaw pymoveit2
)
CORE_BUILD_PACKAGES=(
  robot_config embodied_bringup manipulation_execution skill_library skill_catalog
  task_dispatch action_dispatch inference_service dataset_tools robot_navigation
  robot_teleop safety_guard embodied_agent embodied_common
)

list_workspace_packages() {
  (cd "$MAIN_WORKSPACE" && colcon list --base-paths src --names-only 2>/dev/null)
}

package_dir_for() {
  local pkg="$1" dir
  dir="$(cd "$MAIN_WORKSPACE" && colcon list --base-paths src 2>/dev/null | awk -F'\t' -v p="$pkg" '$1==p {print $2}')"
  [[ -n "$dir" ]] || return 1
  echo "$MAIN_WORKSPACE/$dir"
}

# ---------------------------------------------------------------------------
# Isolated build environment: /opt/ros + the workspace venv, project colcon
# flags (mirrors scripts/build.sh: merge-install, symlink-install, repo mixin).
# ---------------------------------------------------------------------------
isolated_colcon_build() {
  local ws="$1"; shift
  local venv_python="$MAIN_WORKSPACE/venv/bin/python3"
  [[ -x "$venv_python" ]] || { log_error "workspace venv not found at $venv_python"; exit 1; }
  (
    cd "$ws"
    set +u; source /opt/ros/humble/setup.sh; set -u
    # Same flags as scripts/build.sh (dev mixin): merge + symlink install.
    PYTHONNOUSERSITE=1 "$venv_python" -m colcon build \
      --merge-install --symlink-install \
      --parallel-workers "$(nproc)" \
      --cmake-args -Wno-dev \
      "$@" >"$ws/build.log" 2>&1
  ) || { log_error "isolated build failed (see $ws/build.log)"; tail -40 "$ws/build.log"; exit 1; }
}

# ---------------------------------------------------------------------------
# runtime mode
# ---------------------------------------------------------------------------
runtime_gate() {
  local robot_pkg="$1"; shift
  local profile="so101_single_arm"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --profile) profile="$2"; shift 2;;
      *) log_error "unknown option: $1"; exit 2;;
    esac
  done

  local allowlist=("${CONTRACT_PACKAGES[@]}" "${SO101_RUNTIME_MEMBERS[@]}" "${THIRD_PARTY_ALLOWED[@]}")

  log_info "runtime gate: computing dependency closure of ${robot_pkg}"
  local closure
  closure="$(list_workspace_packages | xargs -I{} true; cd "$MAIN_WORKSPACE" && colcon list --base-paths src --packages-up-to "$robot_pkg" --names-only 2>/dev/null)"
  [[ -n "$closure" ]] || { log_error "cannot resolve closure for ${robot_pkg}"; exit 1; }
  grep -Fx "$robot_pkg" <<<"$closure" >/dev/null || { log_error "${robot_pkg} not in its own closure"; exit 1; }

  # Audit against the FIXED allow-list (not the closure itself).
  local offenders=()
  while IFS= read -r pkg; do
    local allowed=false
    for entry in "${allowlist[@]}"; do
      [[ "$pkg" == "$entry" ]] && allowed=true
    done
    # Packages not listed by colcon (third-party system deps) are fine.
    if [[ "$allowed" == false ]] && list_workspace_packages | grep -Fx "$pkg" >/dev/null; then
      offenders+=("$pkg")
    fi
  done <<<"$closure"
  if [[ ${#offenders[@]} -gt 0 ]]; then
    log_error "dependency closure of ${robot_pkg} contains non-member workspace packages:"
    printf '  - %s\n' "${offenders[@]}"
    log_error "allowed members: ${allowlist[*]}"
    exit 1
  fi
  log_info "closure audit OK: $(wc -l <<<"$closure") packages, fixed allow-list honored"

  # Isolated workspace.
  local tmp
  tmp="$(mktemp -d /tmp/runtime_independence_XXXXXX)"
  mkdir -p "$tmp/src"
  trap 'rm -rf "$tmp"' EXIT
  while IFS= read -r pkg; do
    local dir
    dir="$(package_dir_for "$pkg")" || { log_error "no source dir for $pkg"; exit 1; }
    ln -s "$dir" "$tmp/src/$pkg"
  done <<<"$closure"
  # Pinned repo-level libraries under libs/ (Livox-SDK2 for livox).
  [[ -d "$MAIN_WORKSPACE/libs" ]] && ln -s "$MAIN_WORKSPACE/libs" "$tmp/libs"

  log_info "building isolated workspace: $tmp (${robot_pkg} closure)"
  isolated_colcon_build "$tmp"

  # Launch the runtime through its own entry and run conformance.
  local profile_path="$MAIN_WORKSPACE/src/robots/so101/so101_robot/profiles/${profile}.yaml"
  [[ -f "$profile_path" ]] || { log_error "profile not found: $profile_path"; exit 1; }
  export CONFORMANCE_LAUNCH="ros2 launch ${robot_pkg} runtime.launch.py profile:=${profile} simulated:=true"
  export CONFORMANCE_PROFILE="$profile_path"
  export CONFORMANCE_STARTUP_TIMEOUT="${CONFORMANCE_STARTUP_TIMEOUT:-120}"
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-110}" ROS_LOCALHOST_ONLY=1

  log_info "running conformance against ${robot_pkg} (profile=${profile}, simulated transport)"
  (
    cd "$tmp"
    set +u; source /opt/ros/humble/setup.sh; source "$tmp/install/setup.sh"; set -u
    cd "$MAIN_WORKSPACE"
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$MAIN_WORKSPACE/venv/bin/python3" -m pytest \
      "$MAIN_WORKSPACE/src/robot_runtime/test/test_conformance.py" -q -p no:cacheprovider
  )
  log_info "runtime gate PASSED for ${robot_pkg} (fixed allow-list, isolated build, simulated conformance)"
}

# ---------------------------------------------------------------------------
# core mode
# ---------------------------------------------------------------------------
core_gate() {
  local tmp
  tmp="$(mktemp -d /tmp/core_independence_XXXXXX)"
  mkdir -p "$tmp/src"
  trap 'rm -rf "$tmp"' EXIT

  log_info "core gate: building the generic set with robot packages excluded"
  local ignore_count=0
  local pkg name rel_dir
  while IFS=$'\t' read -r pkg rel_dir; do
    name="$pkg"
    local skip=false
    for suffix in "${CORE_EXCLUDED_SUFFIXES[@]}"; do
      [[ "$name" == *"$suffix" ]] && skip=true
    done
    for pkg_name in "${CORE_EXCLUDED_PACKAGES[@]}"; do
      [[ "$name" == "$pkg_name" ]] && skip=true
    done
    if [[ "$skip" == "true" ]]; then
      ignore_count=$((ignore_count + 1))
      continue
    fi
    ln -s "$MAIN_WORKSPACE/$rel_dir" "$tmp/src/$name"
  done < <(cd "$MAIN_WORKSPACE" && colcon list --base-paths src 2>/dev/null)
  log_info "excluded $ignore_count robot-specific packages from the temp workspace"

  isolated_colcon_build "$tmp" --packages-select "${CORE_BUILD_PACKAGES[@]}"
  log_info "generic build OK without robot packages"

  # Contract-chain smoke: mock-runtime provider launch + reconciliation + a
  # mock motion service call + the stop service.
  log_info "core gate: launching robot.launch.py with the mock runtime provider"
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-111}" ROS_LOCALHOST_ONLY=1
  (
    cd "$tmp"
    set +u; source /opt/ros/humble/setup.sh; source install/setup.sh; set -u
    python3 - "$MAIN_WORKSPACE" "$tmp" <<'PY'
import os
import subprocess
import sys
import time

import rclpy
from ibrobot_msgs.srv import ComputeIk, GetRuntimeStatus, StopRuntime
from rclpy.node import Node

main_ws = sys.argv[1]
tmp_dir = sys.argv[2]


def await_future(node, future, what, timeout_s=30.0):
    """Spin until the future resolves or the deadline passes; never spin forever."""
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    assert future.done(), f"{what} did not answer within {timeout_s:.0f}s"
    return future.result()


# 1) The generic launch must compose and reconcile against the mock runtime.
#    Launch output goes to a file: a PIPE that is never read deadlocks the
#    launch process once the kernel pipe buffer fills (same pattern as
#    isolated_colcon_build's build.log).
launch_log_path = os.path.join(tmp_dir, "core_gate_launch.log")
launch_log = open(launch_log_path, "w", encoding="utf-8")
launch = subprocess.Popen(
    ["ros2", "launch", "robot_config", "robot.launch.py",
     "robot_config:=so101_mock_core_gate", "use_sim:=false",
     "control_mode:=teleop"],
    stdout=launch_log, stderr=subprocess.STDOUT, text=True)
try:
    rclpy.init()
    node = Node("core_gate_probe")
    deadline = time.monotonic() + 120
    status = None
    client = node.create_client(GetRuntimeStatus, "/runtime/get_status")
    while time.monotonic() < deadline and status is None:
        if client.service_is_ready():
            future = client.call_async(GetRuntimeStatus.Request())
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.2)
            if future.done() and future.result() is not None:
                status = future.result().status
                break
        rclpy.spin_once(node, timeout_sec=0.5)
    assert status is not None, "mock runtime never reached reconciliation"
    assert status.lifecycle == "ACTIVE", f"lifecycle {status.lifecycle}"
    assert status.runtime_name == "mock_runtime", status.runtime_name

    # 2) One mock motion call: ComputeFk then ComputeIk through the contract.
    from ibrobot_msgs.srv import ComputeFk
    from sensor_msgs.msg import JointState
    fk = node.create_client(ComputeFk, "/motion/compute_fk")
    assert fk.wait_for_service(timeout_sec=30), "mock /motion/compute_fk unavailable"
    fkreq = ComputeFk.Request()
    fkreq.joint_state = JointState()
    fkreq.joint_state.name = [f"joint_{i}" for i in range(1, 6)]
    fkreq.joint_state.position = [0.1, 0.2, 0.3, 0.2, 0.1]
    fkreq.link_names = ["ee"]
    fkres = await_future(node, fk.call_async(fkreq), "mock /motion/compute_fk")
    assert fkres is not None and fkres.success and fkres.poses, "mock FK call failed"

    ik = node.create_client(ComputeIk, "/motion/compute_ik")
    assert ik.wait_for_service(timeout_sec=30), "mock /motion/compute_ik unavailable"
    req = ComputeIk.Request()
    req.target = fkres.poses[0]
    req.seed = JointState()
    req.seed.name = [f"joint_{i}" for i in range(1, 6)]
    req.seed.position = [0.1, 0.2, 0.3, 0.2, 0.1]
    ikres = await_future(node, ik.call_async(req), "mock /motion/compute_ik")
    assert ikres is not None and ikres.success, "mock IK call failed"

    # 3) StopRuntime(HOLD) actually engages through the contract: the call
    #    must succeed, report its latencies, and the post-stop status must
    #    show the latched stop, the STOPPED lifecycle and the idle mode.
    stop = node.create_client(StopRuntime, "/runtime/stop")
    assert stop.wait_for_service(timeout_sec=10), "mock /runtime/stop unavailable"
    stop_req = StopRuntime.Request()
    stop_req.policy = "HOLD"
    stop_res = await_future(node, stop.call_async(stop_req), "mock /runtime/stop")
    assert stop_res is not None and stop_res.success, f"mock stop failed: {stop_res}"
    assert stop_res.cancel_latency_s >= 0.0 and stop_res.idle_latency_s >= 0.0, (
        f"stop latencies not reported: {stop_res}"
    )
    after = await_future(
        node, client.call_async(GetRuntimeStatus.Request()), "mock /runtime/get_status after stop"
    ).status
    assert after.stop_latched, "stop did not latch"
    assert after.lifecycle == "STOPPED", f"lifecycle {after.lifecycle} after stop"
    assert after.active_mode == "idle", f"active_mode {after.active_mode} after stop"

    node.destroy_node()
    rclpy.shutdown()
    launch_log.flush()
    print("core gate: reconciliation ACTIVE + mock ComputeFk/ComputeIk + StopRuntime(HOLD) confirmed")
finally:
    launch.terminate()
    try:
        launch.wait(timeout=15)
    except subprocess.TimeoutExpired:
        launch.kill()
PY
  ) || {
    log_error "core gate contract-chain check failed; tail of the launch log follows"
    tail -60 "$tmp/core_gate_launch.log" 2>/dev/null || true
    exit 1
  }
  log_info "core gate PASSED (generic core independent of robot packages)"
}

# ---------------------------------------------------------------------------
case "$MODE" in
  runtime)
    [[ $# -ge 1 ]] || { log_error "usage: $0 runtime <robot>_robot [--profile NAME]"; exit 2; }
    runtime_gate "$@";;
  core)
    core_gate "$@";;
  *)
    log_error "usage: $0 runtime <robot>_robot | core"
    exit 2;;
esac

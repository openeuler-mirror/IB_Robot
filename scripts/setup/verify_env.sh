#!/bin/bash
# verify_env.sh — Independent environment verification for IB_Robot workspace
#
# This file is designed to be sourced (not executed directly).
# It exposes individual verify_* functions and a top-level verify_env entry point.
#
# Required variables (must be set by the caller before sourcing):
#   WORKSPACE          — absolute path to the IB_Robot workspace root
#   VENV_PYTHON        — absolute path to the venv python interpreter
#   SETUP_PLATFORM_ID  — one of: ubuntu-22.04, openeuler-embedded-24.03, openharmony-5.1.0-musl
#   ROSDEP_BIN         — absolute path to the rosdep binary in the venv
#   USE_SUDO           — "true" or "false"
#   SETUP_ROS_SETUP_PATH — path to ROS 2 setup script (e.g. /opt/ros/humble/setup.bash)
#
# Optional variables (default to the full-workspace behavior when unset):
#   SETUP_PROFILE       — "full" (default) or "inference"; gates checks whose
#                         dependencies are outside the inference profile scope
#
# Colors (imported from setup.sh context):
#   RED, GREEN, YELLOW, NC, log_info, log_warn, log_error, log_done
#
# Fail-fast: if any required variable is missing, abort immediately.

: "${WORKSPACE:?Variable WORKSPACE is not set}"
: "${VENV_PYTHON:?Variable VENV_PYTHON is not set}"
: "${SETUP_PLATFORM_ID:?Variable SETUP_PLATFORM_ID is not set}"
: "${ROSDEP_BIN:?Variable ROSDEP_BIN is not set}"
: "${USE_SUDO:?Variable USE_SUDO is not set}"
: "${SETUP_ROS_SETUP_PATH:?Variable SETUP_ROS_SETUP_PATH is not set}"

verify_ros() {
    local ros_setup="${SETUP_ROS_SETUP_PATH}"
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying ROS 2 connection..."

    if [[ -z "${ros_setup}" || ! -f "${ros_setup}" ]]; then
        log_error "ROS 2 setup script not found at: ${ros_setup:-<empty>}"
        return 1
    fi

    if (set +u; source "${ros_setup}" >/dev/null 2>&1 && set -u && "${venv_python}" -c 'import rclpy; print("ROS 2 Humble connection successful")' >/dev/null 2>&1); then
        log_info "ROS 2 verification: venv can access ROS 2 packages."
        return 0
    fi

    log_error "Verification failed: rclpy not found. Ensure ROS 2 is installed and --system-site-packages was used."
    log_error "If ${WORKSPACE}/venv was created without --system-site-packages, remove it and rerun ./scripts/setup.sh."
    return 1
}

verify_colcon() {
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying colcon..."

    if ! PYTHONNOUSERSITE=1 "${venv_python}" -m colcon --help >/dev/null 2>&1; then
        log_error "Verification failed: 'python3 -m colcon --help' does not work inside the venv."
        log_error "build.sh runs colcon this exact way; please reinstall colcon into the venv:"
        log_error "  source venv/bin/activate"
        log_error "  PYTHONNOUSERSITE=1 python3 -m pip install --upgrade colcon-common-extensions colcon-mixin"
        return 1
    fi

    if ! command -v colcon &>/dev/null; then
        log_warn "colcon is importable from the venv but no 'colcon' CLI is on PATH."
        log_warn "Activate the venv (source venv/bin/activate) before running colcon directly."
    fi

    return 0
}

verify_rosdep() {
    local rosdep_bin="${ROSDEP_BIN}"

    log_info "Verifying rosdep..."

    if ! "${rosdep_bin}" --help >/dev/null 2>&1; then
        log_error "Verification failed: ${rosdep_bin} did not respond to --help."
        log_error "  - Check that rosdep was installed into the workspace venv:"
        log_error "      ${VENV_PYTHON} -m pip show rosdep"
        log_error "  - Re-run setup with VERBOSE=1 to see the install transcript."
        return 1
    fi

    return 0
}

verify_numpy_compat() {
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying NumPy + Empy compatibility..."

    if ! PYTHONNOUSERSITE=1 "${venv_python}" - <<'PY'
import em

raise SystemExit(0 if hasattr(em, "BUFFERED_OPT") else 1)
PY
    then
        log_error "Verification failed: Empy is not ROS 2 Humble compatible."
        log_error "rosidl_adapter requires Empy 3.x (em.BUFFERED_OPT)."
        log_error "Re-run ./scripts/setup.sh to restore empy==3.3.4 in the venv."
        return 1
    fi

    if ! "${venv_python}" -c "import numpy; assert numpy.__version__.startswith('1.26.')" >/dev/null 2>&1; then
        log_error "Verification failed: NumPy is not pinned to the expected ROS-compatible 1.26.x series."
        return 1
    fi

    return 0
}

verify_lerobot() {
    local venv_python="${VENV_PYTHON}"

    log_info "Verifying lerobot..."

    if ! "${venv_python}" -c "import lerobot" >/dev/null 2>&1; then
        log_error "Verification failed: lerobot import failed."
        return 1
    fi

    return 0
}

verify_pygraphviz() {
    local venv_python="${VENV_PYTHON}"

    case "${SETUP_PLATFORM_ID}" in
        openeuler-embedded-24.03)
            log_info "Verifying pygraphviz..."

            if ! PYTHONNOUSERSITE=1 "${venv_python}" -c "import pygraphviz" >/dev/null 2>&1; then
                log_error "Verification failed: pygraphviz is not importable from the workspace venv."
                log_error "Re-run ./scripts/setup.sh after ensuring graphviz and graphviz-devel are installed."
                return 1
            fi
            ;;
    esac

    return 0
}

verify_openeuler_yaml_cpp_abi() {
    case "${SETUP_PLATFORM_ID}" in
        openeuler-embedded-24.03)
            log_info "Verifying openEuler yaml-cpp ABI..."

            if [[ ! -e /usr/lib64/libyaml-cpp.so.0.7 ]]; then
                log_error "Verification failed: missing /usr/lib64/libyaml-cpp.so.0.7."
                log_error "Re-run ./scripts/setup.sh to install yaml-cpp and yaml-cpp-devel."
                return 1
            fi
            ;;
    esac

    return 0
}

verify_tracing() {
    local venv_python="${VENV_PYTHON}"
    local ros_setup="${SETUP_ROS_SETUP_PATH}"

    # Tracing packages (lttng-ust, babeltrace2, tracetools_analysis) are pulled
    # in via robot_config's rosdep keys, which the inference profile does not
    # install; tracing is a full-workspace diagnostics feature.
    if [[ "${SETUP_PROFILE:-full}" == "inference" ]]; then
        log_info "Skipping tracing verification (inference profile)."
        return 0
    fi

    log_info "Verifying tracing tools..."

    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            if ! command -v lttng &>/dev/null; then
                log_error "Verification failed: lttng CLI is not available on PATH."
                return 1
            fi

            if ! command -v babeltrace2 &>/dev/null; then
                log_error "Verification failed: babeltrace2 CLI is not available on PATH."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && ros2 trace --help >/dev/null 2>&1); then
                log_error "Verification failed: ros2 trace CLI is not available from the ROS 2 environment."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import lttngust" >/dev/null 2>&1); then
                log_error "Verification failed: python3-lttngust is not importable from the workspace venv."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import tracetools_analysis" >/dev/null 2>&1); then
                log_error "Verification failed: tracetools-analysis is not importable from the workspace venv."
                return 1
            fi
            ;;
        openeuler-embedded-24.03)
            if ! (set +u; source "${ros_setup}" && set -u && ros2 trace --help >/dev/null 2>&1); then
                log_error "Verification failed: ros2 trace CLI is not available from the ROS 2 environment."
                return 1
            fi

            if ! command -v babeltrace &>/dev/null && ! command -v babeltrace2 &>/dev/null; then
                log_error "Verification failed: neither babeltrace nor babeltrace2 is available on PATH."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import babeltrace" >/dev/null 2>&1); then
                log_error "Verification failed: python3-babeltrace is not importable from the workspace venv."
                return 1
            fi

            if ! (set +u; source "${ros_setup}" && set -u && "${venv_python}" -c "import tracetools_analysis" >/dev/null 2>&1); then
                log_error "Verification failed: tracetools-analysis is not importable from the workspace venv."
                return 1
            fi

            if ! rpm -q lttng-ust >/dev/null 2>&1; then
                log_error "Verification failed: lttng-ust is not installed on openEuler."
                return 1
            fi

            if ! "${venv_python}" -c "import lttngust" >/dev/null 2>&1; then
                log_warn "python3-lttngust is not packaged in the current openEuler repos."
                log_warn "ROS trace CLI is available, but Python-domain ib_trace.* logging remains disabled."
            fi
            ;;
    esac

    return 0
}

verify_runtime_ros_python_bridge() {
    local ros_setup="${SETUP_ROS_SETUP_PATH}"
    local python_bin="$(command -v python3 || true)"
    
    if [[ -z "${ros_setup}" || ! -f "${ros_setup}" || -z "${python_bin}" ]]; then
        return 1
    fi

    (
        set +u
        set +e
        source "${ros_setup}" >/dev/null 2>&1
        "${python_bin}" -c 'import rclpy; print("ROS 2 Humble connection successful")'
    ) 2>/dev/null
}

verify_benchmark_minimal() {
    local venv_python="${VENV_PYTHON}"

    # Keep Setup verification as one fail-closed entry point. Detailed
    # benchmark behavior belongs to the adapter/evaluator, not this gate.
    log_info "Verifying minimal benchmark (LIBERO) environment..."

    local libero_config_dir="${WORKSPACE}/venv/ibrobot_libero"
    if [[ -f "${libero_config_dir}/config.yaml" ]]; then
        export LIBERO_CONFIG_PATH="${libero_config_dir}"
    fi

    PYTHONNOUSERSITE=1 "${venv_python}" - <<'PYBENCH' || return 1
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

workspace = Path(os.environ["WORKSPACE"]).resolve()
os.environ.setdefault("MUJOCO_GL", "osmesa")

# Load the production provider boundary directly from source. Setup runs before
# colcon has installed the benchmark adapter package.
adapter_source = workspace / "src" / "benchmark" / "adapters" / "libero"
probe_path = adapter_source / "benchmark_libero" / "version_probe.py"
probe_spec = importlib.util.spec_from_file_location("benchmark_libero_version_probe", probe_path)
if probe_spec is None or probe_spec.loader is None:
    raise RuntimeError(f"cannot load production provider probe: {probe_path}")
probe_module = importlib.util.module_from_spec(probe_spec)
sys.modules[probe_spec.name] = probe_module
probe_spec.loader.exec_module(probe_module)
identity = probe_module.probe_libero_provider()
if identity.distribution_name != "hf-libero":
    raise RuntimeError(f"unexpected provider distribution: {identity.distribution_name!r}")
if Version(identity.distribution_version) not in SpecifierSet(">=0.1.4,<0.2.0"):
    raise RuntimeError(f"unsupported hf-libero version: {identity.distribution_version}")
bddl_version = importlib.metadata.version("bddl")
if bddl_version != "1.0.1":
    raise RuntimeError(f"bddl is {bddl_version}; expected 1.0.1")
print(f"hf-libero=={identity.distribution_version} ({identity.module_path})")
print(f"bddl=={bddl_version}")
print("legacy libero distribution rejected by provider probe")

# Missing or broken runtime dependencies are Setup failures, never blockers.
import cv2
import libero
import llvmlite
import mujoco
import numba
import numpy as np
import robosuite
import torch
import torchvision
from torchcodec.decoders import VideoDecoder
from libero.libero import benchmark as libero_benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

for module in (cv2, libero, robosuite, mujoco, numba, llvmlite, torch, torchvision):
    print(f"imported {module.__name__} from {getattr(module, '__file__', '<builtin>')}")

expected_torch = os.environ["BENCHMARK_TORCH_VERSION"]
expected_torchvision = os.environ["BENCHMARK_TORCHVISION_VERSION"]
expected_torchcodec = os.environ["BENCHMARK_TORCHCODEC_VERSION"]
expected_numba = os.environ["BENCHMARK_NUMBA_VERSION"]
expected_llvmlite = os.environ["BENCHMARK_LLVM_LITE_VERSION"]
if torch.__version__ != expected_torch:
    raise RuntimeError(f"torch is {torch.__version__}; expected {expected_torch}")
if torchvision.__version__ != expected_torchvision:
    raise RuntimeError(f"torchvision is {torchvision.__version__}; expected {expected_torchvision}")
actual_torchcodec = importlib.metadata.version("torchcodec")
if actual_torchcodec != expected_torchcodec:
    raise RuntimeError(f"torchcodec is {actual_torchcodec}; expected {expected_torchcodec}")
if not callable(VideoDecoder):
    raise RuntimeError("torchcodec VideoDecoder API is unavailable")
if numba.__version__ != expected_numba:
    raise RuntimeError(f"numba is {numba.__version__}; expected {expected_numba}")
if llvmlite.__version__ != expected_llvmlite:
    raise RuntimeError(f"llvmlite is {llvmlite.__version__}; expected {expected_llvmlite}")
if not torch.cuda.is_available():
    raise RuntimeError(
        "torch-cuda is unavailable after Benchmark setup: "
        f"torch={torch.__version__}, torch_cuda={torch.version.cuda}"
    )
device = torch.device("cuda:0")
probe = torch.ones((2, 2), device=device) @ torch.ones((2, 2), device=device)
torch.cuda.synchronize(device)
if not torch.isfinite(probe).all():
    raise RuntimeError("Benchmark CUDA tensor smoke test returned non-finite values")
print(
    f"Benchmark CUDA profile: torch={torch.__version__}, torchvision={torchvision.__version__}, "
    f"torchcodec={actual_torchcodec}, numba={numba.__version__}, llvmlite={llvmlite.__version__}, "
    f"gpu={torch.cuda.get_device_name(device)}"
)

# Validate the provider API and task-0 metadata/resources used by the adapter.
suite_factory = libero_benchmark.get_benchmark_dict().get("libero_10")
if suite_factory is None:
    raise RuntimeError("provider API does not expose libero_10")
suite = suite_factory()
task_count = suite.get_num_tasks() if hasattr(suite, "get_num_tasks") else len(suite.tasks)
if task_count != 10:
    raise RuntimeError(f"libero_10 exposes {task_count} tasks; expected 10")
task = suite.get_task(0)
for field in ("problem_folder", "bddl_file", "init_states_file"):
    if not getattr(task, field, None):
        raise RuntimeError(f"task 0 metadata is missing {field}")

bddl_base = Path(get_libero_path("bddl_files")).resolve()
init_base = Path(get_libero_path("init_states")).resolve()
bddl_path = bddl_base / task.problem_folder / task.bddl_file
init_states_path = init_base / task.problem_folder / task.init_states_file
if not bddl_path.is_file():
    raise RuntimeError(f"task 0 BDDL file is missing: {bddl_path}")
if not init_states_path.is_file():
    raise RuntimeError(f"task 0 init-state file is missing: {init_states_path}")
print(f"libero_10 task_count={task_count}")
print(f"task0 bddl={bddl_path}")
print(f"task0 init_state={init_states_path}")

# Reuse the production trusted loader, including Torch 2.6+ compatibility.
loader_path = adapter_source / "benchmark_libero" / "init_state_loader.py"
loader_spec = importlib.util.spec_from_file_location("benchmark_libero_init_state_loader", loader_path)
if loader_spec is None or loader_spec.loader is None:
    raise RuntimeError(f"cannot load production init-state loader: {loader_path}")
loader_module = importlib.util.module_from_spec(loader_spec)
sys.modules[loader_spec.name] = loader_module
loader_spec.loader.exec_module(loader_module)
resolved_init_path = loader_module.resolve_init_states_path(task, get_libero_path)
init_states = loader_module.load_trusted_init_states(resolved_init_path)
if init_states is None or len(init_states) == 0:
    raise RuntimeError("task 0 has no trusted init states")

# Exercise the real off-screen provider path once, without testing evaluator
# or video writing in Setup.
env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_path),
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=256,
    camera_widths=256,
)
try:
    reset_obs = env.reset()
    if not isinstance(reset_obs, dict):
        raise RuntimeError(f"reset returned {type(reset_obs).__name__}, expected dict")
    init_obs = env.set_init_state(init_states[0])
    if not isinstance(init_obs, dict):
        raise RuntimeError(f"set_init_state returned {type(init_obs).__name__}, expected dict")
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.asarray(init_obs.get(key))
        if image.shape != (256, 256, 3):
            raise RuntimeError(f"{key} shape is {image.shape}, expected (256, 256, 3)")
        print(f"{key} shape={image.shape}")
finally:
    env.close()

# Confirm the final ABI state produced by Setup, including both OpenCV wheels.
if np.__version__ != "1.26.4":
    raise RuntimeError(f"NumPy is {np.__version__}; expected 1.26.4")
if Version(cv2.__version__) not in SpecifierSet("<4.12"):
    raise RuntimeError(f"imported cv2 is {cv2.__version__}; expected <4.12")
print(f"cv2=={cv2.__version__}")
for distribution_name in ("opencv-python-headless", "opencv-python"):
    opencv_version = importlib.metadata.version(distribution_name)
    if Version(opencv_version) not in SpecifierSet("<4.12"):
        raise RuntimeError(f"{distribution_name} is {opencv_version}; expected <4.12")
    print(f"{distribution_name}=={opencv_version}")
print(f"NumPy=={np.__version__}")

# Verify the public LeRobot policy API without constructing a model or loading
# weights in Setup.
from lerobot.configs import PreTrainedConfig
from lerobot.policies.act import ACTConfig, ACTPolicy

if not issubclass(ACTConfig, PreTrainedConfig):
    raise RuntimeError("LeRobot ACTConfig is not a PreTrainedConfig")
if ACTPolicy.config_class is not ACTConfig:
    raise RuntimeError("LeRobot ACTPolicy is not bound to ACTConfig")
if not callable(getattr(ACTPolicy, "reset", None)):
    raise RuntimeError("LeRobot ACTPolicy.reset API is unavailable")
print("LeRobot PreTrainedConfig/ACTConfig/ACTPolicy API verified")
print("MINIMAL_BENCHMARK_VERIFY_OK")
PYBENCH

    log_done "Minimal benchmark (LIBERO) environment verified"
    return 0
}

verify_env() {
    if ! platform_supports_local_workspace_build 2>/dev/null; then
        log_info "Verifying OpenHarmony ROS runtime..."
        if verify_runtime_ros_python_bridge >/dev/null 2>&1; then
            log_done "Verified OpenHarmony ROS runtime"
            return 0
        fi

        log_error "Verification failed: could not import rclpy from the OpenHarmony runtime."
        log_error "Source /data/ros2ohos.env and ensure /data/out/bin/python3.12 is available."
        return 1
    fi

    verify_ros || return 1
    verify_rosdep || return 1
    verify_colcon || return 1
    verify_numpy_compat || return 1
    verify_lerobot || return 1
    verify_pygraphviz || return 1
    verify_openeuler_yaml_cpp_abi || return 1
    verify_tracing || return 1

    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "ubuntu-22.04" ]]; then
        verify_benchmark_minimal || return 1
    fi

    log_done "Verified ROS, rosdep, colcon, lerobot, and NumPy compatibility"
}

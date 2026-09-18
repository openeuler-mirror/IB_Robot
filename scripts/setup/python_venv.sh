#!/bin/bash
# python_venv.sh - Python environment setup, dependency installation, and lerobot management

check_lerobot_python_compat() {
    local toml_path="${WORKSPACE}/libs/lerobot/pyproject.toml"
    local required_python=">=3.10"
    
    if [[ -f "${toml_path}" ]]; then
        local extracted_req
        extracted_req=$(grep -oP '^requires-python\s*=\s*"\K[^"]+' "${toml_path}" || true)
        if [[ -n "${extracted_req}" ]]; then
            required_python="${extracted_req}"
        fi
    fi

    if ! "${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
req = '${required_python}'
if not req:
    sys.exit(0)
spec = SpecifierSet(req)
if version not in spec:
    print(f'ERROR: Python {version} does not satisfy lerobot requirement {req}')
    sys.exit(1)
" 2>/dev/null; then
        local err_msg
        err_msg=$("${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
req = '${required_python}'
if not req:
    sys.exit(0)
spec = SpecifierSet(req)
if version not in spec:
    print(f'ERROR: Python {version} does not satisfy lerobot requirement {req}')
    sys.exit(1)
" 2>&1)
        log_error "${err_msg}"
        return 1
    fi
    return 0
}

check_lerobot_ros_numpy_compat() {
    if ! "${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
spec = SpecifierSet('<3.12')
if version not in spec:
    print(f'WARN: Python {version} detected. ROS 2 Humble binary packages (cv_bridge, image_transport)')
    print('WARN: are compiled against NumPy 1.x. Python 3.12+ typically requires NumPy 2.x, which')
    print('WARN: breaks ABI compatibility. C++ extensions may crash at runtime.')
" 2>/dev/null; then
        local warn_msg
        warn_msg=$("${VENV_PYTHON}" -c "
import sys
from packaging.specifiers import SpecifierSet
version = f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'
spec = SpecifierSet('<3.12')
if version not in spec:
    print(f'WARN: Python {version} detected. ROS 2 Humble binary packages (cv_bridge, image_transport)')
    print('WARN: are compiled against NumPy 1.x. Python 3.12+ typically requires NumPy 2.x, which')
    print('WARN: breaks ABI compatibility. C++ extensions may crash at runtime.')
" 2>&1)
        while IFS= read -r line; do
            log_warn "${line}"
        done <<< "${warn_msg}"
    fi
    return 0
}

_cann_version_from_file() {
    local version_file="$1"
    local line version

    [[ -r "${version_file}" ]] || return 1
    while IFS= read -r line || [[ -n "${line}" ]]; do
        case "${line}" in
            version=*)
                version="${line#version=}"
                ;;
            version_dir=*)
                version="${line#version_dir=}"
                ;;
            *_running_version=\[*:*\])
                version="${line#*:}"
                version="${version%%]*}"
                ;;
            *)
                continue
                ;;
        esac
        version="${version%$'\r'}"
        if [[ "${version}" =~ ^[0-9]+\.[0-9]+([.][A-Za-z0-9]+)*$ ]]; then
            printf '%s\n' "${version}"
            return 0
        fi
    done < "${version_file}"
    return 1
}

detect_cann_version() {
    # CANN's "latest" path can be a merged directory of component symlinks,
    # not a symlink to the release directory. Prefer its public version.cfg,
    # then cover versioned toolkit roots and component metadata.
    local toolkit_root="${IBR_CANN_TOOLKIT_ROOT:-${ASCEND_TOOLKIT_HOME:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}}"
    local version_file version resolved_root
    local version_files=(
        "${toolkit_root}/version.cfg"
        "${toolkit_root}/ascend_toolkit_install.info"
        "${toolkit_root}/aarch64-linux/ascend_toolkit_install.info"
        "${toolkit_root}/arm64-linux/ascend_toolkit_install.info"
        "${toolkit_root}/x86_64-linux/ascend_toolkit_install.info"
        "${toolkit_root}/runtime/version.info"
        "${toolkit_root}/toolkit/version.info"
    )

    for version_file in "${version_files[@]}"; do
        if version="$(_cann_version_from_file "${version_file}")"; then
            printf '%s\n' "${version}"
            return 0
        fi
    done

    resolved_root="$(readlink -f "${toolkit_root}" 2>/dev/null || true)"
    version="${resolved_root##*/}"
    if [[ "${version}" =~ ^[0-9]+\.[0-9]+([.][A-Za-z0-9]+)*$ ]]; then
        printf '%s\n' "${version}"
        return 0
    fi
    return 1
}

openeuler_requirements_for_cann() {
    local cann_version="${1:-}"

    if [[ "${cann_version}" =~ ^8[.]1([.]|$) ]]; then
        printf '%s\n' "openeuler-24.03-cann-8.1.txt"
    else
        printf '%s\n' "openeuler-24.03.txt"
    fi
}

install_openeuler_python_dependencies() {
    local pip_runner=("$@")
    local cann_version=""
    local requirements_file

    if cann_version="$(detect_cann_version)"; then
        log_info "Detected CANN ${cann_version}."
    else
        log_info "No active CANN installation detected."
    fi
    requirements_file="$(openeuler_requirements_for_cann "${cann_version}")"

    if [[ "${cann_version}" =~ ^8[.]1([.]|$) ]]; then
        log_info "Using the CANN 8.1 Torch ABI requirements."
    else
        log_info "Using the default openEuler requirements."
    fi
    run_cmd "${pip_runner[@]}" -r "${WORKSPACE}/requirements/${requirements_file}" --quiet
}

verify_cann_torch_abi() {
    local python_path="$1"
    local cann_version=""

    cann_version="$(detect_cann_version 2>/dev/null || true)"
    [[ "${cann_version}" =~ ^8[.]1([.]|$) ]] || return 0
    "${python_path}" - <<'PY'
import torch
import torch_npu
import torchvision

expected = {"torch": "2.5.1", "torch_npu": "2.5.1", "torchvision": "0.20.1"}
actual = {
    "torch": torch.__version__.split("+", maxsplit=1)[0],
    "torch_npu": torch_npu.__version__.split("+", maxsplit=1)[0],
    "torchvision": torchvision.__version__.split("+", maxsplit=1)[0],
}
if actual != expected:
    raise SystemExit(f"CANN 8.1 requires the exact Torch ABI {expected}, got {actual}")
print(f"CANN 8.1 Torch ABI verified: {actual}")
PY
}

verify_cann_runtime_versions() {
    local python_path="$1"
    local cann_version=""
    cann_version="$(detect_cann_version 2>/dev/null || true)"
    [[ "${cann_version}" =~ ^8[.]1([.]|$) ]] || return 0
    verify_cann_torch_abi "${python_path}" || return 1
    "${python_path}" - "${WORKSPACE}/requirements/constraints-cann-8.1.txt" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement

for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    requirement = Requirement(line)
    try:
        installed = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        # The GUI wheel is optional; the headless wheel is installed by compat.
        if requirement.name == "opencv-python":
            continue
        raise
    if installed not in requirement.specifier:
        raise SystemExit(f"CANN 8.1 requires {requirement}, got {installed}")
    print(f"CANN 8.1 runtime verified: {requirement.name}=={installed}")
numpy_version = importlib.metadata.version("numpy")
if numpy_version != "1.26.4":
    raise SystemExit(f"CANN 8.1 final ROS ABI requires numpy==1.26.4, got {numpy_version}")
PY
}

install_lerobot_editable() {
    local pip_runner=("$@")
    local cann_version=""

    if ! check_lerobot_python_compat; then
        log_error "Cannot install lerobot: Python version is incompatible."
        log_error "Ensure patches 0001/0002 from scripts/setup/lerobot_patches.sh are applied"
        log_error "to downgrade the requirement to >=3.10 if you are on an older system."
        exit 1
    fi

    check_lerobot_ros_numpy_compat

    cann_version="$(detect_cann_version 2>/dev/null || true)"
    if [[ "${cann_version}" =~ ^8[.]1([.]|$) ]]; then
        log_info "Installing the shared CANN 8.1-compatible LeRobot v0.6 runtime..."
        run_cmd "${pip_runner[@]}" install \
            -r "${WORKSPACE}/requirements/lerobot-v0.6-cann-8.1-compat.txt" --quiet
        if [[ "${SETUP_PROFILE:-full}" == "full" ]]; then
            log_info "Installing the CANN 8.1-compatible full-workspace LeRobot dependencies..."
            run_cmd "${pip_runner[@]}" install \
                -r "${WORKSPACE}/requirements/lerobot-v0.6-cann-8.1-extras.txt" --quiet
        fi
        log_info "Installing LeRobot editable without its incompatible upstream Torch constraints..."
        run_cmd "${pip_runner[@]}" install --no-deps -e "${WORKSPACE}/libs/lerobot"
        return 0
    fi

    # [smolvla,pi] extras pull in policy-specific deps; kinematics pulls in
    # placo for SO-101 Placo Cartesian teleop; diffusion pulls in diffusers
    # for Diffusion Policy training/inference; dataset pulls in datasets +
    # torchcodec (video decoding) for training/dataset loading; deepdiff-dep
    # supplies the deepdiff package lerobot's motors_bus needs (v0.6.0+). We
    # deliberately use deepdiff-dep rather than the feetech extra because
    # so101_hardware already provides the Python feetech-servo-sdk via its
    # setup.py install_requires, and the C++ ftservo_sdk is built by
    # so101_hardware's CMake for the ros2_control node — neither should be
    # re-installed via pip here. See libs/lerobot/pyproject.toml. LIBERO is
    # Ubuntu-only and opt-in. Remove the legacy editable ``libero``
    # distribution before installing hf-libero so the two providers never
    # coexist under the same top-level Python package.
    local lerobot_extras=(smolvla pi diffusion dataset deepdiff-dep)
    if [[ "${SETUP_PROFILE:-full}" != "inference" ]]; then
        lerobot_extras+=(kinematics)
    fi
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID:-unknown}" == "ubuntu-22.04" ]]; then
        log_info "Removing the legacy libs/libero editable provider, if installed..."
        "${pip_runner[@]}" uninstall -y libero >/dev/null 2>&1 || true
        lerobot_extras+=(libero)
        log_info "Enabling the local LeRobot libero extra for Ubuntu benchmark setup."
    fi

    local lerobot_extras_csv
    lerobot_extras_csv="$(IFS=,; printf '%s' "${lerobot_extras[*]}")"
    local constraint_args=()
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && -n "${BENCHMARK_PIP_CONSTRAINTS:-}" ]]; then
        constraint_args+=(--constraint "${BENCHMARK_PIP_CONSTRAINTS}")
    fi
    run_cmd "${pip_runner[@]}" install "${constraint_args[@]}" -e \
        "${WORKSPACE}/libs/lerobot[${lerobot_extras_csv}]"
}

install_graspgen_torch_abi() {
    # The checked-in pointnet2_ops wheel is built against this exact Torch ABI.
    # Keep CUDA hosts on their existing Torch installation so the source-build
    # path remains available, but make the no-nvcc path self-contained.
    if [[ "${SETUP_PROFILE:-full}" == "inference" ]]; then
        # The GraspGen pointnet2_ops ABI pin is a manipulation-stack concern;
        # the inference profile keeps the Torch build selected by lerobot.
        return 0
    fi
    if [[ "${SETUP_PLATFORM_ID}" != "ubuntu-22.04" ]]; then
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1 || [[ -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]]; then
        return 0
    fi

    log_info "Pinning Torch 2.7.1+cu126 for the precompiled pointnet2_ops ABI..."
    run_cmd "${VENV_PYTHON}" -m pip install --force-reinstall \
        "torch==2.7.1+cu126" "torchvision==0.22.1+cu126" \
        --index-url "https://download.pytorch.org/whl/cu126" \
        --extra-index-url "${SETUP_PIP_INDEX_URL}" --quiet
}

setup_python_venv() {
    if ! platform_supports_local_workspace_build; then
        log_info "Skipping workspace venv setup on ${SETUP_PLATFORM_ID}."
        log_info "Use the board ROS runtime directly after sourcing $(platform_ros_setup_path)."
        log_info "Cross-build RoboFrame (IB_Robot) OpenHarmony artifacts on the host with scripts/openharmony/build_roboframe_oh.sh."
        PYTHON_ENV_STATUS="skipped"
        log_skipped "Workspace Python virtual environment"
        return 0
    fi

    local venv_path="${WORKSPACE}/venv"
    local lerobot_dir="${WORKSPACE}/libs/lerobot"

    # Convenience flag for profile gating below: the inference profile skips
    # every dependency group that is not required to load policy bundles and
    # serve inference requests (hardware, teleop, perception, grasp, voice,
    # sim, dev tooling).
    local full_profile=true
    if [[ "${SETUP_PROFILE:-full}" == "inference" ]]; then
        full_profile=false
    fi

    # 0. Python interpreter preflight
    local host_python_path host_python_version host_py_major host_py_minor
    host_python_path="$(command -v python3 || true)"
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        host_python_path="${SETUP_BOOTSTRAP_PYTHON_BIN:-${host_python_path}}"
    fi
    if [[ -z "${host_python_path}" ]]; then
        log_error "No bootstrap Python was selected. Install a platform-supported Python before running setup.sh."
        exit 1
    fi
    host_python_version="$(${host_python_path} -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown")"
    log_info "Using setup bootstrap Python: ${host_python_path} (version ${host_python_version})"
    host_py_major="$(${host_python_path} -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
    host_py_minor="$(${host_python_path} -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
    if (( host_py_major < 3 )) || { (( host_py_major == 3 )) && (( host_py_minor < 10 )); }; then
        log_error "Python ${host_python_version} is too old. setup.sh requires Python >= 3.10."
        log_error "On openEuler: 'sudo dnf install -y python3.10 python3.10-devel' and re-run."
        exit 1
    fi

    # Ensure system-level venv tools are installed (done by platform scripts during install_system_deps)

    # Create the virtual environment (must include --system-site-packages to use system rclpy)
    if [[ ! -d "${venv_path}" ]]; then
        run_with_live_output "Creating virtual environment at ${venv_path} with --system-site-packages..." "${SETUP_BOOTSTRAP_PYTHON_BIN:-python3}" -m venv --system-site-packages "${venv_path}"
    else
        log_info "Virtual environment already exists at ${venv_path}."
    fi
    touch "${venv_path}/COLCON_IGNORE"

    # Activate the virtual environment and install dependencies
    log_info "Configuring Python environment and dependencies..."
    source "${venv_path}/bin/activate"

    if [[ -n "${PYTHONPATH:-}" ]]; then
        log_warn "Clearing inherited PYTHONPATH for isolated package installation inside ${venv_path}."
        unset PYTHONPATH
    fi

    if [[ -n "${PYTHONHOME:-}" ]]; then
        log_warn "Clearing inherited PYTHONHOME for isolated package installation inside ${venv_path}."
        unset PYTHONHOME
    fi

    if [[ -z "${VENV_PYTHON}" ]]; then
        log_error "No working virtual environment python was found under ${venv_path}/bin."
        exit 1
    fi

    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        local bootstrap_python_mm venv_python_mm
        bootstrap_python_mm="$(${host_python_path} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        venv_python_mm="$(${VENV_PYTHON} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "${venv_python_mm}" != "${bootstrap_python_mm}" ]]; then
            log_error "Existing workspace venv uses Python ${venv_python_mm}, but Benchmark setup requires Python ${bootstrap_python_mm}."
            log_error "Remove ${venv_path} and rerun setup; setup will not mutate a foreign-ABI venv in place."
            exit 1
        fi
    fi

    # A function-local exported constraint applies to all pip invocations and
    # build subprocesses, and is restored on return. Do not wait until the
    # platform requirements step: extras can otherwise pull a newer Torch first.
    local cann_version=""
    local -x PIP_CONSTRAINT="${PIP_CONSTRAINT:-}"
    cann_version="$(detect_cann_version 2>/dev/null || true)"
    if [[ "${cann_version}" =~ ^8[.]1([.]|$) ]]; then
        PIP_CONSTRAINT="${WORKSPACE}/requirements/constraints-cann-8.1.txt${PIP_CONSTRAINT:+ ${PIP_CONSTRAINT}}"
        log_info "Protecting the validated CANN 8.1 runtime during every dependency install."
    fi
    local pip_install=("${VENV_PYTHON}" -m pip install)
    local ros_abi_constraints="${venv_path}/ros_abi_constraints.txt"
    local ros_abi_pin_packages=("numpy==1.26.4" "opencv-python-headless<4.12")
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "ubuntu-22.04" ]]; then
        # hf-libero requires the GUI OpenCV distribution. Keep both wheels on
        # the same NumPy-1-compatible line in benchmark environments.
        ros_abi_pin_packages+=("opencv-python<4.12")
    fi

    # Upgrade pip
    run_cmd "${VENV_PYTHON}" -m pip install --upgrade pip --quiet

    # Pin and force-reinstall setuptools to a version that satisfies both
    # LeRobot (>=71,<81) and colcon-core (<80), while retaining the legacy
    # `setup.py develop --editable` option used by colcon's symlink install.
    run_cmd "${VENV_PYTHON}" -m pip install --force-reinstall "setuptools==75.8.2" --quiet

    # ------------------------------------------------------------------
    # LeRobot patch stack — MUST run before install_lerobot_editable.
    #
    # Why here: install_lerobot_editable invokes check_lerobot_python_compat
    # which reads libs/lerobot/pyproject.toml. On a fresh clone with py3.10/py3.11
    # hosts, pyproject.toml carries upstream's `requires-python>=3.12`
    # until the patch stack lowers it. Running patches first ensures
    # the compat gate reads the patched pyproject.toml.
    # ------------------------------------------------------------------
    log_info "Installing PyYAML for the lerobot patch dispatcher..."
    run_cmd "${VENV_PYTHON}" -m pip install pyyaml --quiet
    if [[ -d "${lerobot_dir}" ]]; then
        ensure_lerobot_patch_stack_applied
    fi

    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "ubuntu-22.04" ]]; then
        benchmark_prepare_torch_profile "${VENV_PYTHON}" "${venv_path}/benchmark_torch_constraints.txt"
        # Protect the validated Torch/TorchVision/TorchCodec ABI during every
        # subsequent dependency install, not only the LeRobot editable step.
        pip_install+=(--constraint "${BENCHMARK_PIP_CONSTRAINTS}")
    else
        install_graspgen_torch_abi
    fi

    # Install LeRobot in editable mode
    # Note: Do not pass the -c numpy==1.26.4 constraint. The lerobot dependency graph
    # (rerun-sdk, opencv, datasets, etc.) resolves under numpy>=2. A hard constraint
    # causes pip to fail with resolution-too-deep. We allow numpy 2.x here, then force-reinstall 1.26.4 + opencv<4.12 at the end.
    if [[ -d "${lerobot_dir}" ]]; then
        log_info "Installing LeRobot in editable mode with kinematics extra..."
        install_lerobot_editable "${VENV_PYTHON}" -m pip
    fi

    # Install base Python dependencies
    log_info "Installing base Python dependencies..."
    run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/base.txt" --quiet

    # Install hardware dependencies
    if [[ "${full_profile}" == true ]]; then
        log_info "Installing hardware dependencies..."
        run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/hardware.txt" --quiet
    else
        log_info "Skipping hardware dependencies (inference profile)."
    fi

    # Install the optional built-in WebPhone dependency
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping WebPhone teleoperation dependency (inference profile)."
    elif [[ "${AUTO_YES}" == true ]]; then
        log_info "Auto-yes mode: installing WebPhone dependency (websockets)..."
        run_cmd "${pip_install[@]}" websockets --quiet
        log_done "Phone teleoperation dependency installed (websockets)"
    else
        echo ""
        echo "  WebPhone teleoperation (optional):"
        echo "    1) Install websockets (browser WebXR AR + optical-flow fallback)"
        echo "    0) Skip WebPhone"
        echo ""
        while true; do
            read -r -p "  Enter your choice [0-1]: " PHONE_CHOICE
            case "${PHONE_CHOICE}" in
                1)
                    run_cmd "${pip_install[@]}" websockets --quiet
                    log_done "Phone dependencies installed: websockets (WebPhone)"
                    break
                    ;;
                0)
                    log_info "Skipping WebPhone teleoperation dependency."
                    break
                    ;;
                *)
                    echo "  Invalid choice. Please enter 0 or 1."
                    ;;
            esac
        done
    fi

    # Install development and training tools
    if [[ "${full_profile}" == true ]]; then
        log_info "Installing dev-tools (tensorboard, rerun, gitlint, ruff, pre-commit)..."
        run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/dev-tools.txt" --quiet
    else
        log_info "Skipping dev-tools (inference profile)."
    fi

    # Install platform-specific dependencies (ONNX tooling, etc.)
    case "${SETUP_PLATFORM_ID}" in
        ubuntu-22.04)
            if [[ "${full_profile}" == true ]]; then
                run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/ubuntu-22.04.txt" --quiet
            else
                # Inference profile: only the ONNX toolchain; skip mujoco (sim)
                # and atomgit_sdk (collaboration CLI).
                log_info "Installing inference ONNX tooling (inference profile)..."
                run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/inference.txt" --quiet
            fi
            ;;
        openeuler-embedded-24.03)
            # Installed whole in both profiles: onnx/onnxruntime (ONNX policy
            # path), torch_npu (Ascend NPU inference), and pygraphviz (required
            # by verify_env on this platform) are all inference-relevant.
            install_openeuler_python_dependencies "${pip_install[@]}"
            ;;
    esac

    if [[ "${full_profile}" == true ]] && [[ -f "${WORKSPACE}/.pre-commit-config.yaml" ]]; then
        "${VENV_PYTHON}" -m pre_commit install
    fi

    # ------------------------------------------------------------------
    # Environment Patches & Overrides
    # ------------------------------------------------------------------
    
    log_info "Installing colcon-common-extensions + colcon-mixin into the workspace venv..."
    run_cmd "${pip_install[@]}" --ignore-installed --upgrade colcon-common-extensions colcon-mixin --quiet

    if ! PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import setuptools.command.develop as develop

raise SystemExit(
    0 if hasattr(develop, "develop") and hasattr(develop.develop, "install_for_development")
    else 1
)
PY
    then
        log_warn "colcon symlink installation will fail because setuptools>=71 removed setup.py develop."
        log_warn "Please ensure setuptools is downgraded or wait for colcon-core updates."
    fi

    log_info "Pinning Empy 3.3.4 for ROS 2 Humble rosidl compatibility..."
    run_cmd "${pip_install[@]}" --force-reinstall "empy==3.3.4" --quiet

    # rosdep was already installed into this same venv by the early
    # ensure_workspace_venv + ensure_rosdep step. Re-running pip install
    # here is a no-op when the package is current, and acts as a safety net
    # in case the venv was recreated between the two steps.
    log_info "Ensuring rosdep is present in the workspace venv..."
    run_cmd "${pip_install[@]}" rosdep --quiet

    # Force NumPy/OpenCV back to ROS 2 Humble ABI-compatible versions.
    # The lerobot installation brings in numpy 2.x. We unconditionally overwrite it
    # here to ensure ROS packages (cv_bridge, image_transport, etc.) do not trigger binary incompatibility errors at runtime.
    # Only install the headless OpenCV wheel by default; keep opencv-python in
    # the constraints file below so optional dependencies cannot pull 4.12+ and
    # force NumPy 2.x back into the ROS environment.
    log_info "Pinning NumPy/OpenCV to the ROS 2 Humble ABI-compatible versions..."
    run_cmd "${pip_install[@]}" --force-reinstall "${ros_abi_pin_packages[@]}" --quiet
    benchmark_install_runtime_abi "${VENV_PYTHON}" -m pip install

    cat > "${ros_abi_constraints}" <<'EOF'
numpy==1.26.4
opencv-python<4.12
opencv-python-headless<4.12
EOF
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        cat "${BENCHMARK_PIP_CONSTRAINTS}" >> "${ros_abi_constraints}"
    fi

    # Install the ZipVoice frontend after creating the ROS ABI constraints.
    # Vocos itself is maintained in voice_tts_service.vocos_backend because
    # the PyPI package can replace the workspace Torch/torchaudio ABI.
    if [[ "${full_profile}" == true ]]; then
        log_info "Installing voice TTS frontend dependencies..."
        run_cmd "${pip_install[@]}" --constraint "${ros_abi_constraints}" \
            -r "${WORKSPACE}/requirements/voice-tts.txt" --quiet
    else
        log_info "Skipping voice TTS frontend dependencies (inference profile)."
    fi

    # Perception runtime dependencies (SAM2, Grounding-DINO, RAM++, SigLIP2) are
    # part of the default install contract: perception_service and
    # semantic_mapping already build in the default workspace, and the audited
    # RAM++ / GroundingDINO wheels ship in third_party/. Install them on every
    # platform that runs the local workspace build, including openEuler Embedded
    # (the Ascend OM path does not replace the Torch perception runtime).
    # The inference profile skips them: edge-cloud inference verification uses
    # the perception ROS mock instead of the real perception stack.
    local ram_wheel_root="${WORKSPACE}/third_party/wheels/recognize-anything/7cb804a"
    local ram_wheel="${ram_wheel_root}/ibrobot_ram-0.0.1+ibrobot.1-py3-none-any.whl"
    local gdino_wheel_root="${WORKSPACE}/third_party/wheels/groundingdino/313392a"
    local gdino_wheel="${gdino_wheel_root}/ibrobot_groundingdino-0.1.0+ibrobot.1-py3-none-any.whl"
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping perception dependencies (inference profile)."
    else
        log_info "Installing perception dependencies (SAM2, Grounding-DINO, RAM++, SigLIP2)..."
        run_cmd env SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}" SAM2_BUILD_ALLOW_ERRORS=1 \
            "${pip_install[@]}" --no-build-isolation --constraint "${ros_abi_constraints}" \
            -r "${WORKSPACE}/requirements/perception.txt" --quiet
        if ! (cd "${ram_wheel_root}" && sha256sum --check SHA256SUMS); then
            log_error "RAM++ wheel checksum verification failed."
            exit 1
        fi
        run_cmd "${pip_install[@]}" --no-deps "${ram_wheel}" --quiet
        if ! (cd "${gdino_wheel_root}" && sha256sum --check SHA256SUMS); then
            log_error "GroundingDINO wheel checksum verification failed."
            exit 1
        fi
        run_cmd "${pip_install[@]}" --no-deps "${gdino_wheel}" --quiet
    fi

    # FullSubNet speech enhancement model (audio_zen + model.py) as an audited
    # wheel. Pure-Python; the Torch backend loads Model from the installed
    # package instead of cloning the upstream source tree into models/.
    local fullsubnet_wheel_root="${WORKSPACE}/third_party/wheels/fullsubnet/e97448375"
    local fullsubnet_wheel="${fullsubnet_wheel_root}/ibrobot_fullsubnet-0.0.1+ibrobot.1-py3-none-any.whl"
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping FullSubNet wheel (inference profile)."
    elif [[ -f "${fullsubnet_wheel_root}/SHA256SUMS" ]]; then
        if ! (cd "${fullsubnet_wheel_root}" && sha256sum --check SHA256SUMS); then
            log_error "FullSubNet wheel checksum verification failed."
            exit 1
        fi
        run_cmd "${pip_install[@]}" --no-deps "${fullsubnet_wheel}" --quiet
    else
        log_warn "FullSubNet wheel not found at ${fullsubnet_wheel_root}; skipping (speech_direction Torch backend unavailable)."
    fi

    # GraspGen runtime dependencies are part of the default install contract:
    # manipulation_service is part of the default workspace build. Skip on
    # openEuler Embedded because GraspGen's pointnet2_ops CUDA extension is
    # validated on Ubuntu only; the Ascend OM path does not cover Torch grasp.
    # On Ubuntu hosts without nvcc, install_graspgen_pip falls back to the
    # audited pointnet2_ops wheel and validates its Torch/Python ABI contract.
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping grasp dependencies (inference profile)."
    elif [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        log_info "Skipping GraspGen/manipulation dependencies for Benchmark-only setup."
    elif [[ "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_warn "Skipping grasp dependencies on openEuler; GraspGen CUDA extensions are validated on Ubuntu only."
    else
        log_info "Installing grasp dependencies (GraspGen)..."
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/setup/install_graspgen_pip.sh"
        export ROS_ABI_CONSTRAINTS="${ros_abi_constraints}"
        install_graspgen_pip "${VENV_PYTHON}" -m pip install
    fi

    # Optional speech_direction report dependencies (Plotly/Matplotlib). Pure-Python.
    # Only the `speech_direction_report` offline CLI imports these; the runtime node
    # does not, so the default install (without this flag) stays lightweight.
    if [[ "${INSTALL_DIAGNOSTICS_DEPS:-false}" == true ]]; then
        log_info "Installing optional speech_direction report dependencies (Plotly, Matplotlib)..."
        run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/diagnostics.txt" --quiet
        installed_diagnostics_deps=true
    else
        log_info "Skipping optional speech_direction report dependencies. Re-run setup with --with-diagnostics if needed."
    fi

    # ------------------------------------------------------------------
    # Benchmark (LIBERO) optional resource configuration.
    #
    # The conditional LeRobot libero extra above installs the sole hf-libero
    # provider. This layer only prepares the workspace-owned config/assets;
    # provider identity and runtime API checks belong to the LIBERO adapter.
    # ------------------------------------------------------------------
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        if [[ "${SETUP_PLATFORM_ID}" != "ubuntu-22.04" ]]; then
            log_warn "Skipping optional benchmark dependencies on ${SETUP_PLATFORM_ID}; LIBERO/MuJoCo are validated on Ubuntu only."
        else
            log_info "Configuring optional benchmark (LIBERO) runtime..."
            export WORKSPACE="${WORKSPACE}"

            # Generate workspace-owned config and fetch/reuse official assets
            # before importing hf-libero. This prevents the provider's legacy
            # first-import prompt and makes get_assets_path() deterministic.
            local libero_config_script="${SCRIPT_DIR}/setup/libero_config.py"
            local libero_config_dir="${venv_path}/ibrobot_libero"
            log_info "Preparing workspace-owned hf-libero config and assets..."
            if ! PYTHONNOUSERSITE=1 "${VENV_PYTHON}" "${libero_config_script}" \
                > "${venv_path}/benchmark_libero_config_output.txt" 2>&1; then
                log_error "Failed to prepare workspace-owned hf-libero config/assets."
                cat "${venv_path}/benchmark_libero_config_output.txt" 2>/dev/null || true
                exit 1
            fi
            export LIBERO_CONFIG_PATH="${libero_config_dir}"
            log_info "hf-libero config/assets ready at ${libero_config_dir}"
        fi
    fi

    # Optional perception/grasp dependencies can pull OpenCV wheels whose latest
    # releases require NumPy 2.x. Re-apply the final ROS 2 ABI pin before smoke tests.
    log_info "Re-applying NumPy/OpenCV ROS 2 ABI pins after optional dependencies..."
    run_cmd "${pip_install[@]}" --force-reinstall "${ros_abi_pin_packages[@]}" --quiet

    log_info "Running NumPy/OpenCV dependency smoke test..."
    PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import cv2
import numpy

if not numpy.__version__.startswith("1.26"):
    raise SystemExit(f"Expected NumPy 1.26.x after setup, got {numpy.__version__}")
print(f"NumPy/OpenCV smoke test passed: numpy={numpy.__version__}, cv2={cv2.__version__}")
PY
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" != true && "${full_profile}" == true && "${SETUP_PLATFORM_ID}" != "openeuler-embedded-24.03" ]]; then
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import importlib

try:
    importlib.import_module("grasp_gen")
except Exception as exc:
    raise SystemExit(f"Missing grasp_gen after default grasp install: {exc}")
print("Grasp dependencies smoke test passed")
PY
    fi
    # gitlint ships in dev-tools.txt, which the inference profile skips.
    if [[ "${full_profile}" != true ]]; then
        log_info "Skipping gitlint commit-msg hook (inference profile)."
    else
        local commit_msg_hook
        commit_msg_hook="$(git rev-parse --git-path hooks/commit-msg 2>/dev/null || true)"
        if [[ -f "${commit_msg_hook}" ]] && grep -qi "gitlint" "${commit_msg_hook}"; then
            log_warn "gitlint commit-msg hook already exists at ${commit_msg_hook}; keeping it."
        else
            log_info "Installing gitlint commit-msg hook..."
            # gitlint is installed in the venv, which is currently activated
            printf 'y\n' | gitlint install-hook || log_warn "Failed to install gitlint hook"
        fi
    fi

    # Venv summary: print the key facts users need to debug "wrong python /
    # wrong colcon" issues without having to source the venv themselves.
    local venv_numpy_ver="unknown" venv_colcon_path="missing" venv_py_ver="unknown"
    venv_py_ver="$(PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)"
    venv_numpy_ver="$(PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo unknown)"
    venv_colcon_path="$(PYTHONNOUSERSITE=1 "${VENV_PYTHON}" -c 'import colcon, os; print(os.path.dirname(colcon.__file__))' 2>/dev/null || echo missing)"

    # User-site inspection: even though build.sh sets PYTHONNOUSERSITE=1 and
    # we install colcon into the venv, a stale ~/.local/lib/.../colcon left
    # by an old system-wide pip install can still shadow the venv colcon when
    # users run `colcon` interactively outside of build.sh. We explicitly
    # suppress the user-site in build.sh to guarantee isolation, but a user's
    # system-level colcon would silently no-op. We surface this state in
    # the summary and warn explicitly when colcon shadows are detected.
    local user_site user_site_status="not-present" user_site_colcon=""
    user_site="$("${VENV_PYTHON}" -m site --user-site 2>/dev/null || true)"
    if [[ -n "${user_site}" && -d "${user_site}" ]]; then
        local user_pkg_count
        user_pkg_count="$(find "${user_site}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
        user_site_status="active (${user_pkg_count} packages)"
        if find "${user_site}" -maxdepth 1 -name "colcon*" | grep -q .; then
            user_site_colcon="DETECTED"
        fi
    fi

    echo -e "\n${YELLOW}Python Environment Summary:${NC}"
    echo "  Python Version:     ${venv_py_ver} (${VENV_PYTHON})"
    echo "  NumPy Version:      ${venv_numpy_ver} (Target: 1.26.x for ROS 2 ABI)"
    echo "  Colcon Path:        ${venv_colcon_path}"
    echo "  User Site-Packages: ${user_site_status} (${user_site})"

    if [[ -n "${user_site_colcon}" ]]; then
        log_warn "User site-packages contain a colcon installation that may shadow the workspace venv."
        log_warn "If 'colcon build' fails outside of build.sh, consider clearing the user site colcon:"
        log_warn "    rm -rf ${user_site}/colcon* ${user_site%/lib/*}/bin/colcon*"
    fi

    # Optional full-profile installs run after the platform dependencies.
    # Validate at the very end so no later pip transaction can invalidate it.
    if [[ "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        verify_cann_runtime_versions "${VENV_PYTHON}" || return 1
    fi
    PYTHON_ENV_STATUS="done"

}

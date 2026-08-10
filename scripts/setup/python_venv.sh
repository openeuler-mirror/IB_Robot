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

install_lerobot_editable() {
    local pip_runner=("$@")
    
    if ! check_lerobot_python_compat; then
        log_error "Cannot install lerobot: Python version is incompatible."
        log_error "Ensure patches 0001/0002 from scripts/setup/lerobot_patches.sh are applied"
        log_error "to downgrade the requirement to >=3.10 if you are on an older system."
        exit 1
    fi

    check_lerobot_ros_numpy_compat

    # [smolvla,pi] extras pull in policy-specific deps; kinematics pulls in
    # placo for SO-101 Placo Cartesian teleop; diffusion pulls in diffusers
    # for Diffusion Policy training/inference; dataset pulls in datasets +
    # torchcodec (video decoding) for training/dataset loading; deepdiff-dep
    # supplies the deepdiff package lerobot's motors_bus needs (v0.6.0+). We
    # deliberately use deepdiff-dep rather than the feetech extra because
    # so101_hardware already provides the Python feetech-servo-sdk via its
    # setup.py install_requires, and the C++ ftservo_sdk is built by
    # so101_hardware's CMake for the ros2_control node — neither should be
    # re-installed via pip here. See libs/lerobot/pyproject.toml.
    #
    # The inference profile keeps diffusion/dataset (policy inference and
    # video decoding for the streamed observation path) but drops kinematics,
    # which only serves teleop.
    local lerobot_extras="smolvla,pi,kinematics,diffusion,dataset,deepdiff-dep"
    if [[ "${SETUP_PROFILE:-full}" == "inference" ]]; then
        lerobot_extras="smolvla,pi,diffusion,dataset,deepdiff-dep"
    fi
    "${pip_runner[@]}" install -e "${WORKSPACE}/libs/lerobot[${lerobot_extras}]"
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

    # Runtime identity: Outer identity protection via production helpers.
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/setup/outer_identity_guard.sh"
    local outer_identity_script="${SCRIPT_DIR}/setup/torch_identity.py"
    # Pass executable/arguments only. outer_identity_guard injects
    # PYTHONNOUSERSITE=1 while executing the collector.
    local outer_collector_cmd=("${venv_path}/bin/python3" "${outer_identity_script}")

    outer_identity_begin "${venv_path}" "${INSTALL_BENCHMARK_DEPS:-false}" "${outer_collector_cmd[@]}" >&2 || {
        log_error "Outer identity begin failed."
        exit 1
    }
    if [[ "${OUTER_IDENTITY_MODE}" == "protected" ]]; then
        log_info "Outer inference identity protection: protected-existing-runtime mode."
    elif [[ "${OUTER_IDENTITY_MODE}" == "fresh" ]]; then
        log_info "Outer inference identity protection skipped: no pre-existing workspace venv/runtime."
    fi

    # 0. Python interpreter preflight
    local host_python_path host_python_version host_py_major host_py_minor
    host_python_path="$(command -v python3 || true)"
    if [[ -z "${host_python_path}" ]]; then
        log_error "python3 not found on PATH. Install python3 (>=3.10) before running setup.sh."
        exit 1
    fi
    host_python_version="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown")"
    log_info "Using host python3: ${host_python_path} (version ${host_python_version})"
    host_py_major="$(python3 -c 'import sys; print(sys.version_info[0])' 2>/dev/null || echo 0)"
    host_py_minor="$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
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

    local pip_install=("${VENV_PYTHON}" -m pip install)
    local installed_benchmark_deps=false
    local ros_abi_constraints="${venv_path}/ros_abi_constraints.txt"

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

    install_graspgen_torch_abi

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
            run_cmd "${pip_install[@]}" -r "${WORKSPACE}/requirements/openeuler-24.03.txt" --quiet
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
    log_info "Pinning NumPy 1.26.4 + opencv-python-headless<4.12 (ROS 2 Humble ABI)..."
    run_cmd "${pip_install[@]}" --force-reinstall "numpy==1.26.4" \
        "opencv-python-headless<4.12" --quiet

    cat > "${ros_abi_constraints}" <<'EOF'
numpy==1.26.4
opencv-python<4.12
opencv-python-headless<4.12
EOF

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
    # Benchmark (LIBERO) optional dependency layer (benchmark setup).
    #
    # Installs the pinned libs/libero provider and the curated LIBERO
    # runtime deps (robosuite, bddl, gym, imageio, etc.) into the
    # workspace venv. Does NOT install torch/torchvision/robomimic.
    # Records torch/python/torchvision identity before and after;
    # fails-closed if the inference runtime was changed.
    # ------------------------------------------------------------------
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true && "${SETUP_PLATFORM_ID}" == "openeuler-embedded-24.03" ]]; then
        log_warn "Skipping optional benchmark dependencies on openEuler; LIBERO/MuJoCo are validated on Ubuntu only."
    elif [[ "${INSTALL_BENCHMARK_DEPS:-false}" == true ]]; then
        log_info "Installing optional benchmark (LIBERO) dependencies..."
        export WORKSPACE="${WORKSPACE}"

        local benchmark_constraints="${WORKSPACE}/constraints/benchmark.txt"
        local benchmark_req="${WORKSPACE}/requirements/benchmark.txt"
        local benchmark_combined_constraints="${venv_path}/benchmark_combined_constraints.txt"

        # Combine ros_abi_constraints + benchmark constraints into one file
        # so a single --constraint flag covers both.
        cat "${ros_abi_constraints}" > "${benchmark_combined_constraints}"
        if [[ -f "${benchmark_constraints}" ]]; then
            cat "${benchmark_constraints}" >> "${benchmark_combined_constraints}"
        fi

        # 1. Record inference runtime identity BEFORE benchmark install.
        #    Uses the shared torch_identity.py collector to avoid drift
        #    between before/after snapshots. Fail-closed: non-zero exit
        #    from the collector aborts setup (torch is a required component;
        #    installed-but-broken must not be masked as "not installed").
        local torch_identity_script="${SCRIPT_DIR}/setup/torch_identity.py"
        local torch_before_file="${venv_path}/benchmark_torch_identity_before.txt"
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" "${torch_identity_script}" > "${torch_before_file}"
        local torch_before_rc=$?
        if [[ ${torch_before_rc} -ne 0 ]]; then
            log_error "Failed to record torch identity (before). Setup aborts (fail-closed)."
            cat "${torch_before_file}" 2>/dev/null || true
            exit 1
        fi
        log_info "Benchmark torch identity (before) recorded."

        # 2. Verify libs/libero submodule gitlink/HEAD/dirty before install.
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/setup/benchmark_guard.sh"
        if ! verify_libero_gitlink "${WORKSPACE}"; then
            log_error "libs/libero gitlink verification failed. Aborting before editable install."
            exit 1
        fi
        log_info "libs/libero gitlink verified."

        # Workspace-owned LIBERO configuration: Generate workspace-owned LIBERO config BEFORE any import.
        # Without this, the pinned provider calls input() on fresh HOME,
        # which fails non-interactively with EOFError.
        local libero_config_script="${SCRIPT_DIR}/setup/libero_config.py"
        local libero_config_dir="${venv_path}/ibrobot_libero"
        log_info "Generating workspace-owned LIBERO config..."
        if ! PYTHONNOUSERSITE=1 "${VENV_PYTHON}" "${libero_config_script}" \
            > "${venv_path}/benchmark_libero_config_output.txt" 2>&1; then
            log_error "Failed to generate workspace-owned LIBERO config."
            cat "${venv_path}/benchmark_libero_config_output.txt" 2>/dev/null || true
            exit 1
        fi
        export LIBERO_CONFIG_PATH="${libero_config_dir}"
        log_info "LIBERO config generated at ${libero_config_dir}/config.yaml"

        # 3. Install the pinned libs/libero provider in editable mode
        #    with --no-deps so its legacy requirements.txt does not pull
        #    in robomimic/transformers/wandb (which would pull torch).
        local libero_dir_local="${WORKSPACE}/libs/libero"
        log_info "Installing pinned libs/libero (editable compat mode, --no-deps)..."
        # Editable compatibility: Use editable_mode=compat to generate a .pth file that
        # points to libs/libero directly, instead of the default PEP 660
        # finder which produces an empty MAPPING for this package layout.
        run_cmd "${pip_install[@]}" --no-build-isolation --no-deps \
            --config-settings editable_mode=compat \
            -e "${libero_dir_local}" --quiet

        # 4. Install the curated benchmark requirements.
        log_info "Installing benchmark requirements..."
        run_cmd "${pip_install[@]}" --constraint "${benchmark_combined_constraints}" \
            -r "${benchmark_req}" --quiet

        # 5. Record inference runtime identity AFTER benchmark install.
        #    Uses the same shared collector as step 1.
        local torch_after_file="${venv_path}/benchmark_torch_identity_after.txt"
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" "${torch_identity_script}" > "${torch_after_file}"
        local torch_after_rc=$?
        if [[ ${torch_after_rc} -ne 0 ]]; then
            log_error "Failed to record torch identity (after). Setup aborts (fail-closed)."
            cat "${torch_after_file}" 2>/dev/null || true
            exit 1
        fi
        log_info "Benchmark torch identity (after) recorded."

        # 6. Compare before/after; fail-closed on ANY identity field change.
        #    Every field is compared, including torch_file, torchvision_file,
        #    python_version, build suffix, CUDA runtime, and device capability.
        local torch_before_summary torch_after_summary
        torch_before_summary="$(cat "${torch_before_file}" 2>/dev/null)"
        torch_after_summary="$(cat "${torch_after_file}" 2>/dev/null)"
        if [[ "${torch_before_summary}" != "${torch_after_summary}" ]]; then
            log_error "Benchmark dependency installation changed the inference runtime identity."
            log_error "This is a fail-closed condition. The benchmark stack must not modify torch/torchvision."
            log_error "--- BEFORE ---"
            cat "${torch_before_file}" 2>/dev/null || true
            log_error "--- AFTER ---"
            cat "${torch_after_file}" 2>/dev/null || true
            log_error "--- DIFF ---"
            diff "${torch_before_file}" "${torch_after_file}" 2>/dev/null || true
            exit 1
        fi
        log_done "Benchmark dependencies installed; torch/torchvision identity unchanged (all fields compared)"

        # 7. Record provider identity for the benchmark setup report.
        # Provider identity: Use unified standalone collector to avoid contract drift.
        local provider_identity_script="${SCRIPT_DIR}/setup/provider_identity.py"
        local provider_identity_file="${venv_path}/benchmark_provider_identity.txt"
        local provider_rc=0
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" "${provider_identity_script}" \
            > "${provider_identity_file}" 2>&1 || provider_rc=$?
        if [[ ${provider_rc} -ne 0 ]]; then
            log_error "Provider identity collector failed (rc=${provider_rc})."
            cat "${provider_identity_file}" 2>/dev/null || true
            exit 1
        fi
        if ! grep -q "PROVIDER_IDENTITY_OK" "${provider_identity_file}" 2>/dev/null; then
            log_error "Provider identity verification failed (no PROVIDER_IDENTITY_OK marker)."
            cat "${provider_identity_file}" 2>/dev/null || true
            exit 1
        fi
        log_done "Provider identity verified"

        installed_benchmark_deps=true
    else
        log_info "Skipping optional benchmark dependencies. Re-run setup with --with-benchmark if needed."
    fi

    # Optional perception/grasp dependencies can pull OpenCV wheels whose latest
    # releases require NumPy 2.x. Re-apply the final ROS 2 ABI pin before smoke tests.
    log_info "Re-applying NumPy/OpenCV ROS 2 ABI pins after optional dependencies..."
    run_cmd "${pip_install[@]}" --force-reinstall "numpy==1.26.4" \
        "opencv-python-headless<4.12" --quiet

    log_info "Running NumPy/OpenCV dependency smoke test..."
    PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import cv2
import numpy

if not numpy.__version__.startswith("1.26"):
    raise SystemExit(f"Expected NumPy 1.26.x after setup, got {numpy.__version__}")
print(f"NumPy/OpenCV smoke test passed: numpy={numpy.__version__}, cv2={cv2.__version__}")
PY
    if [[ "${full_profile}" == true ]] && [[ "${SETUP_PLATFORM_ID}" != "openeuler-embedded-24.03" ]]; then
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import importlib

try:
    importlib.import_module("grasp_gen")
except Exception as exc:
    raise SystemExit(f"Missing grasp_gen after default grasp install: {exc}")
print("Grasp dependencies smoke test passed")
PY
    fi
    if [[ "${installed_benchmark_deps}" == true ]]; then
        log_info "Running benchmark (LIBERO) import smoke test..."
        PYTHONNOUSERSITE=1 "${VENV_PYTHON}" - <<'PY'
import importlib

required = [
    "libero",
    "libero.libero",
    "robosuite",
    "mujoco",
    "imageio",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Benchmark import smoke test failed:\n  " + "\n  ".join(missing))
print("Benchmark (LIBERO) import smoke test passed")
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

    PYTHON_ENV_STATUS="done"

    # Runtime identity: Finalize outer identity via production helper.
    local outer_finalize_collector=("${VENV_PYTHON}" "${outer_identity_script}")
    outer_identity_finalize "${venv_path}" "${INSTALL_BENCHMARK_DEPS:-false}" "${outer_finalize_collector[@]}" >&2 || {
        log_error "Outer inference identity comparison failed."
        exit 1
    }
    if [[ "${OUTER_IDENTITY_MODE}" == "protected" ]]; then
        log_done "Outer inference identity unchanged (full setup scope)"
    fi
}

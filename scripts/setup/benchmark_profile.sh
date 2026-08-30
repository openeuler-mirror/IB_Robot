#!/bin/bash
# benchmark_profile.sh - PR309 Benchmark-only installation profile.
#
# This file is sourced by setup.sh.  It deliberately owns only the optional
# Benchmark dependency boundary; it does not change the default IB-Robot
# manipulation/CUDA installation path.

BENCHMARK_TORCH_VERSION="2.7.1+cu126"
BENCHMARK_TORCHVISION_VERSION="0.22.1+cu126"
BENCHMARK_TORCH_CUDA_TAG="cu126"
BENCHMARK_TORCHCODEC_VERSION="0.5"
BENCHMARK_NUMBA_VERSION="0.59.1"
BENCHMARK_LLVM_LITE_VERSION="0.42.0"
BENCHMARK_MIN_DRIVER_VERSION="560.28.03"
export BENCHMARK_TORCH_VERSION BENCHMARK_TORCHVISION_VERSION BENCHMARK_TORCH_CUDA_TAG
export BENCHMARK_TORCHCODEC_VERSION BENCHMARK_NUMBA_VERSION BENCHMARK_LLVM_LITE_VERSION BENCHMARK_MIN_DRIVER_VERSION

benchmark_version_at_least() {
    local actual="$1"
    local required="$2"
    awk -v actual="${actual}" -v required="${required}" 'BEGIN {
        split(actual, a, "."); split(required, r, ".")
        for (i = 1; i <= 3; i++) {
            av = (a[i] == "" ? 0 : a[i]) + 0
            rv = (r[i] == "" ? 0 : r[i]) + 0
            if (av > rv) exit 0
            if (av < rv) exit 1
        }
        exit 0
    }'
}

benchmark_detect_driver_cuda() {
    nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n1
}

benchmark_preflight_host() {
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" != true ]]; then
        return 0
    fi

    if [[ "${SETUP_PLATFORM_ID:-unknown}" != "ubuntu-22.04" ]]; then
        log_error "PR309 Benchmark currently supports Ubuntu 22.04 only; detected ${SETUP_PLATFORM_ID}."
        log_error "Do not continue with a partial Benchmark installation on this platform."
        return 1
    fi
    if [[ "${SETUP_ARCH:-$(uname -m)}" != "x86_64" ]]; then
        log_error "PR309 Benchmark CUDA wheels are validated on Ubuntu 22.04 x86_64 only; detected ${SETUP_ARCH:-unknown}."
        log_error "Do not continue with an unvalidated wheel architecture."
        return 1
    fi

    local bootstrap_version="${SETUP_BOOTSTRAP_PYTHON_VERSION:-unknown}"
    if [[ "${bootstrap_version}" != 3.10.* ]]; then
        log_error "Benchmark setup requires the Ubuntu 22.04 Python 3.10 bootstrap interpreter."
        log_error "Detected: ${SETUP_BOOTSTRAP_PYTHON_BIN:-unknown} (${bootstrap_version})"
        return 1
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "Benchmark setup requires a usable NVIDIA driver (nvidia-smi was not found)."
        log_error "The local CUDA Toolkit/nvcc is not required, but the NVIDIA driver is required for torch-cuda."
        return 1
    fi

    local gpu_name driver_version driver_cuda
    IFS=',' read -r gpu_name driver_version < <(
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits 2>/dev/null | head -n1
    )
    gpu_name="${gpu_name# }"
    driver_version="${driver_version// /}"
    driver_cuda="$(benchmark_detect_driver_cuda)"
    if [[ -z "${gpu_name}" || -z "${driver_version}" ]]; then
        log_error "Benchmark setup could not read the NVIDIA GPU and driver from nvidia-smi."
        log_error "The torch-cuda deployment cannot be validated safely."
        return 1
    fi
    if ! benchmark_version_at_least "${driver_version}" "${BENCHMARK_MIN_DRIVER_VERSION}"; then
        log_error "NVIDIA driver ${driver_version} is older than the validated CUDA 12.6 runtime minimum ${BENCHMARK_MIN_DRIVER_VERSION}."
        log_error "Validated Benchmark profile: torch==${BENCHMARK_TORCH_VERSION}, torchvision==${BENCHMARK_TORCHVISION_VERSION}."
        log_error "Upgrade the NVIDIA driver or use a separately validated Benchmark profile."
        return 1
    fi
    log_info "Benchmark host preflight: GPU=${gpu_name}; driver=${driver_version}; driver CUDA=${driver_cuda:-unknown}; local nvcc is not required."
    log_info "Benchmark Torch profile: torch==${BENCHMARK_TORCH_VERSION}; torchvision==${BENCHMARK_TORCHVISION_VERSION}."
}

benchmark_prepare_torch_profile() {
    local venv_python="$1"
    local constraints_file="$2"

    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" != true ]]; then
        return 0
    fi

    local installed_versions
    installed_versions="$(${venv_python} - <<'PY' 2>/dev/null || true
try:
    import torch
    import torchvision
    print(torch.__version__)
    print(torchvision.__version__)
except Exception:
    pass
PY
)"
    local expected_versions="${BENCHMARK_TORCH_VERSION}
${BENCHMARK_TORCHVISION_VERSION}"
    if [[ "${installed_versions}" != "${expected_versions}" ]]; then
        log_info "Installing the validated Benchmark PyTorch CUDA profile..."
        run_cmd "${venv_python}" -m pip install --force-reinstall \
            "torch==${BENCHMARK_TORCH_VERSION}" \
            "torchvision==${BENCHMARK_TORCHVISION_VERSION}" \
            --index-url "https://download.pytorch.org/whl/${BENCHMARK_TORCH_CUDA_TAG}" \
            --extra-index-url "${SETUP_PIP_INDEX_URL}" --quiet
    else
        log_info "Validated Benchmark PyTorch CUDA profile is already installed."
    fi

    cat > "${constraints_file}" <<EOF_CONSTRAINTS
# Generated by --with-benchmark; customers do not edit this file.
torch==${BENCHMARK_TORCH_VERSION}
torchvision==${BENCHMARK_TORCHVISION_VERSION}
torchcodec==${BENCHMARK_TORCHCODEC_VERSION}
EOF_CONSTRAINTS
    export BENCHMARK_PIP_CONSTRAINTS="${constraints_file}"

    PYTHONNOUSERSITE=1 "${venv_python}" - <<PY
import torch
import torchvision

expected_torch = "${BENCHMARK_TORCH_VERSION}"
expected_torchvision = "${BENCHMARK_TORCHVISION_VERSION}"
if torch.__version__ != expected_torch:
    raise SystemExit(f"Benchmark Torch profile mismatch: {torch.__version__} != {expected_torch}")
if torchvision.__version__ != expected_torchvision:
    raise SystemExit(f"Benchmark TorchVision profile mismatch: {torchvision.__version__} != {expected_torchvision}")
if not torch.cuda.is_available():
    raise SystemExit(
        "Benchmark torch-cuda preflight failed: torch.cuda.is_available() is false "
        f"(torch={torch.__version__}, torch_cuda={torch.version.cuda})"
    )
device = torch.device("cuda:0")
probe = torch.ones((2, 2), device=device) @ torch.ones((2, 2), device=device)
torch.cuda.synchronize(device)
if probe.shape != (2, 2) or not torch.isfinite(probe).all():
    raise SystemExit("Benchmark CUDA tensor smoke test returned an invalid result")
print(
    "Benchmark Torch profile verified: "
    f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
    f"cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(device)}"
)
PY
}

benchmark_install_runtime_abi() {
    local pip_runner=("$@")
    if [[ "${INSTALL_BENCHMARK_DEPS:-false}" != true ]]; then
        return 0
    fi
    log_info "Installing the validated Benchmark runtime ABI (TorchCodec/Numba/llvmlite)..."
    # --no-deps is intentional: Torch and the ROS-compatible NumPy are already
    # installed and verified. Re-resolving here could silently replace either
    # side of the ABI contract with an incompatible release.
    run_cmd "${pip_runner[@]}" --force-reinstall --no-deps \
        "torchcodec==${BENCHMARK_TORCHCODEC_VERSION}" \
        "numba==${BENCHMARK_NUMBA_VERSION}" \
        "llvmlite==${BENCHMARK_LLVM_LITE_VERSION}" --quiet
}

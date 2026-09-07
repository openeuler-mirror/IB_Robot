#!/bin/bash
# download_speech_direction_models.sh - 下载 Ubuntu CUDA 环境的 speech_direction 模型依赖
#
# 职责：只负责 Ubuntu 环境依赖（均来自 GitHub，不入库，放 models/ 下）：
#   1. Silero VAD ONNX    — models/silero-vad/assets/silero_vad.onnx (~2.3MB)
#                          官方 snakers4/silero-vad master 分支，与 voice_asr_service 共用
#   2. FullSubNet ckpt    — models/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar (~67MB)
#                          Audio-WestlakeU/FullSubNet v0.2 release，cumulative 218epochs
#
# FullSubNet 源码（model.py + audio_zen）已打包为 ibrobot-fullsubnet wheel，
# 通过 scripts/setup.sh 安装到 venv；本脚本不再 clone 上游源码仓。
#
# 310P 专用 OM 资产（FB/SB/manifest 等）已发布到 HuggingFace openEuler/fullsubnet，
# 可通过 huggingface_hub.snapshot_download 拉取；本脚本只管 Ubuntu CUDA 依赖。
#
# Usage:
#   ./scripts/download_speech_direction_models.sh                 # 下载全部
#   ./scripts/download_speech_direction_models.sh --silero-only   # 仅 Silero VAD
#   ./scripts/download_speech_direction_models.sh --fullsubnet-only # 仅 FullSubNet ckpt
#   ./scripts/download_speech_direction_models.sh --target /custom/path  # 自定义 models 根目录
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_MODELS_DIR="${WORKSPACE}/models"
MODELS_DIR="${DEFAULT_MODELS_DIR}"

SILERO_ONLY=false
FULLSUBNET_ONLY=false

# Silero VAD ONNX（官方 snakers4/silero-vad master 分支；v6 ONNX 不在 release 附件，只在该路径）。
# raw.githubusercontent 在部分网络不稳，jsdelivr CDN 镜像更稳。
SILERO_URL="https://cdn.jsdelivr.net/gh/snakers4/silero-vad@master/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_SIZE=2327524

# FullSubNet cumulative ckpt（Audio-WestlakeU/FullSubNet v0.2 release，218epochs cumulative）。
# 必须与 cumulative_laplace_norm 配对，不能用于 offline_laplace_norm。
# FullSubNet 源码（model.py + audio_zen）由 ibrobot-fullsubnet wheel 提供，不在本脚本范围。
FULLSUBNET_CKPT_URL="https://github.com/Audio-WestlakeU/FullSubNet/releases/download/v0.2/cum_fullsubnet_best_model_218epochs.tar"
FULLSUBNET_CKPT_SHA256="d08d09107eb276b8dc3d2d9fff995f4354a51fa3347125f52f8b9aea7c339f81"
FULLSUBNET_CKPT_SIZE=67667419

usage() {
    cat <<'EOF'
Download Ubuntu CUDA model dependencies for speech_direction.

Usage:
  ./scripts/download_speech_direction_models.sh [OPTIONS]

Options:
  --silero-only         Download Silero VAD ONNX only
  --fullsubnet-only     Download FullSubNet ckpt only
  --target DIR          Target models directory (default: models/)
  -h, --help            Show the help

Models downloaded (all from GitHub, for Ubuntu CUDA):
  - Silero VAD ONNX (~2.3 MB, shared with voice_asr_service)
  - FullSubNet cumulative ckpt 218epochs (~67 MB)

FullSubNet source (model.py + audio_zen) is installed as ibrobot-fullsubnet wheel
via scripts/setup.sh. 310P OM assets: openEuler/fullsubnet on HuggingFace.
Models already present and verified are skipped.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --silero-only) SILERO_ONLY=true ;;
            --fullsubnet-only) FULLSUBNET_ONLY=true ;;
            --target)
                shift
                if [[ $# -eq 0 ]]; then echo "Error: --target requires a path"; exit 1; fi
                MODELS_DIR="$1"
                ;;
            -h|--help) usage; exit 0 ;;
            *) echo "Unknown argument: $1"; usage; exit 1 ;;
        esac
        shift
    done
}

# 校验文件大小和 SHA-256，与 manifest 契约对齐。
verify_file() {
    local path="$1"
    local expected_size="$2"
    local expected_sha="$3"
    local desc="$4"

    if [[ ! -f "${path}" ]]; then
        echo "[error] ${desc} 下载后不存在: ${path}"
        return 1
    fi
    local actual_size
    actual_size=$(stat -c %s "${path}")
    if [[ "${actual_size}" != "${expected_size}" ]]; then
        echo "[error] ${desc} size 不匹配: 实际=${actual_size} 期望=${expected_size}"
        return 1
    fi
    local actual_sha
    actual_sha=$(sha256sum "${path}" | cut -d' ' -f1)
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
        echo "[error] ${desc} SHA-256 不匹配: 实际=${actual_sha} 期望=${expected_sha}"
        return 1
    fi
    echo "[verify] ${desc} size+SHA-256 OK"
}

# 下载单个文件（已存在且校验通过则跳过），带大小和 SHA-256 校验。
download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"
    local expected_size="$4"
    local expected_sha="$5"

    # 已存在且校验通过则跳过。
    if [[ -f "${dest}" ]] && verify_file "${dest}" "${expected_size}" "${expected_sha}" "${desc}" 2>/dev/null; then
        echo "[skip] ${desc} 已存在且校验通过: ${dest}"
        return 0
    fi

    echo "[download] ${desc} -> ${dest}"
    mkdir -p "$(dirname "${dest}")"
    local tmp="${dest}.part"
    rm -f "${tmp}"
    if command -v curl &>/dev/null; then
        curl -fL --progress-bar -o "${tmp}" "${url}"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "${tmp}" "${url}"
    else
        echo "Error: curl or wget required"
        rm -f "${tmp}"
        return 1
    fi

    # 下载后校验，通过才替换目标文件。
    if ! verify_file "${tmp}" "${expected_size}" "${expected_sha}" "${desc}"; then
        rm -f "${tmp}"
        return 1
    fi
    mv "${tmp}" "${dest}"
    echo "[done] ${desc}"
}

download_silero() {
    local dest="${MODELS_DIR}/silero-vad/assets/silero_vad.onnx"
    download_file "${SILERO_URL}" "${dest}" "Silero VAD ONNX" "${SILERO_SIZE}" "${SILERO_SHA256}"
    package_silero_bundle
}

# 独立 Silero VAD bundle（models/silero-vad）：多业务共享的同源部署。
# 打包器按 #378 的 v3 typed API 构造 manifest；缺少 310P OM 时仅生成 torch_cpu 部署。
package_silero_bundle() {
    local bundle_root="${MODELS_DIR}/silero-vad"
    if [[ ! -s "${bundle_root}/assets/silero_vad.onnx" ]]; then
        echo "[skip] silero-vad bundle packaging (ONNX not downloaded)"
        return 0
    fi
    local python_bin="${PYTHON:-python3}"
    echo "[package] refreshing silero-vad bundle manifest"
    PYTHONPATH="${SCRIPT_DIR}/../src/inference_manifest:${SCRIPT_DIR}/../src/voice_asr_service${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_bin}" -m voice_asr_service.package_silero_vad_bundle \
        --bundle-root "${bundle_root}" --workspace "${SCRIPT_DIR}/.."
}

download_fullnet_ckpt() {
    # 独立 fullsubnet bundle（models/fullsubnet）的 torch 部署权重；
    # 310P OM 与 cum manifest 已随 bundle 发布（openEuler/fullsubnet）。
    local dest="${MODELS_DIR}/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar"
    download_file "${FULLSUBNET_CKPT_URL}" "${dest}" "FullSubNet ckpt 218epochs (~67MB)" \
        "${FULLSUBNET_CKPT_SIZE}" "${FULLSUBNET_CKPT_SHA256}"
}

main() {
    parse_args "$@"

    local download_silero=true
    local download_fullsubnet=true

    if [[ "${SILERO_ONLY}" == true ]]; then download_fullsubnet=false; fi
    if [[ "${FULLSUBNET_ONLY}" == true ]]; then download_silero=false; fi

    mkdir -p "${MODELS_DIR}"
    echo "Models directory: ${MODELS_DIR}"
    echo ""

    if [[ "${download_silero}" == true ]]; then
        download_silero
    fi

    if [[ "${download_fullsubnet}" == true ]]; then
        download_fullnet_ckpt
    fi

    echo ""
    echo "Speech direction Ubuntu dependencies setup complete. Directory: ${MODELS_DIR}"
    if [[ "${download_silero}" == true ]]; then
        echo "  Silero VAD:      ${MODELS_DIR}/silero-vad/assets/silero_vad.onnx"
    fi
    if [[ "${download_fullsubnet}" == true ]]; then
        echo "  FullSubNet ckpt: ${MODELS_DIR}/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar"
    fi
    echo ""
    echo "Note: FullSubNet source (model.py + audio_zen) is installed as ibrobot-fullsubnet"
    echo "wheel via scripts/setup.sh. 310P OM assets: openEuler/fullsubnet on HuggingFace."
}

main "$@"

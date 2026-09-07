#!/usr/bin/env bash
# Download ZipVoice-Distill ONNX models for Ubuntu deployment.
#
# Downloads text_encoder.onnx + fm_decoder.onnx from ModelScope
# (k2-fsa/ZipVoice) and reuses tokens/Vocos/prompt assets from the
# 310P bundle when available.  Also generates the zipvoice_onnx.json
# runtime config and merges the ubuntu_onnx deployment into
# inference_manifest.json so the bundle is ready for `deployment:=ubuntu_onnx`.
#
# Usage:
#   ./scripts/download_voice_tts_models.sh [--bundle-dir DIR]
#
# Defaults:
#   BUNDLE_DIR = $WORKSPACE/models/zipvoice
#
# The manifest merge preserves an existing bundle uuid and any existing
# deployments (e.g. ascend_310p).  Run this script in the same bundle
# directory that already contains a 310P bundle to make both deployments
# available; run it in a fresh directory to create an ONNX-only bundle.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
BUNDLE_DIR="${WORKSPACE}/models/zipvoice"

while [ $# -gt 0 ]; do
    case "$1" in
        --bundle-dir)
            [ $# -ge 2 ] || { echo "error: --bundle-dir requires a value" >&2; exit 1; }
            BUNDLE_DIR="$2"
            shift 2
            ;;
        --bundle-dir=*)
            BUNDLE_DIR="${1#--bundle-dir=}"
            shift
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            echo "usage: $0 [--bundle-dir DIR]" >&2
            exit 1
            ;;
    esac
done

MODELSCOPE_BASE="https://www.modelscope.cn/models/k2-fsa/ZipVoice/resolve/master/zipvoice_distill"

ONNX_DIR="${BUNDLE_DIR}/artifacts/onnx"
ASSETS_DIR="${BUNDLE_DIR}/assets"
mkdir -p "${ONNX_DIR}" "${ASSETS_DIR}/vocos" "${ASSETS_DIR}/prompts"

# Expected SHA-256 digests for downloaded assets.  Values recorded from the
# official k2-fsa/ZipVoice ModelScope release; re-verify after any upstream
# bump.  Mirrors scripts/download_speech_direction_models.sh convention.
TEXT_ENCODER_SHA256="495eca2d5f8a911f5c361bcce5bd55cdd2508ccdd26ce3e9bf1d3c29eb974861"
FM_DECODER_SHA256="4510d4f5f049f14ef80207fca695e13c820e2cea61635f402954950bc62b1e3c"
TOKENS_SHA256="ce98c1afc5f7a20c2484dffdd68a1fff0a4a2cc707328833750c4476c37cdbda"
VOCOS_SHA256="97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a"
PROMPT_SHA256="706be5f309ef8e76a323a0640b288d91c6e2a62902e9a343b0e1bde507f125f1"

verify_file() {
    local path="$1" expected_sha="$2" desc="$3"
    local actual_sha actual_size
    actual_sha=$(sha256sum "${path}" | cut -d' ' -f1)
    actual_size=$(stat -c%s "${path}" 2>/dev/null || stat -f%z "${path}")
    if [ -n "${expected_sha}" ]; then
        if [ "${actual_sha}" != "${expected_sha}" ]; then
            echo "[error] ${desc} SHA-256 mismatch: actual=${actual_sha} expected=${expected_sha}" >&2
            exit 1
        fi
        echo "[verify] ${desc} size=${actual_size} SHA-256 OK"
    else
        echo "[info] ${desc} size=${actual_size} SHA-256=${actual_sha}"
        echo "       record this value in the script to enable verification on future runs"
    fi
}

download_and_verify() {
    local url="$1" dest="$2" expected_sha="$3" desc="$4"
    if [ ! -f "${dest}" ]; then
        echo "  downloading ${desc}..."
        curl -fL -o "${dest}" "${url}"
    fi
    verify_file "${dest}" "${expected_sha}" "${desc}"
}

echo "[1/7] Downloading text_encoder.onnx..."
download_and_verify "${MODELSCOPE_BASE}/text_encoder.onnx" "${ONNX_DIR}/text_encoder.onnx" "${TEXT_ENCODER_SHA256}" "text_encoder.onnx"

echo "[2/7] Downloading fm_decoder.onnx..."
download_and_verify "${MODELSCOPE_BASE}/fm_decoder.onnx" "${ONNX_DIR}/fm_decoder.onnx" "${FM_DECODER_SHA256}" "fm_decoder.onnx"

echo "[3/7] Downloading tokens.txt..."
download_and_verify "${MODELSCOPE_BASE}/tokens.txt" "${ASSETS_DIR}/tokens.txt" "${TOKENS_SHA256}" "tokens.txt"

echo "[4/7] Downloading Vocos checkpoint..."
download_and_verify "${MODELSCOPE_BASE}/vocos/pytorch_model.bin" "${ASSETS_DIR}/vocos/pytorch_model.bin" "${VOCOS_SHA256}" "vocos/pytorch_model.bin"

echo "[5/7] Downloading default prompt profile..."
download_and_verify "${MODELSCOPE_BASE}/prompts/default.npz" "${ASSETS_DIR}/prompts/default.npz" "${PROMPT_SHA256}" "prompts/default.npz"

echo "[6/7] Generating zipvoice_onnx.json..."
cat > "${ASSETS_DIR}/zipvoice_onnx.json" <<'JSON'
{
  "text_encoder_path": "artifacts/onnx/text_encoder.onnx",
  "fm_decoder_path": "artifacts/onnx/fm_decoder.onnx",
  "tokens_path": "assets/tokens.txt",
  "vocos_checkpoint_path": "assets/vocos/pytorch_model.bin",
  "prompt_profiles": {"default": "assets/prompts/default.npz"},
  "text_capacity": 256,
  "num_steps": 8,
  "t_shift": 0.5,
  "guidance_scale": 3.0,
  "speed": 1.0,
  "feature_scale": 0.1,
  "sample_rate": 24000,
  "cross_fade_sec": 0.1,
  "seed": 42,
  "inter_op_num_threads": 4,
  "intra_op_num_threads": 4,
  "providers": ["CPUExecutionProvider"]
}
JSON

echo "[7/7] Merging ubuntu_onnx deployment into inference_manifest.json..."
python3 - "$BUNDLE_DIR" <<'PY'
import hashlib
import json
import sys
import uuid
from pathlib import Path

from inference_manifest import BundleFile, TorchDeployment, canonical_bundle_digest, load_inference_manifest

bundle_dir = Path(sys.argv[1])
manifest_path = bundle_dir / "inference_manifest.json"

UBUNTU_ONNX_DEPLOYMENT_NAME = "ubuntu_onnx"
BUNDLE_NAME_ONNX_ONLY = "zipvoice-distill-ubuntu-onnx"

ONNX_ARTIFACTS = {
    "text_encoder": {"format": "onnx", "path": "artifacts/onnx/text_encoder.onnx"},
    "flow_decoder": {"format": "onnx", "path": "artifacts/onnx/fm_decoder.onnx"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


for _spec in ONNX_ARTIFACTS.values():
    _spec["sha256"] = _sha256(bundle_dir / _spec["path"])
ONNX_BINDINGS = {
    "text_encoder": {
        "inputs": [
            {"semantic": "host.zipvoice.tokens", "dtype": "int64", "index": 0, "shape": [1, 256]},
            {"semantic": "host.zipvoice.tokens_len", "dtype": "int64", "index": 1, "shape": []},
            {"semantic": "host.zipvoice.prompt_tokens", "dtype": "int64", "index": 2, "shape": [1, 29]},
            {"semantic": "host.zipvoice.prompt_features_len", "dtype": "int64", "index": 3, "shape": []},
            {"semantic": "host.zipvoice.speed", "dtype": "float32", "index": 4, "shape": []},
        ],
        "outputs": [
            {"semantic": "host.zipvoice.text_condition", "dtype": "float32", "index": 0, "shape": [-1, 100]},
            {"semantic": "host.zipvoice.features_len", "dtype": "int64", "index": 1, "shape": []},
            {"semantic": "host.zipvoice.padding_mask", "dtype": "float32", "index": 2, "shape": [-1]},
        ],
    },
    "flow_decoder": {
        "inputs": [
            {"semantic": "host.zipvoice.t", "dtype": "float32", "index": 0, "shape": []},
            {"semantic": "host.zipvoice.flow_x", "dtype": "float32", "index": 1, "shape": [-1, 100]},
            {"semantic": "host.zipvoice.flow_text_condition", "dtype": "float32", "index": 2, "shape": [-1, 100]},
            {"semantic": "host.zipvoice.speech_condition", "dtype": "float32", "index": 3, "shape": [-1, 100]},
            {"semantic": "host.zipvoice.flow_padding_mask", "dtype": "float32", "index": 4, "shape": [-1]},
            {"semantic": "host.zipvoice.guidance_scale", "dtype": "float32", "index": 5, "shape": []},
        ],
        "outputs": [
            {"semantic": "host.zipvoice.velocity", "dtype": "float32", "index": 0, "shape": [-1, 100]},
        ],
    },
}

model_block = {
    "interface": "tensor_model",
    "model_type": "zipvoice",
    "operation": "synthesize",
    "inputs": [
        {"semantic": "tts.text", "dtype": "uint8", "shape": [-1]},
        {"semantic": "tts.prompt_audio", "dtype": "float32", "shape": [-1]},
        {"semantic": "tts.prompt_sample_rate", "dtype": "int64", "shape": []},
        {"semantic": "tts.prompt_text", "dtype": "uint8", "shape": [-1]},
    ],
    "outputs": [{"semantic": "tts.audio", "dtype": "float32", "shape": [-1]}],
    "semantic_identity": {
        "logical_model_revision": "zipvoice-distill-onnx-dynamic-2026-08-15",
        "preprocessing_contract": "emilia-zh-cn2an-jieba-pypinyin-fixed-golden-prompt-v1",
        "output_semantics": "mono-float32-pcm-24000hz-vocos-cpu",
    },
}

if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise RuntimeError("existing ZipVoice manifest is not schema v3; repackage it before merging")
    bundle = manifest.setdefault("bundle", {})
    bundle_uuid = bundle.get("uuid", str(uuid.uuid4()))
    bundle_revision = bundle.get("revision", 1)
    bundle_name = bundle.get("name", BUNDLE_NAME_ONNX_ONLY)
    manifest["schema_version"] = 3
    manifest["model"] = model_block
    deployments = manifest.setdefault("deployments", {})
    preserved = list(deployments.keys() - {UBUNTU_ONNX_DEPLOYMENT_NAME})
    print(f"merging into existing manifest (preserved deployments: {preserved or 'none'})")
else:
    bundle_uuid = str(uuid.uuid4())
    bundle_revision = 1
    bundle_name = BUNDLE_NAME_ONNX_ONLY
    manifest = {"schema_version": 3, "model": model_block}
    manifest["deployments"] = {}
    print("creating new manifest (ubuntu_onnx only)")

# The schema forbids overlap between bundle.files and deployment artifacts, so
# every path declared by any deployment artifact (existing Ascend OMs included)
# must stay out of the structural file list.
declared_artifacts = {
    spec["path"]
    for deployment in manifest["deployments"].values()
    for spec in (deployment.get("artifacts") or {}).values()
    if isinstance(spec, dict) and spec.get("path")
}
declared_artifacts |= {spec["path"] for spec in ONNX_ARTIFACTS.values()}

file_paths = sorted(
    p.relative_to(bundle_dir).as_posix()
    for sub in ("assets", "artifacts")
    for p in bundle_dir.joinpath(sub).rglob("*")
    if p.is_file() and ".cache" not in p.parts and p.relative_to(bundle_dir).as_posix() not in declared_artifacts
)

manifest["deployments"][UBUNTU_ONNX_DEPLOYMENT_NAME] = {
    "uuid": str(uuid.uuid4()),
    "revision": 1,
    "artifacts": ONNX_ARTIFACTS,
    "bindings": ONNX_BINDINGS,
    "execution": ["text_encoder", "flow_decoder"],
    "execution_contract": {
        "state_scope": "request",
        "execution_structure": "iterative",
        "orchestration_visibility": "session",
        "cancellation_granularity": "checkpoint",
    },
    "runtime_profile": {
        "backend": "torch",
        "target": {"runtime": "torch"},
        "profile": {"device": "cpu"},
    },
    # Synthesis-only audio contract, mirroring the Ascend deployments: the
    # plugin's validate_synthesis_audio_contract checks it fail-closed.
    "audio_contract": {
        "sample_rate_hz": 24000,
        "channels": 1,
        "channel_semantics": "mono",
        "sample_dtype": "float32",
        "execution_mode": "offline",
    },
}

manifest["bundle"] = {
    "uuid": bundle_uuid,
    "revision": bundle_revision,
    "name": bundle_name,
    "files": [{"path": f} for f in file_paths],
    "digest": {
        "algorithm": "sha256",
        "scope": "structure",
        "value": canonical_bundle_digest(bundle_uuid, bundle_revision, bundle_name, [BundleFile(path=p) for p in file_paths]),
    },
}

manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
for deployment_name in manifest["deployments"]:
    load_inference_manifest(bundle_dir, deployment_name)
print(f"manifest written to {manifest_path}")
print(f"bundle uuid: {bundle_uuid}")
print(f"bundle digest: {manifest['bundle']['digest']['value']}")
print(f"deployments: {sorted(manifest['deployments'])}")
PY

echo "Done. Models and bundle metadata generated in ${BUNDLE_DIR}"

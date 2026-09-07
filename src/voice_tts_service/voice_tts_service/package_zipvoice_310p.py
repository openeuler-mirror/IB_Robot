"""Package the verified external ZipVoice 310P delivery as an IB-Robot bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np

from inference_manifest import BundleFile, canonical_bundle_digest, write_inference_manifest

TEXT_OM = "output/om_bucket/text_encoder_t256_p29_f3072.om"
FLOW_OM = "output/om_bucket/fm_decoder_f1537_mask_noif_mixed_linux_aarch64.om"
TOKENS = "models/onnx/tokens.txt"
VOCOS_CHECKPOINT = "models/vocos/pytorch_model.bin"
TEXT_GOLDEN = "golden/text_encoder.npz"
FLOW_GOLDEN = "golden/fm_decoder_step0.npz"
EXPECTED_SHA256 = {
    TEXT_OM: "31a7aa1c06d06abbb9e73d566cc63b38b25b987f13bafeee63f01241a979d287",
    FLOW_OM: "22be611d858e667e0a0d39111fa44c772f2fbdb7ff0b0ea55d708f0a469f4c6c",
    TOKENS: "ce98c1afc5f7a20c2484dffdd68a1fff0a4a2cc707328833750c4476c37cdbda",
    VOCOS_CHECKPOINT: "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
    TEXT_GOLDEN: "0cee022aae56433d57ccf79d53ca936cd046d20eb755bd04660e0019aa117c52",
    FLOW_GOLDEN: "94996ac2d0c78f79abd4dd51c7c22daf9398f8e7f82ef2fb96f398d987375920",
}


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _require_source(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required ZipVoice source asset is unavailable: {path}")
    expected = EXPECTED_SHA256[relative]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(
            f"ZipVoice source asset checksum mismatch for {relative}: expected {expected}, actual {digest.hexdigest()}"
        )
    return path


def _binding(semantic: str, name: str, index: int, dtype: str, shape: list[int]) -> dict[str, object]:
    return {
        "semantic": semantic,
        "runtime_name": name,
        "index": index,
        "dtype": dtype,
        "shape": shape,
    }


def _build_manifest(destination: Path, bundle_uuid: str, deployment_uuid: str) -> dict[str, object]:
    files = tuple(
        BundleFile(path=path.relative_to(destination).as_posix())
        for path in sorted(destination.joinpath("assets").rglob("*"))
        if path.is_file()
    )
    bundle_name = "zipvoice-distill-310p1-fixed-prompt"
    model_inputs = [
        {"semantic": "tts.text", "dtype": "uint8", "shape": [-1]},
        {"semantic": "tts.prompt_audio", "dtype": "float32", "shape": [-1]},
        {"semantic": "tts.prompt_sample_rate", "dtype": "int64", "shape": []},
        {"semantic": "tts.prompt_text", "dtype": "uint8", "shape": [-1]},
    ]
    text_inputs = [
        _binding("host.zipvoice.tokens", "tokens", 0, "int64", [1, 256]),
        _binding("host.zipvoice.tokens_len", "tokens_len", 1, "int64", []),
        _binding("host.zipvoice.prompt_tokens", "prompt_tokens", 2, "int64", [1, 29]),
        _binding("host.zipvoice.prompt_features_len", "prompt_features_len", 3, "int64", []),
        _binding("host.zipvoice.speed", "speed", 4, "float32", []),
    ]
    text_outputs = [
        _binding("host.zipvoice.text_condition", "/Where_5:0:text_condition", 0, "float32", [1, 3072, 100]),
        _binding("host.zipvoice.features_len", "/Reshape_7:0:features_len", 1, "int64", []),
        _binding("host.zipvoice.padding_mask", "/Unsqueeze_3:0:padding_mask", 2, "bool", [1, 3072]),
    ]
    flow_inputs = [
        _binding("host.zipvoice.t", "t", 0, "float32", []),
        _binding("host.zipvoice.flow_x", "x", 1, "float32", [1, 1537, 100]),
        _binding("host.zipvoice.flow_text_condition", "text_condition", 2, "float32", [1, 1537, 100]),
        _binding("host.zipvoice.speech_condition", "speech_condition", 3, "float32", [1, 1537, 100]),
        _binding("host.zipvoice.flow_padding_mask", "padding_mask", 4, "bool", [1, 1537]),
        _binding("host.zipvoice.guidance_scale", "guidance_scale", 5, "float32", []),
    ]
    flow_outputs = [
        _binding(
            "host.zipvoice.velocity",
            "PartitionedCall_/fm_decoder/Transpose_1_Transpose_2643:0:v",
            0,
            "float32",
            [1, 1537, 100],
        )
    ]
    return {
        "schema_version": 3,
        "bundle": {
            "uuid": bundle_uuid,
            "revision": 1,
            "name": bundle_name,
            "files": [entry.model_dump(mode="json") for entry in files],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(bundle_uuid, 1, bundle_name, files),
            },
        },
        "model": {
            "interface": "tensor_model",
            "model_type": "zipvoice",
            "operation": "synthesize",
            "inputs": model_inputs,
            "outputs": [{"semantic": "tts.audio", "dtype": "float32", "shape": [-1]}],
            "semantic_identity": {
                "logical_model_revision": "zipvoice-distill-310p1-bucket-2026-08-03",
                "preprocessing_contract": "emilia-zh-cn2an-jieba-pypinyin-fixed-golden-prompt-v1",
                "output_semantics": "mono-float32-pcm-24000hz-vocos-cpu",
            },
        },
        "deployments": {
            "ascend_310p": {
                "uuid": deployment_uuid,
                "revision": 1,
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "iterative",
                    "orchestration_visibility": "session",
                    "cancellation_granularity": "checkpoint",
                },
                "runtime_profile": {
                    "backend": "ascend",
                    "target": {
                        "soc": "Ascend310P1",
                        "runtime": "acl",
                        "runtime_abi": "cann-8.1.RC1",
                    },
                    "profile": {
                        "device_id": 0,
                    },
                },
                "artifacts": {
                    "text_encoder": {
                        "path": "artifacts/ascend/ascend_310p/text_encoder_t256_p29_f3072.om",
                        "format": "om",
                    },
                    "flow_decoder_1537": {
                        "path": "artifacts/ascend/ascend_310p/fm_decoder_f1537_mask_noif_mixed.om",
                        "format": "om",
                    },
                },
                "execution": ["text_encoder", "flow_decoder_1537"],
                "bindings": {
                    "text_encoder": {"inputs": text_inputs, "outputs": text_outputs},
                    "flow_decoder_1537": {"inputs": flow_inputs, "outputs": flow_outputs},
                },
                "device_links": [],
                # Synthesis-only audio contract: the deployment consumes text and
                # emits mono float32 PCM at 24 kHz. It declares no microphone
                # input frame/chunk constraints because it has no waveform input.
                "audio_contract": {
                    "sample_rate_hz": 24000,
                    "channels": 1,
                    "channel_semantics": "mono",
                    "sample_dtype": "float32",
                    "execution_mode": "offline",
                },
            }
        },
    }


def package_bundle(source: Path, destination: Path) -> Path:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination must be empty for a new immutable bundle: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _copy(
        _require_source(source, TEXT_OM),
        destination / "artifacts/ascend/ascend_310p/text_encoder_t256_p29_f3072.om",
    )
    _copy(
        _require_source(source, FLOW_OM),
        destination / "artifacts/ascend/ascend_310p/fm_decoder_f1537_mask_noif_mixed.om",
    )
    _copy(_require_source(source, TOKENS), destination / "assets/tokens.txt")
    _copy(_require_source(source, VOCOS_CHECKPOINT), destination / "assets/vocos/pytorch_model.bin")

    with np.load(_require_source(source, TEXT_GOLDEN), allow_pickle=False) as text_fixture:
        prompt_tokens = np.asarray(text_fixture["prompt_tokens"], dtype=np.int64)
        prompt_frames = int(np.asarray(text_fixture["prompt_features_len"]).reshape(()))
    with np.load(_require_source(source, FLOW_GOLDEN), allow_pickle=False) as flow_fixture:
        speech_condition = np.asarray(flow_fixture["speech_condition"], dtype=np.float32)
    prompt_features = speech_condition[:, :prompt_frames, :]
    prompt_path = destination / "assets/prompts/default.npz"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(prompt_path, prompt_tokens=prompt_tokens, prompt_features=prompt_features)

    runtime = {
        "text_role": "text_encoder",
        "flow_role": "flow_decoder_1537",
        "tokens_path": "assets/tokens.txt",
        "vocos_checkpoint_path": "assets/vocos/pytorch_model.bin",
        "prompt_profiles": {"default": "assets/prompts/default.npz"},
        "text_capacity": 256,
        "flow_frames": 1537,
        "num_steps": 4,
        "t_shift": 0.5,
        "guidance_scale": 3.0,
        "speed": 1.0,
        "feature_scale": 0.1,
        "sample_rate": 24000,
        "cross_fade_sec": 0.1,
        "seed": 42,
        "request_prompt_supported": False,
    }
    (destination / "assets/zipvoice_310p.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    manifest = _build_manifest(destination, str(uuid4()), str(uuid4()))
    return write_inference_manifest(destination / "inference_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="ZipVoice delivery root")
    parser.add_argument("--destination", required=True, type=Path, help="new bundle directory")
    args = parser.parse_args()
    manifest = package_bundle(args.source, args.destination)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

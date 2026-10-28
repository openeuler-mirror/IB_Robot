from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from inference_manifest import BundleFile, canonical_bundle_digest

TEST_BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
TEST_DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"

POLICY_FEATURES = {
    "input_features": {
        "observation.state": {"type": "STATE", "shape": [6]},
        "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [6]}},
}


def v3_runtime_deployment(value: dict[str, Any], *, default_backend: str = "torch") -> dict[str, Any]:
    """Normalize a test deployment into the strict v3 runtime envelope."""

    deployment = copy.deepcopy(value)
    if "runtime_profile" in deployment:
        return deployment
    backend = deployment.pop("backend", default_backend)
    target = deployment.pop("target", {})
    device = deployment.pop("device", "cpu")
    runtime = target.get(
        "runtime",
        {
            "ascend": "acl",
            "hisilicon": "hisilicon-worker",
            "hmm": "tcim",
            "rknn": "rknn-lite",
        }.get(backend, "torch"),
    )
    profile: dict[str, Any] = {}
    if backend == "torch":
        profile["device"] = device
    elif backend == "ascend":
        profile["device_id"] = 0
    elif backend == "rknn":
        profile.update(target_name=target.get("soc", "rk3588"), core_mask=7, device_id=0)
    elif backend == "hmm":
        profile.update(role="policy", tcim_abi="tcim-v1", device_id=0)
    elif backend == "hisilicon":
        profile["protocol"] = "sd3403"
    deployment["execution_contract"] = {
        "state_scope": "request",
        "execution_structure": "direct",
        "cancellation_granularity": "request_boundary",
        **deployment.pop("execution_contract", {}),
    }
    deployment["runtime_profile"] = {
        "backend": backend,
        "target": {**target, "runtime": runtime},
        "profile": profile,
    }
    return deployment


def policy_model(policy_type: str) -> dict[str, Any]:
    return {
        "interface": "policy",
        "model_type": policy_type,
        "operation": "predict",
        "inputs": [
            {"semantic": "observation.state", "dtype": "float32", "shape": [6]},
            {"semantic": "observation.images.top", "dtype": "float32", "shape": [3, 16, 24]},
        ],
        "outputs": [{"semantic": "action", "dtype": "float32", "shape": [6]}],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def create_policy_bundle(
    root: Path,
    policy_type: str = "act",
    *,
    local_tokenizer: bool = False,
    include_weights: bool = True,
) -> tuple[str, ...]:
    features = POLICY_FEATURES
    config: dict[str, Any] = {
        "type": policy_type,
        **features,
        "device": "cuda",
    }
    preprocessor_steps: list[dict[str, Any]] = [
        {
            "registry_name": "normalizer_processor",
            "config": {"features": {**features["input_features"], **features["output_features"]}},
            "state_file": "policy_preprocessor_step_0_normalizer_processor.safetensors",
        }
    ]
    if policy_type in {"pi05", "smolvla"}:
        tokenizer_name = (
            "tokenizer"
            if local_tokenizer
            else ("bert-base-uncased" if policy_type == "pi05" else "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        )
        preprocessor_steps.append(
            {
                "registry_name": "tokenizer_processor",
                "config": {"tokenizer_name": tokenizer_name, "max_length": 48},
            }
        )
        if policy_type == "smolvla":
            config["vlm_model_name"] = tokenizer_name

    write_json(root / "config.json", config)
    write_json(root / "policy_preprocessor.json", {"name": "policy_preprocessor", "steps": preprocessor_steps})
    write_json(
        root / "policy_postprocessor.json",
        {
            "name": "policy_postprocessor",
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {"features": features["output_features"]},
                    "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                }
            ],
        },
    )
    (root / "policy_preprocessor_step_0_normalizer_processor.safetensors").write_bytes(b"preprocessor-state")
    (root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(b"postprocessor-state")
    if include_weights:
        (root / "model.safetensors").write_bytes(b"native-policy-weights")
    if local_tokenizer:
        tokenizer = root / "tokenizer"
        tokenizer.mkdir()
        write_json(tokenizer / "tokenizer_config.json", {"model_max_length": 48})
        write_json(tokenizer / "tokenizer.json", {"version": "1.0"})

    required = {
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_0_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    if include_weights:
        required.add("model.safetensors")
    if local_tokenizer:
        required.update({"tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"})
    return tuple(sorted(required))


def create_non_policy_bundle(root: Path) -> tuple[str, ...]:
    vocabulary_path = root / "assets" / "tags.txt"
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path.write_text("robot\nworkbench\n", encoding="utf-8")
    return ("assets/tags.txt",)


def make_manifest(
    root: Path,
    bundle_paths: tuple[str, ...],
    *,
    deployment_name: str = "cpu",
    compiled: bool = False,
    backend: str = "rknn",
    policy_type: str = "act",
) -> dict[str, Any]:
    config_path = root / "config.json"
    if config_path.is_file():
        configured_type = json.loads(config_path.read_text(encoding="utf-8")).get("type")
        if isinstance(configured_type, str) and configured_type:
            policy_type = configured_type
    entries = [BundleFile(path=path) for path in bundle_paths]
    if compiled:
        artifact_path = "artifacts/policy.rknn"
        artifact_file = root / artifact_path
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        artifact_file.write_bytes(b"compiled-policy")
        deployment: dict[str, Any] = {
            "uuid": TEST_DEPLOYMENT_UUID,
            "revision": 1,
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {
                "backend": backend,
                "target": {
                    "soc": "sd3403" if backend == "hisilicon" else "rk3588",
                    "runtime": {
                        "ascend": "acl",
                        "hisilicon": "hisilicon-worker",
                        "hmm": "tcim",
                    }.get(backend, "rknn-lite"),
                },
                "profile": {
                    **({"device_id": 0} if backend in {"ascend", "rknn", "hmm"} else {}),
                    **({"target_name": "rk3588", "core_mask": 7} if backend == "rknn" else {}),
                    **({"protocol": "sd3403"} if backend == "hisilicon" else {}),
                    **({"role": "policy", "tcim_abi": "tcim-v1"} if backend == "hmm" else {}),
                },
            },
            "artifacts": {
                "policy": {
                    "path": artifact_path,
                    "format": "rknn",
                }
            },
            "execution": ["policy"],
            "bindings": {
                "policy": {
                    "inputs": [
                        {
                            "semantic": "observation.state",
                            "runtime_name": "state",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 6],
                        },
                        {
                            "semantic": "observation.images.top",
                            "runtime_name": "image",
                            "index": 1,
                            "dtype": "float32",
                            "layout": "NCHW",
                            "shape": [1, 3, 16, 24],
                        },
                    ],
                    "outputs": [
                        {
                            "semantic": "action",
                            "runtime_name": "actions",
                            "index": 0,
                            "dtype": "float32",
                            "shape": [1, 4, 6],
                        }
                    ],
                }
            },
        }
    else:
        deployment = {
            "uuid": TEST_DEPLOYMENT_UUID,
            "revision": 1,
            "execution_contract": {
                "state_scope": "request",
                "execution_structure": "direct",
                "cancellation_granularity": "request_boundary",
            },
            "runtime_profile": {
                "backend": "torch",
                "target": {"runtime": "torch"},
                "profile": {"device": "cpu"},
            },
        }

    return {
        "schema_version": 3,
        "bundle": {
            "uuid": TEST_BUNDLE_UUID,
            "revision": 1,
            "name": f"test-{deployment_name}",
            "files": [entry.model_dump(mode="json") for entry in entries],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, f"test-{deployment_name}", entries),
            },
        },
        "model": {
            "interface": "policy",
            "model_type": policy_type,
            "operation": "predict",
            "inputs": [
                {"semantic": "observation.state", "dtype": "float32", "shape": [6]},
                {"semantic": "observation.images.top", "dtype": "float32", "shape": [3, 16, 24]},
            ],
            "outputs": [{"semantic": "action", "dtype": "float32", "shape": [6]}],
        },
        "deployments": {deployment_name: deployment},
    }


def make_non_policy_manifest(
    root: Path,
    bundle_paths: tuple[str, ...],
    *,
    deployment_name: str = "ascend",
    model_type: str = "ram_plus",
    output_semantic: str = "tag_logits",
) -> dict[str, Any]:
    entries = [BundleFile(path=path) for path in bundle_paths]
    artifact_path = "artifacts/ram_plus.om"
    artifact_file = root / artifact_path
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_bytes(b"compiled-ram-plus")
    return {
        "schema_version": 3,
        "bundle": {
            "uuid": TEST_BUNDLE_UUID,
            "revision": 1,
            "name": "test-ram-plus",
            "files": [entry.model_dump(mode="json") for entry in entries],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, "test-ram-plus", entries),
            },
        },
        "model": {
            "interface": "tensor_model",
            "model_type": model_type,
            "operation": {
                "ram_plus": "recognize_tags",
                "sam2": "prompt",
                "siglip2": "encode",
                "grounding_dino": "detect",
                "dummy_echo": "echo",
                "graspgen": "generate_grasps",
                "fullsubnet": "enhance",
                "silero_vad": "vad",
                "speech_direction": "enhance_and_vad",
            }.get(model_type, "infer"),
            "inputs": [
                {
                    "semantic": "observation.image",
                    "dtype": "float32",
                    "shape": [1, 3, 384, 384],
                    "layout": "NCHW",
                }
            ],
            "outputs": [{"semantic": output_semantic, "dtype": "float32", "shape": [1, 4585]}],
        },
        "deployments": {
            deployment_name: {
                "uuid": TEST_DEPLOYMENT_UUID,
                "revision": 1,
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {
                    "backend": "ascend",
                    "target": {"soc": "Ascend310P3", "runtime": "acl"},
                    "profile": {"device_id": 0},
                },
                "artifacts": {"model": {"path": artifact_path, "format": "om"}},
                "execution": ["model"],
                "bindings": {
                    "model": {
                        "inputs": [
                            {
                                "semantic": "observation.image",
                                "runtime_name": "image",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [1, 3, 384, 384],
                                "layout": "NCHW",
                            }
                        ],
                        "outputs": [
                            {
                                "semantic": output_semantic,
                                "runtime_name": "logits",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [1, 4585],
                            }
                        ],
                    }
                },
            }
        },
    }


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    write_json(root / "inference_manifest.json", manifest)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_manifest import load_inference_manifest
from model_utils.hmm_export import write_hmm_deployment


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_tcim(
    path: Path, inputs: list[tuple[str, str, list[int]]], outputs: list[tuple[str, str, list[int]]]
) -> Path:
    def tensor(value: tuple[str, str, list[int]]) -> dict:
        name, dtype, shape = value
        code = "float" if dtype.startswith("float") else "bool" if dtype == "bool" else "int"
        bits = 8 if dtype == "bool" else int(dtype.removeprefix(code))
        return {"name": name, "shape": shape, "dtype": {"code": code, "bits": bits}}

    _write_json(
        path,
        {
            "Model": {
                "inputs": [tensor(value) for value in inputs],
                "outputs": [tensor(value) for value in outputs],
            }
        },
    )
    return path


def _bundle(root: Path, policy_type: str) -> dict:
    config = {
        "type": policy_type,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 16]},
            "observation.images.wrist": {"type": "VISUAL", "shape": [3, 16, 16]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "tokenizer_max_length": 4,
        "max_state_dim": 8,
        "chunk_size": 2,
        "max_action_dim": 8,
        "num_inference_steps": 2,
        "num_steps": 2,
        "add_image_special_tokens": False,
    }
    _write_json(root / "config.json", config)
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})
    return config


def _artifact(root: Path, name: str) -> Path:
    path = root / name
    path.write_bytes(name.encode())
    return path


def test_write_pi05_hmm_deployment_uses_tcim_abis_and_output_cache_links(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = _bundle(bundle, "pi05")
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    vision = _artifact(compiler, "vision.hmm")
    embedding = _artifact(compiler, "embedding.pt")
    vision_abi = _write_tcim(
        compiler / "vision.json",
        [("pixel_values", "float16", [1, 3, 16, 16])],
        [("image_features", "float16", [1, 2, 4])],
    )
    role_artifacts = {
        "prefill": (
            _artifact(compiler, "prefill.hmm"),
            _write_tcim(
                compiler / "prefill.json",
                [
                    ("prefix_embs", "float16", [1, 9, 4]),
                    ("attention_mask", "float16", [1, 1, 9, 11]),
                    ("position_ids", "int64", [1, 9]),
                ],
                [
                    ("past_key_0", "float16", [1, 1, 9, 2]),
                    ("past_value_0", "float16", [1, 1, 9, 2]),
                ],
            ),
        ),
        "action_in_proj": (
            _artifact(compiler, "action_in_proj.hmm"),
            _write_tcim(
                compiler / "action_in_proj.json",
                [("action_in", "float16", [1, 2, 8])],
                [("action_in_proj_out", "float16", [1, 2, 4])],
            ),
        ),
        "time_mlp": (
            _artifact(compiler, "time_mlp.hmm"),
            _write_tcim(
                compiler / "time_mlp.json",
                [("time_emb", "float16", [1, 4])],
                [("time_mlp_out", "float16", [1, 4])],
            ),
        ),
        "decode": (
            _artifact(compiler, "decode.hmm"),
            _write_tcim(
                compiler / "decode.json",
                [
                    ("action_embs", "float16", [1, 2, 4]),
                    ("attention_mask", "float16", [1, 1, 2, 11]),
                    ("position_ids", "int64", [1, 2]),
                    ("condition", "float16", [1, 4]),
                    ("past_key_0", "float16", [1, 1, 9, 2]),
                    ("past_value_0", "float16", [1, 1, 9, 2]),
                ],
                [("suffix_hidden", "float16", [1, 2, 4])],
            ),
        ),
        "action_out_proj": (
            _artifact(compiler, "action_out_proj.hmm"),
            _write_tcim(
                compiler / "action_out_proj.json",
                [("action_out", "float16", [1, 2, 4])],
                [("action_out_proj_out", "float16", [1, 2, 8])],
            ),
        ),
    }

    import model_utils.inference_manifest_export as exporter

    scans = []
    original = exporter.PublishedArtifactIndex.__init__

    def scan(self, root, manifest):
        scans.append(manifest)
        original(self, root, manifest)

    monkeypatch.setattr(exporter.PublishedArtifactIndex, "__init__", scan)
    manifest_path = write_hmm_deployment(
        bundle,
        config,
        vision_hmm=vision,
        vision_abi_path=vision_abi,
        embedding_path=embedding,
        role_artifacts=role_artifacts,
    )
    assert len(scans) <= 2
    validated = load_inference_manifest(bundle, "hmm")

    assert manifest_path == bundle / "inference_manifest.json"
    assert validated.deployment.execution == (
        "vision_top",
        "vision_wrist",
        "embedding",
        "prefill",
        "action_in_proj",
        "time_mlp",
        "decode",
        "action_out_proj",
    )
    assert validated.deployment.device_links[0].producer_binding == "output"
    assert validated.deployment.device_links[0].semantic == "internal.past_key.0"
    assert validated.deployment.bindings["prefill"].outputs[0].semantic == "internal.past_key.0"


def test_write_smolvla_hmm_deployment_links_only_action_cache_layers(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = _bundle(bundle, "smolvla")
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    vision = _artifact(compiler, "vision.hmm")
    embedding = _artifact(compiler, "embedding.pt")
    state_projection = _artifact(compiler, "state_projection.pt")
    vision_abi = _write_tcim(
        compiler / "vision.json",
        [("pixel_values", "float16", [1, 3, 16, 16])],
        [("image_embeddings", "float16", [1, 2, 4])],
    )
    prefill = _artifact(compiler, "prefill.hmm")
    action = _artifact(compiler, "action.hmm")
    prefill_abi = _write_tcim(
        compiler / "prefill.json",
        [
            ("prefix_embs", "float16", [1, 10, 4]),
            ("attention_mask", "int32", [1, 10, 10]),
            ("position_ids", "int32", [1, 10]),
        ],
        [
            ("past_key_0", "float16", [1, 10, 1, 2]),
            ("past_value_0", "float16", [1, 10, 1, 2]),
            ("past_key_1", "float16", [1, 10, 1, 2]),
            ("past_value_1", "float16", [1, 10, 1, 2]),
        ],
    )
    action_abi = _write_tcim(
        compiler / "action.json",
        [
            ("x_t", "float16", [1, 2, 8]),
            ("timestep", "float16", [1]),
            ("prefix_pad_masks", "int8", [1, 10]),
            ("past_key_0", "float16", [1, 10, 1, 2]),
            ("past_value_0", "float16", [1, 10, 1, 2]),
        ],
        [("v_t", "float16", [1, 2, 8])],
    )

    write_hmm_deployment(
        bundle,
        config,
        vision_hmm=vision,
        vision_abi_path=vision_abi,
        embedding_path=embedding,
        state_projection_path=state_projection,
        role_artifacts={"prefill": (prefill, prefill_abi), "action": (action, action_abi)},
    )
    validated = load_inference_manifest(bundle, "hmm")

    assert validated.deployment.execution == (
        "vision_top",
        "vision_wrist",
        "embedding",
        "prefill",
        "action",
    )
    assert {link.semantic for link in validated.deployment.device_links} == {
        "internal.past_key.0",
        "internal.past_value.0",
    }
    assert tuple(binding.semantic for binding in validated.deployment.bindings["prefill"].outputs[-2:]) == (
        "diagnostic.prefill.cache.2",
        "diagnostic.prefill.cache.3",
    )
    assert "state_projection" in validated.deployment.artifacts


def test_write_hmm_deployment_rejects_act_and_missing_smolvla_projection(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = _bundle(bundle, "act")
    artifact = _artifact(tmp_path, "artifact.hmm")
    abi = _write_tcim(
        tmp_path / "model.json",
        [("input", "float16", [1])],
        [("output", "float16", [1])],
    )

    with pytest.raises(ValueError, match="only PI0.5 and SmolVLA"):
        write_hmm_deployment(
            bundle,
            config,
            vision_hmm=artifact,
            vision_abi_path=abi,
            embedding_path=artifact,
            role_artifacts={},
        )

    smolvla = _bundle(bundle, "smolvla")
    with pytest.raises(ValueError, match="requires state_projection_path"):
        write_hmm_deployment(
            bundle,
            smolvla,
            vision_hmm=artifact,
            vision_abi_path=abi,
            embedding_path=artifact,
            role_artifacts={},
        )

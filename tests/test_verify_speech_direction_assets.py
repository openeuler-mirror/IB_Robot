"""CLI-level tests for scripts/verify_speech_direction_assets.py exit codes."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_speech_direction_assets.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "inference_manifest"))

from inference_manifest import BundleFile, canonical_bundle_digest  # noqa: E402 - scripts dir is not a package


def _build_fullsubnet_bundle(models_root: Path) -> Path:
    bundle = models_root / "fullsubnet"
    om_dir = bundle / "artifacts" / "ascend_310b" / "fullsubnet"
    om_dir.mkdir(parents=True, exist_ok=True)
    payload = b"fake-310b-om"
    digest = hashlib.sha256(payload).hexdigest()
    (om_dir / "fullsubnet_cum_stateful_fb_b4_t2_310b_origin.om").write_bytes(payload)
    adapter = bundle / "assets" / "adapter.json"
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text("{}\n", encoding="utf-8")
    bundle_files = [BundleFile(path="assets/adapter.json")]
    manifest = {
        "schema_version": 3,
        "bundle": {
            "uuid": "123e4567-e89b-42d3-a456-426614174000",
            "revision": 1,
            "name": "fullsubnet",
            "files": [{"path": entry.path} for entry in bundle_files],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest("123e4567-e89b-42d3-a456-426614174000", 1, "fullsubnet", bundle_files),
            },
        },
        "model": {
            "interface": "tensor_model",
            "model_type": "fullsubnet",
            "operation": "enhance",
            "inputs": [{"semantic": "host.fullsubnet.fb_spectrum", "dtype": "float32", "shape": [4, 2, 257]}],
            "outputs": [{"semantic": "host.fullsubnet.fb_features", "dtype": "float32", "shape": [4, 2, 257]}],
            "semantic_identity": {
                "logical_model_revision": "fullsubnet@cumulative-218epochs-v1",
                "preprocessing_contract": "stft-257bins-4frame-batch",
                "output_semantics": "enhanced-spectrum-float32",
            },
        },
        "deployments": {
            "ascend_310b": {
                "uuid": "123e4567-e89b-42d3-a456-426614174001",
                "revision": 1,
                "runtime_profile": {
                    "backend": "ascend",
                    "target": {"runtime": "acl", "runtime_abi": "cann-8.3.RC1", "soc": "Ascend310B1"},
                    "profile": {"device_id": 0},
                },
                "artifacts": {
                    "fullsubnet_fb": {
                        "path": "artifacts/ascend_310b/fullsubnet/fullsubnet_cum_stateful_fb_b4_t2_310b_origin.om",
                        "format": "om",
                        "sha256": digest,
                    }
                },
                "execution": ["fullsubnet_fb"],
                "bindings": {
                    "fullsubnet_fb": {
                        "inputs": [
                            {
                                "semantic": "host.fullsubnet.fb_spectrum",
                                "runtime_name": "frame",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [4, 2, 257],
                            },
                            {
                                "semantic": "host.fullsubnet.fb_hidden_in",
                                "runtime_name": "hidden",
                                "index": 1,
                                "dtype": "float32",
                                "shape": [2, 4, 512],
                            },
                            {
                                "semantic": "host.fullsubnet.fb_cell_in",
                                "runtime_name": "cell",
                                "index": 2,
                                "dtype": "float32",
                                "shape": [2, 4, 512],
                            },
                        ],
                        "outputs": [
                            {
                                "semantic": "host.fullsubnet.fb_features",
                                "runtime_name": "output",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [4, 2, 257],
                            },
                            {
                                "semantic": "host.fullsubnet.fb_hidden_out",
                                "runtime_name": "hidden_out",
                                "index": 1,
                                "dtype": "float32",
                                "shape": [2, 4, 512],
                            },
                            {
                                "semantic": "host.fullsubnet.fb_cell_out",
                                "runtime_name": "cell_out",
                                "index": 2,
                                "dtype": "float32",
                                "shape": [2, 4, 512],
                            },
                        ],
                    },
                },
                "execution_contract": {
                    "state_scope": "stream",
                    "execution_structure": "direct",
                    "cancellation_granularity": "checkpoint",
                    "stateful": True,
                    "state_bank_mode": "runtime_exclusive",
                    "max_open_streams": 1,
                    "state_links": [
                        {
                            "role": "fullsubnet_fb",
                            "state_name": "recurrent",
                            "owner": "session",
                            "source": "state.in",
                            "target": "state.out",
                            "scope": "runtime",
                            "state_bank": "fullsubnet_fb.bank",
                        }
                    ],
                },
            }
        },
    }
    (bundle / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def _run_cli(models_root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--models-root", str(models_root), *extra],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_missing_deployment_selection_fails_closed(tmp_path: Path):
    _build_fullsubnet_bundle(tmp_path)

    result = _run_cli(tmp_path, "--bundle", "fullsubnet", "--deployment", "torch_cpu")

    assert result.returncode != 0
    assert "not in manifest deployments" in result.stdout


def test_matching_deployment_selection_succeeds(tmp_path: Path):
    _build_fullsubnet_bundle(tmp_path)

    result = _run_cli(tmp_path, "--bundle", "fullsubnet", "--deployment", "ascend_310b")

    assert result.returncode == 0
    assert "SHA-256 OK" in result.stdout

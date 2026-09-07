"""Write the schema-v3 manifest for the standalone FullSubNet bundle.

The FullSubNet bundle carries one cumulative checkpoint shared by all
deployments. Artifact roles per deployment:

- ``ascend_310p``: ``fullsubnet_fb``/``fullsubnet_sb`` fixed-ABI OMs plus the
  auxiliary ``state_manifest`` cumulative contract.
- ``torch_cpu``/``torch_cuda``: the shared cumulative checkpoint is mapped to
  the ``fullsubnet_fb`` role and the cumulative contract manifest to the
  ``fullsubnet_sb`` role; FB/SB recurrent state stays inside the Torch
  executor and is declared only through state links.

Run after ``scripts/download_speech_direction_models.sh`` populated
``assets/`` and the 310P OM artifacts are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from inference_manifest import (
    ArtifactBindings,
    AscendRuntimeProfile,
    BundleFile,
    CompiledDeployment,
    DeploymentArtifact,
    DeploymentTarget,
    Digest,
    ExecutionContract,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    RoleRuntimeProfile,
    SemanticIdentity,
    SemanticTensor,
    StateLink,
    TorchRuntimeProfile,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)
from voice_asr_service.speech_direction.contract import speech_direction_bindings

BUNDLE_NAME = "fullsubnet"
_ADAPTER_ASSET = "assets/adapter.json"
_CHECKPOINT_ASSET = "assets/cum_fullsubnet_best_model_218epochs.tar"
_STATE_CONTRACT_ASSET = "assets/cum_fullsubnet_best_model_218epochs.manifest.json"
_FB_OM_REL = "artifacts/ascend/fullsubnet/fullsubnet_cum_stateful_fb_b4_t2_fp16.om"
_SB_OM_REL = "artifacts/ascend/fullsubnet/fullsubnet_cum_stateful_sb_b4_t2_fp16.om"

_STATE_SUFFIXES = ("_hidden_in", "_hidden_out", "_cell_in", "_cell_out", ".state_in", ".state_out")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_bindings(role: str) -> ArtifactBindings:
    full = speech_direction_bindings(role)
    return ArtifactBindings(
        inputs=tuple(b for b in full.inputs if not b.semantic.endswith(_STATE_SUFFIXES)),
        outputs=tuple(b for b in full.outputs if not b.semantic.endswith(_STATE_SUFFIXES)),
    )


def _execution_contract() -> ExecutionContract:
    return ExecutionContract(
        state_scope="stream",
        execution_structure="direct",
        cancellation_granularity="checkpoint",
        stateful=True,
        state_bank_mode="runtime_exclusive",
        max_open_streams=1,
        state_links=(
            StateLink(
                role="__runtime__",
                state_name="hidden",
                owner="session",
                source="state.hidden_in",
                target="state.hidden_out",
                scope="runtime",
                state_bank="fullsubnet_fb.bank",
            ),
            StateLink(
                role="__runtime__",
                state_name="cell",
                owner="session",
                source="state.cell_in",
                target="state.cell_out",
                scope="runtime",
                state_bank="fullsubnet_fb.bank",
            ),
            StateLink(
                role="__runtime__",
                state_name="hidden",
                owner="session",
                source="state.hidden_in",
                target="state.hidden_out",
                scope="runtime",
                state_bank="fullsubnet_sb.bank",
            ),
            StateLink(
                role="__runtime__",
                state_name="cell",
                owner="session",
                source="state.cell_in",
                target="state.cell_out",
                scope="runtime",
                state_bank="fullsubnet_sb.bank",
            ),
        ),
    )


def _model_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        interface="tensor_model",
        model_type="fullsubnet",
        operation="enhance",
        inputs=(SemanticTensor(semantic="observation.audio_4ch", dtype="float32", shape=(-1, 4)),),
        outputs=(SemanticTensor(semantic="voice.audio_enhanced_4ch", dtype="float32", shape=(-1, 4)),),
        semantic_identity=SemanticIdentity(
            logical_model_revision="fullsubnet@cumulative-218epochs-v1",
            output_semantics="cirm_decompressed_ola",
            preprocessing_contract="cumulative_laplace_norm",
        ),
    )


def _artifact(path: str, fmt: str, sha: str) -> DeploymentArtifact:
    return DeploymentArtifact(path=path, format=fmt, sha256=sha)


def _ascend_deployment(root: Path) -> CompiledDeployment:
    return CompiledDeployment(
        execution_contract=_execution_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(runtime="acl", runtime_abi="cann-8.1.RC1", soc="Ascend310P1"),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts={
            "fullsubnet_fb": _artifact(_FB_OM_REL, "om", _sha256(root / _FB_OM_REL)),
            "fullsubnet_sb": _artifact(_SB_OM_REL, "om", _sha256(root / _SB_OM_REL)),
            "state_manifest": _artifact(_STATE_CONTRACT_ASSET, "json", _sha256(root / _STATE_CONTRACT_ASSET)),
        },
        execution=("fullsubnet_fb", "fullsubnet_sb"),
        bindings={
            "fullsubnet_fb": speech_direction_bindings("fullsubnet_fb"),
            "fullsubnet_sb": speech_direction_bindings("fullsubnet_sb"),
        },
    )


def _torch_deployment(root: Path, device: str) -> CompiledDeployment:
    return CompiledDeployment(
        execution_contract=_execution_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device=device),
        ),
        artifacts={
            "fullsubnet_fb": _artifact(_CHECKPOINT_ASSET, "torch", _sha256(root / _CHECKPOINT_ASSET)),
            "fullsubnet_sb": _artifact(_STATE_CONTRACT_ASSET, "json", _sha256(root / _STATE_CONTRACT_ASSET)),
        },
        execution=("fullsubnet_fb", "fullsubnet_sb"),
        bindings={
            "fullsubnet_fb": _torch_bindings("fullsubnet_fb"),
            "fullsubnet_sb": _torch_bindings("fullsubnet_sb"),
        },
    )


def _existing_raw(bundle_root: Path) -> dict | None:
    manifest_path = bundle_root / "inference_manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _deployment_revision(existing: dict | None, name: str, deployment: CompiledDeployment) -> tuple[str, int]:
    previous = None
    if existing is not None:
        raw = existing.get("deployments", {}).get(name)
        if isinstance(raw, dict):
            previous = raw
    if previous is None:
        return str(uuid4()), 1
    uuid = str(previous.get("uuid") or uuid4())
    revision = int(previous.get("revision", 1))
    candidate = json.loads(
        json.dumps(
            deployment.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True, exclude={"uuid": True, "revision": True}
            )
        )
    )

    # Compare every declared field, normalizing null and empty containers so the
    # writer's null-vs-empty serialization drift does not look like a change.
    def _normalized(value: object) -> object:
        if value is None or value == [] or value == {}:
            return None
        return value

    if any(_normalized(previous.get(key)) != _normalized(value) for key, value in candidate.items()):
        revision += 1
    return uuid, revision


def package_fullsubnet_bundle(bundle_root: Path) -> Path:
    """Regenerate the FullSubNet bundle manifest from local bundle assets."""

    root = bundle_root.resolve()
    required = (_ADAPTER_ASSET, _CHECKPOINT_ASSET, _STATE_CONTRACT_ASSET, _FB_OM_REL, _SB_OM_REL)
    missing = [str(root / rel) for rel in required if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            "FullSubNet bundle assets are missing:\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/download_speech_direction_models.sh and fetch the 310P OM artifacts first."
        )

    existing = _existing_raw(root)
    deployments_raw = {
        "ascend_310p": _ascend_deployment(root),
        "torch_cpu": _torch_deployment(root, "cpu"),
        "torch_cuda": _torch_deployment(root, "cuda"),
    }
    deployments: dict[str, CompiledDeployment] = {}
    for name, deployment in deployments_raw.items():
        uuid, revision = _deployment_revision(existing, name, deployment)
        deployments[name] = deployment.model_copy(update={"uuid": uuid, "revision": revision})

    files = (BundleFile(path=_ADAPTER_ASSET),)
    previous_bundle = existing.get("bundle", {}) if existing is not None else {}
    bundle_uuid = str(previous_bundle.get("uuid") or uuid4())
    previous_revision = int(previous_bundle.get("revision", 0))
    changed = existing is not None and previous_bundle.get("files") != [{"path": entry.path} for entry in files]
    bundle_revision = previous_revision + int(changed) if existing is not None else 1

    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name=BUNDLE_NAME,
            files=files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, BUNDLE_NAME, files),
            ),
        ),
        model=_model_descriptor(),
        deployments=deployments,
    )
    manifest_path = root / "inference_manifest.json"
    write_inference_manifest(manifest_path, manifest)
    for name in deployments:
        load_inference_manifest(root, name)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    print(package_fullsubnet_bundle(args.bundle_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

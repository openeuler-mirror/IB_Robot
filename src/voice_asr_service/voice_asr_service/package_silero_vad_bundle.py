"""Write the schema-v3 bundle for the reusable Silero VAD model.

Silero VAD is a speech-domain base component consumed by several business
pipelines (speech_direction DOA gating, ASR endpoint detection).  It is
packaged as its own inference bundle - model boundary equals bundle boundary -
so every consumer references one authoritative deployment instead of keeping
private copies under domain paths.

Deployments:

- ``ascend_310p``: the fixed-ABI 310P OM (sample rate folded to a constant;
  audio + LSTM state in, probability + state out, stream contract).
- ``torch_cpu``: the official v6 ONNX, same ABI, executed through onnxruntime
  on the host.

This bundle is the single authoritative Silero VAD source; business consumers
(speech_direction, ASR) resolve their artifacts from it directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    TensorBinding,
    TorchDeployment,
    TorchRuntimeProfile,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)

BUNDLE_NAME = "silero-vad"
DEFAULT_ONNX_REL = "assets/silero_vad.onnx"
DEFAULT_OM_REL = "artifacts/ascend/ascend_310p/silero_vad_v6_310p_mixed16.om"
DEFAULT_OM_SOURCE_REL = "voice_asr/artifacts/ascend/silero_vad/silero_vad_v6_310p_mixed16.om"
DEFAULT_ONNX_SOURCE_REL = "voice_asr/artifacts/torch/silero-vad/silero_vad.onnx"
_ADAPTER_ASSET = "assets/adapter.json"

SILERO_AUDIO_SEMANTIC = "host.silero.audio"
SILERO_STATE_IN_SEMANTIC = "host.silero.state_in"
SILERO_PROB_SEMANTIC = "host.silero.prob"
SILERO_STATE_OUT_SEMANTIC = "host.silero.state_out"

SILERO_AUDIO_SHAPE = (1, 576)
SILERO_STATE_SHAPE = (2, 1, 128)
SILERO_PROB_SHAPE = (1, 1)

SILERO_PREPROCESSING = "mono-16khz-float32-chunk576-lstm-state-v1"
SILERO_POSTPROCESSING = "speech-probability-float32-per-frame-v1"
SILERO_LOGICAL_REVISION = "silero-vad@v6-openvino-16k"

SILERO_EXECUTION_ROLE = "silero_vad"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        inputs=(
            TensorBinding(semantic=SILERO_AUDIO_SEMANTIC, dtype="float32", shape=SILERO_AUDIO_SHAPE, index=0),
            TensorBinding(semantic=SILERO_STATE_IN_SEMANTIC, dtype="float32", shape=SILERO_STATE_SHAPE, index=1),
        ),
        outputs=(
            TensorBinding(semantic=SILERO_PROB_SEMANTIC, dtype="float32", shape=SILERO_PROB_SHAPE, index=0),
            TensorBinding(semantic=SILERO_STATE_OUT_SEMANTIC, dtype="float32", shape=SILERO_STATE_SHAPE, index=1),
        ),
    )


def _stream_contract(state_role: str = "model") -> ExecutionContract:
    return ExecutionContract(
        state_scope="stream",
        execution_structure="direct",
        cancellation_granularity="checkpoint",
        stateful=True,
        state_bank_mode="runtime_exclusive",
        max_open_streams=1,
        state_links=(
            StateLink(
                role=state_role,
                state_name="hidden",
                owner="session",
                source="state.hidden_in",
                target="state.hidden_out",
                scope="runtime",
                state_bank="silero_vad.bank",
            ),
        ),
    )


def _model_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        interface="tensor_model",
        model_type="silero_vad",
        operation="vad",
        inputs=(
            SemanticTensor(semantic=SILERO_AUDIO_SEMANTIC, dtype="float32", shape=SILERO_AUDIO_SHAPE),
            SemanticTensor(semantic=SILERO_STATE_IN_SEMANTIC, dtype="float32", shape=SILERO_STATE_SHAPE),
        ),
        outputs=(
            SemanticTensor(semantic=SILERO_PROB_SEMANTIC, dtype="float32", shape=SILERO_PROB_SHAPE),
            SemanticTensor(semantic=SILERO_STATE_OUT_SEMANTIC, dtype="float32", shape=SILERO_STATE_SHAPE),
        ),
        semantic_identity=SemanticIdentity(
            logical_model_revision=SILERO_LOGICAL_REVISION,
            preprocessing_contract=SILERO_PREPROCESSING,
            output_semantics=SILERO_POSTPROCESSING,
        ),
    )


def _ascend_deployment(om_path: Path) -> CompiledDeployment:
    return CompiledDeployment(
        execution_contract=_stream_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(runtime="acl", runtime_abi="cann-8.1.RC1", soc="Ascend310P1"),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts={
            "model": DeploymentArtifact(
                path=DEFAULT_OM_REL,
                format="om",
                sha256=_sha256(om_path),
            )
        },
        execution=("model",),
        bindings={"model": _bindings()},
    )


def _torch_deployment() -> TorchDeployment:
    return TorchDeployment(
        execution_contract=_stream_contract(state_role="__runtime__"),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device="cpu"),
        ),
    )


def _materialize(bundle_root: Path, source: Path, relative: str) -> Path:
    destination = bundle_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or _sha256(source) != _sha256(destination):
        shutil.copy2(source, destination)
    return destination


def _resolve_source(explicit: Path | None, workspace: Path, relative: str) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"Silero VAD source not found: {explicit}")
        return explicit
    candidate = workspace / "models" / relative
    return candidate if candidate.is_file() else None


def package_silero_vad_bundle(
    bundle_root: Path,
    *,
    workspace: Path | None = None,
    onnx_source: Path | None = None,
    om_source: Path | None = None,
    include_ascend: bool = True,
) -> Path:
    """Create or refresh the standalone Silero VAD bundle under ``bundle_root``."""
    root = workspace or bundle_root.parent.parent
    bundle_root.mkdir(parents=True, exist_ok=True)

    onnx = _resolve_source(onnx_source, root, DEFAULT_ONNX_SOURCE_REL)
    if onnx is None:
        onnx = bundle_root / DEFAULT_ONNX_REL
    if not onnx.is_file():
        raise FileNotFoundError(
            f"Silero VAD ONNX missing: {onnx}; download it first with "
            "./scripts/download_speech_direction_models.sh --silero-only"
        )
    onnx_path = _materialize(bundle_root, onnx, DEFAULT_ONNX_REL)

    om_path: Path | None = None
    if include_ascend:
        om = _resolve_source(om_source, root, DEFAULT_OM_SOURCE_REL)
        if om is not None:
            om_path = _materialize(bundle_root, om, DEFAULT_OM_REL)

    adapter_path = bundle_root / _ADAPTER_ASSET
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_document = (
        json.dumps(
            {
                "interface": "tensor_model",
                "model_type": "silero_vad",
                "operation": "vad",
                "preprocessing": SILERO_PREPROCESSING,
                "postprocessing": SILERO_POSTPROCESSING,
                "logical_model_revision": SILERO_LOGICAL_REVISION,
                "checkpoint": "silero_vad.onnx",
                "checkpoint_sha256": _sha256(onnx_path),
                "frame_size": 512,
                "context_size": 64,
                "sample_rate": 16000,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    adapter_path.write_text(adapter_document, encoding="utf-8")

    manifest_path = bundle_root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = load_inference_manifest(bundle_root, next(iter(raw["deployments"]))).manifest

    files = (BundleFile(path=DEFAULT_ONNX_REL), BundleFile(path=_ADAPTER_ASSET))
    bundle_uuid = existing.bundle.uuid if existing is not None else str(uuid4())
    previous_revision = existing.bundle.revision if existing is not None else 0
    changed = existing is not None and existing.bundle.files != files
    bundle_revision = previous_revision + int(changed) if existing is not None else 1

    deployments: dict[str, object] = {"torch_cpu": _torch_deployment()}
    if om_path is not None:
        deployments["ascend_310p"] = _ascend_deployment(om_path)

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
    write_inference_manifest(manifest_path, manifest)
    for deployment_name in manifest.deployments:
        load_inference_manifest(bundle_root, deployment_name)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root used to resolve default sources")
    parser.add_argument("--onnx-source", type=Path, default=None)
    parser.add_argument("--om-source", type=Path, default=None)
    parser.add_argument("--skip-ascend", action="store_true", help="Package the Torch deployment only")
    args = parser.parse_args()
    print(
        package_silero_vad_bundle(
            args.bundle_root,
            workspace=args.workspace,
            onnx_source=args.onnx_source,
            om_source=args.om_source,
            include_ascend=not args.skip_ascend,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

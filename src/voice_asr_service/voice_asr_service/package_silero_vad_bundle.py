"""Write the schema-v3 bundle for the reusable Silero VAD model.

Silero VAD is a speech-domain base component consumed by several business
pipelines (speech_direction DOA gating, ASR endpoint detection).  It is
packaged as its own inference bundle - model boundary equals bundle boundary -
so every consumer references one authoritative deployment instead of keeping
private copies under domain paths.

Deployments:

- ``ascend_310p``: the fixed-ABI 310P OM (sample rate folded to a constant;
  audio + LSTM state in, probability + state out, stream contract).
- ``ascend_310b``: the same fixed-ABI OM recompiled for Ascend310B1 from the
  sr-folded no-If graph (fp16; see docs/silero_vad_310b_deployment.md).
- ``torch_cpu``: the host CPU deployment name retained for configuration
  compatibility; its actual backend is ONNX Runtime and it carries the v6
  ONNX artifact and recurrent state ABI.

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
    AudioContract,
    BundleFile,
    CompiledDeployment,
    DeploymentArtifact,
    DeploymentTarget,
    Digest,
    ExecutionContract,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    ONNXRuntimeProfile,
    RoleRuntimeProfile,
    SemanticIdentity,
    SemanticTensor,
    StateLink,
    TensorBinding,
    canonical_bundle_digest,
    load_inference_manifest,
    write_inference_manifest,
)

BUNDLE_NAME = "silero-vad"
DEFAULT_ONNX_REL = "assets/silero_vad.onnx"
DEFAULT_OM_REL = "artifacts/ascend/ascend_310p/silero_vad_v6_310p_mixed16.om"
DEFAULT_OM_SOURCE_REL = "silero-vad/artifacts/ascend/ascend_310p/silero_vad_v6_310p_mixed16.om"
DEFAULT_OM_310B_REL = "artifacts/ascend_310b/silero_vad_v6_310b_fp16.om"
DEFAULT_OM_310B_SOURCE_REL = "silero-vad/artifacts/ascend_310b/silero_vad_v6_310b_fp16.om"
DEFAULT_ONNX_SOURCE_REL = "silero-vad/assets/silero_vad.onnx"
_ADAPTER_ASSET = "assets/adapter.json"

SILERO_AUDIO_SEMANTIC = "host.silero.audio"
SILERO_SAMPLE_RATE_SEMANTIC = "host.silero.sample_rate"
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


def _bindings(*, include_sample_rate: bool) -> ArtifactBindings:
    inputs = [
        TensorBinding(semantic=SILERO_AUDIO_SEMANTIC, dtype="float32", shape=SILERO_AUDIO_SHAPE, index=0),
        TensorBinding(semantic=SILERO_STATE_IN_SEMANTIC, dtype="float32", shape=SILERO_STATE_SHAPE, index=1),
    ]
    if include_sample_rate:
        inputs.append(TensorBinding(semantic=SILERO_SAMPLE_RATE_SEMANTIC, dtype="int64", shape=(), index=2))
    return ArtifactBindings(
        inputs=tuple(inputs),
        outputs=(
            TensorBinding(semantic=SILERO_PROB_SEMANTIC, dtype="float32", shape=SILERO_PROB_SHAPE, index=0),
            TensorBinding(semantic=SILERO_STATE_OUT_SEMANTIC, dtype="float32", shape=SILERO_STATE_SHAPE, index=1),
        ),
    )


def _stream_contract(state_role: str = SILERO_EXECUTION_ROLE) -> ExecutionContract:
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
            SemanticTensor(semantic=SILERO_SAMPLE_RATE_SEMANTIC, dtype="int64", shape=()),
        ),
        outputs=(SemanticTensor(semantic=SILERO_PROB_SEMANTIC, dtype="float32", shape=SILERO_PROB_SHAPE),),
        semantic_identity=SemanticIdentity(
            logical_model_revision=SILERO_LOGICAL_REVISION,
            preprocessing_contract=SILERO_PREPROCESSING,
            output_semantics=SILERO_POSTPROCESSING,
        ),
    )


def _ascend_deployment(om_path: Path, *, rel: str, soc: str, runtime_abi: str) -> CompiledDeployment:
    return CompiledDeployment(
        execution_contract=_stream_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(runtime="acl", runtime_abi=runtime_abi, soc=soc),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts={
            SILERO_EXECUTION_ROLE: DeploymentArtifact(
                path=rel,
                format="om",
                sha256=_sha256(om_path),
            )
        },
        execution=(SILERO_EXECUTION_ROLE,),
        bindings={SILERO_EXECUTION_ROLE: _bindings(include_sample_rate=False)},
        audio_contract=AudioContract(
            sample_rate_hz=16000,
            channels=1,
            channel_semantics="mono",
            sample_dtype="float32",
            frame_size=512,
            chunk_size=576,
            execution_mode="streaming",
        ),
    )


def _onnx_deployment(onnx_path: Path) -> CompiledDeployment:
    return CompiledDeployment(
        execution_contract=_stream_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="onnx",
            target=DeploymentTarget(runtime="onnx"),
            profile=ONNXRuntimeProfile(device="cpu"),
        ),
        artifacts={
            SILERO_EXECUTION_ROLE: DeploymentArtifact(
                path=DEFAULT_ONNX_REL,
                format="onnx",
                sha256=_sha256(onnx_path),
            )
        },
        execution=(SILERO_EXECUTION_ROLE,),
        bindings={SILERO_EXECUTION_ROLE: _bindings(include_sample_rate=True)},
        audio_contract=AudioContract(
            sample_rate_hz=16000,
            channels=1,
            channel_semantics="mono",
            sample_dtype="float32",
            frame_size=512,
            chunk_size=576,
            execution_mode="streaming",
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


def _deployment_revision(existing: dict | None, name: str, deployment: CompiledDeployment) -> tuple[str, int]:
    """Keep one deployment's identity stable across repackaging runs."""
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


def package_silero_vad_bundle(
    bundle_root: Path,
    *,
    workspace: Path | None = None,
    onnx_source: Path | None = None,
    om_source: Path | None = None,
    om_310b_source: Path | None = None,
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
    om_310b_path: Path | None = None
    if include_ascend:
        om = _resolve_source(om_source, root, DEFAULT_OM_SOURCE_REL)
        if om is not None:
            om_path = _materialize(bundle_root, om, DEFAULT_OM_REL)
        om_310b = _resolve_source(om_310b_source, root, DEFAULT_OM_310B_SOURCE_REL)
        if om_310b is not None:
            om_310b_path = _materialize(bundle_root, om_310b, DEFAULT_OM_310B_REL)

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
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = (BundleFile(path=_ADAPTER_ASSET),)
    previous_bundle = existing.get("bundle", {}) if isinstance(existing, dict) else {}
    bundle_uuid = str(previous_bundle.get("uuid") or uuid4())
    previous_revision = int(previous_bundle.get("revision", 0)) if previous_bundle else 0
    changed = isinstance(existing, dict) and previous_bundle.get("files") != [{"path": entry.path} for entry in files]
    bundle_revision = previous_revision + int(changed) if existing is not None else 1

    deployments_raw: dict[str, CompiledDeployment] = {"torch_cpu": _onnx_deployment(onnx_path)}
    if om_path is not None:
        deployments_raw["ascend_310p"] = _ascend_deployment(
            om_path, rel=DEFAULT_OM_REL, soc="Ascend310P1", runtime_abi="cann-8.1.RC1"
        )
    if om_310b_path is not None:
        deployments_raw["ascend_310b"] = _ascend_deployment(
            om_310b_path, rel=DEFAULT_OM_310B_REL, soc="Ascend310B1", runtime_abi="cann-8.3.RC1"
        )
    deployments: dict[str, object] = {}
    for name, deployment in deployments_raw.items():
        uuid, revision = _deployment_revision(existing, name, deployment)
        deployments[name] = deployment.model_copy(update={"uuid": uuid, "revision": revision})

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
    parser.add_argument("--om-source", type=Path, default=None, help="310P OM source")
    parser.add_argument("--om-310b-source", type=Path, default=None, help="310B OM source")
    parser.add_argument("--skip-ascend", action="store_true", help="Package the Torch deployment only")
    args = parser.parse_args()
    print(
        package_silero_vad_bundle(
            args.bundle_root,
            workspace=args.workspace,
            onnx_source=args.onnx_source,
            om_source=args.om_source,
            om_310b_source=args.om_310b_source,
            include_ascend=not args.skip_ascend,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Package explicit sherpa-onnx artifacts as a schema-v3 ASR bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from uuid import uuid4

from inference_manifest import (
    ArtifactBindings,
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

IDENTITY = ("tensor_model", "sherpa_onnx", "recognize")
TOKENS_SEMANTIC = "host.asr.tokens"
MODEL_INPUT_SEMANTIC = "host.asr.audio"
MODEL_OUTPUT_SEMANTIC = "host.asr.text"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bindings(role: str) -> ArtifactBindings:
    if role == "tokens":
        return ArtifactBindings(
            inputs=(TensorBinding(semantic=TOKENS_SEMANTIC, dtype="int32", shape=(1,), index=0),),
            outputs=(TensorBinding(semantic=TOKENS_SEMANTIC, dtype="int32", shape=(1,), index=0),),
        )
    return ArtifactBindings(
        inputs=(TensorBinding(semantic=MODEL_INPUT_SEMANTIC, dtype="float32", shape=(-1,), index=0),),
        outputs=(TensorBinding(semantic=MODEL_OUTPUT_SEMANTIC, dtype="float32", shape=(-1,), index=0),),
    )


def _deployment(artifacts: dict[str, DeploymentArtifact], *, streaming: bool, device: str) -> CompiledDeployment:
    roles = tuple(artifacts)
    contract_values = {
        "state_scope": "stream" if streaming else "request",
        "execution_structure": "iterative" if streaming else "direct",
        "cancellation_granularity": "checkpoint" if streaming else "request_boundary",
        "stateful": streaming,
    }
    if streaming:
        contract_values.update(
            {
                "orchestration_visibility": "session",
                "state_bank_mode": "per_stream",
                "max_open_streams": 1,
                "state_links": (
                    StateLink(
                        role="decoder",
                        state_name="decoder_state",
                        owner="session",
                        source="state.decoder_in",
                        target="state.decoder_out",
                        scope="stream",
                        state_bank="sherpa.decoder.bank",
                    ),
                ),
            }
        )
    return CompiledDeployment(
        execution_contract=ExecutionContract(**contract_values),
        runtime_profile=RoleRuntimeProfile(
            backend="onnx",
            target=DeploymentTarget(runtime="onnx"),
            profile=ONNXRuntimeProfile(device=device),
        ),
        artifacts=artifacts,
        execution=roles,
        bindings={role: _bindings(role) for role in roles},
        audio_contract=AudioContract(
            sample_rate_hz=16000,
            channels=1,
            channel_semantics="mono",
            sample_dtype="float32",
            frame_size=512,
            chunk_size=512,
            execution_mode="streaming" if streaming else "offline",
        ),
    )


def package_sherpa_asr_bundle(
    bundle_root: Path,
    *,
    artifacts: dict[str, Path],
    streaming: bool,
    streaming_kind: str = "transducer",
    include_cuda: bool = True,
) -> Path:
    """Copy explicitly named artifacts and write/validate the manifest."""
    if streaming and streaming_kind not in {"transducer", "paraformer"}:
        raise ValueError("streaming_kind must be transducer or paraformer")
    expected = (
        {"tokens", "encoder", "decoder", "joiner"}
        if streaming and streaming_kind == "transducer"
        else {"tokens", "encoder", "decoder"}
        if streaming
        else {"tokens", "model"}
    )
    missing = sorted(expected - set(artifacts))
    if missing:
        raise ValueError(f"missing ASR artifact roles: {', '.join(missing)}")
    root = bundle_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    deployed: dict[str, DeploymentArtifact] = {}
    for role, source in artifacts.items():
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ASR artifact for role {role!r} not found: {source}")
        relative = f"artifacts/{role}/{source.name}"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        deployed[role] = DeploymentArtifact(
            path=relative, format=source.suffix.lstrip(".") or "file", sha256=_sha256(destination)
        )

    metadata_path = root / "bundle.metadata"
    metadata_path.write_text("sherpa-onnx ASR bundle\n", encoding="utf-8")
    bundle_files = (BundleFile(path="bundle.metadata"),)
    deployments_raw = {"torch_cpu": _deployment(deployed, streaming=streaming, device="cpu")}
    if include_cuda:
        deployments_raw["torch_cuda"] = _deployment(deployed, streaming=streaming, device="cuda")

    # Repackaging keeps a stable bundle identity, mirroring the silero/fullsubnet
    # packagers: reuse the existing uuid and only bump the revision when the
    # bundle files or any deployment artifacts change.
    manifest_path = root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
    previous_bundle = existing.get("bundle", {}) if isinstance(existing, dict) else {}
    bundle_uuid = str(previous_bundle.get("uuid") or uuid4())
    previous_revision = int(previous_bundle.get("revision", 0)) if previous_bundle else 0

    # Per-deployment identity stability (mirrors the silero/fullsubnet
    # packagers): keep the same uuid, bump the revision only when the
    # deployment's semantic content (null/empty normalized) changes.
    deployments: dict[str, object] = {}
    for name, deployment in deployments_raw.items():
        previous_deployment = None
        if isinstance(existing, dict):
            raw = existing.get("deployments", {}).get(name)
            if isinstance(raw, dict):
                previous_deployment = raw
        if previous_deployment is None:
            deployment_uuid, deployment_revision = str(uuid4()), 1
        else:
            deployment_uuid = str(previous_deployment.get("uuid") or uuid4())
            deployment_revision = int(previous_deployment.get("revision", 1))
            candidate = json.loads(
                json.dumps(
                    deployment.model_dump(
                        mode="json", exclude_none=True, exclude_defaults=True, exclude={"uuid": True, "revision": True}
                    )
                )
            )

            def _normalized(value: object) -> object:
                if value is None or value == [] or value == {}:
                    return None
                return value

            if any(_normalized(previous_deployment.get(key)) != _normalized(value) for key, value in candidate.items()):
                deployment_revision += 1
        deployments[name] = deployment.model_copy(update={"uuid": deployment_uuid, "revision": deployment_revision})

    previous_artifacts = (
        {name: deployment.get("artifacts") for name, deployment in existing.get("deployments", {}).items()}
        if isinstance(existing, dict)
        else None
    )
    candidate_artifacts = {
        name: json.loads(deployment.model_dump_json(exclude_none=True)).get("artifacts")
        for name, deployment in deployments_raw.items()
    }
    changed = isinstance(existing, dict) and (
        previous_bundle.get("files") != [{"path": entry.path} for entry in bundle_files]
        or previous_artifacts != candidate_artifacts
    )
    bundle_revision = previous_revision + int(changed) if isinstance(existing, dict) else 1

    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name="sherpa-onnx-asr",
            files=bundle_files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, "sherpa-onnx-asr", bundle_files),
            ),
        ),
        model=ModelDescriptor(
            interface=IDENTITY[0],
            model_type=IDENTITY[1],
            operation=IDENTITY[2],
            inputs=(SemanticTensor(semantic=MODEL_INPUT_SEMANTIC, dtype="float32", shape=(-1,)),),
            outputs=(SemanticTensor(semantic=MODEL_OUTPUT_SEMANTIC, dtype="float32", shape=(-1,)),),
            semantic_identity=SemanticIdentity(
                logical_model_revision="sherpa-onnx-asr@bundle-v1",
                preprocessing_contract="mono-16khz-float32",
                output_semantics="utf8-transcript",
            ),
        ),
        deployments=deployments,
    )
    write_inference_manifest(manifest_path, manifest)
    for name in deployments:
        load_inference_manifest(root, name)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--joiner", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--paraformer", action="store_true", help="Package streaming Paraformer roles")
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()
    artifacts = {"tokens": args.tokens}
    if args.offline:
        if args.model is None:
            parser.error("--model is required with --offline")
        artifacts["model"] = args.model
    else:
        required_roles = (("encoder", args.encoder), ("decoder", args.decoder))
        if not args.paraformer:
            required_roles += (("joiner", args.joiner),)
        for role, path in required_roles:
            if path is None:
                parser.error(f"--{role} is required unless --offline is used")
            artifacts[role] = path
    print(
        package_sherpa_asr_bundle(
            args.bundle_root,
            artifacts=artifacts,
            streaming=not args.offline,
            streaming_kind="paraformer" if args.paraformer else "transducer",
            include_cuda=not args.skip_cuda,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

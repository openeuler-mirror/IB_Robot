# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Package the eight compiled GraspGen OM artifacts as a schema-v3 grasp bundle.

This bridges the Huawei GraspGen toolchain - which emits a flat ``graspgen.onnx.json``
plus one OM per role - and the unified bundle the generic model runtime loads. GraspGen
belongs to the grasp model domain and uses the canonical
``tensor_model/graspgen/generate_grasps`` identity.
The model's own constants live in ``assets/adapter.json`` and no LeRobot policy asset is
written or required.

Only ``observation.object_points``, ``grasp.poses`` and ``grasp.confidence`` are external
semantics. Everything the eight roles exchange is either an ``internal.*`` embedding
handed over a device link, or a ``host.graspgen.*`` tensor the session computes between
roles - PointNet++ grouping, the diffusion sample, the integrated pose. Neither belongs to
the service contract, so neither is enrolled in it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from inference_manifest import (
    ArtifactBindings,
    AscendRuntimeProfile,
    BundleFile,
    CompiledDeployment,
    DeploymentTarget,
    DeviceLink,
    Digest,
    ExecutionContract,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    PublishedArtifactIndex,
    RoleRuntimeProfile,
    SemanticIdentity,
    SemanticTensor,
    TensorBinding,
    TorchDeployment,
    TorchRuntimeProfile,
    canonical_bundle_digest,
    inherit_deployment_version,
    integrity_report_path,
    load_inference_manifest,
    sha256_file,
    write_inference_manifest,
)
from model_utils.acl_abi_inspection import write_acl_om_abi
from model_utils.graspgen_contract import (
    GRASPGEN_CONFIDENCE_SEMANTIC,
    GRASPGEN_CONTRACT_VERSION,
    GRASPGEN_DISCRIMINATOR_EMBEDDING,
    GRASPGEN_EXECUTION,
    GRASPGEN_GENERATOR_EMBEDDING,
    GRASPGEN_POINT_CLOUD_SEMANTIC,
    GRASPGEN_POSE_SEMANTIC,
    graspgen_geometry,
    graspgen_input_semantics,
    graspgen_output_semantics,
)
from model_utils.inference_manifest_export import read_runtime_abi

from .bundle_publication import (
    classify_om_artifacts,
    copy_and_finalize_artifact_roles,
    preflight_publication,
)
from .graspgen_adapter import GRASPGEN_POSTPROCESSING, GRASPGEN_PREPROCESSING, GraspGenAdapter

# The four-dimensional grouped-feature tensors are NCHW; ACL does not always report a
# format for them, and an unlabelled layout would let a future NHWC recompile load
# silently against a session that still groups channel-first.
_GROUPED_FEATURE_ROLES = frozenset(
    {
        "generator_sa1",
        "generator_sa2",
        "generator_encoder_head",
        "discriminator_sa1",
        "discriminator_sa2",
        "discriminator_encoder_head",
    }
)

_DEVICE_LINKS = (
    DeviceLink(
        semantic=GRASPGEN_GENERATOR_EMBEDDING,
        producer="generator_encoder_head",
        consumer="denoiser",
        transport="device_pointer",
        owner="producer",
    ),
    DeviceLink(
        semantic=GRASPGEN_DISCRIMINATOR_EMBEDDING,
        producer="discriminator_encoder_head",
        consumer="discriminator_head",
        transport="device_pointer",
        owner="producer",
    ),
)


@dataclass(frozen=True)
class RoleArtifact:
    """One validated role: where its OM came from, where it lands, how it binds."""

    role: str
    source: Path
    destination: str
    bindings: ArtifactBindings


# Source key -> (adapter key, bundle destination). All asset writers and the
# preflight write set derive from this mapping.
_TORCH_ASSETS = {
    "config": ("gripper_config", "assets/graspgen_config.yml"),
    "generator_checkpoint": ("generator_checkpoint", "assets/generator_checkpoint.pth"),
    "discriminator_checkpoint": ("discriminator_checkpoint", "assets/discriminator_checkpoint.pth"),
}


def require_contract_version(onnx_manifest: Mapping[str, object]) -> None:
    """Refuse an export produced against a different GraspGen contract.

    ``graspgen.onnx.json`` is the handoff between the exporter and this packager, and the
    two agree on the role set, the role order and the sampling geometry only because they
    read the same definition. Compiling a stale export would produce OMs whose static
    shapes no longer match the geometry the session computes, which surfaces on the device
    as an ACL shape error with nothing pointing back at the export that caused it.
    """
    declared = onnx_manifest.get("contract_version")
    if declared != GRASPGEN_CONTRACT_VERSION:
        raise ValueError(
            f"GraspGen ONNX manifest declares contract_version {declared!r}, but this packager "
            f"implements {GRASPGEN_CONTRACT_VERSION}; re-run graspgen-export-onnx against this workspace"
        )


def role_bindings(role: str, abi: Any) -> ArtifactBindings:
    """Map one role's runtime ABI onto manifest semantics, in runtime index order."""
    input_semantics = tuple(graspgen_input_semantics(role).values())
    output_semantics = tuple(graspgen_output_semantics(role).values())
    if len(abi.inputs) != len(input_semantics):
        raise ValueError(f"role {role!r} runtime ABI has {len(abi.inputs)} inputs; expected {len(input_semantics)}")
    if len(abi.outputs) != len(output_semantics):
        raise ValueError(f"role {role!r} runtime ABI has {len(abi.outputs)} outputs; expected {len(output_semantics)}")
    inputs = tuple(
        TensorBinding(
            semantic=semantic,
            runtime_name=tensor.name,
            index=tensor.index,
            dtype=tensor.dtype,
            shape=tensor.shape,
            layout=(
                tensor.layout
                if tensor.layout is not None
                else "NCHW"
                if role in _GROUPED_FEATURE_ROLES and len(tensor.shape) == 4
                else None
            ),
        )
        for tensor, semantic in zip(sorted(abi.inputs, key=lambda item: item.index), input_semantics, strict=True)
    )
    outputs = tuple(
        TensorBinding(
            semantic=semantic,
            runtime_name=tensor.name,
            index=tensor.index,
            dtype=tensor.dtype,
            shape=tensor.shape,
            layout=tensor.layout,
        )
        for tensor, semantic in zip(sorted(abi.outputs, key=lambda item: item.index), output_semantics, strict=True)
    )
    return ArtifactBindings(inputs=inputs, outputs=outputs)


def _resolve_roles(
    *,
    deployment_name: str,
    om_dir: Path,
    om_abi_dir: Path,
    inspect_missing_abi: bool,
    abi_device_id: int,
    acl_config_path: str | None,
) -> tuple[RoleArtifact, ...]:
    """Read and bind all eight roles before the bundle is touched.

    Resolution is deliberately a separate pass: a missing OM or a role whose ABI does not
    match its declared semantics must fail with the bundle exactly as it was, rather than
    leaving a half-copied artifact tree and a manifest describing roles that are not there.
    """
    resolved = []
    for role in GRASPGEN_EXECUTION:
        om_path = om_dir / f"{role}.om"
        if not om_path.is_file():
            raise FileNotFoundError(f"OM artifact for role {role!r} not found: {om_path}")
        abi_path = om_abi_dir / f"{role}.om.abi.json"
        if not abi_path.is_file():
            if not inspect_missing_abi:
                raise FileNotFoundError(f"Runtime-introspected OM ABI for role {role!r} not found: {abi_path}")
            write_acl_om_abi(om_path, abi_path, device_id=abi_device_id, acl_config_path=acl_config_path)
        resolved.append(
            RoleArtifact(
                role=role,
                source=om_path,
                destination=f"artifacts/ascend/{deployment_name}/{role}.om",
                bindings=role_bindings(role, read_runtime_abi(abi_path)),
            )
        )
    return tuple(resolved)


def _adapter_assets(onnx_manifest: Mapping[str, object], *, grasp_batch_size: int, point_count: int) -> dict[str, Any]:
    """Render the adapter identity and the model's own constants for ``assets/adapter.json``.

    ``kappa`` and the diffusion step count come from the checkpoint config the exporter
    read; the sampling geometry does not. The geometry is contract, shared by the exporter,
    this packager and the session through ``graspgen_geometry()``, and the ONNX manifest
    lists it with the encoder head's null stage appended. Copying that list through would
    put two ``null`` entries in the bundle for no reader.
    """
    backend_config = dict(onnx_manifest.get("backend_config") or {})
    return {
        "interface": "tensor_model",
        "model_type": "graspgen",
        "operation": "generate_grasps",
        "preprocessing": GRASPGEN_PREPROCESSING,
        "postprocessing": GRASPGEN_POSTPROCESSING,
        "kappa": float(backend_config.get("kappa", 2.02217)),
        "diffusion_steps": int(backend_config.get("diffusion_steps", 10)),
        "grasp_batch_size": grasp_batch_size,
        "point_count": point_count,
        "geometry": graspgen_geometry(),
        "torch_module_loader": "perception_service.torch_model_loaders:load_graspgen",
        **{adapter_key: path for adapter_key, path in _TORCH_ASSETS.values()},
    }


def _torch_asset_sources(onnx_manifest: Mapping[str, object]) -> dict[str, Path]:
    source = onnx_manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("GraspGen ONNX manifest must declare source config and checkpoints")
    sources = {}
    for name, (_adapter_key, path) in _TORCH_ASSETS.items():
        raw_path = source.get(name)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"GraspGen ONNX manifest source.{name} must be a non-empty path")
        source_path = Path(raw_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"GraspGen Torch source asset is unavailable: {source_path}")
        sources[path] = source_path
    return sources


def _copy_torch_assets(root: Path, sources: Mapping[str, Path]) -> None:
    for path, source_path in sources.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or sha256_file(source_path) != sha256_file(destination):
            destination.write_bytes(source_path.read_bytes())


def _model_descriptor(grasp_batch_size: int) -> ModelDescriptor:
    del grasp_batch_size
    return ModelDescriptor(
        interface="tensor_model",
        model_type="graspgen",
        operation="generate_grasps",
        inputs=(SemanticTensor(semantic=GRASPGEN_POINT_CLOUD_SEMANTIC, dtype="float32", shape=(-1, 3)),),
        outputs=(
            SemanticTensor(semantic=GRASPGEN_POSE_SEMANTIC, dtype="float32", shape=(-1, 4, 4)),
            SemanticTensor(semantic=GRASPGEN_CONFIDENCE_SEMANTIC, dtype="float32", shape=(-1,)),
        ),
        semantic_identity=SemanticIdentity(
            logical_model_revision=f"{GraspGenAdapter.identity.model_type}@v{GRASPGEN_CONTRACT_VERSION}",
            preprocessing_contract=GRASPGEN_PREPROCESSING,
            output_semantics=GRASPGEN_POSTPROCESSING,
        ),
    )


def write_graspgen_ascend_bundle(
    bundle_root: Path,
    *,
    deployment_name: str,
    om_dir: Path,
    om_abi_dir: Path,
    soc_version: str,
    onnx_manifest: Mapping[str, object],
    grasp_batch_size: int,
    point_count: int,
    inspect_missing_abi: bool = True,
    abi_device_id: int = 0,
    acl_config_path: str | None = None,
) -> Path:
    """Package an Ascend bundle; ACL config is accepted only for missing-ABI inspection."""
    require_contract_version(onnx_manifest)
    roles = _resolve_roles(
        deployment_name=deployment_name,
        om_dir=om_dir,
        om_abi_dir=om_abi_dir,
        inspect_missing_abi=inspect_missing_abi,
        abi_device_id=abi_device_id,
        acl_config_path=acl_config_path,
    )
    model = _model_descriptor(grasp_batch_size)
    assets = _adapter_assets(onnx_manifest, grasp_batch_size=grasp_batch_size, point_count=point_count)

    root = bundle_root
    manifest_path = root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        name = next(iter(json.loads(manifest_path.read_text(encoding="utf-8"))["deployments"]))
        existing = load_inference_manifest(root, name).manifest

    previous = existing.deployments.get(deployment_name) if existing is not None else None
    published = PublishedArtifactIndex(root, existing)
    publication = classify_om_artifacts(
        root, {artifact.role: (artifact.destination, artifact.source) for artifact in roles}, published
    )
    deployment = CompiledDeployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(soc=soc_version, runtime="acl"),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts=publication.artifacts,
        execution=GRASPGEN_EXECUTION,
        bindings={artifact.role: artifact.bindings for artifact in roles},
        device_links=_DEVICE_LINKS,
    )
    asset_sources = _torch_asset_sources(onnx_manifest)
    adapter_relative = "assets/adapter.json"
    asset_writes = (adapter_relative, *(path for _, path in _TORCH_ASSETS.values()))
    preserved, publications = preflight_publication(
        root,
        published,
        {deployment_name: publication},
        {deployment_name, "torch_cuda"},
        asset_writes=asset_writes,
        retained_assets=(
            str(path.relative_to(root))
            for path in (root / "assets").rglob("*")
            if path.is_file() and str(path.relative_to(root)) not in asset_writes
        ),
        asset_sources=asset_sources,
    )

    adapter_path = root / adapter_relative
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(json.dumps(assets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _copy_torch_assets(root, asset_sources)
    deployment = copy_and_finalize_artifact_roles(
        root,
        deployment,
        publications[deployment_name],
    )
    deployment = inherit_deployment_version(deployment, previous)

    files = tuple(
        BundleFile(path=str(path.relative_to(root))) for path in sorted((root / "assets").rglob("*")) if path.is_file()
    )
    structure_changed = existing is not None and (
        existing.bundle.name != root.name or existing.bundle.files != files or existing.model != model
    )
    bundle_uuid = existing.bundle.uuid if existing is not None else str(uuid4())
    bundle_revision = existing.bundle.revision + int(structure_changed) if existing is not None else 1

    torch_deployment = TorchDeployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device="cuda"),
        ),
    )
    previous_torch = existing.deployments.get("torch_cuda") if existing is not None else None
    if isinstance(previous_torch, TorchDeployment):
        torch_deployment = torch_deployment.model_copy(
            update={"uuid": previous_torch.uuid, "revision": previous_torch.revision}
        )
    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name=root.name,
            files=files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, root.name, files),
            ),
        ),
        model=model,
        deployments={
            **preserved,
            "torch_cuda": torch_deployment,
            deployment_name: deployment,
        },
    )
    try:
        write_inference_manifest(manifest_path, manifest)
        for name in manifest.deployments:
            load_inference_manifest(root, name)
    except Exception:
        if existing is None:
            manifest_path.unlink(missing_ok=True)
        else:
            write_inference_manifest(manifest_path, existing)
        raise
    integrity_report_path(root).unlink(missing_ok=True)
    return manifest_path


def _load_onnx_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ONNX manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("model_type") != "graspgen":
        raise ValueError(f"not a GraspGen ONNX manifest: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package compiled GraspGen OM artifacts as a schema-v3 grasp-domain bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle-root", required=True, help="Target bundle directory.")
    parser.add_argument("--onnx-manifest", required=True, help="graspgen.onnx.json emitted by graspgen-export-onnx.")
    parser.add_argument(
        "--om-dir",
        default=None,
        help="Directory holding the eight compiled OM files (defaults to bundle_root/artifacts/ascend/<deployment>).",
    )
    parser.add_argument(
        "--om-abi-dir",
        default=None,
        help="Directory containing or receiving <role>.om.abi.json runtime descriptors (defaults to --om-dir).",
    )
    parser.add_argument(
        "--inspect-missing-abi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Inspect OMs with ACL when runtime ABI sidecars are missing.",
    )
    parser.add_argument("--abi-device-id", type=int, default=0, help="Ascend device used for OM ABI inspection.")
    parser.add_argument(
        "--acl-config-path", default=None, help="Optional ACL initialization config for ABI inspection."
    )
    parser.add_argument("--soc-version", required=True, help="Target Ascend SoC, e.g. Ascend310P3.")
    parser.add_argument("--deployment", default="ascend_310p", help="Unified manifest deployment name.")
    parser.add_argument("--grasp-batch-size", type=int, default=None, help="Override the grasp batch size.")
    parser.add_argument("--point-count", type=int, default=None, help="Override the input point-cloud size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve(strict=True)
    onnx_manifest = _load_onnx_manifest(Path(args.onnx_manifest).expanduser().resolve(strict=True))
    backend_config = dict(onnx_manifest.get("backend_config") or {})
    grasp_batch_size = int(
        args.grasp_batch_size if args.grasp_batch_size is not None else backend_config.get("grasp_batch_size", 1000)
    )
    point_count = int(args.point_count if args.point_count is not None else backend_config.get("point_count", 2048))
    om_dir = (
        Path(args.om_dir).expanduser().resolve(strict=True)
        if args.om_dir
        else bundle_root / "artifacts" / "ascend" / args.deployment
    )
    om_abi_dir = Path(args.om_abi_dir).expanduser().resolve(strict=True) if args.om_abi_dir else om_dir
    manifest_path = write_graspgen_ascend_bundle(
        bundle_root,
        deployment_name=args.deployment,
        om_dir=om_dir,
        om_abi_dir=om_abi_dir,
        soc_version=args.soc_version,
        onnx_manifest=onnx_manifest,
        grasp_batch_size=grasp_batch_size,
        point_count=point_count,
        inspect_missing_abi=args.inspect_missing_abi,
        abi_device_id=args.abi_device_id,
        acl_config_path=args.acl_config_path,
    )
    print(f"GraspGen bundle written: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

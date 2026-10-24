"""Finalize downloaded perception assets as schema-v3 inference bundles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from inference_manifest import (
    ARTIFACT_INTEGRITY_UNVERIFIED,
    ArtifactBindings,
    AscendRuntimeProfile,
    BundleFile,
    CompiledDeployment,
    DeploymentArtifact,
    DeploymentTarget,
    Digest,
    EmbeddingMetadata,
    ExecutionContract,
    InferenceManifest,
    ManifestBundle,
    ManifestIntegrityError,
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
    load_inference_manifest_metadata,
    sha256_file,
    write_inference_manifest,
)

from .bundle_publication import (
    classify_om_artifacts,
    copy_and_finalize_artifact_roles,
    preflight_publication,
)
from .ram_plus_adapter import RAM_PLUS_POSTPROCESSING, RAM_PLUS_PREPROCESSING
from .semantic_model_adapters import GroundingDINOAdapter, SAM2Adapter, SigLIP2ImageAdapter


@dataclass(frozen=True)
class BundleSpec:
    model_type: str
    name: str
    adapter: dict[str, object]
    model: ModelDescriptor
    required_paths: tuple[str, ...]
    compiled: tuple[CompiledSpec, ...] = ()


@dataclass(frozen=True)
class CompiledSpec:
    deployment: str
    source: str
    artifact: str
    target_soc: str
    bindings: ArtifactBindings


RAM_PLUS_BINDINGS = ArtifactBindings(
    inputs=(
        TensorBinding(semantic="observation.image", index=0, dtype="float32", shape=(-1, 3, 384, 384), layout="NCHW"),
    ),
    outputs=(TensorBinding(semantic="tag_logits", index=0, dtype="float32", shape=(-1, 4585)),),
)


def _tensor(semantic: str, dtype: str, shape: tuple[int, ...], layout: str | None = None) -> SemanticTensor:
    return SemanticTensor(semantic=semantic, dtype=dtype, shape=shape, layout=layout)


def _request_contract() -> ExecutionContract:
    return ExecutionContract(
        state_scope="request",
        execution_structure="direct",
        cancellation_granularity="request_boundary",
    )


def _torch_deployment(device: str) -> TorchDeployment:
    return TorchDeployment(
        execution_contract=_request_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device=device),
        ),
    )


def _ascend_deployment(
    *,
    target_soc: str,
    artifacts: dict[str, DeploymentArtifact],
    bindings: dict[str, ArtifactBindings],
) -> CompiledDeployment:
    roles = tuple(artifacts)
    return CompiledDeployment(
        execution_contract=_request_contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(soc=target_soc, runtime="acl"),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts=artifacts,
        execution=roles,
        bindings=bindings,
    )


def _identity(model_type: str, preprocessing: str, output_semantics: str, embedding=None) -> SemanticIdentity:
    return SemanticIdentity(
        logical_model_revision=model_type,
        preprocessing_contract=preprocessing,
        output_semantics=output_semantics,
        embedding=embedding,
    )


def _specs() -> dict[str, BundleSpec]:
    siglip_embedding = EmbeddingMetadata(
        embedding_space_id="google/siglip2-so400m-patch14-384@main",
        dimension=1152,
        normalization="l2",
        image_preprocessing="masked-crop-gray127-resize384-bilinear-normalize0.5-v1",
        text_preprocessing="photo-template-gemma-tokenizer-max64-v1",
    )
    return {
        "sam2": BundleSpec(
            model_type="sam2",
            name="sam2.1_hiera_tiny",
            adapter={
                "interface": "tensor_model",
                "model_type": "sam2",
                "operation": "automatic",
                "preprocessing": SAM2Adapter.identity.preprocessing,
                "postprocessing": SAM2Adapter.identity.postprocessing,
                "torch_module_loader": "perception_service.torch_model_loaders:load_sam2",
                "checkpoint": "assets/sam2.1_hiera_tiny.pt",
                "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
                "points_per_batch": 16,
                "points_per_side": 64,
                "pred_iou_thresh": 0.72,
                "stability_score_thresh": 0.90,
            },
            model=ModelDescriptor(
                interface="tensor_model",
                model_type="sam2",
                operation="automatic",
                inputs=(_tensor("observation.image", "uint8", (-1, -1, 3)),),
                outputs=(
                    _tensor("masks", "uint8", (-1, -1, -1)),
                    _tensor("boxes", "float32", (-1, 4)),
                    _tensor("scores", "float32", (-1,)),
                    _tensor("stability_scores", "float32", (-1,)),
                ),
                semantic_identity=_identity(
                    "sam2.1-hiera-tiny@092824",
                    SAM2Adapter.identity.preprocessing,
                    SAM2Adapter.identity.postprocessing,
                ),
            ),
            required_paths=("assets/sam2.1_hiera_tiny.pt",),
        ),
        "ram_plus": BundleSpec(
            model_type="ram_plus",
            name="ram_plus_swin_large_14m",
            adapter={
                "interface": "tensor_model",
                "model_type": "ram_plus",
                "operation": "recognize_tags",
                "preprocessing": RAM_PLUS_PREPROCESSING,
                "postprocessing": RAM_PLUS_POSTPROCESSING,
                "torch_module_loader": "perception_service.torch_model_loaders:load_ram_plus",
                "checkpoint": "assets/ram_plus_swin_large_14m.pth",
                "text_encoder": "assets/bert-base-uncased",
            },
            model=ModelDescriptor(
                interface="tensor_model",
                model_type="ram_plus",
                operation="recognize_tags",
                inputs=(_tensor("observation.image", "float32", (-1, 3, 384, 384), "NCHW"),),
                outputs=(_tensor("tag_logits", "float32", (-1, 4585)),),
                semantic_identity=_identity(
                    "ram-plus-swin-large-14m@v1", RAM_PLUS_PREPROCESSING, RAM_PLUS_POSTPROCESSING
                ),
            ),
            required_paths=(
                "assets/ram_plus_swin_large_14m.pth",
                "assets/bert-base-uncased/config.json",
                "assets/ram_tag_list.txt",
                "assets/ram_tag_list_threshold.txt",
            ),
            compiled=(
                CompiledSpec(
                    deployment="ascend_310p",
                    source="candidates/ascend_310p/ram_plus_310p.om",
                    artifact="artifacts/ascend_310p/ram_plus_310p.om",
                    target_soc="Ascend310P1",
                    bindings=RAM_PLUS_BINDINGS,
                ),
                CompiledSpec(
                    deployment="ascend_310b",
                    source="candidates/ascend_310b/ram_plus_swin_large_14m_fp16.om",
                    artifact="artifacts/ascend_310b/ram_plus_swin_large_14m_fp16.om",
                    target_soc="Ascend310B1",
                    bindings=RAM_PLUS_BINDINGS,
                ),
            ),
        ),
        "siglip2": BundleSpec(
            model_type="siglip2",
            name="siglip2_so400m_patch14_384",
            adapter={
                "interface": "tensor_model",
                "model_type": "siglip2",
                "operation": "encode",
                "preprocessing": SigLIP2ImageAdapter.identity.preprocessing,
                "postprocessing": SigLIP2ImageAdapter.identity.postprocessing,
                "torch_module_loader": "perception_service.torch_model_loaders:load_siglip2",
                "model_path": "assets/model",
            },
            model=ModelDescriptor(
                interface="tensor_model",
                model_type="siglip2",
                operation="encode",
                inputs=(
                    _tensor("masked_images", "float32", (-1, 3, 384, 384), "NCHW"),
                    _tensor("text_tokens", "int64", (-1, 64)),
                    _tensor("text_attention_mask", "int64", (-1, 64)),
                ),
                outputs=(
                    _tensor("image_embeddings", "float32", (-1, 1152)),
                    _tensor("text_embeddings", "float32", (-1, 1152)),
                ),
                semantic_identity=_identity(
                    "google/siglip2-so400m-patch14-384@main",
                    SigLIP2ImageAdapter.identity.preprocessing,
                    SigLIP2ImageAdapter.identity.postprocessing,
                    siglip_embedding,
                ),
            ),
            required_paths=("assets/model/config.json", "assets/model/model.safetensors"),
        ),
        "grounded_sam2": BundleSpec(
            model_type="grounding_dino",
            name="grounded_sam2_swint_ogc",
            adapter={
                "interface": "tensor_model",
                "model_type": "grounding_dino",
                "operation": "detect",
                "preprocessing": GroundingDINOAdapter.identity.preprocessing,
                "postprocessing": GroundingDINOAdapter.identity.postprocessing,
                "torch_module_loader": "perception_service.torch_model_loaders:load_grounded_sam2",
                "gdino_checkpoint": "assets/groundingdino_swint_ogc.pth",
                "text_encoder": "assets/bert-base-uncased",
                "sam_checkpoint": "assets/sam2.1_hiera_tiny.pt",
                "sam_config": "configs/sam2.1/sam2.1_hiera_t.yaml",
            },
            model=ModelDescriptor(
                interface="tensor_model",
                model_type="grounding_dino",
                operation="detect",
                inputs=(
                    _tensor("observation.image", "uint8", (-1, -1, 3)),
                    _tensor("text_prompt", "uint8", (-1,)),
                    _tensor("box_threshold", "float32", (1,)),
                    _tensor("text_threshold", "float32", (1,)),
                ),
                outputs=(
                    _tensor("boxes", "float32", (-1, 4)),
                    _tensor("scores", "float32", (-1,)),
                    _tensor("masks", "uint8", (-1, -1, -1)),
                    _tensor("label_indices", "int32", (-1,)),
                ),
                semantic_identity=_identity(
                    "groundingdino-swint-ogc+sam2.1-hiera-tiny@v1",
                    GroundingDINOAdapter.identity.preprocessing,
                    GroundingDINOAdapter.identity.postprocessing,
                ),
            ),
            required_paths=(
                "assets/groundingdino_swint_ogc.pth",
                "assets/bert-base-uncased/config.json",
                "assets/sam2.1_hiera_tiny.pt",
            ),
        ),
    }


def _asset_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(path.relative_to(root))
            for path in (root / "assets").rglob("*")
            if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    )


def _compiled_artifact_source(root: Path, compiled: CompiledSpec) -> Path | None:
    artifact = root / compiled.artifact
    source = root.parent / "_work" / root.name / compiled.source
    return source if source.is_file() else artifact if artifact.is_file() else None


def package_bundle(root: Path, spec: BundleSpec) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        existing = load_inference_manifest_metadata(
            root, next(iter(json.loads(manifest_path.read_text())["deployments"]))
        ).manifest
    for relative in spec.required_paths:
        if not (root / relative).is_file():
            raise FileNotFoundError(f"required {spec.model_type} bundle asset is missing: {root / relative}")

    torch_deployments = {"torch_cpu": "cpu", "torch_cuda": "cuda"}
    published = PublishedArtifactIndex(root, existing)
    compiled_artifacts = {compiled.deployment: _compiled_artifact_source(root, compiled) for compiled in spec.compiled}
    candidates = {}
    publications = {}
    for compiled in spec.compiled:
        source = compiled_artifacts[compiled.deployment]
        previous = existing.deployments.get(compiled.deployment) if existing else None
        if source is None and previous is not None:
            raise ManifestIntegrityError(
                f"{ARTIFACT_INTEGRITY_UNVERIFIED}: published deployment {compiled.deployment!r} "
                f"is missing artifact {compiled.artifact!r} and its compiler source"
            )
        if source is not None:
            publication = classify_om_artifacts(root, {"model": (compiled.artifact, source)}, published)
            publications[compiled.deployment] = publication
            candidate = _ascend_deployment(
                target_soc=compiled.target_soc,
                artifacts=publication.artifacts,
                bindings={"model": compiled.bindings},
            )
            candidates[compiled.deployment] = candidate

    adapter_relative = "assets/adapter.json"
    digest_relative = "assets/artifact-digests.json"
    removed_assets = ("assets/GroundingDINO_SwinT_OGC.py",) if spec.model_type == "grounding_dino" else ()
    asset_writes = (adapter_relative, digest_relative, *removed_assets)
    preserved, publications = preflight_publication(
        root,
        published,
        publications,
        set(candidates) | set(torch_deployments),
        asset_writes=asset_writes,
        retained_assets=set(spec.required_paths) | (set(_asset_paths(root)) - set(asset_writes)),
    )

    # Every retained artifact and alias dependency passed before any files changed.
    for relative in removed_assets:
        (root / relative).unlink(missing_ok=True)
    adapter_path = root / adapter_relative
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(json.dumps(spec.adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest_path = root / digest_relative
    asset_paths = tuple(path for path in _asset_paths(root) if path != digest_relative)
    digests = {path: sha256_file(root / path) for path in asset_paths}
    for compiled in spec.compiled:
        source = compiled_artifacts[compiled.deployment]
        if source is not None:
            candidate = candidates[compiled.deployment]
            candidate = copy_and_finalize_artifact_roles(root, candidate, publications[compiled.deployment])
            candidates[compiled.deployment] = candidate
            digests[compiled.artifact] = candidate.artifacts["model"].sha256
    previous_digests = digest_path.read_text(encoding="utf-8") if digest_path.is_file() else ""
    digest_document = json.dumps(digests, indent=2, sort_keys=True) + "\n"
    digest_path.write_text(digest_document, encoding="utf-8")

    files = tuple(BundleFile(path=path) for path in _asset_paths(root))
    structure_changed = existing is not None and (
        existing.bundle.name != spec.name
        or existing.bundle.files != files
        or existing.model != spec.model
        or previous_digests != digest_document
    )
    bundle_uuid = existing.bundle.uuid if existing is not None else str(uuid4())
    bundle_revision = existing.bundle.revision + int(structure_changed) if existing is not None else 1
    deployments = dict(preserved)
    for name, device in torch_deployments.items():
        candidate = _torch_deployment(device)
        previous = existing.deployments.get(name) if existing is not None else None
        candidate = inherit_deployment_version(candidate, previous)
        deployments[name] = candidate
    for compiled in spec.compiled:
        name = compiled.deployment
        compiled_artifact = compiled_artifacts[name]
        if compiled_artifact is None:
            deployments.pop(name, None)
        else:
            candidate = candidates[name]
            previous = existing.deployments.get(name) if existing is not None else None
            candidate = inherit_deployment_version(candidate, previous)
            deployments[name] = candidate
    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name=spec.name,
            files=files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, spec.name, files),
            ),
        ),
        model=spec.model,
        deployments=deployments,
    )
    write_inference_manifest(manifest_path, manifest)
    for deployment in sorted(deployments):
        load_inference_manifest(root, deployment)
    integrity_report_path(root).unlink(missing_ok=True)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--model-type", choices=tuple(_specs()) + ("all",), default="all")
    args = parser.parse_args()
    specs = _specs()
    selected = specs if args.model_type == "all" else {args.model_type: specs[args.model_type]}
    for spec in selected.values():
        print(package_bundle(args.models_root / spec.name, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

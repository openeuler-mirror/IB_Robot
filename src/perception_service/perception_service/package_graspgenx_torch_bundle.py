"""Write the schema-v3 bundle used by the CUDA GraspGenX executor.

A GraspGenX bundle pairs the gripper-independent checkpoints with exactly one gripper
description, because the deployed service answers for one physical end effector. The
gripper is copied into the bundle so the runtime never resolves it from the network or
from the official package's own ``ext/`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from uuid import uuid4

import yaml

from inference_manifest import (
    BundleFile,
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
    TorchDeployment,
    TorchRuntimeProfile,
    canonical_bundle_digest,
    load_inference_manifest,
    load_inference_manifest_metadata,
    write_inference_manifest,
)
from model_utils.graspgen_contract import (
    GRASPGEN_CONFIDENCE_SEMANTIC,
    GRASPGEN_POINT_CLOUD_SEMANTIC,
    GRASPGEN_POSE_SEMANTIC,
)
from model_utils.graspgenx_export.binding import generator_binding, validate_binding

from .graspgenx_adapter import (
    GRASPGENX_POSTPROCESSING,
    GRASPGENX_PREPROCESSING,
    GRASPGENX_SAMPLE_COUNT_SEMANTIC,
    GraspGenXConfig,
)

# Bumped whenever the bundle layout or the service semantics change incompatibly.
GRASPGENX_CONTRACT_VERSION = 1

_ADAPTER_ASSET = "assets/adapter.json"
_CHECKPOINTS = "assets/graspgenx"
_GRIPPERS = "assets"
_GENERATOR = f"{_CHECKPOINTS}/gen/generator.pth"
_DISCRIMINATOR = f"{_CHECKPOINTS}/dis/discriminator.pth"
# The offline diffusion loop lives outside assets/ on purpose: _bundle_files
# walks assets/, and a path may be in bundle.files or in a deployment's
# artifacts but never both. Only the artifacts map carries a sha256, so this is
# also the side that gets its content verified rather than only its name.
_DIFFUSION_LOOP_DIR = "artifacts/ascend_310p"
_DIFFUSION_LOOP = f"{_DIFFUSION_LOOP_DIR}/diffusion_loop.om"
_DIFFUSION_LOOP_METADATA = f"{_DIFFUSION_LOOP_DIR}/diffusion_loop.json"
_EPOCH = re.compile(r"epoch_(\d+)\.pth$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_checkpoint(directory: Path) -> Path:
    """The official release ships one ``epoch_<n>.pth`` per network; take the newest."""
    candidates = sorted(
        (int(match.group(1)), path)
        for path in directory.glob("*.pth")
        if (match := _EPOCH.search(path.name)) is not None
    )
    if not candidates:
        raise FileNotFoundError(f"no epoch_<n>.pth checkpoint in {directory}")
    return candidates[-1][1]


def _copy_if_changed(source: Path, target: Path) -> None:
    """Copy unless ``target`` already holds exactly this content.

    Size and mtime are not enough to decide: two checkpoints of the same model
    have the same size by construction, so mtime alone separates them, and any
    copy that preserves it (``cp -p``, ``rsync -t``, an unpacked archive) can
    present new content under the old timestamp. The skip would then be silent
    and the manifest digests, computed later from the destination, would record
    and attest the stale checkpoint. Size is still checked first because it
    rules out most pairs without reading anything; only a size match pays for
    hashing, and only the no-op case hashes twice.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and source.stat().st_size == target.stat().st_size and _sha256(source) == _sha256(target):
        return
    shutil.copy2(source, target)


def _materialize(bundle_root: Path, checkpoint_root: Path, gripper_dir: Path, gripper_name: str) -> None:
    for network, checkpoint in (("gen", _GENERATOR), ("dis", _DISCRIMINATOR)):
        source = checkpoint_root / network
        if not (source / "config.yaml").is_file():
            raise FileNotFoundError(f"GraspGenX {network} config is missing: {source / 'config.yaml'}")
        _copy_if_changed(source / "config.yaml", bundle_root / _CHECKPOINTS / network / "config.yaml")
        _copy_if_changed(_latest_checkpoint(source), bundle_root / checkpoint)
    if not (gripper_dir / "config.json").is_file():
        raise FileNotFoundError(f"GraspGenX gripper description has no config.json: {gripper_dir}")
    # A bundle binds exactly one gripper: drop every other description, including
    # one left by an earlier package run under a different gripper name.
    grippers = bundle_root / _GRIPPERS / "x_grippers"
    if grippers.exists():
        shutil.rmtree(grippers)
    shutil.copytree(gripper_dir, grippers / gripper_name, copy_function=shutil.copy2)


def _model_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        interface="tensor_model",
        model_type="graspgenx",
        operation="generate_grasps",
        inputs=(
            SemanticTensor(semantic=GRASPGEN_POINT_CLOUD_SEMANTIC, dtype="float32", shape=(-1, 3)),
            SemanticTensor(semantic=GRASPGENX_SAMPLE_COUNT_SEMANTIC, dtype="int64", shape=(1,)),
        ),
        outputs=(
            SemanticTensor(semantic=GRASPGEN_POSE_SEMANTIC, dtype="float32", shape=(-1, 4, 4)),
            SemanticTensor(semantic=GRASPGEN_CONFIDENCE_SEMANTIC, dtype="float32", shape=(-1,)),
        ),
        semantic_identity=SemanticIdentity(
            logical_model_revision=f"graspgenx@v{GRASPGENX_CONTRACT_VERSION}",
            preprocessing_contract=GRASPGENX_PREPROCESSING,
            output_semantics=GRASPGENX_POSTPROCESSING,
        ),
    )


def _validate_diffusion_loop(source: Path, num_grasps: int, binding: dict) -> None:
    """Copy the offline diffusion loop in, refusing one the bundle cannot use.

    The graph is compiled for one grasp count and one step count. A mismatch is
    not an error at run time -- the generator declines the offline loop and
    falls back to the eager one -- so it would show up only as the deployment
    being inexplicably slow. Catching it here is the difference between a
    packaging failure and a silent performance cliff.
    """
    metadata_source = source.with_suffix(".json")
    if not source.is_file():
        raise FileNotFoundError(f"offline diffusion loop not found: {source}")
    if not metadata_source.is_file():
        raise FileNotFoundError(f"the offline loop's metadata must sit beside it as {metadata_source.name}")
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    validate_binding(metadata, binding, num_grasps)


def _materialize_diffusion_loop(bundle_root: Path, source: Path) -> None:
    destination = bundle_root / _DIFFUSION_LOOP
    destination.parent.mkdir(parents=True, exist_ok=True)
    _copy_if_changed(source, destination)
    _copy_if_changed(source.with_suffix(".json"), bundle_root / _DIFFUSION_LOOP_METADATA)


def _bundle_files(bundle_root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(bundle_root).as_posix()
            for path in (bundle_root / "assets").rglob("*")
            if path.is_file() and path.relative_to(bundle_root).as_posix() != _ADAPTER_ASSET
        )
    )


def _object_point_budget(config: Path) -> int:
    """The object-cloud size this checkpoint was trained on, from its own config.

    Upstream resamples a served cloud to ``cfg.data.num_points`` to match the training
    distribution (``graspgenx.grasp_server.run_inference_batch``), so the budget belongs
    to the checkpoint, not to this packager. It is copied into adapter.json rather than
    re-read at serve time, the way num_grasps and topk already are; adapter.json's
    source_sha256 covers gen/config.yaml, so editing the config later changes the bundle
    revision.
    """
    data = yaml.safe_load(config.read_text(encoding="utf-8")).get("data") or {}
    budget = data.get("num_points")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError(f"{config} must declare a positive data.num_points, got {budget!r}")
    return budget


def package_graspgenx_torch_bundle(
    bundle_root: Path,
    *,
    checkpoint_root: Path | None = None,
    gripper_dir: Path | None = None,
    gripper_name: str = "omnihand_left_x2",
    num_grasps: int = 400,
    topk: int = 400,
    deployment_name: str = "torch_cuda",
    device: str = "cuda",
    diffusion_loop: Path | None = None,
) -> Path:
    """Create or update a bundle; sources are optional when the bundle already holds them."""
    bundle_root.mkdir(parents=True, exist_ok=True)
    if device not in {"cuda", "npu"}:
        raise ValueError(f"GraspGenX serves cuda and npu deployments, not {device!r}")
    if (device == "npu") != (diffusion_loop is not None):
        raise ValueError("an npu deployment needs --diffusion-loop, and no other deployment takes one")
    if (checkpoint_root is None) != (gripper_dir is None):
        raise ValueError("pass both --checkpoint-root and --gripper-dir, or neither to repackage in place")
    # Preflight the prospective sources before copying any checkpoints or
    # rewriting adapter/manifest digests, so a refused update leaves the bundle as
    # it was: first the point budget the adapter will declare, then every retained
    # NPU deployment (updating only CUDA still changes the assets an existing OM shares).
    config = checkpoint_root / "gen/config.yaml" if checkpoint_root else bundle_root / _CHECKPOINTS / "gen/config.yaml"
    # A bundle missing its config is reported with everything else it lacks below.
    object_points = _object_point_budget(config) if config.is_file() else None
    loops = [diffusion_loop] if diffusion_loop is not None else []
    manifest_path = bundle_root / "inference_manifest.json"
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, deployment in raw["deployments"].items():
            artifact = deployment.get("artifacts", {}).get("diffusion_loop")
            if artifact and name != deployment_name:
                path = bundle_root / artifact["path"]
                loops.append(
                    diffusion_loop if diffusion_loop is not None and path == bundle_root / _DIFFUSION_LOOP else path
                )
    if loops:
        checkpoint = _latest_checkpoint(checkpoint_root / "gen") if checkpoint_root else bundle_root / _GENERATOR
        binding = generator_binding(checkpoint, config)
        for loop in loops:
            _validate_diffusion_loop(loop, num_grasps, binding)
    if checkpoint_root is not None and gripper_dir is not None:
        _materialize(bundle_root, checkpoint_root, gripper_dir, gripper_name)
    required = (
        f"{_CHECKPOINTS}/gen/config.yaml",
        _GENERATOR,
        f"{_CHECKPOINTS}/dis/config.yaml",
        _DISCRIMINATOR,
        f"{_GRIPPERS}/x_grippers/{gripper_name}/config.json",
    )
    missing = [relative for relative in required if not (bundle_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"GraspGenX bundle is missing {missing}; provide --checkpoint-root and --gripper-dir")

    if diffusion_loop is not None:
        _materialize_diffusion_loop(bundle_root, diffusion_loop)

    if object_points is None:
        object_points = _object_point_budget(bundle_root / _CHECKPOINTS / "gen/config.yaml")
    adapter = {
        "interface": "tensor_model",
        "model_type": "graspgenx",
        "operation": "generate_grasps",
        "preprocessing": GRASPGENX_PREPROCESSING,
        "postprocessing": GRASPGENX_POSTPROCESSING,
        "torch_module_loader": "perception_service.torch_model_loaders:load_graspgenx",
        "gripper_name": gripper_name,
        "object_points": object_points,
        "num_grasps": num_grasps,
        "topk": topk,
        "checkpoints_dir": _CHECKPOINTS,
        "gripper_assets_dir": _GRIPPERS,
        "generator_checkpoint": _GENERATOR,
        "discriminator_checkpoint": _DISCRIMINATOR,
        # Every bundle file, so a changed gripper mesh changes adapter.json and with
        # it the bundle revision; bundle.files and the structure digest hold paths only.
        "source_sha256": {relative: _sha256(bundle_root / relative) for relative in _bundle_files(bundle_root)},
    }
    GraspGenXConfig.from_assets(adapter)
    adapter_document = json.dumps(adapter, indent=2, sort_keys=True) + "\n"
    adapter_path = bundle_root / _ADAPTER_ASSET
    previous_adapter = adapter_path.read_text(encoding="utf-8") if adapter_path.is_file() else ""
    adapter_path.write_text(adapter_document, encoding="utf-8")
    files = tuple(BundleFile(path=relative) for relative in (*_bundle_files(bundle_root), _ADAPTER_ASSET))

    manifest_path = bundle_root / "inference_manifest.json"
    existing = None
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Metadata only: the previous file list may name assets this run replaced.
        existing = load_inference_manifest_metadata(bundle_root, next(iter(raw["deployments"]))).manifest
    model = _model_descriptor()
    if existing is None:
        bundle_uuid, revision = str(uuid4()), 1
    else:
        changed = existing.bundle.files != files or existing.model != model or previous_adapter != adapter_document
        bundle_uuid, revision = existing.bundle.uuid, existing.bundle.revision + int(changed)
    # The offline loop is declared as a verified artifact but left out of
    # `execution`: this backend is torch, and the graph is driven by the model
    # the torch_module_loader builds, not by an ACL session. An empty execution
    # list is what lets a torch deployment carry it at all.
    artifacts = {}
    if device == "npu":
        artifacts = {
            "diffusion_loop": DeploymentArtifact(
                path=_DIFFUSION_LOOP,
                format="om",
                sha256=_sha256(bundle_root / _DIFFUSION_LOOP),
            ),
            "diffusion_loop_metadata": DeploymentArtifact(
                path=_DIFFUSION_LOOP_METADATA,
                format="json",
                sha256=_sha256(bundle_root / _DIFFUSION_LOOP_METADATA),
            ),
        }
    deployment = TorchDeployment(
        execution_contract=ExecutionContract(
            state_scope="request",
            execution_structure="direct",
            cancellation_granularity="request_boundary",
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device=device),
        ),
        artifacts=artifacts,
    )
    previous_deployment = existing.deployments.get(deployment_name) if existing is not None else None
    if isinstance(previous_deployment, TorchDeployment):
        deployment = deployment.model_copy(
            update={"uuid": previous_deployment.uuid, "revision": previous_deployment.revision}
        )
    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=revision,
            name=bundle_root.name,
            files=files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, revision, bundle_root.name, files),
            ),
        ),
        model=model,
        deployments={**(existing.deployments if existing is not None else {}), deployment_name: deployment},
    )
    write_inference_manifest(manifest_path, manifest)
    load_inference_manifest(bundle_root, deployment_name)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="GraspGenX release directory containing gen/ and dis/ (config.yaml + epoch_<n>.pth each).",
    )
    parser.add_argument(
        "--gripper-dir",
        type=Path,
        default=None,
        help="One gripper description directory (config.json, gripper.urdf, meshes) to bind into the bundle.",
    )
    parser.add_argument("--gripper-name", default="omnihand_left_x2")
    parser.add_argument("--num-grasps", type=int, default=400)
    parser.add_argument("--topk", type=int, default=400)
    parser.add_argument("--deployment", default="torch_cuda")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "npu"),
        help="Torch device this deployment targets. An npu deployment also needs --diffusion-loop.",
    )
    parser.add_argument(
        "--diffusion-loop",
        type=Path,
        default=None,
        help=(
            "Offline diffusion-loop model (.om) for an npu deployment, with its metadata "
            "JSON beside it. It is declared as a sha256-verified artifact, and its grasp "
            "count must match --num-grasps."
        ),
    )
    args = parser.parse_args()
    print(
        package_graspgenx_torch_bundle(
            args.bundle_root,
            checkpoint_root=args.checkpoint_root,
            gripper_dir=args.gripper_dir,
            gripper_name=args.gripper_name,
            num_grasps=args.num_grasps,
            topk=args.topk,
            deployment_name=args.deployment,
            device=args.device,
            diffusion_loop=args.diffusion_loop,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

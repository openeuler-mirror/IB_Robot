"""Exporter helpers for producing strict unified inference manifests.

This module is a manifest-packaging tool boundary.  ACL ABI inspection lives
in the separate ``acl_abi_inspection`` module so its optional initialization
configuration cannot become part of ``InferenceManifest``,
``DeploymentTarget``, or a runtime profile.  Runtime code imports the
hardware-independent ``inference_manifest`` package and never imports either
tool module.
"""

from __future__ import annotations

import errno
import fcntl
import filecmp
import json
import shutil
import stat
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import onnx

from inference_manifest import (
    ArtifactBindings,
    AscendRuntimeProfile,
    BackendRuntimeProfile,
    BundleFile,
    CompiledDeployment,
    Deployment,
    DeploymentArtifact,
    DeploymentTarget,
    DeviceLink,
    Digest,
    ExecutionContract,
    HisiliconRuntimeProfile,
    HMMRuntimeProfile,
    InferenceManifest,
    ManifestBundle,
    ModelDescriptor,
    RKNNRuntimeProfile,
    RoleRuntimeProfile,
    SemanticTensor,
    StateLink,
    TensorBinding,
    TorchRuntimeProfile,
    ValidatedManifest,
    canonical_bundle_digest,
    is_image_semantic,
    load_inference_manifest,
    load_policy_metadata,
    normalize_bundle_path,
    write_inference_manifest,
)
from inference_manifest.json_utils import load_json_strict
from inference_manifest.schema import validate_manifest_schema

_ONNX_DTYPES = {
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.BFLOAT16: "bfloat16",
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.DOUBLE: "float64",
}


@dataclass(frozen=True)
class RuntimeTensor:
    """One positional tensor exposed by a compiled runtime ABI."""

    name: str
    index: int
    dtype: str
    shape: tuple[int, ...]
    layout: str | None = None


@dataclass(frozen=True)
class RuntimeABI:
    """Input and output tensors in runtime order."""

    inputs: tuple[RuntimeTensor, ...]
    outputs: tuple[RuntimeTensor, ...]


def read_onnx_abi(path: str | Path) -> RuntimeABI:
    """Read the positional tensor ABI from an ONNX graph."""

    model_path = Path(path).expanduser().resolve(strict=True)
    model = onnx.load(str(model_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    inputs = tuple(
        _onnx_tensor(value, index, model_path)
        for index, value in enumerate(item for item in model.graph.input if item.name not in initializer_names)
    )
    outputs = tuple(_onnx_tensor(value, index, model_path) for index, value in enumerate(model.graph.output))
    if not inputs or not outputs:
        raise ValueError(f"ONNX ABI must contain at least one input and one output: {model_path}")
    return RuntimeABI(inputs=inputs, outputs=outputs)


def read_runtime_abi(path: str | Path) -> RuntimeABI:
    """Read runtime-introspected ABI metadata emitted by a compiler toolchain."""

    metadata_path = Path(path).expanduser().resolve(strict=True)
    with metadata_path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Runtime ABI metadata must be a JSON object: {metadata_path}")
    return RuntimeABI(
        inputs=_parse_runtime_tensors(value.get("inputs"), "inputs", metadata_path),
        outputs=_parse_runtime_tensors(value.get("outputs"), "outputs", metadata_path),
    )


def read_tcim_abi(path: str | Path) -> RuntimeABI:
    """Read TCIM compiler ``model.json`` input/output descriptors."""

    metadata_path = Path(path).expanduser().resolve(strict=True)
    with metadata_path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    model = value.get("Model") if isinstance(value, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"TCIM metadata must contain a Model object: {metadata_path}")
    return RuntimeABI(
        inputs=_parse_tcim_tensors(model.get("inputs"), "inputs", metadata_path),
        outputs=_parse_tcim_tensors(model.get("outputs"), "outputs", metadata_path),
    )


def artifact_bindings(
    abi: RuntimeABI,
    *,
    input_semantics: Mapping[str, str],
    output_semantics: Mapping[str, str],
    image_layouts: Mapping[str, str] | None = None,
) -> ArtifactBindings:
    """Map every runtime ABI tensor to one explicit manifest semantic."""

    layouts = dict(image_layouts or {})
    known_semantics = set(input_semantics.values()) | set(output_semantics.values())
    unexpected_layouts = sorted(set(layouts) - known_semantics)
    if unexpected_layouts:
        raise ValueError(f"Layouts reference unknown semantics: {unexpected_layouts}")
    return ArtifactBindings(
        inputs=tuple(_binding(tensor, input_semantics, layouts, "input") for tensor in abi.inputs),
        outputs=tuple(_binding(tensor, output_semantics, layouts, "output") for tensor in abi.outputs),
    )


def deployment_artifact(
    bundle_root: str | Path,
    path: str | Path,
    artifact_format: str,
    *,
    share_group: str | None = None,
) -> DeploymentArtifact:
    """Create a structural artifact descriptor safely contained by the bundle."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    artifact_path = artifact_path.resolve(strict=True)
    if not artifact_path.is_file():
        raise ValueError(f"Deployment artifact is not a regular file: {artifact_path}")
    try:
        relative = artifact_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Deployment artifact escapes bundle root {root}: {artifact_path}") from exc
    return DeploymentArtifact(
        path=normalize_bundle_path(relative),
        format=artifact_format,
        share_group=share_group,
    )


def package_deployment_artifact(
    bundle_root: str | Path,
    source_path: str | Path,
    *,
    backend: str,
    deployment_name: str,
    role: str,
    force_copy: bool = False,
    prefer_hardlink: bool = False,
) -> Path:
    """Copy a compiler output into one immutable artifact generation when needed."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    source = Path(source_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Deployment artifact is not a regular file: {source}")
    del force_copy  # Kept for CLI compatibility; normal publication always creates a generation.
    suffix = "".join(source.suffixes)
    artifacts_root = root / "artifacts" / backend
    manifest = _load_existing_manifest(root / "inference_manifest.json")
    if manifest is not None:
        deployment = manifest.deployments.get(deployment_name)
        if isinstance(deployment, CompiledDeployment) and deployment.backend == backend:
            current = deployment.artifacts.get(role)
            if current is not None:
                current_path = root.joinpath(*current.path.split("/"))
                if _same_file(source, current_path):
                    return current_path
    if artifacts_root.is_dir():
        for existing in artifacts_root.glob(f"*/generations/*/{role}{suffix}"):
            if _same_file(source, existing):
                return existing
    generation = str(uuid4())
    relative = normalize_bundle_path(f"artifacts/{backend}/{deployment_name}/generations/{generation}/{role}{suffix}")
    destination = root.joinpath(*relative.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4()}.tmp")
    try:
        if prefer_hardlink:
            try:
                temporary.hardlink_to(source)
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                    raise
                shutil.copy2(source, temporary)
        else:
            shutil.copy2(source, temporary)
        if stat.S_IMODE(source.stat().st_mode) != stat.S_IMODE(temporary.stat().st_mode):
            temporary.chmod(stat.S_IMODE(source.stat().st_mode))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def compiled_deployment(
    bundle_root: str | Path,
    *,
    backend: str,
    target_soc: str,
    target_runtime: str,
    runtime_abi: str | None = None,
    artifacts: Mapping[str, tuple[str | Path, str]],
    execution: Sequence[str],
    bindings: Mapping[str, ArtifactBindings],
    device_links: Sequence[DeviceLink] = (),
    state_links: Mapping[str, Sequence[StateLink]] | None = None,
    artifact_share_groups: Mapping[str, str] | None = None,
    runtime_profile: BackendRuntimeProfile | None = None,
    execution_contract: ExecutionContract | None = None,
) -> CompiledDeployment:
    """Build a typed compiled deployment from structural artifact descriptors."""

    if state_links is not None:
        raise ValueError("state_links must be declared as typed execution_contract.state_links")
    share_groups = dict(artifact_share_groups or {})
    target = DeploymentTarget(soc=target_soc, runtime=target_runtime, runtime_abi=runtime_abi)
    profile = runtime_profile or _default_runtime_profile(backend, target_soc)
    return CompiledDeployment(
        execution_contract=execution_contract or _request_direct_contract(),
        runtime_profile=RoleRuntimeProfile(backend=backend, target=target, profile=profile),
        artifacts={
            role: deployment_artifact(
                bundle_root,
                path,
                artifact_format,
                share_group=share_groups.get(role),
            )
            for role, (path, artifact_format) in artifacts.items()
        },
        execution=tuple(execution),
        bindings=dict(bindings),
        device_links=tuple(device_links),
    )


def _request_direct_contract() -> ExecutionContract:
    return ExecutionContract(
        state_scope="request",
        execution_structure="direct",
        cancellation_granularity="request_boundary",
    )


def _default_runtime_profile(backend: str, target_soc: str) -> BackendRuntimeProfile:
    """Return the smallest typed profile needed by a packaged compiled deployment."""

    if backend == "ascend":
        return AscendRuntimeProfile(device_id=0)
    if backend == "rknn":
        return RKNNRuntimeProfile(target_name=target_soc, device_id=0)
    if backend == "hmm":
        return HMMRuntimeProfile(role="policy", tcim_abi="tcim-v1", device_id=0)
    if backend == "hisilicon":
        return HisiliconRuntimeProfile(protocol="sd3403")
    if backend == "torch":
        return TorchRuntimeProfile(device="cpu")
    raise ValueError(f"unsupported compiled deployment backend {backend!r}")


def _policy_model_descriptor(policy, *, architecture_class: str | None = None) -> ModelDescriptor:
    def tensor(semantic: str, feature) -> SemanticTensor:
        feature_type = feature.type.upper()
        dtype = "int64" if feature_type in {"LANGUAGE", "TEXT", "TOKEN", "TOKENS"} else "float32"
        return SemanticTensor(semantic=semantic, dtype=dtype, shape=feature.shape)

    return ModelDescriptor(
        interface="policy",
        model_type=policy.policy_type,
        operation="predict",
        inputs=tuple(tensor(name, feature) for name, feature in policy.input_features.items()),
        outputs=tuple(tensor(name, feature) for name, feature in policy.output_features.items()),
        architecture_class=architecture_class,
    )


def upsert_deployment(
    bundle_root: str | Path,
    deployment_name: str,
    deployment: Deployment,
    *,
    bundle_name: str | None = None,
    architecture_class: str | None = None,
) -> ValidatedManifest:
    """Atomically update one deployment under a bundle-local writer lock."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    with _manifest_lock(root):
        return _upsert_deployment_unlocked(
            root,
            deployment_name,
            deployment,
            bundle_name=bundle_name,
            architecture_class=architecture_class,
        )


def update_deployment(
    bundle_root: str | Path,
    deployment_name: str,
    deployment: Deployment,
    *,
    expected_uuid: str,
    expected_revision: int,
) -> ValidatedManifest:
    """Publish a deployment only if its lineage still matches the caller's snapshot."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    with _manifest_lock(root):
        existing = _load_existing_manifest(root / "inference_manifest.json")
        if existing is None or deployment_name not in existing.deployments:
            raise ValueError(f"Deployment {deployment_name!r} no longer exists")
        current = existing.deployments[deployment_name]
        if current.uuid != expected_uuid or current.revision != expected_revision:
            raise ValueError(
                f"Deployment {deployment_name!r} revision conflict: expected "
                f"{expected_uuid}@{expected_revision}, got {current.uuid}@{current.revision}"
            )
        return _upsert_deployment_unlocked(root, deployment_name, deployment)


def _upsert_deployment_unlocked(
    bundle_root: str | Path,
    deployment_name: str,
    deployment: Deployment,
    *,
    bundle_name: str | None = None,
    architecture_class: str | None = None,
) -> ValidatedManifest:
    """Write one deployment and prove the production strict loader accepts it."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    manifest_path = root / "inference_manifest.json"
    existing = _load_existing_manifest(manifest_path)
    deployments = dict(existing.deployments) if existing is not None else {}
    previous_deployment = deployments.get(deployment_name)
    if previous_deployment is not None:
        previous_value = previous_deployment.model_dump(mode="json", exclude={"uuid", "revision"})
        candidate_value = deployment.model_dump(mode="json", exclude={"uuid", "revision"})
        deployment = (
            previous_deployment
            if previous_value == candidate_value
            else deployment.model_copy(
                update={"uuid": previous_deployment.uuid, "revision": previous_deployment.revision + 1}
            )
        )
    deployments[deployment_name] = deployment

    require_native_weights = any(candidate.backend == "torch" for candidate in deployments.values())
    policy = load_policy_metadata(root, require_native_weights=require_native_weights)
    if policy.external_dependencies:
        dependencies = [f"{dependency.source}={dependency.identifier!r}" for dependency in policy.external_dependencies]
        raise ValueError(
            "Unified manifest finalization requires all semantic dependencies to be local bundle assets; "
            f"vendor these references inside the bundle and update the LeRobot metadata: {dependencies}"
        )
    bundle_files = tuple(BundleFile(path=path) for path in policy.required_files)
    name = bundle_name or (existing.bundle.name if existing is not None else root.name)
    if existing is None:
        bundle_uuid = str(uuid4())
        bundle_revision = 1
    else:
        bundle_uuid = existing.bundle.uuid
        previous_structure = (existing.bundle.name, tuple(entry.path for entry in existing.bundle.files))
        candidate_structure = (name, tuple(entry.path for entry in bundle_files))
        bundle_revision = existing.bundle.revision + (candidate_structure != previous_structure)
    model = (
        existing.model
        if existing is not None
        else _policy_model_descriptor(
            policy,
            architecture_class=architecture_class,
        )
    )
    if existing is not None and architecture_class is not None:
        model = model.model_copy(update={"architecture_class": architecture_class})
    if model.architecture_class == "pi05-ascend-310p":
        if model.model_type != "pi05":
            raise ValueError(
                f"architecture_class 'pi05-ascend-310p' requires model_type 'pi05'; got {model.model_type!r}"
            )
        config = load_json_strict(root / "config.json")
        inference_steps = config.get("num_inference_steps") if isinstance(config, dict) else None
        if inference_steps != 10:
            raise ValueError(
                f"architecture_class 'pi05-ascend-310p' requires num_inference_steps=10; got {inference_steps!r}"
            )
        incompatible = sorted(
            name for name, candidate in deployments.items() if candidate.backend != "torch" or candidate.device != "npu"
        )
        if incompatible:
            raise ValueError(
                "architecture_class 'pi05-ascend-310p' requires NPU-only Torch deployments; "
                f"incompatible deployments: {incompatible}"
            )
    manifest = InferenceManifest(
        schema_version=3,
        bundle=ManifestBundle(
            uuid=bundle_uuid,
            revision=bundle_revision,
            name=name,
            files=bundle_files,
            digest=Digest(
                algorithm="sha256",
                scope="structure",
                value=canonical_bundle_digest(bundle_uuid, bundle_revision, name, bundle_files),
            ),
        ),
        model=model,
        deployments=deployments,
    )
    if existing is not None and manifest == existing:
        return load_inference_manifest(root, deployment_name)
    try:
        write_inference_manifest(manifest_path, manifest)
        validated_deployments = {name: load_inference_manifest(root, name) for name in sorted(deployments)}
    except Exception:
        if existing is None:
            manifest_path.unlink(missing_ok=True)
        else:
            write_inference_manifest(manifest_path, existing)
        raise
    return validated_deployments[deployment_name]


@contextmanager
def _manifest_lock(root: Path):
    lock_path = root / ".inference_manifest.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def copy_policy_metadata_bundle(source_root: str | Path, destination_root: str | Path) -> tuple[str, ...]:
    """Transactionally publish required LeRobot semantic files."""

    source = Path(source_root).expanduser().resolve(strict=True)
    destination = Path(destination_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    policy = load_policy_metadata(source, require_native_weights=False)
    with TemporaryDirectory(prefix=".metadata-stage-", dir=destination) as temporary:
        staging = Path(temporary) / "new"
        backup = Path(temporary) / "old"
        for relative in policy.required_files:
            source_path = source.joinpath(*relative.split("/"))
            staged_path = staging.joinpath(*relative.split("/"))
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, staged_path)

        with _manifest_lock(destination):
            changed = tuple(
                relative
                for relative in policy.required_files
                if not _same_file(staging.joinpath(*relative.split("/")), destination.joinpath(*relative.split("/")))
            )
            if not changed:
                return policy.required_files

            previously_missing: set[str] = set()
            for relative in changed:
                current = destination.joinpath(*relative.split("/"))
                if current.is_file():
                    saved = backup.joinpath(*relative.split("/"))
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(current, saved)
                else:
                    previously_missing.add(relative)

            try:
                for relative in changed:
                    _atomic_copy(staging.joinpath(*relative.split("/")), destination.joinpath(*relative.split("/")))
                if (destination / "inference_manifest.json").is_file():
                    _refresh_bundle_revision_unlocked(destination)
            except Exception:
                for relative in changed:
                    current = destination.joinpath(*relative.split("/"))
                    if relative in previously_missing:
                        current.unlink(missing_ok=True)
                    else:
                        _atomic_copy(backup.joinpath(*relative.split("/")), current)
                raise
    return policy.required_files


def refresh_bundle_revision(bundle_root: str | Path) -> InferenceManifest:
    """Publish one new bundle revision after semantic assets were deliberately replaced."""

    root = Path(bundle_root).expanduser().resolve(strict=True)
    with _manifest_lock(root):
        return _refresh_bundle_revision_unlocked(root)


def _refresh_bundle_revision_unlocked(root: Path) -> InferenceManifest:
    manifest_path = root / "inference_manifest.json"
    existing = _load_existing_manifest(manifest_path)
    if existing is None:
        raise ValueError(f"Cannot refresh a bundle without inference_manifest.json: {root}")
    require_native_weights = any(candidate.backend == "torch" for candidate in existing.deployments.values())
    policy = load_policy_metadata(root, require_native_weights=require_native_weights)
    bundle_files = tuple(BundleFile(path=path) for path in policy.required_files)
    revision = existing.bundle.revision + 1
    bundle = ManifestBundle(
        uuid=existing.bundle.uuid,
        revision=revision,
        name=existing.bundle.name,
        files=bundle_files,
        digest=Digest(
            algorithm="sha256",
            scope="structure",
            value=canonical_bundle_digest(existing.bundle.uuid, revision, existing.bundle.name, bundle_files),
        ),
    )
    manifest = existing.model_copy(update={"bundle": bundle})
    try:
        write_inference_manifest(manifest_path, manifest)
        for deployment_name in sorted(manifest.deployments):
            load_inference_manifest(root, deployment_name)
    except Exception:
        write_inference_manifest(manifest_path, existing)
        raise
    return manifest


def _same_file(source: Path, destination: Path) -> bool:
    return (
        destination.is_file()
        and stat.S_IMODE(source.stat().st_mode) == stat.S_IMODE(destination.stat().st_mode)
        and filecmp.cmp(source, destination, shallow=False)
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4()}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _onnx_tensor(value: object, index: int, path: Path) -> RuntimeTensor:
    tensor_type = value.type.tensor_type
    try:
        dtype = _ONNX_DTYPES[tensor_type.elem_type]
    except KeyError as exc:
        dtype_name = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        raise ValueError(f"Unsupported ONNX tensor dtype {dtype_name!r} for {value.name!r} in {path}") from exc
    shape = tuple(int(dimension.dim_value) if dimension.dim_value > 0 else -1 for dimension in tensor_type.shape.dim)
    if not shape:
        raise ValueError(f"Scalar ONNX tensor {value.name!r} cannot be represented as a manifest binding: {path}")
    return RuntimeTensor(name=value.name, index=index, dtype=dtype, shape=shape)


def _parse_runtime_tensors(value: object, direction: str, path: Path) -> tuple[RuntimeTensor, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Runtime ABI {direction} must be a non-empty list: {path}")
    tensors: list[RuntimeTensor] = []
    for expected_index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Runtime ABI {direction}[{expected_index}] must be an object: {path}")
        name = item.get("name")
        dtype = item.get("dtype")
        shape = item.get("shape")
        index = item.get("index", expected_index)
        if not isinstance(name, str) or not name:
            raise ValueError(f"Runtime ABI {direction}[{expected_index}] requires a non-empty name: {path}")
        if not isinstance(dtype, str) or dtype not in set(_ONNX_DTYPES.values()):
            raise ValueError(f"Runtime ABI tensor {name!r} has unsupported dtype {dtype!r}: {path}")
        if type(index) is not int or index < 0:
            raise ValueError(f"Runtime ABI tensor {name!r} has invalid index {index!r}: {path}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension == 0 or dimension < -1 for dimension in shape)
        ):
            raise ValueError(f"Runtime ABI tensor {name!r} has invalid shape {shape!r}: {path}")
        layout = item.get("layout")
        if isinstance(layout, str):
            layout = layout.upper()
        if layout is not None and layout not in {"NCHW", "NHWC"}:
            raise ValueError(f"Runtime ABI tensor {name!r} has invalid layout {layout!r}: {path}")
        tensors.append(RuntimeTensor(name=name, index=index, dtype=dtype, shape=tuple(shape), layout=layout))
    indices = [tensor.index for tensor in tensors]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Runtime ABI {direction} contains duplicate indices: {path}")
    if direction == "inputs" and sorted(indices) != list(range(len(indices))):
        raise ValueError(f"Runtime ABI inputs indices must be contiguous from zero: {path}")
    return tuple(tensors)


def _parse_tcim_tensors(value: object, direction: str, path: Path) -> tuple[RuntimeTensor, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"TCIM Model.{direction} must be a non-empty list: {path}")
    tensors: list[RuntimeTensor] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"TCIM Model.{direction}[{index}] must be an object: {path}")
        name = item.get("name")
        shape = item.get("shape")
        dtype = item.get("dtype")
        if not isinstance(name, str) or not name:
            raise ValueError(f"TCIM Model.{direction}[{index}] requires a non-empty name: {path}")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dimension) is not int or dimension < 1 for dimension in shape)
        ):
            raise ValueError(f"TCIM tensor {name!r} has invalid shape {shape!r}: {path}")
        tensors.append(
            RuntimeTensor(
                name=name,
                index=index,
                dtype=_tcim_dtype(dtype, name, path),
                shape=tuple(shape),
            )
        )
    return tuple(tensors)


def _tcim_dtype(value: object, name: str, path: Path) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"TCIM tensor {name!r} has invalid dtype {value!r}: {path}")
    code = str(value.get("code", "")).lower()
    bits = value.get("bits")
    if code in {"float", "fp"} and bits in {16, 32, 64}:
        return f"float{bits}"
    if code in {"int", "uint"} and bits in {8, 16, 32, 64}:
        return f"{code}{bits}"
    if code == "bool" and bits in {1, 8}:
        return "bool"
    raise ValueError(f"TCIM tensor {name!r} has unsupported dtype code={code!r}, bits={bits!r}: {path}")


def _binding(
    tensor: RuntimeTensor,
    semantics: Mapping[str, str],
    image_layouts: Mapping[str, str],
    direction: str,
) -> TensorBinding:
    try:
        semantic = semantics[tensor.name]
    except KeyError as exc:
        raise ValueError(f"No semantic mapping for runtime {direction} tensor {tensor.name!r}") from exc
    declared_layout = image_layouts.get(semantic)
    runtime_layout = tensor.layout
    if declared_layout is not None and runtime_layout is not None and declared_layout != runtime_layout:
        raise ValueError(
            f"Semantic {semantic!r} declares layout {declared_layout}, but runtime ABI reports {runtime_layout}"
        )
    layout = runtime_layout or declared_layout
    needs_layout = len(tensor.shape) == 4 and is_image_semantic(semantic)
    if needs_layout and layout is None:
        raise ValueError(f"Rank-4 image semantic {semantic!r} requires an explicit runtime layout")
    if len(tensor.shape) != 4 and layout is not None:
        raise ValueError(f"Non-rank-4 semantic {semantic!r} cannot declare a runtime layout")
    return TensorBinding(
        semantic=semantic,
        runtime_name=tensor.name,
        index=tensor.index,
        dtype=tensor.dtype,
        shape=tensor.shape,
        layout=layout,
    )


def _load_existing_manifest(path: Path) -> InferenceManifest | None:
    if not path.is_file():
        return None
    try:
        value = load_json_strict(path)
        validate_manifest_schema(value, str(path))
        return InferenceManifest.model_validate_json(json.dumps(value, ensure_ascii=False))
    except ValueError as exc:
        raise ValueError(f"Existing unified manifest is invalid and cannot be updated: {path}: {exc}") from exc

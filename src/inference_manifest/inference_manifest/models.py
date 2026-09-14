"""Typed, hardware-independent models for schema v3 inference manifests.

The module deliberately contains no backend imports.  A manifest describes a
deployment and its runtime requirements; it never constructs a backend.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, TypeAlias, get_args, get_origin
from uuid import UUID, uuid4

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from inference_manifest.paths import normalize_bundle_path

_ROLE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_DEPLOYMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_IDENTITY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_RETURN_SEMANTICS = frozenset({"action", "grasp.poses", "detection.boxes", "segmentation.masks"})
_LEGACY_IDENTITIES = frozenset(
    {
        "family",
        "lerobot",
        "generic",
        "perception",
        "vla",
    }
)


def _is_legacy_runtime_alias(value: str) -> bool:
    """Recognize retired runtime-shaped aliases without retaining an alias map."""

    lowered = value.lower()
    return (
        lowered.startswith("acl-")
        or lowered == "om"
        or (lowered.isdigit() and len(lowered) == 4)
        or lowered.endswith("_acl")
        or lowered.endswith("_om")
        or "_om_" in lowered
    )


def _is_legacy_model_identity(value: str) -> bool:
    lowered = value.lower()
    return (
        _is_legacy_runtime_alias(value)
        or lowered == "asr"
        or lowered.endswith("_asr")
        or lowered.endswith("_raw")
        or lowered.endswith("_combined")
        or "stateful" in lowered
        or lowered.endswith("_generic")
    )


# Two prefixes carve tensors out of a deployment's external contract, for different reasons.
# ``internal.`` names a tensor one role produces and a later role consumes, so it must have a
# declared producer. ``host.`` names a tensor the host computes between roles and has no
# in-graph producer by construction.
INTERNAL_SEMANTIC_PREFIX = "internal."
HOST_SEMANTIC_PREFIX = "host."


def _validate_sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("must be a lowercase 64-character SHA-256 digest")
    return value


def _validate_uuid(value: str) -> str:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("must be a canonical UUID") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise ValueError("must be a canonical non-nil lowercase UUID")
    return value


def _validate_role(value: str) -> str:
    if not _ROLE_PATTERN.fullmatch(value):
        raise ValueError("must start with a letter and contain only letters, digits, '.', '_', or '-'")
    return value


def _validate_state_role(value: str) -> str:
    if value == "__runtime__":
        return value
    return _validate_role(value)


def _validate_shape(value: tuple[int, ...]) -> tuple[int, ...]:
    if any(dimension == 0 or dimension < -1 for dimension in value):
        raise ValueError("dimensions must be positive integers or -1 for dynamic dimensions")
    return value


def _validate_identifier(value: str, description: str) -> str:
    if not _IDENTITY_PATTERN.fullmatch(value):
        raise ValueError(f"{description} must be a stable logical identifier")
    lowered = value.lower()
    if (
        "/" in value
        or "\\" in value
        or "://" in value
        or value.startswith("/")
        or re.match(r"^[a-z]:", lowered)
        or re.search(r"(?:^|[_.-])(device|stream|lease|provider|pid|socket|resource|handle)(?:[_.-]|$)", lowered)
        or re.fullmatch(r"(?:device|stream|lease|provider|pid|socket|resource|handle)(?:[_.-].*)?", lowered)
    ):
        raise ValueError(f"invalid_state_link_identifier: {description} must be logical and instance-independent")
    return value


def _json_safe(value: Any, description: str = "value") -> Any:
    """Validate and normalize a JSON-safe value without importing a serializer library."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{description} mapping keys must be strings")
            normalized[key] = _json_safe(item, f"{description}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, list | tuple):
        return [_json_safe(item, f"{description}[]") for item in value]
    raise ValueError(f"{description} must be JSON-safe")


StrictString: TypeAlias = Annotated[str, StringConstraints(strict=True, min_length=1)]
BundlePath: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(normalize_bundle_path),
]
Sha256: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=64, max_length=64),
    AfterValidator(_validate_sha256),
]
ManifestUUID: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=36, max_length=36),
    AfterValidator(_validate_uuid),
]
Revision: TypeAlias = Annotated[int, Field(strict=True, ge=1)]
ExecutionRole: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(_validate_role),
]
StateRole: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(_validate_state_role),
]
LogicalIdentifier: TypeAlias = Annotated[
    str,
    StringConstraints(strict=True, min_length=1),
    AfterValidator(lambda value: _validate_identifier(value, "state link identifier")),
]
TensorShape: TypeAlias = Annotated[tuple[int, ...], AfterValidator(_validate_shape)]
TensorDType: TypeAlias = Literal[
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
]
BackendName: TypeAlias = Literal[
    "torch",
    "ascend",
    "rknn",
    "hmm",
    "hisilicon",
    "onnx",
    "tensorrt",
]


class StrictFrozenModel(BaseModel):
    """Base class that rejects coercion and unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @model_validator(mode="before")
    @classmethod
    def normalize_json_arrays(cls, value: Any) -> Any:
        """Accept JSON-style lists while retaining tuple-backed immutable fields."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        for name, item in value.items():
            field = cls.model_fields.get(name)
            annotation = field.annotation if field is not None else None
            while get_origin(annotation) is Annotated:
                annotation = get_args(annotation)[0]
            if get_origin(annotation) is tuple and isinstance(item, list):
                normalized[name] = tuple(item)
        return normalized


class Digest(StrictFrozenModel):
    algorithm: Literal["sha256"]
    scope: Literal["structure"]
    value: Sha256


class BundleFile(StrictFrozenModel):
    path: BundlePath


class ManifestBundle(StrictFrozenModel):
    uuid: ManifestUUID = Field(default_factory=lambda: str(uuid4()))
    revision: Revision = 1
    name: StrictString
    files: tuple[BundleFile, ...] = Field(min_length=1)
    digest: Digest

    @model_validator(mode="after")
    def validate_unique_paths(self) -> ManifestBundle:
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("bundle.files contains duplicate paths")
        return self


class DeploymentTarget(StrictFrozenModel):
    """A canonical runtime family and optional ABI, never an SDK configuration path."""

    soc: StrictString | None = None
    runtime: StrictString
    runtime_abi: StrictString | None = None

    @model_validator(mode="after")
    def validate_runtime_family(self) -> DeploymentTarget:
        if _is_legacy_runtime_alias(self.runtime):
            raise ValueError(
                f"target.runtime={self.runtime!r} is not the canonical runtime family; use target.runtime='acl' and "
                "put the ABI version in target.runtime_abi"
            )
        if self.runtime_abi is not None and _is_legacy_runtime_alias(self.runtime_abi):
            raise ValueError("target.runtime_abi must be a versioned ABI, not a runtime-family alias")
        if self.runtime_abi is not None and not re.search(r"\d", self.runtime_abi):
            raise ValueError("target.runtime_abi must contain a version identifier")
        return self


class DeploymentArtifact(StrictFrozenModel):
    path: BundlePath
    format: StrictString
    share_group: StrictString | None = None
    sha256: Sha256 | None = None


_IMAGE_SEMANTIC_PREFIXES = ("observation.image.", "observation.images.")


def is_image_semantic(semantic: str) -> bool:
    return semantic == "observation.image" or semantic.startswith(_IMAGE_SEMANTIC_PREFIXES)


def _validate_layout(shape: tuple[int, ...], layout: str | None, semantic: str) -> None:
    needs_layout = len(shape) == 4 and is_image_semantic(semantic)
    if needs_layout and layout is None:
        raise ValueError(f"rank-4 image tensor {semantic!r} requires NCHW or NHWC layout")
    if len(shape) != 4 and layout is not None:
        raise ValueError(f"non-rank-4 tensor {semantic!r} must omit layout")


class TensorBinding(StrictFrozenModel):
    semantic: StrictString
    runtime_name: StrictString | None = None
    index: int | None = Field(default=None, ge=0)
    dtype: TensorDType
    shape: TensorShape
    layout: Literal["NCHW", "NHWC"] | None = None

    @model_validator(mode="after")
    def validate_runtime_slot_and_layout(self) -> TensorBinding:
        if self.runtime_name is None and self.index is None:
            raise ValueError("a tensor binding requires runtime_name, index, or both")
        _validate_layout(self.shape, self.layout, self.semantic)
        return self


class SemanticTensor(StrictFrozenModel):
    """Model-owned semantic tensor contract shared by all deployments."""

    semantic: StrictString
    dtype: TensorDType
    shape: TensorShape
    layout: Literal["NCHW", "NHWC"] | None = None

    @model_validator(mode="after")
    def validate_layout(self) -> SemanticTensor:
        _validate_layout(self.shape, self.layout, self.semantic)
        return self


class AudioContract(StrictFrozenModel):
    """Deployment-specific audio boundary constraints."""

    sample_rate_hz: Annotated[int, Field(strict=True, gt=0)] | None = None
    channels: Annotated[int, Field(strict=True, gt=0)] | None = None
    channel_semantics: StrictString | None = None
    sample_dtype: TensorDType | None = None
    frame_size: Annotated[int, Field(strict=True, gt=0)] | None = None
    chunk_size: Annotated[int, Field(strict=True, gt=0)] | None = None
    execution_mode: Literal["streaming", "offline", "both"] | None = None

    @model_validator(mode="after")
    def validate_declared_contract(self) -> AudioContract:
        if not any(
            value is not None
            for value in (
                self.sample_rate_hz,
                self.channels,
                self.channel_semantics,
                self.sample_dtype,
                self.frame_size,
                self.chunk_size,
                self.execution_mode,
            )
        ):
            raise ValueError("audio_contract must declare at least one applicable constraint")
        return self


class EmbeddingMetadata(StrictFrozenModel):
    embedding_space_id: StrictString
    dimension: Annotated[int, Field(strict=True, gt=0)]
    normalization: StrictString
    image_preprocessing: StrictString
    text_preprocessing: StrictString


class SemanticIdentity(StrictFrozenModel):
    """Logical model-space contract independent of its deployment."""

    logical_model_revision: StrictString
    preprocessing_contract: StrictString
    output_semantics: StrictString
    embedding: EmbeddingMetadata | None = None


CANONICAL_MODEL_MAPPING: dict[str, dict[str, Any]] = {
    "act": {"interface": "policy", "operations": ("predict",)},
    "pi05": {"interface": "policy", "operations": ("predict",)},
    "smolvla": {"interface": "policy", "operations": ("predict",)},
    "ram_plus": {"interface": "tensor_model", "operations": ("recognize_tags",)},
    "sam2": {"interface": "tensor_model", "operations": ("prompt", "automatic")},
    "siglip2": {"interface": "tensor_model", "operations": ("encode",)},
    "grounding_dino": {"interface": "tensor_model", "operations": ("detect",)},
    "yolox_person": {"interface": "tensor_model", "operations": ("detect",)},
    "pear_parameter_network": {"interface": "tensor_model", "operations": ("predict_parameters",)},
    "graspgen": {"interface": "tensor_model", "operations": ("generate_grasps",)},
    "dummy_echo": {"interface": "tensor_model", "operations": ("echo",)},
    "zipvoice": {"interface": "tensor_model", "operations": ("synthesize",)},
    "fullsubnet": {"interface": "tensor_model", "operations": ("enhance",)},
    "silero_vad": {"interface": "tensor_model", "operations": ("vad",)},
    "sherpa_onnx": {"interface": "tensor_model", "operations": ("recognize",)},
    "speech_direction": {"interface": "tensor_model", "operations": ("enhance_and_vad",)},
}

# A tuple form is convenient for callers that need a stable, hashable mapping.
CANONICAL_MODEL_IDENTITIES = {
    name: (value["interface"], name, value["operations"]) for name, value in CANONICAL_MODEL_MAPPING.items()
}
CANONICAL_MODEL_TYPE_OPERATION = {
    name: (value["interface"], value["operations"]) for name, value in CANONICAL_MODEL_MAPPING.items()
}


class ModelIdentity(StrictFrozenModel):
    """The only dispatch identity accepted by schema v3."""

    interface: Literal["policy", "tensor_model"]
    model_type: StrictString
    operation: StrictString

    @model_validator(mode="after")
    def validate_identity(self) -> ModelIdentity:
        if not _IDENTITY_PATTERN.fullmatch(self.model_type):
            raise ValueError("model_type must be a concrete stable identifier")
        if self.model_type in _LEGACY_IDENTITIES or _is_legacy_model_identity(self.model_type):
            raise ValueError(f"legacy model identity {self.model_type!r} is not supported in schema v3")
        if self.interface == "policy" and self.operation != "predict":
            raise ValueError("policy operation must be exactly 'predict'")
        if self.operation in {"combined", "raw"}:
            raise ValueError(
                f"model_type {self.model_type!r} cannot use legacy operation variant {self.operation!r} in schema v3"
            )
        if self.interface == "tensor_model" and self.model_type == "vla":
            raise ValueError("model_type='vla' is a classification label, not a runtime identity")

        canonical = CANONICAL_MODEL_MAPPING.get(self.model_type)
        if canonical is not None:
            if self.interface != canonical["interface"]:
                raise ValueError(f"model_type {self.model_type!r} requires interface={canonical['interface']!r}")
            if self.operation not in canonical["operations"]:
                raise ValueError(
                    f"model_type {self.model_type!r} does not support operation {self.operation!r}; "
                    f"supported operations: {list(canonical['operations'])}"
                )
        return self


def canonical_model_identity(model_type: str) -> tuple[str, str, tuple[str, ...]] | None:
    """Return the canonical mapping for a known model type."""

    return CANONICAL_MODEL_IDENTITIES.get(model_type)


class ModelDescriptor(ModelIdentity):
    """Top-level model identity plus its semantic tensor contract."""

    inputs: tuple[SemanticTensor, ...] = ()
    outputs: tuple[SemanticTensor, ...] = ()
    architecture_class: StrictString | None = None
    domain: StrictString | None = None
    lineage: StrictString | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    semantic_identity: SemanticIdentity | None = None

    @model_validator(mode="after")
    def validate_unique_semantics(self) -> ModelDescriptor:
        for direction, descriptors in (("inputs", self.inputs), ("outputs", self.outputs)):
            semantics = [descriptor.semantic for descriptor in descriptors]
            if len(semantics) != len(set(semantics)):
                raise ValueError(f"model.{direction} contains duplicate semantic descriptors")
        if self.interface == "tensor_model" and (not self.inputs or not self.outputs):
            raise ValueError("model.inputs and model.outputs should be non-empty for tensor_model models")
        _json_safe(self.metadata, "model.metadata")
        return self

    @property
    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            interface=self.interface,
            model_type=self.model_type,
            operation=self.operation,
        )


class StateLink(StrictFrozenModel):
    """A role-aware, fingerprint-safe link for persistent stream state."""

    role: StateRole
    state_name: LogicalIdentifier
    owner: Literal["session", "streaming_runtime"]
    source: LogicalIdentifier
    target: LogicalIdentifier
    scope: Literal["stream", "runtime"]
    state_bank: LogicalIdentifier


class ExecutionContract(StrictFrozenModel):
    """The orthogonal public execution contract for one named deployment."""

    state_scope: Literal["request", "stream"]
    execution_structure: Literal["direct", "iterative"]
    orchestration_visibility: Literal["executor", "session"] | None = None
    cancellation_granularity: Literal["stage", "checkpoint", "request_boundary"]
    stateful: bool = False
    state_links: tuple[StateLink, ...] = ()
    state_bank_mode: Literal["per_stream", "runtime_exclusive"] | None = None
    max_open_streams: Annotated[int, Field(strict=True, gt=0)] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_field_presence(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        state_scope = value.get("state_scope")
        structure = value.get("execution_structure")
        if structure == "direct" and "orchestration_visibility" in value:
            raise ValueError("direct execution must omit orchestration_visibility")
        if state_scope == "request":
            present = {key for key in ("state_links", "state_bank_mode", "max_open_streams") if key in value}
            if present:
                raise ValueError(f"request execution contracts must omit {sorted(present)}")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> ExecutionContract:
        if self.execution_structure == "direct" and self.orchestration_visibility is not None:
            raise ValueError("direct execution must omit orchestration_visibility")
        if self.execution_structure == "iterative" and self.orchestration_visibility is None:
            raise ValueError("iterative execution requires orchestration_visibility=executor|session")

        if self.state_scope == "request":
            if self.stateful:
                raise ValueError("request execution contracts must declare stateful=false")
            if self.state_links:
                raise ValueError("request execution contracts must omit state_links")
            if self.state_bank_mode is not None or self.max_open_streams is not None:
                raise ValueError("state bank and max_open_streams are valid only for stream execution contracts")
            return self

        if self.state_bank_mode is None or self.max_open_streams is None:
            raise ValueError("stream execution contracts require state_bank_mode and max_open_streams")
        if self.state_bank_mode == "runtime_exclusive" and self.max_open_streams != 1:
            raise ValueError("runtime_exclusive stream contracts require max_open_streams=1")
        if self.stateful and not self.state_links:
            raise ValueError("stateful stream contracts require explicit state_links")

        canonical_links = [
            (link.role, link.state_name, link.owner, link.source, link.target, link.scope, link.state_bank)
            for link in self.state_links
        ]
        if len(canonical_links) != len(set(canonical_links)):
            raise ValueError("execution_contract.state_links contains duplicate links")
        for link in self.state_links:
            if self.state_bank_mode == "per_stream" and link.scope != "stream":
                raise ValueError("per_stream state links must use scope='stream'")
            if self.state_bank_mode == "runtime_exclusive" and link.owner == "session" and link.scope != "runtime":
                raise ValueError("runtime_exclusive session state links must use scope='runtime'")
        return self

    @property
    def name(self) -> str:
        return f"{self.state_scope}-{self.execution_structure}"

    @property
    def contract_name(self) -> str:
        return self.name

    @property
    def normalized_state_links(self) -> tuple[StateLink, ...]:
        return tuple(
            sorted(
                self.state_links,
                key=lambda link: (
                    link.role,
                    link.state_name,
                    link.owner,
                    link.source,
                    link.target,
                    link.scope,
                    link.state_bank,
                ),
            )
        )


class DeploymentProfileProjection(StrictFrozenModel):
    """Versioned deployment-only projection of a backend profile."""

    projection_version: Literal[1] = 1
    backend: StrictString
    runtime_abi: StrictString | None = None
    fields: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_fields(self) -> DeploymentProfileProjection:
        normalized = _json_safe(self.fields, "deployment profile projection.fields")
        object.__setattr__(self, "fields", normalized)
        if self.runtime_abi is not None and not re.search(r"\d", self.runtime_abi):
            raise ValueError("runtime_abi must contain a version identifier")
        return self

    def canonical_json(self) -> str:
        import json

        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")


class RuntimeInstanceProjection(StrictFrozenModel):
    """Versioned complete typed profile plus local instance metadata."""

    projection_version: Literal[1] = 1
    profile: Any
    instance_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_projection(self) -> RuntimeInstanceProjection:
        if not isinstance(self.profile, BackendRuntimeProfile):
            raise ValueError("runtime instance projection.profile must be a typed BackendRuntimeProfile")
        normalized = _json_safe(
            _without_provider_identity(self.instance_metadata),
            "runtime instance projection.instance_metadata",
        )
        object.__setattr__(self, "instance_metadata", normalized)
        return self

    def canonical_json(self) -> str:
        import json

        value = self.model_dump(mode="json")
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")


class BackendRuntimeProfile(StrictFrozenModel):
    """Base for explicitly classified backend profile values."""

    backend_name: ClassVar[str]
    deployment_field_names: ClassVar[frozenset[str]] = frozenset()
    instance_field_names: ClassVar[frozenset[str]] = frozenset()

    @property
    def backend(self) -> str:
        return self.backend_name

    def _values(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def deployment_projection(self, runtime_abi: str | None = None) -> DeploymentProfileProjection:
        values = self._values()
        classified = self.deployment_field_names | self.instance_field_names
        declared = set(self.__class__.model_fields)
        unknown = declared - classified
        stale = classified - declared
        if unknown or stale:
            names = sorted(unknown or stale)
            raise ValueError(f"unknown_profile_projection_field: {names}")
        fields = {name: values[name] for name in sorted(self.deployment_field_names) if name in values}
        return DeploymentProfileProjection(backend=self.backend_name, runtime_abi=runtime_abi, fields=fields)

    def runtime_instance_projection(
        self,
        runtime_abi: str | None = None,
        instance_metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeInstanceProjection:
        metadata = _without_provider_identity(dict(instance_metadata or {}))
        if runtime_abi is not None:
            metadata = {**metadata, "runtime_abi": runtime_abi}
        return RuntimeInstanceProjection(profile=self, instance_metadata=metadata)


class TorchRuntimeProfile(BackendRuntimeProfile):
    backend_name: ClassVar[str] = "torch"
    deployment_field_names: ClassVar[frozenset[str]] = frozenset({"dtype", "thread_count"})
    instance_field_names: ClassVar[frozenset[str]] = frozenset({"device"})

    device: Literal["cpu", "cuda", "mps", "npu"]
    dtype: StrictString | None = None
    thread_count: Annotated[int, Field(strict=True, gt=0)] | None = None


class AscendRuntimeProfile(BackendRuntimeProfile):
    """Minimal Ascend runtime profile.  ACL configuration paths are forbidden."""

    backend_name: ClassVar[str] = "ascend"
    deployment_field_names: ClassVar[frozenset[str]] = frozenset()
    instance_field_names: ClassVar[frozenset[str]] = frozenset({"device_id"})

    device_id: Annotated[int, Field(strict=True, ge=0)]


class RKNNRuntimeProfile(BackendRuntimeProfile):
    backend_name: ClassVar[str] = "rknn"
    deployment_field_names: ClassVar[frozenset[str]] = frozenset({"target_name", "core_mask"})
    instance_field_names: ClassVar[frozenset[str]] = frozenset({"device", "device_id"})

    target_name: StrictString | None = None
    core_mask: Annotated[int, Field(strict=True, ge=0)] | None = None
    device: StrictString | None = None
    device_id: Annotated[int, Field(strict=True, ge=0)] | None = None


class HMMRuntimeProfile(BackendRuntimeProfile):
    backend_name: ClassVar[str] = "hmm"
    deployment_field_names: ClassVar[frozenset[str]] = frozenset(
        {"role", "link", "quantization", "precision", "tcim_abi"}
    )
    instance_field_names: ClassVar[frozenset[str]] = frozenset({"device_id"})

    role: StrictString | None = None
    link: StrictString | None = None
    quantization: StrictString | None = None
    precision: StrictString | None = None
    tcim_abi: StrictString | None = None
    device_id: Annotated[int, Field(strict=True, ge=0)] | None = None


class HisiliconRuntimeProfile(BackendRuntimeProfile):
    backend_name: ClassVar[str] = "hisilicon"
    deployment_field_names: ClassVar[frozenset[str]] = frozenset({"protocol", "worker_abi"})
    instance_field_names: ClassVar[frozenset[str]] = frozenset({"worker_path", "worker_pid", "worker_socket"})

    protocol: StrictString | None = None
    worker_abi: StrictString | None = None
    worker_path: StrictString | None = None
    worker_pid: Annotated[int, Field(strict=True, ge=1)] | None = None
    worker_socket: StrictString | None = None


class ONNXRuntimeProfile(BackendRuntimeProfile):
    backend_name: ClassVar[str] = "onnx"
    deployment_field_names: ClassVar[frozenset[str]] = frozenset({"provider", "optimization_level"})
    instance_field_names: ClassVar[frozenset[str]] = frozenset({"device"})

    provider: StrictString | None = None
    optimization_level: StrictString | None = None
    device: StrictString | None = None


class TensorRTRuntimeProfile(BackendRuntimeProfile):
    backend_name: ClassVar[str] = "tensorrt"
    deployment_field_names: ClassVar[frozenset[str]] = frozenset({"precision", "engine_version"})
    instance_field_names: ClassVar[frozenset[str]] = frozenset({"device_id"})

    precision: StrictString | None = None
    engine_version: StrictString | None = None
    device_id: Annotated[int, Field(strict=True, ge=0)] | None = None


PROFILE_TYPES: dict[str, type[BackendRuntimeProfile]] = {
    "torch": TorchRuntimeProfile,
    "ascend": AscendRuntimeProfile,
    "rknn": RKNNRuntimeProfile,
    "hmm": HMMRuntimeProfile,
    "hisilicon": HisiliconRuntimeProfile,
    "onnx": ONNXRuntimeProfile,
    "tensorrt": TensorRTRuntimeProfile,
}


def parse_backend_runtime_profile(
    backend: str,
    value: BackendRuntimeProfile | Mapping[str, Any],
) -> BackendRuntimeProfile:
    """Parse one backend profile without importing a backend SDK."""

    profile_type = PROFILE_TYPES.get(backend)
    if profile_type is None:
        raise ValueError(f"unsupported backend runtime profile {backend!r}")
    if isinstance(value, profile_type):
        return value
    if isinstance(value, Mapping):
        return profile_type.model_validate(value)
    raise ValueError(f"backend {backend!r} requires typed profile {profile_type.__name__}")


BACKEND_PROFILE_FIELD_CLASSIFICATIONS = {
    backend: {
        "deployment": tuple(sorted(profile_type.deployment_field_names)),
        "instance": tuple(sorted(profile_type.instance_field_names)),
    }
    for backend, profile_type in PROFILE_TYPES.items()
}


def profile_field_classification(backend: str) -> dict[str, tuple[str, ...]]:
    """Return the explicit deployment/instance field allowlist for a backend."""

    try:
        value = BACKEND_PROFILE_FIELD_CLASSIFICATIONS[backend]
    except KeyError as exc:
        raise ValueError(f"unsupported backend runtime profile {backend!r}") from exc
    return {key: tuple(values) for key, values in value.items()}


def _without_provider_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_provider_identity(item)
            for key, item in value.items()
            if key not in {"provider", "provider_id", "provider_identity", "provider_version"}
        }
    if isinstance(value, list):
        return [_without_provider_identity(item) for item in value]
    return value


class RoleRuntimeProfile(StrictFrozenModel):
    """Backend, canonical target and typed profile for one model role."""

    backend: BackendName
    target: DeploymentTarget
    profile: Any = Field(validation_alias=AliasChoices("profile", "backend_profile"))

    @model_validator(mode="before")
    @classmethod
    def parse_typed_profile(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw = dict(value)
        if "profile" not in raw and "backend_profile" in raw:
            raw["profile"] = raw.pop("backend_profile")
        backend = raw.get("backend")
        profile = raw.get("profile")
        if isinstance(profile, Mapping):
            raw["profile"] = parse_backend_runtime_profile(backend, profile)
        return raw

    @model_validator(mode="after")
    def validate_profile(self) -> RoleRuntimeProfile:
        expected = PROFILE_TYPES.get(self.backend)
        if expected is None:
            raise ValueError(f"unsupported backend runtime profile {self.backend!r}")
        if not isinstance(self.profile, expected):
            raise ValueError(
                f"backend {self.backend!r} requires typed profile {expected.__name__}, "
                f"received {type(self.profile).__name__}"
            )
        if self.profile.backend != self.backend:
            raise ValueError("backend runtime profile identity mismatch")
        if self.backend == "ascend" and self.target.runtime != "acl":
            raise ValueError("backend='ascend' requires target.runtime='acl'")
        return self

    @property
    def target_runtime(self) -> str:
        return self.target.runtime

    @property
    def runtime_abi(self) -> str | None:
        return self.target.runtime_abi

    @property
    def backend_profile(self) -> BackendRuntimeProfile:
        return self.profile

    def deployment_projection(self) -> DeploymentProfileProjection:
        return self.profile.deployment_projection(self.runtime_abi)

    def runtime_instance_projection(
        self,
        instance_metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeInstanceProjection:
        return self.profile.runtime_instance_projection(self.runtime_abi, instance_metadata)


class ArtifactBindings(StrictFrozenModel):
    inputs: tuple[TensorBinding, ...] = Field(min_length=1)
    outputs: tuple[TensorBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_slots(self) -> ArtifactBindings:
        for direction, bindings in (("inputs", self.inputs), ("outputs", self.outputs)):
            semantics = [binding.semantic for binding in bindings]
            if len(semantics) != len(set(semantics)):
                raise ValueError(f"{direction} contains duplicate semantic bindings")

            runtime_names = [binding.runtime_name for binding in bindings if binding.runtime_name is not None]
            if len(runtime_names) != len(set(runtime_names)):
                raise ValueError(f"{direction} contains duplicate runtime_name values")

            indices = [binding.index for binding in bindings if binding.index is not None]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{direction} contains duplicate runtime indices")
            if indices and len(indices) != len(bindings):
                raise ValueError(f"{direction} must declare indices for every binding or omit them from every binding")
            if direction == "inputs" and indices and sorted(indices) != list(range(len(indices))):
                raise ValueError(f"{direction} runtime indices must be contiguous and start at zero")
        return self


class DeviceLink(StrictFrozenModel):
    semantic: Annotated[str, StringConstraints(strict=True, pattern=r"^internal\.[A-Za-z0-9_.-]+$")]
    producer: ExecutionRole
    consumer: ExecutionRole
    producer_binding: Literal["output", "input"] = "output"
    transport: Literal["device_pointer"]
    owner: Literal["producer", "consumer"]
    lifetime: Literal["inference"] = "inference"

    @model_validator(mode="after")
    def validate_input_source_ownership(self) -> DeviceLink:
        if self.producer_binding == "input" and self.owner != "producer":
            raise ValueError("input-sourced device links require owner='producer'")
        return self


class Deployment(StrictFrozenModel):
    """One v3 named deployment, independent of backend construction code."""

    uuid: ManifestUUID = Field(default_factory=lambda: str(uuid4()))
    revision: Revision = 1
    execution_contract: ExecutionContract
    runtime_profile: RoleRuntimeProfile | None = Field(
        default=None,
        validation_alias=AliasChoices("runtime_profile", "profile"),
    )
    role_identities: dict[ExecutionRole, ModelIdentity] | None = None
    role_runtime_profiles: dict[ExecutionRole, RoleRuntimeProfile] | None = None
    artifacts: dict[ExecutionRole, DeploymentArtifact] = Field(default_factory=dict)
    execution: tuple[ExecutionRole, ...] = ()
    bindings: dict[ExecutionRole, ArtifactBindings] = Field(default_factory=dict)
    device_links: tuple[DeviceLink, ...] = ()
    audio_contract: AudioContract | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_composite(self) -> bool:
        return self.role_identities is not None

    @property
    def backend(self) -> str:
        """Return the backend for a single-model deployment.

        Composite deployments keep backend selection per role and therefore do
        not expose a synthetic shared backend.
        """

        if self.runtime_profile is not None:
            return self.runtime_profile.backend
        profiles = self.role_runtime_profiles or {}
        backends = {profile.backend for profile in profiles.values()}
        if len(backends) == 1:
            return next(iter(backends))
        raise ValueError("composite deployments do not have one shared backend")

    @property
    def target(self) -> DeploymentTarget:
        """Return the single-model target profile."""

        if self.runtime_profile is None:
            raise ValueError("composite deployments expose targets per role")
        return self.runtime_profile.target

    @property
    def device(self) -> str | None:
        if self.runtime_profile is None:
            return None
        value = getattr(self.runtime_profile.backend_profile, "device", None)
        return value if isinstance(value, str) else None

    @property
    def contract_name(self) -> str:
        return self.execution_contract.name

    @model_validator(mode="after")
    def validate_deployment(self) -> Deployment:
        execution_roles = set(self.execution)
        if len(execution_roles) != len(self.execution):
            raise ValueError("execution contains duplicate roles")
        binding_roles = set(self.bindings)
        artifact_roles = set(self.artifacts)
        if self.execution and binding_roles != execution_roles:
            raise ValueError(
                "binding roles must exactly match execution roles "
                f"(missing={sorted(execution_roles - binding_roles)}, unexpected={sorted(binding_roles - execution_roles)})"
            )
        if self.bindings and not self.execution:
            raise ValueError("bindings require a non-empty execution role list")
        if self.execution and not execution_roles.issubset(artifact_roles):
            raise ValueError(
                f"artifact roles must contain every execution role (missing={sorted(execution_roles - artifact_roles)})"
            )

        if self.role_identities is None:
            if self.role_runtime_profiles is not None:
                raise ValueError("role_runtime_profiles requires matching role_identities")
            if self.runtime_profile is None:
                raise ValueError("single-model deployments require runtime_profile")
            declared_roles = execution_roles
        else:
            if not self.role_identities:
                raise ValueError("role_identities must be non-empty when present")
            if self.runtime_profile is not None:
                raise ValueError("composite deployments must use role_runtime_profiles, not runtime_profile")
            if self.role_runtime_profiles is None:
                raise ValueError("composite deployments require role_runtime_profiles")
            identity_roles = set(self.role_identities)
            profile_roles = set(self.role_runtime_profiles)
            if identity_roles != profile_roles:
                raise ValueError(
                    "role_identities and role_runtime_profiles must contain exactly the same roles "
                    f"(missing={sorted(identity_roles - profile_roles)}, unexpected={sorted(profile_roles - identity_roles)})"
                )
            declared_roles = identity_roles
            if not declared_roles.issubset(execution_roles):
                raise ValueError("every model role identity must have an execution role")
            if not declared_roles.issubset(artifact_roles):
                raise ValueError("every model role identity must have an artifact binding")
            if not declared_roles.issubset(binding_roles):
                raise ValueError("every model role identity must have tensor bindings")
            model_types = [identity.model_type for identity in self.role_identities.values()]
            if len(model_types) != len(set(model_types)):
                raise ValueError("role model_type values must be globally unique")

        valid_state_roles = declared_roles | {"__runtime__"}
        unknown_state_roles = {link.role for link in self.execution_contract.state_links} - valid_state_roles
        if unknown_state_roles:
            raise ValueError(f"state_links reference undeclared roles: {sorted(unknown_state_roles)}")

        for role, artifact in self.artifacts.items():
            if self.role_identities is None and role not in execution_roles and self.execution:
                # Auxiliary artifacts (workers, metadata) are allowed.
                continue
            if self.role_identities is not None and role in declared_roles and artifact.path == "":
                raise ValueError(f"artifact for role {role!r} must declare a path")
            if self.role_identities is not None and artifact.share_group is not None and role not in declared_roles:
                raise ValueError("auxiliary composite artifacts cannot participate in a role share_group")

        roles_by_path: dict[str, list[str]] = {}
        for role, artifact in self.artifacts.items():
            roles_by_path.setdefault(artifact.path, []).append(role)
            if (
                self.role_identities is None
                and artifact.share_group is not None
                and self.runtime_profile.backend != "rknn"
            ):
                raise ValueError("share_group is only valid for RKNN artifacts")
            if self.role_identities is not None and artifact.share_group is not None:
                profile = self.role_runtime_profiles.get(role) if self.role_runtime_profiles else None
                if profile is not None and profile.backend != "rknn":
                    raise ValueError("share_group is only valid for RKNN artifacts")
        for path, roles in roles_by_path.items():
            if len(roles) < 2:
                continue
            groups = {self.artifacts[role].share_group for role in roles}
            if None in groups or len(groups) != 1:
                raise ValueError(
                    f"duplicate artifact path {path!r} requires one shared non-empty share_group for roles {roles}"
                )
        roles_by_group: dict[str, list[str]] = {}
        for role, artifact in self.artifacts.items():
            if artifact.share_group is not None:
                roles_by_group.setdefault(artifact.share_group, []).append(role)
        for group, roles in roles_by_group.items():
            paths = {self.artifacts[role].path for role in roles}
            if len(paths) != 1:
                raise ValueError(f"RKNN share_group {group!r} must reference one artifact path")

            def signature(role: str) -> tuple[tuple[object, ...], tuple[object, ...]]:
                bindings = self.bindings.get(role)

                def slots(values: tuple[TensorBinding, ...]) -> tuple[object, ...]:
                    return tuple(
                        (value.runtime_name, value.index, value.dtype, value.shape, value.layout) for value in values
                    )

                if bindings is None:
                    return (), ()
                return slots(bindings.inputs), slots(bindings.outputs)

            if len({signature(role) for role in roles}) != 1:
                raise ValueError(f"RKNN share_group {group!r} roles must expose identical runtime ABI")

        if self.execution:
            role_positions = {role: index for index, role in enumerate(self.execution)}
            produced_internal: dict[str, str] = {}
            for role in self.execution:
                for binding in self.bindings[role].outputs:
                    if binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX):
                        produced_internal[binding.semantic] = role

            linked_inputs = {(link.consumer, link.semantic) for link in self.device_links}
            linked_source_inputs = {
                (link.producer, link.semantic) for link in self.device_links if link.producer_binding == "input"
            }
            input_sourced_semantics = {link.semantic for link in self.device_links if link.producer_binding == "input"}
            ambiguous_semantics = sorted(input_sourced_semantics & set(produced_internal))
            if ambiguous_semantics:
                raise ValueError(
                    f"input-sourced device links cannot share semantics with internal outputs: {ambiguous_semantics}"
                )

            for role in self.execution:
                for binding in self.bindings[role].inputs:
                    if not binding.semantic.startswith(INTERNAL_SEMANTIC_PREFIX):
                        continue
                    producer = produced_internal.get(binding.semantic)
                    endpoint = (role, binding.semantic)
                    if producer is None and endpoint not in linked_inputs and endpoint not in linked_source_inputs:
                        raise ValueError(
                            f"internal input {binding.semantic!r} for role {role!r} has no declared producer"
                        )
                    if producer is not None and role_positions[producer] >= role_positions[role]:
                        raise ValueError(
                            f"internal input {binding.semantic!r} must be produced before role {role!r} executes"
                        )

            for link in self.device_links:
                if link.producer not in execution_roles or link.consumer not in execution_roles:
                    raise ValueError(f"device link {link.semantic!r} references an unknown execution role")
                if role_positions[link.producer] >= role_positions[link.consumer]:
                    raise ValueError(f"device link {link.semantic!r} producer must execute before its consumer")
                producer_bindings = (
                    self.bindings[link.producer].inputs
                    if link.producer_binding == "input"
                    else self.bindings[link.producer].outputs
                )
                producer_semantics = {binding.semantic for binding in producer_bindings}
                consumer_inputs = {binding.semantic for binding in self.bindings[link.consumer].inputs}
                if link.semantic not in producer_semantics or link.semantic not in consumer_inputs:
                    raise ValueError(
                        f"device link {link.semantic!r} must match producer {link.producer_binding} "
                        "and consumer input bindings"
                    )

        _json_safe(self.metadata, "deployment.metadata")
        return self


# These names remain as source-level aliases for code that only refers to a deployment
# object.  They do not restore the v2 backend/device constructors or loader behavior.
NamedDeployment = Deployment
CompiledDeployment = Deployment
TorchDeployment = Deployment
DeploymentProfile = RoleRuntimeProfile


class InferenceManifest(StrictFrozenModel):
    schema_version: Literal[3]
    bundle: ManifestBundle
    model: ModelDescriptor
    deployments: dict[str, Deployment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_deployment_names_and_identities(self) -> InferenceManifest:
        invalid = [name for name in self.deployments if not _DEPLOYMENT_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid deployment names: {sorted(invalid)}")
        for name, deployment in self.deployments.items():
            if deployment.role_identities:
                if self.model.model_type in {identity.model_type for identity in deployment.role_identities.values()}:
                    raise ValueError(f"deployment {name!r} reuses top-level model_type as a role identity")
                if self.model.interface != "tensor_model":
                    raise ValueError("composite role identities are valid only for tensor_model services")
        return self


@dataclass(frozen=True)
class ValidatedDeployment:
    """Immutable loader snapshot passed to runtime construction."""

    bundle_root: Path
    manifest_path: Path
    manifest: InferenceManifest
    deployment_name: str
    deployment: Deployment
    top_level_identity: ModelIdentity
    role_identities: Mapping[str, ModelIdentity]
    role_runtime_profiles: Mapping[str, RoleRuntimeProfile]
    selected_deployment: Deployment
    semantic_contract: ModelDescriptor
    resolved_artifacts: Mapping[str, Path]
    role_artifact_bindings: Mapping[str, ArtifactBindings]
    declared_metadata: Mapping[str, Any]
    integrity_status: Any
    deployment_fingerprint: str
    runtime_profile_fingerprint: str
    policy: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_identities", MappingProxyType(dict(self.role_identities)))
        object.__setattr__(self, "role_runtime_profiles", MappingProxyType(dict(self.role_runtime_profiles)))
        object.__setattr__(self, "resolved_artifacts", MappingProxyType(dict(self.resolved_artifacts)))
        object.__setattr__(self, "role_artifact_bindings", MappingProxyType(dict(self.role_artifact_bindings)))
        object.__setattr__(self, "declared_metadata", MappingProxyType(dict(self.declared_metadata)))

    @property
    def fingerprint(self) -> str:
        """Compatibility spelling for the v3 deployment fingerprint."""

        return self.deployment_fingerprint

    @property
    def runtime_instance_fingerprint(self) -> str:
        return self.runtime_profile_fingerprint

    @property
    def identity(self) -> ModelIdentity:
        return self.top_level_identity

    @property
    def runtime_profile(self) -> RoleRuntimeProfile | None:
        return self.deployment.runtime_profile

    @property
    def artifact_handles(self) -> Mapping[str, Path]:
        return self.resolved_artifacts

    @property
    def resolved_artifact_handles(self) -> Mapping[str, Path]:
        return self.resolved_artifacts

    @property
    def role_to_artifact_bindings(self) -> Mapping[str, ArtifactBindings]:
        return self.role_artifact_bindings

    @property
    def integrity(self) -> Any:
        return self.integrity_status

    @property
    def integrity_report(self) -> Any:
        return self.integrity_status

    @property
    def profile_fingerprint(self) -> str:
        return self.runtime_profile_fingerprint

    @property
    def runtime_profile_instance_fingerprint(self) -> str:
        return self.runtime_profile_fingerprint

    def to_runtime_spec(self, execution_policy: ExecutionPolicy | None = None) -> ModelRuntimeSpec:
        return ModelRuntimeSpec.from_validated_deployment(self, execution_policy=execution_policy)


class ExecutionPolicy(StrictFrozenModel):
    """Runtime defaults intentionally limited to one request timeout."""

    default_timeout_seconds: Annotated[float, Field(strict=True, gt=0)] | None = Field(
        default=None,
        validation_alias=AliasChoices("default_timeout_seconds", "default_timeout", "default_request_timeout"),
    )


class RoleRuntimeSpec(StrictFrozenModel):
    deployment_binding: Any
    backend: BackendName
    target_runtime: StrictString
    runtime_abi: StrictString | None = None
    backend_profile: Any

    @model_validator(mode="before")
    @classmethod
    def parse_backend_profile(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw = dict(value)
        backend = raw.get("backend")
        profile = raw.get("backend_profile")
        if isinstance(profile, Mapping):
            raw["backend_profile"] = parse_backend_runtime_profile(backend, profile)
        return raw

    @model_validator(mode="after")
    def validate_typed_profile(self) -> RoleRuntimeSpec:
        expected = PROFILE_TYPES.get(self.backend)
        if expected is None or not isinstance(self.backend_profile, expected):
            expected_name = expected.__name__ if expected else "a supported profile"
            raise ValueError(f"role runtime spec requires typed backend profile {expected_name}")
        if self.backend == "ascend" and self.target_runtime != "acl":
            raise ValueError("backend='ascend' requires target_runtime='acl'")
        if _is_legacy_runtime_alias(self.target_runtime):
            raise ValueError("target_runtime must use the canonical runtime family")
        if self.runtime_abi is not None and not re.search(r"\d", self.runtime_abi):
            raise ValueError("runtime_abi must contain a version identifier")
        return self

    def deployment_projection(self) -> DeploymentProfileProjection:
        return self.backend_profile.deployment_projection(self.runtime_abi)

    def runtime_instance_projection(
        self,
        instance_metadata: Mapping[str, Any] | None = None,
    ) -> RuntimeInstanceProjection:
        return self.backend_profile.runtime_instance_projection(self.runtime_abi, instance_metadata)


class ModelRuntimeSpec(StrictFrozenModel):
    """Typed runtime input: one profile for a single model or one per role."""

    deployment: Any = Field(
        default=None,
        validation_alias=AliasChoices("deployment", "validated_deployment", "deployment_snapshot"),
    )
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    runtime_profile: Any = None
    target_runtime: StrictString | None = None
    runtime_abi: StrictString | None = None
    role_runtime_specs: dict[ExecutionRole, RoleRuntimeSpec] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_single_profile(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw = dict(value)
        profile = raw.get("runtime_profile")
        if isinstance(profile, RoleRuntimeProfile):
            raw.setdefault("target_runtime", profile.target_runtime)
            raw.setdefault("runtime_abi", profile.runtime_abi)
            raw["runtime_profile"] = profile.backend_profile
        elif (
            isinstance(profile, Mapping)
            and {"backend", "target"}.issubset(profile)
            and ("profile" in profile or "backend_profile" in profile)
        ):
            envelope = RoleRuntimeProfile.model_validate(profile)
            raw.setdefault("target_runtime", envelope.target_runtime)
            raw.setdefault("runtime_abi", envelope.runtime_abi)
            raw["runtime_profile"] = envelope.backend_profile
        return raw

    @model_validator(mode="after")
    def validate_shape(self) -> ModelRuntimeSpec:
        if self.runtime_profile is None and not self.role_runtime_specs:
            raise ValueError("ModelRuntimeSpec requires runtime_profile or role_runtime_specs")
        if self.runtime_profile is not None and self.role_runtime_specs:
            raise ValueError("ModelRuntimeSpec cannot mix runtime_profile with role_runtime_specs")
        if self.runtime_profile is not None and not isinstance(self.runtime_profile, BackendRuntimeProfile):
            raise ValueError("ModelRuntimeSpec.runtime_profile must be a typed BackendRuntimeProfile")
        if self.runtime_profile is not None:
            target_runtime = self.target_runtime
            if target_runtime is None:
                target_runtime = "acl" if self.runtime_profile.backend == "ascend" else self.runtime_profile.backend
                object.__setattr__(self, "target_runtime", target_runtime)
            if self.runtime_profile.backend == "ascend" and target_runtime != "acl":
                raise ValueError("backend='ascend' requires target_runtime='acl'")
            if _is_legacy_runtime_alias(target_runtime):
                raise ValueError("target_runtime must use the canonical runtime family")
            if self.runtime_abi is not None and not re.search(r"\d", self.runtime_abi):
                raise ValueError("runtime_abi must contain a version identifier")
        return self

    @property
    def validated_deployment(self) -> Any:
        return self.deployment

    @property
    def deployment_snapshot(self) -> Any:
        return self.deployment

    @property
    def backend(self) -> str | None:
        if self.runtime_profile is None:
            return None
        return self.runtime_profile.backend

    @property
    def backend_profile(self) -> BackendRuntimeProfile | None:
        return self.runtime_profile

    @property
    def runtime_profile_config(self) -> RoleRuntimeProfile | None:
        if self.runtime_profile is None:
            return None
        return RoleRuntimeProfile(
            backend=self.runtime_profile.backend,
            target={"runtime": self.target_runtime or self.runtime_profile.backend, "runtime_abi": self.runtime_abi},
            profile=self.runtime_profile,
        )

    @classmethod
    def from_validated_deployment(
        cls,
        deployment: ValidatedDeployment,
        *,
        execution_policy: ExecutionPolicy | None = None,
    ) -> ModelRuntimeSpec:
        policy = execution_policy or ExecutionPolicy()
        if deployment.role_runtime_profiles:
            specs = {
                role: RoleRuntimeSpec(
                    deployment_binding=role,
                    backend=profile.backend,
                    target_runtime=profile.target_runtime,
                    runtime_abi=profile.runtime_abi,
                    backend_profile=profile.backend_profile,
                )
                for role, profile in deployment.role_runtime_profiles.items()
            }
            return cls(deployment=deployment, execution_policy=policy, role_runtime_specs=specs)
        profile = next(iter(deployment.role_runtime_profiles.values()), None)
        if profile is None:
            profile = deployment.deployment.runtime_profile
        if profile is None:
            raise ValueError("validated single-model deployment has no runtime profile")
        return cls(
            deployment=deployment,
            execution_policy=policy,
            runtime_profile=profile.backend_profile,
            target_runtime=profile.target_runtime,
            runtime_abi=profile.runtime_abi,
        )

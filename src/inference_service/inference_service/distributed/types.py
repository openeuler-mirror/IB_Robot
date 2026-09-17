"""Transport-neutral values for the distributed inference protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from types import MappingProxyType

from inference_manifest import PolicyMetadata, ValidatedManifest

# v6: DistributedInferenceResult, DispatchInfer and ScheduledDispatchInfer add
# the result-level ``execution_horizon`` field. ROS interface definitions are
# not wire-compatible across this boundary, so every edge/cloud deployment and
# all three regenerated interfaces must move together; peers still on v5 are
# rejected during the heartbeat identity handshake.
PROTOCOL_VERSION = 6


class UnsupportedDistributedRuntimeError(ValueError):
    """The distributed wire contract is intentionally policy-only."""

    code = "distributed_tensor_model_unsupported"
    stage = "validation"
    recoverable = False

    def __init__(self, interface: object, model_type: object = "") -> None:
        self.details = {
            "interface": str(interface),
            "model_type": str(model_type),
        }
        super().__init__(
            "distributed inference supports only interface='policy'; "
            f"tensor_model runtime {interface!s}/{model_type!s} is unsupported"
        )


def _immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


class PeerRole(IntEnum):
    EDGE = 1
    CLOUD = 2


class Operation(IntEnum):
    UNKNOWN = 0
    INFER = 1
    RESET = 2
    CANCEL = 3


@dataclass(frozen=True)
class FeatureSummary:
    semantic: str
    feature_type: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.semantic or not self.feature_type:
            raise ValueError("feature semantic and type must be non-empty")
        if not self.shape or any(dimension < 1 for dimension in self.shape):
            raise ValueError("feature summary shape dimensions must be positive")


@dataclass(frozen=True)
class PolicySummary:
    policy_type: str
    inputs: tuple[FeatureSummary, ...]
    outputs: tuple[FeatureSummary, ...]
    action_dimension: int
    nominal_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if not self.policy_type:
            raise ValueError("policy_type must be non-empty")
        if not self.inputs or not self.outputs:
            raise ValueError("policy summaries require non-empty inputs and outputs")
        if self.action_dimension < 1:
            raise ValueError("action_dimension must be positive")
        if self.nominal_chunk_size is not None and self.nominal_chunk_size < 1:
            raise ValueError("nominal_chunk_size must be positive when provided")


@dataclass(frozen=True)
class PipelineIdentity:
    pipeline_id: str
    manifest_schema_version: int
    bundle_uuid: str
    bundle_revision: int
    bundle_digest: str
    deployment_name: str
    deployment_uuid: str
    deployment_revision: int
    deployment_fingerprint: str
    policy: PolicySummary
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "pipeline_id",
            "bundle_uuid",
            "bundle_digest",
            "deployment_name",
            "deployment_uuid",
            "deployment_fingerprint",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if self.protocol_version < 1 or self.manifest_schema_version < 1:
            raise ValueError("protocol and manifest schema versions must be positive")
        if self.bundle_revision < 1 or self.deployment_revision < 1:
            raise ValueError("bundle and deployment revisions must be positive")


@dataclass(frozen=True)
class StructuredError:
    """Policy-only distributed wire error; not a local runtime result type."""

    code: str
    message: str
    stage: str
    recoverable: bool = False
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code or not self.message or not self.stage:
            raise ValueError("structured error code, message, and stage must be non-empty")
        object.__setattr__(self, "details", _immutable_mapping(self.details))


@dataclass(frozen=True)
class PipelineStatus:
    role: PeerRole
    identity: PipelineIdentity
    sequence: int
    session_id: str = ""
    session_generation: int = 0
    ready: bool = False
    runtime_state: str = "created"
    reset_supported: bool = False
    cancellation_supported: bool = False
    error: StructuredError | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("status sequence must be positive")
        if self.session_generation < 0:
            raise ValueError("session_generation cannot be negative")
        if self.ready and (not self.session_id or self.session_generation < 1):
            raise ValueError("ready status requires a live session ID and generation")


@dataclass(frozen=True, slots=True)
class StreamReference:
    observation_key: str
    stream_id: str

    def __post_init__(self) -> None:
        if not self.observation_key or not self.stream_id:
            raise ValueError("stream references require observation_key and stream_id")


@dataclass(frozen=True)
class DistributedRequest:
    operation: Operation
    pipeline_id: str
    request_id: str
    session_id: str
    session_generation: int
    deployment_fingerprint: str
    inputs: Mapping[str, object] = field(default_factory=dict)
    prompt: str | None = None
    deadline: datetime | None = None
    target_request_id: str = ""
    observation_timestamp_ns: int = 0
    stream_references: tuple[StreamReference, ...] = ()
    aligned_timestamps_ns: tuple[int, ...] = ()
    aligned_tensors: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("pipeline_id", "request_id", "session_id", "deployment_fingerprint"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")
        if self.session_generation < 1:
            raise ValueError("session_generation must be positive")
        if self.operation is Operation.UNKNOWN:
            raise ValueError("distributed requests require a known operation")
        if self.operation is Operation.CANCEL and not self.target_request_id:
            raise ValueError("cancel requests require target_request_id")
        if self.operation is not Operation.CANCEL and self.target_request_id:
            raise ValueError("target_request_id is valid only for cancel requests")
        if len(self.aligned_timestamps_ns) != len(self.aligned_tensors):
            raise ValueError("aligned timestamp and tensor arrays must have equal length")
        if self.aligned_timestamps_ns and not self.stream_references:
            raise ValueError("aligned history requires stream references")
        if any(ts < 0 for ts in self.aligned_timestamps_ns):
            raise ValueError("aligned timestamps must be non-negative")
        if self.observation_timestamp_ns < 0:
            raise ValueError("observation_timestamp_ns cannot be negative")
        if self.operation is not Operation.INFER and (self.observation_timestamp_ns or self.stream_references):
            raise ValueError("observation timestamp and stream references are valid only for inference requests")
        if self.stream_references and self.observation_timestamp_ns <= 0:
            raise ValueError("stream-backed inference requires a positive observation timestamp")
        observation_keys = [reference.observation_key for reference in self.stream_references]
        stream_ids = [reference.stream_id for reference in self.stream_references]
        if len(set(observation_keys)) != len(observation_keys):
            raise ValueError("stream references contain duplicate observation keys")
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("stream references contain duplicate stream IDs")
        collisions = sorted(set(observation_keys) & set(self.inputs))
        if collisions:
            raise ValueError(f"stream references collide with tensor input semantics: {collisions}")
        object.__setattr__(self, "inputs", _immutable_mapping(self.inputs))


@dataclass(frozen=True)
class DistributedResult:
    operation: Operation
    pipeline_id: str
    request_id: str
    session_id: str
    session_generation: int
    deployment_fingerprint: str
    success: bool
    action: object | None = None
    actual_chunk_size: int = 0
    backend_latency_ms: float = 0.0
    performance: Mapping[str, object] = field(default_factory=dict)
    backend_ready: bool = False
    backend_state: str = ""
    target_request_id: str = ""
    error: StructuredError | None = None
    execution_horizon: int = 0

    def __post_init__(self) -> None:
        if not self.pipeline_id or not self.request_id or not self.deployment_fingerprint:
            raise ValueError("result pipeline, request, and deployment identity must be non-empty")
        if self.session_generation < 0:
            raise ValueError("session_generation cannot be negative")
        if (
            isinstance(self.actual_chunk_size, bool)
            or not isinstance(self.actual_chunk_size, int)
            or self.actual_chunk_size < 0
        ):
            raise ValueError("actual_chunk_size must be a non-negative integer")
        if not math.isfinite(self.backend_latency_ms) or self.backend_latency_ms < 0:
            raise ValueError("backend_latency_ms must be finite and non-negative")
        object.__setattr__(self, "performance", _immutable_mapping(self.performance))
        if self.success and self.error is not None:
            raise ValueError("successful results cannot contain an error")
        if self.success and self.operation is Operation.UNKNOWN:
            raise ValueError("successful results require a known operation")
        if not self.success and self.error is None:
            raise ValueError("failed results require a structured error")
        if self.operation is Operation.INFER and self.success:
            if self.action is None or self.actual_chunk_size < 1:
                raise ValueError("successful inference results require an action and actual chunk size")
            if (
                isinstance(self.execution_horizon, bool)
                or not isinstance(self.execution_horizon, int)
                or self.execution_horizon < 0
                or self.execution_horizon > self.actual_chunk_size
            ):
                raise ValueError("execution_horizon must be zero or within actual_chunk_size")
        elif self.actual_chunk_size != 0 or self.execution_horizon != 0:
            raise ValueError("non-inference or failed results must report zero chunk and execution horizon")


def summarize_policy(policy: PolicyMetadata) -> PolicySummary:
    def summarize(features: Mapping[str, object]) -> tuple[FeatureSummary, ...]:
        return tuple(
            FeatureSummary(semantic=semantic, feature_type=feature.type, shape=feature.shape)
            for semantic, feature in sorted(features.items())
        )

    action_dimension = policy.output_features["action"].shape[-1]
    return PolicySummary(
        policy_type=policy.policy_type,
        inputs=summarize(policy.input_features),
        outputs=summarize(policy.output_features),
        action_dimension=action_dimension,
        nominal_chunk_size=policy.nominal_chunk_size,
    )


def build_pipeline_identity(pipeline_id: str, validated_manifest: ValidatedManifest) -> PipelineIdentity:
    model = getattr(getattr(validated_manifest, "manifest", None), "model", None)
    interface = getattr(model, "interface", None)
    if interface is not None and interface != "policy":
        raise UnsupportedDistributedRuntimeError(interface, getattr(model, "model_type", ""))
    return PipelineIdentity(
        pipeline_id=pipeline_id,
        manifest_schema_version=validated_manifest.manifest.schema_version,
        bundle_uuid=validated_manifest.manifest.bundle.uuid,
        bundle_revision=validated_manifest.manifest.bundle.revision,
        bundle_digest=validated_manifest.manifest.bundle.digest.value,
        deployment_name=validated_manifest.deployment_name,
        deployment_uuid=validated_manifest.deployment.uuid,
        deployment_revision=validated_manifest.deployment.revision,
        deployment_fingerprint=validated_manifest.fingerprint,
        policy=summarize_policy(validated_manifest.policy),
    )


def deployed_pipeline_fingerprint(
    manifest_fingerprint: str,
    contract_fingerprint: str,
    *,
    execution_mode: str,
    processor_ownership: Mapping[str, str],
) -> str:
    """Compose the model deployment and observation contract identities."""
    if not manifest_fingerprint or not contract_fingerprint or not execution_mode:
        raise ValueError("deployment, contract, and execution mode fingerprints must be non-empty")
    payload = {
        "format": "ibrobot.deployed-pipeline-v1",
        "manifest_fingerprint": manifest_fingerprint,
        "contract_fingerprint": contract_fingerprint,
        "execution_mode": execution_mode,
        "processor_ownership": dict(sorted(processor_ownership.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def identity_error(local: PipelineIdentity, remote: PipelineIdentity) -> StructuredError | None:
    fields = (
        ("protocol_version", "protocol_version_mismatch"),
        ("pipeline_id", "pipeline_id_mismatch"),
        ("manifest_schema_version", "manifest_schema_version_mismatch"),
        ("bundle_uuid", "bundle_uuid_mismatch"),
        ("bundle_revision", "bundle_revision_mismatch"),
        ("bundle_digest", "bundle_digest_mismatch"),
        ("deployment_name", "deployment_mismatch"),
        ("deployment_uuid", "deployment_uuid_mismatch"),
        ("deployment_revision", "deployment_revision_mismatch"),
        ("deployment_fingerprint", "deployment_fingerprint_mismatch"),
        ("policy", "policy_summary_mismatch"),
    )
    for field_name, code in fields:
        local_value = getattr(local, field_name)
        remote_value = getattr(remote, field_name)
        if local_value != remote_value:
            return StructuredError(
                code=code,
                message=f"distributed pipeline identity mismatch for {field_name}",
                stage="handshake",
                details={"field": field_name, "local": local_value, "remote": remote_value},
            )
    return None


def structured_error_from_exception(exc: Exception, stage: str) -> StructuredError:
    details = getattr(exc, "details", {})
    if not isinstance(details, Mapping):
        details = {}
    capability = getattr(exc, "capability", None)
    if capability is not None:
        details = {**details, "capability": capability}
    for field_name in ("operation_started", "outcome_known"):
        value = getattr(exc, field_name, None)
        if isinstance(value, bool):
            details = {**details, field_name: value}
    evidence = getattr(exc, "evidence", None)
    if evidence is not None and callable(getattr(evidence, "to_dict", None)):
        details = {**details, "evidence": evidence.to_dict()}
    recovery = getattr(exc, "recovery", None)
    if recovery is not None and callable(getattr(recovery, "to_dict", None)):
        details = {**details, "recovery": recovery.to_dict()}
    return StructuredError(
        code=str(getattr(exc, "code", "operation_failed")),
        message=str(exc) or type(exc).__name__,
        stage=stage,
        recoverable=bool(getattr(exc, "recoverable", False)),
        details=details,
    )

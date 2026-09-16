"""Hardware-independent backend contracts and immutable runtime values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from inference_manifest import (
    BackendRuntimeProfile,
    CompiledDeployment,
    Deployment,
    DeploymentTarget,
    ModelDescriptor,
    PolicyMetadata,
    RoleRuntimeProfile,
    ValidatedManifest,
    resolve_bundle_file,
)


def _immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


class BackendState(str, Enum):
    CREATED = "created"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class BackendAdmissionEvidence:
    """Evidence gates for concurrency declarations above conservative defaults."""

    overlapping_calls: bool = False
    output_isolation: bool = False
    failure_isolation: bool = False
    deterministic_cleanup: bool = False
    sdk_initialization: bool = False
    multi_instance_execution: bool = False
    independent_close: bool = False


@dataclass(frozen=True)
class BackendPriorityMapping:
    """Backend-owned mapping from generic priorities to native priorities."""

    native_priorities: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.native_priorities) < 2:
            raise ValueError("multi-priority mapping must expose at least two priorities")
        if any(type(priority) is not int or priority < 0 for priority in self.native_priorities):
            raise ValueError("native priorities must be non-negative integers")

    @property
    def generic_level_count(self) -> int:
        return len(self.native_priorities)

    def map_generic(self, priority: int) -> int:
        if type(priority) is not int or priority < 0 or priority >= len(self.native_priorities):
            raise ValueError(f"generic priority must be in [0, {len(self.native_priorities) - 1}], got {priority!r}")
        return self.native_priorities[priority]


@dataclass(frozen=True)
class BackendCapabilities:
    resettable: bool = False
    stateful: bool = False
    thread_safe: bool = False
    max_in_flight_per_instance: int = 1
    supports_multiple_instances: bool = False
    resource_domain: str | None = None
    max_in_flight_per_resource_domain: int | None = None
    supports_attention: bool = False
    supports_cancellation: bool = False
    admission_evidence: BackendAdmissionEvidence | None = None
    # New scheduler fields stay after every legacy field so positional callers
    # retain the pre-scheduler BackendCapabilities constructor contract.
    # Physical runtime identity is distinct from resource_domain, which only
    # controls process-local admission.
    hardware_resource_id: str | None = None
    # None means the backend is single-priority and accepts only generic 0.
    # Multi-priority backends own validation and generic-to-native mapping.
    priority_mapping: BackendPriorityMapping | None = None
    # Loaded backend can asynchronously execute roles with request-owned IO.
    supports_isolated_stage_execution: bool = False

    def __post_init__(self) -> None:
        if self.max_in_flight_per_instance < 1:
            raise ValueError("max_in_flight_per_instance must be at least one")
        if self.max_in_flight_per_instance > 1:
            if not self.thread_safe:
                raise ValueError("max_in_flight_per_instance greater than one requires thread_safe=true")
            self._require_evidence(
                "max_in_flight_per_instance greater than one",
                "overlapping_calls",
                "output_isolation",
                "failure_isolation",
                "deterministic_cleanup",
            )

        if self.hardware_resource_id is not None and not self.hardware_resource_id.strip():
            raise ValueError("hardware_resource_id must be a non-empty string when provided")
        if self.resource_domain is not None and not self.resource_domain.strip():
            raise ValueError("resource_domain must be a non-empty string when provided")
        if self.resource_domain is None and self.max_in_flight_per_resource_domain is not None:
            raise ValueError("max_in_flight_per_resource_domain requires resource_domain")
        if self.max_in_flight_per_resource_domain is not None and self.max_in_flight_per_resource_domain < 1:
            raise ValueError("max_in_flight_per_resource_domain must be at least one")

        if self.supports_multiple_instances or self.resource_domain_limit > 1:
            if self.resource_domain_limit > 1 and not self.supports_multiple_instances:
                raise ValueError("a resource-domain limit greater than one requires supports_multiple_instances=true")
            self._require_evidence(
                "multiple backend instances",
                "sdk_initialization",
                "multi_instance_execution",
                "failure_isolation",
                "independent_close",
            )

    @property
    def resource_domain_limit(self) -> int:
        if self.resource_domain is None:
            return 1
        return self.max_in_flight_per_resource_domain or 1

    def _require_evidence(self, declaration: str, *fields: str) -> None:
        missing = [field_name for field_name in fields if not getattr(self.admission_evidence, field_name, False)]
        if missing:
            raise ValueError(f"{declaration} requires conformance evidence for: {', '.join(missing)}")


@dataclass(frozen=True)
class RuntimeContext:
    """Validated manifest selection and local operational options for a backend.

    ``runtime_options`` is retained for facade/executor options that have not
    moved into the typed v3 profile yet.  Backend identity and device placement
    are resolved from ``runtime_profile`` first, so a builder does not need to
    reconstruct them from an untyped mapping.
    """

    validated_manifest: ValidatedManifest
    runtime_options: Mapping[str, object] = field(default_factory=dict)
    priority_scheduling: bool = False
    runtime_profile: RoleRuntimeProfile | BackendRuntimeProfile | None = None
    role: str | None = None
    resolved_artifacts: Mapping[str, Path] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.priority_scheduling, bool):
            raise TypeError("priority_scheduling must be a bool")
        deployment = self.validated_manifest.deployment
        selected_profile = self.runtime_profile
        if selected_profile is None:
            role_profiles = getattr(self.validated_manifest, "role_runtime_profiles", {})
            if self.role is not None:
                selected_profile = role_profiles.get(self.role)
            if selected_profile is None:
                selected_profile = getattr(deployment, "runtime_profile", None)
        if selected_profile is not None and not isinstance(
            selected_profile, RoleRuntimeProfile | BackendRuntimeProfile
        ):
            raise TypeError("runtime_profile must be a typed v3 backend or role runtime profile")
        artifacts: dict[str, Path] = {}
        if isinstance(deployment, CompiledDeployment):
            for role, artifact in deployment.artifacts.items():
                artifacts[role] = resolve_bundle_file(self.validated_manifest.bundle_root, artifact.path)
        object.__setattr__(self, "runtime_options", _immutable_mapping(self.runtime_options))
        object.__setattr__(self, "runtime_profile", selected_profile)
        object.__setattr__(self, "resolved_artifacts", MappingProxyType(artifacts))

    @property
    def deployment(self) -> Deployment:
        return self.validated_manifest.deployment

    @property
    def deployment_name(self) -> str:
        return self.validated_manifest.deployment_name

    @property
    def policy(self) -> PolicyMetadata:
        policy = self.validated_manifest.policy
        if policy is None:
            raise ValueError(
                f"RuntimeContext.policy is unavailable for {self.interface}/{self.model_type}/{self.operation}"
            )
        return policy

    @property
    def model(self) -> ModelDescriptor:
        return self.validated_manifest.manifest.model

    @property
    def target(self) -> DeploymentTarget | None:
        profile = self.runtime_profile
        if isinstance(profile, RoleRuntimeProfile):
            return profile.target
        deployment = self.deployment
        return getattr(deployment, "target", None) if isinstance(deployment, CompiledDeployment) else None

    @property
    def identity(self):
        """Return the top-level or selected role v3 identity."""

        if self.role is not None:
            role_identities = getattr(self.validated_manifest, "role_identities", {})
            identity = role_identities.get(self.role)
            if identity is not None:
                return identity
        return self.validated_manifest.top_level_identity

    @property
    def interface(self) -> str:
        return self.identity.interface

    @property
    def model_type(self) -> str:
        return self.identity.model_type

    @property
    def operation(self) -> str:
        return self.identity.operation

    @property
    def backend_profile(self) -> BackendRuntimeProfile | None:
        profile = self.runtime_profile
        if isinstance(profile, RoleRuntimeProfile):
            return profile.backend_profile
        if isinstance(profile, BackendRuntimeProfile):
            return profile
        return None

    @property
    def backend(self) -> str:
        profile = self.backend_profile
        if profile is not None:
            return profile.backend
        deployment_backend = getattr(self.deployment, "backend", None)
        if isinstance(deployment_backend, str) and deployment_backend:
            return deployment_backend
        raise ValueError("runtime context does not expose a typed backend profile")

    @property
    def target_runtime(self) -> str | None:
        target = self.target
        return None if target is None else target.runtime

    @property
    def runtime_abi(self) -> str | None:
        target = self.target
        return None if target is None else target.runtime_abi

    @property
    def device_id(self) -> int | None:
        profile = self.backend_profile
        value = getattr(profile, "device_id", None)
        return value if type(value) is int else None

    @property
    def device(self) -> str | None:
        profile = self.backend_profile
        value = getattr(profile, "device", None)
        return value if isinstance(value, str) else None

    @property
    def deployment_fingerprint(self) -> str:
        return self.validated_manifest.fingerprint


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    inputs: Mapping[str, object]
    prompt: str | None = None
    deadline: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    # Keep priority after the legacy metadata field for positional compatibility.
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int) or self.priority < 0:
            raise ValueError("priority must be a non-negative integer")
        object.__setattr__(self, "inputs", _immutable_mapping(self.inputs))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class BackendHealth:
    state: BackendState
    ready: bool
    reason_code: str | None = None
    message: str | None = None
    recoverable: bool = False
    last_successful_inference_time: datetime | None = None
    failure_count: int = 0

    def __post_init__(self) -> None:
        if self.ready != (self.state is BackendState.READY):
            raise ValueError("backend readiness must match the READY state")
        if self.failure_count < 0:
            raise ValueError("failure_count cannot be negative")


__all__ = [
    "BackendAdmissionEvidence",
    "BackendCapabilities",
    "BackendHealth",
    "BackendPriorityMapping",
    "BackendState",
    "InferenceRequest",
    "RuntimeContext",
]

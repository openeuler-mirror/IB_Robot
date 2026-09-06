"""Static, lazy backend conformance registry.

The registry contains identity and capability data only.  Vendor runtimes are
resolved by the session builders after this gate succeeds; importing this
module must never import ACL, RKNNLite, TCIM, Torch, or a worker executable.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from inference_manifest import (
    AscendRuntimeProfile,
    BackendRuntimeProfile,
    Deployment,
    HisiliconRuntimeProfile,
    HMMRuntimeProfile,
    ONNXRuntimeProfile,
    RKNNRuntimeProfile,
    RoleRuntimeProfile,
    TorchRuntimeProfile,
)
from inference_service.backends.errors import BackendCompatibilityError, BackendRegistryError
from inference_service.backends.types import RuntimeContext

CANONICAL_BACKENDS = ("torch", "ascend", "hisilicon", "rknn", "hmm", "onnx")
VALID_INTERFACES = frozenset({"policy", "tensor_model"})
POLICY_MODEL_TYPES = frozenset({"act", "diffusion", "pi05", "smolvla"})

# These are service contracts, not deployment or SDK names.  A concrete
# adapter may still keep a private implementation variant in its asset files.
MODEL_TYPE_OPERATIONS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "act": frozenset({"predict"}),
        "diffusion": frozenset({"predict"}),
        "pi05": frozenset({"predict"}),
        "smolvla": frozenset({"predict"}),
        "ram_plus": frozenset({"recognize_tags"}),
        "sam2": frozenset({"automatic", "prompt"}),
        "siglip2": frozenset({"encode"}),
        "grounding_dino": frozenset({"detect"}),
        "graspgen": frozenset({"generate_grasps"}),
        "dummy_echo": frozenset({"echo"}),
        "zipvoice": frozenset({"synthesize"}),
        "fullsubnet": frozenset({"enhance"}),
        "silero_vad": frozenset({"vad"}),
        "speech_direction": frozenset({"enhance_and_vad"}),
    }
)

_FACTORY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")

IdentityKey = tuple[str, str, str]
TargetValidator = Callable[[Deployment], str | None]


def _canonical_identity(interface: str, model_type: str, operation: str) -> IdentityKey:
    interface = str(interface).strip()
    model_type = str(model_type).strip()
    operation = str(operation).strip()
    if interface not in VALID_INTERFACES:
        raise ValueError(f"interface must be one of {sorted(VALID_INTERFACES)}, got {interface!r}")
    if not model_type:
        raise ValueError("model_type must be non-empty")
    if not operation:
        raise ValueError("operation must be non-empty")
    if interface == "policy" and operation != "predict":
        raise ValueError("policy operation must be exactly 'predict'")
    canonical_operations = MODEL_TYPE_OPERATIONS.get(model_type)
    if canonical_operations is not None and operation not in canonical_operations:
        raise ValueError(
            f"model_type {model_type!r} does not support operation {operation!r}; "
            f"supported operations: {sorted(canonical_operations)}"
        )
    return interface, model_type, operation


def _identity_value(value: object) -> IdentityKey:
    if isinstance(value, tuple | list) and len(value) == 3:
        return _canonical_identity(value[0], value[1], value[2])
    if isinstance(value, Mapping):
        return _canonical_identity(value.get("interface"), value.get("model_type"), value.get("operation"))
    return _canonical_identity(
        getattr(value, "interface", None),
        getattr(value, "model_type", None),
        getattr(value, "operation", None),
    )


def _profile_type(value: object) -> type[BackendRuntimeProfile] | None:
    if isinstance(value, type) and issubclass(value, BackendRuntimeProfile):
        return value
    return None


@dataclass(frozen=True, init=False)
class ConformanceEvidence:
    """Evidence for one concrete v3 identity/backend/session combination."""

    interface: str
    model_type: str
    operation: str
    session_type: str | None
    profile_type: type[BackendRuntimeProfile] | None
    target_runtimes: frozenset[str]
    target_socs: frozenset[str]
    devices: frozenset[str]
    reference: str

    def __init__(
        self,
        interface: str,
        model_type: str,
        operation: str = "predict",
        reference: str = "",
        *,
        session_type: str | None = None,
        profile_type: type[BackendRuntimeProfile] | None = None,
        target_runtimes: frozenset[str] = frozenset(),
        target_socs: frozenset[str] = frozenset(),
        devices: frozenset[str] = frozenset(),
    ) -> None:
        identity = _canonical_identity(interface, model_type, operation)
        if session_type is not None and (not isinstance(session_type, str) or not session_type.strip()):
            raise ValueError("session_type must be non-empty when provided")
        if profile_type is not None and _profile_type(profile_type) is None:
            raise TypeError("profile_type must be a BackendRuntimeProfile type")
        object.__setattr__(self, "interface", identity[0])
        object.__setattr__(self, "model_type", identity[1])
        object.__setattr__(self, "operation", identity[2])
        object.__setattr__(self, "session_type", session_type)
        object.__setattr__(self, "profile_type", profile_type)
        object.__setattr__(self, "target_runtimes", frozenset(target_runtimes))
        object.__setattr__(self, "target_socs", frozenset(target_socs))
        object.__setattr__(self, "devices", frozenset(devices))
        object.__setattr__(self, "reference", reference)

    @property
    def identity(self) -> IdentityKey:
        return self.interface, self.model_type, self.operation

    @property
    def session(self) -> str | None:
        """Short alias used by conformance diagnostics."""

        return self.session_type

    def matches(self, identity: IdentityKey, *, profile: object = None, target: object = None) -> bool:
        if self.identity != identity:
            return False
        if self.profile_type is not None and not isinstance(profile, self.profile_type):
            return False
        if self.target_runtimes and getattr(target, "runtime", None) not in self.target_runtimes:
            return False
        if self.target_socs:
            soc = str(getattr(target, "soc", "")).lower()
            normalized_socs = tuple(value.lower() for value in self.target_socs)
            if not any(soc == value or soc.startswith(value) for value in normalized_socs):
                return False
        return not self.devices or getattr(profile, "device", None) in self.devices


@dataclass(frozen=True)
class BackendDescriptor:
    """Pure conformance declaration for one v3 backend identity set."""

    name: str
    target_validator: TargetValidator
    factory: str | None = None
    supported_identities: frozenset[IdentityKey] = frozenset()
    profile_types: frozenset[type[BackendRuntimeProfile]] = frozenset()
    conformance_evidence: frozenset[ConformanceEvidence] = frozenset()

    def __post_init__(self) -> None:
        normalized = frozenset(_identity_value(identity) for identity in self.supported_identities)
        object.__setattr__(self, "supported_identities", normalized)
        invalid_profiles = [profile for profile in self.profile_types if _profile_type(profile) is None]
        if invalid_profiles:
            raise TypeError(f"backend {self.name!r} declares invalid profile types: {invalid_profiles}")
        profile_types = frozenset(self.profile_types)
        object.__setattr__(self, "profile_types", profile_types)

    @property
    def supported_profile_types(self) -> frozenset[type[BackendRuntimeProfile]]:
        return self.profile_types

    @property
    def profile_type(self) -> type[BackendRuntimeProfile] | None:
        return next(iter(self.profile_types)) if len(self.profile_types) == 1 else None

    @property
    def evidence_identities(self) -> frozenset[IdentityKey]:
        return frozenset(evidence.identity for evidence in self.conformance_evidence)

    @property
    def evidence_pairs(self) -> frozenset[IdentityKey]:
        """Canonical identity triples retained under the historical property name."""

        return self.evidence_identities

    def validate_definition(self) -> None:
        if self.name not in CANONICAL_BACKENDS:
            raise BackendRegistryError(
                f"backend descriptor uses non-canonical name {self.name!r}", code="non_canonical_backend"
            )
        if self.factory is not None and not _FACTORY_PATTERN.fullmatch(self.factory):
            raise BackendRegistryError(
                f"backend {self.name!r} has invalid factory import string {self.factory!r}", code="invalid_factory"
            )
        if not self.supported_identities:
            raise BackendRegistryError(
                f"backend {self.name!r} must declare v3 model identity support", code="empty_model_support"
            )
        for interface, model_type, operation in self.supported_identities:
            try:
                _canonical_identity(interface, model_type, operation)
            except ValueError as exc:
                raise BackendRegistryError(
                    f"backend {self.name!r} declares invalid identity {(interface, model_type, operation)!r}: {exc}",
                    code="invalid_model_identity",
                ) from exc
        if not callable(self.target_validator):
            raise BackendRegistryError(
                f"backend {self.name!r} target validator is not callable", code="invalid_target_validator"
            )
        for evidence in self.conformance_evidence:
            if evidence.identity not in self.supported_identities:
                raise BackendRegistryError(
                    f"backend {self.name!r} conformance evidence claims undeclared identity {evidence.identity!r}",
                    code="evidence_overclaims_support",
                )
            if (
                evidence.profile_type is not None
                and self.profile_types
                and evidence.profile_type not in self.profile_types
            ):
                raise BackendRegistryError(
                    f"backend {self.name!r} evidence uses an undeclared profile type",
                    code="evidence_overclaims_support",
                )
        if not self.conformance_evidence:
            raise BackendRegistryError(
                f"backend {self.name!r} must publish conformance evidence", code="missing_conformance_evidence"
            )


def _context_identity(context: object) -> IdentityKey:
    identity = getattr(context, "identity", None)
    if identity is None:
        model = getattr(context, "model", None)
        identity = model
    if identity is None:
        raise BackendCompatibilityError(
            "runtime context does not expose a v3 model identity", code="invalid_model_identity"
        )
    try:
        return _identity_value(identity)
    except (TypeError, ValueError) as exc:
        raise BackendCompatibilityError(str(exc), code="invalid_model_identity") from exc


def _context_backend(context: object) -> str:
    backend = getattr(context, "backend", None)
    if isinstance(backend, str) and backend.strip():
        return backend.strip()
    profile = getattr(context, "backend_profile", None)
    backend = getattr(profile, "backend", getattr(profile, "backend_name", None))
    if isinstance(backend, str) and backend.strip():
        return backend.strip()
    deployment = getattr(context, "deployment", None)
    backend = getattr(deployment, "backend", None)
    if isinstance(backend, str) and backend.strip():
        return backend.strip()
    raise BackendCompatibilityError("runtime context does not expose a backend", code="backend_not_registered")


def _context_profile(context: object) -> object | None:
    profile = getattr(context, "backend_profile", None)
    if profile is not None:
        return profile
    profile = getattr(context, "runtime_profile", None)
    if isinstance(profile, RoleRuntimeProfile):
        return profile.backend_profile
    return profile


class BackendRegistry:
    """Immutable registry whose validation path is SDK-free and v3 keyed."""

    def __init__(self, descriptors: Mapping[str, BackendDescriptor]) -> None:
        copied = dict(descriptors)
        for key, descriptor in copied.items():
            if key != descriptor.name:
                raise BackendRegistryError(
                    f"backend descriptor key {key!r} does not match descriptor name {descriptor.name!r}",
                    code="descriptor_name_mismatch",
                )
            descriptor.validate_definition()
        self._descriptors = MappingProxyType(copied)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    @property
    def descriptors(self) -> Mapping[str, BackendDescriptor]:
        return self._descriptors

    def validate_static_registry(self) -> None:
        if tuple(self._descriptors) != CANONICAL_BACKENDS:
            raise BackendRegistryError(
                f"static backend registry must contain exactly {list(CANONICAL_BACKENDS)}, "
                f"got {list(self._descriptors)}",
                code="invalid_static_registry",
            )

    def descriptor(self, backend_name: str) -> BackendDescriptor:
        try:
            return self._descriptors[backend_name]
        except KeyError as exc:
            raise BackendRegistryError(
                f"backend {backend_name!r} is not registered; registered backends: {list(self._descriptors)}",
                code="backend_not_registered",
            ) from exc

    def validate(
        self,
        context: RuntimeContext,
        *,
        allowed_deployments: frozenset[str] | None = None,
    ) -> BackendDescriptor:
        backend_name = _context_backend(context)
        descriptor = self.descriptor(backend_name)
        identity = _context_identity(context)
        if identity not in descriptor.supported_identities:
            raise BackendCompatibilityError(
                f"backend {descriptor.name!r} does not support v3 identity {identity!r} "
                f"for deployment {getattr(context, 'deployment_name', '')!r}",
                code="unsupported_model_backend_pair",
            )

        profile = _context_profile(context)
        if descriptor.profile_types and not any(
            isinstance(profile, profile_type) for profile_type in descriptor.profile_types
        ):
            expected = sorted(profile_type.__name__ for profile_type in descriptor.profile_types)
            raise BackendCompatibilityError(
                f"backend {descriptor.name!r} requires a typed profile in {expected}, got "
                f"{type(profile).__name__ if profile is not None else None}",
                code="backend_profile_mismatch",
            )

        target = getattr(context, "target", None)
        evidence = next(
            (
                item
                for item in descriptor.conformance_evidence
                if item.matches(identity, profile=profile, target=target)
            ),
            None,
        )
        if evidence is None:
            raise BackendCompatibilityError(
                f"backend {descriptor.name!r} lacks conformance evidence for identity {identity!r}",
                code="missing_conformance_evidence",
            )

        deployment = getattr(context, "deployment", context)
        incompatibility = descriptor.target_validator(deployment)
        if incompatibility is not None:
            raise BackendCompatibilityError(
                f"deployment {getattr(context, 'deployment_name', '')!r} is incompatible with backend "
                f"{descriptor.name!r}: {incompatibility}",
                code="incompatible_backend_target",
            )
        if allowed_deployments is not None and getattr(context, "deployment_name", None) not in allowed_deployments:
            raise BackendCompatibilityError(
                f"deployment {getattr(context, 'deployment_name', None)!r} is not in the adapter supported "
                f"deployments {sorted(allowed_deployments)} for backend {descriptor.name!r}",
                code="adapter_deployment_mismatch",
            )
        return descriptor


def _target(deployment: object) -> object | None:
    try:
        target = getattr(deployment, "target", None)
    except ValueError:
        target = None
    if target is not None:
        return target
    profiles = getattr(deployment, "role_runtime_profiles", None) or {}
    return next((profile.target for profile in profiles.values()), None)


def _targets(deployment: object) -> tuple[object, ...]:
    profiles = getattr(deployment, "role_runtime_profiles", None) or {}
    if profiles:
        return tuple(profile.target for profile in profiles.values())
    target = _target(deployment)
    return () if target is None else (target,)


def _validate_torch(deployment: Deployment) -> str | None:
    backend = getattr(deployment, "backend", None)
    if backend != "torch":
        return "torch requires a torch runtime profile"
    return None


def _validate_ascend(deployment: Deployment) -> str | None:
    targets = _targets(deployment)
    if not targets:
        return "ascend requires a compiled deployment"
    for target in targets:
        runtime = getattr(target, "runtime", None)
        if runtime != "acl":
            return "target.runtime must be exactly 'acl'; put the ABI version in target.runtime_abi"
        runtime_abi = getattr(target, "runtime_abi", None)
        if runtime_abi is not None and not any(character.isdigit() for character in runtime_abi):
            return "target.runtime_abi must contain a version identifier"
    execution = getattr(deployment, "execution", ())
    artifacts = getattr(deployment, "artifacts", {})
    invalid_formats = sorted({artifacts[role].format for role in execution if role in artifacts} - {"om"})
    if invalid_formats:
        return f"Ascend execution artifacts must use format 'om', got {invalid_formats}"
    return None


def _validate_hisilicon(deployment: Deployment) -> str | None:
    if _target(deployment) is None:
        return "hisilicon requires a compiled deployment"
    target = _target(deployment)
    if str(getattr(target, "soc", "")).lower() != "sd3403":
        return f"target.soc must be 'sd3403', got {getattr(target, 'soc', None)!r}"
    if getattr(target, "runtime", None) != "hisilicon-worker":
        return "target.runtime must be 'hisilicon-worker'"
    if getattr(deployment, "execution", ()) != ("policy",):
        return "execution must be ['policy']"
    artifacts = getattr(deployment, "artifacts", {})
    missing = sorted({"policy", "worker"} - set(artifacts))
    if missing:
        return f"required artifact roles are missing: {missing}"
    if artifacts["policy"].format != "om" or artifacts["worker"].format != "executable":
        return "Hisilicon policy/worker artifact formats are invalid"
    return None


def _validate_rknn(deployment: Deployment) -> str | None:
    if _target(deployment) is None:
        return "rknn requires a compiled deployment"
    target = _target(deployment)
    if not re.fullmatch(r"rk3588[a-z0-9_.-]*", str(getattr(target, "soc", "")).lower()):
        return "target.soc is not in the RK3588 family"
    if not re.fullmatch(r"rknn(?:-lite(?:2)?|-toolkit-lite2)?", str(getattr(target, "runtime", ""))):
        return "target.runtime is not an RKNN runtime"
    return None


def _validate_hmm(deployment: Deployment) -> str | None:
    if _target(deployment) is None:
        return "hmm requires a compiled deployment"
    target = _target(deployment)
    if getattr(target, "runtime", None) not in {"hmm", "tcim"}:
        return "target.runtime must be 'hmm' or 'tcim'"
    artifacts = getattr(deployment, "artifacts", {})
    invalid_formats = sorted({artifact.format for artifact in artifacts.values()} - {"hmm", "pt", "pytorch", "json"})
    if invalid_formats:
        return f"HMM artifacts have unsupported formats: {invalid_formats}"
    return None


def _validate_onnx(deployment: Deployment) -> str | None:
    if _target(deployment) is None:
        return "onnx requires a compiled deployment"
    for target in _targets(deployment):
        if getattr(target, "runtime", None) != "onnx":
            return "target.runtime must be exactly 'onnx'"
    execution = getattr(deployment, "execution", ())
    artifacts = getattr(deployment, "artifacts", {})
    invalid_formats = sorted({artifacts[role].format for role in execution if role in artifacts} - {"onnx"})
    if invalid_formats:
        return f"onnx execution artifacts must use format 'onnx', got {invalid_formats}"
    return None


def _identities(*values: tuple[str, str, str]) -> frozenset[IdentityKey]:
    return frozenset(_canonical_identity(*value) for value in values)


def _evidence(
    *values: tuple[str, str, str],
    session_type: str,
    profile_type: type[BackendRuntimeProfile],
    target_runtimes: frozenset[str] = frozenset(),
    target_socs: frozenset[str] = frozenset(),
    devices: frozenset[str] = frozenset(),
) -> frozenset[ConformanceEvidence]:
    return frozenset(
        ConformanceEvidence(
            *value,
            session_type=session_type,
            profile_type=profile_type,
            target_runtimes=target_runtimes,
            target_socs=target_socs,
            devices=devices,
            reference="software-conformance",
        )
        for value in values
    )


_POLICY_TORCH = tuple(("policy", model_type, "predict") for model_type in POLICY_MODEL_TYPES)
_TENSOR_TORCH = (
    ("tensor_model", "ram_plus", "recognize_tags"),
    ("tensor_model", "sam2", "automatic"),
    ("tensor_model", "grounding_dino", "detect"),
    ("tensor_model", "siglip2", "encode"),
    ("tensor_model", "graspgen", "generate_grasps"),
    ("tensor_model", "dummy_echo", "echo"),
    ("tensor_model", "zipvoice", "synthesize"),
)
_POLICY_ASCEND = (("policy", "act", "predict"), ("policy", "pi05", "predict"))
_TENSOR_ASCEND = (
    ("tensor_model", "ram_plus", "recognize_tags"),
    ("tensor_model", "sam2", "automatic"),
    ("tensor_model", "sam2", "prompt"),
    ("tensor_model", "siglip2", "encode"),
    ("tensor_model", "grounding_dino", "detect"),
    ("tensor_model", "graspgen", "generate_grasps"),
    ("tensor_model", "zipvoice", "synthesize"),
    ("tensor_model", "fullsubnet", "enhance"),
    ("tensor_model", "silero_vad", "vad"),
)
# ONNX Runtime hosts the audio families on Ubuntu hosts where neither Torch
# nor an Ascend ACL runtime is available for the published graphs.
_TENSOR_ONNX = (
    ("tensor_model", "fullsubnet", "enhance"),
    ("tensor_model", "silero_vad", "vad"),
    ("tensor_model", "speech_direction", "enhance_and_vad"),
)


STATIC_BACKEND_DESCRIPTORS: Mapping[str, BackendDescriptor] = MappingProxyType(
    {
        "torch": BackendDescriptor(
            name="torch",
            supported_identities=_identities(*_POLICY_TORCH, *_TENSOR_TORCH),
            profile_types=frozenset({TorchRuntimeProfile}),
            conformance_evidence=_evidence(
                *_POLICY_TORCH,
                session_type="LeRobotTorchModelSession",
                profile_type=TorchRuntimeProfile,
                devices=frozenset({"cpu", "cuda", "mps", "npu"}),
            )
            | _evidence(
                *_TENSOR_TORCH,
                session_type="TorchModelSession",
                profile_type=TorchRuntimeProfile,
                devices=frozenset({"cpu", "cuda"}),
            ),
            target_validator=_validate_torch,
        ),
        "ascend": BackendDescriptor(
            name="ascend",
            supported_identities=_identities(*_POLICY_ASCEND, *_TENSOR_ASCEND),
            profile_types=frozenset({AscendRuntimeProfile}),
            conformance_evidence=_evidence(
                *_POLICY_ASCEND,
                *_TENSOR_ASCEND,
                session_type="AscendOmModelSession",
                profile_type=AscendRuntimeProfile,
                target_runtimes=frozenset({"acl"}),
                target_socs=frozenset({"ascend310p", "ascend310b", "ascend310p1", "ascend310b1"}),
            ),
            target_validator=_validate_ascend,
        ),
        "hisilicon": BackendDescriptor(
            name="hisilicon",
            supported_identities=_identities(("policy", "act", "predict")),
            profile_types=frozenset({HisiliconRuntimeProfile}),
            conformance_evidence=_evidence(
                ("policy", "act", "predict"),
                session_type="HisiliconModelSession",
                profile_type=HisiliconRuntimeProfile,
                target_runtimes=frozenset({"hisilicon-worker"}),
                target_socs=frozenset({"sd3403"}),
            ),
            target_validator=_validate_hisilicon,
        ),
        "rknn": BackendDescriptor(
            name="rknn",
            supported_identities=_identities(
                ("policy", "act", "predict"),
                ("policy", "smolvla", "predict"),
            ),
            profile_types=frozenset({RKNNRuntimeProfile}),
            conformance_evidence=_evidence(
                ("policy", "act", "predict"),
                ("policy", "smolvla", "predict"),
                session_type="RKNNModelSession",
                profile_type=RKNNRuntimeProfile,
                target_runtimes=frozenset({"rknn-lite", "rknn-lite2", "rknn"}),
                target_socs=frozenset({"rk3588"}),
            ),
            target_validator=_validate_rknn,
        ),
        "hmm": BackendDescriptor(
            name="hmm",
            supported_identities=_identities(
                ("policy", "pi05", "predict"),
                ("policy", "smolvla", "predict"),
            ),
            profile_types=frozenset({HMMRuntimeProfile}),
            conformance_evidence=_evidence(
                ("policy", "pi05", "predict"),
                ("policy", "smolvla", "predict"),
                session_type="HMMModelSession",
                profile_type=HMMRuntimeProfile,
                target_runtimes=frozenset({"hmm", "tcim"}),
                target_socs=frozenset({"xh2", "lq50", "m50"}),
            ),
            target_validator=_validate_hmm,
        ),
        "onnx": BackendDescriptor(
            name="onnx",
            supported_identities=_identities(*_TENSOR_ONNX),
            profile_types=frozenset({ONNXRuntimeProfile}),
            conformance_evidence=_evidence(
                *_TENSOR_ONNX,
                session_type="OnnxRuntimeModelSession",
                profile_type=ONNXRuntimeProfile,
                target_runtimes=frozenset({"onnx"}),
                devices=frozenset({"cpu", "cuda"}),
            ),
            target_validator=_validate_onnx,
        ),
    }
)

# Adapter code still uses this name to describe its supported service families;
# it is metadata only and is not a registry dispatch dimension.
PERCEPTION_FAMILIES = frozenset({"ram_plus", "sam2", "siglip2", "grounding_dino", "graspgen", "dummy_echo"})


__all__ = [
    "CANONICAL_BACKENDS",
    "ConformanceEvidence",
    "BackendDescriptor",
    "BackendRegistry",
    "MODEL_TYPE_OPERATIONS",
    "PERCEPTION_FAMILIES",
    "STATIC_BACKEND_DESCRIPTORS",
    "VALID_INTERFACES",
]

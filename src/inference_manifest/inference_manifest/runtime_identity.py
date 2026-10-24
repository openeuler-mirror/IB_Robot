"""Shared deployment identity; independent of robot YAML, ROS and model SDKs."""

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Protocol

from .models import RoleRuntimeProfile, _json_safe, _runtime_profiles

# Bump when the composition payload or its canonical serialization changes.
# Adapter implementations version their own identifier; this is not their ABI.
RUNTIME_COMPOSITION_FINGERPRINT_VERSION = 1


class _DeviceKind(str, Enum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"
    NPU = "npu"
    RKNN = "rknn"
    HMM = "hmm"
    HISILICON = "hisilicon"


_UNINDEXED_DEVICES = {_DeviceKind.CPU, _DeviceKind.MPS}
_DEVICE_PATTERN = re.compile(
    r"local/(?:"
    + "|".join(
        re.escape(kind.value) + ("" if kind in _UNINDEXED_DEVICES else r":(?:0|[1-9][0-9]*)") for kind in _DeviceKind
    )
    + ")"
)


def resolve_hmm_device_id(profile_device_id: int | None, options: Mapping[str, Any]) -> int:
    """Resolve placement for future shared planning/loading integration."""
    value = options.get("device_id", 0 if profile_device_id is None else profile_device_id)
    if type(value) is not int or value < 0:
        raise ValueError("HMM device_id must be a non-negative integer")
    if profile_device_id is not None and value != profile_device_id:
        raise ValueError("HMM device_id does not match the typed runtime profile")
    return value


def _torch(settings: Mapping[str, Any]) -> str:
    device = _DeviceKind(settings.get("device", _DeviceKind.CPU))
    if device in {_DeviceKind.CUDA, _DeviceKind.NPU}:
        index = settings.get("resolved_device_index")
        if type(index) is not int or index < 0:
            raise ValueError("Torch accelerator identity requires an explicit resolved torch_device_index")
        return f"{device.value}:{index}"
    return device.value


def _onnx(settings: Mapping[str, Any]) -> str:
    from .models import ONNXRuntimeProfile

    provider = ONNXRuntimeProfile(**settings).execution_provider
    devices = {"CPUExecutionProvider": _DeviceKind.CPU.value, "CUDAExecutionProvider": f"{_DeviceKind.CUDA.value}:0"}
    if provider not in devices:
        raise ValueError("physical-device identity requires an unambiguous ONNX CPU/CUDA profile")
    return devices[provider]


def _rknn(settings: Mapping[str, Any]) -> str:
    # The current RKNNLite loading path has no device-selection argument.
    device = f"{_DeviceKind.RKNN.value}:0"
    if settings.get("device_id", 0) != 0 or settings.get("device") not in {None, device}:
        raise ValueError("RKNN identity supports only the local default device (rknn:0, device_id=0)")
    return device


DEVICE_RESOLVERS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "torch": _torch,
    "ascend": lambda settings: f"{_DeviceKind.NPU.value}:{settings.get('device_id', 0)}",
    "rknn": _rknn,
    "onnx": _onnx,
    "hmm": lambda settings: f"{_DeviceKind.HMM.value}:{settings.get('device_id', 0)}",
    "hisilicon": lambda settings: f"{_DeviceKind.HISILICON.value}:0",
}


class RuntimePlacement(Protocol):
    """Profiles required by placement helpers, independent of a concrete loader."""

    @property
    def runtime_profile(self) -> RoleRuntimeProfile | None: ...

    @property
    def role_runtime_profiles(self) -> Mapping[str, RoleRuntimeProfile]: ...


def validate_runtime_backend_support(validated: RuntimePlacement) -> None:
    """Check profile presence/backend support only, not placement or visibility."""
    _validate_profiles(_runtime_profiles(validated.runtime_profile, validated.role_runtime_profiles))


def _validate_profiles(profiles: tuple[RoleRuntimeProfile, ...]) -> None:
    if not profiles:
        raise ValueError("missing runtime profile: at least one runtime profile is required")
    unsupported = {profile.backend for profile in profiles} - DEVICE_RESOLVERS.keys()
    if unsupported:
        raise ValueError(f"unsupported runtime backend(s): {sorted(unsupported)}; no physical-device mapping")


def physical_device(
    validated: RuntimePlacement,
    runtime_options: Mapping[str, Any] | None = None,
    *,
    torch_device_index: int | None = None,
) -> str:
    """Resolve placement from profiles and an explicitly resolved Torch index.

    Index-free Torch cuda/npu profiles follow the SDK's current device. Callers
    must supply that index from the context used to load/run the model and keep
    that context stable. No SDK state is queried here and zero is never assumed.
    Visibility remapping must separately pass validate_runtime_device_visibility.
    """
    if runtime_options is not None and not isinstance(runtime_options, Mapping):
        raise ValueError("runtime_options must be a mapping or None")
    options = runtime_options if runtime_options is not None else {}
    profiles = _runtime_profiles(validated.runtime_profile, validated.role_runtime_profiles)
    _validate_profiles(profiles)
    devices = set()
    for profile in profiles:
        settings = profile.profile.model_dump(mode="json", exclude_none=True)
        if profile.backend == "torch":
            settings["resolved_device_index"] = torch_device_index
        if profile.backend == "hmm":
            # Shared identity rule; the existing factory still defaults to zero
            # without an option. Loading integration must reconcile it later.
            settings["device_id"] = resolve_hmm_device_id(settings.get("device_id"), options)
        devices.add("local/" + DEVICE_RESOLVERS[profile.backend](settings))
    # A composite deployment may use CPU helper roles around one accelerator.
    # Its complete call is owned by the primary device; this is not a CPU quota.
    if len(devices) > 1:
        devices.discard(f"local/{_DeviceKind.CPU.value}")
    if len(devices) != 1:
        raise ValueError("runtime operation must resolve exactly one physical device")
    return devices.pop()


def runtime_composition_fingerprint(
    adapter: str, runtime_options: Mapping[str, Any], *, stage: str | None = None
) -> str:
    """Identify the adapter and exact options used to assemble a runtime host."""
    if not isinstance(adapter, str) or not adapter.strip():
        raise ValueError("adapter must be a non-blank string")
    if not isinstance(runtime_options, Mapping):
        raise ValueError("runtime_options must be a mapping")
    if stage is not None and not isinstance(stage, str):
        raise ValueError("stage must be a string or None")
    value = _json_safe(
        {
            "adapter": adapter,
            "fingerprint_format_version": RUNTIME_COMPOSITION_FINGERPRINT_VERSION,
            "options": runtime_options,
            "stage": stage,
        }
    )
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_runtime_device_visibility(device: str, target_id: str) -> None:
    """Require unremapped indices and PCI bus ordering in every CUDA host.

    Hosts must set this environment before SDK initialization and keep the
    same device inventory throughout their lifetime. This checks configuration,
    not SDK state; changing environment after initialization is unsupported.
    """
    if not isinstance(device, str) or _DEVICE_PATTERN.fullmatch(device) is None:
        raise ValueError(f"runtime target {target_id}: invalid physical device {device!r}")
    variables = {_DeviceKind.CUDA: "CUDA_VISIBLE_DEVICES", _DeviceKind.NPU: "ASCEND_RT_VISIBLE_DEVICES"}
    backend = device.removeprefix("local/").split(":", 1)[0]
    variable = variables.get(backend)
    if variable is not None and variable in os.environ:
        raise ValueError(
            f"runtime target {target_id}: unset {variable} (including empty values); "
            "device visibility remapping is unsupported for managed accelerators"
        )
    if backend == _DeviceKind.CUDA and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError(
            f"runtime target {target_id}: set CUDA_DEVICE_ORDER=PCI_BUS_ID before SDK initialization; "
            "other CUDA enumeration orders are unsupported for managed accelerators"
        )

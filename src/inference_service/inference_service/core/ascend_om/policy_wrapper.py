"""PolicyWrapper facades for compiled Ascend OM backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from inference_service.core.compiled_policy import (
    CompiledPolicyWrapper,
    normalize_backend_name,
)
from inference_service.core.pure_inference_engine import PolicyWrapper

ACTWrapper = None
ACT3403Policy = None


def __getattr__(name: str) -> Any:
    if name == "ACTWrapper":
        value = import_module("inference_service.core.ascend_om.ACTWrapper").ACTWrapper
    elif name == "ACT3403Policy":
        value = import_module("inference_service.core.ascend_om.ACTWrapper_3403").ACT3403Policy
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


class AscendOMPolicyWrapper(CompiledPolicyWrapper):
    """PolicyWrapper facade for generic Ascend ACL OM execution."""

    def __init__(self) -> None:
        super().__init__("ascend_om")


class AscendOM3403PolicyWrapper(CompiledPolicyWrapper):
    """PolicyWrapper facade for SD3403 worker-backed OM execution."""

    def __init__(self) -> None:
        super().__init__("ascend_om_3403")


def create_ascend_om_policy_wrapper(device: str, path: str | None = None) -> PolicyWrapper:
    normalized = normalize_backend_name(device)
    if normalized == "ascend_om_3403":
        # SD3403 currently only supports ACT.
        return AscendOM3403PolicyWrapper()
    if normalized == "ascend_om":
        del path
        return AscendOMPolicyWrapper()
    raise ValueError(f"Unsupported Ascend OM inference device: {device}")

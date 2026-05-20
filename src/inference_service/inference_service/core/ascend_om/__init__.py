"""Ascend OM backends for ``inference_service``."""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ACT3403Policy": (
        "inference_service.core.ascend_om.ACTWrapper_3403",
        "ACT3403Policy",
    ),
    "ACTWrapper": (
        "inference_service.core.ascend_om.ACTWrapper",
        "ACTWrapper",
    ),
    "OMmodel": (
        "inference_service.core.ascend_om.OMmodel",
        "OMmodel",
    ),
    "AscendOM3403PolicyWrapper": (
        "inference_service.core.ascend_om.policy_wrapper",
        "AscendOM3403PolicyWrapper",
    ),
    "AscendOMPolicyWrapper": (
        "inference_service.core.ascend_om.policy_wrapper",
        "AscendOMPolicyWrapper",
    ),
    "create_ascend_om_policy_wrapper": (
        "inference_service.core.ascend_om.policy_wrapper",
        "create_ascend_om_policy_wrapper",
    ),
    "resolve_om_model_path": (
        "inference_service.core.compiled_policy",
        "resolve_om_model_path",
    ),
}


def __getattr__(name: str) -> Any:
    module_name, attr_name = _LAZY_EXPORTS.get(name, (None, None))
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "ACT3403Policy",
    "ACTWrapper",
    "AscendOM3403PolicyWrapper",
    "AscendOMPolicyWrapper",
    "OMmodel",
    "create_ascend_om_policy_wrapper",
    "resolve_om_model_path",
]

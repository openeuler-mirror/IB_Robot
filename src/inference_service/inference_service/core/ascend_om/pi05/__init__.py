"""PI05 Ascend OM backend for ``inference_service``."""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "PI05OMModel": (
        "inference_service.core.ascend_om.pi05.PI05OMModel",
        "PI05OMModel",
    ),
    "PI05Wrapper": (
        "inference_service.core.ascend_om.pi05.PI05Wrapper",
        "PI05Wrapper",
    ),
    "AscendOMPi05PolicyWrapper": (
        "inference_service.core.ascend_om.pi05.policy_wrapper",
        "AscendOMPi05PolicyWrapper",
    ),
    "create_ascend_om_pi05_policy_wrapper": (
        "inference_service.core.ascend_om.pi05.policy_wrapper",
        "create_ascend_om_pi05_policy_wrapper",
    ),
    "resolve_pi05_om_paths": (
        "inference_service.core.ascend_om.pi05.policy_wrapper",
        "resolve_pi05_om_paths",
    ),
}


def __getattr__(name: str) -> Any:
    module_name, attr_name = _LAZY_EXPORTS.get(name, (None, None))
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)

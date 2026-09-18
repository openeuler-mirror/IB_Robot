"""Lazy registry for repository-owned Torch policy implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class PolicyProvider:
    """Model-owned hooks consumed by the native Torch session."""

    policy_class: type
    configure_config: Callable[..., object]
    validate: Callable[..., None]
    prepare: Callable[..., object]
    load_options: Mapping[str, object]
    execution_metadata: Callable[[object], Mapping[str, object]] | None = None


_PROVIDERS = {
    ("pi05", "torch", "npu"): "torch_models.pi05_ascend_310p.provider",
}


def resolve_policy_provider(model_type: str, backend: str, device: str) -> PolicyProvider | None:
    """Resolve a repository-owned policy by stable runtime identity."""

    module_name = _PROVIDERS.get((model_type, backend, device))
    if module_name is None:
        return None
    return import_module(module_name).create_provider()


__all__ = ["PolicyProvider", "resolve_policy_provider"]

"""Lazy policy-provider lookup for repository-owned Torch models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class PolicyProvider:
    """A custom policy class and its pre-construction config hook."""

    policy_class: type
    configure_config: Callable[..., object]


def resolve_policy_provider(architecture_class: str | None) -> PolicyProvider | None:
    """Resolve a repository-owned policy without importing optional models eagerly."""

    if architecture_class is None:
        return None
    if architecture_class != "pi05-ascend-310p":
        raise ValueError(f"unknown torch_models architecture_class: {architecture_class!r}")
    module = import_module("torch_models.pi05_ascend_310p")
    return PolicyProvider(
        policy_class=module.PI05Ascend310PPolicy,
        configure_config=module.configure_pi05_ascend_310p_config,
    )


__all__ = ["PolicyProvider", "resolve_policy_provider"]

"""Compatibility facade for PI05 Ascend OM compiled inference."""

from __future__ import annotations

from inference_service.core.compiled_policy import (
    CompiledPolicyWrapper,
)
from inference_service.core.pure_inference_engine import PolicyWrapper


class AscendOMPi05PolicyWrapper(CompiledPolicyWrapper):
    """PolicyWrapper facade for PI05 Ascend OM execution."""

    def __init__(self) -> None:
        super().__init__("ascend_om")


def create_ascend_om_pi05_policy_wrapper() -> PolicyWrapper:
    return AscendOMPi05PolicyWrapper()

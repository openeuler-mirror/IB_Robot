from __future__ import annotations

from inference_service.core.compiled_policy import (
    CompiledPolicyWrapper,
    normalize_backend_name,
)
from inference_service.core.pure_inference_engine import PolicyWrapper


class RKNNPolicyWrapper(CompiledPolicyWrapper):
    def __init__(self) -> None:
        super().__init__("rknn")


def create_rknn_policy_wrapper(device: str) -> PolicyWrapper:
    normalized = normalize_backend_name(device)
    if normalized == "rknn":
        return RKNNPolicyWrapper()
    raise ValueError(f"Unsupported RKNN inference device: {device}")

"""Model-agnostic helpers shared by perception adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inference_service.unified_runtime import ModelResult

from .perception_adapter import AdapterIdentity


def read_adapter_identity(root: Path, expected: AdapterIdentity) -> None:
    import json

    path = root / "assets" / "adapter.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load adapter identity {path}: {exc}") from exc
    required = {
        "interface": "tensor_model",
        "model_type": expected.model_type,
        "preprocessing": expected.preprocessing,
        "postprocessing": expected.postprocessing,
    }
    if any(value.get(name) != expected_value for name, expected_value in required.items()):
        raise ValueError(f"{expected.model_type} adapter identity mismatch: expected {required}, got {value}")
    declared_operation = value.get("operation")
    if declared_operation != expected.operation:
        raise ValueError(
            f"{expected.model_type} adapter operation mismatch: expected {expected.operation!r}, got {declared_operation!r}"
        )


def output_tensor(result: ModelResult, semantic: str, dtype=np.float32) -> np.ndarray:
    try:
        value = np.asarray(result.outputs[semantic], dtype=dtype)
    except KeyError as exc:
        raise RuntimeError(f"runtime result is missing {semantic!r}") from exc
    if not np.isfinite(value).all():
        raise RuntimeError(f"runtime output {semantic!r} contains non-finite values")
    return value


__all__ = ["output_tensor", "read_adapter_identity"]

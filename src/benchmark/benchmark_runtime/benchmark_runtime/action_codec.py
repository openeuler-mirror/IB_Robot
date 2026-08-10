"""Generic action decode helper for the StepBenchmark environment service.

Pre-native validation: this module performs GENERIC VariantsList → dict[str, np.ndarray]
decoding only. It does NOT impose provider-specific action dimension, range or
gripper semantics. Concrete adapters (e.g. ``benchmark_libero.action_codec``)
own the dimension/range/gripper validation.

The generic decoder preserves the original decoded dtype. It does NOT
silently cast int/float64 to float32 — the concrete adapter is responsible
for enforcing its exact dtype contract. The generic decoder still rejects
non-finite float values (NaN/Inf) because non-finite values are a
transport-level error, not a provider-specific constraint.

Generic rules enforced here:

- the VariantsList is decoded through ``tensormsg.TensorMsgConverter.from_variant``
  into a dict of torch tensors, then converted to numpy arrays;
- NaN/Inf float payloads are rejected (non-finite values are a transport
  error, not a provider-specific constraint);
- the decoded numpy array dtype is preserved verbatim — no silent cast to
  float32. Concrete adapters enforce their own dtype contract;
- no silent clipping, normalization, unit conversion or shape/rank
  validation. The concrete adapter validates dimension/range/gripper.

This module does NOT import ``rclpy``; it only needs the ROS message types
that ``TensorMsgConverter`` already imports and NumPy.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ActionDecodeError(ValueError):
    """Raised when a StepBenchmark action payload cannot be decoded."""


def decode_step_action_generic(action_msg: Any) -> dict[str, np.ndarray]:
    """Decode a ``VariantsList`` action payload into a generic dict of numpy arrays.

    Returns a mapping from variant key to contiguous numpy array. The array
    dtype is the original decoded dtype (preserved verbatim, NOT cast to
    float32 — concrete adapters enforce their own dtype contract).

    NaN/Inf float values are rejected at this generic level because
    non-finite values are a transport-level error, not a provider-specific
    constraint. No dimension, range or gripper validation is performed; the
    concrete adapter validates those.
    """
    from tensormsg.converter import TensorMsgConverter

    try:
        decoded = TensorMsgConverter.from_variant(action_msg)
    except Exception as exc:  # noqa: BLE001
        raise ActionDecodeError(f"failed to decode action VariantsList: {exc}") from exc

    if not isinstance(decoded, dict):
        raise ActionDecodeError(f"decoded action payload must be a dict, got {type(decoded).__name__}")

    if not decoded:
        raise ActionDecodeError("action payload is empty")

    result: dict[str, np.ndarray] = {}
    for key, tensor in decoded.items():
        if not isinstance(key, str):
            raise ActionDecodeError(f"action payload variant key must be a string, got {type(key).__name__}")
        try:
            arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        except Exception as exc:  # noqa: BLE001
            raise ActionDecodeError(f"action payload for key '{key}' could not be converted to numpy: {exc}") from exc

        # Preserve the original dtype. Do NOT silently cast to
        # float32 — the concrete adapter enforces its own dtype contract.
        # The generic decoder only rejects non-finite float values because
        # they are a transport-level error.
        if np.issubdtype(arr.dtype, np.floating) and not bool(np.all(np.isfinite(arr))):
            raise ActionDecodeError(f"action payload for key '{key}' contains NaN or Inf")

        result[str(key)] = np.ascontiguousarray(arr)

    return result

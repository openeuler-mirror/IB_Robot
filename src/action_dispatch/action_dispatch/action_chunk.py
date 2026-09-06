"""Shared normalization and validation for policy action chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NormalizedActionChunk:
    tensor: torch.Tensor
    array: np.ndarray


def normalize_action_chunk(action_chunk: object) -> tuple[torch.Tensor, np.ndarray]:
    """Preserve the legacy dispatcher normalization contract exactly."""

    action_chunk_tensor = action_chunk if hasattr(action_chunk, "detach") else torch.from_numpy(action_chunk)
    action_chunk_np = action_chunk_tensor.detach().cpu().numpy() if hasattr(action_chunk, "detach") else action_chunk
    if action_chunk_np.ndim == 3 and action_chunk_np.shape[0] == 1:
        action_chunk_np = action_chunk_np[0]
        action_chunk_tensor = action_chunk_tensor[0]
    if action_chunk_np.ndim == 1:
        action_chunk_np = action_chunk_np.reshape(1, -1)
        action_chunk_tensor = action_chunk_tensor.reshape(1, -1)
    return action_chunk_tensor, action_chunk_np


def validate_action_chunk(
    action_chunk: object,
    *,
    expected_action_dimension: int,
    reported_chunk_size: int | None = None,
) -> NormalizedActionChunk:
    """Normalize and validate action dimension, chunk identity, and finite values."""

    tensor, array = normalize_action_chunk(action_chunk)
    if array.ndim != 2:
        raise ValueError(f"action chunk must have rank 2 after optional batch normalization, got shape {array.shape}")
    if array.shape[0] < 1 or array.shape[1] != expected_action_dimension:
        raise ValueError(f"action chunk must have shape [steps, {expected_action_dimension}], got {array.shape}")
    if reported_chunk_size is not None and (reported_chunk_size < 1 or reported_chunk_size != array.shape[0]):
        raise ValueError(
            f"scheduled result reported chunk_size={reported_chunk_size}, "
            f"but action tensor contains {array.shape[0]} steps"
        )
    if not np.isfinite(array).all():
        raise ValueError("action chunk contains non-finite values")
    return NormalizedActionChunk(tensor=tensor, array=array)


def validate_execution_horizon(execution_horizon: int | None, chunk_size: int) -> int | None:
    """Validate an optional result-level execution prefix.

    Zero and ``None`` mean that the complete predicted chunk is executable.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive when validating execution horizon")
    if execution_horizon is None:
        return None
    if isinstance(execution_horizon, bool) or not isinstance(execution_horizon, int):
        raise ValueError("execution_horizon must be an integer")
    if execution_horizon == 0:
        return None
    if execution_horizon < 1 or execution_horizon > chunk_size:
        raise ValueError(f"execution_horizon={execution_horizon} must be between 1 and {chunk_size}")
    return execution_horizon


__all__ = [
    "NormalizedActionChunk",
    "normalize_action_chunk",
    "validate_execution_horizon",
    "validate_action_chunk",
]

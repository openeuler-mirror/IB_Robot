"""Attention-guided execution horizon estimation from arXiv:2602.21445.

NumPy port of ``_soft_pointer_prefix`` / ``bidir_soft_pointer`` from the
AutoHorizon reference implementation
(https://github.com/hatchetProject/AutoHorizon),
Copyright (c) the AutoHorizon authors, licensed under the Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0). Derivative work keeps the
attribution and license notice required by that license.

Divergence from the reference implementation: ``entropy_quantile=1.0`` is
used as-is, while the reference clamps the quantile with ``min(q, 0.999)``;
the two only differ at that extreme setting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HorizonEstimate:
    """Estimated executable prefix and diagnostics for one action chunk."""

    horizon: int
    forward_horizon: int
    backward_horizon: int
    join_row: int | None
    entropy_threshold: float


def _as_square_attention(attention: np.ndarray) -> np.ndarray:
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("attention must have at least two dimensions")
    matrix = values if values.ndim == 2 else values.mean(axis=tuple(range(values.ndim - 2)))
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError(f"attention must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("attention must contain finite non-negative weights")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("attention rows must have positive mass")
    return matrix / row_sums


def _prefix_pointer(
    attention: np.ndarray,
    *,
    hold_threshold: float,
    entropy_quantile: float,
    run_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    length = attention.shape[0]
    indices = np.arange(length, dtype=np.float64)
    pointer = np.maximum.accumulate((attention * indices).sum(axis=1))
    entropy = -(attention * np.log(np.maximum(attention, 1e-12))).sum(axis=1)
    if length > 1:
        entropy /= np.log(length)
    entropy_threshold = float(np.quantile(entropy, entropy_quantile)) if length > 1 else 0.0
    reliable = entropy <= entropy_threshold
    increments = np.diff(np.concatenate(([0.0], pointer)))
    holds = reliable & (increments < hold_threshold)

    stop = length - 1
    for start in range(0, length - run_length + 1):
        if holds[start : start + run_length].all():
            stop = start
            break
    return pointer, increments, reliable, stop, entropy_threshold


def estimate_execution_horizon(
    attention: np.ndarray,
    *,
    hold_threshold: float = 0.3,
    entropy_quantile: float = 0.9,
    run_length: int = 1,
) -> HorizonEstimate:
    """Estimate how many actions a flow-based policy can execute reliably.

    Attention may be a single ``[T, T]`` map or any tensor with leading layer,
    head, batch, or sampling dimensions. Leading dimensions are averaged before
    applying the paper's low-entropy bidirectional soft-pointer procedure.
    """
    if not np.isfinite(hold_threshold) or hold_threshold < 0.0:
        raise ValueError("hold_threshold must be a finite non-negative number")
    if not np.isfinite(entropy_quantile) or not 0.0 <= entropy_quantile <= 1.0:
        raise ValueError("entropy_quantile must be between 0 and 1")
    if isinstance(run_length, bool) or not isinstance(run_length, int) or run_length < 1:
        raise ValueError("run_length must be a positive integer")

    matrix = _as_square_attention(attention)
    length = matrix.shape[0]
    forward, _, _, stop_forward, entropy_threshold = _prefix_pointer(
        matrix,
        hold_threshold=hold_threshold,
        entropy_quantile=entropy_quantile,
        run_length=run_length,
    )
    backward_reversed, _, _, stop_backward_reversed, _ = _prefix_pointer(
        matrix[::-1, ::-1],
        hold_threshold=hold_threshold,
        entropy_quantile=entropy_quantile,
        run_length=run_length,
    )
    backward = np.flip(length - 1 - backward_reversed)
    forward_horizon = int(np.clip(np.floor(forward[stop_forward]) + 1, 1, length))
    backward_row = length - 1 - stop_backward_reversed
    backward_horizon = int(np.clip(np.floor(backward[backward_row]) + 1, 1, length))
    gap = backward - forward
    meeting_rows = np.flatnonzero(gap <= 1.0)
    join_row = int(meeting_rows[0]) if meeting_rows.size else None
    horizon = length if join_row is not None and forward_horizon + backward_horizon >= length else forward_horizon
    return HorizonEstimate(horizon, forward_horizon, backward_horizon, join_row, entropy_threshold)


__all__ = ["HorizonEstimate", "estimate_execution_horizon"]

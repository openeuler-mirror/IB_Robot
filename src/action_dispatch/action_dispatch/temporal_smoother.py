#!/usr/bin/env python3
"""
Temporal Smoothing Module for Action Chunking.

Provides cross-frame exponential smoothing for action chunks produced by
chunking-based policies (e.g., ACT, Diffusion Policy).

The smoothing ensures continuity between consecutive inference outputs,
preventing abrupt changes when the action plan is updated mid-execution.
"""

from dataclasses import dataclass
from threading import RLock

import numpy as np
import torch


@dataclass
class TemporalSmootherConfig:
    """Configuration for TemporalSmoother.

    Attributes:
        enabled: Whether to enable temporal smoothing. If False, acts as pass-through.
        chunk_size: Maximum number of actions in a chunk (used for weight calculation).
        temporal_ensemble_coeff: Exponential decay coefficient for smoothing weights.
            - 0.0: Uniform weighting (no preference for old/new)
            - Positive: More weight to older actions (stable, conservative)
            - Negative: More weight to newer actions (responsive, may cause jitter)
            Default: 0.01 (as per original ACT paper)
        device: Device for tensor operations. Can be 'cpu', 'cuda', 'npu:0', etc.
            If None, will use the device of input tensors.
    """

    enabled: bool = True
    chunk_size: int = 100
    temporal_ensemble_coeff: float = 0.01
    device: str | None = None

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")


class TemporalSmoother:
    """
    Cross-frame exponential smoother for action chunks.

    This smoother maintains an internal smoothed action plan and updates it
    when new inference results arrive. The key insight is that when a new
    chunk arrives, there's typically overlap with the remaining actions from
    the previous chunk. This overlap region is smoothed using exponential
    weighting to ensure continuity.

    Example:
        >>> config = TemporalSmootherConfig(enabled=True, chunk_size=100)
        >>> smoother = TemporalSmoother(config)
        >>>
        >>> # First inference: 100 actions
        >>> actions1 = np.random.randn(100, 7)  # 7-DOF robot
        >>> smoother.update(actions1, actions_executed=0)
        >>>
        >>> # Get actions one by one
        >>> for _ in range(30):
        ...     action = smoother.get_next_action()
        ...     # execute action...
        >>>
        >>> # Second inference arrives (30 actions executed during inference)
        >>> actions2 = np.random.randn(100, 7)
        >>> smoother.update(actions2, actions_executed=30)

    The smoothing formula for overlapping actions:
        blended = (old_sum + new_term) / cumulative_weights

    where:
        old_sum = old_action * cumulative_weight[old_count - 1]
        new_term = new_action * weight[old_count]
    """

    def __init__(self, config: TemporalSmootherConfig):
        self.config = config
        self._device = config.device
        # Control and inference callbacks can access the plan concurrently.
        self._lock = RLock()

        self._precompute_weights()
        self.reset()

    def _precompute_weights(self):
        """Precompute exponential weights for smoothing."""
        coeff = self.config.temporal_ensemble_coeff
        chunk_size = self.config.chunk_size

        self._weights = torch.exp(-coeff * torch.arange(chunk_size, dtype=torch.float32))
        self._weights_cumsum = torch.cumsum(self._weights, dim=0)

    def _get_device(self, tensor: torch.Tensor | np.ndarray) -> str:
        """Determine the device for tensor operations."""
        if self._device is not None:
            return self._device
        if isinstance(tensor, torch.Tensor):
            return tensor.device
        return "cpu"

    def _to_tensor(self, data: torch.Tensor | np.ndarray, device: str) -> torch.Tensor:
        """Convert input to tensor on specified device."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device)
        return data.float().to(device)

    def reset(self):
        """Reset internal state. Call this at the start of a new episode."""
        with self._lock:
            self._smoothed_actions: torch.Tensor | None = None
            self._action_counts: torch.Tensor | None = None

    @property
    def plan_length(self) -> int:
        """Current number of actions in the smoothed plan."""
        with self._lock:
            if self._smoothed_actions is None:
                return 0
            return self._smoothed_actions.shape[0]

    @property
    def is_enabled(self) -> bool:
        """Check if smoothing is enabled."""
        return self.config.enabled

    def get_next_action(self) -> torch.Tensor | np.ndarray:
        """
        Pop and return the next action from the smoothed plan.

        Returns:
            The next action as a tensor or numpy array.

        Raises:
            IndexError: If the plan is empty.
        """
        with self._lock:
            if self._smoothed_actions is None or self._smoothed_actions.shape[0] == 0:
                raise IndexError("Cannot get action from empty plan. Call update() first.")

            action = self._smoothed_actions[0]
            self._smoothed_actions = self._smoothed_actions[1:]
            if self._action_counts is not None:
                self._action_counts = self._action_counts[1:]

            return action

    def update(
        self,
        new_actions: torch.Tensor | np.ndarray,
        actions_executed_during_inference: int = 0,
    ) -> int:
        """
        Update the smoothed plan with new inference results.

        This is the core method that performs cross-frame smoothing.

        Args:
            new_actions: New action chunk from inference. Shape: (N, action_dim)
            actions_executed_during_inference: Number of actions that were executed
                from the previous plan while the inference was running. This is used
                for temporal alignment.

        Returns:
            The new plan length after update.

        Note:
            If smoothing is disabled (config.enabled=False), this method simply
            replaces the remaining plan with the new actions (after alignment).
        """
        if isinstance(new_actions, np.ndarray) and new_actions.ndim == 1:
            new_actions = new_actions.reshape(1, -1)

        if new_actions.shape[0] == 0:
            return self.plan_length

        device = self._get_device(new_actions)
        new_actions = self._to_tensor(new_actions, device)

        weights = self._weights.to(device)
        weights_cumsum = self._weights_cumsum.to(device)

        relevant_new = new_actions[actions_executed_during_inference:]

        with self._lock:
            if self._smoothed_actions is None or self._smoothed_actions.shape[0] == 0 or not self.config.enabled:
                next_actions = relevant_new.clone()
                next_counts = torch.ones((relevant_new.shape[0], 1), dtype=torch.long, device=device)
            else:
                next_actions, next_counts = self._apply_smoothing(
                    self._smoothed_actions, self._action_counts, relevant_new, weights, weights_cumsum
                )
            self._smoothed_actions, self._action_counts = next_actions, next_counts
            return self._smoothed_actions.shape[0]

    def prepare(self, selected_actions):
        """Prepare owned, aligned storage without changing the current plan.

        The owner serializes prepare/commit and consumption under its lock.
        """
        with self._lock:
            if selected_actions.shape[0] == 0:
                return None, None
            device = self._get_device(selected_actions)
            incoming = self._to_tensor(selected_actions, device).clone()
            if incoming.ndim != 2 or not torch.isfinite(incoming).all():
                raise ValueError("smoother actions must be finite and rank 2")
            if self._smoothed_actions is None or not self.plan_length or not self.config.enabled:
                counts = torch.ones((incoming.shape[0], 1), dtype=torch.long, device=device)
                return incoming, counts
            prepared = self._apply_smoothing(
                self._smoothed_actions,
                self._action_counts,
                incoming,
                self._weights.to(device),
                self._weights_cumsum.to(device),
            )
            if not torch.isfinite(prepared[0]).all():
                raise ValueError("smoothing produced non-finite actions")
            return prepared

    def commit(self, prepared):
        """Swap only previously prepared references; no tensor allocations."""
        with self._lock:
            self._smoothed_actions, self._action_counts = prepared

    def _apply_smoothing(
        self,
        old_actions: torch.Tensor,
        old_counts: torch.Tensor,
        new_actions: torch.Tensor,
        weights: torch.Tensor,
        weights_cumsum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply exponential smoothing to overlapping action regions.

        The smoothing formula:
            blended[i] = (old[i] * cumsum[count[i]-1] + new[i] * weight[count[i]])
                         / cumsum[count[i]]

        This gives exponentially weighted average where older contributions
        have more weight (when coeff > 0).
        """
        overlap_len = min(old_actions.shape[0], new_actions.shape[0])

        old_overlap = old_actions[:overlap_len]
        new_overlap = new_actions[:overlap_len]
        new_tail = new_actions[overlap_len:]

        counts_for_update = old_counts[:overlap_len]

        # Once a position has consumed chunk_size predictions, freeze it. The
        # fixed weight table cannot represent additional contributions.
        can_update = counts_for_update < self.config.chunk_size
        contribution_index = torch.clamp(counts_for_update, max=self.config.chunk_size - 1)
        old_sum = old_overlap * weights_cumsum[contribution_index - 1]
        new_term = new_overlap * weights[contribution_index]
        candidate = (old_sum + new_term) / weights_cumsum[contribution_index]
        blended = torch.where(can_update, candidate, old_overlap)

        updated_counts = torch.where(can_update, counts_for_update + 1, counts_for_update)

        smoothed_actions = torch.cat([blended, new_tail], dim=0)
        action_counts = torch.cat(
            [updated_counts, torch.ones((new_tail.shape[0], 1), dtype=torch.long, device=new_actions.device)], dim=0
        )

        return smoothed_actions, action_counts

    def get_plan(self) -> torch.Tensor | None:
        """Get the current smoothed plan without modifying it."""
        with self._lock:
            return None if self._smoothed_actions is None else self._smoothed_actions.clone()

    def peek_next_action(self) -> torch.Tensor | None:
        """Peek at the next action without removing it."""
        with self._lock:
            if self._smoothed_actions is None or self._smoothed_actions.shape[0] == 0:
                return None
            return self._smoothed_actions[0].clone()


class TemporalSmootherManager:
    """
    Manager class that handles smoothing with optional passthrough mode.

    This class provides a unified interface that can be used regardless of
    whether smoothing is enabled or not, simplifying integration.
    """

    def __init__(
        self,
        enabled: bool = True,
        chunk_size: int = 100,
        temporal_ensemble_coeff: float = 0.01,
        device: str | None = None,
    ):
        self._config = TemporalSmootherConfig(
            enabled=enabled,
            chunk_size=chunk_size,
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            device=device,
        )
        self._smoother = TemporalSmoother(self._config)

    @property
    def config(self) -> TemporalSmootherConfig:
        return self._config

    @property
    def plan_length(self) -> int:
        return self._smoother.plan_length

    @property
    def is_enabled(self) -> bool:
        return self._config.enabled

    def reset(self):
        """Reset the smoother state."""
        self._smoother.reset()

    def prepare(self, selected_actions):
        return self._smoother.prepare(selected_actions)

    def commit(self, prepared):
        self._smoother.commit(prepared)

    def update(
        self,
        new_actions: torch.Tensor | np.ndarray,
        actions_executed_during_inference: int = 0,
    ) -> int:
        """Update with new actions."""
        return self._smoother.update(new_actions, actions_executed_during_inference)

    def get_next_action(self) -> torch.Tensor | np.ndarray:
        """Get the next action."""
        return self._smoother.get_next_action()

    def get_plan(self) -> torch.Tensor | None:
        """Get the current plan."""
        return self._smoother.get_plan()

    def peek_next_action(self) -> torch.Tensor | None:
        """Peek at the next action."""
        return self._smoother.peek_next_action()

    def set_enabled(self, enabled: bool):
        """Enable or disable smoothing at runtime."""
        self._config.enabled = enabled
        self._smoother.config.enabled = enabled

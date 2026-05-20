#!/usr/bin/env python3
"""
TensorPostprocessor - Pure Python tensor denormalization.

Handles the postprocessing step of the inference pipeline:
- Denormalizes action tensors using dataset statistics
- Clamps to physical safety limits
- No ROS dependencies - pure PyTorch operations

Can be used:
1. As part of InferenceCoordinator (zero-copy mode)
2. In PostprocessorComponent (distributed mode)
3. Directly in unit tests
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor


class PostprocessorBase(ABC):
    """Abstract base for postprocessor implementations."""

    @abstractmethod
    def __call__(self, action: Any) -> Any:
        """Apply postprocessing to action."""
        pass


class LeRobotPostprocessor(PostprocessorBase):
    """
    LeRobot-specific postprocessor using make_pre_post_processors.

    Wraps the LeRobot postprocessing pipeline for tensor denormalization.
    """

    def __init__(
        self,
        policy_path: str,
        device: torch.device,
        policy_config: Any | None = None,
    ):
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device
        # See ``LeRobotPreprocessor.__init__`` for why we load a proper
        # ``PreTrainedConfig`` instance instead of a raw dict.
        self._policy_config = policy_config or self._load_policy_config(policy_path)

        _, self._postprocessor = make_pre_post_processors(
            policy_cfg=self._policy_config,
            pretrained_path=policy_path,
            postprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    def _load_policy_config(self, policy_path: str) -> Any:
        from lerobot.configs.policies import PreTrainedConfig

        return PreTrainedConfig.from_pretrained(policy_path)

    def __call__(self, action: Any) -> Any:
        return self._postprocessor(action)


class TensorPostprocessor:
    """
    Pure Python tensor postprocessor.

    Handles denormalization and safety clamping of action tensors:
    - Denormalizes using model's dataset statistics
    - Optionally clamps to physical limits

    Usage:
        postprocessor = TensorPostprocessor(
            policy_path="path/to/policy",
            device="cuda"
        )

        action = torch.randn(7)  # Raw model output
        processed = postprocessor(action)  # Denormalized
    """

    def __init__(
        self,
        policy_path: str | None = None,
        device: str | torch.device = "auto",
        postprocessor: PostprocessorBase | None = None,
        clamp_limits: dict[str, tuple[float, float]] | None = None,
    ):
        from inference_service.core.pure_inference_engine import resolve_device

        self._device = resolve_device(device) if isinstance(device, str) else device
        self._clamp_limits = clamp_limits or {}

        if postprocessor is not None:
            self._postprocessor = postprocessor
        elif policy_path is not None:
            self._postprocessor = LeRobotPostprocessor(policy_path, self._device)
        else:
            self._postprocessor = None

    def __call__(
        self,
        action: Tensor | np.ndarray,
        action_key: str = "action",
    ) -> Tensor:
        """
        Postprocess action tensor.

        Args:
            action: Raw action tensor from model
            action_key: Key for looking up clamp limits

        Returns:
            Denormalized and clamped action tensor
        """
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        action = action.to(self._device)

        if self._postprocessor is not None:
            action = self._postprocessor(action)

        if action_key in self._clamp_limits:
            low, high = self._clamp_limits[action_key]
            action = torch.clamp(action, low, high)

        return action

    def to_numpy(self, action: Tensor | np.ndarray) -> np.ndarray:
        """
        Convert action to numpy array.

        Args:
            action: Action tensor or array

        Returns:
            Numpy array on CPU
        """
        if isinstance(action, Tensor):
            return action.detach().cpu().numpy()
        return np.asarray(action)

    @property
    def device(self) -> torch.device:
        """Get the device used for postprocessing."""
        return self._device


class MockPostprocessor(PostprocessorBase):
    """Mock postprocessor for unit testing."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def __call__(self, action: Any) -> Any:
        if isinstance(action, np.ndarray):
            return torch.from_numpy(action).to(self.device)
        if isinstance(action, Tensor):
            return action.to(self.device)
        return action

#!/usr/bin/env python3
"""
TensorPreprocessor - Pure Python tensor normalization.

Handles the preprocessing step of the inference pipeline:
- Normalizes observation tensors using dataset statistics
- No ROS dependencies - pure PyTorch operations

Can be used:
1. As part of InferenceCoordinator (zero-copy mode)
2. In PreprocessorComponent (distributed mode)
3. Directly in unit tests
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor


class PreprocessorBase(ABC):
    """Abstract base for preprocessor implementations."""

    @abstractmethod
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply preprocessing to batch."""
        pass


class LeRobotPreprocessor(PreprocessorBase):
    """
    LeRobot-specific preprocessor using make_pre_post_processors.

    Wraps the LeRobot preprocessing pipeline for tensor normalization.
    """

    def __init__(
        self,
        policy_path: str,
        device: torch.device,
        policy_config: Any | None = None,
    ):
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device
        # ``make_pre_post_processors`` is typed to accept a ``PreTrainedConfig``
        # instance.  Passing a raw dict happens to work today only because the
        # ``pretrained_path`` branch consults ``policy_cfg`` solely for an
        # ``isinstance(policy_cfg, GrootConfig)`` check, but it would silently
        # mis-dispatch for any future policy-specific branch (e.g. PI05).
        # Load a proper config instance via ``PreTrainedConfig.from_pretrained``.
        self._policy_config = policy_config or self._load_policy_config(policy_path)

        self._preprocessor, _ = make_pre_post_processors(
            policy_cfg=self._policy_config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    def _load_policy_config(self, policy_path: str) -> Any:
        from lerobot.configs.policies import PreTrainedConfig

        return PreTrainedConfig.from_pretrained(policy_path)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._preprocessor(batch)


class TensorPreprocessor:
    """
    Pure Python tensor preprocessor.

    Handles conversion and normalization of observation tensors:
    - numpy arrays -> torch tensors
    - Image format conversion (HWC -> CHW)
    - Normalization using model's dataset statistics

    Usage:
        preprocessor = TensorPreprocessor(
            policy_path="path/to/policy",
            device="cuda"
        )

        obs_frame = {
            "observation.state": np.random.randn(7).astype(np.float32),
            "observation.image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        }

        batch = preprocessor(obs_frame)
    """

    def __init__(
        self,
        policy_path: str | None = None,
        device: str | torch.device = "auto",
        preprocessor: PreprocessorBase | None = None,
    ):
        from inference_service.core.pure_inference_engine import resolve_device

        self._device = resolve_device(device) if isinstance(device, str) else device

        if preprocessor is not None:
            self._preprocessor = preprocessor
        elif policy_path is not None:
            self._preprocessor = LeRobotPreprocessor(policy_path, self._device)
        else:
            self._preprocessor = None

    def __call__(
        self,
        obs_frame: dict[str, Tensor | np.ndarray],
    ) -> dict[str, Tensor]:
        """
        Preprocess observation frame into model-ready batch.

        Args:
            obs_frame: Dictionary of observations.
                       Values can be numpy arrays or tensors.

        Returns:
            Dictionary of preprocessed tensors ready for model input.
        """
        batch = self._prepare_batch(obs_frame)

        if self._preprocessor is not None:
            batch = self._preprocessor(batch)

        return batch

    def _prepare_batch(
        self,
        obs_frame: dict[str, Tensor | np.ndarray],
    ) -> dict[str, Any]:
        """
        Convert observation frame to batch format.

        Handles:
        - numpy -> torch conversion
        - Image format conversion (HWC -> CHW with batch dim)
        - Integer image normalization (0-255 -> 0-1)
        - Device placement
        """
        batch: dict[str, Any] = {}

        for key, value in obs_frame.items():
            if value is None:
                continue

            if isinstance(value, str):
                batch[key] = value
                continue

            if isinstance(value, np.ndarray):
                tensor = self._convert_numpy(value)
            elif isinstance(value, Tensor):
                tensor = self._convert_tensor(value)
            else:
                try:
                    tensor = torch.as_tensor(value, dtype=torch.float32, device=self._device)
                except (ValueError, TypeError, RuntimeError):
                    continue

            if tensor is not None:
                batch[key] = tensor

        return batch

    def _convert_numpy(self, value: np.ndarray) -> Tensor | None:
        """Convert numpy array to tensor with proper format."""
        tensor = torch.from_numpy(value)

        if tensor.ndim == 3 and tensor.shape[2] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous()

            if np.issubdtype(value.dtype, np.integer):
                max_val = float(np.iinfo(value.dtype).max)
                tensor = tensor.to(self._device, dtype=torch.float32) / max_val
            else:
                tensor = tensor.to(self._device, dtype=torch.float32)
        else:
            tensor = tensor.to(self._device, dtype=torch.float32)

        return tensor

    def _convert_tensor(self, value: Tensor) -> Tensor | None:
        """Convert tensor to proper format and device."""
        if value.ndim == 3 and value.shape[2] in (1, 3, 4):
            value = value.permute(2, 0, 1).unsqueeze(0).contiguous()

        return value.to(self._device, dtype=torch.float32)

    @property
    def device(self) -> torch.device:
        """Get the device used for preprocessing."""
        return self._device


class MockPreprocessor(PreprocessorBase):
    """Mock preprocessor for unit testing."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                result[key] = torch.from_numpy(value).to(self.device)
            elif isinstance(value, Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result

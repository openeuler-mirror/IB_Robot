#!/usr/bin/env python3
"""
PureInferenceEngine - Stateless GPU inference engine with zero ROS dependencies.

This is the core building block for model inference. It can be:
1. Used directly in Jupyter/PyTest for unit testing
2. Composed in InferenceCoordinator for zero-copy single-process deployment
3. Wrapped in PureInferenceNode for distributed cloud-edge deployment

Key design principles:
- Zero ROS dependencies (pure PyTorch + NumPy)
- Stateless: all state comes from constructor or method arguments
- Composable: can be freely combined with pre/post processors
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve device string to torch.device.

    Args:
        device: Device string. Options:
            - "auto": Auto-detect (CUDA > MPS > CPU)
            - "cuda" or "cuda:N": Specific CUDA device
            - "mps" or "metal": Apple Metal Performance Shaders
            - "cpu": Force CPU
            - "npu" or "npu:N": Ascend NPU (if available)
            - "ascend_om" or "ascend_om_3403": Ascend OM hardware backend
            - "rknn": Rockchip RKNN NPU backend (RK3588 etc.)

    Returns:
        torch.device instance

    Raises:
        RuntimeError: If requested device is not available
        ValueError: If device string is unknown
    """
    r = device.lower().strip().replace("-", "_")

    def mps_available() -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if r == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available():
            return torch.device("mps")
        try:
            import torch_npu

            if torch_npu.npu.is_available():
                return torch.device("npu:0")
        except ImportError:
            pass
        return torch.device("cpu")

    if r.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device_idx = r[4:].lstrip(":") or "0"
        return torch.device(f"cuda:{device_idx}")

    if r in ("mps", "metal"):
        if not mps_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    if r == "cpu":
        return torch.device("cpu")

    if r in ("ascend_om", "ascend_om_3403", "rknn"):
        return torch.device("cpu")

    if r.startswith("npu"):
        try:
            import torch_npu

            if torch_npu.npu.is_available():
                device_idx = r[3:].lstrip(":") or "0"
                return torch.device(f"npu:{device_idx}")
        except ImportError:
            pass
        raise RuntimeError("NPU requested but torch_npu not available")

    raise ValueError(f"Unknown device: {device}")


@dataclass
class InferenceResult:
    """
    Container for inference results.

    Attributes:
        action: Output action tensor (shape depends on policy)
        chunk_size: Number of actions in chunk (1 for single-step policies)
        latency_ms: Inference latency in milliseconds
        policy_type: Type of policy that produced this result
    """

    action: Tensor
    chunk_size: int = 1
    latency_ms: float = 0.0
    policy_type: str = ""

    def to_numpy(self) -> np.ndarray:
        """Convert action to numpy array."""
        return self.action.detach().cpu().numpy()

    @property
    def shape(self) -> tuple[int, ...]:
        """Get action shape."""
        return tuple(self.action.shape)


class PolicyWrapper(ABC):
    """
    Abstract wrapper for policy models.

    Subclasses handle the specifics of different policy frameworks
    (LeRobot, custom models, etc.)
    """

    @abstractmethod
    def load(self, path: str, device: torch.device) -> None:
        """Load policy from path."""
        pass

    @abstractmethod
    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        """Run inference on batch."""
        pass

    @abstractmethod
    def get_chunk_size(self) -> int:
        """Get action chunk size."""
        pass

    @property
    @abstractmethod
    def policy_type(self) -> str:
        """Get policy type identifier."""
        pass


class LeRobotPolicyWrapper(PolicyWrapper):
    """
    Wrapper for LeRobot policies.

    Supports all LeRobot policy types via unified interface:
    - ACT, Diffusion, TDMPC, VQBeT, Pi0, Pi0.5, SmolVLA
    """

    def __init__(self):
        self._policy: Any = None
        self._policy_type: str = ""
        self._use_action_chunking: bool = False
        self._chunk_size: int = 1
        self._device: torch.device | None = None

    def load(self, path: str, device: torch.device) -> None:
        self._device = device
        is_hf_repo = "/" in path and not os.path.exists(path)

        if is_hf_repo:
            self._load_from_hf(path, device)
        else:
            self._load_from_local(path, device)

        self._use_action_chunking = self._policy_type in ("act", "tdmpc", "vqbet")
        self._chunk_size = self._detect_chunk_size()

    def _load_from_hf(self, repo_id: str, device: torch.device) -> None:
        from lerobot.policies.factory import get_policy_class

        policy_types = ["act", "diffusion", "pi0", "pi05", "smolvla", "tdmpc", "vqbet"]

        for policy_type in policy_types:
            try:
                PolicyCls = get_policy_class(policy_type)
                self._policy = PolicyCls.from_pretrained(repo_id)
                self._policy.to(device)
                self._policy.eval()
                self._policy_type = policy_type
                break
            except Exception:
                continue

        if self._policy is None:
            raise RuntimeError(f"Could not load policy from {repo_id}")

    def _load_from_local(self, path: str, device: torch.device) -> None:
        cfg_json = os.path.join(path, "config.json")
        cfg_type = ""

        if os.path.exists(cfg_json):
            with open(cfg_json, encoding="utf-8") as f:
                cfg = json.load(f)
                cfg_type = str(cfg.get("type", "")).lower()

        if not cfg_type:
            raise RuntimeError(f"Could not determine policy type from {path}")

        from lerobot.policies.factory import get_policy_class

        PolicyCls = get_policy_class(cfg_type)
        self._policy = PolicyCls.from_pretrained(path)
        self._policy.to(device)
        self._policy.eval()
        self._policy_type = cfg_type

    def _detect_chunk_size(self) -> int:
        if hasattr(self._policy.config, "chunk_size"):
            return self._policy.config.chunk_size
        if hasattr(self._policy.config, "action_chunk_size"):
            return self._policy.config.action_chunk_size
        return 1

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        with torch.no_grad():
            if self._use_action_chunking:
                action = self._policy.predict_action_chunk(batch)
                return action.squeeze(0)
            else:
                action = self._policy.select_action(batch)
                return action[0]

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type


class PureInferenceEngine:
    """
    Pure inference engine with zero ROS dependencies.

    This is a stateless GPU inference engine that:
    - Takes preprocessed tensors as input
    - Returns raw action tensors as output
    - Has no knowledge of ROS, timestamps, or hardware

    Usage:
        engine = PureInferenceEngine(
            policy_path="path/to/policy",
            device="cuda"
        )

        batch = {
            "observation.state": torch.randn(1, 7),
            "observation.image": torch.randn(1, 3, 224, 224),
        }

        result = engine(batch)
        print(f"Action shape: {result.shape}")

    For unit testing without a real model:
        engine = PureInferenceEngine(
            policy_wrapper=MockPolicyWrapper()
        )
    """

    def __init__(
        self,
        policy_path: str | None = None,
        device: str = "auto",
        policy_wrapper: PolicyWrapper | None = None,
    ):
        self._device_name = device
        self._device = resolve_device(device)
        self._wrapper: PolicyWrapper | None = None
        self._policy_type: str = ""
        self._chunk_size: int = 1

        if policy_wrapper is not None:
            self._wrapper = policy_wrapper
            self._policy_type = policy_wrapper.policy_type
            self._chunk_size = policy_wrapper.get_chunk_size()
        elif policy_path is not None:
            self._load_policy(policy_path)
        else:
            raise ValueError("Either policy_path or policy_wrapper must be provided")

    def _load_policy(self, path: str) -> None:
        # resolve_device() maps OM backends to CPU so tensor plumbing still
        # works on development hosts; keep the original device string for
        # backend selection before that normalization loses the OM intent.
        device_name = str(self._device_name).lower().strip().replace("-", "_")
        if device_name in ("ascend_om", "ascend_om_3403"):
            from inference_service.core.ascend_om import create_ascend_om_policy_wrapper

            self._wrapper = create_ascend_om_policy_wrapper(device_name, path)
        elif device_name == "rknn":
            from inference_service.core.rknn import create_rknn_policy_wrapper

            self._wrapper = create_rknn_policy_wrapper(device_name)
        else:
            self._wrapper = LeRobotPolicyWrapper()
        self._wrapper.load(path, self._device)
        self._policy_type = self._wrapper.policy_type
        self._chunk_size = self._wrapper.get_chunk_size()

    def __call__(
        self,
        batch: dict[str, Tensor | np.ndarray],
    ) -> InferenceResult:
        """
        Run inference on a batch of observations.

        Args:
            batch: Dictionary of observation tensors.
                   Keys should match model's expected input keys.
                   Values can be Tensor or np.ndarray (will be converted).

        Returns:
            InferenceResult containing action tensor and metadata.
        """
        import time

        start_time = time.perf_counter()

        tensor_batch = self._ensure_tensors(batch)

        action = self._wrapper.infer(tensor_batch)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return InferenceResult(
            action=action,
            chunk_size=self._chunk_size,
            latency_ms=latency_ms,
            policy_type=self._policy_type,
        )

    def _ensure_tensors(
        self,
        batch: dict[str, Tensor | np.ndarray],
    ) -> dict[str, Tensor]:
        """Convert all batch values to tensors on the correct device."""
        result: dict[str, Tensor] = {}

        for key, value in batch.items():
            # Skip non-tensor metadata often injected by LeRobot dataset processors
            # (e.g. ``task`` is a str, but the batch collator wraps it into a
            # ``list[str]`` of length B — neither is convertible to a tensor).
            if value is None or isinstance(value, dict | str | bytes):
                continue
            if isinstance(value, list | tuple) and (len(value) == 0 or isinstance(value[0], str | bytes | dict)):
                continue

            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
            elif isinstance(value, Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value)

            if tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                tensor = tensor.to(dtype=torch.float32)

            result[key] = tensor.to(self._device)

        return result

    @property
    def device(self) -> torch.device:
        """Get the device used for inference."""
        return self._device

    @property
    def policy_type(self) -> str:
        """Get the policy type identifier."""
        return self._policy_type

    @property
    def chunk_size(self) -> int:
        """Get the action chunk size."""
        return self._chunk_size

    @property
    def use_action_chunking(self) -> bool:
        """Check if policy uses action chunking."""
        return self._policy_type in ("act", "tdmpc", "vqbet", "pi0", "pi05", "smolvla")


class MockPolicyWrapper(PolicyWrapper):
    """
    Mock policy wrapper for unit testing.

    Returns random actions with configurable shape.
    """

    def __init__(
        self,
        action_dim: int = 7,
        chunk_size: int = 1,
        policy_type: str = "mock",
    ):
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._policy_type = policy_type
        self._device = torch.device("cpu")

    def load(self, path: str, device: torch.device) -> None:
        self._device = device

    def infer(self, batch: dict[str, Tensor]) -> Tensor:
        if self._chunk_size > 1:
            return torch.randn(self._chunk_size, self._action_dim, device=self._device)
        return torch.randn(self._action_dim, device=self._device)

    def get_chunk_size(self) -> int:
        return self._chunk_size

    @property
    def policy_type(self) -> str:
        return self._policy_type

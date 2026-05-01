#!/usr/bin/env python3
"""
InferenceCoordinator - Zero-copy composition of inference components.

This is the "shell" that composes the three pure Python components:
- TensorPreprocessor: Observation normalization
- PureInferenceEngine: GPU inference
- TensorPostprocessor: Action denormalization

Designed for single-machine deployment where all components run in the
same process, enabling zero-copy tensor passing between stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from inference_service.core.pure_inference_engine import (
    PureInferenceEngine,
    InferenceResult,
    resolve_device,
    PolicyWrapper,
    MockPolicyWrapper,
)
from inference_service.core.preprocessor import (
    TensorPreprocessor,
    PreprocessorBase,
    MockPreprocessor,
)
from inference_service.core.postprocessor import (
    TensorPostprocessor,
    PostprocessorBase,
    MockPostprocessor,
)


@dataclass
class CoordinatorConfig:
    """
    Configuration for InferenceCoordinator.
    
    Attributes:
        policy_path: Path to the policy model
        device: Device for inference ("auto", "cuda", "cpu", etc.)
        use_preprocessing: Whether to apply preprocessing
        use_postprocessing: Whether to apply postprocessing
    """
    policy_path: str
    device: str = "auto"
    use_preprocessing: bool = True
    use_postprocessing: bool = True


@dataclass
class CoordinatorResult:
    """
    Result from coordinator inference.
    
    Attributes:
        action: Final processed action tensor
        chunk_size: Number of actions in chunk
        total_latency_ms: Total pipeline latency
        preprocess_latency_ms: Preprocessing latency
        inference_latency_ms: Model inference latency
        postprocess_latency_ms: Postprocessing latency
        policy_type: Type of policy used
    """
    action: Tensor
    chunk_size: int = 1
    total_latency_ms: float = 0.0
    preprocess_latency_ms: float = 0.0
    inference_latency_ms: float = 0.0
    postprocess_latency_ms: float = 0.0
    policy_type: str = ""
    
    def to_numpy(self) -> np.ndarray:
        """Convert action to numpy array."""
        return self.action.detach().cpu().numpy()
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get action shape."""
        return tuple(self.action.shape)


class InferenceCoordinator:
    """
    Zero-copy coordinator for inference pipeline.
    
    Composes preprocessor, engine, and postprocessor into a single
    pipeline with minimal overhead. All tensor operations happen
    in the same memory space.
    
    Usage:
        coordinator = InferenceCoordinator(
            policy_path="path/to/policy",
            device="cuda"
        )
        
        obs_frame = {
            "observation.state": np.random.randn(7).astype(np.float32),
            "observation.image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        }
        
        result = coordinator(obs_frame)
        print(f"Action: {result.action}, latency: {result.total_latency_ms:.1f}ms")
    
    For unit testing with mock components:
        coordinator = InferenceCoordinator(
            preprocessor=MockPreprocessor(),
            engine=PureInferenceEngine(policy_wrapper=MockPolicyWrapper()),
            postprocessor=MockPostprocessor(),
        )
    """
    
    def __init__(
        self,
        policy_path: Optional[str] = None,
        device: str = "auto",
        config: Optional[CoordinatorConfig] = None,
        preprocessor: Optional[PreprocessorBase] = None,
        engine: Optional[PureInferenceEngine] = None,
        postprocessor: Optional[PostprocessorBase] = None,
    ):
        if config is not None:
            policy_path = config.policy_path
            device = config.device
        
        self._device_name = device
        self._device = resolve_device(device)
        
        if engine is not None:
            self._engine = engine
        elif policy_path is not None:
            self._engine = PureInferenceEngine(policy_path=policy_path, device=self._device_name)
        else:
            raise ValueError("Either policy_path or engine must be provided")
        
        if preprocessor is not None:
            self._preprocessor = TensorPreprocessor(preprocessor=preprocessor, device=self._device)
        elif policy_path is not None:
            self._preprocessor = TensorPreprocessor(policy_path=policy_path, device=self._device)
        else:
            self._preprocessor = TensorPreprocessor(device=self._device)
        
        if postprocessor is not None:
            self._postprocessor = TensorPostprocessor(postprocessor=postprocessor, device=self._device)
        elif policy_path is not None:
            self._postprocessor = TensorPostprocessor(policy_path=policy_path, device=self._device)
        else:
            self._postprocessor = TensorPostprocessor(device=self._device)
        
        self._policy_type = self._engine.policy_type
        self._chunk_size = self._engine.chunk_size
    
    def __call__(
        self,
        obs_frame: Dict[str, Union[Tensor, np.ndarray]],
    ) -> CoordinatorResult:
        """
        Run full inference pipeline on observation frame.
        
        Args:
            obs_frame: Dictionary of observations (numpy arrays or tensors)
        
        Returns:
            CoordinatorResult with processed action and timing info
        """
        total_start = time.perf_counter()
        
        preprocess_start = time.perf_counter()
        batch = self._preprocessor(obs_frame)
        preprocess_latency = (time.perf_counter() - preprocess_start) * 1000.0
        
        inference_result = self._engine(batch)
        inference_latency = inference_result.latency_ms
        
        postprocess_start = time.perf_counter()
        action = self._postprocessor(inference_result.action)
        postprocess_latency = (time.perf_counter() - postprocess_start) * 1000.0
        
        total_latency = (time.perf_counter() - total_start) * 1000.0
        
        return CoordinatorResult(
            action=action,
            chunk_size=self._chunk_size,
            total_latency_ms=total_latency,
            preprocess_latency_ms=preprocess_latency,
            inference_latency_ms=inference_latency,
            postprocess_latency_ms=postprocess_latency,
            policy_type=self._policy_type,
        )
    
    def infer_only(
        self,
        batch: Dict[str, Tensor],
    ) -> InferenceResult:
        """
        Run inference only (skip preprocessing/postprocessing).
        
        Useful when preprocessing is done externally.
        
        Args:
            batch: Preprocessed tensor batch
        
        Returns:
            InferenceResult from engine
        """
        return self._engine(batch)
    
    def preprocess_only(
        self,
        obs_frame: Dict[str, Union[Tensor, np.ndarray]],
    ) -> Dict[str, Tensor]:
        """
        Run preprocessing only.
        
        Useful for distributed mode where preprocessing happens
        on the edge device.
        
        Args:
            obs_frame: Raw observation frame
        
        Returns:
            Preprocessed tensor batch
        """
        return self._preprocessor(obs_frame)
    
    def postprocess_only(
        self,
        action: Union[Tensor, np.ndarray],
    ) -> Tensor:
        """
        Run postprocessing only.
        
        Useful for distributed mode where postprocessing happens
        on the edge device.
        
        Args:
            action: Raw action from inference
        
        Returns:
            Postprocessed action tensor
        """
        return self._postprocessor(action)
    
    @property
    def raw_policy(self):
        """Access the underlying policy object for hook attachment."""
        return self._engine.raw_policy

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
        return self._engine.use_action_chunking

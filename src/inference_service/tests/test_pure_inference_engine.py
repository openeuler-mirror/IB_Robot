#!/usr/bin/env python3
"""
Unit tests for PureInferenceEngine - Zero ROS dependencies.

These tests verify the core inference logic can run independently:
- Device resolution
- Mock policy inference
- Tensor conversion
- InferenceResult structure

Run with: pytest tests/test_pure_inference_engine.py -v
"""

import numpy as np
import pytest
import torch
from torch import Tensor

from inference_service.core import (
    InferenceResult,
    MockPolicyWrapper,
    PureInferenceEngine,
    resolve_device,
)


class TestResolveDevice:
    """Tests for device resolution."""

    def test_auto_returns_valid_device(self):
        """Auto should return a valid torch device."""
        device = resolve_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ("cuda", "mps", "cpu")

    def test_cpu_explicit(self):
        """Explicit CPU should return CPU device."""
        device = resolve_device("cpu")
        assert device.type == "cpu"

    def test_cuda_explicit_if_available(self):
        """Explicit CUDA should work if available."""
        if torch.cuda.is_available():
            device = resolve_device("cuda")
            assert device.type == "cuda"
        else:
            with pytest.raises(RuntimeError, match="CUDA requested but not available"):
                resolve_device("cuda")

    def test_cuda_with_index(self):
        """CUDA with index should parse correctly."""
        if torch.cuda.is_available():
            device = resolve_device("cuda:0")
            assert device.type == "cuda"
            assert device.index == 0

    def test_invalid_device_raises(self):
        """Invalid device string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown device"):
            resolve_device("invalid_device")


class TestMockPolicyWrapper:
    """Tests for mock policy wrapper."""

    def test_single_action_output(self):
        """Mock should produce single action."""
        wrapper = MockPolicyWrapper(action_dim=7, chunk_size=1)
        wrapper.load("", torch.device("cpu"))

        batch = {"observation.state": torch.randn(1, 7)}
        action = wrapper.infer(batch)

        assert action.shape == (7,)
        assert action.dtype == torch.float32

    def test_chunk_action_output(self):
        """Mock should produce action chunk."""
        wrapper = MockPolicyWrapper(action_dim=7, chunk_size=16)
        wrapper.load("", torch.device("cpu"))

        batch = {"observation.state": torch.randn(1, 7)}
        action = wrapper.infer(batch)

        assert action.shape == (16, 7)

    def test_policy_type(self):
        """Mock should report correct policy type."""
        wrapper = MockPolicyWrapper(policy_type="test_policy")
        assert wrapper.policy_type == "test_policy"
        assert wrapper.backend_type == ""
        assert wrapper.uses_action_chunking is False

    def test_chunk_size(self):
        """Mock should report correct chunk size."""
        wrapper = MockPolicyWrapper(chunk_size=32)
        assert wrapper.get_chunk_size() == 32


class TestPureInferenceEngine:
    """Tests for PureInferenceEngine with mock policy."""

    @pytest.fixture
    def mock_engine(self):
        """Create engine with mock policy."""
        return PureInferenceEngine(policy_wrapper=MockPolicyWrapper(action_dim=7, chunk_size=1))

    @pytest.fixture
    def mock_chunking_engine(self):
        """Create engine with mock chunking policy."""
        return PureInferenceEngine(policy_wrapper=MockPolicyWrapper(action_dim=7, chunk_size=16))

    def test_inference_with_tensor_input(self, mock_engine):
        """Engine should accept tensor inputs."""
        batch = {
            "observation.state": torch.randn(1, 7),
        }

        result = mock_engine(batch)

        assert isinstance(result, InferenceResult)
        assert result.action.shape == (7,)
        assert result.chunk_size == 1
        assert result.policy_type == "mock"

    def test_inference_with_numpy_input(self, mock_engine):
        """Engine should accept numpy inputs."""
        batch = {
            "observation.state": np.random.randn(1, 7).astype(np.float32),
        }

        result = mock_engine(batch)

        assert isinstance(result.action, Tensor)
        assert result.action.shape == (7,)

    def test_inference_with_image(self, mock_engine):
        """Engine should handle image tensors."""
        batch = {
            "observation.state": torch.randn(1, 7),
            "observation.image": torch.randn(1, 3, 224, 224),
        }

        result = mock_engine(batch)

        assert result.action.shape == (7,)

    def test_chunking_inference(self, mock_chunking_engine):
        """Chunking policy should return action chunk."""
        batch = {
            "observation.state": torch.randn(1, 7),
        }

        result = mock_chunking_engine(batch)

        assert result.action.shape == (16, 7)
        assert result.chunk_size == 16

    def test_latency_measurement(self, mock_engine):
        """Engine should measure latency."""
        batch = {"observation.state": torch.randn(1, 7)}

        result = mock_engine(batch)

        assert result.latency_ms >= 0

    def test_device_property(self, mock_engine):
        """Engine should expose device property."""
        assert mock_engine.device.type in ("cpu", "cuda", "mps")

    def test_policy_type_property(self, mock_engine):
        """Engine should expose policy type."""
        assert mock_engine.policy_type == "mock"

    def test_chunk_size_property(self, mock_engine):
        """Engine should expose chunk size."""
        assert mock_engine.chunk_size == 1

    def test_use_action_chunking_property(self, mock_engine, mock_chunking_engine):
        """Engine should report chunking status."""
        assert mock_engine.use_action_chunking is False
        assert mock_chunking_engine.use_action_chunking is False  # mock is not act/tdmpc/vqbet

    def test_result_to_numpy(self, mock_engine):
        """InferenceResult should convert to numpy."""
        batch = {"observation.state": torch.randn(1, 7)}

        result = mock_engine(batch)
        action_np = result.to_numpy()

        assert isinstance(action_np, np.ndarray)
        assert action_np.shape == (7,)


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_default_values(self):
        """Result should have sensible defaults."""
        action = torch.randn(7)
        result = InferenceResult(action=action)

        assert result.chunk_size == 1
        assert result.latency_ms == 0.0
        assert result.policy_type == ""
        assert result.backend_type == ""

    def test_shape_property(self):
        """Result should expose shape."""
        action = torch.randn(16, 7)
        result = InferenceResult(action=action, chunk_size=16)

        assert result.shape == (16, 7)

    def test_to_numpy(self):
        """Result should convert to numpy."""
        action = torch.randn(7)
        result = InferenceResult(action=action)

        action_np = result.to_numpy()

        assert isinstance(action_np, np.ndarray)
        np.testing.assert_array_almost_equal(action_np, action.numpy())


class TestTensorConversion:
    """Tests for internal tensor conversion."""

    def test_integer_image_normalization(self):
        """Integer images should be normalized to 0-1."""
        wrapper = MockPolicyWrapper()
        engine = PureInferenceEngine(policy_wrapper=wrapper)

        batch = {"observation.image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)}

        result = engine(batch)

        assert result.action is not None

    def test_float_image_passthrough(self):
        """Float images should pass through."""
        wrapper = MockPolicyWrapper()
        engine = PureInferenceEngine(policy_wrapper=wrapper)

        batch = {"observation.image": np.random.randn(224, 224, 3).astype(np.float32)}

        result = engine(batch)

        assert result.action is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

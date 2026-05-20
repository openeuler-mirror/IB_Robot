#!/usr/bin/env python3
"""
Unit tests for InferenceCoordinator - Zero ROS dependencies.

These tests verify the coordinator can compose all components:
- Preprocessing
- Inference
- Postprocessing
- Timing measurements

Run with: pytest tests/test_inference_coordinator.py -v
"""

import numpy as np
import pytest
import torch
from torch import Tensor

from inference_service.core import (
    CoordinatorConfig,
    CoordinatorResult,
    InferenceCoordinator,
    InferenceResult,
    MockPolicyWrapper,
    MockPostprocessor,
    MockPreprocessor,
    PureInferenceEngine,
)


class TestCoordinatorConfig:
    """Tests for CoordinatorConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = CoordinatorConfig(policy_path="/some/path")

        assert config.device == "auto"
        assert config.use_preprocessing is True
        assert config.use_postprocessing is True


class TestCoordinatorResult:
    """Tests for CoordinatorResult."""

    def test_default_values(self):
        """Result should have sensible defaults."""
        action = torch.randn(7)
        result = CoordinatorResult(action=action)

        assert result.chunk_size == 1
        assert result.total_latency_ms == 0.0
        assert result.preprocess_latency_ms == 0.0
        assert result.inference_latency_ms == 0.0
        assert result.postprocess_latency_ms == 0.0

    def test_to_numpy(self):
        """Result should convert to numpy."""
        action = torch.randn(7)
        result = CoordinatorResult(action=action)

        action_np = result.to_numpy()

        assert isinstance(action_np, np.ndarray)
        np.testing.assert_array_almost_equal(action_np, action.numpy())

    def test_shape_property(self):
        """Result should expose shape."""
        action = torch.randn(16, 7)
        result = CoordinatorResult(action=action, chunk_size=16)

        assert result.shape == (16, 7)


class TestInferenceCoordinator:
    """Tests for InferenceCoordinator with mock components."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create coordinator with all mock components."""
        device = torch.device("cpu")

        return InferenceCoordinator(
            preprocessor=MockPreprocessor(device=device),
            engine=PureInferenceEngine(policy_wrapper=MockPolicyWrapper(action_dim=7, chunk_size=1)),
            postprocessor=MockPostprocessor(device=device),
        )

    @pytest.fixture
    def mock_chunking_coordinator(self):
        """Create coordinator with chunking mock policy."""
        device = torch.device("cpu")

        return InferenceCoordinator(
            preprocessor=MockPreprocessor(device=device),
            engine=PureInferenceEngine(policy_wrapper=MockPolicyWrapper(action_dim=7, chunk_size=16)),
            postprocessor=MockPostprocessor(device=device),
        )

    def test_coordinator_with_numpy_input(self, mock_coordinator):
        """Coordinator should accept numpy inputs."""
        obs_frame = {
            "observation.state": np.random.randn(7).astype(np.float32),
        }

        result = mock_coordinator(obs_frame)

        assert isinstance(result, CoordinatorResult)
        assert result.action.shape == (7,)

    def test_coordinator_with_tensor_input(self, mock_coordinator):
        """Coordinator should accept tensor inputs."""
        obs_frame = {
            "observation.state": torch.randn(7),
        }

        result = mock_coordinator(obs_frame)

        assert isinstance(result.action, Tensor)
        assert result.action.shape == (7,)

    def test_coordinator_with_image(self, mock_coordinator):
        """Coordinator should handle image observations."""
        obs_frame = {
            "observation.state": torch.randn(7),
            "observation.image": torch.randn(3, 224, 224),
        }

        result = mock_coordinator(obs_frame)

        assert result.action.shape == (7,)

    def test_chunking_coordinator(self, mock_chunking_coordinator):
        """Chunking policy should return action chunk."""
        obs_frame = {
            "observation.state": torch.randn(7),
        }

        result = mock_chunking_coordinator(obs_frame)

        assert result.action.shape == (16, 7)
        assert result.chunk_size == 16

    def test_timing_measurements(self, mock_coordinator):
        """Coordinator should measure all latencies."""
        obs_frame = {
            "observation.state": torch.randn(7),
        }

        result = mock_coordinator(obs_frame)

        assert result.total_latency_ms >= 0
        assert result.preprocess_latency_ms >= 0
        assert result.inference_latency_ms >= 0
        assert result.postprocess_latency_ms >= 0

        assert result.total_latency_ms >= result.inference_latency_ms

    def test_device_property(self, mock_coordinator):
        """Coordinator should expose device property."""
        assert mock_coordinator.device.type in ("cpu", "cuda", "mps")

    def test_policy_type_property(self, mock_coordinator):
        """Coordinator should expose policy type."""
        assert mock_coordinator.policy_type == "mock"

    def test_chunk_size_property(self, mock_coordinator):
        """Coordinator should expose chunk size."""
        assert mock_coordinator.chunk_size == 1

    def test_infer_only(self, mock_coordinator):
        """Coordinator should support inference-only mode."""
        batch = {
            "observation.state": torch.randn(1, 7),
        }

        result = mock_coordinator.infer_only(batch)

        assert isinstance(result.action, Tensor)

    def test_preprocess_only(self, mock_coordinator):
        """Coordinator should support preprocessing-only mode."""
        obs_frame = {
            "observation.state": np.random.randn(7).astype(np.float32),
        }

        batch = mock_coordinator.preprocess_only(obs_frame)

        assert "observation.state" in batch
        assert isinstance(batch["observation.state"], Tensor)

    def test_postprocess_only(self, mock_coordinator):
        """Coordinator should support postprocessing-only mode."""
        action = torch.randn(7)

        result = mock_coordinator.postprocess_only(action)

        assert isinstance(result, Tensor)

    def test_dynamic_engine_chunk_size_is_synchronized(self):
        """Coordinator should refresh chunk size after inference."""

        class DynamicChunkEngine:
            def __init__(self):
                self.policy_type = "act"
                self.backend_type = "ascend_om_3403"
                self.chunk_size = 1
                self.device = torch.device("cpu")
                self.use_action_chunking = True

            def __call__(self, batch):
                del batch
                self.chunk_size = 3
                return InferenceResult(
                    action=torch.zeros(3, 2),
                    chunk_size=3,
                    policy_type=self.policy_type,
                    backend_type=self.backend_type,
                )

        device = torch.device("cpu")
        coordinator = InferenceCoordinator(
            preprocessor=MockPreprocessor(device=device),
            engine=DynamicChunkEngine(),
            postprocessor=MockPostprocessor(device=device),
        )

        result = coordinator({"observation.state": torch.randn(2)})

        assert result.chunk_size == 3
        assert coordinator.chunk_size == 3


class TestCoordinatorWithConfig:
    """Tests for coordinator with config object."""

    def test_config_initialization(self):
        """Coordinator should accept config object."""
        config = CoordinatorConfig(
            policy_path="/dummy/path",
            device="cpu",
        )

        with pytest.raises(RuntimeError):
            InferenceCoordinator(config=config)

    def test_config_preserves_hardware_backend_device(self, monkeypatch, tmp_path):
        """Coordinator should pass hardware backend names to PureInferenceEngine."""
        model_path = tmp_path / "model.om"
        model_path.write_bytes(b"om")
        (tmp_path / "config.json").write_text(
            '{"type": "act", "chunk_size": 1, '
            '"input_features": {"observation.state": {"shape": [7]}}, '
            '"output_features": {"action": {"shape": [6]}}}',
            encoding="utf-8",
        )
        (tmp_path / "config.om.json").write_text(
            '{"schema_version": 1, "policy_type": "act", "backend": "ascend_om", '
            '"artifacts": {"policy": "model.om"}, "execution": ["policy"]}',
            encoding="utf-8",
        )

        class FakeRuntimeSession:
            def load(self, policy_path, config, device):
                pass

            def execute(self, inputs):
                return [np.arange(6, dtype=np.float32)]

            def release(self):
                pass

        monkeypatch.setattr(
            "inference_service.core.compiled_policy.create_runtime_session",
            lambda backend, config=None: FakeRuntimeSession(),
        )

        coordinator = InferenceCoordinator(
            policy_path=str(tmp_path),
            device="ascend_om",
            preprocessor=MockPreprocessor(device=torch.device("cpu")),
            postprocessor=MockPostprocessor(device=torch.device("cpu")),
        )

        assert coordinator.device.type == "cpu"
        assert coordinator.policy_type == "act"
        assert coordinator.backend_type == "ascend_om"


class TestEndToEndPipeline:
    """End-to-end tests for the full pipeline."""

    def test_full_pipeline_shape_preservation(self):
        """Action dimension should be preserved through pipeline."""
        device = torch.device("cpu")
        action_dim = 14

        coordinator = InferenceCoordinator(
            preprocessor=MockPreprocessor(device=device),
            engine=PureInferenceEngine(policy_wrapper=MockPolicyWrapper(action_dim=action_dim, chunk_size=1)),
            postprocessor=MockPostprocessor(device=device),
        )

        obs_frame = {
            "observation.state": torch.randn(action_dim),
        }

        result = coordinator(obs_frame)

        assert result.action.shape == (action_dim,)

    def test_multiple_observations(self):
        """Coordinator should handle multiple observation types."""
        device = torch.device("cpu")

        coordinator = InferenceCoordinator(
            preprocessor=MockPreprocessor(device=device),
            engine=PureInferenceEngine(policy_wrapper=MockPolicyWrapper(action_dim=7, chunk_size=1)),
            postprocessor=MockPostprocessor(device=device),
        )

        obs_frame = {
            "observation.state": torch.randn(7),
            "observation.image": torch.randn(3, 224, 224),
            "observation.image_secondary": torch.randn(3, 128, 128),
        }

        result = coordinator(obs_frame)

        assert result.action.shape == (7,)

    def test_empty_observation_handling(self):
        """Coordinator should handle empty observations gracefully."""
        device = torch.device("cpu")

        coordinator = InferenceCoordinator(
            preprocessor=MockPreprocessor(device=device),
            engine=PureInferenceEngine(policy_wrapper=MockPolicyWrapper(action_dim=7, chunk_size=1)),
            postprocessor=MockPostprocessor(device=device),
        )

        obs_frame = {}

        result = coordinator(obs_frame)

        assert result.action.shape == (7,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

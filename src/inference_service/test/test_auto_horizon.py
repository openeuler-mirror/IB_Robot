from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from inference_service.auto_horizon import estimate_execution_horizon
from inference_service.auto_horizon_runtime import (
    ActionSelfAttentionCollector,
    build_action_attention_collector,
)


def _diagonal_attention(size: int) -> np.ndarray:
    return np.eye(size, dtype=np.float64)


def test_estimator_accepts_leading_layer_head_and_batch_dimensions():
    attention = np.stack([np.stack([_diagonal_attention(4), _diagonal_attention(4)])])
    estimate = estimate_execution_horizon(attention)
    assert estimate.horizon == 4
    assert estimate.forward_horizon == 1


def test_estimator_returns_full_horizon_when_pointers_cover_chunk():
    size = 8
    attention = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        attention[row, row] = 0.8
        attention[row, min(row + 1, size - 1)] += 0.2
    estimate = estimate_execution_horizon(attention, hold_threshold=0.3)
    assert estimate.horizon == size
    assert estimate.join_row is not None


@pytest.mark.parametrize(
    "attention",
    [np.ones((3, 4)), np.zeros((3, 3)), np.full((3, 3), -1.0)],
)
def test_estimator_rejects_invalid_attention(attention):
    with pytest.raises(ValueError):
        estimate_execution_horizon(attention)


def test_attention_collector_averages_layers_and_exposes_horizon():
    class FakeAttention(nn.Module):
        def forward(self, values):
            return values, torch.eye(4).reshape(1, 1, 4, 4)

    class FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.gemma_expert = nn.Module()
            self.gemma_expert.model = nn.Module()
            self.gemma_expert.model.layers = nn.ModuleList([nn.Module()])
            self.gemma_expert.model.layers[0].self_attn = FakeAttention()

        def forward(self, values):
            for _ in range(3):
                for layer in self.gemma_expert.model.layers:
                    values, _ = layer.self_attn(values)
            return values

    collector = ActionSelfAttentionCollector(
        hold_threshold=0.3,
        entropy_quantile=0.9,
        run_length=1,
        prediction_horizon=4,
        sampling_step=1,
    )
    policy = FakePolicy()
    assert collector.install(policy)
    collector.begin()
    policy(torch.zeros(1, 4))
    assert collector.estimate()["execution_horizon"] == 4
    collector.close()
    assert not collector._handles


def test_build_collector_fails_closed_without_action_expert_modules():
    class NoExpertPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(chunk_size=4, num_inference_steps=10)
            self.encoder = nn.Module()
            self.encoder.self_attn = nn.Identity()  # not under gemma_expert/action_expert

    with pytest.raises(ValueError, match="no action-expert self_attn modules"):
        build_action_attention_collector(
            NoExpertPolicy(),
            {"auto_horizon_enabled": True, "auto_horizon_sampling_step": 1},
        )


def test_build_collector_rejects_sampling_step_beyond_num_inference_steps():
    class FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(chunk_size=4, num_inference_steps=10)

    with pytest.raises(ValueError, match="exceeds the policy's num_inference_steps"):
        build_action_attention_collector(
            FakePolicy(),
            {"auto_horizon_enabled": True, "auto_horizon_sampling_step": 11},
        )


def test_build_collector_returns_none_when_disabled():
    class FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(chunk_size=4, num_inference_steps=10)

    # Disabled: no collector, no validation, no module scan.
    assert build_action_attention_collector(FakePolicy(), {}) is None

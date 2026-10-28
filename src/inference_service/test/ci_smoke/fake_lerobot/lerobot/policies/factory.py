from __future__ import annotations

from types import SimpleNamespace

import torch


class _FakePolicy:
    supports_attention = False

    def __init__(self, config) -> None:
        self.config = config
        self.model = SimpleNamespace(supports_attention=False)

    @classmethod
    def from_pretrained(cls, _path: str, *, config, **_kwargs):
        return cls(config)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def predict_action_chunk(self, _batch):
        return torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

    def reset(self) -> None:
        return None


def get_policy_class(policy_type: str):
    if policy_type != "act":
        raise ValueError(f"unsupported fake policy type {policy_type!r}")
    return _FakePolicy


def make_pre_post_processors(**_kwargs):
    return _identity, _identity


def _identity(value):
    return value

import sys
from types import ModuleType

import pytest

import perception_service.grounded_sam2_wrapper as wrapper_module
from perception_service.grounding_dino_config import GROUNDING_DINO_SWINT_OGC_CONFIG


def test_grounding_dino_swint_ogc_config_matches_checkpoint_architecture() -> None:
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["backbone"] == "swin_T_224_1k"
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["enc_layers"] == 6
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["dec_layers"] == 6
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["hidden_dim"] == 256
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["num_queries"] == 900
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["text_encoder_type"] == "bert-base-uncased"


class _FakeIncompatibleKeys:
    """Minimal stand-in for torch.nn.modules._IncompatibleKeys."""

    def __init__(self, missing_keys=None, unexpected_keys=None):
        self.missing_keys = list(missing_keys or [])
        self.unexpected_keys = list(unexpected_keys or [])


def test_grounding_model_builds_from_a_copy_of_source_config(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakeSLConfig:
        def __init__(self, values):
            captured["values"] = values
            self.__dict__.update(values)

    class FakeModel:
        def load_state_dict(self, state, strict):
            captured["state"] = state
            captured["strict"] = strict
            return _FakeIncompatibleKeys()

        def eval(self):
            captured["evaluated"] = True

    groundingdino = ModuleType("groundingdino")
    models = ModuleType("groundingdino.models")
    util = ModuleType("groundingdino.util")
    slconfig = ModuleType("groundingdino.util.slconfig")
    utils = ModuleType("groundingdino.util.utils")
    fake_model = FakeModel()

    def build_model(args):
        captured["args"] = args
        return fake_model

    models.build_model = build_model
    slconfig.SLConfig = FakeSLConfig
    utils.clean_state_dict = lambda state: state
    monkeypatch.setitem(sys.modules, "groundingdino", groundingdino)
    monkeypatch.setitem(sys.modules, "groundingdino.models", models)
    monkeypatch.setitem(sys.modules, "groundingdino.util", util)
    monkeypatch.setitem(sys.modules, "groundingdino.util.slconfig", slconfig)
    monkeypatch.setitem(sys.modules, "groundingdino.util.utils", utils)
    monkeypatch.setattr(wrapper_module.torch, "load", lambda *_args, **_kwargs: {"model": {}})

    model = wrapper_module.GroundedSAM2Wrapper._load_grounding_model(
        tmp_path / "groundingdino.pth", "/models/bert-base-uncased", "cpu"
    )

    assert model is fake_model
    assert captured["values"] is not GROUNDING_DINO_SWINT_OGC_CONFIG
    assert captured["args"].device == "cpu"
    assert captured["args"].text_encoder_type == "/models/bert-base-uncased"
    assert GROUNDING_DINO_SWINT_OGC_CONFIG["text_encoder_type"] == "bert-base-uncased"
    assert captured["strict"] is False
    assert captured["evaluated"] is True


def test_verify_grounding_dino_checkpoint_compatible_passes_official_checkpoint_keys() -> None:
    """Keys measured from the official groundingdino_swint_ogc.pth loaded via
    load_state_dict(strict=False) with the audited wheel."""
    incompatible = _FakeIncompatibleKeys(
        missing_keys=[],
        unexpected_keys=["label_enc.weight", "bert.embeddings.position_ids"],
    )
    wrapper_module._verify_grounding_dino_checkpoint_compatible(incompatible)


def test_verify_grounding_dino_checkpoint_compatible_rejects_unaudited_missing() -> None:
    incompatible = _FakeIncompatibleKeys(
        missing_keys=["bert.pooler.dense.weight"],
    )
    with pytest.raises(RuntimeError, match="missing="):
        wrapper_module._verify_grounding_dino_checkpoint_compatible(incompatible)


def test_verify_grounding_dino_checkpoint_compatible_rejects_unaudited_unexpected() -> None:
    incompatible = _FakeIncompatibleKeys(
        unexpected_keys=["backbone.0.body.stem.conv.weight"],
    )
    with pytest.raises(RuntimeError, match="unexpected="):
        wrapper_module._verify_grounding_dino_checkpoint_compatible(incompatible)


def test_verify_grounding_dino_checkpoint_compatible_handles_none_return() -> None:
    wrapper_module._verify_grounding_dino_checkpoint_compatible(None)


def _has_groundingdino_wheel() -> bool:
    import importlib.util

    return importlib.util.find_spec("groundingdino") is not None


@pytest.mark.skipif(
    wrapper_module.torch is None or not _has_groundingdino_wheel(),
    reason="requires torch and the audited ibrobot-groundingdino wheel",
)
def test_bert_model_warper_does_not_register_full_bert_model() -> None:
    """BertModelWarper must not register the full BertModel as a sub-module.

    The Grounding-DINO checkpoint contract only carries ``bert.embeddings.*``,
    ``bert.encoder.*``, ``bert.pooler.*`` keys. If the patched BertModelWarper
    stored the underlying BertModel via ``self._bert_model = bert_model`` (any
    normal nn.Module attribute assignment), the same parameters would also
    appear under ``_bert_model.*`` in state_dict, breaking the checkpoint ABI
    and producing duplicated weight paths on save. The reference must stay
    outside nn.Module sub-module registration.
    """
    import torch
    import torch.nn as nn
    from groundingdino.models.GroundingDINO.bertwarper import BertModelWarper

    class FakeBertModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Linear(8, 8)
            self.encoder = nn.Linear(8, 8)
            self.pooler = nn.Linear(8, 8)
            self.config = object()
            self.dtype = torch.float32

        def get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
            return attention_mask

        def invert_attention_mask(self, encoder_attention_mask):
            # BertModelWarper.__init__ copies this across alongside embeddings,
            # encoder, pooler and config; a stand-in missing it fails at
            # construction before the state_dict leak can be checked.
            return encoder_attention_mask

    bert = FakeBertModel()
    warper = BertModelWarper(bert)
    state_dict_keys = set(warper.state_dict().keys())
    leaked = {key for key in state_dict_keys if key.startswith("_bert_model")}
    assert not leaked, f"BertModelWarper leaked internal BertModel reference into state_dict: {leaked}"


def test_cuda_autocast_uses_fp16_for_grounding_dino_extension(monkeypatch):
    calls = []

    class FakeAutocast:
        def __enter__(self):
            return self

    monkeypatch.setattr(
        wrapper_module.torch,
        "autocast",
        lambda **kwargs: calls.append(kwargs) or FakeAutocast(),
    )

    wrapper_module._enable_grounding_dino_autocast()

    assert calls == [{"device_type": "cuda", "dtype": wrapper_module.torch.float16}]

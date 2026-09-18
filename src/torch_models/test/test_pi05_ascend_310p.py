#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from torch_models.pi05_ascend_310p.modeling_pi05_ascend_310p import (
    PaliGemmaWithExpertModel,
    PI05Ascend310PPolicy,
    PI05Pytorch,
    _cann_build_identity,
    _cann_version,
    _log_torchair_cache_status,
    _pi05_no_init_weights,
    _torchair_cache_dir,
    configure_pi05_ascend_310p_config,
)


def test_cann_version_reads_arch_specific_install_metadata(monkeypatch, tmp_path):
    toolkit = tmp_path / "latest"
    metadata = toolkit / "aarch64-linux" / "ascend_toolkit_install.info"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("version=8.1.RC1\n", encoding="utf-8")
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)

    assert _cann_version() == "8.1.RC1"

    first_identity = _cann_build_identity()
    metadata.write_text("version=8.1.RC1\ninnerversion=8.1.RC1.B238\n", encoding="utf-8")

    assert _cann_version() == "8.1.RC1"
    assert _cann_build_identity() != first_identity


def test_prefix_graph_defaults_to_compiled_vision(monkeypatch):
    monkeypatch.delenv("LEROBOT_PI05_COMPILE_VISION_EMBED", raising=False)
    model = PI05Pytorch.__new__(PI05Pytorch)
    model.parameters = lambda: iter((SimpleNamespace(device=SimpleNamespace(type="npu")),))

    assert model._prefix_graph_mode() == "compiled_vision_split"


def test_config_rejects_non_ten_step_denoise() -> None:
    config = SimpleNamespace(num_inference_steps=5, dtype="float32")

    with pytest.raises(ValueError, match="requires num_inference_steps=10"):
        configure_pi05_ascend_310p_config(config)


def test_torchair_cache_separates_npu_graph_variants(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IBROBOT_TORCHAIR_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path / "missing-cann"))
    module = SimpleNamespace(__version__="2.5.1")
    common = {
        "deployment_fingerprint": "fingerprint",
        "torch_module": module,
        "torch_npu_module": module,
        "device_name": "Ascend310P1",
        "model_dtype": "float16",
        "prefix_graph_mode": "compiled_vision_split",
        "npu_fused_ops": True,
    }

    fused = _torchair_cache_dir(**common, vision_npu_fused_ops=True)
    unfused = _torchair_cache_dir(**common, vision_npu_fused_ops=False)

    assert fused != unfused
    assert "Ascend310P1" in fused.parts


@pytest.mark.parametrize(
    ("prefix_mode", "expected_graph_path", "expected_targets", "vision_compiled"),
    [
        (
            "full_prefix",
            "compiled_full_prefix_denoise",
            ["_action_prefix_forward_for_compile", "_action_denoise_10_steps_for_compile"],
            True,
        ),
        (
            "compiled_vision_split",
            "compiled_vision_compiled_prefill_denoise",
            ["embed_image", "_action_prefix_prefill_for_compile", "_action_denoise_10_steps_for_compile"],
            True,
        ),
        (
            "eager_vision_split",
            "eager_vision_compiled_prefill_denoise",
            ["_action_prefix_prefill_for_compile", "_action_denoise_10_steps_for_compile"],
            False,
        ),
    ],
)
def test_graph_compile_selects_platform_prefix_boundary(
    monkeypatch,
    prefix_mode,
    expected_graph_path,
    expected_targets,
    vision_compiled,
):
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.register_parameter("device_anchor", nn.Parameter(torch.zeros(())))
    model.config = SimpleNamespace(
        compile_inference_fullgraph=True,
        compile_inference_dynamic=False,
        compile_frozen_parameter=True,
        compile_tiling_schedule_optimize=True,
    )
    model.paligemma_with_expert = SimpleNamespace(embed_image=lambda images: images)
    model._action_prefix_forward_for_compile = lambda *args: args
    model._action_prefix_prefill_for_compile = lambda *args: args
    model._action_denoise_10_steps_for_compile = lambda *args: args
    model._force_eager_attention_for_graph_compile = MethodType(lambda self: None, model)
    model._prefix_graph_mode = MethodType(lambda self: prefix_mode, model)
    model._build_inference_compile_kwargs = MethodType(
        lambda self, **kwargs: ({"fullgraph": True, "dynamic": False}, "test"),
        model,
    )

    compiled_targets = []

    def fake_compile(target, **kwargs):
        compiled_targets.append(target)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)

    info = model.enable_sample_actions_graph_compile()

    expected_callables = {
        "embed_image": model.paligemma_with_expert.embed_image,
        "_action_prefix_forward_for_compile": model._action_prefix_forward_for_compile,
        "_action_prefix_prefill_for_compile": model._action_prefix_prefill_for_compile,
        "_action_denoise_10_steps_for_compile": model._action_denoise_10_steps_for_compile,
    }
    assert compiled_targets == [expected_callables[target] for target in expected_targets]
    assert info["graph_path"] == expected_graph_path
    assert info["prefix_vision_compiled"] is vision_compiled
    assert info["targets"] == expected_targets


def test_graph_compile_uses_persistent_torchair_cache(monkeypatch, tmp_path):
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.register_parameter("device_anchor", nn.Parameter(torch.zeros(())))
    model.config = SimpleNamespace(
        compile_inference_fullgraph=True,
        compile_inference_dynamic=False,
        compile_frozen_parameter=True,
        compile_tiling_schedule_optimize=True,
    )
    model.paligemma_with_expert = SimpleNamespace(embed_image=lambda images: images)
    model._action_prefix_forward_for_compile = lambda *args: args
    model._action_prefix_prefill_for_compile = lambda *args: args
    model._action_denoise_10_steps_for_compile = lambda *args: args
    model._force_eager_attention_for_graph_compile = MethodType(lambda self: None, model)
    model._prefix_graph_mode = MethodType(lambda self: "full_prefix", model)
    model._build_inference_compile_kwargs = MethodType(
        lambda self, **kwargs: ({"fullgraph": True, "dynamic": False}, "torchair"),
        model,
    )
    compiler_config = object()
    model._build_torchair_compiler_config = MethodType(lambda self, **kwargs: compiler_config, model)
    cached_targets = []

    def fake_cached(self, target, **kwargs):
        cached_targets.append((target, kwargs))
        return target

    model._compile_torchair_cached = MethodType(fake_cached, model)

    info = model.enable_sample_actions_graph_compile(torchair_cache_dir=tmp_path)

    assert [target for target, _kwargs in cached_targets] == [
        model._action_prefix_forward_for_compile,
        model._action_denoise_10_steps_for_compile,
    ]
    assert all(kwargs["compiler_config"] is compiler_config for _target, kwargs in cached_targets)
    assert all(kwargs["cache_root"] == tmp_path for _target, kwargs in cached_targets)
    assert info["torchair_cache_enabled"] is True
    assert info["torchair_cache_dir"] == str(tmp_path.resolve())


def test_pi05_no_init_context_restores_initializers():
    original = torch.nn.init.uniform_
    tensor = torch.full((2,), 3.0)

    with _pi05_no_init_weights():
        torch.nn.init.uniform_(tensor)
        torch.testing.assert_close(tensor, torch.full((2,), 3.0))

    assert torch.nn.init.uniform_ is original


def test_torchair_cache_log_reports_status_path_size_and_cleanup(capsys, tmp_path):
    cache_bin = tmp_path / "prefix" / "compiled_module"
    cache_bin.parent.mkdir()
    cache_bin.write_bytes(b"cache")

    _log_torchair_cache_status("HIT", tmp_path, cache_bin)

    message = capsys.readouterr().err
    assert "status=HIT" in message
    assert f"path={tmp_path}" in message
    assert f"cache_file={cache_bin}" in message
    assert "total_size=5.00 B" in message
    assert f"rm -rf -- {tmp_path}" in message


def test_pi05_from_pretrained_no_init_assigns_checkpoint_tensors(monkeypatch, tmp_path):
    import safetensors.torch

    checkpoint_tensor = torch.tensor([[7.0]])
    calls = {}

    def fake_init(self, config, **kwargs):
        del kwargs
        nn.Module.__init__(self)
        calls["construction_device"] = str(config.device)
        self.config = config
        self.model = nn.Linear(1, 1, bias=False)

    original_load_state_dict = PI05Ascend310PPolicy.load_state_dict

    def record_load_state_dict(self, state_dict, **kwargs):
        calls["assign"] = kwargs.get("assign")
        return original_load_state_dict(self, state_dict, **kwargs)

    monkeypatch.setattr(PI05Ascend310PPolicy, "__init__", fake_init)
    monkeypatch.setattr(PI05Ascend310PPolicy, "load_state_dict", record_load_state_dict)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *_args, **_kwargs: {"weight": checkpoint_tensor})
    (tmp_path / "model.safetensors").touch()
    config = SimpleNamespace(device="npu")

    policy = PI05Ascend310PPolicy.from_pretrained(tmp_path, config=config, skip_weight_init=True)

    assert calls == {"construction_device": "cpu", "assign": True}
    assert config.device == "npu"
    assert policy.model.weight.data_ptr() == checkpoint_tensor.data_ptr()
    torch.testing.assert_close(policy.model.weight, checkpoint_tensor)


def test_pi05_from_pretrained_no_init_fails_closed(monkeypatch, tmp_path):
    import safetensors.torch

    def fake_init(self, config, **kwargs):
        del kwargs
        nn.Module.__init__(self)
        self.config = config
        self.model = nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(PI05Ascend310PPolicy, "__init__", fake_init)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad")))
    (tmp_path / "model.safetensors").touch()

    with pytest.raises(RuntimeError, match="Could not load PI05 checkpoint"):
        PI05Ascend310PPolicy.from_pretrained(tmp_path, config=SimpleNamespace(device="npu"), skip_weight_init=True)


def test_prefix_mask_uses_pad_outer_product_without_broadcast_compare():
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    pad_masks = torch.tensor([[True, True, False]])
    block_masks = torch.zeros_like(pad_masks)

    actual, positions = model._action_prefix_masks_for_compile(pad_masks, block_masks)

    expected = pad_masks.unsqueeze(1) & pad_masks.unsqueeze(2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(positions, torch.tensor([[0, 1, 1]]))


def test_action_compute_dtype_supports_float16():
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.register_parameter("device_anchor", nn.Parameter(torch.zeros(())))
    model.config = SimpleNamespace(dtype="float16")

    assert model._action_compute_dtype() is torch.float16


def test_npu_qkv_projection_does_not_require_fused_weights():
    model = PaliGemmaWithExpertModel.__new__(PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model._qkv_weights_fused = False

    attn = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=2, num_key_value_heads=1),
        head_dim=4,
        q_proj=nn.Linear(8, 8, bias=False),
        k_proj=nn.Linear(8, 4, bias=False),
        v_proj=nn.Linear(8, 4, bias=False),
    )
    hidden_states = torch.randn(1, 3, 8)

    expected = (attn.q_proj(hidden_states), attn.k_proj(hidden_states), attn.v_proj(hidden_states))
    actual = model._project_qkv(attn, hidden_states)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_compiled_denoise_forward_does_not_mutate_transformers_config():
    source = PI05Pytorch._run_action_denoise_forward.__code__

    assert "_attn_implementation" not in source.co_names


def test_prefix_preserves_v051_language_embedding_scale():
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.paligemma_with_expert = SimpleNamespace(
        embed_image=lambda images: torch.ones(images.shape[0], 2, 4),
        embed_language_tokens=lambda tokens: torch.ones(tokens.shape[0], tokens.shape[1], 4),
    )
    model._apply_checkpoint = MethodType(lambda self, function, *args: function(*args), model)

    embeddings, _, _ = model.embed_prefix(
        images=[torch.zeros(1, 3, 2, 2)],
        img_masks=[torch.ones(1, dtype=torch.bool)],
        tokens=torch.zeros(1, 3, dtype=torch.int64),
        masks=torch.ones(1, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(embeddings[:, :2], torch.ones(1, 2, 4))
    torch.testing.assert_close(embeddings[:, 2:], torch.full((1, 3, 4), 2.0))


def test_internal_weight_format_targets_static_fp16_linear_and_conv():
    assert PI05Pytorch._npu_internal_weight_format(nn.Linear(4, 4).half()) == 29
    assert PI05Pytorch._npu_internal_weight_format(nn.Conv2d(3, 4, 2).half()) == 4
    assert PI05Pytorch._npu_internal_weight_format(nn.Linear(4, 4).float()) is None
    assert PI05Pytorch._npu_internal_weight_format(nn.Conv2d(4, 4, 2, groups=2).half()) is None


def test_prefix_mlp_addmm_matches_down_projection_plus_residual():
    mlp = SimpleNamespace(
        gate_proj=nn.Linear(4, 8, bias=False),
        up_proj=nn.Linear(4, 8, bias=False),
        down_proj=nn.Linear(8, 4, bias=False),
        act_fn=lambda tensor: nn.functional.gelu(tensor, approximate="tanh"),
    )
    hidden_states = torch.randn(2, 3, 4)
    residual = torch.randn(2, 3, 4)

    expected = mlp.down_proj(mlp.act_fn(mlp.gate_proj(hidden_states)) * mlp.up_proj(hidden_states)) + residual
    actual = PaliGemmaWithExpertModel._npu_mlp_with_residual(mlp, hidden_states, residual)

    torch.testing.assert_close(actual, expected)

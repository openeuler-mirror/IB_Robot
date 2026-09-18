#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
#
# Modified by the IB-Robot project in 2026 to provide an independently owned
# Ascend310P inference implementation, TorchAir graphs, and persistent caching.
# Based on LeRobot v0.5.1 PI0.5 and the linked OpenPI implementation.

import builtins
import hashlib
import logging
import math
import os
import shlex
import sys
import time as time_module
import types
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pi_gemma import (
    PaliGemmaForConditionalGenerationWithPiGemma,
    PiGemmaForCausalLM,
    _gated_residual,
    layernorm_forward,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.import_utils import _transformers_available, require_package
from torch import Tensor, nn
from typing_extensions import Unpack

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

else:
    CONFIG_MAPPING = None
    DynamicCache = None
    DynamicLayer = None
    modeling_gemma = None
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

from .vision_siglip_npu import PI05SiglipVisionModel


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


PI05_GRAPH_DENOISE_STEPS = 10
PI05_COMPILE_VISION_EMBED_ENV = "LEROBOT_PI05_COMPILE_VISION_EMBED"
PI05_ENABLE_VISION_NPU_FUSED_OPS_ENV = "LEROBOT_PI05_ENABLE_VISION_NPU_FUSED_OPS"
PI05_TORCHAIR_CACHE_ABI = "pi05-ascend-310p-two-graph-v2"
PI05_ASCEND_310P_ARCHITECTURE = "pi05-ascend-310p"
_TORCHAIR_CACHE_ENV = "IBROBOT_PI05_TORCHAIR_CACHE"
_TORCHAIR_CACHE_HOME_ENV = "IBROBOT_PI05_TORCHAIR_CACHE_HOME"
_GRAPH_COMPILE_ENV = "IBROBOT_PI05_GRAPH_COMPILE"
_NPU_FUSED_OPS_ENV = "IBROBOT_PI05_NPU_FUSED_OPS"
_STAGE_TIMING_ENV = "IBROBOT_PI05_STAGE_TIMING"
_CONFIG_DEFAULTS = {
    "compile_inference_graph": True,
    "compile_inference_backend": "torchair",
    "compile_inference_fullgraph": True,
    "compile_inference_dynamic": False,
    "compile_frozen_parameter": True,
    "compile_tiling_schedule_optimize": True,
}
_PI05_NO_INIT_FUNCTIONS = (
    "normal_",
    "uniform_",
    "constant_",
    "zeros_",
    "ones_",
    "trunc_normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "sparse_",
    "dirac_",
    "eye_",
    "_no_grad_normal_",
    "_no_grad_uniform_",
    "_no_grad_trunc_normal_",
    "_no_grad_fill_",
    "_no_grad_zero_",
)


def configure_pi05_ascend_310p_config(config: PI05Config, *, model_dtype: str = "fp16") -> PI05Config:
    """Apply the fixed Ascend310P inference contract to a standard PI0.5 config."""

    if config.num_inference_steps != PI05_GRAPH_DENOISE_STEPS:
        raise ValueError(
            f"PI05 Ascend310P requires num_inference_steps={PI05_GRAPH_DENOISE_STEPS}; got {config.num_inference_steps}"
        )
    dtype_by_runtime = {
        "native": config.dtype,
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
    }
    try:
        config.dtype = dtype_by_runtime[model_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported PI05 Ascend310P model dtype: {model_dtype!r}") from exc
    for name, value in _CONFIG_DEFAULTS.items():
        setattr(config, name, value)
    return config


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got {raw!r}")


def _cache_component(value: object) -> str:
    normalized = "".join(character if character.isalnum() or character in "_.-" else "-" for character in str(value))
    return normalized.strip("-.") or "unknown"


def _cann_version() -> str:
    roots = (
        os.environ.get("ASCEND_TOOLKIT_HOME"),
        os.environ.get("ASCEND_HOME_PATH"),
        "/usr/local/Ascend/ascend-toolkit/latest",
    )
    for raw_root in roots:
        if not raw_root:
            continue
        root = Path(raw_root).expanduser()
        for candidate in (
            root / "version.cfg",
            root / "ascend_toolkit_install.info",
            root / "aarch64-linux" / "ascend_toolkit_install.info",
            root / "arm64-linux" / "ascend_toolkit_install.info",
            root / "x86_64-linux" / "ascend_toolkit_install.info",
            root / "runtime" / "version.info",
            root / "toolkit" / "version.info",
        ):
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if "running_version" in line and ":" in line:
                    return line.rsplit(":", maxsplit=1)[-1].rstrip("]").strip()
                if line.startswith("version="):
                    return line.partition("=")[2].strip()
                if line.startswith("version_dir="):
                    return line.partition("=")[2].strip()
    return "unknown"


def _cann_build_identity() -> str:
    """Include toolkit metadata contents so CANN hotfixes cannot share frozen graphs."""

    roots = {
        Path(raw_root).expanduser().resolve()
        for raw_root in (
            os.environ.get("ASCEND_TOOLKIT_HOME"),
            os.environ.get("ASCEND_HOME_PATH"),
            "/usr/local/Ascend/ascend-toolkit/latest",
        )
        if raw_root
    }
    digest = hashlib.sha256()
    found = False
    relative_paths = (
        "version.cfg",
        "ascend_toolkit_install.info",
        "aarch64-linux/ascend_toolkit_install.info",
        "arm64-linux/ascend_toolkit_install.info",
        "x86_64-linux/ascend_toolkit_install.info",
        "runtime/version.info",
        "toolkit/version.info",
    )
    for root in sorted(roots, key=str):
        digest.update(str(root).encode())
        for relative_path in relative_paths:
            candidate = root / relative_path
            try:
                contents = candidate.read_bytes()
            except OSError:
                continue
            found = True
            digest.update(relative_path.encode())
            digest.update(contents)
    version = _cann_version()
    return f"{version}-{digest.hexdigest()[:16]}" if found else version


def _torchair_cache_dir(
    *,
    deployment_fingerprint: str,
    torch_module: object,
    torch_npu_module: object,
    device_name: str,
    model_dtype: str,
    prefix_graph_mode: str,
    npu_fused_ops: bool,
    vision_npu_fused_ops: bool,
) -> Path | None:
    if not _env_flag(_TORCHAIR_CACHE_ENV, default=True):
        return None
    root = Path(os.environ.get(_TORCHAIR_CACHE_HOME_ENV, str(Path.home() / ".cache" / "ibrobot" / "torchair")))
    try:
        import transformers

        transformers_version = transformers.__version__
    except (AttributeError, ImportError):
        transformers_version = "unknown"
    components = (
        PI05_TORCHAIR_CACHE_ABI,
        device_name,
        f"torch-{getattr(torch_module, '__version__', 'unknown')}",
        f"torch-npu-{getattr(torch_npu_module, '__version__', 'unknown')}",
        f"transformers-{transformers_version}",
        f"cann-{_cann_build_identity()}",
        f"dtype-{model_dtype}",
        f"prefix-{prefix_graph_mode}",
        f"npu-fused-{npu_fused_ops}",
        f"vision-fused-{vision_npu_fused_ops}",
        deployment_fingerprint,
    )
    return root.expanduser().joinpath(*(_cache_component(value) for value in components)).resolve()


@contextmanager
def _pi05_no_init_weights():
    """Skip temporary random parameter fills during strict checkpoint loading."""

    originals = {name: getattr(torch.nn.init, name, None) for name in _PI05_NO_INIT_FUNCTIONS}

    def no_initialize(tensor, *args, **kwargs):
        del args, kwargs
        return tensor

    try:
        for name, function in originals.items():
            if function is not None:
                setattr(torch.nn.init, name, no_initialize)
        yield
    finally:
        for name, function in originals.items():
            if function is not None:
                setattr(torch.nn.init, name, function)


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _format_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _log_torchair_cache_status(status: str, cache_root: Path, cache_bin: Path) -> None:
    cleanup = f"rm -rf -- {shlex.quote(str(cache_root))}"
    print(
        "PI05 TorchAir cache "
        f"status={status} path={cache_root} cache_file={cache_bin} "
        f"total_size={_format_size(_directory_size_bytes(cache_root))}; "
        f"stop inference processes before manual cleanup: {cleanup}",
        file=sys.stderr,
        flush=True,
    )


class _TorchAirCachedCallable:
    def __init__(self, target, *, status: str, cache_root: Path, cache_bin: Path) -> None:
        self._target = target
        self._status = status
        self._cache_root = cache_root
        self._cache_bin = cache_bin
        self._reported_first_call = False

    def __call__(self, *args, **kwargs):
        result = self._target(*args, **kwargs)
        if not self._reported_first_call:
            self._reported_first_call = True
            final_status = "HIT" if self._status == "HIT" else "MISS_GENERATED"
            _log_torchair_cache_status(final_status, self._cache_root, self._cache_bin)
        return result


def _device_type_from_config(device: str | torch.device | None) -> str:
    if device is None:
        return ""
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", maxsplit=1)[0]


def _device_supports_bfloat16(device: torch.device | str | None) -> bool:
    device_type = _device_type_from_config(device)
    if device_type == "npu":
        return True
    if device_type == "cuda":
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return False


def _module_device(module: nn.Module) -> torch.device:
    """Return a module's local parameter/buffer device without recursing into children."""
    for param in module.parameters(recurse=False):
        return param.device
    for buffer in module.buffers(recurse=False):
        return buffer.device
    return torch.device("cpu")


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "npu" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def clone_past_key_values(past_key_values):
    """Clone the DynamicCache returned by prefix prefill for compiled denoising."""
    if isinstance(past_key_values, dict):
        return past_key_values

    cloned_cache = DynamicCache()
    for keys, values, sliding_window in past_key_values:
        layer = DynamicLayer()
        layer.keys = keys.clone()
        layer.values = values.clone()
        layer.is_initialized = True
        if sliding_window is not None:
            layer._sliding_window_tensor = sliding_window
        cloned_cache.layers.append(layer)
    return cloned_cache


def _npu_graph_safe_linear_forward(linear: nn.Linear, input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.ndim <= 2 or input_tensor.device.type != "npu":
        return F.linear(input_tensor, linear.weight, linear.bias)

    output_shape = (*input_tensor.shape[:-1], linear.out_features)
    output = F.linear(
        input_tensor.reshape(-1, input_tensor.shape[-1]),
        linear.weight,
        linear.bias,
    )
    return output.reshape(output_shape)


def _npu_graph_safe_gemma_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)

    batch_size, num_heads, query_len, head_dim = query.shape
    key_len = key_states.shape[-2]
    query_3d = query.reshape(batch_size * num_heads, query_len, head_dim)
    key_3d = key_states.reshape(batch_size * num_heads, key_len, head_dim)
    attn_weights = torch.bmm(query_3d, key_3d.transpose(1, 2)) * scaling
    attn_weights = attn_weights.reshape(batch_size, num_heads, query_len, key_len)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_weights_3d = attn_weights.reshape(batch_size * num_heads, query_len, key_len)
    value_3d = value_states.reshape(batch_size * num_heads, key_len, head_dim)
    attn_output = torch.bmm(attn_weights_3d, value_3d)
    attn_output = attn_output.reshape(batch_size, num_heads, query_len, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _npu_graph_safe_gemma_attention_module_forward(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)

    query_states = attention.q_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    key_states = attention.k_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    value_states = attention.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        while len(past_key_values.layers) <= attention.layer_idx:
            past_key_values.layers.append(DynamicLayer())
        cache_layer = past_key_values.layers[attention.layer_idx]
        if not cache_layer.is_initialized:
            cache_layer.keys = key_states
            cache_layer.values = value_states
            cache_layer.is_initialized = True
        else:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                attention.layer_idx,
            )

    attn_output, attn_weights = _npu_graph_safe_gemma_attention_forward(
        attention,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=attention.scaling,
        dropout=0.0 if not attention.training else attention.attention_dropout,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attention.o_proj(attn_output)
    return attn_output, attn_weights


def _npu_graph_safe_siglip_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, query_len, head_dim = query.shape
    key_len = key.shape[-2]
    query_3d = query.reshape(batch_size * num_heads, query_len, head_dim)
    key_3d = key.reshape(batch_size * num_heads, key_len, head_dim)
    attn_weights = torch.bmm(query_3d, key_3d.transpose(1, 2)) * scaling
    attn_weights = attn_weights.reshape(batch_size, num_heads, query_len, key_len)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    value_3d = value.reshape(batch_size * num_heads, key_len, head_dim)
    attn_output = torch.bmm(attn_weights.reshape(batch_size * num_heads, query_len, key_len), value_3d)
    attn_output = attn_output.reshape(batch_size, num_heads, query_len, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _npu_graph_safe_siglip_attention_module_forward(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)

    queries = attention.q_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    keys = attention.k_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)
    values = attention.v_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

    attn_output, attn_weights = _npu_graph_safe_siglip_attention_forward(
        attention,
        queries,
        keys,
        values,
        attention_mask,
        scaling=attention.scale,
        dropout=0.0 if not attention.training else attention.dropout,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attention.out_proj(attn_output)
    return attn_output, attn_weights


def _npu_graph_safe_gemma_rotary_forward(
    rotary_emb: nn.Module,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = position_ids[:, :, None].to(dtype=torch.float32) * rotary_emb.inv_freq[None, None, :].to(
        device=x.device, dtype=torch.float32
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * rotary_emb.attention_scaling
    sin = emb.sin() * rotary_emb.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb):
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    scaling = paligemma_layer.self_attn.scaling
    # Attention computation
    att_output, _ = _npu_graph_safe_gemma_attention_forward(
        paligemma_layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


def _import_torch_npu():
    try:
        import torch_npu  # type: ignore
    except ModuleNotFoundError:
        return None
    return torch_npu


def _is_npu_tensor_list(inputs_embeds: list[torch.Tensor | None]) -> bool:
    for tensor in inputs_embeds:
        if tensor is not None:
            return tensor.device.type == "npu"
    return False


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()
        self._npu_fused_inference_enabled = False
        self._qkv_weights_fused = False

    def to_bfloat16_for_selected_params(
        self,
        precision: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        *,
        vision_precision: Literal["bfloat16", "float16", "float32"] = "float32",
    ):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float16":
            self.to(dtype=torch.float16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")
        if vision_precision not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Invalid vision_precision: {vision_precision}")

        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        if vision_precision == "float32":
            params_to_keep_float32.extend(["vision_tower", "multi_modal_projector"])

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

        if vision_precision in {"bfloat16", "float16"}:
            vision_dtype = torch.bfloat16 if vision_precision == "bfloat16" else torch.float16
            self.paligemma.model.vision_tower.to(dtype=vision_dtype)
            self.paligemma.model.multi_modal_projector.to(dtype=vision_dtype)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    @torch.no_grad()
    def fuse_qkv_weights(self):
        """Create fused qkv Linear layers used by the Ascend NPU attention path."""
        if self._qkv_weights_fused:
            return

        # Fuse q/k/v projections once after loading to reduce inference-time launches.
        for model in (self.paligemma.model.language_model, self.gemma_expert.model):
            for layer in model.layers:
                attn = layer.self_attn
                qkv_weight = torch.cat(
                    [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                    dim=0,
                ).contiguous()
                attn.qkv = nn.Linear(
                    qkv_weight.shape[1],
                    qkv_weight.shape[0],
                    bias=False,
                    device=qkv_weight.device,
                    dtype=qkv_weight.dtype,
                )
                attn.qkv.weight.copy_(qkv_weight)
                attn.qkv.weight.requires_grad_(False)

        self._qkv_weights_fused = True

    def _prepare_vision_pixels(self, image: torch.Tensor) -> torch.Tensor:
        vision_dtype = next(self.paligemma.model.vision_tower.parameters()).dtype
        if image.dtype != vision_dtype:
            return image.to(vision_dtype)
        return image

    def embed_image_vision_tower(self, image: torch.Tensor) -> torch.Tensor:
        image = self._prepare_vision_pixels(image)
        image_outputs = self.paligemma.model.vision_tower(image)
        return image_outputs.last_hidden_state

    def embed_image_projector(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.paligemma.model.multi_modal_projector(image_features)

    def embed_image(self, image: torch.Tensor):
        image_features = self.embed_image_vision_tower(image)
        return self.embed_image_projector(image_features)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.get_input_embeddings()(tokens)

    def prepare_vision_tower_npu_fused_ops(self, *, enable_qkv_fusion: bool = True) -> dict[str, Any]:
        """Replace transformers SigLIP with the local eager-attention implementation."""
        old_vision_tower = self.paligemma.model.vision_tower
        replaced = False
        if isinstance(old_vision_tower, PI05SiglipVisionModel):
            vision_tower = old_vision_tower
        else:
            first_param = next(old_vision_tower.parameters())
            device = first_param.device
            dtype = first_param.dtype
            training = old_vision_tower.training
            requires_grad_by_name = {name: param.requires_grad for name, param in old_vision_tower.named_parameters()}

            vision_tower = PI05SiglipVisionModel(old_vision_tower.config)
            vision_tower.to(device=device, dtype=dtype)
            vision_tower.load_state_dict(old_vision_tower.state_dict(), strict=True)
            for name, param in vision_tower.named_parameters():
                if name in requires_grad_by_name:
                    param.requires_grad_(requires_grad_by_name[name])
            vision_tower.train(training)
            self.paligemma.model.vision_tower = vision_tower
            replaced = True

        if enable_qkv_fusion:
            vision_tower.fuse_qkv_weights()
        return {
            "local_siglip_vision_tower": True,
            "replaced_transformers_vision_tower": replaced,
            "attention": "eager",
            "qkv_weights_fused": bool(enable_qkv_fusion),
        }

    def should_enable_vision_tower_npu_fused_ops(self, *, default: bool = True) -> bool:
        raw_value = os.environ.get(PI05_ENABLE_VISION_NPU_FUSED_OPS_ENV)
        if raw_value is None or raw_value.strip() == "":
            return bool(default)
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{PI05_ENABLE_VISION_NPU_FUSED_OPS_ENV} must be a boolean-like value, got {value!r}")

    def _is_npu_modulation_tree(self, modulation) -> bool:
        if isinstance(modulation, torch.Tensor):
            return modulation.device.type == "npu"
        # Precomputed AdaRMS modulation is a static tuple of module buffers.
        # Recursively walking 10*37*3 tensors inside torch.compile creates
        # excessive Dynamo guards; the buffers are refreshed on the target
        # device before graph capture, so accepting the tuple shape is enough.
        return isinstance(modulation, tuple)

    def _can_use_npu_fused_inference(
        self,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        inputs_embeds: list[torch.FloatTensor] | None,
        adarms_cond: list[torch.Tensor | None] | None,
        adarms_modulations: list[torch.Tensor | None] | None = None,
    ) -> bool:
        if self.training or attention_mask is None or position_ids is None:
            return False
        if not self._npu_fused_inference_enabled:
            return False
        if inputs_embeds is None or not _is_npu_tensor_list(inputs_embeds):
            return False
        if any(embed is not None and not embed.dtype.is_floating_point for embed in inputs_embeds):
            return False
        if _import_torch_npu() is None:
            return False
        if adarms_cond is None:
            adarms_cond = [None, None]
        if adarms_modulations is None:
            adarms_modulations = [None, None]

        active_model_specs = (
            (inputs_embeds[0], self.paligemma.model.language_model, adarms_cond[0], adarms_modulations[0]),
            (inputs_embeds[1], self.gemma_expert.model, adarms_cond[1], adarms_modulations[1]),
        )
        for embed, model, cond, modulation in active_model_specs:
            if embed is None:
                continue
            if getattr(model.config, "use_adarms", False) and cond is None and modulation is None:
                return False
            if cond is not None and cond.device.type != "npu":
                return False
            if modulation is not None and not self._is_npu_modulation_tree(modulation):
                return False
        return True

    def _project_qkv(self, attn, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._qkv_weights_fused:
            q_out = attn.q_proj.out_features
            kv_out = attn.k_proj.out_features
            return attn.qkv(hidden_states).split([q_out, kv_out, kv_out], dim=-1)
        return attn.q_proj(hidden_states), attn.k_proj(hidden_states), attn.v_proj(hidden_states)

    def _attention_projection_dtype(self, attn) -> torch.dtype:
        projection = attn.qkv if self._qkv_weights_fused else attn.q_proj
        return projection.weight.dtype

    def _build_npu_rotary_cache(
        self,
        position_ids: torch.LongTensor,
        head_dim: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_half = head_dim // 2
        freq_exponents = (2.0 / head_dim) * torch.arange(d_half, dtype=torch.float32, device=position_ids.device)
        timescale = 10_000**freq_exponents
        radians = position_ids[..., None].to(torch.float32) / timescale[None, None, :]
        radians = radians[..., None, :]
        cos = torch.cat([torch.cos(radians), torch.cos(radians)], dim=-1)
        sin = torch.cat([torch.sin(radians), torch.sin(radians)], dim=-1)
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def _npu_rotary_emb(
        self,
        torch_npu,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_q_heads = query_states.shape[2]
        num_kv_heads = key_states.shape[2]
        merged_states = torch.cat([query_states, key_states], dim=2)
        merged_states = torch_npu.npu_rotary_mul(merged_states, cos, sin)
        return merged_states.split([num_q_heads, num_kv_heads], dim=2)

    def _npu_rms_norm(self, torch_npu, layernorm, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_weight = layernorm.weight.add(1.0).to(dtype=hidden_states.dtype)
        return torch_npu.npu_rms_norm(hidden_states, norm_weight, layernorm.eps)[0]

    def _npu_adarms_layernorm(
        self,
        torch_npu,
        layernorm,
        hidden_states: torch.Tensor,
        cond: torch.Tensor | None = None,
        modulation: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        *,
        return_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        precomputed_modulation = modulation is not None
        if not precomputed_modulation:
            if cond is None:
                raise ValueError("AdaRMS fused layernorm requires cond or precomputed modulation")
            if cond.shape[-1] != layernorm.cond_dim:
                raise ValueError(f"Expected cond dim {layernorm.cond_dim}, got {cond.shape[-1]}")

            modulation = layernorm.dense(cond)
            if len(hidden_states.shape) == 3:
                modulation = modulation.unsqueeze(1)
            scale, shift, gate = modulation.chunk(3, dim=-1)
            if not return_gate:
                gate = None
        else:
            if isinstance(modulation, tuple):
                scale_weight, shift, gate = modulation
                if not return_gate:
                    gate = None
            else:
                scale_weight = modulation[0, 0, 0]
                shift = modulation[1]
                gate = modulation[2] if return_gate else None

        if not precomputed_modulation:
            scale_weight = 1 + scale.reshape(-1)
        dynamic_weight = scale_weight.to(dtype=hidden_states.dtype).contiguous()
        normed = torch_npu.npu_rms_norm(hidden_states, dynamic_weight, layernorm.eps)[0]
        normed = normed + shift if shift.dtype == normed.dtype else normed.float() + shift.float()
        if gate is not None and not precomputed_modulation and gate.dtype != hidden_states.dtype:
            gate = gate.to(dtype=hidden_states.dtype)
        return normed.to(hidden_states.dtype), gate

    def _npu_or_adarms_layernorm(
        self,
        torch_npu,
        layernorm,
        hidden_states: torch.Tensor,
        cond: torch.Tensor | None,
        modulation: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        *,
        return_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cond is None and getattr(layernorm, "dense", None) is None:
            return self._npu_rms_norm(torch_npu, layernorm, hidden_states), None
        if modulation is not None and getattr(layernorm, "dense", None) is not None:
            return self._npu_adarms_layernorm(
                torch_npu,
                layernorm,
                hidden_states,
                cond=cond,
                modulation=modulation,
                return_gate=return_gate,
            )
        if cond is not None and getattr(layernorm, "dense", None) is not None:
            return self._npu_adarms_layernorm(
                torch_npu,
                layernorm,
                hidden_states,
                cond=cond,
                return_gate=return_gate,
            )
        return layernorm_forward(layernorm, hidden_states, cond)

    @staticmethod
    def _npu_mlp_with_residual(mlp: nn.Module, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        activation = mlp.act_fn(mlp.gate_proj(hidden_states)) * mlp.up_proj(hidden_states)
        output_shape = residual.shape
        output = torch.addmm(
            residual.reshape(-1, output_shape[-1]),
            activation.reshape(-1, activation.shape[-1]),
            mlp.down_proj.weight.t(),
        )
        return output.reshape(output_shape)

    def _cache_layer_tensors(self, past_key_values, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(past_key_values, dict):
            cache = past_key_values[layer_idx]
            return cache["key_states"], cache["value_states"]
        keys, values, _sliding_window = list(past_key_values)[layer_idx]
        return keys, values

    def _forward_npu_optimized(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: dict | None,
        inputs_embeds: list[torch.FloatTensor],
        use_cache: bool | None,
        adarms_cond: list[torch.Tensor | None],
        adarms_modulations: list[torch.Tensor | None] | None = None,
    ):
        # Keep QKV, RoPE and RMSNorm optimizations, but use eager GQA because
        # Prompt Flash Attention is unavailable on Ascend 310P.
        torch_npu = _import_torch_npu()
        if torch_npu is None:
            raise RuntimeError("torch_npu is required for PI0.5 NPU fused inference")
        if adarms_modulations is None:
            adarms_modulations = [None, None]

        active_models = [
            (
                0,
                self.paligemma.model.language_model,
                inputs_embeds[0],
                adarms_cond[0],
                adarms_modulations[0],
            ),
            (
                1,
                self.gemma_expert.model,
                inputs_embeds[1],
                adarms_cond[1],
                adarms_modulations[1],
            ),
        ]
        active_models = [
            (idx, model, embeds, cond, modulation_table)
            for idx, model, embeds, cond, modulation_table in active_models
            if embeds is not None
        ]
        adarms_modulation_offsets = [0, 0]

        num_layers = active_models[0][1].config.num_hidden_layers
        outputs_by_index: list[torch.Tensor | None] = [inputs_embeds[0], inputs_embeds[1]]
        prefix_past_key_values = past_key_values
        if use_cache and prefix_past_key_values is None:
            prefix_past_key_values = {}

        attention_model = active_models[0][1]
        first_attn = attention_model.layers[0].self_attn
        attention_head_dim = first_attn.head_dim
        attention_num_heads = attention_model.config.num_attention_heads
        attention_scale_value = 1.0 / math.sqrt(attention_head_dim)
        rotary_cos, rotary_sin = self._build_npu_rotary_cache(
            position_ids,
            attention_head_dim,
            self._attention_projection_dtype(first_attn),
        )

        for layer_idx in range(num_layers):
            query_states_parts = []
            key_states_parts = []
            value_states_parts = []
            residual_parts = []

            for model_idx, model, hidden_states, cond, modulation_table in active_models:
                layer = model.layers[layer_idx]
                attn = layer.self_attn
                projection_dtype = self._attention_projection_dtype(attn)
                residual = hidden_states.to(dtype=projection_dtype)
                input_modulation = None
                if modulation_table is not None:
                    input_modulation = modulation_table[adarms_modulation_offsets[model_idx]]
                    adarms_modulation_offsets[model_idx] = adarms_modulation_offsets[model_idx] + 1
                normed, gate = self._npu_or_adarms_layernorm(
                    torch_npu, layer.input_layernorm, residual, cond, input_modulation
                )
                if normed.dtype != projection_dtype:
                    normed = normed.to(dtype=projection_dtype)

                hidden_shape = (*normed.shape[:-1], -1, attention_head_dim)
                query_states, key_states, value_states = self._project_qkv(attn, normed)
                query_states_parts.append(query_states.view(hidden_shape))
                key_states_parts.append(key_states.view(hidden_shape))
                value_states_parts.append(value_states.view(hidden_shape))
                residual_parts.append((model_idx, residual, gate, cond))

            query_states = torch.cat(query_states_parts, dim=1)
            key_states = torch.cat(key_states_parts, dim=1)
            value_states = torch.cat(value_states_parts, dim=1)

            query_states, key_states = self._npu_rotary_emb(torch_npu, query_states, key_states, rotary_cos, rotary_sin)

            if use_cache and len(active_models) == 1 and active_models[0][0] == 0:
                prefix_past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            elif past_key_values is not None:
                cached_key_states, cached_value_states = self._cache_layer_tensors(past_key_values, layer_idx)
                key_states = torch.cat([cached_key_states, key_states], dim=1)
                value_states = torch.cat([cached_value_states, value_states], dim=1)

            batch_size = query_states.shape[0]
            att_output, _ = _npu_graph_safe_gemma_attention_forward(
                first_attn,
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask,
                attention_scale_value,
            )
            att_output = att_output.reshape(batch_size, -1, attention_num_heads * attention_head_dim)

            next_active_models = []
            start = 0
            for model_idx, model, hidden_states, cond, modulation_table in active_models:
                layer = model.layers[layer_idx]
                _residual_model_idx, residual, gate, _cond = residual_parts.pop(0)
                end = start + hidden_states.shape[1]
                out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                if cond is None and getattr(layer.post_attention_layernorm, "dense", None) is None:
                    # npu_add_rms_norm is numerically unstable on Ascend 310P with FP16.
                    after_first_residual = out_emb + residual.to(out_emb.dtype)
                    out_emb = self._npu_rms_norm(
                        torch_npu,
                        layer.post_attention_layernorm,
                        after_first_residual,
                    )
                    out_emb = self._npu_mlp_with_residual(layer.mlp, out_emb, after_first_residual)
                else:
                    after_first_residual = _gated_residual(residual, out_emb, gate)
                    post_modulation = None
                    if modulation_table is not None:
                        post_modulation = modulation_table[adarms_modulation_offsets[model_idx]]
                        adarms_modulation_offsets[model_idx] = adarms_modulation_offsets[model_idx] + 1
                    out_emb, gate = self._npu_or_adarms_layernorm(
                        torch_npu,
                        layer.post_attention_layernorm,
                        after_first_residual,
                        cond,
                        post_modulation,
                    )
                    if out_emb.dtype != layer.mlp.up_proj.weight.dtype:
                        out_emb = out_emb.to(dtype=layer.mlp.up_proj.weight.dtype)
                    out_emb = layer.mlp(out_emb)
                    out_emb = _gated_residual(after_first_residual, out_emb, gate)

                outputs_by_index[model_idx] = out_emb
                next_active_models.append((model_idx, model, out_emb, cond, modulation_table))
                start = end
            active_models = next_active_models

        for model_idx, model, hidden_states, cond, modulation_table in active_models:
            if cond is None and getattr(model.norm, "dense", None) is None:
                outputs_by_index[model_idx] = self._npu_rms_norm(torch_npu, model.norm, hidden_states)
            else:
                final_modulation = None
                if modulation_table is not None:
                    final_modulation = modulation_table[adarms_modulation_offsets[model_idx]]
                    adarms_modulation_offsets[model_idx] = adarms_modulation_offsets[model_idx] + 1
                outputs_by_index[model_idx], _ = self._npu_or_adarms_layernorm(
                    torch_npu,
                    model.norm,
                    hidden_states,
                    cond,
                    final_modulation,
                    return_gate=False,
                )

        return outputs_by_index, prefix_past_key_values if use_cache else None

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        adarms_modulations: list[torch.Tensor | None] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if adarms_modulations is None:
            adarms_modulations = [None, None]
        if self._can_use_npu_fused_inference(
            attention_mask, position_ids, inputs_embeds, adarms_cond, adarms_modulations
        ):
            return self._forward_npu_optimized(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
                adarms_modulations=adarms_modulations,
            )
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            paligemma_layers = self.paligemma.model.language_model.layers
            gemma_expert_layers = self.gemma_expert.model.layers
            rotary_emb = self.paligemma.model.language_model.rotary_emb

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layers in zip(paligemma_layers, gemma_expert_layers, strict=True):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        layers=layers,
                        rotary_emb=rotary_emb,
                    )

            # final norm
            final_norms = (
                self.paligemma.model.language_model.norm,
                self.gemma_expert.model.norm,
            )

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(final_norms[i], hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.register_buffer("_action_denoise_dt", torch.empty(0, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_action_denoise_timestep_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_cond_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_suffix_position_ids",
            torch.empty(0, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_scale_weight_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_shift_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_action_denoise_adarms_gate_table",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self._action_denoise_adarms_modulation_steps: tuple[
            tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
            ...,
        ] = ()

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False
        self._sample_actions_graph_compile_enabled = False
        self._compiled_action_vision_embed = None
        self._compiled_action_prefix_prefill = None
        self._compiled_action_prefix_forward = None
        self._compiled_action_denoise_10_steps = None
        self._sample_actions_graph_compile_info: dict[str, Any] = {}
        self._action_fused_stage_timing_enabled = False
        self._action_fused_stage_timing_call_index = 0
        self._action_fused_stage_timing_last_record: dict[str, Any] | None = None
        self._action_fused_stage_timing_records: list[dict[str, Any]] = []
        self._npu_internal_format_weights_info = {
            "enabled": False,
            "linear_weights": 0,
            "conv2d_weights": 0,
        }

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _apply(self, fn):
        result = super()._apply(fn)
        self._rebuild_adarms_modulation_step_refs()
        return result

    def prepare_inference_optimizations(
        self,
        *,
        enable_npu_fused_ops: bool = False,
        enable_graph_compile: bool | None = None,
        enable_qkv_fusion: bool | None = None,
        torchair_cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Finalize optional PI0.5 inference optimizations after weights are loaded."""
        self._prepare_action_compute_dtype()
        enable_graph_compile = (
            bool(getattr(self.config, "compile_inference_graph", False))
            if enable_graph_compile is None
            else bool(enable_graph_compile)
        )
        enable_qkv_fusion = (
            bool(enable_npu_fused_ops or enable_graph_compile) if enable_qkv_fusion is None else bool(enable_qkv_fusion)
        )
        self.paligemma_with_expert._npu_fused_inference_enabled = bool(enable_npu_fused_ops or enable_graph_compile)
        enable_vision_npu_fused_ops = bool(enable_npu_fused_ops or enable_graph_compile)
        optimization_info = {
            "npu_fused_ops_enabled": self.paligemma_with_expert._npu_fused_inference_enabled,
            "qkv_fusion_requested": bool(enable_qkv_fusion),
            "qkv_weights_fused": self.paligemma_with_expert._qkv_weights_fused,
            "action_compute_dtype": str(self._action_compute_dtype()),
            "vision_compute_dtype": str(self._vision_compute_dtype()),
            "adarms_npu_mode": "dynamic_weight",
            "npu_internal_format_weights": dict(self._npu_internal_format_weights_info),
            "fixed_denoise_lookup_tables": self._fixed_denoise_lookup_tables_info(),
            "graph_compile": dict(self._sample_actions_graph_compile_info),
            "vision_tower_npu_fused_ops": {
                "local_siglip_vision_tower": isinstance(
                    self.paligemma_with_expert.paligemma.model.vision_tower,
                    PI05SiglipVisionModel,
                ),
                "enabled": False,
                "default_enabled": True,
                "enable_env": PI05_ENABLE_VISION_NPU_FUSED_OPS_ENV,
                "attention": "transformers_default",
                "qkv_weights_fused": False,
            },
        }
        if enable_qkv_fusion:
            # Graph and eager inference both reuse the fused QKV projections.
            self.paligemma_with_expert.fuse_qkv_weights()
            optimization_info["qkv_weights_fused"] = self.paligemma_with_expert._qkv_weights_fused
        if enable_vision_npu_fused_ops and self.paligemma_with_expert.should_enable_vision_tower_npu_fused_ops(
            default=True
        ):
            optimization_info["vision_tower_npu_fused_ops"] = (
                self.paligemma_with_expert.prepare_vision_tower_npu_fused_ops(enable_qkv_fusion=enable_qkv_fusion)
            )
            optimization_info["vision_tower_npu_fused_ops"]["enabled"] = True
            optimization_info["vision_tower_npu_fused_ops"]["default_enabled"] = True
            optimization_info["vision_tower_npu_fused_ops"]["enable_env"] = PI05_ENABLE_VISION_NPU_FUSED_OPS_ENV
        if enable_npu_fused_ops or enable_graph_compile:
            optimization_info["action_compute_dtype"] = str(self._action_compute_dtype())
            optimization_info["vision_compute_dtype"] = str(self._vision_compute_dtype())
            optimization_info["adarms_npu_mode"] = "dynamic_weight"
        if enable_graph_compile:
            self._refresh_fixed_denoise_lookup_tables()
        if enable_npu_fused_ops or enable_graph_compile:
            optimization_info["npu_internal_format_weights"] = self._prepare_npu_internal_format_weights()
        if enable_graph_compile:
            if not self._sample_actions_graph_compile_enabled:
                self.enable_sample_actions_graph_compile(torchair_cache_dir=torchair_cache_dir)
            optimization_info["fixed_denoise_lookup_tables"] = self._fixed_denoise_lookup_tables_info()
            optimization_info["graph_compile"] = dict(self._sample_actions_graph_compile_info)
        return optimization_info

    def _action_compute_dtype(self, device: torch.device | str | None = None) -> torch.dtype:
        target_device = device
        if target_device is None:
            try:
                target_device = next(self.parameters()).device
            except StopIteration:
                target_device = self.config.device
        requested = getattr(self.config, "dtype", "bfloat16")
        if requested == "bfloat16" and _device_supports_bfloat16(target_device):
            return torch.bfloat16
        if requested == "float16":
            return torch.float16
        return torch.float32

    def _cast_action_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = self._action_compute_dtype(tensor.device)
        if tensor.dtype.is_floating_point and tensor.dtype != target_dtype:
            return tensor.to(dtype=target_dtype)
        return tensor

    def _vision_compute_dtype(self) -> torch.dtype:
        return self._action_compute_dtype()

    def _prepare_action_compute_dtype(self) -> None:
        target_dtype = self._action_compute_dtype()
        precision: Literal["bfloat16", "float16", "float32"] = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
        }.get(target_dtype, "float32")
        vision_precision: Literal["bfloat16", "float16", "float32"] = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
        }.get(self._vision_compute_dtype(), "float32")
        self.paligemma_with_expert.to_bfloat16_for_selected_params(
            precision,
            vision_precision=vision_precision,
        )
        modules = [
            self.action_in_proj,
            self.action_out_proj,
            self.time_mlp_in,
            self.time_mlp_out,
        ]
        for module in modules:
            module.to(dtype=target_dtype)

    @staticmethod
    def _npu_internal_weight_format(module: nn.Module) -> int | None:
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype != torch.float16:
            return None
        if isinstance(module, nn.Linear):
            return 29  # ACL_FORMAT_FRACTAL_NZ
        if isinstance(module, nn.Conv2d) and module.groups == 1:
            return 4  # ACL_FORMAT_FRACTAL_Z
        return None

    @torch.no_grad()
    def _prepare_npu_internal_format_weights(self) -> dict[str, Any]:
        """Prepack static FP16 weights so inference does not repeat TransData each request."""
        if self._npu_internal_format_weights_info["enabled"]:
            return dict(self._npu_internal_format_weights_info)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            return dict(self._npu_internal_format_weights_info)
        if device.type != "npu":
            return dict(self._npu_internal_format_weights_info)
        torch_npu = _import_torch_npu()
        if torch_npu is None:
            return dict(self._npu_internal_format_weights_info)

        counts = {"linear_weights": 0, "conv2d_weights": 0}
        for module in self.modules():
            target_format = self._npu_internal_weight_format(module)
            if target_format is None:
                continue
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, target_format)
            key = "linear_weights" if isinstance(module, nn.Linear) else "conv2d_weights"
            counts[key] += 1
        self._npu_internal_format_weights_info = {"enabled": True, **counts}
        return dict(self._npu_internal_format_weights_info)

    def _set_denoise_lookup_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Register or update a non-persistent fixed denoise lookup buffer."""
        if name in self._buffers:
            setattr(self, name, tensor)
        else:
            self.register_buffer(name, tensor, persistent=False)

    def _fixed_denoise_lookup_tables_info(self) -> dict[str, Any]:
        scale_table = self._action_denoise_adarms_scale_weight_table
        shift_table = self._action_denoise_adarms_shift_table
        gate_table = self._action_denoise_adarms_gate_table
        return {
            "denoise_steps": PI05_GRAPH_DENOISE_STEPS,
            "dt_shape": list(self._action_denoise_dt.shape),
            "timestep_table_shape": list(self._action_denoise_timestep_table.shape),
            "adarms_cond_table_shape": list(self._action_denoise_adarms_cond_table.shape),
            "suffix_position_ids_shape": list(self._action_denoise_suffix_position_ids.shape),
            "adarms_modulation_layout": "split_scale_shift_gate_buffers",
            "adarms_scale_weight_table_shape": list(scale_table.shape),
            "adarms_shift_table_shape": list(shift_table.shape),
            "adarms_gate_table_shape": list(gate_table.shape),
            "adarms_modulation_step_buffers": sum(
                len(step_modulations) for step_modulations in self._action_denoise_adarms_modulation_steps
            ),
            "dtype": str(scale_table.dtype),
            "device": str(scale_table.device),
        }

    def _action_expert_adarms_layernorms(self) -> list[nn.Module]:
        layernorms: list[nn.Module] = []
        for layer in self.paligemma_with_expert.gemma_expert.model.layers:
            layernorms.append(layer.input_layernorm)
            layernorms.append(layer.post_attention_layernorm)
        layernorms.append(self.paligemma_with_expert.gemma_expert.model.norm)
        return layernorms

    def _refresh_adarms_modulation_step_buffers(
        self,
        scale_weight_table: torch.Tensor,
        shift_table: torch.Tensor,
        gate_table: torch.Tensor,
    ) -> None:
        """Split scale/shift/gate tables into per-step/layer buffers for graph capture."""
        step_modulations: list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]] = []
        num_steps, num_layernorms = scale_weight_table.shape[:2]
        for step in range(num_steps):
            layer_modulations: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(num_layernorms):
                scale_name = f"_action_denoise_adarms_scale_weight_s{step}_l{layer_idx}"
                shift_name = f"_action_denoise_adarms_shift_s{step}_l{layer_idx}"
                gate_name = f"_action_denoise_adarms_gate_s{step}_l{layer_idx}"
                self._set_denoise_lookup_buffer(
                    scale_name,
                    scale_weight_table[step, layer_idx].contiguous(),
                )
                self._set_denoise_lookup_buffer(
                    shift_name,
                    shift_table[step, layer_idx].contiguous(),
                )
                self._set_denoise_lookup_buffer(
                    gate_name,
                    gate_table[step, layer_idx].contiguous(),
                )
                layer_modulations.append(
                    (
                        getattr(self, scale_name),
                        getattr(self, shift_name),
                        getattr(self, gate_name),
                    )
                )
            step_modulations.append(tuple(layer_modulations))
        self._action_denoise_adarms_modulation_steps = tuple(step_modulations)

    def _rebuild_adarms_modulation_step_refs(self) -> None:
        """Rebuild Python tuple references after module device/dtype application."""
        scale_table = self._buffers.get("_action_denoise_adarms_scale_weight_table")
        if scale_table is None or scale_table.numel() == 0:
            self._action_denoise_adarms_modulation_steps = ()
            return
        step_modulations: list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]] = []
        num_steps, num_layernorms = scale_table.shape[:2]
        for step in range(num_steps):
            layer_modulations: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(num_layernorms):
                scale_name = f"_action_denoise_adarms_scale_weight_s{step}_l{layer_idx}"
                shift_name = f"_action_denoise_adarms_shift_s{step}_l{layer_idx}"
                gate_name = f"_action_denoise_adarms_gate_s{step}_l{layer_idx}"
                if scale_name not in self._buffers or shift_name not in self._buffers or gate_name not in self._buffers:
                    self._action_denoise_adarms_modulation_steps = ()
                    return
                layer_modulations.append(
                    (
                        getattr(self, scale_name),
                        getattr(self, shift_name),
                        getattr(self, gate_name),
                    )
                )
            step_modulations.append(tuple(layer_modulations))
        self._action_denoise_adarms_modulation_steps = tuple(step_modulations)

    @torch.no_grad()
    def _refresh_fixed_denoise_lookup_tables(self) -> None:
        """Precompute fixed PI0.5 denoise timestep, AdaRMS condition and modulation tables."""
        device = _module_device(self.time_mlp_in)
        target_dtype = self._action_compute_dtype(device)
        timestep_table_f32 = (
            1.0
            - torch.arange(
                PI05_GRAPH_DENOISE_STEPS,
                dtype=torch.float32,
                device=device,
            )
            / PI05_GRAPH_DENOISE_STEPS
        )
        time_embedding_f32 = create_sinusoidal_pos_embedding(
            timestep_table_f32,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=device,
        )
        time_embedding = time_embedding_f32.to(dtype=self.time_mlp_in.weight.dtype)
        adarms_cond = self.time_mlp_in(time_embedding)
        adarms_cond = F.silu(adarms_cond)
        adarms_cond = self.time_mlp_out(adarms_cond)
        adarms_cond = F.silu(adarms_cond).to(dtype=target_dtype).contiguous()

        scale_weight_tables: list[torch.Tensor] = []
        shift_tables: list[torch.Tensor] = []
        gate_tables: list[torch.Tensor] = []
        for layernorm in self._action_expert_adarms_layernorms():
            modulation = layernorm.dense(adarms_cond)
            scale, shift, gate = modulation.reshape(PI05_GRAPH_DENOISE_STEPS, 3, -1).unbind(dim=1)
            scale_weight_tables.append((1 + scale).to(dtype=target_dtype).contiguous())
            shift_tables.append(shift[:, None, None, :].to(dtype=target_dtype).contiguous())
            gate_tables.append(gate[:, None, None, :].to(dtype=target_dtype).contiguous())
        adarms_scale_weight_table = torch.stack(scale_weight_tables, dim=1).contiguous()
        adarms_shift_table = torch.stack(shift_tables, dim=1).contiguous()
        adarms_gate_table = torch.stack(gate_tables, dim=1).contiguous()

        self._action_denoise_dt = torch.tensor(
            -1.0 / PI05_GRAPH_DENOISE_STEPS,
            dtype=target_dtype,
            device=device,
        )
        self._action_denoise_timestep_table = timestep_table_f32.to(dtype=target_dtype)
        self._action_denoise_suffix_position_ids = torch.arange(
            self.config.chunk_size,
            dtype=torch.int64,
            device=device,
        )
        self._action_denoise_adarms_cond_table = adarms_cond
        self._action_denoise_adarms_scale_weight_table = adarms_scale_weight_table
        self._action_denoise_adarms_shift_table = adarms_shift_table
        self._action_denoise_adarms_gate_table = adarms_gate_table
        self._refresh_adarms_modulation_step_buffers(
            adarms_scale_weight_table,
            adarms_shift_table,
            adarms_gate_table,
        )

    def _force_eager_attention_for_graph_compile(self) -> None:
        """Force graph-safe eager attention kernels before TorchAir capture."""
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        except Exception:
            all_attention_functions = None
        else:
            all_attention_functions = ALL_ATTENTION_FUNCTIONS
        if all_attention_functions is not None:
            all_attention_functions.register("eager_bmm", _npu_graph_safe_gemma_attention_forward)
        gemma_attention_functions = getattr(modeling_gemma, "ALL_ATTENTION_FUNCTIONS", None)
        if gemma_attention_functions is not None:
            gemma_attention_functions.register("eager_bmm", _npu_graph_safe_gemma_attention_forward)

        modules = [
            self.paligemma_with_expert.paligemma,
            self.paligemma_with_expert.paligemma.model.vision_tower,
        ]
        for module in modules:
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"

        gemma_modules = [
            self.paligemma_with_expert.paligemma.model.language_model,
            self.paligemma_with_expert.gemma_expert.model,
        ]
        for module in gemma_modules:
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager_bmm"

    def _patch_linear_for_npu_graph_compile(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and not getattr(module, "_pi05_graph_safe_linear", False):
                module.forward = types.MethodType(_npu_graph_safe_linear_forward, module)
                module._pi05_graph_safe_linear = True

    def _patch_gemma_attention_for_npu_graph_compile(self) -> None:
        gemma_attention_cls = getattr(modeling_gemma, "GemmaAttention", None)
        if gemma_attention_cls is None:
            return
        for module in self.modules():
            if isinstance(module, gemma_attention_cls) and not getattr(module, "_pi05_graph_safe_attention", False):
                module.forward = types.MethodType(_npu_graph_safe_gemma_attention_module_forward, module)
                module._pi05_graph_safe_attention = True

    def _patch_siglip_attention_for_npu_graph_compile(self) -> None:
        try:
            from transformers.models.siglip.modeling_siglip import SiglipAttention
        except Exception:
            return
        for module in self.modules():
            if isinstance(module, SiglipAttention) and not getattr(module, "_pi05_graph_safe_attention", False):
                module.forward = types.MethodType(_npu_graph_safe_siglip_attention_module_forward, module)
                module._pi05_graph_safe_attention = True

    def _patch_gemma_rotary_for_npu_graph_compile(self) -> None:
        rotary_cls = getattr(modeling_gemma, "GemmaRotaryEmbedding", None)
        if rotary_cls is None:
            return
        for module in self.modules():
            if isinstance(module, rotary_cls) and not getattr(module, "_pi05_graph_safe_rotary", False):
                module.forward = types.MethodType(_npu_graph_safe_gemma_rotary_forward, module)
                module._pi05_graph_safe_rotary = True

    def _prefix_graph_mode(self) -> str:
        device = next(self.parameters()).device
        if device.type != "npu":
            return "full_prefix"
        raw_value = os.environ.get(PI05_COMPILE_VISION_EMBED_ENV)
        if raw_value is not None and raw_value.strip():
            value = raw_value.strip().lower()
            if value in {"1", "true", "yes", "on"}:
                return "compiled_vision_split"
            if value in {"0", "false", "no", "off"}:
                return "eager_vision_split"
            raise ValueError(f"{PI05_COMPILE_VISION_EMBED_ENV} must be a boolean-like value, got {value!r}")
        return "compiled_vision_split"

    def enable_sample_actions_graph_compile(
        self,
        *,
        backend=None,
        fullgraph: bool | None = None,
        dynamic: bool | None = None,
        frozen_parameter: bool | None = None,
        tiling_schedule_optimize: bool | None = None,
        torchair_cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compile PI0.5 prefix and fixed 10-step denoise fragments."""
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        self._force_eager_attention_for_graph_compile()
        if next(self.parameters()).device.type == "npu":
            self._patch_linear_for_npu_graph_compile()
            self._patch_gemma_attention_for_npu_graph_compile()
            self._patch_siglip_attention_for_npu_graph_compile()
            self._patch_gemma_rotary_for_npu_graph_compile()

        fullgraph = self.config.compile_inference_fullgraph if fullgraph is None else fullgraph
        dynamic = self.config.compile_inference_dynamic if dynamic is None else dynamic
        frozen_parameter = self.config.compile_frozen_parameter if frozen_parameter is None else frozen_parameter
        tiling_schedule_optimize = (
            self.config.compile_tiling_schedule_optimize
            if tiling_schedule_optimize is None
            else tiling_schedule_optimize
        )
        compile_kwargs, backend_name = self._build_inference_compile_kwargs(
            backend=backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
            frozen_parameter=frozen_parameter,
            tiling_schedule_optimize=tiling_schedule_optimize,
        )

        prefix_graph_mode = self._prefix_graph_mode()
        use_torchair_cache = torchair_cache_dir is not None and backend_name == "torchair" and backend is None
        compiler_config = (
            self._build_torchair_compiler_config(
                frozen_parameter=frozen_parameter,
                tiling_schedule_optimize=tiling_schedule_optimize,
            )
            if use_torchair_cache
            else None
        )

        def compile_target(target):
            if not use_torchair_cache:
                return torch.compile(target, **compile_kwargs)
            return self._compile_torchair_cached(
                target,
                compiler_config=compiler_config,
                dynamic=bool(dynamic),
                cache_root=Path(torchair_cache_dir),
            )

        if prefix_graph_mode == "full_prefix":
            # Prefix block IDs are constant, so the prefix mask is the pad-mask outer product.
            # Avoid the large broadcast-bool LogicalAnd path that is miscompiled by CANN 8.1.
            self._compiled_action_prefix_forward = compile_target(self._action_prefix_forward_for_compile)
        else:
            if prefix_graph_mode == "compiled_vision_split":
                self._compiled_action_vision_embed = compile_target(self.paligemma_with_expert.embed_image)
            self._compiled_action_prefix_prefill = compile_target(self._action_prefix_prefill_for_compile)
        self._compiled_action_denoise_10_steps = compile_target(self._action_denoise_10_steps_for_compile)
        self._sample_actions_graph_compile_enabled = True
        self._sample_actions_graph_compile_info = {
            "enabled": True,
            "backend": backend_name,
            "dynamic": dynamic,
            "fullgraph": fullgraph,
            "graph_path": {
                "full_prefix": "compiled_full_prefix_denoise",
                "compiled_vision_split": "compiled_vision_compiled_prefill_denoise",
                "eager_vision_split": "eager_vision_compiled_prefill_denoise",
            }[prefix_graph_mode],
            "prefix_compiled": True,
            "prefix_vision_compiled": prefix_graph_mode != "eager_vision_split",
            "denoise_steps": PI05_GRAPH_DENOISE_STEPS,
            "denoise_compilation": "single_graph",
            "frozen_parameter": frozen_parameter,
            "tiling_schedule_optimize": tiling_schedule_optimize,
            "torchair_cache_enabled": use_torchair_cache,
            "torchair_cache_dir": str(Path(torchair_cache_dir).resolve()) if use_torchair_cache else None,
            "targets": [
                *(["embed_image"] if prefix_graph_mode == "compiled_vision_split" else []),
                *(
                    ["_action_prefix_forward_for_compile"]
                    if prefix_graph_mode == "full_prefix"
                    else ["_action_prefix_prefill_for_compile"]
                ),
                "_action_denoise_10_steps_for_compile",
            ],
        }
        return dict(self._sample_actions_graph_compile_info)

    def _build_inference_compile_kwargs(
        self,
        *,
        backend=None,
        fullgraph: bool | None = None,
        dynamic: bool | None = None,
        frozen_parameter: bool | None = None,
        tiling_schedule_optimize: bool | None = None,
    ) -> tuple[dict[str, Any], str]:
        fullgraph = self.config.compile_inference_fullgraph if fullgraph is None else fullgraph
        dynamic = self.config.compile_inference_dynamic if dynamic is None else dynamic
        frozen_parameter = self.config.compile_frozen_parameter if frozen_parameter is None else frozen_parameter
        tiling_schedule_optimize = (
            self.config.compile_tiling_schedule_optimize
            if tiling_schedule_optimize is None
            else tiling_schedule_optimize
        )

        compile_kwargs: dict[str, Any] = {"dynamic": dynamic, "fullgraph": fullgraph}
        backend_name = self.config.compile_inference_backend or "auto"
        if backend is None:
            device_type = next(self.parameters()).device.type
            if backend_name in {None, "auto"}:
                if device_type == "npu":
                    backend = self._build_torchair_backend(
                        frozen_parameter=frozen_parameter,
                        tiling_schedule_optimize=tiling_schedule_optimize,
                    )
                    backend_name = "torchair"
                else:
                    compile_kwargs["mode"] = self.config.compile_mode
                    backend_name = "inductor"
            elif backend_name == "torchair":
                backend = self._build_torchair_backend(
                    frozen_parameter=frozen_parameter,
                    tiling_schedule_optimize=tiling_schedule_optimize,
                )
            elif backend_name == "npugraph_ex":
                backend = self._build_npugraph_ex_backend()
            elif backend_name == "inductor":
                compile_kwargs["mode"] = self.config.compile_mode
            else:
                backend = backend_name
        else:
            backend_name = type(backend).__name__

        if backend is not None:
            compile_kwargs["backend"] = backend
        return compile_kwargs, backend_name

    def _build_npugraph_ex_backend(self):
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("npugraph_ex backend requires torch_npu") from exc

        try:
            import npugraph_ex  # type: ignore
        except ModuleNotFoundError:
            try:
                from torch_npu.dynamo import npugraph_ex  # type: ignore
            except Exception as exc:
                raise RuntimeError("npugraph_ex backend requires torch_npu.dynamo.npugraph_ex") from exc
            sys.modules.setdefault("npugraph_ex", npugraph_ex)

        config_cls = getattr(npugraph_ex, "CompilerConfig", None)
        if config_cls is None:
            from npugraph_ex.configs.compiler_config import CompilerConfig  # type: ignore

            config_cls = CompilerConfig

        compiler_config = config_cls()
        if hasattr(compiler_config, "mode"):
            compiler_config.mode = "npugraph_ex"
        return npugraph_ex.get_npu_backend(compiler_config=compiler_config)

    def _build_torchair_backend(
        self,
        *,
        frozen_parameter: bool = True,
        tiling_schedule_optimize: bool = True,
    ):
        try:
            import torch_npu  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("TorchAir backend requires torch_npu") from exc

        try:
            import torchair  # type: ignore
        except ModuleNotFoundError:
            try:
                from torch_npu.dynamo import torchair  # type: ignore
            except Exception as exc:
                raise RuntimeError("TorchAir backend requires torchair") from exc
            sys.modules.setdefault("torchair", torchair)

        config_cls = getattr(torchair, "CompilerConfig", None)
        if config_cls is None:
            from torchair.configs.compiler_config import CompilerConfig  # type: ignore

            config_cls = CompilerConfig

        compiler_config = self._build_torchair_compiler_config(
            frozen_parameter=frozen_parameter,
            tiling_schedule_optimize=tiling_schedule_optimize,
            config_cls=config_cls,
        )
        return torchair.get_npu_backend(compiler_config=compiler_config)

    def _build_torchair_compiler_config(
        self,
        *,
        frozen_parameter: bool = True,
        tiling_schedule_optimize: bool = True,
        config_cls=None,
    ):
        if config_cls is None:
            try:
                from torch_npu.dynamo.torchair.configs.compiler_config import CompilerConfig
            except ImportError as exc:
                raise RuntimeError("TorchAir compiler config requires torch_npu") from exc
            config_cls = CompilerConfig

        compiler_config = config_cls()
        experimental_config = getattr(compiler_config, "experimental_config", None)
        if experimental_config is not None:
            if hasattr(experimental_config, "frozen_parameter"):
                experimental_config.frozen_parameter = frozen_parameter
            if hasattr(experimental_config, "tiling_schedule_optimize"):
                experimental_config.tiling_schedule_optimize = tiling_schedule_optimize
        return compiler_config

    def _compile_torchair_cached(
        self,
        target,
        *,
        compiler_config,
        dynamic: bool,
        cache_root: Path,
    ):
        try:
            from torch_npu.dynamo.torchair.inference import cache_compile
            from torch_npu.dynamo.torchair.inference._cache_compiler import CompiledModel, ModelCacheSaver
        except ImportError as exc:
            raise RuntimeError("TorchAir cache requires torch_npu.dynamo.torchair.inference") from exc

        cache_root = cache_root.expanduser().resolve()
        original_repr = type(self).__repr__

        def stable_repr(_instance):
            return f"{type(self).__module__}.{type(self).__qualname__}(cache_abi={PI05_TORCHAIR_CACHE_ABI})"

        type(self).__repr__ = stable_repr
        try:
            cache_bin = Path(
                CompiledModel.get_cache_bin(
                    target,
                    config=compiler_config,
                    dynamic=dynamic,
                    cache_dir=str(cache_root),
                    ge_cache=True,
                )
            )
            cache_existed = cache_bin.is_file()
            cached = cache_compile(
                target,
                config=compiler_config,
                dynamic=dynamic,
                cache_dir=str(cache_root),
                ge_cache=True,
            )
            cached._compiled_model = cached.compile()
        finally:
            type(self).__repr__ = original_repr

        status = "HIT" if cache_existed and not isinstance(cached._compiled_model, ModelCacheSaver) else "MISS"
        _log_torchair_cache_status(status, cache_root, cache_bin)
        return _TorchAirCachedCallable(cached, status=status, cache_root=cache_root, cache_bin=cache_bin)

    def _can_use_compiled_action_inference(self, rtc_kwargs: dict[str, Any]) -> bool:
        return (
            self._sample_actions_graph_compile_enabled
            and (self._compiled_action_prefix_forward is not None or self._compiled_action_prefix_prefill is not None)
            and self._compiled_action_denoise_10_steps is not None
            and self.rtc_processor is None
            and not rtc_kwargs
        )

    def _make_action_graph_att_2d_masks(self, pad_masks, att_masks):
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum.unsqueeze(1) <= cumsum.unsqueeze(2)
        pad_2d_masks = pad_masks.unsqueeze(1) & pad_masks.unsqueeze(2)
        return att_2d_masks & pad_2d_masks

    def enable_action_fused_stage_timing(self, enabled: bool = True, *, clear: bool = True) -> None:
        """Enable synchronized model-side timing for the fused PI0.5 inference path."""
        self._action_fused_stage_timing_enabled = bool(enabled)
        if clear:
            self.clear_action_fused_stage_timing()

    def clear_action_fused_stage_timing(self) -> None:
        self._action_fused_stage_timing_call_index = 0
        self._action_fused_stage_timing_last_record = None
        self._action_fused_stage_timing_records = []

    def get_last_action_fused_stage_timing(self) -> dict[str, Any] | None:
        return self._copy_action_fused_stage_record(self._action_fused_stage_timing_last_record)

    def get_action_fused_stage_timing_records(self) -> list[dict[str, Any]]:
        return [self._copy_action_fused_stage_record(record) for record in self._action_fused_stage_timing_records]

    def _copy_action_fused_stage_record(self, record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        copied = dict(record)
        copied["denoise_step_ms"] = list(record.get("denoise_step_ms", []))
        return copied

    def _sync_action_fused_stage_timing_device(self, device: torch.device) -> None:
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.synchronize(device)
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

    def _start_action_fused_stage_timing(
        self,
        *,
        num_steps: int,
        prefix_mode: str,
        prefix_tokens: int,
        denoise_suffix_tokens: int,
    ) -> dict[str, Any] | None:
        if not self._action_fused_stage_timing_enabled:
            return None
        self._action_fused_stage_timing_call_index += 1
        return {
            "call_index": self._action_fused_stage_timing_call_index,
            "prefix_mode": prefix_mode,
            "num_steps": int(num_steps),
            "prefix_tokens": int(prefix_tokens),
            "denoise_suffix_tokens": int(denoise_suffix_tokens),
            "prefix_ms": None,
            "denoise_step_ms": [],
            "denoise_graph_ms": None,
            "denoise_total_ms": None,
            "denoise_steps": int(num_steps),
            "npu_fused_ops": bool(self.paligemma_with_expert._npu_fused_inference_enabled),
        }

    def _finish_action_fused_stage_timing(self, record: dict[str, Any] | None) -> None:
        if record is None:
            return
        denoise_step_ms = record["denoise_step_ms"]
        if record.get("denoise_total_ms") is None:
            if record.get("denoise_graph_ms") is not None:
                record["denoise_total_ms"] = float(record["denoise_graph_ms"])
            else:
                record["denoise_total_ms"] = float(sum(denoise_step_ms))
                record["denoise_steps"] = len(denoise_step_ms)
        self._action_fused_stage_timing_last_record = record
        self._action_fused_stage_timing_records.append(record)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        target_dtype = self._action_compute_dtype(device)
        normal_dtype = target_dtype
        if torch.device(device).type == "npu" and target_dtype == torch.bfloat16:
            normal_dtype = torch.float32
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=normal_dtype,
            device=device,
        )
        if noise.dtype != target_dtype:
            return noise.to(dtype=target_dtype)
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(self, images, img_masks, tokens, masks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []

        def image_embed_func(stacked_images):
            bsize, num_views = stacked_images.shape[:2]
            flat_images = stacked_images.reshape(bsize * num_views, *stacked_images.shape[2:])
            embed_image = getattr(self, "_compiled_action_vision_embed", None)
            if embed_image is None:
                embed_image = self.paligemma_with_expert.embed_image
            flat_img_emb = embed_image(flat_images)
            num_img_embs, hidden_dim = flat_img_emb.shape[1:]
            return flat_img_emb.reshape(bsize, num_views * num_img_embs, hidden_dim)

        stacked_images = torch.stack(images, dim=1)
        img_emb = self._apply_checkpoint(image_embed_func, stacked_images)
        bsize = img_emb.shape[0]
        num_img_embs = img_emb.shape[1] // len(images)

        embs.append(img_emb)
        stacked_img_masks = torch.stack(img_masks, dim=1)
        pad_masks.append(stacked_img_masks[:, :, None].expand(bsize, len(images), num_img_embs).reshape(bsize, -1))

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        bsize = pad_masks.shape[0]
        att_masks = torch.zeros(bsize, pad_masks.shape[1], dtype=torch.bool, device=pad_masks.device)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.to(dtype=self.time_mlp_in.weight.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        first_action_att_mask = torch.ones(bsize, 1, dtype=embs.dtype, device=embs.device)
        remaining_action_att_mask = torch.zeros(
            bsize,
            action_time_dim - 1,
            dtype=embs.dtype,
            device=embs.device,
        )
        att_masks = torch.cat([first_action_att_mask, remaining_action_att_mask], dim=1)

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise, time) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def _action_prefix_embed_for_compile(self, images, img_masks, tokens, masks):
        return self.embed_prefix(images, img_masks, tokens, masks)

    def _action_prefix_masks_for_compile(self, prefix_pad_masks, prefix_att_masks):
        del prefix_att_masks
        prefix_att_2d_masks = prefix_pad_masks.unsqueeze(1) & prefix_pad_masks.unsqueeze(2)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        return prefix_att_2d_masks, prefix_position_ids

    def _action_prefix_prefill_for_compile(
        self,
        prefix_embs,
        prefix_att_2d_masks_4d,
        prefix_position_ids,
    ):
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return past_key_values

    def _action_prefix_forward_for_compile(self, images, img_masks, tokens, masks):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._action_prefix_embed_for_compile(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks, prefix_position_ids = self._action_prefix_masks_for_compile(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        past_key_values = self._action_prefix_prefill_for_compile(
            prefix_embs,
            prefix_att_2d_masks_4d,
            prefix_position_ids,
        )
        return prefix_pad_masks, past_key_values

    def _action_prefix_with_compiled_prefill(self, images, img_masks, tokens, masks):
        if self._compiled_action_prefix_forward is not None:
            return self._compiled_action_prefix_forward(images, img_masks, tokens, masks)
        if self._compiled_action_prefix_prefill is None:
            raise RuntimeError("compiled prefix is not initialized")
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._action_prefix_embed_for_compile(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks, prefix_position_ids = self._action_prefix_masks_for_compile(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        past_key_values = self._compiled_action_prefix_prefill(
            prefix_embs,
            prefix_att_2d_masks_4d,
            prefix_position_ids,
        )
        return prefix_pad_masks, past_key_values

    def _embed_suffix_embs_with_adarms_cond(self, noisy_actions, adarms_cond):
        """Embed PI0.5 action suffix using a precomputed fixed-step AdaRMS condition."""
        noisy_actions = noisy_actions.to(dtype=self.action_in_proj.weight.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        device = action_emb.device

        if adarms_cond.ndim == 1:
            adarms_cond = adarms_cond[None, :].expand(action_emb.shape[0], -1)
        adarms_cond = adarms_cond.to(dtype=action_emb.dtype, device=device)
        return action_emb, adarms_cond

    def embed_suffix_with_adarms_cond(self, noisy_actions, adarms_cond):
        """Embed suffix and build eager denoise masks for one PI0.5 action step."""
        action_emb, adarms_cond = self._embed_suffix_embs_with_adarms_cond(noisy_actions, adarms_cond)
        bsize, action_time_dim = action_emb.shape[:2]
        device = action_emb.device

        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        first_action_att_mask = torch.ones(bsize, 1, dtype=action_emb.dtype, device=device)
        remaining_action_att_mask = torch.zeros(
            bsize,
            action_time_dim - 1,
            dtype=action_emb.dtype,
            device=device,
        )
        att_masks = torch.cat([first_action_att_mask, remaining_action_att_mask], dim=1)
        return action_emb, action_time_mask, att_masks, adarms_cond

    def _denoise_step_from_suffix_embs(
        self,
        prefix_pad_masks,
        past_key_values,
        suffix_embs,
        suffix_pad_masks,
        suffix_att_masks,
        adarms_cond,
        adarms_modulation=None,
    ):
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (  # noqa: SLF001
            "eager_bmm" if self._sample_actions_graph_compile_enabled and suffix_embs.device.type == "npu" else "eager"
        )

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            adarms_modulations=[None, adarms_modulation],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    def denoise_step_with_adarms_cond(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        adarms_cond,
        adarms_modulation=None,
    ):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix_with_adarms_cond(
            x_t,
            adarms_cond,
        )
        return self._denoise_step_from_suffix_embs(
            prefix_pad_masks,
            past_key_values,
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond,
            adarms_modulation,
        )

    def _run_action_denoise_forward(
        self,
        past_key_values,
        suffix_embs,
        adarms_cond,
        adarms_modulation,
        full_att_2d_masks_4d,
        position_ids,
    ):
        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
            adarms_modulations=[None, adarms_modulation],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    def _prepare_denoise_mask_context_for_compile(self, prefix_pad_masks, x_t):
        batch_size, suffix_len = x_t.shape[:2]
        prefix_len = prefix_pad_masks.shape[1]
        device = x_t.device
        suffix_pad_masks = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=device)
        first_action_att_mask = torch.ones(batch_size, 1, dtype=x_t.dtype, device=device)
        remaining_action_att_mask = torch.zeros(
            batch_size,
            suffix_len - 1,
            dtype=x_t.dtype,
            device=device,
        )
        suffix_att_masks = torch.cat([first_action_att_mask, remaining_action_att_mask], dim=1)

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        if self._action_denoise_suffix_position_ids.numel() != suffix_len:
            suffix_position_ids = torch.arange(suffix_len, dtype=torch.int64, device=device)
        else:
            suffix_position_ids = self._action_denoise_suffix_position_ids
        position_ids = prefix_offsets + suffix_position_ids[None, :]
        return full_att_2d_masks_4d, position_ids

    def _denoise_step_with_prepared_mask_context(
        self,
        past_key_values,
        x_t,
        adarms_cond,
        adarms_modulation,
        full_att_2d_masks_4d,
        position_ids,
    ):
        suffix_embs, adarms_cond = self._embed_suffix_embs_with_adarms_cond(x_t, adarms_cond)
        return self._run_action_denoise_forward(
            past_key_values=past_key_values,
            suffix_embs=suffix_embs,
            adarms_cond=adarms_cond,
            adarms_modulation=adarms_modulation,
            full_att_2d_masks_4d=full_att_2d_masks_4d,
            position_ids=position_ids,
        )

    def _action_denoise_10_steps_for_compile(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
    ):
        dt_tensor = self._action_denoise_dt.to(dtype=x_t.dtype)
        adarms_cond_table = self._action_denoise_adarms_cond_table
        adarms_modulation_steps = self._action_denoise_adarms_modulation_steps
        full_att_2d_masks_4d, position_ids = self._prepare_denoise_mask_context_for_compile(prefix_pad_masks, x_t)
        for step in range(PI05_GRAPH_DENOISE_STEPS):
            v_t = self._denoise_step_with_prepared_mask_context(
                past_key_values=past_key_values,
                x_t=x_t,
                adarms_cond=adarms_cond_table[step],
                adarms_modulation=adarms_modulation_steps[step],
                full_att_2d_masks_4d=full_att_2d_masks_4d,
                position_ids=position_ids,
            )
            x_t = x_t + dt_tensor * v_t
        return x_t

    def _sample_actions_graph_inference(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise,
        num_steps,
    ):
        if int(num_steps) != PI05_GRAPH_DENOISE_STEPS:
            raise ValueError(
                "PI0.5 graph inference compiles denoise as one fixed 10-step TorchAir graph; "
                f"got num_steps={num_steps}."
            )
        device = tokens.device
        x_t = self._cast_action_tensor(noise)
        if (
            self._compiled_action_prefix_forward is None and self._compiled_action_prefix_prefill is None
        ) or self._compiled_action_denoise_10_steps is None:
            raise RuntimeError("graph inference requires compiled prefix and 10-step denoise callables")

        stage_record = self._start_action_fused_stage_timing(
            num_steps=num_steps,
            prefix_mode="compiled",
            prefix_tokens=0,
            denoise_suffix_tokens=self.config.chunk_size,
        )
        if stage_record is not None:
            self._sync_action_fused_stage_timing_device(device)
            prefix_start = time_module.perf_counter()
            prefix_pad_masks, past_key_values = self._action_prefix_with_compiled_prefill(
                images, img_masks, tokens, masks
            )
            self._sync_action_fused_stage_timing_device(device)
            stage_record["prefix_ms"] = (time_module.perf_counter() - prefix_start) * 1000.0
            stage_record["prefix_tokens"] = int(prefix_pad_masks.shape[1])
        else:
            prefix_pad_masks, past_key_values = self._action_prefix_with_compiled_prefill(
                images, img_masks, tokens, masks
            )

        if stage_record is not None:
            self._sync_action_fused_stage_timing_device(device)
            denoise_start = time_module.perf_counter()
            x_t = self._compiled_action_denoise_10_steps(prefix_pad_masks, past_key_values, x_t)
            self._sync_action_fused_stage_timing_device(device)
            denoise_ms = (time_module.perf_counter() - denoise_start) * 1000.0
            stage_record["denoise_graph_ms"] = denoise_ms
            stage_record["denoise_total_ms"] = denoise_ms
            stage_record["denoise_steps"] = PI05_GRAPH_DENOISE_STEPS
        else:
            x_t = self._compiled_action_denoise_10_steps(prefix_pad_masks, past_key_values, x_t)
        self._finish_action_fused_stage_timing(stage_record)
        return x_t

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)
        elif noise.dtype.is_floating_point:
            noise = self._cast_action_tensor(noise)

        if self._can_use_compiled_action_inference(kwargs):
            # Use the compiled prefix graph and fixed ten-step denoise graph.
            return self._sample_actions_graph_inference(
                images,
                img_masks,
                tokens,
                masks,
                noise,
                num_steps,
            ).to(dtype=torch.float32)

        prefix_timing_enabled = self._action_fused_stage_timing_enabled
        if prefix_timing_enabled:
            self._sync_action_fused_stage_timing_device(device)
            prefix_start = time_module.perf_counter()

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        prefix_mode = "npu_fused_ops" if self.paligemma_with_expert._npu_fused_inference_enabled else "baseline_eager"
        stage_record = self._start_action_fused_stage_timing(
            num_steps=num_steps,
            prefix_mode=prefix_mode,
            prefix_tokens=prefix_pad_masks.shape[1],
            denoise_suffix_tokens=self.config.chunk_size,
        )

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        if stage_record is not None:
            self._sync_action_fused_stage_timing_device(device)
            stage_record["prefix_ms"] = (time_module.perf_counter() - prefix_start) * 1000.0

            self._sync_action_fused_stage_timing_device(device)
            denoise_start = time_module.perf_counter()

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        if stage_record is not None:
            self._sync_action_fused_stage_timing_device(device)
            denoise_ms = (time_module.perf_counter() - denoise_start) * 1000.0
            stage_record["denoise_graph_ms"] = denoise_ms
            stage_record["denoise_total_ms"] = denoise_ms
            stage_record["denoise_steps"] = int(num_steps)

        self._finish_action_fused_stage_timing(stage_record)
        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = clone_past_key_values(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)


class PI05Ascend310PPolicy(PreTrainedPolicy):
    """PI0.5 policy with an Ascend310P-owned inference implementation."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        require_package("transformers", extra="pi")
        configure_pi05_ascend_310p_config(
            config, model_dtype={"float16": "fp16", "bfloat16": "bf16"}.get(config.dtype, "fp32")
        )
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()
        # Inference is the only supported runtime for this hardware-specific policy.
        self.eval()

    def prepare_inference_optimizations(
        self,
        *,
        enable_npu_fused_ops: bool = False,
        enable_graph_compile: bool | None = None,
        enable_qkv_fusion: bool | None = None,
        torchair_cache_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Finalize optional PI0.5 inference optimizations after loading or device placement."""
        # Expose one policy-level entry point for eager NPU and TorchAir graph setup.
        self.eval()
        return self.model.prepare_inference_optimizations(
            enable_npu_fused_ops=enable_npu_fused_ops,
            enable_graph_compile=enable_graph_compile,
            enable_qkv_fusion=enable_qkv_fusion,
            torchair_cache_dir=torchair_cache_dir,
        )

    def prepare_for_inference(
        self,
        *,
        deployment_fingerprint: str,
        torch_module: object,
        torch_npu_module: object,
        device_name: str,
    ) -> dict[str, Any]:
        """Finalize the fixed Ascend310P fused-op and two-graph runtime."""

        enable_graph_compile = _env_flag(_GRAPH_COMPILE_ENV, default=True)
        enable_npu_fused_ops = _env_flag(_NPU_FUSED_OPS_ENV, default=True)
        enable_vision_npu_fused_ops = _env_flag(PI05_ENABLE_VISION_NPU_FUSED_OPS_ENV, default=True)
        self.model.enable_action_fused_stage_timing(_env_flag(_STAGE_TIMING_ENV, default=False))
        prefix_graph_mode = self.model._prefix_graph_mode()
        cache_dir = _torchair_cache_dir(
            deployment_fingerprint=deployment_fingerprint,
            torch_module=torch_module,
            torch_npu_module=torch_npu_module,
            device_name=device_name,
            model_dtype=self.config.dtype,
            prefix_graph_mode=prefix_graph_mode,
            npu_fused_ops=enable_npu_fused_ops,
            vision_npu_fused_ops=enable_vision_npu_fused_ops,
        )
        return self.prepare_inference_optimizations(
            enable_graph_compile=enable_graph_compile,
            enable_npu_fused_ops=enable_npu_fused_ops,
            enable_qkv_fusion=False,
            torchair_cache_dir=cache_dir if enable_graph_compile else None,
        )

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        skip_weight_init: bool = False,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 Ascend310P model is based on the LeRobot/OpenPI implementation. \n"
            "It owns the hardware-specific inference graph without patching LeRobot. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        if skip_weight_init and not strict:
            raise ValueError("skip_weight_init requires strict=True")

        original_device = config.device
        if skip_weight_init:
            config.device = torch.device("cpu")
        try:
            context = _pi05_no_init_weights() if skip_weight_init else nullcontext()
            with context:
                model = cls(config, **kwargs)
        finally:
            config.device = original_device

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from safetensors.torch import load_file

                checkpoint = Path(pretrained_name_or_path).expanduser().resolve() / "model.safetensors"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                original_state_dict = load_file(checkpoint)
                print("Loaded state dict from model.safetensors")
            except Exception as e:
                if skip_weight_init:
                    raise RuntimeError(f"Could not load PI05 checkpoint: {e}") from e
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            missing_keys, unexpected_keys = model.load_state_dict(
                remapped_state_dict,
                strict=strict,
                assign=skip_weight_init,
            )

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            if skip_weight_init:
                raise RuntimeError(f"Could not load PI05 checkpoint: {e}") from e
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = (
                    value.clone()
                )

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), "RTC is not supported for select_action, use it with predict_action_chunk"

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

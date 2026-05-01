#!/usr/bin/env python3
"""
PyTorch 运行时 Hook —— 从 ACT decoder 捕获 cross-attention 权重。

不修改 lerobot 源码，通过 monkey-patch + forward_hook 方式
在推理时从 nn.MultiheadAttention 模块提取注意力权重。

lerobot 的 ACTDecoderLayer 默认使用 need_weights=False，
因此需要 monkey-patch forward 以强制开启权重返回。
"""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class AttentionWeightHook:
    """
    通过 forward hook 从 ACT decoder 的 cross-attention 层捕获注意力权重。

    用法::

        hook = AttentionWeightHook()
        hook.install(policy)      # policy 是 ACTPolicy 实例
        # ... 运行推理 ...
        data = hook.get_latest()  # 获取注意力数据
        hook.uninstall()          # 清理
    """

    def __init__(self):
        self._handles: list = []          # forward hook handles
        self._original_forwards: list = []  # 原始 forward 方法
        self._latest_weights: Tensor | None = None
        self._feature_map_size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._enabled: bool = True
        self._camera_keys: list[str] = []
        self._num_non_image_tokens: int = 0
        self._decoder_attention_mask: Tensor | None = None
        self._encoder_key_padding_mask: Tensor | None = None

    # ---- 公共接口 ----

    def install(self, policy: Any) -> bool:
        """
        安装 hook 到 ACT policy 模型。

        遍历 policy.model.decoder.layers，对每个 multihead_attn：
        1. 保存原始 forward
        2. monkey-patch 为 need_weights=True
        3. 注册 forward_hook 捕获 attn_weights

        Args:
            policy: LeRobot ACTPolicy 实例。

        Returns:
            是否成功安装。
        """
        try:
            model = getattr(policy, "model", policy)

            # 找到 decoder layers
            decoder = getattr(model, "decoder", None)
            if decoder is None:
                logger.warning("Policy has no 'decoder' attribute.")
                return False

            layers = getattr(decoder, "layers", None)
            if layers is None:
                logger.warning("Decoder has no 'layers' attribute.")
                return False

            # 获取 camera keys（从 policy config）
            config = getattr(policy, "config", None)
            if config and hasattr(config, "image_features"):
                self._camera_keys = list(config.image_features.keys())

            decoder_count = 0
            for layer in layers:
                mha = getattr(layer, "multihead_attn", None)
                if mha is None:
                    continue

                # 保存原始 forward
                original_forward = mha.forward
                self._original_forwards.append((mha, original_forward))

                # LeRobot's ACT decoder normally calls MultiheadAttention with
                # need_weights=False for speed. The hook only needs inference
                # telemetry, so force weight emission without touching
                # third-party source files.
                def make_patched_forward(orig_fwd):
                    def patched_forward(
                        query, key, value,
                        key_padding_mask=None,
                        need_weights=True,       # 强制 True
                        attn_mask=None,
                        average_attn_weights=False,  # 强制 False
                        is_causal=False,
                    ):
                        if attn_mask is None:
                            attn_mask = self._mask_for_device(
                                self._decoder_attention_mask,
                                query.device,
                            )
                        return orig_fwd(
                            query, key, value,
                            key_padding_mask=key_padding_mask,
                            need_weights=True,
                            attn_mask=attn_mask,
                            average_attn_weights=False,
                            is_causal=is_causal,
                        )
                    return patched_forward

                mha.forward = make_patched_forward(original_forward)

                # 注册 forward_hook 捕获注意力权重
                handle = mha.register_forward_hook(self._make_hook_fn())
                self._handles.append(handle)

                decoder_count += 1

            encoder_count = self._patch_encoder_self_attention(model)
            installed_count = decoder_count + encoder_count

            logger.info(
                "AttentionWeightHook installed on "
                f"{decoder_count} decoder layers and {encoder_count} "
                f"encoder layers, cameras={self._camera_keys}",
            )
            return installed_count > 0

        except Exception as e:
            logger.error(f"Failed to install attention hook: {e}")
            self.uninstall()
            return False

    def get_latest(self) -> dict[str, Any] | None:
        """
        获取最近一次推理的注意力权重数据。

        Returns:
            字典包含:
            - attn_weights: Tensor (num_heads, query_len, key_len) 或
              Tensor (num_layers, batch, num_heads, query_len, key_len)
            - feature_map_size: (h, w) 或 None
            - camera_keys: list[str]
            - num_non_image_tokens: int
            如果没有数据则返回 None。
        """
        with self._lock:
            if self._latest_weights is None:
                return None
            return {
                "attn_weights": self._latest_weights,
                "feature_map_size": self._feature_map_size,
                "camera_keys": list(self._camera_keys),
                "num_non_image_tokens": self._num_non_image_tokens,
            }

    def set_feature_map_size(self, size: tuple[int, int]):
        """设置特征图尺寸（由外部推理流程计算后传入）。"""
        with self._lock:
            self._feature_map_size = size

    def set_camera_keys(self, keys: list[str]):
        """设置相机键名。"""
        with self._lock:
            self._camera_keys = keys

    def set_num_non_image_tokens(self, n: int):
        """设置非图像 token 数量。"""
        with self._lock:
            self._num_non_image_tokens = n

    def set_attention_masks(
        self,
        decoder_attention_mask: Tensor | None,
        encoder_key_padding_mask: Tensor | None,
    ) -> None:
        """Set optional transformer masks created from user-drawn regions."""
        with self._lock:
            self._decoder_attention_mask = decoder_attention_mask
            self._encoder_key_padding_mask = encoder_key_padding_mask

    def clear_attention_masks(self) -> None:
        """Clear any user-supplied transformer masks."""
        with self._lock:
            self._decoder_attention_mask = None
            self._encoder_key_padding_mask = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def uninstall(self):
        """恢复原始 forward 方法，移除所有 hooks。"""
        for handle in self._handles:
            with suppress(Exception):
                handle.remove()
        self._handles.clear()

        for mha, original_forward in self._original_forwards:
            with suppress(Exception):
                mha.forward = original_forward
        self._original_forwards.clear()

        with self._lock:
            self._latest_weights = None
            self._decoder_attention_mask = None
            self._encoder_key_padding_mask = None

        logger.info("AttentionWeightHook uninstalled.")

    # ---- 内部方法 ----

    def _make_hook_fn(self):
        """创建 forward_hook 回调函数。"""
        hook_ref = self

        def hook_fn(module, input, output):
            if not hook_ref._enabled:
                return

            # nn.MultiheadAttention with need_weights=True returns
            # (attn_output, attn_weights) where attn_weights shape is
            # (batch*num_heads, query_len, key_len) or (batch, num_heads, query_len, key_len)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    with hook_ref._lock:
                        hook_ref._latest_weights = attn_weights.detach().cpu()

        return hook_fn

    def _patch_encoder_self_attention(self, model: Any) -> int:
        """Patch ACT encoder self-attention so pixel masks hide source tokens."""
        encoder = getattr(model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            return 0

        patched_count = 0
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue
            original_forward = self_attn.forward
            self._original_forwards.append((self_attn, original_forward))

            def make_patched_encoder_forward(orig_fwd):
                def patched_forward(
                    query, key, value,
                    key_padding_mask=None,
                    need_weights=True,
                    attn_mask=None,
                    average_attn_weights=True,
                    is_causal=False,
                ):
                    if key_padding_mask is None:
                        key_padding_mask = self._mask_for_device(
                            self._encoder_key_padding_mask,
                            query.device,
                        )
                    return orig_fwd(
                        query,
                        key,
                        value,
                        key_padding_mask=key_padding_mask,
                        need_weights=need_weights,
                        attn_mask=attn_mask,
                        average_attn_weights=average_attn_weights,
                        is_causal=is_causal,
                    )
                return patched_forward

            self_attn.forward = make_patched_encoder_forward(original_forward)
            patched_count += 1
        return patched_count

    def _mask_for_device(
        self,
        mask: Tensor | None,
        device: torch.device,
    ) -> Tensor | None:
        if mask is None:
            return None
        return mask.to(device=device, dtype=torch.bool)

    def __del__(self):
        self.uninstall()


class StackedAttentionHook(AttentionWeightHook):
    """
    增强版 Hook —— 收集所有 decoder 层的注意力权重并 stack。

    用于需要多层注意力分析的场景。
    """

    def __init__(self):
        super().__init__()
        self._all_layer_weights: list = []
        self._expected_layer_count: int = 0

    def install(self, policy: Any) -> bool:
        """安装 hook 并重置状态。"""
        with self._lock:
            self._all_layer_weights.clear()
            self._latest_weights = None
            self._expected_layer_count = 0
        installed = super().install(policy)
        with self._lock:
            self._expected_layer_count = len(self._handles) if installed else 0
        return installed

    def _make_hook_fn(self):
        hook_ref = self

        def hook_fn(module, input, output):
            if not hook_ref._enabled:
                return

            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    with hook_ref._lock:
                        hook_ref._all_layer_weights.append(attn_weights.detach().cpu())
                        if (
                            hook_ref._expected_layer_count
                            and len(hook_ref._all_layer_weights)
                            >= hook_ref._expected_layer_count
                        ):
                            hook_ref._latest_weights = torch.stack(
                                hook_ref._all_layer_weights[
                                    -hook_ref._expected_layer_count:
                                ]
                            )

        return hook_fn

    def get_latest(self) -> dict[str, Any] | None:
        """返回所有层的堆叠注意力权重。"""
        result = super().get_latest()
        return result

    def reset_for_new_inference(self):
        """在新的推理步骤前重置层收集。"""
        with self._lock:
            self._all_layer_weights.clear()
            self._latest_weights = None

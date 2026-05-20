"""
PI05Wrapper.py

加载 PI05 VLM 和 Action Expert 两个 OM 模型并协调推理.

使用 PI05OMModel 实现 buffer 共享，减少 kv_cache 的 D2H + H2D 开销.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .PI05OMModel import PI05OMModel
from .prefix_mask_utils import build_prefix_att_2d_masks_4d_np as _build_prefix_mask

# --- 昇腾板上关闭算子 JIT 编译以提速 ---
if hasattr(torch, "npu"):
    with contextlib.suppress(Exception):  # pragma: no cover - non-Ascend hosts may shim torch.npu
        torch.npu.set_compile_mode(jit_compile=False)


# Debug flag: set ``PI05_OM_DEBUG=1`` to enable verbose diagnostics.
_DEBUG = os.environ.get("PI05_OM_DEBUG", "").lower() in ("1", "true", "yes")


def logger(msg: str):
    print(f"[PI05Wrapper]: {msg}")


def _t_stats(t: Tensor) -> str:
    """Short tensor stats string (shape/dtype/min/max/mean)."""
    try:
        if t.dtype in (torch.bool, torch.int32, torch.int64):
            return f"shape={tuple(t.shape)} dtype={t.dtype} min={int(t.min())} max={int(t.max())} sum={int(t.sum())}"
        f = t.detach().float()
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={float(f.min()):+.4g} max={float(f.max()):+.4g} "
            f"mean={float(f.mean()):+.4g} std={float(f.std()):+.4g}"
        )
    except Exception as exc:
        return f"shape={tuple(t.shape)} dtype={t.dtype} (stats failed: {exc})"


# Match constants in lerobot/utils/constants.py without importing the module
# at top level (keeps this file usable in unit tests that stub lerobot).
_OBS_LANGUAGE_TOKENS = "observation.language.tokens"
_OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


class PI05Wrapper:
    """
    Wrapper for PI05 Ascend OM inference.

    Coordinates two OM models:
    - VLM: processes images + language tokens → KV cache
    - Action Expert: takes KV cache + noise → denoised actions
    """

    def __init__(
        self,
        vlm_model_path: str,
        action_expert_model_path: str,
        config: Any,
    ):
        """
        Initialize the PI05 OM wrapper.

        Args:
            vlm_model_path: Path to the VLM OM model file
            action_expert_model_path: Path to the Action Expert OM model file
            config: duck-typed config exposing ``chunk_size``, ``max_action_dim``,
                ``num_inference_steps`` and ``image_features`` (an iterable
                mapping whose key order matches the exported ONNX/OM input order).
        """
        self.config = config
        self.chunk_size = config.chunk_size
        self.max_action_dim = config.max_action_dim
        self.num_inference_steps = config.num_inference_steps

        logger("Initializing PI05OMModel with buffer sharing...")
        self.model = PI05OMModel(vlm_model_path, action_expert_model_path, config)
        logger("PI05OMModel initialized successfully")

        # Output shape: (batch_size, chunk_size, max_action_dim)
        self.output_shape = [1, self.chunk_size, self.max_action_dim]
        logger(f"Output shape: {self.output_shape}")

    def predict(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Perform PI05 inference using VLM and Action Expert OM models.

        Accepts the raw observation batch directly. The OM model's computation
        graph already includes image preprocessing (resize, normalize, etc.),
        so no preprocessing should be done before calling this method.

        Args:
            batch: Raw observation batch dict containing image tensors,
                   language tokens and attention masks.
                   Optionally contains ``_noise`` (Tensor) for deterministic
                   cross-machine comparison.

        Returns:
            Action tensor of shape (B, chunk_size, action_dim)
        """
        # Extract optional external noise
        noise_tensor = batch.pop("_noise", None)

        if _DEBUG:
            logger(f"predict() called. batch keys = {list(batch.keys())}")
            for k, v in batch.items():
                if isinstance(v, Tensor):
                    logger(f"  batch[{k!r}]: {_t_stats(v)}")
                else:
                    logger(f"  batch[{k!r}]: type={type(v).__name__}")
            if noise_tensor is not None:
                logger(f"  external noise: {_t_stats(noise_tensor)}")

        # Extract raw data from batch
        images, tokens, masks = self._extract_from_batch(batch)
        device = tokens.device

        if _DEBUG:
            for i, img in enumerate(images):
                logger(f"  extracted image[{i}]: {_t_stats(img)}")
            logger(f"  extracted tokens: {_t_stats(tokens)}")
            logger(f"  extracted masks:  {_t_stats(masks)}")

        # Prepare inputs as numpy arrays
        images_np = [img.cpu().numpy().astype(np.float32) for img in images]
        tokens_np = tokens.cpu().numpy().astype(np.int64)
        # NOTE: ``np.bool8`` was removed in numpy>=1.24; use ``np.bool_``.
        masks_np = masks.cpu().numpy().astype(np.bool_)

        # Build the 4D additive prefix attention mask on host so
        # OPENPI_ATTENTION_MASK_VALUE never enters the OM graph
        # (ATC fp16 corrupts that constant otherwise).
        prefix_mask_np = self._build_prefix_att_2d_masks_4d_np(
            num_cameras=len(images_np),
            lang_masks=masks_np,
            prefix_seq_len=self.model.prefix_seq_len,
        )
        if _DEBUG:
            logger(
                f"  built prefix_att_2d_masks_4d: shape={prefix_mask_np.shape} "
                f"dtype={prefix_mask_np.dtype} "
                f"valid={int((prefix_mask_np == 0.0).sum())} "
                f"masked={int((prefix_mask_np != 0.0).sum())}"
            )

        noise_np = noise_tensor.cpu().numpy() if noise_tensor is not None else None

        actions = self.model.forward(images_np, tokens_np, masks_np, prefix_mask_np, noise=noise_np)

        actions = actions.to(device).reshape(*self.output_shape)
        return actions

    def _extract_from_batch(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], Tensor, Tensor]:
        """
        Extract all camera images, language tokens and masks from observation batch.

        Images are returned in the same order as ``config.image_features`` so
        that they match the positional VLM ONNX/OM inputs.
        """
        images: list[Tensor] = []
        for key in self.config.image_features:
            if key in batch:
                images.append(batch[key])
            else:
                raise ValueError(f"Missing image key {key!r} in batch. Available keys: {list(batch.keys())}")

        if not images:
            raise ValueError(f"No image found in batch. Expected: {list(self.config.image_features)}")

        tokens = batch.get(_OBS_LANGUAGE_TOKENS, batch.get("lang_tokens"))
        masks = batch.get(_OBS_LANGUAGE_ATTENTION_MASK, batch.get("lang_masks"))
        if tokens is None or masks is None:
            raise ValueError("Missing language tokens or attention masks in batch")

        return images, tokens, masks

    @staticmethod
    def _build_prefix_att_2d_masks_4d_np(
        *,
        num_cameras: int,
        lang_masks: np.ndarray,
        prefix_seq_len: int,
    ) -> np.ndarray:
        """Delegate to :func:`prefix_mask_utils.build_prefix_att_2d_masks_4d_np`."""
        return _build_prefix_mask(
            num_cameras=num_cameras,
            lang_masks=lang_masks,
            prefix_seq_len=prefix_seq_len,
        )

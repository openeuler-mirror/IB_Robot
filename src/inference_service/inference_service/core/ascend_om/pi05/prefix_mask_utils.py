# Copyright (c) 2025 Syslong Technology Co., Ltd. All Rights Reserved.
# Licensed under the Mulan PSL v2.
"""Pure-numpy utility for building the PI05 prefix attention mask.

This module has **zero hardware dependencies** (no ``acl``, no device
libs) so it can be imported safely on any machine — PC, GPU server, or
NPU board.

The single public function :func:`build_prefix_att_2d_masks_4d_np`
constructs the ``(B, 1, S, S)`` fp32 additive mask that the VLM ONNX /
OM model expects as its ``prefix_att_2d_masks_4d`` input (Plan A: this
constant was extracted from the ONNX graph to avoid ATC fp16 corruption
of the ``-2.38e38`` masking value).
"""

from __future__ import annotations

import numpy as np
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE


def build_prefix_att_2d_masks_4d_np(
    *,
    num_cameras: int,
    lang_masks: np.ndarray,  # (B, lang_seq_len) bool
    prefix_seq_len: int,  # S, fixed by the OM input slot
) -> np.ndarray:
    """Build the 4D additive prefix attention mask in pure numpy.

    For PI05 the per-token ``att_masks`` produced by ``embed_prefix``
    are **all zero** (every prefix block is bidirectional), so the
    Big-Vision ``make_att_2d_masks`` cumsum trick collapses to the
    outer product of ``pad_masks``.  That means we only need:

      * one ``True`` for every image token  (cameras are always
        present once they're in ``config.image_features``);
      * the language ``lang_masks`` for the trailing language block.

    Args:
        num_cameras: Number of camera tensors fed to the VLM (sets
            the image-token block count).
        lang_masks: ``(B, lang_seq_len)`` bool mask for language tokens.
        prefix_seq_len: Total prefix length ``S`` (queried from the
            OM input slot for ``prefix_att_2d_masks_4d``).

    Returns:
        ``(B, 1, S, S)`` fp32 array with ``0.0`` on visible positions
        and ``OPENPI_ATTENTION_MASK_VALUE`` on masked ones.
    """
    if lang_masks.ndim != 2:
        raise ValueError(f"lang_masks must be 2D, got shape {lang_masks.shape}")
    bsize, lang_seq_len = lang_masks.shape

    image_tokens_total = prefix_seq_len - lang_seq_len
    if image_tokens_total <= 0 or image_tokens_total % num_cameras != 0:
        raise ValueError(
            f"Cannot derive num_image_tokens: prefix_seq_len={prefix_seq_len}, "
            f"lang_seq_len={lang_seq_len}, num_cameras={num_cameras}"
        )

    img_pad = np.ones((bsize, image_tokens_total), dtype=bool)
    pad_masks = np.concatenate([img_pad, lang_masks.astype(bool, copy=False)], axis=1)
    # att_masks all-zero ⇒ att_2d == pad outer product
    att_2d = pad_masks[:, None, :] & pad_masks[:, :, None]  # (B, S, S)
    out = np.where(
        att_2d[:, None, :, :],
        np.float32(0.0),
        np.float32(OPENPI_ATTENTION_MASK_VALUE),
    ).astype(np.float32, copy=False)
    return out

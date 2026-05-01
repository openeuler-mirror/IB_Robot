#!/usr/bin/env python3
"""Interactive attention mask helpers for ACT inference debugging."""

from __future__ import annotations

import math
import os
import time
from contextlib import suppress
from typing import Any

import cv2
import numpy as np

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:
    torch = None
    Tensor = Any


def tensor_image_to_rgb(image_tensor: Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor or ndarray image to an RGB uint8 image."""
    if torch is not None and torch.is_tensor(image_tensor):
        image = image_tensor.detach().cpu()
        if image.dim() == 4:
            image = image[0]
        if image.dim() != 3:
            raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")
        if image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image_np = image.numpy()
    else:
        image_np = np.asarray(image_tensor)
        if image_np.ndim == 4:
            image_np = image_np[0]
        if image_np.ndim != 3:
            raise ValueError(f"Unsupported image shape: {tuple(image_np.shape)}")
        if image_np.shape[0] in (1, 3) and image_np.shape[-1] not in (1, 3):
            image_np = np.moveaxis(image_np, 0, -1)

    if image_np.dtype != np.uint8:
        max_value = float(np.nanmax(image_np)) if image_np.size else 0.0
        if max_value <= 2.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    return image_np


def create_interactive_masks(
    initial_observation: dict[str, Tensor],
    camera_keys: list[str] | None = None,
    gui_save_dir: str = "gui_interactions",
    line_thickness: int = 15,
) -> dict[str, np.ndarray]:
    """Open a Matplotlib GUI and let the user draw pixel ignore masks."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected_keys = camera_keys or [
        key
        for key in initial_observation
        if str(key).startswith("observation.images.")
    ]
    camera_images: dict[str, np.ndarray] = {}
    for camera_key in selected_keys:
        if camera_key not in initial_observation:
            continue
        camera_images[camera_key] = tensor_image_to_rgb(
            initial_observation[camera_key]
        )

    if not camera_images:
        return {}

    class _MaskDrawer:
        def __init__(self, ax):
            self.ax = ax
            self.mask: np.ndarray | None = None
            self.press: tuple[float, float] | None = None
            self.lines: list[Line2D] = []

        def set_image(self, image: np.ndarray) -> None:
            self.mask = np.zeros(image.shape[:2], dtype=np.uint8)

        def connect(self) -> None:
            canvas = self.ax.figure.canvas
            canvas.mpl_connect("button_press_event", self.on_press)
            canvas.mpl_connect("button_release_event", self.on_release)
            canvas.mpl_connect("motion_notify_event", self.on_motion)

        def on_press(self, event) -> None:
            if event.inaxes != self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            self.press = (event.xdata, event.ydata)

        def on_motion(self, event) -> None:
            if self.press is None or event.inaxes != self.ax or self.mask is None:
                return
            if event.xdata is None or event.ydata is None:
                return
            x, y = event.xdata, event.ydata
            line = Line2D(
                [self.press[0], x],
                [self.press[1], y],
                color="white",
                linewidth=max(2, line_thickness // 2),
                alpha=0.7,
            )
            self.ax.add_line(line)
            self.lines.append(line)
            cv2.line(
                self.mask,
                (int(self.press[0]), int(self.press[1])),
                (int(x), int(y)),
                255,
                thickness=line_thickness,
            )
            self.press = (x, y)
            self.ax.figure.canvas.draw_idle()

        def on_release(self, _event) -> None:
            self.press = None

        def clear(self) -> None:
            for line in self.lines:
                line.remove()
            self.lines.clear()
            if self.mask is not None:
                self.mask[:] = 0
            self.ax.figure.canvas.draw_idle()

    plt.rcParams["toolbar"] = "none"
    num_cameras = len(camera_images)
    max_cols = 3 if num_cameras > 4 else 2
    num_rows = math.ceil(num_cameras / max_cols)
    fig_width = 5 * max_cols
    fig_height = 4.5 * num_rows
    fig, axes = plt.subplots(
        num_rows,
        max_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    fig.suptitle(
        "Interactive Attention Mask\n"
        "Drag: mark ignored region | s/Enter: save | r: reset | q/Esc: skip",
        fontsize=13,
    )

    drawers: dict[str, _MaskDrawer] = {}
    axes_flat = axes.flatten()
    for idx, ax in enumerate(axes_flat):
        if idx >= num_cameras:
            ax.axis("off")
            ax.set_visible(False)
            continue
        camera_key, image = list(camera_images.items())[idx]
        ax.imshow(image)
        ax.set_title(camera_key.replace("observation.images.", ""), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        drawer = _MaskDrawer(ax)
        drawer.set_image(image)
        drawer.connect()
        drawers[camera_key] = drawer

    final_masks: dict[str, np.ndarray] = {}

    def on_key_press(event) -> None:
        nonlocal final_masks
        if event.key in ("s", "enter"):
            os.makedirs(gui_save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(gui_save_dir, f"mask_{timestamp}.png")
            with suppress(Exception):
                fig.savefig(save_path, bbox_inches="tight", dpi=150)
            final_masks = {
                key: drawer.mask
                for key, drawer in drawers.items()
                if drawer.mask is not None and np.any(drawer.mask)
            }
            plt.close(fig)
        elif event.key in ("q", "escape"):
            final_masks = {}
            plt.close(fig)
        elif event.key == "r":
            for drawer in drawers.values():
                drawer.clear()

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()
    return final_masks


def process_pixel_mask_to_feature_mask(
    mask_image: np.ndarray | None,
    feature_map_size: tuple[int, int] | None,
) -> Tensor | None:
    """Convert a pixel-space mask to a flattened feature-token mask."""
    if torch is None:
        raise ModuleNotFoundError("torch is required for attention masking")
    if mask_image is None or feature_map_size is None:
        return None
    h_feat, w_feat = feature_map_size
    mask_resized = cv2.resize(
        mask_image,
        (w_feat, h_feat),
        interpolation=cv2.INTER_NEAREST,
    )
    return torch.from_numpy(mask_resized > 0).flatten()


def build_transformer_attention_masks(
    feature_masks: dict[str, Tensor],
    camera_keys: list[str],
    feature_map_size: tuple[int, int],
    num_non_image_tokens: int,
    chunk_size: int,
    batch_size: int,
    device: torch.device | str,
) -> tuple[Tensor | None, Tensor | None]:
    """Build decoder and encoder masks matching ACT token layout."""
    if torch is None:
        raise ModuleNotFoundError("torch is required for attention masking")
    if not feature_masks:
        return None, None

    feature_map_area = feature_map_size[0] * feature_map_size[1]
    visual_masks: list[Tensor] = []
    for camera_key in camera_keys:
        mask = feature_masks.get(camera_key)
        if mask is None:
            mask = torch.zeros(feature_map_area, dtype=torch.bool)
        elif mask.numel() != feature_map_area:
            raise ValueError(
                f"Feature mask for {camera_key} has {mask.numel()} tokens, "
                f"expected {feature_map_area}"
            )
        visual_masks.append(mask.to(device=device, dtype=torch.bool))

    non_image_mask = torch.zeros(
        max(0, num_non_image_tokens),
        dtype=torch.bool,
        device=device,
    )
    mask_1d = torch.cat([non_image_mask, *visual_masks])
    decoder_attention_mask = mask_1d.unsqueeze(0).expand(chunk_size, -1)
    encoder_key_padding_mask = mask_1d.unsqueeze(0).expand(batch_size, -1)
    return decoder_attention_mask, encoder_key_padding_mask


def create_feature_masks_from_pixel_masks(
    pixel_masks: dict[str, np.ndarray],
    feature_map_size: tuple[int, int],
) -> dict[str, Tensor]:
    """Convert all drawn pixel masks to feature-token masks."""
    feature_masks: dict[str, Tensor] = {}
    for camera_key, pixel_mask in pixel_masks.items():
        feature_mask = process_pixel_mask_to_feature_mask(
            pixel_mask,
            feature_map_size,
        )
        if feature_mask is not None and bool(feature_mask.any()):
            feature_masks[camera_key] = feature_mask
    return feature_masks

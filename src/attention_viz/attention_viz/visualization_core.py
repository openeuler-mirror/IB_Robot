#!/usr/bin/env python3
# ruff: noqa: I001
"""
注意力可视化核心逻辑模块。

从 lerobot_ros2 迁移的可视化功能，无 ROS 依赖，纯 Python 实现。
包含：
- visualize_attention_single: 单图热力图生成
- visualize_attention_maps: 批量文件模式保存热力图
- RealTimeVisualizer: 实时交互 Matplotlib GUI
- RealTimeVisualizerROS: 继承 RealTimeVisualizer，适配 ROS 消息
"""

import math
import os
import traceback

import cv2
import matplotlib
if (
    not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    and matplotlib.get_backend().lower() != "agg"
):
    matplotlib.use("Agg")  # headless fallback for file/offscreen rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.widgets import CheckButtons
from torch import Tensor

from attention_viz.utils import extract_attention_per_camera


# ---------------------------------------------------------------------------
#  单图热力图生成
# ---------------------------------------------------------------------------

def visualize_attention_single(
    original_image: np.ndarray,
    attn_weights_single_head_cam: Tensor,
    feature_map_size: tuple[int, int],
    blend_alpha: float = 0.4,
) -> np.ndarray:
    """
    将单个注意力图叠加到原始图像上，生成热力图。

    Args:
        original_image: BGR 格式的原始图像 (H, W, 3)。
        attn_weights_single_head_cam: 扁平化的注意力权重张量。
        feature_map_size: 特征图 (高度, 宽度)。
        blend_alpha: 热力图透明度。

    Returns:
        叠加了热力图的 BGR 图像。
    """
    h_feat, w_feat = feature_map_size
    attention_map = (
        attn_weights_single_head_cam.reshape(h_feat, w_feat).cpu().numpy()
    )

    h_orig, w_orig = original_image.shape[:2]
    resized_attention_map = cv2.resize(
        attention_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC,
    )
    norm_attention_map = cv2.normalize(
        resized_attention_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,
    )
    heatmap = cv2.applyColorMap(norm_attention_map, cv2.COLORMAP_JET)

    output_image = original_image.astype(np.uint8)
    superimposed_img = cv2.addWeighted(
        heatmap, blend_alpha, output_image, 1 - blend_alpha, 0,
    )
    return superimposed_img


# ---------------------------------------------------------------------------
#  批量文件模式
# ---------------------------------------------------------------------------

def visualize_attention_maps(
    original_images: dict,
    attn_weights: Tensor,
    camera_keys: list[str],
    feature_map_size: tuple[int, int],
    step_counter: int,
    save_dir: str,
    queries_to_visualize: list[int],
    layer_idx: int = -1,
    batch_idx: int = 0,
    average_heads: bool = True,
    blend_alpha: float = 0.4,
    num_non_image_tokens: int | None = None,
):
    """
    批量保存注意力热力图到文件。

    Args:
        original_images: {camera_key: BGR ndarray} 字典。
        attn_weights: 注意力权重张量
            (num_layers, batch, num_heads, num_queries, seq_len)。
        camera_keys: 相机键名列表。
        feature_map_size: 特征图 (高度, 宽度)。
        step_counter: 当前推理步骤。
        save_dir: 保存目录。
        queries_to_visualize: 要可视化的查询索引列表。
        layer_idx: 注意力层索引。
        batch_idx: 批次索引。
        average_heads: 是否平均所有注意力头。
        blend_alpha: 热力图透明度。
        num_non_image_tokens: cross-attention key 中非图像 token 数量；不传时按
            seq_len 和相机特征图尺寸推导。
    """
    print(
        f"\n--- Step {step_counter}: Generating attention heatmaps "
        f"for actions {queries_to_visualize} ---",
    )

    try:
        if not feature_map_size:
            print("[VIZ-WARN] feature_map_size is None, skipping.")
            return

        feature_map_area = feature_map_size[0] * feature_map_size[1]
        num_visual_tokens = len(camera_keys) * feature_map_area
        total_tokens = attn_weights.shape[-1]
        if num_non_image_tokens is None:
            num_non_image_tokens = total_tokens - num_visual_tokens

        if num_non_image_tokens < 0:
            print(
                f"[VIZ-WARN] Negative non_image_tokens ({num_non_image_tokens}). "
                "Skipping.",
            )
            return

        for action_query_idx in queries_to_visualize:
            try:
                per_cam = extract_attention_per_camera(
                    attn_weights,
                    camera_keys,
                    feature_map_size,
                    num_non_image_tokens,
                    query_idx=action_query_idx,
                    batch_idx=batch_idx,
                    layer_idx=layer_idx,
                    average_heads=average_heads,
                )
            except Exception as exc:
                print(
                    f"    - Warning: failed to extract action_query_idx "
                    f"{action_query_idx}: {exc}",
                )
                continue

            for cam_key, weights in per_cam.items():
                if cam_key not in original_images:
                    print(
                        f"    - Warning: missing original image for {cam_key}, "
                        "skipping.",
                    )
                    continue
                heatmap_img = visualize_attention_single(
                    original_images[cam_key],
                    weights,
                    feature_map_size,
                    blend_alpha,
                )

                action_dir = os.path.join(
                    save_dir,
                    cam_key.replace(".", "_"),
                    f"action_{action_query_idx:03d}",
                )
                os.makedirs(action_dir, exist_ok=True)
                output_path = os.path.join(
                    action_dir,
                    f"step_{step_counter:04d}_attn.jpg",
                )
                if not cv2.imwrite(output_path, heatmap_img):
                    print(f"    - Warning: failed to save {output_path}")

            print(f"    - Saved heatmaps for action_query={action_query_idx}")

    except Exception:
        print(
            f"\n--- Step {step_counter}: Unexpected error during visualization ---",
        )
        traceback.print_exc()


# ---------------------------------------------------------------------------
#  实时交互可视化
# ---------------------------------------------------------------------------

class RealTimeVisualizer:
    """
    实时注意力可视化 Matplotlib GUI。

    支持 CheckButtons 切换不同 action query，
    实时更新热力图叠加到相机图像上。
    """

    def __init__(
        self,
        camera_keys: list[str],
        queries_to_visualize: list[int],
        layer_idx: int,
        batch_idx: int,
        average_heads: bool,
        blend_alpha: float,
    ):
        self.camera_keys = camera_keys
        self.queries_to_visualize = queries_to_visualize
        self.layer_idx = layer_idx
        self.batch_idx = batch_idx
        self.average_heads = average_heads
        self.blend_alpha = blend_alpha

        self.fig = None
        self.axes_map: dict = {}
        self.image_artists: dict = {}
        self.check_buttons = None
        self.labels_text: list[str] = []

    def _setup_gui(self, initial_batch: dict):
        """创建并显示 GUI 窗口（仅在第一次 update 时调用）。"""
        print("--- [Visualizer] Creating persistent GUI window... ---")

        plt.rcParams["toolbar"] = "none"
        num_cameras = len(self.camera_keys)
        max_cols = 2
        num_rows = math.ceil(num_cameras / max_cols)
        num_cols = min(num_cameras, max_cols)
        fig_width = 5 * num_cols + 2

        self.fig, axes_flat = plt.subplots(
            num_rows, num_cols,
            figsize=(fig_width, 5.5 * num_rows),
            squeeze=False,
        )
        axes_flat = axes_flat.flatten()
        self.fig.suptitle("Real-time Attention Visualizer", fontsize=16)

        for i, cam_key in enumerate(self.camera_keys):
            ax = axes_flat[i]
            img_tensor = initial_batch[cam_key][0].cpu()
            img_np_rgb = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(
                np.uint8,
            )
            self.image_artists[cam_key] = ax.imshow(img_np_rgb)
            ax.set_title(cam_key, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            self.axes_map[cam_key] = ax

        for i in range(num_cameras, len(axes_flat)):
            axes_flat[i].axis("off")

        # CheckButtons
        self.labels_text = [f"Action {q}" for q in self.queries_to_visualize]
        actives = [False] * len(self.labels_text)

        rax = self.fig.add_axes([0.9, 0.4, 0.08, 0.3])
        self.check_buttons = CheckButtons(rax, self.labels_text, actives)

        for label in self.check_buttons.labels:
            label.set_fontsize(10)

        rects = [
            p for p in self.check_buttons.ax.patches
            if isinstance(p, Rectangle)
        ]
        if len(rects) == len(self.labels_text):
            for i, rect in enumerate(rects):
                rect.set_edgecolor("black")
                rect.set_linewidth(1.5)
                if not actives[i]:
                    rect.set_facecolor("lightgray")

        self.check_buttons.on_clicked(self._on_check_clicked)

        plt.tight_layout(rect=[0, 0, 0.88, 0.95])
        plt.show(block=False)

        print("--- [Visualizer] GUI initialized. Waiting 2s for rendering... ---")
        plt.pause(2.0)
        print("--- [Visualizer] Ready. ---")

    def _on_check_clicked(self, label):
        """单选按钮逻辑，允许取消所有选择（显示原图）。"""
        if not self.check_buttons:
            return

        self.check_buttons.eventson = False
        try:
            idx_clicked = self.labels_text.index(label)
            current_status = self.check_buttons.get_status()
            is_checked_now = current_status[idx_clicked]

            if is_checked_now:
                for i, status in enumerate(current_status):
                    if i != idx_clicked and status:
                        self.check_buttons.set_active(i)
            # 取消勾选时不做额外操作

            rects = [
                p for p in self.check_buttons.ax.patches
                if isinstance(p, Rectangle)
            ]
            new_status = self.check_buttons.get_status()
            if len(rects) == len(new_status):
                for i, rect in enumerate(rects):
                    rect.set_facecolor(
                        "white" if new_status[i] else "lightgray",
                    )
        except Exception as e:
            print(f"[Visualizer Error] Button click failed: {e}")
        finally:
            self.check_buttons.eventson = True

    @torch.no_grad()
    def update(
        self,
        attn_weights: Tensor,
        batch: dict,
        feature_map_size: tuple[int, int],
        num_non_image_tokens: int,
    ):
        """用新的注意力权重更新可视化窗口。"""
        if self.fig is None:
            self._setup_gui(batch)

        # 获取激活的 query
        active_queries = []
        if self.check_buttons:
            for i, status in enumerate(self.check_buttons.get_status()):
                if status:
                    active_queries.append(self.queries_to_visualize[i])

        # 准备图像
        current_images_bgr: dict = {}
        for cam_key in self.camera_keys:
            img_tensor = batch[cam_key][0].cpu()
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(
                np.uint8,
            )
            current_images_bgr[cam_key] = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        final_display_images: dict = {}

        if active_queries:
            preprocessed_attn = self._preprocess_attn(
                attn_weights,
                feature_map_size,
                num_non_image_tokens,
            )
            for cam_key in self.camera_keys:
                if (active_queries[0], cam_key) in preprocessed_attn:
                    combined_attn = torch.zeros_like(
                        preprocessed_attn[(active_queries[0], cam_key)],
                    )
                    for query_idx in active_queries:
                        if (query_idx, cam_key) in preprocessed_attn:
                            attn_map = preprocessed_attn[(query_idx, cam_key)]
                            combined_attn = torch.max(combined_attn, attn_map)

                    heatmap_img_bgr = self._visualize_single(
                        current_images_bgr[cam_key],
                        combined_attn,
                        feature_map_size,
                    )
                    final_display_images[cam_key] = cv2.cvtColor(
                        heatmap_img_bgr, cv2.COLOR_BGR2RGB,
                    )
                else:
                    final_display_images[cam_key] = cv2.cvtColor(
                        current_images_bgr[cam_key], cv2.COLOR_BGR2RGB,
                    )
        else:
            for cam_key in self.camera_keys:
                final_display_images[cam_key] = cv2.cvtColor(
                    current_images_bgr[cam_key], cv2.COLOR_BGR2RGB,
                )

        for cam_key in self.camera_keys:
            if cam_key in self.image_artists and cam_key in final_display_images:
                self.image_artists[cam_key].set_data(final_display_images[cam_key])

        plt.pause(0.001)

    def _preprocess_attn(self, attn_weights, feature_map_size, num_non_image_tokens):
        preprocessed_attn: dict = {}
        for query_idx in self.queries_to_visualize:
            try:
                per_cam = extract_attention_per_camera(
                    attn_weights,
                    self.camera_keys,
                    feature_map_size,
                    num_non_image_tokens,
                    query_idx=query_idx,
                    batch_idx=self.batch_idx,
                    layer_idx=self.layer_idx,
                    average_heads=self.average_heads,
                )
            except Exception:
                continue
            for cam_key, weights in per_cam.items():
                preprocessed_attn[(query_idx, cam_key)] = weights
        return preprocessed_attn

    def _visualize_single(self, original_image, attn_weights_flat, feature_map_size):
        h_feat, w_feat = feature_map_size
        attention_map = attn_weights_flat.reshape(h_feat, w_feat).cpu().numpy()
        h_orig, w_orig = original_image.shape[:2]
        resized_map = cv2.resize(
            attention_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC,
        )
        norm_map = cv2.normalize(
            resized_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,
        )
        heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        return cv2.addWeighted(
            heatmap,
            self.blend_alpha,
            original_image.astype(np.uint8),
            1 - self.blend_alpha,
            0,
        )

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes_map = {}
            self.image_artists = {}
            print("--- [Visualizer] GUI closed. ---")


# ---------------------------------------------------------------------------
#  ROS 适配子类
# ---------------------------------------------------------------------------

class RealTimeVisualizerROS(RealTimeVisualizer):
    """
    继承 RealTimeVisualizer，适配 ROS 消息格式。

    通过 update_from_ros_msg() 接收 AttentionWeights 消息和相机图像，
    转换为张量后调用父类 update() 方法。
    """

    def __init__(
        self,
        camera_keys: list[str],
        queries_to_visualize: list[int],
        layer_idx: int = -1,
        batch_idx: int = 0,
        average_heads: bool = True,
        blend_alpha: float = 0.4,
    ):
        super().__init__(
            camera_keys=camera_keys,
            queries_to_visualize=queries_to_visualize,
            layer_idx=layer_idx,
            batch_idx=batch_idx,
            average_heads=average_heads,
            blend_alpha=blend_alpha,
        )

    def update_from_ros_msg(
        self,
        attn_weights_tensor: Tensor,
        batch: dict,
        feature_map_size: tuple[int, int],
        num_non_image_tokens: int,
    ):
        """
        从已转换的注意力权重张量和图像 batch 更新可视化。

        Args:
            attn_weights_tensor: 注意力权重张量
                (num_layers, batch, heads, queries, seq_len)。
            batch: 图像 batch 字典，key 为 camera_key，
                value 为 (1, C, H, W) tensor。
            feature_map_size: 特征图 (高度, 宽度)。
            num_non_image_tokens: cross-attention key 中非图像 token 数量。
        """
        self.update(
            attn_weights_tensor,
            batch,
            feature_map_size,
            num_non_image_tokens,
        )

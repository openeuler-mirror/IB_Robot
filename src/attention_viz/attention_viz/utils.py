#!/usr/bin/env python3
"""
消息转换工具函数。

提供 AttentionWeights 消息与张量之间的双向转换。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor
except ModuleNotFoundError:
    torch = None
    Tensor = Any


def normalize_visualization_mode(mode: Any, default: str = "file") -> str:
    """Return the canonical visualization mode used by the ROS node."""
    normalized = str(mode or default).strip().lower()
    if normalized == "interactive":
        return "realtime"
    if normalized in {"file", "realtime"}:
        return normalized
    return default


def default_camera_topic_from_key(camera_key: str) -> str:
    """将 observation.images.<name> 转换为 IB-Robot 的默认图像话题。"""
    prefix = "observation.images."
    if camera_key.startswith(prefix):
        return f"/camera/{camera_key[len(prefix):]}/image_raw"
    return camera_key


def build_camera_topic_map(
    camera_keys: list[str],
    message_camera_topics: list[str] | None = None,
    configured_camera_topics: list[str] | None = None,
) -> dict[str, str]:
    """组合消息内元数据和参数覆写，生成 camera_key -> topic 映射。"""
    explicit_overrides: dict[str, str] = {}
    ordered_topics: list[str] = []

    for entry in configured_camera_topics or []:
        if not entry:
            continue
        if ":=" in entry:
            key, topic = entry.split(":=", 1)
            explicit_overrides[key.strip()] = topic.strip()
        else:
            ordered_topics.append(entry.strip())

    topic_map: dict[str, str] = {}
    ordered_idx = 0
    for idx, cam_key in enumerate(camera_keys):
        topic = explicit_overrides.get(cam_key, "")
        if not topic and message_camera_topics and idx < len(message_camera_topics):
            topic = str(message_camera_topics[idx]).strip()
        if not topic and ordered_idx < len(ordered_topics):
            topic = ordered_topics[ordered_idx]
            ordered_idx += 1
        if not topic:
            topic = default_camera_topic_from_key(cam_key)
        topic_map[cam_key] = topic

    return topic_map


def attention_data_to_msg(
    attn_data: dict[str, Any],
    stamp_sec: int = 0,
    stamp_nanosec: int = 0,
) -> Any:
    """
    将注意力数据字典转换为 AttentionWeights ROS 消息。

    Args:
        attn_data: get_latest() 返回的字典，包含:
            - attn_weights: Tensor
            - feature_map_size: (h, w)
            - camera_keys: list[str]
            - num_non_image_tokens: int
        stamp_sec: 时间戳秒
        stamp_nanosec: 时间戳纳秒

    Returns:
        AttentionWeights 消息实例。
    """
    from ibrobot_msgs.msg import AttentionWeights

    msg = AttentionWeights()
    msg.header.stamp.sec = stamp_sec
    msg.header.stamp.nanosec = stamp_nanosec

    weights = attn_data.get("attn_weights")
    if weights is not None:
        if torch is not None and torch.is_tensor(weights):
            msg.tensor_shape = [int(v) for v in weights.shape]
            msg.attention_weights = weights.flatten().tolist()
        elif isinstance(weights, (list, np.ndarray)):
            weights_array = np.asarray(weights)
            flat = weights_array.flatten()
            msg.tensor_shape = [int(v) for v in weights_array.shape]
            msg.attention_weights = flat.tolist()

    fms = attn_data.get("feature_map_size")
    if fms is not None:
        msg.feature_map_height = int(fms[0])
        msg.feature_map_width = int(fms[1])
    else:
        msg.feature_map_height = 0
        msg.feature_map_width = 0

    msg.camera_keys = attn_data.get("camera_keys", [])
    msg.camera_topics = attn_data.get("camera_topics", [])
    msg.num_non_image_tokens = attn_data.get("num_non_image_tokens", 0)
    msg.average_heads = bool(attn_data.get("average_heads", True))
    msg.blend_alpha = float(attn_data.get("blend_alpha", 0.4))
    if msg.tensor_shape and len(msg.tensor_shape) >= 3:
        msg.num_heads = int(msg.tensor_shape[-3])

    return msg


def msg_to_attention_data(msg: Any) -> dict[str, Any]:
    """
    将 AttentionWeights ROS 消息转换回注意力数据字典。

    Args:
        msg: AttentionWeights 消息实例。

    Returns:
        字典包含:
            - attention_weights: Tensor
            - feature_map_size: (h, w)
            - camera_keys: list[str]
            - num_non_image_tokens: int
            - average_heads: bool
            - blend_alpha: float
    """
    if torch is None:
        raise ModuleNotFoundError("torch is required for attention message deserialization")

    weights_tensor = torch.tensor(msg.attention_weights, dtype=torch.float32)
    tensor_shape = [int(v) for v in msg.tensor_shape if int(v) > 0]
    if tensor_shape and int(np.prod(tensor_shape)) == weights_tensor.numel():
        weights_tensor = weights_tensor.reshape(tensor_shape)

    feature_map_size = None
    if msg.feature_map_height > 0 and msg.feature_map_width > 0:
        feature_map_size = (msg.feature_map_height, msg.feature_map_width)

    return {
        "attention_weights": weights_tensor,
        "feature_map_size": feature_map_size,
        "camera_keys": list(msg.camera_keys),
        "camera_topics": list(msg.camera_topics),
        "num_non_image_tokens": msg.num_non_image_tokens,
        "average_heads": msg.average_heads,
        "blend_alpha": msg.blend_alpha,
    }


def extract_attention_per_camera(
    attn_weights: Tensor,
    camera_keys: list[str],
    feature_map_size: tuple[int, int],
    num_non_image_tokens: int,
    query_idx: int = 0,
    batch_idx: int = 0,
    layer_idx: int = -1,
    average_heads: bool = True,
) -> dict[str, Tensor]:
    """
    从完整注意力权重中提取每个相机的注意力图。

    Args:
        attn_weights: 注意力权重 (num_layers, batch, num_heads, query_len, key_len)、
                      (num_layers, num_heads, query_len, key_len) 或
                      (num_heads, query_len, key_len)。
        camera_keys: 相机键名列表。
        feature_map_size: 特征图 (高度, 宽度)。
        num_non_image_tokens: 非图像 token 数量。
        query_idx: 要提取的 query 索引。
        layer_idx: 要提取的层索引。
        average_heads: 是否平均注意力头。

    Returns:
        {camera_key: 1D Tensor} 字典。
    """
    if torch is None:
        raise ModuleNotFoundError("torch is required for attention tensor extraction")

    if attn_weights.dim() == 5:
        attn = attn_weights[layer_idx, batch_idx]
    elif attn_weights.dim() == 4:
        attn = attn_weights[layer_idx]
    elif attn_weights.dim() == 3:
        attn = attn_weights
    else:
        raise ValueError(
            f"Unsupported attention tensor shape: {tuple(attn_weights.shape)}"
        )

    feature_map_area = feature_map_size[0] * feature_map_size[1]
    visual_tokens = attn[:, query_idx, num_non_image_tokens:]

    if average_heads:
        weights = visual_tokens.mean(dim=0)
    else:
        weights = visual_tokens[0]

    result: dict[str, Tensor] = {}
    for i, cam_key in enumerate(camera_keys):
        start = i * feature_map_area
        end = start + feature_map_area
        if end <= weights.shape[0]:
            result[cam_key] = weights[start:end]

    return result

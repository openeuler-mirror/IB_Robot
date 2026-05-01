#!/usr/bin/env python3
"""
AttentionVisualizationNode —— 注意力可视化 ROS 节点。

订阅:
  - /attention/weights (AttentionWeights): 注意力权重数据
  - /camera/{name}/image_raw (sensor_msgs/Image): 原始相机图像

参数:
  - attention_topic: 注意力权重话题
  - visualization_mode: 'interactive' | 'realtime' | 'file'
  - save_dir: 文件模式保存目录
  - queries_to_visualize: 要可视化的 query 索引列表
  - layer_idx: 注意力层索引
  - batch_idx: 批次索引
  - average_heads: 是否平均注意力头
  - blend_alpha: 热力图透明度
  - update_frequency: 最大更新频率 (Hz)
  - heatmap_topic_prefix: 热力图输出话题前缀

发布:
  - /visualization/heatmap/{camera_key} (sensor_msgs/Image): 热力图叠加结果
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from typing import Any

import cv2
import numpy as np
import rclpy
import torch
from rclpy._rclpy_pybind11 import RCLError
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from attention_viz.utils import (
    build_camera_topic_map,
    extract_attention_per_camera,
    msg_to_attention_data,
    normalize_visualization_mode,
)
from attention_viz.visualization_core import (
    RealTimeVisualizerROS,
    visualize_attention_single,
)
from ibrobot_msgs.msg import AttentionWeights


class AttentionVisualizationNode(Node):
    """注意力可视化 ROS 节点。"""

    def __init__(self):
        super().__init__("attention_visualization_node")

        # 声明参数
        self.declare_parameter("attention_topic", "/attention/weights")
        self.declare_parameter("visualization_mode", "file")
        self.declare_parameter("save_dir", "attention_visualizations")
        self.declare_parameter("queries_to_visualize", [0, 20, 40, 60, 80])
        self.declare_parameter("layer_idx", -1)
        self.declare_parameter("batch_idx", 0)
        self.declare_parameter("average_heads", True)
        self.declare_parameter("blend_alpha", 0.4)
        self.declare_parameter("update_frequency", 10.0)
        self.declare_parameter("camera_topics", [""])
        self.declare_parameter("headless", False)
        self.declare_parameter("heatmap_topic_prefix", "/visualization/heatmap")

        # 读取参数
        self._attention_topic = self.get_parameter("attention_topic").value
        raw_mode = self.get_parameter("visualization_mode").value
        self._mode = normalize_visualization_mode(raw_mode)
        raw_mode_normalized = str(raw_mode or "").strip().lower()
        if self._mode == "file" and raw_mode_normalized not in {"", "file"}:
            self.get_logger().warn(
                f"Unsupported visualization_mode={raw_mode!r}; using file mode"
            )
        self._save_dir = self.get_parameter("save_dir").value
        self._queries = self.get_parameter("queries_to_visualize").value
        self._layer_idx = self.get_parameter("layer_idx").value
        self._batch_idx = self.get_parameter("batch_idx").value
        self._average_heads = self.get_parameter("average_heads").value
        self._blend_alpha = self.get_parameter("blend_alpha").value
        self._update_freq = self.get_parameter("update_frequency").value
        self._headless = self.get_parameter("headless").value
        self._heatmap_topic_prefix = (
            str(self.get_parameter("heatmap_topic_prefix").value).rstrip("/")
            or "/visualization/heatmap"
        )
        camera_topic_param = self.get_parameter("camera_topics").value
        if isinstance(camera_topic_param, str):
            camera_topic_overrides = [camera_topic_param]
        else:
            camera_topic_overrides = list(camera_topic_param)
        self._camera_topic_overrides = [
            entry for entry in camera_topic_overrides if entry
        ]

        # 状态
        self._step_counter = 0
        self._last_update_time: float = 0.0
        self._min_interval = 1.0 / max(self._update_freq, 0.1)
        self._image_cache: dict[str, np.ndarray] = {}
        self._image_lock = threading.Lock()
        self._camera_keys: list[str] = []
        self._camera_topic_map: dict[str, str] = {}
        self._feature_map_size: tuple | None = None
        self._num_non_image_tokens: int = 0
        self._first_attention_logged = False
        self._first_heatmap_logged = False
        self._last_waiting_log_time = 0.0
        self._cv_bridge = None

        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if self._mode == "realtime" and (self._headless or not has_display):
            reason = (
                "headless=true"
                if self._headless
                else "no graphical display is available"
            )
            self.get_logger().warn(
                f"Realtime visualization requested but {reason}; using file mode"
            )
            self._mode = "file"

        if self._mode == "file":
            os.makedirs(self._save_dir, exist_ok=True)

        # 实时可视化器 (延迟初始化)
        self._realtime_viz: RealTimeVisualizerROS | None = None

        # 订阅注意力权重
        self._attn_sub = self.create_subscription(
            AttentionWeights,
            self._attention_topic,
            self._attention_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        # 发布者字典 (动态创建)
        self._heatmap_pubs: dict[str, Any] = {}

        # 相机图像订阅 (动态创建，在收到第一条注意力消息时确定相机)
        self._camera_subs: dict[str, Any] = {}

        self.get_logger().info(
            f"AttentionVisualizationNode started: mode={self._mode}, "
            f"attention_topic={self._attention_topic}, "
            f"heatmap_topic_prefix={self._heatmap_topic_prefix}, "
            f"save_dir={os.path.abspath(self._save_dir)}, "
            f"queries={self._queries}, headless={self._headless}"
        )
        self.get_logger().info(
            f"Waiting for AttentionWeights on {self._attention_topic}"
        )

    def _attention_callback(self, msg: AttentionWeights):
        """处理注意力权重消息。"""
        now = time.time()
        if now - self._last_update_time < self._min_interval:
            return

        self._step_counter += 1

        # 解析消息
        try:
            data = msg_to_attention_data(msg)
        except Exception as e:
            self.get_logger().warn(f"Invalid AttentionWeights message: {e}")
            return
        attn_weights = data["attention_weights"]
        feature_map_size = data["feature_map_size"]
        camera_keys = data["camera_keys"]
        camera_topics = data.get("camera_topics", [])
        num_non_image_tokens = data["num_non_image_tokens"]

        if not camera_keys or feature_map_size is None:
            self.get_logger().warn(
                "AttentionWeights missing camera_keys or feature_map_size; "
                "cannot render heatmaps"
            )
            return

        self._last_update_time = now

        if not self._first_attention_logged:
            self.get_logger().info(
                "First AttentionWeights received: "
                f"shape={tuple(attn_weights.shape)}, "
                f"cameras={camera_keys}, feature_map={feature_map_size}"
            )
            self._first_attention_logged = True

        # 首次收到消息时设置相机订阅
        if not self._camera_keys:
            self._setup_camera_subscriptions(camera_keys, camera_topics)
            self._camera_keys = camera_keys
            self._feature_map_size = feature_map_size
            self._num_non_image_tokens = num_non_image_tokens

        # 模式分发
        if self._mode == "file":
            self._handle_file_mode(
                attn_weights,
                camera_keys,
                feature_map_size,
                num_non_image_tokens,
            )
        elif self._mode == "realtime":
            self._handle_realtime_mode(
                attn_weights,
                camera_keys,
                feature_map_size,
                num_non_image_tokens,
            )

    def _setup_camera_subscriptions(
        self,
        camera_keys: list[str],
        camera_topics: list[str] | None = None,
    ):
        """根据注意力消息中的相机键名创建图像订阅。"""
        self._camera_topic_map = build_camera_topic_map(
            camera_keys,
            message_camera_topics=camera_topics,
            configured_camera_topics=self._camera_topic_overrides,
        )
        for cam_key, topic_name in self._camera_topic_map.items():
            sub = self.create_subscription(
                Image,
                topic_name,
                lambda msg, key=cam_key: self._image_callback(msg, key),
                qos_profile_sensor_data,
                callback_group=ReentrantCallbackGroup(),
            )
            self._camera_subs[cam_key] = sub
            self.get_logger().info(
                f"Subscribed to camera: {topic_name} (key={cam_key})"
            )
        self.get_logger().info(
            "Waiting for camera images before heatmaps can be generated"
        )

    def _image_callback(self, msg: Image, cam_key: str):
        """缓存最新相机图像。"""
        try:
            cv_img = self._get_cv_bridge().imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            with self._image_lock:
                self._image_cache[cam_key] = cv_img
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image for {cam_key}: {e}")

    def _get_cv_bridge(self):
        if self._cv_bridge is None:
            from cv_bridge import CvBridge

            self._cv_bridge = CvBridge()
        return self._cv_bridge

    def _get_publisher(self, cam_key: str):
        """获取或创建热力图发布者。"""
        if cam_key not in self._heatmap_pubs:
            safe_key = self._safe_topic_suffix(cam_key)
            pub = self.create_publisher(
                Image,
                f"{self._heatmap_topic_prefix}/{safe_key}",
                10,
            )
            self._heatmap_pubs[cam_key] = pub
        return self._heatmap_pubs[cam_key]

    @staticmethod
    def _safe_topic_suffix(camera_key: str) -> str:
        suffix = "".join(
            ch if ch.isalnum() or ch == "_" else "_"
            for ch in camera_key.replace(".", "_")
        ).strip("_")
        return suffix or "camera"

    def _handle_file_mode(
        self,
        attn_weights,
        camera_keys,
        feature_map_size,
        num_non_image_tokens,
    ):
        """文件模式：保存热力图到磁盘。"""
        with self._image_lock:
            if not self._image_cache:
                self._log_waiting_for_images()
                return
            images = dict(self._image_cache)

        missing = [key for key in camera_keys if key not in images]
        if missing:
            self._log_waiting_for_images(missing)
            return

        saved_count = 0
        first_path = ""
        for query_idx in self._queries:
            try:
                per_cam = extract_attention_per_camera(
                    attn_weights,
                    camera_keys,
                    feature_map_size,
                    num_non_image_tokens,
                    query_idx=query_idx,
                    batch_idx=self._batch_idx,
                    layer_idx=self._layer_idx,
                    average_heads=self._average_heads,
                )
            except Exception as e:
                self.get_logger().warn(
                    f"Failed to extract attention for query {query_idx}: {e}"
                )
                continue
            for cam_key, cam_attn in per_cam.items():
                if cam_key not in images:
                    continue
                heatmap = visualize_attention_single(
                    images[cam_key],
                    cam_attn,
                    feature_map_size,
                    self._blend_alpha,
                )
                # 保存到文件
                action_dir = os.path.join(
                    self._save_dir,
                    cam_key.replace(".", "_"),
                    f"action_{query_idx:03d}",
                )
                os.makedirs(action_dir, exist_ok=True)
                filepath = os.path.join(
                    action_dir,
                    f"step_{self._step_counter:04d}_attn.jpg",
                )
                if not cv2.imwrite(filepath, heatmap):
                    self.get_logger().warn(f"Failed to save heatmap: {filepath}")
                    continue
                saved_count += 1
                if not first_path:
                    first_path = filepath

                # 同时发布到 ROS 话题
                self._publish_heatmap(heatmap, cam_key)

        if saved_count == 0:
            self._log_waiting_for_images()
            return

        if not self._first_heatmap_logged:
            self.get_logger().info(
                f"First heatmap saved: {os.path.abspath(first_path)}"
            )
            self._first_heatmap_logged = True
        self.get_logger().info(
            f"Step {self._step_counter}: saved {saved_count} heatmaps under "
            f"{os.path.abspath(self._save_dir)}"
        )

    def _handle_realtime_mode(
        self,
        attn_weights,
        camera_keys,
        feature_map_size,
        num_non_image_tokens,
    ):
        """实时模式：使用 RealTimeVisualizerROS 交互显示。"""
        if self._headless:
            # headless 模式下退化为文件保存 + ROS 发布
            self._handle_file_mode(
                attn_weights,
                camera_keys,
                feature_map_size,
                num_non_image_tokens,
            )
            return

        # 构造 batch 格式 (和 lerobot 的 batch 格式兼容)
        with self._image_lock:
            if not self._image_cache:
                self._log_waiting_for_images()
                return
            images = dict(self._image_cache)

        missing = [key for key in camera_keys if key not in images]
        if missing:
            self._log_waiting_for_images(missing)
            return

        batch = {}
        for cam_key in camera_keys:
            img_bgr = images[cam_key]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(img_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            batch[cam_key] = tensor

        if self._realtime_viz is None:
            self._realtime_viz = RealTimeVisualizerROS(
                camera_keys=camera_keys,
                queries_to_visualize=self._queries,
                layer_idx=self._layer_idx,
                batch_idx=self._batch_idx,
                average_heads=self._average_heads,
                blend_alpha=self._blend_alpha,
            )

        self._realtime_viz.update_from_ros_msg(
            attn_weights,
            batch,
            feature_map_size,
            num_non_image_tokens,
        )

    def _publish_heatmap(self, heatmap_bgr: np.ndarray, cam_key: str):
        """发布热力图为 ROS Image 消息。"""
        try:
            img_msg = self._get_cv_bridge().cv2_to_imgmsg(
                heatmap_bgr,
                encoding="bgr8",
            )
            img_msg.header.stamp = self.get_clock().now().to_msg()
            pub = self._get_publisher(cam_key)
            pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish heatmap for {cam_key}: {e}")

    def _log_waiting_for_images(self, missing_camera_keys: list[str] | None = None):
        now = time.time()
        if now - self._last_waiting_log_time < 5.0:
            return
        self._last_waiting_log_time = now
        if missing_camera_keys:
            topics = [
                self._camera_topic_map.get(key, key)
                for key in missing_camera_keys
            ]
        else:
            topics = list(self._camera_topic_map.values())
        expected = (
            ", ".join(topics)
            or "(camera topics not known yet)"
        )
        self.get_logger().warn(
            "Attention received, but required camera images are not cached yet. "
            f"Expected image topics: {expected}"
        )

    def destroy_node(self):
        if self._realtime_viz is not None:
            self._realtime_viz.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = AttentionVisualizationNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        executor.shutdown()
        with suppress(Exception):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

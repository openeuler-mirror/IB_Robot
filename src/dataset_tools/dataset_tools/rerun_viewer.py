#!/usr/bin/python3
"""
Rerun visualization sidecar for IB-Robot recording pipeline.

Subscribes to all contract-defined topics (cameras, joint states, actions)
and logs them to a Rerun viewer in real-time.  Designed to run alongside
``episode_recorder`` as an **optional** visualization companion — it never
touches the recording itself.

The set of topics is derived from the same ``robot_config`` contract used by
the recorder, so the two nodes always stay in sync.

Parameters
----------
robot_config_path : str (required)
    Path to the robot_config YAML (Single Source of Truth).
rerun_app_name : str, default "IB-Robot Recording"
    Application title shown in the Rerun viewer window.
rerun_mode : str, default "spawn"
    How to connect to the viewer:
    - ``spawn``  — launch a local viewer process (default).
    - ``connect`` — connect to a running viewer via gRPC (use ``rerun_addr``).
    - ``save``   — write an ``.rrd`` file to ``rerun_save_path``.
rerun_addr : str, default "rerun+http://127.0.0.1:9090/proxy"
    gRPC address when ``rerun_mode`` is ``connect``.
rerun_save_path : str, default "/tmp/ib_recording.rrd"
    Output path when ``rerun_mode`` is ``save``.
image_max_fps : float, default 5.0
    Per-image-stream logging cap inside the sidecar. This reduces viewer memory
    growth without affecting rosbag recording.
image_max_long_edge : int, default 320
    Downscale large images before logging when their longest edge exceeds this
    value. Set to ``0`` to disable downscaling.

Usage
-----
Launched automatically when ``record_visualizer:=rerun`` is passed to the
main launch file, or manually::

    ros2 run dataset_tools rerun_viewer --ros-args \\
        -p robot_config_path:=/path/to/so101_single_arm.yaml
"""

from __future__ import annotations

import os
import signal
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rosidl_runtime_py.utilities import get_message

from robot_config.contract_utils import (
    Contract,
    ObservationSpec,
    qos_profile_from_dict,
)
from tensormsg import TensorMsgConverter

DEFAULT_QOS_DEPTH = 10
DEFAULT_IMAGE_MAX_FPS = 5.0
DEFAULT_IMAGE_MAX_LONG_EDGE = 320

# --------------- Color palette for joint/action curves ---------------
_COLORS = [
    (0x1F, 0x77, 0xB4, 0xFF),  # blue
    (0xFF, 0x7F, 0x0E, 0xFF),  # orange
    (0x2C, 0xA0, 0x2C, 0xFF),  # green
    (0xD6, 0x27, 0x28, 0xFF),  # red
    (0x94, 0x67, 0xBD, 0xFF),  # purple
    (0x8C, 0x56, 0x4B, 0xFF),  # brown
    (0xE3, 0x77, 0xC2, 0xFF),  # pink
    (0x7F, 0x7F, 0x7F, 0xFF),  # gray
    (0xBC, 0xBD, 0x22, 0xFF),  # olive
    (0x17, 0xBE, 0xCF, 0xFF),  # cyan
]


def _color_for(idx: int) -> tuple[int, int, int, int]:
    return _COLORS[idx % len(_COLORS)]


def _default_rerun_memory_limit() -> str:
    """Match LeRobot's default viewer memory cap and allow env override."""
    return str(os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%") or "10%")


def _image_msg_to_numpy(msg: Any) -> np.ndarray | None:
    """Decode common ``sensor_msgs/Image`` encodings into numpy arrays."""
    encoding = str(getattr(msg, "encoding", "") or "").lower()
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    step = int(getattr(msg, "step", 0) or 0)
    data = getattr(msg, "data", None)

    if width <= 0 or height <= 0 or step <= 0 or data is None:
        return None

    encoding_spec: dict[str, tuple[type[np.generic], int, str | None]] = {
        "rgb8": (np.uint8, 3, None),
        "bgr8": (np.uint8, 3, "bgr"),
        "rgba8": (np.uint8, 4, None),
        "bgra8": (np.uint8, 4, "bgra"),
        "mono8": (np.uint8, 1, None),
        "8uc1": (np.uint8, 1, None),
        "16uc1": (np.uint16, 1, None),
        "mono16": (np.uint16, 1, None),
        "32fc1": (np.float32, 1, None),
    }
    spec = encoding_spec.get(encoding)
    if spec is None:
        return None

    dtype, channels, reorder = spec
    bytes_per_channel = np.dtype(dtype).itemsize
    expected_row_bytes = width * channels * bytes_per_channel
    if step < expected_row_bytes:
        return None

    raw = memoryview(data)
    expected_total_bytes = step * height
    if len(raw) < expected_total_bytes:
        return None

    rows = np.frombuffer(raw[:expected_total_bytes], dtype=np.uint8).reshape(height, step)
    pixel_bytes = rows[:, :expected_row_bytes]
    arr = pixel_bytes.view(dtype)

    if channels == 1:
        arr = arr.reshape(height, width)
    else:
        arr = arr.reshape(height, width, channels)

    if reorder == "bgr":
        arr = arr[..., ::-1]
    elif reorder == "bgra":
        arr = arr[..., [2, 1, 0, 3]]

    return np.ascontiguousarray(arr)


def _decode_image_for_rerun(msg: Any, spec: Any = None) -> np.ndarray | None:
    """Decode images via fast-path raw decoding, then tensormsg as fallback."""
    arr = _image_msg_to_numpy(msg)
    if arr is None:
        try:
            arr = np.asarray(TensorMsgConverter.decode(msg, spec=spec))
        except (ValueError, TypeError):
            return None

    if arr.size == 0:
        return None

    if np.issubdtype(arr.dtype, np.floating):
        # TensorMsgConverter normalizes floating image outputs to [0, 1]
        # (including preview-friendly depth decodes), so Rerun only needs
        # the final uint8 conversion here.
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return np.ascontiguousarray(arr)


def _downscale_image_for_rerun(arr: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Downscale large images before shipping them to Rerun."""
    if max_long_edge <= 0 or arr.ndim < 2:
        return arr

    height, width = arr.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return arr

    scale = max_long_edge / float(long_edge)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    y_idx = np.linspace(0, height - 1, new_height, dtype=np.int32)
    x_idx = np.linspace(0, width - 1, new_width, dtype=np.int32)
    reduced = np.take(arr, y_idx, axis=0)
    reduced = np.take(reduced, x_idx, axis=1)
    return np.ascontiguousarray(reduced)


def _should_log_sample(
    last_timestamp_s: float | None,
    current_timestamp_s: float,
    min_interval_s: float,
) -> bool:
    """Return whether a timestamp is outside the current throttle window."""
    if min_interval_s <= 0.0 or last_timestamp_s is None:
        return True
    return current_timestamp_s >= (last_timestamp_s + min_interval_s)


# --------------- Main Node ---------------


class RerunViewer(Node):
    """Contract-driven Rerun visualization sidecar.

    Reads the same ``robot_config`` contract as ``episode_recorder`` and
    creates subscriptions for every observation, action, and task topic.
    Incoming messages are logged to Rerun in real-time.
    """

    def __init__(self) -> None:
        super().__init__("rerun_viewer")

        # Lazy import — allows the node to declare params before crashing
        # if rerun is missing.
        try:
            import rerun as rr  # noqa: F811
        except ImportError as exc:
            self.get_logger().fatal(
                f"Failed to import rerun-sdk: {exc}. Rebuild from the workspace "
                "venv with `source .shrc_local && python3 -m colcon build "
                "--symlink-install --merge-install --packages-select dataset_tools`."
            )
            raise RuntimeError(
                "rerun_viewer requires a compatible rerun-sdk installation in the workspace venv"
            ) from exc
        self._rr = rr
        self._rerun_cleanup_done = False

        # ---- Parameters ----
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("rerun_app_name", "IB-Robot Recording")
        self.declare_parameter("rerun_mode", "spawn")
        self.declare_parameter("rerun_addr", "rerun+http://127.0.0.1:9090/proxy")
        self.declare_parameter("rerun_save_path", "/tmp/ib_recording.rrd")
        self.declare_parameter("rerun_memory_limit", _default_rerun_memory_limit())
        self.declare_parameter("image_max_fps", DEFAULT_IMAGE_MAX_FPS)
        self.declare_parameter("image_max_long_edge", DEFAULT_IMAGE_MAX_LONG_EDGE)

        robot_config_path = self.get_parameter("robot_config_path").get_parameter_value().string_value
        if not robot_config_path:
            raise RuntimeError("The 'robot_config_path' parameter is required for rerun_viewer.")

        # ---- Load contract ----
        from robot_config.loader import load_robot_config

        cfg_path = Path(robot_config_path).expanduser().resolve()
        robot_config = load_robot_config(str(cfg_path))
        self._contract: Contract = robot_config.to_contract()

        # ---- Initialize Rerun ----
        app_name = self.get_parameter("rerun_app_name").get_parameter_value().string_value
        mode = self.get_parameter("rerun_mode").get_parameter_value().string_value.lower()
        self._rerun_mode = mode

        rr.init(app_name, spawn=False)

        if mode == "connect":
            addr = self.get_parameter("rerun_addr").get_parameter_value().string_value
            rr.connect_grpc(addr)
            self.get_logger().info(f"Rerun: connected to {addr}")
        elif mode == "save":
            save_path = self.get_parameter("rerun_save_path").get_parameter_value().string_value
            rr.save(save_path)
            self.get_logger().info(f"Rerun: saving to {save_path}")
        else:
            memory_limit = (
                self.get_parameter("rerun_memory_limit").get_parameter_value().string_value.strip()
                or _default_rerun_memory_limit()
            )
            rr.spawn(memory_limit=memory_limit, detach_process=False)
            self.get_logger().info(f"Rerun: spawning local viewer (memory_limit={memory_limit})")

        # ---- Set up static SeriesLines style for scalar panels ----
        self._setup_series_styles()

        # ---- Create subscriptions ----
        self._cbg = ReentrantCallbackGroup()
        self._subs: list[Any] = []
        self._lock = threading.Lock()
        self._image_decode_warned: set[str] = set()
        self._last_image_log_time_s: dict[str, float] = {}
        self._joint_state_name_warned = False
        configured_image_max_fps = self.get_parameter("image_max_fps").get_parameter_value().double_value
        self._image_min_interval_s = 0.0 if configured_image_max_fps <= 0.0 else 1.0 / configured_image_max_fps
        self._image_max_long_edge = max(
            0,
            int(self.get_parameter("image_max_long_edge").get_parameter_value().integer_value),
        )

        self._setup_observation_subs()
        self._setup_action_subs()
        self._setup_feedback_sub()

        self.get_logger().info(
            f"Rerun viewer ready — contract '{self._contract.name}' | {len(self._subs)} subscriptions"
        )
        self.get_logger().info(
            "Rerun image safeguards enabled: "
            f"max_fps={'unlimited' if self._image_min_interval_s <= 0.0 else f'{configured_image_max_fps:.2f}'} "
            f"| max_long_edge={self._image_max_long_edge or 'disabled'}"
        )

    def _shutdown_rerun(self) -> None:
        """Close rerun transports and spawned viewer exactly once."""
        if self._rerun_cleanup_done:
            return

        self._rerun_cleanup_done = True
        rr = getattr(self, "_rr", None)
        if rr is None:
            return

        try:
            rr.disconnect()
        except Exception as exc:
            self.get_logger().warning(f"Failed to disconnect rerun cleanly: {exc}")

        try:
            rr.rerun_shutdown()
        except Exception as exc:
            self.get_logger().warning(f"Failed to fully shut down rerun: {exc}")

    def destroy_node(self) -> bool:
        self._shutdown_rerun()
        return super().destroy_node()

    # ================================================================
    #  Static style setup
    # ================================================================

    def _setup_series_styles(self) -> None:
        """Pre-configure rerun line-series styles for joint/action panels."""
        rr = self._rr

        # Joint panel: one series per joint name
        obs_state = [
            o for o in self._contract.observations if o.image is None and o.selector and o.selector.get("names")
        ]
        for o in obs_state:
            names = o.selector["names"]
            short_key = o.key.replace("observation.", "")
            for i, name in enumerate(names):
                entity = f"joints/{short_key}/{name}"
                rr.log(
                    entity,
                    rr.SeriesLines(
                        colors=[_color_for(i)],
                        names=[name],
                        widths=[1.5],
                    ),
                    static=True,
                )

        # Action panel: one series per action dim
        for a in self._contract.actions:
            if a.selector and a.selector.get("names"):
                names = a.selector["names"]
                for i, name in enumerate(names):
                    entity = f"actions/{a.key}/{name}"
                    rr.log(
                        entity,
                        rr.SeriesLines(
                            colors=[_color_for(i)],
                            names=[name],
                            widths=[1.5],
                        ),
                        static=True,
                    )

    # ================================================================
    #  Observation subscriptions (images + state vectors)
    # ================================================================

    def _setup_observation_subs(self) -> None:
        for obs in self._contract.observations:
            if obs.image is not None:
                self._subscribe_image(obs)
            else:
                self._subscribe_state(obs)

    def _subscribe_image(self, obs: ObservationSpec) -> None:
        """Subscribe to an image topic and log frames to rerun."""
        msg_cls = get_message(obs.type)
        qos = qos_profile_from_dict(obs.qos) or QoSProfile(depth=DEFAULT_QOS_DEPTH)

        # Derive a short camera name from the key, e.g. "observation.images.top" → "top"
        parts = obs.key.split(".")
        cam_name = parts[-1] if len(parts) > 1 else obs.key
        entity_path = f"cameras/{cam_name}"

        # Default-argument binding avoids Python's late-binding closure pitfall.
        def cb(msg: Any, _path: str = entity_path) -> None:
            self._log_image(msg, _path)

        sub = self.create_subscription(
            msg_cls,
            obs.topic,
            cb,
            qos,
            callback_group=self._cbg,
        )
        self._subs.append(sub)
        self.get_logger().info(f"  Image sub: {obs.topic} → rerun:/{entity_path}")

    def _subscribe_state(self, obs: ObservationSpec) -> None:
        """Subscribe to a state topic (e.g. JointState) and log scalars."""
        msg_cls = get_message(obs.type)
        qos = qos_profile_from_dict(obs.qos) or QoSProfile(depth=DEFAULT_QOS_DEPTH)

        names = list((obs.selector or {}).get("names", []))
        short_key = obs.key.replace("observation.", "")

        # Default-argument binding avoids Python's late-binding closure pitfall.
        def cb(msg: Any, _key: str = short_key, _names: list = names, _type: str = obs.type) -> None:
            self._log_state(msg, _key, _names, _type)

        sub = self.create_subscription(
            msg_cls,
            obs.topic,
            cb,
            qos,
            callback_group=self._cbg,
        )
        self._subs.append(sub)
        self.get_logger().info(f"  State sub: {obs.topic} → rerun:/joints/{short_key} ({len(names)} channels)")

    # ================================================================
    #  Action subscriptions
    # ================================================================

    def _setup_action_subs(self) -> None:
        for act in self._contract.actions:
            msg_cls = get_message(act.type)
            qos = qos_profile_from_dict(act.publish_qos) or QoSProfile(depth=DEFAULT_QOS_DEPTH)
            names = list((act.selector or {}).get("names", []))
            key = act.key

            # Default-argument binding avoids Python's late-binding closure pitfall.
            def cb(msg: Any, _key: str = key, _names: list = names, _type: str = act.type) -> None:
                self._log_action(msg, _key, _names, _type)

            sub = self.create_subscription(
                msg_cls,
                act.publish_topic,
                cb,
                qos,
                callback_group=self._cbg,
            )
            self._subs.append(sub)
            self.get_logger().info(f"  Action sub: {act.publish_topic} → rerun:/actions/{key} ({len(names)} channels)")

    # ================================================================
    #  Episode feedback subscription
    # ================================================================

    def _setup_feedback_sub(self) -> None:
        """Subscribe to the RecordEpisode action feedback topic."""
        try:
            from action_msgs.msg import GoalStatusArray

            from ibrobot_msgs.action import RecordEpisode
        except ImportError:
            self.get_logger().warning("ibrobot_msgs not available — episode feedback disabled")
            return

        # Action feedback topic follows the ROS 2 convention:
        # /<action_name>/_action/feedback
        namespace = self.get_namespace().rstrip("/")
        action_name = f"{namespace}/record_episode" if namespace else "record_episode"
        feedback_topic = f"{action_name}/_action/feedback"
        status_topic = f"{action_name}/_action/status"
        feedback_type = RecordEpisode.Impl.FeedbackMessage
        sub = self.create_subscription(
            feedback_type,
            feedback_topic,
            self._on_episode_feedback,
            QoSProfile(depth=10),
            callback_group=self._cbg,
        )
        self._subs.append(sub)

        # Also subscribe to status for recording state detection
        sub2 = self.create_subscription(
            GoalStatusArray,
            status_topic,
            self._on_episode_status,
            QoSProfile(depth=10),
            callback_group=self._cbg,
        )
        self._subs.append(sub2)
        self.get_logger().info(f"  Episode feedback sub: {action_name}/_action/{{feedback,status}}")

    # ================================================================
    #  Logging callbacks
    # ================================================================

    def _set_ros_time(self, msg: Any) -> float:
        """Set the rerun timeline to the message's ROS timestamp."""
        rr = self._rr
        try:
            stamp = msg.header.stamp
            t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except AttributeError:
            t = float(self.get_clock().now().nanoseconds) * 1e-9
        rr.set_time("ros_time", duration=t)
        return t

    def _log_image(self, msg: Any, entity_path: str) -> None:
        """Convert a ROS Image message and log to rerun."""
        rr = self._rr
        timestamp_s = self._set_ros_time(msg)

        with self._lock:
            last_timestamp_s = self._last_image_log_time_s.get(entity_path)
            should_log = _should_log_sample(
                last_timestamp_s=last_timestamp_s,
                current_timestamp_s=timestamp_s,
                min_interval_s=self._image_min_interval_s,
            )
            if should_log:
                self._last_image_log_time_s[entity_path] = timestamp_s

        if not should_log:
            return

        arr = _decode_image_for_rerun(msg, spec=None)
        if arr is None:
            if entity_path not in self._image_decode_warned:
                self._image_decode_warned.add(entity_path)
                self.get_logger().warning(
                    "Skipping image for "
                    f"{entity_path}: unsupported or malformed frame "
                    f"(encoding={getattr(msg, 'encoding', '?')}, "
                    f"width={getattr(msg, 'width', '?')}, "
                    f"height={getattr(msg, 'height', '?')}, "
                    f"step={getattr(msg, 'step', '?')})"
                )
            return

        arr = _downscale_image_for_rerun(arr, self._image_max_long_edge)

        # Determine color model
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            rr.log(entity_path, rr.Image(arr))
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rr.log(entity_path, rr.Image(arr, color_model=rr.ColorModel.RGB))
        elif arr.ndim == 3 and arr.shape[2] == 4:
            rr.log(entity_path, rr.Image(arr, color_model=rr.ColorModel.RGBA))
        else:
            rr.log(entity_path, rr.Image(arr))

    def _log_state(self, msg: Any, key: str, names: list[str], ros_type: str) -> None:
        """Log a state observation (e.g. JointState) as scalar time series."""
        rr = self._rr
        self._set_ros_time(msg)

        values = self._extract_values(msg, names, ros_type)
        if values is None:
            return

        for name, val in zip(names, values, strict=False):
            rr.log(f"joints/{key}/{name}", rr.Scalars([val]))

    def _log_action(self, msg: Any, key: str, names: list[str], ros_type: str) -> None:
        """Log an action command as scalar time series."""
        rr = self._rr
        # Actions typically lack headers; use node clock
        t = float(self.get_clock().now().nanoseconds) * 1e-9
        rr.set_time("ros_time", duration=t)

        values = self._extract_values(msg, names, ros_type)
        if values is None:
            return

        for name, val in zip(names, values, strict=False):
            rr.log(f"actions/{key}/{name}", rr.Scalars([val]))

    def _on_episode_feedback(self, msg: Any) -> None:
        """Log episode recording progress."""
        rr = self._rr
        t = float(self.get_clock().now().nanoseconds) * 1e-9
        rr.set_time("ros_time", duration=t)

        fb = msg.feedback
        rr.log(
            "episode/remaining_s",
            rr.Scalars([float(fb.seconds_remaining)]),
        )
        rr.log(
            "episode/status",
            rr.TextLog(fb.feedback_message),
        )

    def _on_episode_status(self, msg: Any) -> None:
        """Log recording state changes (active/idle)."""
        rr = self._rr
        t = float(self.get_clock().now().nanoseconds) * 1e-9
        rr.set_time("ros_time", duration=t)

        # GoalStatusArray.status_list is non-empty when a goal is active
        is_recording = False
        for status in msg.status_list:
            # STATUS_EXECUTING = 2, STATUS_CANCELING = 5
            if status.status in (2, 5):
                is_recording = True
                break

        label = "● RECORDING" if is_recording else "○ IDLE"
        rr.log("episode/recording_state", rr.TextLog(label))

    # ================================================================
    #  Value extraction helpers
    # ================================================================

    def _extract_values(self, msg: Any, names: list[str], ros_type: str) -> list[float] | None:
        """Extract numeric values from a ROS message based on type and selector.

        Supports:
        - ``sensor_msgs/msg/JointState``: extracts ``position`` by joint name
        - ``std_msgs/msg/Float64MultiArray``: extracts ``data`` by index
        - ``sensor_msgs/msg/JointState`` with ``position.N`` selectors
        """
        if not names:
            return None

        if ros_type == "sensor_msgs/msg/JointState":
            return self._extract_joint_state(msg, names)
        elif "Float64MultiArray" in ros_type or "Float32MultiArray" in ros_type:
            return self._extract_multi_array(msg, names)
        else:
            # Best-effort: try .data attribute
            data = getattr(msg, "data", None)
            if data is not None and len(data) >= len(names):
                return [float(data[i]) for i in range(len(names))]
            return None

    def _extract_joint_state(self, msg: Any, names: list[str]) -> list[float] | None:
        """Extract values from JointState using selector names.

        Selector names follow the pattern ``position.N`` where N is
        a 1-based index, or direct joint names matched against msg.name.
        """
        values: list[float] = []
        joint_names = list(getattr(msg, "name", []))
        if not joint_names and any("." not in sel_name for sel_name in names) and not self._joint_state_name_warned:
            self._joint_state_name_warned = True
            self.get_logger().warning("JointState.name is empty; named selectors will fall back to 0.0 values")

        for sel_name in names:
            # Pattern: "position.N" → index into msg.position
            if "." in sel_name:
                parts = sel_name.split(".", 1)
                field_name = parts[0]  # "position", "velocity", "effort"
                try:
                    idx = int(parts[1]) - 1  # 1-based → 0-based
                except (ValueError, IndexError):
                    values.append(0.0)
                    continue

                field_data = getattr(msg, field_name, None)
                if field_data is not None and idx < len(field_data):
                    values.append(float(field_data[idx]))
                else:
                    values.append(0.0)
            else:
                # Direct joint name match
                if sel_name in joint_names:
                    idx = joint_names.index(sel_name)
                    if idx < len(msg.position):
                        values.append(float(msg.position[idx]))
                    else:
                        values.append(0.0)
                else:
                    values.append(0.0)

        return values if values else None

    def _extract_multi_array(self, msg: Any, names: list[str]) -> list[float] | None:
        """Extract values from Float64MultiArray/Float32MultiArray by index.

        Selector names follow the pattern ``action.N`` where N is a 0-based
        index, or simple integers.
        """
        data = msg.data
        if not data:
            return None

        values: list[float] = []
        for i, name in enumerate(names):
            # Try to parse index from name (e.g. "action.0" → 0)
            idx = i
            if "." in name:
                with suppress(ValueError):
                    idx = int(name.split(".")[-1])

            if idx < len(data):
                values.append(float(data[idx]))
            else:
                values.append(0.0)

        return values if values else None


# ================================================================
#  Entry point
# ================================================================


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RerunViewer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, _frame) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        node.get_logger().info("Rerun viewer shutting down")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

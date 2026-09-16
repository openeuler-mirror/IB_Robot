"""Move to a fixed release pose, release, reposition, verify, and return."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState

from embodied_common.dispatch_binding import (
    copy_binding,
    delegated_executor_identity,
    delegated_executor_identity_matches,
    fill_delegated_executor_identity,
)
from embodied_common.wire_contracts import validate_public_request_wire_contracts
from ibrobot_msgs.action import PlaceObject, PrimitiveCommand
from ibrobot_msgs.msg import DetectionArray
from ibrobot_msgs.srv import GroundingDetect, SegmentDetections


class PlacementFlowError(RuntimeError):
    """A placement phase failed with a stable public error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PlacementCancelled(PlacementFlowError):
    def __init__(self) -> None:
        super().__init__("PLACE_CANCELLED", "placement execution cancelled before release")


class PrimitiveFlowError(PlacementFlowError):
    """A primitive failed with explicit terminal-state knowledge."""

    def __init__(self, code: str, message: str, *, terminal_known: bool) -> None:
        super().__init__(code, message)
        self.terminal_known = terminal_known


PRIMITIVE_CANCEL_CLEANUP_TIMEOUT = "CANCEL_CLEANUP_TIMEOUT"


@dataclass(frozen=True)
class MaskContainment:
    inside: bool
    target_pixel_count: int
    inside_pixel_count: int
    inside_fraction: float
    center_inside: bool


@dataclass
class PlacementState:
    completed_phases: list[str] = field(default_factory=list)
    diagnostic_details: list[str] = field(default_factory=list)
    pipeline_timings: dict[str, float] = field(default_factory=dict)
    release_status: int = PlaceObject.Result.RELEASE_NOT_RELEASED
    verification_status: int = PlaceObject.Result.VERIFICATION_NOT_RUN
    place_succeeded: bool = False
    release_completed_ros_ns: int = 0
    verification_started_ros_ns: int = 0
    active_phase: str = ""
    active_started: float = 0.0
    debug_output_dir: str = ""
    target_query: str = ""
    container_query: str = ""

    def phase(self, name: str) -> None:
        now = time.monotonic()
        if self.active_phase:
            key = f"phase_{self.active_phase}"
            self.pipeline_timings[key] = self.pipeline_timings.get(key, 0.0) + max(0.0, now - self.active_started)
        self.active_phase = name
        self.active_started = now
        if not self.completed_phases or self.completed_phases[-1] != name:
            self.completed_phases.append(name)

    def finish(self) -> None:
        if not self.active_phase:
            return
        now = time.monotonic()
        key = f"phase_{self.active_phase}"
        self.pipeline_timings[key] = self.pipeline_timings.get(key, 0.0) + max(0.0, now - self.active_started)
        self.active_phase = ""


def _finite_positive(value: object) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError, OverflowError):
        return False


def _mask_to_bool(message) -> np.ndarray:
    """Decode a mono8 ROS image without depending on OpenCV."""
    encoding = str(message.encoding).lower()
    if encoding not in {"mono8", "8uc1"}:
        raise ValueError(f"unsupported segmentation encoding: {message.encoding}")
    height = int(message.height)
    width = int(message.width)
    row_items = int(message.step)
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if height <= 0 or width <= 0 or row_items < width or raw.size < height * row_items:
        raise ValueError("segmentation mask payload is inconsistent with its dimensions")
    return raw[: height * row_items].reshape((height, row_items))[:, :width] > 0


def _filled_container_region(mask: np.ndarray, inset_ratio: float) -> np.ndarray:
    """Fill the segmented container silhouette row-by-row and inset its edges."""
    source = np.asarray(mask, dtype=bool)
    region = np.zeros_like(source)
    rows = np.flatnonzero(np.any(source, axis=1))
    if rows.size == 0:
        return region
    y_min = int(rows[0])
    y_max = int(rows[-1])
    y_inset = int(round((y_max - y_min + 1) * float(inset_ratio)))
    for y in rows:
        if int(y) < y_min + y_inset or int(y) > y_max - y_inset:
            continue
        columns = np.flatnonzero(source[int(y)])
        if columns.size == 0:
            continue
        x_min = int(columns[0])
        x_max = int(columns[-1])
        x_inset = int(round((x_max - x_min + 1) * float(inset_ratio)))
        start = x_min + x_inset
        stop = x_max - x_inset + 1
        if start < stop:
            region[int(y), start:stop] = True
    return region


def _normalized_polygon_mask(shape: tuple[int, int], polygons: object) -> np.ndarray:
    """Rasterize normalized ``[x, y]`` polygons into a boolean image mask.

    The placement camera can publish a different resolution on different robots.  Keeping
    the exclusion geometry normalized makes the same SSOT configuration valid for both
    640x360 PC frames and the 1280x720 310P frames.
    """
    height, width = (int(shape[0]), int(shape[1])) if len(shape) == 2 else (0, 0)
    if height <= 0 or width <= 0:
        raise ValueError("exclusion mask shape must contain positive height and width")
    if not isinstance(polygons, list):
        raise ValueError("exclusion mask polygons must be a list")
    result = np.zeros((height, width), dtype=bool)
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid += 0.5
    y_grid += 0.5
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("each exclusion mask polygon must contain at least three points")
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("exclusion mask polygon points must be finite [x, y] pairs")
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError("exclusion mask polygon points must be normalized to [0, 1]")
        points[:, 0] *= width
        points[:, 1] *= height
        inside = np.zeros((height, width), dtype=bool)
        for index in range(len(points)):
            x1, y1 = points[index]
            x2, y2 = points[(index + 1) % len(points)]
            if y1 == y2:
                continue
            crosses = (y1 > y_grid) != (y2 > y_grid)
            x_intersection = (x2 - x1) * (y_grid - y1) / (y2 - y1) + x1
            inside ^= crosses & (x_grid < x_intersection)
        result |= inside
    return result


def _load_exclusion_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    """Load a mono mask file and resize it to the current RGB frame."""
    configured_path = str(path).strip()
    configured_path = re.sub(r"\$\(env\s+(\w+)\)", lambda match: os.environ.get(match.group(1), ""), configured_path)
    if "$(find robot_config)" in configured_path:
        try:
            from ament_index_python.packages import get_package_share_directory

            configured_path = configured_path.replace(
                "$(find robot_config)", get_package_share_directory("robot_config")
            )
        except (ImportError, KeyError, RuntimeError):
            configured_path = configured_path.replace(
                "$(find robot_config)", str(Path(__file__).resolve().parents[2] / "robot_config")
            )
    mask_path = Path(configured_path).expanduser()
    if not mask_path.is_file():
        raise FileNotFoundError(f"target exclusion mask not found: {mask_path}")
    try:
        from PIL import Image as PILImage

        source = np.asarray(PILImage.open(mask_path).convert("L"), dtype=np.uint8)
    except ImportError:
        import cv2

        source = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise ValueError(f"unable to read target exclusion mask: {mask_path}") from None
    if source.ndim != 2 or source.size == 0:
        raise ValueError(f"target exclusion mask must be a non-empty grayscale image: {mask_path}")
    height, width = int(shape[0]), int(shape[1])
    if source.shape != (height, width):
        try:
            from PIL import Image as PILImage

            source = np.asarray(
                PILImage.fromarray(source).resize((width, height), PILImage.Resampling.NEAREST), dtype=np.uint8
            )
        except ImportError:
            import cv2

            source = cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)
    return source > 0


def _bbox_exclusion_overlap(bbox: object, exclusion_mask: np.ndarray) -> float:
    """Return the fraction of a detection bbox covered by an exclusion mask."""
    values = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if values.size != 4:
        return 0.0
    height, width = exclusion_mask.shape
    x1, y1, x2, y2 = values
    left = max(0, min(width, int(math.floor(float(min(x1, x2))))))
    right = max(0, min(width, int(math.ceil(float(max(x1, x2))))))
    top = max(0, min(height, int(math.floor(float(min(y1, y2))))))
    bottom = max(0, min(height, int(math.ceil(float(max(y1, y2))))))
    if right <= left or bottom <= top:
        return 0.0
    area = (right - left) * (bottom - top)
    return float(np.count_nonzero(exclusion_mask[top:bottom, left:right])) / float(area)


def evaluate_mask_containment(
    container_mask,
    target_mask,
    *,
    min_target_pixels: int,
    min_inside_fraction: float,
    container_inset_ratio: float,
) -> MaskContainment:
    """Evaluate a target mask against the filled 2-D container region."""
    container = _mask_to_bool(container_mask)
    target = _mask_to_bool(target_mask)
    return evaluate_mask_arrays(
        container,
        target,
        min_target_pixels=min_target_pixels,
        min_inside_fraction=min_inside_fraction,
        container_inset_ratio=container_inset_ratio,
    )


def evaluate_mask_arrays(
    container_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    min_target_pixels: int,
    min_inside_fraction: float,
    container_inset_ratio: float,
) -> MaskContainment:
    """Evaluate replayable boolean masks without requiring live ROS messages."""
    container = np.asarray(container_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if container.shape != target.shape:
        raise ValueError("container and target masks have different dimensions")
    if container.ndim != 2 or target.ndim != 2:
        raise ValueError("container and target masks must be 2-D")
    target_pixel_count = int(np.count_nonzero(target))
    if target_pixel_count < int(min_target_pixels):
        return MaskContainment(False, target_pixel_count, 0, 0.0, False)
    region = _filled_container_region(container, container_inset_ratio)
    target_y, target_x = np.nonzero(target)
    center_y = int(round(float(np.mean(target_y))))
    center_x = int(round(float(np.mean(target_x))))
    center_y = min(max(center_y, 0), region.shape[0] - 1)
    center_x = min(max(center_x, 0), region.shape[1] - 1)
    center_inside = bool(region[center_y, center_x])
    inside_pixel_count = int(np.count_nonzero(target & region))
    inside_fraction = inside_pixel_count / target_pixel_count
    return MaskContainment(
        inside=center_inside and inside_fraction >= float(min_inside_fraction),
        target_pixel_count=target_pixel_count,
        inside_pixel_count=inside_pixel_count,
        inside_fraction=inside_fraction,
        center_inside=center_inside,
    )


class PlacementExecutorNode(Node):
    """Execute the fixed-pose release, single-joint verification, and return action."""

    _PROGRESS = {
        "preflight": 0.05,
        "move_to_place": 0.15,
        "release": 0.35,
        "move_to_verify": 0.50,
        "verify_place": 0.65,
        "return_to_place": 0.90,
        "completed": 1.0,
    }

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("placement_executor_node", parameter_overrides=parameter_overrides)
        validate_public_request_wire_contracts()
        self.declare_parameter("action_name", "/manipulation/execute_place")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("placement_execution_json", "{}")
        self.declare_parameter("gripper_joint_name", "")
        self.declare_parameter("gripper_open_position", 1.0)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("gripper_position_tolerance", 0.05)
        self.declare_parameter("rpc_timeout_sec", 5.0)

        self._action_name = str(self.get_parameter("action_name").value)
        primitive_action_name = str(self.get_parameter("primitive_action_name").value)
        self._config = self._load_json(self.get_parameter("placement_execution_json").value)
        self._executor_identity = delegated_executor_identity(
            name="placement_pipeline",
            endpoint_name=self._action_name,
            configuration=self._config,
        )
        self._dispatch_binding = None
        self._gripper_joint_name = str(self.get_parameter("gripper_joint_name").value).strip()
        if not self._gripper_joint_name:
            raise RuntimeError(
                "gripper_joint_name parameter is required: the SO-101 '6' default "
                "was removed; declare the gripper joint explicitly"
            )
        self._gripper_open = float(self.get_parameter("gripper_open_position").value)
        self._gripper_closed = float(self.get_parameter("gripper_closed_position").value)
        self._gripper_tolerance = float(self.get_parameter("gripper_position_tolerance").value)
        self._rpc_timeout = float(self.get_parameter("rpc_timeout_sec").value)
        self._detect_service = str(self._config.get("detect_service", "/perception/grasp/grounding_detect"))
        self._segment_service = str(self._config.get("segment_service", ""))
        self._debug_output_root = str(self._config.get("debug_output_root", ""))
        self._debug_sample_index = 0
        # The wrist camera and gripper are rigidly mounted.  Cache the static
        # image-space exclusion mask for each input resolution instead of
        # reloading/rasterizing it for every post-release verification sample.
        self._target_exclusion_mask_cache: dict[tuple[object, ...], np.ndarray] = {}

        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._state_lock = threading.Lock()
        self._latest_rgb: Image | None = None
        self._latest_joint_state: JointState | None = None
        self._latest_joint_receipt_monotonic = 0.0
        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            Image,
            str(self._config.get("rgb_topic", "/camera/wrist/image_raw")),
            self._rgb_cb,
            qos_profile_sensor_data,
            callback_group=callback_group,
        )
        self.create_subscription(
            JointState,
            str(self._config.get("joint_state_topic", "/joint_states")),
            self._joint_cb,
            10,
            callback_group=callback_group,
        )
        self._detect_client = self.create_client(GroundingDetect, self._detect_service, callback_group=callback_group)
        self._segment_client = (
            self.create_client(SegmentDetections, self._segment_service, callback_group=callback_group)
            if self._segment_service
            else None
        )
        self._primitive_client = ActionClient(
            self, PrimitiveCommand, primitive_action_name, callback_group=callback_group
        )
        self._action_server = ActionServer(
            self,
            PlaceObject,
            self._action_name,
            execute_callback=self._execute_place,
            goal_callback=self._handle_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=callback_group,
        )

    @staticmethod
    def _load_json(value: object) -> dict[str, Any]:
        parsed = json.loads(str(value or "{}"))
        if not isinstance(parsed, dict):
            raise ValueError("placement_execution_json must contain a JSON object")
        return parsed

    @staticmethod
    def _safe_component(value: str) -> str:
        component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
        return component.strip("._") or "task"

    def _start_debug_record(
        self,
        state: PlacementState,
        task_id: str,
        target_query: str,
        container_query: str,
    ) -> None:
        """Create a replayable evidence directory before any irreversible action."""
        state.target_query = str(target_query)
        state.container_query = str(container_query)
        root = str(getattr(self, "_debug_output_root", "") or "").strip()
        missing_environment: list[str] = []

        def replace_environment(match: re.Match[str]) -> str:
            name = match.group(1)
            value = os.environ.get(name, "")
            if not value:
                missing_environment.append(name)
            return value

        root = re.sub(r"\$\(env\s+(\w+)\)", replace_environment, root)
        if missing_environment:
            self.get_logger().warning(
                f"placement evidence disabled because environment is unset: {', '.join(missing_environment)}"
            )
            return
        root = os.path.expanduser(root)
        if not root:
            return
        directory = Path(root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}_{self._safe_component(task_id)}"
        try:
            directory.mkdir(parents=True, exist_ok=False)
            state.debug_output_dir = str(directory)
            self._debug_sample_index = 0
            self._write_debug_json(
                directory / "placement_manifest.json",
                {
                    "schema_version": 1,
                    "pipeline": "placement_pipeline",
                    "pipeline_version": 3,
                    "task_id": str(task_id),
                    "target_query": str(target_query),
                    "container_query": str(container_query),
                    "executor_identity": dict(getattr(self, "_executor_identity", {})),
                    "configuration": getattr(self, "_config", {}),
                    "gripper": {
                        "joint_name": self._gripper_joint_name,
                        "open_position": self._gripper_open,
                        "position_tolerance": self._gripper_tolerance,
                    },
                    "evidence_policy": "post_release_rgb_and_joint_state_replay",
                },
            )
        except (OSError, TypeError, ValueError) as exc:
            self.get_logger().warning(f"failed to create placement evidence directory: {exc}")

    @staticmethod
    def _write_debug_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    @staticmethod
    def _image_array(image: Image) -> np.ndarray:
        encoding = str(image.encoding).lower()
        channels = 1 if encoding in {"mono8", "8uc1"} else 3 if encoding in {"rgb8", "bgr8"} else 0
        height, width, step = int(image.height), int(image.width), int(image.step)
        if channels == 0 or height <= 0 or width <= 0 or step < width * channels:
            raise ValueError(f"unsupported RGB image payload: encoding={image.encoding} size={width}x{height}")
        raw = np.frombuffer(bytes(image.data), dtype=np.uint8)
        if raw.size < height * step:
            raise ValueError("RGB image payload is shorter than height*step")
        rows = raw[: height * step].reshape((height, step))[:, : width * channels]
        return rows.reshape((height, width, channels)) if channels == 3 else rows.reshape((height, width))

    @staticmethod
    def _image_with_exclusion(image: Image, exclusion_mask: np.ndarray) -> Image:
        """Return a copy of ``image`` with excluded pixels replaced by neutral gray."""
        array = PlacementExecutorNode._image_array(image)
        exclusion = np.asarray(exclusion_mask, dtype=bool)
        if array.ndim != 3 or exclusion.shape != array.shape[:2]:
            raise ValueError("target exclusion mask dimensions do not match the RGB image")
        masked = array.copy()
        masked[exclusion] = 127
        output = deepcopy(image)
        raw = np.frombuffer(bytearray(bytes(output.data)), dtype=np.uint8)
        rows = raw[: int(output.height) * int(output.step)].reshape((int(output.height), int(output.step)))
        rows[:, : int(output.width) * 3] = masked.reshape((int(output.height), int(output.width) * 3))
        output.data = bytes(raw)
        return output

    def _target_exclusion_mask(self, image: Image) -> np.ndarray | None:
        """Build the configured normalized gripper exclusion mask for one frame."""
        settings = self._config.get("verification", {}).get("target_exclusion", {})
        if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
            return None
        shape = (int(image.height), int(image.width))
        mask_path = str(settings.get("mask_path", "")).strip()
        polygons = settings.get("polygons", [])
        # Include the source geometry and frame shape in the key.  This keeps
        # the cache valid if a test/runtime swaps configuration or if a camera
        # changes resolution, while still reusing the fixed mask for all normal
        # frames in a placement action.
        source_key = ("path", mask_path) if mask_path else ("polygons", repr(polygons))
        cache_key = (*source_key, *shape)
        cache = getattr(self, "_target_exclusion_mask_cache", None)
        if cache is None:
            cache = {}
            self._target_exclusion_mask_cache = cache
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if mask_path:
            try:
                exclusion = _load_exclusion_mask(mask_path, shape)
            except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
                raise PlacementFlowError(
                    "PLACE_VERIFICATION_UNCERTAIN",
                    f"target exclusion mask file unavailable: {exc}",
                ) from exc
        else:
            exclusion = _normalized_polygon_mask(shape, polygons)
        cache[cache_key] = exclusion
        return exclusion

    @staticmethod
    def _detection_exclusion_overlap(detection: Any, exclusion_mask: np.ndarray) -> float:
        try:
            mask = _mask_to_bool(detection.mask)
            if mask.shape == exclusion_mask.shape and np.any(mask):
                return float(np.count_nonzero(mask & exclusion_mask)) / float(np.count_nonzero(mask))
        except (AttributeError, TypeError, ValueError):
            pass
        return _bbox_exclusion_overlap(getattr(detection, "bbox", None), exclusion_mask)

    @staticmethod
    def _remove_exclusion_from_detection(detection: Any, exclusion_mask: np.ndarray) -> Any:
        """Copy a detection and remove excluded pixels from its segmentation mask."""
        try:
            mask = _mask_to_bool(detection.mask)
        except (AttributeError, TypeError, ValueError):
            return detection
        if mask.shape != exclusion_mask.shape:
            return detection
        output = deepcopy(detection)
        filtered = np.where(mask & ~exclusion_mask, 255, 0).astype(np.uint8)
        output.mask.data = bytes(filtered)
        output.mask.step = int(filtered.shape[1])
        return output

    def _filter_target_detections(self, detections: list[Any], exclusion_mask: np.ndarray | None) -> list[Any]:
        if exclusion_mask is None:
            return detections
        settings = self._config.get("verification", {}).get("target_exclusion", {})
        overlap_threshold = float(settings.get("min_detection_overlap", 0.35))
        filtered = []
        for detection in detections:
            overlap = self._detection_exclusion_overlap(detection, exclusion_mask)
            if overlap >= overlap_threshold:
                continue
            filtered.append(self._remove_exclusion_from_detection(detection, exclusion_mask))
        return filtered

    @staticmethod
    def _detection_metadata(detections: list[Any]) -> list[dict[str, Any]]:
        records = []
        for detection in detections:
            records.append(
                {
                    "label": str(detection.label),
                    "confidence": float(detection.confidence),
                    "bbox": [float(value) for value in detection.bbox],
                    "mask_shape": [int(detection.mask.height), int(detection.mask.width)],
                }
            )
        return records

    def _record_verification_sample(
        self,
        *,
        image: Image,
        containers: list[Any],
        targets: list[Any],
        outcome: bool | None,
        detail: str,
        debug_output_dir: str,
        sample_index: int = 0,
    ) -> None:
        if not debug_output_dir:
            return
        try:
            index = int(sample_index or getattr(self, "_debug_sample_index", 0))
            if index <= 0:
                index = int(getattr(self, "_debug_sample_index", 0)) + 1
                self._debug_sample_index = index
            directory = Path(debug_output_dir)
            records = {"sample_index": index, "stamp_ns": self._stamp_ns(image), "outcome": outcome, "detail": detail}
            for kind, detections in (("container", containers), ("target", targets)):
                records[f"{kind}_detections"] = self._detection_metadata(detections)
                for detection_index, detection in enumerate(detections):
                    np.save(
                        directory / f"sample_{index:02d}_{kind}_{detection_index:02d}_mask.npy",
                        _mask_to_bool(detection.mask),
                        allow_pickle=False,
                    )
            self._write_debug_json(directory / f"sample_{index:02d}.json", records)
        except (OSError, ValueError, TypeError) as exc:
            self.get_logger().warning(f"failed to save placement verification sample: {exc}")

    def _record_image_snapshot(self, image: Image, debug_output_dir: str) -> int:
        """Persist each fresh post-release RGB frame, even when detection is uncertain."""
        if not debug_output_dir:
            return 0
        try:
            index = int(getattr(self, "_debug_sample_index", 0)) + 1
            self._debug_sample_index = index
            directory = Path(debug_output_dir)
            np.save(directory / f"sample_{index:02d}_rgb.npy", self._image_array(image), allow_pickle=False)
            return index
        except (OSError, ValueError, TypeError) as exc:
            self.get_logger().warning(f"failed to save placement RGB evidence: {exc}")
            return 0

    def _record_exclusion_snapshot(
        self, exclusion_mask: np.ndarray | None, debug_output_dir: str, sample_index: int
    ) -> None:
        if exclusion_mask is None or not debug_output_dir or not sample_index:
            return
        try:
            np.save(
                Path(debug_output_dir) / f"sample_{int(sample_index):02d}_target_exclusion_mask.npy",
                np.asarray(exclusion_mask, dtype=bool),
                allow_pickle=False,
            )
        except (OSError, ValueError, TypeError) as exc:
            self.get_logger().warning(f"failed to save target exclusion mask evidence: {exc}")

    def _record_masked_image_snapshot(
        self, image: Image, exclusion_mask: np.ndarray | None, debug_output_dir: str, sample_index: int
    ) -> None:
        if exclusion_mask is None or not debug_output_dir or not sample_index:
            return
        try:
            masked = self._image_with_exclusion(image, exclusion_mask)
            np.save(
                Path(debug_output_dir) / f"sample_{int(sample_index):02d}_target_masked_rgb.npy",
                self._image_array(masked),
                allow_pickle=False,
            )
        except (OSError, ValueError, TypeError) as exc:
            self.get_logger().warning(f"failed to save masked placement RGB evidence: {exc}")

    def _record_open_feedback(self, state: PlacementState) -> None:
        if not state.debug_output_dir:
            return
        with self._state_lock:
            joint_state = self._latest_joint_state
            receipt = self._latest_joint_receipt_monotonic
        if joint_state is None:
            return
        try:
            self._write_debug_json(
                Path(state.debug_output_dir) / "open_gripper_joint_state.json",
                {
                    "receipt_monotonic": float(receipt),
                    "header_stamp_ns": self._stamp_ns(joint_state),
                    "name": [str(name) for name in joint_state.name],
                    "position": [float(value) for value in joint_state.position],
                    "velocity": [float(value) for value in joint_state.velocity],
                    "effort": [float(value) for value in joint_state.effort],
                },
            )
        except (OSError, TypeError, ValueError) as exc:
            self.get_logger().warning(f"failed to save open-gripper feedback: {exc}")

    def _handle_goal(self, request):
        if not str(request.target_query).strip():
            return GoalResponse.REJECT
        if not str(request.container_query).strip():
            return GoalResponse.REJECT
        if not _finite_positive(request.timeout_sec):
            return GoalResponse.REJECT
        if not delegated_executor_identity_matches(request.expected_executor, self._executor_identity):
            return GoalResponse.REJECT
        binding = request.dispatch_binding
        budget = binding.task_budget
        if (
            binding.schema_version != 1
            or not str(binding.task_id).strip()
            or not str(binding.root_task_id).strip()
            or not str(binding.dispatch_nonce).strip()
            or not str(binding.expected_registry_epoch).strip()
            or int(binding.expected_registry_generation) <= 0
            or not str(binding.expected_registry_digest).strip()
            or budget.schema_version != 1
        ):
            return GoalResponse.REJECT
        started = budget.started_at.sec + budget.started_at.nanosec / 1_000_000_000
        deadline = budget.deadline.sec + budget.deadline.nanosec / 1_000_000_000
        now = self.get_clock().now().nanoseconds / 1_000_000_000
        if (
            budget.started_at.sec < 0
            or budget.deadline.sec < 0
            or not 0 <= budget.started_at.nanosec < 1_000_000_000
            or not 0 <= budget.deadline.nanosec < 1_000_000_000
            or not math.isfinite(started)
            or not math.isfinite(deadline)
            or deadline <= started
            or deadline <= now
            or float(request.timeout_sec) > deadline - now
        ):
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active:
                return GoalResponse.REJECT
            self._goal_active = True
            self._dispatch_binding = copy_binding(binding)
        return GoalResponse.ACCEPT

    def _joint_cb(self, message: JointState) -> None:
        with self._state_lock:
            self._latest_joint_state = message
            self._latest_joint_receipt_monotonic = time.monotonic()

    def _rgb_cb(self, message: Image) -> None:
        with self._state_lock:
            self._latest_rgb = message

    def _wait_future(self, future, goal_handle, deadline: float, label: str, *, honor_cancel: bool = True):
        while rclpy.ok() and not future.done():
            if honor_cancel and goal_handle.is_cancel_requested:
                raise PlacementCancelled()
            if time.monotonic() >= deadline:
                future.cancel()
                raise PlacementFlowError("DEPENDENCY_UNAVAILABLE", f"{label} timed out")
            time.sleep(0.02)
        try:
            return future.result()
        except Exception as exc:
            raise PlacementFlowError("DEPENDENCY_UNAVAILABLE", f"{label} failed: {exc}") from exc

    def _cancel_primitive_and_wait(self, handle, result_future, primitive_name: str) -> None:
        """Cancel one admitted primitive and require a known terminal state."""
        cleanup_deadline = time.monotonic() + self._rpc_timeout
        try:
            cancel_future = handle.cancel_goal_async()
            while rclpy.ok() and not cancel_future.done() and time.monotonic() < cleanup_deadline:
                time.sleep(0.02)
            if not cancel_future.done() or cancel_future.result() is None:
                raise PlacementFlowError(
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    f"primitive cancellation state is unknown: {primitive_name}",
                )
            cancel_response = cancel_future.result()
            while rclpy.ok() and not result_future.done() and time.monotonic() < cleanup_deadline:
                time.sleep(0.02)
            if not result_future.done() or result_future.result() is None:
                raise PlacementFlowError(
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    f"primitive terminal state is unknown: {primitive_name}",
                )
            wrapped = result_future.result()
            result = getattr(wrapped, "result", None)
            error_code = str(getattr(result, "error_code", ""))
            if error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                raise PlacementFlowError(
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    str(getattr(result, "message", "")) or f"primitive stop state is unknown: {primitive_name}",
                )
            if hasattr(cancel_response, "goals_canceling") and not cancel_response.goals_canceling and result is None:
                raise PlacementFlowError(
                    PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                    f"primitive cancellation was not accepted: {primitive_name}",
                )
        except PlacementFlowError:
            raise
        except Exception as exc:
            raise PlacementFlowError(
                PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                f"primitive cleanup failed: {primitive_name}",
            ) from exc

    def _cancel_pending_primitive_and_wait(self, send_future, primitive_name: str) -> None:
        """Resolve a pending goal response, then cancel any admitted primitive."""
        cleanup_deadline = time.monotonic() + self._rpc_timeout
        while rclpy.ok() and not send_future.done() and time.monotonic() < cleanup_deadline:
            time.sleep(0.02)
        if not send_future.done():
            raise PlacementFlowError(
                PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                f"primitive goal state is unknown: {primitive_name}",
            )
        try:
            handle = send_future.result()
        except Exception as exc:
            raise PlacementFlowError(
                PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
                f"primitive goal response failed during cleanup: {primitive_name}",
            ) from exc
        if handle is None or not handle.accepted:
            return
        self._cancel_primitive_and_wait(handle, handle.get_result_async(), primitive_name)

    def _feedback(
        self,
        goal_handle,
        state: PlacementState,
        phase: str,
        detail: str,
        *,
        honor_cancel: bool = True,
    ) -> None:
        if honor_cancel and goal_handle.is_cancel_requested:
            raise PlacementCancelled()
        state.phase(phase)
        feedback = PlaceObject.Feedback()
        feedback.phase = phase
        feedback.progress = float(self._PROGRESS[phase])
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)

    def _preflight(self, goal_handle, deadline: float, state: PlacementState) -> None:
        self._feedback(goal_handle, state, "preflight", "checking the motion and release dependency")
        wait_timeout = min(self._rpc_timeout, max(0.1, deadline - time.monotonic()))
        if not self._primitive_client.wait_for_server(timeout_sec=wait_timeout):
            raise PlacementFlowError("DEPENDENCY_UNAVAILABLE", "primitive action server unavailable")

    @staticmethod
    def _execution_token(goal_handle) -> str:
        raw_uuid = getattr(getattr(goal_handle, "goal_id", None), "uuid", None)
        try:
            return bytes(raw_uuid).hex() if raw_uuid is not None else ""
        except (TypeError, ValueError):
            return ""

    def _move_to_joint_positions(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        joint_names: list[str],
        joint_positions: list[float],
        *,
        duration_sec: float,
        honor_cancel: bool,
    ) -> None:
        """Move the configured joints through the guarded primitive."""
        goal = PrimitiveCommand.Goal()
        goal.schema_version = 1
        if self._dispatch_binding is not None:
            goal.dispatch_binding = copy_binding(self._dispatch_binding)
        if goal.dispatch_binding.task_id != task_id:
            raise PlacementFlowError("DISPATCH_BINDING_MISMATCH", "delegated primitive task ID mismatch")
        goal.execution_token = self._execution_token(goal_handle)
        goal.primitive_name = "move_to_joint_positions"
        goal.joint_names = [str(name) for name in joint_names]
        goal.joint_positions = [float(position) for position in joint_positions]
        goal.primitive_duration_sec = float(duration_sec)
        goal.timeout_sec = max(0.1, deadline - time.monotonic())
        send_future = self._primitive_client.send_goal_async(goal)
        try:
            handle = self._wait_future(
                send_future,
                goal_handle,
                deadline,
                "move_to_joint_positions dispatch",
                honor_cancel=honor_cancel,
            )
        except (PlacementCancelled, PlacementFlowError):
            self._cancel_pending_primitive_and_wait(send_future, "move_to_joint_positions")
            raise
        if handle is None or not handle.accepted:
            raise PlacementFlowError("PRIMITIVE_FAILED", "move_to_joint_positions was rejected")
        result_future = handle.get_result_async()
        try:
            wrapped = self._wait_future(
                result_future,
                goal_handle,
                deadline,
                "move_to_joint_positions result",
                honor_cancel=honor_cancel,
            )
        except (PlacementCancelled, PlacementFlowError):
            self._cancel_primitive_and_wait(handle, result_future, "move_to_joint_positions")
            raise
        result = getattr(wrapped, "result", None)
        if result is None or not bool(result.success):
            raise PlacementFlowError(
                str(getattr(result, "error_code", "")) or "PRIMITIVE_FAILED",
                str(getattr(result, "message", "")) or "move_to_joint_positions failed",
            )

    def _open_gripper(self, goal_handle, deadline: float, task_id: str) -> float:
        goal = PrimitiveCommand.Goal()
        goal.schema_version = 1
        if self._dispatch_binding is not None:
            goal.dispatch_binding = copy_binding(self._dispatch_binding)
        if goal.dispatch_binding.task_id != task_id:
            raise PrimitiveFlowError(
                "DISPATCH_BINDING_MISMATCH",
                "delegated primitive task ID mismatch",
                terminal_known=True,
            )
        goal.execution_token = self._execution_token(goal_handle)
        goal.primitive_name = "open_gripper"
        goal.gripper_position = self._gripper_open
        goal.timeout_sec = max(0.1, deadline - time.monotonic())
        try:
            handle = self._wait_future(
                self._primitive_client.send_goal_async(goal), goal_handle, deadline, "open_gripper dispatch"
            )
        except Exception as exc:
            raise PrimitiveFlowError(
                "RELEASE_STATE_UNKNOWN",
                f"open_gripper dispatch state is unknown: {exc}",
                terminal_known=False,
            ) from exc
        if handle is None or not handle.accepted:
            raise PrimitiveFlowError("PRIMITIVE_FAILED", "open_gripper primitive was rejected", terminal_known=True)
        try:
            wrapped = self._wait_future(
                handle.get_result_async(),
                goal_handle,
                deadline,
                "open_gripper result",
                honor_cancel=False,
            )
        except PlacementFlowError as exc:
            raise PrimitiveFlowError(
                "RELEASE_STATE_UNKNOWN", f"open_gripper terminal state is unknown: {exc}", terminal_known=False
            ) from exc
        result = getattr(wrapped, "result", None)
        if result is None or not bool(result.success):
            error_code = str(getattr(result, "error_code", ""))
            if error_code == "CANCEL_CLEANUP_TIMEOUT":
                raise PrimitiveFlowError(
                    "RELEASE_STATE_UNKNOWN",
                    str(getattr(result, "message", "")) or "open_gripper physical state is unknown",
                    terminal_known=False,
                )
            raise PrimitiveFlowError(
                error_code or "PRIMITIVE_FAILED",
                str(getattr(result, "message", "")) or "open_gripper primitive failed",
                terminal_known=False,
            )
        return time.monotonic()

    def _gripper_open_feedback_is_fresh(self, *, newer_than_monotonic: float) -> bool:
        with self._state_lock:
            joint_state = self._latest_joint_state
            receipt = self._latest_joint_receipt_monotonic
        if receipt <= newer_than_monotonic:
            return False
        if joint_state is None or self._gripper_joint_name not in joint_state.name:
            return False
        index = joint_state.name.index(self._gripper_joint_name)
        if index >= len(joint_state.position):
            return False
        position = float(joint_state.position[index])
        return math.isfinite(position) and abs(position - self._gripper_open) <= self._gripper_tolerance

    def _wait_for_open_feedback(self, open_completed: float, deadline: float) -> bool:
        timeout = float(self._config.get("sensor", {}).get("gripper_feedback_timeout_sec", 1.0))
        feedback_deadline = min(deadline, time.monotonic() + timeout)
        while rclpy.ok() and time.monotonic() < feedback_deadline:
            if self._gripper_open_feedback_is_fresh(newer_than_monotonic=open_completed):
                return True
            time.sleep(0.02)
        return False

    def _sleep_until(self, deadline: float, duration_sec: float) -> None:
        end = min(deadline, time.monotonic() + max(0.0, duration_sec))
        while rclpy.ok() and time.monotonic() < end:
            time.sleep(min(0.05, end - time.monotonic()))
        if time.monotonic() >= deadline:
            raise PlacementFlowError("PLACE_VERIFICATION_UNCERTAIN", "placement deadline expired before verification")

    def _wait_for_rgb(self, *, newer_than_ns: int, deadline: float) -> Image:
        while rclpy.ok() and time.monotonic() < deadline:
            with self._state_lock:
                image = self._latest_rgb
            if image is not None and self._stamp_ns(image) > int(newer_than_ns):
                return image
            time.sleep(0.02)
        raise PlacementFlowError("PLACE_VERIFICATION_UNCERTAIN", "fresh post-release RGB image is unavailable")

    def _detect(
        self,
        goal_handle,
        deadline: float,
        image: Image,
        query: str,
        *,
        honor_cancel: bool = False,
        exclusion_mask: np.ndarray | None = None,
    ):
        request_image = image
        if exclusion_mask is not None:
            request_image = self._image_with_exclusion(image, exclusion_mask)
        request = GroundingDetect.Request()
        request.image = request_image
        request.text_prompt = query
        threshold = float(self._config.get("verification", {}).get("confidence_threshold", 0.30))
        request.box_threshold = threshold
        request.text_threshold = threshold
        response = self._wait_future(
            self._detect_client.call_async(request),
            goal_handle,
            deadline,
            f"detect {query}",
            honor_cancel=honor_cancel,
        )
        if response is None or not bool(response.success):
            raise PlacementFlowError(
                "PLACE_VERIFICATION_UNCERTAIN",
                str(getattr(response, "message", "")) or f"detection failed for {query!r}",
            )
        detections = [
            detection for detection in response.detections.detections if float(detection.confidence) >= threshold
        ]
        if not detections or self._segment_client is None:
            return detections
        segment_request = SegmentDetections.Request()
        segment_request.image = request_image
        segment_request.detections = DetectionArray(header=response.detections.header, detections=detections)
        segmented = self._wait_future(
            self._segment_client.call_async(segment_request),
            goal_handle,
            deadline,
            f"segment {query}",
            honor_cancel=honor_cancel,
        )
        if segmented is None or not bool(segmented.success):
            raise PlacementFlowError(
                "PLACE_VERIFICATION_UNCERTAIN",
                str(getattr(segmented, "message", "")) or f"segmentation failed for {query!r}",
            )
        return [detection for detection in segmented.detections.detections if float(detection.confidence) >= threshold]

    @staticmethod
    def _stamp_ns(message) -> int:
        stamp = getattr(getattr(message, "header", None), "stamp", None)
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec) if stamp is not None else 0

    def _sample_verification(
        self,
        goal_handle,
        deadline: float,
        target_query: str,
        container_query: str,
        *,
        minimum_stamp_ns: int,
        debug_output_dir: str = "",
    ) -> tuple[bool | None, str, tuple[int, int] | None]:
        verification = self._config.get("verification", {})
        sample_index = 0
        image: Image | None = None
        try:
            image = self._wait_for_rgb(newer_than_ns=minimum_stamp_ns, deadline=deadline)
            sample_index = self._record_image_snapshot(image, debug_output_dir)
            exclusion_mask = self._target_exclusion_mask(image)
            self._record_exclusion_snapshot(exclusion_mask, debug_output_dir, sample_index)
            self._record_masked_image_snapshot(image, exclusion_mask, debug_output_dir, sample_index)
            containers = self._detect(goal_handle, deadline, image, container_query)
            if exclusion_mask is None:
                targets = self._detect(goal_handle, deadline, image, target_query)
            else:
                targets = self._detect(
                    goal_handle,
                    deadline,
                    image,
                    target_query,
                    exclusion_mask=exclusion_mask,
                )
            targets = self._filter_target_detections(targets, exclusion_mask)
            # Keep all candidates for evidence, but make the confidence-led
            # choice deterministic and align sample_XX_*_00_mask.npy with the
            # candidate actually used for containment verification.
            containers = sorted(containers, key=lambda detection: float(detection.confidence), reverse=True)
            targets = sorted(targets, key=lambda detection: float(detection.confidence), reverse=True)
        except PlacementFlowError as exc:
            if debug_output_dir and sample_index and image is not None:
                try:
                    self._write_debug_json(
                        Path(debug_output_dir) / f"sample_{sample_index:02d}.json",
                        {
                            "sample_index": sample_index,
                            "stamp_ns": self._stamp_ns(image),
                            "outcome": None,
                            "detail": str(exc),
                        },
                    )
                except (OSError, TypeError, ValueError) as write_exc:
                    self.get_logger().warning(f"failed to save uncertain placement sample: {write_exc}")
            return None, str(exc), None
        if not containers:
            detail = "no visible container was detected"
            self._record_verification_sample(
                image=image,
                containers=containers,
                targets=targets,
                outcome=None,
                detail=detail,
                debug_output_dir=debug_output_dir,
                sample_index=sample_index,
            )
            return None, detail, (image_stamp_ns := self._stamp_ns(image), image_stamp_ns)
        if not targets:
            detail = "no visible target object was detected"
            self._record_verification_sample(
                image=image,
                containers=containers,
                targets=targets,
                outcome=None,
                detail=detail,
                debug_output_dir=debug_output_dir,
                sample_index=sample_index,
            )
            return None, detail, (image_stamp_ns := self._stamp_ns(image), image_stamp_ns)
        # Grounding-DINO may return several candidates for the same visible
        # object (or several physical target objects).  Placement verification
        # is intentionally confidence-led: retain every candidate in the
        # evidence record, but use the highest-confidence container and target
        # for the containment decision instead of treating multiplicity as an
        # uncertain result.
        selected_container = max(containers, key=lambda detection: float(detection.confidence))
        selected_target = max(targets, key=lambda detection: float(detection.confidence))
        image_stamp_ns = self._stamp_ns(image)
        sample_key = (image_stamp_ns, image_stamp_ns)

        minimum_container_pixels = int(verification.get("min_container_mask_pixels", 500))
        try:
            container_pixels = int(np.count_nonzero(_mask_to_bool(selected_container.mask)))
        except ValueError as exc:
            self._record_verification_sample(
                image=image,
                containers=containers,
                targets=targets,
                outcome=None,
                detail=str(exc),
                debug_output_dir=debug_output_dir,
                sample_index=sample_index,
            )
            return None, str(exc), sample_key
        if container_pixels < minimum_container_pixels:
            detail = f"container mask is too small: {container_pixels} pixels"
            self._record_verification_sample(
                image=image,
                containers=containers,
                targets=targets,
                outcome=None,
                detail=detail,
                debug_output_dir=debug_output_dir,
                sample_index=sample_index,
            )
            return None, detail, sample_key

        try:
            result = evaluate_mask_containment(
                selected_container.mask,
                selected_target.mask,
                min_target_pixels=int(verification.get("min_target_mask_pixels", 100)),
                min_inside_fraction=float(verification.get("min_inside_mask_fraction", 0.70)),
                container_inset_ratio=float(verification.get("container_inset_ratio", 0.05)),
            )
        except ValueError as exc:
            self._record_verification_sample(
                image=image,
                containers=containers,
                targets=targets,
                outcome=None,
                detail=str(exc),
                debug_output_dir=debug_output_dir,
                sample_index=sample_index,
            )
            return None, str(exc), sample_key
        if result.target_pixel_count < int(verification.get("min_target_mask_pixels", 100)):
            detail = "target masks were too small for a reliable decision"
            self._record_verification_sample(
                image=image,
                containers=containers,
                targets=targets,
                outcome=None,
                detail=detail,
                debug_output_dir=debug_output_dir,
                sample_index=sample_index,
            )
            return None, detail, sample_key
        detail = (
            f"containers={len(containers)} selected_container_confidence={float(selected_container.confidence):.3f} "
            f"targets={len(targets)} selected_target_confidence={float(selected_target.confidence):.3f} "
            f"target_pixels={result.target_pixel_count} inside_pixels={result.inside_pixel_count} "
            f"inside_fraction={result.inside_fraction:.3f} center_inside={result.center_inside}"
        )
        outcome = result.inside
        self._record_verification_sample(
            image=image,
            containers=containers,
            targets=targets,
            outcome=outcome,
            detail=detail,
            debug_output_dir=debug_output_dir,
            sample_index=sample_index,
        )
        return outcome, detail, sample_key

    def _verify_post_release(
        self,
        goal_handle,
        deadline: float,
        target_query: str,
        container_query: str,
        state: PlacementState,
    ) -> int:
        verification = self._config.get("verification", {})
        wait_timeout = min(self._rpc_timeout, max(0.1, deadline - time.monotonic()))
        if not self._detect_client.wait_for_service(timeout_sec=wait_timeout):
            state.diagnostic_details.append("verification: detection service unavailable after release")
            return PlaceObject.Result.VERIFICATION_UNCERTAIN
        if self._segment_client is not None and not self._segment_client.wait_for_service(timeout_sec=wait_timeout):
            state.diagnostic_details.append("verification: segmentation service unavailable after release")
            return PlaceObject.Result.VERIFICATION_UNCERTAIN
        attempts = int(verification.get("max_resamples", 3)) + 1
        required = int(verification.get("required_confirmations", 2))
        previous: bool | None = None
        previous_sample_key: tuple[int, int] | None = None
        consecutive = 0
        minimum_stamp_ns = max(state.release_completed_ros_ns, state.verification_started_ros_ns)
        for attempt in range(1, attempts + 1):
            sample_kwargs = {"minimum_stamp_ns": minimum_stamp_ns}
            if state.debug_output_dir:
                sample_kwargs["debug_output_dir"] = state.debug_output_dir
            outcome, detail, sample_key = self._sample_verification(
                goal_handle, deadline, target_query, container_query, **sample_kwargs
            )
            state.diagnostic_details.append(f"verification_sample_{attempt}: {detail}")
            if sample_key is not None:
                minimum_stamp_ns = max(minimum_stamp_ns, *sample_key)
            if sample_key is not None and sample_key == previous_sample_key:
                state.diagnostic_details.append(f"verification_sample_{attempt}: repeated image stamps ignored")
            elif outcome is None:
                previous = None
                consecutive = 0
            else:
                if outcome == previous:
                    consecutive += 1
                else:
                    previous = outcome
                    consecutive = 1
                previous_sample_key = sample_key
            if consecutive >= required:
                return (
                    PlaceObject.Result.VERIFICATION_SUCCESS
                    if outcome is True
                    else PlaceObject.Result.VERIFICATION_FAILED
                )
            if attempt < attempts:
                self._sleep_until(deadline, float(verification.get("resample_interval_sec", 0.25)))
        return PlaceObject.Result.VERIFICATION_UNCERTAIN

    def _result(self, state: PlacementState, *, success: bool, code: str = "", message: str = ""):
        state.finish()
        result = PlaceObject.Result()
        result.success = bool(success)
        result.place_succeeded = bool(state.place_succeeded)
        result.release_status = int(state.release_status)
        result.verification_status = int(state.verification_status)
        result.error_code = code
        result.message = message
        result.completed_phases = list(state.completed_phases)
        result.diagnostic_details = list(state.diagnostic_details)
        result.pipeline_timings_json = json.dumps(state.pipeline_timings, sort_keys=True)
        result.debug_output_dir = state.debug_output_dir
        if state.debug_output_dir:
            try:
                self._write_debug_json(
                    Path(state.debug_output_dir) / "placement_result.json",
                    {
                        "schema_version": 1,
                        "pipeline": "placement_pipeline",
                        "pipeline_version": 3,
                        "target_query": state.target_query,
                        "container_query": state.container_query,
                        "success": bool(success),
                        "place_succeeded": bool(state.place_succeeded),
                        "release_status": int(state.release_status),
                        "verification_status": int(state.verification_status),
                        "error_code": str(code),
                        "message": str(message),
                        "completed_phases": list(state.completed_phases),
                        "diagnostic_details": list(state.diagnostic_details),
                        "pipeline_timings": dict(state.pipeline_timings),
                    },
                )
            except (OSError, TypeError, ValueError) as exc:
                self.get_logger().warning(f"failed to save placement result evidence: {exc}")
        fill_delegated_executor_identity(result.actual_executor, self._executor_identity)
        return result

    def _return_to_release_pose(
        self,
        goal_handle,
        deadline: float,
        task_id: str,
        state: PlacementState,
        *,
        joint_names: list[str],
        joint_positions: list[float],
        changed_joint_name: str,
        duration_sec: float,
    ) -> None:
        self._feedback(
            goal_handle,
            state,
            "return_to_place",
            f"returning joint {changed_joint_name} to the fixed release position",
            honor_cancel=False,
        )
        try:
            self._move_to_joint_positions(
                goal_handle,
                deadline,
                task_id,
                joint_names,
                joint_positions,
                duration_sec=duration_sec,
                honor_cancel=False,
            )
        except PlacementFlowError as exc:
            if exc.code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                raise
            raise PlacementFlowError(
                "PLACE_RETURN_FAILED",
                f"could not return joint {changed_joint_name} to the fixed release position: {exc}",
            ) from exc

    def _execute_place(self, goal_handle):
        goal = goal_handle.request
        state = PlacementState()
        task_deadline_unix = (
            goal.dispatch_binding.task_budget.deadline.sec
            + goal.dispatch_binding.task_budget.deadline.nanosec / 1_000_000_000
        )
        remaining_budget = task_deadline_unix - self.get_clock().now().nanoseconds / 1_000_000_000
        timeout_sec = min(float(goal.timeout_sec), remaining_budget)
        if timeout_sec <= 0.0:
            goal_handle.abort()
            result = self._result(
                state,
                success=False,
                code="TASK_TIMEOUT",
                message="shared task budget expired before placement execution",
            )
            with self._goal_lock:
                self._goal_active = False
                self._dispatch_binding = None
            return result
        deadline = time.monotonic() + timeout_sec
        task_id = str(goal.dispatch_binding.task_id).strip()
        target_query = str(goal.target_query).strip()
        container_query = str(goal.container_query).strip()
        motion = self._config.get("motion", {})
        place_pose = str(motion.get("place_pose", "place_container")).strip()
        place_joint_names = [str(name) for name in motion.get("place_joint_names", [])]
        place_joint_mapping = motion.get("place_joint_positions", {})
        place_duration_sec = float(motion.get("place_duration_sec", 10.0))
        post_release = motion.get("post_release", {})
        verify_joint_name = str(post_release.get("verify_joint_name", "")).strip()
        verify_joint_position = float(post_release.get("verify_joint_position", 0.0))
        verify_duration_sec = float(post_release.get("verify_duration_sec", 2.0))
        return_duration_sec = float(post_release.get("return_duration_sec", 2.0))
        release_joint_positions = [float(place_joint_mapping[name]) for name in place_joint_names]
        verify_joint_positions = list(release_joint_positions)
        verify_joint_positions[place_joint_names.index(verify_joint_name)] = verify_joint_position
        pre_return_deadline = deadline - return_duration_sec - self._rpc_timeout
        try:
            self._start_debug_record(state, task_id, target_query, container_query)
            self._preflight(goal_handle, deadline, state)
            self._feedback(goal_handle, state, "move_to_place", f"moving to fixed release pose {place_pose}")
            try:
                self._move_to_joint_positions(
                    goal_handle,
                    deadline,
                    task_id,
                    place_joint_names,
                    release_joint_positions,
                    duration_sec=place_duration_sec,
                    honor_cancel=True,
                )
            except PlacementCancelled:
                raise
            except PlacementFlowError as exc:
                if exc.code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                    raise
                raise PlacementFlowError("PLACE_POSITIONING_FAILED", f"could not reach {place_pose}: {exc}") from exc

            self._feedback(goal_handle, state, "release", f"opening the gripper at {place_pose}")
            try:
                open_completed = self._open_gripper(goal_handle, deadline, task_id)
            except PrimitiveFlowError as exc:
                state.release_status = (
                    PlaceObject.Result.RELEASE_NOT_RELEASED
                    if exc.terminal_known
                    else PlaceObject.Result.RELEASE_UNKNOWN
                )
                raise
            if not self._wait_for_open_feedback(open_completed, deadline):
                state.release_status = PlaceObject.Result.RELEASE_UNKNOWN
                raise PlacementFlowError(
                    "RELEASE_STATE_UNKNOWN", "fresh joint feedback did not confirm the gripper open target"
                )
            self._record_open_feedback(state)
            state.release_status = PlaceObject.Result.RELEASE_RELEASED
            state.release_completed_ros_ns = int(self.get_clock().now().nanoseconds)
            state.verification_status = PlaceObject.Result.VERIFICATION_UNCERTAIN

            self._feedback(
                goal_handle,
                state,
                "move_to_verify",
                f"moving only joint {verify_joint_name} to the visual verification position",
                honor_cancel=False,
            )
            try:
                self._move_to_joint_positions(
                    goal_handle,
                    pre_return_deadline,
                    task_id,
                    place_joint_names,
                    verify_joint_positions,
                    duration_sec=verify_duration_sec,
                    honor_cancel=False,
                )
            except PlacementFlowError as exc:
                if exc.code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT:
                    raise
                self._return_to_release_pose(
                    goal_handle,
                    deadline,
                    task_id,
                    state,
                    joint_names=place_joint_names,
                    joint_positions=release_joint_positions,
                    changed_joint_name=verify_joint_name,
                    duration_sec=return_duration_sec,
                )
                raise PlacementFlowError(
                    "PLACE_VERIFICATION_POSITIONING_FAILED",
                    f"could not move joint {verify_joint_name} to the visual verification position: {exc}",
                ) from exc

            verification_error: Exception | None = None
            try:
                self._sleep_until(
                    pre_return_deadline,
                    float(self._config.get("verification", {}).get("post_release_wait_sec", 1.0)),
                )
                state.verification_started_ros_ns = int(self.get_clock().now().nanoseconds)
                self._feedback(
                    goal_handle,
                    state,
                    "verify_place",
                    f"checking the released object with joint {verify_joint_name} at the verification position",
                    honor_cancel=False,
                )
                state.verification_status = self._verify_post_release(
                    goal_handle,
                    pre_return_deadline,
                    target_query,
                    container_query,
                    state,
                )
                state.place_succeeded = state.verification_status == PlaceObject.Result.VERIFICATION_SUCCESS
            except Exception as exc:
                verification_error = exc

            try:
                self._return_to_release_pose(
                    goal_handle,
                    deadline,
                    task_id,
                    state,
                    joint_names=place_joint_names,
                    joint_positions=release_joint_positions,
                    changed_joint_name=verify_joint_name,
                    duration_sec=return_duration_sec,
                )
            except PlacementFlowError:
                if verification_error is not None:
                    state.diagnostic_details.append(f"verification before return failed: {verification_error}")
                raise

            if verification_error is not None:
                raise verification_error
            if state.verification_status != PlaceObject.Result.VERIFICATION_SUCCESS:
                code = (
                    "PLACE_VERIFICATION_FAILED"
                    if state.verification_status == PlaceObject.Result.VERIFICATION_FAILED
                    else "PLACE_VERIFICATION_UNCERTAIN"
                )
                raise PlacementFlowError(code, "gripper opened, but placement inside the container was not confirmed")

            self._feedback(
                goal_handle,
                state,
                "completed",
                "object released, visually confirmed, and joint returned to the release position",
                honor_cancel=False,
            )
            goal_handle.succeed()
            return self._result(
                state,
                success=True,
                message="object released and verified inside the container; verification joint returned",
            )
        except PlacementCancelled as exc:
            goal_handle.canceled()
            return self._result(state, success=False, code=exc.code, message=str(exc))
        except PlacementFlowError as exc:
            goal_handle.abort()
            return self._result(state, success=False, code=exc.code, message=str(exc))
        except Exception as exc:
            self.get_logger().exception("unexpected placement execution failure")
            goal_handle.abort()
            return self._result(state, success=False, code="PLACE_FAILED", message=str(exc))
        finally:
            with self._goal_lock:
                self._goal_active = False
                self._dispatch_binding = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlacementExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

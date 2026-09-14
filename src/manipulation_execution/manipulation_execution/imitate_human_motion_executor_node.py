"""ROS Action server for the internal imitate_human_motion executor."""

from __future__ import annotations

import contextlib
import json
import math
import queue
import threading
import time

import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from embodied_common.dispatch_binding import (
    copy_binding,
    delegated_executor_identity,
    delegated_executor_identity_matches,
    fill_delegated_executor_identity,
)
from ibrobot_msgs.action import ImitateHumanMotion, PrimitiveCommand
from ibrobot_msgs.msg import Detection2D, DetectionArray
from ibrobot_msgs.srv import PearParameterPredict, YoloXDetect
from manipulation_execution.imitate_human_motion_executor import (
    CANCEL_CLEANUP_TIMEOUT,
    AnimationPlan,
    MockExecutor,
    MockGoal,
    MockResult,
    PrimitiveStateUnknown,
)

_PREPARE_DURATION_SEC = 2.0
_PEAR_CROP_EXPANSION = 1.25


def _validated_person_confidence_threshold(value: object) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("person_confidence_threshold must be finite and within [0, 1]")
    return threshold


def _validated_yolox_refresh_interval(value: object) -> float:
    interval = float(value)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("yolox_refresh_interval_sec must be finite and greater than zero")
    return interval


def _select_person_detection(detections: list[Detection2D], threshold: float) -> Detection2D | None:
    persons = [
        detection
        for detection in detections
        if detection.label == "person"
        and math.isfinite(float(detection.confidence))
        and float(detection.confidence) >= threshold
    ]
    return max(persons, key=lambda detection: float(detection.confidence), default=None)


def _center_fallback_detection(image: Image) -> Detection2D:
    width = int(image.width)
    height = int(image.height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    center_x = width / 2.0
    center_y = height / 2.0
    half_side = min(width, height) / _PEAR_CROP_EXPANSION / 2.0
    detection = Detection2D()
    detection.header = image.header
    detection.label = "person"
    detection.confidence = 0.0
    detection.bbox = [
        center_x - half_side,
        center_y - half_side,
        center_x + half_side,
        center_y + half_side,
    ]
    return detection


class _GuardedPrimitivePlayer:
    """Translate the HRI lifecycle into the existing guarded primitives."""

    def __init__(
        self,
        client,
        goal_handle,
        *,
        joint_names: tuple[str, ...],
        reset_positions: dict[str, float],
        rpc_timeout_sec: float,
        deadline: float,
        ros_now_sec,
    ) -> None:
        self._client = client
        self._goal_handle = goal_handle
        self._binding = copy_binding(goal_handle.request.dispatch_binding)
        self._joint_names = joint_names
        self._reset_positions = reset_positions
        self._rpc_timeout = rpc_timeout_sec
        self._ros_now_sec = ros_now_sec
        budget = self._binding.task_budget
        self._task_deadline_ros = budget.deadline.sec + budget.deadline.nanosec / 1_000_000_000
        self._deadline = self._bounded_deadline(deadline)
        raw_uuid = getattr(getattr(goal_handle, "goal_id", None), "uuid", None)
        self._execution_token = bytes(raw_uuid).hex() if raw_uuid is not None else ""

    @property
    def deadline(self) -> float:
        """The task deadline after the dispatch budget ceiling has been applied."""
        return self._deadline

    def _bounded_deadline(self, requested_deadline: float) -> float:
        remaining_budget = max(0.0, self._task_deadline_ros - self._ros_now_sec())
        return min(requested_deadline, time.monotonic() + remaining_budget)

    def _wait(self, future, deadline: float, *, honor_cancel: bool) -> bool:
        while not future.done():
            if honor_cancel and self._goal_handle.is_cancel_requested:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def _cancel(self, handle, result_future=None) -> bool:
        try:
            result_future = result_future or handle.get_result_async()
            cancel_future = handle.cancel_goal_async()
        except Exception:
            return False
        cleanup_deadline = self._bounded_deadline(time.monotonic() + self._rpc_timeout)
        if not self._wait(cancel_future, cleanup_deadline, honor_cancel=False):
            return False
        try:
            response = cancel_future.result()
            if response is None or not bool(response.goals_canceling):
                return False
            if not self._wait(result_future, cleanup_deadline, honor_cancel=False):
                return False
            wrapped = result_future.result()
        except Exception:
            return False
        result = getattr(wrapped, "result", None)
        return result is not None and str(getattr(result, "error_code", "")) != CANCEL_CLEANUP_TIMEOUT

    def _cancel_pending(self, send_future) -> bool:
        cleanup_deadline = self._bounded_deadline(time.monotonic() + self._rpc_timeout)
        if not self._wait(send_future, cleanup_deadline, honor_cancel=False):
            send_future.add_done_callback(self._cancel_when_ready)
            return False
        try:
            handle = send_future.result()
        except Exception:
            return False
        return handle is None or not handle.accepted or self._cancel(handle)

    def _cancel_when_ready(self, send_future) -> None:
        try:
            handle = send_future.result()
        except Exception:
            return
        if handle is not None and handle.accepted:
            self._cancel(handle)

    def _run(
        self,
        *,
        primitive_name: str,
        deadline: float,
        duration_sec: float = 0.0,
        joint_positions: tuple[float, ...] = (),
        pose_name: str = "",
        honor_cancel: bool = True,
    ) -> str:
        deadline = self._bounded_deadline(deadline)
        wait_sec = min(self._rpc_timeout, max(0.0, deadline - time.monotonic()))
        if wait_sec <= 0.0 or not self._client.wait_for_server(timeout_sec=wait_sec):
            return "TIMEOUT" if time.monotonic() >= deadline else "FAILED"
        goal = PrimitiveCommand.Goal()
        goal.schema_version = 1
        goal.dispatch_binding = copy_binding(self._binding)
        goal.execution_token = self._execution_token
        goal.primitive_name = primitive_name
        goal.pose_name = pose_name
        goal.joint_names = list(self._joint_names) if joint_positions else []
        goal.joint_positions = list(joint_positions)
        goal.primitive_duration_sec = float(duration_sec)
        goal.timeout_sec = deadline - time.monotonic()
        try:
            send_future = self._client.send_goal_async(goal)
        except Exception:
            return "UNKNOWN"
        if not self._wait(send_future, deadline, honor_cancel=honor_cancel):
            requested_cancel = honor_cancel and self._goal_handle.is_cancel_requested
            if not self._cancel_pending(send_future):
                return "UNKNOWN"
            return "CANCELED" if requested_cancel else "TIMEOUT"
        try:
            handle = send_future.result()
        except Exception:
            return "UNKNOWN"
        if handle is None or not handle.accepted:
            return "FAILED"
        try:
            result_future = handle.get_result_async()
        except Exception:
            return "UNKNOWN"
        if not self._wait(result_future, deadline, honor_cancel=honor_cancel):
            requested_cancel = honor_cancel and self._goal_handle.is_cancel_requested
            if not self._cancel(handle, result_future):
                return "UNKNOWN"
            return "CANCELED" if requested_cancel else "TIMEOUT"
        try:
            wrapped = result_future.result()
        except Exception:
            return "UNKNOWN"
        result = getattr(wrapped, "result", None)
        if result is None or str(getattr(result, "error_code", "")) == CANCEL_CLEANUP_TIMEOUT:
            return "UNKNOWN"
        return "COMPLETED" if bool(result.success) else "FAILED"

    def prepare(self) -> bool:
        outcome = self._run(
            primitive_name="move_to_joint_positions",
            joint_positions=tuple(self._reset_positions[name] for name in self._joint_names),
            duration_sec=_PREPARE_DURATION_SEC,
            deadline=self._deadline,
        )
        if outcome == "UNKNOWN":
            raise PrimitiveStateUnknown("prepare primitive execution state is unknown")
        return outcome == "COMPLETED"

    def play(self, plan: AnimationPlan, duration_sec: float, *, feedback, is_cancel_requested, deadline) -> str:
        segment_duration = plan.duration_sec / (len(plan.waypoints) - 1)
        remaining = duration_sec
        for start, end in zip(plan.waypoints, plan.waypoints[1:], strict=True):
            if remaining <= 0.0:
                break
            active_duration = min(segment_duration, remaining)
            ratio = active_duration / segment_duration
            target = tuple(a + (b - a) * ratio for a, b in zip(start, end, strict=True))
            outcome = self._run(
                primitive_name="move_to_joint_positions",
                joint_positions=target,
                duration_sec=active_duration,
                deadline=deadline,
            )
            if outcome != "COMPLETED":
                return outcome
            remaining -= active_duration
            progress = min(1.0, (duration_sec - remaining) / duration_sec)
            feedback("mock_playback", progress, f"Executing {plan.animation_id}")
        return "COMPLETED"

    def reset(self) -> bool:
        outcome = self._run(
            primitive_name="move_to_named_pose",
            pose_name="home",
            deadline=self._bounded_deadline(time.monotonic() + max(30.0, self._rpc_timeout)),
            honor_cancel=False,
        )
        if outcome == "UNKNOWN":
            raise PrimitiveStateUnknown("reset primitive execution state is unknown")
        return outcome == "COMPLETED"


class ImitateHumanMotionExecutorNode(Node):
    """Serve the internal delegated HRI Action as a launch-managed runtime."""

    def __init__(self, parameter_overrides=None) -> None:
        super().__init__("imitate_human_motion_executor_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("action_name", "/hri/imitate_human_motion")
        self.declare_parameter("primitive_action_name", "/embodied/execute_primitive")
        self.declare_parameter("rpc_timeout_sec", 5.0)
        self.declare_parameter("startup_warmup", True)
        self.declare_parameter("arm_joint_names_json", "[]")
        self.declare_parameter("reset_positions_json", "{}")
        self.declare_parameter("joint_limits_json", "{}")
        self.declare_parameter("rgb_topic", "/camera/wrist/image_raw")
        self.declare_parameter("person_confidence_threshold", 0.30)
        self.declare_parameter("yolox_refresh_interval_sec", 0.25)
        self.declare_parameter("yolox_detect_service", "/perception/hri/yolox_detect")
        self.declare_parameter("pear_parameters_service", "/perception/hri/pear_parameters")
        self._action_name = str(self.get_parameter("action_name").value)
        primitive_action_name = str(self.get_parameter("primitive_action_name").value)
        self._rpc_timeout = float(self.get_parameter("rpc_timeout_sec").value)
        self._startup_warmup = bool(self.get_parameter("startup_warmup").value)
        self._joint_names = tuple(str(name) for name in json.loads(self.get_parameter("arm_joint_names_json").value))
        self._reset_positions = {
            str(name): float(value)
            for name, value in json.loads(self.get_parameter("reset_positions_json").value).items()
        }
        self._joint_limits = json.loads(self.get_parameter("joint_limits_json").value)
        self._rgb_topic = str(self.get_parameter("rgb_topic").value).strip()
        if not self._rgb_topic:
            raise ValueError("rgb_topic must be non-empty")
        self._person_confidence_threshold = _validated_person_confidence_threshold(
            self.get_parameter("person_confidence_threshold").value
        )
        self._yolox_refresh_interval_sec = _validated_yolox_refresh_interval(
            self.get_parameter("yolox_refresh_interval_sec").value
        )
        self._yolox_service = str(self.get_parameter("yolox_detect_service").value).strip()
        self._pear_service = str(self.get_parameter("pear_parameters_service").value).strip()
        for name, value in (
            ("yolox_detect_service", self._yolox_service),
            ("pear_parameters_service", self._pear_service),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        self._executor_identity = delegated_executor_identity(
            name="imitate_human_motion",
            endpoint_name=self._action_name,
            configuration={"implementation": "mock_v1"},
        )
        self._mock = MockExecutor(
            joint_names=self._joint_names,
            reset_positions=self._reset_positions,
            joint_limits=self._joint_limits,
            warmup_ready=False,
        )
        self._startup_warmup_attempted = False
        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._capture_lock = threading.Lock()
        self._capture_active = False
        self._capture_epoch = 0
        self._capture_deadline = 0.0
        self._vision_ok = 0
        self._vision_detected = 0
        self._vision_fallback = 0
        self._vision_failed = 0
        self._vision_sample = ""
        self._frame_count = 0
        self._first_frame_stamp = ""
        self._last_frame_stamp = ""
        self._last_frame_id = ""
        self._last_frame_received_at = 0.0
        # PEAR consumes the latest available image while YOLOX refreshes the
        # cached bbox at a lower rate on its own queue.
        self._vision_queue = queue.Queue(maxsize=1)
        self._yolox_queue = queue.Queue(maxsize=1)
        self._yolox_detection: Detection2D | None = None
        self._yolox_result_ready = False
        self._yolox_calls = 0
        self._pear_calls = 0
        self._vision_stop = threading.Event()
        self._yolox_client = self.create_client(YoloXDetect, self._yolox_service)
        self._pear_client = self.create_client(PearParameterPredict, self._pear_service)
        self._vision_worker = threading.Thread(target=self._vision_loop, name="hri-pear", daemon=True)
        self._yolox_worker = threading.Thread(target=self._yolox_loop, name="hri-yolox", daemon=True)
        self._vision_worker.start()
        self._yolox_worker.start()
        self._rgb_subscription = self.create_subscription(
            Image,
            self._rgb_topic,
            self._on_rgb_frame,
            qos_profile_sensor_data,
        )
        callback_group = ReentrantCallbackGroup()
        self._primitive_client = ActionClient(
            self,
            PrimitiveCommand,
            primitive_action_name,
            callback_group=callback_group,
        )
        self._action_server = ActionServer(
            self,
            ImitateHumanMotion,
            self._action_name,
            execute_callback=self._execute,
            goal_callback=self._handle_goal,
            cancel_callback=self._handle_cancel,
            callback_group=callback_group,
        )
        self.get_logger().info(f"imitate_human_motion executor ready: action={self._action_name}")
        self.get_logger().info(f"imitate_human_motion RGB input: topic={self._rgb_topic}")
        if self._startup_warmup:
            self._run_startup_warmup()

    def _on_rgb_frame(self, message: Image) -> None:
        """Record RGB frames only while an HRI imitation task is active."""
        with self._capture_lock:
            if not self._capture_active:
                return
            now = time.monotonic()
            if now >= self._capture_deadline:
                self._capture_active = False
                return
            stamp = message.header.stamp
            stamp_text = f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"
            self._frame_count += 1
            if not self._first_frame_stamp:
                self._first_frame_stamp = stamp_text
            self._last_frame_stamp = stamp_text
            self._last_frame_id = str(message.header.frame_id)
            self._last_frame_received_at = now
            frame = (self._capture_epoch, message)
            self._put_latest(self._vision_queue, frame)
            self._put_latest(self._yolox_queue, frame)

    @staticmethod
    def _put_latest(frame_queue, frame) -> None:
        while True:
            try:
                frame_queue.put_nowait(frame)
                return
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    continue

    _PEAR_OUTPUT_FIELDS = (
        "smplx_pose_raw",
        "smplx_scale",
        "smplx_shape",
        "smplx_expression",
        "flame_pose",
        "flame_shape",
        "flame_expression",
        "camera_raw",
    )

    def _vision_loop(self) -> None:
        while not self._vision_stop.is_set():
            try:
                epoch, message = self._vision_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._capture_lock:
                # A frame queued for an earlier task must never be attributed to this
                # one. Draining at _begin_capture is not enough on its own: the worker
                # may already be blocked inside a call when the next task starts, and
                # the frame it is holding predates that task.
                deadline = self._capture_deadline
                expired = time.monotonic() >= deadline
                stale = epoch != self._capture_epoch or not self._capture_active or expired
                if expired:
                    self._capture_active = False
            if stale:
                continue
            try:
                self._process_pear_frame(message, epoch, deadline)
            except Exception as exc:
                self._record_vision_failure(epoch, f"{type(exc).__name__}: {exc}")

    def _yolox_loop(self) -> None:
        next_refresh = 0.0
        scheduled_epoch = None
        pending_frame = None
        while not self._vision_stop.is_set():
            timeout = 0.1
            if pending_frame is not None:
                timeout = min(timeout, max(0.0, next_refresh - time.monotonic()))
            with contextlib.suppress(queue.Empty):
                pending_frame = self._yolox_queue.get(timeout=timeout)
            if pending_frame is None:
                continue
            epoch, message = pending_frame
            if epoch != scheduled_epoch:
                scheduled_epoch = epoch
                next_refresh = 0.0
            if time.monotonic() < next_refresh:
                continue
            pending_frame = None
            with self._capture_lock:
                deadline = self._capture_deadline
                expired = time.monotonic() >= deadline
                stale = epoch != self._capture_epoch or not self._capture_active or expired
                if expired:
                    self._capture_active = False
            if stale:
                continue
            refresh_started = time.monotonic()
            try:
                self._process_yolox_frame(message, epoch, deadline)
            except Exception as exc:
                self._record_vision_failure(epoch, f"{type(exc).__name__}: {exc}")
            next_refresh = refresh_started + self._yolox_refresh_interval_sec

    def _process_yolox_frame(self, message: Image, epoch: int, deadline: float) -> bool:
        """Refresh the task-local bbox cache. Preprocessing lives in model adapters."""
        try:
            _center_fallback_detection(message)
        except ValueError as exc:
            self._record_vision_failure(epoch, str(exc))
            return False
        request = YoloXDetect.Request()
        request.image = message
        request.confidence_threshold = self._person_confidence_threshold
        request.nms_threshold = 0.65
        response = self._call_vision(self._yolox_client, request, epoch, deadline, metric="yolox")
        if response is None:
            return False
        if not response.success:
            self._record_vision_failure(epoch, f"yolox: {response.message}")
            return False
        detection = _select_person_detection(list(response.detections.detections), self._person_confidence_threshold)
        cached_detection = None
        if detection is not None:
            cached_detection = Detection2D()
            cached_detection.header = message.header
            cached_detection.label = detection.label
            cached_detection.confidence = detection.confidence
            cached_detection.bbox = [float(value) for value in detection.bbox]
        with self._capture_lock:
            if epoch != self._capture_epoch or not self._capture_active or time.monotonic() >= deadline:
                return False
            self._yolox_detection = cached_detection
            self._yolox_result_ready = True
        return True

    def _process_pear_frame(self, message: Image, epoch: int, deadline: float) -> None:
        """Run PEAR on the newest RGB frame using the cached YOLOX result."""
        try:
            fallback_detection = _center_fallback_detection(message)
        except ValueError as exc:
            self._record_vision_failure(epoch, str(exc))
            return
        with self._capture_lock:
            if (
                epoch != self._capture_epoch
                or not self._capture_active
                or time.monotonic() >= deadline
                or not self._yolox_result_ready
            ):
                return
            cached_detection = self._yolox_detection
        if cached_detection is None:
            detection = fallback_detection
            source = "center_fallback"
        else:
            detection = Detection2D()
            detection.header = message.header
            detection.label = cached_detection.label
            detection.confidence = cached_detection.confidence
            detection.bbox = [float(value) for value in cached_detection.bbox]
            source = "yolox"

        pear_request = PearParameterPredict.Request()
        pear_request.image = message
        # The compiled PEAR deployment is batch 1: exactly one person crop per call.
        pear_request.detections = DetectionArray(header=message.header, detections=[detection])
        result = self._call_vision(self._pear_client, pear_request, epoch, deadline, metric="pear")
        if result is None:
            return
        if not result.success:
            self._record_vision_failure(epoch, f"pear: {result.message}")
            return

        values = {name: list(getattr(result, name)) for name in self._PEAR_OUTPUT_FIELDS}
        missing = [name for name, value in values.items() if not value]
        nonfinite = [name for name, value in values.items() if any(not math.isfinite(float(v)) for v in value)]
        if missing or nonfinite:
            self._record_vision_failure(epoch, f"pear returned empty={missing} nonfinite={nonfinite}")
            return

        # Sampled values, not just lengths: a length check passes on an all-zero
        # response, which is exactly the failure this log exists to catch.
        sample = "source={} bbox=[{}] conf={:.4f} pose[0:4]={} camera={} scale[0:3]={} latency={:.1f}ms".format(
            source,
            ", ".join(f"{float(value):.1f}" for value in detection.bbox),
            float(detection.confidence),
            [round(float(value), 4) for value in values["smplx_pose_raw"][:4]],
            [round(float(value), 4) for value in values["camera_raw"]],
            [round(float(value), 4) for value in values["smplx_scale"][:3]],
            float(result.inference_time_ms),
        )
        with self._capture_lock:
            if epoch != self._capture_epoch or not self._capture_active or time.monotonic() >= deadline:
                return
            self._vision_ok += 1
            if source == "yolox":
                self._vision_detected += 1
            else:
                self._vision_fallback += 1
            self._vision_sample = sample
        self.get_logger().info(f"hri vision {sample}")

    def _process_vision_frame(self, message: Image, epoch: int, deadline: float) -> None:
        """Run one synchronous YOLOX -> PEAR pass for isolated test callers."""
        if self._process_yolox_frame(message, epoch, deadline):
            self._process_pear_frame(message, epoch, deadline)

    def _call_vision(self, client, request, epoch: int, deadline: float, *, metric: str = ""):
        """Call a vision service without misreporting task expiry as RPC timeout."""
        started_at = time.monotonic()
        if started_at >= deadline:
            return None
        if not client.service_is_ready():
            self._record_vision_failure(epoch, f"{client.srv_name} is not available")
            return None
        with self._capture_lock:
            if epoch != self._capture_epoch or not self._capture_active or time.monotonic() >= deadline:
                return None
            future = client.call_async(request)
            if metric == "yolox":
                self._yolox_calls = getattr(self, "_yolox_calls", 0) + 1
            elif metric == "pear":
                self._pear_calls = getattr(self, "_pear_calls", 0) + 1
        rpc_deadline = started_at + self._rpc_timeout
        call_deadline = min(rpc_deadline, deadline)
        while not future.done():
            if self._vision_stop.is_set() or not rclpy.ok():
                future.cancel()
                return None
            with self._capture_lock:
                if epoch != self._capture_epoch or not self._capture_active:
                    future.cancel()
                    return None
            if time.monotonic() >= call_deadline:
                future.cancel()
                if rpc_deadline < deadline:
                    self._record_vision_failure(epoch, f"{client.srv_name} timed out")
                return None
            time.sleep(0.01)
        if time.monotonic() >= deadline:
            return None
        with self._capture_lock:
            if epoch != self._capture_epoch or not self._capture_active:
                return None
        return future.result()

    def _record_vision_failure(self, epoch: int, detail: str) -> None:
        with self._capture_lock:
            if epoch != self._capture_epoch or not self._capture_active or time.monotonic() >= self._capture_deadline:
                return
            self._vision_failed += 1
            count = self._vision_failed
        if count <= 3 or count % 30 == 0:
            self.get_logger().warning(f"hri vision frame failed ({count}): {detail}")

    def _begin_capture(self, deadline: float) -> None:
        with self._capture_lock:
            self._capture_active = True
            self._capture_epoch += 1
            self._capture_deadline = deadline
            self._vision_ok = 0
            self._vision_detected = 0
            self._vision_fallback = 0
            self._vision_failed = 0
            self._vision_sample = ""
            self._yolox_detection = None
            self._yolox_result_ready = False
            self._yolox_calls = 0
            self._pear_calls = 0
            self._frame_count = 0
            self._first_frame_stamp = ""
            self._last_frame_stamp = ""
            self._last_frame_id = ""
            self._last_frame_received_at = 0.0
        # Frames buffered before this task started belong to the previous one.
        while True:
            try:
                self._vision_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._yolox_queue.get_nowait()
            except queue.Empty:
                break

    def destroy_node(self):
        self._vision_stop.set()
        for worker in (self._vision_worker, self._yolox_worker):
            if worker.is_alive():
                worker.join(timeout=1.0)
        return super().destroy_node()

    def _end_capture(self) -> dict[str, object]:
        with self._capture_lock:
            self._capture_active = False
            count = self._frame_count
            summary = {
                "topic": self._rgb_topic,
                "frames": count,
                "vision_ok": self._vision_ok,
                "vision_detected": self._vision_detected,
                "vision_fallback": self._vision_fallback,
                "vision_failed": self._vision_failed,
                "yolox_calls": self._yolox_calls,
                "pear_calls": self._pear_calls,
                "vision_sample": self._vision_sample,
                "first_stamp": self._first_frame_stamp,
                "last_stamp": self._last_frame_stamp,
                "last_frame_id": self._last_frame_id,
            }
            if self._last_frame_received_at:
                summary["last_age_sec"] = max(0.0, time.monotonic() - self._last_frame_received_at)
            else:
                summary["last_age_sec"] = None
            return summary

    def _run_startup_warmup(self) -> bool:
        """Perform the single launch-time warmup; tasks never repeat it."""
        if self._startup_warmup_attempted:
            return self._mock.status.warmup_ready
        self._startup_warmup_attempted = True
        ready = self._mock.warmup()
        if ready:
            self.get_logger().info("imitate_human_motion warmup READY")
        else:
            self.get_logger().error("imitate_human_motion warmup FAILED")
        # Pre-warm only. A vision service that is still coming up at launch is not a
        # node failure: each task re-checks readiness against its own deadline, and
        # the model services are fail-closed at load time anyway.
        for client in (self._yolox_client, self._pear_client):
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(
                    f"imitate_human_motion vision service {client.srv_name} not up yet; "
                    "it will be re-checked when a task starts"
                )
        return ready

    def _wait_for_vision_services(self, deadline: float) -> list[str]:
        """Return the names of the vision services still missing at task start."""
        missing = []
        for client in (self._yolox_client, self._pear_client):
            budget = min(self._rpc_timeout, max(0.0, deadline - time.monotonic()))
            if not client.wait_for_service(timeout_sec=budget):
                missing.append(client.srv_name)
        return missing

    def _binding_is_valid(self, request) -> bool:
        binding = request.dispatch_binding
        budget = binding.task_budget
        started = budget.started_at.sec + budget.started_at.nanosec / 1_000_000_000
        deadline = budget.deadline.sec + budget.deadline.nanosec / 1_000_000_000
        timeout_sec = float(request.timeout_sec)
        now = self.get_clock().now().nanoseconds / 1_000_000_000
        return bool(
            binding.schema_version == 1
            and str(binding.task_id).strip()
            and str(binding.root_task_id).strip()
            and str(binding.expected_registry_epoch).strip()
            and int(binding.expected_registry_generation) > 0
            and str(binding.expected_registry_digest).strip()
            and str(binding.dispatch_nonce).strip()
            and budget.schema_version == 1
            and budget.started_at.sec >= 0
            and budget.deadline.sec >= 0
            and 0 <= budget.started_at.nanosec < 1_000_000_000
            and 0 <= budget.deadline.nanosec < 1_000_000_000
            and math.isfinite(started)
            and math.isfinite(deadline)
            and deadline > started
            and deadline > now
            and math.isfinite(timeout_sec)
            and timeout_sec > 0.0
            and timeout_sec <= deadline - now
        )

    def _handle_goal(self, request):
        goal = MockGoal(
            arm_side=str(request.arm_side).strip().lower(),
            imitation_duration_sec=float(request.imitation_duration_sec),
            timeout_sec=float(request.timeout_sec),
        )
        if not self._binding_is_valid(request):
            return GoalResponse.REJECT
        if not delegated_executor_identity_matches(request.expected_executor, self._executor_identity):
            return GoalResponse.REJECT
        accepted, reason = self._mock.can_accept(goal)
        if not accepted:
            if "warmup" in reason:
                self.get_logger().info("imitate_human_motion warmup is in progress")
            return GoalResponse.REJECT
        with self._goal_lock:
            if self._goal_active:
                return GoalResponse.REJECT
            self._goal_active = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _handle_cancel(_goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        request = goal_handle.request
        goal = MockGoal(
            arm_side=str(request.arm_side).strip().lower(),
            imitation_duration_sec=float(request.imitation_duration_sec),
            timeout_sec=float(request.timeout_sec),
        )

        last_logged_phase = ""

        def publish_feedback(phase: str, progress: float, detail: str) -> None:
            nonlocal last_logged_phase
            feedback = ImitateHumanMotion.Feedback()
            feedback.phase = phase
            feedback.progress = float(progress)
            feedback.detail = detail
            goal_handle.publish_feedback(feedback)
            if phase != last_logged_phase:
                self.get_logger().info(f"imitate_human_motion phase={phase}: {detail}")
                last_logged_phase = phase

        runner = _GuardedPrimitivePlayer(
            self._primitive_client,
            goal_handle,
            joint_names=self._joint_names,
            reset_positions=self._reset_positions,
            rpc_timeout_sec=self._rpc_timeout,
            deadline=time.monotonic() + goal.timeout_sec,
            ros_now_sec=lambda: self.get_clock().now().nanoseconds / 1_000_000_000,
        )
        # Readiness is a task-time question, not a launch-time one: the check must be
        # here rather than in _handle_goal, which runs on the executor's callback
        # thread and must not block. It shares the task's deadline like every other
        # dependency wait, matching placement_executor_node's DEPENDENCY_UNAVAILABLE.
        unavailable = self._wait_for_vision_services(runner.deadline)
        if unavailable:
            message = f"vision services unavailable: {', '.join(unavailable)}"
            self.get_logger().error(f"imitate_human_motion {message}")
            with self._goal_lock:
                self._goal_active = False
            result = ImitateHumanMotion.Result()
            result.success = False
            result.error_code = "DEPENDENCY_UNAVAILABLE"
            result.message = message
            result.requested_duration_sec = goal.imitation_duration_sec
            fill_delegated_executor_identity(result.actual_executor, self._executor_identity)
            goal_handle.abort()
            return result

        self._begin_capture(runner.deadline)
        try:
            result_value = self._mock.execute(
                goal,
                feedback=lambda phase, progress, detail: publish_feedback(
                    phase, progress, f"{detail}; rgb_frames={self._frame_count}"
                ),
                is_cancel_requested=lambda: bool(goal_handle.is_cancel_requested),
                player=runner,
                prepare=runner.prepare,
                recover_safe_pose=runner.reset,
            )
        except PrimitiveStateUnknown as exc:
            result_value = MockResult(
                success=False,
                error_code=CANCEL_CLEANUP_TIMEOUT,
                message=str(exc),
                animation_id="",
                requested_duration_sec=goal.imitation_duration_sec,
                actual_duration_sec=0.0,
                completed_phases=(),
            )
        except Exception as exc:
            self.get_logger().error(f"imitate_human_motion execution failed: {exc}")
            result_value = MockResult(
                success=False,
                error_code="MOCK_PLAYBACK_FAILED",
                message=str(exc),
                animation_id="",
                requested_duration_sec=goal.imitation_duration_sec,
                actual_duration_sec=0.0,
                completed_phases=(),
            )
        finally:
            capture_summary = self._end_capture()
            with self._goal_lock:
                self._goal_active = False
        result = ImitateHumanMotion.Result()
        result.success = result_value.success
        result.error_code = result_value.error_code
        result.message = f"{result_value.message}; rgb_input={json.dumps(capture_summary, sort_keys=True)}"
        result.animation_id = result_value.animation_id
        result.requested_duration_sec = result_value.requested_duration_sec
        result.actual_duration_sec = result_value.actual_duration_sec
        result.completed_phases = list(result_value.completed_phases)
        fill_delegated_executor_identity(result.actual_executor, self._executor_identity)
        if result_value.success:
            goal_handle.succeed()
        elif result_value.error_code == "CANCELED":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImitateHumanMotionExecutorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

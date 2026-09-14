import queue
import threading
import time
from types import SimpleNamespace

import pytest
from sensor_msgs.msg import Image

import manipulation_execution.imitate_human_motion_executor_node as vision_module  # noqa: F401
from ibrobot_msgs.msg import Detection2D, DetectionArray
from manipulation_execution.imitate_human_motion_executor_node import (
    ImitateHumanMotionExecutorNode,
    _center_fallback_detection,
    _select_person_detection,
    _validated_person_confidence_threshold,
    _validated_yolox_refresh_interval,
)


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result
        self.cancelled = False

    def done(self):
        return True

    def result(self):
        return self._result

    def cancel(self):
        self.cancelled = True


class _Client:
    def __init__(self, name, responses=(), *, ready=True, future_factory=None):
        self.srv_name = name
        self._responses = list(responses)
        self._ready = ready
        self._future_factory = future_factory
        self.requests = []

    def service_is_ready(self):
        return self._ready

    def call_async(self, request):
        self.requests.append(request)
        if self._future_factory is not None:
            return self._future_factory(request)
        return _ImmediateFuture(self._responses.pop(0))


class _VisionHarness:
    _PEAR_OUTPUT_FIELDS = ImitateHumanMotionExecutorNode._PEAR_OUTPUT_FIELDS
    _on_rgb_frame = ImitateHumanMotionExecutorNode._on_rgb_frame
    _put_latest = staticmethod(ImitateHumanMotionExecutorNode._put_latest)
    _vision_loop = ImitateHumanMotionExecutorNode._vision_loop
    _yolox_loop = ImitateHumanMotionExecutorNode._yolox_loop
    _process_vision_frame = ImitateHumanMotionExecutorNode._process_vision_frame
    _process_yolox_frame = ImitateHumanMotionExecutorNode._process_yolox_frame
    _process_pear_frame = ImitateHumanMotionExecutorNode._process_pear_frame
    _call_vision = ImitateHumanMotionExecutorNode._call_vision
    _record_vision_failure = ImitateHumanMotionExecutorNode._record_vision_failure
    _end_capture = ImitateHumanMotionExecutorNode._end_capture

    def get_logger(self):
        return self._logger


def _detection(label, confidence, bbox):
    return Detection2D(label=label, confidence=confidence, bbox=bbox)


def _yolox_response(*detections, success=True, message=""):
    return SimpleNamespace(
        success=success,
        message=message,
        detections=DetectionArray(detections=list(detections)),
    )


def _pear_response(*, success=True, message="", empty_field=None, nonfinite_field=None):
    values = {name: [1.0] for name in ImitateHumanMotionExecutorNode._PEAR_OUTPUT_FIELDS}
    if empty_field is not None:
        values[empty_field] = []
    if nonfinite_field is not None:
        values[nonfinite_field] = [float("nan")]
    return SimpleNamespace(
        success=success,
        message=message,
        inference_time_ms=12.5,
        **values,
    )


def _harness(yolox_response, pear_response=None):
    node = _VisionHarness()
    node._logger = _Logger()
    node._person_confidence_threshold = 0.30
    node._rpc_timeout = 1.0
    node._capture_lock = threading.Lock()
    node._capture_active = True
    node._capture_epoch = 1
    node._capture_deadline = time.monotonic() + 10.0
    node._vision_ok = 0
    node._vision_detected = 0
    node._vision_fallback = 0
    node._vision_failed = 0
    node._vision_sample = ""
    node._rgb_topic = "/camera/test/image_raw"
    node._frame_count = 0
    node._first_frame_stamp = ""
    node._last_frame_stamp = ""
    node._last_frame_id = ""
    node._last_frame_received_at = 0.0
    node._vision_stop = threading.Event()
    node._vision_queue = queue.Queue(maxsize=1)
    node._yolox_queue = queue.Queue(maxsize=1)
    node._yolox_detection = None
    node._yolox_result_ready = False
    node._yolox_calls = 0
    node._pear_calls = 0
    node._yolox_client = _Client("/yolox", [yolox_response])
    node._pear_client = _Client("/pear", [] if pear_response is None else [pear_response])
    return node


@pytest.mark.parametrize("value", [0.0, 0.30, 1.0])
def test_person_confidence_threshold_accepts_finite_probability(value):
    assert _validated_person_confidence_threshold(value) == value


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_person_confidence_threshold_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="person_confidence_threshold"):
        _validated_person_confidence_threshold(value)


@pytest.mark.parametrize("value", [0.001, 0.25, 10.0])
def test_yolox_refresh_interval_accepts_positive_finite_value(value):
    assert _validated_yolox_refresh_interval(value) == value


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_yolox_refresh_interval_rejects_invalid_value(value):
    with pytest.raises(ValueError, match="yolox_refresh_interval_sec"):
        _validated_yolox_refresh_interval(value)


def test_person_selection_uses_highest_qualifying_finite_confidence():
    selected = _select_person_detection(
        [
            _detection("person", 0.29, [1.0, 1.0, 2.0, 2.0]),
            _detection("chair", 0.99, [2.0, 2.0, 3.0, 3.0]),
            _detection("person", float("nan"), [3.0, 3.0, 4.0, 4.0]),
            _detection("person", 0.71, [4.0, 4.0, 5.0, 5.0]),
            _detection("person", 0.83, [5.0, 5.0, 6.0, 6.0]),
        ],
        0.30,
    )

    assert selected is not None
    assert selected.confidence == pytest.approx(0.83)
    assert list(selected.bbox) == pytest.approx([5.0, 5.0, 6.0, 6.0])


def test_person_selection_returns_none_without_qualifying_person():
    detections = [
        _detection("chair", 0.9, [1.0, 1.0, 2.0, 2.0]),
        _detection("person", 0.29, [2.0, 2.0, 3.0, 3.0]),
    ]

    assert _select_person_detection(detections, 0.30) is None


@pytest.mark.parametrize(
    ("width", "height", "expected_bbox"),
    [
        (640, 480, [128.0, 48.0, 512.0, 432.0]),
        (640, 360, [176.0, 36.0, 464.0, 324.0]),
    ],
)
def test_center_fallback_geometry(width, height, expected_bbox):
    image = Image(width=width, height=height)
    image.header.frame_id = "camera"

    detection = _center_fallback_detection(image)

    assert detection.header.frame_id == "camera"
    assert detection.label == "person"
    assert detection.confidence == 0.0
    assert list(detection.bbox) == pytest.approx(expected_bbox)


@pytest.mark.parametrize(("width", "height"), [(0, 480), (640, 0)])
def test_center_fallback_rejects_invalid_image_dimensions(width, height):
    with pytest.raises(ValueError, match="image dimensions"):
        _center_fallback_detection(Image(width=width, height=height))


def test_invalid_image_dimensions_count_as_failure_without_pear_call():
    node = _harness(_yolox_response())

    node._process_vision_frame(Image(width=0, height=480), 1, node._capture_deadline)

    assert node._pear_client.requests == []
    assert node._vision_ok == 0
    assert node._vision_failed == 1


def test_invalid_image_dimensions_with_detected_person_fail_without_pear_call():
    node = _harness(
        _yolox_response(_detection("person", 0.83, [5.0, 6.0, 100.0, 200.0])),
        _pear_response(),
    )

    node._process_vision_frame(Image(width=0, height=480), 1, node._capture_deadline)

    assert node._pear_client.requests == []
    assert node._vision_ok == 0
    assert node._vision_detected == 0
    assert node._vision_fallback == 0
    assert node._vision_failed == 1


def test_detected_person_is_sent_to_pear_and_counted():
    selected = _detection("person", 0.83, [5.0, 6.0, 100.0, 200.0])
    node = _harness(
        _yolox_response(
            _detection("person", 0.29, [1.0, 2.0, 3.0, 4.0]),
            _detection("chair", 0.99, [2.0, 3.0, 4.0, 5.0]),
            selected,
        ),
        _pear_response(),
    )

    node._process_vision_frame(Image(width=640, height=480), 1, node._capture_deadline)

    assert node._yolox_client.requests[0].confidence_threshold == pytest.approx(0.30)
    sent = node._pear_client.requests[0].detections.detections
    assert len(sent) == 1
    assert sent[0].confidence == pytest.approx(0.83)
    assert node._vision_ok == 1
    assert node._vision_detected == 1
    assert node._vision_fallback == 0
    assert node._vision_ok == node._vision_detected + node._vision_fallback
    assert node._vision_sample.startswith("source=yolox")
    assert node._yolox_calls == 1
    assert node._pear_calls == 1


def test_pear_uses_newest_rgb_frame_with_cached_yolox_detection():
    node = _harness(
        _yolox_response(_detection("person", 0.83, [5.0, 6.0, 100.0, 200.0])),
        _pear_response(),
    )
    first = Image(width=640, height=480)
    first.header.frame_id = "first"
    second = Image(width=640, height=480)
    second.header.frame_id = "second"

    assert node._process_yolox_frame(first, 1, node._capture_deadline)
    node._process_pear_frame(second, 1, node._capture_deadline)

    assert node._yolox_client.requests[0].image is first
    assert node._pear_client.requests[0].image is second
    sent = node._pear_client.requests[0].detections.detections
    assert sent[0].header.frame_id == "second"
    assert sent[0].confidence == pytest.approx(0.83)
    assert list(sent[0].bbox) == pytest.approx([5.0, 6.0, 100.0, 200.0])


def test_yolox_refresh_failure_keeps_last_successful_bbox_for_pear():
    node = _harness(_yolox_response(), _pear_response())
    selected = _detection("person", 0.83, [5.0, 6.0, 100.0, 200.0])
    node._yolox_client = _Client(
        "/yolox",
        [_yolox_response(selected), _yolox_response(success=False, message="temporary failure")],
    )
    node._pear_client = _Client("/pear", [_pear_response(), _pear_response()])
    first = Image(width=640, height=480)
    second = Image(width=640, height=480)

    assert node._process_yolox_frame(first, 1, node._capture_deadline)
    node._process_pear_frame(first, 1, node._capture_deadline)
    assert not node._process_yolox_frame(second, 1, node._capture_deadline)
    node._process_pear_frame(second, 1, node._capture_deadline)

    assert node._vision_failed == 1
    assert node._vision_ok == 2
    assert node._vision_detected == 2
    assert all(
        list(request.detections.detections[0].bbox) == pytest.approx([5.0, 6.0, 100.0, 200.0])
        for request in node._pear_client.requests
    )


def test_pear_waits_for_first_successful_yolox_result():
    node = _harness(_yolox_response(), _pear_response())

    node._process_pear_frame(Image(width=640, height=480), 1, node._capture_deadline)

    assert node._pear_client.requests == []
    assert node._vision_ok == 0


def test_latest_frame_queue_replaces_older_frame():
    frame_queue = queue.Queue(maxsize=1)
    first = object()
    second = object()

    ImitateHumanMotionExecutorNode._put_latest(frame_queue, first)
    ImitateHumanMotionExecutorNode._put_latest(frame_queue, second)

    assert frame_queue.get_nowait() is second


def test_rgb_frame_is_admitted_to_both_latest_frame_queues():
    node = _harness(_yolox_response())
    first = Image(width=640, height=480)
    first.header.frame_id = "first"
    second = Image(width=640, height=480)
    second.header.frame_id = "second"

    node._on_rgb_frame(first)
    node._on_rgb_frame(second)

    assert node._vision_queue.get_nowait()[1] is second
    assert node._yolox_queue.get_nowait()[1] is second


def test_no_qualifying_person_sends_center_fallback_to_pear():
    node = _harness(
        _yolox_response(
            _detection("person", 0.29, [1.0, 2.0, 3.0, 4.0]),
            _detection("chair", 0.95, [5.0, 6.0, 7.0, 8.0]),
        ),
        _pear_response(),
    )

    node._process_vision_frame(Image(width=640, height=360), 1, node._capture_deadline)

    sent = node._pear_client.requests[0].detections.detections
    assert len(sent) == 1
    assert sent[0].confidence == 0.0
    assert list(sent[0].bbox) == pytest.approx([176.0, 36.0, 464.0, 324.0])
    assert node._vision_ok == 1
    assert node._vision_detected == 0
    assert node._vision_fallback == 1
    assert node._vision_ok == node._vision_detected + node._vision_fallback
    assert node._vision_sample.startswith("source=center_fallback")


def test_yolox_failure_does_not_invoke_fallback_or_pear():
    node = _harness(_yolox_response(success=False, message="model failed"))

    node._process_vision_frame(Image(width=640, height=480), 1, node._capture_deadline)

    assert node._pear_client.requests == []
    assert node._vision_ok == 0
    assert node._vision_failed == 1


def test_yolox_service_unavailable_does_not_invoke_fallback_or_pear():
    node = _harness(_yolox_response())
    node._yolox_client = _Client("/yolox", ready=False)

    node._process_vision_frame(Image(width=640, height=480), 1, node._capture_deadline)

    assert node._pear_client.requests == []
    assert node._vision_ok == 0
    assert node._vision_failed == 1


@pytest.mark.parametrize(
    "response",
    [
        _pear_response(success=False, message="model failed"),
        _pear_response(empty_field="camera_raw"),
        _pear_response(nonfinite_field="camera_raw"),
    ],
)
def test_pear_failure_or_malformed_output_is_counted(response):
    node = _harness(_yolox_response(), response)

    node._process_vision_frame(Image(width=640, height=480), 1, node._capture_deadline)

    assert len(node._pear_client.requests) == 1
    assert node._vision_ok == 0
    assert node._vision_failed == 1


def test_end_capture_reports_source_counters_and_invariant():
    node = _harness(_yolox_response())
    node._vision_ok = 7
    node._vision_detected = 2
    node._vision_fallback = 5
    node._yolox_calls = 3
    node._pear_calls = 7

    summary = node._end_capture()

    assert summary["vision_ok"] == 7
    assert summary["vision_detected"] == 2
    assert summary["vision_fallback"] == 5
    assert summary["vision_ok"] == summary["vision_detected"] + summary["vision_fallback"]
    assert summary["yolox_calls"] == 3
    assert summary["pear_calls"] == 7


class _Clock:
    def __init__(self, now=0.0):
        self.now = float(now)

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class _PendingFuture:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


class _CompletingFuture(_PendingFuture):
    def __init__(self, clock, complete_at, result):
        super().__init__()
        self._clock = clock
        self._complete_at = complete_at
        self._result = result

    def done(self):
        return self._clock.now >= self._complete_at

    def result(self):
        return self._result


class _SingleFrameQueue:
    def __init__(self, stop_event, frame):
        self._stop_event = stop_event
        self._frame = frame

    def get(self, timeout):
        self._stop_event.set()
        return self._frame


class _TimedFrameQueue:
    def __init__(self, clock, stop_event, frames, *, stop_at):
        self._clock = clock
        self._stop_event = stop_event
        self._frames = list(frames)
        self._stop_at = stop_at

    def get(self, timeout):
        wait_deadline = self._clock.now + timeout
        if self._frames and self._frames[0][0] <= wait_deadline:
            received_at, frame = self._frames.pop(0)
            self._clock.now = received_at
            return frame
        self._clock.now = wait_deadline
        if not self._frames and self._clock.now >= self._stop_at:
            self._stop_event.set()
        raise queue.Empty


def _deadline_harness(rpc_timeout, deadline):
    node = _VisionHarness()
    node._logger = _Logger()
    node._rpc_timeout = rpc_timeout
    node._capture_lock = threading.Lock()
    node._capture_active = True
    node._capture_epoch = 1
    node._capture_deadline = deadline
    node._vision_failed = 0
    node._vision_stop = threading.Event()
    return node


def test_yolox_worker_rate_limits_calls_and_uses_latest_pending_frame(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    node = _deadline_harness(rpc_timeout=1.0, deadline=10.0)
    node._yolox_refresh_interval_sec = 0.25
    first = Image(width=640, height=480)
    first.header.frame_id = "first"
    second = Image(width=640, height=480)
    second.header.frame_id = "second"
    third = Image(width=640, height=480)
    third.header.frame_id = "third"
    node._yolox_queue = _TimedFrameQueue(
        clock,
        node._vision_stop,
        [(0.0, (1, first)), (0.05, (1, second)), (0.10, (1, third))],
        stop_at=0.30,
    )
    calls = []
    node._process_yolox_frame = lambda message, _epoch, _deadline: calls.append((clock.now, message))

    node._yolox_loop()

    assert [called_at for called_at, _message in calls] == pytest.approx([0.0, 0.25])
    assert [message.header.frame_id for _called_at, message in calls] == ["first", "third"]


def test_yolox_worker_first_frame_of_new_epoch_is_immediate(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    node = _deadline_harness(rpc_timeout=1.0, deadline=10.0)
    node._yolox_refresh_interval_sec = 0.25
    first = Image(width=640, height=480)
    second = Image(width=640, height=480)
    node._capture_epoch = 2
    node._yolox_queue = _TimedFrameQueue(
        clock,
        node._vision_stop,
        [(0.0, (1, first)), (0.05, (2, second))],
        stop_at=0.10,
    )
    calls = []
    node._process_yolox_frame = lambda message, epoch, _deadline: calls.append((clock.now, epoch, message))

    node._yolox_loop()

    assert [(called_at, epoch) for called_at, epoch, _message in calls] == [(0.05, 2)]


def test_expired_task_does_not_submit_vision_request(monkeypatch):
    clock = _Clock(now=1.0)
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    node = _deadline_harness(rpc_timeout=1.0, deadline=1.0)
    client = _Client("/vision", [_yolox_response()])

    result = node._call_vision(client, object(), epoch=1, deadline=1.0)

    assert result is None
    assert client.requests == []
    assert node._vision_failed == 0


def test_readiness_crossing_deadline_does_not_submit_vision_request(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    node = _deadline_harness(rpc_timeout=1.0, deadline=0.2)
    client = _Client("/vision", [_yolox_response()])

    def service_is_ready():
        clock.now = 0.2
        return True

    client.service_is_ready = service_is_ready

    result = node._call_vision(client, object(), epoch=1, deadline=0.2)

    assert result is None
    assert client.requests == []
    assert node._vision_failed == 0


def test_task_deadline_cancels_without_timeout_failure(monkeypatch):
    clock = _Clock()
    future = _PendingFuture()
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(vision_module.time, "sleep", clock.sleep)
    monkeypatch.setattr(vision_module.rclpy, "ok", lambda: True)
    node = _deadline_harness(rpc_timeout=1.0, deadline=0.2)
    client = _Client("/vision", future_factory=lambda _request: future)

    result = node._call_vision(client, object(), epoch=1, deadline=0.2)

    assert result is None
    assert len(client.requests) == 1
    assert future.cancelled
    assert node._vision_failed == 0


def test_rpc_deadline_records_exactly_one_timeout_failure(monkeypatch):
    clock = _Clock()
    future = _PendingFuture()
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(vision_module.time, "sleep", clock.sleep)
    monkeypatch.setattr(vision_module.rclpy, "ok", lambda: True)
    node = _deadline_harness(rpc_timeout=0.2, deadline=2.0)
    client = _Client("/vision", future_factory=lambda _request: future)

    result = node._call_vision(client, object(), epoch=1, deadline=2.0)

    assert result is None
    assert future.cancelled
    assert node._vision_failed == 1


def test_frame_at_task_deadline_is_not_counted_or_queued(monkeypatch):
    clock = _Clock(now=2.0)
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    node = _deadline_harness(rpc_timeout=1.0, deadline=2.0)
    node._frame_count = 7
    node._first_frame_stamp = ""
    node._last_frame_stamp = ""
    node._last_frame_id = ""
    node._last_frame_received_at = 0.0
    node._vision_queue = queue.Queue(maxsize=2)

    node._on_rgb_frame(Image(width=640, height=480))

    assert node._frame_count == 7
    assert node._vision_queue.empty()
    assert not node._capture_active


def test_worker_discards_dequeued_frame_at_task_deadline(monkeypatch):
    clock = _Clock(now=2.0)
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    node = _deadline_harness(rpc_timeout=1.0, deadline=2.0)
    processed = []
    node._vision_queue = _SingleFrameQueue(node._vision_stop, (1, Image(width=640, height=480)))
    node._process_vision_frame = lambda *args: processed.append(args)

    node._vision_loop()

    assert processed == []
    assert not node._capture_active


def test_failure_observed_at_task_deadline_is_not_counted(monkeypatch):
    clock = _Clock(now=2.0)
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    node = _deadline_harness(rpc_timeout=1.0, deadline=2.0)

    node._record_vision_failure(1, "late failure")

    assert node._vision_failed == 0


def test_response_observed_at_task_deadline_is_not_recorded(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(vision_module.time, "sleep", clock.sleep)
    monkeypatch.setattr(vision_module.rclpy, "ok", lambda: True)
    node = _harness(_yolox_response())
    node._capture_deadline = 0.2
    node._pear_client = _Client(
        "/pear",
        future_factory=lambda _request: _CompletingFuture(clock, 0.2, _pear_response()),
    )

    node._process_vision_frame(Image(width=640, height=480), 1, node._capture_deadline)

    assert node._vision_ok == 0
    assert node._vision_failed == 0

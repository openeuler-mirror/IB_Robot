import queue
import threading
import time
from types import SimpleNamespace

import pytest
from sensor_msgs.msg import Image

import manipulation_execution.imitate_human_motion_executor_node as vision_module  # noqa: F401
from ibrobot_msgs.msg import Detection2D, DetectionArray, DispatchBinding
from manipulation_execution.imitate_human_motion_executor import (
    MAX_IMITATION_DURATION_SEC,
    MockExecutor,
    MockGoal,
    MockResult,
)
from manipulation_execution.imitate_human_motion_executor_node import (
    _MAX_CAPTURED_PEAR_FRAMES,
    ImitateHumanMotionExecutorNode,
    _center_fallback_detection,
    _GuardedPrimitivePlayer,
    _select_person_detection,
    _validated_person_confidence_threshold,
    _validated_yolox_refresh_interval,
)


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
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
    _reset_capture_stats = ImitateHumanMotionExecutorNode._reset_capture_stats
    _execute = ImitateHumanMotionExecutorNode._execute
    _begin_capture = ImitateHumanMotionExecutorNode._begin_capture
    _end_capture = ImitateHumanMotionExecutorNode._end_capture
    _captured_pear_frames = ImitateHumanMotionExecutorNode._captured_pear_frames

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
    node._pear_frames = []
    node._pear_frames_dropped = 0
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
    # A frame that failed validation must not reach the recording either. The
    # retargeting stage reads this buffer without re-checking it, so a single
    # NaN pose smuggled in here would come back as a joint command.
    assert node._pear_frames == []


def _stamped_image(*, width=640, height=480, sec=0, nanosec=0, frame_id=""):
    """An RGB frame carrying a header, which the recording copies verbatim."""
    message = Image(width=width, height=height)
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = nanosec
    message.header.frame_id = frame_id
    return message


def test_pear_output_is_recorded_frame_by_frame():
    """The captured window is the only place the per-frame pose survives."""
    node = _harness(
        _yolox_response(_detection("person", 0.875, [10.0, 20.0, 110.0, 220.0])),
        _pear_response(),
    )

    node._process_vision_frame(
        _stamped_image(sec=17, nanosec=250000000, frame_id="wrist_camera_color_optical_frame"),
        1,
        node._capture_deadline,
    )

    assert len(node._pear_frames) == 1
    frame = node._pear_frames[0]
    # All eight outputs, not a subset: the retargeter reads the whole set, and
    # a window that is missing fields cannot be recorded a second time.
    for name in ImitateHumanMotionExecutorNode._PEAR_OUTPUT_FIELDS:
        assert frame[name] == [1.0], name
    # Raw sec/nanosec rather than the display string used for the summary. PEAR
    # answers at roughly 16 Hz against a ~23 fps camera and neither rate is
    # steady, so the series is non-uniformly sampled and the consumer needs
    # every frame's own time as a number.
    assert frame["stamp_sec"] == 17
    assert frame["stamp_nanosec"] == 250000000
    assert frame["frame_id"] == "wrist_camera_color_optical_frame"
    assert frame["source"] == "yolox"
    assert frame["bbox"] == [10.0, 20.0, 110.0, 220.0]
    assert frame["confidence"] == pytest.approx(0.875)
    assert frame["inference_time_ms"] == pytest.approx(12.5)


def test_recorded_frame_flags_the_centred_crop_guess():
    """A fallback crop must stay distinguishable from a real person box.

    On a normal run a good share of the window is inferred from a centred
    guess, not from a detection. Without the flag the retargeting stage cannot
    tell those frames from the trustworthy ones.
    """
    node = _harness(_yolox_response(), _pear_response())

    node._process_vision_frame(_stamped_image(), 1, node._capture_deadline)

    assert [frame["source"] for frame in node._pear_frames] == ["center_fallback"]
    # The crop travels with the pose, so the consumer can see which region the
    # pose was inferred from instead of re-deriving it.
    assert len(node._pear_frames[0]["bbox"]) == 4


def test_recording_is_bounded_and_overflow_is_counted():
    """The buffer is capped so a deadline defect cannot exhaust a swapless board."""
    node = _harness(_yolox_response(), _pear_response())
    node._pear_frames = [{"stamp_sec": index} for index in range(_MAX_CAPTURED_PEAR_FRAMES)]

    node._process_vision_frame(_stamped_image(sec=99), 1, node._capture_deadline)

    assert len(node._pear_frames) == _MAX_CAPTURED_PEAR_FRAMES
    assert node._pear_frames[-1] == {"stamp_sec": _MAX_CAPTURED_PEAR_FRAMES - 1}
    assert node._pear_frames_dropped == 1
    # PEAR did answer and the frame was good, so it still counts as a good
    # frame; only its storage was refused. Conflating the two would make an
    # overflow read as a vision failure in the summary.
    assert node._vision_ok == 1
    assert node._vision_failed == 0


def test_a_frame_arriving_after_the_window_closes_is_not_recorded():
    """The append is guarded on its own, not only by the call before it.

    ``_call_vision`` re-checks the window before handing a response back, but
    the window can still close in the gap between that check and the append --
    a new task starting while PEAR is still thinking. Closing it exactly in
    that gap is the only way to reach the guard deterministically, and without
    the guard the frame would land in the next task's recording.
    """
    node = _harness(_yolox_response(), _pear_response())
    inner = node._call_vision

    def _call_then_close(client, request, epoch, deadline, *, metric=""):
        result = inner(client, request, epoch, deadline, metric=metric)
        if metric == "pear":
            with node._capture_lock:
                node._capture_epoch += 1
        return result

    node._call_vision = _call_then_close

    node._process_vision_frame(_stamped_image(sec=5), 1, node._capture_deadline)

    assert node._pear_frames == []
    # Counted nowhere either: the frame belongs to a window that no longer exists.
    assert node._vision_ok == 0


def test_captured_frames_are_handed_over_as_a_snapshot():
    """The consumer works on the recording while the node moves on."""
    node = _harness(_yolox_response(), _pear_response())
    node._process_vision_frame(_stamped_image(sec=3), 1, node._capture_deadline)

    handed = node._captured_pear_frames()
    node._pear_frames.append({"stamp_sec": 99})

    assert [frame["stamp_sec"] for frame in handed] == [3]


def test_begin_capture_clears_the_previous_recording():
    """A new task must not inherit the last task's frames."""
    node = _harness(_yolox_response())
    node._pear_frames = [{"stamp_sec": 1}]
    node._pear_frames_dropped = 2

    node._begin_capture(time.monotonic() + 5.0)

    assert node._pear_frames == []
    assert node._pear_frames_dropped == 0


def test_end_capture_reports_recording_size_without_the_recording():
    node = _harness(_yolox_response())
    node._pear_frames = [{"smplx_pose_raw": [1.0]}, {"smplx_pose_raw": [2.0]}]
    node._pear_frames_dropped = 3

    summary = node._end_capture()

    assert summary["pear_frames"] == 2
    assert summary["pear_dropped"] == 3
    # Counts only. The frames are orders of magnitude too large to travel in
    # the result message, and nothing outside the node consumes them.
    assert all(not isinstance(value, (list | dict)) for value in summary.values())


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
    # The window closed while PEAR was still thinking. The frame is counted in
    # neither direction, so it must not be recorded in either -- a late answer
    # belongs to no recording at all.
    assert node._pear_frames == []


def test_capture_recorder_publishes_the_recording_when_the_window_closes(monkeypatch):
    """The recording leaves the node through the recorder, not the summary.

    ``_CaptureRecorder.frames`` is the seam the retargeting stage will read: it
    is published in the ``finally``, after ``_end_capture`` has closed the
    window, so what lands there can no longer grow and is exactly the recording.
    """
    node = _harness(_yolox_response(), _pear_response())
    clock = _Clock(now=1000.0)
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)

    def _sleep(_seconds):
        # One PEAR frame arrives while the window is open, then the clock jumps
        # past its end: a camera publishing into a recording that then closes.
        node._process_vision_frame(_stamped_image(sec=7), node._capture_epoch, node._capture_deadline)
        clock.sleep(5.0)

    monkeypatch.setattr(vision_module.time, "sleep", _sleep)
    recorder = vision_module._CaptureRecorder(node)

    outcome = recorder.record(
        2.0,
        feedback=lambda *_args: None,
        is_cancel_requested=lambda: False,
        deadline=clock.monotonic() + 30.0,
    )

    assert outcome == "COMPLETED"
    assert [frame["stamp_sec"] for frame in recorder.frames] == [7]
    # The count travels in the summary; the frames themselves never do.
    assert recorder.summary["pear_frames"] == 1


def _lifecycle_harness():
    """A harness carrying the counters of a task that has already run."""
    node = _harness(_yolox_response())
    node._frame_count = 375
    node._vision_ok = 370
    node._vision_detected = 5
    node._vision_fallback = 365
    node._vision_failed = 2
    node._vision_sample = "bbox=[176,36,464,324] conf=0.0000"
    node._yolox_calls = 59
    node._pear_calls = 370
    node._first_frame_stamp = "100.000000000"
    node._last_frame_stamp = "115.000000000"
    node._last_frame_id = "wrist_camera_color_optical_frame"
    node._last_frame_received_at = time.monotonic()
    return node


class _GoalHandle:
    """Minimal action goal handle: records the feedback _execute publishes."""

    def __init__(self, node, imitation_duration_sec=15.0):
        self.request = SimpleNamespace(
            dispatch_binding=DispatchBinding(),
            arm_side="right",
            imitation_duration_sec=imitation_duration_sec,
            timeout_sec=300.0,
        )
        self.is_cancel_requested = False
        self._node = node
        self.outcome = ""

    def publish_feedback(self, feedback):
        self._node._feedback.append((feedback.phase, feedback.detail))

    def succeed(self):
        self.outcome = "succeed"

    def abort(self):
        self.outcome = "abort"

    def canceled(self):
        self.outcome = "canceled"


class _FakeRunner:
    """Stands in for _GuardedPrimitivePlayer: prepare succeeds, nothing moves."""

    def __init__(self, *_args, **kwargs):
        self.deadline = kwargs.get("deadline", time.monotonic() + 30.0)

    def prepare(self):
        return True

    def play(self, _plan, _duration_sec, *, feedback, is_cancel_requested, deadline):
        self.node._deliver_frames(50)
        return "COMPLETED"

    def reset(self):
        return True


class _StubMock:
    """Replays the phase order MockExecutor uses, minus the animation itself.

    RGB frames are delivered in every phase, so the test sees which of them the
    node actually counts. Only the ``start`` phase is the recording: prepare
    moves the arm to the imitation start pose, playback moves it through the
    animation and reset stows it, and in all three the wrist camera is sweeping
    across the room rather than holding on the person.
    """

    def __init__(self, node):
        self._node = node

    def execute(self, goal, *, feedback, is_cancel_requested, player, recorder, prepare, recover_safe_pose):
        feedback("prepare", 0.05, "moving to imitation start pose")
        self._node._deliver_frames(20)
        prepared = prepare()
        self._node._deliver_frames(20)
        feedback("prepare", 0.25, "imitation start pose reached")
        if not prepared:
            return MockResult(
                success=False,
                error_code="PREPARE_FAILED",
                message="prepare failed",
                animation_id="",
                requested_duration_sec=goal.imitation_duration_sec,
                actual_duration_sec=0.0,
                completed_phases=(),
            )
        feedback("start", 0.3, f"capturing human motion for {goal.imitation_duration_sec:.1f}s")
        recorder.record(
            goal.imitation_duration_sec,
            feedback=feedback,
            is_cancel_requested=is_cancel_requested,
            deadline=time.monotonic() + goal.timeout_sec,
        )
        feedback("mock_playback", 0.5, "playing animation")
        player.play(
            _StubPlan(),
            goal.imitation_duration_sec,
            feedback=feedback,
            is_cancel_requested=is_cancel_requested,
            deadline=time.monotonic() + goal.timeout_sec,
        )
        feedback("reset", 0.9, "returning to safe pose")
        self._node._deliver_frames(30)
        recover_safe_pose()
        return MockResult(
            success=True,
            error_code="",
            message="ok",
            animation_id="stub",
            requested_duration_sec=goal.imitation_duration_sec,
            actual_duration_sec=goal.imitation_duration_sec,
            completed_phases=("prepare", "start", "mock_playback", "reset"),
        )


class _StubPlan:
    animation_id = "stub"
    waypoints = ()
    duration_sec = 10.0


def _execute_harness(monkeypatch):
    """A harness that can run _execute, carrying a finished task's counters."""
    node = _lifecycle_harness()
    node._end_capture()
    node._feedback = []
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._joint_names = ["1", "2", "3", "4", "5"]
    node._reset_positions = {name: 0.0 for name in node._joint_names}
    node._executor_identity = {}
    node._primitive_client = None
    node._mock = _StubMock(node)
    node._wait_for_vision_services = lambda _deadline: []
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))
    node._deliver_frames = lambda count: [node._on_rgb_frame(Image(width=640, height=480)) for _ in range(count)]

    def _make_runner(*args, **kwargs):
        runner = _FakeRunner(*args, **kwargs)
        runner.node = node
        return runner

    monkeypatch.setattr(vision_module, "_GuardedPrimitivePlayer", _make_runner)
    return node


def _capture_clock(monkeypatch, node, *, frames_per_tick=10, jump_sec=6.0, frames_after_jump=0):
    """Drive _CaptureRecorder on a fake clock that delivers frames as it waits.

    The recorder waits in ``time.sleep`` between feedback ticks, so replacing
    sleep with "hand the node some frames, then jump the clock" is what a camera
    publishing into an open window looks like, without the test depending on
    real elapsed time. The jump is deliberately coarser than the recorder's own
    feedback period so the window is crossed in a couple of ticks.

    ``frames_after_jump`` delivers a second batch once the clock has already
    moved, which is how frames that arrive past the window's end -- while the
    loop still believes it is recording -- reach the node.
    """
    clock = _Clock(now=1000.0)
    monkeypatch.setattr(vision_module.time, "monotonic", clock.monotonic)

    def _sleep(_seconds):
        node._deliver_frames(frames_per_tick)
        clock.sleep(jump_sec)
        if frames_after_jump:
            node._deliver_frames(frames_after_jump)

    monkeypatch.setattr(vision_module.time, "sleep", _sleep)
    return clock


def test_only_the_capture_window_is_recorded(monkeypatch):
    """Counters belong to the capture window, and to nothing around it.

    ``imitation_duration_sec`` is the data-collection window: it opens once
    prepare has parked the arm at the imitation start pose and closes before the
    animation plays. Frames from prepare, from playback and from reset are all
    arm-motion frames with the wrist camera sweeping across the room, and none
    of them may reach the recording.

    Driven through ``_execute`` rather than by calling the helpers directly: the
    defect this covers was never in a helper but in where it was called from, so
    a test that calls them itself would keep passing if the call moved again.
    """
    node = _execute_harness(monkeypatch)
    _capture_clock(monkeypatch, node)

    result = node._execute(_GoalHandle(node, imitation_duration_sec=10.0))

    counts = {}
    for phase, detail in node._feedback:
        counts.setdefault(phase, []).append(int(detail.rsplit("rgb_frames=", 1)[1]))
    assert counts.get("prepare"), "prepare phase published no feedback"
    # The first feedback line is the one the operator reads to tell whether
    # capture has started; before the window opens it must be this task's zero,
    # not the previous task's total.
    assert counts["prepare"] == [0, 0]
    # Two ticks of ten frames land inside the ten-second window: the first
    # feedback line predates any of them, the next two watch them arrive.
    assert counts["start"] == [0, 0, 10, 20]
    # 50 frames arrive during playback and 30 during reset. Neither phase is
    # part of the recording, so the total never moves past the window's 20.
    assert counts["mock_playback"] == [20]
    assert counts["reset"] == [20]
    assert '"frames": 20' in result.message


def test_reset_capture_stats_does_not_start_capture():
    """Zeroing the counters must not make the node start consuming frames."""
    node = _lifecycle_harness()
    node._end_capture()
    epoch_before = node._capture_epoch
    deadline_before = node._capture_deadline

    with node._capture_lock:
        node._reset_capture_stats()

    assert node._capture_active is False
    assert node._capture_epoch == epoch_before
    assert node._capture_deadline == deadline_before


def test_begin_capture_clears_stale_counters():
    """Extracting the reset helper must not cost _begin_capture its own reset."""
    node = _lifecycle_harness()
    node._end_capture()
    epoch_before = node._capture_epoch

    node._begin_capture(time.monotonic() + 10.0)

    assert node._capture_active is True
    assert node._capture_epoch == epoch_before + 1
    assert node._frame_count == 0
    assert node._vision_detected == 0
    assert node._yolox_calls == 0

    node._on_rgb_frame(Image(width=640, height=480))
    assert node._frame_count == 1


def test_capture_window_closes_at_the_requested_duration(monkeypatch):
    """The recording is bounded by time, not by how long the loop actually took.

    The animation is built from the window's wall-clock length, so a recorder
    that overshoots its own budget must not stretch the recording with it.
    ``_CaptureRecorder`` arms the capture deadline from the requested duration,
    and ``_on_rgb_frame`` stops counting once that passes.
    """
    node = _execute_harness(monkeypatch)
    # One tick overshoots the 10-second window by 2 seconds. The batch before
    # the jump is inside the window; the batch after it arrives while the loop
    # still believes it is recording, and must be rejected on the deadline
    # alone -- which only happens if that deadline came from the requested
    # duration rather than from the task's own much later timeout.
    _capture_clock(monkeypatch, node, jump_sec=12.0, frames_after_jump=10)

    result = node._execute(_GoalHandle(node, imitation_duration_sec=10.0))

    assert '"frames": 10' in result.message


def _motion_config():
    names = ["1", "2", "3", "4", "5"]
    return names, {name: 0.0 for name in names}, {name: (-2.0, 2.0) for name in names}


def test_capture_runs_to_completion_before_the_animation_plays():
    """Data collection is a phase of its own, ahead of playback, not alongside it.

    The animation is what the captured motion gets solved into -- today a preset
    stands in for the solver -- so it cannot begin until the capture window has
    closed. Playing it during the window would also move the arm, which is
    exactly what the window exists to prevent: the wrist camera has to hold on
    the person for the whole recording.
    """
    names, reset, limits = _motion_config()
    executor = MockExecutor(joint_names=names, reset_positions=reset, joint_limits=limits, warmup_ready=True)
    events = []

    class _Recorder:
        def record(self, duration_sec, *, feedback, is_cancel_requested, deadline):
            events.append(f"record_start:{duration_sec}")
            events.append("record_end")
            return "COMPLETED"

    class _Player:
        def play(self, _plan, duration_sec, *, feedback, is_cancel_requested, deadline):
            events.append(f"play_start:{duration_sec}")
            return "COMPLETED"

    result = executor.execute(
        MockGoal(arm_side="auto", imitation_duration_sec=7.0, timeout_sec=300.0),
        recorder=_Recorder(),
        player=_Player(),
        prepare=lambda: events.append("prepare") or True,
        recover_safe_pose=lambda: events.append("reset") or True,
    )

    assert events == ["prepare", "record_start:7.0", "record_end", "play_start:7.0", "reset"]
    assert result.completed_phases == ("prepare", "start", "mock_playback", "reset")
    assert result.success is True


@pytest.mark.parametrize(
    ("outcome", "error_code"),
    [("CANCELED", "CANCELED"), ("TIMEOUT", "SKILL_TIMEOUT"), ("FAILED", "CAPTURE_FAILED")],
)
def test_capture_failure_skips_playback_but_still_resets(outcome, error_code):
    """A window that did not complete has nothing to animate, but the arm still moved.

    Prepare has already driven the arm out to the imitation start pose by this
    point, so returning an error and stopping there would leave it standing
    there. Reset runs on every path that got past prepare; only a goal rejected
    before prepare leaves the arm untouched, and there is nothing to undo.
    """
    names, reset, limits = _motion_config()
    executor = MockExecutor(joint_names=names, reset_positions=reset, joint_limits=limits, warmup_ready=True)
    played = []
    reset_calls = []

    class _Recorder:
        def record(self, _duration_sec, *, feedback, is_cancel_requested, deadline):
            return outcome

    class _Player:
        def play(self, *_args, **_kwargs):
            played.append(True)
            return "COMPLETED"

    result = executor.execute(
        MockGoal(arm_side="auto", imitation_duration_sec=5.0, timeout_sec=300.0),
        recorder=_Recorder(),
        player=_Player(),
        prepare=lambda: True,
        recover_safe_pose=lambda: reset_calls.append(True) or True,
    )

    assert result.success is False
    assert result.error_code == error_code
    assert played == [], "a failed capture must not be animated"
    assert reset_calls == [True], "the arm is out at the start pose and must be stowed"
    assert result.completed_phases == ("prepare", "start", "reset")


def test_playback_spans_every_segment_at_the_maximum_duration():
    """The longest legal request must play the animation, not raise on it.

    The waypoint list is one longer than the segment list, so pairing waypoint
    n with waypoint n+1 is correctly uneven. Pairing them strictly meant that
    any request long enough to consume every segment -- which the 20 s maximum
    does exactly -- ended in "zip() argument 2 is shorter than argument 1"
    instead of a finished animation.
    """
    names, reset, limits = _motion_config()
    executor = MockExecutor(joint_names=names, reset_positions=reset, joint_limits=limits, warmup_ready=True)
    plan = executor.animations["mock_auto_v1"]
    runner = _GuardedPrimitivePlayer(
        None,
        _GoalHandle(SimpleNamespace(_feedback=[])),
        joint_names=names,
        reset_positions=reset,
        rpc_timeout_sec=5.0,
        deadline=time.monotonic() + 300.0,
        ros_now_sec=lambda: 0.0,
    )
    segments = []
    runner._run = lambda **kwargs: segments.append(kwargs["duration_sec"]) or "COMPLETED"

    outcome = runner.play(
        plan,
        MAX_IMITATION_DURATION_SEC,
        feedback=lambda *_args: None,
        is_cancel_requested=lambda: False,
        deadline=time.monotonic() + 300.0,
    )

    assert outcome == "COMPLETED"
    assert len(segments) == len(plan.waypoints) - 1
    assert sum(segments) == pytest.approx(plan.duration_sec)

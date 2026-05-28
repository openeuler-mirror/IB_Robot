"""Tests for camera ISP calibrator helpers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_tools.camera_isp import v4l2_ctl  # noqa: E402
from dataset_tools.camera_isp_calibrator import (  # noqa: E402
    OpenCvBridge,
    _camera_name_from_source,
    _derived_camera_name_notice,
    _video_device_from_source,
    build_parser,
    normalize_camera_source,
)


class DummyCapture:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.released = False
        self.set_calls = []

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(0.001)
        return True, self.frame.copy()

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


class DummyOpenCV:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, frame: np.ndarray):
        self.capture = DummyCapture(frame)
        self.source = None

    def VideoCapture(self, source):
        self.source = source
        return self.capture


class FailingCapture:
    def __init__(self):
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        self.read_count += 1
        return False, None

    def release(self) -> None:
        self.released = True


class FailingOpenCV:
    def __init__(self):
        self.capture = FailingCapture()

    def VideoCapture(self, _source):
        return self.capture


def test_normalize_camera_source_supports_index_or_path():
    assert normalize_camera_source("0") == 0
    assert normalize_camera_source("/dev/video2") == "/dev/video2"


def test_video_device_and_camera_name_derive_from_source():
    assert _video_device_from_source(0) == "/dev/video0"
    assert _video_device_from_source("3") == "/dev/video3"
    assert _video_device_from_source("/dev/video2") == "/dev/video2"
    assert _video_device_from_source("/tmp/reference.mp4") is None

    assert _camera_name_from_source("0") == "video0"
    assert _camera_name_from_source("/dev/video2") == "video2"


def test_parser_accepts_direct_source_without_camera_name():
    args = build_parser().parse_args([
        "--camera_index",
        "/dev/video0",
        "--reference",
        "ref.png",
    ])

    assert args.camera is None
    assert args.camera_index == "/dev/video0"


def test_derived_camera_name_notice_points_to_override_path(monkeypatch):
    monkeypatch.setenv("ROS_HOME", "/tmp/ibrobot-ros")

    notice = _derived_camera_name_notice("video0")

    assert "--camera omitted" in notice
    assert "/tmp/ibrobot-ros/ibrobot/camera_isp_overrides/video0.json" in notice
    assert "robot.launch.py" in notice


def test_opencv_bridge_reads_latest_frame_and_reports_no_v4l2(monkeypatch):
    monkeypatch.setattr(v4l2_ctl, "have_v4l2_ctl", lambda: False)
    frame = np.full((2, 3, 3), 17, dtype=np.uint8)
    opencv = DummyOpenCV(frame)
    bridge = OpenCvBridge("video0", 0, opencv)

    try:
        deadline = time.monotonic() + 1.0
        latest = None
        while time.monotonic() < deadline:
            latest, stamp = bridge.latest_frame()
            if latest is not None and stamp > 0.0:
                break
            time.sleep(0.01)

        assert opencv.source == 0
        assert latest is not None
        assert np.array_equal(latest, frame)

        params, err = bridge.get_params(("video_device", "exposure", "auto_white_balance"))
        expected_defaults = OpenCvBridge._default_params(("exposure", "auto_white_balance"))
        assert params["video_device"] == "/dev/video0"
        assert params["exposure"] == expected_defaults["exposure"]
        assert params["auto_white_balance"] is expected_defaults["auto_white_balance"]
        assert "v4l2-ctl not available" in err
    finally:
        bridge.shutdown()

    assert opencv.capture.released is True


def test_opencv_bridge_requests_reference_capture_size():
    frame = np.full((480, 640, 3), 17, dtype=np.uint8)
    opencv = DummyOpenCV(frame)
    bridge = OpenCvBridge("video0", 0, opencv, capture_size=(640, 480))

    try:
        assert opencv.capture.set_calls == [
            (DummyOpenCV.CAP_PROP_FRAME_WIDTH, 640),
            (DummyOpenCV.CAP_PROP_FRAME_HEIGHT, 480),
        ]
    finally:
        bridge.shutdown()


def test_opencv_bridge_stops_after_consecutive_capture_failures(monkeypatch):
    monkeypatch.setattr(OpenCvBridge, "_MAX_CAPTURE_FAILURES", 3)
    monkeypatch.setattr(OpenCvBridge, "_CAPTURE_RETRY_S", 0.001)
    opencv = FailingOpenCV()
    bridge = OpenCvBridge("video0", 0, opencv)

    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and bridge.capture_error is None:
            time.sleep(0.01)

        assert opencv.capture.read_count >= 3
        assert "stopped after 3 consecutive read failures" in bridge.capture_error
        latest, stamp = bridge.latest_frame()
        assert latest is None
        assert stamp == 0.0
    finally:
        bridge.shutdown()

    assert opencv.capture.released is True


def test_opencv_bridge_writes_numeric_params_before_auto_toggles(monkeypatch):
    monkeypatch.setattr(v4l2_ctl, "have_v4l2_ctl", lambda: True)
    resolved = {"resolved": object()}
    monkeypatch.setattr(v4l2_ctl, "resolve_ctrls", lambda _device: resolved)
    calls = []

    def fake_apply_params(device, resolved, params):
        calls.append(("apply_params", device, resolved, params))
        return True, "ok", set(params)

    def fake_set_ctrl(device, logical, value, resolved):
        calls.append(("set_ctrl", device, resolved, logical, value))
        return True, logical

    monkeypatch.setattr(v4l2_ctl, "apply_params", fake_apply_params)
    monkeypatch.setattr(v4l2_ctl, "set_ctrl", fake_set_ctrl)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    bridge = OpenCvBridge("video0", 0, DummyOpenCV(frame))

    try:
        ok, err = bridge.set_params({
            "autoexposure": False,
            "exposure": 200,
            "auto_white_balance": False,
            "white_balance": 4500,
        })
    finally:
        bridge.shutdown()

    assert ok is True
    assert err is None
    assert calls == [
        (
            "apply_params",
            "/dev/video0",
            resolved,
            {"exposure": 200, "white_balance": 4500},
        ),
        ("set_ctrl", "/dev/video0", resolved, "exposure_auto", 1),
        ("set_ctrl", "/dev/video0", resolved, "white_balance_auto", 0),
    ]

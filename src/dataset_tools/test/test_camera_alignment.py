"""Tests for camera alignment helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset_tools.camera_alignment as camera_alignment  # noqa: E402
from dataset_tools.camera_alignment import (  # noqa: E402
    CaptureSettings,
    CaptureStatus,
    OpenCVFrameSource,
    build_parser,
    capture_setting_warnings,
    compute_alignment_error,
    decode_fourcc,
    get_status_color,
    normalize_camera_source,
    parse_reference_payload,
    reference_size_warning,
    serialize_reference_payload,
)
from dataset_tools.opencv_utils import (  # noqa: E402
    opencv_has_gui_support,
    path_has_cv2_module,
)


def test_compute_alignment_error_averages_marker_corner_distance():
    reference = {
        1: np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    }
    detected = {
        1: np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    }

    error, status = compute_alignment_error(reference, detected)

    assert error == 1.0
    assert status == "Error: 1.00px (IDs:[1])"


def test_compute_alignment_error_handles_missing_reference_and_missing_targets():
    error, status = compute_alignment_error(None, {})
    assert error is None
    assert status == "No Reference (Press 's')"

    reference = {
        7: np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    }
    error, status = compute_alignment_error(reference, {})
    assert error is None
    assert status == "All Markers Lost"

    error, status = compute_alignment_error(reference, {1: reference[7]})
    assert error is None
    assert status == "Target IDs [7] not found"


def test_get_status_color_uses_expected_thresholds():
    assert get_status_color(None) == (0, 255, 255)
    assert get_status_color(2.99) == (0, 255, 0)
    assert get_status_color(3.0) == (0, 0, 255)


def test_normalize_camera_source_supports_video_device():
    assert normalize_camera_source("0") == 0
    assert normalize_camera_source("/dev/video2") == "/dev/video2"


def test_build_parser_accepts_explicit_capture_settings():
    parsed = build_parser().parse_args(
        [
            "--cameras_index_or_path",
            "/dev/video0",
            "--width",
            "640",
            "--height",
            "480",
            "--fps",
            "60",
            "--format",
            "mjpg",
        ]
    )

    assert parsed.width == 640
    assert parsed.height == 480
    assert parsed.fps == 60.0
    assert parsed.capture_format == "MJPG"


def test_build_parser_keeps_fourcc_alias_for_compatibility():
    parsed = build_parser().parse_args(
        [
            "--cameras_index_or_path",
            "/dev/video0",
            "--fourcc",
            "yuyv",
        ]
    )

    assert parsed.capture_format == "YUYV"


def _encode_fourcc(text: str) -> int:
    return sum(ord(char) << (8 * index) for index, char in enumerate(text))


def _encode_fourcc_bytes(values: list[int]) -> int:
    return sum(value << (8 * index) for index, value in enumerate(values))


def test_decode_fourcc_handles_opencv_integer_values():
    assert decode_fourcc(_encode_fourcc("MJPG")) == "MJPG"
    assert decode_fourcc(0) is None


def test_decode_fourcc_escapes_non_printable_bytes():
    assert decode_fourcc(_encode_fourcc_bytes([ord("M"), 0, ord("P"), 255])) == ("M\\x00P\\xff")


def test_capture_setting_warnings_reports_mismatches():
    requested = CaptureSettings(
        width=1280,
        height=720,
        fps=60.0,
        capture_format="MJPG",
    )
    actual = CaptureStatus(width=640, height=480, fps=30.0, capture_format="YUYV")

    assert capture_setting_warnings(requested, actual) == [
        "width requested 1280, actual 640",
        "height requested 720, actual 480",
        "fps requested 60, actual 30",
        "format requested MJPG, actual YUYV",
    ]


def test_opencv_frame_source_sets_requested_capture_properties(monkeypatch, capsys):
    class DummyCapture:
        def __init__(self):
            self.set_calls = []
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.set_calls.append((prop, value))
            return True

        def get(self, prop):
            if prop == DummyOpenCV.CAP_PROP_FPS:
                return 60.0
            if prop == DummyOpenCV.CAP_PROP_FOURCC:
                return _encode_fourcc("MJPG")
            return 0

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class DummyOpenCV:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        CAP_PROP_FOURCC = 6

        def __init__(self):
            self.capture = DummyCapture()

        def VideoCapture(self, source):
            self.source = source
            return self.capture

        def VideoWriter_fourcc(self, *chars):
            return _encode_fourcc("".join(chars))

    dummy_opencv = DummyOpenCV()
    monkeypatch.setattr(camera_alignment, "cv2", dummy_opencv)

    source = OpenCVFrameSource(
        "/dev/video0",
        width=640,
        height=480,
        fps=60.0,
        capture_format="MJPG",
    )
    ok, frame = source.read()
    output = capsys.readouterr().out

    assert ok is True
    assert frame.shape == (480, 640, 3)
    assert dummy_opencv.capture.set_calls == [
        (DummyOpenCV.CAP_PROP_FRAME_WIDTH, 640),
        (DummyOpenCV.CAP_PROP_FRAME_HEIGHT, 480),
        (DummyOpenCV.CAP_PROP_FPS, 60.0),
        (DummyOpenCV.CAP_PROP_FOURCC, _encode_fourcc("MJPG")),
    ]
    assert "requested 640x480@60 MJPG" in output
    assert "actual 640x480@60 MJPG" in output
    assert "mismatch" not in output

    source.release()
    assert dummy_opencv.capture.released is True


def test_opencv_frame_source_reports_zero_fps(monkeypatch, capsys):
    class DummyCapture:
        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def get(self, prop):
            if prop == DummyOpenCV.CAP_PROP_FPS:
                return 0.0
            if prop == DummyOpenCV.CAP_PROP_FOURCC:
                return _encode_fourcc("MJPG")
            return 0

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            pass

    class DummyOpenCV:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        CAP_PROP_FOURCC = 6

        def __init__(self):
            self.capture = DummyCapture()

        def VideoCapture(self, source):
            return self.capture

    monkeypatch.setattr(camera_alignment, "cv2", DummyOpenCV())

    source = OpenCVFrameSource("/dev/video0")
    source.read()
    output = capsys.readouterr().out

    assert "actual 640x480@0 MJPG" in output


def test_opencv_frame_source_releases_capture_when_open_fails(monkeypatch):
    class DummyCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    class DummyOpenCV:
        def __init__(self):
            self.capture = DummyCapture()

        def VideoCapture(self, source):
            return self.capture

    dummy_opencv = DummyOpenCV()
    monkeypatch.setattr(camera_alignment, "cv2", dummy_opencv)

    with pytest.raises(RuntimeError):
        OpenCVFrameSource("/dev/video0")

    assert dummy_opencv.capture.released is True


def test_reference_payload_records_frame_size_and_markers():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    payload = serialize_reference_payload(frame, {7: corners})
    parsed_markers, parsed_size = parse_reference_payload(payload)

    assert payload["image_width"] == 640
    assert payload["image_height"] == 480
    assert parsed_size == (640, 480)
    np.testing.assert_array_equal(parsed_markers[7], corners)


def test_parse_reference_payload_supports_legacy_marker_map():
    corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    parsed_markers, parsed_size = parse_reference_payload({"7": corners})

    assert parsed_size is None
    np.testing.assert_array_equal(
        parsed_markers[7],
        np.array(corners, dtype=np.float32),
    )


def test_reference_size_warning_flags_different_pixel_coordinate_system():
    assert reference_size_warning((640, 480), np.zeros((480, 640, 3))) is None

    warning = reference_size_warning((640, 480), np.zeros((720, 1280, 3)))

    assert warning is not None
    assert "ref 640x480 current 1280x720" in warning
    assert "alignment error is unreliable" in warning


def test_path_has_cv2_module_handles_binary_extension_layout(tmp_path):
    assert path_has_cv2_module(tmp_path) is False

    (tmp_path / "cv2.cpython-310-x86_64-linux-gnu.so").write_text("")

    assert path_has_cv2_module(tmp_path) is True


def test_opencv_has_gui_support_parses_build_information():
    class DummyOpenCV:
        def __init__(self, build_information: str):
            self.build_information = build_information

        def getBuildInformation(self) -> str:
            return self.build_information

    assert opencv_has_gui_support(DummyOpenCV("GUI: GTK3")) is True
    assert opencv_has_gui_support(DummyOpenCV("GUI: NONE")) is False

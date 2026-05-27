"""Camera alignment helper based on ArUco markers."""

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_tools.opencv_utils import require_opencv_gui

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

cv2 = None


@dataclass(frozen=True)
class CaptureSettings:
    """Requested OpenCV capture settings."""

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    capture_format: str | None = None


@dataclass(frozen=True)
class CaptureStatus:
    """Effective OpenCV capture settings observed after the first frame."""

    width: int
    height: int
    fps: float | None
    capture_format: str | None


def get_status_color(error_value: float | None) -> tuple[int, int, int]:
    """Map alignment error to a UI color."""
    if error_value is None:
        return YELLOW
    if error_value < 3.0:
        return GREEN
    return RED


def compute_alignment_error(
    reference_data: dict[int, np.ndarray] | None,
    detected_markers: dict[int, np.ndarray],
) -> tuple[float | None, str]:
    """Compute average marker corner error against saved reference data."""
    if reference_data is None:
        return None, "No Reference (Press 's')"
    if not detected_markers:
        return None, "All Markers Lost"

    errors: list[float] = []
    matched_ids: list[int] = []

    for marker_id, reference_corners in reference_data.items():
        detected_corners = detected_markers.get(marker_id)
        if detected_corners is None:
            continue
        error = np.mean(np.linalg.norm(detected_corners - reference_corners, axis=1))
        errors.append(float(error))
        matched_ids.append(marker_id)

    if not errors:
        return None, f"Target IDs {sorted(reference_data.keys())} not found"

    average_error = float(np.mean(errors))
    return average_error, f"Error: {average_error:.2f}px (IDs:{matched_ids})"


def _require_opencv():
    global cv2
    if cv2 is None:
        cv2 = require_opencv_gui()
    return cv2


def _safe_destroy_window(window_name: str) -> None:
    if cv2 is None:
        return

    with contextlib.suppress(Exception):  # pragma: no cover - cleanup should not hide real failure
        cv2.destroyWindow(window_name)


def _safe_destroy_all_windows() -> None:
    if cv2 is None:
        return

    with contextlib.suppress(Exception):  # pragma: no cover - cleanup should not hide real failure
        cv2.destroyAllWindows()


def normalize_camera_source(camera_source: str) -> str | int:
    """Normalize the original camera source CLI option."""
    return int(camera_source) if camera_source.isdigit() else camera_source


def positive_int(value: str) -> int:
    """argparse type for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """argparse type for positive floating point values."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def capture_format_text(value: str) -> str:
    """argparse type for four-character OpenCV capture format strings."""
    if len(value) != 4 or not value.isascii():
        raise argparse.ArgumentTypeError("must be exactly four ASCII characters")
    return value.upper()


def frame_size(frame) -> tuple[int, int]:
    """Return frame size as width, height."""
    height, width = frame.shape[:2]
    return int(width), int(height)


def decode_fourcc(value: float | int | None) -> str | None:
    """Decode an OpenCV CAP_PROP_FOURCC value to a readable string."""
    try:
        code = int(value) if value is not None else 0
    except (TypeError, ValueError, OverflowError):
        return None
    if code <= 0:
        return None

    decoded = []
    for index in range(4):
        byte = (code >> (8 * index)) & 0xFF
        if 32 <= byte <= 126:
            decoded.append(chr(byte))
        else:
            decoded.append(f"\\x{byte:02x}")
    return "".join(decoded)


def format_fps(fps: float | None) -> str:
    """Format FPS without noisy trailing decimals."""
    if fps is None:
        return "unknown"
    if float(fps).is_integer():
        return str(int(fps))
    return f"{fps:.2f}".rstrip("0").rstrip(".")


def format_requested_capture(settings: CaptureSettings) -> str:
    """Describe requested capture settings for logs."""
    resolution = "default"
    if settings.width is not None or settings.height is not None:
        width = settings.width if settings.width is not None else "?"
        height = settings.height if settings.height is not None else "?"
        resolution = f"{width}x{height}"

    fps = f"@{format_fps(settings.fps)}" if settings.fps is not None else ""
    capture_format = f" {settings.capture_format}" if settings.capture_format else ""
    return f"{resolution}{fps}{capture_format}"


def format_capture_status(status: CaptureStatus) -> str:
    """Describe effective capture settings for logs."""
    capture_format = status.capture_format or "unknown"
    return f"{status.width}x{status.height}@{format_fps(status.fps)} {capture_format}"


def capture_setting_warnings(
    requested: CaptureSettings,
    actual: CaptureStatus,
) -> list[str]:
    """Return warnings for requested capture settings that did not take effect."""
    warnings: list[str] = []
    if requested.width is not None and requested.width != actual.width:
        warnings.append(f"width requested {requested.width}, actual {actual.width}")
    if requested.height is not None and requested.height != actual.height:
        warnings.append(f"height requested {requested.height}, actual {actual.height}")
    if requested.fps is not None:
        if actual.fps is None:
            warnings.append(f"fps requested {format_fps(requested.fps)}, actual unknown")
        elif abs(requested.fps - actual.fps) > 0.5:
            warnings.append(f"fps requested {format_fps(requested.fps)}, actual {format_fps(actual.fps)}")
    if requested.capture_format is not None and requested.capture_format != actual.capture_format:
        warnings.append(f"format requested {requested.capture_format}, actual {actual.capture_format or 'unknown'}")
    return warnings


def serialize_reference_payload(
    frame,
    detected_markers: dict[int, np.ndarray],
) -> dict[str, object]:
    """Serialize reference markers with the frame size used for pixel coordinates."""
    width, height = frame_size(frame)
    return {
        "image_width": width,
        "image_height": height,
        "markers": {marker_id: corners.tolist() for marker_id, corners in detected_markers.items()},
    }


def parse_reference_payload(data: object) -> tuple[dict[int, np.ndarray], tuple[int, int] | None]:
    """Parse new and legacy reference JSON payloads."""
    if not isinstance(data, dict):
        raise ValueError("reference JSON must be an object")

    image_size = None
    markers = data
    if "markers" in data:
        markers = data["markers"]
        if not isinstance(markers, dict):
            raise ValueError("reference JSON field 'markers' must be an object")
        width = data.get("image_width")
        height = data.get("image_height")
        if width is not None and height is not None:
            image_size = (int(width), int(height))

    reference_data: dict[int, np.ndarray] = {}
    for marker_id, corners in markers.items():
        reference_data[int(marker_id)] = np.array(corners, dtype=np.float32)

    return reference_data, image_size


def reference_size_status(
    reference_size: tuple[int, int] | None,
    frame,
) -> str | None:
    """Return a compact status if frame size differs from the reference size."""
    if reference_size is None:
        return None

    current_width, current_height = frame_size(frame)
    reference_width, reference_height = reference_size
    if (current_width, current_height) == (reference_width, reference_height):
        return None

    return f"Size mismatch ref {reference_width}x{reference_height} current {current_width}x{current_height}"


def reference_size_warning(
    reference_size: tuple[int, int] | None,
    frame,
) -> str | None:
    """Return a warning when pixel-coordinate alignment errors are unreliable."""
    status = reference_size_status(reference_size, frame)
    if status is None:
        return None
    return format_reference_size_warning(status)


def format_reference_size_warning(size_status: str) -> str:
    """Expand a compact size status into a user-facing warning."""
    return f"{size_status}; alignment error is unreliable. Press 's' to save a new reference."


class OpenCVFrameSource:
    """Frame source backed by cv2.VideoCapture."""

    def __init__(
        self,
        camera_source: str | int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        capture_format: str | None = None,
    ):
        opencv = _require_opencv()
        self.camera_source = camera_source
        self.requested = CaptureSettings(
            width=width,
            height=height,
            fps=fps,
            capture_format=capture_format,
        )
        self._reported_capture = False
        self.capture = opencv.VideoCapture(camera_source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开摄像头 {camera_source}")

        self._apply_requested_settings(opencv)

    def _apply_requested_settings(self, opencv) -> None:
        if self.requested.width is not None:
            self.capture.set(opencv.CAP_PROP_FRAME_WIDTH, self.requested.width)
        if self.requested.height is not None:
            self.capture.set(opencv.CAP_PROP_FRAME_HEIGHT, self.requested.height)
        if self.requested.fps is not None:
            self.capture.set(opencv.CAP_PROP_FPS, self.requested.fps)
        if self.requested.capture_format is not None:
            fourcc_value = opencv.VideoWriter_fourcc(*self.requested.capture_format)
            self.capture.set(opencv.CAP_PROP_FOURCC, fourcc_value)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()
        if ok and frame is not None and not self._reported_capture:
            self._reported_capture = True
            self._report_effective_capture(frame)
        return ok, frame

    def _report_effective_capture(self, frame) -> None:
        opencv = _require_opencv()
        width, height = frame_size(frame)
        fps = self.capture.get(opencv.CAP_PROP_FPS)
        actual = CaptureStatus(
            width=width,
            height=height,
            fps=fps if fps is not None else None,
            capture_format=decode_fourcc(self.capture.get(opencv.CAP_PROP_FOURCC)),
        )
        print(
            f"camera_alignment opened {self.camera_source}: "
            f"requested {format_requested_capture(self.requested)}, "
            f"actual {format_capture_status(actual)}"
        )
        warnings = capture_setting_warnings(self.requested, actual)
        if warnings:
            print(f"⚠️ camera_alignment capture request mismatch: {'; '.join(warnings)}")

    def release(self) -> None:
        self.capture.release()


def create_aruco_detector():
    """Create a detector that works across OpenCV ArUco API versions."""
    opencv = _require_opencv()
    if hasattr(opencv.aruco, "DetectorParameters"):
        parameters = opencv.aruco.DetectorParameters()
    else:
        parameters = opencv.aruco.DetectorParameters_create()

    if hasattr(opencv.aruco, "ArucoDetector"):
        detector = opencv.aruco.ArucoDetector(
            opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_4X4_50),
            parameters,
        )
        return detector, parameters

    return None, parameters


class MultiCameraAligner:
    """Interactive marker-based camera alignment helper."""

    def __init__(
        self,
        reference_path: str | Path = "camera_reference_multi.json",
        reference_image_path: str | Path = "reference_img.png",
    ):
        opencv = _require_opencv()
        self.reference_path = Path(reference_path)
        self.reference_image_path = Path(reference_image_path)
        self.dictionary = opencv.aruco.getPredefinedDictionary(opencv.aruco.DICT_4X4_50)
        self.detector, self.parameters = create_aruco_detector()
        self.reference_size: tuple[int, int] | None = None
        self._reference_size_warning_printed = False
        self.reference_data = self.load_reference()

    def load_reference(self) -> dict[int, np.ndarray] | None:
        if not self.reference_path.exists():
            self.reference_size = None
            return None

        with open(self.reference_path, encoding="utf-8") as file:
            data = json.load(file)

        reference_data, self.reference_size = parse_reference_payload(data)
        return reference_data

    def detect_markers(self, frame) -> tuple[dict[int, np.ndarray], np.ndarray | None, list]:
        opencv = _require_opencv()
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(frame)
        else:
            corners, ids, rejected = opencv.aruco.detectMarkers(
                frame,
                self.dictionary,
                parameters=self.parameters,
            )
        if ids is None:
            return {}, None, rejected
        marker_ids = ids.flatten()
        detected = {int(marker_ids[index]): corners[index][0] for index in range(len(marker_ids))}
        return detected, ids, rejected

    def save_reference(self, frame) -> bool:
        opencv = _require_opencv()
        detected, _, _ = self.detect_markers(frame)
        if not detected:
            print("❌ 错误：当前画面没看到任何 ArUco 码，无法保存！")
            return False

        serialized = serialize_reference_payload(frame, detected)
        with open(self.reference_path, "w", encoding="utf-8") as file:
            json.dump(serialized, file, indent=2, ensure_ascii=False)
        opencv.imwrite(str(self.reference_image_path), frame)
        self.reference_data = self.load_reference()
        print(f"✅ 基准已更新，保存了 {len(detected)} 个 marker。")
        return True

    def get_alignment_error(self, frame) -> tuple[float | None, str]:
        detected, _, _ = self.detect_markers(frame)
        return self.get_alignment_status(detected, frame)

    def get_alignment_status(
        self,
        detected_markers: dict[int, np.ndarray],
        frame,
    ) -> tuple[float | None, str]:
        error_value, status = compute_alignment_error(self.reference_data, detected_markers)
        size_status = reference_size_status(self.reference_size, frame)
        if size_status is None:
            return error_value, status

        warning = format_reference_size_warning(size_status)
        if not self._reference_size_warning_printed:
            self._reference_size_warning_printed = True
            print(f"⚠️ {warning}")
        return error_value, f"{status} | {size_status}"

    def run_ghosting_ui(self, capture) -> None:
        opencv = _require_opencv()
        reference_image = opencv.imread(str(self.reference_image_path))
        if reference_image is None:
            print("❌ 找不到参考图，请先按 's' 保存")
            return

        print(">>> 虚影模式开启，按 'q' 退出")
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            error_value, status = self.get_alignment_error(frame)
            color = get_status_color(error_value)
            reference_resized = opencv.resize(reference_image, (frame.shape[1], frame.shape[0]))
            ghost = opencv.addWeighted(frame, 0.5, reference_resized, 0.5, 0)
            opencv.putText(
                ghost,
                f"GHOST MODE: {status}",
                (20, 50),
                opencv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            opencv.imshow("Ghosting_Mode", ghost)

            if opencv.waitKey(1) & 0xFF == ord("q"):
                break

        _safe_destroy_window("Ghosting_Mode")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Marker-based camera alignment helper",
    )
    parser.add_argument(
        "--cameras_index_or_path",
        required=True,
        help="Camera index or video device path",
    )
    parser.add_argument(
        "--reference-path",
        default="camera_reference_multi.json",
        help="Path to the saved reference marker JSON",
    )
    parser.add_argument(
        "--reference-image-path",
        default="reference_img.png",
        help="Path to the saved reference image",
    )
    parser.add_argument(
        "--width",
        type=positive_int,
        help="Requested capture width in pixels",
    )
    parser.add_argument(
        "--height",
        type=positive_int,
        help="Requested capture height in pixels",
    )
    parser.add_argument(
        "--fps",
        type=positive_float,
        help="Requested capture frame rate",
    )
    parser.add_argument(
        "--format",
        dest="capture_format",
        type=capture_format_text,
        help="Requested capture format, for example MJPG or YUYV",
    )
    parser.add_argument(
        "--fourcc",
        dest="capture_format",
        type=capture_format_text,
        help=argparse.SUPPRESS,
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    opencv = _require_opencv()

    capture = OpenCVFrameSource(
        normalize_camera_source(parsed.cameras_index_or_path),
        width=parsed.width,
        height=parsed.height,
        fps=parsed.fps,
        capture_format=parsed.capture_format,
    )

    aligner = MultiCameraAligner(
        reference_path=parsed.reference_path,
        reference_image_path=parsed.reference_image_path,
    )

    print("s: 保存当前 marker 作为基准")
    print("v: 进入虚影对齐模式")
    print("q: 退出")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detected, ids, _ = aligner.detect_markers(frame)
            error_value, status = aligner.get_alignment_status(detected, frame)
            color = get_status_color(error_value)

            display_frame = frame.copy()
            if ids is not None:
                marker_corners = [corners.reshape(1, 4, 2) for corners in detected.values()]
                opencv.aruco.drawDetectedMarkers(
                    display_frame,
                    marker_corners,
                    ids,
                )

            opencv.putText(
                display_frame,
                status,
                (10, 30),
                opencv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            opencv.imshow("Calibration_Monitor", display_frame)

            key = opencv.waitKey(1) & 0xFF
            if key == ord("s"):
                aligner.save_reference(frame)
            elif key == ord("v"):
                aligner.run_ghosting_ui(capture)
            elif key == ord("q"):
                break
    finally:
        capture.release()
        _safe_destroy_all_windows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

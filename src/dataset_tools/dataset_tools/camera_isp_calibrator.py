"""Interactive ISP color calibrator.

Runs alongside a live ``usb_cam`` node and lets the user converge on
exposure / white-balance / gain / brightness-contrast-saturation-sharpness
values that make the live feed match a reference image.

Designed for **idiot-proof, robust** operation:

* Single-window cv2 GUI — no extra dependencies (no PyQt).
* Visible HUD always shows current mode + key bindings.
* Sliders are debounced (param-set fires 400 ms after the last drag, never
  during drag) so we don't flood the camera node.
* Auto mode runs the solver in a worker thread; sliders stay locked while
  it is computing and the HUD turns yellow.
* Initial param snapshot is taken at startup; ``r`` always restores it,
  and ``q`` prompts before quitting if there are unsaved changes.
* All ROS / cv2 / camera-node failures are caught and displayed as a banner;
  the GUI never disappears mid-session. Persistent reconnect attempts are
  made silently in the background.
* ``s`` saves a JSON override at
  ``~/.ros/ibrobot/camera_isp_overrides/{camera}.json`` so the result is
  re-applied automatically on the next ``robot.launch.py``.

CLI:
    ros2 run dataset_tools camera_isp_calibrator \\
        --camera top --reference ref.png

Public surface kept narrow on purpose; future "Manual + bbox" mode will be
added as an additional keyboard mode without changing any current handler.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_tools.camera_isp.solver import auto_match_lab
from dataset_tools.camera_isp.sw_isp import SwIspParams, apply_sw_isp
from dataset_tools.opencv_utils import require_opencv_gui


# --------------------------------------------------------------------------
# Constants — keep param names in lock-step with usb_cam_node.cpp:65-85 and
# robot_config.launch_builders.camera_isp_overrides._ALLOWED_KEYS.
# --------------------------------------------------------------------------

_ALL_KEYS = (
    "exposure",
    "white_balance",
    "gain",
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "auto_white_balance",
    "autoexposure",
    "autofocus",
    "focus",
)

# Slider definitions: (display_label, key, min, max, default).
# Order matters — top-to-bottom in the UI.
_SLIDERS: tuple[tuple[str, str, int, int, int], ...] = (
    ("exposure",    "exposure",      1,  10000,  312),
    ("wb_kelvin",   "white_balance", 2000, 8000, 4600),
    ("gain",        "gain",          0,    255,    0),
    # Hardware brightness is one signed constant pixel offset. HighGUI
    # trackbars render signed ranges poorly across backends, so the UI
    # presents it as two mutually exclusive 0..64 controls:
    #   blacklevel  -> writes negative brightness
    #   brightness  -> writes positive brightness
    ("brightness",  "brightness",   -64,     64,    0),
    ("contrast",    "contrast",      0,    255,  128),
    ("saturation",  "saturation",    0,    255,  128),
    ("sharpness",   "sharpness",     0,    255,  128),
)

_SLIDER_KEYS = {key for _, key, _, _, _ in _SLIDERS}
_BOOL_ISP_KEYS = {"auto_white_balance", "autoexposure", "autofocus"}

_DEBOUNCE_S = 0.4
_RECONNECT_S = 2.0
_SOLVER_RETRIES = 4

WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
GREY = (128, 128, 128)


# --------------------------------------------------------------------------
# Override path — must mirror the loader on the robot_config side.
# Single source of truth for the path is intentionally NOT shared via import,
# because dataset_tools must work in environments where robot_config is not
# installed (e.g. dataset post-processing on a workstation). Both sides
# agree on this constant; a unit test pins the convention.
# --------------------------------------------------------------------------


def _override_path(camera_name: str) -> Path:
    ros_home = os.environ.get("ROS_HOME") or str(Path.home() / ".ros")
    return Path(ros_home) / "ibrobot" / "camera_isp_overrides" / f"{camera_name}.json"


# --------------------------------------------------------------------------
# ROS bridge — frame source + parameter client.
# Kept thin so the rest of the file stays testable without a live ROS stack.
# --------------------------------------------------------------------------


@dataclass
class _LatestFrame:
    """Mutable single-slot frame buffer guarded by a lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: np.ndarray | None = None
    stamp: float = 0.0


class RosBridge:
    """Encapsulates rclpy lifecycle: subscribe to image, get/set params.

    Constructed lazily — failure to import rclpy is reported with a clear
    message instead of a stack trace.
    """

    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.node_fqn = f"/{camera_name}_camera"
        self.image_topic = f"/camera/{camera_name}/image_raw"
        self._latest = _LatestFrame()
        self._spin_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connection_error: str | None = None
        # video_device path is resolved lazily after the node spins up so we
        # can talk to v4l2-ctl directly when usb_cam's hardcoded V4L2
        # control names mismatch the running kernel.
        self.video_device: str | None = None

        try:
            import rclpy  # noqa: F401
            from rclpy.node import Node  # noqa: F401
            from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: F401
            from sensor_msgs.msg import Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - install-time only
            raise RuntimeError(
                "rclpy / sensor_msgs not available — source your ROS 2 "
                "workspace before running camera_isp_calibrator."
            ) from exc

        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self._node: Node = rclpy.create_node(f"camera_isp_calibrator_{camera_name}")
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self._node.create_subscription(Image, self.image_topic, self._on_image, qos)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    # ------------------------------------------------------------------
    # Frame intake
    # ------------------------------------------------------------------

    def _on_image(self, msg) -> None:
        try:
            frame = self._decode(msg)
        except Exception as exc:  # noqa: BLE001 — robustness
            self._connection_error = f"decode failed: {exc}"
            return
        if frame is None:
            return
        now = time.monotonic()
        with self._latest.lock:
            self._latest.frame = frame
            self._latest.stamp = now
        # [PublishRate-debug] commented out — was spamming terminal every second.
        # st = self.__dict__.setdefault("_rate_state", {"t": now, "n": 0})
        # st["n"] += 1
        # if now - st["t"] >= 1.0:
        #     print(f"[PublishRate] {st['n']} fps", flush=True)
        #     st["t"] = now
        #     st["n"] = 0

    @staticmethod
    def _decode(msg) -> np.ndarray | None:
        h, w, enc = msg.height, msg.width, msg.encoding
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in ("bgr8", "rgb8"):
            arr = data.reshape(h, w, 3)
            return arr[:, :, ::-1].copy() if enc == "rgb8" else arr.copy()
        if enc in ("bgra8", "rgba8"):
            arr = data.reshape(h, w, 4)[:, :, :3]
            return arr[:, :, ::-1].copy() if enc == "rgba8" else arr.copy()
        if enc == "mono8":
            mono = data.reshape(h, w)
            return np.stack([mono] * 3, axis=-1)
        return None

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._rclpy.spin_once(self._node, timeout_sec=0.05)
            except Exception:  # noqa: BLE001
                # Don't let spin failures kill the thread; the GUI thread
                # will surface the staleness via "live feed lost" banner.
                time.sleep(0.1)

    def latest_frame(self) -> tuple[np.ndarray | None, float]:
        with self._latest.lock:
            return (
                None if self._latest.frame is None else self._latest.frame.copy(),
                self._latest.stamp,
            )

    # ------------------------------------------------------------------
    # Parameter get/set — synchronous service calls with bounded timeout.
    # Failures are reported as bool + message; never raise.
    # ------------------------------------------------------------------

    def get_params(self, keys: tuple[str, ...]) -> tuple[dict[str, Any], str | None]:
        """Return current values for the requested keys via service call."""
        from rcl_interfaces.srv import GetParameters

        client = self._node.create_client(GetParameters, f"{self.node_fqn}/get_parameters")
        try:
            if not client.wait_for_service(timeout_sec=2.0):
                return {}, f"camera node {self.node_fqn} not responding"
            req = GetParameters.Request()
            req.names = list(keys)
            future = client.call_async(req)
            self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=2.0)
            if not future.done():
                return {}, "get_parameters timed out"
            resp = future.result()
            out: dict[str, Any] = {}
            for k, val in zip(keys, resp.values):
                out[k] = self._unwrap(val)
            return out, None
        except Exception as exc:  # noqa: BLE001
            return {}, f"get_parameters failed: {exc}"
        finally:
            self._node.destroy_client(client)

    def set_params(self, params: Mapping[str, Any]) -> tuple[bool, str | None]:
        """Apply a batch of parameters. Returns (all_succeeded, error_msg)."""
        if not params:
            return True, None
        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
        from rcl_interfaces.srv import SetParameters

        client = self._node.create_client(SetParameters, f"{self.node_fqn}/set_parameters")
        try:
            if not client.wait_for_service(timeout_sec=2.0):
                return False, f"camera node {self.node_fqn} not responding"
            req = SetParameters.Request()
            for k, v in params.items():
                p = Parameter()
                p.name = k
                pv = ParameterValue()
                if isinstance(v, bool):
                    pv.type = ParameterType.PARAMETER_BOOL
                    pv.bool_value = bool(v)
                elif isinstance(v, int):
                    pv.type = ParameterType.PARAMETER_INTEGER
                    pv.integer_value = int(v)
                elif isinstance(v, float):
                    pv.type = ParameterType.PARAMETER_DOUBLE
                    pv.double_value = float(v)
                else:
                    return False, f"unsupported param type for {k}: {type(v).__name__}"
                p.value = pv
                req.parameters.append(p)
            future = client.call_async(req)
            self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=3.0)
            if not future.done():
                return False, "set_parameters timed out"
            results = future.result().results
            failures = [
                f"{p.name}: {r.reason}"
                for p, r in zip(req.parameters, results)
                if not r.successful
            ]
            if failures:
                return False, "; ".join(failures)
            return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"set_parameters failed: {exc}"
        finally:
            self._node.destroy_client(client)

    @staticmethod
    def _unwrap(v) -> Any:
        from rcl_interfaces.msg import ParameterType

        t = v.type
        if t == ParameterType.PARAMETER_BOOL:
            return bool(v.bool_value)
        if t == ParameterType.PARAMETER_INTEGER:
            return int(v.integer_value)
        if t == ParameterType.PARAMETER_DOUBLE:
            return float(v.double_value)
        if t == ParameterType.PARAMETER_STRING:
            return str(v.string_value)
        return None

    def shutdown(self) -> None:
        self._stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        try:
            self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# Reference loader — image OR video; auto-resize to live frame size.
# --------------------------------------------------------------------------


def load_reference(path: str | Path, opencv) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"reference not found: {p}")
    suffix = p.suffix.lower()
    if suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        cap = opencv.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open reference video: {p}")
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"reference video has no readable frame: {p}")
            return frame
        finally:
            cap.release()
    img = opencv.imread(str(p))
    if img is None:
        raise RuntimeError(f"cannot decode reference image: {p}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError(f"reference must be 3-channel BGR, got shape {img.shape}")
    return img


# --------------------------------------------------------------------------
# StageBridge adapter — connects ``hw_pipeline.run_full_pipeline`` to the
# live ROS / V4L2 surface owned by :class:`CalibratorWindow`. Kept as a
# thin private class so the orchestrator stays GUI-agnostic and unit
# tests inject their own fakes.
# --------------------------------------------------------------------------


class _CalibratorStageBridge:
    """Adapt :class:`CalibratorWindow` to the ``hw_pipeline.StageBridge`` protocol."""

    #: Settling delay after a hardware write before the next ``grab_frame``
    #: call. Plan §1: each stage must observe the *post-write* state, not
    #: the pre-write one.
    _SETTLING_S = 0.25

    #: Maximum age (seconds) of a frame still considered "live" — older
    #: than this and ``grab_frame`` returns ``None`` so the orchestrator
    #: aborts the stage rather than consuming a stale frame.
    _STALE_S = 1.0

    def __init__(self, calibrator: "CalibratorWindow") -> None:
        self._calib = calibrator
        # ROI boxes (live-side coordinates) the bridge applies to every
        # ``grab_frame`` call. ``None`` / empty ⇒ full frame is returned
        # (legacy behaviour). Set by :meth:`CalibratorWindow._run_auto_pipeline`
        # before invoking ``run_full_pipeline`` whenever ``_roi_pairs``
        # is non-empty.
        self._live_roi_boxes: list[tuple[int, int, int, int]] | None = None

    # --- StageBridge --------------------------------------------------

    def grab_frame(self) -> "np.ndarray | None":
        live, stamp = self._calib._bridge.latest_frame()
        if live is None or time.monotonic() - stamp > self._STALE_S:
            return None
        if self._live_roi_boxes:
            from dataset_tools.camera_isp.hw_stages import extract_roi_frame
            return extract_roi_frame(live, self._live_roi_boxes)
        return live

    def write_v4l2(self, params: Mapping[str, int]) -> None:
        if not params:
            return
        if getattr(self._calib, "_protect_brightness_in_auto_pipeline", False):
            params = {k: v for k, v in params.items() if k != "brightness"}
            if not params:
                time.sleep(self._SETTLING_S * 0.5)
                return
        # Write-diff: skip keys whose value matches what we last applied
        # successfully. Coarse coordinate-descent search varies one of
        # K/C/Sat per eval, so 2/3 of writes are pure no-ops that still
        # pause usb_cam's stream → frozen GUI preview + stuck grabs.
        diff = {
            k: v for k, v in params.items()
            if self._calib._applied.get(k) != v
        }
        if not diff:
            # Nothing changed → no V4L2 thrash, but still settle a bit
            # so the next grab gate has time to receive a publisher
            # heartbeat; halve the settle since the camera pipeline
            # is undisturbed.
            time.sleep(self._SETTLING_S * 0.5)
            return
        ok, err = self._calib._apply_via_v4l2_or_ros(dict(diff))
        if not ok:
            # Surface but don't abort — orchestrator records the stage
            # outcome from the grabbed frame on the next iteration.
            self._calib._notify(f"v4l2 write failed: {err}", RED, 4.0)
            return
        self._calib._applied.update(diff)
        time.sleep(self._SETTLING_S)

    def get_caps(self, key: str):
        from dataset_tools.camera_isp.hw_stages import CtrlCaps
        cap = self._calib._device_caps.get(key)
        if not cap:
            return None
        lo = int(cap["min"])
        hi = int(cap["max"])
        # Plan §2.3 brightness fallback / §3.3 sat-con band require a
        # ``default``; if v4l2-ctl didn't report one, fall back to the
        # midpoint as a neutral anchor.
        default = int(cap.get("default", (lo + hi) // 2))
        return CtrlCaps(minimum=lo, maximum=hi, default=default)

    def get_current(self, key: str) -> int:
        # Falls back to the v4l2 default (and finally 0) so the orchestrator
        # never reads a phantom 0 — propose_exposure multiplies by current
        # value, and "0" would collapse the entire stage to exp_min.
        applied = self._calib._applied
        if key in applied and isinstance(applied[key], int):
            return int(applied[key])
        cap = self._calib._device_caps.get(key, {})
        if "default" in cap:
            return int(cap["default"])
        return 0


# --------------------------------------------------------------------------
# Calibrator window — pure UI orchestration.
# --------------------------------------------------------------------------


class CalibratorWindow:
    WINDOW_NAME = "Camera ISP Calibrator"

    def __init__(
        self,
        bridge: RosBridge,
        reference: np.ndarray,
        opencv,
        *,
        colorchecker: bool = False,
        max_exposure_ms: float | None = None,
    ):
        self._bridge = bridge
        self._opencv = opencv
        self._reference = reference
        # ColorChecker24 mode: AUTO disabled, SW-only forced, manual
        # routes to the 24-patch wizard instead of pair-drag.
        self._colorchecker_mode: bool = bool(colorchecker)
        # AUTO Stage 1 exposure ceiling in ms; None ⇒ project default
        # (15 ms). Pulled by _run_auto_pipeline when computing exp_max.
        self._max_exposure_ms: float | None = max_exposure_ms

        # ROI pairs collected by `_run_manual` (pair-drag). When
        # populated, the next AUTO run computes ALL stage statistics
        # (Y_mean / chroma / Kelvin) on these regions only — both the
        # live frame *and* the reference image are masked to the
        # paired boxes. This lets the user point the calibrator at a
        # specific subject (color chart, robot head, …) instead of
        # averaging the whole scene. Boxes are kept across AUTO runs
        # until cleared via the dedicated key (see `_run_loop`).
        # Format: list of (ref_box, live_box) tuples; each box is
        # (x1, y1, x2, y2) in the source image's pixel coordinates.
        self._roi_pairs: list[
            tuple[tuple[int, int, int, int], tuple[int, int, int, int]]
        ] = []

        # Param state -- triplet model: initial / current applied / pending UI value.
        self._initial: dict[str, Any] = {}
        self._applied: dict[str, Any] = {}
        self._pending: dict[str, int] = {}
        self._last_change: dict[str, float] = {}
        # Set True while we are programmatically moving sliders so the
        # trackbar callback can short-circuit and not bounce values back.
        self._suppress_cb = False
        self._trackbars_ready = False
        self._main_window_seen = False
        self._brightness_user_locked = False
        self._protect_brightness_in_auto_pipeline = False
        # Updated by _force_manual_modes(): True if AWB really turned off at
        # the driver level (not just the ROS parameter store).
        self._wb_writable = True
        self._device_caps: dict[str, dict[str, int]] = {}
        self._slider_range: dict[str, tuple[int, int]] = {}
        self._v4l2_resolved: dict = {}

        # Single-step undo stack: each entry is a snapshot of
        # ``self._applied`` (numeric / bool keys only) taken just before
        # a top-level user action that may write hardware (a / m / r /
        # cc24 wizard apply). ``u`` pops one entry and writes it back.
        # Capped to avoid unbounded growth in long sessions.
        self._undo_stack: list[dict[str, Any]] = []
        self._undo_cap: int = 16

        # Mode flags (set by handlers, read by render loop).
        self._mode = "IDLE"
        self._banner: tuple[str, tuple[int, int, int]] | None = None
        self._banner_until = 0.0
        self._auto_thread: threading.Thread | None = None
        self._dirty_save = False
        self._compute_progress = (0, 0)
        self._last_pedestal_delta: int | None = None
        self._last_pedestal_used: str = ""
        self._stop_render = False

        # Reference image display rotation (degrees CW: 0 / 90 / 180 / 270).
        # Only affects display and user-facing selectROI; self._reference
        # is always the original unrotated image for solver accuracy.
        self._ref_rotation: int = 0
        # Canvas-space bounding rect of the [R] rotate button, updated each
        # _render() call.  None until the first render.
        self._ref_rot_btn_rect: tuple[int, int, int, int] | None = None

        # SW-ISP debug pane state.
        # `_sw_isp` mirrors whatever the last solver call recommended, in
        # the linear-domain form the SW-ISP applies pixel-wise. It is
        # ALWAYS kept in sync with solver output (regardless of whether
        # we wrote to hardware), so the third pane is a faithful
        # "this is what the solution should look like" preview.
        # `_sw_only` is toggled by 'd': when True we render the third
        # pane and skip *all* hardware writes from m/a (the user wants
        # to compare cleanly without the camera moving under them).
        self._sw_isp: SwIspParams = SwIspParams()
        # The SW-ISP preview pane is the third on-canvas window. It used
        # to be auto-enabled in cc24 mode (when the workflow was
        # SW-only). The cc24 path now drives the camera through
        # cost_24card so the live pane already shows the corrected
        # output -- the SW-ISP pane would just duplicate it. Keep the
        # flag at False; the user can still toggle the pane manually
        # with 'd' in REF mode.
        self._sw_only: bool = False
        self._sw_isp_neutral_n: int = 0       # px count that drove WB (0 = no neutral box)
        self._sw_isp_ref_saturated: bool = False  # True = ref pixels were all clipped
        self._sw_isp_ccm_pairs: int = 0       # >0 means CCM mode active (3x3 in _sw_isp.ccm)
        # CCM variant cache (populated by _update_sw_isp_from_dbg). Keys:
        # "linear" / "rpcc2" / "rpcc2_ridge" / "rpcc2_als". Each value is a
        # dict {"M","feat_dim","n_pairs","lambda","iters","delta_e_median"}
        # or None when that variant was unavailable for the last solve.
        self._ccm_variants: dict[str, dict | None] = {
            "linear": None,
            "rpcc2": None,
            "rpcc2_ridge": None,
            "rpcc2_als": None,
        }
        # Currently-selected variant (toggled by '0' / '1' / '2' / '3').
        # Default: "linear" matches pre-RPCC behaviour bit-for-bit.
        self._ccm_variant: str = "linear"
        # True once any solver call has populated _sw_isp with real values.
        # Used to distinguish "never run" (show placeholder) from
        # "ran and gave kr=1" (show corrected frame, which may be identical).
        self._sw_isp_computed: bool = False

    # ------------------------------------------------------------------
    # Init / teardown
    # ------------------------------------------------------------------

    def _snapshot_initial(self) -> bool:
        keys = _ALL_KEYS + ("video_device",)
        params, err = self._bridge.get_params(keys)
        if err:
            self._notify(f"Init read failed: {err}", RED, 5.0)
            # Fall back to slider defaults.
            params = {key: default for _, key, _, _, default in _SLIDERS}
            params["auto_white_balance"] = True
            params["autoexposure"] = True
        # Surface video_device to the bridge for v4l2-ctl direct access.
        vdev = params.pop("video_device", None)
        if isinstance(vdev, str) and vdev:
            self._bridge.video_device = vdev
        self._initial = dict(params)
        self._applied = dict(params)
        self._pending = {
            key: int(params.get(key, default))
            for _, key, _, _, default in _SLIDERS
        }
        # Pull real device-capability ranges (min/max/default) for every
        # control the 4-stage hardware calibrator drives. Without this we
        # would push wb to 2000 K on a camera whose real range is
        # 2800-6500 K and look like Auto is broken when it's just clamped
        # to slider rails. ``default`` is also required by Stage 2 / 3 to
        # size the brightness fallback band and the ``default ± 30%``
        # saturation/contrast band per plan §2.1 / §2.3.
        self._device_caps: dict[str, dict[str, int]] = {}
        if self._bridge.video_device:
            try:
                from dataset_tools.camera_isp import v4l2_ctl as _v4l2
                if _v4l2.have_v4l2_ctl():
                    resolved = _v4l2.resolve_ctrls(self._bridge.video_device)
                    self._v4l2_resolved = resolved
                    for logical_key in (
                        "white_balance", "exposure",
                        "gain", "brightness", "saturation", "contrast",
                        "sharpness", "focus",
                    ):
                        info = resolved.get(logical_key)
                        if info and info.minimum is not None and info.maximum is not None:
                            cap = {
                                "min": int(info.minimum),
                                "max": int(info.maximum),
                            }
                            if info.default is not None:
                                cap["default"] = int(info.default)
                            if info.value is not None:
                                cap["value"] = int(info.value)
                            self._device_caps[logical_key] = cap
            except Exception as exc:  # noqa: BLE001
                self._notify(f"v4l2-ctl probe failed: {exc}", YELLOW, 4.0)
        self._normalize_initial_snapshot()
        return err is None

    def _normalize_isp_value(self, key: str, value: Any) -> Any | None:
        if key in _BOOL_ISP_KEYS:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            return None
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None
        if key != "brightness" and ivalue < 0:
            cap = self._device_caps.get(key, {})
            cap_value = cap.get("value")
            cap_default = cap.get("default")
            if cap_value is not None:
                ivalue = int(cap_value)
            elif cap_default is not None:
                ivalue = int(cap_default)
            else:
                return None
        cap = self._device_caps.get(key)
        if cap is not None:
            ivalue = max(int(cap["min"]), min(int(cap["max"]), ivalue))
        return ivalue

    def _normalize_initial_snapshot(self) -> None:
        for key in _ALL_KEYS:
            if key not in self._applied:
                continue
            normalized = self._normalize_isp_value(key, self._applied[key])
            if normalized is None:
                self._applied.pop(key, None)
                self._initial.pop(key, None)
                continue
            self._applied[key] = normalized
            self._initial[key] = normalized
            if key in _SLIDER_KEYS:
                self._pending[key] = int(normalized)

    def _force_manual_modes(self) -> None:
        """Disable AWB / AE / AF at the V4L2 driver level (bypassing usb_cam).

        usb_cam (apt-installed) hardcodes legacy control names that newer
        kernels rejected as 'unknown control'. Going straight to v4l2-ctl
        with new-name->legacy-name fallback guarantees the driver actually
        engages manual mode, otherwise every later white_balance write is
        denied with 'Permission denied' and Auto mode looks broken.

        We deliberately do NOT touch the usb_cam ROS parameters
        (`auto_white_balance` / `autoexposure`) here — setting them via
        ROS makes the node re-issue its full broken setup chain on every
        call, stalling the SetParameters service and breaking Auto/Reset.
        """
        from dataset_tools.camera_isp import v4l2_ctl as _v4l2
        device = self._bridge.video_device
        if not device or not _v4l2.have_v4l2_ctl():
            self._notify(
                "v4l2-ctl not available; manual WB cannot be guaranteed",
                YELLOW, 4.0,
            )
            self._wb_writable = False
            return
        resolved = getattr(self, "_v4l2_resolved", None) or _v4l2.resolve_ctrls(device)
        self._v4l2_resolved = resolved
        msgs = _v4l2.force_manual_modes(device, resolved)
        if _v4l2.verify_manual_wb(device, resolved):
            self._wb_writable = True
            self._applied["auto_white_balance"] = False
            self._applied["autoexposure"] = False
            self._notify("Manual WB / AE engaged at driver", GREEN, 3.0)
        else:
            self._wb_writable = False
            self._notify(
                "AWB still on at driver -- manual WB will be denied. "
                + " | ".join(msgs)[:120],
                RED, 6.0,
            )

    def _ui_exposure_max_ticks(self) -> int | None:
        """Compute the UI-side exposure ceiling in raw V4L2 ticks.

        Mirrors the formula in :meth:`_run_auto_pipeline` so the slider
        cannot offer values that AUTO would refuse to use anyway. We
        deliberately apply this clamp at the *UI* layer (rather than
        only inside the AUTO pipeline) because users routinely launch
        the calibrator after a previous run left exposure parked above
        the safe rail — the slider would otherwise show 300+ ticks on
        a budget that physically tops out at 150, leading to confusion
        and to Stage 1 having to pull exposure down on its first iter
        (wasting an iteration / a frame).

        Returns ``None`` when the device cap is unknown — caller falls
        back to ``_SLIDERS`` defaults.
        """
        from dataset_tools.camera_isp.exposure_units import (
            DEFAULT_MAX_EXPOSURE_MS,
            compute_exposure_max_us,
            probe_exposure,
            ticks_from_us,
        )
        device = self._bridge.video_device
        probe = probe_exposure(device) if device else None
        fps = probe.fps if (probe and probe.fps) else None
        unit_us = probe.unit_us if probe else None
        cli_max_ms = float(getattr(self, "_max_exposure_ms", None)
                           or DEFAULT_MAX_EXPOSURE_MS)
        exp_max_us = compute_exposure_max_us(cli_max_ms, fps)
        if unit_us:
            ticks = ticks_from_us(exp_max_us, unit_us)
        else:
            return None
        device_max = self._device_caps.get("exposure", {}).get("max")
        if device_max is not None:
            ticks = min(ticks, device_max)
        return int(ticks)

    def _build_window(self) -> None:
        cv = self._opencv
        cv.namedWindow(self.WINDOW_NAME, cv.WINDOW_AUTOSIZE)
        self._trackbars_ready = False
        # Cache effective (lo, hi) per slider so render / sync code uses the
        # device-capability ranges instead of the conservative built-in ones.
        self._slider_range: dict[str, tuple[int, int]] = {}
        # Apply the AUTO-mode exposure ceiling to the UI as well so the
        # slider never offers values the pipeline would refuse.
        exp_ui_cap = self._ui_exposure_max_ticks()
        for label, key, lo, hi, _default in _SLIDERS:
            cap = getattr(self, "_device_caps", {}).get(key)
            if cap is not None:
                lo, hi = cap["min"], cap["max"]
            if key == "exposure" and exp_ui_cap is not None:
                hi = min(hi, exp_ui_cap)
            self._slider_range[key] = (lo, hi)
            if key == "brightness":
                signed_bri = self._brightness_param_to_signed(
                    int(self._pending.get(key, 0)),
                )
                blacklevel_value = max(0, -signed_bri)
                brightness_value = max(0, signed_bri)
                cv.createTrackbar(
                    "blacklevel", self.WINDOW_NAME,
                    int(blacklevel_value), 64,
                    self._make_brightness_split_cb("blacklevel"),
                )
                cv.createTrackbar(
                    "brightness", self.WINDOW_NAME,
                    int(brightness_value), 64,
                    self._make_brightness_split_cb("brightness"),
                )
                continue
            ui_lo, ui_hi = self._slider_ui_range(key, lo, hi)
            value_raw = self._pending.get(key, lo)
            value = self._param_to_slider_value(key, int(value_raw), lo, hi)
            value = max(ui_lo, min(ui_hi, value))
            # Clamp _applied / _pending for exposure so the slider
            # snapshot is consistent and the next AUTO sees the same
            # current_exp the user does. We also push the clamped
            # value to hardware below if it's out of range.
            if key == "exposure" and exp_ui_cap is not None:
                applied_exp = self._applied.get("exposure")
                if isinstance(applied_exp, int) and applied_exp > hi:
                    self._applied["exposure"] = hi
                    self._pending["exposure"] = hi
                    value = self._param_to_slider_value(key, hi, lo, hi)
                    # Best-effort write — it's fine if it fails because
                    # the driver hasn't engaged manual mode yet; the
                    # next AUTO run will re-clamp.
                    try:
                        self._apply_via_v4l2_or_ros({"exposure": hi})
                    except Exception:  # noqa: BLE001
                        pass
            cv.createTrackbar(
                label, self.WINDOW_NAME, max(0, value), max(0, ui_hi),
                self._make_trackbar_cb(key, lo, hi),
            )
        cv.setMouseCallback(self.WINDOW_NAME, self._on_main_mouse)
        self._trackbars_ready = True

    def _slider_ui_range(self, key: str, lo: int, hi: int) -> tuple[int, int]:
        if key != "brightness":
            return int(lo), int(hi)
        return 0, 64

    def _brightness_is_signed_control(self) -> bool:
        _bri_default, bri_min, bri_max = self._bri_default_minmax()
        return int(bri_min) < 0 < int(bri_max)

    def _brightness_param_to_signed(self, value: int) -> int:
        if self._brightness_is_signed_control():
            return max(-64, min(64, int(value)))
        bri_default, _bri_min, _bri_max = self._bri_default_minmax()
        return max(-64, min(64, int(value) - int(bri_default)))

    def _brightness_signed_to_param(self, signed_value: int) -> int:
        signed_value = max(-64, min(64, int(signed_value)))
        bri_default, bri_min, bri_max = self._bri_default_minmax()
        if self._brightness_is_signed_control():
            return max(bri_min, min(bri_max, signed_value))
        return max(bri_min, min(bri_max, int(bri_default) + signed_value))

    def _param_to_slider_value(self, key: str, value: int, lo: int, hi: int) -> int:
        if key != "brightness":
            return max(int(lo), min(int(hi), int(value)))
        return abs(self._brightness_param_to_signed(int(value)))

    def _slider_value_to_param(self, key: str, value: int, lo: int, hi: int) -> int:
        if key != "brightness":
            return max(int(lo), min(int(hi), int(value)))
        return self._brightness_signed_to_param(int(value))

    def _make_brightness_split_cb(self, kind: str):
        def cb(val: int) -> None:
            if self._mode != "IDLE" or self._suppress_cb:
                return
            magnitude = max(0, min(64, int(val)))
            signed_value = -magnitude if kind == "blacklevel" else magnitude
            opposite = "brightness" if kind == "blacklevel" else "blacklevel"
            if magnitude > 0:
                self._suppress_cb = True
                try:
                    self._opencv.setTrackbarPos(opposite, self.WINDOW_NAME, 0)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    self._suppress_cb = False
            self._pending["brightness"] = self._brightness_signed_to_param(
                signed_value,
            )
            self._last_change["brightness"] = time.monotonic()
            if getattr(self, "_trackbars_ready", False):
                self._brightness_user_locked = True
            self._dirty_save = True
        return cb

    def _make_trackbar_cb(self, key: str, lo: int, hi: int):
        def cb(val: int) -> None:
            # Don't apply yet — just record. The render loop debounces.
            if self._mode != "IDLE":
                # While AUTO/SEARCH/wizard is running we ignore user
                # fiddling; sliders will snap back to the applied values
                # on the next render tick (see render-loop sync below).
                return
            if self._suppress_cb:
                # Ignore callbacks fired by our own setTrackbarPos() calls,
                # otherwise reset/sync would round-trip into _pending and
                # clobber the value we just restored.
                return
            self._pending[key] = self._slider_value_to_param(key, val, lo, hi)
            self._last_change[key] = time.monotonic()
            self._dirty_save = True
        return cb

    # ------------------------------------------------------------------
    # Notification banner
    # ------------------------------------------------------------------

    def _notify(self, text: str, color: tuple[int, int, int], duration: float) -> None:
        self._banner = (text, color)
        self._banner_until = time.monotonic() + duration

    # ------------------------------------------------------------------
    # Main-window mouse callback
    # ------------------------------------------------------------------

    def _on_main_mouse(
        self,
        event: int,
        x: int,
        y: int,
        _flags: int,
        _userdata: object,
    ) -> None:
        """Handle left-clicks on the [R] rotate button in the ref pane."""
        if event != self._opencv.EVENT_LBUTTONDOWN:
            return
        btn = self._ref_rot_btn_rect
        if btn is None:
            return
        if btn[0] <= x <= btn[2] and btn[1] <= y <= btn[3]:
            self._ref_rotation = (self._ref_rotation + 90) % 360
            if self._roi_pairs:
                self._roi_pairs = []
                self._notify(
                    f"Ref rotated to {self._ref_rotation}deg — ROI pairs cleared",
                    YELLOW, 3.0,
                )
            else:
                self._notify(
                    f"Ref rotated to {self._ref_rotation}deg",
                    WHITE, 2.0,
                )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self, live: np.ndarray | None, live_stale: bool) -> np.ndarray:
        cv = self._opencv
        ref = self._get_rotated_ref()

        # Resize live to reference height for side-by-side.
        target_h = 480
        ref_view = self._fit(ref, target_h)
        live_view = (
            self._fit(live, target_h)
            if live is not None
            else self._placeholder(target_h, "WAITING FOR LIVE FRAME...")
        )

        # SW-ISP debug pane (only when toggled on with 'd').
        #
        # Two rendering modes share this pane:
        # * **CCM mode** (preferred when solver returns a 3x3): apply the
        #   full matrix straight from cur→ref. The CCM already absorbs
        #   exposure and per-channel gains, so nothing needs pinning.
        # * **Diag mode** (fallback): apply only kr/kb in linear light
        #   with exp_scale pinned to 1.0 — channel gains evaluated at
        #   unit exposure so the pane doesn't wash out from clipping.
        sw_view: np.ndarray | None = None
        if self._sw_only and live is not None:
            if not self._sw_isp_computed:
                # Solver has not been run yet — guide the user explicitly
                # rather than showing an identity-transform copy of live
                # (which looks indistinguishable and is confusing).
                sw_view = self._placeholder(
                    target_h,
                    "Press [m] or [a] to compute WB",
                )
            else:
                try:
                    if self._sw_isp.ccm is not None:
                        # CCM mode: feed the matrix as-is.
                        sw_params = SwIspParams(ccm=self._sw_isp.ccm)
                    else:
                        # Diag mode: WB-only preview, pin exp=1.
                        sw_params = SwIspParams(
                            kr=self._sw_isp.kr,
                            kb=self._sw_isp.kb,
                            exp_scale=1.0,
                        )
                    sw_full = apply_sw_isp(live, sw_params)
                    sw_view = self._fit(sw_full, target_h)
                except Exception as exc:  # noqa: BLE001
                    sw_view = self._placeholder(target_h, f"SW-ISP error: {exc}")

        gap = 8
        canvas_w = ref_view.shape[1] + live_view.shape[1] + gap
        if sw_view is not None:
            canvas_w += sw_view.shape[1] + gap
        canvas = np.full((target_h + 80, canvas_w, 3), 30, dtype=np.uint8)
        canvas[:target_h, : ref_view.shape[1]] = ref_view
        live_x0 = ref_view.shape[1] + gap
        canvas[:target_h, live_x0 : live_x0 + live_view.shape[1]] = live_view
        if sw_view is not None:
            sw_x0 = live_x0 + live_view.shape[1] + gap
            canvas[:target_h, sw_x0 : sw_x0 + sw_view.shape[1]] = sw_view

        # Labels under each pane.
        cv.putText(canvas, "REFERENCE", (10, target_h + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        # [R] rotate-90 button — bottom-right corner of the ref pane.
        rot_label = f"[R{self._ref_rotation}]"
        (rtw, rth), _ = cv.getTextSize(
            rot_label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1,
        )
        btn_x = ref_view.shape[1] - rtw - 10
        btn_y_top = target_h + 4
        btn_y_bot = target_h + 4 + rth + 8
        cv.rectangle(
            canvas,
            (btn_x - 4, btn_y_top),
            (btn_x + rtw + 4, btn_y_bot),
            (70, 70, 70), -1,
        )
        cv.putText(
            canvas, rot_label,
            (btn_x, btn_y_bot - 4),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv.LINE_AA,
        )
        self._ref_rot_btn_rect = (
            btn_x - 4, btn_y_top,
            btn_x + rtw + 4, btn_y_bot,
        )
        live_label = "LIVE  (stale!)" if live_stale else "LIVE"
        if self._sw_only:
            live_label += "  [hw frozen]"
        cv.putText(canvas, live_label,
                   (live_x0 + 10, target_h + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   YELLOW if (live_stale or self._sw_only) else WHITE, 2)
        if sw_view is not None:
            if not self._sw_isp_computed:
                sw_label = "SW-ISP  [press m or a first]"
                sw_label_color = YELLOW
            elif self._sw_isp.ccm is not None:
                # CCM mode (linear or RPCC2). Surface variant name + ΔE so
                # the user can compare 0/1/2/3 quickly.
                M = self._sw_isp.ccm
                variant_name = self._ccm_variant
                entry = self._ccm_variants.get(variant_name) or {}
                feat_dim = int(entry.get("feat_dim", M.shape[1]))
                de = entry.get("delta_e_median")
                # NB: cv2 HERSHEY fonts only render ASCII — do NOT use λΔ→
                # here, they show as `???` and confuse the user.
                de_str = f"  dE={de:.2f}" if de is not None else ""
                lam = entry.get("lambda")
                lam_str = f"  lam={lam:.0e}" if lam is not None else ""
                it = entry.get("iters")
                it_str = f"  it={it}" if it is not None else ""
                if M.shape == (3, 3):
                    diag = (M[0, 0], M[1, 1], M[2, 2])
                    detail = (
                        f"diag=({diag[0]:.2f},{diag[1]:.2f},{diag[2]:.2f})"
                    )
                else:  # (3, 6) RPCC2
                    diag = (M[0, 0], M[1, 1], M[2, 2])  # linear part
                    detail = (
                        f"lin=({diag[0]:.2f},{diag[1]:.2f},{diag[2]:.2f})"
                    )
                sw_label = (
                    f"SW-ISP[{variant_name}]  feat={feat_dim}  {detail}"
                    f"  pairs={self._sw_isp_ccm_pairs}{lam_str}{it_str}{de_str}"
                )
                if self._colorchecker_mode:
                    sw_label += "  cc24"
                sw_label_color = GREEN
            else:
                kr, kb = self._sw_isp.kr, self._sw_isp.kb
                chroma_ok = abs(kr - 1.0) < 0.02 and abs(kb - 1.0) < 0.02
                if self._sw_isp_neutral_n == 0:
                    note = "NO neutral boxes"
                    sw_label_color = RED
                elif self._sw_isp_ref_saturated:
                    note = "REF saturated"
                    sw_label_color = YELLOW
                elif chroma_ok:
                    note = "chroma OK"
                    sw_label_color = GREEN
                else:
                    note = f"{self._sw_isp_neutral_n}px"
                    sw_label_color = GREEN
                sw_label = (
                    f"SW-ISP[diag]  kr={kr:.3f} kb={kb:.3f}"
                    f"  exp=1(pinned)  [{note}]"
                )
            cv.putText(canvas, sw_label, (sw_x0 + 10, target_h + 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, sw_label_color, 1)

        # Mode HUD on live pane.
        mode_color = {
            "IDLE": GREEN,
            "AUTO": YELLOW,
            "SEARCH": YELLOW,
            "ERROR": RED,
        }.get(self._mode.split()[0] if self._mode else "IDLE", WHITE)
        hud_x = live_x0 + 10
        # Mode label can grow long during SEARCH because it appends a
        # human-readable metric (e.g. "SEARCH 12/82 | dE=1.87 (best
        # 1.42)"). Split it into two lines so it never overlaps the
        # right-side param HUD (wb_kelvin / contrast / saturation).
        mode_main = self._mode
        mode_metric = ""
        if self._mode == "AUTO":
            cur, total = self._compute_progress
            mode_main = f"AUTO {cur}/{total}"
        elif self._mode and self._mode.startswith("SEARCH") and "|" in self._mode:
            head, _, tail = self._mode.partition("|")
            mode_main = head.strip()
            mode_metric = tail.strip()
        cv.putText(canvas, f"[{mode_main}]", (hud_x, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        if mode_metric:
            cv.putText(canvas, mode_metric, (hud_x, 56),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # Param HUD (top-right of live). Anchor to the live pane's
        # right edge instead of a fixed offset from the mode label,
        # so a long SEARCH banner can't push into it.
        live_w = canvas.shape[1] - live_x0
        params_x = live_x0 + max(220, live_w - 180)
        for i, (label, key, _, _, _) in enumerate(_SLIDERS[:3]):
            v = self._applied.get(key, "?")
            cv.putText(canvas, f"{label}={v}",
                       (params_x, 30 + i * 24),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        bri_raw = self._applied.get("brightness")
        try:
            if bri_raw is None:
                bri_text = "brightness d=?"
            elif self._brightness_is_signed_control():
                bri_text = f"brightness d={int(bri_raw)}"
            else:
                bri_default, _bri_min, _bri_max = self._bri_default_minmax()
                bri_text = f"brightness d={int(bri_raw) - int(bri_default)}"
        except (TypeError, ValueError):
            bri_text = "brightness d=?"
        cv.putText(canvas, bri_text,
                   (params_x, 30 + 3 * 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        # Pedestal HUD shows the last *estimated* signed d, not the raw
        # brightness register offset. This avoids confusing a user-set
        # or driver-restored raw value (e.g. 0) with the estimator's
        # bounded output (auto d is clamped to [-15, 0]).
        ped_delta = self._last_pedestal_delta
        ped_used = self._last_pedestal_used
        if ped_delta is None:
            ped_text = "ped d=--"
            ped_color = GREY
        else:
            sign = "+" if ped_delta > 0 else ""
            ped_text = f"ped d={sign}{ped_delta}"
            if ped_used:
                ped_text += f" ({ped_used})"
            ped_color = WHITE if ped_delta == 0 else YELLOW
        cv.putText(canvas, ped_text,
                   (params_x, 30 + 4 * 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, ped_color, 1)

        # Bottom status bar.
        bar_y = target_h + 50
        save_marker = "* UNSAVED" if self._dirty_save else "saved"
        save_color = YELLOW if self._dirty_save else GREY
        cv.putText(canvas, save_marker, (10, bar_y + 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, save_color, 1)
        if self._colorchecker_mode:
            keys_text = (
                "[m] ColorChecker wizard  [s] Save  [r] Reset  [p] Snap  [q] Quit  "
                "| TIP: set exposure/gain/WB to taste BEFORE pressing [m]"
            )
        else:
            keys_text = (
                "[a] Auto  [m] Manual  [s] Save  [r] Reset  "
                "[p] Snap  [?] Help  [q] Quit"
            )
        cv.putText(canvas, keys_text, (170, bar_y + 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

        # Banner overlay (transient).
        if self._banner is not None and time.monotonic() < self._banner_until:
            text, color = self._banner
            (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            box_w, box_h = tw + 24, th + 18
            x0 = (canvas.shape[1] - box_w) // 2
            y0 = 8
            overlay = canvas.copy()
            cv.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                         (40, 40, 40), -1)
            cv.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
            cv.putText(canvas, text, (x0 + 12, y0 + box_h - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif self._banner is not None:
            self._banner = None

        return canvas

    @staticmethod
    def _fit(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        import cv2 as _cv2
        interp = _cv2.INTER_AREA if scale < 1.0 else _cv2.INTER_LINEAR
        return _cv2.resize(img, (new_w, target_h), interpolation=interp)

    @staticmethod
    def _placeholder(h: int, text: str) -> np.ndarray:
        img = np.full((h, h * 4 // 3, 3), 50, dtype=np.uint8)
        import cv2 as _cv2
        (tw, _), _ = _cv2.getTextSize(text, _cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        _cv2.putText(img, text, ((img.shape[1] - tw) // 2, h // 2),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
        return img

    # ------------------------------------------------------------------
    # Reference rotation helpers
    # ------------------------------------------------------------------

    def _get_rotated_ref(self) -> np.ndarray:
        """Return self._reference rotated by self._ref_rotation degrees CW."""
        import cv2 as _cv2
        rot = self._ref_rotation % 360
        if rot == 0:
            return self._reference
        if rot == 90:
            return _cv2.rotate(self._reference, _cv2.ROTATE_90_CLOCKWISE)
        if rot == 180:
            return _cv2.rotate(self._reference, _cv2.ROTATE_180)
        return _cv2.rotate(self._reference, _cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _rotate_box(
        self,
        box: tuple[int, int, int, int],
        orig_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Map a box from original-image coords to rotated-display coords."""
        x1, y1, x2, y2 = box
        H, W = orig_hw
        rot = self._ref_rotation % 360
        if rot == 0:
            return box
        if rot == 90:
            # rotated image size: (W, H) — new_h=W, new_w=H
            nx1, ny1, nx2, ny2 = H - 1 - y2, x1, H - 1 - y1, x2
            rH, rW = W, H
        elif rot == 180:
            nx1, ny1, nx2, ny2 = W - 1 - x2, H - 1 - y2, W - 1 - x1, H - 1 - y1
            rH, rW = H, W
        else:  # 270
            # rotated image size: (W, H) — new_h=W, new_w=H
            nx1, ny1, nx2, ny2 = y1, W - 1 - x2, y2, W - 1 - x1
            rH, rW = W, H
        return (
            max(0, min(rW - 1, nx1)),
            max(0, min(rH - 1, ny1)),
            max(0, min(rW - 1, nx2)),
            max(0, min(rH - 1, ny2)),
        )

    def _unrotate_box(
        self,
        box: tuple[int, int, int, int],
        orig_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Map a box from rotated-display coords back to original-image coords."""
        x1, y1, x2, y2 = box
        H, W = orig_hw
        rot = self._ref_rotation % 360
        if rot == 0:
            return box
        if rot == 90:
            # inverse of 90 CW: x_orig=y_rot, y_orig=H-1-x_rot
            nx1, ny1, nx2, ny2 = y1, H - 1 - x2, y2, H - 1 - x1
        elif rot == 180:
            nx1, ny1, nx2, ny2 = W - 1 - x2, H - 1 - y2, W - 1 - x1, H - 1 - y1
        else:  # 270
            # inverse of 270 CW: x_orig=W-1-y_rot, y_orig=x_rot
            nx1, ny1, nx2, ny2 = W - 1 - y2, x1, W - 1 - y1, x2
        return (
            max(0, min(W - 1, nx1)),
            max(0, min(H - 1, ny1)),
            max(0, min(W - 1, nx2)),
            max(0, min(H - 1, ny2)),
        )

    @staticmethod
    def _fit_for_roi(
        img: np.ndarray,
        max_h: int = 900,
        max_w: int = 1400,
    ) -> tuple[np.ndarray, float]:
        """Scale *img* to fit within max_h × max_w, preserving aspect ratio.

        Returns ``(scaled_img, scale)`` where ``scale <= 1``.  Callers
        multiply back by ``1/scale`` to recover native-pixel coordinates
        from the display-space ROI returned by ``cv.selectROI``.
        """
        import cv2 as _cv2
        h, w = img.shape[:2]
        scale = min(max_h / float(h), max_w / float(w), 1.0)
        if scale >= 1.0:
            return img, 1.0
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return _cv2.resize(img, (new_w, new_h), interpolation=_cv2.INTER_AREA), scale

    # ------------------------------------------------------------------
    # Debounced apply + sync trackbars to applied state
    # ------------------------------------------------------------------

    def _maybe_apply_pending(self) -> None:
        # Skip whenever an automatic worker (legacy AUTO pipeline or
        # color search) is driving the device. The auto thread updates
        # ``_applied`` directly, but ``_pending`` still holds whatever
        # the trackbars showed at the moment the user pressed 'a' — so
        # without this guard the GUI thread would compare the two and
        # immediately write the trackbar value back to V4L2, undoing
        # every search candidate one frame later.
        if self._mode != "IDLE":
            return
        now = time.monotonic()
        ready: dict[str, Any] = {}
        for key, val in self._pending.items():
            applied_val = self._applied.get(key)
            last = self._last_change.get(key, 0.0)
            if applied_val == val:
                continue
            if now - last < _DEBOUNCE_S:
                continue
            ready[key] = int(val)

        if not ready:
            return

        ok, err = self._apply_via_v4l2_or_ros(ready)
        if ok:
            self._applied.update(ready)
            self._dirty_save = True
        else:
            self._notify(f"set failed: {err}", RED, 4.0)

    def _flush_locked_brightness(self) -> None:
        if not self._brightness_user_locked:
            return
        pending = self._pending.get("brightness")
        if pending is None or self._applied.get("brightness") == pending:
            return
        ok, err = self._apply_via_v4l2_or_ros({"brightness": int(pending)})
        if not ok:
            self._notify(f"blacklevel apply failed: {err}", YELLOW, 4.0)
            return
        self._applied["brightness"] = int(pending)
        self._last_change.pop("brightness", None)
        self._dirty_save = True

    def _sync_trackbars_to_applied(self) -> None:
        cv = self._opencv
        # Suppress trackbar callbacks during the mass setTrackbarPos so they
        # don't round-trip into _pending and re-trigger debounced apply.
        self._suppress_cb = True
        try:
            for label, key, _lo, _hi, _default in _SLIDERS:
                v = self._applied.get(key)
                if v is None:
                    continue
                lo, hi = self._slider_range.get(key, (_lo, _hi))
                if key == "brightness":
                    signed_bri = self._brightness_param_to_signed(int(v))
                    blacklevel_value = max(0, -signed_bri)
                    brightness_value = max(0, signed_bri)
                    try:
                        cv.setTrackbarPos(
                            "blacklevel", self.WINDOW_NAME,
                            int(blacklevel_value),
                        )
                        cv.setTrackbarPos(
                            "brightness", self.WINDOW_NAME,
                            int(brightness_value),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    self._pending[key] = int(v)
                    self._last_change.pop(key, None)
                    continue
                ui_lo, ui_hi = self._slider_ui_range(key, lo, hi)
                ui_v = self._param_to_slider_value(key, int(v), lo, hi)
                ui_v = max(ui_lo, min(ui_hi, ui_v))
                try:
                    cv.setTrackbarPos(label, self.WINDOW_NAME, ui_v)
                except Exception:  # noqa: BLE001
                    pass
                self._pending[key] = int(v)
                # Clear the debounce timer so the next render won't think the
                # user just changed this slider.
                self._last_change.pop(key, None)
        finally:
            self._suppress_cb = False

    # ------------------------------------------------------------------
    # Undo stack + ROI visualisation helpers
    # ------------------------------------------------------------------

    def _push_undo(self) -> None:
        """Snapshot ``self._applied`` for single-step undo.

        Only persisted (int / bool) keys are captured so the snapshot
        is safe to re-apply via :meth:`_apply_via_v4l2_or_ros`. Stack
        is capped at :attr:`_undo_cap` (oldest entry dropped).
        """
        snap = {
            k: v for k, v in self._applied.items()
            if isinstance(v, (int, bool))
        }
        if not snap:
            return
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._undo_cap:
            self._undo_stack.pop(0)

    def _undo_last(self) -> None:
        """Pop one undo entry and write it back to the device.

        IDLE-only -- never interrupts AUTO/SEARCH/wizard. Clears its
        own snapshot from the stack (no redo) and syncs sliders to the
        restored values.
        """
        if self._mode != "IDLE":
            self._notify("Cannot undo while computing", YELLOW, 2.0)
            return
        if not self._undo_stack:
            self._notify("Nothing to undo", YELLOW, 2.0)
            return
        snap = self._undo_stack.pop()
        ok, err = self._apply_via_v4l2_or_ros(snap)
        if not ok:
            # Restore stack entry so the user can try again.
            self._undo_stack.append(snap)
            self._notify(f"Undo failed: {err}", RED, 4.0)
            return
        self._applied.update(snap)
        self._sync_trackbars_to_applied()
        self._dirty_save = True
        remaining = len(self._undo_stack)
        self._notify(
            f"Undone -- {remaining} more available"
            if remaining else "Undone -- stack empty",
            GREEN, 3.0,
        )

    def _overlay_boxes(
        self,
        img: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        *,
        labels: list[str] | None = None,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """Return *img* with translucent rectangles + numbered labels.

        Used by the manual / cc24 wizards so the user always sees the
        boxes they have already drawn while picking the next one and
        during the final review modal.
        """
        cv = self._opencv
        out = img.copy()
        if not boxes:
            return out
        overlay = out.copy()
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            col = colors[i] if (colors and i < len(colors)) else (0, 200, 255)
            cv.rectangle(
                overlay, (int(x0), int(y0)), (int(x1), int(y1)), col, -1,
            )
        cv.addWeighted(overlay, 0.20, out, 0.80, 0, out)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            col = colors[i] if (colors and i < len(colors)) else (0, 200, 255)
            lbl = labels[i] if (labels and i < len(labels)) else str(i + 1)
            cv.rectangle(
                out, (int(x0), int(y0)), (int(x1), int(y1)), col, 2,
            )
            (tw, th), _bl = cv.getTextSize(
                lbl, cv.FONT_HERSHEY_SIMPLEX, 0.55, 2,
            )
            ly = max(int(y0), th + 6)
            cv.rectangle(
                out,
                (int(x0), ly - th - 6),
                (int(x0) + tw + 8, ly + 2),
                (20, 20, 20), -1,
            )
            cv.rectangle(
                out,
                (int(x0), ly - th - 6),
                (int(x0) + tw + 8, ly + 2),
                col, 1,
            )
            cv.putText(
                out, lbl, (int(x0) + 4, ly - 2),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                cv.LINE_AA,
            )
        return out

    def _review_boxes_modal(
        self,
        snap: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        *,
        title: str = "Review boxes",
        labels: list[str] | None = None,
        colors: list[tuple[int, int, int]] | None = None,
        show_clear: bool = True,
    ) -> tuple[str, int | None]:
        """Show *snap* with all *boxes* annotated; wait for the user.

        Returns one of:
        * ``("apply", None)``  -- Enter pressed, accept the selection.
        * ``("cancel", None)`` -- Esc pressed, abort the wizard.
        * ``("clear", None)``  -- 'c' pressed (only when *show_clear*).
        * ``("redraw", i)``    -- left-click inside the i-th box.
        """
        cv = self._opencv
        annotated = self._overlay_boxes(
            snap, boxes, labels=labels, colors=colors,
        )
        # Cap displayed size so the modal fits on smaller screens.
        max_w = 1280
        scale = 1.0
        if annotated.shape[1] > max_w:
            scale = max_w / annotated.shape[1]
            new_w = int(round(annotated.shape[1] * scale))
            new_h = int(round(annotated.shape[0] * scale))
            annotated = cv.resize(annotated, (new_w, new_h), interpolation=cv.INTER_AREA)
        h_ann, w_ann = annotated.shape[:2]
        footer_h = 70
        canvas = np.full((h_ann + footer_h, w_ann, 3), 30, dtype=np.uint8)
        canvas[:h_ann, :w_ann] = annotated
        line1 = (
            "[Enter] apply   [Esc] cancel"
            + ("   [c] clear all" if show_clear else "")
        )
        line2 = "Click a numbered box to redraw it"
        cv.putText(canvas, line1, (10, h_ann + 26),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv.LINE_AA)
        cv.putText(canvas, line2, (10, h_ann + 54),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv.LINE_AA)

        click_idx: dict[str, int | None] = {"v": None}

        # Hit-test in the *displayed* (possibly scaled) coordinate space
        # so a click on a drawn rectangle hits its source-pixel box.
        scaled_boxes = [
            (
                int(round(x0 * scale)),
                int(round(y0 * scale)),
                int(round(x1 * scale)),
                int(round(y1 * scale)),
            )
            for (x0, y0, x1, y1) in boxes
        ]

        def _on_mouse(event, x, y, _flags, _userdata):
            if event != cv.EVENT_LBUTTONDOWN:
                return
            if y >= h_ann:
                return
            for idx in range(len(scaled_boxes) - 1, -1, -1):
                bx0, by0, bx1, by1 = scaled_boxes[idx]
                if bx0 <= x <= bx1 and by0 <= y <= by1:
                    click_idx["v"] = idx
                    return

        cv.imshow(title, canvas)
        try:
            cv.setMouseCallback(title, _on_mouse)
        except Exception:  # noqa: BLE001
            pass
        try:
            while True:
                k = cv.waitKey(50) & 0xFF
                if click_idx["v"] is not None:
                    return ("redraw", click_idx["v"])
                if k == 27:
                    return ("cancel", None)
                if k in (13, 10):
                    return ("apply", None)
                if show_clear and k == ord("c"):
                    return ("clear", None)
        finally:
            try:
                cv.destroyWindow(title)
            except Exception:  # noqa: BLE001
                pass

    def _compose_with_preview(
        self,
        target: np.ndarray,
        preview: np.ndarray,
        *,
        target_h: int = 540,
        gap: int = 8,
        preview_scale: float = 0.5,
    ) -> tuple[np.ndarray, int, int]:
        """Stack TARGET (left, drag here) + PREVIEW (right, read-only).

        TARGET is resized to *target_h*; PREVIEW is shown at
        ``target_h * preview_scale`` so the user can tell at a glance
        which side is editable. Returns ``(composite, preview_x0,
        target_view_w)``: ``preview_x0`` is the left edge of the
        preview region (target half ends at ``preview_x0 - gap``);
        ``target_view_w`` is the *actual* width of the resized target
        in the composite — callers must use this (not ``preview_x0``)
        when computing the back-projection scale, otherwise the gap
        leaks into the scale factor and ROIs get mapped a few pixels
        off-center.
        """
        cv = self._opencv

        def _fit_h(img: np.ndarray, h: int) -> np.ndarray:
            scale = h / float(img.shape[0])
            new_w = max(1, int(round(img.shape[1] * scale)))
            interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
            return cv.resize(img, (new_w, h), interpolation=interp)

        t_view = _fit_h(target, target_h)
        p_h = max(1, int(round(target_h * preview_scale)))
        p_view = _fit_h(preview, p_h)
        canvas_h = target_h + 30
        canvas_w = t_view.shape[1] + gap + p_view.shape[1]
        composite = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
        composite[:target_h, : t_view.shape[1]] = t_view
        preview_x0 = t_view.shape[1] + gap
        # Centre the smaller preview vertically inside the target band.
        p_y0 = (target_h - p_h) // 2
        composite[p_y0 : p_y0 + p_h, preview_x0 : preview_x0 + p_view.shape[1]] = p_view
        cv.putText(
            composite, "DRAG HERE",
            (10, target_h + 22),
            cv.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1, cv.LINE_AA,
        )
        cv.putText(
            composite, "preview (read-only)",
            (preview_x0 + 10, target_h + 22),
            cv.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv.LINE_AA,
        )
        return composite, preview_x0, int(t_view.shape[1])

    def _review_pairs_modal(
        self,
        ref: np.ndarray,
        live: np.ndarray,
        pairs: list[
            tuple[
                tuple[int, int, int, int], tuple[int, int, int, int],
            ]
        ],
        *,
        title: str = "Manual review",
        target_h: int = 540,
    ) -> tuple[str, int | None]:
        """Side-by-side review for paired (ref, live) boxes.

        Renders both panes annotated with matching numbers. Returns:
        * ``("apply", None)``  -- Enter accepted.
        * ``("cancel", None)`` -- Esc.
        * ``("clear", None)``  -- 'c'.
        * ``("redraw", i)``    -- click on the i-th numbered box on
          either pane.
        """
        cv = self._opencv
        labels = [str(i + 1) for i in range(len(pairs))]
        ref_ann = self._overlay_boxes(
            ref, [p[0] for p in pairs], labels=labels,
        )
        live_ann = self._overlay_boxes(
            live, [p[1] for p in pairs], labels=labels,
        )

        def _fit_h(img: np.ndarray) -> tuple[np.ndarray, float]:
            h, _w = img.shape[:2]
            scale = target_h / float(h)
            interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
            return cv.resize(
                img,
                (max(1, int(round(img.shape[1] * scale))), target_h),
                interpolation=interp,
            ), scale

        ref_view, ref_scale = _fit_h(ref_ann)
        live_view, live_scale = _fit_h(live_ann)
        gap = 8
        footer_h = 70
        canvas_w = ref_view.shape[1] + gap + live_view.shape[1]
        canvas = np.full(
            (target_h + footer_h, canvas_w, 3), 30, dtype=np.uint8,
        )
        canvas[:target_h, : ref_view.shape[1]] = ref_view
        live_x0 = ref_view.shape[1] + gap
        canvas[:target_h, live_x0 : live_x0 + live_view.shape[1]] = live_view
        cv.putText(canvas, "REFERENCE", (10, target_h + 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv.LINE_AA)
        cv.putText(canvas, "LIVE", (live_x0 + 10, target_h + 22),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv.LINE_AA)
        cv.putText(
            canvas,
            "[Enter] apply   [Esc] cancel   [c] clear all   "
            "click a numbered box to redraw that pair",
            (10, target_h + 54),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv.LINE_AA,
        )

        # Pre-scale boxes into displayed coordinate spaces so click
        # hit-testing is exact even when the panes were resized.
        ref_scaled = [
            (
                int(round(p[0][0] * ref_scale)),
                int(round(p[0][1] * ref_scale)),
                int(round(p[0][2] * ref_scale)),
                int(round(p[0][3] * ref_scale)),
            )
            for p in pairs
        ]
        live_scaled = [
            (
                int(round(p[1][0] * live_scale)),
                int(round(p[1][1] * live_scale)),
                int(round(p[1][2] * live_scale)),
                int(round(p[1][3] * live_scale)),
            )
            for p in pairs
        ]

        click_idx: dict[str, int | None] = {"v": None}

        def _hit(boxes_local, x_local, y_local) -> int | None:
            # Last-drawn-wins: iterate from the end so the most recent
            # rectangle takes precedence on overlap.
            for i in range(len(boxes_local) - 1, -1, -1):
                bx0, by0, bx1, by1 = boxes_local[i]
                if bx0 <= x_local <= bx1 and by0 <= y_local <= by1:
                    return i
            return None

        def _on_mouse(event, x, y, _flags, _userdata):
            if event != cv.EVENT_LBUTTONDOWN:
                return
            if y >= target_h:
                return
            if x < ref_view.shape[1]:
                hit = _hit(ref_scaled, x, y)
            elif x >= live_x0:
                hit = _hit(live_scaled, x - live_x0, y)
            else:
                return
            if hit is not None:
                click_idx["v"] = hit

        cv.imshow(title, canvas)
        try:
            cv.setMouseCallback(title, _on_mouse)
        except Exception:  # noqa: BLE001
            pass
        try:
            while True:
                k = cv.waitKey(50) & 0xFF
                if click_idx["v"] is not None:
                    return ("redraw", click_idx["v"])
                if k == 27:
                    return ("cancel", None)
                if k in (13, 10):
                    return ("apply", None)
                if k == ord("c"):
                    return ("clear", None)
        finally:
            try:
                cv.destroyWindow(title)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Auto mode (background thread)
    # ------------------------------------------------------------------

    def _trigger_auto(self) -> None:
        if self._mode == "AUTO":
            self._notify("Auto already running", YELLOW, 2.0)
            return
        # Snapshot current device state so the user can `u` back to it
        # if AUTO + chained color search drift somewhere they don't like.
        self._push_undo()
        self._mode = "AUTO"
        self._compute_progress = (0, _SOLVER_RETRIES)
        thread = threading.Thread(target=self._run_auto, daemon=True)
        self._auto_thread = thread
        thread.start()

    def _update_sw_isp_from_dbg(self, dbg: Mapping[str, Any] | None) -> None:
        """Refresh the SW-ISP debug-pane params from a solver debug dict.

        Pulls ``kr_sw`` / ``kb_sw`` / ``exp_scale_sw`` (linear-domain
        channel gains the SW-ISP applies pixel-wise). Missing or
        non-finite values default to 1.0 (identity).
        """
        if not isinstance(dbg, dict):
            return

        def _safe(name: str) -> float:
            v = dbg.get(name, 1.0)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return 1.0
            if not np.isfinite(fv) or fv <= 0.0:
                return 1.0
            return fv

        kr = _safe("kr_sw")
        kb = _safe("kb_sw")
        exp = _safe("exp_scale_sw")
        n_neutral = int(dbg.get("_sw_neutral_n", 0))
        ref_sat = bool(dbg.get("_sw_ref_saturated", False))

        # CCM (3x3) — populated by manual_match_neutral / manual_match_patches
        # when ≥ 3 chromatic-distinct pairs are available. None means fall
        # back to diag-only rendering.
        ccm = dbg.get("ccm_sw", None)
        ccm_pairs = int(dbg.get("_sw_ccm_pairs", 0))
        if ccm is not None:
            try:
                ccm_arr = np.asarray(ccm, dtype=np.float64)
                if ccm_arr.shape != (3, 3) or not np.all(np.isfinite(ccm_arr)):
                    ccm_arr = None
            except (TypeError, ValueError):
                ccm_arr = None
        else:
            ccm_arr = None

        self._sw_isp = SwIspParams(kr=kr, kb=kb, exp_scale=exp, ccm=ccm_arr)
        self._sw_isp_computed = True
        self._sw_isp_neutral_n = n_neutral
        self._sw_isp_ref_saturated = ref_sat
        self._sw_isp_ccm_pairs = ccm_pairs
        # [SW-ISP] solver dbg — suppressed (CCM no longer visualized)
        # mode_tag = "CCM" if ccm_arr is not None else "diag"
        # print(f"[SW-ISP] solver dbg → mode={mode_tag} ...")

        # Cache all four CCM variants and re-apply whichever the user has
        # currently selected (key '0' / '1' / '2' / '3'). On first call this
        # is "linear" so behaviour is bit-identical to pre-RPCC.
        variants = dbg.get("ccm_variants")
        if isinstance(variants, dict):
            self._ccm_variants = {
                name: variants.get(name) for name in
                ("linear", "rpcc2", "rpcc2_ridge", "rpcc2_als")
            }
        else:
            self._ccm_variants = {
                "linear": None, "rpcc2": None,
                "rpcc2_ridge": None, "rpcc2_als": None,
            }
        # Re-apply current selection (no notify; auto fallback chain).
        self._apply_ccm_variant(self._ccm_variant, notify=False)

    # ------------------------------------------------------------------
    # CCM variant switcher (keys '0' / '1' / '2' / '3')
    # ------------------------------------------------------------------

    _VARIANT_BY_KEY: dict[str, str] = {
        "0": "linear",
        "1": "rpcc2",
        "2": "rpcc2_ridge",
        "3": "rpcc2_als",
    }

    def _apply_ccm_variant(self, name: str, *, notify: bool = True) -> None:
        """Select a CCM variant and push its M into ``self._sw_isp.ccm``.

        Falls back to ``"linear"`` (then to identity) if the requested
        variant is not currently solvable. ``notify=True`` shows a
        banner; ``False`` is used during silent re-application after
        ``_update_sw_isp_from_dbg``.
        """
        chain = [name, "linear"]  # always try linear as a fallback
        chosen: str | None = None
        entry: dict | None = None
        for candidate in chain:
            ent = self._ccm_variants.get(candidate)
            if ent is not None and isinstance(ent.get("M"), np.ndarray):
                chosen = candidate
                entry = ent
                break

        if chosen is None or entry is None:
            # No variant is solvable yet — keep current state, just
            # remember the selection for the next solve.
            self._ccm_variant = name
            if notify:
                self._notify(
                    f"variant '{name}' unavailable (run [m] or [a] first)",
                    YELLOW, 2.5,
                )
            return

        self._ccm_variant = chosen
        M = np.asarray(entry["M"], dtype=np.float64)
        self._sw_isp = SwIspParams(
            kr=self._sw_isp.kr,
            kb=self._sw_isp.kb,
            exp_scale=self._sw_isp.exp_scale,
            ccm=M,
        )
        self._sw_isp_ccm_pairs = int(entry.get("n_pairs", 0))

        if notify:
            de = entry.get("delta_e_median")
            # ASCII only — the banner is rendered through cv2.putText.
            de_str = f"  dE_med={de:.2f}" if de is not None else ""
            lam = entry.get("lambda")
            lam_str = f"  lam={lam:.1e}" if lam is not None else ""
            it = entry.get("iters")
            it_str = f"  iters={it}" if it is not None else ""
            note = (
                f"CCM variant -> {chosen}  feat={entry['feat_dim']}  "
                f"pairs={entry['n_pairs']}{lam_str}{it_str}{de_str}"
            )
            if chosen != name:
                note = f"'{name}' unavailable, fell back to '{chosen}'"
                self._notify(note, YELLOW, 2.5)
            else:
                self._notify(note, GREEN, 2.5)
        # [SW-ISP] applied variant — suppressed
        # print(f"[SW-ISP] applied variant={chosen}  M.shape={M.shape}")

    def _toggle_sw_only(self) -> None:
        self._sw_only = not self._sw_only
        if self._sw_only:
            self._notify(
                "SW-only mode ON  (m/a will not write hardware; "
                "third pane shows the digital answer)",
                YELLOW, 4.0,
            )
        else:
            self._notify("SW-only mode OFF  (hardware writes resumed)",
                         GREEN, 3.0)

    def _zero_brightness_for_calibration(self, *, force: bool = False) -> None:
        """Write brightness=default at the very start of any calibration run.

        Ensures the pipeline, color search, and solver operate from a
        neutral baseline unless the user has explicitly tuned blacklevel.
        No-op if brightness is already at the device default or if the
        instance is not fully initialised (e.g. in unit tests).
        """
        if not hasattr(self, "_applied"):
            return
        if self._brightness_user_locked and not force:
            return
        bri_default, _, _ = self._bri_default_minmax()
        cur_bri = int(self._applied.get("brightness", bri_default))
        if cur_bri == bri_default:
            return
        print(
            f"[Pedestal] pre-calibration: zeroing brightness "
            f"{cur_bri} -> {bri_default}", flush=True,
        )
        ok, _err = self._apply_via_v4l2_or_ros({"brightness": bri_default})
        if ok:
            self._applied["brightness"] = bri_default
            self._sync_trackbars_to_applied()
        time.sleep(0.25)  # allow camera to settle

    def _run_auto(self) -> None:
        """4-stage hardware ISP calibration (plan v2 §2).

        Stages:
          1. Exposure  — drive Y_mean to mid-gray (128) with overexp guard.
          2. Gain      — half-step bump; SNR fallback to brightness offset.
          3. Sat / Con — ratio-match chroma + luma std vs. reference.
          4. Kelvin    — chromaticity → McCamy → delta-CCT (no CCM here).

        SW-only mode bypasses hardware writes (third pane preview only)
        and falls back to the legacy single-shot solver.
        """
        # ColorChecker24 mode disables AUTO entirely (the "reference" is
        # 24 known sRGB triples — there is no full-frame image to LAB-
        # match against). Stage 4 is delegated to the manual 24-patch
        # wizard ('m' key).
        if self._colorchecker_mode:
            self._notify(
                "AUTO disabled in colorchecker mode. Press 'm' to run "
                "the 24-patch wizard.",
                YELLOW, 4.0,
            )
            self._mode = "IDLE"
            return
        # SW-only path: keep using the legacy solver because the new
        # 4-stage orchestrator only writes hardware. We surface the SW
        # answer (kr/kb/exp_scale + CCM) without moving the camera.
        if self._sw_only:
            self._run_auto_legacy()
            return
        # Guard: if we know the driver is not honouring manual WB, refuse
        # to run — otherwise the solver pushes wb to a slider rail because
        # the camera never reacts.
        if not getattr(self, "_wb_writable", True):
            self._notify(
                "Auto disabled: camera driver rejects manual WB. "
                "Check 'v4l2-ctl --list-ctrls' control names.",
                RED, 6.0,
            )
            self._mode = "IDLE"
            return
        self._flush_locked_brightness()
        self._zero_brightness_for_calibration()
        saved_protect = self._protect_brightness_in_auto_pipeline
        try:
            self._protect_brightness_in_auto_pipeline = self._brightness_user_locked
            self._run_auto_pipeline()
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Auto crashed: {exc}", RED, 6.0)
            return
        finally:
            self._protect_brightness_in_auto_pipeline = saved_protect
            self._mode = "IDLE"
            self._compute_progress = (0, 0)

        # Chain color search after auto succeeds (REF mode only). The
        # 4-stage pipeline gives a good initial estimate; the unified
        # K/C/Sat search refines K, contrast and saturation jointly
        # against the reference. Skip in SW-only mode (no hardware
        # writes) and ColorChecker24 mode (uses dedicated wizard).
        if self._sw_only or self._colorchecker_mode:
            return
        # Pedestal SSOT: estimate signed Δ against the reference and
        # write it to ``brightness`` BEFORE color search so K/C/Sat
        # converge on the corrected dark level (otherwise the search
        # spends evals fighting a constant offset).
        try:
            if self._brightness_user_locked:
                self._notify(
                    "Auto: keeping user blacklevel; pedestal skipped",
                    GREEN, 4.0,
                )
            else:
                self._run_pedestal_stage(mode="ref")
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Auto: pedestal failed: {exc}", YELLOW, 4.0)
        try:
            self._run_color_search()
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Auto: color search failed: {exc}", YELLOW, 5.0)

    def _run_auto_pipeline(self) -> None:
        """Drive :func:`run_full_pipeline` against the live camera."""
        from dataset_tools.camera_isp.exposure_units import (
            DEFAULT_MAX_EXPOSURE_MS,
            compute_exposure_max_us,
            probe_exposure,
            ticks_from_us,
        )
        from dataset_tools.camera_isp.hw_pipeline import run_full_pipeline
        from dataset_tools.camera_isp.hw_stages import (
            compute_chroma_stats,
            compute_y_stats,
            extract_roi_frame,
        )

        live, stamp = self._bridge.latest_frame()
        if live is None or time.monotonic() - stamp > 1.0:
            self._notify("No live frame -- is the camera streaming?", RED, 4.0)
            return
        ref_resized = self._resize_ref_to(live.shape)

        # ROI plumbing — when the user has dragged paired boxes via 'm',
        # every stage runs on the union of those boxes (NOT the full
        # frame). Box coordinates were captured against the original
        # reference image and a snapshot of the live frame; live boxes
        # are still valid as long as the camera/scene haven't moved
        # since selection. Ref boxes get scaled to ref_resized's pixel
        # grid; live boxes are forwarded verbatim because they were
        # already drawn against a frame of the same shape as the live
        # stream.
        ref_boxes_native = [pair[0] for pair in self._roi_pairs]
        live_boxes = [pair[1] for pair in self._roi_pairs]
        if ref_boxes_native and ref_resized.shape != self._reference.shape:
            sx = ref_resized.shape[1] / float(self._reference.shape[1])
            sy = ref_resized.shape[0] / float(self._reference.shape[0])
            ref_boxes = [
                (int(b[0] * sx), int(b[1] * sy),
                 int(b[2] * sx), int(b[3] * sy))
                for b in ref_boxes_native
            ]
        else:
            ref_boxes = list(ref_boxes_native)

        # Stats sources — when ROIs are active the prelude stats AND
        # everything inside ``run_full_pipeline`` operate on the ROI
        # strip. Otherwise we fall back to the legacy full-frame view.
        if ref_boxes and live_boxes:
            ref_for_stats = extract_roi_frame(ref_resized, ref_boxes)
            live_for_stats = extract_roi_frame(live, live_boxes)
            roi_active = True
        else:
            ref_for_stats = ref_resized
            live_for_stats = live
            roi_active = False

        # Stage 1 needs an upper bound on exposure that respects both the
        # device caps AND the active streaming fps so we don't propose a
        # value that physically can't fit in one frame.
        device = self._bridge.video_device
        probe = probe_exposure(device) if device else None
        fps = probe.fps if (probe and probe.fps) else None
        unit_us = probe.unit_us if probe else None
        cli_max_ms = float(getattr(self, "_max_exposure_ms", None)
                           or DEFAULT_MAX_EXPOSURE_MS)
        exp_max_us = compute_exposure_max_us(cli_max_ms, fps)
        if unit_us:
            exp_max_ticks = ticks_from_us(exp_max_us, unit_us)
        else:
            # Unknown unit ⇒ trust the device cap; better than guessing.
            exp_max_ticks = self._device_caps.get("exposure", {}).get(
                "max", 1000,
            )
        device_max = self._device_caps.get("exposure", {}).get("max")
        if device_max is not None:
            exp_max_ticks = min(exp_max_ticks, device_max)

        # Reference-frame statistics drive the Stage 3 ratio match.
        y_stats_ref = compute_y_stats(ref_for_stats)
        chroma_ref = compute_chroma_stats(ref_for_stats)
        y_stats_live = compute_y_stats(live_for_stats)

        # Per-stage diagnostic prelude — printing the live-vs-ref deltas
        # before any write makes it cheap to spot why the pipeline picked
        # a particular branch (e.g. overexposed vs. underexposed).
        print("[Auto] === run_full_pipeline starting ===")
        print(f"[Auto] device={device} fps={fps} unit_us={unit_us} "
              f"cli_max_ms={cli_max_ms:.1f} exp_max_ticks={int(exp_max_ticks)}")
        if roi_active:
            print(f"[Auto] ROI active: {len(self._roi_pairs)} pair(s); "
                  f"all stages computed on ROI pixels only")
        print(f"[Auto] live  Y_mean={y_stats_live.y_mean_excl_clip:.1f} "
              f"Y_p99={y_stats_live.y_p99:.1f} clip={y_stats_live.clip_ratio:.3f}")
        print(f"[Auto] ref   Y_mean={y_stats_ref.y_mean_excl_clip:.1f} "
              f"Y_p99={y_stats_ref.y_p99:.1f} clip={y_stats_ref.clip_ratio:.3f} "
              f"chroma_mag={chroma_ref.chroma_mag_median:.2f}")
        print(f"[Auto] start params={dict(self._applied)}")

        bridge = _CalibratorStageBridge(self)
        # Tell the bridge to ROI-extract every grab_frame — Stage 1/2
        # (run_stage_exposure / run_stage_gain) and Stage 4 (kelvin)
        # all consume bridge.grab_frame(); routing the ROI through the
        # bridge keeps the pipeline itself agnostic.
        if roi_active:
            bridge._live_roi_boxes = list(live_boxes)
        self._compute_progress = (1, 4)
        # Drive Stage 1/2 toward the *reference* Y-mean so the live
        # frame matches the reference brightness (not a fixed mid-gray
        # 128 cd/m²). Clamp to [40,180] to defend against an obviously
        # mis-exposed reference: outside that band we fall back to 128
        # rather than chase a broken target.
        ref_y_mean = float(y_stats_ref.y_mean_excl_clip)
        if 40.0 <= ref_y_mean <= 180.0:
            target_y_mean = ref_y_mean
        else:
            target_y_mean = 128.0
        print(f"[Auto] target_y_mean={target_y_mean:.1f} "
              f"(ref={ref_y_mean:.1f}, "
              f"{'from-ref' if target_y_mean == ref_y_mean else 'fallback-128'})")
        result = run_full_pipeline(
            bridge,
            mode="ref",
            target_y_mean=target_y_mean,
            exp_max=int(exp_max_ticks),
            chroma_mag_ref=chroma_ref.chroma_mag_median,
            y_std_ref=y_stats_ref.y_std_excl_clip,
            ref_bgr=ref_for_stats,
            device_caps=self._device_caps or None,
        )
        self._dirty_save = True

        # Surface a concise summary banner.
        passed = sum(1 for s in result.stages if s.passed)
        params = result.final_params
        msg = (
            f"[{passed}/4 stages] exp={params.get('exposure', '?')} "
            f"gain={params.get('gain', '?')} bri={params.get('brightness', '?')} "
            f"sat={params.get('saturation', '?')} con={params.get('contrast', '?')} "
            f"K={params.get('white_balance', '?')}"
        )
        self._notify(msg, GREEN if result.all_passed else YELLOW, 6.0)
        # Console echo: per-stage rationale + last_stats for diagnosis.
        for s in result.stages:
            stats_repr = ""
            if s.last_stats is not None:
                stats_repr = f" stats={s.last_stats}"
            print(f"[Auto] {s.name}: iters={s.iters} converged={s.converged} "
                  f"note={s.note}{stats_repr}")
        print(f"[Auto] final params={params}")
        print("[Auto] === run_full_pipeline done ===")

    def _run_auto_legacy(self) -> None:
        """Legacy single-shot ``auto_match_lab`` path (SW-only fallback)."""
        rail_streak = 0  # consecutive iterations where wb hits a rail
        try:
            for i in range(_SOLVER_RETRIES):
                self._compute_progress = (i + 1, _SOLVER_RETRIES)
                live, stamp = self._bridge.latest_frame()
                if live is None or time.monotonic() - stamp > 1.0:
                    self._notify("No live frame -- is the camera streaming?", RED, 4.0)
                    return
                ref_resized = self._resize_ref_to(live.shape)
                caps = getattr(self, "_device_caps", None) or None
                step = auto_match_lab(ref_resized, live, self._applied, caps)
                self._update_sw_isp_from_dbg(step.debug)
                if not step.proposed:
                    if step.debug.get("_warning"):
                        self._notify(
                            f"Auto: {step.debug['_warning']}", YELLOW, 4.0,
                        )
                    return
                if self._sw_only:
                    sw = self._sw_isp
                    self._notify(
                        f"SW-only auto: kr={sw.kr:.3f} kb={sw.kb:.3f} "
                        f"exp={sw.exp_scale:.3f}  (hardware unchanged)",
                        GREEN, 5.0,
                    )
                    return
                wb_lo, wb_hi = self._slider_range.get("white_balance", (2000, 8000))
                wb_proposed = int(step.proposed.get("white_balance", -1))
                if wb_proposed in (wb_lo, wb_hi):
                    rail_streak += 1
                else:
                    rail_streak = 0
                if rail_streak >= 2:
                    self._notify(
                        "Auto stopped: white-balance keeps hitting slider rail. "
                        "Driver may be ignoring manual WB.",
                        RED, 6.0,
                    )
                    return
                ok, err = self._apply_via_v4l2_or_ros(step.proposed)
                if not ok:
                    self._notify(f"Auto apply failed: {err}", RED, 5.0)
                    return
                self._applied.update(step.proposed)
                self._dirty_save = True
                if step.converged:
                    self._notify(f"Auto converged in {i + 1} iter(s)", GREEN, 4.0)
                    return
                time.sleep(0.25)
            self._notify(
                f"Auto reached max {_SOLVER_RETRIES} iters (still improving)",
                YELLOW, 4.0,
            )
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Auto crashed: {exc}", RED, 6.0)
        finally:
            self._mode = "IDLE"
            self._compute_progress = (0, 0)

    def _resize_ref_to(self, shape: tuple[int, ...]) -> np.ndarray:
        """Resize reference image to match (h, w, 3) of the live frame."""
        h, w = shape[:2]
        if self._reference.shape[:2] == (h, w):
            return self._reference
        import cv2 as _cv2
        return _cv2.resize(self._reference, (w, h))

    # ------------------------------------------------------------------
    # Save / reset
    # ------------------------------------------------------------------

    def _save_override(self) -> None:
        path = _override_path(self._bridge.camera_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save the full reproducible ISP state, including usb_cam defaults
        # after sentinel normalization and the manual auto_* flags engaged
        # by _force_manual_modes().
        payload: dict[str, Any] = {}
        for k in _ALL_KEYS:
            if k not in self._applied:
                continue
            value = self._normalize_isp_value(k, self._applied[k])
            if value is not None:
                payload[k] = value
        payload["_camera"] = self._bridge.camera_name
        payload["_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            self._notify(f"Save failed: {exc}", RED, 5.0)
            return
        self._dirty_save = False
        self._notify(f"Saved -> {path}", GREEN, 5.0)
        # Also print to stdout so headless invocations have a record.
        print(f"[camera_isp_calibrator] saved override to {path}")

    def _reset(self) -> None:
        if not self._initial:
            self._notify("No initial snapshot to restore", YELLOW, 3.0)
            return
        # Snapshot pre-reset state so 'u' can roll back to it.
        self._push_undo()
        # Re-assert manual modes at the driver level first; otherwise the
        # subsequent white_balance write will be denied by V4L2.
        self._force_manual_modes()
        # Only restore the calibration targets (exposure + white_balance).
        # Aesthetic params (brightness/contrast/saturation/sharpness/gain)
        # are user-tuned via sliders and are *not* the calibrator's job;
        # forcing them back to whatever usb_cam booted with (often 0)
        # turns the live image into a flat grey wash, which is the
        # opposite of helpful.
        targets = {
            k: int(self._initial[k])
            for k in ("exposure", "white_balance")
            if k in self._initial
        }
        if not targets:
            self._notify("Nothing to reset", YELLOW, 3.0)
            return
        ok, err = self._apply_via_v4l2_or_ros(targets)
        if not ok:
            self._notify(f"Reset failed: {err}", RED, 5.0)
            return
        self._applied.update(targets)
        self._sync_trackbars_to_applied()
        self._dirty_save = False
        summary = ", ".join(f"{k}={v}" for k, v in targets.items())
        self._notify(f"Reset (calibration only): {summary}", GREEN, 3.0)

    def _direct_write_wb_exposure(self) -> None:
        """Push every slider key through v4l2-ctl directly (used by Reset)."""
        from dataset_tools.camera_isp import v4l2_ctl as _v4l2
        device = self._bridge.video_device
        resolved = self._v4l2_resolved
        if not device or not resolved or not _v4l2.have_v4l2_ctl():
            return
        payload = {
            key: int(self._applied[key])
            for _, key, _, _, _ in _SLIDERS
            if isinstance(self._applied.get(key), int)
        }
        _v4l2.apply_params(device, resolved, payload)

    def _apply_via_v4l2_or_ros(self, params: dict) -> tuple[bool, str | None]:
        """Apply *params* preferring v4l2-ctl, falling back to ROS.

        Direct v4l2-ctl avoids the usb_cam parameter-callback cascade that
        otherwise re-writes every control (and trips 'unknown control' on
        renamed UVCs), making the ROS service call slow and unreliable.
        """
        if not params:
            return True, None
        from dataset_tools.camera_isp import v4l2_ctl as _v4l2
        device = self._bridge.video_device
        resolved = self._v4l2_resolved
        # Split: bool keys (auto_white_balance / autoexposure) -> ROS only.
        # Numeric V4L2-mapped keys -> v4l2-ctl direct (skips ROS entirely).
        v4l2_payload = {
            k: v for k, v in params.items()
            if not isinstance(v, bool)
        }
        ros_payload = {
            k: v for k, v in params.items()
            if isinstance(v, bool)
        }
        v4l2_ok = True
        v4l2_msg = ""
        if device and resolved and _v4l2.have_v4l2_ctl() and v4l2_payload:
            v4l2_ok, v4l2_msg, handled = _v4l2.apply_params(
                device, resolved, v4l2_payload
            )
            # [WriteTrace] commented out — fires on every V4L2 write, too noisy.
            # tag = "OK" if v4l2_ok else f"FAIL:{v4l2_msg}"
            # print(f"[WriteTrace] v4l2 mode={self._mode} payload={v4l2_payload} {tag}",
            #       flush=True)
            if v4l2_ok:
                # Drop only the keys the V4L2 layer actually wrote; keys it
                # didn't recognise (e.g. ``focus``) must still fall through
                # to ROS instead of being silently dropped.
                v4l2_payload = {
                    k: v for k, v in v4l2_payload.items() if k not in handled
                }
        # Anything still in v4l2_payload (failure or v4l2-ctl unavailable) +
        # the bool keys go through ROS as fallback.
        ros_payload.update(v4l2_payload)
        if not ros_payload:
            return v4l2_ok, (None if v4l2_ok else v4l2_msg)
        ok, err = self._bridge.set_params(ros_payload)
        if not ok and not v4l2_ok:
            return False, f"v4l2: {v4l2_msg}; ros: {err}"
        return ok, err

    def _snapshot_screenshot(self, frame: np.ndarray | None) -> None:
        if frame is None:
            self._notify("No frame to snapshot", YELLOW, 2.0)
            return
        out = Path.cwd() / time.strftime("isp_capture_%Y%m%d_%H%M%S.png")
        try:
            self._opencv.imwrite(str(out), frame)
            self._notify(f"Saved {out.name}", GREEN, 3.0)
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Snapshot failed: {exc}", RED, 4.0)

    # ------------------------------------------------------------------
    # ColorChecker24 wizard
    # ------------------------------------------------------------------

    def _run_color_checker(self) -> None:
        """24-patch wizard: drag a LIVE box for each chart patch in turn.

        For each of the 24 X-Rite ColorChecker patches, we show a small
        prompt window highlighting the patch on the chart thumbnail and
        ask the user to draw a box on the LIVE pane covering the
        corresponding patch on the physical chart in front of the camera.
        Patches the user does not want to use can be skipped with 's'.
        At the end we feed all collected (patch_index, live_box) tuples
        into :func:`manual_match_colorchecker` and update the SW-ISP
        debug pane (no hardware writes — cc24 mode is SW-only).
        """
        cv = self._opencv
        from dataset_tools.camera_isp.colorchecker24 import (
            COLORCHECKER24_NAMES,
            COLORCHECKER24_SRGB,
            make_checker_thumbnail,
        )

        live, stamp = self._bridge.latest_frame()
        if live is None or (stamp and time.monotonic() - stamp > 1.5):
            self._notify("No fresh live frame for colorchecker wizard",
                         RED, 4.0)
            return
        live_snap = live.copy()

        self._notify(
            "ColorChecker: for each highlighted patch, drag a box on LIVE."
            "  Enter=accept  s=skip  Esc=finish early",
            WHITE, 6.0,
        )

        prompt_win = "ColorChecker patch"
        live_win = "Drag LIVE box for highlighted patch (Enter=ok, c=cancel)"
        # Sparse mapping patch_idx (0..23) -> live box. Re-pick simply
        # overwrites the entry. Order is preserved by sorting the keys.
        boxes_by_patch: dict[int, tuple[int, int, int, int]] = {}

        def _patch_color(idx: int) -> tuple[int, int, int]:
            rgb = COLORCHECKER24_SRGB[idx]
            return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

        def _annotated_live(skip_patch: int | None = None) -> np.ndarray:
            ordered = sorted(
                k for k in boxes_by_patch.keys() if k != skip_patch
            )
            return self._overlay_boxes(
                live_snap,
                [boxes_by_patch[k] for k in ordered],
                labels=[str(k + 1) for k in ordered],
                colors=[_patch_color(k) for k in ordered],
            )

        def _show_patch_prompt(patch_idx: int) -> int:
            """Show chart prompt for *patch_idx*; return waitKey result."""
            thumb = make_checker_thumbnail(
                360, 540, with_index=True, highlight=patch_idx + 1,
            )
            top_h, bot_h = 34, 30
            canvas = np.full(
                (thumb.shape[0] + top_h + bot_h, thumb.shape[1], 3),
                30, dtype=np.uint8,
            )
            canvas[top_h : top_h + thumb.shape[0], : thumb.shape[1]] = thumb
            cv.putText(
                canvas,
                f"Patch {patch_idx + 1}/24: "
                f"{COLORCHECKER24_NAMES[patch_idx]}",
                (10, top_h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1,
                cv.LINE_AA,
            )
            cv.putText(
                canvas,
                "[Enter] draw box on LIVE   [s] skip   [Esc] finish",
                (10, canvas.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv.LINE_AA,
            )
            cv.imshow(prompt_win, canvas)
            k = 0xFF
            while k == 0xFF:
                k = cv.waitKey(50) & 0xFF
                if k in (13, 10, ord("s"), 27):
                    break
                k = 0xFF
            try:
                cv.destroyWindow(prompt_win)
            except Exception:  # noqa: BLE001
                pass
            return k

        def _pick_one(patch_idx: int) -> str:
            """Run prompt + ROI for one patch.

            Returns ``"add"`` / ``"skip"`` / ``"finish"``.
            """
            k = _show_patch_prompt(patch_idx)
            if k == 27:
                return "finish"
            if k == ord("s"):
                return "skip"
            l = cv.selectROI(
                live_win, _annotated_live(skip_patch=patch_idx),
                showCrosshair=True, fromCenter=False,
            )
            try:
                cv.destroyWindow(live_win)
            except Exception:  # noqa: BLE001
                pass
            if l is None or l[2] < 4 or l[3] < 4:
                return "skip"
            lx, ly, lw, lh = (
                int(l[0]), int(l[1]), int(l[2]), int(l[3]),
            )
            boxes_by_patch[patch_idx] = (lx, ly, lx + lw, ly + lh)
            return "add"

        # Phase 1: linear pick of all 24.
        for i in range(24):
            if _pick_one(i) == "finish":
                break

        # Phase 2: review / re-edit loop.
        while True:
            if not boxes_by_patch:
                self._notify("ColorChecker: no patches picked", YELLOW, 3.0)
                return
            ordered = sorted(boxes_by_patch.keys())
            review_boxes = [boxes_by_patch[k] for k in ordered]
            labels = [str(k + 1) for k in ordered]
            colors = [_patch_color(k) for k in ordered]
            action, which = self._review_boxes_modal(
                live_snap, review_boxes,
                title="ColorChecker review",
                labels=labels, colors=colors,
                show_clear=True,
            )
            if action == "cancel":
                self._notify("ColorChecker cancelled", YELLOW, 3.0)
                return
            if action == "clear":
                boxes_by_patch.clear()
                for i in range(24):
                    if _pick_one(i) == "finish":
                        break
                continue
            if action == "redraw":
                # ``which`` is an index into ``ordered`` (the displayed
                # list). Map back to the patch slot and re-pick.
                patch_idx = ordered[int(which)]
                _pick_one(patch_idx)
                continue
            if action == "apply":
                break

        ref_indices = sorted(boxes_by_patch.keys())
        boxes = [boxes_by_patch[k] for k in ref_indices]

        if len(boxes) < 3:
            self._notify(
                f"ColorChecker: only {len(boxes)} boxes -- need >=3 for "
                "linear, >=6 (12 for plain RPCC2) for full 4-variant set.",
                YELLOW, 5.0,
            )
            if not boxes:
                return

        from dataset_tools.camera_isp.solver import (
            manual_match_colorchecker,
        )
        ref_subset = COLORCHECKER24_SRGB[np.asarray(ref_indices, dtype=np.int64)]
        try:
            _proposed, dbg = manual_match_colorchecker(
                ref_subset, live_snap, boxes,
                self._applied, self._device_caps,
            )
        except Exception as exc:  # noqa: BLE001
            self._notify(f"ColorChecker solver failed: {exc}", RED, 5.0)
            return

        self._update_sw_isp_from_dbg(dbg)
        sw = self._sw_isp
        self._notify(
            f"cc24 ({len(boxes)} patches) applied to SW-ISP: kr={sw.kr:.3f} "
            f"kb={sw.kb:.3f}  variant={self._ccm_variant}",
            GREEN, 5.0,
        )

        # Seed the hardware (exposure / gain / WB / contrast / sat)
        # from the cc24 truth before launching the K/C/Sat search.
        # Without this step the search starts from whatever values
        # happen to be in self._applied (often device defaults that
        # have nothing to do with the actual scene), which makes the
        # grid search waste evals climbing back to a sane region.
        # Snapshot pre-cc24 device state so 'u' can roll back the whole
        # cc24 apply (HW seed + chained color search) in one step.
        self._push_undo()
        try:
            self._run_cc24_hw_seed(list(boxes), list(ref_indices))
        except Exception as exc:  # noqa: BLE001
            self._notify(f"cc24: HW seed failed: {exc}", YELLOW, 4.0)

        # Chain HW K/C/Sat search using cc24 boxes + X-Rite truth so
        # the camera itself converges (not just the SW-ISP CCM). This
        # mirrors the manual REF-pair workflow where _run_manual ends
        # with _run_color_search; in cc24 mode we go through the same
        # search infrastructure but with cost_24card.
        try:
            # Pedestal SSOT: cc24 has no full reference image to
            # compare against, so use the auto (live-only) estimator.
            # Δ stays clamped ≤ 0 — cc24 only ever subtracts pedestal.
            if self._brightness_user_locked:
                self._notify(
                    "cc24: keeping user blacklevel; auto pedestal skipped",
                    GREEN, 4.0,
                )
            else:
                self._run_pedestal_stage(mode="auto")
        except Exception as exc:  # noqa: BLE001
            self._notify(f"cc24: pedestal failed: {exc}", YELLOW, 4.0)
        try:
            self._cc24_search = {
                "boxes": list(boxes),
                "ref_indices": list(ref_indices),
            }
            self._run_color_search()
        except Exception as exc:  # noqa: BLE001
            self._notify(f"cc24: color search failed: {exc}", YELLOW, 5.0)

    def _run_cc24_hw_seed(
        self,
        boxes: list[tuple[int, int, int, int]],
        ref_indices: list[int],
    ) -> None:
        """Seed HW params from cc24 truth via the AUTO pipeline.

        cc24 mode has no real reference image, so the standard AUTO
        pipeline would target the checker thumbnail's brightness (a
        synthetic gradient) instead of what the scene should look like.

        We synthesize a reference by painting each user-selected
        patch's X-Rite truth BGR into a copy of the live frame, then
        run :meth:`_run_auto_pipeline` with that synthetic reference
        and the user-drawn boxes acting as ROI pairs. The 4-stage
        pipeline (exposure / gain / brightness-contrast-sat /
        white-balance) converges on values matching the truth on the
        selected patches; those settled values become the seed for
        the subsequent K/C/Sat color search.
        """
        from dataset_tools.camera_isp.colorchecker24 import (
            COLORCHECKER24_SRGB,
        )

        live, stamp = self._bridge.latest_frame()
        if live is None or (stamp and time.monotonic() - stamp > 1.5):
            self._notify("cc24 seed: no fresh live frame", YELLOW, 3.0)
            return

        truth_srgb = COLORCHECKER24_SRGB[
            np.asarray(ref_indices, dtype=np.int64)
        ]
        truth_bgr = truth_srgb[:, ::-1].astype(np.uint8)  # (n, 3) BGR

        synthetic = live.copy()
        h, w = synthetic.shape[:2]
        valid_pairs: list[
            tuple[
                tuple[int, int, int, int],
                tuple[int, int, int, int],
            ]
        ] = []
        for slot, (x0, y0, x1, y1) in enumerate(boxes):
            x0c = max(0, min(int(x0), w))
            x1c = max(0, min(int(x1), w))
            y0c = max(0, min(int(y0), h))
            y1c = max(0, min(int(y1), h))
            if x1c <= x0c or y1c <= y0c:
                continue
            synthetic[y0c:y1c, x0c:x1c] = truth_bgr[slot]
            box = (x0c, y0c, x1c, y1c)
            valid_pairs.append((box, box))

        if not valid_pairs:
            self._notify("cc24 seed: no valid boxes", YELLOW, 3.0)
            return

        self._notify(
            f"cc24: seeding HW from {len(valid_pairs)} truth patches "
            "(running AUTO pipeline)...",
            GREEN, 4.0,
        )

        # Temporarily swap reference + ROI so _run_auto_pipeline scopes
        # its stats and pipeline writes to the user's truth patches.
        # Restore originals in finally so cc24 mode behaviour elsewhere
        # (live pane, save flow) is unaffected.
        saved_ref = self._reference
        saved_pairs = self._roi_pairs
        saved_protect = self._protect_brightness_in_auto_pipeline
        try:
            self._reference = synthetic
            self._roi_pairs = valid_pairs
            self._protect_brightness_in_auto_pipeline = self._brightness_user_locked
            self._run_auto_pipeline()
        finally:
            self._reference = saved_ref
            self._roi_pairs = saved_pairs
            self._protect_brightness_in_auto_pipeline = saved_protect

    # ------------------------------------------------------------------
    # Pedestal (signed-Δ ``brightness``) stage — runs BEFORE color search
    # ------------------------------------------------------------------

    def _bri_default_minmax(self) -> tuple[int, int, int]:
        """Resolve (default, min, max) for the ``brightness`` register.

        Falls back to the ``_SLIDERS`` baseline when the device caps
        haven't been probed yet — the pedestal stage must still be
        able to compose a valid register write.
        """
        builtin = next(
            ((d, lo, hi) for _, k, lo, hi, d in _SLIDERS if k == "brightness"),
            (128, 0, 255),
        )
        cap = (getattr(self, "_device_caps", {}) or {}).get("brightness")
        if cap is None:
            return builtin
        return (
            int(cap.get("default", builtin[0])),
            int(cap.get("min", builtin[1])),
            int(cap.get("max", builtin[2])),
        )

    def _run_pedestal_stage(
        self,
        *,
        mode: str,
        ref_box: tuple[int, int, int, int] | None = None,
        live_box: tuple[int, int, int, int] | None = None,
        announce: bool = True,
    ) -> "DarkLevelEstimate | None":  # noqa: F821
        """Estimate + apply the signed-Δ pedestal offset.

        Args:
            mode: One of ``"auto"`` (cc24 / no reference), ``"ref"``
                (full-frame ref-vs-live), ``"manual"`` (user nominated
                a black patch on both ref and live). The selector is
                explicit so a future mode (e.g. dark-frame capture)
                can be added without regressing the existing paths.
            ref_box: REF-frame XYXY box in REF native pixels — only
                used when ``mode == "manual"``.
            live_box: LIVE-frame XYXY box in live native pixels —
                only used when ``mode == "manual"``.
            announce: When ``True`` the stage publishes a banner /
                ``_notify`` summarising the chosen Δ. Set ``False``
                when the caller wants to suppress UI noise (e.g.
                tests).

        Returns the :class:`DarkLevelEstimate` for downstream logging,
        or ``None`` when the stage is skipped (no live frame, etc.).
        """
        from dataset_tools.camera_isp.pedestal import (
            apply_pedestal_offset,
            estimate_pedestal_offset_auto,
            estimate_pedestal_offset_manual,
            estimate_pedestal_offset_ref_mode,
        )

        live, stamp = self._bridge.latest_frame()
        if live is None or time.monotonic() - stamp > 1.0:
            self._notify("Pedestal: no live frame", YELLOW, 3.0)
            return None

        bri_default, bri_min, bri_max = self._bri_default_minmax()
        cur_bri = int(self._applied.get("brightness", bri_default))

        # Zero brightness before sampling so that gain*pedestal compound
        # error is eliminated.  Write bri_default (= offset-zero) to the
        # hardware, settle for one exposure cycle, then re-grab a fresh
        # live frame with a known neutral starting point.
        if cur_bri != bri_default:
            print(
                f"[Pedestal] zeroing brightness {cur_bri} -> {bri_default}"
                f" before sampling", flush=True,
            )
            ok_z, err_z = self._apply_via_v4l2_or_ros({"brightness": bri_default})
            if ok_z:
                self._applied["brightness"] = bri_default
                cur_bri = bri_default
            else:
                print(
                    f"[Pedestal] WARNING: brightness zero failed ({err_z});"
                    f" sampling at bri={cur_bri}", flush=True,
                )
            time.sleep(0.25)  # one exposure settle
            fresh, _ = self._bridge.latest_frame()
            if fresh is None:
                self._notify("Pedestal: lost live after brightness reset", YELLOW, 3.0)
                return None
            live = fresh

        if mode == "ref":
            ref_resized = self._resize_ref_to(live.shape)
            est = estimate_pedestal_offset_ref_mode(live, ref_resized)
        elif mode == "manual":
            if ref_box is None:
                self._notify("Pedestal: no ref box", YELLOW, 3.0)
                return None
            if live_box is None:
                self._notify("Pedestal: no live box", YELLOW, 3.0)
                return None
            ref_resized = self._resize_ref_to(live.shape)
            # Scale ref_box to ref_resized's coordinate frame if the
            # reference image was downscaled.
            if ref_resized.shape != self._reference.shape:
                sx = ref_resized.shape[1] / float(self._reference.shape[1])
                sy = ref_resized.shape[0] / float(self._reference.shape[0])
                ref_box = (
                    int(ref_box[0] * sx), int(ref_box[1] * sy),
                    int(ref_box[2] * sx), int(ref_box[3] * sy),
                )
            # live_box is already in live-frame pixel coordinates — no scaling.
            est = estimate_pedestal_offset_manual(ref_resized, live, ref_box, live_box)
        else:
            # Default / cc24 path: scan the live frame.
            est = estimate_pedestal_offset_auto(live)

        new_bri = apply_pedestal_offset(est.delta, bri_default, bri_min, bri_max)

        print(
            f"[Pedestal] mode={mode}  used={est.used}  delta={est.delta:+d}\n"
            f"           measured_y={est.measured_y:.2f}"
            f"  dark_frac={est.dark_pixel_frac:.4f} ({est.n_dark_pixels} px)\n"
            f"           confidence={est.confidence:.2f}  warn={est.warn!r}\n"
            f"           bri_default={bri_default}  cur_bri={cur_bri}"
            f"  -> new_bri={new_bri}",
            flush=True,
        )
        self._last_pedestal_delta = int(est.delta)
        self._last_pedestal_used = str(est.used)

        if new_bri != cur_bri:
            ok, err = self._apply_via_v4l2_or_ros({"brightness": int(new_bri)})
            if ok:
                self._applied["brightness"] = int(new_bri)
                self._dirty_save = True
            else:
                self._notify(
                    f"Pedestal: write failed ({err}); kept d=0",
                    YELLOW, 4.0,
                )
                return est

        if announce:
            sign = "+" if est.delta > 0 else ""
            head = f"Pedestal d={sign}{est.delta} ({est.used})"
            if est.warn:
                self._notify(f"{head} — {est.warn}", YELLOW, 6.0)
            else:
                colour = GREEN if est.delta != 0 else GREY
                self._notify(head, colour, 3.0)
        return est

    def _run_pedestal_for_manual(
        self,
        ref_img: np.ndarray,
        live_snap: np.ndarray,
        pairs: list[tuple[tuple[int, int, int, int],
                          tuple[int, int, int, int]]],
    ) -> None:
        """Three-branch pedestal prompt for the ``m`` workflow.

        Layout:
          ``[1..N]`` reuse pair #k as the black reference (digit key)
          ``[n]``    pick a NEW box on REF only (no LIVE pair)
          ``[s]``    skip → fall back to auto pedestal estimation

        Default action = darkest existing pair (Enter). Implementation
        notes:
          * Numbered list capped at the 5 darkest pairs to keep the
            modal readable when the user has many ROI pairs.
          * The "n" branch reuses ``cv.selectROI`` on a fit-to-screen
            REF canvas — the new box is **not** appended to ``pairs``
            (pedestal metadata is intentionally orthogonal to K/C/Sat
            search ROIs).
          * Sanity warnings are non-blocking; the user-chosen Δ wins.
        """
        if self._brightness_user_locked:
            self._notify(
                "Manual: keeping user blacklevel; pedestal skipped",
                GREEN, 4.0,
            )
            return

        from dataset_tools.camera_isp.pedestal import darkest_pair_index

        cv = self._opencv

        # Pre-compute "darkest pair" suggestion so the modal can name
        # it. Also rank all pairs by ref-Y so we can list at most 5.
        if pairs:
            ranked: list[tuple[int, float]] = []
            for i, (rb, _) in enumerate(pairs):
                x0, y0, x1, y1 = rb
                x0 = max(0, min(int(x0), ref_img.shape[1]))
                x1 = max(0, min(int(x1), ref_img.shape[1]))
                y0 = max(0, min(int(y0), ref_img.shape[0]))
                y1 = max(0, min(int(y1), ref_img.shape[0]))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = ref_img[y0:y1, x0:x1].astype(np.float64)
                y_mean = float(
                    0.114 * crop[..., 0].mean()
                    + 0.587 * crop[..., 1].mean()
                    + 0.299 * crop[..., 2].mean()
                )
                ranked.append((i, y_mean))
            ranked.sort(key=lambda t: t[1])
            shown = ranked[:5]
            darkest_idx = shown[0][0] if shown else None
        else:
            shown = []
            darkest_idx = darkest_pair_index(pairs, ref_img)  # → None

        # Build the prompt window. We deliberately use a small,
        # self-contained modal (not the main canvas) so the existing
        # HUD discipline isn't disturbed; it auto-closes on key press.
        win = "Pedestal: black reference?"
        text_w = 620
        thumb_max_w, thumb_max_h = 400, 300
        # Build REF thumbnail with boxes highlighted and numbered.
        rh, rw = ref_img.shape[:2]
        scale = min(thumb_max_w / max(rw, 1), thumb_max_h / max(rh, 1), 1.0)
        th, tw = max(1, int(rh * scale)), max(1, int(rw * scale))
        import cv2 as _cv2_thumb
        thumb = _cv2_thumb.resize(ref_img, (tw, th))
        # Draw each ranked pair box on the thumbnail (white border, yellow for suggested).
        _COLOURS = [(255, 255, 255)] * 10
        for rank, (idx, _ym) in enumerate(shown):
            rb, _ = pairs[idx]
            bx0 = int(rb[0] * scale)
            by0 = int(rb[1] * scale)
            bx1 = int(rb[2] * scale)
            by1 = int(rb[3] * scale)
            col = (0, 200, 255) if idx == darkest_idx else (200, 200, 200)
            cv.rectangle(thumb, (bx0, by0), (bx1, by1), col, 2)
            cv.putText(thumb, str(idx + 1), (bx0 + 3, by0 + 18),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        canvas_h = max(th, max(180, 90 + 24 * (len(shown) + 2)))
        canvas_w = text_w + tw + 8
        canvas = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
        # Paste thumbnail on the right.
        canvas[:th, text_w + 4:text_w + 4 + tw] = thumb
        cv.putText(canvas, "Reference looks washed?",
                   (10, 32), cv.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        y_line = 64
        for rank, (idx, y_mean) in enumerate(shown):
            tag = f"[{idx + 1}] reuse pair #{idx + 1}  (ref Y={y_mean:.0f})"
            color = YELLOW if idx == darkest_idx else WHITE
            if idx == darkest_idx:
                tag += "   <- suggested (Enter)"
            cv.putText(canvas, tag, (16, y_line),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y_line += 24
        cv.putText(canvas, "[n] pick a NEW black box on REF",
                   (16, y_line), cv.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
        y_line += 24
        cv.putText(canvas, "[s] skip - use auto pedestal estimation",
                   (16, y_line), cv.FONT_HERSHEY_SIMPLEX, 0.55, GREY, 1)
        cv.imshow(win, canvas)

        # Accept digits 1..N (only those listed), n / s, Enter, Esc.
        listed_digits = {ord(str(idx + 1)) for idx, _ in shown}
        chosen: str | None = None
        chosen_idx: int | None = None
        while chosen is None:
            k = cv.waitKey(50) & 0xFF
            if k == 0xFF:
                continue
            if k == 27:  # Esc → skip silently
                chosen = "skip_silent"
                break
            if k in (13, 10):  # Enter → darkest pair, or skip if no pairs
                if darkest_idx is not None:
                    chosen = "reuse"
                    chosen_idx = darkest_idx
                else:
                    chosen = "skip"
                break
            if k in (ord("s"), ord("S")):
                chosen = "skip"
                break
            if k in (ord("n"), ord("N")):
                chosen = "new"
                break
            if k in listed_digits:
                chosen = "reuse"
                chosen_idx = int(chr(k)) - 1
                break

        try:
            cv.destroyWindow(win)
        except Exception:  # noqa: BLE001
            pass

        if chosen == "skip_silent":
            return
        if chosen == "skip":
            self._run_pedestal_stage(mode="ref")
            return
        if chosen == "reuse" and chosen_idx is not None:
            ref_box_rot, live_box = pairs[chosen_idx]
            ref_box = self._unrotate_box(ref_box_rot, self._reference.shape[:2])
            self._run_pedestal_stage(
                mode="manual",
                ref_box=ref_box,
                live_box=live_box,
            )
            return
        if chosen == "new":
            # Two-window REF+LIVE selection. The user picks a black box
            # on the ref first, then a matching black box on the live.
            # Neither box enters ``pairs``; pedestal metadata is
            # intentionally orthogonal to K/C/Sat search ROIs.
            new_win = "Pedestal: drag a black box on REF"
            disp_ref, roi_scale = self._fit_for_roi(ref_img)
            sel = cv.selectROI(new_win, disp_ref,
                               showCrosshair=True, fromCenter=False)
            try:
                cv.destroyWindow(new_win)
            except Exception:  # noqa: BLE001
                pass
            if sel is None:
                return
            sx, sy, sw, sh = (int(sel[0]), int(sel[1]),
                              int(sel[2]), int(sel[3]))
            if sw < 4 or sh < 4:
                self._notify("Pedestal: ref box too small; skipped", YELLOW, 3.0)
                return
            # Scale back to full-size rotated-ref coords, then unrotate.
            inv = 1.0 / max(roi_scale, 1e-9)
            rot_box = (
                int(round(sx * inv)), int(round(sy * inv)),
                int(round((sx + sw) * inv)), int(round((sy + sh) * inv)),
            )
            ref_box = self._unrotate_box(rot_box, self._reference.shape[:2])

            # Second window: pick the matching black box on the live frame.
            live_win = "Pedestal: drag a black box on LIVE"
            disp_live, live_roi_scale = self._fit_for_roi(live_snap)
            lsel = cv.selectROI(live_win, disp_live,
                                showCrosshair=True, fromCenter=False)
            try:
                cv.destroyWindow(live_win)
            except Exception:  # noqa: BLE001
                pass
            if lsel is None:
                return
            lx, ly, lw, lh = (int(lsel[0]), int(lsel[1]),
                               int(lsel[2]), int(lsel[3]))
            if lw < 4 or lh < 4:
                self._notify("Pedestal: live box too small; skipped", YELLOW, 3.0)
                return
            linv = 1.0 / max(live_roi_scale, 1e-9)
            live_box: tuple[int, int, int, int] = (
                int(round(lx * linv)), int(round(ly * linv)),
                int(round((lx + lw) * linv)), int(round((ly + lh) * linv)),
            )
            self._run_pedestal_stage(mode="manual", ref_box=ref_box, live_box=live_box)
            return

    # ------------------------------------------------------------------
    # Unified K/C/Sat color search ('c' key) — plan v4
    # ------------------------------------------------------------------

    def _run_color_search(self) -> None:
        """Run the unified K/C/Sat search (color_search.search_KCS).

        Mode dispatch:
          * cc24 boxes present (``self._cc24_search``) → cost_24card
            against X-Rite truth Lab values (chained from cc24 wizard).
          * ROI pairs present → m / ROI cost (cost_manual_roi).
          * Otherwise         → AUTO SWD palette cost.

        Seed is the *currently applied* (white_balance, contrast,
        saturation). Driver guarantees the device is left at the seed
        if no candidate beats it (legacy fallback contract).

        UX:
          * Live HUD shows ``[SEARCH n/N J=...]`` and updates per eval.
          * Press ``q`` or ``Esc`` during search to cancel — seed is
            re-applied to the camera and no settings are committed.
        """
        # cc24 mode normally freezes hardware. The cc24 wizard sets
        # ``self._cc24_search`` right before calling us so the search
        # leg can run; we still bail out for plain cc24 'a'/'c' presses.
        cc24_data = getattr(self, "_cc24_search", None)
        # Pop immediately so subsequent searches don't mistakenly
        # reuse the boxes — this slot is one-shot per wizard run.
        if cc24_data is not None:
            self._cc24_search = None
        if self._colorchecker_mode and cc24_data is None:
            self._notify(
                "ColorSearch unavailable in ColorChecker24 mode", YELLOW, 4.0,
            )
            return
        live, stamp = self._bridge.latest_frame()
        if live is None or (stamp and time.monotonic() - stamp > 1.5):
            self._notify("No fresh live frame for color search", RED, 4.0)
            return
        ref = self._reference

        from dataset_tools.camera_isp import color_search as cs

        seed = cs.KCS(
            K=int(self._applied.get(
                "white_balance",
                self._device_caps.get("white_balance", {}).get("default", 4600),
            )),
            C=int(self._applied.get(
                "contrast",
                self._device_caps.get("contrast", {}).get("default", 32),
            )),
            Sat=int(self._applied.get(
                "saturation",
                self._device_caps.get("saturation", {}).get("default", 64),
            )),
        )

        # Build cost function.
        roi_pairs = list(self._roi_pairs) if self._roi_pairs else []
        if cc24_data is not None:
            # cc24: per-patch ΔE2000 against X-Rite truth Lab values.
            # Each live box is a stable rectangle drawn during the
            # wizard; we just average BGR inside it on every fresh
            # frame and let cost_24card map to Lab + ΔE.
            from dataset_tools.camera_isp.colorchecker24 import (
                COLORCHECKER24_SRGB,
            )
            from dataset_tools.camera_isp.color_search import bgr_to_lab
            boxes = cc24_data["boxes"]
            ref_idx = np.asarray(cc24_data["ref_indices"], dtype=np.int64)
            # COLORCHECKER24_SRGB is sRGB uint8 (24, 3); convert to BGR
            # then to Lab to match cost_24card's truth_lab contract.
            truth_srgb = COLORCHECKER24_SRGB[ref_idx]
            truth_bgr = truth_srgb[:, ::-1].copy()  # RGB -> BGR
            truth_lab = bgr_to_lab(truth_bgr.astype(np.uint8)[None, :, :])[0]
            # Pad/truncate to (24, 3) for the strict cost_24card check.
            n_pairs = truth_lab.shape[0]

            def _patch_means(frame_bgr: np.ndarray) -> np.ndarray:
                h, w = frame_bgr.shape[:2]
                out = np.zeros((24, 3), dtype=np.float64)
                for slot, (x0, y0, x1, y1) in enumerate(boxes):
                    x0 = max(0, min(int(x0), w))
                    x1 = max(0, min(int(x1), w))
                    y0 = max(0, min(int(y0), h))
                    y1 = max(0, min(int(y1), h))
                    if x1 > x0 and y1 > y0:
                        out[slot] = frame_bgr[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
                return out

            # Truth padded to (24, 3); unused slots stay at 0 with weight 0.
            truth_full = np.zeros((24, 3), dtype=np.float64)
            truth_full[:n_pairs] = truth_lab
            weights = np.zeros(24, dtype=np.float64)
            weights[:n_pairs] = 1.0
            cost_fn = cs.cost_24card(_patch_means, truth_full, weights=weights)
            mode_label = f"cc24({n_pairs} patch)"
            self._notify(
                f"cc24 search: dE2000 across {n_pairs} patches",
                GREEN, 4.0,
            )
        elif roi_pairs:
            cost_fn = cs.cost_manual_roi(
                roi_pairs, ref,
                prev_kcs=seed, lam_reg=0.0,
            )
            mode_label = f"m({len(roi_pairs)} ROI)"
        else:
            # AUTO mode: chroma-weighted Sliced Wasserstein on (a*, b*).
            #
            # No clustering, no spatial correspondence — we compare the
            # color-palette *distribution shapes* of ref vs. live. L*
            # is dropped entirely (lab[:, 1:]), so exposure/contrast
            # mismatch can't pollute white-balance/saturation search.
            # Chroma-weighted importance sampling makes saturated
            # patches dominate over grey background.
            #
            # Cost is a deterministic function of params (fixed RNG
            # seeds for ref + live samplers and the SWD direction set),
            # which is the precondition for grid+refine search.
            cost_fn = cs.cost_palette_swd(ref)
            mode_label = "AUTO-swd"
            self._notify(
                "AUTO: SWD on (a*, b*) -- searching K/C/Sat",
                GREEN, 4.0,
            )

        # Adapter: search_KCS expects HwWriter.write / FrameGrabber.grab,
        # while _CalibratorStageBridge exposes write_v4l2 / grab_frame.
        bridge = _CalibratorStageBridge(self)

        # [ColorSearch-grab] Track grab-call freshness: print a one-shot warning
        # if two consecutive grabs return literally the same array (means
        # the spin thread hasn't received a new ROS frame between writes).
        debug_grab_state = {"prev_first_pixel": None, "stuck": 0, "ok": 0}

        # Fresh-frame gate: every write records its monotonic stamp,
        # every grab waits until ``latest_frame`` carries a stamp
        # strictly newer than the previous gate target. This stops
        # ``frame_capture`` from returning the same cached frame
        # n_capture times in a row when the camera publishes at ~30 fps
        # (without this, only the *first* grab after a write blocks for
        # a fresh frame; the following 6 grabs return instantly with
        # the same array).
        write_state = {
            # Updated by both write() and grab(); each grab() must see
            # a stamp > this value to be considered fresh.
            "gate_t": 0.0,
            # Hard ceiling so a hung publisher can't deadlock the search.
            "wait_timeout_s": 0.35,
            # Sub-frame poll interval (~60 Hz).
            "poll_s": 0.015,
            # Eval-counter heartbeat (incremented in write()).
            "eval_n": 0,
            "t0": time.monotonic(),
        }

        class _Adapter:
            def write(self_inner, params):
                bridge.write_v4l2(params)
                # Stamp *after* the bridge's settle sleep so the gate
                # only waits for genuinely post-settle frames.
                write_state["gate_t"] = time.monotonic()
                # Eval-counter heartbeat: one write == one eval.
                write_state["eval_n"] += 1
                n = write_state["eval_n"]
                if n == 1 or n % 20 == 0:
                    pass  # [ColorSearch] eval# — suppressed
            def grab(self_inner):
                # Wait for a frame whose ROS receive-timestamp is
                # strictly newer than the last write OR the last
                # successfully-grabbed frame. Falls through after a
                # timeout so we never deadlock; the [ColorSearch-grab]
                # stuck counter will surface the issue if it happens.
                deadline = time.monotonic() + write_state["wait_timeout_s"]
                target_t = write_state["gate_t"]
                fresh_stamp = 0.0
                while True:
                    _, stamp = self._bridge.latest_frame()
                    if stamp and stamp > target_t:
                        fresh_stamp = stamp
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(write_state["poll_s"])
                f = bridge.grab_frame()
                if f is not None and fresh_stamp > 0.0:
                    # Advance the gate so the *next* grab in this
                    # capture burst also waits for a brand-new frame.
                    write_state["gate_t"] = fresh_stamp
                if f is not None:
                    fp = (int(f[0, 0, 0]), int(f[0, 0, 1]), int(f[0, 0, 2]))
                    if fp == debug_grab_state["prev_first_pixel"]:
                        debug_grab_state["stuck"] += 1
                    else:
                        debug_grab_state["ok"] += 1
                    debug_grab_state["prev_first_pixel"] = fp
                return f

        adapter = _Adapter()

        # Eval-budget estimate so the HUD can show progress.
        s = cs.SearchConfig()
        if s.strategy == "coord":
            # Coord descent: passes * (n_K + n_C + n_S) + refine.
            per_pass = s.n_K + s.n_C + s.n_S
            total_est = min(s.max_evals, max(1, s.coord_passes) * per_pass) + 1
        else:
            total_est = min(s.max_evals, s.n_K * s.n_C * s.n_S)
        if s.final_refine:
            total_est += 27

        cv = self._opencv
        cancelled = {"flag": False}
        seen = {"n": 0, "best": float("inf"),
                "best_metric": float("nan"), "metric_label": ""}

        # In-flight banner on first paint so the user sees it immediately.
        # Duration is short (3s) because the mode HUD on the live pane
        # carries the same progress + metric live; keeping the banner
        # up the whole search would just overlap the HUD.
        self._mode = f"SEARCH 0/{total_est}"
        self._notify(
            f"ColorSearch[{mode_label}] running -- q/Esc to cancel",
            YELLOW, 3.0,
        )

        def _pump_gui() -> None:
            """Repaint one frame + read one key; non-blocking."""
            live_now, stamp_now = self._bridge.latest_frame()
            live_stale = (
                live_now is None
                or (time.monotonic() - stamp_now > 1.5 if stamp_now else True)
            )
            canvas = self._render(live_now, live_stale)
            cv.imshow(self.WINDOW_NAME, canvas)
            k = cv.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                cancelled["flag"] = True

        # Initial paint so "SEARCH 0/N" appears before the first eval.
        _pump_gui()

        class _UserCancelled(Exception):
            pass

        def _fmt_metric(value: float, label: str) -> str:
            if not label or not np.isfinite(value):
                return ""
            return f"{label}={value:.2f}"

        def _on_trace(entry) -> None:
            seen["n"] += 1
            if entry.J < seen["best"]:
                seen["best"] = entry.J
                seen["best_metric"] = float(entry.metric_value)
                seen["metric_label"] = str(entry.metric_label)
            cur = _fmt_metric(float(entry.metric_value),
                              str(entry.metric_label))
            best_txt = _fmt_metric(seen["best_metric"], seen["metric_label"])
            tail = ""
            if cur and best_txt:
                tail = f" | {cur} (best {seen['best_metric']:.2f})"
            elif cur:
                tail = f" | {cur}"
            self._mode = f"SEARCH {seen['n']}/{total_est}{tail}"
            _pump_gui()
            if cancelled["flag"]:
                raise _UserCancelled()

        try:
            result = cs.search_KCS(
                seed=seed, cost_fn=cost_fn,
                writer=adapter, grabber=adapter,
                device_caps=self._device_caps,
                on_trace=_on_trace,
            )
        except _UserCancelled:
            # Restore seed on the device so the user is back where they started.
            try:
                adapter.write(seed.as_params())
            except Exception:  # noqa: BLE001
                pass
            self._mode = "IDLE"
            self._notify(
                f"ColorSearch[{mode_label}] cancelled -- restored "
                f"K={seed.K} C={seed.C} Sat={seed.Sat}",
                YELLOW, 5.0,
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._mode = "IDLE"
            self._notify(f"ColorSearch failed: {exc}", RED, 5.0)
            return

        self._mode = "IDLE"

        # [ColorSearch] mode summary — suppressed (print spam)
        # [ColorSearch-grab] summary — suppressed
        # [trace] — suppressed

        if result.fallback_used:
            self._notify(
                f"ColorSearch[{mode_label}] no improvement (J={result.seed_J:.3f}); "
                "kept seed",
                YELLOW, 5.0,
            )
            return

        # Adopt the new triple as authoritative.
        new_params = result.best.as_params()
        self._applied.update(new_params)
        self._dirty_save = True
        self._sync_trackbars_to_applied()

        # Refresh the SW-ISP / CCM cache with the new live state. The
        # stored CCM was solved against the *pre-search* live snapshot;
        # now that K/C/Sat have moved, that CCM is applied to a frame it
        # was never fit to and the right-hand SW pane shows nonsense.
        # Re-running the m solver with the same ROI pairs against a
        # fresh frame keeps the CCM consistent with the camera state.
        if roi_pairs and self._sw_isp_computed:
            self._refresh_ccm_after_search(roi_pairs, ref)

        self._notify(
            f"ColorSearch[{mode_label}] J {result.seed_J:.3f}->{result.best_J:.3f}"
            + (f" ({seen['metric_label']} best={seen['best_metric']:.2f})"
               if seen["metric_label"] and np.isfinite(seen["best_metric"])
               else "")
            + f" K={result.best.K} C={result.best.C} Sat={result.best.Sat}",
            GREEN, 6.0,
        )

    def _refresh_ccm_after_search(
        self,
        roi_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
        ref: np.ndarray,
    ) -> None:
        """Re-solve the SW-ISP / CCM from a fresh frame.

        Called by ``_run_color_search`` after K/C/Sat have moved, so
        the previously cached CCM no longer matches the live data.
        Settles briefly to let the camera reach the new state before
        snapping the live frame.
        """
        time.sleep(0.15)
        for _ in range(2):                  # drop two stale frames
            self._bridge.latest_frame()
        live_now, stamp = self._bridge.latest_frame()
        if live_now is None or (stamp and time.monotonic() - stamp > 1.5):
            # Don't crash; just leave the (now stale) CCM and warn.
            self._notify(
                "CCM refresh skipped: no fresh frame after color search",
                YELLOW, 4.0,
            )
            return
        try:
            from dataset_tools.camera_isp.solver import manual_match_ref
            _proposed, dbg = manual_match_ref(
                ref, live_now.copy(), roi_pairs,
                self._applied, self._device_caps,
            )
        except Exception as exc:  # noqa: BLE001
            self._notify(f"CCM refresh failed: {exc}", YELLOW, 4.0)
            return
        self._update_sw_isp_from_dbg(dbg)

    # ------------------------------------------------------------------
    # Help overlay
    # ------------------------------------------------------------------

    def _show_help(self) -> None:
        if self._colorchecker_mode:
            self._notify(
                "m=ColorChecker wizard  s=Save  r=Reset  u=Undo  "
                "p=Snap  q=Quit  |  0=linear  1=RPCC2  2=RPCC2+ridge  "
                "3=ALS-LAB",
                WHITE, 6.0,
            )
            return
        self._notify(
            "a=Auto+ColorSearch  m=Manual+ColorSearch  x=ClearROI  "
            "s=Save  r=Reset  u=Undo  p=Snap  q=Quit",
            WHITE, 6.0,
        )

    # ------------------------------------------------------------------
    # Manual box-pair mode
    # ------------------------------------------------------------------

    def _run_manual(self) -> None:
        """Pair-drag manual color match (REF mode).

        Workflow:
          1. User drags a box on the REFERENCE pane.
          2. User drags the corresponding box on the LIVE pane.
          3. Press [Space] to add another pair, [Enter] to apply, [Esc] to cancel.

        Every (ref, live) pair feeds exposure / WB / saturation / contrast.
        No "is this neutral gray?" labelling is needed in REF mode — the
        user's correspondences are the ground truth.
        """
        # Zero brightness before dispatch unless the user has explicitly
        # tuned the split blacklevel/brightness sliders.
        self._flush_locked_brightness()
        self._zero_brightness_for_calibration()
        # ColorChecker24 mode → run the 24-patch wizard instead.
        if self._colorchecker_mode:
            return self._run_color_checker()
        cv = self._opencv
        live, stamp = self._bridge.latest_frame()
        if live is None or (stamp and time.monotonic() - stamp > 1.5):
            self._notify("No fresh live frame for manual match", RED, 4.0)
            return
        live_snap = live.copy()
        ref = self._get_rotated_ref()
        # Original image shape for box coordinate back-projection at save time.
        _orig_ref_hw = self._reference.shape[:2]

        self._notify(
            "Manual: drag REF box -> drag LIVE box -> [Space] more / [Enter] apply / [Esc] cancel.",
            WHITE, 6.0,
        )

        pairs: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []

        ref_win = "Pick REF box  (LIVE preview on the right)"
        live_win = "Pick LIVE box  (REF preview on the right)"
        prompt_win = "Pairs / continue"

        def _annotated(
            target: str,
            skip_idx: int | None = None,
            extra_box: tuple[int, int, int, int] | None = None,
            extra_label: str | None = None,
        ) -> np.ndarray:
            """Return REF or LIVE annotated with previously-picked boxes.

            *extra_box* / *extra_label* let callers slip in a freshly
            picked-but-not-yet-committed box (e.g. show the just-drawn
            REF box while the user is picking the matching LIVE box).
            """
            base = ref if target == "ref" else live_snap
            shown_boxes: list[tuple[int, int, int, int]] = []
            labels: list[str] = []
            for j, (rb, lb) in enumerate(pairs):
                if skip_idx is not None and j == skip_idx:
                    continue
                shown_boxes.append(rb if target == "ref" else lb)
                labels.append(str(j + 1))
            if extra_box is not None:
                shown_boxes.append(extra_box)
                labels.append(extra_label or "?")
            return self._overlay_boxes(base, shown_boxes, labels=labels)

        def _select_on_target(
            window_title: str,
            target_img: np.ndarray,
            preview_img: np.ndarray,
        ) -> tuple[int, int, int, int] | None:
            """Show TARGET (drag) | PREVIEW (read-only) and return ROI.

            ROI is clipped to the target half **and** to the target's
            vertical band (the canvas is 30 px taller than the target
            for the "DRAG HERE" caption — without clipping, a drag
            that strays into the caption strip would map y past the
            real frame).
            """
            composite, preview_x0, target_view_w = self._compose_with_preview(
                target_img, preview_img,
            )
            # Use the *actual* resized target width (NOT preview_x0,
            # which includes the 8 px gap) so the back-projection
            # scale exactly matches the on-screen pixel-to-source
            # ratio. Composite is fit-to-height so X and Y share the
            # same scale.
            target_scale = target_view_w / max(1, target_img.shape[1])
            target_h_view = composite.shape[0] - 30  # mirrors _compose_with_preview
            sel = cv.selectROI(
                window_title, composite,
                showCrosshair=True, fromCenter=False,
            )
            try:
                cv.destroyWindow(window_title)
            except Exception:  # noqa: BLE001
                pass
            if sel is None:
                return None
            sx, sy, sw, sh = (
                int(sel[0]), int(sel[1]), int(sel[2]), int(sel[3]),
            )
            if sw < 4 or sh < 4:
                return None
            # Clip ROI to the target half (left side) and target band
            # (top portion before the caption strip).
            x_max = target_view_w
            y_max = target_h_view
            sx2 = min(sx + sw, x_max)
            sy2 = min(sy + sh, y_max)
            sx = max(0, min(sx, x_max - 1))
            sy = max(0, min(sy, y_max - 1))
            sw2 = sx2 - sx
            sh2 = sy2 - sy
            if sw2 < 4 or sh2 < 4:
                return None
            # Map composite-space coords back to source-image pixels.
            inv = 1.0 / max(target_scale, 1e-9)
            x0 = int(round(sx * inv))
            y0 = int(round(sy * inv))
            x1 = int(round((sx + sw2) * inv))
            y1 = int(round((sy + sh2) * inv))
            x0 = max(0, min(x0, target_img.shape[1]))
            x1 = max(0, min(x1, target_img.shape[1]))
            y0 = max(0, min(y0, target_img.shape[0]))
            y1 = max(0, min(y1, target_img.shape[0]))
            if x1 - x0 < 4 or y1 - y0 < 4:
                return None
            return (x0, y0, x1, y1)

        def _pick_pair(replace_idx: int | None = None) -> str:
            """Pick (ref, live) box pair.

            * ``replace_idx is None`` -> append a new pair.
            * Otherwise -> overwrite ``pairs[replace_idx]``.

            Returns ``"ok"`` (committed) or ``"abort"`` (user cancelled).
            """
            ref_box = _select_on_target(
                ref_win,
                _annotated("ref", skip_idx=replace_idx),
                _annotated("live", skip_idx=replace_idx),
            )
            if ref_box is None:
                return "abort"
            # Compute the label this pair will end up with so the LIVE
            # preview shows the just-picked REF box with the correct
            # number (matches what the review modal will display).
            new_label = (
                str(replace_idx + 1)
                if replace_idx is not None
                else str(len(pairs) + 1)
            )
            live_box = _select_on_target(
                live_win,
                _annotated("live", skip_idx=replace_idx),
                _annotated(
                    "ref",
                    skip_idx=replace_idx,
                    extra_box=ref_box,
                    extra_label=new_label,
                ),
            )
            if live_box is None:
                return "abort"
            new_pair = (ref_box, live_box)
            if replace_idx is None:
                pairs.append(new_pair)
            else:
                pairs[replace_idx] = new_pair
            return "ok"

        while True:
            if _pick_pair() == "abort":
                break
            # Continue?
            cont = np.full((130, 520, 3), 30, dtype=np.uint8)
            cv.putText(cont,
                       f"Pairs: {len(pairs)}",
                       (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
            cv.putText(cont,
                       "[Space] add another   [Enter] review/apply   [Esc] cancel",
                       (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)
            cv.imshow(prompt_win, cont)
            k2 = 0xFF
            while k2 == 0xFF:
                k2 = cv.waitKey(50) & 0xFF
                if k2 in (ord(" "), 13, 10, 27):
                    break
                k2 = 0xFF
            try:
                cv.destroyWindow(prompt_win)
            except Exception:  # noqa: BLE001
                pass
            if k2 == 27:
                self._notify("Manual cancelled", YELLOW, 3.0)
                return
            if k2 in (13, 10):  # Enter -> proceed to review
                break
            # Space -> loop again

        if not pairs:
            self._notify("Manual: no boxes selected", YELLOW, 3.0)
            return

        # Combined REF+LIVE review modal. Click a numbered box on
        # either pane to re-pick that pair; Enter accepts.
        while True:
            if not pairs:
                self._notify("Manual: no boxes left", YELLOW, 3.0)
                return
            action, which = self._review_pairs_modal(
                ref, live_snap, pairs,
                title="Manual review",
            )
            if action == "cancel":
                self._notify("Manual cancelled", YELLOW, 3.0)
                return
            if action == "clear":
                pairs.clear()
                if _pick_pair() == "abort":
                    return
                continue
            if action == "redraw":
                _pick_pair(replace_idx=int(which))
                continue
            if action == "apply":
                break

        # Pedestal SSOT (signed-Δ ``brightness``) — user has just
        # confirmed the ROI pairs, so this is the right moment to ask
        # "is one of these the black reference?" before any color
        # search runs. Three branches: reuse an existing pair, draw a
        # fresh REF-only box, or skip → auto.
        try:
            self._run_pedestal_for_manual(ref, live_snap, pairs)
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Pedestal prompt failed: {exc}", YELLOW, 4.0)

        # Persist the boxes for the next AUTO run — every stage will
        # use these ROIs (not just neutral-WB). User clears them with
        # the dedicated key in the run loop.
        # Ref boxes were picked on the rotated display; unrotate them
        # back to self._reference (original) pixel coords for the solver.
        self._roi_pairs = [
            (self._unrotate_box(rb, _orig_ref_hw), lb)
            for rb, lb in pairs
        ]
        self._notify(
            f"ROI saved: {len(self._roi_pairs)} pair(s) — next 'a' will "
            f"calibrate on these regions only ('x' to clear)",
            GREEN, 5.0,
        )

        from dataset_tools.camera_isp.solver import manual_match_ref
        try:
            proposed, dbg = manual_match_ref(
                ref, live_snap, pairs,
                self._applied, self._device_caps,
            )
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Manual solver failed: {exc}", RED, 5.0)
            return

        # Always refresh the SW-ISP debug pane params from the solver's
        # linear-domain *_sw fields, regardless of whether we'll actually
        # write to the hardware below.
        self._update_sw_isp_from_dbg(dbg)

        if not proposed:
            warn = dbg.get("warnings", []) if isinstance(dbg, dict) else []
            self._notify(
                "Manual: nothing to apply" + (f" ({warn})" if warn else ""),
                YELLOW, 4.0,
            )
            return

        # SW-only mode: do NOT push to hardware. The third pane already
        # reflects the new self._sw_isp on the next render tick.
        if self._sw_only:
            sw = self._sw_isp
            self._notify(
                f"SW-only manual: kr={sw.kr:.3f} kb={sw.kb:.3f} "
                f"exp={sw.exp_scale:.3f}  (hardware unchanged)",
                GREEN, 5.0,
            )
            return

        # Snapshot pre-apply state so 'u' rolls back manual + the
        # chained color search in one step.
        self._push_undo()
        ok, err = self._apply_via_v4l2_or_ros(proposed)
        if ok:
            self._applied.update(proposed)
            self._dirty_save = True
            self._sync_trackbars_to_applied()
            summary = ", ".join(f"{k}={v}" for k, v in proposed.items())
            self._notify(f"Manual applied: {summary}", GREEN, 5.0)
        else:
            # Apply failed -> drop the phantom undo entry we just pushed.
            if self._undo_stack:
                self._undo_stack.pop()
            self._notify(f"Manual apply failed: {err}", RED, 5.0)
            return

        # Chain color search after manual ROI commit. The user already
        # provided ground-truth ROI pairs; running the unified K/C/Sat
        # search now (m/ROI cost mode) refines on those exact patches
        # plus the global SWD background term.
        try:
            self._run_color_search()
        except Exception as exc:  # noqa: BLE001
            self._notify(f"Manual: color search failed: {exc}", YELLOW, 5.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> int:
        cv = self._opencv
        self._snapshot_initial()
        self._build_window()
        self._force_manual_modes()
        # Zero brightness on startup so the initial state is always the
        # calibration baseline (brightness=default), not whatever the
        # device happened to have stored (e.g. 3 from a previous session).
        self._zero_brightness_for_calibration()
        self._sync_trackbars_to_applied()

        # Install Ctrl-C handler so users can shut the GUI down from the
        # terminal that launched it. cv.waitKey is invoked with a short
        # poll interval below, which lets the Python interpreter run the
        # handler between iterations.
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigtstp = signal.getsignal(signal.SIGTSTP)

        def _on_stop_signal(_signum, _frame):
            """Ctrl-C (SIGINT) and Ctrl-Z (SIGTSTP) both request a clean exit."""
            self._stop_render = True

        try:
            signal.signal(signal.SIGINT, _on_stop_signal)
            signal.signal(signal.SIGTSTP, _on_stop_signal)
        except (ValueError, OSError):
            # Not on the main thread (e.g. embedded test harness) —
            # signal install is best-effort.
            prev_sigint = None
            prev_sigtstp = None

        if self._colorchecker_mode:
            self._notify(
                "ColorChecker24 mode. AUTO disabled. "
                "Press 'm' to start the 24-patch wizard, 's' to save.",
                WHITE, 6.0,
            )
        else:
            self._notify(
                f"Ready. Reference {self._reference.shape[1]}x{self._reference.shape[0]}. "
                "Press 'a' for Auto, drag sliders, 's' to save.",
                WHITE, 6.0,
            )

        # [GuiRate-debug] One-shot per-second instrumentation:
        #   * gui fps    = how often the main loop iterated
        #   * fresh%     = fraction of iterations where latest_frame()
        #                  stamp advanced since the previous iteration
        # If gui<<33 → main loop blocked (sync_trackbars / waitKey).
        # If gui~33 but fresh%~0 → spin thread updates _latest but the
        #   GIL or lock keeps GUI from observing the new stamp.
        gui_dbg = {"t": time.monotonic(), "frames": 0,
                   "fresh": 0, "last_stamp": 0.0}
        try:
            while not self._stop_render:
                live, stamp = self._bridge.latest_frame()
                # [GuiRate-debug] commented out — was spamming terminal every second.
                # now = time.monotonic()
                # gui_dbg["frames"] += 1
                # if stamp != gui_dbg["last_stamp"]:
                #     gui_dbg["fresh"] += 1
                #     gui_dbg["last_stamp"] = stamp
                # if now - gui_dbg["t"] >= 1.0:
                #     print(
                #         f"[GuiRate] gui={gui_dbg['frames']}fps "
                #         f"fresh={gui_dbg['fresh']}/{gui_dbg['frames']} "
                #         f"mode={self._mode} "
                #         f"stamp_age={now - stamp:.2f}s",
                #         flush=True,
                #     )
                #     gui_dbg["t"] = now
                #     gui_dbg["frames"] = 0
                #     gui_dbg["fresh"] = 0
                live_stale = (
                    live is None
                    or (time.monotonic() - stamp > 1.5 if stamp else True)
                )
                self._maybe_apply_pending()
                if self._mode != "IDLE":
                    # Snap sliders back to applied values during any
                    # compute so user drags during AUTO/SEARCH visibly
                    # bounce back instead of silently being dropped.
                    self._sync_trackbars_to_applied()
                canvas = self._render(live, live_stale)
                cv.imshow(self.WINDOW_NAME, canvas)
                key = cv.waitKey(30) & 0xFF
                if self._main_window_closed() or self._trackbar_panel_closed():
                    self._stop_render = True
                    break

                if key == 0xFF:
                    continue
                if key == ord("q"):
                    if self._dirty_save:
                        # Show warning and require a second 'q' within 3 s.
                        if (
                            self._banner
                            and self._banner[0].startswith("Press 'q' again")
                            and time.monotonic() < self._banner_until
                        ):
                            break
                        self._notify(
                            "Press 'q' again to quit without saving (or 's' to save first)",
                            YELLOW, 3.0,
                        )
                    else:
                        break
                elif key == ord("a"):
                    self._trigger_auto()
                elif key == ord("m"):
                    self._run_manual()
                elif key == ord("x"):
                    # Clear ROI pairs so subsequent AUTO runs use the
                    # full frame again. Useful when the camera/scene
                    # has moved and the previously-dragged live boxes
                    # no longer cover the intended subject.
                    if self._roi_pairs:
                        n = len(self._roi_pairs)
                        self._roi_pairs = []
                        self._notify(
                            f"Cleared {n} ROI pair(s) — AUTO will use full frame",
                            YELLOW, 3.0,
                        )
                    else:
                        self._notify("No ROI pairs to clear", WHITE, 2.0)
                # NOTE: the `d` key (toggle SW-only / CCM preview mode) has
                # been disabled to keep the GUI focused on hardware
                # calibration. The underlying ``_toggle_sw_only`` and CCM
                # pipeline code are preserved for future re-enablement.
                elif key == ord("s"):
                    self._save_override()
                elif key == ord("r"):
                    self._reset()
                elif key == ord("u"):
                    self._undo_last()
                elif key == ord("p"):
                    self._snapshot_screenshot(live)
                elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
                    self._apply_ccm_variant(self._VARIANT_BY_KEY[chr(key)])
                elif key in (ord("?"), ord("h")):
                    self._show_help()
        finally:
            if prev_sigint is not None:
                try:
                    signal.signal(signal.SIGINT, prev_sigint)
                except (ValueError, OSError):
                    pass
            if prev_sigtstp is not None:
                try:
                    signal.signal(signal.SIGTSTP, prev_sigtstp)
                except (ValueError, OSError):
                    pass
            self._join_auto_thread()
            self._destroy_all_windows()
        return 0

    def _join_auto_thread(self) -> None:
        """Best-effort wait for AUTO/search worker before tearing down GUI.

        The worker is daemon, so process exit will reap it regardless,
        but joining briefly here prevents it from writing one last
        parameter set after the user already closed the window.
        """
        thread = self._auto_thread
        if thread is None or not thread.is_alive():
            return
        try:
            thread.join(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass

    def _destroy_all_windows(self) -> None:
        cv = self._opencv
        for _ in range(8):
            try:
                cv.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
            try:
                cv.waitKey(20)
            except Exception:  # noqa: BLE001
                pass

    def _main_window_closed(self) -> bool:
        """True only when the main window has been outright destroyed.

        OpenCV+GTK exposes three states via ``WND_PROP_VISIBLE``:

        * ``>= 1.0`` -- window is mapped and visible;
        * ``== 0.0`` -- minimized / temporarily hidden by the WM;
        * ``< 0.0``  -- window no longer exists (user clicked X, or it
          was never registered).

        Treating ``0.0`` as "closed" caused false positives whenever the
        user minimised the calibrator or the WM briefly unmapped it,
        making the process self-exit. We only react to a true destroy
        (``< 0.0``) here; the X-click signal is also covered by
        :meth:`_trackbar_panel_closed`, which is more reliable on GTK.
        """
        cv = self._opencv
        try:
            visible = cv.getWindowProperty(
                self.WINDOW_NAME,
                cv.WND_PROP_VISIBLE,
            )
        except Exception:  # noqa: BLE001
            return self._main_window_seen
        if visible >= 1.0:
            self._main_window_seen = True
            return False
        if visible < 0.0:
            return self._main_window_seen
        # 0.0: minimized / transiently unmapped — not a close.
        return False

    def _trackbar_panel_closed(self) -> bool:
        if not self._trackbars_ready:
            return False
        cv = self._opencv
        try:
            cv.getTrackbarPos("exposure", self.WINDOW_NAME)
        except Exception:  # noqa: BLE001
            return True
        return False


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camera_isp_calibrator",
        description=(
            "Interactive ISP color calibrator. Connects to a running usb_cam "
            "node, lets you match its output to a reference image, and saves "
            "the result to ~/.ros/ibrobot/camera_isp_overrides/{camera}.json."
        ),
    )
    parser.add_argument(
        "--camera", required=True,
        help="Camera name as declared in robot_config YAML (e.g. 'top', 'wrist').",
    )
    parser.add_argument(
        "--reference", required=False, default=None,
        help=(
            "Path to reference image or video. First frame is used for video. "
            "Required unless --colorchecker is set."
        ),
    )
    parser.add_argument(
        "--colorchecker", action="store_true",
        help=(
            "Use the X-Rite ColorChecker Classic 24 sRGB reference values "
            "instead of an image. Disables AUTO mode; the 24-patch "
            "wizard ('m') drives the camera through the K/C/Sat search."
        ),
    )
    parser.add_argument(
        "--max-exposure-ms", type=float, default=None,
        help=(
            "Upper bound on exposure time in milliseconds for AUTO mode. "
            "Default 15 ms (≈safe at 30+ fps). Raise this when the scene "
            "is dark and the reference is dimmer than what 15 ms can "
            "reach; the pipeline will not exceed 1/fps minus a small "
            "frame-time headroom."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.colorchecker and not args.reference:
        print(
            "[camera_isp_calibrator] --reference is required unless "
            "--colorchecker is set"
        )
        return 2
    if args.colorchecker and args.reference:
        print(
            "[camera_isp_calibrator] --colorchecker overrides --reference; "
            "ignoring --reference"
        )
    opencv = require_opencv_gui()

    if args.colorchecker:
        # Generate a built-in thumbnail of the 24-patch chart so the
        # second pane has something meaningful to show.
        from dataset_tools.camera_isp.colorchecker24 import (
            make_checker_thumbnail,
        )
        reference = make_checker_thumbnail(360, 540, with_index=True)
    else:
        try:
            reference = load_reference(args.reference, opencv)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[camera_isp_calibrator] {exc}")
            return 2

    try:
        bridge = RosBridge(args.camera)
    except RuntimeError as exc:
        print(f"[camera_isp_calibrator] {exc}")
        return 3

    try:
        # Wait briefly for the first frame so the user sees something
        # immediately rather than a "WAITING" screen during the first
        # second.
        deadline = time.monotonic() + _RECONNECT_S
        while time.monotonic() < deadline:
            f, _ = bridge.latest_frame()
            if f is not None:
                break
            time.sleep(0.05)
        win = CalibratorWindow(
            bridge, reference, opencv, colorchecker=bool(args.colorchecker),
            max_exposure_ms=args.max_exposure_ms,
        )
        return win.run()
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

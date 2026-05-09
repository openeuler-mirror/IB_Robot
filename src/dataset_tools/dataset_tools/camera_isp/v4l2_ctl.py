"""Lightweight ``v4l2-ctl`` shell wrapper for camera_isp_calibrator.

The upstream ``usb_cam`` node hardcodes legacy V4L2 control names
(``white_balance_temperature_auto``, ``exposure_auto``, ``focus_auto``)
that newer Linux UVC drivers have renamed to
(``white_balance_automatic``, ``auto_exposure``, ``focus_automatic``).
On those kernels, ``ros2 param set /xxx_camera auto_white_balance false``
silently fails inside the node — it triggers
``v4l2-ctl -c white_balance_temperature_auto=0`` which the driver rejects
with ``unknown control``. The driver's AWB stays on, and any subsequent
manual ``white_balance`` write is denied with ``Permission denied``.

This module bypasses that by talking to ``v4l2-ctl`` directly with a
new-name → legacy-name fallback chain, so the calibrator can guarantee
manual modes are actually engaged before it asks the solver to drive
the parameters.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable


# Ordered new-first / legacy-fallback names for each logical control.
# The first one that v4l2-ctl accepts wins.
_NAME_FALLBACKS: dict[str, tuple[str, ...]] = {
    "white_balance_auto": ("white_balance_automatic", "white_balance_temperature_auto"),
    "white_balance":      ("white_balance_temperature",),
    "exposure_auto":      ("auto_exposure", "exposure_auto"),
    "exposure":           ("exposure_time_absolute", "exposure_absolute"),
    "focus_auto":         ("focus_automatic", "focus_auto"),
    # Standard V4L2 controls used by the 4-stage hardware pipeline. They
    # carry no name aliases on the kernels we target, but listing them
    # here lets ``resolve_ctrls`` populate min/max/default uniformly so
    # the calibrator can drive every stage off a single caps dictionary.
    "gain":               ("gain",),
    "brightness":         ("brightness",),
    "saturation":         ("saturation",),
    "contrast":           ("contrast",),
    "sharpness":          ("sharpness",),
    "focus":              ("focus_absolute", "focus"),
}

# Special menu values for `auto_exposure`:
#   1 = Manual Mode
#   3 = Aperture Priority Mode (= "auto" in old API).
_AUTO_EXPOSURE_MANUAL = 1
_AUTO_EXPOSURE_AUTO   = 3


@dataclass
class CtrlInfo:
    """Resolved info for a single logical control on a given device."""

    real_name: str
    minimum: int | None
    maximum: int | None
    default: int | None = None
    value: int | None = None


def have_v4l2_ctl() -> bool:
    """Return True iff ``v4l2-ctl`` is on PATH."""
    return shutil.which("v4l2-ctl") is not None


def _run(args: list[str], timeout: float = 2.0) -> tuple[int, str, str]:
    """Run a v4l2-ctl invocation. Returns (rc, stdout, stderr)."""
    try:
        cp = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return -1, "", str(exc)


def list_ctrls(device: str) -> dict[str, dict]:
    """Return ``{ctrl_name: {min, max, default, value}}`` for *device*."""
    rc, out, _err = _run(["v4l2-ctl", "--device", device, "--list-ctrls"])
    if rc != 0:
        return {}
    result: dict[str, dict] = {}
    for line in out.splitlines():
        # Example line:
        #   white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=10 default=4600 value=4600 flags=inactive
        s = line.strip()
        if " : " not in s:
            continue
        head, body = s.split(" : ", 1)
        name = head.split()[0]
        info: dict = {}
        for tok in body.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    info[k] = int(v)
                except ValueError:
                    info[k] = v
        result[name] = info
    return result


def resolve_ctrls(device: str) -> dict[str, CtrlInfo]:
    """Resolve every logical key in :data:`_NAME_FALLBACKS` on *device*."""
    available = list_ctrls(device)
    out: dict[str, CtrlInfo] = {}
    for logical, candidates in _NAME_FALLBACKS.items():
        for cand in candidates:
            if cand in available:
                info = available[cand]
                out[logical] = CtrlInfo(
                    real_name=cand,
                    minimum=info.get("min"),
                    maximum=info.get("max"),
                    default=info.get("default"),
                    value=info.get("value"),
                )
                break
    return out


def set_ctrl(device: str, logical: str, value: int,
             resolved: dict[str, CtrlInfo]) -> tuple[bool, str]:
    """Write a single logical control. Returns (ok, message)."""
    info = resolved.get(logical)
    if info is None:
        return False, f"control '{logical}' not exposed by {device}"
    rc, _out, err = _run([
        "v4l2-ctl", "--device", device,
        "-c", f"{info.real_name}={int(value)}",
    ])
    if rc != 0:
        return False, err.strip() or f"v4l2-ctl rc={rc}"
    return True, info.real_name


def force_manual_modes(device: str,
                       resolved: dict[str, CtrlInfo]) -> list[str]:
    """Disable AWB / AE / AF on *device*. Returns list of human messages.

    Always returns — never raises. Any failure is reported in the message
    list so the GUI can surface it without crashing the calibrator.
    """
    msgs: list[str] = []
    if "white_balance_auto" in resolved:
        ok, msg = set_ctrl(device, "white_balance_auto", 0, resolved)
        msgs.append(("AWB off via " if ok else "AWB off failed: ") + msg)
    if "exposure_auto" in resolved:
        ok, msg = set_ctrl(device, "exposure_auto",
                           _AUTO_EXPOSURE_MANUAL, resolved)
        msgs.append(("AE off via " if ok else "AE off failed: ") + msg)
    if "focus_auto" in resolved:
        ok, msg = set_ctrl(device, "focus_auto", 0, resolved)
        msgs.append(("AF off via " if ok else "AF off failed: ") + msg)
    return msgs


def verify_manual_wb(device: str,
                     resolved: dict[str, CtrlInfo]) -> bool:
    """Return True iff the AWB control reads back 0 (i.e. AWB is off)."""
    info = resolved.get("white_balance_auto")
    if info is None:
        return False
    rc, out, _err = _run([
        "v4l2-ctl", "--device", device, "-C", info.real_name,
    ])
    if rc != 0:
        return False
    # Output format: "white_balance_automatic: 0"
    if ":" not in out:
        return False
    try:
        return int(out.split(":", 1)[1].strip()) == 0
    except (ValueError, IndexError):
        return False


# Mapping from calibrator slider keys (== usb_cam ROS parameter names) to
# logical v4l2_ctl keys / direct V4L2 control names. Anything not listed here
# is left to the ROS path.
_SLIDER_TO_LOGICAL: dict[str, str] = {
    "white_balance": "white_balance",
    "exposure":      "exposure",
}
# Keys whose V4L2 control name was never renamed across kernels — we just
# write them directly with no fallback list.
_DIRECT_KEYS: tuple[str, ...] = (
    "brightness", "contrast", "saturation", "sharpness", "gain",
)


def apply_params(device: str,
                 resolved: dict[str, CtrlInfo],
                 params: dict) -> tuple[bool, str, set[str]]:
    """Best-effort: write *params* directly via v4l2-ctl.

    Bypasses the usb_cam ROS parameter callback entirely so it never has a
    chance to cascade-rewrite every control (which on broken-fork kernels
    spams 'unknown control' and stalls the set_parameters service for
    seconds, causing Auto / Reset timeouts).

    Returns ``(all_ok, summary, handled_keys)``:

    * ``all_ok``       -- True iff every attempted write succeeded.
    * ``summary``      -- human-readable failure list, or ``"ok"``.
    * ``handled_keys`` -- the set of input keys we actually attempted via
      v4l2-ctl (succeeded **or** failed). Anything in ``params`` but not
      in this set was skipped (bool-typed, non-numeric, or unknown to
      this layer) and the caller must still route it via ROS.
    """
    failures: list[str] = []
    handled: set[str] = set()
    for k, v in params.items():
        if isinstance(v, bool):
            continue  # handled by force_manual_modes
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if k in _SLIDER_TO_LOGICAL:
            handled.add(k)
            ok, msg = set_ctrl(device, _SLIDER_TO_LOGICAL[k], iv, resolved)
            if not ok:
                failures.append(f"{k}: {msg}")
            continue
        if k in _DIRECT_KEYS:
            handled.add(k)
            rc, _out, err = _run([
                "v4l2-ctl", "--device", device, "-c", f"{k}={iv}",
            ])
            if rc != 0:
                failures.append(f"{k}: {err.strip() or rc}")
            continue
        # Unknown key — leave it to the ROS path (caller sees it absent
        # from ``handled`` and re-routes).
    if failures:
        return False, "; ".join(failures)[:200], handled
    return True, "ok", handled

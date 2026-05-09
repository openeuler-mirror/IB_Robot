"""Pure-function hardware ISP stage solvers for camera_isp_calibrator.

This module implements the four-stage hardware calibration pipeline
described in ``临时/hardware_isp_calibration_plan.md`` (v2):

    Stage 1 — Exposure  : push Y_mean to ~128 (50% code value),
                          fps-aware exposure ceiling.
    Stage 2 — Gain      : Step A bumps gain if still under-bright; then
                          Step B verifies SNR ≥ 15 dB and falls back to
                          ``halved gain + brightness offset`` when noise
                          floor is breached.
    Stage 3 — Sat/Con   : in CC24 mode, write driver default; in REF
                          mode, scale toward ref statistics but clip to
                          ``default ± 30%``.
    Stage 4 — White Bal : delegated to ``solver.solve_kelvin_only``.

Every function here is a *pure proposer* — given a frame and current
hardware values, return what to write next. Iteration / IO / ROS state
machine all live in ``camera_isp_calibrator.py``.

Design invariants (do **not** weaken without a plan revision):

* All luminance statistics ``y_mean / y_std / SNR`` must be computed
  with the **overexposure mask** ``Y >= CLIP_THRESHOLD`` removed first
  (ISO 12232 / DxO standard practice — plan §8.3).
* ``BRIGHTNESS`` is touched **only** by ``evaluate_gain_step_b`` when
  the SNR floor is breached (plan §8.5). All other code paths must
  treat brightness as locked at ``default_value``.
* ``cv2.cvtColor(BGR2YCrCb)`` returns channels in the order
  **[Y, Cr, Cb]** — not YCbCr. This module only uses the chroma
  magnitude ``sqrt((Cb-128)² + (Cr-128)²)`` which is symmetric, but
  callers that touch the raw image must remember the order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants — shared with the plan; do not hardcode at the call site.
# ---------------------------------------------------------------------------

#: Pixels with Y >= this are treated as saturated (ISO 12232 mask).
CLIP_THRESHOLD: float = 250.0
#: Stage 1 / Stage 2 luminance target in 8-bit code values (≈ 50%, plan §8.2).
TARGET_Y_MEAN_CODE: float = 128.0
#: Stage 1 redirect target when overexposed (255 × 0.80, plan §2.1).
P99_TARGET_CODE: float = 204.0
#: Overexposure trigger; clip_ratio above this → switch to p99 target.
#:
#: Set to 10% per ISO 12232 SOS exposure index (which allows up to
#: 12.5% saturation against an 18%-gray reference). Below this we trust
#: ``y_mean_excl_clip`` and accept the small specular-clip overhead —
#: typical white-card scenes (white card occupies 15–25% of frame and
#: barely clips at correct exposure) would otherwise loop the
#: "overexposed" branch every iteration and underexpose the scene.
OVEREXP_CLIP_RATIO: float = 0.10
#: SNR floor in dB (plan §8.4); below this we refuse to keep the gain bump.
#:
#: Set to 20 dB — the project-wide "worst acceptable image quality"
#: threshold (10× signal-to-noise ratio = noticeable but not crippling
#: noise). Below this the Step B fallback halves gain and shifts
#: brightness to recover the deficit. Tightening from the historical
#: 15 dB is intentional; do not relax without a plan revision.
SNR_FLOOR_DB: float = 20.0
#: Stage 3 sat/con band around driver default (plan §3.3).
SAT_CON_BAND: float = 0.30
#: Stage 2 Step A "half-step" coefficient on gain proposal (plan §2.1).
_GAIN_STEP_COEFF: float = 0.5
#: Gate fraction: gain only engages when current_exp ≥ exp_max × this.
#: Professional ISP rule — exhaust analog exposure first because gain
#: amplifies noise (1/√(gain) penalty per ISO 12232). Plan §2.1 refinement.
GAIN_EXPOSURE_GATE: float = 0.80
#: Brightness fallback band as fraction of (max - min).
_BRIGHTNESS_BAND: float = 0.20
#: SNR patch geometry.
_SNR_PATCH_DEFAULT_N: int = 8
_SNR_PATCH_DEFAULT_SIZE: int = 32
#: Patch is "flat" when its low-pass-filtered range (max-min) is below
#: this. Range-after-blur attenuates pixel noise but preserves real
#: edges, so this gate is robust to noisy uniform target patches.
_SNR_BLUR_RANGE_FLAT_MAX: float = 30.0
_SNR_BLUR_KERNEL: int = 9
_SNR_BLUR_SIGMA: float = 2.0
_SNR_MIN_KEPT_PATCHES: int = 4
_SNR_MIN_PATCH_STD: float = 0.5  # below this we treat the patch as a frozen reading.


# ---------------------------------------------------------------------------
# Lightweight value objects.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CtrlCaps:
    """Min / max / default for a single V4L2 control."""

    minimum: int
    maximum: int
    default: int

    @property
    def span(self) -> int:
        return max(0, self.maximum - self.minimum)


@dataclass(frozen=True)
class YStats:
    """Luminance statistics with overexposure pixels masked out."""

    y_mean_excl_clip: float    # mean of Y where Y < CLIP_THRESHOLD (0..255).
    y_p99: float               # 99th percentile of full Y (used to size pull-down).
    clip_ratio: float          # fraction of pixels with Y >= CLIP_THRESHOLD.
    y_std_excl_clip: float     # std of Y where Y < CLIP_THRESHOLD, normalized to [0,1].
    valid_pixel_count: int     # |Y_valid|, useful to detect "everything clipped".


@dataclass(frozen=True)
class ChromaStats:
    """Chroma statistics in YCbCr (BT.601)."""

    chroma_mag_median: float   # median of sqrt((Cb-128)² + (Cr-128)²).


@dataclass(frozen=True)
class SnrPatch:
    x: int
    y: int
    mean: float
    std: float


@dataclass(frozen=True)
class SnrResult:
    """SNR estimate via random flat-patch sampling.

    ``snr_db`` is ``None`` when fewer than :data:`_SNR_MIN_KEPT_PATCHES`
    flat patches were found — typically because the scene is heavily
    textured or saturated. Callers must treat ``None`` as
    "unmeasurable, do not gamble" rather than as a value < floor.
    """

    snr_db: float | None
    n_patches_used: int
    patches: tuple[SnrPatch, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExposureProposal:
    new_exp: int
    rationale: str   # "overexposed" | "underexposed" | "ok" | "totally_black"
    ratio: float
    hit_max: bool
    hit_min: bool
    converged: bool


@dataclass(frozen=True)
class GainProposal:
    new_gain: int
    ratio: float
    hit_max: bool
    no_op: bool       # True when Y_mean_live is already at/above target (no bump needed).


@dataclass(frozen=True)
class GainStepBResult:
    final_gain: int
    final_brightness: int | None   # None ⇒ caller must NOT touch BRIGHTNESS.
    snr_db: float | None
    snr_ok: bool
    fallback_active: bool          # True ⇒ "halved gain + brightness offset" was applied.
    note: str


@dataclass(frozen=True)
class SatConProposal:
    new_saturation: int
    new_contrast: int
    sat_ratio: float | None    # None when degenerate (live is grayscale or cc24 mode).
    con_ratio: float | None
    rationale: str             # "cc24_default" | "ref_match" | "ref_match_partial".


# ---------------------------------------------------------------------------
# Frame statistics.
# ---------------------------------------------------------------------------

#: ``(x1, y1, x2, y2)`` rectangle in pixel coordinates; half-open like
#: NumPy slicing. Convention: ``x2 > x1 and y2 > y1``.
RoiBox = tuple[int, int, int, int]


def extract_roi_frame(
    bgr: np.ndarray,
    boxes: list[RoiBox] | tuple[RoiBox, ...] | None,
) -> np.ndarray:
    """Stack ROI pixels into a (1, N, 3) virtual BGR frame.

    When *boxes* is ``None`` or empty, returns the input unchanged.
    Out-of-bounds boxes are clipped; degenerate boxes are dropped. The
    returned array preserves only the ROI pixels, so all downstream
    statistics (``compute_y_stats`` / ``compute_chroma_stats``) operate
    purely on the user-selected regions. The 1-pixel-tall shape is
    intentional: many SNR / spatial estimators that need 2-D structure
    will gracefully degrade (returning ``snr_db=None``) rather than
    silently mis-report on stitched ROIs.

    Empty result (e.g. all boxes clipped to zero area) returns the
    original *bgr* — refusing to fabricate a 0-pixel frame downstream.
    """
    if not boxes:
        return bgr
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return bgr
    h, w = bgr.shape[:2]
    parts: list[np.ndarray] = []
    for box in boxes:
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        x1 = max(0, min(w, min(x1, x2)))
        x2 = max(0, min(w, max(x1, x2)))
        y1 = max(0, min(h, min(y1, y2)))
        y2 = max(0, min(h, max(y1, y2)))
        if x2 > x1 and y2 > y1:
            parts.append(bgr[y1:y2, x1:x2].reshape(-1, 3))
    if not parts:
        return bgr
    flat = np.concatenate(parts, axis=0)
    return flat.reshape(1, -1, 3)


def compute_y_stats(bgr: np.ndarray) -> YStats:
    """Compute luminance statistics with overexposure mask applied.

    Parameters
    ----------
    bgr : np.ndarray
        ``(H, W, 3)`` uint8 OpenCV BGR frame. Other dtypes are accepted
        but the clip threshold ``CLIP_THRESHOLD`` is in 8-bit units, so
        non-uint8 inputs should already be in 0..255 range.

    Returns
    -------
    YStats
        See class docstring. Always returns; never raises.
    """
    if bgr.size == 0:
        return YStats(0.0, 0.0, 0.0, 0.0, 0)
    arr = bgr.astype(np.float32, copy=False)
    # BT.601 luminance (BGR order in OpenCV).
    y = (0.114 * arr[..., 0] + 0.587 * arr[..., 1]
         + 0.299 * arr[..., 2])
    clip_mask = y >= CLIP_THRESHOLD
    clip_ratio = float(clip_mask.mean())
    valid = y[~clip_mask]
    if valid.size == 0:
        # Everything clipped: signal "max overexposure" so the proposer
        # pulls exposure down hard.
        return YStats(
            y_mean_excl_clip=255.0,
            y_p99=float(y.max()),
            clip_ratio=clip_ratio,
            y_std_excl_clip=0.0,
            valid_pixel_count=0,
        )
    return YStats(
        y_mean_excl_clip=float(valid.mean()),
        y_p99=float(np.percentile(y, 99)),
        clip_ratio=clip_ratio,
        y_std_excl_clip=float(valid.std() / 255.0),
        valid_pixel_count=int(valid.size),
    )


def compute_chroma_stats(bgr: np.ndarray) -> ChromaStats:
    """Compute median chroma magnitude in BT.601 YCbCr."""
    if bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
        return ChromaStats(chroma_mag_median=0.0)
    # OpenCV returns channels [Y, Cr, Cb] — symmetric for chroma_mag.
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[..., 1].astype(np.float32) - 128.0
    cb = ycrcb[..., 2].astype(np.float32) - 128.0
    mag = np.sqrt(cr * cr + cb * cb)
    return ChromaStats(chroma_mag_median=float(np.median(mag)))


def compute_snr_db(
    bgr: np.ndarray,
    *,
    n_patches: int = _SNR_PATCH_DEFAULT_N,
    patch_size: int = _SNR_PATCH_DEFAULT_SIZE,
    flat_range_max: float = _SNR_BLUR_RANGE_FLAT_MAX,
    seed: int | None = 0,
) -> SnrResult:
    """Estimate per-patch SNR (median μ/σ across flat regions, in dB).

    Algorithm:
      1. Convert to grayscale and Gaussian-blur with a 9x9 σ=2 kernel —
         this attenuates pixel-level noise while preserving structural
         edges. The blurred image is used **only** for the flatness
         gate; ``μ`` and ``σ`` are computed on the raw grayscale to
         keep noise in the SNR signal.
      2. Sample ``n_patches`` random ``patch_size×patch_size`` boxes;
         keep those whose blurred range ``max - min < flat_range_max``
         AND whose own mean is below the saturation cliff AND whose
         std is large enough to be a real reading.
      3. SNR_db = ``20 * log10(median(μ_k / σ_k))`` over kept patches.

    Returns ``snr_db = None`` when fewer than
    :data:`_SNR_MIN_KEPT_PATCHES` patches survived; callers must treat
    that as "unmeasurable".
    """
    if bgr.size == 0:
        return SnrResult(snr_db=None, n_patches_used=0)

    if bgr.ndim == 3 and bgr.shape[2] == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.astype(np.uint8, copy=False)

    h, w = gray.shape[:2]
    if h <= patch_size or w <= patch_size:
        return SnrResult(snr_db=None, n_patches_used=0)

    blurred = cv2.GaussianBlur(
        gray, (_SNR_BLUR_KERNEL, _SNR_BLUR_KERNEL), _SNR_BLUR_SIGMA,
    )

    rng = np.random.default_rng(seed)
    kept: list[SnrPatch] = []
    max_attempts = max(n_patches * 8, 32)
    for _ in range(max_attempts):
        if len(kept) >= n_patches:
            break
        x = int(rng.integers(0, w - patch_size + 1))
        y0 = int(rng.integers(0, h - patch_size + 1))
        b_block = blurred[y0:y0 + patch_size, x:x + patch_size]
        flat_score = float(b_block.max()) - float(b_block.min())
        if flat_score >= flat_range_max:
            continue
        p = gray[y0:y0 + patch_size, x:x + patch_size].astype(np.float32)
        m = float(p.mean())
        s = float(p.std())
        if m >= CLIP_THRESHOLD or s < _SNR_MIN_PATCH_STD:
            continue
        kept.append(SnrPatch(x=x, y=y0, mean=m, std=s))

    if len(kept) < _SNR_MIN_KEPT_PATCHES:
        return SnrResult(
            snr_db=None,
            n_patches_used=len(kept),
            patches=tuple(kept),
        )

    ratios = np.fromiter((p.mean / p.std for p in kept),
                        dtype=np.float64, count=len(kept))
    snr_linear = float(np.median(ratios))
    snr_db = 20.0 * float(np.log10(max(snr_linear, 1e-9)))
    return SnrResult(
        snr_db=snr_db,
        n_patches_used=len(kept),
        patches=tuple(kept),
    )


# ---------------------------------------------------------------------------
# Stage 1 — Exposure.
# ---------------------------------------------------------------------------

def propose_exposure(
    y_stats: YStats,
    current_exp: int,
    *,
    target_y_mean: float = TARGET_Y_MEAN_CODE,
    exp_min: int,
    exp_max: int,
    overexp_clip_ratio: float = OVEREXP_CLIP_RATIO,
    p99_target: float = P99_TARGET_CODE,
) -> ExposureProposal:
    """Propose the next exposure value (in raw V4L2 ticks).

    Branches (plan §2.1, refined):

      * **overexposed** — fires only when BOTH (a) ``clip_ratio ≥
        overexp_clip_ratio`` AND (b) ``y_mean_excl_clip > target_y_mean
        × 1.05``. This double-gate prevents the runaway underexposure
        that happens on white-card scenes where a small bright region
        chronically clips even at correct exposure (the non-clipped
        majority is already at the target — the clipping is structural
        scene content, not an exposure error). When clipping coexists
        with a normal mean we treat the frame as ``ok``.
      * **mean-match** — push ``y_mean_excl_clip`` toward ``target_y_mean``.
      * **totally_black** — drive exposure to ``exp_max``.

    Convergence is reported when ``|Δexp| / cur_exp < 5%`` *or* the
    rationale is ``ok``.
    """
    if exp_max < exp_min:
        raise ValueError("exp_max must be >= exp_min")

    overexp_signal = (
        y_stats.clip_ratio >= overexp_clip_ratio
        and y_stats.y_p99 > 0
        and y_stats.y_mean_excl_clip > target_y_mean * 1.05
    )

    if overexp_signal:
        ratio = p99_target / y_stats.y_p99
        rationale = "overexposed"
    elif y_stats.y_mean_excl_clip > 0:
        ratio = target_y_mean / y_stats.y_mean_excl_clip
        if ratio >= 1.05:
            rationale = "underexposed"
        elif ratio <= 0.95:
            rationale = "overexposed"
        else:
            rationale = "ok"
    else:
        ratio = float("inf")
        rationale = "totally_black"

    if rationale == "ok":
        new_exp = current_exp
    elif not np.isfinite(ratio):
        new_exp = exp_max
    else:
        new_exp = int(round(current_exp * ratio))

    new_exp = max(exp_min, min(exp_max, new_exp))
    # Convergence — measurement-driven, NOT step-size-driven.
    #
    # Earlier rule was ``|Δexp|/cur_exp < 5%`` which falsely declared
    # convergence whenever the proposer was making small relative
    # corrections, even if the measured Y_mean was still far from
    # target (e.g. ratio=1.05 → Δexp=5% → "converged" while actually
    # 5% off). The proper signal is: did the *measurement* match
    # target? That is exactly what ``rationale == "ok"`` means after
    # the next ``compute_y_stats``. Until then we keep iterating, but
    # we still call it converged when the rail is hit because no
    # further movement is physically possible.
    hit_max = new_exp == exp_max
    hit_min = new_exp == exp_min
    converged = (
        rationale == "ok"
        or (hit_max and rationale in ("underexposed", "totally_black"))
        or (hit_min and rationale == "overexposed")
    )
    return ExposureProposal(
        new_exp=new_exp,
        rationale=rationale,
        ratio=float(ratio) if np.isfinite(ratio) else float("inf"),
        hit_max=hit_max,
        hit_min=hit_min,
        converged=bool(converged),
    )


# ---------------------------------------------------------------------------
# Stage 2 — Gain (Step A) + SNR gate (Step B) + brightness fallback.
# ---------------------------------------------------------------------------

def propose_gain_step_a(
    y_mean_live: float,
    y_mean_target: float,
    current_gain: int,
    gain_caps: CtrlCaps,
    *,
    current_exp: int | None = None,
    exp_max: int | None = None,
    exposure_gate: float = GAIN_EXPOSURE_GATE,
) -> GainProposal:
    """Step A: half-step gain bump toward ``y_mean_target``.

    Coefficient ``0.5`` is the conservative half-step from plan §2.1 —
    do not change without a plan revision.

    Parameters
    ----------
    current_exp / exp_max:
        When both are provided, gain stays a no-op until the exposure
        rail is at least ``exposure_gate`` (default 80%) full. This
        enforces the professional ISP rule of exhausting analog
        exposure before turning on gain (which amplifies sensor
        noise). Pass ``None`` for either to disable the gate.
    exposure_gate:
        Fraction of ``exp_max`` at which gain becomes eligible. Plan
        §2.1 default is 0.80; lowering this trades SNR for headroom.
    """
    if gain_caps.maximum < gain_caps.minimum:
        raise ValueError("gain_caps invalid")

    # Exposure gate: do not engage gain while there is exposure headroom.
    # The convention is "no_op" with a synthetic ratio so callers can
    # distinguish gate-trip from already-bright frames in the trace.
    if current_exp is not None and exp_max is not None and exp_max > 0:
        if current_exp < exposure_gate * exp_max:
            return GainProposal(
                new_gain=current_gain,
                ratio=float("nan"),
                hit_max=False,
                no_op=True,
            )

    if y_mean_live <= 0:
        return GainProposal(
            new_gain=gain_caps.maximum,
            ratio=float("inf"),
            hit_max=True,
            no_op=False,
        )

    ratio = y_mean_target / y_mean_live
    if ratio <= 1.05:
        # Already bright enough; do not bump gain.
        return GainProposal(
            new_gain=current_gain,
            ratio=float(ratio),
            hit_max=False,
            no_op=True,
        )

    delta = (ratio - 1.0) * gain_caps.span * _GAIN_STEP_COEFF
    proposed = current_gain + int(round(delta))
    new_gain = max(gain_caps.minimum, min(gain_caps.maximum, proposed))
    return GainProposal(
        new_gain=new_gain,
        ratio=float(ratio),
        hit_max=(new_gain == gain_caps.maximum),
        no_op=(new_gain == current_gain),
    )


def evaluate_gain_step_b(
    y_mean_after_step_a: float,
    y_mean_target: float,
    gain_before_step_a: int,
    gain_after_step_a: int,
    snr_result: SnrResult,
    brightness_caps: CtrlCaps,
    gain_caps: CtrlCaps,
    *,
    snr_floor_db: float = SNR_FLOOR_DB,
) -> GainStepBResult:
    """Step B: verify SNR; if it dropped below the floor, halve gain
    and let brightness pick up the slack.

    The brightness fallback follows plan §2.3:

      * ``new_brightness = brightness.default + round(deficit)``
      * clipped to ``default ± 20% × range``
      * applied **only** when SNR is measured and below floor.

    When :class:`SnrResult` reports ``snr_db is None`` (not enough flat
    patches) we accept Step A as-is rather than gamble on a halve+
    brightness move that we cannot verify.
    """
    if gain_after_step_a == gain_before_step_a:
        # No-op Step A: nothing to roll back; pass SNR through for HUD.
        snr_ok = snr_result.snr_db is None or snr_result.snr_db >= snr_floor_db
        return GainStepBResult(
            final_gain=gain_after_step_a,
            final_brightness=None,
            snr_db=snr_result.snr_db,
            snr_ok=snr_ok,
            fallback_active=False,
            note="no_gain_change",
        )

    if snr_result.snr_db is None:
        return GainStepBResult(
            final_gain=gain_after_step_a,
            final_brightness=None,
            snr_db=None,
            snr_ok=True,    # benefit of doubt — calibrator HUD will warn separately.
            fallback_active=False,
            note=f"snr_unmeasurable_n_patches={snr_result.n_patches_used}",
        )

    if snr_result.snr_db >= snr_floor_db:
        return GainStepBResult(
            final_gain=gain_after_step_a,
            final_brightness=None,
            snr_db=snr_result.snr_db,
            snr_ok=True,
            fallback_active=False,
            note=f"snr_ok_{snr_result.snr_db:.1f}dB",
        )

    # SNR too low → halve gain back toward Step A's starting point.
    halved = (gain_before_step_a + gain_after_step_a) // 2
    halved = max(gain_caps.minimum, min(gain_caps.maximum, halved))

    deficit = y_mean_target - y_mean_after_step_a   # may be negative if already bright.
    proposed_bri = brightness_caps.default + int(round(deficit))
    band = int(round(_BRIGHTNESS_BAND * brightness_caps.span))
    bri_lo = max(brightness_caps.minimum, brightness_caps.default - band)
    bri_hi = min(brightness_caps.maximum, brightness_caps.default + band)
    final_bri = max(bri_lo, min(bri_hi, proposed_bri))
    # Pedestal SSOT: ``brightness`` is a constant offset added to every
    # pixel — auto modes are forbidden from pushing it positive (would
    # wash blacks). The signed Δ is owned by the pedestal stage; this
    # fallback may only *subtract* from default.
    final_bri = min(final_bri, brightness_caps.default)

    return GainStepBResult(
        final_gain=halved,
        final_brightness=final_bri,
        snr_db=snr_result.snr_db,
        snr_ok=False,
        fallback_active=True,
        note=(f"snr_{snr_result.snr_db:.1f}dB_below_{snr_floor_db:.1f}dB"
              "_halve_and_fallback"),
    )


# ---------------------------------------------------------------------------
# Stage 3 — Saturation / Contrast.
# ---------------------------------------------------------------------------

def _clip_headroom_cap(bgr: np.ndarray, kind: str) -> float:
    """Return the max ratio (new/current) that avoids *newly* introducing
    per-channel clips to 255 or 0.

    Model — camera ISP applies a linear stretch around the neutral midpoint
    (128 code): ``ch' = 128 + (ch - 128) * ratio``.  Already-clipped
    pixels (ch ≥ 255 or ch ≤ 0) are excluded from the headroom estimate so
    that the guard is strictly about *new* clipping.

    *kind* ``"sat"`` uses all three BGR channels; *kind* ``"con"`` uses the
    BT.601 Y channel.  Returns ``inf`` when there is no binding constraint
    (e.g. all channels near 128, or image is empty).
    """
    if bgr is None or bgr.size == 0:
        return float("inf")
    arr = bgr.reshape(-1, 3).astype(np.float32)
    cap = float("inf")
    if kind == "sat":
        for c in range(3):
            ch = arr[:, c]
            hi = ch[ch < 254.5]    # exclude already-clipped to 255
            lo = ch[ch > 0.5]      # exclude already-clipped to 0
            if hi.size > 0:
                p99 = float(np.percentile(hi, 99))
                if p99 > 128.0:
                    cap = min(cap, 127.0 / (p99 - 128.0))
            if lo.size > 0:
                p01 = float(np.percentile(lo, 1))
                if p01 < 128.0:
                    cap = min(cap, 128.0 / (128.0 - p01))
    elif kind == "con":
        y = 0.114 * arr[:, 0] + 0.587 * arr[:, 1] + 0.299 * arr[:, 2]
        hi = y[y < 250.0]
        lo = y[y > 5.0]
        if hi.size > 0:
            p99 = float(np.percentile(hi, 99))
            if p99 > 128.0:
                cap = min(cap, 127.0 / (p99 - 128.0))
        if lo.size > 0:
            p01 = float(np.percentile(lo, 1))
            if p01 < 128.0:
                cap = min(cap, 128.0 / (128.0 - p01))
    return cap


def propose_sat_con(
    *,
    mode: str,
    chroma_mag_live: float,
    chroma_mag_ref: float | None,
    y_std_live: float,
    y_std_ref: float | None,
    current_sat: int,
    current_con: int,
    sat_caps: CtrlCaps,
    con_caps: CtrlCaps,
    band: float = SAT_CON_BAND,
    cur_bgr: np.ndarray | None = None,
) -> SatConProposal:
    """Propose the next ``saturation`` and ``contrast`` values.

    ``mode``:
      * ``"cc24"`` — write ``sat_caps.default`` / ``con_caps.default``
        (plan §3.1; CC24 patches are non-natural so ratio reasoning
        would over-fit).
      * ``"ref"`` — scale current values by ``ref/live`` ratio, then
        clip to ``default ± band`` (default 30%, plan §3.3).

    ``cur_bgr``: optional live frame used for per-channel clipping guard.
    When provided, any ratio > 1 (boost) is capped so that no currently-
    unclipped channel would be newly pushed to 255 or 0.

    Degenerate guards:
      * ``chroma_mag_live < 2.0``                  → keep ``current_sat``.
      * ``y_std_live < 0.02``                      → keep ``current_con``.
      * ``chroma_mag_ref / y_std_ref is None``     → keep current value.
    """
    if mode not in ("cc24", "ref"):
        raise ValueError(f"unknown mode: {mode!r}")

    if mode == "cc24":
        return SatConProposal(
            new_saturation=int(sat_caps.default),
            new_contrast=int(con_caps.default),
            sat_ratio=None,
            con_ratio=None,
            rationale="cc24_default",
        )

    sat_lo = max(sat_caps.minimum,
                 int(round(sat_caps.default * (1.0 - band))))
    sat_hi = min(sat_caps.maximum,
                 int(round(sat_caps.default * (1.0 + band))))
    con_lo = max(con_caps.minimum,
                 int(round(con_caps.default * (1.0 - band))))
    con_hi = min(con_caps.maximum,
                 int(round(con_caps.default * (1.0 + band))))

    sat_partial = False
    con_partial = False

    if chroma_mag_live < 2.0 or chroma_mag_ref is None or chroma_mag_ref <= 0:
        new_sat = current_sat
        sat_ratio: float | None = None
        sat_partial = True
    else:
        sat_ratio = float(chroma_mag_ref / chroma_mag_live)
        if cur_bgr is not None and sat_ratio > 1.0:
            sat_ratio = min(sat_ratio, _clip_headroom_cap(cur_bgr, "sat"))
        new_sat = int(round(current_sat * sat_ratio))
    new_sat = max(sat_lo, min(sat_hi, new_sat))

    if y_std_live < 0.02 or y_std_ref is None or y_std_ref <= 0:
        new_con = current_con
        con_ratio: float | None = None
        con_partial = True
    else:
        con_ratio = float(y_std_ref / y_std_live)
        if cur_bgr is not None and con_ratio > 1.0:
            con_ratio = min(con_ratio, _clip_headroom_cap(cur_bgr, "con"))
        new_con = int(round(current_con * con_ratio))
    new_con = max(con_lo, min(con_hi, new_con))

    rationale = "ref_match_partial" if (sat_partial or con_partial) else "ref_match"
    return SatConProposal(
        new_saturation=new_sat,
        new_contrast=new_con,
        sat_ratio=sat_ratio,
        con_ratio=con_ratio,
        rationale=rationale,
    )


__all__ = [
    "CLIP_THRESHOLD",
    "TARGET_Y_MEAN_CODE",
    "P99_TARGET_CODE",
    "OVEREXP_CLIP_RATIO",
    "SNR_FLOOR_DB",
    "SAT_CON_BAND",
    "GAIN_EXPOSURE_GATE",
    "CtrlCaps",
    "YStats",
    "ChromaStats",
    "SnrPatch",
    "SnrResult",
    "ExposureProposal",
    "GainProposal",
    "GainStepBResult",
    "SatConProposal",
    "RoiBox",
    "extract_roi_frame",
    "compute_y_stats",
    "compute_chroma_stats",
    "compute_snr_db",
    "propose_exposure",
    "propose_gain_step_a",
    "evaluate_gain_step_b",
    "propose_sat_con",
]

"""Stage orchestrator for the 4-stage hardware ISP calibration pipeline.

This module wires the pure proposers in :mod:`hw_stages` into a
sequential state machine while keeping ROS / OpenCV / hardware IO
behind a small protocol so the orchestrator stays trivially testable.

Layering (plan §1):

    camera_isp_calibrator.py  (GUI, ROS, key bindings, HUD)
                │     calls
                ▼
    hw_pipeline.run_full_pipeline()    ◄── this module
                │     calls
                ▼
    hw_stages.propose_*(...)           ◄── pure functions
                │     uses
                ▼
    solver.solve_kelvin_only(...)      ◄── chromaticity → kelvin

Per plan §1: each stage acquires a fresh frame *after* the previous
write has settled. The orchestrator never sleeps directly — settling
delays are the responsibility of the injected ``StageBridge``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import numpy as np

from .hw_stages import (
    ChromaStats,
    CtrlCaps,
    ExposureProposal,
    GainProposal,
    GainStepBResult,
    SatConProposal,
    SnrResult,
    YStats,
    compute_chroma_stats,
    compute_snr_db,
    compute_y_stats,
    evaluate_gain_step_b,
    propose_exposure,
    propose_gain_step_a,
    propose_sat_con,
)
from .solver import KelvinResult, solve_kelvin_only


# ---------------------------------------------------------------------------
# Verbose iteration prints — enabled when ``IBROBOT_HW_PIPELINE_VERBOSE=1``.
# Off by default so unit tests stay quiet; the calibrator turns it on so
# real-camera runs always emit the per-iteration trace needed to diagnose
# why a stage picked a particular branch. The env var is checked on every
# call so toggling it after import (the calibrator does this) takes effect
# immediately.
# ---------------------------------------------------------------------------


def _trace(stage: str, iters: int, msg: str) -> None:
    if os.environ.get("IBROBOT_HW_PIPELINE_VERBOSE", "0") == "1":
        print(f"[hw_pipeline] {stage} iter={iters} {msg}")


# ---------------------------------------------------------------------------
# Bridge protocol — calibrator implements this; tests pass a fake.
# ---------------------------------------------------------------------------

class StageBridge(Protocol):
    """Side-effect surface the orchestrator drives.

    Implementations must:
      * apply hardware writes synchronously (settling delay included),
      * always return a fresh frame from :meth:`grab_frame`.
    """

    def grab_frame(self) -> np.ndarray | None:
        """Return the most recent live BGR frame, or None on no signal."""

    def write_v4l2(self, params: Mapping[str, int]) -> None:
        """Apply ``params`` to the hardware. Block until settled."""

    def get_caps(self, key: str) -> CtrlCaps | None:
        """Return :class:`CtrlCaps` for ``key`` or None when unknown."""

    def get_current(self, key: str) -> int:
        """Return the live hardware value for ``key``."""


# ---------------------------------------------------------------------------
# Stage / pipeline result objects.
# ---------------------------------------------------------------------------

@dataclass
class StageRecord:
    """Per-stage outcome captured for the HUD and the save metadata."""

    name: str
    iters: int
    converged: bool
    proposal: object | None        # one of the four Proposal types.
    last_stats: object | None      # YStats / ChromaStats / KelvinResult / etc.
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.converged and self.proposal is not None


@dataclass
class PipelineResult:
    """End-to-end summary of one ``run_full_pipeline`` invocation."""

    stage1_exposure: StageRecord
    stage2_gain: StageRecord
    stage3_satcon: StageRecord
    stage4_kelvin: StageRecord
    final_params: dict[str, int]
    notes: list[str] = field(default_factory=list)

    @property
    def stages(self) -> tuple[StageRecord, ...]:
        return (
            self.stage1_exposure,
            self.stage2_gain,
            self.stage3_satcon,
            self.stage4_kelvin,
        )

    @property
    def all_passed(self) -> bool:
        return all(s.passed for s in self.stages)


# ---------------------------------------------------------------------------
# Iteration / safety constants — single source of truth, plan §2.
# ---------------------------------------------------------------------------

#: Stage 1 max iterations (plan §2.1).
EXPOSURE_MAX_ITERS = 3
#: Stage 2 hardware writes ≤ 2 (Step A then optional roll-back, plan §2.1).
GAIN_MAX_WRITES = 2
#: Stage 3 max iterations for ref mode (plan §3.3).
SATCON_MAX_ITERS = 2
#: Stage 4 max iterations (plan §2 Stage 4).
KELVIN_MAX_ITERS = 2


# ---------------------------------------------------------------------------
# Stage runners — small wrappers around the proposers that drive the bridge.
# ---------------------------------------------------------------------------

def run_stage_exposure(
    bridge: StageBridge,
    *,
    target_y_mean: float,
    exp_max: int,
    max_iters: int = EXPOSURE_MAX_ITERS,
) -> StageRecord:
    """Stage 1 — drive exposure toward ``target_y_mean`` with overexp guard."""
    caps = bridge.get_caps("exposure")
    if caps is None:
        return StageRecord(
            name="exposure", iters=0, converged=False,
            proposal=None, last_stats=None,
            note="exposure caps unavailable",
        )

    last_stats: YStats | None = None
    last_proposal: ExposureProposal | None = None
    iters = 0
    for iters in range(1, max_iters + 1):
        frame = bridge.grab_frame()
        if frame is None:
            return StageRecord(
                name="exposure", iters=iters - 1, converged=False,
                proposal=last_proposal, last_stats=last_stats,
                note="no live frame",
            )
        last_stats = compute_y_stats(frame)
        current_exp = bridge.get_current("exposure")
        last_proposal = propose_exposure(
            last_stats,
            current_exp=current_exp,
            exp_min=caps.minimum,
            exp_max=min(caps.maximum, exp_max),
        )
        _trace("exposure", iters,
               f"cur={current_exp} Y_mean={last_stats.y_mean_excl_clip:.1f} "
               f"Y_p99={last_stats.y_p99:.1f} clip={last_stats.clip_ratio:.3f} "
               f"-> new={last_proposal.new_exp} ({last_proposal.rationale})")
        if last_proposal.new_exp != current_exp:
            bridge.write_v4l2({"exposure": last_proposal.new_exp})
        if last_proposal.converged:
            break

    return StageRecord(
        name="exposure",
        iters=iters,
        converged=bool(last_proposal and last_proposal.converged),
        proposal=last_proposal,
        last_stats=last_stats,
        note=last_proposal.rationale if last_proposal else "no proposal",
    )


def run_stage_gain(
    bridge: StageBridge,
    *,
    target_y_mean: float,
    exp_max: int | None = None,
) -> StageRecord:
    """Stage 2 — Step A propose + optional Step B SNR fallback (plan §2).

    ``exp_max`` (raw V4L2 ticks) is forwarded to
    :func:`propose_gain_step_a` so that gain stays a no-op while the
    exposure rail is below ``GAIN_EXPOSURE_GATE``. Pass ``None`` to
    disable the gate (legacy behaviour).
    """
    gain_caps = bridge.get_caps("gain")
    bri_caps = bridge.get_caps("brightness")
    if gain_caps is None or bri_caps is None:
        return StageRecord(
            name="gain", iters=0, converged=False,
            proposal=None, last_stats=None,
            note="gain/brightness caps unavailable",
        )

    frame = bridge.grab_frame()
    if frame is None:
        return StageRecord(
            name="gain", iters=0, converged=False,
            proposal=None, last_stats=None,
            note="no live frame",
        )
    pre_stats = compute_y_stats(frame)
    current_gain = bridge.get_current("gain")
    current_exp = bridge.get_current("exposure")

    step_a = propose_gain_step_a(
        y_mean_live=pre_stats.y_mean_excl_clip,
        y_mean_target=target_y_mean,
        current_gain=current_gain,
        gain_caps=gain_caps,
        current_exp=current_exp,
        exp_max=exp_max,
    )
    _trace("gain", 1, f"stepA cur_gain={current_gain} cur_exp={current_exp} "
           f"exp_max={exp_max} Y_mean={pre_stats.y_mean_excl_clip:.1f} "
           f"target={target_y_mean:.1f} -> new={step_a.new_gain} "
           f"no_op={step_a.no_op}")

    if step_a.no_op:
        # Distinguish "exposure-gated" from "already bright enough" in
        # the trace note for diagnosis.
        gate_note = (
            "exposure_below_gate"
            if (exp_max and current_exp < exp_max * 0.80)
            else "already_bright_enough"
        )
        return StageRecord(
            name="gain", iters=1, converged=True,
            proposal=step_a, last_stats=pre_stats,
            note=gate_note,
        )

    bridge.write_v4l2({"gain": step_a.new_gain})

    frame_after = bridge.grab_frame()
    if frame_after is None:
        return StageRecord(
            name="gain", iters=1, converged=False,
            proposal=step_a, last_stats=pre_stats,
            note="no frame after step A",
        )
    post_stats = compute_y_stats(frame_after)
    snr = compute_snr_db(frame_after)

    step_b = evaluate_gain_step_b(
        y_mean_after_step_a=post_stats.y_mean_excl_clip,
        y_mean_target=target_y_mean,
        gain_before_step_a=current_gain,
        gain_after_step_a=step_a.new_gain,
        snr_result=snr,
        brightness_caps=bri_caps,
        gain_caps=gain_caps,
    )
    _trace("gain", 2, f"stepB Y_mean_after={post_stats.y_mean_excl_clip:.1f} "
           f"snr_db={snr.snr_db} -> gain={step_b.final_gain} "
           f"brightness={step_b.final_brightness} note={step_b.note}")

    write: dict[str, int] = {}
    if step_b.final_gain != step_a.new_gain:
        write["gain"] = step_b.final_gain
    if step_b.final_brightness is not None:
        write["brightness"] = step_b.final_brightness
    if write:
        bridge.write_v4l2(write)

    converged = step_b.snr_ok or not step_b.fallback_active
    return StageRecord(
        name="gain",
        iters=1 + (1 if step_b.fallback_active else 0),
        converged=converged,
        proposal=step_b,
        last_stats=post_stats,
        note=step_b.note,
    )


def run_stage_satcon(
    bridge: StageBridge,
    *,
    mode: str,
    chroma_mag_ref: float | None,
    y_std_ref: float | None,
    max_iters: int = SATCON_MAX_ITERS,
) -> StageRecord:
    """Stage 3 — write driver default in ``cc24`` mode; ratio-match in ``ref``."""
    if mode not in ("ref", "cc24"):
        raise ValueError(f"unknown mode: {mode!r}")
    sat_caps = bridge.get_caps("saturation")
    con_caps = bridge.get_caps("contrast")
    if sat_caps is None or con_caps is None:
        return StageRecord(
            name="satcon", iters=0, converged=False,
            proposal=None, last_stats=None,
            note="saturation/contrast caps unavailable",
        )

    if mode == "cc24":
        # One-shot write: driver defaults (plan §3.1).
        proposal = propose_sat_con(
            mode="cc24",
            chroma_mag_live=0.0, chroma_mag_ref=None,
            y_std_live=0.0, y_std_ref=None,
            current_sat=bridge.get_current("saturation"),
            current_con=bridge.get_current("contrast"),
            sat_caps=sat_caps, con_caps=con_caps,
        )
        bridge.write_v4l2({
            "saturation": proposal.new_saturation,
            "contrast": proposal.new_contrast,
        })
        return StageRecord(
            name="satcon", iters=1, converged=True,
            proposal=proposal, last_stats=None,
            note="cc24_default",
        )

    # ref mode: iterate until proposals converge or budget exhausted.
    last_proposal: SatConProposal | None = None
    last_chroma: ChromaStats | None = None
    last_y: YStats | None = None
    iters = 0
    for iters in range(1, max_iters + 1):
        frame = bridge.grab_frame()
        if frame is None:
            return StageRecord(
                name="satcon", iters=iters - 1, converged=False,
                proposal=last_proposal, last_stats=last_chroma,
                note="no live frame",
            )
        last_chroma = compute_chroma_stats(frame)
        last_y = compute_y_stats(frame)
        current_sat = bridge.get_current("saturation")
        current_con = bridge.get_current("contrast")
        last_proposal = propose_sat_con(
            mode="ref",
            chroma_mag_live=last_chroma.chroma_mag_median,
            chroma_mag_ref=chroma_mag_ref,
            y_std_live=last_y.y_std_excl_clip,
            y_std_ref=y_std_ref,
            current_sat=current_sat,
            current_con=current_con,
            sat_caps=sat_caps,
            con_caps=con_caps,
            cur_bgr=frame,
        )
        _trace("satcon", iters, f"chroma_live={last_chroma.chroma_mag_median:.2f} "
               f"chroma_ref={chroma_mag_ref} y_std_live={last_y.y_std_excl_clip:.4f} "
               f"y_std_ref={y_std_ref} -> sat={last_proposal.new_saturation} "
               f"con={last_proposal.new_contrast} ({last_proposal.rationale})")
        write: dict[str, int] = {}
        if last_proposal.new_saturation != current_sat:
            write["saturation"] = last_proposal.new_saturation
        if last_proposal.new_contrast != current_con:
            write["contrast"] = last_proposal.new_contrast
        if not write:
            break
        bridge.write_v4l2(write)

    return StageRecord(
        name="satcon",
        iters=iters,
        converged=bool(last_proposal),
        proposal=last_proposal,
        last_stats=last_chroma,
        note=last_proposal.rationale if last_proposal else "no proposal",
    )


def run_stage_kelvin(
    bridge: StageBridge,
    *,
    ref_bgr: np.ndarray | None,
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
    max_iters: int = KELVIN_MAX_ITERS,
) -> StageRecord:
    """Stage 4 — delegate to :func:`solve_kelvin_only`.

    ``ref_bgr=None`` (cc24 mode) is handled separately by the GUI 24-patch
    wizard; this orchestrator's stage 4 is for ref mode only.
    """
    if ref_bgr is None:
        return StageRecord(
            name="kelvin", iters=0, converged=False,
            proposal=None, last_stats=None,
            note="ref_bgr=None — GUI must use cc24 wizard for kelvin",
        )

    last_result: KelvinResult | None = None
    iters = 0
    for iters in range(1, max_iters + 1):
        frame = bridge.grab_frame()
        if frame is None:
            return StageRecord(
                name="kelvin", iters=iters - 1, converged=False,
                proposal=last_result, last_stats=last_result,
                note="no live frame",
            )
        # Resize ref to match live frame on the fly so the saturated mask
        # is per-pixel coherent (cheap; already done elsewhere too).
        if frame.shape != ref_bgr.shape:
            import cv2  # lazy: keep this module import-light for tests.
            ref_resized = cv2.resize(
                ref_bgr,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        else:
            ref_resized = ref_bgr
        current_k = bridge.get_current("white_balance")
        last_result = solve_kelvin_only(
            ref_resized, frame, current_kelvin=current_k,
            device_caps=device_caps,
        )
        _trace("kelvin", iters, f"cur={current_k} -> new={last_result.new_kelvin} "
               f"delta={last_result.delta:.1f} ({last_result.rationale})")
        if last_result.new_kelvin != current_k:
            bridge.write_v4l2({"white_balance": last_result.new_kelvin})
        if last_result.converged:
            break

    return StageRecord(
        name="kelvin",
        iters=iters,
        converged=bool(last_result and last_result.converged),
        proposal=last_result,
        last_stats=last_result,
        note=last_result.rationale if last_result else "no result",
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------

def run_full_pipeline(
    bridge: StageBridge,
    *,
    mode: str,
    target_y_mean: float,
    exp_max: int,
    chroma_mag_ref: float | None = None,
    y_std_ref: float | None = None,
    ref_bgr: np.ndarray | None = None,
    device_caps: Mapping[str, Mapping[str, int]] | None = None,
) -> PipelineResult:
    """Run all four hardware stages in order.

    Stage failures do not abort the pipeline (plan §8.8) — the result
    records the per-stage outcome and the GUI surfaces the full picture
    via the HUD.

    ``mode``:
      * ``"ref"``  — Stage 3 reads ``chroma_mag_ref`` / ``y_std_ref`` and
                     Stage 4 reads ``ref_bgr``.
      * ``"cc24"`` — Stage 3 writes driver defaults; Stage 4 is left to
                     the GUI 24-patch wizard (this orchestrator returns a
                     placeholder StageRecord noting the delegation).
    """
    if mode not in ("ref", "cc24"):
        raise ValueError(f"unknown mode: {mode!r}")

    notes: list[str] = []
    s1 = run_stage_exposure(bridge, target_y_mean=target_y_mean, exp_max=exp_max)
    s2 = run_stage_gain(bridge, target_y_mean=target_y_mean, exp_max=exp_max)
    s3 = run_stage_satcon(
        bridge, mode=mode,
        chroma_mag_ref=chroma_mag_ref,
        y_std_ref=y_std_ref,
    )
    if mode == "cc24":
        s4 = StageRecord(
            name="kelvin", iters=0, converged=False,
            proposal=None, last_stats=None,
            note="cc24_wizard_delegated",
        )
        notes.append("Stage 4 delegated to 24-patch wizard (cc24 mode)")
    else:
        s4 = run_stage_kelvin(
            bridge, ref_bgr=ref_bgr, device_caps=device_caps,
        )

    final_params: dict[str, int] = {
        "exposure": bridge.get_current("exposure"),
        "gain": bridge.get_current("gain"),
        "brightness": bridge.get_current("brightness"),
        "saturation": bridge.get_current("saturation"),
        "contrast": bridge.get_current("contrast"),
        "white_balance": bridge.get_current("white_balance"),
    }

    return PipelineResult(
        stage1_exposure=s1,
        stage2_gain=s2,
        stage3_satcon=s3,
        stage4_kelvin=s4,
        final_params=final_params,
        notes=notes,
    )


__all__ = [
    "StageBridge",
    "StageRecord",
    "PipelineResult",
    "EXPOSURE_MAX_ITERS",
    "GAIN_MAX_WRITES",
    "SATCON_MAX_ITERS",
    "KELVIN_MAX_ITERS",
    "run_stage_exposure",
    "run_stage_gain",
    "run_stage_satcon",
    "run_stage_kelvin",
    "run_full_pipeline",
]

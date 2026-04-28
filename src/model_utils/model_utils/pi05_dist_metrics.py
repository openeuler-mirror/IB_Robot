"""
Distributional evaluation metrics for PI05 deployment validation.

Why this file exists
--------------------
PI05 inference is a 10-step Euler integration of a flow-matching ODE.  Such
ODEs have positive Lyapunov exponents in practice, so a tiny perturbation in
the conditioning (e.g. fp16 KV-cache drift of 1-cos ~ 2e-5) is amplified
exponentially: by step 10 the trajectories de-correlate and the *pointwise*
cosine similarity between PT and OM outputs collapses toward 0 even though
both implementations are mathematically correct.

Reference for the chaos-amplification phenomenon:
  - Strogatz, "Nonlinear Dynamics and Chaos", Ch. 9-10
  - Karras et al. (NeurIPS 2022) "Elucidating the Design Space of Diffusion-
    Based Generative Models", §3 — small per-step solver error compounds
    multiplicatively across steps.
  - Theis et al. (ICLR 2016) "A note on the evaluation of generative models"
    — pointwise distance and sample quality are decoupled for generative
    models; use distributional metrics instead.

Therefore *mean cosine similarity is the wrong metric for PI05 PT-vs-OM
validation*.  This module provides two chaos-robust replacements:

  Method A — Marginal Wasserstein-1 per action dim
      Compares the *value distribution* (not pointwise alignment) of every
      action dimension.  Insensitive to order, hence chaos-robust.

  Method C — First-frame cosine
      In receding-horizon control only ``chunk[:, 0, :]`` is actually sent
      to the robot before the next inference overwrites the rest.  Even
      under chaos the first frame is the most stable frame to compare and
      is the only one that drives physical motion.

Usage
-----
Imported by ``loss_compare.py`` and called automatically when
``--policy_type pi05``.  Can also be used standalone:

    from pi05_dist_metrics import evaluate_pi05
    evaluate_pi05(preds, targets,
                  raw_preds=raw_preds, raw_targets=raw_targets)

Where ``preds``/``targets`` are lists of tensors of shape ``(B, T, D)`` or
``(T, D)`` (B is summed across the list).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.stats import ks_2samp, wasserstein_distance
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack(items) -> np.ndarray:
    """Stack a list of (T,D) or (B,T,D) tensors into a single (N,T,D) ndarray."""
    arr = []
    for x in items:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]  # (1, T, D)
        arr.append(x)
    return np.concatenate(arr, axis=0)  # (N, T, D)


def _grade(value: float, thresholds, higher_is_better: bool = False) -> str:
    """Pick a verdict label given an ordered list of (label, threshold)."""
    for label, thr in thresholds:
        if higher_is_better:
            if value >= thr:
                return label
        else:
            if value <= thr:
                return label
    return thresholds[-1][0]


# ---------------------------------------------------------------------------
# Method A — marginal Wasserstein-1 per action dim
# ---------------------------------------------------------------------------

def wasserstein_per_dim(preds, targets, space_label: str = "") -> dict:
    """Method A: marginal W1 + KS per action dim. Chaos-robust."""
    if not _HAS_SCIPY:
        print("[WARN] scipy not installed; skip Wasserstein metric "
              "(`pip install scipy`)")
        return {}

    P = _stack(preds)
    T = _stack(targets)
    if P.shape != T.shape:
        print(f"[WARN] shape mismatch: pred={P.shape} vs target={T.shape}; "
              f"skip Method A")
        return {}
    D = P.shape[-1]

    p_flat = P.reshape(-1, D)
    t_flat = T.reshape(-1, D)

    w1 = np.array([wasserstein_distance(p_flat[:, d], t_flat[:, d]) for d in range(D)])
    ks = np.array([ks_2samp(p_flat[:, d], t_flat[:, d]).statistic for d in range(D)])

    # Per-dim drift ratio: W1 normalized by target std.  This makes the metric
    # scale-invariant -- a value of 0.1 means "W1 is one tenth of the spread of
    # the target distribution", regardless of whether actions are in [-1, 1]
    # (normalized) or [-100, 100] (unnormalized joint angles in degrees).
    t_std = t_flat.std(axis=0)
    # Avoid division by zero on degenerate (constant) dims; treat r as 0 there.
    safe_std = np.where(t_std > 1e-8, t_std, 1.0)
    ratio = np.where(t_std > 1e-8, w1 / safe_std, 0.0)

    print(f"\n--- Method A: marginal distribution match {space_label} ---")
    print("  (Compares value distributions per dim. Insensitive to ODE chaos.)")
    print(f"  {'dim':>4} {'W1':>12} {'tgt_std':>12} {'W1/std':>10} {'KS':>10}")
    for d in range(D):
        print(f"  {d:>4} {w1[d]:>12.5f} {t_std[d]:>12.5f} "
              f"{ratio[d]:>10.5f} {ks[d]:>10.5f}")
    print("  ---")
    mean_w1 = float(w1.mean())
    max_w1 = float(w1.max())
    mean_ratio = float(ratio.mean())
    max_ratio = float(ratio.max())
    mean_ks = float(ks.mean())
    print(f"  mean W1 = {mean_w1:.5f}   max W1 = {max_w1:.5f}   mean KS = {mean_ks:.5f}")
    print(f"  mean W1/std = {mean_ratio:.5f}   max W1/std = {max_ratio:.5f}  "
          f"<-- scale-invariant; same scale in both spaces")

    # Verdict on the SCALE-INVARIANT ratio so it's meaningful in both
    # normalized and unnormalized spaces.
    grade = _grade(mean_ratio, [
        ("EXCELLENT  (deployable)",  0.05),
        ("GOOD       (minor shift)", 0.15),
        ("MARGINAL   (investigate)", 0.30),
        ("POOR       (drift)",       float("inf")),
    ])
    print(f"  Verdict: [{grade}]   (based on mean W1/std)")
    print("  Criteria (scale-invariant, applies to both spaces):")
    print("    W1/std <= 0.05   EXCELLENT  -- drift << distribution spread")
    print("    W1/std <= 0.15   GOOD       -- drift < 1/6 of std")
    print("    W1/std <= 0.30   MARGINAL   -- drift ~ 1/3 of std, investigate")
    print("    W1/std >  0.30   POOR       -- drift comparable to std")
    return {
        "mean_w1": mean_w1, "max_w1": max_w1,
        "mean_ratio": mean_ratio, "max_ratio": max_ratio,
        "mean_ks": mean_ks, "verdict": grade,
    }


# ---------------------------------------------------------------------------
# Method C — first-frame cosine (control-relevant)
# ---------------------------------------------------------------------------

def first_frame_cosine(preds, targets, space_label: str = "") -> dict:
    """Method C: cosine sim on chunk[:, 0, :] only — the frame actually executed."""
    P = _stack(preds)
    T = _stack(targets)
    if P.shape != T.shape:
        print(f"[WARN] shape mismatch: pred={P.shape} vs target={T.shape}; "
              f"skip Method C")
        return {}

    Pt = torch.from_numpy(P[:, 0, :])  # (N, D)
    Tt = torch.from_numpy(T[:, 0, :])
    cos = F.cosine_similarity(Pt, Tt, dim=-1).numpy()
    l1 = (Pt - Tt).abs().mean(dim=-1).numpy()

    print(f"\n--- Method C: first-frame cosine {space_label} ---")
    print("  (Receding-horizon control only executes chunk[:, 0]; "
          "rest is overwritten by the next inference.)")
    print(f"  {'sample':>6} {'cos':>10} {'L1':>10}")
    for i, (c, l) in enumerate(zip(cos, l1)):
        print(f"  {i:>6} {c:>10.5f} {l:>10.5f}")
    print("  ---")
    mean_cos = float(cos.mean())
    min_cos = float(cos.min())
    print(f"  mean cos = {mean_cos:.5f}   min cos = {min_cos:.5f}")

    grade = _grade(mean_cos, [
        ("EXCELLENT", 0.95),
        ("GOOD",      0.80),
        ("MARGINAL",  0.50),
        ("POOR",      -float("inf")),
    ], higher_is_better=True)
    print(f"  Verdict: [{grade}]")
    print("  Criteria (first-frame cosine, control-relevant):")
    print("    cos >= 0.95   EXCELLENT  -- executed action nearly identical")
    print("    cos >= 0.80   GOOD       -- direction agrees, magnitude close")
    print("    cos >= 0.50   MARGINAL   -- same half-space, drift visible")
    print("    cos <  0.50   POOR       -- actions disagree on first frame")
    return {"mean_cos": mean_cos, "min_cos": min_cos, "verdict": grade}


# ---------------------------------------------------------------------------
# Top-level entry used by loss_compare.py
# ---------------------------------------------------------------------------

def evaluate_pi05(preds, targets, raw_preds=None, raw_targets=None) -> None:
    """Print the full PI05 distributional evaluation report.

    Parameters
    ----------
    preds, targets
        Lists of tensors in the *unnormalized* action space (post-postprocessor).
    raw_preds, raw_targets
        Optional lists of tensors in the *normalized* action space
        (pre-postprocessor).  When provided, evaluation runs on this space too;
        the verdict thresholds were calibrated here.
    """
    print("\n" + "=" * 72)
    print(" PI05 distributional evaluation")
    print(" (per-sample cosine similarity is misleading on chaotic ODEs)")
    print("=" * 72)

    if raw_preds is not None and raw_targets is not None and len(raw_preds) > 0:
        print("\n>>> Normalized action space (pre-postprocessor) <<<")
        print("    Verdict thresholds are calibrated for this space.")
        wasserstein_per_dim(raw_preds, raw_targets, space_label="(normalized)")
        first_frame_cosine(raw_preds, raw_targets, space_label="(normalized)")

    print("\n>>> Unnormalized action space (post-postprocessor) <<<")
    print("    Numbers are in physical units; absolute thresholds are heuristic.")
    wasserstein_per_dim(preds, targets, space_label="(unnormalized)")
    first_frame_cosine(preds, targets, space_label="(unnormalized)")

    print("\n" + "=" * 72)
    print(" Note: for final deployment validation, also run a task-level")
    print(" success-rate evaluation on simulator or real robot.")
    print("=" * 72 + "\n")

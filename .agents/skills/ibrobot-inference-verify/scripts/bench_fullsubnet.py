#!/usr/bin/env python3
"""Micro-benchmark FullSubNet stateful torch executor fb/sb latency.

Complements the ROS E2E test with per-hop numbers (the node's
[FullSubNetTiming] INFO logs are suppressed by default python logging).

Usage: python3 bench_fullsubnet.py  # benchmarks cuda then cpu
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src/voice_asr_service"))

from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful_executor import (  # noqa: E402
    FB_FRAME_SHAPE,
    SB_FRAME_SHAPE,
)
from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful_torch import (  # noqa: E402
    StatefulTorchFullSubNetExecutor,
)

CKPT = REPO_ROOT / "models/fullsubnet/assets/cum_fullsubnet_best_model_218epochs.tar"
MANIFEST = CKPT.with_name(CKPT.name.replace(".tar", ".manifest.json"))


def main() -> None:
    rng = np.random.default_rng(0)
    fb = (rng.standard_normal(FB_FRAME_SHAPE) * 0.1).astype(np.float32)
    sb = (rng.standard_normal(SB_FRAME_SHAPE) * 0.1).astype(np.float32)
    for device in ("cuda", "cpu"):
        ex = StatefulTorchFullSubNetExecutor(str(CKPT), str(MANIFEST), device=device, timing_enabled=True)
        for _ in range(20):
            ex.run_fb(fb)
            ex.run_sb(sb)
        for _ in range(60):
            ex.run_fb(fb)
            ex.run_sb(sb)
        t = ex.last_timing_ms
        print(f"{device}: fb={t['fb_infer_ms']:.3f}ms sb={t['sb_infer_ms']:.3f}ms (per 512-sample hop, 32ms audio)")
        ex.close()


if __name__ == "__main__":
    main()

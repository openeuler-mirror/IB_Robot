#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Score one fixed grasp set on either device and compare the ranking it induces.

The pose-parity check compares two *independent* draws, so it can say the two
devices put grasps in the same places but nothing about ranking: there is no
"same grasp" to follow between runs. Ranking is what the deployment actually
depends on -- the reach filter walks candidates in confidence order, and on the
2026-09-14 X2 bottle recordings the arm-feasible ones sat at confidence ranks
8-36 and 45-99 -- and a measured confidence shift between devices (W1 up to
0.065 against a same-device band of 0.042) is exactly the kind of thing that
reorders that range while leaving translation agreement untouched.

So this drives the discriminator alone, on a fixed cloud and a fixed set of
poses, which makes the two devices' confidences *paired* and the comparison a
rank question rather than a distribution question.

    DEPLOYMENT=torch_cuda ... score --cloud c.npz --poses p.npz out_cuda.npz
    DEPLOYMENT=torch_npu  ... score --cloud c.npz --poses p.npz out_npu.npz
    ... compare out_cuda.npz out_npu.npz

Both devices must be given the same cloud and the same poses; their digests are
recorded and the comparison refuses runs that disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE") or Path(__file__).resolve().parents[3])
for package in sorted(p.name for p in (WORKSPACE / "src").iterdir() if p.is_dir()):
    sys.path.insert(0, str(WORKSPACE / "src" / package))

DEPLOYMENT = os.environ.get("DEPLOYMENT", "torch_npu")
BUNDLE = Path(os.environ.get("BUNDLE", WORKSPACE / "models/graspgenx_omnihand_left_x2"))
# The ranks the reach filter was observed to depend on, from the X2 run reports.
REACH_BANDS = ((8, 36), (45, 99))


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def _array(path: Path, keys: tuple[str, ...]) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    raise SystemExit(f"{path} has none of {keys} (found {sorted(data.keys())})")


def score(cloud_path: Path, poses_path: Path) -> dict:
    """Confidences the discriminator gives these poses for this cloud, on this device."""
    if DEPLOYMENT == "torch_npu":
        import torch_npu  # noqa: F401 -- registers the device before anything places a tensor

    import torch

    from inference_manifest.loader import load_inference_manifest
    from perception_service.torch_model_loaders import load_graspgenx

    cloud = np.ascontiguousarray(_array(cloud_path, ("points", "cloud", "object_points")), dtype=np.float32)
    poses = np.ascontiguousarray(_array(poses_path, ("pose", "poses", "grasp_poses")), dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise SystemExit(f"cloud must be [N, 3], got {cloud.shape}")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise SystemExit(f"poses must be [K, 4, 4], got {poses.shape}")

    validated = load_inference_manifest(BUNDLE, DEPLOYMENT)
    artifacts = BUNDLE / "artifacts" / "ascend_310p"
    resolved = dict(validated.resolved_artifacts or {})
    context = SimpleNamespace(
        validated_manifest=SimpleNamespace(bundle_root=BUNDLE, resolved_artifacts=resolved),
        deployment=SimpleNamespace(device=validated.deployment.device),
    )
    if validated.deployment.device == "npu" and not resolved:
        resolved.update(
            diffusion_loop=artifacts / "diffusion_loop.om",
            diffusion_loop_metadata=artifacts / "diffusion_loop.json",
        )
    sampler = load_graspgenx(context).sampler
    # Only now: importing graspgenx runs a hook that clones gripper descriptions and
    # checkpoints from Hugging Face unless both asset roots exist, and load_graspgenx
    # is what points them into the bundle before its own first import.
    from graspgenx.dataset.dataset import collate

    # The sampler centres the cloud and restores the source frame on the way out, so
    # the poses have to be centred the same way before the discriminator sees them.
    center = cloud.mean(axis=0)
    centred_cloud = torch.from_numpy(cloud - center)
    centred_poses = poses.copy()
    centred_poses[:, :3, 3] -= center

    # Not from next(model.parameters()): on the npu path assembly leaves the PTV3
    # encoders resident on the host behind an EncoderDeviceBridge, so the first
    # parameter can legitimately be a CPU one and placing the inputs there puts the
    # cloud on cpu while the rest of the model is on npu.
    device = torch.device("npu:0" if validated.deployment.device == "npu" else "cuda")
    data = {"task": "pick"}
    colour = torch.zeros_like(centred_cloud)
    data["inputs"] = torch.cat([centred_cloud, colour], dim=-1).float().to(device)
    data["points"] = centred_cloud.float().to(device)
    data = sampler.load_gripper_input(data)
    batch = collate([data])
    batch["grasps_pred"] = [torch.from_numpy(centred_poses).float().to(device)]
    batch["grasp_key"] = "grasps_pred"

    with torch.inference_mode():
        outputs, _, _ = sampler.model.grasp_discriminator.infer(batch)
    confidence = outputs["grasp_confidence"][0][:, 0].float().cpu().numpy()
    if len(confidence) != len(poses):
        raise SystemExit(f"scored {len(confidence)} of {len(poses)} poses")
    return {
        "deployment": DEPLOYMENT,
        "confidence": confidence.astype(np.float64),
        "cloud_sha256": digest(cloud),
        "poses_sha256": digest(poses),
    }


def compare(left_path: Path, right_path: Path) -> int:
    left, right = (np.load(p, allow_pickle=False) for p in (left_path, right_path))
    for field in ("cloud_sha256", "poses_sha256"):
        values = [str(run[field]) for run in (left, right)]
        if values[0] != values[1]:
            raise SystemExit(f"these runs scored different inputs ({field} differs):\n  " + "\n  ".join(values))
    a, b = left["confidence"], right["confidence"]
    if len(a) != len(b):
        raise SystemExit(f"different pose counts: {len(a)} and {len(b)}")

    rank_a, rank_b = (np.empty(len(a), dtype=np.int64) for _ in range(2))
    rank_a[np.argsort(-a)] = np.arange(len(a))
    rank_b[np.argsort(-b)] = np.arange(len(b))
    # Spearman without scipy: Pearson on the ranks.
    spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
    shift = np.abs(rank_a - rank_b)

    print(f"{left_path.name} vs {right_path.name}: {len(a)} paired poses, one cloud\n")
    print(f"confidence   {str(left['deployment']):10s} [{a.min():.4f}, {a.max():.4f}] median {np.median(a):.4f}")
    print(f"             {str(right['deployment']):10s} [{b.min():.4f}, {b.max():.4f}] median {np.median(b):.4f}")
    print(f"per-pose     max |delta| {np.abs(a - b).max():.4f}   mean |delta| {np.abs(a - b).mean():.4f}")
    print(f"\nSpearman rank correlation   {spearman:.5f}")
    print(
        f"rank displacement           median {np.median(shift):.0f}  p90 {np.percentile(shift, 90):.0f}  max {shift.max()}"
    )

    print("\ntop-k overlap (what the reach filter walks)")
    for k in (10, 36, 50, 99, 200):
        if k > len(a):
            continue
        top_a = set(np.argsort(-a)[:k].tolist())
        top_b = set(np.argsort(-b)[:k].tolist())
        print(f"  top-{k:<4} {len(top_a & top_b)}/{k}")

    print("\nreach bands observed on the X2 recordings")
    for low, high in REACH_BANDS:
        if high > len(a):
            continue
        band = set(np.argsort(-a)[low - 1 : high].tolist())
        other = set(np.argsort(-b)[low - 1 : high].tolist())
        print(f"  ranks {low}-{high:<4} {len(band & other)}/{len(band)} of this band is still in it")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("score")
    one.add_argument("output", type=Path)
    one.add_argument("--cloud", type=Path, required=True)
    one.add_argument("--poses", type=Path, required=True)
    two = sub.add_parser("compare")
    two.add_argument("left", type=Path)
    two.add_argument("right", type=Path)
    arguments = parser.parse_args()

    if arguments.action == "compare":
        return compare(arguments.left, arguments.right)

    produced = score(arguments.cloud, arguments.poses)
    np.savez(arguments.output, **produced)
    confidence = produced["confidence"]
    print(f"deployment   {produced['deployment']}")
    print(f"cloud sha    {produced['cloud_sha256']}")
    print(f"poses sha    {produced['poses_sha256']}")
    print(f"scored       {len(confidence)} poses")
    print(f"confidence   [{confidence.min():.4f}, {confidence.max():.4f}] median {np.median(confidence):.4f}")
    print(f"wrote        {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

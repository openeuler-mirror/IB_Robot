"""GraspGenX point-cloud preprocessing and grasp decoding.

GraspGenX is NVlabs' cross-gripper successor to GraspGen: one gripper-independent
diffusion model conditioned on a gripper description, so the same checkpoint serves the
X2 OmniHand and any other described gripper. Its contract differs from GraspGen's in one
way that matters here: the official sampler centres the cloud itself and returns poses
in the frame the points arrived in, so this adapter must not centre or kappa-scale.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inference_service.unified_runtime import ModelResult
from model_utils.graspgen_contract import (
    GRASPGEN_CONFIDENCE_SEMANTIC,
    GRASPGEN_POINT_CLOUD_SEMANTIC,
    GRASPGEN_POSE_SEMANTIC,
)

from .graspgen_adapter import GraspCandidatePose
from .graspgen_geometry import positive_float, positive_int, remove_point_cloud_outliers
from .perception_adapter import AdapterIdentity, PerceptionAdapter

GRASPGENX_PREPROCESSING = "object-points-finite-outlier-filtered-resampled-source-frame-v1"
GRASPGENX_POSTPROCESSING = "grasp-pose-matrix-orthonormal-confidence-sorted-v1"
# How many grasps the diffusion sampler draws for one request. It is an input rather
# than a load-time constant because a caller asking for more candidates than the default
# draw must get a larger draw, not a silently truncated one.
GRASPGENX_SAMPLE_COUNT_SEMANTIC = "observation.grasp_sample_count"

# The official sampler falls back to the unfiltered cloud when outlier removal leaves
# fewer points than this, and treats a smaller cloud as ungraspable.
GRASPGENX_MIN_POINTS = 10
#: Upper bound on one diffusion draw. The sample count is a GPU batch size, so it
#: must not be set by an unbounded request field on a shared model service.
GRASPGENX_MAX_SAMPLE_COUNT = 1000
_ORTHONORMAL_TOLERANCE = 2e-3


@dataclass(frozen=True)
class GraspGenXConfig:
    """Bundle constants read from ``assets/adapter.json``.

    The defaults draw 400 grasps and return all of them, which is where the
    X2 pipeline's own sweep over 200, 400 and 800 settled and what the Ascend
    310P loop graph is compiled for. Returning fewer than the draw only
    discards candidates already paid for, and the reach filter consumes them:
    on the 2026-09-14 X2 bottle recordings the arm-feasible candidates sat at
    confidence ranks 8-36 and 45-99, so a cut below 100 removed every
    reachable grasp.
    """

    gripper_name: str
    object_points: int
    num_grasps: int = 400
    topk: int = 400
    outlier_threshold: float = 0.014
    outlier_neighbors: int = 20

    @classmethod
    def from_assets(cls, assets: Mapping[str, object]) -> GraspGenXConfig:
        gripper_name = assets.get("gripper_name")
        if not isinstance(gripper_name, str) or not gripper_name:
            raise ValueError("GraspGenX adapter.json must name the gripper")
        if "object_points" not in assets:
            raise ValueError(
                "GraspGenX adapter.json must declare object_points (the checkpoint's "
                "data.num_points); repackage the bundle with package_graspgenx_torch_bundle"
            )
        config = cls(
            gripper_name=gripper_name,
            object_points=positive_int(assets.get("object_points"), "object_points"),
            num_grasps=positive_int(assets.get("num_grasps", 400), "num_grasps"),
            topk=positive_int(assets.get("topk", 400), "topk"),
            outlier_threshold=positive_float(assets.get("outlier_threshold", 0.014), "outlier_threshold"),
            outlier_neighbors=positive_int(assets.get("outlier_neighbors", 20), "outlier_neighbors"),
        )
        if config.object_points < GRASPGENX_MIN_POINTS:
            raise ValueError(f"GraspGenX object_points {config.object_points} is below {GRASPGENX_MIN_POINTS}")
        if config.num_grasps > GRASPGENX_MAX_SAMPLE_COUNT:
            raise ValueError(f"GraspGenX num_grasps {config.num_grasps} exceeds {GRASPGENX_MAX_SAMPLE_COUNT}")
        if config.topk > config.num_grasps:
            raise ValueError(f"GraspGenX topk {config.topk} exceeds the sampled count {config.num_grasps}")
        return config


class GraspGenXAdapter(PerceptionAdapter):
    identity = AdapterIdentity(
        model_type="graspgenx",
        preprocessing=GRASPGENX_PREPROCESSING,
        postprocessing=GRASPGENX_POSTPROCESSING,
        supported_deployments=frozenset({"torch_cuda", "torch_npu"}),
        operation="generate_grasps",
    )

    def __init__(self, config: GraspGenXConfig, *, compiled_sample_count: int | None = None) -> None:
        self.config = config
        # Set when the deployment ships an offline diffusion loop: that graph is
        # compiled for one row count, so the draw cannot follow a request.
        self.compiled_sample_count = compiled_sample_count

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None) -> GraspGenXAdapter:
        del model
        assets = json.loads((Path(bundle_root) / "assets" / "adapter.json").read_text(encoding="utf-8"))
        expected = {
            "interface": "tensor_model",
            "model_type": cls.identity.model_type,
            "operation": cls.identity.operation,
            "preprocessing": cls.identity.preprocessing,
            "postprocessing": cls.identity.postprocessing,
        }
        mismatched = {name: assets.get(name) for name, value in expected.items() if assets.get(name) != value}
        if mismatched:
            raise ValueError(f"GraspGenX adapter identity mismatch: expected {expected}, got {mismatched}")
        config = GraspGenXConfig.from_assets(assets)
        # A deployment that carries a compiled diffusion loop can only run the
        # row count the graph was built for; see the draw cap in prepare.
        artifacts = getattr(deployment, "artifacts", None) or {}
        return cls(config, compiled_sample_count=config.num_grasps if "diffusion_loop" in artifacts else None)

    def preprocess(self, points_xyz: np.ndarray) -> dict[str, np.ndarray]:
        prepared = self.prepare(points_xyz)
        if prepared is None:
            raise ValueError(f"GraspGenX needs at least {GRASPGENX_MIN_POINTS} object points")
        return prepared

    def filter_points(self, points_xyz: np.ndarray) -> np.ndarray | None:
        """The cloud the sampler should see, before it is resampled to the budget.

        Non-finite rows go first, then outliers, with the sampler's own rule (mean L1
        distance to the 20 nearest neighbours under 1.4 cm) computed on the host so a
        dense cloud does not allocate the official implementation's N x N GPU buffer.
        Points stay in the source frame: the sampler centres them and puts the poses
        back itself. ``None`` means too few points to grasp.

        Separate from the resampling in ``prepare`` because this part is deterministic
        and is what decides which points exist at all.
        """
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"GraspGenX object points must have shape [N, 3], got {points.shape}")
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) == 0:
            raise ValueError("GraspGenX object point cloud contains no finite points")
        if len(finite) < GRASPGENX_MIN_POINTS:
            return None
        kept, _ = remove_point_cloud_outliers(
            finite, threshold=self.config.outlier_threshold, k=self.config.outlier_neighbors
        )
        if len(kept) < GRASPGENX_MIN_POINTS:
            kept = finite
        return kept

    def prepare(self, points_xyz: np.ndarray, *, max_grasps: int = 0) -> dict[str, np.ndarray] | None:
        """Filter and resample an object cloud, or ``None`` when it is too small to grasp.

        Filtering is ``filter_points``. The cloud is then resampled to the checkpoint's
        own point budget, as upstream's batched entry point does
        (``graspgenx.grasp_server.run_inference_batch``: "Each input PC is resampled
        (with replacement) to ``grasp_sampler.cfg.data.num_points`` ... this matches the
        training distribution"). Upstream's single-cloud path -- ``run_inference`` ->
        ``GraspGenXSampler.sample``, which is the one this adapter drives -- omits it, so
        without this a served frame reached the model several times over its training
        budget: a segmented bottle at 0.53 m gives ~12k points against a budget of 3500.
        The count also varies per frame with object size, distance and depth validity, so
        resampling is what makes the input, and the latency, a constant.
        """
        kept = self.filter_points(points_xyz)
        if kept is None:
            return None
        if len(kept) != self.config.object_points:
            # Upstream's own draw: with replacement, after outlier removal. Farthest-point
            # sampling or a stride would cover the surface more evenly than the training
            # data did, which is a different input distribution, not a better one. This
            # uses numpy's global stream deliberately, so a caller that seeds it (the
            # parity verifiers do) still gets a reproducible cloud.
            kept = kept[np.random.randint(0, len(kept), self.config.object_points)]
        if max_grasps > GRASPGENX_MAX_SAMPLE_COUNT:
            raise ValueError(f"GraspGenX max_grasps {max_grasps} exceeds {GRASPGENX_MAX_SAMPLE_COUNT}")
        sample_count = max(self.config.num_grasps, int(max_grasps))
        if self.compiled_sample_count is not None:
            # The offline diffusion loop is compiled for one row count, and the
            # generator declines a graph that does not match it -- silently, by
            # running the eager loop: ~320 ms against ~196 ms on Ascend310P1
            # (torch_models/graspgenx_ascend_310p/README.md). Nothing is given
            # up by pinning the draw, because the eager loop's cost barely
            # depends on the count, so a larger draw buys slower and not more.
            # max_grasps still bounds what is returned, which is what it means
            # on every other deployment anyway.
            sample_count = self.compiled_sample_count
        return {
            GRASPGEN_POINT_CLOUD_SEMANTIC: np.ascontiguousarray(kept, dtype=np.float32),
            GRASPGENX_SAMPLE_COUNT_SEMANTIC: np.asarray([sample_count], dtype=np.int64),
        }

    def postprocess(
        self,
        result: ModelResult,
        *,
        max_grasps: int = 0,
        min_confidence: float = 0.0,
    ) -> list[GraspCandidatePose]:
        poses = self._required(result, GRASPGEN_POSE_SEMANTIC)
        confidence = self._required(result, GRASPGEN_CONFIDENCE_SEMANTIC).reshape(-1)
        if poses.size == 0:
            return []
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise RuntimeError(f"GraspGenX poses must have shape [N, 4, 4], got {poses.shape}")
        if confidence.shape[0] != poses.shape[0]:
            raise RuntimeError(f"GraspGenX returned {poses.shape[0]} poses but {confidence.shape[0]} confidences")
        rotations = poses[:, :3, :3]
        orthonormal = np.abs(np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3)).max(axis=(1, 2))
        # R^T R is the identity for a reflection as much as for a rotation, so
        # the determinant is what separates them. det = -1 is a real failure
        # mode for a quantised or mis-transposed graph, and it reaches the
        # executor as a left-handed frame the gripper would act on.
        right_handed = np.linalg.det(rotations.astype(np.float64)) > 0
        valid = (
            np.isfinite(poses).all(axis=(1, 2))
            & np.isfinite(confidence)
            & (orthonormal <= _ORTHONORMAL_TOLERANCE)
            & right_handed
            & (confidence >= min_confidence)
        )
        order = [index for index in np.argsort(-confidence, kind="stable") if valid[index]]
        limit = int(max_grasps) if max_grasps > 0 else self.config.topk
        return [
            GraspCandidatePose(np.ascontiguousarray(poses[index], dtype=np.float32), float(confidence[index]))
            for index in order[:limit]
        ]

    @staticmethod
    def _required(result: ModelResult, semantic: str) -> np.ndarray:
        try:
            return np.asarray(result.outputs[semantic], dtype=np.float32)
        except KeyError as exc:
            raise RuntimeError(f"GraspGenX runtime result is missing {semantic!r}") from exc

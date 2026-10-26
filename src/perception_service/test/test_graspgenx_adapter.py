# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GraspGenX adapter semantics, pinned to the behaviour validated on X2 recordings.

The adapter reproduces the X2 grasp pipeline's GraspGenX wrapper: host-side outlier
removal with the sampler's own rule, the sampler's 10-point fallback, a 200-draw /
top-100 default and orthonormality screening of every returned pose.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from model_utils.graspgen_contract import (
    GRASPGEN_CONFIDENCE_SEMANTIC,
    GRASPGEN_POINT_CLOUD_SEMANTIC,
    GRASPGEN_POSE_SEMANTIC,
)
from perception_service.graspgenx_adapter import (
    GRASPGENX_POSTPROCESSING,
    GRASPGENX_PREPROCESSING,
    GRASPGENX_SAMPLE_COUNT_SEMANTIC,
    GraspGenXAdapter,
    GraspGenXConfig,
)


def _adapter(**overrides) -> GraspGenXAdapter:
    return GraspGenXAdapter(GraspGenXConfig(gripper_name="omnihand_left_x2", **{"object_points": 3500, **overrides}))


def _result(poses, confidence):
    return SimpleNamespace(
        outputs={
            GRASPGEN_POSE_SEMANTIC: np.asarray(poses, dtype=np.float32),
            GRASPGEN_CONFIDENCE_SEMANTIC: np.asarray(confidence, dtype=np.float32),
        }
    )


def _poses(count: int) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    poses[:, :3, 3] = np.arange(count, dtype=np.float32)[:, None]
    return poses


def _identity(**overrides) -> dict:
    identity = {
        "interface": "tensor_model",
        "model_type": "graspgenx",
        "operation": "generate_grasps",
        "preprocessing": GRASPGENX_PREPROCESSING,
        "postprocessing": GRASPGENX_POSTPROCESSING,
        "gripper_name": "omnihand_left_x2",
        "object_points": 3500,
    }
    identity.update(overrides)
    return identity


def test_outlier_removal_matches_the_full_mean_l1_rule_with_duplicates_and_boundary():
    """Same fixture as the X2 pipeline's tree-filter test: duplicates stay, the far point goes."""
    rng = np.random.default_rng(42)
    points = np.concatenate(
        [rng.normal(0, 0.003, (100, 3)), np.zeros((25, 3)), [[0.014, 0.0, 0.0], [0.014001, 0.0, 0.0], [0.3, 0.0, 0.0]]]
    ).astype(np.float32)
    distances = np.abs(points[:, None] - points[None]).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    expected = np.sort(distances, axis=1)[:, :20].mean(axis=1) < 0.014

    kept = _adapter().filter_points(points)

    np.testing.assert_array_equal(kept, points[expected])
    assert expected[:125].all() and not expected[-1]


def test_outlier_removal_keeps_up_with_a_whole_object_mask():
    """A real X2 bottle mask is ~27k points; filtering it must not cost the request its deadline."""
    rng = np.random.default_rng(7)
    # A 6 cm x 20 cm bottle side seen at 2 mm pixel spacing, plus stray depth far behind it.
    angle, height = rng.uniform(0, np.pi, 30000), rng.uniform(0, 0.2, 30000)
    surface = np.column_stack([0.03 * np.cos(angle), height, 0.6 + 0.03 * np.sin(angle)])
    stray = rng.uniform(-0.3, 0.3, (40, 3)) + [0.0, 0.0, 1.5]
    points = np.concatenate([surface, stray]).astype(np.float32)

    started = time.perf_counter()
    kept = _adapter().filter_points(points)
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert len(kept) == len(surface)
    assert (kept[:, 2] < 1.0).all()


def test_points_stay_in_the_source_frame_because_the_sampler_centres_them_itself():
    points = np.random.default_rng(1).normal(0, 0.002, (64, 3)).astype(np.float32) + np.float32([0.2, -0.1, 0.6])

    kept = _adapter().filter_points(points)

    np.testing.assert_array_equal(kept, points)


def test_an_over_filtered_cloud_falls_back_to_every_finite_point():
    """Sparse points all fail the neighbour test; the sampler then uses the unfiltered cloud."""
    points = np.arange(36, dtype=np.float32).reshape(12, 3)

    kept = _adapter().filter_points(points)

    np.testing.assert_array_equal(kept, points)


def test_non_finite_points_are_dropped_and_a_tiny_cloud_is_ungraspable():
    points = np.zeros((9, 3), dtype=np.float32)

    assert _adapter().prepare(np.concatenate([points, [[np.nan, 0, 0]]])) is None
    assert _adapter().filter_points(np.concatenate([points, [[np.nan, 0, 0]]])) is None
    with pytest.raises(ValueError, match="no finite points"):
        _adapter().prepare(np.full((4, 3), np.inf, dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        _adapter().prepare(np.zeros((4, 2), dtype=np.float32))


def _cloud(count: int, seed: int = 3) -> np.ndarray:
    """A compact blob that survives outlier filtering intact."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.002, (count, 3)) + [0.2, -0.1, 0.6]).astype(np.float32)


def test_a_served_cloud_is_resampled_to_the_checkpoints_point_budget():
    """Upstream resamples to cfg.data.num_points to match the training distribution.

    A real X2 bottle mask gives ~12k points against a budget of 3500, and the count
    varies per frame with object size, distance and depth validity, so without this the
    model sees a different input size every request -- and on Ascend anything over 4096
    points has no working path at all.
    """
    points = _cloud(12201)

    prepared = _adapter(object_points=3500).prepare(points)[GRASPGEN_POINT_CLOUD_SEMANTIC]

    assert len(prepared) == 3500
    # Whole rows are drawn from the filtered cloud, not synthesised or mixed across
    # points: an elementwise check would pass even if coordinates were shuffled.
    drawn = {row.tobytes() for row in prepared}
    assert drawn <= {row.tobytes() for row in points}


def test_a_cloud_under_the_budget_is_drawn_up_to_it_with_replacement():
    """collate stacks fixed-size clouds, so a small cloud is padded by redrawing."""
    points = _cloud(300)

    prepared = _adapter(object_points=3500).prepare(points)[GRASPGEN_POINT_CLOUD_SEMANTIC]

    assert len(prepared) == 3500
    assert len(np.unique(prepared, axis=0)) <= 300


def test_a_cloud_already_at_the_budget_is_left_alone():
    points = _cloud(64)

    prepared = _adapter(object_points=64).prepare(points)[GRASPGEN_POINT_CLOUD_SEMANTIC]

    np.testing.assert_array_equal(prepared, points)


def test_the_resampling_draw_follows_the_seeded_numpy_stream():
    """The parity verifiers seed numpy to make a device comparison reproducible."""
    points = _cloud(9000)
    adapter = _adapter(object_points=3500)

    np.random.seed(11)
    first = adapter.prepare(points)[GRASPGEN_POINT_CLOUD_SEMANTIC]
    np.random.seed(11)
    second = adapter.prepare(points)[GRASPGEN_POINT_CLOUD_SEMANTIC]

    np.testing.assert_array_equal(first, second)


def test_a_bundle_that_does_not_declare_its_point_budget_is_rejected():
    """Guessing it would serve a cloud the checkpoint was not trained on."""
    assets = _identity()
    del assets["object_points"]

    with pytest.raises(ValueError, match="must declare object_points"):
        GraspGenXConfig.from_assets(assets)


def test_the_draw_grows_to_cover_a_larger_request_but_never_shrinks():
    points = np.zeros((32, 3), dtype=np.float32)
    adapter = _adapter()

    def drawn(max_grasps):
        return int(adapter.prepare(points, max_grasps=max_grasps)[GRASPGENX_SAMPLE_COUNT_SEMANTIC][0])

    assert drawn(0) == 400
    assert drawn(50) == 400
    assert drawn(700) == 700
    # The draw is a GPU batch size, so a request field may not set it without bound.
    assert drawn(1000) == 1000
    with pytest.raises(ValueError, match="exceeds 1000"):
        drawn(4_000_000_000)


def test_postprocess_sorts_thresholds_and_keeps_the_default_top_k():
    confidence = np.linspace(0.0, 1.0, 500, dtype=np.float32)

    candidates = _adapter().postprocess(_result(_poses(500), confidence))

    assert len(candidates) == 400
    assert [c.confidence for c in candidates] == sorted((c.confidence for c in candidates), reverse=True)
    assert candidates[0].confidence == pytest.approx(1.0)
    floor = _adapter().postprocess(_result(_poses(500), confidence), min_confidence=0.9, max_grasps=500)
    assert floor and min(c.confidence for c in floor) >= 0.9


def test_postprocess_drops_non_finite_and_non_orthonormal_poses():
    poses = _poses(3)
    poses[0, 0, 0] = np.nan
    poses[1, :3, :3] *= 2.0

    candidates = _adapter().postprocess(_result(poses, [0.9, 0.8, 0.1]))

    assert [c.confidence for c in candidates] == [pytest.approx(0.1)]
    np.testing.assert_array_equal(candidates[0].pose_matrix, poses[2])


def test_postprocess_accepts_an_empty_draw_and_rejects_inconsistent_outputs():
    assert _adapter().postprocess(_result(np.empty((0, 4, 4)), np.empty((0,)))) == []
    with pytest.raises(RuntimeError, match="2 poses but 1 confidences"):
        _adapter().postprocess(_result(_poses(2), [0.5]))
    with pytest.raises(RuntimeError, match="missing"):
        _adapter().postprocess(SimpleNamespace(outputs={}))


def test_config_rejects_an_unbounded_draw_a_top_k_larger_than_it_and_an_unnamed_gripper():
    with pytest.raises(ValueError, match="exceeds 1000"):
        GraspGenXConfig.from_assets({"gripper_name": "g", "object_points": 3500, "num_grasps": 5000, "topk": 10})
    with pytest.raises(ValueError, match="exceeds the sampled count"):
        GraspGenXConfig.from_assets({"gripper_name": "g", "object_points": 3500, "num_grasps": 10, "topk": 20})
    with pytest.raises(ValueError, match="must name the gripper"):
        GraspGenXConfig.from_assets({})


def test_from_bundle_validates_the_adapter_identity(tmp_path):
    (tmp_path / "assets").mkdir()
    path = tmp_path / "assets" / "adapter.json"
    path.write_text(json.dumps(_identity(num_grasps=300, topk=120)), encoding="utf-8")

    adapter = GraspGenXAdapter.from_bundle(tmp_path)

    assert adapter.config == GraspGenXConfig(
        gripper_name="omnihand_left_x2", object_points=3500, num_grasps=300, topk=120
    )
    path.write_text(json.dumps(_identity(model_type="graspgen")), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        GraspGenXAdapter.from_bundle(tmp_path)


def test_a_reflection_is_rejected_although_it_passes_the_orthonormality_check():
    """R^T R is the identity for a reflection too, so the determinant decides.

    A det=-1 pose is what a quantised or mis-transposed graph produces, and it
    would otherwise reach the executor as a left-handed frame.
    """
    poses = _poses(2)
    poses[1, :3, :3] = np.diag([1.0, 1.0, -1.0])  # orthonormal, det = -1
    confidence = np.asarray([0.5, 0.9], dtype=np.float32)

    kept = _adapter().postprocess(_result(poses, confidence))

    assert np.abs(poses[1, :3, :3].T @ poses[1, :3, :3] - np.eye(3)).max() == 0.0
    assert [c.confidence for c in kept] == [pytest.approx(0.5)]


def test_a_compiled_diffusion_loop_pins_the_draw_to_the_graph():
    """A request may not push the draw off the compiled row count.

    The offline loop is built for one count and the generator declines any
    other -- silently, by running the eager loop, which cost ~320 ms against the
    graph's ~196 ms when both were measured (before the 2026-10-26 recompile).
    Pinning gives nothing up: the eager loop's cost barely depends on the
    count, so the larger draw would be slower rather than richer. max_grasps
    still bounds what is returned, which is all it means on a deployment
    without a compiled loop too.
    """
    points = np.zeros((32, 3), dtype=np.float32)
    # What the 310P README has an npu bundle declare, to match its graph.
    adapter = GraspGenXAdapter(
        GraspGenXConfig(gripper_name="omnihand_left_x2", object_points=3500, num_grasps=400, topk=400),
        compiled_sample_count=400,
    )

    def drawn(max_grasps):
        return int(adapter.prepare(points, max_grasps=max_grasps)[GRASPGENX_SAMPLE_COUNT_SEMANTIC][0])

    assert drawn(0) == 400
    assert drawn(100) == 400
    # Without the pin these would draw 500 and 1000 and leave the graph behind.
    assert drawn(500) == 400
    assert drawn(1000) == 400
    # The contract bound still applies: it is the service's limit, not the graph's.
    with pytest.raises(ValueError, match="exceeds 1000"):
        drawn(1001)


def test_only_a_deployment_carrying_the_loop_pins_the_draw(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "adapter.json").write_text(json.dumps(_identity()), encoding="utf-8")

    without = GraspGenXAdapter.from_bundle(tmp_path, deployment=SimpleNamespace(artifacts={}))
    with_loop = GraspGenXAdapter.from_bundle(
        tmp_path, deployment=SimpleNamespace(artifacts={"diffusion_loop": object()})
    )

    assert without.compiled_sample_count is None
    assert with_loop.compiled_sample_count == without.config.num_grasps


def test_a_bundle_read_without_a_deployment_does_not_pin_the_draw(tmp_path):
    """from_bundle is also called without a deployment, e.g. by packaging tools."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "adapter.json").write_text(json.dumps(_identity()), encoding="utf-8")

    assert GraspGenXAdapter.from_bundle(tmp_path).compiled_sample_count is None

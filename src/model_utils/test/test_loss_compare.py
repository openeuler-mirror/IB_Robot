from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import model_utils.loss_compare as loss_compare
from model_utils.observation_batch import FieldSpec, save_observation_batch


class _FakeEngine:
    policy_type = "pi05"
    backend_type = "ascend"
    nominal_chunk_size = 2
    max_action_dimension = 8

    def __init__(self, *, stateful: bool = True) -> None:
        self.capabilities = SimpleNamespace(stateful=stateful, resettable=stateful)
        self.calls = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(
        self,
        batch,
        *,
        request_id=None,
        control_inputs=None,
        capture_raw_action=False,
    ):
        noise = None if control_inputs is None else control_inputs.get("noise")
        value = float(torch.as_tensor(noise).mean()) if noise is not None else float(len(self.calls) + 1)
        raw = torch.full((2, 6), value)
        final = raw + 10.0
        self.calls.append(
            {
                "batch": batch,
                "request_id": request_id,
                "control_inputs": control_inputs,
                "capture_raw_action": capture_raw_action,
            }
        )
        return SimpleNamespace(raw_action=raw, action=final)


def _args(tmp_path, *, generate_target: bool = False) -> Namespace:
    return Namespace(
        policy_path=str(tmp_path / "bundle"),
        deployment="ascend",
        model_dtype="native",
        batch_path=str(tmp_path / "batches.json"),
        target_path=str(tmp_path / "target.json"),
        raw_target_path=str(tmp_path / "target_raw.json"),
        noise_dir=str(tmp_path / "noise"),
        generate_target=generate_target,
        task="pick",
        seed=42,
        policy_type="pi05",
        metrics_json=None,
        schedule_override_path=None,
        curvature_log_path=None,
    )


def _utils(tmp_path, engine: _FakeEngine, *, generate_target: bool = False):
    utils = object.__new__(loss_compare.LossUtils)
    utils.args = _args(tmp_path, generate_target=generate_target)
    utils.engine = engine
    return utils


def test_forward_resets_stateful_pipeline_and_separates_raw_from_final_actions(monkeypatch, tmp_path):
    engine = _FakeEngine()
    utils = _utils(tmp_path, engine)
    utils.args.noise_dir = ""
    monkeypatch.setattr(loss_compare, "np", np)
    monkeypatch.setattr(loss_compare, "torch", torch)
    monkeypatch.setattr(loss_compare, "tqdm", lambda values, **_kwargs: values)

    outputs = utils.forward([{"observation.state": np.zeros(6)}, {"observation.state": np.ones(6)}])

    assert engine.reset_calls == 2
    assert [call["request_id"] for call in engine.calls] == ["loss-compare-0", "loss-compare-1"]
    assert all(call["capture_raw_action"] is True for call in engine.calls)
    assert torch.equal(
        utils._raw_preds[0], torch.full((2, 6), float(engine.calls[0]["control_inputs"]["noise"].mean()))
    )
    assert torch.equal(outputs[0], utils._raw_preds[0] + 10.0)


def _runtime_dependencies():
    """Stand in for the lazily built runtime dependencies.

    ``loss_compare`` resolves PureInferenceEngine and runtime_dependencies in the
    same deferred-import step, so a test that replaces only the engine leaves the
    module global at None and prepare_policy fails reaching .registry_set.
    """
    return SimpleNamespace(registry_set="registry-set-sentinel", providers="providers-sentinel")


def test_prepare_policy_forwards_exact_named_deployment(monkeypatch, tmp_path):
    created = []

    class Engine(_FakeEngine):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(kwargs)

    monkeypatch.setattr(loss_compare, "PureInferenceEngine", Engine)
    monkeypatch.setattr(loss_compare, "runtime_dependencies", _runtime_dependencies())
    args = _args(tmp_path)
    args.deployment = "lab.ascend-310p3"

    engine = loss_compare.LossUtils.prepare_policy(SimpleNamespace(args=args))

    assert engine is not None
    assert created[0]["deployment"] == "lab.ascend-310p3"
    assert created[0]["runtime_options"] == {}
    assert created[0]["registry_set"] == "registry-set-sentinel"
    assert created[0]["providers"] == "providers-sentinel"


def test_prepare_policy_forwards_transient_ascend_diagnostics(monkeypatch, tmp_path):
    created = []

    class Engine(_FakeEngine):
        def __init__(self, **kwargs):
            super().__init__()
            created.append(kwargs)

    monkeypatch.setattr(loss_compare, "PureInferenceEngine", Engine)
    monkeypatch.setattr(loss_compare, "runtime_dependencies", _runtime_dependencies())
    args = _args(tmp_path)
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "format": "pi05-denoising-schedule-v1",
                "name": "test",
                "algorithm": "euler",
                "model_output": "velocity",
                "timesteps": [1.0, 0.0],
            }
        )
    )
    args.schedule_override_path = str(schedule_path)
    args.curvature_log_path = "/tmp/curvature.jsonl"

    loss_compare.LossUtils.prepare_policy(SimpleNamespace(args=args))

    assert created[0]["runtime_options"] == {"curvature_log_path": "/tmp/curvature.jsonl"}
    assert created[0]["pi05_diagnostic_schedule"].name == "test"
    assert created[0]["pi05_diagnostic_schedule_source"] == str(schedule_path.resolve())


def test_pi05_noise_is_external_and_deterministic_without_noise_dir(monkeypatch, tmp_path):
    utils = _utils(tmp_path, _FakeEngine())
    utils.args.noise_dir = ""
    monkeypatch.setattr(loss_compare, "torch", torch)

    first = utils._resolve_noise(3)
    torch.manual_seed(999)
    second = utils._resolve_noise(3)

    assert torch.equal(first, second)
    assert tuple(first.shape) == (1, 2, 8)


def test_pi05_target_generation_requires_noise_persistence(monkeypatch, tmp_path):
    utils = _utils(tmp_path, _FakeEngine(), generate_target=True)
    utils.args.noise_dir = ""
    monkeypatch.setattr(loss_compare, "torch", torch)

    with pytest.raises(RuntimeError, match="noise-dir"):
        utils._resolve_noise(0)


def test_pi05_noise_self_check_uses_metadata_shape_and_resets_each_run(monkeypatch, tmp_path):
    engine = _FakeEngine()
    utils = _utils(tmp_path, engine)
    monkeypatch.setattr(loss_compare, "torch", torch)

    utils._assert_noise_effective({"observation.state": np.zeros(6)})

    assert engine.reset_calls == 3
    assert [tuple(call["control_inputs"]["noise"].shape) for call in engine.calls] == [(1, 2, 8)] * 3
    assert torch.equal(engine.calls[0]["control_inputs"]["noise"], engine.calls[2]["control_inputs"]["noise"])
    assert not torch.equal(engine.calls[0]["control_inputs"]["noise"], engine.calls[1]["control_inputs"]["noise"])


def test_generate_target_writes_final_and_raw_action_documents(monkeypatch, tmp_path):
    engine = _FakeEngine(stateful=False)
    utils = _utils(tmp_path, engine, generate_target=True)
    monkeypatch.setattr(loss_compare, "torch", torch)
    monkeypatch.setattr(loss_compare, "tqdm", lambda values, **_kwargs: values)
    monkeypatch.setattr(utils, "load_batches_as_tensors", lambda: [{"observation.state": np.zeros(6)}])
    monkeypatch.setattr(utils, "_assert_noise_effective", lambda _batch: None)
    monkeypatch.setattr(utils, "_resolve_noise", lambda _index: None)

    utils.generate_target()

    assert json.loads((tmp_path / "target.json").read_text(encoding="utf-8")) == [[[11.0] * 6, [11.0] * 6]]
    assert json.loads((tmp_path / "target_raw.json").read_text(encoding="utf-8")) == [[[1.0] * 6, [1.0] * 6]]


def test_pi05_noise_shape_requires_bundle_metadata(tmp_path):
    engine = _FakeEngine()
    engine.max_action_dimension = None
    utils = _utils(tmp_path, engine)

    with pytest.raises(RuntimeError, match="max_action_dim"):
        utils._pi05_noise_shape()


def test_load_batches_normalizes_unbatched_json_observations(monkeypatch, tmp_path):
    utils = _utils(tmp_path, _FakeEngine())
    batch = {
        "observation.state": [1, 2, 3, 4, 5, 6],
        "observation.images.top_view": np.full((2, 3, 3), 255).tolist(),
        "observation.images.hand_view": np.zeros((2, 3, 3)).tolist(),
    }
    (tmp_path / "batches.json").write_text(json.dumps([batch]), encoding="utf-8")
    monkeypatch.setattr(loss_compare, "np", np)

    loaded = utils.load_batches_as_tensors()[0]

    assert loaded["observation.state"].shape == (1, 6)
    assert loaded["observation.images.top"].shape == (1, 3, 2, 3)
    assert loaded["observation.images.wrist"].shape == (1, 3, 2, 3)
    assert loaded["observation.images.top"].max() == 1.0
    assert loaded["task"] == "pick"


def test_load_batches_normalizes_safetensors_observations(monkeypatch, tmp_path):
    pytest.importorskip("safetensors")
    utils = _utils(tmp_path, _FakeEngine())
    utils.args.batch_path = str(tmp_path / "batches.safetensors")
    save_observation_batch(
        utils.args.batch_path,
        [
            {
                "observation.state": np.arange(1, 7, dtype=np.float32),
                "observation.images.top_view": np.full((2, 3, 3), 255, dtype=np.uint8),
                "observation.images.hand_view": np.zeros((2, 3, 3), dtype=np.uint8),
                "task": "pick",
            }
        ],
        field_specs=[
            FieldSpec("observation.state", (6,), "float32"),
            FieldSpec("observation.images.top_view", (2, 3, 3), "uint8", semantic="image", layout="HWC"),
            FieldSpec("observation.images.hand_view", (2, 3, 3), "uint8", semantic="image", layout="HWC"),
        ],
    )
    monkeypatch.setattr(loss_compare, "np", np)

    loaded = utils.load_batches_as_tensors()[0]

    assert loaded["observation.state"].shape == (1, 6)
    assert loaded["observation.images.top"].shape == (1, 3, 2, 3)
    assert loaded["observation.images.top"].max() == 1.0
    assert loaded["task"] == "pick"


def test_compute_loss_writes_structured_aggregate_metrics_json(monkeypatch, tmp_path):
    utils = _utils(tmp_path, _FakeEngine(stateful=False))
    utils.args.metrics_json = str(tmp_path / "reports" / "metrics.json")
    (tmp_path / "target.json").write_text(json.dumps([[[1.0, 0.0], [1.0, 0.0]]]), encoding="utf-8")
    (tmp_path / "target_raw.json").write_text(json.dumps([[[1.0, 0.0], [1.0, 0.0]]]), encoding="utf-8")
    monkeypatch.setattr(loss_compare, "np", np)
    monkeypatch.setattr(loss_compare, "torch", torch)
    monkeypatch.setattr(utils, "load_batches_as_tensors", lambda: [{}])

    def forward(_batches):
        utils._raw_preds = [torch.tensor([[2.0, 0.0], [2.0, 0.0]])]
        utils._inference_latencies_ms = [12.5]
        return [torch.tensor([[1.0, 0.0], [1.0, 0.0]])]

    monkeypatch.setattr(utils, "forward", forward)
    monkeypatch.setattr(
        "model_utils.pi05_dist_metrics.evaluate_pi05",
        lambda **_kwargs: {
            "normalized": {
                "wasserstein": {"mean_ratio": 0.25},
                "first_frame": {"mean_cos": 0.75},
            },
            "unnormalized": {},
        },
    )

    metrics = utils.compute_loss()
    written = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))

    assert metrics == written
    assert written["inference"]["average_latency_ms"] == 12.5
    assert written["unnormalized"]["l1"] == 0.0
    assert written["unnormalized"]["cosine"] == pytest.approx(1.0)
    assert written["normalized"]["l1"] == 0.5
    assert written["normalized"]["cosine"] == pytest.approx(1.0)
    assert written["aggregates"]["raw_l1"] == 0.5
    assert written["aggregates"]["normalized_mean_w1_std"] == 0.25
    assert written["aggregates"]["normalized_first_frame_cos"] == 0.75

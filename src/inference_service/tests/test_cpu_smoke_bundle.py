from __future__ import annotations

import json
import logging
import sys
import time
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ibrobot_tracing import TraceEmitter
from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import InferenceRequest
from inference_service.backends.errors import BackendInferenceError
from inference_service.distributed.runtime import CloudBackendRuntime, EdgeProcessorRuntime
from inference_service.pipeline import create_pipeline_manager
from inference_service.pipeline import runtime as pipeline_runtime
from inference_service.pipeline.stages import ModelStage
from inference_service.runtime_composition import build_policy_runtime_dependencies
from inference_service.unified_runtime import ExecutionFailure
from robot_config.inference_config import parse_inference_config
from tests.manifest_fixtures import TEST_BUNDLE_UUID, TEST_DEPLOYMENT_UUID

MODEL_NAME = "ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515"


class _TraceRecorder(TraceEmitter):
    def __init__(self, enabled=True) -> None:
        logger = logging.Logger(__name__, logging.INFO)
        super().__init__(logger, enabled=enabled)
        self.events: list[tuple[str, str, dict[str, object]]] = []
        handler = logging.Handler()
        handler.emit = self._record
        logger.addHandler(handler)

    def _record(self, record):
        payload = json.loads(record.getMessage().split(" ", 1)[1])
        name, fields = payload["event"], payload["fields"]
        kind = "span" if name == "span_begin" else name
        self.events.append((kind, fields.get("span_name", fields.get("edge_id", name)), fields))


def _workspace() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _create_tracked_equivalent_bundle(root: Path) -> Path:
    fixture = Path(__file__).parent / "assets" / MODEL_NAME
    root.mkdir()
    paths = (
        "config.json",
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
    )
    for path in paths:
        (root / path).write_bytes((fixture / path).read_bytes())
    entries = [BundleFile(path=path) for path in paths]
    _write_json(
        root / "inference_manifest.json",
        {
            "schema_version": 3,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": MODEL_NAME,
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, MODEL_NAME, entries),
                },
            },
            "model": {
                "interface": "policy",
                "model_type": "act",
                "operation": "predict",
                "inputs": [
                    {"semantic": "observation.state", "dtype": "float32", "shape": [6]},
                    {"semantic": "observation.images.top", "dtype": "float32", "shape": [3, 16, 24]},
                ],
                "outputs": [{"semantic": "action", "dtype": "float32", "shape": [6]}],
            },
            "deployments": {
                "cpu": {
                    "uuid": TEST_DEPLOYMENT_UUID,
                    "revision": 1,
                    "execution_contract": {
                        "state_scope": "request",
                        "execution_structure": "direct",
                        "cancellation_granularity": "request_boundary",
                    },
                    "runtime_profile": {
                        "backend": "torch",
                        "target": {"runtime": "torch"},
                        "profile": {"device": "cpu"},
                    },
                }
            },
        },
    )
    return root


def _install_fake_lerobot(monkeypatch, torch_module, calls):
    lerobot_module = ModuleType("lerobot")
    configs_module = ModuleType("lerobot.configs")
    config_module = ModuleType("lerobot.configs.policies")
    policies_module = ModuleType("lerobot.policies")
    factory_module = ModuleType("lerobot.policies.factory")

    class FakeConfig:
        def __init__(self, policy_type: str, device: str) -> None:
            self.type = policy_type
            self.device = device

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
            calls["metadata_path"] = Path(path)
            return cls(raw["type"], raw["device"])

    class FakePolicy:
        supports_attention = False

        def __init__(self, config) -> None:
            self.config = config
            self.model = SimpleNamespace(supports_attention=False)
            self.closed = False

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["policy_path"] = Path(path)
            calls["policy_kwargs"] = kwargs
            return cls(kwargs["config"])

        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            return self

        def predict_action_chunk(self, batch):
            calls["batch"] = dict(batch)
            calls["model_calls"] = calls.get("model_calls", 0) + 1
            if calls.get("model_delay"):
                time.sleep(float(calls["model_delay"]))
            if calls.get("model_error"):
                raise calls["model_error"]
            return torch_module.zeros((1, 4, 6), dtype=torch_module.float32)

        def reset(self):
            calls["reset"] = calls.get("reset", 0) + 1

    def get_policy_class(policy_type):
        assert policy_type == "act"
        return FakePolicy

    def make_pre_post_processors(**kwargs):
        calls["processor_path"] = Path(kwargs["pretrained_path"])

        def preprocess(batch):
            calls["preprocess_calls"] = calls.get("preprocess_calls", 0) + 1
            if calls.get("preprocess_error"):
                raise calls["preprocess_error"]
            return batch

        def postprocess(action):
            calls["postprocess_calls"] = calls.get("postprocess_calls", 0) + 1
            if calls.get("postprocess_error"):
                raise calls["postprocess_error"]
            return action

        return preprocess, postprocess

    config_module.PreTrainedConfig = FakeConfig
    factory_module.get_policy_class = get_policy_class
    factory_module.make_pre_post_processors = make_pre_post_processors
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs_module)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", config_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_module)


def test_named_ignored_bundle_manifest_validates() -> None:
    bundle = _workspace() / "models" / MODEL_NAME
    if not bundle.is_dir():
        pytest.skip(f"local smoke bundle is unavailable: {bundle}")

    validated = load_inference_manifest(bundle, "cpu")

    assert validated.manifest.bundle.name == MODEL_NAME
    assert validated.policy.policy_type == "act"
    assert validated.deployment.backend == "torch"


def test_tracked_equivalent_cpu_bundle_runs_unified_registry_pipeline_end_to_end(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    inference = parse_inference_config(
        {
            "control_modes": {
                "model_inference": {
                    "inference": {
                        "enabled": True,
                        "pipelines": {
                            "cpu_smoke": {
                                "model_path": str(bundle),
                                "deployment": "cpu",
                                "execution_mode": "monolithic",
                                "request_timeout": 2.0,
                            }
                        },
                    }
                }
            }
        },
        "model_inference",
    )
    pipeline = inference.pipelines["cpu_smoke"]
    validated = pipeline.validated_manifest
    dependencies = build_policy_runtime_dependencies()
    manager = create_pipeline_manager(
        "cpu_smoke",
        validated,
        request_timeout=2.0,
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )

    result = manager.infer(
        "cpu_smoke",
        InferenceRequest(
            request_id="smoke-1",
            inputs={
                "observation.state": np.zeros(6, dtype=np.float32),
                "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
                "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            },
        ),
    )

    assert result.pipeline_id == "cpu_smoke"
    assert result.bundle == MODEL_NAME
    assert result.backend == "torch"
    assert result.actual_chunk_size == 4
    assert tuple(result.action.shape) == (4, 6)
    assert calls["metadata_path"] == bundle
    assert calls["policy_path"] == bundle
    assert calls["processor_path"] == bundle
    assert calls["device"] == "cpu"
    manager.reset("cpu_smoke")
    manager.close()
    manager.close()
    dependencies.providers.close()


def test_policy_runtime_traces_aggregate_call_and_keeps_backend_latency(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    calls = {"model_delay": 0.01}
    recorder = _TraceRecorder()
    # Deliberately zero trace durations: business latency must use its own clock.
    recorder.monotonic_clock = lambda: 0
    monkeypatch.setattr(pipeline_runtime, "trace", recorder)
    _install_fake_lerobot(monkeypatch, torch, calls)
    validated = load_inference_manifest(bundle, "cpu")
    dependencies = build_policy_runtime_dependencies()
    manager = create_pipeline_manager(
        "traced",
        validated,
        request_timeout=2.0,
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )
    request = InferenceRequest(
        request_id="trace-request",
        inputs={"observation.state": np.zeros(6, dtype=np.float32)},
        metadata={"trace_flow_id": "trace-flow"},
    )
    try:
        result = manager.infer("traced", request)

        assert result.backend_latency_ms >= 8.0
        assert result.metadata["latency_ms"]["backend"] == result.backend_latency_ms
        assert all(fields["duration_ns"] == 0 for kind, _, fields in recorder.events if kind == "span_end")
        assert [name for kind, name, _fields in recorder.events if kind == "span"] == [
            "preprocess",
            "model_call",
            "postprocess",
        ]
        assert [(kind, name) for kind, name, _fields in recorder.events if kind in {"flow_send", "flow_receive"}] == [
            ("flow_receive", "observation_to_preprocess"),
            ("flow_send", "preprocess_to_inference"),
            ("flow_receive", "preprocess_to_inference"),
            ("flow_send", "inference_to_postprocess"),
            ("flow_receive", "inference_to_postprocess"),
        ]
        assert all(
            fields["flow_id"] == "trace-request"
            for kind, _name, fields in recorder.events
            if kind in {"flow_send", "flow_receive"}
        )
    finally:
        manager.close()
        dependencies.providers.close()


@pytest.mark.parametrize("failure_phase", [None, "preprocess", "model", "postprocess"])
def test_trace_toggle_preserves_policy_output_errors_and_call_counts(monkeypatch, tmp_path, failure_phase):
    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    outcomes = []
    for enabled in (False, True):
        monkeypatch.setenv("IB_TRACE_ENABLED", "1" if enabled else "0")
        recorder = _TraceRecorder(enabled)
        monkeypatch.setattr(pipeline_runtime, "trace", recorder)
        fault = RuntimeError("same business failure")
        calls = {} if failure_phase is None else {f"{failure_phase}_error": fault}
        _install_fake_lerobot(monkeypatch, torch, calls)
        dependencies = build_policy_runtime_dependencies()
        manager = create_pipeline_manager(
            "toggle",
            load_inference_manifest(bundle, "cpu"),
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
        try:
            if failure_phase is None:
                result = manager.infer("toggle", InferenceRequest("request", {}))
                payload = (result.action.tolist(), result.actual_chunk_size)
            else:
                with pytest.raises(RuntimeError) as error:
                    manager.infer("toggle", InferenceRequest("request", {}))
                assert error.value is fault
                payload = vars(fault).copy()
            counts = tuple(calls.get(f"{phase}_calls", 0) for phase in ("preprocess", "model", "postprocess"))
            outcomes.append((payload, counts, manager.health("toggle").state))
            if enabled:
                ends = [fields for kind, _, fields in recorder.events if kind == "span_end"]
                assert len(ends) == sum(counts)
                assert [event["status"] for event in ends] == (
                    ["ok"] * len(ends)
                    if failure_phase is None
                    else ["ok"] * (len(ends) - 1) + ["incomplete" if failure_phase == "model" else "error"]
                )
            else:
                assert recorder.events == []
        finally:
            manager.close()
            dependencies.providers.close()
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("failure_role", [None, "encoder", "step", "decoder"])
def test_aggregate_span_keeps_flat_role_order_exception_and_evidence(monkeypatch, tmp_path, failure_role):
    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    original_builder = pipeline_runtime.InferencePipeline._build_session_executor
    roles = ("encoder", "step", "decoder")
    outcomes = []
    for enabled in (False, True):
        recorder = _TraceRecorder(enabled)
        monkeypatch.setattr(pipeline_runtime, "trace", recorder)
        _install_fake_lerobot(monkeypatch, torch, {})
        calls = []
        fault = BackendInferenceError("role failed", operation_started=True, outcome_known=False)

        def build(facade, session_handle, calls=calls, fault=fault):
            # Exercise multiple flat stages without fixing the PI05 registry.
            model_executor = session_handle.model_executor
            session = model_executor.stages[0].session

            def execute(role_request, context, *, role):
                calls.append((role, tuple(role_request.inputs), dict(role_request.metadata)))
                if role == failure_role:
                    raise fault
                return session.execute(role_request, context)

            model_executor._stages = tuple(
                ModelStage(role, SimpleNamespace(execute=partial(execute, role=role))) for role in roles
            )
            return original_builder(facade, session_handle)

        monkeypatch.setattr(pipeline_runtime.InferencePipeline, "_build_session_executor", build)
        dependencies = build_policy_runtime_dependencies()
        manager = create_pipeline_manager(
            "multi",
            load_inference_manifest(bundle, "cpu"),
            registry_set=dependencies.registry_set,
            providers=dependencies.providers,
        )
        outer_span = object()
        outer_context = pipeline_runtime._model_span.set(outer_span)
        try:
            request = InferenceRequest("multi-request", {}, metadata={"trace_flow_id": "ignored"})
            if failure_role is None:
                result = manager.infer("multi", request)
                outcome = (result.action.tolist(), result.metadata["outcome_evidence"])
            else:
                with pytest.raises(BackendInferenceError) as error:
                    manager.infer("multi", request)
                assert error.value is fault
                assert isinstance(fault.__cause__, ExecutionFailure)
                outcome = (vars(fault).copy(), fault.__cause__.evidence.to_dict())
            outcomes.append((calls, outcome, manager.health("multi").state))
            expected_roles = roles if failure_role is None else roles[: roles.index(failure_role) + 1]
            assert tuple(role for role, _, _ in calls) == expected_roles
            assert pipeline_runtime._model_span.get() is outer_span
            model_events = [(kind, fields) for kind, name, fields in recorder.events if name == "model_call"]
            if enabled:
                assert [kind for kind, _ in model_events] == ["span", "span_end"]
                assert model_events[1][1]["status"] == ("ok" if failure_role is None else "incomplete")
                assert model_events[0][1]["span_id"] == model_events[1][1]["span_id"]
            else:
                assert recorder.events == []
        finally:
            pipeline_runtime._model_span.reset(outer_context)
            manager.close()
            dependencies.providers.close()
    assert outcomes[0] == outcomes[1]


def test_disabled_policy_tracing_does_not_query_tokens_fields_or_coerce_flags(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    _install_fake_lerobot(monkeypatch, torch, {})
    monkeypatch.setattr(pipeline_runtime, "trace", SimpleNamespace(enabled=False))
    monkeypatch.setattr(pipeline_runtime, "_model_span", object())

    class Uncoercible:
        def __bool__(self):
            pytest.fail("an observation flag must not coerce user objects")

        def __str__(self):
            pytest.fail("disabled tracing must not serialize observation fields")

    dependencies = build_policy_runtime_dependencies()
    manager = create_pipeline_manager(
        "disabled",
        load_inference_manifest(bundle, "cpu"),
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
        instrument_processors=Uncoercible(),
        model_component_id=Uncoercible(),
    )
    try:
        result = manager.infer("disabled", InferenceRequest("request", {}))
        assert result.actual_chunk_size == 4
    finally:
        manager.close()
        dependencies.providers.close()


def test_cloud_traces_real_processors_and_edge_identity_has_no_processor_spans(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    recorder = _TraceRecorder()
    monkeypatch.setattr(pipeline_runtime, "trace", recorder)
    dependencies = build_policy_runtime_dependencies()
    validated = load_inference_manifest(bundle, "cpu")
    cloud = CloudBackendRuntime(
        "cloud",
        validated,
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )
    edge = EdgeProcessorRuntime("edge", validated)
    edge.load()
    try:
        inputs = edge.preprocess({"observation.state": np.zeros(6, dtype=np.float32)})
        assert recorder.events == []
        result = cloud.infer("distributed-1", inputs)
        assert calls["processor_path"] == bundle
        assert [(name, fields["component_id"]) for kind, name, fields in recorder.events if kind == "span"] == [
            ("preprocess", "policy.preprocess"),
            ("model_call", "cloud_inference"),
            ("postprocess", "policy.postprocess"),
        ]
        flows = [(kind, name, fields) for kind, name, fields in recorder.events if kind.startswith("flow_")]
        assert [(kind, name) for kind, name, _ in flows] == [
            ("flow_receive", "observation_to_preprocess"),
            ("flow_send", "preprocess_to_inference"),
            ("flow_receive", "preprocess_to_inference"),
            ("flow_send", "inference_to_postprocess"),
            ("flow_receive", "inference_to_postprocess"),
        ]
        assert {fields["flow_id"] for _, _, fields in flows} == {"distributed-1"}
        before = list(recorder.events)
        edge.postprocess(result.action, actual_chunk_size=result.actual_chunk_size)
        assert recorder.events == before
    finally:
        edge.close()
        cloud.close()
        dependencies.providers.close()


@pytest.mark.parametrize(
    "scheduler", [None, {"enable": False}, {"enable": True}], ids=["absent", "disabled", "enabled"]
)
@pytest.mark.parametrize("reset_fails", [False, True])
def test_policy_late_result_recovery_is_scheduler_only(monkeypatch, tmp_path, scheduler, reset_fails):
    from inference_service.pipeline.errors import PipelineNotReadyError, PipelineTimeoutError

    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    inference_config = {
        "enabled": True,
        "pipelines": {
            "policy": {
                "model_path": str(bundle),
                "deployment": "cpu",
                "execution_mode": "monolithic",
            }
        },
    }
    scheduled = bool(scheduler and scheduler["enable"])
    if scheduler is not None:
        inference_config["scheduler"] = scheduler
    if scheduled:
        inference_config["scheduler"] = {
            **scheduler,
            "global_endpoints": {
                "readiness": "/inference/scheduler/ready",
                "open_session": "/inference/session/open",
                "dispatch": "/inference/dispatch",
                "close_session": "/inference/session/close",
            },
        }
        inference_config["pipelines"]["policy"].update(
            compatibility_group="policy",
            hardware_resource_id="cpu:0",
            hardware_profile_fingerprint="a" * 64,
            transport={
                endpoint: f"/inference/policy/scheduled/{endpoint}"
                for endpoint in ("open_session", "dispatch", "close_session", "serving_status")
            },
            public_capacity={
                "session_control": {"max_in_flight": 1},
                "action_generation": {"max_in_flight": 1},
            },
        )
    config = parse_inference_config(
        {
            "control_modes": {
                "model_inference": {"inference": inference_config, "executor": {"inference_pipeline": "policy"}}
            }
        },
        "model_inference",
    )
    dependencies = build_policy_runtime_dependencies()
    manager = create_pipeline_manager(
        "policy",
        config.pipelines["policy"].validated_manifest,
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
        priority_scheduling=bool(config.scheduler and config.scheduler.enable),
    )
    pipeline = manager.pipelines["policy"]
    request = InferenceRequest(
        "late",
        inputs={
            "observation.state": np.zeros(6, dtype=np.float32),
            "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
        },
    )
    postprocess = pipeline._postprocessor

    def late_result(_action):
        raise pipeline._timeout_error("postprocess", backend_completed=True)

    try:
        assert pipeline.capabilities.stateful and pipeline.capabilities.resettable
        monkeypatch.setattr(pipeline, "_postprocessor", late_result)
        with pytest.raises(PipelineTimeoutError):
            manager.infer("policy", request)
        failure = pipeline._policy_failure
        assert failure is not None
        monkeypatch.setattr(pipeline, "_postprocessor", postprocess)
        if not scheduled:
            assert manager.infer("policy", request).actual_chunk_size == 4
            resets_before = calls.get("reset", 0)
            manager.reset("policy")
            assert calls.get("reset", 0) == resets_before
            assert pipeline._policy_failure is failure
            return
        with pytest.raises(PipelineNotReadyError, match="session left READY"):
            manager.infer("policy", request)
        if reset_fails:

            def fail_reset():
                raise RuntimeError("reset failed")

            monkeypatch.setattr(pipeline.runtime_handle.assembly.session, "_reset", fail_reset)
            with pytest.raises(RuntimeError, match="reset failed"):
                manager.reset("policy")
            assert pipeline._policy_failure is failure
            assert not pipeline.runtime_handle.health.ready
        else:
            resets_before = calls.get("reset", 0)
            manager.reset("policy")
            assert calls["reset"] == resets_before + 1
            assert pipeline._policy_failure is None
            assert manager.infer("policy", request).actual_chunk_size == 4
    finally:
        manager.close()
        dependencies.providers.close()

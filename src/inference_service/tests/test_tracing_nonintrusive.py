"""Tracing observes the existing validators and control boundaries, not new ones."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from ibrobot_tracing import TraceEmitter
from inference_service.backends import BackendCapabilities, BackendState
from inference_service.backends.errors import BackendInferenceError
from inference_service.codecs import build_execution_plan
from inference_service.model_sessions import OnnxRuntimeModelSession
from inference_service.model_sessions.base import ModelSession
from inference_service.model_sessions.lerobot_torch import LeRobotTorchModelSession
from inference_service.pipeline import SequentialModelExecutor
from inference_service.pipeline import runtime as pipeline_runtime
from inference_service.pipeline.runtime_core import StageFrame
from inference_service.pipeline.stages import ModelStage
from inference_service.unified_runtime import ExecutionContext, ModelRequest
from tests.test_onnx_model_session import _context, _fake_ort, _FakeGraph, _stateless_manifest


@pytest.mark.parametrize("enabled", [False, True])
def test_public_request_can_prepare_internal_role_inputs_before_execution(tmp_path, enabled):
    class OrchestratedSession(ModelSession):
        def __init__(self):
            super().__init__("host-orchestrated", BackendCapabilities())
            self.received = []

        def _load(self, context, rollback):
            pass

        def _execute(self, request, context):
            self.received.append(request)
            prepared = np.asarray(request.inputs["observation.public_audio"], dtype=np.float32)
            return {"host.silero.prob": np.asarray([prepared.mean()], dtype=np.float32)}

        def _close(self):
            pass

    manifest = _stateless_manifest()
    manifest["model"]["inputs"][0]["semantic"] = "observation.public_audio"
    session = OrchestratedSession()
    session.load(_context(tmp_path, manifest))
    request = ModelRequest({"observation.public_audio": np.ones(4, dtype=np.float32)}, {"caller": "business"})
    logger = logging.Logger("nonintrusive", logging.INFO)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    emitter = TraceEmitter(logger, enabled=enabled)
    try:
        with emitter.span("public.execute"):
            outputs = session.execute(request, ExecutionContext("public"))
        assert session.received == [request]
        assert dict(request.metadata) == {"caller": "business"}
        np.testing.assert_array_equal(outputs["host.silero.prob"], [1.0])
        assert len(records) == (2 if enabled else 0)
    finally:
        session.close()


@pytest.mark.parametrize("failure", ["shape", "dtype", "missing", "sdk", None])
def test_trace_toggle_preserves_role_input_validation_position(monkeypatch, tmp_path, failure):
    outcomes = []
    for enabled in (False, True):
        monkeypatch.setattr(pipeline_runtime, "trace", TraceEmitter(logging.getLogger(__name__), enabled=enabled))
        fault = BackendInferenceError("vendor error")

        def run(_feed, fault=fault):
            if failure == "sdk":
                raise fault
            return [np.ones(1, dtype=np.float32)]

        graph = _FakeGraph(("input",), ("prob",), run)
        session = OnnxRuntimeModelSession(
            ort_loader=lambda graph=graph: _fake_ort({str(tmp_path / "artifacts/model.onnx"): graph})
        )
        context = _context(tmp_path, _stateless_manifest())
        session.load(context)
        executor = SequentialModelExecutor(
            (ModelStage("model", session),),
            SimpleNamespace(adapt=lambda frame: frame.values["host.silero.prob"]),
            execution_plan=build_execution_plan(context.deployment.execution, context.deployment.bindings),
        )
        # Lists bypass ModelStage's pre-existing NumPy dtype coercion.
        value = [1.0] * 4 if failure == "dtype" else np.ones(3 if failure == "shape" else 4, dtype=np.float32)
        inputs = {} if failure == "missing" else {"host.silero.audio": value}
        try:
            if failure:
                with pytest.raises(BackendInferenceError) as error:
                    executor.execute(ModelRequest(inputs), ExecutionContext("role"))
                if failure == "sdk":
                    assert error.value is fault
                if failure in {"shape", "dtype"}:
                    assert failure in error.value.code
                outcome = (error.value.code, str(error.value), vars(error.value).copy())
            else:
                outcome = executor.execute(ModelRequest(inputs), ExecutionContext("role")).tolist()
            # Baseline validates role shape/dtype after the vendor call. Missing
            # input is rejected while building the feed, before the vendor call.
            assert len(graph.feeds) == (0 if failure == "missing" else 1)
            outcomes.append((outcome, len(graph.feeds)))
        finally:
            session.close()
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("action_method", ["forward", "select_action"])
def test_trace_toggle_keeps_policy_validation_after_input_preparation(monkeypatch, action_method):
    outcomes = []
    for enabled in (False, True):
        monkeypatch.setattr(pipeline_runtime, "trace", TraceEmitter(logging.getLogger(__name__), enabled=enabled))
        session = LeRobotTorchModelSession("cpu")
        session._state = BackendState.READY
        session._torch = SimpleNamespace()
        session._policy = SimpleNamespace()
        session._context = SimpleNamespace(
            validated_manifest=SimpleNamespace(manifest=SimpleNamespace(model=SimpleNamespace(inputs=())))
        )
        prepared = []
        monkeypatch.setattr(session, "_move_input", prepared.append)
        request = ModelRequest({"state": 1}, {"action_method": action_method})
        frame = StageFrame(
            request, values={"_model_request": request, "_execution_context": ExecutionContext("validation")}
        )
        try:
            with pytest.raises(BackendInferenceError) as error:
                ModelStage("policy", session).execute(frame, deadline=None)
            assert prepared == [1]
            outcomes.append((error.value.code, vars(error.value).copy()))
        finally:
            frame.close()
            session.close()
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("device", ["cpu", "cuda", "npu", "mps"])
def test_trace_toggle_does_not_add_processor_sync_or_device_walk(monkeypatch, device):
    class Output:
        @property
        def device(self):
            pytest.fail("tracing must not inspect processor output devices")

    def unexpected(*args):
        pytest.fail("tracing must not synchronize or walk processor outputs")

    result = {"nested": [Output()]}
    for enabled in (False, True):
        monkeypatch.setenv("IB_TRACE_ENABLED", "1" if enabled else "0")
        session = LeRobotTorchModelSession(device)
        session._torch = SimpleNamespace(is_tensor=unexpected, **{device: SimpleNamespace(synchronize=unexpected)})
        session._device = device
        session._preprocessor = lambda inputs: result
        session._postprocessor = lambda action: result
        assert session.preprocess({}) is result
        assert session.postprocess(result) is result

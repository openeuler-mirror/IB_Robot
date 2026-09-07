"""Readiness and lifecycle gate tests for the audio runtimes.

Covers the fail-closed gates that audio services must expose before model
execution: runtime readiness failures, state-bank reset, stream-scoped
admission, and reverse-order cleanup for the audio session assemblies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1]
_WORKSPACE_SRC = _SRC.parent
for package_root in (_SRC, _WORKSPACE_SRC / "inference_manifest", _WORKSPACE_SRC / "inference_service"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from inference_service.unified_runtime import (  # noqa: E402
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
    RuntimeAssembly,
)
from voice_asr_service.asr_inference_module import ASRInferenceModule, ASRState  # noqa: E402
from voice_asr_service.vad_runtime import ManifestVadRuntime  # noqa: E402

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_SILERO_BUNDLE = _WORKSPACE_ROOT / "models" / "silero-vad"


class _RecordingStreamingRuntime:
    """Stream-scoped runtime recording resets and exposing a state bank."""

    def __init__(self, *, fail_steps: bool = False):
        self.bank = 0
        self.resets = 0
        self.streams_closed = 0
        self.fail_steps = fail_steps

    def load(self, _context=None):
        return None

    def open_stream(self, _context):
        self.bank = 0
        return None

    def step(self, _stream, request, context):
        context.check("backend")
        if self.fail_steps:
            raise RuntimeError("backend exploded mid-step")
        self.bank += 100
        return {"outputs": {"value": self.bank}, "latency": 1.0}

    def reset_stream(self, _stream, _context):
        self.bank = 0
        self.resets += 1

    def close_stream(self, _stream, _context):
        self.streams_closed += 1

    def close(self):
        return None


def _stream_handle(**kwargs) -> tuple[ModelRuntimeHandle, _RecordingStreamingRuntime]:
    streaming = _RecordingStreamingRuntime(**kwargs)
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=streaming,
            streaming_runtime=streaming,
            execution_contract=ExecutionContract(
                state_scope="stream",
                state_bank_mode="per_stream",
                max_open_streams=1,
            ),
            stateful=True,
            resettable=True,
            state_scope="stream",
            state_bank_mode="per_stream",
            max_open_streams=1,
        )
    )
    return handle, streaming


def test_stream_scoped_runtime_rejects_request_execution():
    handle, _streaming = _stream_handle()
    handle.load()

    with pytest.raises(ExecutionFailure) as exc:
        handle.execute(ModelRequest({"x": 1}), ExecutionContext("request"))
    assert exc.value.code == "stream_required"

    handle.close()


def test_execution_before_load_fails_closed_with_lifecycle_error():
    handle, _streaming = _stream_handle()

    with pytest.raises(ExecutionFailure) as exec_failure:
        handle.execute(ModelRequest({"x": 1}), ExecutionContext("early"))
    assert exec_failure.value.code == "runtime_not_ready"

    with pytest.raises(ExecutionFailure) as stream_failure:
        handle.open_stream(ExecutionContext("early-stream"))
    assert stream_failure.value.code == "runtime_not_ready"


def test_stateful_stream_failure_requires_reset_then_recovers():
    handle, streaming = _stream_handle(fail_steps=True)
    handle.load()
    stream = handle.open_stream(ExecutionContext("open"))

    with pytest.raises(ExecutionFailure) as failed:
        handle.step(stream, ModelRequest({"x": 1}), ExecutionContext("step-1"))
    assert failed.value.code == "execution_failed"
    assert failed.value.recovery is not None

    with pytest.raises(ExecutionFailure) as blocked:
        handle.step(stream, ModelRequest({"x": 1}), ExecutionContext("step-2"))
    assert blocked.value.code == "recovery_required"

    handle.reset_stream(stream, ExecutionContext("reset"))
    assert streaming.resets == 1

    streaming.fail_steps = False
    recovered = handle.step(stream, ModelRequest({"x": 1}), ExecutionContext("step-3"))
    assert recovered.outputs["value"] == 100

    handle.close_stream(stream, ExecutionContext("close"))
    handle.close()


def test_state_bank_reset_clears_stream_state_without_reload():
    handle, _streaming = _stream_handle()
    handle.load()
    stream = handle.open_stream(ExecutionContext("open"))

    first = handle.step(stream, ModelRequest({"x": 1}), ExecutionContext("step-1"))
    second = handle.step(stream, ModelRequest({"x": 1}), ExecutionContext("step-2"))
    assert first.outputs["value"] == 100
    assert second.outputs["value"] == 200

    handle.reset_stream(stream, ExecutionContext("reset"))
    after_reset = handle.step(stream, ModelRequest({"x": 1}), ExecutionContext("step-3"))
    assert after_reset.outputs["value"] == 100, "reset must rewind the state bank to its initial value"

    handle.close()


def test_asr_module_gates_before_initialization():
    asr = ASRInferenceModule()

    with pytest.raises(RuntimeError, match="recognizer is not initialized"):
        asr.recognize_file(np.zeros(512, dtype=np.float32))

    assert asr.accept_waveform(np.zeros(512, dtype=np.float32)) is None
    assert asr.get_partial_result().text == ""
    assert asr.get_final_result().is_final is True

    with pytest.raises(RuntimeError, match="streaming ASR runtime not initialized"):
        asr.create_stream()

    asr.reset()
    asr.cleanup()
    assert asr.state is ASRState.IDLE


def test_asr_start_streaming_rejects_offline_model():
    asr = ASRInferenceModule()
    asr._model_type = type(asr._model_type)("offline")
    asr.state = ASRState.READY

    with pytest.raises(RuntimeError, match="only for streaming models"):
        asr.start_streaming()


@pytest.mark.skipif(
    not (_SILERO_BUNDLE / "inference_manifest.json").is_file(),
    reason="local silero-vad bundle is not present",
)
def test_manifest_vad_runtime_real_load_reset_and_close():
    runtime = ManifestVadRuntime(_SILERO_BUNDLE, "torch_cpu")

    with pytest.raises(RuntimeError, match="not initialized"):
        runtime.infer(np.zeros(512, dtype=np.float32))

    runtime.initialize()
    assert runtime.sample_rate_hz == 16000
    assert runtime.frame_size == 512

    probability = runtime.infer(np.zeros(512, dtype=np.float32))
    assert 0.0 <= probability <= 1.0

    with pytest.raises(RuntimeError, match="already initialized"):
        runtime.initialize()

    runtime.reset()
    probability_after_reset = runtime.infer(np.zeros(512, dtype=np.float32))
    assert 0.0 <= probability_after_reset <= 1.0

    runtime.close()
    runtime.close()

    with pytest.raises(RuntimeError, match="not initialized"):
        runtime.infer(np.zeros(512, dtype=np.float32))

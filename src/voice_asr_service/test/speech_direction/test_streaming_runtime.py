"""Focused unified-runtime ownership tests for Speech Direction streaming."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2]
_WORKSPACE_SRC = _SRC.parent
for package_root in (_SRC, _WORKSPACE_SRC / "inference_service"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from inference_service.unified_runtime import (  # noqa: E402
    ExecutionContext,
    ExecutionContract,
    ExecutionFailure,
    ModelRequest,
    ModelRuntimeHandle,
    OwnedComponent,
    RuntimeAssembly,
)
from voice_asr_service.speech_direction.model_sessions import (  # noqa: E402
    SpeechDirectionRoleRunner,
    SpeechDirectionSessionResources,
)
from voice_asr_service.speech_direction.runtime import SpeechDirectionRuntime  # noqa: E402
from voice_asr_service.speech_direction.speech_gate import SileroVadEngine  # noqa: E402
from voice_asr_service.speech_direction.streaming_runtime import (  # noqa: E402
    SpeechDirectionStreamingRuntime,
)


class _Pipeline:
    hop_size = 2
    frame_size = 4
    sr = 16000

    def __init__(self) -> None:
        self.vad_state = SimpleNamespace(get=lambda: (0.8, True))
        self.doa_state = SimpleNamespace()
        self.reset_calls = 0
        self.processed: list[tuple[np.ndarray, int | None]] = []
        self.close_calls: list[bool] = []
        self.processed_event = threading.Event()

    def reset(self) -> None:
        self.reset_calls += 1

    def process_block(self, audio: np.ndarray, *, capture_start_sample: int | None = None):
        self.processed.append((audio.copy(), capture_start_sample))
        self.processed_event.set()
        return SimpleNamespace(seg_output=90, seg_seq=len(self.processed))

    def close(self, *, close_backends: bool = True) -> None:
        self.close_calls.append(close_backends)


class _Resource:
    def __init__(self) -> None:
        self.load_calls = 0
        self.load_contexts: list[object] = []
        self.close_calls = 0

    def load(self, _context: object) -> None:
        self.load_calls += 1
        self.load_contexts.append(_context)

    def close(self) -> None:
        self.close_calls += 1


class _VadBackend:
    def __init__(self) -> None:
        self.close_calls = 0

    def infer(self, _audio: np.ndarray) -> float:
        return 0.5

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1


def _handle() -> tuple[ModelRuntimeHandle, SpeechDirectionStreamingRuntime, _Pipeline, _Resource]:
    pipeline = _Pipeline()
    streaming = SpeechDirectionStreamingRuntime(pipeline, close_backends=False)
    session_resource = _Resource()
    contract = ExecutionContract(
        state_scope="stream",
        execution_structure="direct",
        cancellation_granularity="checkpoint",
        state_bank_mode="runtime_exclusive",
        max_open_streams=1,
    )
    role_context = object()
    handle = ModelRuntimeHandle(
        RuntimeAssembly(
            runtime_executor=streaming,
            streaming_runtime=streaming,
            session=session_resource,
            owned_components=(
                OwnedComponent(session_resource, "session", load_context=role_context),
                OwnedComponent(streaming, "streaming-runtime"),
            ),
            stateful=True,
            resettable=True,
            state_scope="stream",
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
            execution_contract=contract,
        )
    )
    return handle, streaming, pipeline, session_resource


def test_speech_direction_stream_lifecycle_and_exclusive_admission() -> None:
    handle, _streaming, pipeline, session_resource = _handle()
    handle.load()
    assert session_resource.load_calls == 1
    session_entry = next(entry for entry in handle.owned_components if entry.name == "session")
    assert session_resource.load_contexts == [session_entry.load_context]

    first = handle.open_stream(ExecutionContext("open-1"))
    assert first.stream_id == "speech-direction-1"
    with pytest.raises(ExecutionFailure) as capacity:
        handle.open_stream(ExecutionContext("open-2"))
    assert capacity.value.code == "stream_capacity_exhausted"

    audio = np.arange(8, dtype=np.float32).reshape(4, 2)
    result = handle.step(
        first,
        ModelRequest({"audio": audio, "capture_start_sample": 128}),
        ExecutionContext("step-1"),
    )
    assert result.outputs["segment_angle"] == 90
    assert result.to_dict()["outputs"]["segment_seq"] == 1
    np.testing.assert_array_equal(pipeline.processed[0][0], audio)
    assert pipeline.processed[0][1] == 128

    handle.reset_stream(first, ExecutionContext("reset-1"))
    assert pipeline.reset_calls >= 3  # load, open, and stream reset
    handle.close_stream(first, ExecutionContext("close-1"))
    handle.close_stream(first, ExecutionContext("close-1-again"))
    assert first.closed
    with pytest.raises(ExecutionFailure) as closed:
        handle.step(first, ModelRequest({"audio": audio}), ExecutionContext("step-closed"))
    assert closed.value.code == "stream_closed"

    second = handle.open_stream(ExecutionContext("open-3"))
    assert second.stream_id == "speech-direction-2"
    handle.close()
    assert pipeline.close_calls == [False]
    assert session_resource.close_calls == 1


def test_session_resources_and_role_runner_do_not_close_sessions_from_host_layer() -> None:
    session = _Resource()
    resources = SpeechDirectionSessionResources({"fullsubnet": (session, object())})
    resources.load()
    resources.close()
    resources.close()
    assert session.load_calls == 1
    assert session.close_calls == 1

    runner = SpeechDirectionRoleRunner(session, context=None)
    runner.close()
    assert session.close_calls == 1


def test_invalid_stream_input_is_rejected_before_pipeline_execution() -> None:
    handle, _streaming, pipeline, session_resource = _handle()
    handle.load()
    stream = handle.open_stream(ExecutionContext("open"))
    with pytest.raises(ExecutionFailure) as invalid:
        handle.step(stream, ModelRequest({"other": np.zeros((2, 2), dtype=np.float32)}), ExecutionContext("bad"))
    assert invalid.value.code == "invalid_stream_input"
    assert not pipeline.processed
    with pytest.raises(ExecutionFailure) as invalid_shape:
        handle.step(
            stream,
            ModelRequest({"audio": np.zeros((2, 3), dtype=np.float32)}),
            ExecutionContext("bad-shape"),
        )
    assert invalid_shape.value.code == "invalid_stream_input"
    assert not pipeline.processed
    handle.close()
    assert session_resource.close_calls == 1


def test_capture_runtime_routes_worker_blocks_through_handle() -> None:
    handle, _streaming, pipeline, session_resource = _handle()
    config = SimpleNamespace(
        audio=SimpleNamespace(sample_rate=16000, channels=2),
        speech_direction_max_age_ms=1000,
    )
    runtime = SpeechDirectionRuntime(
        config,
        pipeline,
        enable_capture=False,
        model_runtime_handle=handle,
    )
    runtime.start()
    runtime.feed_audio(np.ones((2, 2), dtype=np.float32))
    assert pipeline.processed_event.wait(1.0)
    assert runtime.stream_handle is not None
    runtime.stop()
    assert runtime.stream_handle is None
    assert session_resource.close_calls == 1
    assert pipeline.close_calls == [False]
    # Give the worker a scheduling opportunity before the test exits; a live
    # daemon worker would otherwise hide an ownership/race regression.
    time.sleep(0.01)


def test_silero_host_close_leaves_backend_close_for_the_owner(tmp_path) -> None:
    model_path = tmp_path / "silero.om"
    model_path.write_bytes(b"mock")
    backend = _VadBackend()
    engine = SileroVadEngine(str(model_path), inference_runner=backend)
    engine.close(close_runner=False)
    assert backend.close_calls == 0
    engine.close()
    assert backend.close_calls == 1

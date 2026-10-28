from __future__ import annotations

import io
import subprocess

import numpy as np
import pytest

from inference_service.backends.hisilicon import sd3403_protocol as protocol_module
from inference_service.backends.hisilicon.sd3403_protocol import (
    DIM_STRUCT,
    INPUT_ENTRY_STRUCT,
    OUTPUT_ENTRY_STRUCT,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    REQUEST_HEADER_STRUCT,
    RESPONSE_HEADER_STRUCT,
    WORKER_ELEM_FLOAT32,
    WORKER_STATUS_OK,
    SD3403Protocol,
    SD3403ProtocolError,
    SD3403WorkerError,
    SD3403WorkerExitedError,
)


class FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode: int | None = None
        self.pid = 1234
        self.terminated = False
        self.killed = False
        self.wait_timeouts: list[float | None] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class NonExitingProcess(FakeProcess):
    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        raise subprocess.TimeoutExpired("worker", timeout)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def response_frame(
    outputs: list[tuple[int, np.ndarray]],
    *,
    request_id: int,
    prefix: bytes = b"",
    status: int = WORKER_STATUS_OK,
    error_code: int = 0,
    error_message: str = "",
) -> bytes:
    message = error_message.encode()
    frame = bytearray(prefix)
    frame.extend(
        RESPONSE_HEADER_STRUCT.pack(
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            status,
            request_id,
            len(outputs),
            1250,
            error_code,
            len(message),
        )
    )
    for output_index, value in outputs:
        array = np.ascontiguousarray(value, dtype=np.float32)
        frame.extend(
            OUTPUT_ENTRY_STRUCT.pack(
                output_index,
                WORKER_ELEM_FLOAT32,
                array.size,
                array.nbytes,
                array.ndim,
                0,
            )
        )
        for dimension in array.shape:
            frame.extend(DIM_STRUCT.pack(dimension))
        frame.extend(memoryview(array).cast("B"))
    frame.extend(message)
    return bytes(frame)


def create_protocol(
    tmp_path,
    process: FakeProcess,
    monkeypatch,
    **protocol_options,
) -> tuple[SD3403Protocol, dict[str, object]]:
    worker = tmp_path / "worker"
    model = tmp_path / "model.om"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    model.write_bytes(b"om")
    for key in protocol_module._LEGACY_WORKER_ENV_KEYS:
        monkeypatch.setenv(key, f"legacy-{key}")
    call: dict[str, object] = {}

    def process_factory(args, **kwargs):
        call["args"] = args
        call["kwargs"] = kwargs
        return process

    return SD3403Protocol(worker, model, process_factory=process_factory, **protocol_options), call


def test_protocol_preserves_back_to_back_frames_sparse_outputs_and_request_ids(tmp_path, monkeypatch):
    first_action = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    second_action = np.arange(18, dtype=np.float32).reshape(1, 3, 6)
    process = FakeProcess(
        response_frame([(1, first_action)], request_id=1, prefix=b"load runtime OK\n")
        + response_frame([(1, second_action)], request_id=2)
    )
    protocol, call = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    first = protocol.execute((np.ones((1, 6), dtype=np.float64),))
    second = protocol.execute((np.zeros((1, 6), dtype=np.float32),))

    assert first.request_id == 1
    assert second.request_id == 2
    assert first.worker_latency_us == 1250
    np.testing.assert_array_equal(first.outputs[1], first_action)
    np.testing.assert_array_equal(second.outputs[1], second_action)
    assert call["args"] == [str(protocol.worker_path), "--model", str(protocol.model_path)]
    environment = call["kwargs"]["env"]
    assert all(key not in environment for key in protocol_module._LEGACY_WORKER_ENV_KEYS)

    request_bytes = process.stdin.getvalue()
    offset = 0
    for request_id in (1, 2):
        magic, version, input_count, actual_request_id, reserved = REQUEST_HEADER_STRUCT.unpack_from(
            request_bytes, offset
        )
        offset += REQUEST_HEADER_STRUCT.size
        assert (magic, version, input_count, actual_request_id, reserved) == (
            PROTOCOL_MAGIC,
            PROTOCOL_VERSION,
            1,
            request_id,
            0,
        )
        input_index, byte_size, reserved = INPUT_ENTRY_STRUCT.unpack_from(request_bytes, offset)
        offset += INPUT_ENTRY_STRUCT.size
        assert input_index == 0
        assert byte_size == 24
        assert reserved == 0
        offset += byte_size
    assert offset == len(request_bytes)
    protocol.close()


def test_protocol_reports_worker_error_and_mismatched_request_id(tmp_path, monkeypatch):
    process = FakeProcess(
        response_frame([], request_id=1, status=1, error_code=7, error_message="extract output failed")
    )
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    with pytest.raises(SD3403WorkerError, match="extract output failed") as error:
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))
    assert error.value.error_code == 7
    protocol.close()

    mismatched = FakeProcess(response_frame([(1, np.zeros((1, 1, 6)))], request_id=9))
    protocol, _ = create_protocol(tmp_path, mismatched, monkeypatch)
    protocol.start()
    with pytest.raises(SD3403ProtocolError, match="mismatched response id"):
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))
    protocol.close()


def test_protocol_restarts_dead_worker_before_request_without_replaying_partial_io(tmp_path, monkeypatch):
    dead = FakeProcess()
    dead.returncode = 3
    live = FakeProcess(response_frame([(1, np.zeros((1, 1, 6)))], request_id=1))
    processes = iter((dead, live))
    protocol, _ = create_protocol(tmp_path, dead, monkeypatch)
    protocol._process_factory = lambda *args, **kwargs: next(processes)
    protocol.start()

    response = protocol.execute((np.zeros((1, 6), dtype=np.float32),))

    assert response.request_id == 1
    protocol.close()


def test_protocol_maps_stream_loss_to_worker_exit_and_force_closes(tmp_path, monkeypatch):
    process = NonExitingProcess()
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    with pytest.raises(SD3403WorkerExitedError, match="stream closed"):
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))

    protocol.close()
    protocol.close()
    assert process.terminated is True
    assert process.killed is True


def test_protocol_drains_stderr_and_records_model_load_time(tmp_path, monkeypatch):
    process = FakeProcess(stderr=b"worker booted\nmodel_load_ms=12.75\n")
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()
    stderr_thread = protocol._stderr_thread
    assert stderr_thread is not None
    stderr_thread.join(timeout=1.0)

    assert protocol.stderr_tail == ("worker booted", "model_load_ms=12.75")
    assert protocol.model_load_ms == 12.75
    protocol.close()


def test_protocol_can_leave_non_exiting_worker_when_force_close_is_disabled(tmp_path, monkeypatch):
    process = NonExitingProcess()
    protocol, _ = create_protocol(
        tmp_path,
        process,
        monkeypatch,
        graceful_close_timeout=0.0,
        force_close=False,
    )
    protocol.start()

    protocol.close()

    assert process.terminated is False
    assert process.killed is False


def test_protocol_rejects_unreasonable_output_metadata(tmp_path, monkeypatch):
    frame = bytearray(RESPONSE_HEADER_STRUCT.pack(PROTOCOL_MAGIC, PROTOCOL_VERSION, WORKER_STATUS_OK, 1, 1, 0, 0, 0))
    frame.extend(OUTPUT_ENTRY_STRUCT.pack(1, WORKER_ELEM_FLOAT32, 2, 4, 0, 0))
    process = FakeProcess(bytes(frame))
    protocol, _ = create_protocol(tmp_path, process, monkeypatch)
    protocol.start()

    with pytest.raises(SD3403ProtocolError, match="byte size does not match"):
        protocol.execute((np.zeros((1, 6), dtype=np.float32),))
    protocol.close()

from __future__ import annotations

import logging

from embodied_common.tracing import current_trace_id, sanitize_trace_id, trace_scope, trace_stage


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_sanitize_trace_id_bounds_and_rejects_credentials() -> None:
    assert sanitize_trace_id(" request-1 ") == "request-1"
    assert sanitize_trace_id("Bearer secret") == "unknown"
    assert sanitize_trace_id("api key: secret") == "unknown"
    assert sanitize_trace_id("request\nforged") == "unknown"
    assert sanitize_trace_id("", fallback="") == ""
    assert len(sanitize_trace_id("x" * 200)) == 128


def test_trace_scope_and_stage_emit_safe_timing_record() -> None:
    logger = logging.getLogger("test.ib_trace")
    logger.setLevel(logging.INFO)
    handler = _CaptureHandler()
    logger.addHandler(handler)
    try:
        with trace_scope("request-1"):
            assert current_trace_id() == "request-1"
            with trace_stage(logger, "agent.test", task_id="task-1"):
                pass
        assert current_trace_id() == ""
        assert len(handler.records) == 1
        message = handler.records[0].getMessage()
        assert "trace_id=request-1" in message
        assert "stage=agent.test" in message
        assert "task_id=task-1" in message
        assert "latency_ms=" in message
    finally:
        logger.removeHandler(handler)


def test_stage_fields_cannot_reintroduce_credential_ids() -> None:
    logger = logging.getLogger("test.ib_trace.fields")
    logger.setLevel(logging.INFO)
    handler = _CaptureHandler()
    logger.addHandler(handler)
    request_id = "bearer synthetic-marker"
    try:
        with trace_scope(request_id), trace_stage(logger, "test", request_id=request_id, task_id="task\nforged"):
            pass
        message = handler.records[0].getMessage()
        assert "synthetic-marker" not in message
        assert "forged" not in message
        assert "trace_id=unknown" in message
        assert "request_id=unknown" in message
        assert "task_id=unknown" in message
    finally:
        logger.removeHandler(handler)

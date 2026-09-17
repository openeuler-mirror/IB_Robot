"""Low-overhead stage timing helpers for the agent control plane.

This module is deliberately separate from ``ibrobot_tracing``:

- ``ibrobot_tracing`` owns structured ``IBTRACE1`` instrumentation with its
  analysis toolchain (parsing, critical path, workbench). Its trace-id
  ContextVar is named ``ibrobot_trace_id``.
- This module only emits bounded ``[hop]`` stage-latency records for the
  agent request lifecycle; its ContextVar is named ``embodied_stage_trace_id``
  so the two in-process contexts cannot be confused.

Cross-subsystem correlation uses the ``DispatchBinding.trace_id`` message
field on the wire, never a shared in-process ContextVar.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

try:
    from lttngust import loghandler as _lttng_loghandler
except ImportError:
    _lttng_loghandler = None

_TRACE_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_trace_id: ContextVar[str] = ContextVar("embodied_stage_trace_id", default="")
_hop_count: ContextVar[int] = ContextVar("ibrobot_stage_hop_count", default=0)


def sanitize_trace_id(value: object, *, fallback: str = "unknown") -> str:
    """Bound trace IDs and reject values that look like credentials."""
    text = str(value or "").strip()
    if not text:
        return fallback
    lowered = text.casefold()
    if "bearer " in lowered or "api key" in lowered or "apikey" in lowered or not text.isprintable():
        return fallback
    return text[:128]


def create_trace_logger(name: str) -> logging.Logger:
    """Create an ``ib_trace.*`` logger compatible with the LTTng launch setup."""
    logger = logging.getLogger(name)
    level = os.environ.get("IB_TRACE_LOG_LEVEL", "").upper()
    logger.setLevel(_TRACE_LEVELS.get(level, logging.WARNING))
    if _lttng_loghandler is not None and not any(
        isinstance(handler, _lttng_loghandler._Handler) for handler in logger.handlers
    ):
        try:
            logger.addHandler(_lttng_loghandler._Handler())
            logger.propagate = False
        except OSError:
            pass
    return logger


def current_trace_id() -> str:
    """Return the trace ID bound to the current execution context."""
    return _trace_id.get()


@contextmanager
def trace_scope(trace_id: object) -> Iterator[str]:
    """Bind a trace ID and reset the per-process hop counter for one request."""
    trace_token = _trace_id.set(sanitize_trace_id(trace_id))
    hop_token = _hop_count.set(0)
    try:
        yield _trace_id.get()
    finally:
        _hop_count.reset(hop_token)
        _trace_id.reset(trace_token)


@contextmanager
def trace_stage(logger: logging.Logger, stage: str, **fields: object) -> Iterator[None]:
    """Emit one bounded stage timing record without logging payloads or secrets."""
    hop = _hop_count.get() + 1
    _hop_count.set(hop)
    started = time.perf_counter()
    try:
        yield
    finally:
        safe_fields = " ".join(f"{key}={sanitize_trace_id(value)}" for key, value in fields.items())
        logger.info(
            "[hop] trace_id=%s stage=%s hop=%d latency_ms=%.2f%s",
            current_trace_id() or "unknown",
            stage,
            hop,
            (time.perf_counter() - started) * 1000.0,
            f" {safe_fields}" if safe_fields else "",
        )


__all__ = [
    "create_trace_logger",
    "current_trace_id",
    "sanitize_trace_id",
    "trace_scope",
    "trace_stage",
]

"""Compatibility entrypoint for historical tracing imports.

Deprecated: new code must import directly from the owning package:

- stage timing helpers (``create_trace_logger``, ``current_trace_id``,
  ``sanitize_trace_id``, ``trace_scope``, ``trace_stage``) live in
  ``embodied_common.tracing`` (agent-plane ``[hop]`` stage records);
- structured instrumentation (``get_trace_emitter``, spans, ``IBTRACE1``
  events and the analysis toolchain) lives in ``ibrobot_tracing``.

This shim re-exports both homes so historical imports keep working; it is
slated for removal once callers have migrated.
"""

from embodied_common.tracing import (
    create_trace_logger,
    current_trace_id,
    sanitize_trace_id,
    trace_scope,
    trace_stage,
)
from ibrobot_tracing.instrumentation import get_trace_emitter

__all__ = [
    "create_trace_logger",
    "current_trace_id",
    "get_trace_emitter",
    "sanitize_trace_id",
    "trace_scope",
    "trace_stage",
]

"""Compatibility entrypoint for historical tracing imports.

New code should import tracing instrumentation directly from ``ibrobot_tracing``.
"""

from ibrobot_tracing.instrumentation import create_trace_logger, get_trace_emitter

__all__ = ["create_trace_logger", "get_trace_emitter"]

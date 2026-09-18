"""Structured Python instrumentation carried by the LTTng logging handler."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

_TRACE_PREFIX = "IBTRACE1 "
_MAX_DEPTH = 6
_MAX_FIELDS = 128
_MAX_STRING_LENGTH = 1024
_MAX_RECORD_BYTES = 16_384
_MAX_DEFINITIONS = 256
_IDENTITY_FIELDS = (
    "trace_id",
    "request_id",
    "inference_id",
    "span_id",
    "parent_span_id",
    "flow_id",
    "edge_id",
    "component_id",
    "origin",
    "span_name",
)
_NULL_CONTEXT = nullcontext()
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("ibrobot_trace_id", default="")
_component_var: contextvars.ContextVar[str] = contextvars.ContextVar("ibrobot_component_id", default="")
_span_stack_var: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("ibrobot_span_stack", default=())

P = ParamSpec("P")
R = TypeVar("R")


def _level() -> int:
    level_name = os.environ.get("IB_TRACE_LOG_LEVEL", "WARNING").upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(level_name, logging.WARNING)


def _enabled() -> bool:
    return os.environ.get("IB_TRACE_ENABLED", "0").lower() in {"1", "true", "yes", "on"}


def create_trace_logger(name: str) -> logging.Logger:
    """Explicitly opt in to LTTng, without configuring the caller's business logger."""
    logger = logging.getLogger(f"{name}._ibrobot_trace")
    logger.setLevel(_level())
    logger.propagate = False
    if any(getattr(handler, "_ibrobot_lttng_handler", False) for handler in logger.handlers):
        return logger
    root_level = logging.root.level
    try:
        from lttngust.loghandler import _Handler

        handler = _Handler()
        handler._ibrobot_lttng_handler = True
        logger.addHandler(handler)
    except (MemoryError, SystemError):
        raise
    except Exception:
        pass
    finally:
        # The LTTng Python agent sets the root logger to NOTSET while importing.
        # Restore the host application's level so enabling trace does not expose
        # unrelated DEBUG logs, notably ROS 2 Launch event dispatch internals.
        logging.root.setLevel(root_level)
    return logger


def _json_value(value: Any) -> Any:
    """Snapshot only builtin JSON values; never dispatch to business object hooks."""
    remaining = _MAX_FIELDS
    byte_budget = _MAX_RECORD_BYTES // 2
    ancestors = set()

    def visit(item, depth):
        nonlocal remaining, byte_budget
        remaining -= 1
        byte_budget -= 32
        kind = type(item)
        if item is None or kind is bool:
            return item
        if kind is str:
            text = item[: min(_MAX_STRING_LENGTH, max(0, byte_budget) // 12)]
            byte_budget -= len(json.dumps(text))
            return text
        if kind is int:
            return item if item.bit_length() <= 64 else "<unsupported>"
        if kind is float:
            return item if math.isfinite(item) else "<unsupported>"
        if kind is not list and kind is not dict:
            return "<unsupported>"
        if depth >= _MAX_DEPTH or id(item) in ancestors:
            return "<truncated>"
        ancestors.add(id(item))
        try:
            if kind is list:
                result = []
                for child in item:
                    if remaining <= 0 or byte_budget <= 0:
                        break
                    result.append(visit(child, depth + 1))
            else:
                result = {}
                for key, child in item.items():
                    if remaining <= 0 or byte_budget <= 0:
                        break
                    if type(key) is str and len(key) <= _MAX_STRING_LENGTH:
                        key_bytes = len(json.dumps(key))
                        if key_bytes + 64 > byte_budget:
                            break
                        byte_budget -= key_bytes
                        result[key] = visit(child, depth + 1)
                    else:
                        remaining -= 1
            return result
        finally:
            ancestors.remove(id(item))

    return visit(value, 0)


def _text(value: Any) -> str:
    return value[:_MAX_STRING_LENGTH] if type(value) is str else ""


def _identity(value: Any) -> str:
    # Never alias different requests by shortening an identifier to a shared prefix.
    if value is None:
        return ""
    if type(value) is not str or len(value) > _MAX_STRING_LENGTH:
        raise ValueError("Unsupported trace identity")
    return value


def _restore_context(tokens) -> None:
    for variable, token, previous in reversed(tokens):
        try:
            variable.reset(token)
        except (MemoryError, SystemError):
            raise
        except Exception:
            try:
                variable.set(previous)
            except (MemoryError, SystemError):
                raise
            except Exception:
                pass


def _set_context(*values):
    tokens = []
    try:
        for variable, value in values:
            previous = variable.get()
            tokens.append((variable, variable.set(value), previous))
    except (MemoryError, SystemError):
        raise
    except Exception:
        _restore_context(tokens)
        tokens.clear()
    return tokens


@dataclass(slots=True)
class _SpanToken:
    fields: dict[str, Any]
    started_ns: int
    ended: bool = False


class _TraceContext:
    def __init__(self, emitter, trace_id, component_id):
        self.emitter = emitter
        self.trace_id = trace_id
        self.component_id = component_id
        self.tokens = []

    def __enter__(self):
        if not self.emitter.enabled:
            return
        try:
            try:
                trace_id = _identity(self.trace_id)
                component_id = (
                    _identity(self.component_id)
                    or _identity(_component_var.get())
                    or _identity(self.emitter.default_component)
                )
            except ValueError:
                # Do not accidentally attribute an invalid nested identity to its parent.
                trace_id = ""
                component_id = ""
            self.tokens = _set_context(
                (_trace_id_var, trace_id),
                (_component_var, component_id),
            )
        except (MemoryError, SystemError):
            raise
        except Exception:
            pass

    def __exit__(self, exc_type, exc, traceback):
        _restore_context(self.tokens)
        self.tokens.clear()
        return False


class _SpanScope:
    """Scope tracing only; __exit__ observes the exception type, never the object."""

    def __init__(self, emitter, name, fields):
        self.emitter = emitter
        self.name = name
        self.fields = fields
        self.token = None
        self.context_tokens = []

    def __enter__(self):
        if not self.emitter.enabled:
            self.fields = {}
            return
        self.token = self.emitter.start_span(self.name, **self.fields)
        self.fields = {}
        if self.token is not None:
            try:
                self.context_tokens = _set_context(
                    (_span_stack_var, (*_span_stack_var.get(), self.token.fields["span_id"])),
                    (_component_var, self.token.fields["component_id"]),
                )
            except (MemoryError, SystemError):
                raise
            except Exception:
                pass

    def __exit__(self, exc_type, exc, traceback):
        _restore_context(self.context_tokens)
        self.context_tokens.clear()
        if self.token is not None:
            self.emitter.end_span(
                self.token,
                status="ok" if exc_type is None else "error",
                exception_name="" if exc_type is None else type.__dict__["__name__"].__get__(exc_type),
            )
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, traceback):
        return self.__exit__(exc_type, exc, traceback)


@dataclass(slots=True)
class TraceEmitter:
    """Startup-configured tracing; disabled scopes reuse a no-op context.

    Argument evaluation and Python's kwargs binding still happen at call sites.
    Changing enabled between scope creation and entry is not supported.
    """

    logger: logging.Logger | None
    default_component: str = ""
    clock: Callable[[], int] = time.time_ns
    monotonic_clock: Callable[[], int] = time.monotonic_ns
    enabled: bool = True
    _emitted_definitions: set[tuple[str, str, str, str]] | None = field(default=None, init=False, repr=False)

    def _write_record(self, name, fields, timestamp_ns=None) -> bool:
        identities = {key: _identity(fields[key]) for key in _IDENTITY_FIELDS if key in fields}
        attributes = _json_value({key: value for key, value in fields.items() if key not in identities})
        # Identity fields are never subject to the ordinary attribute budget. If the
        # complete record does not fit, drop it rather than manufacture an identity.
        protected_fields = {**attributes, **identities}
        payload = {
            "schema_version": 1,
            "event": _identity(name),
            "timestamp_ns": timestamp_ns if timestamp_ns is not None else self.clock(),
            "monotonic_ns": self.monotonic_clock(),
            "clock": "realtime",
            "fields": protected_fields,
        }
        for key in ("timestamp_ns", "monotonic_ns"):
            if type(payload[key]) is not int or payload[key].bit_length() > 64:
                return False
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)
        if len(encoded) + len(_TRACE_PREFIX) > _MAX_RECORD_BYTES:
            return False
        self.logger.info("%s%s", _TRACE_PREFIX, encoded)
        # No cache for absent, filtered or indirect sinks: retrying is harmless.
        return (
            self.logger.isEnabledFor(logging.INFO)
            and not self.logger.filters
            and any(
                not isinstance(handler, logging.NullHandler) and not handler.filters and handler.level <= logging.INFO
                for handler in self.logger.handlers
            )
        )

    def _log_failure(self, message: str, name: str) -> None:
        try:
            # A tracer failure may chain to a business error; never format that traceback.
            self.logger.debug(message, _text(name))
        except (MemoryError, SystemError):
            raise
        except Exception:
            pass

    def _emit_tracepoint_definition(
        self,
        kind: str,
        component_id: str,
        name: str,
        origin: str,
        description: str,
    ) -> None:
        try:
            if not self.enabled or not _text(description):
                return
            identity = tuple(_identity(value) for value in (kind, component_id, name, origin))
            if self._emitted_definitions is not None and identity in self._emitted_definitions:
                return
            sent = self._write_record(
                "_tracepoint_definition",
                {
                    "kind": identity[0],
                    "component_id": identity[1],
                    "name": identity[2],
                    "origin": identity[3],
                    "description": description,
                },
            )
            if sent:
                if self._emitted_definitions is None:
                    self._emitted_definitions = set()
                if len(self._emitted_definitions) < _MAX_DEFINITIONS:
                    self._emitted_definitions.add(identity)
        except (MemoryError, SystemError):
            raise
        except Exception:
            self._log_failure("Failed to serialize tracepoint definition %s", name)

    def event(
        self,
        name: str,
        /,
        *,
        timestamp_ns: int | None = None,
        tracepoint_description: str = "",
        **fields: Any,
    ) -> None:
        if not self.enabled:
            return
        try:
            trace_id = _identity(fields.pop("trace_id", "")) or _identity(_trace_id_var.get())
            component_id = (
                _identity(fields.pop("component_id", ""))
                or _identity(_component_var.get())
                or _identity(self.default_component)
            )
            origin = _text(fields.get("origin", "built-in"))
            self._emit_tracepoint_definition("event", component_id, name, origin, tracepoint_description)
            stack = _span_stack_var.get()
            if trace_id:
                fields["trace_id"] = trace_id
                fields.setdefault("request_id", trace_id)
            if component_id:
                fields["component_id"] = component_id
            if stack:
                fields.setdefault("span_id", stack[-1])
            self._write_record(name, fields, timestamp_ns)
        except (MemoryError, SystemError):
            raise
        except Exception:
            self._log_failure("Failed to serialize trace event %s", name)

    def trace_context(self, trace_id: str, *, component_id: str = "") -> _TraceContext | nullcontext:
        if not self.enabled:
            return _NULL_CONTEXT
        return _TraceContext(self, trace_id, component_id)

    def start_span(
        self,
        name: str,
        /,
        *,
        component_id: str = "",
        origin: str = "user",
        tracepoint_description: str = "",
        **fields: Any,
    ) -> _SpanToken | None:
        """Begin a detached span. Store only this trace token across flat stages.

        Does not change ambient context or retain caller frames/business objects.
        Pass the token to end_span, optionally with status and an exception name.
        Emission is immediate, not deferred: call outside business locks.
        """
        if not self.enabled:
            return None
        try:
            stack = _span_stack_var.get()
            component = _identity(component_id) or _identity(_component_var.get()) or _identity(self.default_component)
            self._emit_tracepoint_definition("span", component, name, origin, tracepoint_description)
            common = {
                "span_id": uuid.uuid4().hex,
                "parent_span_id": stack[-1] if stack else "",
                "component_id": component,
                "origin": _identity(origin),
                "span_name": _identity(name),
                "trace_id": _identity(fields.get("trace_id")) or _identity(_trace_id_var.get()),
            }
            if common["trace_id"]:
                common["request_id"] = common["trace_id"]
            extra_ids = {key: _identity(fields[key]) for key in _IDENTITY_FIELDS if key in fields and key not in common}
            attributes = _json_value({key: value for key, value in fields.items() if key not in _IDENTITY_FIELDS})
            common.update(extra_ids)
            common.update((key, value) for key, value in attributes.items() if key not in common)
            location = inspect.currentframe()
            caller = None
            try:
                caller = location.f_back if location is not None else None
                if caller is not None and caller.f_code is _SpanScope.__enter__.__code__:
                    caller = caller.f_back
                if caller is not None:
                    common.setdefault("file", caller.f_code.co_filename)
                    common.setdefault("function", caller.f_code.co_name)
                    common.setdefault("line", caller.f_lineno)
            finally:
                del caller, location
            started_ns = self.monotonic_clock()
            if type(started_ns) is not int or started_ns.bit_length() > 64:
                return None
            token = _SpanToken(common, started_ns)
            self._write_record("span_begin", common)
            return token
        except (MemoryError, SystemError):
            raise
        except Exception:
            self._log_failure("Failed to begin trace span %s", _text(name))
            return None

    def end_span(self, token: _SpanToken | None, *, status: str = "ok", exception_name: str = "") -> None:
        """End a detached span once; no business exception object is needed."""
        if not self.enabled or token is None or token.ended:
            return
        try:
            token.ended = True
            ended_ns = self.monotonic_clock()
            if type(ended_ns) is not int or ended_ns.bit_length() > 64:
                return
            fields = {"status": _text(status), "duration_ns": ended_ns - token.started_ns}
            if _text(exception_name):
                fields["error_type"] = _text(exception_name)
            fields.update((key, value) for key, value in token.fields.items() if key not in fields)
            self._write_record("span_end", fields)
        except (MemoryError, SystemError):
            raise
        except Exception:
            self._log_failure("Failed to end trace span %s", token.fields.get("span_name", ""))

    def span(self, name: str, /, **fields: Any) -> _SpanScope | nullcontext:
        if not self.enabled:
            return _NULL_CONTEXT
        return _SpanScope(self, name, fields)

    def async_span(self, name: str, /, **fields: Any) -> _SpanScope | nullcontext:
        if not self.enabled:
            return _NULL_CONTEXT
        return self.span(name, **fields)

    def span_decorator(
        self,
        name: str,
        /,
        *,
        component_id: str = "",
        origin: str = "user",
        tracepoint_description: str = "",
        **fields: Any,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorate(function: Callable[P, R]) -> Callable[P, R]:
            if inspect.iscoroutinefunction(function):

                @functools.wraps(function)
                async def async_wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
                    async with self.async_span(
                        name,
                        component_id=component_id,
                        origin=origin,
                        tracepoint_description=tracepoint_description,
                        **fields,
                    ):
                        return await function(*args, **kwargs)

                return async_wrapped

            @functools.wraps(function)
            def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
                with self.span(
                    name,
                    component_id=component_id,
                    origin=origin,
                    tracepoint_description=tracepoint_description,
                    **fields,
                ):
                    return function(*args, **kwargs)

            return wrapped

        return decorate

    def flow_send(self, edge_id: str, flow_id: str, **fields: Any) -> None:
        self.event("flow_send", edge_id=edge_id, flow_id=flow_id, **fields)

    def flow_receive(self, edge_id: str, flow_id: str, **fields: Any) -> None:
        self.event("flow_receive", edge_id=edge_id, flow_id=flow_id, **fields)


def get_trace_emitter(name: str, *, component_id: str = "", enabled: bool | None = None) -> TraceEmitter:
    enabled = _enabled() if enabled is None else enabled
    logger = create_trace_logger(name) if enabled else None
    if enabled:
        # Structured events are INFO; the legacy logger factory keeps WARNING.
        logger.setLevel(_level() if "IB_TRACE_LOG_LEVEL" in os.environ else logging.INFO)
    return TraceEmitter(
        logger,
        default_component=component_id,
        enabled=enabled,
    )


_default_emitter = get_trace_emitter("ib_trace.user")


def trace_context(trace_id: str, *, component_id: str = ""):
    return _default_emitter.trace_context(trace_id, component_id=component_id)


def span(name: str, /, *, component_id: str = "", tracepoint_description: str = "", **fields: Any):
    return _default_emitter.span(
        name,
        component_id=component_id,
        tracepoint_description=tracepoint_description,
        **fields,
    )


def start_span(name: str, /, **fields: Any):
    return _default_emitter.start_span(name, **fields)


def end_span(token, *, status: str = "ok", exception_name: str = "") -> None:
    _default_emitter.end_span(token, status=status, exception_name=exception_name)


def mark(name: str, /, *, tracepoint_description: str = "", **fields: Any) -> None:
    fields.setdefault("origin", "user")
    _default_emitter.event(name, tracepoint_description=tracepoint_description, **fields)


def flow_send(edge_id: str, flow_id: str, **fields: Any) -> None:
    fields.setdefault("origin", "user")
    _default_emitter.flow_send(edge_id, flow_id, **fields)


def flow_receive(edge_id: str, flow_id: str, **fields: Any) -> None:
    fields.setdefault("origin", "user")
    _default_emitter.flow_receive(edge_id, flow_id, **fields)

import asyncio
import contextvars
import json
import logging
import sys
import types
import weakref
from collections.abc import Mapping

import pytest

from ibrobot_tracing import instrumentation
from ibrobot_tracing.instrumentation import TraceEmitter, create_trace_logger, get_trace_emitter


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def emitter():
    logger = logging.Logger("test", level=logging.INFO)
    handler = ListHandler()
    logger.addHandler(handler)
    ticks = iter(range(1, 100))
    return TraceEmitter(logger, clock=lambda: next(ticks), monotonic_clock=lambda: next(ticks)), handler


def payloads(handler):
    return [json.loads(message.removeprefix("IBTRACE1 ")) for message in handler.messages]


def test_nested_spans_maintain_parent_and_restore_context():
    trace, handler = emitter()

    with trace.trace_context("req-1", component_id="policy"):
        with trace.span("outer"), trace.span("inner"):
            trace.event("mark")
        trace.event("after")

    records = payloads(handler)
    outer_id = records[0]["fields"]["span_id"]
    inner_id = records[1]["fields"]["span_id"]
    assert records[1]["fields"]["parent_span_id"] == outer_id
    assert records[2]["fields"]["span_id"] == inner_id
    assert "span_id" not in records[-1]["fields"]


def test_span_records_errors():
    trace, handler = emitter()

    with pytest.raises(ValueError), trace.span("broken"):
        raise ValueError("bad")

    end = payloads(handler)[-1]
    assert end["fields"]["status"] == "error"
    assert end["fields"]["error_type"] == "ValueError"


def test_large_attributes_cannot_consume_identity_budget():
    trace, stream = emitter()
    with trace.trace_context("request-original", component_id="owner"):
        trace.flow_send("edge", "flow-original", huge=["x" * 1024] * 500)
    row = payloads(stream)[0]
    assert row["fields"]["trace_id"] == row["fields"]["request_id"] == "request-original"
    assert row["fields"]["flow_id"] == "flow-original"
    assert row["fields"]["edge_id"] == "edge"
    assert len(stream.messages[0]) <= 16_384


def test_detached_span_preserves_ids_after_large_attributes():
    trace, handler = emitter()
    token = trace.start_span("work", huge=["x" * 1024] * 500, flow_id="flow-original", edge_id="edge-original")
    trace.end_span(token)
    records = payloads(handler)
    assert len(records) == 2
    for record in records:
        assert record["fields"]["flow_id"] == "flow-original"
        assert record["fields"]["edge_id"] == "edge-original"


def test_oversized_identity_drops_record_without_aliased_id():
    trace, stream = emitter()
    trace.event("bad", trace_id="same-prefix" * 500)
    trace.event("valid", trace_id="original")
    rows = payloads(stream)
    assert [row["event"] for row in rows] == ["valid"]


def test_invalid_nested_context_never_borrows_parent_identity():
    trace, handler = emitter()
    with trace.trace_context("parent"):
        with trace.trace_context("oversize" * 200):
            trace.event("unassociated")
        trace.event("parent-again")
    records = payloads(handler)
    assert "trace_id" not in records[0]["fields"]
    assert records[1]["fields"]["trace_id"] == "parent"


@pytest.mark.parametrize("configured", [None, "0", "false", "1"])
def test_disabled_emitter_does_not_initialize_optional_handler(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv("IB_TRACE_ENABLED", raising=False)
    else:
        monkeypatch.setenv("IB_TRACE_ENABLED", configured)

    def unexpected_handler(_name):
        raise AssertionError("disabled tracing must not initialize LTTng")

    monkeypatch.setattr("ibrobot_tracing.instrumentation.create_trace_logger", unexpected_handler)
    with monkeypatch.context() as patch:
        patch.setattr(logging, "getLogger", unexpected_handler)
        trace = get_trace_emitter("ib_trace.disabled", enabled=False if configured == "1" else None)
    assert trace.enabled is False
    assert trace.logger is None
    with trace.span("ignored"):
        trace.event("ignored")


def test_enabled_emitter_initializes_handler(monkeypatch):
    names = []
    logger = logging.Logger("enabled")

    def handler(name):
        names.append(name)
        return logger

    monkeypatch.setattr("ibrobot_tracing.instrumentation.create_trace_logger", handler)
    trace = get_trace_emitter("ib_trace.enabled", enabled=True)
    assert trace.enabled and trace.logger is logger
    assert names == ["ib_trace.enabled"]


class Unprintable:
    def __str__(self):
        raise ValueError("cannot stringify trace field")


@pytest.mark.parametrize("failure", ["list_cycle", "dict_cycle", "str"])
@pytest.mark.parametrize("diagnostic_fails", [False, True])
def test_span_exit_serialization_does_not_mask_business_error(monkeypatch, failure, diagnostic_fails):
    trace, handler = emitter()
    values = []
    business_error = RuntimeError("business failure")
    diagnostics = []

    def debug(*args, **kwargs):
        diagnostics.append(args)
        if diagnostic_fails:
            raise OSError("diagnostic handler failed")

    monkeypatch.setattr(trace.logger, "debug", debug)
    with trace.trace_context("req-1", component_id="outer"):
        with pytest.raises(RuntimeError) as caught, trace.span("broken", component_id="inner", values=values):
            if failure == "list_cycle":
                values.append(values)
            elif failure == "dict_cycle":
                cycle = {}
                cycle["self"] = cycle
                values.append(cycle)
            else:
                values.append(Unprintable())
            raise business_error
        assert caught.value is business_error
        trace.event("after")

    records = payloads(handler)
    assert [record["event"] for record in records] == ["span_begin", "span_end", "after"]
    assert records[1]["fields"]["values"] == []
    assert records[1]["fields"]["error_type"] == "RuntimeError"
    assert records[-1]["fields"] == {"trace_id": "req-1", "request_id": "req-1", "component_id": "outer"}
    assert not diagnostics


@pytest.mark.parametrize("key", ["value", "trace_id", "component_id", "origin"])
def test_event_isolates_all_field_conversions(key):
    trace, handler = emitter()

    trace.event("bad", **{key: Unprintable()})
    trace.event("after")

    records = payloads(handler)
    assert [record["event"] for record in records] == (
        ["after"] if key in {"trace_id", "component_id", "origin"} else ["bad", "after"]
    )
    assert records[0]["fields"].get(key, "<unsupported>") == "<unsupported>"


@pytest.mark.parametrize("definition", [False, True])
@pytest.mark.parametrize("failure", ["clock", "handler"])
def test_emission_isolates_payload_and_diagnostic_failures(monkeypatch, definition, failure):
    trace, handler = emitter()
    diagnostics = []

    def fail(*args, **kwargs):
        raise OSError("trace unavailable")

    def debug(*args, **kwargs):
        diagnostics.append(args)
        raise ValueError("diagnostics unavailable")

    monkeypatch.setattr(trace.logger, "debug", debug)
    if failure == "clock":
        monkeypatch.setattr(trace, "clock", fail)
    else:
        monkeypatch.setattr(trace.logger, "info", fail)

    if definition:
        trace._emit_tracepoint_definition("span", "component", "work", "user", "description")
    else:
        trace.event("work")

    assert not handler.messages
    assert len(diagnostics) == 1


@pytest.mark.parametrize("error_type", [MemoryError, SystemError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("definition", [False, True])
@pytest.mark.parametrize("failure", ["handler", "diagnostic"])
def test_emission_propagates_fatal_errors(monkeypatch, error_type, definition, failure):
    trace, _ = emitter()
    fatal_error = error_type("fatal")
    diagnostics = []

    def info(*args, **kwargs):
        if failure == "handler":
            raise fatal_error
        raise ValueError("serialization failed")

    def debug(*args, **kwargs):
        diagnostics.append(args)
        raise fatal_error

    monkeypatch.setattr(trace.logger, "info", info)
    monkeypatch.setattr(trace.logger, "debug", debug)

    with pytest.raises(error_type) as caught:
        if definition:
            trace._emit_tracepoint_definition("span", "component", "work", "user", "description")
        else:
            trace.event("work")
    assert caught.value is fatal_error
    assert len(diagnostics) == (1 if failure == "diagnostic" else 0)


def test_span_releases_caller_locals_before_yield():
    trace, handler = emitter()

    class Resource:
        pass

    def enter_span():
        resource = Resource()
        reference = weakref.ref(resource)
        manager = trace.span("work")
        line = sys._getframe().f_lineno + 1
        manager.__enter__()
        return manager, reference, line

    manager, reference, line = enter_span()
    try:
        assert reference() is None
        fields = payloads(handler)[0]["fields"]
        assert fields["file"] == __file__
        assert fields["function"] == "enter_span"
        assert fields["line"] == line
    finally:
        manager.__exit__(None, None, None)


def test_async_decorator_measures_coroutine_execution():
    trace, handler = emitter()

    @trace.span_decorator("async-work")
    async def work():
        trace.event("inside")
        await asyncio.sleep(0)

    asyncio.run(work())

    assert [record["event"] for record in payloads(handler)] == ["span_begin", "inside", "span_end"]


def test_trace_context_arguments_are_optional_for_unscoped_events():
    trace, handler = emitter()

    trace.event("unscoped")
    with trace.trace_context("req-1"):
        trace.event("request-scoped")

    records = payloads(handler)
    assert records[0]["fields"] == {}
    assert records[1]["fields"]["trace_id"] == "req-1"
    assert "component_id" not in records[1]["fields"]


def test_disabled_emitter_is_a_noop():
    trace, handler = emitter()
    trace.enabled = False

    with trace.trace_context("req-1"), trace.span("work"):
        trace.event("mark")

    assert not handler.messages


def test_create_trace_logger_preserves_root_log_level(monkeypatch):
    logger = logging.getLogger("ib_trace.test_root_level")
    logger.handlers.clear()
    root_level = logging.WARNING
    monkeypatch.setattr(logging.root, "level", root_level)

    class Handler(logging.Handler):
        def __init__(self):
            super().__init__()
            logging.root.setLevel(logging.NOTSET)

    module = types.ModuleType("lttngust.loghandler")
    module._Handler = Handler
    monkeypatch.setitem(sys.modules, "lttngust.loghandler", module)

    create_trace_logger(logger.name)

    assert logging.root.level == root_level


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("business_fails", [False, True])
def test_span_keyword_collisions_preserve_body_and_business_error(enabled, business_fails):
    trace, handler = emitter()
    trace.enabled = enabled
    attrs = {"status": "supplied", "duration_ns": 123, "timestamp_ns": 456, "span_name": "supplied"}
    error = ValueError("business")
    error.details = attrs
    ran = []

    def work():
        with trace.span("actual", **attrs):
            ran.append(True)
            if business_fails:
                raise error

    if business_fails:
        with pytest.raises(ValueError) as caught:
            work()
        assert caught.value is error
        assert error.details is attrs
    else:
        work()
    assert ran == [True]
    assert attrs == {"status": "supplied", "duration_ns": 123, "timestamp_ns": 456, "span_name": "supplied"}
    if enabled:
        end = payloads(handler)[-1]["fields"]
        assert end["span_name"] == "actual"
        assert end["status"] == ("error" if business_fails else "ok")
        assert end["timestamp_ns"] == 456
        assert end["duration_ns"] != 123


def test_module_helpers_allow_origin_override(monkeypatch):
    trace, handler = emitter()
    monkeypatch.setattr(instrumentation, "_default_emitter", trace)
    instrumentation.mark("work", origin="custom")
    instrumentation.flow_send("edge", "flow", origin="custom")
    instrumentation.flow_receive("edge", "flow", origin="custom")
    with instrumentation.span("work", origin="custom"):
        pass
    assert all(record["fields"]["origin"] == "custom" for record in payloads(handler))


def test_disabled_paths_do_not_snapshot_inspect_generate_ids_or_read_clocks(monkeypatch):
    trace, handler = emitter()
    trace.enabled = False

    def unexpected(*args, **kwargs):
        pytest.fail("disabled tracer performed work")

    for owner, name in (
        (instrumentation, "_json_value"),
        (instrumentation.uuid, "uuid4"),
        (instrumentation.inspect, "currentframe"),
        (trace, "clock"),
        (trace, "monotonic_clock"),
        (instrumentation, "_set_context"),
        (instrumentation, "_SpanScope"),
        (instrumentation, "_TraceContext"),
        (instrumentation, "_SpanToken"),
        (instrumentation, "nullcontext"),
    ):
        monkeypatch.setattr(owner, name, unexpected)

    @trace.span_decorator("decorated", value=Unprintable())
    def work():
        return 42

    async def run():
        async with trace.async_span("async", value=Unprintable()):
            return work()

    with trace.trace_context(Unprintable(), component_id=Unprintable()), trace.span("sync", value=Unprintable()):
        trace.event("event", value=Unprintable())
        trace._emit_tracepoint_definition("span", "worker", "work", "user", "description")
        assert trace.start_span("detached", value=Unprintable()) is None
        trace.end_span(None)
        assert asyncio.run(run()) == 42
    assert trace._emitted_definitions is None
    assert not handler.messages


def test_disabled_scopes_reuse_one_sync_and_async_context(monkeypatch):
    trace = get_trace_emitter("disabled", enabled=False)
    other = get_trace_emitter("other", enabled=False)
    monkeypatch.setattr(instrumentation, "_default_emitter", trace)
    scope = trace.span("outer")
    assert trace.span("again") is scope
    assert trace.async_span("async") is scope
    assert trace.trace_context("request") is scope
    assert other.span("other-emitter") is scope
    assert instrumentation.span("module") is scope
    assert instrumentation.trace_context("module-request") is scope

    async def run():
        async with scope as first, scope as second:
            assert first is second is None

    with scope as outer, scope as inner:
        assert outer is inner is None
        asyncio.run(run())
    with scope:
        pass


class Hostile(Mapping):
    def fail(self, *args, **kwargs):
        pytest.fail("tracing called a business object hook")

    __str__ = __bool__ = __iter__ = __len__ = __getitem__ = items = fail


class HostileList(list):
    __iter__ = __bool__ = Hostile.fail


class HostileDict(dict):
    items = __iter__ = __bool__ = Hostile.fail


class HostileString(str):
    __str__ = __bool__ = Hostile.fail


class HostileInt(int):
    __str__ = __bool__ = Hostile.fail


@pytest.mark.parametrize(
    "value",
    [Hostile(), HostileList(), HostileDict(), HostileString("text"), HostileInt(1)],
    ids=["mapping", "list", "dict", "string", "int"],
)
def test_unknown_values_never_execute_business_hooks(value):
    trace, handler = emitter()
    attrs = {"value": value, "nested": [value, {"value": value}]}
    trace.event("event", trace_id=value, component_id=value, origin=value, **attrs)
    with trace.trace_context(value, component_id=value), trace.span("span", origin=value, **attrs):
        pass
    assert attrs["value"] is value
    for record in payloads(handler):
        assert record["fields"]["value"] == "<unsupported>"
        assert record["fields"]["nested"] == ["<unsupported>", {"value": "<unsupported>"}]


def test_json_snapshot_is_bounded_and_keeps_input_untouched():
    trace, handler = emitter()
    cycle = []
    cycle.append(cycle)
    mapping = {}
    mapping["self"] = mapping
    deep = 0
    for _ in range(100):
        deep = [deep]
    attrs = {
        "cycle": cycle,
        "mapping": mapping,
        "deep": deep,
        "nonfinite": float("inf"),
        "huge_int": 1 << 10_000,
        "many": list(range(10_000)),
        "huge": "\U0001f642" * 100_000,
    }
    trace.event("bounded", **attrs)
    fields = payloads(handler)[0]["fields"]
    assert fields["cycle"] == ["<truncated>"]
    assert fields["mapping"] == {"self": "<truncated>"}
    assert fields["nonfinite"] == fields["huge_int"] == "<unsupported>"
    assert len(fields["many"]) < instrumentation._MAX_FIELDS
    assert cycle[0] is cycle and mapping["self"] is mapping
    assert len(attrs["many"]) == 10_000
    assert len(handler.messages[0].encode()) <= instrumentation._MAX_RECORD_BYTES
    trace.event("large_strings", values=["\U0001f642" * 100_000] * 10_000)
    assert len(handler.messages) == 2
    assert len(handler.messages[1].encode()) <= instrumentation._MAX_RECORD_BYTES


@pytest.mark.parametrize("clock_name", ["clock", "monotonic_clock"])
@pytest.mark.parametrize("when", ["begin", "end"])
@pytest.mark.parametrize("business_fails", [False, True])
def test_span_clock_failure_only_drops_records(monkeypatch, clock_name, when, business_fails):
    trace, handler = emitter()
    error = ValueError("business")
    error.details = {"unchanged": []}
    original_attrs = dict(error.__dict__)
    ran = []
    original_clock = getattr(trace, clock_name)

    def fail():
        raise OSError("clock unavailable")

    def work():
        if when == "begin":
            monkeypatch.setattr(trace, clock_name, fail)
        with trace.span("work", component_id="inner"):
            ran.append(True)
            if when == "end":
                monkeypatch.setattr(trace, clock_name, fail)
            if business_fails:
                raise error

    with trace.trace_context("request", component_id="outer"):
        if business_fails:
            with pytest.raises(ValueError) as caught:
                work()
            assert caught.value is error
            assert error.__dict__ == original_attrs
            assert error.__context__ is None and error.__cause__ is None
        else:
            work()
        monkeypatch.setattr(trace, clock_name, original_clock)
        trace.event("after")
    assert ran == [True]
    assert payloads(handler)[-1]["fields"] == {
        "trace_id": "request",
        "request_id": "request",
        "component_id": "outer",
    }


@pytest.mark.parametrize("scope", ["span", "trace_context"])
@pytest.mark.parametrize("operation", ["get", "set", "reset"])
def test_context_failures_are_best_effort_and_restore_other_tokens(monkeypatch, scope, operation):
    trace, _ = emitter()
    variable = instrumentation._component_var

    class FaultyContext:
        def get(self):
            if operation == "get":
                raise RuntimeError("context get failed")
            return variable.get()

        def set(self, value):
            if operation == "set":
                raise RuntimeError("context set failed")
            return variable.set(value)

        def reset(self, token):
            if operation == "reset":
                raise RuntimeError("context reset failed")
            return variable.reset(token)

    with trace.trace_context("outer", component_id="outer"):
        monkeypatch.setattr(instrumentation, "_component_var", FaultyContext())
        error = ValueError("business")
        with pytest.raises(ValueError) as caught, getattr(trace, scope)("inner", component_id="inner"):
            raise error
        assert caught.value is error
        assert instrumentation._trace_id_var.get() == "outer"
        assert instrumentation._span_stack_var.get() == ()
        assert variable.get() == "outer"


@pytest.mark.parametrize(
    "error_type", [ValueError, StopIteration, KeyboardInterrupt, SystemExit, asyncio.CancelledError]
)
@pytest.mark.parametrize("enabled", [False, True])
def test_business_exception_object_is_never_modified(error_type, enabled):
    trace, handler = emitter()
    trace.enabled = enabled

    class ErrorMeta(type):
        @property
        def __name__(cls):
            pytest.fail("tracing invoked an exception metaclass hook")

    class BusinessError(error_type, metaclass=ErrorMeta):
        __str__ = __bool__ = Hostile.fail

        def __setattr__(self, name, value):
            if name in {"__traceback__", "__context__", "__cause__"}:
                pytest.fail("tracing modified business exception attributes")
            super().__setattr__(name, value)

    error = BusinessError("unchanged", {"payload": []})
    attrs = {"detail": []}
    error.attrs = attrs
    with pytest.raises(BusinessError) as caught, trace.trace_context("request"), trace.span("work"):
        raise error
    assert caught.value is error
    assert error.attrs is attrs
    assert error.args == ("unchanged", {"payload": []})
    assert error.__context__ is None and error.__cause__ is None
    if enabled:
        assert payloads(handler)[-1]["fields"]["error_type"] == "BusinessError"
    else:
        assert not handler.messages


@pytest.mark.parametrize("decorated", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_async_error_preserves_identity_and_restores_context(decorated, enabled):
    trace, handler = emitter()
    trace.enabled = enabled
    error = asyncio.CancelledError("business cancellation")
    error.attrs = {"data": []}

    async def body():
        await asyncio.sleep(0)
        raise error

    async def run():
        with trace.trace_context("outer", component_id="outer"):
            with pytest.raises(asyncio.CancelledError) as caught:
                if decorated:
                    await trace.span_decorator("async", component_id="inner")(body)()
                else:
                    async with trace.async_span("async", component_id="inner", status="collision"):
                        await body()
            assert caught.value is error
            assert error.__dict__ == {"attrs": {"data": []}}
            trace.event("after")

    asyncio.run(run())
    records = payloads(handler)
    if enabled:
        assert records[-2]["fields"]["error_type"] == "CancelledError"
        assert records[-1]["fields"] == {"trace_id": "outer", "request_id": "outer", "component_id": "outer"}
    else:
        assert not records


def test_tracer_diagnostics_do_not_format_chained_business_exceptions(monkeypatch):
    trace, handler = emitter()
    trace.logger.setLevel(logging.DEBUG)
    monkeypatch.setattr(handler, "emit", lambda record: handler.messages.append(handler.format(record)))

    class BusinessError(ValueError):
        __str__ = Hostile.fail

    def fail():
        raise OSError("trace clock failed")

    error = BusinessError("business")
    with pytest.raises(BusinessError) as caught, trace.span("work"):
        trace.clock = fail
        raise error
    assert caught.value is error
    assert handler.messages[-1] == "Failed to end trace span work"


def test_detached_span_uses_original_context_without_leaking_or_retaining_attrs():
    trace, handler = emitter()
    resource = Unprintable()
    reference = weakref.ref(resource)
    attrs = {"values": [1], "resource": resource}
    with trace.trace_context("first", component_id="worker"), trace.span("parent"):
        token = trace.start_span("detached", **attrs)
        assert instrumentation._span_stack_var.get() == (payloads(handler)[0]["fields"]["span_id"],)
    del resource, attrs
    assert reference() is None

    def finish():
        with trace.trace_context("second", component_id="other"):
            trace.end_span(token, status="error", exception_name="ValueError")
            trace.end_span(token)
            trace.event("after")

    contextvars.Context().run(finish)
    records = payloads(handler)
    begin, end = records[1], records[3]
    for name in ("span_id", "parent_span_id", "trace_id", "request_id", "component_id", "values"):
        assert begin["fields"][name] == end["fields"][name]
    assert end["fields"]["status"] == "error"
    assert end["fields"]["error_type"] == "ValueError"
    assert records[-1]["fields"] == {"trace_id": "second", "request_id": "second", "component_id": "other"}


@pytest.mark.parametrize("failure", ["handler", "no_handler", "filtered", "level"])
def test_definition_retries_after_unsuccessful_output(monkeypatch, failure):
    trace, handler = emitter()
    if failure == "handler":

        def fail(record):
            raise OSError("sink unavailable")

        monkeypatch.setattr(handler, "emit", fail)
    elif failure == "no_handler":
        trace.logger.removeHandler(handler)
    elif failure == "filtered":
        handler.addFilter(lambda record: False)
    else:
        handler.setLevel(logging.WARNING)
    trace.event("ready", tracepoint_description="description")
    assert not trace._emitted_definitions
    monkeypatch.undo()
    trace.logger.addHandler(handler)
    handler.filters.clear()
    handler.setLevel(logging.INFO)
    trace.event("ready", tracepoint_description="description")
    assert len(trace._emitted_definitions) == 1
    assert len([record for record in payloads(handler) if record["event"] == "_tracepoint_definition"]) == 1


def test_definition_cache_is_lazy_and_bounded():
    trace, handler = emitter()
    trace.clock = trace.monotonic_clock = lambda: 1
    assert trace._emitted_definitions is None
    for index in range(instrumentation._MAX_DEFINITIONS + 10):
        trace.event(f"event-{index}", tracepoint_description="description")
    assert len(trace._emitted_definitions) == instrumentation._MAX_DEFINITIONS
    assert len(payloads(handler)) == 2 * (instrumentation._MAX_DEFINITIONS + 10)


@pytest.mark.parametrize("level", [None, "invalid", "INFO"])
def test_logger_opt_in_preserves_business_configuration_and_legacy_default(monkeypatch, level):
    if level is None:
        monkeypatch.delenv("IB_TRACE_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("IB_TRACE_LOG_LEVEL", level)
    business = logging.getLogger(f"business.logger.{level}")
    business.setLevel(logging.ERROR)
    business.propagate = True
    handler = ListHandler()
    business.addHandler(handler)
    module = types.ModuleType("lttngust.loghandler")
    module._Handler = ListHandler
    monkeypatch.setitem(sys.modules, "lttngust.loghandler", module)
    trace_logger = create_trace_logger(business.name)
    assert trace_logger is not business
    assert trace_logger.name == f"{business.name}._ibrobot_trace"
    assert trace_logger.level == (logging.INFO if level == "INFO" else logging.WARNING)
    assert business.level == logging.ERROR and business.propagate
    assert business.handlers == [handler]
    assert create_trace_logger(business.name) is trace_logger
    assert len(trace_logger.handlers) == 1

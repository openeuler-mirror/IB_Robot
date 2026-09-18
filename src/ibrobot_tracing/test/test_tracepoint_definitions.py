import asyncio
import json
import logging

import pytest

from ibrobot_tracing import instrumentation
from ibrobot_tracing.analysis import AnalysisService
from ibrobot_tracing.cli import main
from ibrobot_tracing.definitions import BUILTIN_TRACEPOINT_REGISTRY
from ibrobot_tracing.instrumentation import TraceEmitter
from ibrobot_tracing.model import EventOrigin, TraceDataset, TraceEvent, TracepointDefinition, TraceTopology
from ibrobot_tracing.parsing import parse_lines
from ibrobot_tracing.query import QueryService, TracepointQuery, stable_tracepoint_id
from ibrobot_tracing.topology import bind_robot_topology, read_topology_manifest, write_topology_manifest


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def emitter():
    logger = logging.Logger("ib_trace.test", level=logging.INFO)
    handler = ListHandler()
    logger.addHandler(handler)
    ticks = iter(range(1, 1_000))
    return TraceEmitter(logger, clock=lambda: next(ticks), monotonic_clock=lambda: next(ticks)), handler


def test_scheduler_tracepoints_have_builtin_descriptions():
    assert (
        "event",
        "action_dispatcher.execute",
        "safe_stop_topic_publish",
        "built-in",
    ) in BUILTIN_TRACEPOINT_REGISTRY
    assert ("span", "global_scheduler", "scheduler_dispatch", "built-in") in BUILTIN_TRACEPOINT_REGISTRY
    assert ("span", "policy.postprocess", "result_encoding", "built-in") in BUILTIN_TRACEPOINT_REGISTRY


def payloads(handler):
    return [json.loads(message.removeprefix("IBTRACE1 ")) for message in handler.messages]


def test_event_definition_is_private_context_free_once_per_identity():
    trace, handler = emitter()

    with trace.trace_context("req-1", component_id="policy.preprocess"), trace.span("parent"):
        trace.event("ready", origin="user", tracepoint_description="输入张量已准备完成")
        trace.event("ready", origin="user", tracepoint_description="不会覆盖首个定义")

    records = payloads(handler)
    definitions = [record for record in records if record["event"] == "_tracepoint_definition"]
    ready_events = [record for record in records if record["event"] == "ready"]

    assert len(definitions) == 1
    assert definitions[0]["fields"] == {
        "kind": "event",
        "component_id": "policy.preprocess",
        "name": "ready",
        "origin": "user",
        "description": "输入张量已准备完成",
    }
    assert all("tracepoint_description" not in record["fields"] for record in ready_events)
    assert all(record["fields"]["trace_id"] == "req-1" for record in ready_events)


def test_span_async_decorator_and_module_helpers_accept_descriptions(monkeypatch):
    trace, handler = emitter()
    monkeypatch.setattr(instrumentation, "_default_emitter", trace)

    with trace.span("sync", component_id="worker", tracepoint_description="同步处理"):
        pass

    async def run():
        async with trace.async_span("async", component_id="worker", tracepoint_description="异步处理"):
            pass

        @trace.span_decorator("decorated", component_id="worker", tracepoint_description="装饰器处理")
        async def decorated():
            return None

        await decorated()

    asyncio.run(run())
    instrumentation.mark("module_mark", tracepoint_description="模块事件")
    with instrumentation.span("module_span", tracepoint_description="模块区间"):
        pass

    definitions = {
        (record["fields"]["kind"], record["fields"]["name"]): record["fields"]["description"]
        for record in payloads(handler)
        if record["event"] == "_tracepoint_definition"
    }
    assert definitions == {
        ("event", "module_mark"): "模块事件",
        ("span", "async"): "异步处理",
        ("span", "decorated"): "装饰器处理",
        ("span", "module_span"): "模块区间",
        ("span", "sync"): "同步处理",
    }


def test_empty_description_and_disabled_emitter_do_not_emit_metadata():
    trace, handler = emitter()
    trace.event("plain")
    with trace.span("plain"):
        pass
    trace.enabled = False
    trace.event("disabled", tracepoint_description="不可见")
    with trace.span("disabled", tracepoint_description="不可见"):
        pass

    assert all(record["event"] != "_tracepoint_definition" for record in payloads(handler))
    assert not trace._emitted_definitions


def test_parser_extracts_definition_and_excludes_private_record_from_events():
    trace, handler = emitter()
    trace.event(
        "ready",
        component_id="policy.preprocess",
        origin="user",
        tracepoint_description="输入张量已准备完成",
    )

    dataset = parse_lines(handler.messages)

    assert [event.name for event in dataset.events] == ["ready"]
    assert dataset.definitions == [
        TracepointDefinition("event", "policy.preprocess", "ready", "user", "输入张量已准备完成")
    ]
    assert "description" not in dataset.events[0].fields


def test_definition_metadata_does_not_consume_event_limit():
    trace, handler = emitter()
    trace.event("ready", component_id="worker", origin="user", tracepoint_description="准备完成")

    dataset = parse_lines(handler.messages, max_events=1)

    assert [event.name for event in dataset.events] == ["ready"]
    assert [definition.name for definition in dataset.definitions] == ["ready"]
    assert not dataset.warnings


def test_definition_component_is_inferred_from_logger_bound_event():
    dataset = TraceDataset(
        events=[
            TraceEvent(
                1,
                "ready",
                {"origin": "user"},
                origin=EventOrigin(provider="ib_trace.worker"),
                schema_version=1,
            )
        ],
        definitions=[TracepointDefinition("event", "", "ready", "user", "准备完成")],
    )
    topology = TraceTopology(logger_to_component={"ib_trace.worker": "worker"})

    result = AnalysisService().analyze_dataset(dataset, topology=topology)

    assert result.definitions == [TracepointDefinition("event", "worker", "ready", "user", "准备完成")]


def test_analysis_resolves_precedence_empty_fallback_and_deterministic_conflicts():
    runtime = [
        TracepointDefinition("event", "action_dispatcher.request", "dispatch_request", "built-in", "运行时乙"),
        TracepointDefinition("event", "action_dispatcher.request", "dispatch_request", "built-in", "运行时甲"),
    ]
    topology = TraceTopology(
        definitions=[
            TracepointDefinition("event", "action_dispatcher.request", "dispatch_request", "built-in", "清单说明"),
            TracepointDefinition("event", "worker", "manifest_only", "user", "仅清单说明"),
        ]
    )
    dataset = TraceDataset(
        events=[
            TraceEvent(
                1,
                "dispatch_request",
                {"component_id": "action_dispatcher.request", "origin": "built-in"},
                schema_version=1,
            ),
            TraceEvent(2, "obs_frame", {"component_id": "policy.observation", "origin": "built-in"}, schema_version=1),
            TraceEvent(3, "undocumented", {"component_id": "worker", "origin": "user"}, schema_version=1),
        ],
        definitions=runtime,
    )

    result = AnalysisService().analyze_dataset(dataset, topology=topology)
    by_identity = {definition.identity: definition for definition in result.definitions}

    assert by_identity[("event", "action_dispatcher.request", "dispatch_request", "built-in")].description == "运行时乙"
    assert by_identity[("event", "worker", "manifest_only", "user")].description == "仅清单说明"
    assert (
        by_identity[("event", "policy.observation", "obs_frame", "built-in")].description
        == (BUILTIN_TRACEPOINT_REGISTRY[("event", "policy.observation", "obs_frame", "built-in")])
    )
    assert by_identity[("event", "worker", "undocumented", "user")].description == ""
    assert len(result.warnings) == 1
    assert "selected runtime: '运行时乙'" in result.warnings[0]
    assert "运行时甲" in result.warnings[0]
    assert result.dataset.events == dataset.events
    assert all("description" not in event.fields for event in result.dataset.events)

    reversed_result = AnalysisService().analyze_dataset(
        TraceDataset(events=dataset.events, definitions=list(reversed(runtime))),
        topology=topology,
    )
    assert reversed_result.definitions == result.definitions
    assert reversed_result.warnings == result.warnings


def test_span_description_is_resolved_without_copying_it_to_span_record():
    definition = TracepointDefinition("span", "worker", "work", "user", "执行工作")
    dataset = TraceDataset(
        events=[
            TraceEvent(
                1,
                "span_begin",
                {"span_id": "one", "span_name": "work", "component_id": "worker", "origin": "user"},
                schema_version=1,
            ),
            TraceEvent(
                2,
                "span_end",
                {"span_id": "one", "span_name": "work", "component_id": "worker", "origin": "user"},
                schema_version=1,
            ),
        ],
        definitions=[definition],
    )

    result = AnalysisService().analyze_dataset(dataset)

    assert result.definitions == [definition]
    assert "description" not in result.spans[0].fields
    assert result.to_dict()["tracepoints"] == [definition.to_dict()]


def test_manifest_tracepoints_are_additive_in_schema_v2_and_optional_in_old_files(tmp_path):
    topology = bind_robot_topology({"name": "robot"})
    path = write_topology_manifest(tmp_path / "v2.json", topology)
    document = json.loads(path.read_text())

    assert document["schema_version"] == 2
    assert document["tracepoints"]
    assert read_topology_manifest(path).definitions == topology.definitions

    for version in (1, 2):
        legacy = tmp_path / f"old-v{version}.json"
        legacy.write_text(
            json.dumps(
                {
                    "schema_version": version,
                    "components": [],
                    "edges": [],
                    "logger_to_component": {},
                    "metadata": {},
                }
            )
        )
        assert read_topology_manifest(legacy).definitions == []


def test_builtin_registry_covers_current_action_dispatch_and_inference_tracepoints():
    expected = {
        ("event", "action_dispatcher.decode", "dispatch_result", "built-in"),
        ("event", "action_dispatcher.execute", "action_execute", "built-in"),
        ("event", "action_dispatcher.execute", "action_topic_publish", "built-in"),
        ("event", "action_dispatcher.execute", "first_action_execute", "built-in"),
        ("event", "action_dispatcher.queue", "queue_refill", "built-in"),
        ("event", "action_dispatcher.request", "dispatch_request", "built-in"),
        ("event", "policy", "dispatch_result", "built-in"),
        ("event", "policy.observation", "obs_frame", "built-in"),
        ("event", "policy.observation", "obs_receive", "built-in"),
        ("event", "policy.observation", "obs_sample", "built-in"),
        ("span", "action_dispatcher.decode", "dispatch_decode", "built-in"),
        ("span", "action_dispatcher.execute", "action_execute", "built-in"),
        ("span", "action_dispatcher.execute", "first_action_execute", "built-in"),
        ("span", "action_dispatcher.queue", "queue_refill", "built-in"),
        ("span", "cloud_inference", "model_call", "built-in"),
        ("span", "policy", "cloud_roundtrip", "built-in"),
        ("span", "policy", "policy_pipeline", "built-in"),
        ("span", "policy", "policy_total", "built-in"),
        ("span", "policy.inference", "model_call", "built-in"),
        ("span", "policy.observation", "observation_sampling", "built-in"),
        ("span", "policy.postprocess", "action_chunk_publish", "built-in"),
        ("span", "policy.postprocess", "postprocess", "built-in"),
        ("span", "policy.preprocess", "preprocess", "built-in"),
    }

    assert expected <= BUILTIN_TRACEPOINT_REGISTRY.keys()
    assert all(
        description and any("\u4e00" <= character <= "\u9fff" for character in description)
        for description in BUILTIN_TRACEPOINT_REGISTRY.values()
    )


def test_tracepoint_definition_rejects_unknown_kinds_and_empty_names():
    with pytest.raises(ValueError, match="kind"):
        TracepointDefinition("counter", "worker", "value", "user")
    with pytest.raises(ValueError, match="name"):
        TracepointDefinition("event", "worker", "", "user")


def test_tracepoint_query_stable_id_and_cli(tmp_path, capsys):
    definitions = [
        TracepointDefinition("event", "worker", "ready", "user", "准备完成"),
        TracepointDefinition("span", "worker", "work", "user", "执行工作"),
    ]
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[TraceEvent(1, "ready", {"component_id": "worker", "origin": "user"}, schema_version=1)],
            definitions=definitions,
        )
    )
    service = QueryService(result)

    page = service.tracepoints(TracepointQuery(kind="event", component_id="worker", limit=1))
    assert page.items == definitions[:1]
    assert stable_tracepoint_id(definitions[0]) == stable_tracepoint_id(
        TracepointDefinition("event", "worker", "ready", "user", "更新后的说明")
    )

    trace, handler = emitter()
    trace.event("ready", component_id="worker", origin="user", tracepoint_description="准备完成")
    source = tmp_path / "trace.log"
    source.write_text("\n".join(handler.messages))

    assert main(["tracepoints", str(source), "--kind", "event", "--component", "worker"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["id"].startswith("tracepoint:")
    assert output["description"] == "准备完成"

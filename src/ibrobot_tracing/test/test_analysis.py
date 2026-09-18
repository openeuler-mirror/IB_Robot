import json
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest

from ibrobot_tracing.analysis import AnalysisRequest, AnalysisService
from ibrobot_tracing.model import Component, DataFlowEdge, EventOrigin, TraceDataset, TraceEvent, TraceTopology
from ibrobot_tracing.parsing import parse_log
from ibrobot_tracing.projection import LatencyDistributionQuery, project_graph, project_latency_distribution
from ibrobot_tracing.query import EventQuery, QueryService, SpanQuery
from ibrobot_tracing.rendering import render_call_tree, render_graph, render_summary


def event(timestamp_ns, name, trace_id, **fields):
    return TraceEvent(timestamp_ns, name, {"trace_id": trace_id, **fields})


def builtin_event(timestamp_ns, name, trace_id, **fields):
    components = {
        "dispatch_request": "action_dispatcher.request",
        "first_action_execute": "action_dispatcher.execute",
        "dispatch_result": "policy",
    }
    return TraceEvent(
        timestamp_ns,
        name,
        {
            "trace_id": trace_id,
            "component_id": components.get(name, "policy.observation"),
            "origin": "built-in",
            **fields,
        },
        schema_version=1,
    )


def span_events(name, component_id, start_ms, end_ms, *, span_id="", origin="built-in", status="ok"):
    return [
        TraceEvent(
            timestamp_ms * 1_000_000,
            event_name,
            {
                "trace_id": "req-1",
                "span_id": span_id or name,
                "span_name": name,
                "component_id": component_id,
                "origin": origin,
                "status": status,
            },
            schema_version=1,
        )
        for timestamp_ms, event_name in ((start_ms, "span_begin"), (end_ms, "span_end"))
    ]


@pytest.fixture
def readable_trace(tmp_path, monkeypatch):
    log = tmp_path / "events.log"
    log.write_text(
        "\n".join(
            "IBTRACE1 " + json.dumps({"timestamp_ns": item.timestamp_ns, "event": item.name, "fields": item.fields})
            for item in span_events("model_call", "policy.inference", 1, 3)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("ibrobot_tracing.parsing.parse_ctf", lambda path, **kwargs: parse_log(path / "events.log"))
    return tmp_path


def test_analysis_without_optional_sidecar_keeps_events_spans_and_summary(readable_trace):
    result = AnalysisService().analyze(AnalysisRequest(readable_trace))

    assert QueryService(result).events().total == 2
    assert QueryService(result).spans().total == 1
    assert result.stage_summary["inference_ms"].mean == 2.0
    assert result.topology is None
    assert project_graph(result).topology_source == "observed"
    assert not result.warnings


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "contents",
    [
        b"{",
        b"\xff",
        b"null",
        b"[]",
        b'{"schema_version": 99}',
        b'{"schema_version": null}',
        b'{"components": null}',
        b'{"components": [{"component_id": []}]}',
        b'{"edges": [{}]}',
        b'{"logger_to_component": {"ib_trace.policy": []}}',
        b'{"metadata": []}',
        b'{"tracepoints": [{"kind": "unknown", "name": "work", "component_id": "c", "origin": "user"}]}',
    ],
)
def test_bad_optional_sidecar_keeps_summary_and_warns(readable_trace, contents, explicit):
    sidecar = readable_trace / "ibrobot-topology.json"
    sidecar.write_bytes(contents)

    result = AnalysisService().analyze(AnalysisRequest(readable_trace, topology_path=sidecar if explicit else None))

    assert QueryService(result).events().total == 2
    assert QueryService(result).spans().items[0].duration_ms == 2.0
    assert result.to_dict()["stages"]["inference_ms"]["mean"] == 2.0
    assert result.topology is None
    assert project_graph(result).topology_source == "observed"
    assert any("Ignoring topology manifest" in warning for warning in result.warnings)
    output = StringIO()
    render_summary(result, output)
    assert "Model call" in output.getvalue()


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError("read failed"), RuntimeError("unexpected")])
def test_sidecar_io_and_unexpected_errors_are_not_hidden(readable_trace, monkeypatch, error):
    sidecar = readable_trace / "ibrobot-topology.json"
    sidecar.write_text("{}")

    def fail_read(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(type(error), match=str(error)):
        AnalysisService().analyze(AnalysisRequest(readable_trace))


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), ValueError("unsafe source"), RuntimeError("reader failed")]
)
def test_bad_sidecar_does_not_hide_source_errors(readable_trace, monkeypatch, error):
    (readable_trace / "ibrobot-topology.json").write_text("{")

    def fail_read(*args, **kwargs):
        raise error

    monkeypatch.setattr("ibrobot_tracing.analysis.parse_trace", fail_read)
    with pytest.raises(type(error), match=str(error)):
        AnalysisService().analyze(AnalysisRequest(readable_trace))


@pytest.mark.parametrize("model_component", ["policy.inference", "cloud_inference"])
@pytest.mark.parametrize("origin, foreign_component", [("user", None), ("built-in", "custom"), ("built-in", "")])
@pytest.mark.parametrize("include_builtin", [False, True])
def test_summary_span_identity_prevents_impersonation_and_overwrite(
    model_component, origin, foreign_component, include_builtin
):
    events = [
        TraceEvent(
            0,
            "dispatch_request",
            {"trace_id": "req-1", "component_id": "action_dispatcher.request", "origin": "built-in"},
            schema_version=1,
        )
    ]
    for name, component_id, start_ms, end_ms in (
        ("preprocess", "policy.preprocess", 10, 12),
        ("model_call", model_component, 20, 23),
        ("postprocess", "policy.postprocess", 30, 34),
        ("action_chunk_publish", "policy.postprocess", 40, 45),
        ("dispatch_decode", "action_dispatcher.decode", 50, 56),
        ("queue_refill", "action_dispatcher.queue", 60, 67),
        ("first_action_execute", "action_dispatcher.execute", 70, 78),
        ("cloud_roundtrip", "policy", 80, 89),
    ):
        if include_builtin:
            events.extend(span_events(name, component_id, start_ms, end_ms))
        events.extend(
            span_events(
                name,
                component_id if foreign_component is None else foreign_component,
                100,
                200,
                span_id=f"impostor-{name}",
                origin=origin,
            )
        )

    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    expected = (
        {
            "preprocess_ms": 2.0,
            "inference_ms": 3.0,
            "postprocess_ms": 4.0,
            "action_chunk_publish_ms": 5.0,
            "dispatch_decode_ms": 6.0,
            "execute_publish_ms": 8.0,
            "cloud_roundtrip_ms": 9.0,
            "dispatch_to_infer_ms": 20.0,
            "queue_refill_ms": 67.0,
            "refill_to_execute_ms": 11.0,
            "total_ms": 78.0,
        }
        if include_builtin
        else {}
    )

    assert result.request_rows == [
        {"request_id": "req-1", **expected, **{f"{key}_source": "structured" for key in expected}}
    ]
    assert set(result.stage_summary) == set(expected)
    assert len(result.custom_span_summary) == (8 if origin == "user" else 0)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("model_component", ["policy.inference", "cloud_inference"])
def test_summary_retries_are_ambiguous_without_discarding_spans(reverse, model_component):
    pairs = []
    for name, component_id in (
        ("preprocess", "policy.preprocess"),
        ("model_call", model_component),
        ("queue_refill", "action_dispatcher.queue"),
        ("first_action_execute", "action_dispatcher.execute"),
    ):
        for suffix, start_ms, end_ms, status in (
            ("error", 0, 1, "error"),
            ("first", 10, 12, "ok"),
            ("same-start", 10, 14, "ok"),
            ("retry", 20, 25, "ok"),
        ):
            pairs.append(span_events(name, component_id, start_ms, end_ms, span_id=f"{name}-{suffix}", status=status))
        pairs.append(span_events(name, component_id, 0, 100, span_id=f"{name}-incomplete")[:1])
        tied_pair = span_events(name, component_id, 10, 12, span_id=f"{name}-z-tie")
        pairs.append([replace(item, fields={**item.fields, "duration_ns": 9_000_000}) for item in tied_pair])
    other_model_component = "cloud_inference" if model_component == "policy.inference" else "policy.inference"
    pairs.append(span_events("model_call", other_model_component, 30, 37, span_id="other-model"))
    if reverse:
        pairs.reverse()
    events = [
        TraceEvent(
            0,
            "dispatch_request",
            {"trace_id": "req-1", "component_id": "action_dispatcher.request", "origin": "built-in"},
            schema_version=1,
        ),
        *(item for pair in pairs for item in pair),
    ]

    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    row = QueryService(result).requests().items[0]

    for metric in (
        "preprocess_ms",
        "inference_ms",
        "execute_publish_ms",
        "dispatch_to_infer_ms",
        "queue_refill_ms",
        "total_ms",
        "refill_to_execute_ms",
    ):
        assert metric not in row
        assert row[f"{metric}_status"] == "ambiguous,error,incomplete"
        assert row[f"{metric}_source"] == "structured"
        assert any(f"{metric}: ambiguous" in warning for warning in result.warnings)
    assert not result.stage_summary
    assert QueryService(result).spans().total == 25
    assert QueryService(result).spans(SpanQuery(status="error")).total == 4
    assert QueryService(result).spans(SpanQuery(status="incomplete")).total == 4
    stats = next(
        item
        for item in result.span_summary
        if item["name"] == "model_call" and item["component_id"] == model_component and item["status"] == "ok"
    )
    assert stats["count"] == 4
    assert stats["mean"] == 5.0


@pytest.mark.parametrize("status", ["error", "cancelled", "timeout"])
def test_unique_failed_span_keeps_duration_and_status_but_not_success_metric(status):
    events = [
        *span_events("model_call", "policy.inference", 1, 3, status=status),
        event(0, "inference_begin", "req-1"),
        event(10_000_000, "inference_end", "req-1"),
        *span_events("postprocess", "policy.postprocess", 4, 5),
    ]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    row = result.request_rows[0]

    assert "inference_ms" not in row
    assert row["inference_ms_status"] == status
    assert row["inference_ms_source"] == "structured"
    assert "inference_ms" not in result.stage_summary
    assert result.stage_summary["postprocess_ms"].count == 1
    failed = QueryService(result).spans(SpanQuery(status=status)).items[0]
    assert failed.duration_ms == 2.0
    stats = next(item for item in result.to_dict()["span_summary"] if item["name"] == "model_call")
    assert (stats["status"], stats["count"], stats["mean"]) == (status, 1, 2.0)
    assert any(f"inference_ms: {status}" in warning for warning in result.warnings)
    legacy_row = result.to_dict(compatibility=True)["requests"][0]
    assert legacy_row == {"request_id": "req-1", "postprocess_ms": 1.0}


@pytest.mark.parametrize("incomplete", [False, True])
def test_ambiguous_structured_metric_cannot_fall_back_to_legacy_or_reported(incomplete):
    retry = span_events("first_action_execute", "action_dispatcher.execute", 3, 4, span_id="retry")
    events = [
        *span_events("first_action_execute", "action_dispatcher.execute", 1, 2),
        *(retry[:1] if incomplete else retry),
        event(0, "dispatch_request", "req-1"),
        event(10_000_000, "action_execute", "req-1", publish_ms=42),
    ]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))

    for metric in ("execute_publish_ms", "total_ms", "refill_to_execute_ms"):
        assert metric not in result.request_rows[0]
        assert result.request_rows[0][f"{metric}_status"] == ("ambiguous,incomplete" if incomplete else "ambiguous")
        assert metric not in result.stage_summary
    assert len(result.spans) == 2


@pytest.mark.parametrize("structured", [False, True])
def test_repeated_boundary_events_do_not_pick_an_arbitrary_request_interval(structured):
    events = [event(0, "dispatch_request", "req-1"), event(10_000_000, "dispatch_request", "req-1")]
    if structured:
        events = [
            replace(item, schema_version=1, fields={**item.fields, "component_id": "action_dispatcher.request"})
            for item in events
        ]
        events.extend(span_events("first_action_execute", "action_dispatcher.execute", 20, 22))
    else:
        events.append(event(22_000_000, "action_execute", "req-1"))
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))

    assert "total_ms" not in result.request_rows[0]
    if structured:
        assert result.request_rows[0]["total_ms_status"] == "ambiguous"
    else:
        assert "total_ms_status" not in result.request_rows[0]
    assert "total_ms" not in result.stage_summary
    if structured:
        assert result.request_rows[0]["execute_publish_ms"] == 2.0


def test_repeated_reported_metrics_are_ambiguous_and_cannot_overwrite_a_span():
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[
                *span_events("dispatch_decode", "action_dispatcher.decode", 1, 3),
                event(4_000_000, "dispatch_decode", "req-1", decode_ms=90),
                builtin_event(5_000_000, "dispatch_result", "req-1", policy_total_ms=10),
                builtin_event(6_000_000, "dispatch_result", "req-1", policy_total_ms=20),
            ]
        )
    )

    assert result.request_rows[0]["dispatch_decode_ms"] == 2.0
    assert result.request_rows[0]["dispatch_decode_ms_source"] == "structured"
    assert "policy_total_reported_ms" not in result.stage_summary
    assert result.request_rows[0]["policy_total_reported_ms_status"] == "ambiguous"


def test_failed_reported_result_is_not_a_success_metric():
    result = AnalysisService().analyze_dataset(
        TraceDataset(events=[builtin_event(1, "dispatch_result", "req-1", success=False, policy_total_ms=12)])
    )

    assert "policy_total_reported_ms" not in result.stage_summary
    assert result.request_rows[0]["policy_total_reported_ms_status"] == "error"


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_span_duration_is_queryable_but_not_aggregated(duration):
    events = span_events("model_call", "policy.inference", 1, 3)
    events[1] = replace(events[1], fields={**events[1].fields, "duration_ns": duration})
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))

    assert QueryService(result).spans().total == 1
    assert result.request_rows[0]["inference_ms_status"] == "invalid"
    assert not result.stage_summary
    assert not result.span_summary


def test_generic_span_summary_includes_non_request_and_user_spans_by_identity_and_status():
    events = []
    for origin, status, end in (("built-in", "ok", 3), ("built-in", "error", 5), ("user", "error", 7)):
        events.extend(span_events("work", "custom", 1, end, span_id=f"{origin}-{status}", origin=origin, status=status))
    events = [replace(item, fields={**item.fields, "trace_id": ""}) for item in events]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))

    assert not result.request_rows
    assert {(item["origin"], item["status"], item["mean"]) for item in result.span_summary} == {
        ("built-in", "ok", 2.0),
        ("built-in", "error", 4.0),
        ("user", "error", 6.0),
    }


@pytest.mark.parametrize("origin, foreign_component", [("user", None), ("built-in", "custom"), ("built-in", "")])
@pytest.mark.parametrize("include_builtin", [False, True])
def test_summary_boundary_event_identity_preserves_builtin_or_legacy(origin, foreign_component, include_builtin):
    events = [event(0, "dispatch_request", "req-1"), event(9_000_000, "obs_frame", "req-1")]
    for name, component_id, timestamp_ms in (
        ("dispatch_request", "action_dispatcher.request", 10),
        ("obs_frame", "policy.observation", 15),
    ):
        builtin = TraceEvent(
            timestamp_ms * 1_000_000,
            name,
            {"trace_id": "req-1", "component_id": component_id, "origin": "built-in"},
            schema_version=1,
        )
        if include_builtin:
            events.append(builtin)
        events.append(
            replace(
                builtin,
                timestamp_ns=(timestamp_ms - 1) * 1_000_000,
                fields={
                    **builtin.fields,
                    "origin": origin,
                    "component_id": component_id if foreign_component is None else foreign_component,
                },
            )
        )
    events.extend(span_events("model_call", "policy.inference", 20, 23))
    events.extend(span_events("queue_refill", "action_dispatcher.queue", 30, 31))
    events.extend(span_events("first_action_execute", "action_dispatcher.execute", 40, 41))

    row = AnalysisService().analyze_dataset(TraceDataset(events=events)).request_rows[0]

    if include_builtin:
        assert row["obs_frame_ms"] == 5.0
        assert row["obs_frame_ms_source"] == "structured"
        assert row["dispatch_to_infer_ms"] == 10.0
        assert row["queue_refill_ms"] == 21.0
        assert row["total_ms"] == 31.0
    else:
        assert not {"obs_frame_ms", "dispatch_to_infer_ms", "queue_refill_ms", "total_ms"}.intersection(row)


@pytest.fixture
def dispatcher_events():
    return [
        TraceEvent(
            timestamp_ms * 1_000_000,
            name,
            {"trace_id": "req-1", "component_id": component, "origin": "built-in", **fields},
            schema_version=1,
        )
        for timestamp_ms, name, component, fields in (
            (0, "dispatch_request", "action_dispatcher.request", {}),
            (10, "queue_refill", "action_dispatcher.queue", {"new": 8, "skipped": 0, "after": 8}),
            (
                20,
                "first_action_execute",
                "action_dispatcher.execute",
                {"consumed_index": 0, "execute_index": 0, "publish_ms": 4.0, "publish_end_ns": 25_000_000},
            ),
        )
    ]


@pytest.mark.parametrize("execute_timestamp_ms", [20, 30], ids=["scheduled-start", "legacy-after-publish"])
@pytest.mark.parametrize("reverse_emission", [False, True])
def test_dispatcher_events_use_explicit_publish_end_and_capture_order(
    dispatcher_events, execute_timestamp_ms, reverse_emission
):
    epoch_ns = 1_700_000_000_000_000_123
    dispatcher_events[-1] = replace(dispatcher_events[-1], timestamp_ns=execute_timestamp_ms * 1_000_000)
    events = [
        replace(
            item,
            timestamp_ns=item.timestamp_ns + epoch_ns,
            fields={**item.fields, "publish_end_ns": item.field("publish_end_ns") + epoch_ns}
            if "publish_end_ns" in item.fields
            else item.fields,
        )
        for item in dispatcher_events
    ]
    events.extend(
        TraceEvent(
            epoch_ns + timestamp_ms * 1_000_000,
            name,
            {
                "trace_id": "req-1",
                "component_id": component,
                "edge_id": "queue_to_execute",
                "flow_id": "req-1",
            },
            schema_version=1,
        )
        for timestamp_ms, name, component in (
            (10, "flow_send", "action_dispatcher.queue"),
            (20, "flow_receive", "action_dispatcher.execute"),
        )
    )
    if reverse_emission:
        events.reverse()
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    expected = {"queue_refill_ms": 10.0, "refill_to_execute_ms": 15.0, "total_ms": 25.0, "execute_publish_ms": 4.0}

    assert QueryService(result).requests().items == [
        {"request_id": "req-1", **expected, **{f"{metric}_source": "structured" for metric in expected}}
    ]
    assert {name: stats.mean for name, stats in result.stage_summary.items()} == expected
    assert result.dataset.events == events
    assert QueryService(result).events(EventQuery(event_name="first_action_execute")).items[0].timestamp_ns == (
        epoch_ns + execute_timestamp_ms * 1_000_000
    )
    assert not result.spans and not result.span_summary
    assert QueryService(result).flows().items[0].duration_ms == 10.0
    assert result.flows[0].status == "complete"
    assert project_latency_distribution(result, LatencyDistributionQuery(metric="total_ms")).summary.mean == 25.0
    assert not result.warnings
    assert ("event", "action_dispatcher.queue", "queue_refill", "built-in") in {
        definition.identity for definition in result.definitions
    }
    assert ("event", "action_dispatcher.execute", "first_action_execute", "built-in") in {
        definition.identity for definition in result.definitions
    }


@pytest.mark.parametrize("origin, component", [("user", None), ("built-in", "custom"), ("built-in", "")])
@pytest.mark.parametrize("include_builtin", [False, True])
def test_dispatcher_boundary_event_identity_cannot_be_impersonated(
    dispatcher_events, origin, component, include_builtin
):
    events = dispatcher_events if include_builtin else dispatcher_events[:1]
    events = events + [
        replace(
            item,
            fields={
                **item.fields,
                "origin": origin,
                "component_id": item.component_id if component is None else component,
            },
        )
        for item in dispatcher_events[1:]
    ]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))

    if include_builtin:
        assert result.request_rows[0]["queue_refill_ms"] == 10.0
        assert result.request_rows[0]["refill_to_execute_ms"] == 15.0
        assert result.request_rows[0]["total_ms"] == 25.0
        assert result.request_rows[0]["execute_publish_ms"] == 4.0
    else:
        assert not result.stage_summary
    assert not result.warnings
    assert QueryService(result).events().total == len(events)


@pytest.mark.parametrize("name", ["queue_refill", "first_action_execute"])
def test_duplicate_dispatcher_events_are_ambiguous(dispatcher_events, name):
    duplicate = next(item for item in dispatcher_events if item.name == name)
    result = AnalysisService().analyze_dataset(TraceDataset(events=[*dispatcher_events, duplicate]))
    affected = (
        ("queue_refill_ms", "refill_to_execute_ms")
        if name == "queue_refill"
        else ("execute_publish_ms", "refill_to_execute_ms", "total_ms")
    )

    for metric in affected:
        assert metric not in result.stage_summary
        assert result.request_rows[0][f"{metric}_status"] == "ambiguous"
    unaffected = "total_ms" if name == "queue_refill" else "queue_refill_ms"
    assert unaffected in result.stage_summary
    assert QueryService(result).events().total == 4


@pytest.mark.parametrize(
    "name, component",
    [("queue_refill", "action_dispatcher.queue"), ("first_action_execute", "action_dispatcher.execute")],
)
@pytest.mark.parametrize("status", ["error", "incomplete", "ambiguous", "invalid"])
def test_dispatcher_events_do_not_replace_rejected_spans(dispatcher_events, name, component, status):
    events = span_events(name, component, 5, 7, status="error" if status == "error" else "ok")
    if status == "incomplete":
        events = events[:1]
    elif status == "ambiguous":
        events.extend(span_events(name, component, 8, 9, span_id="retry"))
    elif status == "invalid":
        events[-1] = replace(events[-1], fields={**events[-1].fields, "duration_ns": -1})
    result = AnalysisService().analyze_dataset(TraceDataset(events=[*dispatcher_events, *events]))
    affected = (
        ("queue_refill_ms", "refill_to_execute_ms")
        if name == "queue_refill"
        else ("execute_publish_ms", "refill_to_execute_ms", "total_ms")
    )

    for metric in affected:
        assert metric not in result.stage_summary
        assert result.request_rows[0][f"{metric}_status"] == status
    assert QueryService(result).spans().total == (2 if status == "ambiguous" else 1)
    assert result.span_summary == AnalysisService().analyze_dataset(TraceDataset(events=events)).span_summary


@pytest.mark.parametrize("span_name", ["queue_refill", "first_action_execute", "both"])
def test_dispatcher_event_fallback_preserves_existing_span_endpoints(dispatcher_events, span_name):
    events = list(dispatcher_events)
    if span_name in {"queue_refill", "both"}:
        events.extend(span_events("queue_refill", "action_dispatcher.queue", 5, 7))
    if span_name in {"first_action_execute", "both"}:
        events.extend(span_events("first_action_execute", "action_dispatcher.execute", 15, 19))
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    refill_ms = 7.0 if span_name in {"queue_refill", "both"} else 10.0
    end_ms = 19.0 if span_name in {"first_action_execute", "both"} else 25.0

    assert result.request_rows[0]["queue_refill_ms"] == refill_ms
    assert result.request_rows[0]["total_ms"] == end_ms
    assert result.request_rows[0]["refill_to_execute_ms"] == end_ms - refill_ms
    assert QueryService(result).spans().total == (2 if span_name == "both" else 1)


@pytest.mark.parametrize("publish_end_ns", [None, True, "25000000", float("nan"), float("inf"), -1])
@pytest.mark.parametrize("complete_boundaries", [False, True])
def test_dispatcher_event_missing_or_invalid_end_does_not_guess_or_use_legacy(
    dispatcher_events, publish_end_ns, complete_boundaries
):
    fields = dict(dispatcher_events[-1].fields)
    if publish_end_ns is None:
        fields.pop("publish_end_ns")
    else:
        fields["publish_end_ns"] = publish_end_ns
    dispatcher_events[-1] = replace(dispatcher_events[-1], fields=fields)
    events = (dispatcher_events if complete_boundaries else dispatcher_events[-1:]) + [
        event(0, "dispatch_request", "req-1"),
        event(10_000_000, "queue_refill", "req-1"),
        event(99_000_000, "action_execute", "req-1", publish_ms=90),
    ]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))

    for metric in ("refill_to_execute_ms", "total_ms"):
        assert metric not in result.stage_summary
        assert result.request_rows[0][f"{metric}_source"] == "structured"
        assert result.request_rows[0][f"{metric}_status"] == ("incomplete" if publish_end_ns is None else "invalid")
    if complete_boundaries:
        assert result.request_rows[0]["queue_refill_ms"] == 10.0
    else:
        assert "queue_refill_ms" not in result.request_rows[0]
    assert result.request_rows[0]["execute_publish_ms"] == 4.0
    assert result.warnings


@pytest.mark.parametrize("publish_ms", [None, -1, float("nan"), float("inf")])
def test_dispatcher_event_publish_duration_is_independent_of_end_boundary(dispatcher_events, publish_ms):
    dispatcher_events[-1] = replace(
        dispatcher_events[-1], fields={**dispatcher_events[-1].fields, "publish_ms": publish_ms}
    )
    result = AnalysisService().analyze_dataset(TraceDataset(events=dispatcher_events))

    assert "execute_publish_ms" not in result.stage_summary
    assert result.request_rows[0]["execute_publish_ms_status"] == ("incomplete" if publish_ms is None else "invalid")
    assert result.request_rows[0]["total_ms"] == 25.0
    assert result.request_rows[0]["refill_to_execute_ms"] == 15.0


def test_registered_spans_outside_summary_mapping_do_not_add_metrics():
    events = []
    for name, component_id in (
        ("scheduler_dispatch", "global_scheduler"),
        ("observation_sampling", "policy.observation"),
        ("result_encoding", "policy.postprocess"),
        ("policy_pipeline", "policy"),
    ):
        events.extend(span_events(name, component_id, 0, 1))

    result = AnalysisService().analyze_dataset(TraceDataset(events=events))

    assert len(result.spans) == 4
    assert result.request_rows == [{"request_id": "req-1"}]
    assert result.stage_summary == {}


def test_request_metrics_spans_flows_and_legacy_json():
    dataset = TraceDataset(
        events=[
            builtin_event(0, "dispatch_request", "req-1"),
            event(1_000_000, "span_begin", "req-1", span_id="same", span_name="outer"),
            event(2_000_000, "flow_send", "req-1", edge_id="edge", flow_id="same"),
            event(3_000_000, "span_begin", "req-2", span_id="same", span_name="other"),
            event(4_000_000, "span_end", "req-1", span_id="same", span_name="outer"),
            event(5_000_000, "flow_receive", "req-1", edge_id="edge", flow_id="same"),
            event(6_000_000, "span_end", "req-2", span_id="same", span_name="other"),
            builtin_event(10_000_000, "first_action_execute", "req-1", publish_ms=0.5, publish_end_ns=10_000_000),
        ]
    )

    result = AnalysisService().analyze_dataset(dataset)

    assert result.request_rows[0]["total_ms"] == 10.0
    assert sorted(span.duration_ms for span in result.spans) == [3.0, 3.0]
    assert result.flows[0].duration_ms == 3.0
    assert result.to_dict(compatibility=True) == {
        "requests": [
            {"request_id": "req-1", "total_ms": 10.0, "execute_publish_ms": 0.5},
            {"request_id": "req-2"},
        ],
        "observations": {},
    }


def test_topology_binds_legacy_logger_to_component():
    dataset = TraceDataset(
        events=[
            TraceEvent(
                1,
                "dispatch_request",
                {"request_id": "req-1"},
                EventOrigin(provider="ib_trace.dispatch"),
            )
        ]
    )
    topology = TraceTopology(
        components=[Component("dispatcher", "Dispatcher", "ros_node")],
        edges=[DataFlowEdge("edge", "dispatcher", "dispatcher", "loop", "internal")],
        logger_to_component={"ib_trace.dispatch": "dispatcher"},
    )

    result = AnalysisService().analyze_dataset(dataset, topology=topology)

    assert result.dataset.events[0].component_id == "dispatcher"


def test_monolithic_coverage_does_not_require_cloud_events():
    result = AnalysisService().analyze_dataset(TraceDataset(events=[event(0, "dispatch_request", "req-1")]))

    assert "cloud_roundtrip_ms" not in result.coverage["missing_events"]


def test_partial_cloud_coverage_requires_both_edge_events():
    result = AnalysisService().analyze_dataset(
        TraceDataset(events=[event(0, "dispatch_request", "req-1")]),
        topology=TraceTopology(execution_mode="distributed"),
    )

    assert "cloud_roundtrip_ms" in result.coverage["missing_events"]


def test_summary_renders_mean_after_max():
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[
                builtin_event(0, "dispatch_request", "req-1"),
                builtin_event(10_000_000, "first_action_execute", "req-1", publish_ms=1.0, publish_end_ns=10_000_000),
                builtin_event(20_000_000, "dispatch_request", "req-2"),
                builtin_event(50_000_000, "first_action_execute", "req-2", publish_ms=1.0, publish_end_ns=50_000_000),
            ]
        )
    )
    stream = StringIO()

    render_summary(result, stream)

    lines = stream.getvalue().splitlines()
    header = next(line for line in lines if line.strip().startswith("Stage"))
    total = next(line for line in lines if "Dispatch->Execute" in line)
    assert header.split() == ["Stage", "p50", "p95", "p99", "max", "mean", "n"]
    assert "20.0ms" in total


def test_summary_labels_inference_boundary_as_model_call():
    result = AnalysisService().analyze_dataset(
        TraceDataset(
            events=[
                *span_events("model_call", "policy.inference", 1, 11),
            ]
        )
    )
    stream = StringIO()

    render_summary(result, stream)

    assert "Model call" in stream.getvalue()
    assert result.request_rows[0]["inference_ms"] == 10.0


def test_request_exposes_reported_policy_total_by_explicit_name():
    result = AnalysisService().analyze_dataset(
        TraceDataset(events=[builtin_event(1, "dispatch_result", "req-1", policy_total_ms=17.5)])
    )

    assert result.request_rows[0]["policy_total_reported_ms"] == 17.5


def test_reported_policy_latency_keeps_its_source_identity_without_dropping_events():
    events = [
        TraceEvent(
            index,
            "dispatch_result",
            {"trace_id": "r", "component_id": component, "origin": origin, "policy_total_ms": value},
            schema_version=1,
        )
        for index, (component, origin, value) in enumerate(
            [("policy", "built-in", 12.0), ("action_dispatcher.decode", "built-in", 12.0), ("policy", "user", 900.0)]
        )
    ]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    assert result.request_rows[0]["policy_total_reported_ms"] == 12.0
    assert len(result.dataset.events) == 3
    assert result.warnings == []

    result = AnalysisService().analyze_dataset(TraceDataset(events=[events[0], replace(events[0], timestamp_ns=10)]))
    assert result.request_rows[0]["policy_total_reported_ms_status"] == "ambiguous"
    assert "policy_total_reported_ms" not in result.request_rows[0]


def test_empty_call_tree_explains_that_structured_spans_are_required():
    stream = StringIO()

    render_call_tree([], stream)

    assert stream.getvalue().startswith("No structured spans found.")


def test_summary_separates_user_spans_and_marks():
    dataset = TraceDataset(
        events=[
            event(
                0,
                "span_begin",
                "req-1",
                span_id="built-in",
                span_name="preprocess",
                component_id="policy.preprocess",
                origin="built-in",
            ),
            event(
                1_000_000,
                "span_begin",
                "req-1",
                span_id="user",
                parent_span_id="built-in",
                span_name="resize_image",
                component_id="policy.preprocess",
                origin="user",
            ),
            event(
                3_000_000,
                "span_end",
                "req-1",
                span_id="user",
                parent_span_id="built-in",
                span_name="resize_image",
                component_id="policy.preprocess",
                origin="user",
            ),
            event(
                4_000_000,
                "tensor_ready",
                "req-1",
                component_id="policy.preprocess",
                origin="user",
            ),
            event(
                5_000_000,
                "span_end",
                "req-1",
                span_id="built-in",
                span_name="preprocess",
                component_id="policy.preprocess",
                origin="built-in",
            ),
        ]
    )
    dataset.events[3] = TraceEvent(
        dataset.events[3].timestamp_ns,
        dataset.events[3].name,
        dataset.events[3].fields,
        schema_version=1,
    )
    result = AnalysisService().analyze_dataset(dataset)
    stream = StringIO()

    render_summary(result, stream)

    output = stream.getvalue()
    assert "Custom Span Summary" in output
    assert "resize_image" in output
    assert "preprocess" not in [item["name"] for item in result.custom_span_summary]
    assert "Custom Marks" in output
    assert "tensor_ready" in output


def test_tracepoint_graph_includes_declared_and_observed_operations():
    topology = TraceTopology(
        components=[
            Component("policy", "Policy", "ros_node"),
            Component("policy.preprocess", "Preprocessor", "module", "policy"),
        ]
    )
    dataset = TraceDataset(
        events=[
            event(
                0,
                "span_begin",
                "req-1",
                span_id="user",
                span_name="resize_image",
                component_id="policy.preprocess",
                origin="user",
            ),
            event(
                2_000_000,
                "span_end",
                "req-1",
                span_id="user",
                span_name="resize_image",
                component_id="policy.preprocess",
                origin="user",
            ),
        ]
    )
    result = AnalysisService().analyze_dataset(dataset, topology=topology)
    stream = StringIO()

    render_graph(result, stream, metric="mean", view="tracepoints")

    output = stream.getvalue()
    assert "Resize Image*" in output
    assert "processing=2.000ms" in output


@pytest.mark.parametrize(
    "component_id, origin, inference_ms, source",
    [
        ("policy.inference", "built-in", 2.0, "structured"),
        ("cloud_inference", "built-in", 2.0, "structured"),
        ("policy.inference", "user", 10.0, "legacy"),
        ("custom", "built-in", 10.0, "legacy"),
        ("", "built-in", 10.0, "legacy"),
    ],
)
def test_structured_metrics_override_legacy_and_preserve_compatibility_shape(
    component_id, origin, inference_ms, source
):
    dataset = TraceDataset(
        events=[
            TraceEvent(
                0,
                "dispatch_request",
                {"trace_id": "req-1", "component_id": "action_dispatcher.request", "origin": "built-in"},
                schema_version=1,
            ),
            event(100_000, "dispatch_request", "req-1"),
            event(
                1_000_000,
                "span_begin",
                "req-1",
                span_id="model",
                span_name="model_call",
                component_id=component_id,
                origin=origin,
            ),
            event(
                4_000_000,
                "span_end",
                "req-1",
                span_id="model",
                span_name="model_call",
                component_id=component_id,
                origin=origin,
                duration_ns=2_000_000,
            ),
            event(9_000_000, "inference_begin", "req-1"),
            event(19_000_000, "inference_end", "req-1"),
        ]
    )

    result = AnalysisService().analyze_dataset(dataset)

    if source == "structured":
        assert result.request_rows[0]["inference_ms"] == inference_ms
        assert result.request_rows[0]["inference_ms_source"] == source
        assert result.request_rows[0]["dispatch_to_infer_ms"] == 1.0
    else:
        assert "inference_ms" not in result.request_rows[0]
        assert "dispatch_to_infer_ms" not in result.request_rows[0]
    assert "inference_ms_source" not in result.to_dict(compatibility=True)["requests"][0]


def test_dual_emitted_observations_are_not_double_counted():
    dataset = TraceDataset(
        events=[
            event(0, "obs_receive", "", key="camera", transport_ms=2.0),
            TraceEvent(
                1,
                "obs_receive",
                {"key": "camera", "transport_ms": 2.0, "origin": "built-in"},
                schema_version=1,
            ),
            event(2, "obs_sample", "req-1", key="camera", ready=1, age_ms=3.0),
            TraceEvent(
                3,
                "obs_sample",
                {"trace_id": "req-1", "key": "camera", "ready": 1, "age_ms": 3.0, "origin": "built-in"},
                schema_version=1,
            ),
        ]
    )

    result = AnalysisService().analyze_dataset(dataset)

    assert result.observations["camera"] == {"transport_ms": [2.0], "age_ms": [3.0]}


def test_non_finite_observations_are_ignored():
    dataset = TraceDataset(
        events=[
            builtin_event(1, "obs_receive", "req", key="camera", transport_ms="nan"),
            builtin_event(2, "obs_sample", "req", key="camera", ready="true", age_ms="inf"),
        ]
    )

    result = AnalysisService().analyze_dataset(dataset)

    assert result.observations["camera"] == {"transport_ms": [], "age_ms": []}

import json

import pytest

from ibrobot_tracing.analysis import AnalysisService
from ibrobot_tracing.model import TraceDataset, TraceEvent
from ibrobot_tracing.parsing import parse_lines


@pytest.mark.parametrize("timestamp", [None, 0, -1, True, "42", float("inf")])
def test_missing_epoch_never_falls_back_to_midnight(timestamp):
    record = {"event": "dispatch_request", "timestamp_ns": timestamp}
    result = parse_lines(["[23:59:59.000000000] ib_trace.test: IBTRACE1 " + json.dumps(record)])
    assert not result.events and result.warnings


def test_legacy_midnight_and_bad_ts_do_not_poison_epoch_records():
    epoch = 1_760_000_000_000_000_000
    lines = [
        "[23:59:59.000000000] ib_trace.test: [dispatch_request] request_id=r _ts_ns=bad",
        "IBTRACE1 " + json.dumps({"event": "mark", "timestamp_ns": epoch, "fields": {"trace_id": "r"}}),
        "[00:00:01.000000000] ib_trace.test: [action_execute] request_id=r _ts_ns=[1]",
    ]
    result = parse_lines(lines)
    assert [event.timestamp_ns for event in result.events] == [epoch]
    assert len(result.warnings) == 2


@pytest.mark.parametrize(
    "value,status", [(None, "incomplete"), ("n/a", "invalid"), (-42, "invalid"), (float("nan"), "invalid")]
)
def test_reported_value_is_retained_as_invalid_not_a_negative_sample(value, status):
    event = TraceEvent(
        42,
        "dispatch_result",
        {"trace_id": "r", "component_id": "policy", "origin": "built-in", "policy_total_ms": value},
        schema_version=1,
    )
    result = AnalysisService().analyze_dataset(TraceDataset(events=[event]))
    row = result.request_rows[0]
    assert row["policy_total_reported_ms_status"] == status
    assert row["policy_total_reported_ms_source"] == "reported"
    assert "policy_total_reported_ms" not in result.stage_summary
    assert result.warnings


def test_sampled_later_actions_are_not_promoted_to_first_action():
    events = [
        TraceEvent(
            i,
            "action_execute",
            {"trace_id": "r", "component_id": "action_dispatcher.execute", "publish_ms": i},
            schema_version=1,
        )
        for i in range(1, 4)
    ]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    assert result.request_rows == [{"request_id": "r"}]
    assert len(result.dataset.events) == 3 and not result.warnings


def test_large_invalid_set_has_bounded_diagnostics_and_keeps_raw_events():
    events = [
        TraceEvent(
            i + 1,
            "dispatch_result",
            {"trace_id": str(i), "component_id": "policy", "policy_total_ms": -1},
            schema_version=1,
        )
        for i in range(1200)
    ]
    result = AnalysisService().analyze_dataset(TraceDataset(events=events))
    assert len(result.dataset.events) == 1200 and len(result.request_rows) == 1200
    assert len(result.warnings) == 1001
    assert result.warnings[-1] == "200 additional analysis diagnostics suppressed"

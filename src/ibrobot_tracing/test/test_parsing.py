import json

import pytest

from ibrobot_tracing import parsing
from ibrobot_tracing.parsing import parse_lines


def test_parse_legacy_event():
    dataset = parse_lines(["[12:34:56.123456789] ib_trace.dispatch: [dispatch_request] request_id=req-1 queue_size=4"])

    assert not dataset.events
    assert "Unsupported legacy" in dataset.warnings[0]


def test_parse_babeltrace_python_message_without_ctf_field_leakage():
    line = (
        "[15:16:58.741234932] host lttng_python:event: { cpu_id = 9 }, "
        '{ msg = "[preprocess_begin] request_id=req-1", logger_name = "ib_trace.policy", lineno = 892 }'
    )

    dataset = parse_lines([line])

    assert not dataset.events
    assert "Unsupported legacy" in dataset.warnings[0]


def test_parse_structured_babeltrace_python_message():
    payload = {
        "schema_version": 1,
        "event": "span_begin",
        "timestamp_ns": 42,
        "fields": {"trace_id": "req-1", "span_id": "span-1"},
    }
    message = json.dumps(f"IBTRACE1 {json.dumps(payload)}")
    line = f'[15:16:58.741234932] lttng_python:event: {{ msg = {message}, logger_name = "ib_trace.user" }}'

    dataset = parse_lines([line])

    assert dataset.events[0].name == "span_begin"
    assert dataset.events[0].request_id == "req-1"
    assert dataset.events[0].field("span_id") == "span-1"


def test_parse_structured_babeltrace1_unescaped_python_message():
    payload = {
        "schema_version": 1,
        "event": "dispatch_request",
        "timestamp_ns": 42,
        "fields": {"trace_id": "req-1", "queue_size": 0},
    }
    line = (
        '[15:16:58.741234932] lttng_python:event: { msg = "IBTRACE1 '
        f'{json.dumps(payload)}", logger_name = "ib_trace.dispatch", lineno = 150 }}'
    )

    dataset = parse_lines([line])

    assert dataset.events[0].name == "dispatch_request"
    assert dataset.events[0].request_id == "req-1"
    assert dataset.events[0].field("queue_size") == 0


def test_structured_payload_timestamp_takes_precedence():
    payload = {
        "schema_version": 1,
        "event": "custom_mark",
        "timestamp_ns": 42,
        "fields": {"trace_id": "req-1", "value": True},
    }
    dataset = parse_lines([f"[12:34:56.123456789] ib_trace.user: IBTRACE1 {json.dumps(payload)}"])

    assert dataset.events[0].timestamp_ns == 42
    assert dataset.events[0].schema_version == 1
    assert dataset.events[0].field("value") is True


def test_invalid_structured_record_is_reported():
    dataset = parse_lines(["ib_trace.user: IBTRACE1 not-json"])

    assert not dataset.events
    assert len(dataset.warnings) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        "text",
        42,
        True,
        {},
        {"event": "bad", "timestamp_ns": None},
        {"event": "bad", "timestamp_ns": "invalid"},
        {"event": "bad", "timestamp_ns": []},
        {"event": "bad", "timestamp_ns": {}},
        {"event": "bad", "timestamp_ns": float("inf")},
        {"event": "bad", "timestamp_ns": 42, "schema_version": None},
        {"event": "bad", "timestamp_ns": 42, "schema_version": "invalid"},
        {"event": "bad", "timestamp_ns": 42, "schema_version": []},
        {"event": "bad", "timestamp_ns": 42, "schema_version": float("inf")},
        {"event": "bad", "timestamp_ns": 42, "fields": []},
        {"event": "bad", "timestamp_ns": 42, "fields": None},
        {"event": "bad", "timestamp_ns": 42, "fields": "wrongtype"},
        {"event": "bad", "timestamp_ns": 42, "fields": 1},
        {"event": ["bad"], "timestamp_ns": 42},
        {"event": 1, "timestamp_ns": 42},
        {"event": "bad", "timestamp_ns": 42, "clock": []},
    ],
)
@pytest.mark.parametrize("transport", ["log", "babeltrace1", "babeltrace2"])
def test_bad_structured_record_warns_and_does_not_stop_following_records(payload, transport):
    valid = {"event": "valid", "timestamp_ns": 42, "fields": {"request_id": "00123"}}
    message = f"IBTRACE1 {json.dumps(payload)}"
    if transport == "log":
        line = f"[12:34:56.123456789] ib_trace.user: {message}"
    else:
        quoted = f'"{message}"' if transport == "babeltrace1" else json.dumps(message)
        line = f'[12:34:56.123456789] lttng_python:event: {{ msg = {quoted}, logger_name = "ib_trace.user" }}'
    dataset = parse_lines(
        [
            f"IBTRACE1 {json.dumps(valid)}",
            line,
            f"IBTRACE1 {json.dumps(valid)}",
        ],
        source="records.log",
    )

    assert [event.name for event in dataset.events] == ["valid", "valid"]
    assert [event.sequence for event in dataset.events] == [0, 2]
    assert dataset.warnings == ["Could not parse trace record records.log:2"]


def test_invalid_structured_record_is_not_reinterpreted_as_legacy():
    payload = {"event": "bad", "timestamp_ns": 42, "fields": "[fake] request_id=00123"}

    dataset = parse_lines([f"[12:34:56.123456789] ib_trace.user: IBTRACE1 {json.dumps(payload)}"])

    assert not dataset.events
    assert dataset.warnings == ["Could not parse trace record <memory>:1"]


@pytest.mark.parametrize("error_type", [MemoryError, SystemError, KeyboardInterrupt, SystemExit])
def test_parser_propagates_fatal_errors(monkeypatch, error_type):
    fatal_error = error_type("fatal")

    def fail(*args, **kwargs):
        raise fatal_error

    monkeypatch.setattr(parsing.json, "loads", fail)

    with pytest.raises(error_type) as caught:
        parse_lines(['IBTRACE1 {"event":"valid","timestamp_ns":42}'])
    assert caught.value is fatal_error


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("00123", "00123"),
        ("1e3", "1e3"),
        ("true", "true"),
        ("nan", "nan"),
        ('"00123"', "00123"),
        ('"id\\""', 'id\\"'),
    ],
)
def test_existing_text_reader_preserves_opaque_ids(value, expected):
    keys = ("trace_id", "request_id", "inference_id", "flow_id", "span_id", "parent_span_id")
    fields = {**dict.fromkeys(keys, expected), "queue_size": 4}
    dataset = parse_lines(["IBTRACE1 " + json.dumps({"event": "mark", "timestamp_ns": 42, "fields": fields})])

    assert dataset.events[0].fields == {**dict.fromkeys(keys, expected), "queue_size": 4}
    assert not dataset.warnings


def test_trace_record_limit_stops_parsing_with_a_warning():
    dataset = parse_lines(
        [
            'IBTRACE1 {"timestamp_ns":1000000000,"event":"first","fields":{"request_id":"req"}}',
            'IBTRACE1 {"timestamp_ns":2000000000,"event":"second","fields":{"request_id":"req"}}',
        ],
        max_events=1,
    )

    assert [event.name for event in dataset.events] == ["first"]
    assert dataset.warnings == ["Trace record limit (1) reached"]


def test_malformed_record_warnings_are_bounded():
    dataset = parse_lines(["ib_trace.user: IBTRACE1 not-json"] * 1_100)

    assert len(dataset.warnings) == 1_001
    assert dataset.warnings[-1] == "Additional trace parse warnings suppressed"

import json
from io import StringIO

import pytest

from ibrobot_tracing import (
    LatencyDistributionProjection,
    LatencyDistributionQuery,
    project_latency_distribution,
)
from ibrobot_tracing.analysis import AnalysisResult
from ibrobot_tracing.cli import main
from ibrobot_tracing.model import TraceDataset, TraceEvent
from ibrobot_tracing.rendering import render_latency_distribution


def _result(rows, *, with_event=False):
    events = [TraceEvent(0, "marker")] if with_event else []
    return AnalysisResult(TraceDataset(events=events), request_rows=rows)


def test_metric_discovery_priority_default_and_invalid_samples():
    result = _result(
        [
            {
                "request_id": "two",
                "zeta_ms": 2,
                "model_call_ms": 20,
                "inference_ms": 12,
                "total_ms": 22,
                "total_ms_source": "structured",
                "request_id_ms": 123,
            },
            {
                "request_id": "one",
                "alpha_ms": 1.0,
                "model_call_ms": 10,
                "inference_ms": float("nan"),
                "total_ms": 11,
            },
            {"request_id": "invalid", "total_ms": float("inf")},
        ]
    )

    projection = project_latency_distribution(result, LatencyDistributionQuery(bins=5))

    assert projection.metric == "total_ms"
    assert projection.available_metrics == [
        "total_ms",
        "inference_ms",
        "alpha_ms",
        "model_call_ms",
        "zeta_ms",
    ]
    assert projection.sample_count == projection.summary.count == 2
    assert projection.invalid_count == 1
    assert projection.minimum == 11
    assert projection.maximum == 22
    assert "request_id_ms" not in projection.available_metrics


def test_histogram_conserves_samples_clamps_max_and_returns_highest_outliers():
    rows = [
        {"request_id": request_id, "latency_ms": value}
        for request_id, value in zip("abcdef", [-5, 0, 5, 10, 15, 20], strict=True)
    ]

    projection = project_latency_distribution(
        _result(rows),
        LatencyDistributionQuery(metric="latency_ms", bins=5, outlier_limit=3),
    )

    assert projection.bin_count == 5
    assert sum(bucket.count for bucket in projection.buckets) == projection.sample_count == 6
    assert projection.buckets[0].start == -5
    assert projection.buckets[-1].end == 20
    assert sum("f" in bucket.request_ids for bucket in projection.buckets) == 1
    assert [outlier.value for outlier in projection.outliers] == [20, 15, 10]
    assert [outlier.rank for outlier in projection.outliers] == [1, 2, 3]
    assert projection.outliers[0].percentile == 100
    assert projection.outliers[0].p95_tail
    assert not projection.outliers[1].p95_tail
    assert not projection.outliers[2].p95_tail


def test_equal_single_empty_and_invalid_only_values_are_deterministic():
    equal = project_latency_distribution(
        _result([{"request_id": request_id, "same_ms": 4.5} for request_id in ("d", "b", "a", "c")]),
        LatencyDistributionQuery(metric="same_ms", bucket_request_limit=2),
    )
    single = project_latency_distribution(
        _result([{"request_id": "only", "single_ms": -2}]),
        LatencyDistributionQuery(metric="single_ms"),
    )
    empty = project_latency_distribution(_result([]))
    invalid = project_latency_distribution(
        _result(
            [
                {"request_id": "nan", "broken_ms": float("nan")},
                {"request_id": "inf", "broken_ms": float("-inf")},
                {"request_id": "huge", "broken_ms": 10**10_000},
            ]
        ),
        LatencyDistributionQuery(metric="broken_ms"),
    )

    assert equal.bin_count == 1
    assert equal.buckets[0].start == equal.buckets[0].end == 4.5
    assert equal.buckets[0].request_ids == ["a", "b"]
    assert equal.buckets[0].returned == 2
    assert equal.buckets[0].truncated
    assert single.bin_count == 1
    assert single.minimum == single.maximum == -2
    assert single.outliers[0].p95_tail
    assert empty == LatencyDistributionProjection(None, "ms", [], empty.summary, 0, 0, None, None, 0, [], [])
    assert invalid.metric == "broken_ms"
    assert invalid.available_metrics == []
    assert invalid.sample_count == invalid.bin_count == 0
    assert invalid.invalid_count == 3
    assert invalid.to_dict()["summary"]["mean"] is None


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"bins": 4}, "bins"),
        ({"bins": 101}, "bins"),
        ({"outlier_limit": 0}, "outlier_limit"),
        ({"outlier_limit": 101}, "outlier_limit"),
        ({"bucket_request_limit": 0}, "bucket_request_limit"),
        ({"bucket_request_limit": 1001}, "bucket_request_limit"),
        ({"metric": "request_id"}, "metric"),
    ],
)
def test_distribution_query_rejects_unbounded_or_non_latency_inputs(arguments, message):
    with pytest.raises(ValueError, match=message):
        LatencyDistributionQuery(**arguments)


def test_unknown_metric_rejected_and_text_renderer_handles_empty():
    result = _result([{"request_id": "one", "known_ms": 1}])
    with pytest.raises(ValueError, match="not available"):
        project_latency_distribution(result, LatencyDistributionQuery(metric="unknown_ms"))

    stream = StringIO()
    render_latency_distribution(project_latency_distribution(_result([])), stream)
    assert "No finite samples" in stream.getvalue()


def test_distribution_cli_supports_text_json_and_validation(monkeypatch, capsys):
    result = _result(
        [
            {"request_id": "fast", "latency_ms": 1},
            {"request_id": "slow", "latency_ms": 9},
        ],
        with_event=True,
    )
    monkeypatch.setattr("ibrobot_tracing.cli._load", lambda _args: result)

    assert main(["distribution", "trace.log", "--metric", "latency_ms", "--bins", "5"]) == 0
    assert "Latency Distribution: latency_ms (ms)" in capsys.readouterr().out

    assert main(["distribution", "trace.log", "--format", "json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["metric"] == "latency_ms"
    assert document["sample_count"] == 2

    assert main(["distribution", "trace.log", "--bins", "4"]) == 2
    assert "bins must be between" in capsys.readouterr().err

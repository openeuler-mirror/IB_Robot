import json
from pathlib import Path

import pytest

from ibrobot_tracing import (
    SpanProfileProjection as PublicSpanProfileProjection,
)
from ibrobot_tracing import (
    SpanProfileQuery as PublicSpanProfileQuery,
)
from ibrobot_tracing import cli
from ibrobot_tracing import project_span_profile as public_project_span_profile
from ibrobot_tracing import projection as projection_module
from ibrobot_tracing._span_hierarchy import build_span_hierarchy
from ibrobot_tracing.analysis import AnalysisResult
from ibrobot_tracing.model import SpanRecord, TraceDataset, TraceEvent
from ibrobot_tracing.projection import (
    AggregateSpanProfileProjection,
    RequestSpanProfileProjection,
    SpanProfileQuery,
    project_span_profile,
    span_profile_speedscope,
)


def span(
    name,
    span_id,
    start_ns,
    end_ns,
    *,
    parent_span_id="",
    request_id="request",
    component_id="component",
    origin="user",
    status="ok",
    duration_ns=None,
):
    fields = {} if duration_ns is None else {"duration_ns": duration_ns}
    return SpanRecord(
        name,
        request_id,
        span_id,
        parent_span_id,
        component_id,
        start_ns,
        end_ns,
        status,
        origin,
        fields,
    )


def result(*spans):
    return AnalysisResult(TraceDataset(events=[TraceEvent(0, "marker")]), spans=list(spans))


def by_span_id(profile):
    return {node.span_id: node for node in profile.nodes}


def by_name(profile):
    return {node.name: node for node in profile.nodes}


def test_request_profile_uses_monotonic_geometry_and_sequential_uncovered_wall():
    profile = project_span_profile(
        result(
            span("root", "root", 0, 12, duration_ns=10),
            span("child", "child", 2, 6, parent_span_id="root", duration_ns=4),
        ),
        SpanProfileQuery(request_id="request"),
    )
    assert isinstance(profile, RequestSpanProfileProjection)
    nodes = by_span_id(profile)

    assert profile.measurement == "instrumented_wall"
    assert profile.not_cpu
    assert nodes["root"].end_ns == 10
    assert nodes["root"].observed_end_ns == 12
    assert nodes["root"].duration_source == "monotonic"
    assert nodes["root"].uncovered_wall_ns == 6
    assert nodes["child"].uncovered_wall_ns == 4
    assert nodes["child"].parent_id == nodes["root"].id
    assert nodes["child"].depth == 1
    assert profile.totals["sampled_uncovered_wall_ns"] == 10
    assert profile.totals["concurrency_factor"] == 1.0


def test_overlapping_siblings_get_tracks_and_parent_uses_interval_union():
    profile = project_span_profile(
        result(
            span("root", "root", 0, 10),
            span("left", "left", 1, 7, parent_span_id="root"),
            span("right", "right", 3, 9, parent_span_id="root"),
        ),
        SpanProfileQuery(request_id="request"),
    )
    nodes = by_span_id(profile)

    assert nodes["root"].uncovered_wall_ns == 2
    assert {nodes["left"].track, nodes["right"].track} == {0, 1}
    assert "overlapping_siblings" in nodes["left"].diagnostics
    assert "overlapping_siblings" in nodes["right"].diagnostics
    assert profile.totals["sampled_uncovered_wall_ns"] == 14
    assert profile.totals["concurrency_factor"] == pytest.approx(1.4)


def test_large_overlap_group_is_bounded_and_counts_every_affected_node():
    count = 256
    profile = project_span_profile(
        result(*(span(f"span-{index}", str(index), 0, 10) for index in range(count))),
        SpanProfileQuery(request_id="request"),
    )
    overlap_diagnostics = [item for item in profile.diagnostics if item.code == "overlapping_siblings"]

    assert len(overlap_diagnostics) == 1
    assert len(overlap_diagnostics[0].occurrence_ids) == 20
    assert len(overlap_diagnostics[0].message) <= 512
    assert f"affecting {count} spans" in overlap_diagnostics[0].message
    assert profile.totals["diagnostic_counts"]["overlapping_siblings"] == count
    assert len({node.track for node in profile.nodes}) == count


def test_non_contained_async_child_is_linked_but_does_not_cover_parent_wall():
    profile = project_span_profile(
        result(
            span("parent", "parent", 0, 5),
            span("async", "async", 6, 10, parent_span_id="parent"),
        ),
        SpanProfileQuery(request_id="request"),
    )
    nodes = by_span_id(profile)

    assert nodes["async"].parent_id == nodes["parent"].id
    assert nodes["parent"].uncovered_wall_ns == 5
    assert "child_outside_parent" in nodes["async"].diagnostics


def test_structural_and_duration_diagnostics_are_explicit():
    profile = project_span_profile(
        result(
            span("orphan", "orphan", 0, 1, parent_span_id="missing"),
            span("cycle-a", "cycle-a", 2, 3, parent_span_id="cycle-b"),
            span("cycle-b", "cycle-b", 2, 3, parent_span_id="cycle-a"),
            span("duplicate-one", "duplicate", 4, 5),
            span("duplicate-two", "duplicate", 6, 7),
            span("incomplete", "incomplete", 8, None, status="incomplete"),
            span("negative", "negative", 10, 9),
        ),
        SpanProfileQuery(request_id="request"),
    )
    codes = {diagnostic.code for diagnostic in profile.diagnostics}

    assert {
        "orphan",
        "cycle",
        "duplicate_span_id",
        "incomplete",
        "negative_duration",
    } <= codes
    assert profile.totals["incomplete_count"] == 1
    assert profile.totals["invalid_count"] == 1
    assert profile.totals["excluded_weight_count"] == 2
    duplicate_ids = [node.id for node in profile.nodes if node.span_id == "duplicate"]
    assert len(duplicate_ids) == len(set(duplicate_ids)) == 2


def test_component_filter_reports_filtered_parent_without_reparenting():
    profile = project_span_profile(
        result(
            span("root", "root", 0, 10, component_id="outer"),
            span("child", "child", 2, 8, parent_span_id="root", component_id="inner"),
        ),
        SpanProfileQuery(request_id="request", component_id="inner"),
    )

    assert len(profile.roots) == 1
    assert profile.nodes[0].parent_id == ""
    assert "filtered_parent" in profile.nodes[0].diagnostics
    assert {diagnostic.code for diagnostic in profile.diagnostics} == {"filtered_parent"}


def test_request_mode_builds_only_the_selected_request_before_profile_filters(monkeypatch):
    built_request_ids = []
    original = projection_module.build_span_hierarchy

    def recording_build(spans):
        built_request_ids.append({item.trace_id for item in spans})
        return original(spans)

    monkeypatch.setattr(projection_module, "build_span_hierarchy", recording_build)
    profile = project_span_profile(
        result(
            span("root", "root", 0, 10, component_id="outer"),
            span("child", "child", 1, 9, parent_span_id="root", component_id="inner"),
            *(span(f"other-{index}", str(index), 0, 10, request_id="other") for index in range(50)),
        ),
        SpanProfileQuery(request_id="request", component_id="inner"),
    )

    assert built_request_ids == [{"request"}]
    assert "filtered_parent" in profile.nodes[0].diagnostics


def test_request_filters_and_multiple_logical_roots_are_preserved():
    profile = project_span_profile(
        result(
            span("first", "first", 0, 2, component_id="one", origin="built-in"),
            span("second", "second", 4, 8, component_id="two", status="error"),
            span("other-request", "other", 0, 10, request_id="other"),
        ),
        SpanProfileQuery(request_id="request", start_ns=1, end_ns=7),
    )
    assert len(profile.roots) == 2
    assert {node.span_id for node in profile.nodes} == {"first", "second"}

    filtered = project_span_profile(
        result(
            span("first", "first", 0, 2, origin="built-in"),
            span("second", "second", 4, 8, status="error"),
        ),
        SpanProfileQuery(request_id="request", origin="user", status="error"),
    )
    assert [node.span_id for node in filtered.nodes] == ["second"]


def test_aggregate_merges_full_paths_without_double_counting():
    profile = project_span_profile(
        result(
            span("root", "r1", 0, 10, request_id="one", component_id="root-component"),
            span("child", "c1", 2, 6, parent_span_id="r1", request_id="one", component_id="child-component"),
            span("root", "r2", 20, 40, request_id="two", component_id="root-component"),
            span("child", "c2", 25, 30, parent_span_id="r2", request_id="two", component_id="child-component"),
        ),
        SpanProfileQuery(mode="aggregate"),
    )
    assert isinstance(profile, AggregateSpanProfileProjection)
    nodes = by_name(profile)

    assert len(profile.roots) == 1
    assert nodes["root"].self_value_ns == 21
    assert nodes["child"].self_value_ns == 9
    assert nodes["root"].value_ns == 30
    assert nodes["root"].occurrence_count == 2
    assert nodes["root"].request_count == 2
    assert nodes["root"].percentage == 100.0
    assert nodes["child"].percentage == 30.0
    assert profile.totals["value_ns"] == 30
    assert profile.totals["concurrency_factor"] == 1.0


def test_aggregate_path_identity_counts_invalid_occurrences_and_concurrency():
    profile = project_span_profile(
        result(
            span("root", "root", 0, 10, component_id="a"),
            span("work", "left", 1, 7, parent_span_id="root", component_id="b", status="error"),
            span("work", "right", 3, 9, parent_span_id="root", component_id="c"),
            span("work", "incomplete", 4, None, component_id="different", status="incomplete"),
            span("work", "negative", 12, 11, component_id="different"),
        ),
        SpanProfileQuery(mode="aggregate"),
    )

    work_nodes = [node for node in profile.nodes if node.name == "work"]
    assert len(work_nodes) == 3
    assert next(node for node in work_nodes if node.component_id == "b").error_count == 1
    excluded = next(node for node in work_nodes if node.component_id == "different")
    assert excluded.occurrence_count == 2
    assert excluded.incomplete_count == 1
    assert excluded.invalid_count == 1
    assert profile.totals["error_count"] == 1
    assert profile.totals["incomplete_count"] == 1
    assert profile.totals["invalid_count"] == 1
    assert profile.totals["concurrency_factor"] == pytest.approx(1.4)


def test_aggregate_identity_includes_the_complete_parent_path():
    profile = project_span_profile(
        result(
            span("root-a", "root-a", 0, 10, component_id="a"),
            span("work", "work-a", 1, 3, parent_span_id="root-a", component_id="leaf"),
            span("root-b", "root-b", 20, 30, component_id="b"),
            span("work", "work-b", 21, 23, parent_span_id="root-b", component_id="leaf"),
        ),
        SpanProfileQuery(mode="aggregate"),
    )
    work_nodes = [node for node in profile.nodes if node.name == "work"]

    assert len(work_nodes) == 2
    assert len({node.id for node in work_nodes}) == 2
    assert {node.path[0]["component_id"] for node in work_nodes} == {"a", "b"}


def test_aggregate_totals_scale_across_many_requests():
    request_count = 3_000
    profile = project_span_profile(
        result(*(span("work", str(index), 0, 10, request_id=f"request-{index}") for index in range(request_count))),
        SpanProfileQuery(mode="aggregate"),
    )

    assert profile.totals["request_count"] == request_count
    assert profile.totals["occurrence_count"] == request_count
    assert profile.totals["selected_wall_union_ns"] == request_count * 10
    assert profile.nodes[0].request_count == request_count


@pytest.mark.parametrize("mode", ["request", "aggregate"])
def test_truncation_is_explicit_deterministic_and_preserves_parents(mode):
    profile = project_span_profile(
        result(
            span("root", "root", 0, 10),
            span("child", "child", 1, 9, parent_span_id="root"),
            span("leaf", "leaf", 2, 8, parent_span_id="child"),
        ),
        SpanProfileQuery(mode=mode, request_id="request" if mode == "request" else "", max_nodes=2),
    )

    assert profile.total_nodes == 3
    assert profile.returned_nodes == 2
    assert profile.truncated
    assert profile.truncation_reason == "max_nodes"
    returned = {node.id for node in profile.nodes}
    assert all(not node.parent_id or node.parent_id in returned for node in profile.nodes)
    assert (
        profile.to_dict()
        == project_span_profile(
            result(
                span("root", "root", 0, 10),
                span("child", "child", 1, 9, parent_span_id="root"),
                span("leaf", "leaf", 2, 8, parent_span_id="child"),
            ),
            SpanProfileQuery(mode=mode, request_id="request" if mode == "request" else "", max_nodes=2),
        ).to_dict()
    )


def test_request_mode_requires_request_id_and_valid_node_limit():
    with pytest.raises(ValueError, match="requires request_id"):
        SpanProfileQuery()
    with pytest.raises(ValueError, match="positive"):
        SpanProfileQuery(mode="aggregate", max_nodes=0)


def test_span_profile_query_and_projection_are_public_exports():
    assert PublicSpanProfileQuery is SpanProfileQuery
    assert PublicSpanProfileProjection == RequestSpanProfileProjection | AggregateSpanProfileProjection
    assert public_project_span_profile is project_span_profile


def test_speedscope_uses_full_path_samples_and_instrumented_wall_schema():
    profile = project_span_profile(
        result(
            span("root", "root", 0, 10),
            span("child", "child", 2, 6, parent_span_id="root"),
        ),
        SpanProfileQuery(mode="aggregate"),
    )
    document = span_profile_speedscope(profile)
    sampled = document["profiles"][0]

    assert document["$schema"].endswith("file-format-schema.json")
    assert "Instrumented Span Wall Time" in sampled["name"]
    assert "CPU Time" not in json.dumps(document)
    assert sampled["type"] == "sampled"
    assert sampled["unit"] == "nanoseconds"
    assert sampled["weights"] == [6, 4]
    assert sorted(len(sample) for sample in sampled["samples"]) == [1, 2]

    truncated = project_span_profile(
        result(
            span("root", "root", 0, 10),
            span("child", "child", 2, 6, parent_span_id="root"),
        ),
        SpanProfileQuery(mode="aggregate", max_nodes=1),
    )
    with pytest.raises(ValueError, match="truncated"):
        span_profile_speedscope(truncated)


def test_ten_thousand_deep_hierarchies_do_not_depend_on_python_recursion():
    count = 10_050
    deep_spans = [
        span(
            f"span-{index}",
            str(index),
            index,
            count * 2 - index,
            parent_span_id=str(index - 1) if index else "",
        )
        for index in range(count)
    ]

    request_profile = project_span_profile(
        result(*deep_spans),
        SpanProfileQuery(request_id="request", max_nodes=25),
    )
    aggregate_profile = project_span_profile(
        result(*deep_spans),
        SpanProfileQuery(mode="aggregate", max_nodes=25),
    )

    assert request_profile.total_nodes == aggregate_profile.total_nodes == count
    assert request_profile.returned_nodes == aggregate_profile.returned_nodes == 25
    assert request_profile.nodes[-1].depth == aggregate_profile.nodes[-1].depth == 24


def test_ten_thousand_node_cycle_is_detected_without_recursion():
    count = 10_050
    hierarchy = build_span_hierarchy(
        [
            span(
                f"cycle-{index}",
                str(index),
                index,
                index,
                parent_span_id=str((index + 1) % count),
            )
            for index in range(count)
        ]
    )

    assert len(hierarchy.roots) == count
    assert sum("cycle" in node.diagnostics for node in hierarchy.nodes) == count


def test_cli_span_profile_json_and_speedscope_export(monkeypatch, capsys):
    analysis = result(
        span("root", "root", 0, 10),
        span("child", "child", 2, 6, parent_span_id="root"),
    )
    monkeypatch.setattr(cli, "_load", lambda _args: analysis)

    assert cli.main(["span-profile", str(Path("trace")), "--request-id", "request", "--format", "json"]) == 0
    profile_document = json.loads(capsys.readouterr().out)
    assert profile_document["mode"] == "request"

    assert cli.main(["export", str(Path("trace")), "--format", "speedscope"]) == 0
    speedscope_document = json.loads(capsys.readouterr().out)
    assert speedscope_document["profiles"][0]["type"] == "sampled"

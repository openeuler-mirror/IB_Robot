"""Internal span hierarchy and instrumented wall-interval model."""

from __future__ import annotations

import heapq
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from .model import SpanRecord
from .query import stable_span_id

DIAGNOSTIC_ORDER = (
    "orphan",
    "cycle",
    "duplicate_span_id",
    "incomplete",
    "negative_duration",
    "child_outside_parent",
    "overlapping_siblings",
    "filtered_parent",
)
_MAX_DIAGNOSTIC_OCCURRENCE_IDS = 20
_MAX_DIAGNOSTIC_MESSAGE_NODES = 8
_MAX_DIAGNOSTIC_MESSAGE_LENGTH = 512


@dataclass(slots=True)
class SpanHierarchyDiagnostic:
    code: str
    occurrence_ids: tuple[str, ...]
    message: str
    member_ids: frozenset[str] = field(repr=False)


@dataclass(slots=True)
class SpanHierarchyNode:
    occurrence_id: str
    span: SpanRecord
    candidate_parent_id: str = ""
    parent_id: str = ""
    child_ids: list[str] = field(default_factory=list)
    depth: int = 0
    track: int = 0
    end_ns: int | None = None
    observed_end_ns: int | None = None
    duration_ns: int | None = None
    duration_source: str = "unavailable"
    valid_for_weight: bool = False
    incomplete: bool = False
    negative_duration: bool = False
    uncovered_wall_ns: int | None = None
    compatibility_duration_ns: int | None = None
    compatibility_uncovered_ns: int | None = None
    diagnostics: set[str] = field(default_factory=set)


@dataclass(slots=True)
class SpanHierarchy:
    roots: list[str]
    nodes: list[SpanHierarchyNode]
    diagnostics: list[SpanHierarchyDiagnostic]

    @property
    def by_id(self) -> dict[str, SpanHierarchyNode]:
        return {node.occurrence_id: node for node in self.nodes}


def _compatibility_duration(span: SpanRecord) -> int | None:
    duration = span.fields.get("duration_ns")
    try:
        if duration is not None:
            return max(0, int(duration))
    except (TypeError, ValueError):
        pass
    if span.end_ns is None:
        return None
    return max(0, span.end_ns - span.start_ns)


def _geometry(node: SpanHierarchyNode) -> None:
    span = node.span
    node.observed_end_ns = span.end_ns
    node.incomplete = span.status == "incomplete" or span.end_ns is None
    raw_duration: Any = span.fields.get("duration_ns")
    monotonic_duration = None
    if raw_duration is not None:
        with suppress(TypeError, ValueError):
            monotonic_duration = int(raw_duration)
    if monotonic_duration is not None:
        node.duration_source = "monotonic"
        if monotonic_duration < 0:
            node.negative_duration = True
        else:
            node.duration_ns = monotonic_duration
            node.end_ns = span.start_ns + monotonic_duration
    elif span.end_ns is not None:
        node.duration_source = "observed"
        observed_duration = span.end_ns - span.start_ns
        if observed_duration < 0:
            node.negative_duration = True
        else:
            node.duration_ns = observed_duration
            node.end_ns = span.end_ns
    node.valid_for_weight = node.duration_ns is not None and not node.incomplete and not node.negative_duration
    node.compatibility_duration_ns = _compatibility_duration(span)
    if node.incomplete:
        node.diagnostics.add("incomplete")
    if node.negative_duration:
        node.diagnostics.add("negative_duration")


def _union_length(intervals: list[tuple[int, int]]) -> int:
    covered = 0
    current_start = current_end = None
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
        elif start > current_end:
            covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        covered += current_end - current_start
    return covered


def interval_union_ns(nodes: list[SpanHierarchyNode]) -> int:
    """Return the union of valid visual wall intervals."""
    return _union_length(
        [
            (node.span.start_ns, node.end_ns)
            for node in nodes
            if node.valid_for_weight and node.end_ns is not None and node.end_ns > node.span.start_ns
        ]
    )


def assign_tracks(
    node_ids: list[str],
    by_id: dict[str, SpanHierarchyNode],
) -> dict[str, int]:
    """Assign the lowest non-overlapping track in deterministic interval order."""
    tracks: dict[str, int] = {}
    active: list[tuple[int, int]] = []
    available: list[int] = []
    next_track = 0
    ordered = sorted(node_ids, key=lambda node_id: (by_id[node_id].span.start_ns, node_id))
    for node_id in ordered:
        node = by_id[node_id]
        if node.end_ns is None or node.end_ns <= node.span.start_ns:
            tracks[node_id] = 0
            continue
        while active and active[0][0] <= node.span.start_ns:
            _, track = heapq.heappop(active)
            heapq.heappush(available, track)
        if available:
            track = heapq.heappop(available)
        else:
            track = next_track
            next_track += 1
        tracks[node_id] = track
        heapq.heappush(active, (node.end_ns, track))
    return tracks


def _message(code: str, nodes: list[SpanHierarchyNode]) -> str:
    shown = nodes[:_MAX_DIAGNOSTIC_MESSAGE_NODES]
    names = ", ".join(f"{node.span.trace_id}/{node.span.span_id}" for node in shown)
    if len(nodes) > len(shown):
        names += f", ... (+{len(nodes) - len(shown)} more)"
    messages = {
        "orphan": "parent span was not observed",
        "cycle": "span parent relationship is cyclic",
        "duplicate_span_id": "span_id occurs more than once in one request",
        "incomplete": "span has no complete end interval",
        "negative_duration": "span duration is negative",
        "child_outside_parent": "child interval is not contained by its parent interval",
        "overlapping_siblings": f"sibling intervals form an overlap group affecting {len(nodes)} spans",
    }
    message = f"{messages[code]}: {names}"
    if len(message) > _MAX_DIAGNOSTIC_MESSAGE_LENGTH:
        return message[: _MAX_DIAGNOSTIC_MESSAGE_LENGTH - 3] + "..."
    return message


def _diagnostic(code: str, nodes: list[SpanHierarchyNode]) -> SpanHierarchyDiagnostic:
    member_ids = tuple(node.occurrence_id for node in nodes)
    return SpanHierarchyDiagnostic(
        code,
        member_ids[:_MAX_DIAGNOSTIC_OCCURRENCE_IDS],
        _message(code, nodes),
        frozenset(member_ids),
    )


def build_span_hierarchy(spans: list[SpanRecord]) -> SpanHierarchy:
    """Build deterministic occurrence identities, parent links, geometry, and diagnostics."""
    if not spans:
        return SpanHierarchy([], [], [])

    ordered = sorted(spans, key=lambda span: (span.start_ns, stable_span_id(span)))
    used: dict[str, int] = defaultdict(int)
    by_key: dict[tuple[str, str], str] = {}
    by_id: dict[str, SpanHierarchyNode] = {}
    duplicate_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for span in ordered:
        base_id = stable_span_id(span)
        used[base_id] += 1
        occurrence_id = base_id if used[base_id] == 1 else f"{base_id}:{used[base_id]}"
        node = SpanHierarchyNode(occurrence_id, span)
        _geometry(node)
        by_key.setdefault((span.trace_id, span.span_id), occurrence_id)
        by_id[occurrence_id] = node
        duplicate_groups[(span.trace_id, span.span_id)].append(occurrence_id)

    diagnostics: list[SpanHierarchyDiagnostic] = []
    for occurrence_ids in duplicate_groups.values():
        if len(occurrence_ids) < 2:
            continue
        duplicate_nodes = [by_id[occurrence_id] for occurrence_id in occurrence_ids]
        for node in duplicate_nodes:
            node.diagnostics.add("duplicate_span_id")
        diagnostics.append(_diagnostic("duplicate_span_id", duplicate_nodes))

    candidate_parent: dict[str, str] = {}
    for occurrence_id, node in by_id.items():
        span = node.span
        if not span.parent_span_id:
            continue
        parent_id = by_key.get((span.trace_id, span.parent_span_id), "")
        if parent_id:
            node.candidate_parent_id = parent_id
            candidate_parent[occurrence_id] = parent_id
        else:
            node.diagnostics.add("orphan")
            diagnostics.append(_diagnostic("orphan", [node]))

    cycles: set[str] = set()
    processed: set[str] = set()
    for occurrence_id in by_id:
        if occurrence_id in processed:
            continue
        path = []
        path_index: dict[str, int] = {}
        current_id = occurrence_id
        while current_id and current_id not in processed and current_id not in path_index:
            path_index[current_id] = len(path)
            path.append(current_id)
            current_id = candidate_parent.get(current_id, "")
        if current_id in path_index:
            cycles.update(path[path_index[current_id] :])
        processed.update(path)
    for occurrence_id in sorted(cycles):
        node = by_id[occurrence_id]
        node.diagnostics.add("cycle")
        diagnostics.append(_diagnostic("cycle", [node]))

    parent_by_id = {
        occurrence_id: parent_id for occurrence_id, parent_id in candidate_parent.items() if occurrence_id not in cycles
    }
    for occurrence_id, parent_id in parent_by_id.items():
        by_id[occurrence_id].parent_id = parent_id
        by_id[parent_id].child_ids.append(occurrence_id)
    for node in by_id.values():
        node.child_ids.sort(key=lambda occurrence_id: (by_id[occurrence_id].span.start_ns, occurrence_id))

    roots = [occurrence_id for occurrence_id in by_id if occurrence_id not in parent_by_id]
    roots.sort(key=lambda occurrence_id: (by_id[occurrence_id].span.start_ns, occurrence_id))

    stack = [(root_id, 0) for root_id in reversed(roots)]
    while stack:
        occurrence_id, depth = stack.pop()
        node = by_id[occurrence_id]
        node.depth = depth
        stack.extend((child_id, depth + 1) for child_id in reversed(node.child_ids))

    roots_by_request: dict[str, list[str]] = defaultdict(list)
    for root_id in roots:
        roots_by_request[by_id[root_id].span.trace_id].append(root_id)
    sibling_groups = [*roots_by_request.values(), *(node.child_ids for node in by_id.values())]
    for sibling_ids in sibling_groups:
        tracks = assign_tracks(sibling_ids, by_id)
        for occurrence_id, track in tracks.items():
            by_id[occurrence_id].track = track
        valid_siblings = [
            by_id[occurrence_id]
            for occurrence_id in sorted(sibling_ids, key=lambda item: (by_id[item].span.start_ns, item))
            if by_id[occurrence_id].end_ns is not None
            and by_id[occurrence_id].end_ns > by_id[occurrence_id].span.start_ns
        ]
        overlap_group: list[SpanHierarchyNode] = []
        group_end_ns: int | None = None

        def finish_overlap_group(group: list[SpanHierarchyNode]) -> None:
            if len(group) < 2:
                return
            for sibling in group:
                sibling.diagnostics.add("overlapping_siblings")
            diagnostics.append(_diagnostic("overlapping_siblings", group))

        for sibling in valid_siblings:
            if group_end_ns is None or sibling.span.start_ns >= group_end_ns:
                finish_overlap_group(overlap_group)
                overlap_group = [sibling]
                group_end_ns = sibling.end_ns
            else:
                overlap_group.append(sibling)
                if sibling.end_ns is not None:
                    group_end_ns = max(group_end_ns, sibling.end_ns)
        finish_overlap_group(overlap_group)

    for node in by_id.values():
        if node.incomplete:
            diagnostics.append(_diagnostic("incomplete", [node]))
        if node.negative_duration:
            diagnostics.append(_diagnostic("negative_duration", [node]))
        if node.parent_id:
            parent = by_id[node.parent_id]
            if (
                node.end_ns is not None
                and parent.end_ns is not None
                and (node.span.start_ns < parent.span.start_ns or node.end_ns > parent.end_ns)
            ):
                node.diagnostics.add("child_outside_parent")
                diagnostics.append(_diagnostic("child_outside_parent", [node, parent]))

        if node.valid_for_weight and node.end_ns is not None and node.duration_ns is not None:
            intervals = []
            for child_id in node.child_ids:
                child = by_id[child_id]
                if not child.valid_for_weight or child.end_ns is None:
                    continue
                start = max(node.span.start_ns, child.span.start_ns)
                end = min(node.end_ns, child.end_ns)
                if end > start:
                    intervals.append((start, end))
            node.uncovered_wall_ns = max(0, node.duration_ns - _union_length(intervals))

        compatibility_duration = node.compatibility_duration_ns
        if compatibility_duration is not None:
            compatibility_end = node.span.start_ns + compatibility_duration
            compatibility_intervals = []
            for child_id in node.child_ids:
                child = by_id[child_id]
                child_duration = child.compatibility_duration_ns
                if child_duration is None:
                    continue
                start = max(node.span.start_ns, child.span.start_ns)
                end = min(compatibility_end, child.span.start_ns + child_duration)
                if end > start:
                    compatibility_intervals.append((start, end))
            node.compatibility_uncovered_ns = max(0, compatibility_duration - _union_length(compatibility_intervals))

    diagnostic_rank = {code: index for index, code in enumerate(DIAGNOSTIC_ORDER)}
    diagnostics.sort(
        key=lambda diagnostic: (
            diagnostic_rank.get(diagnostic.code, len(diagnostic_rank)),
            diagnostic.occurrence_ids,
        )
    )
    return SpanHierarchy(roots, list(by_id.values()), diagnostics)

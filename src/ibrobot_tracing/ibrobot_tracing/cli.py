"""Command-line and lightweight interactive explorer adapters."""

from __future__ import annotations

import argparse
import cmd
import json
import shlex
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .analysis import AnalysisRequest, AnalysisResult, AnalysisService
from .critical_path import CriticalPathQuery, project_critical_path
from .projection import (
    AggregateSpanProfileProjection,
    CallTreeQuery,
    GraphQuery,
    LatencyDistributionQuery,
    SpanProfileQuery,
    TimelineQuery,
    project_call_tree,
    project_graph,
    project_latency_distribution,
    project_span_profile,
    project_timeline,
    span_profile_speedscope,
)
from .query import (
    MAX_PAGE_SIZE,
    REQUEST_SORT_FIELDS,
    ComponentQuery,
    EventQuery,
    FlowQuery,
    QueryService,
    RequestQuery,
    SpanQuery,
    TracepointQuery,
    stable_tracepoint_id,
)
from .rendering import (
    render_call_tree,
    render_critical_path,
    render_graph,
    render_json,
    render_latency_distribution,
    render_span_profile,
    render_summary,
    render_timeline,
)


def _load(args: argparse.Namespace) -> AnalysisResult:
    return AnalysisService().analyze(
        AnalysisRequest(source=args.source, source_kind=args.source_kind, topology_path=args.topology)
    )


def _take(fetch: Any, query: Any, limit: int | None = None) -> list[Any]:
    remaining = limit
    items = []
    offset = query.offset
    while remaining is None or remaining > 0:
        page_limit = MAX_PAGE_SIZE if remaining is None else min(MAX_PAGE_SIZE, remaining)
        page = fetch(replace(query, offset=offset, limit=page_limit))
        items.extend(page.items)
        if page.next_offset is None:
            break
        offset = page.next_offset
        if remaining is not None:
            remaining -= len(page.items)
    return items


def _print_json_line(value: Any) -> None:
    print(json.dumps(value, sort_keys=True))


class TraceExplorer(cmd.Cmd):
    intro = "IB-Robot Tracing. Type help or ? to list commands."
    prompt = "trace[all]> "

    def __init__(self, result: AnalysisResult):
        super().__init__()
        self.result = result
        self.query = QueryService(result)
        self.request_id = ""
        self.component_id = ""

    def _prompt(self) -> None:
        context = self.request_id or "all"
        if self.component_id:
            context += ":" + self.component_id
        self.prompt = f"trace[{context}]> "

    def do_summary(self, _line: str) -> None:
        """Show aggregate performance summary."""
        render_summary(self.result, sys.stdout)

    def do_requests(self, line: str) -> None:
        """List requests: requests [limit]."""
        limit = int(line or "20")
        for row in _take(self.query.requests, RequestQuery(), limit):
            _print_json_line(row)

    def do_use(self, line: str) -> None:
        """Select context: use request ID | use component ID | use all."""
        words = shlex.split(line)
        if words == ["all"]:
            self.request_id = self.component_id = ""
        elif len(words) == 2 and words[0] == "request":
            self.request_id = words[1]
        elif len(words) == 2 and words[0] == "component":
            self.component_id = words[1]
        else:
            print("Usage: use request ID | use component ID | use all")
        self._prompt()

    def do_graph(self, line: str) -> None:
        """Show topology: graph [p50|p95|p99|minimum|maximum|mean] [nodes|components|tracepoints]."""
        words = shlex.split(line)
        metric = words[0] if words else "p95"
        view = words[1] if len(words) > 1 else "components"
        valid_metrics = {"p50", "p95", "p99", "minimum", "maximum", "mean"}
        if metric not in valid_metrics or view not in {"nodes", "components", "tracepoints"}:
            print("Usage: graph [p50|p95|p99|minimum|maximum|mean] [nodes|components|tracepoints]")
            return
        render_graph(
            project_graph(self.result, GraphQuery(request_id=self.request_id, metric=metric, view=view)),
            sys.stdout,
        )

    def do_timeline(self, _line: str) -> None:
        """Show the current request/component timeline."""
        render_timeline(
            project_timeline(
                self.result,
                TimelineQuery(request_id=self.request_id, component_id=self.component_id),
            ),
            sys.stdout,
        )

    def do_call_tree(self, _line: str) -> None:
        """Show structured span hierarchy for the current context."""
        render_call_tree(
            project_call_tree(
                self.result,
                CallTreeQuery(request_id=self.request_id, component_id=self.component_id),
            ),
            sys.stdout,
        )

    def do_components(self, _line: str) -> None:
        """List topology components and their IDs."""
        if self.result.topology is None:
            print("No topology manifest is available.")
            return
        for component in _take(self.query.components, ComponentQuery()):
            print(f"{component.component_id:<36} {component.kind:<12} {component.name}")

    def do_tracepoints(self, line: str) -> None:
        """List tracepoint definitions: tracepoints [limit]."""
        limit = int(line or "100")
        for definition in _take(
            self.query.tracepoints,
            TracepointQuery(component_id=self.component_id),
            limit,
        ):
            _print_json_line({"id": stable_tracepoint_id(definition), **definition.to_dict()})

    def do_spans(self, line: str) -> None:
        """List structured spans: spans [limit]."""
        limit = int(line or "20")
        spans = _take(
            self.query.spans,
            SpanQuery(request_id=self.request_id, component_id=self.component_id),
            limit,
        )
        if not spans:
            print("No structured spans found.")
            return
        for span in spans:
            _print_json_line(asdict(span) | {"duration_ms": span.duration_ms})

    def do_flows(self, line: str) -> None:
        """List correlated flows: flows [limit]."""
        limit = int(line or "20")
        flows = _take(self.query.flows, FlowQuery(request_id=self.request_id), limit)
        if not flows:
            print("No correlated flows found.")
            return
        for flow in flows:
            _print_json_line(asdict(flow) | {"duration_ms": flow.duration_ms})

    def do_warnings(self, _line: str) -> None:
        """Show parser and analysis diagnostics."""
        print("\n".join(self.result.warnings) or "No warnings.")

    def do_quit(self, _line: str) -> bool:
        """Exit the explorer."""
        return True

    do_exit = do_quit
    do_EOF = do_quit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ibrobot-trace", description="IB-Robot offline performance analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "summary",
        "distribution",
        "requests",
        "timeline",
        "call-tree",
        "critical-path",
        "span-profile",
        "graph",
        "events",
        "spans",
        "flows",
        "components",
        "tracepoints",
        "export",
        "explore",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("source", type=Path)
        command.add_argument("--source-kind", choices=("auto", "ctf", "log"), default="auto")
        command.add_argument("--topology", type=Path)
        command.add_argument("--request-id", required=name == "critical-path", default="")
        command.add_argument("--component", default="")
    subparsers.choices["requests"].add_argument("--sort", choices=sorted(REQUEST_SORT_FIELDS), default="total_ms")
    subparsers.choices["requests"].add_argument("--limit", type=int, default=20)
    subparsers.choices["events"].add_argument("--event", default="")
    subparsers.choices["events"].add_argument("--provider", default="")
    subparsers.choices["events"].add_argument("--limit", type=int, default=100)
    subparsers.choices["spans"].add_argument("--limit", type=int, default=100)
    subparsers.choices["flows"].add_argument("--limit", type=int, default=100)
    subparsers.choices["tracepoints"].add_argument("--kind", choices=("event", "span"), default="")
    subparsers.choices["tracepoints"].add_argument("--name", default="")
    subparsers.choices["tracepoints"].add_argument("--origin", default="")
    subparsers.choices["tracepoints"].add_argument("--limit", type=int, default=100)
    subparsers.choices["distribution"].add_argument("--metric")
    subparsers.choices["distribution"].add_argument("--bins", type=int, default=30)
    subparsers.choices["distribution"].add_argument("--outlier-limit", type=int, default=20)
    subparsers.choices["distribution"].add_argument("--format", choices=("text", "json"), default="text")
    subparsers.choices["span-profile"].add_argument("--mode", choices=("request", "aggregate"), default="request")
    subparsers.choices["span-profile"].add_argument("--format", choices=("text", "json"), default="text")
    subparsers.choices["span-profile"].add_argument("--start-ns", type=int)
    subparsers.choices["span-profile"].add_argument("--end-ns", type=int)
    subparsers.choices["span-profile"].add_argument("--origin", default="")
    subparsers.choices["span-profile"].add_argument("--status", default="")
    subparsers.choices["span-profile"].add_argument("--max-nodes", type=int, default=10_000)
    subparsers.choices["critical-path"].add_argument("--start-ns", type=int)
    subparsers.choices["critical-path"].add_argument("--end-ns", type=int)
    subparsers.choices["critical-path"].add_argument("--no-flows", action="store_true")
    subparsers.choices["critical-path"].add_argument("--max-segments", type=int, default=1_000)
    subparsers.choices["critical-path"].add_argument("--format", choices=("text", "json"), default="text")
    subparsers.choices["export"].add_argument("--format", choices=("json", "legacy-json", "speedscope"), default="json")
    subparsers.choices["graph"].add_argument(
        "--metric", choices=("p50", "p95", "p99", "minimum", "maximum", "mean"), default="p95"
    )
    subparsers.choices["graph"].add_argument(
        "--view", choices=("nodes", "components", "tracepoints"), default="components"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _load(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    if not result.dataset.events and args.command not in {"critical-path", "tracepoints"}:
        print("No ib_trace events found.", file=sys.stderr)
        return 1
    if args.command == "summary":
        render_summary(result, sys.stdout)
    elif args.command == "distribution":
        try:
            distribution = project_latency_distribution(
                result,
                LatencyDistributionQuery(
                    metric=args.metric,
                    bins=args.bins,
                    outlier_limit=args.outlier_limit,
                ),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        if args.format == "json":
            print(json.dumps(distribution.to_dict(), indent=2))
        else:
            render_latency_distribution(distribution, sys.stdout)
    elif args.command == "requests":
        for row in _take(QueryService(result).requests, RequestQuery(sort_by=args.sort), args.limit):
            _print_json_line(row)
    elif args.command == "timeline":
        render_timeline(
            project_timeline(result, TimelineQuery(request_id=args.request_id, component_id=args.component)),
            sys.stdout,
        )
    elif args.command == "call-tree":
        render_call_tree(
            project_call_tree(result, CallTreeQuery(request_id=args.request_id, component_id=args.component)),
            sys.stdout,
        )
    elif args.command == "span-profile":
        try:
            profile = project_span_profile(
                result,
                SpanProfileQuery(
                    mode=args.mode,
                    request_id=args.request_id,
                    component_id=args.component,
                    start_ns=args.start_ns,
                    end_ns=args.end_ns,
                    origin=args.origin,
                    status=args.status,
                    max_nodes=args.max_nodes,
                ),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        if args.format == "json":
            print(json.dumps(profile.to_dict(), indent=2))
        else:
            render_span_profile(profile, sys.stdout)
    elif args.command == "critical-path":
        try:
            critical_path = project_critical_path(
                result,
                CriticalPathQuery(
                    request_id=args.request_id,
                    component_id=args.component,
                    start_ns=args.start_ns,
                    end_ns=args.end_ns,
                    include_flows=not args.no_flows,
                    max_segments=args.max_segments,
                ),
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        if args.format == "json":
            print(json.dumps(critical_path.to_dict(), indent=2))
        else:
            render_critical_path(critical_path, sys.stdout)
    elif args.command == "graph":
        render_graph(
            project_graph(result, GraphQuery(request_id=args.request_id, metric=args.metric, view=args.view)),
            sys.stdout,
        )
    elif args.command == "events":
        events = _take(
            QueryService(result).events,
            EventQuery(
                request_id=args.request_id,
                component_id=args.component,
                event_name=args.event,
                provider=args.provider,
            ),
            args.limit,
        )
        for event in events:
            _print_json_line(event.to_dict())
    elif args.command == "spans":
        spans = _take(
            QueryService(result).spans,
            SpanQuery(request_id=args.request_id, component_id=args.component),
            args.limit,
        )
        if not spans:
            print("No structured spans found.")
        for span in spans:
            _print_json_line(asdict(span) | {"duration_ms": span.duration_ms})
    elif args.command == "flows":
        flows = _take(QueryService(result).flows, FlowQuery(request_id=args.request_id), args.limit)
        if not flows:
            print("No correlated flows found.")
        for flow in flows:
            _print_json_line(asdict(flow) | {"duration_ms": flow.duration_ms})
    elif args.command == "components":
        if result.topology is None:
            print("No topology manifest is available.")
        else:
            for component in _take(QueryService(result).components, ComponentQuery()):
                print(f"{component.component_id:<36} {component.kind:<12} {component.name}")
    elif args.command == "tracepoints":
        definitions = _take(
            QueryService(result).tracepoints,
            TracepointQuery(
                kind=args.kind,
                component_id=args.component,
                name=args.name,
                origin=args.origin,
            ),
            args.limit,
        )
        if not definitions:
            print("No tracepoint definitions found.")
        for definition in definitions:
            _print_json_line({"id": stable_tracepoint_id(definition), **definition.to_dict()})
    elif args.command == "export":
        if args.format == "speedscope":
            profile = project_span_profile(
                result,
                SpanProfileQuery(
                    mode="aggregate",
                    request_id=args.request_id,
                    component_id=args.component,
                    max_nodes=max(1, len(result.spans)),
                ),
            )
            assert isinstance(profile, AggregateSpanProfileProjection)
            print(json.dumps(span_profile_speedscope(profile), indent=2))
        else:
            render_json(result, sys.stdout, compatibility=args.format == "legacy-json")
    elif args.command == "explore":
        TraceExplorer(result).cmdloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

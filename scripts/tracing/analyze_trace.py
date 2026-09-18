#!/usr/bin/env python3
"""Compatibility CLI for the IB-Robot offline trace analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ibrobot_tracing import AnalysisRequest, AnalysisService
from ibrobot_tracing.rendering import render_json, render_legacy_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze IB-Robot trace latency")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace-dir", type=Path, help="CTF trace directory")
    source.add_argument("--log-file", type=Path, help="Log file (IB_TRACE_MODE=log)")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    path = args.trace_dir or args.log_file
    source_kind = "ctf" if args.trace_dir else "log"
    try:
        result = AnalysisService().analyze(AnalysisRequest(source=path, source_kind=source_kind))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3

    if not result.dataset.events:
        print("No ib_trace events found.", file=sys.stderr)
        return 1

    print(f"Parsed {len(result.dataset.events)} ib_trace events", file=sys.stderr)
    print(f"Found {len(result.request_rows)} requests", file=sys.stderr)
    if args.format == "json":
        render_json(result, sys.stdout, compatibility=True)
    else:
        render_legacy_summary(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

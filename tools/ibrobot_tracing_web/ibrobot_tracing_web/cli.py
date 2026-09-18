"""Command-line entry point for the tracing web service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from .app import create_app
from .config import WebConfig, is_loopback_host, validate_lan_exposure


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibrobot-tracing-web",
        description="Serve the IB-Robot offline tracing API",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--trace-root",
        type=Path,
        action="append",
        help="Allowed trace catalog root; repeat for multiple roots (default: ~/.ros/tracing)",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        help="Exact HTTP Host header accepted by TrustedHostMiddleware; repeat as needed",
    )
    parser.add_argument("--cors-origin", action="append", help="Exact allowed CORS origin; CORS is off by default")
    parser.add_argument(
        "--allow-unauthenticated-lan",
        action="store_true",
        help="Acknowledge that the no-auth API will be reachable beyond loopback",
    )
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--max-analyses", type=int, default=8)
    parser.add_argument("--max-job-history", type=int, default=64)
    parser.add_argument("--max-sources", type=int, default=256)
    parser.add_argument("--max-scan-entries", type=int, default=100_000)
    parser.add_argument("--max-source-bytes", type=int, default=WebConfig().max_source_bytes)
    parser.add_argument("--max-events", type=int, default=WebConfig().max_events)
    parser.add_argument("--max-result-bytes", type=int, default=WebConfig().max_result_bytes)
    parser.add_argument("--web-root", type=Path, help="Vue dist directory; auto-discovered when omitted")
    parser.add_argument("--certfile", type=Path)
    parser.add_argument("--keyfile", type=Path)
    parser.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug"), default="info")
    return parser


def _default_allowed_hosts(host: str) -> tuple[str, ...]:
    hosts = ["localhost", "127.0.0.1"]
    if host not in {"0.0.0.0", "::"} and host not in hosts:
        hosts.append(host)
    return tuple(hosts)


def parse_config(argv: list[str] | None = None) -> WebConfig:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.host in {"0.0.0.0", "::"} and not args.allowed_host:
        parser.error("wildcard bind addresses require at least one exact --allowed-host for LAN clients")
    try:
        config = WebConfig(
            host=args.host,
            port=args.port,
            trace_roots=tuple(args.trace_root) if args.trace_root else WebConfig().trace_roots,
            allowed_hosts=tuple(args.allowed_host) if args.allowed_host else _default_allowed_hosts(args.host),
            cors_origins=tuple(args.cors_origin or ()),
            allow_unauthenticated_lan=args.allow_unauthenticated_lan,
            queue_size=args.queue_size,
            max_analyses=args.max_analyses,
            max_job_history=args.max_job_history,
            max_sources=args.max_sources,
            max_scan_entries=args.max_scan_entries,
            max_source_bytes=args.max_source_bytes,
            max_events=args.max_events,
            max_result_bytes=args.max_result_bytes,
            web_root=args.web_root if args.web_root else WebConfig().web_root,
            certfile=args.certfile,
            keyfile=args.keyfile,
            log_level=args.log_level,
        )
        validate_lan_exposure(config)
    except ValueError as exc:
        parser.error(str(exc))
    for label, path in (("certificate", config.certfile), ("private key", config.keyfile)):
        if path is not None and not path.is_file():
            parser.error(f"TLS {label} file does not exist: {path}")
    if config.web_root is not None and not (config.web_root / "index.html").is_file():
        parser.error(f"Web root does not contain index.html: {config.web_root}")
    return config


def main(argv: list[str] | None = None) -> int:
    config = parse_config(argv)
    if not is_loopback_host(config.host):
        print(
            "WARNING: exposing the unauthenticated IB-Robot tracing API to the LAN; "
            "restrict network access and use TLS for untrusted networks.",
            file=sys.stderr,
        )
    if config.web_root is None:
        print("WARNING: Vue assets were not found; serving the analysis API only.", file=sys.stderr)
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,
        log_level=config.log_level,
        ssl_certfile=str(config.certfile) if config.certfile else None,
        ssl_keyfile=str(config.keyfile) if config.keyfile else None,
        proxy_headers=False,
        server_header=False,
        ws="none",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

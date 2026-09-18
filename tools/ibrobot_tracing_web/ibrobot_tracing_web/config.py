"""Server configuration and LAN exposure safeguards."""

from __future__ import annotations

import ipaddress
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


def default_trace_roots() -> tuple[Path, ...]:
    return (Path.home() / ".ros" / "tracing",)


def default_web_root() -> Path | None:
    candidates = [
        Path(sys.prefix) / "share" / "ibrobot_tracing_web" / "web",
        Path(__file__).resolve().parents[3] / "web" / "ibrobot_tracing_ui" / "dist",
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.insert(0, Path(get_package_share_directory("ibrobot_tracing_web")) / "web")
    except (ImportError, LookupError):
        pass
    return next((path for path in candidates if (path / "index.html").is_file()), None)


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _validate_allowed_host(host: str) -> str:
    value = host.strip()
    if not value or "*" in value or "/" in value or "://" in value or any(character.isspace() for character in value):
        raise ValueError(f"Allowed hosts must be exact host names or addresses: {host!r}")
    return value


def _validate_cors_origin(origin: str) -> str:
    value = origin.strip()
    parsed = urlsplit(value)
    if (
        not value
        or "*" in value
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or value != f"{parsed.scheme}://{parsed.netloc}"
    ):
        raise ValueError(f"CORS origins must be exact HTTP(S) origins without paths: {origin!r}")
    return value


@dataclass(frozen=True, slots=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    trace_roots: tuple[Path, ...] = field(default_factory=default_trace_roots)
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    cors_origins: tuple[str, ...] = ()
    allow_unauthenticated_lan: bool = False
    queue_size: int = 8
    max_analyses: int = 8
    max_job_history: int = 64
    max_sources: int = 256
    max_scan_entries: int = 100_000
    max_source_bytes: int = 67_108_864
    max_events: int = 200_000
    max_result_bytes: int = 268_435_456
    web_root: Path | None = field(default_factory=default_web_root)
    certfile: Path | None = None
    keyfile: Path | None = None
    log_level: str = "info"

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("Host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        if not self.trace_roots:
            raise ValueError("At least one trace root is required")
        if not self.allowed_hosts:
            raise ValueError("At least one exact allowed host is required")
        if self.queue_size < 1 or self.max_analyses < 1 or self.max_job_history < 1:
            raise ValueError("Queue, analysis, and job-history limits must be positive")
        if self.max_job_history < self.max_analyses:
            raise ValueError("Job-history limit must be at least the analysis limit")
        if self.max_sources < 1 or self.max_scan_entries < 1 or self.max_source_bytes < 1 or self.max_events < 1:
            raise ValueError("Catalog limits must be positive")
        if self.max_result_bytes < 1:
            raise ValueError("Result memory budget must be positive")
        if (self.certfile is None) != (self.keyfile is None):
            raise ValueError("--certfile and --keyfile must be provided together")

        roots = tuple(Path(root).expanduser() for root in self.trace_roots)
        for root in roots:
            if ".." in root.parts:
                raise ValueError(f"Trace roots must not contain '..': {root}")
        object.__setattr__(self, "trace_roots", roots)
        object.__setattr__(self, "allowed_hosts", tuple(_validate_allowed_host(host) for host in self.allowed_hosts))
        object.__setattr__(self, "cors_origins", tuple(_validate_cors_origin(origin) for origin in self.cors_origins))
        object.__setattr__(self, "web_root", self.web_root.expanduser().resolve() if self.web_root else None)
        object.__setattr__(self, "certfile", self.certfile.expanduser() if self.certfile else None)
        object.__setattr__(self, "keyfile", self.keyfile.expanduser() if self.keyfile else None)

    @property
    def tls_enabled(self) -> bool:
        return self.certfile is not None


def validate_lan_exposure(config: WebConfig) -> None:
    if not is_loopback_host(config.host) and not config.allow_unauthenticated_lan:
        raise ValueError("Binding the unauthenticated API to a non-loopback host requires --allow-unauthenticated-lan")

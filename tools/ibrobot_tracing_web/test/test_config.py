from pathlib import Path

import pytest

from ibrobot_tracing_web.cli import parse_config
from ibrobot_tracing_web.config import WebConfig, validate_lan_exposure


def test_non_loopback_requires_explicit_no_auth_acknowledgement(tmp_path):
    config = WebConfig(host="0.0.0.0", trace_roots=(tmp_path,), allowed_hosts=("robot.local",))

    with pytest.raises(ValueError, match="allow-unauthenticated-lan"):
        validate_lan_exposure(config)

    validate_lan_exposure(
        WebConfig(
            host="0.0.0.0",
            trace_roots=(tmp_path,),
            allowed_hosts=("robot.local",),
            allow_unauthenticated_lan=True,
        )
    )


@pytest.mark.parametrize("host", ["*", "*.example.com", "http://robot.local", "robot.local/path"])
def test_allowed_hosts_must_be_exact(tmp_path, host):
    with pytest.raises(ValueError, match="exact"):
        WebConfig(trace_roots=(tmp_path,), allowed_hosts=(host,))


@pytest.mark.parametrize("origin", ["*", "http://example.com/path", "file:///tmp", "http://*.example.com"])
def test_cors_origins_must_be_exact_http_origins(tmp_path, origin):
    with pytest.raises(ValueError, match="exact"):
        WebConfig(trace_roots=(tmp_path,), cors_origins=(origin,))


def test_trace_root_traversal_and_partial_tls_configuration_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not contain"):
        WebConfig(trace_roots=(Path("safe") / ".." / "other",))
    with pytest.raises(ValueError, match="together"):
        WebConfig(trace_roots=(tmp_path,), certfile=tmp_path / "cert.pem")
    with pytest.raises(ValueError, match="limits"):
        WebConfig(trace_roots=(tmp_path,), max_events=0)


def test_removed_baseline_root_option_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        parse_config(["--trace-root", str(tmp_path), "--baseline-root", str(tmp_path / "baselines")])

import asyncio
import importlib.util
import json
from urllib.parse import urlsplit

import pytest

_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")

if _FASTAPI_AVAILABLE:
    from ibrobot_tracing_web.app import create_app
    from ibrobot_tracing_web.config import WebConfig


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.content = body
        self.text = body.decode(errors="replace")

    def json(self):
        return json.loads(self.content)


async def _request(app, method, path, *, body=None):
    url = urlsplit(path)
    payload = json.dumps(body).encode() if body is not None else b""
    request_sent = False
    messages = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    headers = [(b"host", b"testserver")]
    if body is not None:
        headers.append((b"content-type", b"application/json"))
    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": url.path,
            "raw_path": url.path.encode(),
            "query_string": url.query.encode(),
            "headers": headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return _Response(start["status"], response_body)


async def _wait_for_job(app, endpoint, job_id):
    for _ in range(200):
        response = await _request(app, "GET", f"/api/v1/{endpoint}/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"queued", "running"}:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"{endpoint} job did not finish")


def _write_trace(path, latency_ms, request_id):
    path.write_text(
        "\n".join(
            (
                "IBTRACE1 "
                + json.dumps(
                    {
                        "timestamp_ns": 1000000000,
                        "event": "dispatch_request",
                        "fields": {"trace_id": request_id, "component_id": "action_dispatcher.request"},
                    }
                ),
                "IBTRACE1 "
                + json.dumps(
                    {
                        "timestamp_ns": 1000000000 + latency_ms * 1000000,
                        "event": "first_action_execute",
                        "fields": {
                            "trace_id": request_id,
                            "component_id": "action_dispatcher.execute",
                            "publish_ms": 1,
                            "publish_end_ns": 1000000000 + latency_ms * 1000000,
                        },
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _comparison_body(baseline, analysis_id, **overrides):
    return {
        "baseline_source_id": baseline["id"],
        "baseline_source_version": baseline["version"],
        "candidate_analysis_id": analysis_id,
        "statistic": "p95",
        "relative_threshold_percent": 5.0,
        "absolute_threshold_ms": 1.0,
        "metric": "total_ms",
        "bins": 5,
        **overrides,
    }


async def _analyze(app, source_id):
    created = await _request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source_id})
    return await _wait_for_job(app, "analysis-jobs", created.json()["id"])


def test_capabilities_expose_direct_trace_comparison_without_baseline_registry(tmp_path):
    async def run():
        trace_root = tmp_path / "traces"
        trace_root.mkdir()
        app = create_app(WebConfig(trace_roots=(trace_root,), allowed_hosts=("testserver",)))

        async with app.router.lifespan_context(app):
            capabilities = (await _request(app, "GET", "/api/v1/capabilities")).json()
            assert capabilities["baseline_compare"]
            assert capabilities["limits"]["comparison_queue_size"] == 8
            assert all(not key.startswith("baseline_max_") for key in capabilities["limits"])

            openapi = (await _request(app, "GET", "/openapi.json")).json()
            assert "/api/v1/baselines" not in openapi["paths"]
            assert "/api/v1/comparison-jobs" in openapi["paths"]

    asyncio.run(run())


def test_comparison_api_uses_two_catalog_traces_dedupes_and_never_leaks_paths(tmp_path):
    async def run():
        trace_root = tmp_path / "traces"
        trace_root.mkdir()
        baseline_path = _write_trace(trace_root / "baseline.log", 10, "baseline")
        candidate_path = _write_trace(trace_root / "candidate.log", 20, "candidate")
        app = create_app(
            WebConfig(
                trace_roots=(trace_root,),
                allowed_hosts=("testserver",),
                max_analyses=2,
                max_job_history=4,
            )
        )

        async with app.router.lifespan_context(app):
            sources = (await _request(app, "GET", "/api/v1/sources")).json()["items"]
            by_name = {source["name"]: source for source in sources}
            analysis = await _analyze(app, by_name["candidate.log"]["id"])
            body = _comparison_body(by_name["baseline.log"], analysis["analysis_id"])

            created = await _request(app, "POST", "/api/v1/comparison-jobs", body=body)
            assert created.status_code == 202, created.text
            job = await _wait_for_job(app, "comparison-jobs", created.json()["id"])
            assert job["status"] == "completed", job

            duplicate = await _request(app, "POST", "/api/v1/comparison-jobs", body=body)
            assert duplicate.status_code == 200
            assert duplicate.json()["deduplicated"]

            response = await _request(app, "GET", f"/api/v1/comparisons/{job['comparison_id']}")
            comparison = response.json()
            assert comparison["baseline_source_id"] == by_name["baseline.log"]["id"]
            assert comparison["baseline_source_version"] == by_name["baseline.log"]["version"]
            assert comparison["comparable"]
            assert comparison["has_regression"]
            assert comparison["histogram"]["baseline_count"] == 1
            assert comparison["histogram"]["candidate_count"] == 1
            assert str(tmp_path) not in response.text
            assert str(baseline_path) not in response.text
            assert str(candidate_path) not in response.text

            deleted = await _request(app, "DELETE", f"/api/v1/comparisons/{job['comparison_id']}")
            assert deleted.status_code == 204
            assert baseline_path.is_file()
            assert (await _request(app, "GET", f"/api/v1/comparisons/{job['comparison_id']}")).status_code == 404

    asyncio.run(run())


def test_zero_baseline_and_non_comparable_results_remain_json_safe(tmp_path):
    async def run():
        trace_root = tmp_path / "traces"
        trace_root.mkdir()
        _write_trace(trace_root / "baseline.log", 0, "baseline")
        _write_trace(trace_root / "candidate.log", 10, "candidate")
        (trace_root / "empty.log").write_text("[1000000000] ib_trace.test [unrelated]\n", encoding="utf-8")
        app = create_app(WebConfig(trace_roots=(trace_root,), allowed_hosts=("testserver",), max_analyses=2))

        async with app.router.lifespan_context(app):
            sources = (await _request(app, "GET", "/api/v1/sources")).json()["items"]
            by_name = {source["name"]: source for source in sources}
            candidate = await _analyze(app, by_name["candidate.log"]["id"])
            created = await _request(
                app,
                "POST",
                "/api/v1/comparison-jobs",
                body=_comparison_body(
                    by_name["baseline.log"],
                    candidate["analysis_id"],
                    relative_threshold_percent=1_000_000.0,
                ),
            )
            job = await _wait_for_job(app, "comparison-jobs", created.json()["id"])
            result = (await _request(app, "GET", f"/api/v1/comparisons/{job['comparison_id']}")).json()
            metric = next(item for item in result["metrics"] if item["metric"] == "total_ms")
            assert result["comparable"]
            assert metric["deltas"]["p95"]["percent_delta"] is None

            empty = await _analyze(app, by_name["empty.log"]["id"])
            created = await _request(
                app,
                "POST",
                "/api/v1/comparison-jobs",
                body=_comparison_body(by_name["baseline.log"], empty["analysis_id"]),
            )
            job = await _wait_for_job(app, "comparison-jobs", created.json()["id"])
            result = (await _request(app, "GET", f"/api/v1/comparisons/{job['comparison_id']}")).json()
            assert not result["comparable"]
            assert result["blocking_reasons"]

    asyncio.run(run())


def test_comparison_rejects_invalid_stale_missing_and_same_source_inputs(tmp_path):
    async def run():
        trace_root = tmp_path / "traces"
        trace_root.mkdir()
        _write_trace(trace_root / "baseline.log", 10, "baseline")
        _write_trace(trace_root / "candidate.log", 20, "candidate")
        app = create_app(WebConfig(trace_roots=(trace_root,), allowed_hosts=("testserver",)))

        async with app.router.lifespan_context(app):
            sources = (await _request(app, "GET", "/api/v1/sources")).json()["items"]
            by_name = {source["name"]: source for source in sources}
            candidate = await _analyze(app, by_name["candidate.log"]["id"])
            valid = _comparison_body(by_name["baseline.log"], candidate["analysis_id"])

            for body in (
                valid | {"bins": 4},
                valid | {"relative_threshold_percent": -1.0},
                valid | {"baseline_source_id": "/tmp/baseline"},
                valid | {"baseline_source_version": "stale"},
                valid | {"filesystem_path": "/tmp/trace"},
            ):
                assert (await _request(app, "POST", "/api/v1/comparison-jobs", body=body)).status_code == 422

            stale = valid | {"baseline_source_version": "0" * 64}
            assert (await _request(app, "POST", "/api/v1/comparison-jobs", body=stale)).status_code == 409
            missing = valid | {"baseline_source_id": "f" * 32}
            assert (await _request(app, "POST", "/api/v1/comparison-jobs", body=missing)).status_code == 404
            same = _comparison_body(by_name["candidate.log"], candidate["analysis_id"])
            assert (await _request(app, "POST", "/api/v1/comparison-jobs", body=same)).status_code == 409

    asyncio.run(run())

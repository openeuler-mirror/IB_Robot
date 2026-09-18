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
    def __init__(self, status_code, headers, body):
        self.status_code = status_code
        self.headers = headers
        self.content = body
        self.text = body.decode(errors="replace")

    def json(self):
        return json.loads(self.content)


async def _request(app, method, path, *, body=None, host="testserver"):
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

    headers = [(b"host", host.encode())]
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
            "server": (host, 80),
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return _Response(start["status"], dict(start.get("headers", [])), response_body)


async def _wait_for_job(app, job_id):
    for _ in range(200):
        response = await _request(app, "GET", f"/api/v1/analysis-jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"queued", "running"}:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("analysis job did not finish")


def test_full_api_lifecycle_and_host_protection(tmp_path):
    async def run():
        trace = tmp_path / "robot.log"
        records = [
            {
                "schema_version": 1,
                "timestamp_ns": 1,
                "event": "_tracepoint_definition",
                "fields": {
                    "kind": "event",
                    "component_id": "worker",
                    "name": "ready",
                    "origin": "user",
                    "description": "Ready for work.",
                },
            },
            {
                "schema_version": 1,
                "timestamp_ns": 2,
                "event": "_tracepoint_definition",
                "fields": {
                    "kind": "span",
                    "component_id": "worker",
                    "name": "work",
                    "origin": "user",
                    "description": "Runs one work item.",
                },
            },
            {
                "schema_version": 1,
                "timestamp_ns": 1_000,
                "event": "ready",
                "fields": {"trace_id": "req-1", "component_id": "worker", "origin": "user"},
            },
            {
                "schema_version": 1,
                "timestamp_ns": 1_100,
                "event": "plain",
                "fields": {"trace_id": "req-1", "component_id": "worker", "origin": "user"},
            },
            {
                "schema_version": 1,
                "timestamp_ns": 1_200,
                "event": "span_begin",
                "fields": {
                    "trace_id": "req-1",
                    "span_id": "work-1",
                    "span_name": "work",
                    "component_id": "worker",
                    "origin": "user",
                },
            },
            {
                "schema_version": 1,
                "timestamp_ns": 1_300,
                "event": "span_end",
                "fields": {
                    "trace_id": "req-1",
                    "span_id": "work-1",
                    "span_name": "work",
                    "component_id": "worker",
                    "origin": "user",
                },
            },
        ]
        trace.write_text("\n".join(f"IBTRACE1 {json.dumps(record)}" for record in records), encoding="utf-8")
        config = WebConfig(trace_roots=(tmp_path,), allowed_hosts=("testserver",), max_analyses=2, max_job_history=4)
        app = create_app(config)

        async with app.router.lifespan_context(app):
            assert (await _request(app, "GET", "/healthz")).json()["status"] == "ok"
            capabilities = (await _request(app, "GET", "/api/v1/capabilities")).json()
            assert capabilities["authentication"] == "none"
            assert not capabilities["cors_enabled"]
            assert "tracepoints" in capabilities["analysis_views"]
            assert "span-profile" in capabilities["analysis_views"]
            assert "critical-path" in capabilities["analysis_views"]
            assert "distribution" in capabilities["analysis_views"]
            assert capabilities["limits"]["max_span_profile_nodes"] == 10_000
            assert capabilities["limits"]["max_critical_path_segments"] == 10_000
            assert capabilities["limits"]["max_distribution_bins"] == 100

            sources = (await _request(app, "GET", "/api/v1/sources")).json()
            assert sources["items"][0]["name"] == "robot.log"
            assert str(tmp_path) not in str(sources)
            source_id = sources["items"][0]["id"]
            assert (await _request(app, "GET", f"/api/v1/sources/{source_id}")).status_code == 200
            assert (await _request(app, "GET", "/api/v1/sources/../../etc/passwd")).status_code == 404
            assert (await _request(app, "POST", "/api/v1/sources/refresh")).status_code == 200

            created = await _request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source_id})
            assert created.status_code == 202
            job = await _wait_for_job(app, created.json()["id"])
            assert job["status"] == "completed", job
            analysis_id = job["analysis_id"]

            duplicate = await _request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source_id})
            assert duplicate.status_code == 200
            assert duplicate.json()["deduplicated"]
            analyses = (await _request(app, "GET", "/api/v1/analyses")).json()
            assert analyses["items"][0]["id"] == analysis_id
            assert (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}")).status_code == 200
            filtered = await _request(
                app,
                "GET",
                f"/api/v1/analyses/{analysis_id}/requests?request_id=missing&limit=1",
            )
            assert filtered.json()["total"] == 0

            for view in (
                "summary",
                "distribution",
                "requests",
                "events",
                "spans",
                "flows",
                "components",
                "tracepoints",
                "timeline",
                "call-tree",
                "graph",
                "warnings",
            ):
                response = await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/{view}")
                assert response.status_code == 200, (view, response.text)
            summary = (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/summary")).json()
            assert summary["span_summary_total"] == 1
            assert summary["span_summary_limit"] == 100
            assert summary["span_summary_truncated"] is False
            assert summary["span_summary"][0] == {
                "component_id": "worker",
                "name": "work",
                "origin": "user",
                "status": "ok",
                "count": 1,
                "minimum": 0.0001,
                "p50": 0.0001,
                "p95": 0.0001,
                "p99": 0.0001,
                "maximum": 0.0001,
                "mean": 0.0001,
            }
            entry = app.state.analysis_manager.get_analysis(analysis_id)
            entry.result.span_summary = [dict(summary["span_summary"][0], name=f"work-{index}") for index in range(150)]
            bounded = (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/summary?limit=10000")).json()
            assert bounded["span_summary_total"] == 150
            assert bounded["span_summary_limit"] == 100
            assert bounded["span_summary_truncated"] is True
            assert len(bounded["span_summary"]) == 100
            assert bounded["span_summary"] == entry.result.span_summary[:100]
            events = (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/events")).json()
            assert "path" not in events["items"][0]["origin"]
            assert isinstance(events["items"][0]["timestamp_ns"], str)

            tracepoints = (
                await _request(
                    app,
                    "GET",
                    f"/api/v1/analyses/{analysis_id}/tracepoints"
                    "?kind=event&component_id=worker&name=ready&origin=user&offset=0&limit=1",
                )
            ).json()
            assert tracepoints["total"] == 1
            assert tracepoints["items"][0]["id"].startswith("tracepoint:")
            assert tracepoints["items"][0]["description"] == "Ready for work."
            undocumented = (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/tracepoints?name=plain")).json()
            assert undocumented["items"][0]["description"] == ""
            page = (
                await _request(
                    app,
                    "GET",
                    f"/api/v1/analyses/{analysis_id}/tracepoints?component_id=worker&offset=0&limit=1",
                )
            ).json()
            assert page["total"] == 3
            assert page["next_offset"] == 1
            assert str(tmp_path) not in str(page)

            for graph_view in ("nodes", "components", "tracepoints"):
                graph_response = await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/graph?view={graph_view}")
                assert graph_response.status_code == 200, (graph_view, graph_response.text)
            graph = (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/graph?view=tracepoints")).json()
            assert all("description" in node for node in graph["nodes"])
            assert all("directed" in edge for edge in graph["edges"])
            assert any(not edge["directed"] for edge in graph["edges"] if edge["kind"] == "contains")
            assert str(tmp_path) not in str(graph)
            assert (
                await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/tracepoints?kind=counter")
            ).status_code == 422
            assert (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}/graph?view=systems")).status_code == 422
            distribution_endpoint = f"/api/v1/analyses/{analysis_id}/distribution"
            for query in (
                "bins=4",
                "bins=101",
                "outlier_limit=0",
                "outlier_limit=101",
                "bucket_request_limit=0",
                "bucket_request_limit=1001",
                "metric=unknown_ms",
            ):
                response = await _request(app, "GET", f"{distribution_endpoint}?{query}")
                assert response.status_code == 422, (query, response.text)

            assert (await _request(app, "DELETE", f"/api/v1/analyses/{analysis_id}")).status_code == 204
            assert (await _request(app, "GET", f"/api/v1/analyses/{analysis_id}")).status_code == 404
            assert (await _request(app, "GET", "/healthz", host="evil.example")).status_code == 400

    asyncio.run(run())


def test_span_profile_api_modes_filters_limits_and_path_safety(tmp_path):
    async def run():
        trace = tmp_path / "span-profile.log"
        base_ns = 9_007_199_254_741_000

        def record(timestamp_ns, event, **fields):
            return {
                "schema_version": 1,
                "timestamp_ns": timestamp_ns,
                "event": event,
                "fields": fields,
            }

        records = [
            record(
                base_ns,
                "span_begin",
                trace_id="req-1",
                span_id="root",
                span_name="root",
                component_id="outer",
                origin="user",
            ),
            record(
                base_ns + 10,
                "span_begin",
                trace_id="req-1",
                span_id="child",
                parent_span_id="root",
                span_name="child",
                component_id="inner",
                origin="user",
            ),
            record(
                base_ns + 40,
                "span_end",
                trace_id="req-1",
                span_id="child",
                parent_span_id="root",
                span_name="child",
                component_id="inner",
                origin="user",
                status="error",
                debug_path=str(trace),
            ),
            record(
                base_ns + 100,
                "span_end",
                trace_id="req-1",
                span_id="root",
                span_name="root",
                component_id="outer",
                origin="user",
                debug_path=str(trace),
            ),
            record(
                base_ns + 200,
                "span_begin",
                trace_id="req-2",
                span_id="other",
                span_name="other",
                component_id="other",
                origin="built-in",
            ),
            record(
                base_ns + 250,
                "span_end",
                trace_id="req-2",
                span_id="other",
                span_name="other",
                component_id="other",
                origin="built-in",
            ),
        ]
        trace.write_text("\n".join(f"IBTRACE1 {json.dumps(item)}" for item in records), encoding="utf-8")
        app = create_app(WebConfig(trace_roots=(tmp_path,), allowed_hosts=("testserver",)))

        async with app.router.lifespan_context(app):
            source_id = (await _request(app, "GET", "/api/v1/sources")).json()["items"][0]["id"]
            created = await _request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source_id})
            job = await _wait_for_job(app, created.json()["id"])
            assert job["status"] == "completed", job
            endpoint = f"/api/v1/analyses/{job['analysis_id']}/span-profile"

            missing_request = await _request(app, "GET", endpoint)
            assert missing_request.status_code == 422
            assert "request_id" in missing_request.text

            request_response = await _request(app, "GET", f"{endpoint}?mode=request&request_id=req-1")
            assert request_response.status_code == 200, request_response.text
            request_profile = request_response.json()
            assert request_profile["mode"] == "request"
            assert request_profile["start_ns"] == str(base_ns)
            assert request_profile["end_ns"] == str(base_ns + 100)
            assert request_profile["total_nodes"] == 2
            assert str(tmp_path) not in request_response.text

            filtered_response = await _request(
                app,
                "GET",
                f"{endpoint}?mode=request&request_id=req-1&component_id=inner"
                f"&start_ns={base_ns + 20}&end_ns={base_ns + 35}&origin=user&status=error",
            )
            assert filtered_response.status_code == 200, filtered_response.text
            filtered = filtered_response.json()
            assert [node["span_id"] for node in filtered["nodes"]] == ["child"]
            assert filtered["nodes"][0]["start_ns"] == str(base_ns + 10)
            assert str(tmp_path) not in filtered_response.text

            aggregate_response = await _request(app, "GET", f"{endpoint}?mode=aggregate&origin=user&max_nodes=1")
            assert aggregate_response.status_code == 200, aggregate_response.text
            aggregate = aggregate_response.json()
            assert aggregate["mode"] == "aggregate"
            assert aggregate["total_nodes"] == 2
            assert aggregate["returned_nodes"] == 1
            assert aggregate["truncated"]
            assert aggregate["truncation_reason"] == "max_nodes"
            assert str(tmp_path) not in aggregate_response.text

            for query in (
                "mode=invalid",
                "mode=aggregate&max_nodes=0",
                "mode=aggregate&max_nodes=10001",
                "mode=aggregate&start_ns=20&end_ns=10",
            ):
                response = await _request(app, "GET", f"{endpoint}?{query}")
                assert response.status_code == 422, (query, response.text)

    asyncio.run(run())


def test_critical_path_api_partition_validation_empty_and_path_safety(tmp_path):
    async def run():
        trace = tmp_path / "critical-path.log"
        base_ns = 9_007_199_254_741_000

        def record(timestamp_ns, event, **fields):
            return {
                "schema_version": 1,
                "timestamp_ns": timestamp_ns,
                "event": event,
                "fields": fields,
            }

        records = [
            record(base_ns, "dispatch_request", trace_id="request", component_id="dispatcher"),
            record(
                base_ns,
                "span_begin",
                trace_id="request",
                span_id="root",
                span_name="root",
                component_id="outer",
                origin="user",
            ),
            record(
                base_ns + 20,
                "span_begin",
                trace_id="request",
                span_id="child",
                parent_span_id="root",
                span_name="child",
                component_id="inner",
                origin="user",
            ),
            record(
                base_ns + 30,
                "flow_send",
                trace_id="request",
                edge_id="edge",
                flow_id="transfer",
                component_id="outer",
            ),
            record(
                base_ns + 40,
                "flow_receive",
                trace_id="request",
                edge_id="edge",
                flow_id="transfer",
                component_id="inner",
            ),
            record(
                base_ns + 80,
                "span_end",
                trace_id="request",
                span_id="child",
                parent_span_id="root",
                span_name="child",
                component_id="inner",
                origin="user",
                debug_path=str(trace),
            ),
            record(
                base_ns + 100,
                "span_end",
                trace_id="request",
                span_id="root",
                span_name="root",
                component_id="outer",
                origin="user",
                debug_path=str(trace),
            ),
            record(base_ns + 100, "first_action_execute", trace_id="request", component_id="dispatcher"),
        ]
        trace.write_text("\n".join(f"IBTRACE1 {json.dumps(item)}" for item in records), encoding="utf-8")
        app = create_app(WebConfig(trace_roots=(tmp_path,), allowed_hosts=("testserver",)))

        async with app.router.lifespan_context(app):
            capabilities = (await _request(app, "GET", "/api/v1/capabilities")).json()
            assert "critical-path" in capabilities["analysis_views"]
            assert capabilities["limits"]["max_critical_path_segments"] == 10_000

            source_id = (await _request(app, "GET", "/api/v1/sources")).json()["items"][0]["id"]
            created = await _request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source_id})
            job = await _wait_for_job(app, created.json()["id"])
            assert job["status"] == "completed", job
            endpoint = f"/api/v1/analyses/{job['analysis_id']}/critical-path"

            missing_request = await _request(app, "GET", endpoint)
            assert missing_request.status_code == 422
            assert "request_id" in missing_request.text

            response = await _request(app, "GET", f"{endpoint}?request_id=request")
            assert response.status_code == 200, response.text
            document = response.json()
            assert document["start_ns"] == str(base_ns)
            assert document["end_ns"] == str(base_ns + 100)
            assert document["duration_ns"] == 100
            assert [
                (item["kind"], item["label"], item["start_ns"], item["end_ns"]) for item in document["segments"]
            ] == [
                ("span", "root", str(base_ns), str(base_ns + 20)),
                ("span", "child", str(base_ns + 20), str(base_ns + 30)),
                ("flow", "edge", str(base_ns + 30), str(base_ns + 40)),
                ("span", "child", str(base_ns + 40), str(base_ns + 80)),
                ("span", "root", str(base_ns + 80), str(base_ns + 100)),
            ]
            assert document["totals"]["partition_ns"] == 100
            assert str(tmp_path) not in response.text

            bounded = await _request(app, "GET", f"{endpoint}?request_id=request&max_segments=1")
            assert bounded.status_code == 200, bounded.text
            assert bounded.json()["total_segments"] == 5
            assert bounded.json()["returned_segments"] == 1
            assert bounded.json()["truncated"]

            empty = await _request(app, "GET", f"{endpoint}?request_id=missing")
            assert empty.status_code == 200, empty.text
            assert empty.json()["start_ns"] is None
            assert empty.json()["segments"] == []
            assert empty.json()["total_segments"] == 0

            for query in (
                "request_id=request&start_ns=20&end_ns=10",
                "request_id=request&max_segments=0",
                "request_id=request&max_segments=10001",
            ):
                invalid = await _request(app, "GET", f"{endpoint}?{query}")
                assert invalid.status_code == 422, (query, invalid.text)

    asyncio.run(run())


def test_latency_distribution_api_returns_histogram_and_outliers(tmp_path):
    async def run():
        trace = tmp_path / "distribution.log"

        def record(timestamp_ns, event, request_id, span_id):
            return {
                "schema_version": 1,
                "timestamp_ns": timestamp_ns,
                "event": event,
                "fields": {
                    "trace_id": request_id,
                    "span_id": span_id,
                    "span_name": "model_call",
                    "component_id": "policy.inference",
                    "origin": "built-in",
                },
            }

        records = [
            record(1, "span_begin", "fast", "fast-call"),
            record(1_000_001, "span_end", "fast", "fast-call"),
            record(2_000_001, "span_begin", "slow", "slow-call"),
            record(12_000_001, "span_end", "slow", "slow-call"),
        ]
        trace.write_text("\n".join(f"IBTRACE1 {json.dumps(item)}" for item in records), encoding="utf-8")
        app = create_app(WebConfig(trace_roots=(tmp_path,), allowed_hosts=("testserver",)))

        async with app.router.lifespan_context(app):
            source_id = (await _request(app, "GET", "/api/v1/sources")).json()["items"][0]["id"]
            created = await _request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source_id})
            job = await _wait_for_job(app, created.json()["id"])
            endpoint = f"/api/v1/analyses/{job['analysis_id']}/distribution"

            response = await _request(
                app,
                "GET",
                f"{endpoint}?bins=5&outlier_limit=1&bucket_request_limit=1",
            )
            assert response.status_code == 200, response.text
            document = response.json()
            assert document["metric"] == "inference_ms"
            assert document["unit"] == "ms"
            assert document["sample_count"] == 2
            assert document["invalid_count"] == 0
            assert document["bin_count"] == 2
            assert sum(bucket["count"] for bucket in document["buckets"]) == 2
            assert document["outliers"] == [
                {
                    "request_id": "slow",
                    "value": 10.0,
                    "rank": 1,
                    "percentile": 100.0,
                    "p95_tail": True,
                }
            ]
            assert str(tmp_path) not in response.text

    asyncio.run(run())


def test_vue_assets_and_spa_routes_are_served(tmp_path):
    async def run():
        web_root = tmp_path / "web"
        web_root.mkdir()
        (web_root / "index.html").write_text("<main>IB-Robot Trace Explorer</main>", encoding="utf-8")
        (web_root / "app.js").write_text("console.log('trace')", encoding="utf-8")
        trace_root = tmp_path / "traces"
        trace_root.mkdir()
        app = create_app(WebConfig(trace_roots=(trace_root,), allowed_hosts=("testserver",), web_root=web_root))

        root = await _request(app, "GET", "/")
        route = await _request(app, "GET", "/analysis/abc")
        asset = await _request(app, "GET", "/app.js")
        missing_asset = await _request(app, "GET", "/assets/missing.js")
        missing_api = await _request(app, "GET", "/api/v1/missing")
        missing_health = await _request(app, "GET", "/healthz/missing")

        assert root.status_code == 200
        assert route.status_code == 200
        assert "Trace Explorer" in root.text
        assert route.content == root.content
        assert b"console.log" in asset.content
        assert missing_asset.status_code == 404
        assert missing_api.status_code == 404
        assert missing_health.status_code == 404

    asyncio.run(run())


def test_symlink_install_web_assets_are_served(tmp_path):
    async def run():
        dist = tmp_path / "source" / "dist"
        dist.mkdir(parents=True)
        (dist / "index.html").write_text("<main>Trace Explorer</main>", encoding="utf-8")
        (dist / "app.js").write_text("console.log('trace')", encoding="utf-8")
        web_root = tmp_path / "install" / "web"
        web_root.mkdir(parents=True)
        (web_root / "index.html").symlink_to(dist / "index.html")
        (web_root / "app.js").symlink_to(dist / "app.js")
        trace_root = tmp_path / "traces"
        trace_root.mkdir()
        app = create_app(WebConfig(trace_roots=(trace_root,), allowed_hosts=("testserver",), web_root=web_root))

        assert (await _request(app, "GET", "/")).status_code == 200
        asset = await _request(app, "GET", "/app.js")
        assert asset.status_code == 200
        assert b"console.log" in asset.content

    asyncio.run(run())

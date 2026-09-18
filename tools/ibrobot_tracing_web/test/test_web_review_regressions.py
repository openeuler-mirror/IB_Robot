import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from test_api import _request

from ibrobot_tracing.analysis import AnalysisResult
from ibrobot_tracing.model import TraceDataset
from ibrobot_tracing_web.app import _MutationOriginMiddleware, create_app
from ibrobot_tracing_web.catalog import SourceCatalog
from ibrobot_tracing_web.config import WebConfig
from ibrobot_tracing_web.jobs import AnalysisEntry, AnalysisManager, AnalysisStore, JobNotFoundError


def test_cache_evicts_by_bytes_even_below_item_limit():
    now = datetime.now(timezone.utc)

    def entry(name):
        return AnalysisEntry(name, name, name, name, name, "log", "v", now, b"x" * 400)

    store = AnalysisStore(10, max_bytes=1000)
    a, b, c = entry("a"), entry("b"), entry("c")
    assert store.add(a) == []
    assert store.add(b) == []
    assert store.add(c) == [a]
    assert len(store) == 2
    with pytest.raises(ValueError, match="budget"):
        store.add(replace(a, result=b"z" * 2000))
    assert len(store) == 2


def test_refresh_does_not_queue_a_second_scan(tmp_path, monkeypatch):
    catalog = SourceCatalog((tmp_path,))
    entered, release = threading.Event(), threading.Event()
    scans = []

    def scan():
        scans.append(1)
        entered.set()
        assert release.wait(2)
        return catalog.snapshot()

    monkeypatch.setattr(catalog, "_refresh", scan)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(catalog.refresh_coalesced)
        try:
            assert entered.wait(2)
            assert catalog.refresh_coalesced().generation == 0
        finally:
            release.set()
        future.result()
    catalog.refresh_coalesced()
    assert scans == [1]


@pytest.mark.parametrize(
    "origin,status", [("http://evil.example", 403), ("null", 403), ("http://local", 200), (None, 200)]
)
def test_simple_cross_site_post_is_rejected_before_work(origin, status):
    called = []
    messages = []

    async def inner(scope, receive, send):
        called.append(1)
        await send({"type": "http.response.start", "status": 200})

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/sources/refresh",
        "query_string": b"",
        "headers": [(b"host", b"local")],
        "server": ("local", 80),
    }
    if origin is not None:
        scope["headers"].append((b"origin", origin.encode()))
    asyncio.run(_MutationOriginMiddleware(inner, origins=())(scope, None, send))
    assert messages[0]["status"] == status
    assert bool(called) == (status == 200)


def test_create_maps_evicted_job_to_404(tmp_path, monkeypatch):
    (tmp_path / "a.log").write_text('IBTRACE1 {"timestamp_ns":1,"event":"ready"}')
    catalog = SourceCatalog((tmp_path,))
    source = catalog.refresh().sources[0]
    manager = AnalysisManager(catalog, queue_size=1, max_analyses=1, max_job_history=2)

    def gone(_id):
        raise JobNotFoundError(_id)

    monkeypatch.setattr(manager, "get_job", gone)
    app = create_app(
        WebConfig(trace_roots=(tmp_path,), allowed_hosts=("testserver",), web_root=None),
        catalog=catalog,
        manager=manager,
    )
    response = asyncio.run(_request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source.source_id}))
    assert response.status_code == 404


def test_unexpected_error_text_is_private():
    message = AnalysisManager._safe_error(RuntimeError("secret /home/private/file"), None)
    assert "secret" not in message and "/home" not in message


def test_warning_endpoint_is_explicitly_truncated(tmp_path):
    class Analyzer:
        def analyze(self, *_args):
            return AnalysisResult(TraceDataset(), warnings=[f"warning-{i}" for i in range(110)])

    (tmp_path / "a.log").write_text("placeholder")
    catalog = SourceCatalog((tmp_path,))
    source = catalog.refresh().sources[0]
    manager = AnalysisManager(catalog, queue_size=1, max_analyses=1, max_job_history=2, analyzer=Analyzer())
    app = create_app(
        WebConfig(trace_roots=(tmp_path,), allowed_hosts=("testserver",), web_root=None),
        catalog=catalog,
        manager=manager,
    )

    async def run():
        async with app.router.lifespan_context(app):
            response = await _request(app, "POST", "/api/v1/analysis-jobs", body={"source_id": source.source_id})
            for _ in range(200):
                job = (await _request(app, "GET", "/api/v1/analysis-jobs/" + response.json()["id"])).json()
                if job["status"] == "completed":
                    break
                await asyncio.sleep(0.01)
            assert job["status"] == "completed"
            result = (await _request(app, "GET", "/api/v1/analyses/" + job["analysis_id"] + "/warnings")).json()
            assert result["total"] == 110 and result["limit"] == len(result["items"]) == 100
            assert result["truncated"] is True

    asyncio.run(run())

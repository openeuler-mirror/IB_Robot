import json
import threading
import time

import pytest

from ibrobot_tracing_web.catalog import SourceCatalog
from ibrobot_tracing_web.jobs import (
    AnalysisManager,
    AnalysisNotFoundError,
    ComparisonJobNotFoundError,
    ComparisonManager,
    ComparisonNotFoundError,
    ComparisonQueueFullError,
    JobNotFoundError,
    QueueFullError,
    SameSourceComparisonError,
)


def _write_trace(path, request_id):
    path.write_text(
        "\n".join(
            [
                "IBTRACE1 "
                + json.dumps(
                    {
                        "timestamp_ns": 1_000_000_000,
                        "event": "dispatch_request",
                        "fields": {"trace_id": request_id, "component_id": "action_dispatcher.request"},
                    }
                ),
                "IBTRACE1 "
                + json.dumps(
                    {
                        "timestamp_ns": 1_010_000_000,
                        "event": "first_action_execute",
                        "fields": {
                            "trace_id": request_id,
                            "component_id": "action_dispatcher.execute",
                            "publish_end_ns": 1_010_000_000,
                            "publish_ms": 1,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _wait(manager, job_id):
    for _ in range(200):
        job, _ = manager.get_job(job_id)
        if job.status not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def _comparison_inputs(tmp_path):
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    _write_trace(trace_root / "baseline.log", "baseline")
    _write_trace(trace_root / "candidate.log", "candidate")
    catalog = SourceCatalog((trace_root,))
    sources = {source.name: source for source in catalog.refresh().sources}
    analysis_manager = AnalysisManager(catalog, queue_size=2, max_analyses=2, max_job_history=4)
    analysis_manager.start()
    analysis_job, _ = analysis_manager.submit(sources["candidate.log"].source_id)
    analysis_job = _wait(analysis_manager, analysis_job.job_id)
    assert analysis_job.status == "completed"
    return catalog, sources["baseline.log"], sources["candidate.log"], analysis_manager, analysis_job.analysis_id


def _submit(manager, baseline, analysis_id, *, bins=5):
    return manager.submit(
        baseline.source_id,
        baseline.fingerprint,
        analysis_id,
        statistic="p95",
        relative_threshold_percent=5,
        absolute_threshold_ms=1,
        metric="total_ms",
        bins=bins,
    )


def test_jobs_dedupe_and_analysis_store_is_bounded(tmp_path):
    _write_trace(tmp_path / "one.log", "one")
    _write_trace(tmp_path / "two.log", "two")
    catalog = SourceCatalog((tmp_path,))
    sources = catalog.refresh().sources
    manager = AnalysisManager(catalog, queue_size=2, max_analyses=1, max_job_history=4)
    manager.start()
    try:
        first, deduplicated = manager.submit(sources[0].source_id)
        assert not deduplicated
        first = _wait(manager, first.job_id)
        duplicate, deduplicated = manager.submit(sources[0].source_id)
        assert deduplicated
        assert duplicate.job_id == first.job_id

        second, _ = manager.submit(sources[1].source_id)
        second = _wait(manager, second.job_id)
        assert manager.list_analyses()[0].analysis_id == second.analysis_id
        assert manager.get_job(first.job_id)[0].status == "expired"
    finally:
        manager.close()


class _BlockingAnalyzer:
    def __init__(self, release, result=None):
        self.release = release
        self.result = result or {"requests": [], "warnings": []}

    def analyze(self, _source, _source_kind):
        self.release.wait(timeout=2)
        return self.result


def test_queued_analysis_job_can_be_cancelled(tmp_path):
    _write_trace(tmp_path / "one.log", "one")
    _write_trace(tmp_path / "two.log", "two")
    catalog = SourceCatalog((tmp_path,))
    sources = catalog.refresh().sources
    release = threading.Event()
    manager = AnalysisManager(
        catalog,
        queue_size=1,
        max_analyses=1,
        max_job_history=4,
        analyzer=_BlockingAnalyzer(release),
    )
    manager.start()
    try:
        running, _ = manager.submit(sources[0].source_id)
        for _ in range(100):
            if manager.get_job(running.job_id)[0].status == "running":
                break
            time.sleep(0.01)
        queued, _ = manager.submit(sources[1].source_id)
        assert manager.cancel(queued.job_id)[0].status == "cancelled"
    finally:
        release.set()
        manager.close()


@pytest.mark.parametrize("start_worker", [False, True])
def test_analysis_job_history_rejects_when_active_jobs_fill_limit(tmp_path, start_worker):
    for name in ("one", "two", "three"):
        _write_trace(tmp_path / f"{name}.log", name)
    catalog = SourceCatalog((tmp_path,))
    sources = catalog.refresh().sources
    release = threading.Event()
    manager = AnalysisManager(
        catalog, queue_size=3, max_analyses=1, max_job_history=2, analyzer=_BlockingAnalyzer(release)
    )
    if start_worker:
        manager.start()
    try:
        first, _ = manager.submit(sources[0].source_id)
        if start_worker:
            for _ in range(100):
                if manager.get_job(first.job_id)[0].status == "running":
                    break
                time.sleep(0.01)
            assert manager.get_job(first.job_id)[0].status == "running"
        second, _ = manager.submit(sources[1].source_id)
        for source, job in zip(sources, (first, second), strict=False):
            duplicate, deduplicated = manager.submit(source.source_id)
            assert deduplicated
            assert duplicate.job_id == job.job_id

        for _ in range(2):
            with pytest.raises(QueueFullError, match="history is full"):
                manager.submit(sources[2].source_id)
            assert list(manager._jobs) == [first.job_id, second.job_id]
            assert manager._dedupe == {first.dedupe_key: first.job_id, second.dedupe_key: second.job_id}
            assert manager.queued_count == (1 if start_worker else 2)

        manager.cancel(second.job_id)
        third, deduplicated = manager.submit(sources[2].source_id)
        assert not deduplicated
        assert list(manager._jobs) == [first.job_id, third.job_id]
        assert manager._dedupe == {first.dedupe_key: first.job_id, third.dedupe_key: third.job_id}
        with pytest.raises(JobNotFoundError):
            manager.get_job(second.job_id)
    finally:
        release.set()
        manager.close()


@pytest.mark.parametrize("previous_status", ["completed", "failed", "expired"])
def test_analysis_job_history_reclaims_terminal_jobs_before_execution(tmp_path, previous_status):
    for name in ("one", "two"):
        _write_trace(tmp_path / f"{name}.log", name)
    catalog = SourceCatalog((tmp_path,))
    sources = catalog.refresh().sources
    release = threading.Event()
    release.set()
    manager = AnalysisManager(
        catalog, queue_size=2, max_analyses=1, max_job_history=1, analyzer=_BlockingAnalyzer(release)
    )
    first, _ = manager.submit(sources[0].source_id)
    if previous_status == "failed":
        sources[0].path.write_text("changed", encoding="utf-8")
    manager.start()
    try:
        first = _wait(manager, first.job_id)
        assert first.status == ("failed" if previous_status == "failed" else "completed")
        if previous_status == "completed":
            duplicate, deduplicated = manager.submit(sources[0].source_id)
            assert deduplicated
            assert duplicate.job_id == first.job_id
        elif previous_status == "expired":
            manager.delete_analysis(first.analysis_id)
        release.clear()

        second, deduplicated = manager.submit(sources[1].source_id)

        assert not deduplicated
        assert list(manager._jobs) == [second.job_id]
        assert manager._dedupe == {second.dedupe_key: second.job_id}
        assert manager.analysis_count == 0
        with pytest.raises(JobNotFoundError):
            manager.get_job(first.job_id)
        if first.analysis_id is not None:
            with pytest.raises(AnalysisNotFoundError):
                manager.get_analysis(first.analysis_id)
        release.set()
        assert _wait(manager, second.job_id).status == "completed"
    finally:
        release.set()
        manager.close()


def test_comparison_jobs_use_catalog_sources_dedupe_expire_and_delete(tmp_path):
    catalog, baseline, _, analysis_manager, analysis_id = _comparison_inputs(tmp_path)
    manager = ComparisonManager(
        catalog,
        analysis_manager,
        queue_size=2,
        max_comparisons=1,
        max_job_history=4,
        max_events=100,
    )
    manager.start()
    try:
        first, deduplicated = _submit(manager, baseline, analysis_id)
        assert not deduplicated
        first = _wait(manager, first.job_id)
        assert first.status == "completed"
        duplicate, deduplicated = _submit(manager, baseline, analysis_id)
        assert deduplicated
        assert duplicate.job_id == first.job_id

        second, _ = _submit(manager, baseline, analysis_id, bins=6)
        second = _wait(manager, second.job_id)
        assert second.status == "completed"
        assert manager.get_job(first.job_id)[0].status == "expired"
        with pytest.raises(ComparisonNotFoundError):
            manager.get_comparison(first.comparison_id)

        manager.delete_comparison(second.comparison_id)
        assert baseline.path.is_file()
        with pytest.raises(ComparisonNotFoundError):
            manager.get_comparison(second.comparison_id)
    finally:
        manager.close()
        analysis_manager.close()


def test_comparison_rejects_candidate_source_as_its_own_baseline(tmp_path):
    catalog, _, candidate, analysis_manager, analysis_id = _comparison_inputs(tmp_path)
    manager = ComparisonManager(
        catalog,
        analysis_manager,
        queue_size=1,
        max_comparisons=1,
        max_job_history=2,
        max_events=100,
    )
    try:
        with pytest.raises(SameSourceComparisonError):
            _submit(manager, candidate, analysis_id)
    finally:
        analysis_manager.close()


def test_comparison_pending_queue_is_bounded(tmp_path):
    catalog, baseline, _, analysis_manager, analysis_id = _comparison_inputs(tmp_path)
    candidate = analysis_manager.get_analysis(analysis_id)
    release = threading.Event()
    manager = ComparisonManager(
        catalog,
        analysis_manager,
        queue_size=1,
        max_comparisons=2,
        max_job_history=4,
        max_events=100,
        analyzer=_BlockingAnalyzer(release, candidate.result),
    )
    manager.start()
    try:
        running, _ = _submit(manager, baseline, analysis_id)
        for _ in range(100):
            if manager.get_job(running.job_id)[0].status == "running":
                break
            time.sleep(0.01)
        queued, _ = _submit(manager, baseline, analysis_id, bins=6)
        assert manager.get_job(queued.job_id)[1] == 1
        with pytest.raises(ComparisonQueueFullError, match="full"):
            _submit(manager, baseline, analysis_id, bins=7)
    finally:
        release.set()
        manager.close()
        analysis_manager.close()


def test_comparison_detects_source_changes_without_leaking_paths(tmp_path):
    catalog, baseline, _, analysis_manager, analysis_id = _comparison_inputs(tmp_path)
    manager = ComparisonManager(
        catalog,
        analysis_manager,
        queue_size=1,
        max_comparisons=1,
        max_job_history=2,
        max_events=100,
    )
    job, _ = _submit(manager, baseline, analysis_id)
    baseline.path.write_text("changed", encoding="utf-8")
    manager.start()
    try:
        failed = _wait(manager, job.job_id)
        assert failed.status == "failed"
        assert "changed" in failed.error
        assert str(tmp_path) not in failed.error
    finally:
        manager.close()
        analysis_manager.close()


class _FailingComparisonAnalyzer:
    def analyze(self, source, _source_kind):
        raise RuntimeError(f"failed at {source}")


def test_comparison_failure_history_is_bounded_and_sanitized(tmp_path):
    catalog, baseline, _, analysis_manager, analysis_id = _comparison_inputs(tmp_path)
    manager = ComparisonManager(
        catalog,
        analysis_manager,
        queue_size=1,
        max_comparisons=1,
        max_job_history=1,
        max_events=100,
        analyzer=_FailingComparisonAnalyzer(),
    )
    manager.start()
    try:
        first, _ = _submit(manager, baseline, analysis_id)
        first = _wait(manager, first.job_id)
        assert first.status == "failed"
        assert str(tmp_path) not in first.error
        second, _ = _submit(manager, baseline, analysis_id, bins=6)
        assert _wait(manager, second.job_id).status == "failed"
        with pytest.raises(ComparisonJobNotFoundError):
            manager.get_job(first.job_id)
    finally:
        manager.close()
        analysis_manager.close()


@pytest.mark.parametrize("manager_kind", ["analysis", "comparison"])
@pytest.mark.parametrize("source_kind", ["log", "ctf"])
@pytest.mark.parametrize("swap_parent", [False, True])
def test_jobs_read_only_snapshot_after_parent_symlink_swap(tmp_path, manager_kind, source_kind, swap_parent):
    catalog, baseline, _, analysis_manager, analysis_id = _comparison_inputs(tmp_path)
    if source_kind == "ctf":
        trace = baseline.root / "session.ctf"
        (trace / "ust").mkdir(parents=True)
        (trace / "ust" / "metadata").write_text("safe metadata", encoding="utf-8")
        baseline = next(source for source in catalog.refresh().sources if source.name == "session.ctf")
    expected = (baseline.path / "ust" / "metadata").read_text() if source_kind == "ctf" else baseline.path.read_text()
    outside = tmp_path / "outside"
    outside.mkdir()
    if source_kind == "ctf":
        (outside / baseline.name / "ust").mkdir(parents=True)
        (outside / baseline.name / "ust" / "metadata").write_text("secret", encoding="utf-8")
    else:
        (outside / baseline.name).write_text("secret", encoding="utf-8")
    seen = []
    paths = []
    candidate = analysis_manager.get_analysis(analysis_id).result
    candidate_source = candidate.dataset.metadata["source"]

    class ReadingAnalyzer:
        def analyze(self, path, kind):
            paths.append(path)
            if swap_parent:
                baseline.root.rename(tmp_path / "original")
                baseline.root.symlink_to(outside)
            seen.append((path / "ust" / "metadata").read_text() if kind == "ctf" else path.read_text())
            return candidate

    if manager_kind == "analysis":
        manager = AnalysisManager(catalog, queue_size=1, max_analyses=1, max_job_history=2, analyzer=ReadingAnalyzer())
        job, _ = manager.submit(baseline.source_id)
    else:
        manager = ComparisonManager(
            catalog,
            analysis_manager,
            queue_size=1,
            max_comparisons=1,
            max_job_history=2,
            max_events=100,
            analyzer=ReadingAnalyzer(),
        )
        job, _ = _submit(manager, baseline, analysis_id)
    manager.start()
    try:
        assert _wait(manager, job.job_id).status == "completed"
        assert seen == [expected]
        assert paths[0] != baseline.path
        assert not paths[0].exists()
        assert catalog.get(baseline.source_id) == baseline
        assert candidate.dataset.metadata["source"] == candidate_source
    finally:
        manager.close()
        analysis_manager.close()


@pytest.mark.parametrize("outcome", ["completed", "failed", "timeout", "cancelled"])
def test_analysis_snapshot_lifetime_and_source_identity(tmp_path, outcome):
    _write_trace(tmp_path / "robot.log", "one")
    catalog = SourceCatalog((tmp_path,))
    source = catalog.refresh().sources[0]
    paths = []

    class SnapshotAnalyzer:
        def analyze(self, path, kind):
            paths.append(path)
            assert path != source.path
            assert path.read_bytes() == source.path.read_bytes()
            if outcome == "failed":
                raise RuntimeError(f"failed at {path}")
            if outcome == "timeout":
                raise TimeoutError(f"timed out reading {path}")
            if outcome == "cancelled":
                manager.cancel(job.job_id)
            from ibrobot_tracing_web.jobs import TracingAnalyzer

            return TracingAnalyzer().analyze(path, kind)

    manager = AnalysisManager(catalog, queue_size=1, max_analyses=1, max_job_history=2, analyzer=SnapshotAnalyzer())
    job, _ = manager.submit(source.source_id)
    manager.start()
    try:
        finished = _wait(manager, job.job_id)
        assert finished.status == ("failed" if outcome == "timeout" else outcome)
        assert paths
        assert not paths[0].exists()
        assert not paths[0].parent.exists()
        if finished.status == "completed":
            entry = manager.get_analysis(finished.analysis_id)
            assert (entry.source_id, entry.source_name, entry.source_version) == (
                source.source_id,
                source.name,
                source.fingerprint,
            )
            assert entry.result.dataset.metadata["source"] == str(source.path)
            assert all(event.origin.path == str(source.path) for event in entry.result.dataset.events)
        else:
            assert manager.analysis_count == 0
            assert str(paths[0].parent) not in (finished.error or "")
    finally:
        manager.close()


def test_analysis_snapshot_copy_failure_is_cleaned_and_redacted(tmp_path, monkeypatch):
    from pathlib import Path

    _write_trace(tmp_path / "robot.log", "one")
    catalog = SourceCatalog((tmp_path,))
    source = catalog.refresh().sources[0]
    paths = []
    original_open = Path.open

    def fail_snapshot_write(path, mode="r", *args, **kwargs):
        if mode == "xb":
            paths.append(path)
            raise OSError(28, "No space left on device", str(path))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_snapshot_write)
    manager = AnalysisManager(catalog, queue_size=1, max_analyses=1, max_job_history=2)
    job, _ = manager.submit(source.source_id)
    manager.start()
    try:
        finished = _wait(manager, job.job_id)
        assert finished.status == "failed"
        assert paths
        assert not paths[0].parent.exists()
        assert str(paths[0].parent) not in finished.error
    finally:
        manager.close()

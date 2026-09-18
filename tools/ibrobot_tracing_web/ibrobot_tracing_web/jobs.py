"""Bounded single-worker analysis scheduling and in-memory result storage."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ibrobot_tracing.analysis import AnalysisRequest, AnalysisResult, AnalysisService
from ibrobot_tracing.comparison import DEFAULT_MAX_FLAME_NODES, TraceComparisonService

from .catalog import SourceCatalog, SourceChangedError, SourceNotFoundError, SourceRecord, UnsafeSourceError
from .queries import ResultQueries


class JobNotFoundError(KeyError):
    pass


class AnalysisNotFoundError(KeyError):
    pass


class QueueFullError(RuntimeError):
    pass


class ComparisonJobNotFoundError(KeyError):
    pass


class ComparisonNotFoundError(KeyError):
    pass


class ComparisonQueueFullError(RuntimeError):
    pass


class ComparisonSourceUnavailableError(RuntimeError):
    pass


class SameSourceComparisonError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _result_bytes(value: Any, limit: int) -> int:
    """Bounded resident-size estimate, not an OS memory-limit guarantee."""
    pending = [value]
    seen = set()
    total = 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if total > limit or len(seen) > 2_000_000:
            return limit + 1
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list | tuple | set | frozenset | deque):
            pending.extend(item)
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, f.name) for f in dataclass_fields(item))
    return total


@dataclass(slots=True)
class JobRecord:
    job_id: str
    source_id: str
    source_version: str
    dedupe_key: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    analysis_id: str | None = None
    error: str | None = None
    cancel_requested: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisEntry:
    analysis_id: str
    job_id: str
    dedupe_key: str
    source_id: str
    source_name: str
    source_kind: str
    source_version: str
    created_at: datetime
    result: Any


@dataclass(frozen=True, slots=True)
class ComparisonParameters:
    statistic: str
    relative_threshold_percent: float
    absolute_threshold_ms: float
    metric: str | None
    bins: int


@dataclass(slots=True)
class ComparisonJobRecord:
    job_id: str
    baseline_source_id: str
    baseline_source_version: str
    candidate_analysis_id: str
    parameters: ComparisonParameters
    dedupe_key: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    comparison_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonEntry:
    comparison_id: str
    job_id: str
    dedupe_key: str
    baseline_source_id: str
    baseline_source_version: str
    candidate_analysis_id: str
    created_at: datetime
    result: Any


class AnalysisStore:
    def __init__(self, max_items: int, max_bytes: int = 256 * 1024 * 1024):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._sizes: dict[str, int] = {}
        self._items: OrderedDict[str, AnalysisEntry] = OrderedDict()

    def add(self, entry: AnalysisEntry, *, size: int | None = None) -> list[AnalysisEntry]:
        if size is None:
            size = _result_bytes(entry.result, self.max_bytes)
        if size > self.max_bytes:
            raise ValueError("Analysis exceeds the result memory budget")
        self._items[entry.analysis_id] = entry
        self._sizes[entry.analysis_id] = size
        self._items.move_to_end(entry.analysis_id)
        evicted = []
        while len(self._items) > self.max_items or sum(self._sizes.values()) > self.max_bytes:
            key, item = self._items.popitem(last=False)
            self._sizes.pop(key)
            evicted.append(item)
        return evicted

    def get(self, analysis_id: str) -> AnalysisEntry:
        try:
            return self._items[analysis_id]
        except KeyError as exc:
            raise AnalysisNotFoundError(analysis_id) from exc

    def delete(self, analysis_id: str) -> AnalysisEntry:
        try:
            entry = self._items.pop(analysis_id)
            self._sizes.pop(analysis_id, None)
            return entry
        except KeyError as exc:
            raise AnalysisNotFoundError(analysis_id) from exc

    def list(self) -> list[AnalysisEntry]:
        return list(reversed(self._items.values()))

    def contains(self, analysis_id: str | None) -> bool:
        return analysis_id is not None and analysis_id in self._items

    def __len__(self) -> int:
        return len(self._items)


class ComparisonStore:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self._items: OrderedDict[str, ComparisonEntry] = OrderedDict()

    def add(self, entry: ComparisonEntry) -> ComparisonEntry | None:
        self._items[entry.comparison_id] = entry
        self._items.move_to_end(entry.comparison_id)
        if len(self._items) > self.max_items:
            _, evicted = self._items.popitem(last=False)
            return evicted
        return None

    def get(self, comparison_id: str) -> ComparisonEntry:
        try:
            return self._items[comparison_id]
        except KeyError as exc:
            raise ComparisonNotFoundError(comparison_id) from exc

    def delete(self, comparison_id: str) -> ComparisonEntry:
        try:
            return self._items.pop(comparison_id)
        except KeyError as exc:
            raise ComparisonNotFoundError(comparison_id) from exc

    def list(self) -> list[ComparisonEntry]:
        return list(reversed(self._items.values()))

    def contains(self, comparison_id: str | None) -> bool:
        return comparison_id is not None and comparison_id in self._items

    def __len__(self) -> int:
        return len(self._items)


class TracingAnalyzer:
    def __init__(self, max_events: int | None = None) -> None:
        self._service = AnalysisService()
        self._max_events = max_events

    def analyze(self, source: Path, source_kind: str) -> Any:
        try:
            request = AnalysisRequest(source=source, source_kind=source_kind, max_events=self._max_events)
        except TypeError:
            request = AnalysisRequest(source=source)
        return self._service.analyze(request)


def _restore_source_identity(result: Any, snapshot: Path, source: SourceRecord) -> None:
    if not isinstance(result, AnalysisResult):
        return
    original = str(source.path)
    temporary = str(snapshot)
    if result.dataset.metadata.get("source") == temporary:
        result.dataset.metadata["source"] = original
    origins = {}
    for index, event in enumerate(result.dataset.events):
        if event.origin.path != temporary:
            continue
        origin = origins.get(event.origin)
        if origin is None:
            origin = replace(event.origin, path=original)
            if len(origins) < 1024:
                origins[event.origin] = origin
        result.dataset.events[index] = replace(event, origin=origin)
    result.dataset.warnings = [warning.replace(temporary, original) for warning in result.dataset.warnings]
    result.warnings = [warning.replace(temporary, original) for warning in result.warnings]


class AnalysisManager:
    def __init__(
        self,
        catalog: SourceCatalog,
        *,
        queue_size: int,
        max_analyses: int,
        max_job_history: int,
        analyzer: Any | None = None,
        max_result_bytes: int = 256 * 1024 * 1024,
    ):
        self.catalog = catalog
        self.queue_size = queue_size
        self.max_job_history = max_job_history
        self._analyzer = analyzer or TracingAnalyzer()
        self._store = AnalysisStore(max_analyses, max_result_bytes)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pending: deque[str] = deque()
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._dedupe: dict[str, str] = {}
        self._thread: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._worker, name="ibrobot-tracing-analysis", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            while self._pending:
                job = self._jobs[self._pending.popleft()]
                job.status = "cancelled"
                job.finished_at = _now()
                self._dedupe.pop(job.dedupe_key, None)
            for job in self._jobs.values():
                if job.status == "running":
                    job.cancel_requested = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=5)

    @property
    def worker_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and not self._stopping

    @property
    def queued_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def analysis_count(self) -> int:
        with self._lock:
            return len(self._store)

    def submit(self, source_id: str) -> tuple[JobRecord, bool]:
        source = self.catalog.get(source_id)
        self.catalog.validate(source)
        dedupe_key = f"{source.source_id}:{source.fingerprint}"
        with self._condition:
            existing_id = self._dedupe.get(dedupe_key)
            existing = self._jobs.get(existing_id) if existing_id else None
            if existing is not None and (
                existing.status in {"queued", "running"}
                or (existing.status == "completed" and self._store.contains(existing.analysis_id))
            ):
                return replace(existing), True
            if self._stopping:
                raise RuntimeError("Analysis worker is stopping")
            if len(self._pending) >= self.queue_size:
                raise QueueFullError("Analysis queue is full")
            self._make_job_room()
            job = JobRecord(
                job_id=uuid.uuid4().hex,
                source_id=source.source_id,
                source_version=source.fingerprint,
                dedupe_key=dedupe_key,
                status="queued",
                created_at=_now(),
            )
            self._jobs[job.job_id] = job
            self._dedupe[dedupe_key] = job.job_id
            self._pending.append(job.job_id)
            self._condition.notify()
            return replace(job), False

    def get_job(self, job_id: str) -> tuple[JobRecord, int | None]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise JobNotFoundError(job_id) from exc
            try:
                queue_position = list(self._pending).index(job_id) + 1
            except ValueError:
                queue_position = None
            return replace(job), queue_position

    def cancel(self, job_id: str) -> tuple[JobRecord, int | None]:
        with self._condition:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise JobNotFoundError(job_id) from exc
            if job.status == "queued":
                self._pending.remove(job_id)
                job.status = "cancelled"
                job.cancel_requested = True
                job.finished_at = _now()
                self._dedupe.pop(job.dedupe_key, None)
            elif job.status == "running":
                job.cancel_requested = True
            self._prune_jobs()
            return replace(job), None

    def list_analyses(self) -> list[AnalysisEntry]:
        with self._lock:
            return self._store.list()

    def get_analysis(self, analysis_id: str) -> AnalysisEntry:
        with self._lock:
            return self._store.get(analysis_id)

    def delete_analysis(self, analysis_id: str) -> None:
        with self._lock:
            entry = self._store.delete(analysis_id)
            if self._dedupe.get(entry.dedupe_key) == entry.job_id:
                self._dedupe.pop(entry.dedupe_key, None)
            job = self._jobs.get(entry.job_id)
            if job is not None:
                job.status = "expired"
                job.analysis_id = None
                job.error = "Analysis result is no longer available"
            self._prune_jobs()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending:
                    return
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                if job.status == "cancelled":
                    continue
                job.status = "running"
                job.started_at = _now()
            self._run(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
        snapshot = None
        try:
            source = self.catalog.get(job.source_id)
            if source.fingerprint != job.source_version:
                raise RuntimeError("Trace source version is no longer cataloged")
            with self.catalog.source_snapshot(source, cancelled=lambda: job.cancel_requested) as snapshot:
                result = self._analyzer.analyze(snapshot, source.kind)
            _restore_source_identity(result, snapshot, source)
            result_size = _result_bytes(result, self._store.max_bytes)
            if result_size > self._store.max_bytes:
                raise ValueError("Analysis exceeds the result memory budget")
        except Exception as exc:  # The worker must preserve queue service after parser/tool failures.
            with self._lock:
                job = self._jobs[job_id]
                job.finished_at = _now()
                if job.cancel_requested:
                    job.status = "cancelled"
                else:
                    job.status = "failed"
                    job.error = self._safe_error(exc, source if "source" in locals() else None, snapshot)
                self._dedupe.pop(job.dedupe_key, None)
                self._prune_jobs()
            return

        with self._lock:
            job = self._jobs[job_id]
            job.finished_at = _now()
            if job.cancel_requested:
                job.status = "cancelled"
                self._dedupe.pop(job.dedupe_key, None)
                self._prune_jobs()
                return
            analysis_id = uuid.uuid4().hex
            entry = AnalysisEntry(
                analysis_id=analysis_id,
                job_id=job.job_id,
                dedupe_key=job.dedupe_key,
                source_id=source.source_id,
                source_name=source.name,
                source_kind=source.kind,
                source_version=source.fingerprint,
                created_at=job.finished_at,
                result=result,
            )
            job.status = "completed"
            job.analysis_id = analysis_id
            for evicted in self._store.add(entry, size=result_size):
                if self._dedupe.get(evicted.dedupe_key) == evicted.job_id:
                    self._dedupe.pop(evicted.dedupe_key, None)
                evicted_job = self._jobs.get(evicted.job_id)
                if evicted_job is not None:
                    evicted_job.status = "expired"
                    evicted_job.analysis_id = None
                    evicted_job.error = "Analysis result was evicted"
            self._prune_jobs()

    def _prune_jobs(self) -> None:
        while len(self._jobs) > self.max_job_history:
            removable_id = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.status in {"failed", "cancelled", "expired"}
                    or (job.status == "completed" and not self._store.contains(job.analysis_id))
                ),
                None,
            )
            if removable_id is None:
                return
            removed = self._jobs.pop(removable_id)
            if self._dedupe.get(removed.dedupe_key) == removable_id:
                self._dedupe.pop(removed.dedupe_key, None)

    def _make_job_room(self) -> None:
        while len(self._jobs) >= self.max_job_history:
            removable_id = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.status in {"failed", "cancelled", "expired"}
                    or (job.status == "completed" and not self._store.contains(job.analysis_id))
                ),
                None,
            )
            if removable_id is None:
                completed = next(
                    (job for job in self._jobs.values() if job.status == "completed" and job.analysis_id),
                    None,
                )
                if completed is None:
                    raise QueueFullError("Analysis job history is full")
                self._store.delete(completed.analysis_id)
                removable_id = completed.job_id
            removed = self._jobs.pop(removable_id)
            if self._dedupe.get(removed.dedupe_key) == removable_id:
                self._dedupe.pop(removed.dedupe_key, None)

    @staticmethod
    def _safe_error(exc: Exception, source: SourceRecord | None, snapshot: Path | None = None) -> str:
        logging.getLogger(__name__).error(
            "Trace analysis failed for source %s", source.source_id if source else "unknown", exc_info=exc
        )
        if isinstance(exc, SourceChangedError | SourceNotFoundError | UnsafeSourceError | OSError):
            return "Selected trace source is unavailable, changed, unsafe or exceeds snapshot limits"
        if isinstance(exc, ValueError):
            return "Analysis input is invalid or exceeds the configured result budget"
        return "Analysis could not be completed; inspect the service log"


class ComparisonManager:
    """Bounded single-worker trace comparison scheduler and result cache."""

    def __init__(
        self,
        catalog: SourceCatalog,
        analysis_manager: AnalysisManager,
        *,
        queue_size: int,
        max_comparisons: int,
        max_job_history: int,
        max_events: int,
        max_flame_nodes: int = DEFAULT_MAX_FLAME_NODES,
        analyzer: Any | None = None,
        comparison_service: Any | None = None,
    ):
        if min(queue_size, max_comparisons, max_job_history, max_events, max_flame_nodes) < 1:
            raise ValueError("Comparison limits must be positive")
        if max_job_history < max_comparisons:
            raise ValueError("Comparison job-history limit must be at least the result limit")
        self.catalog = catalog
        self.analysis_manager = analysis_manager
        self.queue_size = queue_size
        self.max_job_history = max_job_history
        self.max_flame_nodes = max_flame_nodes
        self._analyzer = analyzer or TracingAnalyzer(max_events)
        self._comparison_service = comparison_service or TraceComparisonService()
        self._store = ComparisonStore(max_comparisons)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pending: deque[str] = deque()
        self._jobs: OrderedDict[str, ComparisonJobRecord] = OrderedDict()
        self._inputs: dict[str, tuple[SourceRecord, str]] = {}
        self._dedupe: dict[str, str] = {}
        self._thread: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(target=self._worker, name="ibrobot-tracing-comparison", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            while self._pending:
                job = self._jobs[self._pending.popleft()]
                self._inputs.pop(job.job_id, None)
                job.status = "failed"
                job.finished_at = _now()
                job.error = "Comparison worker stopped before execution"
                self._dedupe.pop(job.dedupe_key, None)
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=5)

    @property
    def worker_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and not self._stopping

    @property
    def queued_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def comparison_count(self) -> int:
        with self._lock:
            return len(self._store)

    def submit(
        self,
        baseline_source_id: str,
        baseline_source_version: str,
        candidate_analysis_id: str,
        *,
        statistic: str,
        relative_threshold_percent: float,
        absolute_threshold_ms: float,
        metric: str | None,
        bins: int,
    ) -> tuple[ComparisonJobRecord, bool]:
        baseline_source = self.catalog.get(baseline_source_id)
        if baseline_source.fingerprint != baseline_source_version:
            raise ComparisonSourceUnavailableError("Selected baseline trace version is no longer cataloged")
        self.catalog.validate(baseline_source)
        candidate = self.analysis_manager.get_analysis(candidate_analysis_id)
        if baseline_source.source_id == candidate.source_id:
            raise SameSourceComparisonError("Baseline and candidate must be different trace sources")
        parameters = ComparisonParameters(
            statistic=statistic,
            relative_threshold_percent=float(relative_threshold_percent),
            absolute_threshold_ms=float(absolute_threshold_ms),
            metric=metric,
            bins=bins,
        )
        dedupe_key = self._dedupe_key(
            baseline_source.source_id,
            baseline_source.fingerprint,
            candidate_analysis_id,
            parameters,
        )

        with self._condition:
            existing_id = self._dedupe.get(dedupe_key)
            existing = self._jobs.get(existing_id) if existing_id else None
            if existing is not None and (
                existing.status in {"queued", "running"}
                or (existing.status == "completed" and self._store.contains(existing.comparison_id))
            ):
                return replace(existing), True
            if self._stopping:
                raise RuntimeError("Comparison worker is stopping")
            if len(self._pending) >= self.queue_size:
                raise ComparisonQueueFullError("Comparison queue is full")
            self._make_job_room()
            job = ComparisonJobRecord(
                job_id=uuid.uuid4().hex,
                baseline_source_id=baseline_source.source_id,
                baseline_source_version=baseline_source.fingerprint,
                candidate_analysis_id=candidate_analysis_id,
                parameters=parameters,
                dedupe_key=dedupe_key,
                status="queued",
                created_at=_now(),
            )
            self._jobs[job.job_id] = job
            # Queued jobs must not pin evicted, potentially large analysis results.
            self._inputs[job.job_id] = (baseline_source, candidate.analysis_id)
            self._dedupe[dedupe_key] = job.job_id
            self._pending.append(job.job_id)
            self._prune_jobs()
            self._condition.notify()
            return replace(job), False

    def get_job(self, job_id: str) -> tuple[ComparisonJobRecord, int | None]:
        with self._lock:
            try:
                job = self._jobs[job_id]
            except KeyError as exc:
                raise ComparisonJobNotFoundError(job_id) from exc
            try:
                queue_position = list(self._pending).index(job_id) + 1
            except ValueError:
                queue_position = None
            return replace(job), queue_position

    def list_comparisons(self) -> list[ComparisonEntry]:
        with self._lock:
            return self._store.list()

    def get_comparison(self, comparison_id: str) -> ComparisonEntry:
        with self._lock:
            return self._store.get(comparison_id)

    def delete_comparison(self, comparison_id: str) -> None:
        with self._lock:
            entry = self._store.delete(comparison_id)
            if self._dedupe.get(entry.dedupe_key) == entry.job_id:
                self._dedupe.pop(entry.dedupe_key, None)
            job = self._jobs.get(entry.job_id)
            if job is not None:
                job.status = "expired"
                job.comparison_id = None
                job.error = "Comparison result is no longer available"
            self._prune_jobs()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping and not self._pending:
                    return
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                job.status = "running"
                job.started_at = _now()
            self._run(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            baseline_source, candidate_id = self._inputs.pop(job_id)
        try:
            candidate = self.analysis_manager.get_analysis(candidate_id)
            current_source = self.catalog.get(job.baseline_source_id)
            if current_source.fingerprint != job.baseline_source_version:
                raise ComparisonSourceUnavailableError("Selected baseline trace version changed before comparison")
            with self.catalog.source_snapshot(baseline_source, cancelled=lambda: self._stopping) as snapshot:
                baseline_result = self._analyzer.analyze(snapshot, baseline_source.kind)
            _restore_source_identity(baseline_result, snapshot, baseline_source)
            budget = self.analysis_manager._store.max_bytes
            if _result_bytes(baseline_result, budget) > budget:
                raise ValueError("Baseline exceeds the result memory budget")
            parameters = job.parameters
            comparison = self._comparison_service.compare(
                baseline_result,
                candidate.result,
                statistic=parameters.statistic,
                relative_percent=parameters.relative_threshold_percent,
                absolute_ms=parameters.absolute_threshold_ms,
                metric=parameters.metric,
                bins=parameters.bins,
                max_flame_nodes=self.max_flame_nodes,
            )
            if self._stopping:
                raise RuntimeError("Comparison worker stopped")
            if _result_bytes(comparison, budget // self._store.max_items) > budget // self._store.max_items:
                raise ValueError("Comparison exceeds the result memory budget")
        except Exception as exc:  # The worker must preserve queue service after parser/tool failures.
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.finished_at = _now()
                job.error = self._safe_comparison_error(exc)
                self._dedupe.pop(job.dedupe_key, None)
                self._prune_jobs()
            return

        with self._lock:
            job = self._jobs[job_id]
            job.finished_at = _now()
            comparison_id = uuid.uuid4().hex
            entry = ComparisonEntry(
                comparison_id=comparison_id,
                job_id=job.job_id,
                dedupe_key=job.dedupe_key,
                baseline_source_id=job.baseline_source_id,
                baseline_source_version=job.baseline_source_version,
                candidate_analysis_id=job.candidate_analysis_id,
                created_at=job.finished_at,
                result=comparison,
            )
            evicted = self._store.add(entry)
            job.status = "completed"
            job.comparison_id = comparison_id
            if evicted is not None:
                self._expire(evicted, "Comparison result was evicted")
            self._prune_jobs()

    def _expire(self, entry: ComparisonEntry, message: str) -> None:
        if self._dedupe.get(entry.dedupe_key) == entry.job_id:
            self._dedupe.pop(entry.dedupe_key, None)
        job = self._jobs.get(entry.job_id)
        if job is not None:
            job.status = "expired"
            job.comparison_id = None
            job.error = message

    def _prune_jobs(self) -> None:
        while len(self._jobs) > self.max_job_history:
            removable_id = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.status in {"failed", "expired"}
                    or (job.status == "completed" and not self._store.contains(job.comparison_id))
                ),
                None,
            )
            if removable_id is None:
                return
            removed = self._jobs.pop(removable_id)
            if self._dedupe.get(removed.dedupe_key) == removable_id:
                self._dedupe.pop(removed.dedupe_key, None)

    def _make_job_room(self) -> None:
        while len(self._jobs) >= self.max_job_history:
            removable_id = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.status in {"failed", "expired"}
                    or (job.status == "completed" and not self._store.contains(job.comparison_id))
                ),
                None,
            )
            if removable_id is None:
                completed = next(
                    (job for job in self._jobs.values() if job.status == "completed" and job.comparison_id),
                    None,
                )
                if completed is None:
                    raise ComparisonQueueFullError("Comparison job history is full")
                entry = self._store.delete(completed.comparison_id)
                self._expire(entry, "Comparison result was evicted")
                removable_id = completed.job_id
            removed = self._jobs.pop(removable_id)
            if self._dedupe.get(removed.dedupe_key) == removable_id:
                self._dedupe.pop(removed.dedupe_key, None)

    @staticmethod
    def _dedupe_key(
        baseline_source_id: str,
        baseline_source_version: str,
        candidate_analysis_id: str,
        parameters: ComparisonParameters,
    ) -> str:
        payload = json.dumps(
            {
                "absolute_threshold_ms": parameters.absolute_threshold_ms,
                "baseline_source_id": baseline_source_id,
                "baseline_source_version": baseline_source_version,
                "bins": parameters.bins,
                "candidate_analysis_id": candidate_analysis_id,
                "metric": parameters.metric,
                "relative_threshold_percent": parameters.relative_threshold_percent,
                "statistic": parameters.statistic,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _safe_comparison_error(exc: Exception) -> str:
        if isinstance(
            exc, ComparisonSourceUnavailableError | SourceChangedError | SourceNotFoundError | UnsafeSourceError
        ):
            return "Selected baseline trace is unavailable or changed during comparison"
        if isinstance(exc, AnalysisNotFoundError):
            return "Candidate analysis is no longer available"
        if isinstance(exc, ValueError):
            return "Comparison inputs are not valid for the selected analyses"
        return f"{type(exc).__name__}: Comparison could not be completed"[:1000]


def analysis_counts(entry: AnalysisEntry) -> dict[str, int]:
    queries = ResultQueries(entry.result)
    return {
        "event_count": len(queries.events()),
        "request_count": len(queries.request_rows()),
        "span_count": len(queries.spans()),
        "flow_count": len(queries.flows()),
        "warning_count": len(queries.result.warnings),
    }

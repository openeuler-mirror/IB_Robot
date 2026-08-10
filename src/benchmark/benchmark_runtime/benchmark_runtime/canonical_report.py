"""Durable, provider-agnostic canonical benchmark report writer.

The evaluator owns these files while a concrete adapter owns native artifacts.
This module writes metadata and references only.  Every JSON document is
strict/deterministic, manifest and summary replacement is atomic, and each
JSONL episode line is flushed and fsynced before the caller may advance.
"""

from __future__ import annotations

import contextlib
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from benchmark_runtime._strict_json import StrictJSONError, dumps_strict, loads_strict, require_json_object
from benchmark_runtime.canonical_models import (
    ArtifactStatus,
    CanonicalCounters,
    CanonicalEpisodeRecord,
    CanonicalRunSummary,
    CanonicalTaskSummary,
)


class CanonicalReportError(RuntimeError):
    """Raised for durable report lifecycle or consistency failures."""


class DuplicateEpisodeError(CanonicalReportError):
    """Raised when the same episode identity is appended more than once."""


def _json_payload(value: Any) -> Any:
    if isinstance(value, CanonicalEpisodeRecord):
        return value.to_dict()
    if isinstance(value, CanonicalRunSummary):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported canonical payload type {type(value).__name__}")


def _strict_object(value: Any, name: str) -> dict[str, Any]:
    payload = _json_payload(value)
    try:
        parsed = loads_strict(dumps_strict(payload))
    except StrictJSONError as exc:
        raise CanonicalReportError(f"{name} is not strict JSON: {exc}") from exc
    return dict(require_json_object(parsed, name))


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability after ``os.replace``."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    indent: int | None = None,
) -> None:
    """Write a strict JSON object through a same-directory atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = dumps_strict(payload, indent=indent) + "\n"
    except StrictJSONError as exc:
        raise CanonicalReportError(f"payload for {destination} is not strict JSON: {exc}") from exc

    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except OSError as exc:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
        raise CanonicalReportError(f"atomic write failed for {destination}: {exc}") from exc


def _episode_identity(record: CanonicalEpisodeRecord) -> tuple[str, str, int, int]:
    return record.run_id, record.suite, record.task_id, record.episode_index


def _coerce_episode(record: CanonicalEpisodeRecord | Mapping[str, Any]) -> CanonicalEpisodeRecord:
    if isinstance(record, CanonicalEpisodeRecord):
        return record
    if not isinstance(record, Mapping):
        raise TypeError("episode must be CanonicalEpisodeRecord or mapping")
    raw = dict(record)
    artifacts = tuple(_coerce_artifact(item) for item in raw.pop("artifacts", ()))
    return CanonicalEpisodeRecord(**raw, artifacts=artifacts)


def _coerce_artifact(value: ArtifactStatus | Mapping[str, Any]) -> ArtifactStatus:
    if isinstance(value, ArtifactStatus):
        return value
    if isinstance(value, Mapping):
        return ArtifactStatus(**dict(value))
    raise TypeError("artifact must be ArtifactStatus or mapping")


def _read_jsonl(path: Path) -> list[CanonicalEpisodeRecord]:
    if not path.exists():
        return []
    records: list[CanonicalEpisodeRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = require_json_object(loads_strict(line), f"episodes.jsonl line {line_number}")
                records.append(_coerce_episode(raw))
            except (StrictJSONError, TypeError, ValueError) as exc:
                raise CanonicalReportError(f"invalid episodes.jsonl line {line_number}: {exc}") from exc
    return records


def _planned_for_task(
    task_id: int,
    task_name: str,
    planned_episodes: int,
    planned_tasks: Mapping[int, int | Mapping[str, Any]] | None,
) -> tuple[int, str]:
    if planned_tasks is None or task_id not in planned_tasks:
        return planned_episodes, task_name
    item = planned_tasks[task_id]
    if isinstance(item, Mapping):
        count = int(item.get("planned_episodes", item.get("episodes", 0)))
        name = str(item.get("task_name", task_name))
    else:
        count = int(item)
        name = task_name
    if count < 0:
        raise ValueError("planned task episode count must be non-negative")
    return count, name


def reduce_episode_records(
    records: Iterable[CanonicalEpisodeRecord],
    *,
    run_id: str | None = None,
    suite: str | None = None,
    planned_episodes: int,
    planned_tasks: Mapping[int, int | Mapping[str, Any]] | None = None,
    run_status: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CanonicalRunSummary:
    """Reduce ordered episode facts without changing native success semantics."""
    materialized = [_coerce_episode(record) for record in records]
    seen: set[tuple[str, str, int, int]] = set()
    for record in materialized:
        identity = _episode_identity(record)
        if identity in seen:
            raise DuplicateEpisodeError(f"duplicate episode identity: {identity}")
        seen.add(identity)
        if run_id is not None and record.run_id != run_id:
            raise CanonicalReportError("episode run_id does not match reducer run_id")
        if suite is not None and record.suite != suite:
            raise CanonicalReportError("episode suite does not match reducer suite")
    if not isinstance(planned_episodes, int) or isinstance(planned_episodes, bool) or planned_episodes < 0:
        raise ValueError("planned_episodes must be a non-negative int")
    effective_run_id = run_id or (materialized[0].run_id if materialized else "unknown-run")
    effective_suite = suite or (materialized[0].suite if materialized else "unknown-suite")

    completed = sum(record.native_result_available for record in materialized)
    successful = sum(record.native_result_available and record.is_success is True for record in materialized)
    native_failed = completed - successful
    infrastructure = sum(record.infrastructure_error is not None for record in materialized)
    artifact_failures = sum(bool(record.artifact_failures) for record in materialized)
    counters = CanonicalCounters(
        planned_episodes=planned_episodes,
        started_episodes=len(materialized),
        completed_native_episodes=completed,
        successful_native_episodes=successful,
        native_failed_episodes=native_failed,
        infrastructure_error_episodes=infrastructure,
        artifact_failure_episodes=artifact_failures,
    )

    grouped: dict[int, list[CanonicalEpisodeRecord]] = defaultdict(list)
    for record in materialized:
        grouped[record.task_id].append(record)
    task_ids = sorted(set(grouped) | set(planned_tasks or {}))
    task_summaries: list[CanonicalTaskSummary] = []
    for task_id in task_ids:
        task_records = grouped.get(task_id, [])
        fallback_name = task_records[0].task_name if task_records else f"task_{task_id:03d}"
        planned_for_task, task_name = _planned_for_task(task_id, fallback_name, len(task_records), planned_tasks)
        task_completed = sum(record.native_result_available for record in task_records)
        task_successful = sum(record.native_result_available and record.is_success is True for record in task_records)
        task_artifact_failures = sum(bool(record.artifact_failures) for record in task_records)
        task_status = "complete"
        if len(task_records) < planned_for_task or task_artifact_failures:
            task_status = "partial"
        task_summaries.append(
            CanonicalTaskSummary(
                task_id=task_id,
                task_name=task_name,
                planned_episodes=planned_for_task,
                records=len(task_records),
                completed_native_episodes=task_completed,
                successful_native_episodes=task_successful,
                artifact_failure_episodes=task_artifact_failures,
                status=task_status,
            )
        )

    if run_status is None:
        if len(materialized) == planned_episodes and infrastructure == 0 and artifact_failures == 0:
            effective_status = "complete"
        elif materialized:
            effective_status = "partial"
        else:
            effective_status = "failed" if planned_episodes else "complete"
    else:
        if run_status not in {"running", "complete", "partial", "failed"}:
            raise ValueError("run_status must be running, complete, partial, or failed")
        task_complete = all(task.status == "complete" for task in task_summaries)
        exact_record_count = len(materialized) == planned_episodes
        effective_status = (
            "partial" if run_status == "complete" and (not task_complete or not exact_record_count) else run_status
        )
    failures = tuple(artifact for record in materialized for artifact in record.artifact_failures)
    return CanonicalRunSummary(
        run_id=effective_run_id,
        suite=effective_suite,
        run_status=effective_status,
        counters=counters,
        tasks=task_summaries,
        artifact_failures=failures,
        metadata=metadata or {},
    )


class CanonicalReportWriter:
    """Own the canonical run files and enforce durable episode append order."""

    def __init__(self, run_root: str | Path) -> None:
        self.run_root = Path(run_root)
        self.manifest_path = self.run_root / "run_manifest.json"
        self.episodes_path = self.run_root / "episodes.jsonl"
        self.summary_path = self.run_root / "summary.json"
        self._started = False
        self._seen: set[tuple[str, str, int, int]] = set()

    @property
    def records(self) -> tuple[CanonicalEpisodeRecord, ...]:
        return tuple(_read_jsonl(self.episodes_path))

    def start_run(self, manifest: Mapping[str, Any]) -> None:
        """Create the run root and initial running manifest before reset."""
        if self._started:
            raise CanonicalReportError("run has already started")
        payload = _strict_object(manifest, "run_manifest")
        if payload.get("status") in {"complete", "partial", "failed"}:
            raise CanonicalReportError("initial manifest status must not be terminal")
        if self.run_root.exists():
            existing = list(self.run_root.iterdir())
            unexpected = [item for item in existing if item.name != "native" or not item.is_dir()]
            if unexpected:
                raise CanonicalReportError(f"run root contains unexpected entries: {unexpected}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_path, payload)
        with self.episodes_path.open("a", encoding="utf-8"):
            pass
        self._started = True

    def append_episode(self, record: CanonicalEpisodeRecord | Mapping[str, Any]) -> None:
        """Append one strict JSON line and fsync it before returning."""
        if not self._started:
            raise CanonicalReportError("start_run must be called first")
        episode = _coerce_episode(record)
        identity = _episode_identity(episode)
        if identity in self._seen:
            raise DuplicateEpisodeError(f"duplicate episode identity: {identity}")
        encoded = dumps_strict(episode.to_dict()) + "\n"
        try:
            with self.episodes_path.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise CanonicalReportError(f"episode append failed: {exc}") from exc
        self._seen.add(identity)

    def write_summary(self, summary: CanonicalRunSummary | Mapping[str, Any]) -> None:
        """Atomically replace ``summary.json`` with a readable strict object."""
        payload = _strict_object(summary, "summary")
        atomic_write_json(self.summary_path, payload, indent=2)

    def finalize_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Atomically replace the run manifest with its terminal snapshot."""
        payload = _strict_object(manifest, "run_manifest")
        atomic_write_json(self.manifest_path, payload)

    def finalize(
        self,
        summary: CanonicalRunSummary,
        *,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        """Write summary and, optionally, the terminal manifest."""
        self.write_summary(summary)
        if manifest is not None:
            self.finalize_manifest(manifest)

    def abort(self, summary: CanonicalRunSummary, *, manifest: Mapping[str, Any] | None = None) -> None:
        """Persist a partial/failed summary; never manufacture ``complete``."""
        if summary.run_status == "complete":
            raise CanonicalReportError("abort cannot persist a complete summary")
        self.finalize(summary, manifest=manifest)


def load_episode_records(path: str | Path) -> tuple[CanonicalEpisodeRecord, ...]:
    """Load and validate an append-only episode JSONL file."""
    return tuple(_read_jsonl(Path(path)))

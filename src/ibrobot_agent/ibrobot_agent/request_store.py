"""SQLite request ledger for ibrobot_agent.

The store owns de-duplication, request state transitions, and audit events.
It keeps the contract narrow so higher layers can remain testable without ROS.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from embodied_common.canon import to_canonical_json
from ibrobot_agent.contracts import ExecutionResult, Presentation, RequestKey, RequestRecord, RequestStore, TaskRef

_ACTIVE_STATES = {
    "RECEIVED",
    "PLANNING",
    "PROPOSAL_READY",
    "PREPARING",
    "MAY_EXECUTE",
    "RUNNING",
    "STOPPING",
}


class RequestStoreError(ValueError):
    """Raised when the SQLite ledger cannot satisfy a request mutation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _task_ref_to_json(task_ref: TaskRef | None) -> str:
    if task_ref is None:
        return ""
    return to_canonical_json(task_ref.to_dict())


def _task_ref_from_json(raw: str | None) -> TaskRef | None:
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RequestStoreError("REQUEST_SCHEMA_INVALID", "task_ref_json must decode to an object")
    return _task_ref_from_value(payload)


def _task_ref_from_value(value: Any) -> TaskRef | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RequestStoreError("REQUEST_SCHEMA_INVALID", "task_ref_json must decode to an object")
    return TaskRef(
        task_id=str(value["task_id"]),
        plan_id=str(value["plan_id"]),
        plan_digest=str(value["plan_digest"]),
        registry_epoch=str(value["registry_epoch"]),
        registry_generation=int(value["registry_generation"]),
        registry_digest=str(value["registry_digest"]),
        expected_step_count=int(value["expected_step_count"]),
    )


def _execution_result_to_json(result: ExecutionResult) -> str:
    return to_canonical_json(result.to_dict())


def _execution_result_from_json(raw: str | None) -> ExecutionResult | None:
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RequestStoreError("REQUEST_SCHEMA_INVALID", "terminal_json must decode to an object")
    task_ref = _task_ref_from_value(payload.get("task_ref"))
    detail = payload.get("detail", {})
    if not isinstance(detail, dict):
        raise RequestStoreError("REQUEST_SCHEMA_INVALID", "terminal detail must be a mapping")
    return ExecutionResult(
        status=str(payload["status"]),
        task_ref=task_ref,
        error_code=str(payload.get("error_code", "")),
        message=str(payload.get("message", "")),
        detail=detail,
    )


def _request_key_tuple(key: RequestKey) -> tuple[str, str, str, str]:
    return key.robot_scope, key.channel_id, key.principal_id, key.request_id


class SQLiteRequestStore(RequestStore):
    """SQLite-backed request ledger with fail-closed conditional updates."""

    def __init__(self, path: str | Path, *, clock: Callable[[], str] | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or _now_text
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(self._path, check_same_thread=False, timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0].lower() != "wal":
            raise RequestStoreError("STORAGE_UNAVAILABLE", "request store requires SQLite WAL mode")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize_schema()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def admit(self, key: RequestKey, *, input_hash: str, session_id: str) -> RequestRecord:
        if not input_hash:
            raise RequestStoreError("REQUEST_SCHEMA_INVALID", "input_hash must be non-empty")
        with self._transaction():
            row = self._fetch_row(key)
            if row is not None:
                if row["input_hash"] != input_hash:
                    raise RequestStoreError("REQUEST_ID_CONFLICT", "request_id payload conflicts with stored request")
                return self._row_to_record(row)
            now = self._clock()
            self._connection.execute(
                """
                INSERT INTO requests (
                    robot_scope, channel_id, principal_id, request_id,
                    input_hash, session_id, state, planning_generation,
                    stop_requested, may_have_submitted, task_ref_json, proposal_json,
                    terminal_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'RECEIVED', 0, 0, 0, NULL, NULL, NULL, ?, ?)
                """,
                (*_request_key_tuple(key), input_hash, session_id, now, now),
            )
            row = self._fetch_row(key)
            assert row is not None
            self._append_event_locked(key, event_type="admit", state="RECEIVED", detail={"input_hash": input_hash})
            return self._row_to_record(row)

    def get_request(self, key: RequestKey) -> RequestRecord:
        with self._transaction(read_only=True):
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            return self._row_to_record(row)

    def is_robot_quarantined(self, robot_scope: str) -> bool:
        with self._transaction(read_only=True):
            row = self._connection.execute(
                "SELECT 1 FROM requests WHERE robot_scope = ? AND state = 'UNKNOWN' LIMIT 1",
                (robot_scope,),
            ).fetchone()
            return row is not None

    def begin_planning(self, key: RequestKey, *, expected_generation: int) -> RequestRecord:
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            current_generation = int(row["planning_generation"])
            if row["state"] == "PLANNING" and current_generation == expected_generation + 1:
                return self._row_to_record(row)
            self._ensure_generation(row, expected_generation)
            self._ensure_state(row, {"RECEIVED"})
            now = self._clock()
            self._connection.execute(
                """
                UPDATE requests
                SET state = 'PLANNING', planning_generation = planning_generation + 1, updated_at = ?
                WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
                  AND planning_generation = ? AND state = 'RECEIVED'
                """,
                (now, *_request_key_tuple(key), expected_generation),
            )
            updated = self._fetch_row(key)
            if updated is None or updated["state"] != "PLANNING":
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request cannot begin planning")
            self._append_event_locked(
                key, event_type="begin_planning", state="PLANNING", detail={"generation": expected_generation + 1}
            )
            return self._row_to_record(updated)

    def mark_preparing(self, key: RequestKey, *, expected_generation: int) -> RequestRecord:
        return self._transition_state(
            key,
            expected_generation=expected_generation,
            from_states={"PROPOSAL_READY"},
            to_state="PREPARING",
            event_type="mark_preparing",
        )

    def mark_proposal_ready(self, key: RequestKey, *, expected_generation: int) -> RequestRecord:
        return self._transition_state(
            key,
            expected_generation=expected_generation,
            from_states={"PLANNING"},
            to_state="PROPOSAL_READY",
            event_type="mark_proposal_ready",
        )

    def bind_task(
        self,
        key: RequestKey,
        *,
        expected_generation: int,
        task_ref: TaskRef,
        presentation: Presentation,
        proposal_json: str,
    ) -> RequestRecord:
        if not proposal_json:
            raise RequestStoreError("REQUEST_SCHEMA_INVALID", "proposal_json must be non-empty")
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            self._ensure_generation(row, expected_generation)
            if row["state"] == "MAY_EXECUTE" and row["task_ref_json"]:
                stored_task_ref = _task_ref_from_json(row["task_ref_json"])
                if stored_task_ref == task_ref and row["proposal_json"] == proposal_json:
                    return self._row_to_record(row)
            self._ensure_state(row, {"PREPARING"})
            now = self._clock()
            self._connection.execute(
                """
                UPDATE requests
                SET state = 'MAY_EXECUTE', task_ref_json = ?, proposal_json = ?, updated_at = ?
                WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
                  AND planning_generation = ? AND state = 'PREPARING'
                """,
                (_task_ref_to_json(task_ref), proposal_json, now, *_request_key_tuple(key), expected_generation),
            )
            updated = self._fetch_row(key)
            if updated is None or updated["state"] != "MAY_EXECUTE":
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request cannot bind a task")
            self._append_event_locked(
                key,
                event_type="bind_task",
                state="MAY_EXECUTE",
                detail={"task_ref": task_ref.to_dict(), "summary": getattr(presentation, "summary", "")},
            )
            return self._row_to_record(updated)

    def mark_stop(self, key: RequestKey, *, expected_generation: int) -> RequestRecord:
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            self._ensure_generation(row, expected_generation)
            if row["stop_requested"]:
                return self._row_to_record(row)
            if row["state"] not in _ACTIVE_STATES:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request is no longer active")
            new_state = "STOPPING" if row["state"] != "RECEIVED" else "STOPPING"
            now = self._clock()
            self._connection.execute(
                """
                UPDATE requests
                SET stop_requested = 1, state = ?, updated_at = ?
                WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
                  AND planning_generation = ? AND state IN ('RECEIVED', 'PLANNING', 'PROPOSAL_READY', 'PREPARING', 'MAY_EXECUTE', 'RUNNING', 'STOPPING')
                """,
                (new_state, now, *_request_key_tuple(key), expected_generation),
            )
            updated = self._fetch_row(key)
            if updated is None or not updated["stop_requested"]:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request stop could not be recorded")
            self._append_event_locked(
                key, event_type="mark_stop", state=str(updated["state"]), detail={"stop_requested": True}
            )
            return self._row_to_record(updated)

    def mark_submitted(self, key: RequestKey, *, expected_generation: int, task_ref: TaskRef) -> RequestRecord:
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            self._ensure_generation(row, expected_generation)
            stored_task_ref = _task_ref_from_json(row["task_ref_json"])
            if stored_task_ref is not None and stored_task_ref != task_ref:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "task_ref does not match the stored proposal")
            if row["state"] not in {"MAY_EXECUTE", "RUNNING", "STOPPING"}:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request cannot be marked submitted")
            new_state = "STOPPING" if row["stop_requested"] else "RUNNING"
            now = self._clock()
            self._connection.execute(
                """
                UPDATE requests
                SET may_have_submitted = 1, state = ?, task_ref_json = COALESCE(task_ref_json, ?), updated_at = ?
                WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
                  AND planning_generation = ? AND state IN ('MAY_EXECUTE', 'RUNNING', 'STOPPING')
                """,
                (new_state, _task_ref_to_json(task_ref), now, *_request_key_tuple(key), expected_generation),
            )
            updated = self._fetch_row(key)
            if updated is None or not updated["may_have_submitted"]:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request submission could not be recorded")
            self._append_event_locked(
                key, event_type="mark_submitted", state=str(updated["state"]), detail={"task_ref": task_ref.to_dict()}
            )
            return self._row_to_record(updated)

    def record_confirmation(
        self, key: RequestKey, *, expected_generation: int, detail: Mapping[str, object]
    ) -> RequestRecord:
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            self._ensure_generation(row, expected_generation)
            self._append_event_locked(
                key, event_type="record_confirmation", state=str(row["state"]), detail=dict(detail)
            )
            return self._row_to_record(row)

    def finish(self, key: RequestKey, *, expected_generation: int, result: ExecutionResult) -> RequestRecord:
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            self._ensure_generation(row, expected_generation)
            self._ensure_terminal_compatibility(row, result)
            target_state = self._state_from_execution_result(result.status, result.task_ref is not None)
            now = self._clock()
            terminal_json = _execution_result_to_json(result)
            self._connection.execute(
                """
                UPDATE requests
                SET state = ?, terminal_json = ?, updated_at = ?
                WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
                  AND planning_generation = ?
                """,
                (target_state, terminal_json, now, *_request_key_tuple(key), expected_generation),
            )
            updated = self._fetch_row(key)
            if updated is None or str(updated["terminal_json"]) != terminal_json:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request terminal could not be recorded")
            self._append_event_locked(key, event_type="finish", state=target_state, detail=result.to_dict())
            return self._row_to_record(updated)

    def mark_answered(
        self,
        key: RequestKey,
        *,
        expected_generation: int,
        message: str,
        error_code: str = "",
    ) -> RequestRecord:
        return self.finish(
            key,
            expected_generation=expected_generation,
            result=ExecutionResult(
                status="succeeded",
                task_ref=None,
                error_code=error_code,
                message=message,
                detail={},
            ),
        )

    def quarantine(self, key: RequestKey, *, expected_generation: int, reason: str) -> RequestRecord:
        if not reason:
            raise RequestStoreError("REQUEST_SCHEMA_INVALID", "reason must be non-empty")
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            self._ensure_generation(row, expected_generation)
            now = self._clock()
            terminal = ExecutionResult(
                status="unknown",
                task_ref=_task_ref_from_json(row["task_ref_json"]),
                error_code="QUARANTINED",
                message=reason,
                detail={"reason": reason},
            )
            terminal_json = _execution_result_to_json(terminal)
            self._connection.execute(
                """
                UPDATE requests
                SET state = 'UNKNOWN', terminal_json = ?, updated_at = ?
                WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
                  AND planning_generation = ?
                """,
                (terminal_json, now, *_request_key_tuple(key), expected_generation),
            )
            updated = self._fetch_row(key)
            if updated is None:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "request quarantine could not be recorded")
            self._append_event_locked(key, event_type="quarantine", state="UNKNOWN", detail={"reason": reason})
            return self._row_to_record(updated)

    def record_event(self, key: RequestKey, *, event_type: str, state: str, detail_json: str) -> int:
        if not event_type:
            raise RequestStoreError("REQUEST_SCHEMA_INVALID", "event_type must be non-empty")
        if not state:
            raise RequestStoreError("REQUEST_SCHEMA_INVALID", "state must be non-empty")
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            return self._append_event_locked(key, event_type=event_type, state=state, detail_json=detail_json)

    def _initialize_schema(self) -> None:
        with self._lock:
            user_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if user_version not in {0, 1}:
                raise RequestStoreError(
                    "STORAGE_UNAVAILABLE", f"unsupported request store schema version: {user_version}"
                )
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS requests (
                    robot_scope TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    planning_generation INTEGER NOT NULL,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    may_have_submitted INTEGER NOT NULL DEFAULT 0,
                    task_ref_json TEXT,
                    proposal_json TEXT,
                    terminal_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (robot_scope, channel_id, principal_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS request_events (
                    robot_scope TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    detail_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (robot_scope, channel_id, principal_id, request_id, sequence)
                );
                """
            )
            if user_version == 0:
                self._connection.execute("PRAGMA user_version = 1")

    @contextmanager
    def _transaction(self, *, read_only: bool = False) -> Iterator[None]:
        with self._lock:
            if self._closed:
                raise RequestStoreError("STORAGE_UNAVAILABLE", "request store is closed")
            if read_only:
                try:
                    yield
                finally:
                    pass
                return
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _fetch_row(self, key: RequestKey):
        row = self._connection.execute(
            """
            SELECT * FROM requests
            WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
            """,
            _request_key_tuple(key),
        ).fetchone()
        return row

    def _row_to_record(self, row) -> RequestRecord:
        key = RequestKey(
            robot_scope=str(row["robot_scope"]),
            channel_id=str(row["channel_id"]),
            principal_id=str(row["principal_id"]),
            request_id=str(row["request_id"]),
        )
        return RequestRecord(
            key=key,
            session_id=str(row["session_id"]),
            state=str(row["state"]),
            planning_generation=int(row["planning_generation"]),
            stop_requested=bool(row["stop_requested"]),
            may_have_submitted=bool(row["may_have_submitted"]),
            input_hash=str(row["input_hash"]),
            task_ref=_task_ref_from_json(row["task_ref_json"]),
            proposal_json=str(row["proposal_json"] or ""),
            terminal=_execution_result_from_json(row["terminal_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _ensure_generation(self, row, expected_generation: int) -> None:
        if int(row["planning_generation"]) != expected_generation:
            raise RequestStoreError("REQUEST_ID_CONFLICT", "request generation does not match")

    @staticmethod
    def _ensure_state(row, allowed: set[str]) -> None:
        if str(row["state"]) not in allowed:
            raise RequestStoreError("REQUEST_ID_CONFLICT", "request is in the wrong state")

    def _ensure_terminal_compatibility(self, row, result: ExecutionResult) -> None:
        existing = row["terminal_json"]
        if existing:
            stored = _execution_result_from_json(existing)
            if stored != result:
                raise RequestStoreError("REQUEST_ID_CONFLICT", "terminal result is immutable")
        stored_task_ref = _task_ref_from_json(row["task_ref_json"])
        if stored_task_ref is not None and result.task_ref is not None and stored_task_ref != result.task_ref:
            raise RequestStoreError("REQUEST_ID_CONFLICT", "task_ref does not match stored request")

    @staticmethod
    def _state_from_execution_result(status: str, has_task_ref: bool) -> str:
        if not has_task_ref and status == "cancelled":
            return "CANCELLED_BEFORE_EXECUTION"
        if not has_task_ref and status == "succeeded":
            return "ANSWERED"
        if status == "succeeded":
            return "SUCCEEDED"
        if status == "failed":
            return "FAILED"
        if status == "cancelled":
            return "CANCELLED"
        return "UNKNOWN"

    def _transition_state(
        self,
        key: RequestKey,
        *,
        expected_generation: int,
        from_states: set[str],
        to_state: str,
        event_type: str,
    ) -> RequestRecord:
        with self._transaction():
            row = self._fetch_row(key)
            if row is None:
                raise RequestStoreError("REQUEST_NOT_FOUND", "request is not known")
            self._ensure_generation(row, expected_generation)
            if str(row["state"]) == to_state:
                return self._row_to_record(row)
            self._ensure_state(row, from_states)
            now = self._clock()
            self._connection.execute(
                """
                UPDATE requests
                SET state = ?, updated_at = ?
                WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
                  AND planning_generation = ? AND state IN ({states})
                """.format(states=", ".join(f"'{state}'" for state in sorted(from_states))),
                (to_state, now, *_request_key_tuple(key), expected_generation),
            )
            updated = self._fetch_row(key)
            if updated is None or str(updated["state"]) != to_state:
                raise RequestStoreError("REQUEST_ID_CONFLICT", f"request cannot transition to {to_state}")
            self._append_event_locked(
                key, event_type=event_type, state=to_state, detail={"expected_generation": expected_generation}
            )
            return self._row_to_record(updated)

    def _append_event_locked(
        self,
        key: RequestKey,
        *,
        event_type: str,
        state: str,
        detail: Mapping[str, object] | None = None,
        detail_json: str | None = None,
    ) -> int:
        if detail_json is None:
            detail_json = to_canonical_json(detail or {})
        now = self._clock()
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
            FROM request_events
            WHERE robot_scope = ? AND channel_id = ? AND principal_id = ? AND request_id = ?
            """,
            _request_key_tuple(key),
        ).fetchone()
        assert row is not None
        sequence = int(row[0])
        self._connection.execute(
            """
            INSERT INTO request_events (
                robot_scope, channel_id, principal_id, request_id, sequence,
                event_type, state, detail_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*_request_key_tuple(key), sequence, event_type, state, detail_json, now),
        )
        return sequence


__all__ = ["RequestStoreError", "SQLiteRequestStore"]

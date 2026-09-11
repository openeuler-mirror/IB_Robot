"""Persistent bounded conversation memory for the Agent incubator."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ibrobot_agent.contracts import AgentRequest


class SQLiteConversationStore:
    """Store session-isolated messages without importing ROS."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_turns: int = 12,
        clarification_ttl_sec: float = 300.0,
    ) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_messages = max_turns * 2
        if clarification_ttl_sec <= 0.0:
            raise ValueError("clarification_ttl_sec must be positive")
        self._clarification_ttl_sec = float(clarification_ttl_sec)
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(self._path, check_same_thread=False, timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        if self._connection.execute("PRAGMA journal_mode = WAL").fetchone()[0].lower() != "wal":
            raise RuntimeError("conversation store requires SQLite WAL mode")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        schema_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if schema_version not in {0, 1}:
            raise RuntimeError(f"unsupported conversation store schema version: {schema_version}")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                principal_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                robot_scope TEXT NOT NULL,
                request_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS conversation_session_idx
            ON conversation_messages(session_id, principal_id, channel_id, robot_scope, sequence);
            CREATE TABLE IF NOT EXISTS clarification_contexts (
                request_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                principal_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                robot_scope TEXT NOT NULL,
                original_text TEXT NOT NULL,
                missing_fields_json TEXT NOT NULL,
                expires_at REAL NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (request_id, session_id, principal_id, channel_id, robot_scope)
            );
            """
        )
        if schema_version == 0:
            self._connection.execute("PRAGMA user_version = 1")
            self._connection.commit()

    def append(self, request: AgentRequest, *, role: str = "user") -> None:
        self._append(request, role=role, content=request.text)

    def append_assistant(self, request: AgentRequest, content: str) -> None:
        self._append(request, role="assistant", content=content)

    def context(self, request: AgentRequest) -> list[dict[str, str]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT role, content FROM conversation_messages
                WHERE session_id = ? AND principal_id = ? AND channel_id = ? AND robot_scope = ?
                ORDER BY sequence DESC LIMIT ?
                """,
                (*self._session_key(request), self._max_messages),
            ).fetchall()
            return [{"role": str(row["role"]), "content": str(row["content"])} for row in reversed(rows)]

    def save_clarification(self, request: AgentRequest, missing_fields) -> None:
        with self._lock:
            self._ensure_open()
            self._connection.execute(
                """
                INSERT OR REPLACE INTO clarification_contexts (
                    request_id, session_id, principal_id, channel_id, robot_scope,
                    original_text, missing_fields_json, expires_at, consumed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    request.request_id,
                    *self._session_key(request),
                    request.text,
                    json.dumps(list(missing_fields), ensure_ascii=False, sort_keys=True),
                    time.time() + self._clarification_ttl_sec,
                ),
            )
            self._connection.commit()

    def consume_clarification(self, request: AgentRequest) -> dict[str, object] | None:
        if request.reply_to_request_id is None:
            return None
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT * FROM clarification_contexts
                    WHERE request_id = ? AND session_id = ? AND principal_id = ? AND channel_id = ?
                      AND robot_scope = ? AND consumed = 0 AND expires_at > ?
                    """,
                    (request.reply_to_request_id, *self._session_key(request), time.time()),
                ).fetchone()
                if row is None:
                    self._connection.rollback()
                    return None
                self._connection.execute(
                    """
                    UPDATE clarification_contexts SET consumed = 1
                    WHERE request_id = ? AND session_id = ? AND principal_id = ? AND channel_id = ?
                      AND robot_scope = ? AND consumed = 0
                    """,
                    (request.reply_to_request_id, *self._session_key(request)),
                )
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()
            return {
                "request_id": str(row["request_id"]),
                "original_text": str(row["original_text"]),
                "missing_fields": json.loads(str(row["missing_fields_json"])),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def _append(self, request: AgentRequest, *, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("conversation role must be user or assistant")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("conversation content must be non-empty")
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    INSERT INTO conversation_messages (
                        session_id, principal_id, channel_id, robot_scope, request_id, role, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*self._session_key(request), request.request_id, role, content.strip()),
                )
                self._connection.execute(
                    """
                    DELETE FROM conversation_messages WHERE sequence IN (
                        SELECT sequence FROM conversation_messages
                        WHERE session_id = ? AND principal_id = ? AND channel_id = ? AND robot_scope = ?
                        ORDER BY sequence DESC LIMIT -1 OFFSET ?
                    )
                    """,
                    (*self._session_key(request), self._max_messages),
                )
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @staticmethod
    def _session_key(request: AgentRequest) -> tuple[str, str, str, str]:
        return request.session_id, request.principal_id, request.channel_id, request.robot_scope

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("conversation store is closed")


__all__ = ["SQLiteConversationStore"]

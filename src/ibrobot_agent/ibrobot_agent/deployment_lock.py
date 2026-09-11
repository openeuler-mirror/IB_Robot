"""Single-owner deployment lock for an Agent robot scope."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import TextIO


class DeploymentLock:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: TextIO | None = self._path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._stream.close()
            self._stream = None
            raise RuntimeError(f"Agent deployment lock is already held: {self._path}") from exc

    def close(self) -> None:
        if self._stream is None:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None


__all__ = ["DeploymentLock"]

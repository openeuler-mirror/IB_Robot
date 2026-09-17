"""Exact-plan display receipts for the Agent's immediate execution mode."""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from embodied_common.canon import sha256_text, to_canonical_json
from ibrobot_agent.contracts import Presentation, RequestKey


class PresentationError(RuntimeError):
    """No valid display receipt was received before execution."""


@dataclass
class _PendingPresentation:
    digest: str
    token: str
    deadline: float
    cancel_token: threading.Event
    rendered: threading.Event = field(default_factory=threading.Event)


def presentation_digest(value: Mapping[str, object]) -> str:
    """Bind all displayed plan fields, including the exact task and registry tuple."""
    return sha256_text(to_canonical_json(value))


class PresentationGate:
    """Wait for a bounded, correlated receipt from the presenting client.

    The receipt proves display completion, not user approval or motion authority.
    There is deliberately no no-client or transport-failure fallback.
    """

    def __init__(self, timeout_sec: float) -> None:
        if isinstance(timeout_sec, bool) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("presentation_timeout_sec must be finite and positive")
        self._timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._pending: dict[RequestKey, _PendingPresentation] = {}

    def present(
        self,
        key: RequestKey,
        presentation: Presentation,
        *,
        publish: Callable[[Mapping[str, object]], None],
        cancel_token: threading.Event,
    ) -> None:
        value = presentation.to_dict()
        pending = _PendingPresentation(
            presentation_digest(value),
            secrets.token_urlsafe(32),
            time.monotonic() + self._timeout_sec,
            cancel_token,
        )
        with self._lock:
            if key in self._pending:
                raise PresentationError("request already has a pending presentation")
            self._pending[key] = pending
        try:
            if cancel_token.is_set():
                raise PresentationError("stopped before presentation")
            publish({"presentation": value, "presentation_digest": pending.digest, "receipt_token": pending.token})
            while True:
                if cancel_token.is_set():
                    raise PresentationError("stopped while awaiting presentation")
                if pending.rendered.is_set():
                    return
                remaining = pending.deadline - time.monotonic()
                if remaining <= 0:
                    raise PresentationError("exact plan presentation receipt timed out")
                pending.rendered.wait(min(remaining, 0.05))
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def acknowledge(self, key: RequestKey, *, digest: str, token: str) -> bool:
        if not isinstance(digest, str) or not isinstance(token, str):
            return False
        with self._lock:
            pending = self._pending.get(key)
            if pending is None or pending.cancel_token.is_set() or time.monotonic() >= pending.deadline:
                return False
            if digest != pending.digest or not secrets.compare_digest(token, pending.token):
                return False
            pending.rendered.set()
            return True

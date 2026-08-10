"""Completion-aware ActionExecutor contract (v2).

completion-aware executor contract evolves the executor registry contract synchronous ABC into a v2 contract that supports both
synchronous (topic) and asynchronous (future benchmark step) executors via
``submit`` + ``drain_completions``. The legacy synchronous ``execute`` API is
retained as a concrete method so existing TopicExecutor callers and tests keep
working; async-only executors need not override it.

This module still imports only NumPy and the Python standard library plus the
frozen completion envelope defined in ``executors/completion.py``. It must not
import rclpy, ROS messages, benchmark runtime, sim backend or any concrete
executor. Concrete executors live in sibling modules and are wired through the
registry.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from .completion import ExecutionCompletion, ExecutionContext, ExecutionReceipt


class ActionExecutor(ABC):
    """Last-mile action output contract.

    An executor only answers "where does the already-decided single-step
    action go?". It must not select models, modify chunks, smooth actions,
    trigger inference, interpret suites/tasks or generate reports.

    completion-aware executor contract adds ``submit`` and ``drain_completions`` so that asynchronous
    executors can deliver completions without faking a synchronous ``execute``
    return value. The legacy ``execute`` method is retained as a concrete
    NotImplementedError for synchronous callers (e.g. TopicExecutor overrides
    it to publish exactly once and return True).
    """

    @property
    @abstractmethod
    def executor_type(self) -> str:
        """Return the exact SSOT executor type."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize resources and return whether initialization succeeded."""

    @abstractmethod
    def submit(
        self,
        action: np.ndarray,
        context: ExecutionContext,
    ) -> ExecutionReceipt:
        """Submit one already-selected action for execution.

        The dispatcher creates a fresh ``ExecutionContext`` (with a new
        correlation_id) per submission. The returned receipt indicates whether
        the executor accepted the action. If the executor completes
        synchronously it may set ``immediate_completion`` on the receipt; in
        that case ``drain_completions`` must not return the same completion
        again.
        """

    @abstractmethod
    def drain_completions(self) -> tuple[ExecutionCompletion, ...]:
        """Return any completions that have arrived since the last drain.

        For synchronous executors this always returns an empty tuple. For
        asynchronous executors it returns completions enqueued by background
        callbacks (e.g. ROS Future done callbacks). The dispatcher control timer
        is the sole caller of this method; callbacks must not modify dispatcher
        state directly.
        """

    def execute(
        self,
        action: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Legacy synchronous API; async-only executors need not override.

        TopicExecutor overrides this to publish exactly once and return True.
        The dispatcher continuous path keeps calling this method so its
        observable behaviour is byte-for-byte identical to executor registry contract. The
        wait-for-feedback path uses ``submit`` instead.
        """
        raise NotImplementedError

    def invalidate_pending(self) -> None:
        """Invalidate local pending/completions without claiming remote cancellation.

        executor lifecycle isolation adds this concrete no-op lifecycle hook so the dispatcher can
        isolate local state after a timeout or reset without assuming the
        remote side has cancelled, observed or even received the in-flight
        request. Synchronous executors (e.g. TopicExecutor) have no pending
        state, so the base implementation is a no-op and returns ``None``;
        they inherit it unchanged. Asynchronous executors override it to bump
        a local generation counter and drop stale completions.

        The hook is synchronous, takes no arguments beyond ``self`` and always
        returns ``None``. It must not wait, spin, sleep, call
        ``Future.cancel()``, recreate clients, contact a remote service or
        claim that the remote action was cancelled. Repeated calls are safe.
        Only ``reset`` on the dispatcher (which calls this hook) and the
        timeout path are authorised callers.
        """
        return None

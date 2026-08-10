"""Asynchronous benchmark step transport executor.

This module implements the ``BenchmarkStepExecutor``: a generic ROS async
service client that sends one action to a resolved ``StepBenchmark`` endpoint
and converts the response into an immutable ``ExecutionCompletion``.

benchmark step transport boundary:
- This module does not select benchmarks, read YAML, or import any environment
  adapter. The ``step_service`` endpoint is an internal resolved string passed
  via the config dict.
- Only one submission may be in-flight at a time. Responses arrive via the
  ROS Future done callback and are drained by the dispatcher control timer.
- No timeout, no retry, no reset, no episode/summary logic is implemented here.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from rclpy.node import Node

from ibrobot_msgs.srv import StepBenchmark
from tensormsg.converter import TensorMsgConverter

from .base import ActionExecutor
from .completion import (
    CompletionStatus,
    ExecutionCompletion,
    ExecutionContext,
    ExecutionReceipt,
)

_ALLOWED_DTYPE_TYPES = (np.float32, np.float64, np.int32, np.int64)
_NATIVE_BYTEORDER_CHAR = "<" if np.little_endian else ">"
_TRANSPORT_SUBMIT_KEY = "transport.submit_monotonic_ns"
_TRANSPORT_COMPLETION_KEY = "transport.completion_monotonic_ns"
_TRANSPORT_LATENCY_KEY = "transport.round_trip_latency_ms"


@dataclass
class _PendingSubmission:
    """Internal record of a single in-flight submission."""

    token: int
    correlation_id: str
    episode_id: int | None
    submit_monotonic_ns: int
    generation: int


class BenchmarkStepExecutor(ActionExecutor):
    """Generic async ``StepBenchmark`` transport executor.

    Receives an already-resolved ``step_service`` endpoint string, an
    already-selected action, and an explicit ``ExecutionContext``. Sends the
    action via ``call_async`` exactly once and converts the Future response
    into an immutable ``ExecutionCompletion`` drained by the dispatcher.
    """

    def __init__(self, node: Node, config: dict[str, Any]):
        self._node = node
        self._config = config
        self._lock = threading.Lock()
        self._completions: deque[ExecutionCompletion] = deque()
        self._client = None
        self._initialized = False
        self._step_service: str | None = None
        self._pending_token_counter = 0
        self._pending: _PendingSubmission | None = None
        # executor lifecycle isolation: local generation counter. Bumped by invalidate_pending() so
        # that late callbacks whose captured generation is now stale are
        # ignored at both the claim and append stages. Submit captures the
        # current generation into the pending record and the done-callback
        # lambda; the callback checks it twice (before claim, before append).
        self._current_generation: int = 0

    @property
    def executor_type(self) -> str:
        return "benchmark"

    def initialize(self) -> bool:
        """Validate config and create the ``StepBenchmark`` client exactly once.

        Idempotent: a second call returns ``True`` without creating a second
        client. Does not block on service readiness. Only logs
        ``[IBROBOT_BENCHMARK_STEP][CLIENT_CREATED]``.
        """
        with self._lock:
            if self._initialized:
                return True
            step_service = self._config.get("step_service")
            if not self._is_valid_step_service(step_service):
                self._node.get_logger().error(f"[IBROBOT_BENCHMARK_STEP][CONFIG_INVALID] step_service={step_service!r}")
                return False
            try:
                self._client = self._node.create_client(StepBenchmark, step_service)
            except Exception as exc:
                self._node.get_logger().error(
                    f"[IBROBOT_BENCHMARK_STEP][CREATE_CLIENT_FAILED] service={step_service} err={exc}"
                )
                self._client = None
                return False
            self._step_service = step_service
            self._initialized = True
            self._node.get_logger().info(f"[IBROBOT_BENCHMARK_STEP][CLIENT_CREATED] service={step_service}")
            return True

    @staticmethod
    def _is_valid_step_service(value: object) -> bool:
        """Exact ``str`` type (no subclass), non-empty, no surrounding whitespace, starts with ``/``."""
        if type(value) is not str:
            return False
        if not value:
            return False
        if value.strip() != value:
            return False
        return value.startswith("/")

    def submit(
        self,
        action: np.ndarray,
        context: ExecutionContext,
    ) -> ExecutionReceipt:
        """Validate, reserve pending, encode, ``call_async`` once, register callback."""
        rejected = self._pre_validate(action, context)
        if rejected is not None:
            return rejected

        with self._lock:
            if self._pending is not None:
                return ExecutionReceipt(
                    correlation_id=context.correlation_id,
                    accepted=False,
                    message="pending submission exists",
                )
            self._pending_token_counter += 1
            token = self._pending_token_counter
            submit_monotonic_ns = time.monotonic_ns()
            captured_generation = self._current_generation
            self._pending = _PendingSubmission(
                token=token,
                correlation_id=context.correlation_id,
                episode_id=context.episode_id,
                submit_monotonic_ns=submit_monotonic_ns,
                generation=captured_generation,
            )

        try:
            request = StepBenchmark.Request()
            request.episode_id = context.episode_id
            request.expected_step_id = context.expected_step_id
            request.action = TensorMsgConverter.to_variant({"action": action})
        except Exception as exc:
            self._clear_pending_if_match(token, captured_generation)
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message=f"request encoding failed: {exc}",
            )

        try:
            future = self._client.call_async(request)
        except Exception as exc:
            self._clear_pending_if_match(token, captured_generation)
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message=f"call_async failed: {exc}",
            )

        try:
            future.add_done_callback(lambda f, t=token, g=captured_generation: self._on_done(f, t, g))
        except Exception as exc:
            completion_monotonic_ns = time.monotonic_ns()
            with self._lock:
                # Only clear pending and enqueue UNCERTAIN if this generation
                # is still current. A concurrent invalidate/reset must not be
                # disturbed, and a stale UNCERTAIN must not pollute a fresh
                # generation's completion deque.
                if (
                    self._pending is not None
                    and self._pending.token == token
                    and self._pending.generation == captured_generation
                ):
                    self._pending = None
                    self._completions.append(
                        ExecutionCompletion(
                            correlation_id=context.correlation_id,
                            status=CompletionStatus.UNCERTAIN,
                            episode_id=context.episode_id,
                            step_id=None,
                            message=f"callback registration failed: {exc}",
                            details=self._transport_details(submit_monotonic_ns, completion_monotonic_ns),
                        )
                    )
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=True,
                message=f"accepted; callback registration failed: {exc}",
            )

        return ExecutionReceipt(
            correlation_id=context.correlation_id,
            accepted=True,
            message="accepted",
        )

    def drain_completions(self) -> tuple[ExecutionCompletion, ...]:
        """Snapshot and clear the completion deque under lock."""
        with self._lock:
            result = tuple(self._completions)
            self._completions.clear()
            return result

    def invalidate_pending(self) -> None:
        """Invalidate local pending and queued completions via generation bump.

        executor lifecycle isolation lifecycle hook. Atomically under the same lock:

        - increment ``_current_generation`` so any in-flight or late callback
          whose captured generation is now stale is ignored at both the claim
          and append stages of ``_on_done``;
        - clear the pending submission (if any);
        - clear the completion deque (drops any queued-but-undrained
          completions so they cannot pollute a fresh generation).

        Does NOT call ``Future.cancel()``, wait, spin, sleep, contact a
        remote service, recreate the client, or claim the remote action was
        cancelled. Repeated calls are safe: each call bumps the generation
        again and clears the (already empty) pending/deque.

        Logs the exact marker::

            [IBROBOT_BENCHMARK_STEP][PENDING_INVALIDATED] generation=<new value>
        """
        with self._lock:
            self._current_generation += 1
            self._pending = None
            self._completions.clear()
            new_generation = self._current_generation
        self._node.get_logger().info(f"[IBROBOT_BENCHMARK_STEP][PENDING_INVALIDATED] generation={new_generation}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear_pending_if_match(self, token: int, generation: int) -> None:
        """Clear pending only if it matches the given token AND generation.

        This prevents a concurrent ``invalidate_pending``/reset from causing
        an exception cleanup path (encoding/call_async/callback-registration
        failure) to wipe a NEW pending that was created after invalidation.
        If the current pending belongs to a different token or a different
        (newer) generation, it is left untouched.
        """
        with self._lock:
            if self._pending is not None and self._pending.token == token and self._pending.generation == generation:
                self._pending = None

    def _pre_validate(
        self,
        action: np.ndarray,
        context: ExecutionContext,
    ) -> ExecutionReceipt | None:
        """Return a rejected receipt if any pre-send check fails, else ``None``."""
        if not self._initialized:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="executor not initialized",
            )

        if type(context) is not ExecutionContext:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="context must be an ExecutionContext",
            )

        if context.episode_id is None:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="context.episode_id is required",
            )

        if context.expected_step_id is None:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="context.expected_step_id is required",
            )

        if type(action) is not np.ndarray:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message=f"action must be np.ndarray, got {type(action).__name__}",
            )

        if action.ndim != 1:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message=f"action must be 1-D, got ndim={action.ndim}",
            )

        if action.size == 0:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="action must be non-empty",
            )

        if action.dtype.type not in _ALLOWED_DTYPE_TYPES:
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message=f"action dtype {action.dtype} not in allowed {_ALLOWED_DTYPE_TYPES}",
            )

        if action.dtype.byteorder not in ("=", "|", _NATIVE_BYTEORDER_CHAR):
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message=f"action dtype byte order {action.dtype.byteorder!r} is not native",
            )

        if np.issubdtype(action.dtype, np.floating) and not bool(np.all(np.isfinite(action))):
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="action contains NaN or Inf",
            )

        if not self._client.service_is_ready():
            return ExecutionReceipt(
                correlation_id=context.correlation_id,
                accepted=False,
                message="step service is not ready",
            )

        return None

    def _on_done(self, future, pending_token: int, pending_generation: int) -> None:
        """Future done callback: claim pending, build completion, enqueue.

        Two-stage generation check:

        1. Before claiming pending: if the captured generation no longer
           matches ``_current_generation``, the callback is stale (invalidate
           or reset happened after submit). Log the stale marker and return
           without claiming or enqueuing.
        2. After response conversion, before appending the completion: if the
           generation advanced during the lock-free conversion window, log the
           stale marker and return without appending.

        A generation match with no pending means a duplicate callback in the
        same generation; silently ignore (benchmark step transport frozen semantics). Only
        generation-based staleness produces the explicit stale marker.
        """
        with self._lock:
            # Stage 1: check generation before claiming pending.
            if pending_generation != self._current_generation:
                self._node.get_logger().warn(
                    f"[IBROBOT_BENCHMARK_STEP][STALE_CALLBACK] token={pending_token} "
                    f"generation={pending_generation} current_generation={self._current_generation}"
                )
                return
            # Generation matches but pending is gone: duplicate callback in
            # the same generation. Silently ignore (benchmark step transport frozen semantics).
            if self._pending is None:
                return
            # Token mismatch in the same generation: should not happen in
            # single-in-flight mode, but silently ignore to be safe.
            if self._pending.token != pending_token:
                return
            # Claim the pending.
            pending = self._pending
            self._pending = None
            submit_monotonic_ns = pending.submit_monotonic_ns
            completion_monotonic_ns = time.monotonic_ns()

        try:
            completion = self._map_future_to_completion(future, pending, submit_monotonic_ns, completion_monotonic_ns)
        except Exception as exc:
            completion = ExecutionCompletion(
                correlation_id=pending.correlation_id,
                status=CompletionStatus.UNCERTAIN,
                episode_id=pending.episode_id,
                step_id=None,
                message=f"callback conversion exception: {exc}",
                details=self._transport_details(submit_monotonic_ns, completion_monotonic_ns),
            )

        # Stage 2: re-check generation before appending completion.
        with self._lock:
            if pending_generation != self._current_generation:
                self._node.get_logger().warn(
                    f"[IBROBOT_BENCHMARK_STEP][STALE_CALLBACK] token={pending_token} "
                    f"generation={pending_generation} current_generation={self._current_generation}"
                )
                return
            self._completions.append(completion)

    def _map_future_to_completion(
        self,
        future,
        pending: _PendingSubmission,
        submit_monotonic_ns: int,
        completion_monotonic_ns: int,
    ) -> ExecutionCompletion:
        """Map a done Future into an immutable ``ExecutionCompletion``."""
        if future.cancelled():
            return ExecutionCompletion(
                correlation_id=pending.correlation_id,
                status=CompletionStatus.UNCERTAIN,
                episode_id=pending.episode_id,
                step_id=None,
                message="future cancelled",
                details=self._transport_details(submit_monotonic_ns, completion_monotonic_ns),
            )

        exception = future.exception()
        if exception is not None:
            return ExecutionCompletion(
                correlation_id=pending.correlation_id,
                status=CompletionStatus.UNCERTAIN,
                episode_id=pending.episode_id,
                step_id=None,
                message=f"future exception: {exception}",
                details=self._transport_details(submit_monotonic_ns, completion_monotonic_ns),
            )

        try:
            response = future.result()
        except Exception as exc:
            return ExecutionCompletion(
                correlation_id=pending.correlation_id,
                status=CompletionStatus.UNCERTAIN,
                episode_id=pending.episode_id,
                step_id=None,
                message=f"future.result() raised: {exc}",
                details=self._transport_details(submit_monotonic_ns, completion_monotonic_ns),
            )

        if response is None:
            return ExecutionCompletion(
                correlation_id=pending.correlation_id,
                status=CompletionStatus.UNCERTAIN,
                episode_id=pending.episode_id,
                step_id=None,
                message="future result is None",
                details=self._transport_details(submit_monotonic_ns, completion_monotonic_ns),
            )

        if response.success:
            return self._map_success_response(response, pending, submit_monotonic_ns, completion_monotonic_ns)
        return self._map_failure_response(response, pending, submit_monotonic_ns, completion_monotonic_ns)

    def _map_success_response(
        self,
        response,
        pending: _PendingSubmission,
        submit_monotonic_ns: int,
        completion_monotonic_ns: int,
    ) -> ExecutionCompletion:
        """Map a ``success=True`` response into a COMPLETED or UNCERTAIN completion."""
        sec = int(response.obs_timestamp.sec)
        nanosec = int(response.obs_timestamp.nanosec)
        if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
            return self._uncertain_from_response(
                response,
                pending,
                submit_monotonic_ns,
                completion_monotonic_ns,
                message=f"invalid obs_timestamp sec={sec} nanosec={nanosec}",
            )
        timestamp_ns = sec * 1_000_000_000 + nanosec
        if timestamp_ns <= 0:
            return self._uncertain_from_response(
                response,
                pending,
                submit_monotonic_ns,
                completion_monotonic_ns,
                message="obs_timestamp computes to zero",
            )

        for json_field in ("standard_metrics_json", "native_metrics_json", "info_json"):
            raw = getattr(response, json_field)
            try:
                parsed = json.loads(raw)
            except Exception:
                return self._uncertain_from_response(
                    response,
                    pending,
                    submit_monotonic_ns,
                    completion_monotonic_ns,
                    message=f"{json_field} is not valid JSON",
                )
            if not isinstance(parsed, dict):
                return self._uncertain_from_response(
                    response,
                    pending,
                    submit_monotonic_ns,
                    completion_monotonic_ns,
                    message=f"{json_field} is not a JSON object",
                )

        return ExecutionCompletion(
            correlation_id=pending.correlation_id,
            status=CompletionStatus.COMPLETED,
            observation_timestamp_ns=timestamp_ns,
            episode_id=pending.episode_id,
            step_id=int(response.step_id),
            message=str(response.message),
            details=self._response_details(response, submit_monotonic_ns, completion_monotonic_ns),
        )

    def _map_failure_response(
        self,
        response,
        pending: _PendingSubmission,
        submit_monotonic_ns: int,
        completion_monotonic_ns: int,
    ) -> ExecutionCompletion:
        """Map a ``success=False`` response into a FAILED completion."""
        return ExecutionCompletion(
            correlation_id=pending.correlation_id,
            status=CompletionStatus.FAILED,
            observation_timestamp_ns=None,
            episode_id=pending.episode_id,
            step_id=int(response.step_id),
            message=str(response.message),
            details=self._response_details(response, submit_monotonic_ns, completion_monotonic_ns),
        )

    def _uncertain_from_response(
        self,
        response,
        pending: _PendingSubmission,
        submit_monotonic_ns: int,
        completion_monotonic_ns: int,
        *,
        message: str,
    ) -> ExecutionCompletion:
        """Build an UNCERTAIN completion that preserves raw response details."""
        return ExecutionCompletion(
            correlation_id=pending.correlation_id,
            status=CompletionStatus.UNCERTAIN,
            observation_timestamp_ns=None,
            episode_id=pending.episode_id,
            step_id=int(response.step_id),
            message=message,
            details=self._response_details(response, submit_monotonic_ns, completion_monotonic_ns),
        )

    @staticmethod
    def _transport_details(
        submit_monotonic_ns: int,
        completion_monotonic_ns: int,
    ) -> dict[str, Any]:
        return {
            _TRANSPORT_SUBMIT_KEY: int(submit_monotonic_ns),
            _TRANSPORT_COMPLETION_KEY: int(completion_monotonic_ns),
            _TRANSPORT_LATENCY_KEY: (completion_monotonic_ns - submit_monotonic_ns) / 1_000_000.0,
        }

    @staticmethod
    def _response_details(
        response,
        submit_monotonic_ns: int,
        completion_monotonic_ns: int,
    ) -> dict[str, Any]:
        details = BenchmarkStepExecutor._transport_details(submit_monotonic_ns, completion_monotonic_ns)
        details["has_reward"] = bool(response.has_reward)
        details["reward"] = float(response.reward)
        details["terminated"] = bool(response.terminated)
        details["truncated"] = bool(response.truncated)
        details["has_is_success"] = bool(response.has_is_success)
        details["is_success"] = bool(response.is_success)
        details["standard_metrics_json"] = str(response.standard_metrics_json)
        details["native_metrics_json"] = str(response.native_metrics_json)
        details["info_json"] = str(response.info_json)
        return details

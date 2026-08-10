"""Provider-independent atomic observation routing.

The router owns delivery ordering and Episode transaction isolation, but it
knows nothing about ROS, RTP, benchmark providers, or inference internals.
Concrete routes prepare transport-specific payloads without side effects and
commit them only after every route has prepared successfully.
"""

from __future__ import annotations

import contextlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from benchmark_runtime.io_descriptor import ObservationBatch


class ObservationRouterError(RuntimeError):
    """Base error for observation routing failures."""


class ObservationPrepareError(ObservationRouterError):
    """Raised when one route cannot prepare its payload."""

    def __init__(self, route_key: str, message: str) -> None:
        super().__init__(f"route {route_key!r} prepare failed: {message}")
        self.route_key = route_key
        self.message = message


class ObservationCommitError(ObservationRouterError):
    """Raised when transport commit fails, carrying the uncertain receipt."""

    def __init__(self, receipt: DeliveryReceipt) -> None:
        failure = receipt.failure_route or "unknown"
        super().__init__(f"observation delivery failed at route {failure!r}: {receipt.error}")
        self.receipt = receipt


@dataclass(frozen=True, slots=True)
class DeliveryContext:
    """Environment-owned delivery timestamp shared by every route."""

    timestamp_sec: int
    timestamp_nanosec: int

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_sec, int) or isinstance(self.timestamp_sec, bool) or self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be a non-negative int")
        if (
            not isinstance(self.timestamp_nanosec, int)
            or isinstance(self.timestamp_nanosec, bool)
            or not 0 <= self.timestamp_nanosec < 1_000_000_000
        ):
            raise ValueError("timestamp_nanosec must be an int in [0, 1000000000)")
        if self.timestamp_sec == 0 and self.timestamp_nanosec == 0:
            raise ValueError("delivery timestamp must be non-zero")


@dataclass(frozen=True, slots=True)
class PreparedRoute:
    """One side-effect-free transport payload prepared for commit."""

    key: str
    transport: str
    payload: Any

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("prepared route key must be a non-empty string")
        if not isinstance(self.transport, str) or not self.transport.strip():
            raise ValueError("prepared route transport must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RouteDeliveryReceipt:
    """Commit result for one route."""

    key: str
    transport: str
    status: str
    committed_monotonic_ns: int | None = None
    prepare_start_monotonic_ns: int | None = None
    prepare_end_monotonic_ns: int | None = None
    commit_start_monotonic_ns: int | None = None
    commit_end_monotonic_ns: int | None = None
    performance: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    VALID_STATUSES = frozenset({"committed", "failed", "skipped"})

    def __post_init__(self) -> None:
        if self.status not in self.VALID_STATUSES:
            raise ValueError(f"invalid route delivery status {self.status!r}")
        if self.status == "committed" and self.committed_monotonic_ns is None:
            raise ValueError("committed route receipt requires committed_monotonic_ns")
        if self.status == "failed" and not self.error:
            raise ValueError("failed route receipt requires error")
        object.__setattr__(self, "performance", MappingProxyType(dict(self.performance)))


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """Structured delivery result for a complete canonical batch."""

    episode_transaction_id: str
    sequence_id: int
    capture_timestamp_ns: int
    clock_domain: str
    delivery_timestamp_sec: int
    delivery_timestamp_nanosec: int
    status: str
    routes: tuple[RouteDeliveryReceipt, ...]
    router_prepare_start_monotonic_ns: int | None = None
    router_prepare_end_monotonic_ns: int | None = None
    router_commit_start_monotonic_ns: int | None = None
    router_commit_end_monotonic_ns: int | None = None
    failure_route: str | None = None
    error: str | None = None

    VALID_STATUSES = frozenset({"committed", "failed", "cancelled"})

    def __post_init__(self) -> None:
        if self.status not in self.VALID_STATUSES:
            raise ValueError(f"invalid delivery status {self.status!r}")
        if self.status == "failed" and (not self.failure_route or not self.error):
            raise ValueError("failed delivery receipt requires failure_route and error")

    @property
    def committed_keys(self) -> tuple[str, ...]:
        return tuple(route.key for route in self.routes if route.status == "committed")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for reports and wire metadata."""
        return {
            "episode_transaction_id": self.episode_transaction_id,
            "sequence_id": self.sequence_id,
            "capture_timestamp_ns": self.capture_timestamp_ns,
            "clock_domain": self.clock_domain,
            "delivery_timestamp_sec": self.delivery_timestamp_sec,
            "delivery_timestamp_nanosec": self.delivery_timestamp_nanosec,
            "status": self.status,
            "failure_route": self.failure_route,
            "error": self.error,
            "routes": [
                {
                    "key": route.key,
                    "transport": route.transport,
                    "status": route.status,
                    "committed_monotonic_ns": route.committed_monotonic_ns,
                    "prepare_start_monotonic_ns": route.prepare_start_monotonic_ns,
                    "prepare_end_monotonic_ns": route.prepare_end_monotonic_ns,
                    "commit_start_monotonic_ns": route.commit_start_monotonic_ns,
                    "commit_end_monotonic_ns": route.commit_end_monotonic_ns,
                    "performance": dict(route.performance),
                    "error": route.error,
                }
                for route in self.routes
            ],
            "router_timing": {
                "clock_domain": "monotonic",
                "prepare_start_monotonic_ns": self.router_prepare_start_monotonic_ns,
                "prepare_end_monotonic_ns": self.router_prepare_end_monotonic_ns,
                "commit_start_monotonic_ns": self.router_commit_start_monotonic_ns,
                "commit_end_monotonic_ns": self.router_commit_end_monotonic_ns,
            },
        }


class ObservationSink(ABC):
    """Provider-neutral destination for one canonical observation.

    The sink contract deliberately combines the identity metadata needed by the
    router (``key``, ``transport`` and ``optional``) with the transactional
    lifecycle used by every destination (``prepare`` -> ``commit`` or
    ``cancel`` -> ``close``).  ``transport`` is an opaque receipt label; this
    interface never interprets a protocol name or owns a transport.

    A sink's :meth:`prepare` must be side-effect free from the caller's point
    of view.  :meth:`commit` means *admitted to the owned transport*, not that
    an asynchronous encoder, sender, or receiver has finished.  Any later
    progress is correlated through the returned performance mapping.
    """

    @property
    @abstractmethod
    def key(self) -> str:
        """Canonical observation key owned by this sink."""

    @property
    @abstractmethod
    def transport(self) -> str:
        """Stable transport name, for example ``dds`` or ``rtp``."""

    @property
    def optional(self) -> bool:
        """Whether this sink may be absent from a batch."""
        return False

    @abstractmethod
    def prepare(self, batch: ObservationBatch, context: DeliveryContext) -> PreparedRoute:
        """Validate and prepare without externally visible side effects."""

    @abstractmethod
    def commit(self, prepared: PreparedRoute) -> Mapping[str, Any] | None:
        """Commit one payload and optionally return transport performance data."""

    def cancel(self, prepared: PreparedRoute) -> None:
        """Release a prepared payload that will not be committed."""
        del prepared

    def close(self) -> None:
        """Release route-owned resources. Repeated calls must be safe."""
        return None


@dataclass(slots=True)
class PreparedBatch:
    """All routes prepared for one batch, ready for exactly-once commit."""

    batch: ObservationBatch
    context: DeliveryContext
    routes: tuple[PreparedRoute, ...]
    route_prepare_timings: Mapping[str, tuple[int, int]]
    router_prepare_start_monotonic_ns: int
    router_prepare_end_monotonic_ns: int
    _commit_fn: Callable[[PreparedBatch], DeliveryReceipt] = field(repr=False)
    _cancel_fn: Callable[[PreparedBatch], DeliveryReceipt] = field(repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _receipt: DeliveryReceipt | None = field(default=None, init=False, repr=False)

    def commit(self) -> DeliveryReceipt:
        """Commit once; repeated calls return the same successful receipt."""
        with self._lock:
            if self._receipt is not None:
                if self._receipt.status == "failed":
                    raise ObservationCommitError(self._receipt)
                if self._receipt.status == "cancelled":
                    raise ObservationRouterError("cancelled PreparedBatch cannot be committed")
                return self._receipt
            try:
                receipt = self._commit_fn(self)
            except ObservationCommitError as exc:
                self._receipt = exc.receipt
                raise
            self._receipt = receipt
            return receipt

    def cancel(self) -> DeliveryReceipt:
        """Cancel before commit; repeated cancellation is idempotent."""
        with self._lock:
            if self._receipt is None:
                self._receipt = self._cancel_fn(self)
            return self._receipt


# ``ObservationRoute`` was the name introduced in the earlier Phase 3 API.
# Keep it as an exact compatibility alias, rather than a second interface:
# the semantic contract is now explicitly named ``ObservationSink``.
ObservationRoute = ObservationSink


class ObservationRouter:
    """Order-preserving sink prepare/commit router with Episode isolation."""

    def __init__(self, routes: tuple[ObservationSink, ...] | list[ObservationSink]) -> None:
        ordered = tuple(routes)
        if not ordered:
            raise ValueError("ObservationRouter requires at least one route")
        seen: set[str] = set()
        for route in ordered:
            if not isinstance(route, ObservationSink):
                raise TypeError("routes must contain ObservationSink values")
            if route.key in seen:
                raise ValueError(f"duplicate observation route key {route.key!r}")
            seen.add(route.key)
        self._routes = ordered
        self._route_map = MappingProxyType({route.key: route for route in ordered})
        self._lock = threading.RLock()
        self._closed = False
        self._active_transaction_id: str | None = None
        self._last_committed_sequence: int | None = None
        self._last_receipt: DeliveryReceipt | None = None
        self._poisoned_transaction_id: str | None = None

    @property
    def route_keys(self) -> tuple[str, ...]:
        return tuple(route.key for route in self._routes)

    @property
    def last_receipt(self) -> DeliveryReceipt | None:
        with self._lock:
            return self._last_receipt

    def prepare(self, batch: ObservationBatch, context: DeliveryContext) -> PreparedBatch:
        """Prepare every route in contract order without committing any."""
        if not isinstance(batch, ObservationBatch):
            raise TypeError("batch must be an ObservationBatch")
        if not isinstance(context, DeliveryContext):
            raise TypeError("context must be a DeliveryContext")
        with self._lock:
            self._require_open()
            self._validate_identity(batch)

        router_prepare_start_monotonic_ns = time.monotonic_ns()
        prepared: list[PreparedRoute] = []
        route_prepare_timings: dict[str, tuple[int, int]] = {}
        try:
            for route in self._routes:
                if route.key not in batch:
                    if route.optional:
                        continue
                    raise ObservationPrepareError(route.key, "required observation is absent")
                try:
                    route_prepare_start_monotonic_ns = time.monotonic_ns()
                    item = route.prepare(batch, context)
                    route_prepare_end_monotonic_ns = time.monotonic_ns()
                except ObservationPrepareError:
                    raise
                except Exception as exc:
                    raise ObservationPrepareError(route.key, f"{type(exc).__name__}: {exc}") from exc
                if item.key != route.key or item.transport != route.transport:
                    raise ObservationPrepareError(route.key, "prepared route identity does not match route")
                prepared.append(item)
                route_prepare_timings[route.key] = (
                    route_prepare_start_monotonic_ns,
                    route_prepare_end_monotonic_ns,
                )
        except Exception:
            self._cancel_prepared(tuple(prepared))
            raise

        router_prepare_end_monotonic_ns = time.monotonic_ns()
        return PreparedBatch(
            batch=batch,
            context=context,
            routes=tuple(prepared),
            route_prepare_timings=MappingProxyType(route_prepare_timings),
            router_prepare_start_monotonic_ns=router_prepare_start_monotonic_ns,
            router_prepare_end_monotonic_ns=router_prepare_end_monotonic_ns,
            _commit_fn=self._commit,
            _cancel_fn=self._cancel,
        )

    def _commit(self, prepared: PreparedBatch) -> DeliveryReceipt:
        route_receipts: list[RouteDeliveryReceipt] = []
        router_commit_start_monotonic_ns = time.monotonic_ns()
        with self._lock:
            self._require_open()
            self._validate_identity(prepared.batch)
            for item in prepared.routes:
                route = self._route_map[item.key]
                route_commit_start_monotonic_ns = time.monotonic_ns()
                try:
                    commit_performance = route.commit(item)
                except Exception as exc:
                    route_commit_end_monotonic_ns = time.monotonic_ns()
                    error = f"{type(exc).__name__}: {exc}"
                    route_receipts.append(
                        RouteDeliveryReceipt(
                            item.key,
                            item.transport,
                            "failed",
                            prepare_start_monotonic_ns=prepared.route_prepare_timings[item.key][0],
                            prepare_end_monotonic_ns=prepared.route_prepare_timings[item.key][1],
                            commit_start_monotonic_ns=route_commit_start_monotonic_ns,
                            commit_end_monotonic_ns=route_commit_end_monotonic_ns,
                            error=error,
                        )
                    )
                    remaining = prepared.routes[len(route_receipts) :]
                    self._cancel_uncommitted(remaining)
                    route_receipts.extend(
                        RouteDeliveryReceipt(remaining_item.key, remaining_item.transport, "skipped")
                        for remaining_item in remaining
                    )
                    receipt = self._receipt(
                        prepared,
                        status="failed",
                        routes=tuple(route_receipts),
                        failure_route=item.key,
                        error=error,
                        router_commit_start_monotonic_ns=router_commit_start_monotonic_ns,
                        router_commit_end_monotonic_ns=route_commit_end_monotonic_ns,
                    )
                    self._last_receipt = receipt
                    self._poisoned_transaction_id = prepared.batch.episode_transaction_id
                    raise ObservationCommitError(receipt) from exc
                route_commit_end_monotonic_ns = time.monotonic_ns()
                route_performance = {} if commit_performance is None else commit_performance
                route_receipts.append(
                    RouteDeliveryReceipt(
                        item.key,
                        item.transport,
                        "committed",
                        committed_monotonic_ns=time.monotonic_ns(),
                        prepare_start_monotonic_ns=prepared.route_prepare_timings[item.key][0],
                        prepare_end_monotonic_ns=prepared.route_prepare_timings[item.key][1],
                        commit_start_monotonic_ns=route_commit_start_monotonic_ns,
                        commit_end_monotonic_ns=route_commit_end_monotonic_ns,
                        performance=route_performance,
                    )
                )

            router_commit_end_monotonic_ns = time.monotonic_ns()
            receipt = self._receipt(
                prepared,
                status="committed",
                routes=tuple(route_receipts),
                router_commit_start_monotonic_ns=router_commit_start_monotonic_ns,
                router_commit_end_monotonic_ns=router_commit_end_monotonic_ns,
            )
            self._active_transaction_id = prepared.batch.episode_transaction_id
            self._last_committed_sequence = prepared.batch.sequence_id
            self._last_receipt = receipt
            self._poisoned_transaction_id = None
            return receipt

    def _cancel(self, prepared: PreparedBatch) -> DeliveryReceipt:
        self._cancel_prepared(prepared.routes)
        return self._receipt(prepared, status="cancelled", routes=())

    def _receipt(
        self,
        prepared: PreparedBatch,
        *,
        status: str,
        routes: tuple[RouteDeliveryReceipt, ...],
        failure_route: str | None = None,
        error: str | None = None,
        router_commit_start_monotonic_ns: int | None = None,
        router_commit_end_monotonic_ns: int | None = None,
    ) -> DeliveryReceipt:
        return DeliveryReceipt(
            episode_transaction_id=prepared.batch.episode_transaction_id,
            sequence_id=prepared.batch.sequence_id,
            capture_timestamp_ns=prepared.batch.capture_timestamp_ns,
            clock_domain=prepared.batch.clock_domain,
            delivery_timestamp_sec=prepared.context.timestamp_sec,
            delivery_timestamp_nanosec=prepared.context.timestamp_nanosec,
            status=status,
            routes=routes,
            router_prepare_start_monotonic_ns=prepared.router_prepare_start_monotonic_ns,
            router_prepare_end_monotonic_ns=prepared.router_prepare_end_monotonic_ns,
            router_commit_start_monotonic_ns=router_commit_start_monotonic_ns,
            router_commit_end_monotonic_ns=router_commit_end_monotonic_ns,
            failure_route=failure_route,
            error=error,
        )

    def _validate_identity(self, batch: ObservationBatch) -> None:
        if batch.episode_transaction_id == self._poisoned_transaction_id:
            raise ObservationRouterError(
                "observation transaction is transport-poisoned; a new reset transaction is required"
            )
        if batch.sequence_id == 0:
            if (
                self._active_transaction_id == batch.episode_transaction_id
                and self._last_committed_sequence is not None
            ):
                raise ObservationRouterError("duplicate reset batch for active Episode transaction")
            return
        if self._active_transaction_id is None or self._last_committed_sequence is None:
            raise ObservationRouterError("step batch has no committed reset transaction")
        if batch.episode_transaction_id != self._active_transaction_id:
            raise ObservationRouterError(
                "stale or cross-Episode observation transaction: "
                f"expected {self._active_transaction_id!r}, got {batch.episode_transaction_id!r}"
            )
        expected = self._last_committed_sequence + 1
        if batch.sequence_id != expected:
            raise ObservationRouterError(f"observation sequence mismatch: expected {expected}, got {batch.sequence_id}")

    def _cancel_prepared(self, prepared: tuple[PreparedRoute, ...]) -> None:
        for item in reversed(prepared):
            route = self._route_map.get(item.key)
            if route is not None:
                with contextlib.suppress(Exception):
                    route.cancel(item)

    def _cancel_uncommitted(self, prepared: tuple[PreparedRoute, ...]) -> None:
        self._cancel_prepared(prepared)

    def _require_open(self) -> None:
        if self._closed:
            raise ObservationRouterError("ObservationRouter is closed")

    def close(self) -> None:
        """Close every route once. Repeated calls are bounded and idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for route in reversed(self._routes):
                with contextlib.suppress(Exception):
                    route.close()

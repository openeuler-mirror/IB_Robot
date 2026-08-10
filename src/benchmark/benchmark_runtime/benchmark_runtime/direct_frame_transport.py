"""Thin Benchmark observation route for the public IB-Robot FrameIngress."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmark_runtime.io_descriptor import ObservationBatch
from benchmark_runtime.observation_router import DeliveryContext, ObservationSink, PreparedRoute
from observation_transport.frame_ingress import (
    FrameAdmissionDisposition,
    FrameIngress,
    FrameIngressError,
)

DIRECT_FRAME_COMMIT_TAG = "[IBROBOT_BENCHMARK][DIRECT_FRAME_ADMISSION]"


@dataclass(frozen=True, slots=True)
class _PreparedFrame:
    frame: np.ndarray
    capture_timestamp_ns: int
    receive_timestamp_ns: int
    provider_capture_timestamp_ns: int
    prepare_start_monotonic_ns: int
    prepare_end_monotonic_ns: int


class FrameIngressObservationSink(ObservationSink):
    """Adapt a canonical NumPy observation to ``FrameIngress.submit_frame``."""

    def __init__(
        self,
        ingress: FrameIngress,
        key: str,
        *,
        optional: bool = False,
        logger: Any | None = None,
    ) -> None:
        if key not in ingress.observation_keys:
            raise ValueError(f"no frame ingress stream configured for {key!r}")
        self._ingress = ingress
        self._key = key
        self._optional = optional
        self._logger = logger
        self._closed = False

    @property
    def key(self) -> str:
        return self._key

    @property
    def transport(self) -> str:
        return "rtp"

    @property
    def optional(self) -> bool:
        return self._optional

    def prepare(self, batch: ObservationBatch, context: DeliveryContext) -> PreparedRoute:
        if self._closed:
            raise RuntimeError("direct-frame observation route is closed")
        start = time.monotonic_ns()
        payload = batch[self.key]
        if not isinstance(payload, np.ndarray):
            raise TypeError(f"direct-frame payload for {self.key!r} must be numpy.ndarray")
        capture_ns = context.timestamp_sec * 1_000_000_000 + context.timestamp_nanosec
        prepared = _PreparedFrame(
            payload,
            capture_ns,
            time.time_ns(),
            batch.capture_timestamp_ns,
            start,
            time.monotonic_ns(),
        )
        return PreparedRoute(self.key, self.transport, prepared)

    def commit(self, prepared: PreparedRoute) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("direct-frame observation route is closed")
        if prepared.key != self.key or prepared.transport != self.transport:
            raise ValueError("prepared direct-frame route identity mismatch")
        payload = prepared.payload
        if not isinstance(payload, _PreparedFrame):
            raise TypeError("prepared direct-frame payload type mismatch")
        admission_start = time.monotonic_ns()
        receipt = self._ingress.submit_frame(
            self.key,
            payload.frame,
            capture_timestamp_ns=payload.capture_timestamp_ns,
            receive_timestamp_ns=payload.receive_timestamp_ns,
            pixel_format="rgb24",
        )
        admission_end = time.monotonic_ns()
        if receipt.disposition is not FrameAdmissionDisposition.ACCEPTED:
            raise FrameIngressError(
                receipt.reason,
                f"frame admission rejected: {receipt.reason}",
                observation_key=receipt.observation_key,
                stream_id=receipt.stream_id,
                recoverable=True,
            )
        performance = {
            "provider_capture_timestamp_ns": payload.provider_capture_timestamp_ns,
            "capture_timestamp_ns": receipt.capture_timestamp_ns,
            "prepare_start_monotonic_ns": payload.prepare_start_monotonic_ns,
            "prepare_end_monotonic_ns": payload.prepare_end_monotonic_ns,
            "admission_start_monotonic_ns": admission_start,
            "admission_monotonic_ns": receipt.admission_monotonic_ns,
            "admission_end_monotonic_ns": admission_end,
            "prepare_latency_ns": payload.prepare_end_monotonic_ns - payload.prepare_start_monotonic_ns,
            "admission_latency_ns": admission_end - admission_start,
            "admission_id": receipt.admission_id,
            "transport_correlation_id": receipt.admission_id,
            "session_generation": receipt.session_generation,
            "queue_depth": receipt.queue_depth,
            "disposition": receipt.disposition.value,
        }
        sequence = int(receipt.admission_id.rsplit(":", 1)[-1])
        if self._logger is not None and (sequence == 1 or sequence % 25 == 0):
            self._logger.info(
                f"{DIRECT_FRAME_COMMIT_TAG} observation={receipt.observation_key} stream={receipt.stream_id} "
                f"capture_ns={receipt.capture_timestamp_ns} admission_id={receipt.admission_id} "
                f"admission_latency_ns={performance['admission_latency_ns']} queue_depth={receipt.queue_depth}"
            )
        return performance

    def close(self) -> None:
        self._closed = True


# Compatibility name from the route-oriented Phase 4 API.  This is an exact
# alias; there is no parallel transport implementation.
DirectFrameObservationRoute = FrameIngressObservationSink

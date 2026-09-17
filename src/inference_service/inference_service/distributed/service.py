"""Cloud request execution with immediate structured distributed results."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, replace

from inference_service.distributed.runtime import CloudBackendRuntime
from inference_service.distributed.session import CloudSession, DistributedProtocolError
from inference_service.distributed.types import (
    DistributedRequest,
    DistributedResult,
    Operation,
    PipelineIdentity,
    PipelineStatus,
    StructuredError,
    UnsupportedDistributedRuntimeError,
    structured_error_from_exception,
)
from inference_service.distributed.video_streams import VideoStreamNegotiator


class _StreamManager:
    negotiator: VideoStreamNegotiator

    def reset_session(self, session_id: str, session_generation: int) -> None: ...

    def assemble_inputs(self, target_timestamp_ns: int, *, now_ns: int | None = None) -> dict[str, object]: ...

    def statuses(self) -> tuple[object, ...]: ...

    def selection_anchor_ns(self) -> int: ...

    def state_alignment_tolerance_ns(self) -> int: ...

    def record_state_alignment(self, delta_ns: int) -> None: ...

    def close(self) -> None: ...


def align_inputs_to_selection(
    inputs: dict[str, object],
    *,
    aligned_timestamps_ns: tuple[int, ...],
    aligned_tensors: tuple[Mapping[str, object], ...],
    selected_capture_ns: int,
    tolerance_ns: int,
) -> tuple[dict[str, object], int]:
    """Replace tick-anchored inputs with the history entry matching the capture.

    The cloud can only select video frames that are already decoded, so the
    selected capture lags the request tick. The request carries a timestamped
    history of the small observations; the entry closest to the selected
    capture replaces the tick-anchored samples so the policy consumes one
    consistent instant. No entry within tolerance is a hard failure: the
    dispatcher retries instead of silently pairing mismatched instants.
    """
    if not aligned_timestamps_ns or selected_capture_ns <= 0:
        return inputs, 0
    best_index = min(
        range(len(aligned_timestamps_ns)),
        key=lambda index: abs(aligned_timestamps_ns[index] - selected_capture_ns),
    )
    delta_ns = abs(aligned_timestamps_ns[best_index] - selected_capture_ns)
    if tolerance_ns > 0 and delta_ns > tolerance_ns:
        raise DistributedProtocolError(
            StructuredError(
                code="state_alignment_unavailable",
                message=(
                    "no aligned observation history within "
                    f"{tolerance_ns / 1e6:.0f}ms of the selected capture "
                    f"(delta {delta_ns / 1e6:.1f}ms)"
                ),
                stage="backend",
                recoverable=True,
            )
        )
    aligned = dict(inputs)
    aligned.update(aligned_tensors[best_index])
    return aligned, delta_ns


class DistributedCloudService:
    def __init__(
        self,
        identity: PipelineIdentity,
        runtime: CloudBackendRuntime | None,
        *,
        startup_error: StructuredError | None = None,
        stream_negotiator: VideoStreamNegotiator | None = None,
        stream_manager: _StreamManager | None = None,
        runtime_interface: str = "policy",
        runtime_model_type: str = "",
    ) -> None:
        if runtime_interface != "policy":
            raise UnsupportedDistributedRuntimeError(runtime_interface, runtime_model_type)
        runtime_identity = getattr(runtime, "identity", None) if runtime is not None else None
        runtime_interface = getattr(runtime_identity, "interface", None)
        if runtime_interface is None and runtime is not None:
            runtime_interface = getattr(runtime, "interface", None)
        if runtime_interface is not None and runtime_interface != "policy":
            raise UnsupportedDistributedRuntimeError(
                runtime_interface,
                getattr(runtime_identity, "model_type", getattr(runtime, "model_type", "")),
            )
        if runtime is None and startup_error is None:
            raise ValueError("startup_error is required when the cloud runtime is unavailable")
        self.identity = identity
        self._lifecycle_lock = threading.RLock()
        self.runtime = runtime
        self.startup_error = startup_error
        if stream_manager is not None and stream_negotiator is not None:
            raise ValueError("provide stream_manager or stream_negotiator, not both")
        self.stream_manager = stream_manager
        self.stream_negotiator = stream_manager.negotiator if stream_manager is not None else stream_negotiator
        self.session = CloudSession(
            identity,
            request_stream_validator=(
                self.stream_negotiator.validate_request if self.stream_negotiator is not None else None
            ),
        )

    def observe_edge(self, status: PipelineStatus) -> PipelineStatus:
        with self._lifecycle_lock:
            if self.runtime is None:
                self.session.observe_edge(status, backend_ready=False)
                return self.status()
            _, backend_available, _ = self._runtime_status()
            self.session.observe_edge(
                status,
                backend_ready=backend_available,
                rollover_barrier=self._reset_stateful_runtime_for_rollover,
            )
            cloud_status = self.status()
            self._sync_stream_negotiator_session(cloud_status)
            return cloud_status

    def status(self) -> PipelineStatus:
        with self._lifecycle_lock:
            if self.runtime is None:
                status = self.session.status(
                    backend_ready=False,
                    backend_state="failed",
                    reset_supported=False,
                    cancellation_supported=False,
                )
                return replace(status, error=self.startup_error)
            _, backend_available, runtime_state = self._runtime_status()
            capabilities = self.runtime.capabilities
            return self.session.status(
                backend_ready=backend_available,
                backend_state=runtime_state,
                reset_supported=capabilities.resettable or not getattr(capabilities, "stateful", True),
                cancellation_supported=capabilities.supports_cancellation,
            )

    def handle(self, request: DistributedRequest) -> DistributedResult:
        request_start_monotonic_ns = time.monotonic_ns()
        stream_assembly_start_monotonic_ns = 0
        stream_assembly_end_monotonic_ns = 0
        inference_start_monotonic_ns = 0
        inference_end_monotonic_ns = 0
        transport_streams: tuple[dict[str, object], ...] = ()
        if self.runtime is None:
            result_monotonic_ns = time.monotonic_ns()
            return DistributedResult(
                operation=request.operation,
                pipeline_id=self.identity.pipeline_id,
                request_id=request.request_id,
                session_id=request.session_id,
                session_generation=request.session_generation,
                deployment_fingerprint=self.identity.deployment_fingerprint,
                success=False,
                performance=self._performance_payload(
                    request_start_monotonic_ns=request_start_monotonic_ns,
                    stream_assembly_start_monotonic_ns=0,
                    stream_assembly_end_monotonic_ns=0,
                    inference_start_monotonic_ns=0,
                    inference_end_monotonic_ns=0,
                    result_monotonic_ns=result_monotonic_ns,
                    transport_streams=(),
                ),
                backend_ready=False,
                backend_state="failed",
                target_request_id=request.target_request_id,
                error=self.startup_error,
            )
        try:
            with self.session.operation(request):
                if request.operation is Operation.INFER:
                    inputs = dict(request.inputs)
                    stream_assembly_start_monotonic_ns = time.monotonic_ns()
                    if self.stream_manager is not None:
                        inputs.update(
                            self.stream_manager.assemble_inputs(
                                request.observation_timestamp_ns,
                                now_ns=time.time_ns(),
                            )
                        )
                        if request.aligned_timestamps_ns:
                            inputs, alignment_delta_ns = align_inputs_to_selection(
                                inputs,
                                aligned_timestamps_ns=request.aligned_timestamps_ns,
                                aligned_tensors=request.aligned_tensors,
                                selected_capture_ns=self.stream_manager.selection_anchor_ns(),
                                tolerance_ns=self.stream_manager.state_alignment_tolerance_ns(),
                            )
                            self.stream_manager.record_state_alignment(alignment_delta_ns)
                    stream_assembly_end_monotonic_ns = time.monotonic_ns()
                    inference_start_monotonic_ns = time.monotonic_ns()
                    pipeline_result = self.runtime.infer(
                        request.request_id,
                        inputs,
                        prompt=request.prompt,
                        deadline=request.deadline,
                    )
                    action = pipeline_result.action
                    chunk_size = pipeline_result.actual_chunk_size
                    metadata = getattr(pipeline_result, "metadata", {})
                    raw_horizon = metadata.get("execution_horizon", 0) if isinstance(metadata, Mapping) else 0
                    if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, int):
                        raise ValueError("pipeline execution_horizon must be an integer")
                    execution_horizon = raw_horizon
                    if execution_horizon < 0 or execution_horizon > chunk_size:
                        raise ValueError("pipeline execution_horizon is outside the action chunk")
                    latency_ms = pipeline_result.backend_latency_ms
                    inference_end_monotonic_ns = time.monotonic_ns()
                    transport_streams = self._transport_performance()
                elif request.operation is Operation.RESET:
                    self.runtime.reset(deadline=request.deadline)
                    # A policy reset does not roll over the distributed session.
                    # Preserve negotiated descriptors, UDP sockets, timestamp
                    # mappings, and the reset observation already delivered by
                    # an external direct-frame producer. Session replacement is
                    # handled exclusively by observe_edge()/rollover.
                    action = None
                    chunk_size = 0
                    execution_horizon = 0
                    latency_ms = 0.0
                elif request.operation is Operation.CANCEL:
                    self.runtime.cancel(request.target_request_id, deadline=request.deadline)
                    action = None
                    chunk_size = 0
                    execution_horizon = 0
                    latency_ms = 0.0
                else:
                    raise ValueError(f"unsupported distributed operation {request.operation!r}")
        except Exception as exc:
            stage = "routing" if isinstance(exc, DistributedProtocolError) else self._operation_stage(request.operation)
            error = (
                exc.error if isinstance(exc, DistributedProtocolError) else structured_error_from_exception(exc, stage)
            )
            _, backend_available, runtime_state = self._runtime_status()
            result_monotonic_ns = time.monotonic_ns()
            return DistributedResult(
                operation=request.operation,
                pipeline_id=self.identity.pipeline_id,
                request_id=request.request_id,
                session_id=request.session_id,
                session_generation=request.session_generation,
                deployment_fingerprint=self.identity.deployment_fingerprint,
                success=False,
                performance=self._performance_payload(
                    request_start_monotonic_ns=request_start_monotonic_ns,
                    stream_assembly_start_monotonic_ns=stream_assembly_start_monotonic_ns,
                    stream_assembly_end_monotonic_ns=stream_assembly_end_monotonic_ns,
                    inference_start_monotonic_ns=inference_start_monotonic_ns,
                    inference_end_monotonic_ns=inference_end_monotonic_ns,
                    result_monotonic_ns=result_monotonic_ns,
                    transport_streams=transport_streams,
                ),
                backend_ready=backend_available,
                backend_state=runtime_state,
                target_request_id=request.target_request_id,
                error=error,
            )

        _, backend_available, runtime_state = self._runtime_status()
        result_monotonic_ns = time.monotonic_ns()
        return DistributedResult(
            operation=request.operation,
            pipeline_id=self.identity.pipeline_id,
            request_id=request.request_id,
            session_id=request.session_id,
            session_generation=request.session_generation,
            deployment_fingerprint=self.identity.deployment_fingerprint,
            success=True,
            action=action,
            actual_chunk_size=chunk_size,
            execution_horizon=execution_horizon,
            backend_latency_ms=latency_ms,
            performance=self._performance_payload(
                request_start_monotonic_ns=request_start_monotonic_ns,
                stream_assembly_start_monotonic_ns=stream_assembly_start_monotonic_ns,
                stream_assembly_end_monotonic_ns=stream_assembly_end_monotonic_ns,
                inference_start_monotonic_ns=inference_start_monotonic_ns,
                inference_end_monotonic_ns=inference_end_monotonic_ns,
                result_monotonic_ns=result_monotonic_ns,
                transport_streams=transport_streams,
            ),
            backend_ready=backend_available,
            backend_state=runtime_state,
            target_request_id=request.target_request_id,
        )

    @staticmethod
    def _performance_payload(
        *,
        request_start_monotonic_ns: int,
        stream_assembly_start_monotonic_ns: int,
        stream_assembly_end_monotonic_ns: int,
        inference_start_monotonic_ns: int,
        inference_end_monotonic_ns: int,
        result_monotonic_ns: int,
        transport_streams: tuple[dict[str, object], ...],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "clock_domain": "monotonic",
            "request_start_monotonic_ns": request_start_monotonic_ns,
            "stream_assembly_start_monotonic_ns": stream_assembly_start_monotonic_ns,
            "stream_assembly_end_monotonic_ns": stream_assembly_end_monotonic_ns,
            "inference_start_monotonic_ns": inference_start_monotonic_ns,
            "inference_end_monotonic_ns": inference_end_monotonic_ns,
            "result_monotonic_ns": result_monotonic_ns,
        }
        if transport_streams:
            payload["transport_streams"] = list(transport_streams)
        return payload

    def _transport_performance(self) -> tuple[dict[str, object], ...]:
        if self.stream_manager is None:
            return ()
        snapshots = []
        for status in self.stream_manager.statuses():
            snapshot = asdict(status)
            snapshot["clock_domain"] = "monotonic"
            encode_start = int(snapshot["encode_start_monotonic_ns"])
            encode_end = int(snapshot["encode_end_monotonic_ns"])
            send_start = int(snapshot["send_start_monotonic_ns"])
            send_end = int(snapshot["send_end_monotonic_ns"])
            receive = int(snapshot["receive_monotonic_ns"])
            decode_start = int(snapshot["decode_start_monotonic_ns"])
            decode_end = int(snapshot["decode_end_monotonic_ns"])
            if encode_end >= encode_start > 0:
                snapshot["encode_latency_ns"] = encode_end - encode_start
            if send_end >= send_start > 0:
                snapshot["send_latency_ns"] = send_end - send_start
            if decode_end >= decode_start > 0:
                snapshot["decode_latency_ns"] = decode_end - decode_start
            if (
                int(snapshot["mapping_capture_timestamp_ns"]) == int(snapshot["last_decoded_capture_timestamp_ns"])
                and receive >= send_end > 0
            ):
                snapshot["network_latency_ns"] = receive - send_end
            snapshots.append(snapshot)
        return tuple(snapshots)

    def close(self) -> None:
        error: Exception | None = None
        if self.stream_manager is not None:
            try:
                self.stream_manager.close()
            except Exception as exc:
                error = exc
        if self.runtime is not None:
            try:
                self.runtime.close()
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def _sync_stream_negotiator_session(self, status: PipelineStatus) -> None:
        if self.stream_negotiator is None or not status.session_id or status.session_generation < 1:
            return
        requirements = self.stream_negotiator.requirements
        if requirements.session_id != status.session_id or requirements.session_generation != status.session_generation:
            if self.stream_manager is not None:
                self.stream_manager.reset_session(status.session_id, status.session_generation)
            else:
                self.stream_negotiator.reset(status.session_id, status.session_generation)

    def _runtime_status(self) -> tuple[object, bool, str]:
        assert self.runtime is not None
        health = self.runtime.health()
        backend_health = getattr(health, "backend_health", health)
        runtime_state = health.state.value
        backend_available = bool(backend_health.ready) and runtime_state in {"ready", "resetting"}
        return health, backend_available, runtime_state

    def _reset_stateful_runtime_for_rollover(self) -> StructuredError | None:
        assert self.runtime is not None
        capabilities = self.runtime.capabilities
        if not getattr(capabilities, "stateful", True):
            return None
        if not capabilities.resettable:
            return StructuredError(
                code="session_reset_unsupported",
                message="stateful cloud backend cannot reset before session rollover",
                stage="reset",
            )
        try:
            self.runtime.reset()
        except Exception as exc:
            return structured_error_from_exception(exc, "reset")
        _, backend_available, runtime_state = self._runtime_status()
        if not backend_available:
            return StructuredError(
                code="session_reset_failed",
                message=f"cloud backend is not ready after session rollover reset ({runtime_state})",
                stage="reset",
            )
        return None

    @staticmethod
    def _operation_stage(operation: Operation) -> str:
        return {
            Operation.INFER: "backend",
            Operation.RESET: "reset",
            Operation.CANCEL: "cancel",
        }[operation]

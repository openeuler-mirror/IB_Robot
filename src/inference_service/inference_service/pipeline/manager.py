"""Atomic owner and router for multiple independent named pipelines."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from datetime import datetime
from types import MappingProxyType

from inference_service.backends import BackendCapabilities, InferenceRequest
from inference_service.pipeline.errors import PipelineManagerError, PipelineNotFoundError
from inference_service.pipeline.runtime import InferencePipeline
from inference_service.pipeline.types import PipelineDiagnostics, PipelineResult


class InferencePipelineManager:
    def __init__(self, pipelines: Iterable[InferencePipeline] = (), *, retry_pending_close: bool = False) -> None:
        owned: dict[str, InferencePipeline] = {}
        for pipeline in pipelines:
            if pipeline.pipeline_id in owned:
                raise PipelineManagerError(
                    f"duplicate pipeline ID {pipeline.pipeline_id!r}",
                    code="duplicate_pipeline_id",
                    pipeline_id=pipeline.pipeline_id,
                )
            owned[pipeline.pipeline_id] = pipeline
        self._pipelines = owned
        self._lock = threading.RLock()
        self._cleanup_lock = threading.RLock() if retry_pending_close else None
        self._unreleased_pipelines = set(owned) if retry_pending_close else None
        self._started = False
        self._starting = False
        self._closed = False

    @property
    def pipelines(self) -> Mapping[str, InferencePipeline]:
        return MappingProxyType(self._pipelines)

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise PipelineManagerError("pipeline manager is closed", code="manager_closed")
            if self._started or self._starting:
                raise PipelineManagerError("pipeline manager has already started", code="manager_already_started")
            self._starting = True

        failed_pipeline: str | None = None
        try:
            for pipeline in self._pipelines.values():
                failed_pipeline = pipeline.pipeline_id
                pipeline.load()
        except Exception as exc:
            cleanup_errors = self._close_owned_pipelines()
            with self._lock:
                self._starting = False
                self._closed = True
            raise PipelineManagerError(
                f"pipeline manager startup failed while loading {failed_pipeline!r}: {exc}",
                code="startup_failed",
                pipeline_id=failed_pipeline,
                details={"cleanup_errors": tuple(str(error) for error in cleanup_errors)},
            ) from exc
        else:
            with self._lock:
                self._starting = False
                if self._closed:
                    raise PipelineManagerError(
                        "pipeline manager startup was interrupted by shutdown",
                        code="startup_interrupted",
                    )
                self._started = True

    def infer(
        self,
        pipeline_id: str,
        request: InferenceRequest,
        *,
        control_inputs: Mapping[str, object] | None = None,
        capture_raw_action: bool = False,
    ) -> PipelineResult:
        self._require_started()
        return self._pipeline(pipeline_id).infer(
            request,
            control_inputs=control_inputs,
            capture_raw_action=capture_raw_action,
        )

    def submit_frame(self, pipeline_id: str, request: object, *, deadline: datetime | None = None):
        self._require_started()
        pipeline = self._pipeline(pipeline_id)
        submit = getattr(pipeline, "submit_frame", None)
        if not callable(submit):
            return None
        return submit(request, deadline=deadline)

    def reset(self, pipeline_id: str, deadline: datetime | None = None) -> None:
        self._require_started()
        self._pipeline(pipeline_id).reset(deadline)

    def cancel(self, pipeline_id: str, request_id: str, deadline: datetime | None = None) -> None:
        self._require_started()
        self._pipeline(pipeline_id).cancel(request_id, deadline)

    def capabilities(self, pipeline_id: str) -> BackendCapabilities:
        self._require_started()
        return self._pipeline(pipeline_id).capabilities

    def supports_priority_zero_deadline_admission(self, pipeline_id: str) -> bool:
        self._require_started()
        return self._pipeline(pipeline_id).supports_priority_zero_deadline_admission

    def runtime_handle(self, pipeline_id: str):
        """Return the migrated unified handle, if this pipeline has one."""

        self._require_started()
        return self._pipeline(pipeline_id).runtime_handle

    def health(self, pipeline_id: str) -> PipelineDiagnostics:
        self._require_started()
        return self._pipeline(pipeline_id).diagnostics()

    def diagnostics(self, pipeline_id: str | None = None) -> PipelineDiagnostics | Mapping[str, PipelineDiagnostics]:
        self._require_started()
        if pipeline_id is not None:
            return self._pipeline(pipeline_id).diagnostics()
        return MappingProxyType({name: pipeline.diagnostics() for name, pipeline in self._pipelines.items()})

    def close(self, pipeline_id: str | None = None) -> None:
        if pipeline_id is not None:
            self._require_started()
            self._pipeline(pipeline_id).close()
            return
        with self._lock:
            if self._closed and self._unreleased_pipelines is None:
                return
            self._closed = True
            self._started = False
        errors = self._close_owned_pipelines()
        if errors:
            raise PipelineManagerError(
                "pipeline manager shutdown encountered errors: " + "; ".join(str(error) for error in errors),
                code="shutdown_failed",
                details={"errors": tuple(str(error) for error in errors)},
            )

    def _pipeline(self, pipeline_id: str) -> InferencePipeline:
        try:
            return self._pipelines[pipeline_id]
        except KeyError as exc:
            raise PipelineNotFoundError(pipeline_id, tuple(self._pipelines)) from exc

    def _require_started(self) -> None:
        with self._lock:
            if not self._started or self._starting or self._closed:
                raise PipelineManagerError(
                    "pipeline manager is not ready",
                    code="manager_not_ready",
                    details={
                        "started": self._started,
                        "starting": self._starting,
                        "closed": self._closed,
                    },
                )

    def _close_owned_pipelines(self) -> tuple[Exception, ...]:
        errors: list[Exception] = []
        with self._cleanup_lock or nullcontext():
            for pipeline in reversed(tuple(self._pipelines.values())):
                if self._unreleased_pipelines is not None and pipeline.pipeline_id not in self._unreleased_pipelines:
                    continue
                try:
                    pipeline.close()
                except Exception as exc:
                    errors.append(exc)
                    if getattr(exc, "close_pending", False):
                        continue
                if self._unreleased_pipelines is not None:
                    self._unreleased_pipelines.discard(pipeline.pipeline_id)
        return tuple(errors)

"""Native model runtime resources used by the unified runtime factory.

The classes in this package are backend/model resources, not lifecycle owners.
``ModelRuntimeHandle`` owns admission, deadlines, cancellation evidence,
recovery, and public lifecycle state.  A model resource only loads vendor
objects, executes one request, and releases those objects.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from inference_manifest import CompiledDeployment, SemanticTensor
from inference_service.backends.errors import (
    BackendCapabilityError,
    BackendError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
)
from inference_service.backends.types import BackendCapabilities, BackendHealth, BackendState, RuntimeContext
from inference_service.unified_runtime import ExecutionContext, LoadRollback, ModelRequest


class ModelSession(ABC):
    """Execute one manifest-bound model resource behind the native runtime API."""

    supports_reset: bool = False
    is_stateful: bool = False

    def __init__(self, name: str, capabilities: BackendCapabilities) -> None:
        if not name:
            raise ValueError("model runtime name must be non-empty")
        self._name = name
        if self.supports_reset or self.is_stateful:
            capabilities = replace(capabilities, resettable=self.supports_reset, stateful=self.is_stateful)
        self._capabilities = capabilities
        self._condition = threading.Condition(threading.RLock())
        self._state = BackendState.CREATED
        self._reason_code: str | None = None
        self._message: str | None = None
        self._failure_count = 0
        self._context: RuntimeContext | None = None
        self._last_successful_inference_time: datetime | None = None
        self._loading = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def _update_loaded_capabilities(self, *, priority_mapping=None, **changes: object) -> None:
        """Publish backend capabilities discovered while vendor objects load."""

        with self._condition:
            if not self._loading:
                raise BackendLifecycleError(
                    f"model runtime {self.name!r} can update capabilities only while loading",
                    code="invalid_capability_update_state",
                )
            self._capabilities = replace(self._capabilities, priority_mapping=priority_mapping, **changes)

    @property
    def runtime_version(self) -> str:
        return ""

    @staticmethod
    def _runtime_version(runtime: object | None) -> str:
        if runtime is None:
            return ""
        version = getattr(runtime, "__version__", None)
        if version is not None and str(version).strip():
            return str(version).strip()
        getter = getattr(runtime, "get_version", None)
        if callable(getter):
            with suppress(Exception):
                value = getter()
                if isinstance(value, tuple):
                    value = ".".join(str(part) for part in value)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""

    def load(self, context: RuntimeContext) -> None:
        if not isinstance(context, RuntimeContext):
            raise TypeError("model runtimes require a validated RuntimeContext")
        with self._condition:
            if self._state is not BackendState.CREATED:
                raise BackendLifecycleError(
                    f"model runtime {self.name!r} cannot load from state {self._state.value}",
                    code="invalid_load_state",
                )
            self._state = BackendState.LOADING
            self._loading = True

        rollback = LoadRollback()
        try:
            self._load(context, rollback)
            self._context = context
            rollback.commit()
        except Exception as exc:
            rollback_errors = rollback.rollback()
            error = self._load_error(exc, rollback_errors)
            with self._condition:
                self._failure_count += 1
                self._reason_code = error.code
                self._message = str(error)
                self._state = BackendState.FAILED
            if error is exc:
                raise
            raise error from exc
        else:
            with self._condition:
                self._state = BackendState.READY
        finally:
            with self._condition:
                self._loading = False

    def execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]:
        """Execute one native request; handle-level lifecycle wraps this call."""

        if not isinstance(request, ModelRequest):
            raise TypeError("model runtimes require a ModelRequest")
        if not isinstance(context, ExecutionContext):
            raise TypeError("model runtimes require an ExecutionContext")
        self._require_ready()
        context.check("backend")
        self._validate_request(request, context)
        outputs = self._execute(request, context)
        context.check("backend")
        self._validate_values(outputs, self._require_context().validated_manifest.manifest.model.outputs, "output")
        with self._condition:
            self._last_successful_inference_time = datetime.now(timezone.utc)
        return outputs

    def execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: ModelRequest,
        context: ExecutionContext,
    ) -> Mapping[str, object]:
        """Execute one manifest role inside an already admitted handle request."""

        if not isinstance(request, ModelRequest) or not isinstance(context, ExecutionContext):
            raise TypeError("role execution requires native ModelRequest and ExecutionContext")
        self._require_ready()
        context.check(f"model.{role}")
        outputs = self._execute_role(role, inputs, request, context)
        self._validate_role_values(role, inputs, outputs)
        context.check(f"model.{role}")
        return outputs

    def reset(self, context: ExecutionContext | None = None) -> None:
        if not self.capabilities.resettable:
            raise BackendCapabilityError(f"model runtime {self.name!r} does not support reset", capability="reset")
        if context is not None:
            context.check("reset")
        self._reset()
        if context is not None:
            context.check("reset")

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        if not self.capabilities.supports_cancellation:
            return
        self._cancel(request_id)

    def health(self) -> BackendHealth:
        with self._condition:
            return BackendHealth(
                state=self._state,
                ready=self._state is BackendState.READY,
                reason_code=self._reason_code,
                message=self._message,
                recoverable=False,
                last_successful_inference_time=self._last_successful_inference_time,
                failure_count=self._failure_count,
            )

    def close(self) -> None:
        with self._condition:
            if self._state is BackendState.CLOSED:
                return
            if self._state is BackendState.CLOSING:
                return
            self._state = BackendState.CLOSING
        error: Exception | None = None
        try:
            self._close()
        except Exception as exc:
            error = exc
        finally:
            self._context = None
            with self._condition:
                if error is not None:
                    self._failure_count += 1
                    self._reason_code = getattr(error, "code", "close_failed")
                    self._message = str(error)
                self._state = BackendState.CLOSED
        if error is not None:
            raise error

    @abstractmethod
    def _load(self, context: RuntimeContext, rollback: LoadRollback) -> None: ...

    @abstractmethod
    def _execute(self, request: ModelRequest, context: ExecutionContext) -> Mapping[str, object]: ...

    def _validate_request(self, request: ModelRequest, context: ExecutionContext) -> None:
        del request, context

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: ModelRequest,
        context: ExecutionContext,
    ) -> Mapping[str, object]:
        raise BackendCapabilityError(
            f"model runtime {self.name!r} does not implement manifest role {role!r}", capability="role_execution"
        )

    @abstractmethod
    def _close(self) -> None: ...

    def _reset(self) -> None:
        raise BackendCapabilityError(f"model runtime {self.name!r} does not implement reset", capability="reset")

    def _cancel(self, request_id: str) -> None:
        del request_id

    def _require_context(self) -> RuntimeContext:
        if self._context is None:
            raise BackendInferenceError("model runtime is not loaded", code="runtime_not_loaded")
        return self._context

    def _require_ready(self) -> None:
        with self._condition:
            if self._state is not BackendState.READY:
                raise BackendInferenceError(
                    f"model runtime {self.name!r} is {self._state.value}", code="runtime_not_ready"
                )

    def _validate_role_values(
        self,
        role: str,
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
    ) -> None:
        deployment = self._require_context().deployment
        if not isinstance(deployment, CompiledDeployment):
            raise BackendInferenceError("role execution requires a compiled deployment", code="invalid_deployment")
        try:
            bindings = deployment.bindings[role]
        except KeyError as exc:
            raise BackendInferenceError(f"unknown execution role {role!r}", code="unknown_execution_role") from exc
        linked_inputs = {
            link.semantic
            for link in deployment.device_links
            if link.consumer == role and link.transport == "device_pointer"
        }
        host_inputs = tuple(binding for binding in bindings.inputs if binding.semantic not in linked_inputs)
        linked_outputs = {
            link.semantic
            for link in deployment.device_links
            if link.producer == role and link.transport == "device_pointer"
        }
        host_outputs = tuple(binding for binding in bindings.outputs if binding.semantic not in linked_outputs)
        self._validate_values(inputs, host_inputs, f"role_{role}_input")
        self._validate_values(outputs, host_outputs, f"role_{role}_output")

    @staticmethod
    def _validate_values(values: Mapping[str, object], descriptors: tuple[SemanticTensor, ...], direction: str) -> None:
        expected = {descriptor.semantic: descriptor for descriptor in descriptors}
        missing = sorted(set(expected) - set(values))
        unexpected = sorted(set(values) - set(expected))
        if missing or unexpected:
            raise BackendInferenceError(
                f"named {direction} semantics mismatch: missing={missing}, unexpected={unexpected}",
                code=f"{direction}_semantic_mismatch",
            )
        for semantic, descriptor in expected.items():
            try:
                array = np.asarray(values[semantic])
            except Exception as exc:
                raise BackendInferenceError(
                    f"{direction} {semantic!r} is not tensor-like: {exc}", code=f"invalid_{direction}_tensor"
                ) from exc
            expected_dtype = np.dtype(descriptor.dtype)
            if array.dtype != expected_dtype:
                raise BackendInferenceError(
                    f"{direction} {semantic!r} dtype {array.dtype} does not match {expected_dtype}",
                    code=f"{direction}_dtype_mismatch",
                )
            if len(array.shape) != len(descriptor.shape) or any(
                declared != -1 and declared != actual
                for declared, actual in zip(descriptor.shape, array.shape, strict=True)
            ):
                raise BackendInferenceError(
                    f"{direction} {semantic!r} shape {array.shape} does not match {descriptor.shape}",
                    code=f"{direction}_shape_mismatch",
                )

    def _load_error(self, exc: Exception, rollback_errors: tuple[Exception, ...]) -> BackendError:
        suffix = ""
        if rollback_errors:
            suffix = "; rollback errors: " + "; ".join(str(error) for error in rollback_errors)
        if isinstance(exc, BackendError) and not suffix:
            return exc
        code = exc.code if isinstance(exc, BackendError) else "load_failed"
        return BackendLoadError(f"model runtime {self.name!r} failed to load: {exc}{suffix}", code=code)

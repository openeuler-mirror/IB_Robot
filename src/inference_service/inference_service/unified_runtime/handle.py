"""Lifecycle owner and public execution boundary for unified runtimes."""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .adapters import ResultAdapter
from .admission import NativeAdmission
from .assembly import OwnedComponent, RuntimeAssembly
from .contracts import (
    Deadline,
    ExecutionContext,
    LifecycleState,
    ModelRequest,
    ModelResult,
    OutcomeEvidence,
    OutcomeEvidenceTracker,
    OutcomeState,
    RuntimeDiagnostics,
    RuntimeHealth,
    StreamState,
)
from .errors import (
    CancellationRequested,
    DeadlineExceeded,
    ExecutionFailure,
    ExecutionFailureFactory,
    RecoveryAction,
    RecoveryRequirement,
    RecoveryScope,
)
from .streaming import StreamDiagnostics, StreamHandle, StreamingRuntime

_LEGAL_TRANSITIONS: dict[LifecycleState, frozenset[LifecycleState]] = {
    LifecycleState.CREATED: frozenset({LifecycleState.LOADING, LifecycleState.CLOSING}),
    LifecycleState.LOADING: frozenset({LifecycleState.READY, LifecycleState.FAILED, LifecycleState.CLOSING}),
    LifecycleState.READY: frozenset(
        {LifecycleState.RESET_REQUIRED, LifecycleState.RESETTING, LifecycleState.FAILED, LifecycleState.CLOSING}
    ),
    LifecycleState.RESET_REQUIRED: frozenset({LifecycleState.RESETTING, LifecycleState.FAILED, LifecycleState.CLOSING}),
    LifecycleState.RESETTING: frozenset(
        {LifecycleState.READY, LifecycleState.RESET_REQUIRED, LifecycleState.FAILED, LifecycleState.CLOSING}
    ),
    LifecycleState.FAILED: frozenset({LifecycleState.CLOSING}),
    LifecycleState.CLOSING: frozenset({LifecycleState.CLOSED}),
    LifecycleState.CLOSED: frozenset(),
}


def _compatible_call(method: object, candidates: tuple[tuple[tuple[object, ...], dict[str, object]], ...]) -> object:
    """Call a dependency using the first signature-compatible spelling."""

    if not callable(method):
        raise TypeError("runtime dependency method is not callable")
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        args, kwargs = candidates[0]
        return method(*args, **kwargs)  # type: ignore[misc]
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return method(*args, **kwargs)  # type: ignore[misc]
    args, kwargs = candidates[0]
    return method(*args, **kwargs)  # type: ignore[misc]


def _call_load(resource: object, context: object) -> None:
    method = getattr(resource, "load", None)
    if not callable(method):
        return
    _compatible_call(
        method,
        (
            ((context,), {}),
            ((), {"context": context}),
            ((), {"execution_context": context}),
            ((), {}),
        ),
    )


def _call_reset(resource: object, context: ExecutionContext) -> None:
    method = getattr(resource, "reset", None)
    if not callable(method):
        return
    _compatible_call(
        method,
        (
            ((context,), {}),
            ((), {"context": context}),
            ((), {"execution_context": context}),
            ((), {"deadline": context.deadline}),
            ((), {}),
        ),
    )


def _is_superseded_frame(exc: BaseException) -> bool:
    """Duck-typed coalescing marker (circular import: staged executor)."""

    return bool(getattr(exc, "superseded_frame", False))


def _call_close(resource: object) -> None:
    method = getattr(resource, "close", None)
    if callable(method):
        _compatible_call(method, ((((), {})),))
        return
    _call_release(resource)


def _call_release(resource: object) -> None:
    for name in ("release", "unregister", "close"):
        method = getattr(resource, name, None)
        if callable(method):
            _compatible_call(method, ((((), {})),))
            return


def _call_executor(executor: object, request: ModelRequest, context: ExecutionContext) -> object:
    method = getattr(executor, "execute", None)
    if not callable(method) and callable(executor):
        method = executor
    if not callable(method):
        raise TypeError("runtime executor must expose execute(request, context)")
    return _compatible_call(
        method,
        (
            ((request, context), {}),
            ((request,), {"context": context}),
            ((request,), {"execution_context": context}),
            ((request,), {"deadline": context.deadline, "cancellation_token": context.cancellation_token}),
            ((request,), {}),
        ),
    )


def _call_adapter(
    adapter: object,
    frame: object,
    context: ExecutionContext,
    evidence: OutcomeEvidence,
    latency_ms: float,
) -> object:
    method = getattr(adapter, "adapt", None)
    if not callable(method):
        raise TypeError("result adapter must expose adapt(frame, ...)")
    return _compatible_call(
        method,
        (
            (
                (frame,),
                {"context": context, "evidence": evidence, "latency": latency_ms},
            ),
            ((frame, context, evidence, latency_ms), {}),
            ((frame, context), {}),
            ((frame,), {"context": context, "evidence": evidence}),
            ((frame,), {}),
        ),
    )


def _call_stream_open(streaming_runtime: object, context: ExecutionContext) -> object:
    method = getattr(streaming_runtime, "open_stream", None)
    if not callable(method):
        raise TypeError("streaming runtime must expose open_stream(context)")
    return _compatible_call(method, (((context,), {}), ((), {"context": context}), ((), {})))


def _call_stream_step(
    streaming_runtime: object, stream_handle: object, request: ModelRequest, context: ExecutionContext
) -> object:
    method = getattr(streaming_runtime, "step", None)
    if not callable(method):
        raise TypeError("streaming runtime must expose step(stream_handle, request, context)")
    return _compatible_call(
        method,
        (
            ((stream_handle, request, context), {}),
            ((stream_handle, request), {"context": context}),
            ((stream_handle, request), {"execution_context": context}),
            ((stream_handle, request), {}),
        ),
    )


def _call_stream_control(
    streaming_runtime: object,
    name: str,
    stream_handle: object,
    context: ExecutionContext,
) -> None:
    method = getattr(streaming_runtime, name, None)
    if not callable(method):
        return
    _compatible_call(
        method,
        (
            ((stream_handle, context), {}),
            ((stream_handle,), {"context": context}),
            ((stream_handle,), {"execution_context": context}),
            ((stream_handle,), {}),
        ),
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _StreamRecord:
    public_handle: StreamHandle
    backend_handle: object
    state_bank_mode: str
    created_at: datetime
    last_step_at: datetime | None = None
    active_step: bool = False
    active_thread: int | None = None
    resetting: bool = False
    closing: bool = False
    recovery_requirement: RecoveryRequirement | None = None
    last_failure_code: str | None = None
    host_state: dict[str, object] = field(default_factory=dict)
    control_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def diagnostics(self) -> StreamDiagnostics:
        return StreamDiagnostics(
            stream_id=self.public_handle.stream_id,
            state=self.state,
            active_step=self.active_step,
            state_bank_mode=self.state_bank_mode,
            created_at=self.created_at,
            last_step_at=self.last_step_at,
            recovery_requirement=self.recovery_requirement,
            last_failure_code=self.last_failure_code,
        )

    @property
    def state(self) -> StreamState:
        return self.public_handle.state

    def set_state(self, state: StreamState) -> None:
        self.public_handle._set_state(state)


class ModelRuntimeHandle:
    """Own an assembled runtime and expose one normalized execution boundary."""

    def __init__(
        self,
        assembly: RuntimeAssembly | object | None = None,
        *,
        executor: object | None = None,
        runtime_executor: object | None = None,
        streaming_runtime: StreamingRuntime | object | None = None,
        session: object | None = None,
        role_assemblies: Mapping[str, object] | None = None,
        artifact_bindings: Mapping[str, object] | None = None,
        adapter: object | None = None,
        request_adapter: object | None = None,
        result_adapter: object | None = None,
        failure_factory: ExecutionFailureFactory | None = None,
        processor: object | None = None,
        worker: object | None = None,
        host_resources: Iterable[object] = (),
        host_resource: object | None = None,
        device_lease: object | None = None,
        provider_leases: Iterable[object] = (),
        provider_lease: object | None = None,
        provider_registrations: Iterable[object] = (),
        provider_registration: object | None = None,
        provider_leases_and_registrations: Iterable[object] = (),
        owned_components: Iterable[object] | None = None,
        components: Iterable[object] | None = None,
        stateful: bool | None = None,
        resettable: bool | None = None,
        state_scope: str | None = None,
        state_bank_mode: str | None = None,
        max_open_streams: int | None = None,
        max_active_executions: int | None = None,
        reloadable: bool | None = None,
        runtime_id: str | None = None,
        identity: object | None = None,
        execution_contract: object | None = None,
        declared_capabilities: Mapping[str, object] | None = None,
        capabilities: object | None = None,
        deployment_fingerprint: str | None = None,
        runtime_profile_fingerprint: str | None = None,
        artifact_integrity: object | None = None,
        runtime_version: str | None = None,
    ) -> None:
        if isinstance(assembly, RuntimeAssembly):
            source = assembly
            self._assembly = source
            selected_executor = source.runtime_executor
            selected_streaming = source.streaming_runtime
            selected_adapter = source.request_adapter or source.result_adapter
            selected_factory = source.failure_factory
            capability_source = source.capabilities or source.declared_capabilities
            selected_stateful = bool(source.stateful or self._capability_value(capability_source, "stateful", False))
            selected_resettable = bool(
                source.resettable or self._capability_value(capability_source, "resettable", False)
            )
            selected_scope = source.state_scope
            selected_bank = source.state_bank_mode
            selected_max_streams = source.max_open_streams
            selected_granularity = source.cancellation_granularity
            selected_max_active = source.max_active_executions
            selected_reloadable = source.reloadable
            selected_runtime_id = source.runtime_id
            selected_identity = source.identity
            selected_contract = source.execution_contract
            selected_capabilities = source.capabilities or source.declared_capabilities
            selected_deployment_fp = source.deployment_fingerprint
            selected_profile_fp = source.runtime_profile_fingerprint
            selected_integrity = source.artifact_integrity
            selected_version = source.runtime_version
            entries = source.component_entries()
            provider = getattr(source.providers, "resource_admission_provider", None)
            capabilities = source.session or source.capabilities or source.declared_capabilities
        else:
            if assembly is not None and executor is None and runtime_executor is None:
                executor = assembly
            selected_executor = runtime_executor or executor
            if selected_executor is None:
                raise ValueError("ModelRuntimeHandle requires a runtime executor")
            selected_streaming = streaming_runtime
            selected_adapter = request_adapter or result_adapter or adapter
            selected_factory = failure_factory or ExecutionFailureFactory()
            capability_source = capabilities if capabilities is not None else declared_capabilities
            selected_stateful = (
                self._capability_value(capability_source, "stateful", False) if stateful is None else stateful
            )
            selected_resettable = (
                self._capability_value(capability_source, "resettable", False) if resettable is None else resettable
            )
            selected_scope = "request" if state_scope is None else state_scope
            selected_bank = state_bank_mode
            selected_max_streams = max_open_streams
            selected_granularity = self._capability_value(
                capability_source, "cancellation_granularity", "request_boundary"
            )
            selected_max_active = (
                self._capability_value(capability_source, "max_active_executions", 1)
                if max_active_executions is None
                else max_active_executions
            )
            selected_reloadable = False if reloadable is None else reloadable
            selected_runtime_id = runtime_id
            selected_identity = identity
            selected_contract = execution_contract
            selected_capabilities = declared_capabilities or (
                dict(capabilities) if isinstance(capabilities, Mapping) else {}
            )
            selected_deployment_fp = deployment_fingerprint
            selected_profile_fp = runtime_profile_fingerprint
            selected_integrity = artifact_integrity
            selected_version = runtime_version
            generated_assembly = RuntimeAssembly(
                runtime_executor=selected_executor,
                streaming_runtime=selected_streaming,
                session=session,
                role_assemblies=dict(role_assemblies or {}),
                artifact_bindings=dict(artifact_bindings or {}),
                request_adapter=selected_adapter,
                failure_factory=selected_factory,
                processor=processor,
                worker=worker,
                host_resources=tuple(host_resources),
                host_resource=host_resource,
                device_lease=device_lease,
                provider_leases=tuple(provider_leases),
                provider_lease=provider_lease,
                provider_registrations=tuple(provider_registrations),
                provider_registration=provider_registration,
                provider_leases_and_registrations=tuple(provider_leases_and_registrations),
                owned_components=tuple(owned_components or ()),
                components=tuple(components or ()),
                stateful=bool(selected_stateful),
                resettable=bool(selected_resettable),
                state_scope=selected_scope,
                state_bank_mode=selected_bank,
                max_open_streams=selected_max_streams,
                cancellation_granularity=selected_granularity,
                max_active_executions=selected_max_active,
                reloadable=bool(selected_reloadable),
                runtime_id=selected_runtime_id,
                identity=selected_identity,
                execution_contract=selected_contract,
                declared_capabilities=selected_capabilities,
                capabilities=capabilities,
                deployment_fingerprint=selected_deployment_fp,
                runtime_profile_fingerprint=selected_profile_fp,
                artifact_integrity=selected_integrity,
                runtime_version=selected_version,
            )
            self._assembly = generated_assembly
            entries = generated_assembly.component_entries()
            provider = getattr(generated_assembly.providers, "resource_admission_provider", None)
            capabilities = session or capabilities

        self._executor = selected_executor
        self._streaming_runtime = selected_streaming
        self._adapter = selected_adapter or ResultAdapter()
        self._failure_factory = selected_factory or ExecutionFailureFactory()
        self._stateful = bool(
            selected_stateful if stateful is None or isinstance(assembly, RuntimeAssembly) else stateful
        )
        self._resettable = bool(
            selected_resettable if resettable is None or isinstance(assembly, RuntimeAssembly) else resettable
        )
        self._state_scope = self._contract_value(selected_contract, "state_scope", selected_scope)
        self._state_bank_mode = self._contract_value(selected_contract, "state_bank_mode", selected_bank)
        self._max_open_streams = self._contract_value(selected_contract, "max_open_streams", selected_max_streams)
        self._cancellation_granularity = self._contract_value(
            selected_contract, "cancellation_granularity", selected_granularity
        )
        self._max_active_executions = int(
            selected_max_active
            if max_active_executions is None or isinstance(assembly, RuntimeAssembly)
            else max_active_executions
        )
        self._reloadable = bool(
            selected_reloadable if reloadable is None or isinstance(assembly, RuntimeAssembly) else reloadable
        )
        self._runtime_id = selected_runtime_id or f"runtime-{uuid.uuid4().hex}"
        self._identity = selected_identity
        self._execution_contract = selected_contract
        self._declared_capabilities = self._capabilities_mapping(selected_capabilities)
        self._deployment_fingerprint = selected_deployment_fp
        self._runtime_profile_fingerprint = selected_profile_fp
        self._artifact_integrity = selected_integrity
        self._runtime_version = selected_version
        self._entries = tuple(entries)
        self._admission = (
            NativeAdmission(provider, str(getattr(capabilities, "name", self._runtime_id)), capabilities)
            if provider is not None
            and callable(getattr(provider, "register_instance", None))
            and capabilities is not None
            else None
        )

        self._validate_configuration()

        # Claim only after all handle configuration checks have succeeded.  A
        # factory can therefore release an assembly when construction fails
        # before ownership is actually transferred.
        if isinstance(assembly, RuntimeAssembly):
            assembly.claim_ownership()

        self._condition = threading.Condition(threading.RLock())
        self._control_lock = threading.Lock()
        self._state = LifecycleState.CREATED
        self._admission_open = False
        self._loading = False
        self._active_executions = 0
        self._active_frame_executions = 0
        self._active_stream_steps = 0
        self._active_threads: dict[int, int] = {}
        self._execution_contexts: dict[str, ExecutionContext] = {}
        self._loaded_entries: tuple[OwnedComponent, ...] = ()
        self._released_ids: set[int] = set()
        self._failure_count = 0
        self._last_failure: ExecutionFailure | None = None
        self._last_outcome: OutcomeEvidence | None = None
        self._last_successful_at: datetime | None = None
        self._recovery_requirement: RecoveryRequirement | None = None
        self._recovery_available = False
        self._close_error: ExecutionFailure | None = None
        self._stream_owner = object()
        self._streams: dict[str, _StreamRecord] = {}
        self._stream_counter = 0
        self._stream_openings = 0

    @staticmethod
    def _contract_value(contract: object | None, name: str, fallback: Any) -> Any:
        if contract is None:
            return fallback
        if isinstance(contract, Mapping):
            return contract.get(name, fallback)
        return getattr(contract, name, fallback)

    @staticmethod
    def _capability_value(capabilities: object | None, name: str, fallback: Any) -> Any:
        if capabilities is None:
            return fallback
        if isinstance(capabilities, Mapping):
            return capabilities.get(name, fallback)
        return getattr(capabilities, name, fallback)

    @staticmethod
    def _capabilities_mapping(capabilities: object | None) -> dict[str, object]:
        if capabilities is None:
            return {}
        if isinstance(capabilities, Mapping):
            return dict(capabilities)
        names = (
            "stateful",
            "resettable",
            "max_open_streams",
            "state_bank_mode",
            "supports_cancellation",
            "cancellation_granularity",
        )
        return {name: getattr(capabilities, name) for name in names if hasattr(capabilities, name)}

    def _validate_configuration(self) -> None:
        structure = self._contract_value(self._execution_contract, "execution_structure", "direct")
        visibility = self._contract_value(self._execution_contract, "orchestration_visibility", None)
        if structure not in {"direct", "iterative"}:
            raise ValueError("execution_structure must be direct or iterative")
        if structure == "direct" and visibility not in {None, ""}:
            raise ValueError("direct execution cannot declare orchestration_visibility")
        if structure == "iterative" and visibility not in {"executor", "session"}:
            raise ValueError("iterative execution requires executor or session visibility")
        if self._state_scope not in {"request", "stream"}:
            raise ValueError("state_scope must be request or stream")
        if self._cancellation_granularity not in {"stage", "checkpoint", "request_boundary"}:
            raise ValueError("invalid cancellation_granularity")
        if self._state_scope == "request" and self._stateful:
            raise ValueError("stateful request contracts are not supported")
        if self._state_scope == "request":
            if self._state_bank_mode is not None or self._max_open_streams is not None:
                raise ValueError("request contracts cannot declare stream admission")
            self._max_open_streams = None
        else:
            if self._streaming_runtime is None:
                raise ValueError("stream contracts require a StreamingRuntime")
            if self._state_bank_mode is None:
                self._state_bank_mode = "per_stream"
            if self._state_bank_mode not in {"per_stream", "runtime_exclusive"}:
                raise ValueError("state_bank_mode must be per_stream or runtime_exclusive")
            if self._max_open_streams is None:
                self._max_open_streams = 1
            if (
                isinstance(self._max_open_streams, bool)
                or not isinstance(self._max_open_streams, int)
                or self._max_open_streams < 1
            ):
                raise ValueError("max_open_streams must be a positive integer")
            if self._state_bank_mode == "runtime_exclusive" and self._max_open_streams != 1:
                raise ValueError("runtime_exclusive state banks require max_open_streams=1")
        if isinstance(self._max_active_executions, bool) or self._max_active_executions < 1:
            raise ValueError("max_active_executions must be a positive integer")

    @property
    def state(self) -> LifecycleState:
        with self._condition:
            return self._state

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def executor(self) -> object:
        return self._executor

    @property
    def assembly(self) -> RuntimeAssembly:
        return self._assembly

    @property
    def result_adapter(self) -> object:
        return self._adapter

    @property
    def failure_factory(self) -> ExecutionFailureFactory:
        return self._failure_factory

    @property
    def owned_components(self) -> tuple[OwnedComponent, ...]:
        return self._entries

    @property
    def streaming_runtime(self) -> object | None:
        return self._streaming_runtime

    @property
    def lifecycle_state(self) -> LifecycleState:
        return self.state

    @property
    def status(self) -> LifecycleState:
        return self.state

    @property
    def ready(self) -> bool:
        return self.state is LifecycleState.READY

    @property
    def reset_required(self) -> bool:
        return self.state is LifecycleState.RESET_REQUIRED

    @property
    def closed(self) -> bool:
        return self.state is LifecycleState.CLOSED

    @property
    def active_executions(self) -> int:
        with self._condition:
            return self._active_executions

    @property
    def last_failure(self) -> ExecutionFailure | None:
        with self._condition:
            return self._last_failure

    @property
    def recovery_requirement(self) -> RecoveryRequirement | None:
        with self._condition:
            return self._recovery_requirement

    def _transition_locked(self, target: LifecycleState) -> None:
        if target is self._state:
            return
        if target not in _LEGAL_TRANSITIONS[self._state]:
            raise RuntimeError(f"illegal runtime transition {self._state.value} -> {target.value}")
        self._state = target

    def _lifecycle_failure_locked(self, operation: str) -> ExecutionFailure:
        state = self._state
        if state is LifecycleState.CLOSED or state is LifecycleState.CLOSING:
            code = "runtime_closed"
            message = f"runtime {self.runtime_id!r} is closed"
        elif state is LifecycleState.RESET_REQUIRED:
            code = "recovery_required"
            message = f"runtime {self.runtime_id!r} requires recovery before {operation}"
        else:
            code = "runtime_not_ready"
            message = f"runtime {self.runtime_id!r} is not ready for {operation} (state={state.value})"
        recovery = self._recovery_requirement or RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction.NONE)
        return self._failure_factory.create(
            code,
            message,
            evidence=OutcomeEvidence.not_started("admission", state=state.value, operation=operation),
            recovery=recovery,
            recoverable=self._recovery_available,
            details={"runtime_id": self.runtime_id, "state": state.value, "operation": operation},
        )

    def _context_or_type_error(self, context: ExecutionContext) -> ExecutionContext:
        if not isinstance(context, ExecutionContext):
            raise TypeError("unified runtime operations require an ExecutionContext")
        return context

    def load(self, context: object = None) -> None:
        if context is None:
            context = getattr(self._assembly, "load_context", None)
        with self._condition:
            if self._state is not LifecycleState.CREATED:
                raise self._failure_factory.create(
                    "runtime_lifecycle_invalid",
                    f"runtime {self.runtime_id!r} cannot load from {self._state.value}",
                    evidence=OutcomeEvidence.not_started("admission", operation="load"),
                    details={"state": self._state.value, "operation": "load"},
                )
            self._transition_locked(LifecycleState.LOADING)
            self._loading = True

        attempted: list[OwnedComponent] = []
        try:
            for entry in self._entries:
                with self._condition:
                    if self._state is LifecycleState.CLOSING:
                        raise RuntimeError("runtime load was interrupted by close")
                load_context = entry.load_context if entry.load_context is not None else context
                _call_load(entry.resource, load_context)
                attempted.append(entry)
            with self._condition:
                if self._state is not LifecycleState.LOADING:
                    raise RuntimeError(f"runtime load ended in {self._state.value}")
                # Some sessions discover reset support only after loading the model.
                session_capabilities = getattr(self._assembly.session, "capabilities", None)
                if self._assembly.retry_failed_reset:
                    self._resettable = self._resettable or bool(getattr(session_capabilities, "resettable", False))
                self._loaded_entries = tuple(attempted)
                self._admission_open = True
                self._transition_locked(LifecycleState.READY)
                self._condition.notify_all()
        except Exception as exc:
            rollback_errors = self._release_entries(tuple(reversed(attempted)))
            evidence = OutcomeEvidence.not_started("admission", operation="load")
            details: dict[str, object] = {"runtime_id": self.runtime_id}
            if rollback_errors:
                details["rollback_errors"] = tuple(str(error) for error in rollback_errors)
            failure = self._failure_factory.from_exception(
                exc,
                code="runtime_load_failed",
                message=f"runtime {self.runtime_id!r} failed to load: {exc}",
                evidence=evidence,
                scope=RecoveryScope.RUNTIME,
                action=RecoveryAction.RELOAD,
                recoverable=self._reloadable,
                details=details,
            )
            with self._condition:
                self._admission_open = False
                if any(getattr(error, "close_pending", False) for error in rollback_errors):
                    self._loaded_entries = tuple(attempted)
                if self._state is not LifecycleState.CLOSED and self._state is not LifecycleState.CLOSING:
                    self._transition_locked(LifecycleState.FAILED)
                self._record_failure_locked(failure)
                self._loading = False
                self._condition.notify_all()
            raise failure from exc
        finally:
            with self._condition:
                self._loading = False
                self._condition.notify_all()

    def _register_execution(self, context: ExecutionContext, *, frame: bool = False) -> None:
        with self._condition:
            if self._state is not LifecycleState.READY or not self._admission_open:
                raise self._lifecycle_failure_locked("execute")
            if self._state_scope != "request":
                raise self._failure_factory.create(
                    "stream_required",
                    "stream-scoped runtimes require open_stream/step",
                    evidence=OutcomeEvidence.not_started("admission"),
                    details={"state_scope": self._state_scope},
                )
            # Frames have one running and one pending slot; the third record
            # owns a replacement until the executor settles the old pending frame.
            active = self._active_frame_executions if frame else self._active_executions - self._active_frame_executions
            limit = 3 if frame else self._max_active_executions
            if active >= limit:
                raise self._failure_factory.create(
                    "admission_rejected",
                    "runtime request admission limit is exhausted",
                    evidence=OutcomeEvidence.not_started("admission"),
                    details={"active_executions": active, "limit": limit},
                )
            if (frame or self._active_frame_executions) and context.request_id in self._execution_contexts:
                raise self._failure_factory.create(
                    "request_conflict",
                    "runtime request ID is already active",
                    evidence=OutcomeEvidence.not_started("admission"),
                )
            context.check("admission")
            self._active_executions += 1
            self._active_frame_executions += int(frame)
            if not frame:
                thread_id = threading.get_ident()
                self._active_threads[thread_id] = self._active_threads.get(thread_id, 0) + 1
            self._execution_contexts[context.request_id] = context

    def _release_execution(self, context: ExecutionContext, *, frame: bool = False) -> None:
        with self._condition:
            self._active_executions -= 1
            self._active_frame_executions -= int(frame)
            if not frame:
                thread_id = threading.get_ident()
                count = self._active_threads.get(thread_id, 0) - 1
                if count > 0:
                    self._active_threads[thread_id] = count
                else:
                    self._active_threads.pop(thread_id, None)
            self._execution_contexts.pop(context.request_id, None)
            self._condition.notify_all()

    def _execution_failure(
        self,
        exc: Exception,
        context: ExecutionContext,
        *,
        started: bool,
        record: bool = True,
    ) -> ExecutionFailure:
        if isinstance(exc, ExecutionFailure):
            failure = exc
        else:
            started = bool(getattr(exc, "operation_started", started))
            failure = self._failure_factory.from_exception(
                exc,
                evidence=OutcomeEvidence(
                    OutcomeState.STARTED if started else OutcomeState.NOT_STARTED,
                    bool(getattr(exc, "outcome_known", not (started and self._stateful))),
                    bool(getattr(exc, "state_mutated", started and self._stateful)),
                    str(getattr(exc, "phase", "backend")),
                ),
                details={"request_id": context.request_id},
            )
        if record:
            self._remember_failure(failure)
            self._apply_uncertain_request_failure(failure, started)
        return failure

    def submit_frame(self, request: ModelRequest, context: ExecutionContext) -> Future:
        """Own a coalesced producer request until its executor completion fence."""
        if not isinstance(request, ModelRequest):
            raise TypeError("unified runtime frame submission requires a ModelRequest")
        context = self._context_or_type_error(context)
        submit = getattr(self._executor, "submit_frame", None)
        if not callable(submit):
            raise TypeError("runtime executor does not support frame submission")
        self._register_execution(context, frame=True)
        result: Future = Future()
        # Device work cannot be canceled by dropping the Future. cancel(request_id)
        # signals the shared context while ownership lasts until completion.
        result.set_running_or_notify_cancel()

        def complete(done: Future) -> None:
            value = error = None
            try:
                value = done.result()
                context.check("frame_result")
                with self._condition:
                    self._last_successful_at = _now()
                    self._last_outcome = OutcomeEvidence.completed("frame_result")
            except Exception as exc:
                if _is_superseded_frame(exc):
                    # Normal coalescing: the newer frame owns the producer.
                    # Propagated to the caller but excluded from health
                    # accounting (no failure_count, no _last_failure).
                    error = self._execution_failure(exc, context, started=True, record=False)
                else:
                    error = self._execution_failure(exc, context, started=True)
            finally:
                self._release_execution(context, frame=True)
            if error is not None:
                result.set_exception(error)
            else:
                result.set_result(value)

        try:
            submitted = submit(request, context=context)
        except Exception as exc:
            failure = self._execution_failure(exc, context, started=False)
            self._release_execution(context, frame=True)
            result.set_exception(failure)
        else:
            submitted.add_done_callback(complete)
        return result

    def execute(self, request: ModelRequest, context: ExecutionContext) -> ModelResult:
        if not isinstance(request, ModelRequest):
            raise TypeError("unified runtime execute requires a ModelRequest")
        context = self._context_or_type_error(context)
        tracker = OutcomeEvidenceTracker()
        try:
            self._register_execution(context)
        except (DeadlineExceeded, CancellationRequested) as exc:
            failure = self._failure_factory.from_exception(
                exc,
                evidence=tracker.mark_failed(exc.phase, state=OutcomeState.NOT_STARTED),
                details={"request_id": context.request_id},
            )
            self._remember_failure(failure)
            raise failure from exc

        started = False
        started_at = time.perf_counter()
        try:
            context.check("request")
            tracker.mark_started("backend", state_mutated=self._stateful, outcome_known=not self._stateful)
            started = True
            if self._admission is not None:
                with self._admission.admit(context.deadline):
                    frame = _call_executor(self._executor, request, context)
            else:
                frame = _call_executor(self._executor, request, context)
            if isinstance(frame, ExecutionFailure):
                raise frame
            tracker.mark_completed("backend")
            try:
                result = _call_adapter(
                    self._adapter, frame, context, tracker.snapshot(), (time.perf_counter() - started_at) * 1000.0
                )
                if not isinstance(result, ModelResult):
                    raise TypeError("result adapter did not return ModelResult")
            except ExecutionFailure:
                raise
            except Exception as exc:
                failure_code = str(getattr(exc, "code", "adaptation_failed"))
                failure_phase = "output_validation" if failure_code == "output_validation_failed" else "adaptation"
                evidence = tracker.mark_failed(
                    failure_phase,
                    state=OutcomeState.COMPLETED,
                    outcome_known=True,
                    state_mutated=self._stateful,
                )
                failure = self._failure_factory.from_exception(
                    exc,
                    code=failure_code,
                    message=f"runtime result adaptation failed: {exc}",
                    evidence=evidence,
                    details={"request_id": context.request_id},
                )
                raise failure from exc
            result_evidence = tracker.mark_completed("adaptation", request_id=context.request_id)
            result = result.with_evidence(result_evidence)
            try:
                context.check("result")
            except (DeadlineExceeded, CancellationRequested) as exc:
                evidence = tracker.mark_failed(
                    exc.phase,
                    state=OutcomeState.COMPLETED,
                    outcome_known=True,
                    state_mutated=self._stateful,
                )
                failure = self._failure_factory.from_exception(
                    exc,
                    evidence=evidence,
                    details={"request_id": context.request_id},
                )
                raise failure from exc
            with self._condition:
                self._last_successful_at = _now()
                self._last_outcome = result.evidence
            return result
        except ExecutionFailure as failure:
            self._remember_failure(failure)
            self._apply_uncertain_request_failure(failure, started)
            raise
        except (DeadlineExceeded, CancellationRequested) as exc:
            evidence = tracker.mark_failed(
                exc.phase,
                state=OutcomeState.STARTED if started else OutcomeState.NOT_STARTED,
                outcome_known=not started,
                state_mutated=started and self._stateful,
            )
            failure = self._failure_factory.from_exception(
                exc,
                evidence=evidence,
                details={"request_id": context.request_id},
            )
            self._remember_failure(failure)
            self._apply_uncertain_request_failure(failure, started)
            raise failure from exc
        except Exception as exc:
            failure = self._execution_failure(exc, context, started=started)
            raise failure from exc
        finally:
            self._release_execution(context)

    def cancel(self, request_id: str, *, deadline: Deadline | None = None) -> None:
        """Cancel an active request through the shared native cancellation token."""

        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("runtime cancellation requires a non-empty request_id")
        with self._condition:
            context = self._execution_contexts.get(request_id)
            if context is None:
                return
            context.cancellation_token.cancel("runtime cancellation requested")
        executor_cancel = getattr(self._executor, "cancel", None)
        if not callable(executor_cancel):
            return
        try:
            executor_cancel(request_id, deadline=deadline.deadline_at if deadline is not None else None)
        except TypeError:
            executor_cancel(request_id)

    def _apply_uncertain_request_failure(self, failure: ExecutionFailure, started: bool) -> None:
        evidence = failure.evidence
        if not (started and evidence.state is OutcomeState.STARTED and not evidence.outcome_known):
            return
        with self._condition:
            if self._state is not LifecycleState.READY:
                return
            if self._resettable:
                self._recovery_requirement = RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction.RESET_RUNTIME)
                self._recovery_available = True
                self._transition_locked(LifecycleState.RESET_REQUIRED)
            else:
                self._recovery_requirement = RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction.RELOAD)
                self._recovery_available = self._reloadable
                self._transition_locked(LifecycleState.FAILED)
            self._admission_open = False

    def mark_reset_required(
        self,
        *,
        evidence: OutcomeEvidence | None = None,
        recovery: RecoveryRequirement | None = None,
    ) -> None:
        """Enter fail-closed recovery state for an externally observed outcome."""

        selected = recovery or RecoveryRequirement(
            RecoveryScope.RUNTIME,
            RecoveryAction.RESET_RUNTIME if self._resettable else RecoveryAction.RELOAD,
        )
        with self._condition:
            if self._state is LifecycleState.READY:
                target = LifecycleState.RESET_REQUIRED if self._resettable else LifecycleState.FAILED
                self._transition_locked(target)
            self._admission_open = False
            self._recovery_requirement = selected
            self._recovery_available = (
                self._resettable if selected.action is RecoveryAction.RESET_RUNTIME else self._reloadable
            )
            self._last_outcome = evidence or OutcomeEvidence.started("backend", outcome_known=False, state_mutated=True)

    require_reset = mark_reset_required

    def _remember_failure(self, failure: ExecutionFailure) -> None:
        with self._condition:
            self._record_failure_locked(failure)

    def _record_failure_locked(self, failure: ExecutionFailure) -> None:
        self._failure_count += 1
        self._last_failure = failure
        self._last_outcome = failure.evidence
        if failure.recovery.action is not RecoveryAction.NONE and failure.recovery.scope is not RecoveryScope.STREAM:
            self._recovery_requirement = failure.recovery
            self._recovery_available = failure.recoverable

    def _reset_failure_state(self) -> LifecycleState:
        """Scheduled pipelines may retry drain; other runtimes keep legacy failure."""

        return (
            LifecycleState.RESET_REQUIRED
            if self._resettable and self._assembly.retry_failed_reset
            else LifecycleState.FAILED
        )

    def _control_context(self, context: ExecutionContext | None, deadline: Deadline | None) -> ExecutionContext:
        if context is not None and not isinstance(context, ExecutionContext):
            raise TypeError("control operations require an ExecutionContext")
        if context is None:
            effective = Deadline.unbounded() if deadline is None else deadline
            return ExecutionContext("runtime-control", effective)
        if deadline is None:
            return context
        return context.with_deadline(deadline)

    def _wait_quiescent_locked(self, context: ExecutionContext, *, streams: bool = True) -> None:
        while (
            self._active_executions or self._loading or self._stream_openings or (streams and self._active_stream_steps)
        ):
            context.check("control_drain")
            remaining = context.deadline.remaining_seconds()
            self._condition.wait(timeout=0.05 if remaining is None else min(0.05, remaining))

    def reset(self, context: ExecutionContext | None = None, *, deadline: Deadline | None = None) -> None:
        if not self._stateful and not self._resettable:
            with self._condition:
                if self._state in {LifecycleState.CLOSED, LifecycleState.CLOSING}:
                    raise self._lifecycle_failure_locked("reset")
            return
        control_context = self._control_context(context, deadline)
        with self._control_lock:
            with self._condition:
                if self._state is LifecycleState.CLOSED or self._state is LifecycleState.CLOSING:
                    raise self._lifecycle_failure_locked("reset")
                if self._state not in {LifecycleState.READY, LifecycleState.RESET_REQUIRED}:
                    raise self._lifecycle_failure_locked("reset")
                if self._stateful and not self._resettable:
                    failure = self._failure_factory.create(
                        "reset_unsupported",
                        "stateful runtime does not support reset",
                        evidence=OutcomeEvidence.not_started("reset"),
                        details={"runtime_id": self.runtime_id},
                    )
                    self._record_failure_locked(failure)
                    raise failure
                previous = self._state
                self._admission_open = False
                self._transition_locked(LifecycleState.RESETTING)
                try:
                    self._wait_quiescent_locked(control_context)
                except (DeadlineExceeded, CancellationRequested) as exc:
                    self._transition_locked(previous)
                    self._admission_open = previous is LifecycleState.READY
                    failure = self._failure_factory.from_exception(
                        exc,
                        evidence=OutcomeEvidence.not_started("reset"),
                        details={"runtime_id": self.runtime_id},
                    )
                    self._record_failure_locked(failure)
                    raise failure from exc

            reset_started = False
            try:
                control_context.check("reset")
                if self._state_scope == "stream":
                    for record in self._open_stream_records():
                        self._reset_stream_record(record, control_context, from_runtime=True)
                for entry in reversed(self._entries):
                    if entry.resource is self._streaming_runtime:
                        continue
                    capabilities = getattr(entry.resource, "capabilities", None)
                    if (
                        self._assembly.retry_failed_reset
                        and entry.resource is not self._executor
                        and getattr(capabilities, "resettable", None) is False
                        and getattr(capabilities, "stateful", None) is False
                    ):
                        continue
                    if entry.resource is self._executor or callable(getattr(entry.resource, "reset", None)):
                        reset_started = True
                        _call_reset(entry.resource, control_context)
                control_context.check("reset")
                if self._assembly.reset_complete is not None:
                    self._assembly.reset_complete()
            except ExecutionFailure as failure:
                with self._condition:
                    if self._state is LifecycleState.RESETTING:
                        self._transition_locked(self._reset_failure_state())
                    self._admission_open = False
                    self._record_failure_locked(failure)
                raise
            except (DeadlineExceeded, CancellationRequested) as exc:
                evidence = OutcomeEvidence(
                    OutcomeState.STARTED if reset_started else OutcomeState.NOT_STARTED,
                    not reset_started,
                    reset_started and self._stateful,
                    "reset",
                )
                failure = self._failure_factory.from_exception(
                    exc, evidence=evidence, details={"runtime_id": self.runtime_id}
                )
                with self._condition:
                    if self._state is LifecycleState.RESETTING:
                        self._transition_locked(self._reset_failure_state())
                    self._admission_open = False
                    self._record_failure_locked(failure)
                raise failure from exc
            except Exception as exc:
                failure = self._failure_factory.from_exception(
                    exc,
                    code="reset_failed",
                    message=f"runtime reset failed: {exc}",
                    evidence=OutcomeEvidence.started("reset", outcome_known=False, state_mutated=self._stateful),
                    scope=RecoveryScope.RUNTIME,
                    action=RecoveryAction.RELOAD,
                    recoverable=self._reloadable,
                    details={"runtime_id": self.runtime_id},
                )
                with self._condition:
                    if self._state is LifecycleState.RESETTING:
                        self._transition_locked(self._reset_failure_state())
                    self._admission_open = False
                    self._record_failure_locked(failure)
                raise failure from exc
            else:
                with self._condition:
                    if self._state is LifecycleState.RESETTING:
                        self._transition_locked(LifecycleState.READY)
                    self._admission_open = True
                    self._recovery_requirement = None
                    self._recovery_available = False
                    self._condition.notify_all()

    def close(self, context: ExecutionContext | None = None, *, deadline: Deadline | None = None) -> None:
        control_context = self._control_context(context, deadline)
        with self._control_lock:
            with self._condition:
                if self._state is LifecycleState.CLOSED:
                    return
                release_unloaded = self._state is LifecycleState.CREATED or (
                    self._state is LifecycleState.CLOSING and not self._loading and not self._loaded_entries
                )
                self._admission_open = False
                if self._state is not LifecycleState.CLOSING:
                    self._transition_locked(LifecycleState.CLOSING)
                try:
                    self._wait_quiescent_locked(control_context)
                except (DeadlineExceeded, CancellationRequested) as exc:
                    failure = self._failure_factory.from_exception(
                        exc,
                        code="close_drain_failed",
                        message=f"runtime close could not drain active work: {exc}",
                        evidence=OutcomeEvidence.started("reset", outcome_known=False, state_mutated=False),
                        scope=RecoveryScope.RUNTIME,
                        action=RecoveryAction.RELOAD,
                        recoverable=self._reloadable,
                        details={"runtime_id": self.runtime_id, "close_pending": True},
                    )
                    self._close_error = failure
                    # Keep CLOSING so a later close can retry the drain.
                    raise failure from exc

            errors: list[BaseException] = []
            for record in self._open_stream_records():
                try:
                    self._close_stream_record(record, control_context, internal=True)
                except BaseException as exc:
                    errors.append(exc)
            entries_to_release = self._loaded_entries or (self._entries if release_unloaded else ())
            errors.extend(self._release_entries(tuple(reversed(entries_to_release))))
            if any(getattr(error, "close_pending", False) for error in errors):
                failure = self._failure_factory.create(
                    "close_drain_failed",
                    f"runtime {self.runtime_id!r} retains resources pending close drain",
                    evidence=OutcomeEvidence.started("reset", outcome_known=False, state_mutated=False),
                    details={
                        "runtime_id": self.runtime_id,
                        "close_pending": True,
                        "errors": tuple(str(error) for error in errors),
                    },
                )
                with self._condition:
                    self._loaded_entries = entries_to_release
                    self._close_error = failure
                raise failure
            if self._admission is not None:
                with suppress(BaseException):
                    self._admission.close()
            with self._condition:
                self._loaded_entries = ()
                self._admission_open = False
                self._transition_locked(LifecycleState.CLOSED)
                self._condition.notify_all()
            if errors:
                failure = self._failure_factory.create(
                    "close_failed",
                    f"runtime {self.runtime_id!r} close completed with {len(errors)} cleanup error(s)",
                    evidence=OutcomeEvidence.completed("reset"),
                    scope=RecoveryScope.RUNTIME,
                    action=RecoveryAction.RELOAD,
                    recoverable=self._reloadable,
                    details={"errors": tuple(str(error) for error in errors), "runtime_id": self.runtime_id},
                )
                with self._condition:
                    self._close_error = failure
                    self._record_failure_locked(failure)
                raise failure

    def _release_entries(self, entries: tuple[OwnedComponent, ...]) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        for entry in entries:
            resource_id = id(entry.resource)
            with self._condition:
                if resource_id in self._released_ids:
                    continue
                self._released_ids.add(resource_id)
            try:
                if entry.release is not None:
                    entry.release()
                elif entry.release_method == "release":
                    method = getattr(entry.resource, "release", None)
                    if callable(method):
                        _compatible_call(method, ((((), {})),))
                    else:
                        _call_release(entry.resource)
                elif entry.release_method == "unregister":
                    method = getattr(entry.resource, "unregister", None)
                    if callable(method):
                        _compatible_call(method, ((((), {})),))
                    else:
                        _call_release(entry.resource)
                else:
                    _call_close(entry.resource)
            except BaseException as exc:
                errors.append(exc)
                if getattr(exc, "close_pending", False):
                    with self._condition:
                        self._released_ids.discard(resource_id)
                    # Earlier acquisitions may provide buffers or leases to this resource.
                    break
        return tuple(errors)

    def _health_snapshot(self) -> RuntimeHealth:
        with self._condition:
            return RuntimeHealth(
                state=self._state,
                ready=self._state is LifecycleState.READY,
                reason_code=self._last_failure.code if self._last_failure is not None else None,
                message=self._last_failure.message if self._last_failure is not None else None,
                recoverable=(
                    (self._last_failure.recoverable if self._last_failure is not None else False)
                    or self._recovery_available
                ),
                failure_count=self._failure_count,
                last_successful_at=self._last_successful_at,
                recovery_requirement=self._recovery_requirement,
            )

    @property
    def health(self) -> RuntimeHealth:
        return self._health_snapshot()

    @property
    def health_status(self) -> RuntimeHealth:
        return self.health

    get_health = _health_snapshot

    def diagnostics(self) -> RuntimeDiagnostics:
        with self._condition:
            health = self.health
            streams = tuple(record.diagnostics() for record in self._streams.values())
            return RuntimeDiagnostics(
                runtime_id=self.runtime_id,
                state=self._state,
                health=health,
                active_executions=self._active_executions,
                open_streams=sum(record.state is not StreamState.CLOSED for record in self._streams.values()),
                recovery_requirement=self._recovery_requirement,
                last_failure=self._last_failure,
                last_outcome=self._last_outcome,
                stream_diagnostics=streams,
                identity=self._identity,
                execution_contract=self._execution_contract,
                deployment_fingerprint=self._deployment_fingerprint,
                runtime_profile_fingerprint=self._runtime_profile_fingerprint,
                artifact_integrity=self._artifact_integrity,
                runtime_version=self._runtime_version,
                capabilities={
                    **self._declared_capabilities,
                    "stateful": self._stateful,
                    "resettable": self._resettable,
                    "state_scope": self._state_scope,
                    "state_bank_mode": self._state_bank_mode,
                    "max_open_streams": self._max_open_streams,
                    "cancellation_granularity": self._cancellation_granularity,
                },
            )

    get_diagnostics = diagnostics

    # ---- Streaming boundary -------------------------------------------------

    def open_stream(self, context: ExecutionContext) -> StreamHandle:
        context = self._context_or_type_error(context)
        tracker = OutcomeEvidenceTracker()
        with self._condition:
            if self._state is not LifecycleState.READY or not self._admission_open:
                raise self._lifecycle_failure_locked("open_stream")
            if self._state_scope != "stream" or self._streaming_runtime is None:
                raise self._failure_factory.create(
                    "stream_unsupported",
                    "runtime does not expose a StreamingRuntime",
                    evidence=OutcomeEvidence.not_started("admission"),
                )
            open_count = sum(record.state is not StreamState.CLOSED for record in self._streams.values())
            if open_count + self._stream_openings >= int(self._max_open_streams or 1):
                raise self._failure_factory.create(
                    "stream_capacity_exhausted",
                    "stream admission limit is exhausted",
                    evidence=OutcomeEvidence.not_started("admission"),
                    scope=RecoveryScope.STREAM,
                    action=RecoveryAction.NONE,
                    details={"max_open_streams": self._max_open_streams},
                )
            self._stream_openings += 1
        try:
            try:
                context.check("stream_admission")
                backend_handle = _call_stream_open(self._streaming_runtime, context)
            except (DeadlineExceeded, CancellationRequested) as exc:
                failure = self._failure_factory.from_exception(
                    exc,
                    evidence=tracker.mark_failed(exc.phase, state=OutcomeState.NOT_STARTED),
                    scope=RecoveryScope.STREAM,
                    details={"operation": "open_stream"},
                )
                self._remember_failure(failure)
                raise failure from exc
            except ExecutionFailure as failure:
                self._remember_failure(failure)
                raise
            except Exception as exc:
                failure = self._failure_factory.from_exception(
                    exc,
                    evidence=tracker.mark_failed("backend", state=OutcomeState.NOT_STARTED),
                    scope=RecoveryScope.STREAM,
                    details={"operation": "open_stream"},
                )
                self._remember_failure(failure)
                raise failure from exc

            public_handle: StreamHandle
            if isinstance(backend_handle, StreamHandle):
                public_handle = backend_handle
            elif isinstance(backend_handle, str) and backend_handle:
                public_handle = StreamHandle(backend_handle)
            elif backend_handle is None:
                with self._condition:
                    self._stream_counter += 1
                    stream_id = f"stream-{self._stream_counter}"
                public_handle = StreamHandle(stream_id)
                backend_handle = public_handle
            else:
                stream_id = getattr(backend_handle, "stream_id", None)
                if not isinstance(stream_id, str) or not stream_id:
                    with self._condition:
                        self._stream_counter += 1
                        stream_id = f"stream-{self._stream_counter}"
                public_handle = StreamHandle(stream_id)

            try:
                public_handle._bind_owner(self._stream_owner)
            except ValueError:
                # A provider may return a handle it previously owned.  Preserve
                # the provider handle internally but expose a fresh owner-bound ID.
                public_handle = StreamHandle(public_handle.stream_id)
                public_handle._bind_owner(self._stream_owner)
            record = _StreamRecord(public_handle, backend_handle, str(self._state_bank_mode), _now())
            with self._condition:
                if self._state is not LifecycleState.READY or not self._admission_open:
                    with suppress(Exception):
                        _call_stream_control(self._streaming_runtime, "close_stream", backend_handle, context)
                    raise self._lifecycle_failure_locked("open_stream")
                if public_handle.stream_id in self._streams:
                    with suppress(Exception):
                        _call_stream_control(self._streaming_runtime, "close_stream", backend_handle, context)
                    raise self._failure_factory.create(
                        "stream_id_conflict",
                        f"stream ID {public_handle.stream_id!r} is already open",
                        evidence=OutcomeEvidence.not_started("admission"),
                        scope=RecoveryScope.STREAM,
                    )
                self._streams[public_handle.stream_id] = record
                self._condition.notify_all()
            return public_handle
        finally:
            with self._condition:
                self._stream_openings -= 1
                self._condition.notify_all()

    def _resolve_stream(self, stream_handle: StreamHandle | str, operation: str) -> _StreamRecord:
        stream_id = stream_handle.stream_id if isinstance(stream_handle, StreamHandle) else stream_handle
        if not isinstance(stream_id, str) or not stream_id:
            raise self._failure_factory.create(
                "stream_not_found",
                f"{operation} received an invalid stream handle",
                evidence=OutcomeEvidence.not_started("admission"),
                scope=RecoveryScope.STREAM,
            )
        with self._condition:
            record = self._streams.get(stream_id)
            if record is None:
                raise self._failure_factory.create(
                    "stream_not_found",
                    f"stream {stream_id!r} was not found",
                    evidence=OutcomeEvidence.not_started("admission"),
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": stream_id, "operation": operation},
                )
            if isinstance(stream_handle, StreamHandle) and not stream_handle._belongs_to(self._stream_owner):
                raise self._failure_factory.create(
                    "stream_not_found",
                    f"stream {stream_id!r} is not owned by this runtime",
                    evidence=OutcomeEvidence.not_started("admission"),
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": stream_id, "operation": operation},
                )
            if record.state is StreamState.CLOSED:
                raise self._failure_factory.create(
                    "stream_closed",
                    f"stream {stream_id!r} is closed",
                    evidence=OutcomeEvidence.not_started("admission"),
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": stream_id, "operation": operation},
                )
            return record

    def step(self, stream_handle: StreamHandle | str, request: ModelRequest, context: ExecutionContext) -> ModelResult:
        if not isinstance(request, ModelRequest):
            raise TypeError("stream step requires a ModelRequest")
        context = self._context_or_type_error(context)
        record = self._resolve_stream(stream_handle, "step")
        with self._condition:
            if self._state is not LifecycleState.READY or not self._admission_open:
                raise self._lifecycle_failure_locked("step")
            if record.state is StreamState.RESET_REQUIRED:
                recovery = record.recovery_requirement or RecoveryRequirement(
                    RecoveryScope.STREAM, RecoveryAction.RESET_STREAM
                )
                raise self._failure_factory.create(
                    "recovery_required",
                    f"stream {record.public_handle.stream_id!r} requires reset",
                    evidence=OutcomeEvidence.not_started("admission"),
                    recovery=recovery,
                    recoverable=recovery.action is not RecoveryAction.NONE,
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": record.public_handle.stream_id},
                )
            if record.state is StreamState.FAILED:
                recovery = record.recovery_requirement or RecoveryRequirement(
                    RecoveryScope.STREAM, RecoveryAction.RELOAD
                )
                raise self._failure_factory.create(
                    "recovery_required",
                    f"stream {record.public_handle.stream_id!r} has failed",
                    evidence=OutcomeEvidence.not_started("admission"),
                    recovery=recovery,
                    recoverable=(
                        self._reloadable
                        if recovery.action is RecoveryAction.RELOAD
                        else recovery.action is not RecoveryAction.NONE
                    ),
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": record.public_handle.stream_id},
                )
            if record.resetting or record.closing:
                raise self._failure_factory.create(
                    "stream_reentrant",
                    f"stream {record.public_handle.stream_id!r} is in a control operation",
                    evidence=OutcomeEvidence.not_started("admission"),
                    scope=RecoveryScope.STREAM,
                )
            if record.active_step:
                raise self._failure_factory.create(
                    "stream_reentrant",
                    f"stream {record.public_handle.stream_id!r} already has an active step",
                    evidence=OutcomeEvidence.not_started("admission"),
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": record.public_handle.stream_id},
                )
        try:
            context.check("stream_step_admission")
        except (DeadlineExceeded, CancellationRequested) as exc:
            failure = self._failure_factory.from_exception(
                exc,
                evidence=OutcomeEvidence.not_started(exc.phase),
                scope=RecoveryScope.STREAM,
                details={"stream_id": record.public_handle.stream_id},
            )
            self._remember_failure(failure)
            raise failure from exc

        with self._condition:
            if record.active_step or record.resetting or record.closing:
                raise self._failure_factory.create(
                    "stream_reentrant",
                    f"stream {record.public_handle.stream_id!r} already has an active operation",
                    evidence=OutcomeEvidence.not_started("admission"),
                    scope=RecoveryScope.STREAM,
                )
            record.active_step = True
            record.active_thread = threading.get_ident()
            record.set_state(StreamState.STEPPING)
            self._active_stream_steps += 1

        tracker = OutcomeEvidenceTracker()
        started = False
        started_at = time.perf_counter()
        try:
            tracker.mark_started("backend", state_mutated=self._stateful, outcome_known=not self._stateful)
            started = True
            frame = _call_stream_step(self._streaming_runtime, record.backend_handle, request, context)
            if isinstance(frame, ExecutionFailure):
                raise frame
            tracker.mark_completed("backend")
            try:
                result = _call_adapter(
                    self._adapter, frame, context, tracker.snapshot(), (time.perf_counter() - started_at) * 1000.0
                )
                if not isinstance(result, ModelResult):
                    raise TypeError("result adapter did not return ModelResult")
            except ExecutionFailure:
                raise
            except Exception as exc:
                failure_code = str(getattr(exc, "code", "adaptation_failed"))
                failure_phase = "output_validation" if failure_code == "output_validation_failed" else "adaptation"
                evidence = tracker.mark_failed(
                    failure_phase,
                    state=OutcomeState.COMPLETED,
                    outcome_known=True,
                    state_mutated=self._stateful,
                )
                failure = self._failure_factory.from_exception(
                    exc,
                    code=failure_code,
                    message=f"stream result adaptation failed: {exc}",
                    evidence=evidence,
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": record.public_handle.stream_id},
                )
                raise failure from exc
            result = result.with_evidence(
                tracker.mark_completed("adaptation", stream_id=record.public_handle.stream_id)
            )
            try:
                context.check("stream_step_result")
            except (DeadlineExceeded, CancellationRequested) as exc:
                evidence = tracker.mark_failed(
                    exc.phase,
                    state=OutcomeState.COMPLETED,
                    outcome_known=True,
                    state_mutated=self._stateful,
                )
                failure = self._failure_factory.from_exception(
                    exc,
                    evidence=evidence,
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": record.public_handle.stream_id},
                )
                raise failure from exc
            with self._condition:
                record.last_step_at = _now()
                record.recovery_requirement = None
                self._last_successful_at = record.last_step_at
                self._last_outcome = result.evidence
            return result
        except ExecutionFailure as failure:
            normalized = self._handle_stream_failure(record, failure, started)
            raise normalized from None
        except (DeadlineExceeded, CancellationRequested) as exc:
            evidence = tracker.mark_failed(
                exc.phase,
                state=OutcomeState.STARTED if started else OutcomeState.NOT_STARTED,
                outcome_known=not started,
                state_mutated=started and self._stateful,
            )
            failure = self._failure_factory.from_exception(
                exc,
                evidence=evidence,
                scope=RecoveryScope.STREAM,
                details={"stream_id": record.public_handle.stream_id},
            )
            normalized = self._handle_stream_failure(record, failure, started)
            raise normalized from exc
        except Exception as exc:
            started_flag = bool(getattr(exc, "operation_started", started))
            known = bool(getattr(exc, "outcome_known", not (started_flag and self._stateful)))
            mutated = bool(getattr(exc, "state_mutated", started_flag and self._stateful))
            evidence = tracker.mark_failed(
                str(getattr(exc, "phase", "backend")),
                state=OutcomeState.STARTED if started_flag else OutcomeState.NOT_STARTED,
                outcome_known=known,
                state_mutated=mutated,
            )
            failure = self._failure_factory.from_exception(
                exc,
                evidence=evidence,
                scope=RecoveryScope.STREAM,
                details={"stream_id": record.public_handle.stream_id},
            )
            normalized = self._handle_stream_failure(record, failure, started_flag)
            raise normalized from exc
        finally:
            with self._condition:
                record.active_step = False
                record.active_thread = None
                if record.state is StreamState.STEPPING:
                    record.set_state(StreamState.OPEN)
                self._active_stream_steps -= 1
                self._condition.notify_all()

    def _handle_stream_failure(
        self, record: _StreamRecord, failure: ExecutionFailure, started: bool
    ) -> ExecutionFailure:
        evidence = failure.evidence
        uncertain = started and evidence.state is OutcomeState.STARTED and not evidence.outcome_known
        if uncertain:
            if record.state_bank_mode == "per_stream":
                if self._resettable:
                    recovery = RecoveryRequirement(RecoveryScope.STREAM, RecoveryAction.RESET_STREAM)
                    recoverable = True
                else:
                    recovery = RecoveryRequirement(RecoveryScope.STREAM, RecoveryAction.CLOSE_STREAM)
                    recoverable = True
                with self._condition:
                    record.set_state(StreamState.RESET_REQUIRED)
                    record.recovery_requirement = recovery
                    record.last_failure_code = failure.code
                normalized = self._failure_factory.create(
                    failure.code,
                    failure.message,
                    evidence=evidence,
                    recovery=recovery,
                    recoverable=recoverable,
                    cause=failure.cause or failure,
                    details={**dict(failure.details), "stream_id": record.public_handle.stream_id},
                )
            else:
                if self._resettable:
                    recovery = RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction.RESET_RUNTIME)
                    recoverable = True
                    target_state = LifecycleState.RESET_REQUIRED
                else:
                    recovery = RecoveryRequirement(RecoveryScope.RUNTIME, RecoveryAction.RELOAD)
                    recoverable = self._reloadable
                    target_state = LifecycleState.FAILED
                with self._condition:
                    record.set_state(
                        StreamState.RESET_REQUIRED
                        if target_state is LifecycleState.RESET_REQUIRED
                        else StreamState.FAILED
                    )
                    record.recovery_requirement = recovery
                    record.last_failure_code = failure.code
                    self._admission_open = False
                    if self._state is LifecycleState.READY:
                        self._transition_locked(target_state)
                    self._recovery_requirement = recovery
                    self._recovery_available = recoverable
                normalized = self._failure_factory.create(
                    failure.code,
                    failure.message,
                    evidence=evidence,
                    recovery=recovery,
                    recoverable=recoverable,
                    cause=failure.cause or failure,
                    details={**dict(failure.details), "stream_id": record.public_handle.stream_id},
                )
        else:
            normalized = failure
            with self._condition:
                record.last_failure_code = failure.code
        self._remember_failure(normalized)
        return normalized

    def reset_stream(
        self,
        stream_handle: StreamHandle | str,
        context: ExecutionContext,
        *,
        _from_runtime: bool = False,
    ) -> None:
        context = self._context_or_type_error(context)
        record = self._resolve_stream(stream_handle, "reset_stream")
        was_reset_required = record.state is StreamState.RESET_REQUIRED
        runtime_resetting = False
        with record.control_lock:
            with self._condition:
                if self._state in {LifecycleState.CLOSED, LifecycleState.CLOSING, LifecycleState.FAILED} or (
                    self._state is LifecycleState.RESETTING and not _from_runtime
                ):
                    raise self._lifecycle_failure_locked("reset_stream")
                if record.state is StreamState.FAILED:
                    recovery = record.recovery_requirement or RecoveryRequirement(
                        RecoveryScope.STREAM, RecoveryAction.RELOAD
                    )
                    raise self._failure_factory.create(
                        "recovery_required",
                        f"stream {record.public_handle.stream_id!r} cannot be reset after failure",
                        evidence=OutcomeEvidence.not_started("reset"),
                        recovery=recovery,
                        recoverable=self._reloadable if recovery.action is RecoveryAction.RELOAD else False,
                        scope=RecoveryScope.STREAM,
                        details={"stream_id": record.public_handle.stream_id},
                    )
                if record.active_step:
                    if record.active_thread == threading.get_ident():
                        raise self._failure_factory.create(
                            "stream_reentrant",
                            "cannot reset a stream from its active step",
                            evidence=OutcomeEvidence.not_started("reset"),
                            scope=RecoveryScope.STREAM,
                        )
                    try:
                        self._wait_for_stream_quiescence_locked(record, context)
                    except (DeadlineExceeded, CancellationRequested) as exc:
                        failure = self._failure_factory.from_exception(
                            exc,
                            evidence=OutcomeEvidence.not_started("reset"),
                            scope=RecoveryScope.STREAM,
                            details={"stream_id": record.public_handle.stream_id},
                        )
                        self._remember_failure(failure)
                        raise failure from exc
                record.resetting = True
                record.set_state(StreamState.RESETTING)
                if self._state is LifecycleState.RESET_REQUIRED and record.state_bank_mode == "runtime_exclusive":
                    self._transition_locked(LifecycleState.RESETTING)
                    self._admission_open = False
                    runtime_resetting = True
            if not self._resettable and self._stateful:
                failure = self._failure_factory.create(
                    "reset_unsupported",
                    f"stream {record.public_handle.stream_id!r} does not support reset",
                    evidence=OutcomeEvidence.not_started("reset"),
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": record.public_handle.stream_id},
                )
                with self._condition:
                    record.resetting = False
                    record.set_state(StreamState.RESET_REQUIRED if was_reset_required else StreamState.OPEN)
                    record.recovery_requirement = failure.recovery
                    if runtime_resetting:
                        self._transition_locked(LifecycleState.FAILED)
                self._remember_failure(failure)
                raise failure
            reset_started = False
            try:
                context.check("reset")
                reset_started = True
                _call_stream_control(self._streaming_runtime, "reset_stream", record.backend_handle, context)
                context.check("reset")
            except ExecutionFailure as failure:
                with self._condition:
                    record.resetting = False
                    record.set_state(StreamState.FAILED)
                    record.recovery_requirement = RecoveryRequirement(RecoveryScope.STREAM, RecoveryAction.RELOAD)
                    record.last_failure_code = failure.code
                    if runtime_resetting:
                        self._transition_locked(LifecycleState.FAILED)
                        self._admission_open = False
                self._remember_failure(failure)
                raise
            except (DeadlineExceeded, CancellationRequested) as exc:
                evidence = OutcomeEvidence(
                    OutcomeState.STARTED if reset_started else OutcomeState.NOT_STARTED,
                    not reset_started,
                    reset_started and self._stateful,
                    "reset",
                )
                failure = self._failure_factory.from_exception(
                    exc,
                    evidence=evidence,
                    scope=RecoveryScope.STREAM,
                    details={"stream_id": record.public_handle.stream_id},
                )
                with self._condition:
                    record.resetting = False
                    record.set_state(StreamState.FAILED if reset_started else StreamState.RESET_REQUIRED)
                    record.recovery_requirement = RecoveryRequirement(RecoveryScope.STREAM, RecoveryAction.RELOAD)
                    record.last_failure_code = failure.code
                    if runtime_resetting:
                        self._transition_locked(LifecycleState.FAILED)
                        self._admission_open = False
                self._remember_failure(failure)
                raise failure from exc
            except Exception as exc:
                failure = self._failure_factory.from_exception(
                    exc,
                    code="reset_failed",
                    message=f"stream reset failed: {exc}",
                    evidence=OutcomeEvidence.started("reset", outcome_known=False, state_mutated=self._stateful),
                    scope=RecoveryScope.STREAM,
                    action=RecoveryAction.RELOAD,
                    recoverable=self._reloadable,
                    details={"stream_id": record.public_handle.stream_id},
                )
                with self._condition:
                    record.resetting = False
                    record.set_state(StreamState.FAILED)
                    record.recovery_requirement = failure.recovery
                    record.last_failure_code = failure.code
                    if runtime_resetting:
                        self._transition_locked(LifecycleState.FAILED)
                        self._admission_open = False
                self._remember_failure(failure)
                raise failure from exc
            else:
                with self._condition:
                    record.host_state.clear()
                    record.last_step_at = None
                    record.resetting = False
                    record.recovery_requirement = None
                    record.last_failure_code = None
                    record.set_state(StreamState.OPEN)
                    if runtime_resetting:
                        self._transition_locked(LifecycleState.READY)
                        self._admission_open = True
                    self._condition.notify_all()

    def _wait_for_stream_quiescence_locked(self, record: _StreamRecord, context: ExecutionContext) -> None:
        while record.active_step:
            context.check("control_drain")
            remaining = context.deadline.remaining_seconds()
            self._condition.wait(timeout=0.05 if remaining is None else min(0.05, remaining))

    def _reset_stream_record(
        self,
        record: _StreamRecord,
        context: ExecutionContext,
        *,
        from_runtime: bool = False,
    ) -> None:
        del from_runtime
        self.reset_stream(record.public_handle, context, _from_runtime=True)

    def close_stream(
        self,
        stream_handle: StreamHandle | str,
        context: ExecutionContext | None = None,
        *,
        deadline: Deadline | None = None,
    ) -> None:
        stream_id = stream_handle.stream_id if isinstance(stream_handle, StreamHandle) else stream_handle
        with self._condition:
            record = self._streams.get(stream_id) if isinstance(stream_id, str) else None
            if record is not None and record.state is StreamState.CLOSED:
                return
        control_context = self._control_context(context, deadline)
        record = self._resolve_stream(stream_handle, "close_stream")
        self._close_stream_record(record, control_context)

    def _close_stream_record(self, record: _StreamRecord, context: ExecutionContext, *, internal: bool = False) -> None:
        with record.control_lock:
            with self._condition:
                if record.state is StreamState.CLOSED:
                    return
                if record.active_step:
                    if record.active_thread == threading.get_ident():
                        raise self._failure_factory.create(
                            "stream_reentrant",
                            "cannot close a stream from its active step",
                            evidence=OutcomeEvidence.not_started("reset"),
                            scope=RecoveryScope.STREAM,
                        )
                    self._wait_for_stream_quiescence_locked(record, context)
                record.closing = True
            close_error: BaseException | None = None
            try:
                if not internal:
                    context.check("close_stream")
                _call_stream_control(self._streaming_runtime, "close_stream", record.backend_handle, context)
            except BaseException as exc:
                close_error = exc
            finally:
                with self._condition:
                    record.host_state.clear()
                    record.closing = False
                    record.set_state(StreamState.CLOSED)
                    record.recovery_requirement = None
                    record.last_failure_code = getattr(close_error, "code", None)
                    self._condition.notify_all()
            if close_error is not None:
                failure = (
                    close_error
                    if isinstance(close_error, ExecutionFailure)
                    else self._failure_factory.from_exception(
                        close_error,
                        code="stream_close_failed",
                        message=f"stream {record.public_handle.stream_id!r} close failed: {close_error}",
                        evidence=OutcomeEvidence.completed("reset"),
                        scope=RecoveryScope.STREAM,
                        details={"stream_id": record.public_handle.stream_id},
                    )
                )
                self._remember_failure(failure)
                raise failure from close_error

    def _open_stream_records(self) -> tuple[_StreamRecord, ...]:
        with self._condition:
            return tuple(record for record in self._streams.values() if record.state is not StreamState.CLOSED)

    def stream_diagnostics(
        self, stream_handle: StreamHandle | str | None = None
    ) -> StreamDiagnostics | tuple[StreamDiagnostics, ...]:
        with self._condition:
            if stream_handle is None:
                return tuple(record.diagnostics() for record in self._streams.values())
        record = self._resolve_stream(stream_handle, "diagnostics")
        return record.diagnostics()

    get_stream_diagnostics = stream_diagnostics

    @property
    def open_stream_count(self) -> int:
        with self._condition:
            return sum(record.state is not StreamState.CLOSED for record in self._streams.values())


__all__ = ["LifecycleState", "ModelRuntimeHandle"]

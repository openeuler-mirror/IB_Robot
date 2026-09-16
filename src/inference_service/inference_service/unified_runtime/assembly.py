"""Transferable runtime assembly and ownership descriptions."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .adapters import ResultAdapter
from .contracts import ExecutionContext, ModelRequest
from .errors import ExecutionFailureFactory


@runtime_checkable
class RuntimeExecutor(Protocol):
    """Minimal executor protocol used by ``ModelRuntimeHandle``."""

    def execute(self, request: ModelRequest, context: ExecutionContext) -> object: ...


@runtime_checkable
class Loadable(Protocol):
    def load(self, context: object) -> None: ...


@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


@dataclass(frozen=True)
class OwnedComponent:
    """One resource in the assembly acquisition order."""

    resource: object
    name: str = "component"
    release_method: str | None = None
    release: Callable[[], object] | None = None
    # Composite assemblies may load each role Session with its own validated
    # RuntimeContext while retaining one handle-level load barrier.
    load_context: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.resource is None:
            raise ValueError("owned component resource cannot be None")
        if not self.name:
            raise ValueError("owned component name must be non-empty")
        if callable(self.release_method) and self.release is None:  # type: ignore[arg-type]
            object.__setattr__(self, "release", self.release_method)
            object.__setattr__(self, "release_method", None)
        if self.release is not None and not callable(self.release):
            raise TypeError("owned component release must be callable")


@dataclass
class RuntimeAssembly:
    """Public transfer object consumed by ``ModelRuntimeHandle``.

    ``owned_components`` is authoritative when supplied and is interpreted as
    acquisition order.  Cleanup always traverses that order in reverse.  The
    named fields are convenient for factory implementations and are converted
    to the same order when no explicit list is provided.

    ``process_providers``/``providers`` are diagnostic references only.  They
    are deliberately excluded from ``component_entries`` and are closed by
    the composition root after all handles release their leases.
    """

    runtime_executor: object | None = None
    executor: object | None = None
    streaming_runtime: object | None = None
    session: object | None = None
    role_assemblies: Mapping[str, object] = field(default_factory=dict)
    artifact_bindings: Mapping[str, object] = field(default_factory=dict)
    request_adapter: object | None = None
    adapter: object | None = None
    result_adapter: object | None = None
    failure_factory: ExecutionFailureFactory | None = None
    processor: object | None = None
    worker: object | None = None
    host_resources: tuple[object, ...] = ()
    host_resource: object | None = None
    device_lease: object | None = None
    provider_leases: tuple[object, ...] = ()
    provider_lease: object | None = None
    provider_registrations: tuple[object, ...] = ()
    provider_registration: object | None = None
    provider_leases_and_registrations: tuple[object, ...] = ()
    process_providers: tuple[object, ...] = ()
    providers: object | None = None
    owned_components: tuple[object, ...] = ()
    components: tuple[object, ...] = ()
    stateful: bool = False
    resettable: bool = False
    # Scheduled drain recovery discovers reset support and skips stateless
    # non-resettable resources; disabled runtimes retain legacy reset behavior.
    retry_failed_reset: bool = False
    # Runs after component reset succeeds, before request admission reopens.
    reset_complete: Callable[[], None] | None = None
    state_scope: str = "request"
    state_bank_mode: str | None = None
    max_open_streams: int | None = None
    cancellation_granularity: str = "request_boundary"
    max_active_executions: int = 1
    reloadable: bool = False
    runtime_id: str | None = None
    identity: object | None = None
    execution_contract: object | None = None
    declared_capabilities: Mapping[str, object] = field(default_factory=dict)
    capabilities: object | None = None
    deployment_fingerprint: str | None = None
    runtime_profile_fingerprint: str | None = None
    artifact_integrity: object | None = None
    runtime_version: str | None = None
    load_context: object | None = None
    _claimed: bool = field(default=False, init=False, repr=False)
    _claim_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.runtime_executor is None:
            self.runtime_executor = self.executor
        elif self.executor is not None and self.executor is not self.runtime_executor:
            raise ValueError("runtime_executor and executor refer to different objects")
        if self.runtime_executor is None:
            raise ValueError("RuntimeAssembly requires a runtime executor")
        if self.request_adapter is None:
            self.request_adapter = self.result_adapter or self.adapter
        if self.failure_factory is None:
            self.failure_factory = ExecutionFailureFactory()
        self.host_resources = tuple(self.host_resources) + (
            (self.host_resource,) if self.host_resource is not None else ()
        )
        self.provider_leases = tuple(self.provider_leases) + (
            (self.provider_lease,) if self.provider_lease is not None else ()
        )
        self.provider_registrations = tuple(self.provider_registrations) + (
            (self.provider_registration,) if self.provider_registration is not None else ()
        )
        self.provider_leases_and_registrations = tuple(self.provider_leases_and_registrations)
        self.process_providers = tuple(self.process_providers)
        if isinstance(self.owned_components, Mapping):
            self.owned_components = tuple(self.owned_components.items())
        else:
            self.owned_components = tuple(self.owned_components)
        if isinstance(self.components, Mapping):
            self.components = tuple(self.components.items())
        else:
            self.components = tuple(self.components)
        if self.state_scope not in {"request", "stream"}:
            raise ValueError("state_scope must be request or stream")
        if self.max_active_executions < 1:
            raise ValueError("max_active_executions must be positive")

    @property
    def resolved_executor(self) -> object:
        return self.runtime_executor

    @property
    def resolved_adapter(self) -> object:
        return self.request_adapter or ResultAdapter()

    def claim_ownership(self) -> None:
        with self._claim_lock:
            if self._claimed:
                raise RuntimeError("runtime assembly ownership has already been transferred")
            self._claimed = True

    transfer_ownership = claim_ownership
    transfer = claim_ownership

    @property
    def ownership_transferred(self) -> bool:
        with self._claim_lock:
            return self._claimed

    @property
    def result_adapter_instance(self) -> object:
        return self.request_adapter or ResultAdapter()

    def component_entries(self) -> tuple[OwnedComponent, ...]:
        raw: list[OwnedComponent | object] = []
        if self.owned_components:
            raw.extend(self.owned_components)
        elif self.components:
            raw.extend(self.components)
        else:
            # Provider leases are acquired before dependent runtime resources
            # so reverse-order cleanup releases sessions/workers before their
            # leases.  A factory may still provide an explicit list when its
            # dependency graph is more detailed.
            raw.extend(("provider_lease", lease, "release") for lease in self.provider_leases)
            raw.extend(
                ("provider_registration", registration, "release") for registration in self.provider_registrations
            )
            raw.extend(
                ("provider_lease_or_registration", item, "release") for item in self.provider_leases_and_registrations
            )
            if self.device_lease is not None:
                raw.append(("device_lease", self.device_lease, "release"))
            raw.append(("session", self.session, None))
            raw.extend((f"role_assembly:{role}", resource, None) for role, resource in self.role_assemblies.items())
            raw.append(("runtime_executor", self.runtime_executor, None))
            raw.append(("streaming_runtime", self.streaming_runtime, None))
            raw.append(("adapter", self.request_adapter, None))
            raw.extend(
                (f"artifact_binding:{role}", resource, None) for role, resource in self.artifact_bindings.items()
            )
            raw.append(("processor", self.processor, None))
            raw.append(("worker", self.worker, None))
            raw.extend(("host_resource", resource, None) for resource in self.host_resources)

        entries: list[OwnedComponent] = []
        seen: set[int] = set()
        process_provider_ids = {id(provider) for provider in self.process_providers}
        if self.providers is not None:
            for provider_name in ("acl_runtime_provider", "resource_admission_provider"):
                provider = getattr(self.providers, provider_name, None)
                if provider is not None:
                    process_provider_ids.add(id(provider))
        for item in raw:
            if isinstance(item, OwnedComponent):
                entry = item
            elif isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], str):
                name, resource, release_method = item
                if resource is None:
                    continue
                entry = OwnedComponent(resource, name, release_method)
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                name, resource = item
                if resource is None:
                    continue
                entry = OwnedComponent(resource, name)
            else:
                if item is None:
                    continue
                entry = OwnedComponent(item, type(item).__name__)
            if id(entry.resource) in process_provider_ids:
                continue
            if id(entry.resource) in seen:
                continue
            seen.add(id(entry.resource))
            entries.append(entry)
        return tuple(entries)


@dataclass
class _RuntimeProviderState:
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    closed: bool = False


@dataclass(frozen=True)
class RuntimeProviders:
    """Composition-root dependencies; providers themselves are never owned by a handle."""

    acl_runtime_provider: object
    resource_admission_provider: object
    _state: _RuntimeProviderState = field(default_factory=_RuntimeProviderState, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.acl_runtime_provider is None:
            raise TypeError("acl_runtime_provider must be explicitly supplied")
        if self.resource_admission_provider is None:
            raise TypeError("resource_admission_provider must be explicitly supplied")

    @classmethod
    def create(cls, acl_runtime_provider: object, resource_admission_provider: object) -> RuntimeProviders:
        """Construct the process-level dependency value at the composition root."""

        return cls(acl_runtime_provider, resource_admission_provider)

    @property
    def closed(self) -> bool:
        with self._state.lock:
            return self._state.closed

    @staticmethod
    def _close_provider(provider: object) -> None:
        for method_name in ("close", "shutdown", "finalize"):
            method = getattr(provider, method_name, None)
            if callable(method):
                method()
                return

    def close(self) -> None:
        """Close providers once, in composition-root shutdown order.

        Runtime handles never call this method.  The resource provider is
        closed before ACL so all admission registrations are gone before the
        process-level runtime is finalized.
        """

        with self._state.lock:
            if self._state.closed:
                return
            self._state.closed = True
        errors: list[BaseException] = []
        seen: set[int] = set()
        for provider in (self.resource_admission_provider, self.acl_runtime_provider):
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            try:
                self._close_provider(provider)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"runtime provider shutdown failed: {errors[0]}") from errors[0]

    shutdown = close

    def __enter__(self) -> RuntimeProviders:
        return self

    def __exit__(self, _exc_type: object, _exc_value: object, _traceback: object) -> None:
        self.close()


__all__ = ["Closable", "Loadable", "OwnedComponent", "RuntimeAssembly", "RuntimeExecutor", "RuntimeProviders"]

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from inference_service.unified_runtime import (
    CompositeRuntimeDescriptor,
    CompositeRuntimeKey,
    ModelRuntimeHandle,
    ModelRuntimeKey,
    RegistryFrozenError,
    RegistrySet,
    RoleBackendProfile,
    RuntimeAssemblerRegistry,
    RuntimeAssembly,
    RuntimeDescriptor,
    RuntimeProviders,
    RuntimeRegistryError,
    RuntimeRoleSelector,
    SessionBuilderKey,
    SessionBuilderRegistry,
)


class FakeBackendRegistry:
    names = ("torch", "ascend")


class TorchProfile:
    backend_name = "torch"


class AscendProfile:
    backend_name = "ascend"


class ClosableResource:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def load(self, _context: object) -> None:
        self.events.append(f"load:{self.name}")

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


def _role_key(
    model_type: str = "act",
    *,
    operation: str = "predict",
    backend: str = "torch",
    contract: str = "request-direct",
    visibility: str | None = None,
) -> ModelRuntimeKey:
    return ModelRuntimeKey("policy", model_type, operation, backend, contract, visibility)


def _session_key(
    model_type: str = "act",
    *,
    operation: str = "predict",
    backend: str = "torch",
) -> SessionBuilderKey:
    return SessionBuilderKey("policy", model_type, operation, backend)


def _assembly(*, contract: str | None = None, events: list[str] | None = None) -> RuntimeAssembly:
    events = events if events is not None else []
    executor = ClosableResource("executor", events)
    return RuntimeAssembly(
        runtime_executor=executor,
        execution_contract=contract,
        runtime_id="test-runtime",
    )


def _role_descriptor(
    key: ModelRuntimeKey,
    session_key: SessionBuilderKey | None = None,
    *,
    profile_type: type = TorchProfile,
    assembler: object | None = None,
) -> RuntimeDescriptor:
    selected_assembler = assembler
    if selected_assembler is None:

        def selected_assembler(*, providers: RuntimeProviders) -> RuntimeAssembly:
            del providers
            return _assembly(contract=key.execution_contract)

        selected_assembler.execution_contract = key.execution_contract  # type: ignore[attr-defined]
    return RuntimeDescriptor(
        key=key,
        session_builder_key=session_key or key.session_builder_key,
        profile_type=profile_type,
        assembler=selected_assembler,
    )


def _registry_set(
    *,
    add_session: bool = True,
    runtime: RuntimeAssemblerRegistry | None = None,
) -> tuple[RegistrySet, SessionBuilderRegistry, RuntimeAssemblerRegistry]:
    sessions = SessionBuilderRegistry()
    if add_session:
        sessions.register(_session_key(), lambda _context: object())
    assemblers = runtime or RuntimeAssemblerRegistry()
    assemblers.register(_role_descriptor(_role_key()))
    return RegistrySet(FakeBackendRegistry(), sessions, assemblers), sessions, assemblers


def test_runtime_keys_canonicalize_direct_visibility_and_serialize_null() -> None:
    omitted = ModelRuntimeKey("policy", "act", "predict", "torch", "request-direct")
    explicit = ModelRuntimeKey("policy", "act", "predict", "torch", "request-direct", None)
    empty = ModelRuntimeKey("policy", "act", "predict", "torch", "request-direct", "")

    assert omitted == explicit == empty
    assert omitted.to_dict()["orchestration_visibility"] is None
    assert json.loads(omitted.canonical_json())["orchestration_visibility"] is None
    assert omitted.fingerprint == explicit.fingerprint == empty.fingerprint

    composite = CompositeRuntimeKey("tensor_model", "speech_direction", "enhance_and_vad", "stream-direct", "")
    assert composite.orchestration_visibility is None
    assert json.loads(composite.canonical_json())["orchestration_visibility"] is None

    with pytest.raises(ValueError, match="iterative execution requires"):
        ModelRuntimeKey("policy", "act", "predict", "torch", "request-iterative")


def test_session_builder_key_is_v3_identity_and_one_builder_can_back_two_runtimes() -> None:
    session_key = _session_key()
    sessions = SessionBuilderRegistry()

    def builder(_context: object) -> object:
        return object()

    sessions.register(session_key, builder)

    direct = _role_descriptor(_role_key(), session_key)
    iterative_key = _role_key(contract="request-iterative", visibility="executor")
    iterative = _role_descriptor(iterative_key, session_key)
    assemblers = RuntimeAssemblerRegistry()
    assemblers.register(direct)
    assemblers.register(iterative)

    registry_set = RegistrySet(FakeBackendRegistry(), sessions, assemblers)
    registry_set.freeze()

    assert sessions.get(SessionBuilderKey("policy", "act", "predict", "torch")) is builder
    assert assemblers.get(direct.key) is direct
    assert assemblers.get(iterative.key) is iterative


def test_session_builder_key_derives_v3_identity_from_context() -> None:
    context = SimpleNamespace(
        model=SimpleNamespace(interface="tensor_model", model_type="zipvoice", operation="synthesize"),
        deployment=SimpleNamespace(runtime_profile=SimpleNamespace(backend="torch")),
    )

    assert SessionBuilderKey.from_context(context) == SessionBuilderKey(
        "tensor_model", "zipvoice", "synthesize", "torch"
    )


def test_duplicate_keys_are_rejected_and_frozen_registries_are_immutable() -> None:
    sessions = SessionBuilderRegistry()
    key = _session_key()

    def builder(_context: object) -> object:
        return object()

    sessions.register(key, builder)
    with pytest.raises(RuntimeRegistryError) as duplicate_session:
        sessions.register(key, builder)
    assert duplicate_session.value.code == "duplicate_session_builder_key"

    assemblers = RuntimeAssemblerRegistry()
    descriptor = _role_descriptor(_role_key())
    assemblers.register(descriptor)
    with pytest.raises(RuntimeRegistryError) as duplicate_runtime:
        assemblers.register(descriptor)
    assert duplicate_runtime.value.code == "duplicate_runtime_key"

    registry_set, _sessions, _assemblers = _registry_set()
    calls: list[str] = []

    def register_builtins(_set: RegistrySet) -> None:
        calls.append("registered")

    registry_set.bootstrap(register_builtins)
    registry_set.bootstrap(register_builtins)
    assert calls == ["registered"]
    assert registry_set.backends is registry_set.backend_registry
    assert registry_set.session_builders is registry_set.session_builder_registry
    assert registry_set.runtime_assemblers is registry_set.runtime_assembler_registry
    registry_set.freeze()
    assert registry_set.frozen
    with pytest.raises(RegistryFrozenError):
        registry_set.runtime_assembler_registry.register(_role_descriptor(_role_key("pi05")))
    with pytest.raises(RegistryFrozenError):
        registry_set.session_builder_registry.register(
            SessionBuilderKey("policy", "pi05", "predict", "torch"), lambda _context: object()
        )


def test_freeze_rejects_missing_session_builder_before_exposing_the_set() -> None:
    registry_set, _sessions, _assemblers = _registry_set(add_session=False)

    with pytest.raises(RuntimeRegistryError) as error:
        registry_set.freeze()

    assert error.value.code == "missing_session_builder"
    assert not registry_set.frozen


def _composite_descriptor(
    *,
    selectors: tuple[RuntimeRoleSelector, ...] | None = None,
    matrix: object | None = None,
    assembler: object | None = None,
) -> CompositeRuntimeDescriptor:
    selected_selectors = selectors or (
        RuntimeRoleSelector("enhancer", "tensor_model", "fullsubnet", "enhance", "request-direct"),
        RuntimeRoleSelector("vad", "tensor_model", "silero_vad", "vad", "request-direct"),
    )
    selected_matrix = matrix or {
        "enhancer": RoleBackendProfile("torch", TorchProfile),
        "vad": RoleBackendProfile("torch", TorchProfile),
    }
    selected_assembler = assembler or (lambda *, providers: _assembly(contract="request-direct"))
    selected_assembler.execution_contract = "request-direct"  # type: ignore[attr-defined]
    return CompositeRuntimeDescriptor(
        key=CompositeRuntimeKey("tensor_model", "speech_direction", "enhance_and_vad", "request-direct"),
        assembler=selected_assembler,
        required_role_selectors=selected_selectors,
        role_compatibility_matrix=selected_matrix,
    )


def _composite_registry_set(descriptor: CompositeRuntimeDescriptor) -> RegistrySet:
    sessions = SessionBuilderRegistry()
    for model_type, operation in (("fullsubnet", "enhance"), ("silero_vad", "vad")):
        sessions.register(
            SessionBuilderKey("tensor_model", model_type, operation, "torch"),
            lambda _context: object(),
        )
    assemblers = RuntimeAssemblerRegistry()
    for selector in descriptor.required_role_selectors:
        role_key = selector.runtime_key("torch")
        if assemblers.get_runtime(role_key) is None:
            assemblers.register(_role_descriptor(role_key, profile_type=TorchProfile))
    assemblers.register(descriptor)
    return RegistrySet(FakeBackendRegistry(), sessions, assemblers)


def test_composite_selectors_must_be_unique_and_matrix_must_match_roles() -> None:
    duplicate_selector = RuntimeRoleSelector("enhancer", "tensor_model", "fullsubnet", "enhance", "request-direct")
    descriptor = _composite_descriptor(selectors=(duplicate_selector, duplicate_selector))
    with pytest.raises(RuntimeRegistryError) as duplicate_error:
        _composite_registry_set(descriptor).freeze()
    assert duplicate_error.value.code == "invalid_composite_role_selectors"

    missing_role = _composite_descriptor(
        matrix={"enhancer": RoleBackendProfile("torch", TorchProfile)},
    )
    with pytest.raises(RuntimeRegistryError) as missing_error:
        _composite_registry_set(missing_role).freeze()
    assert missing_error.value.code == "invalid_composite_role_selectors"

    mismatched_profile = _composite_descriptor(
        matrix={
            "enhancer": RoleBackendProfile("torch", AscendProfile),
            "vad": RoleBackendProfile("torch", TorchProfile),
        }
    )
    with pytest.raises(RuntimeRegistryError) as matrix_error:
        _composite_registry_set(mismatched_profile).freeze()
    assert matrix_error.value.code == "composite_role_backend_mismatch"


def test_descriptor_and_assembler_contracts_are_checked_before_freeze() -> None:
    def mismatched_assembler(*, providers: RuntimeProviders) -> RuntimeAssembly:
        del providers
        return _assembly(contract="request-direct")

    mismatched_assembler.execution_contract = "stream-direct"  # type: ignore[attr-defined]
    sessions = SessionBuilderRegistry()
    sessions.register(_session_key(), lambda _context: object())
    assemblers = RuntimeAssemblerRegistry()
    assemblers.register(_role_descriptor(_role_key(), assembler=mismatched_assembler))

    with pytest.raises(RuntimeRegistryError) as error:
        RegistrySet(FakeBackendRegistry(), sessions, assemblers).freeze()

    assert error.value.code == "descriptor_contract_mismatch"


def test_descriptor_aliases_preserve_the_public_role_shape() -> None:
    key = _role_key()

    def assemble(*, providers: RuntimeProviders) -> RuntimeAssembly:
        del providers
        return _assembly(contract=key.execution_contract)

    descriptor = RuntimeDescriptor(
        runtime_key=key,
        session_builder_reference=key.session_builder_key,
        backend_profile_type=TorchProfile,
        assembler=assemble,
        capabilities={"stateful": False},
    )
    assert descriptor.runtime_key == key
    assert descriptor.backend_profile_type is TorchProfile
    assert descriptor.session_builder_reference == key.session_builder_key


def test_assembler_requires_injected_providers_and_handles_do_not_close_them() -> None:
    sessions = SessionBuilderRegistry()
    sessions.register(_session_key(), lambda _context: object())
    assemblers = RuntimeAssemblerRegistry()
    seen: list[RuntimeProviders] = []
    events: list[str] = []

    class FakeProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(f"provider:{self.name}")

    class Lease:
        def release(self) -> None:
            events.append("lease")

    def assemble(*, providers: RuntimeProviders) -> RuntimeAssembly:
        seen.append(providers)
        return RuntimeAssembly(
            runtime_executor=ClosableResource("executor", events),
            session=ClosableResource("session", events),
            provider_leases=(Lease(),),
            providers=providers,
            runtime_id="provider-runtime",
        )

    assemble.execution_contract = "request-direct"  # type: ignore[attr-defined]
    key = _role_key()
    assemblers.register(_role_descriptor(key, assembler=assemble))
    registry_set = RegistrySet(FakeBackendRegistry(), sessions, assemblers)
    registry_set.freeze()
    providers = RuntimeProviders.create(FakeProvider("acl"), FakeProvider("resource"))

    with pytest.raises(RuntimeRegistryError) as missing:
        assemblers.assemble(key)
    assert missing.value.code == "runtime_providers_required"

    assembly = assemblers.assemble(key, providers=providers)
    assert seen == [providers]
    assert not hasattr(registry_set, "acl_runtime_provider")
    assert not hasattr(registry_set, "resource_admission_provider")

    handle = ModelRuntimeHandle(assembly)
    handle.load()
    handle.close()
    assert events[:3] == ["load:session", "load:executor", "close:executor"]
    assert "lease" in events
    assert "provider:resource" not in events
    assert "provider:acl" not in events
    providers.close()
    providers.close()
    assert events[-2:] == ["provider:resource", "provider:acl"]

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_manifest import ModelIdentity, ModelRuntimeSpec, TorchRuntimeProfile, ValidatedDeployment
from inference_service.unified_runtime import (
    CompositeRuntimeDescriptor,
    CompositeRuntimeKey,
    ExecutionContext,
    ExecutionContract,
    ModelRequest,
    ModelRuntimeFactory,
    ModelRuntimeKey,
    RegistrySet,
    RequestDirectAssembler,
    RequestIterativeAssembler,
    RoleBackendProfile,
    RuntimeAssemblerRegistry,
    RuntimeAssembly,
    RuntimeDescriptor,
    RuntimeFactoryError,
    RuntimeProviders,
    RuntimeRoleSelector,
    SessionBuilderKey,
    SessionBuilderRegistry,
)


class TorchProfile:
    backend_name = "torch"


class OtherProfile:
    backend_name = "other"


class FakeBackendRegistry:
    names = ("torch", "other")


class RecordingResource:
    def __init__(self, name: str, events: list[str], *, output: object | None = None) -> None:
        self.name = name
        self.events = events
        self.output = output
        self.close_count = 0

    def load(self, _context: object) -> None:
        self.events.append(f"load:{self.name}")

    def execute(self, request: ModelRequest, _context: ExecutionContext) -> object:
        self.events.append(f"execute:{self.name}")
        return self.output if self.output is not None else {"outputs": dict(request.inputs)}

    def close(self) -> None:
        self.close_count += 1
        self.events.append(f"close:{self.name}")


class Lease:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def release(self) -> None:
        self.events.append("release:lease")


def _contract(name: str = "request-direct") -> ExecutionContract:
    state_scope, structure = name.split("-", 1)
    return ExecutionContract(
        state_scope=state_scope,
        execution_structure=structure,
        orchestration_visibility="executor" if structure == "iterative" else None,
        cancellation_granularity="request_boundary",
        state_bank_mode="per_stream" if state_scope == "stream" else None,
        max_open_streams=1 if state_scope == "stream" else None,
    )


def _identity(interface: str, model_type: str, operation: str) -> SimpleNamespace:
    return SimpleNamespace(interface=interface, model_type=model_type, operation=operation)


def _profile_envelope(profile: object, backend: str = "torch") -> SimpleNamespace:
    return SimpleNamespace(
        backend=backend,
        target=SimpleNamespace(runtime=backend, runtime_abi=None),
        target_runtime=backend,
        runtime_abi=None,
        backend_profile=profile,
        profile=profile,
    )


def _single_spec(*, contract: str = "request-direct", profile: object | None = None) -> SimpleNamespace:
    profile = profile or TorchProfile()
    return SimpleNamespace(
        identity=_identity("policy", "act", "predict"),
        execution_contract=_contract(contract),
        runtime_profile=_profile_envelope(profile),
    )


def _role_descriptor(
    key: ModelRuntimeKey,
    *,
    assembler: object | None = None,
    profile_type: type = TorchProfile,
) -> RuntimeDescriptor:
    return RuntimeDescriptor(
        key=key,
        session_builder_key=key.session_builder_key,
        profile_type=profile_type,
        assembler=assembler or RequestDirectAssembler(),
        declared_capabilities={"stateful": False},
    )


def _single_registry(
    *,
    assembler: object | None = None,
    profile_type: type = TorchProfile,
    builder=None,
    contract: str = "request-direct",
) -> RegistrySet:
    sessions = SessionBuilderRegistry()
    session_key = SessionBuilderKey("policy", "act", "predict", "torch")
    sessions.register(session_key, builder or (lambda _runtime_spec: RecordingResource("session", [])))
    runtimes = RuntimeAssemblerRegistry()
    key = ModelRuntimeKey(
        "policy", "act", "predict", "torch", contract, "executor" if contract.endswith("iterative") else None
    )
    runtimes.register(_role_descriptor(key, assembler=assembler, profile_type=profile_type))
    registry_set = RegistrySet(FakeBackendRegistry(), sessions, runtimes)
    registry_set.freeze()
    return registry_set


def _providers() -> RuntimeProviders:
    return RuntimeProviders.create(SimpleNamespace(), SimpleNamespace())


def test_factory_builds_single_request_direct_and_transfers_ownership() -> None:
    events: list[str] = []
    session = RecordingResource("session", events, output={"outputs": {"value": 7}})
    registry_set = _single_registry(builder=lambda _runtime_spec: session)

    runtime = ModelRuntimeFactory.create(_single_spec(), registry_set, _providers())

    assert isinstance(runtime.assembly, RuntimeAssembly)
    assert runtime.assembly.ownership_transferred
    assert runtime.state.value == "created"
    runtime.load()
    result = runtime.execute(ModelRequest({"value": 1}), ExecutionContext("request-1"))
    assert result.outputs["value"] == 7
    runtime.close()
    assert session.close_count == 1
    assert events == ["load:session", "execute:session", "close:session"]


def test_factory_accepts_typed_runtime_spec_and_validated_deployment() -> None:
    identity = ModelIdentity(interface="policy", model_type="act", operation="predict")
    contract = _contract()
    deployment = SimpleNamespace(execution_contract=contract, execution=())
    validated = ValidatedDeployment(
        bundle_root=Path("/tmp"),
        manifest_path=Path("/tmp/inference_manifest.json"),
        manifest=object(),
        deployment_name="cpu",
        deployment=deployment,
        top_level_identity=identity,
        role_identities={},
        role_runtime_profiles={},
        selected_deployment=deployment,
        semantic_contract=object(),
        resolved_artifacts={},
        role_artifact_bindings={},
        declared_metadata={},
        integrity_status=None,
        deployment_fingerprint="deployment-fingerprint",
        runtime_profile_fingerprint="runtime-profile-fingerprint",
    )
    spec = ModelRuntimeSpec(
        deployment=validated,
        runtime_profile=TorchRuntimeProfile(device="cpu"),
    )
    session = RecordingResource("typed-session", [])
    sessions = SessionBuilderRegistry()
    session_key = SessionBuilderKey("policy", "act", "predict", "torch")
    sessions.register(session_key, lambda runtime_spec: session)
    runtimes = RuntimeAssemblerRegistry()
    runtime_key = ModelRuntimeKey("policy", "act", "predict", "torch", "request-direct")
    runtimes.register(
        RuntimeDescriptor(
            key=runtime_key,
            session_builder_key=session_key,
            profile_type=TorchRuntimeProfile,
            assembler=RequestDirectAssembler(),
        )
    )
    registry_set = RegistrySet(SimpleNamespace(names=("torch",)), sessions, runtimes)
    registry_set.freeze()

    runtime = ModelRuntimeFactory.create(spec, registry_set, _providers())

    assert runtime.assembly.identity == identity
    assert runtime.assembly.deployment_fingerprint == "deployment-fingerprint"
    runtime.close()


def test_factory_supports_explicit_request_iterative_assembler() -> None:
    session = RecordingResource("iterative", [])
    registry_set = _single_registry(
        assembler=RequestIterativeAssembler(),
        builder=lambda _runtime_spec: session,
        contract="request-iterative",
    )

    runtime = ModelRuntimeFactory.create(_single_spec(contract="request-iterative"), registry_set, _providers())
    runtime.load()
    assert runtime.execute(ModelRequest({"value": 2}), ExecutionContext("iterative-1")).outputs["value"] == 2
    runtime.close()


def test_transferred_assembly_releases_resources_before_provider_lease() -> None:
    events: list[str] = []
    session = RecordingResource("session", events)
    executor = RecordingResource("executor", events)
    lease = Lease(events)

    def assembler(*, session, providers):
        del providers
        return RuntimeAssembly(
            runtime_executor=executor,
            session=session,
            provider_leases=(lease,),
            execution_contract="request-direct",
        )

    registry_set = _single_registry(assembler=assembler, builder=lambda _runtime_spec: session)
    runtime = ModelRuntimeFactory.create(_single_spec(), registry_set, _providers())
    assert runtime.assembly.ownership_transferred

    runtime.close()

    assert events == ["close:executor", "close:session", "release:lease"]


def _composite_setup(*, include_descriptor: bool = True, profile: object | None = None):
    events: list[str] = []
    top_identity = _identity("tensor_model", "speech_direction", "enhance_and_vad")
    role_identities = {
        "enhancer": _identity("tensor_model", "fullsubnet", "enhance"),
        "vad": _identity("tensor_model", "silero_vad", "vad"),
    }
    role_profiles = {
        "enhancer": _profile_envelope(profile or TorchProfile()),
        "vad": _profile_envelope(TorchProfile()),
    }
    spec = SimpleNamespace(
        identity=top_identity,
        role_identities=role_identities,
        role_runtime_specs=role_profiles,
        execution_contract=_contract(),
    )
    sessions = SessionBuilderRegistry()
    runtimes = RuntimeAssemblerRegistry()
    selectors = tuple(
        RuntimeRoleSelector(
            role,
            identity.interface,
            identity.model_type,
            identity.operation,
            "request-direct",
        )
        for role, identity in role_identities.items()
    )
    matrix = {
        "enhancer": RoleBackendProfile("torch", TorchProfile),
        "vad": RoleBackendProfile("torch", TorchProfile),
    }
    for role, identity in role_identities.items():
        key = ModelRuntimeKey(
            identity.interface,
            identity.model_type,
            identity.operation,
            "torch",
            "request-direct",
        )
        sessions.register(key.session_builder_key, lambda _runtime_spec, role=role: RecordingResource(role, events))
        runtimes.register(_role_descriptor(key))

    aggregate_calls: list[Mapping[str, RuntimeAssembly]] = []

    def aggregate(*, role_assemblies, providers):
        del providers
        aggregate_calls.append(role_assemblies)
        return RuntimeAssembly(
            runtime_executor=RecordingResource("aggregate", events),
            execution_contract="request-direct",
        )

    if include_descriptor:
        composite_key = CompositeRuntimeKey("tensor_model", "speech_direction", "enhance_and_vad", "request-direct")
        runtimes.register(
            CompositeRuntimeDescriptor(
                key=composite_key,
                assembler=aggregate,
                required_role_selectors=selectors,
                role_compatibility_matrix=matrix,
            )
        )
    registry_set = RegistrySet(FakeBackendRegistry(), sessions, runtimes)
    registry_set.freeze()
    return spec, registry_set, events, aggregate_calls


def test_factory_selects_composite_descriptor_before_role_assemblers() -> None:
    spec, registry_set, _events, aggregate_calls = _composite_setup()
    runtime = ModelRuntimeFactory.create(spec, registry_set, _providers())

    assert set(runtime.assembly.role_assemblies) == {"enhancer", "vad"}
    assert len(aggregate_calls) == 1
    runtime.close()


def test_missing_composite_descriptor_fails_before_session_creation() -> None:
    spec, registry_set, events, _aggregate_calls = _composite_setup(include_descriptor=False)

    with pytest.raises(RuntimeFactoryError) as error:
        ModelRuntimeFactory.create(spec, registry_set, _providers())

    assert error.value.code == "composite_assembler_unavailable"
    assert events == []


def test_composite_profile_matrix_mismatch_is_clear_and_rolls_back_nothing() -> None:
    spec, registry_set, events, _aggregate_calls = _composite_setup(profile=OtherProfile())

    with pytest.raises(RuntimeFactoryError) as error:
        ModelRuntimeFactory.create(spec, registry_set, _providers())

    assert error.value.code == "composite_role_backend_mismatch"
    assert events == []


def test_factory_rolls_back_sessions_and_leases_before_transfer() -> None:
    events: list[str] = []
    session = RecordingResource("session", events)
    executor = RecordingResource("executor", events)
    lease = Lease(events)

    def invalid_assembler(*, session, providers):
        del session, providers
        return RuntimeAssembly(
            runtime_executor=executor,
            provider_leases=(lease,),
            execution_contract="request-iterative",
        )

    registry_set = _single_registry(assembler=invalid_assembler, builder=lambda _runtime_spec: session)

    with pytest.raises(RuntimeFactoryError) as error:
        ModelRuntimeFactory.create(_single_spec(), registry_set, _providers())

    assert error.value.code == "descriptor_assembly_contract_mismatch"
    assert events == ["close:executor", "release:lease", "close:session"] or events == [
        "release:lease",
        "close:executor",
        "close:session",
    ]

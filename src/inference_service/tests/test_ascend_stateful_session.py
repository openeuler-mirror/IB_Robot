"""Validation tests for stateful Ascend OM sessions."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_manifest import (
    ArtifactBindings,
    AscendRuntimeProfile,
    Deployment,
    DeploymentArtifact,
    DeploymentTarget,
    ExecutionContract,
    RoleRuntimeProfile,
    StateLink,
    TensorBinding,
    ValidatedDeployment,
)
from inference_service.backends.errors import BackendLoadError
from inference_service.backends.types import RuntimeContext
from inference_service.model_sessions import StatefulAscendOmModelSession


def _binding(semantic: str, index: int, shape: tuple[int, ...] = (2, 1, 128)) -> TensorBinding:
    return TensorBinding(semantic=semantic, index=index, dtype="float32", shape=shape)


def _role_bindings(hidden_out_shape=(2, 1, 128), hidden_in_shape=(2, 1, 128)) -> ArtifactBindings:
    return ArtifactBindings(
        inputs=(
            _binding("host.model.audio", 0, (1, 576)),
            _binding("host.model.lstm_hidden_in", 1, hidden_in_shape),
            _binding("host.model.lstm_cell_in", 2),
        ),
        outputs=(
            _binding("host.model.prob", 0, (1, 1)),
            _binding("host.model.lstm_hidden_out", 1, hidden_out_shape),
            _binding("host.model.lstm_cell_out", 2),
        ),
    )


def _state_link(role="model", state_name="hidden", state_bank="model.bank") -> StateLink:
    return StateLink(
        role=role,
        state_name=state_name,
        owner="session",
        source=f"state.{state_name}_in",
        target=f"state.{state_name}_out",
        scope="runtime",
        state_bank=state_bank,
    )


def _contract(state_bank_mode="runtime_exclusive", links=None, stateful=True) -> ExecutionContract:
    if links is None:
        links = (_state_link(),)
    return ExecutionContract(
        state_scope="stream",
        execution_structure="direct",
        cancellation_granularity="checkpoint",
        stateful=stateful,
        state_bank_mode=state_bank_mode,
        max_open_streams=1,
        state_links=tuple(links),
    )


def _deployment(bindings=None, contract=None, execution=("model",), extra_bindings=None) -> Deployment:
    deployment_bindings = {"model": bindings or _role_bindings()}
    if extra_bindings:
        deployment_bindings.update(extra_bindings)
    artifacts = {role: DeploymentArtifact(path=f"{role}.om", format="om") for role in execution}
    return Deployment(
        execution_contract=contract or _contract(),
        runtime_profile=RoleRuntimeProfile(
            backend="ascend",
            target=DeploymentTarget(runtime="acl", runtime_abi="cann-8.1.RC1"),
            profile=AscendRuntimeProfile(device_id=0),
        ),
        artifacts=artifacts,
        execution=execution,
        bindings=deployment_bindings,
    )


def _context(tmp_path: Path, deployment: Deployment) -> RuntimeContext:
    for role in deployment.artifacts:
        (tmp_path / f"{role}.om").write_bytes(b"mock-om")
    validated = ValidatedDeployment(
        bundle_root=tmp_path,
        manifest_path=tmp_path / "inference_manifest.json",
        manifest=SimpleNamespace(),
        deployment_name="ascend_310p",
        deployment=deployment,
        top_level_identity=SimpleNamespace(interface="tensor_model", model_type="silero_vad", operation="vad"),
        role_identities={},
        role_runtime_profiles={},
        selected_deployment=deployment,
        semantic_contract=SimpleNamespace(),
        resolved_artifacts={},
        role_artifact_bindings={},
        declared_metadata={},
        integrity_status=None,
        deployment_fingerprint="fingerprint",
        runtime_profile_fingerprint="profile-fingerprint",
    )
    return RuntimeContext(validated)


def _session() -> StatefulAscendOmModelSession:
    return StatefulAscendOmModelSession(device_id=0, runtime_manager=object())


def test_state_link_resolves_recurrent_abi_pair() -> None:
    pairs = StatefulAscendOmModelSession._state_indices_for_role(
        _role_bindings(), _state_link(state_name="hidden"), "model"
    )
    assert pairs == ((1, 1),)
    pairs = StatefulAscendOmModelSession._state_indices_for_role(
        _role_bindings(), _state_link(state_name="cell"), "model"
    )
    assert pairs == ((2, 2),)


def test_state_link_rejects_shape_mismatch() -> None:
    bindings = _role_bindings(hidden_out_shape=(2, 1, 64))
    with pytest.raises(BackendLoadError) as exc:
        StatefulAscendOmModelSession._state_indices_for_role(bindings, _state_link(), "model")
    assert exc.value.code == "state_size_mismatch"


def test_state_link_rejects_dynamic_shape() -> None:
    bindings = _role_bindings(hidden_in_shape=(-1, 128), hidden_out_shape=(-1, 128))
    with pytest.raises(BackendLoadError) as exc:
        StatefulAscendOmModelSession._state_indices_for_role(bindings, _state_link(), "model")
    assert exc.value.code == "invalid_state_link_abi"


def test_state_link_rejects_ambiguous_abi() -> None:
    bindings = ArtifactBindings(
        inputs=(_binding("host.model.audio", 0, (1, 576)),),
        outputs=(_binding("host.model.prob", 0, (1, 1)),),
    )
    with pytest.raises(BackendLoadError) as exc:
        StatefulAscendOmModelSession._state_indices_for_role(bindings, _state_link(), "model")
    assert exc.value.code == "invalid_state_link_abi"


def test_load_rejects_per_stream_state_banks(tmp_path: Path) -> None:
    links = (_state_link().model_copy(update={"scope": "stream"}),)
    deployment = _deployment(contract=_contract(state_bank_mode="per_stream", links=links))
    with pytest.raises(BackendLoadError) as exc:
        _session().load(_context(tmp_path, deployment))
    assert exc.value.code == "invalid_state_contract"
    assert "runtime_exclusive" in str(exc.value)


def test_load_rejects_execution_role_without_state_link(tmp_path: Path) -> None:
    extra_bindings = {
        "extra": ArtifactBindings(
            inputs=(_binding("host.extra.frame", 0, (1, 16)),),
            outputs=(_binding("host.extra.out", 0, (1, 16)),),
        )
    }
    deployment = _deployment(
        execution=("model", "extra"),
        extra_bindings=extra_bindings,
    )
    with pytest.raises(BackendLoadError) as exc:
        _session().load(_context(tmp_path, deployment))
    assert exc.value.code == "invalid_state_contract"
    assert "extra" in str(exc.value)


def test_load_rejects_state_link_to_non_execution_role(tmp_path: Path) -> None:
    deployment = _deployment(contract=_contract(links=(_state_link(role="__runtime__", state_bank="other.bank"),)))
    with pytest.raises(BackendLoadError) as exc:
        _session().load(_context(tmp_path, deployment))
    assert exc.value.code == "invalid_state_contract"


def test_load_rejects_missing_state_links(tmp_path: Path) -> None:
    deployment = _deployment(contract=_contract(stateful=False, links=()))
    with pytest.raises(BackendLoadError) as exc:
        _session().load(_context(tmp_path, deployment))
    assert exc.value.code == "invalid_state_contract"


def test_runtime_bank_link_resolves_through_state_bank_name() -> None:
    bindings = _role_bindings()
    link = _state_link(role="__runtime__", state_bank="model.bank")
    pairs = StatefulAscendOmModelSession._state_indices_for_role(bindings, link, "model")
    assert pairs == ((1, 1),)

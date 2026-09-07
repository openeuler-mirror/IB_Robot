"""Tests for the manifest-backed FullSubNet Torch session."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2]
_WORKSPACE_SRC = _SRC.parent
for package_root in (
    _SRC,
    _WORKSPACE_SRC / "inference_manifest",
    _WORKSPACE_SRC / "inference_service",
):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

import voice_asr_service.speech_direction.enhancement.fullsubnet_stateful_torch as torch_executor_module  # noqa: E402
from inference_manifest import (  # noqa: E402
    ArtifactBindings,
    Deployment,
    DeploymentArtifact,
    DeploymentTarget,
    ExecutionContract,
    RoleRuntimeProfile,
    StateLink,
    TorchRuntimeProfile,
    ValidatedDeployment,
)
from inference_service.backends.errors import BackendInferenceError, BackendLoadError  # noqa: E402
from inference_service.backends.types import RuntimeContext  # noqa: E402
from inference_service.unified_runtime import ExecutionContext, ModelRequest  # noqa: E402
from voice_asr_service.speech_direction.contract import speech_direction_bindings  # noqa: E402
from voice_asr_service.speech_direction.fullsubnet_torch_session import (  # noqa: E402
    FullSubNetTorchSession,
    build_fullsubnet_torch_session,
)

_STATE_SUFFIXES = ("_hidden_in", "_hidden_out", "_cell_in", "_cell_out", ".state_in", ".state_out")


class _FakeExecutor:
    backend = "stateful_torch_cpu"

    def __init__(self, checkpoint, state_contract, *, device, timing_enabled):
        self.checkpoint = checkpoint
        self.state_contract = state_contract
        self.device = device
        self.timing_enabled = timing_enabled
        self.resets = 0
        self.closed = False

    def run_fb(self, frame):
        return np.zeros((4, 2, 257), dtype=np.float32)

    def run_sb(self, frame):
        return np.zeros((1028, 2, 2), dtype=np.float32)

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True

    @property
    def last_timing_ms(self):
        return {"fb_infer_ms": 1.0}


def _torch_bindings(role: str) -> ArtifactBindings:
    full = speech_direction_bindings(role)
    return ArtifactBindings(
        inputs=tuple(b for b in full.inputs if not b.semantic.endswith(_STATE_SUFFIXES)),
        outputs=tuple(b for b in full.outputs if not b.semantic.endswith(_STATE_SUFFIXES)),
    )


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "fullsubnet"
    (root / "assets").mkdir(parents=True)
    (root / "assets" / "cum_fullsubnet_best_model_218epochs.tar").write_bytes(b"checkpoint")
    (root / "assets" / "cum_fullsubnet_best_model_218epochs.manifest.json").write_bytes(b"{}")
    return root


def _deployment(device: str = "cpu") -> Deployment:
    links = tuple(
        StateLink(
            role="__runtime__",
            state_name=state_name,
            owner="session",
            source=f"state.{state_name}_in",
            target=f"state.{state_name}_out",
            scope="runtime",
            state_bank=f"fullsubnet_{role}.bank",
        )
        for role in ("fb", "sb")
        for state_name in ("hidden", "cell")
    )
    return Deployment(
        execution_contract=ExecutionContract(
            state_scope="stream",
            execution_structure="direct",
            cancellation_granularity="checkpoint",
            stateful=True,
            state_bank_mode="runtime_exclusive",
            max_open_streams=1,
            state_links=links,
        ),
        runtime_profile=RoleRuntimeProfile(
            backend="torch",
            target=DeploymentTarget(runtime="torch"),
            profile=TorchRuntimeProfile(device=device),
        ),
        artifacts={
            "fullsubnet_fb": DeploymentArtifact(
                path="assets/cum_fullsubnet_best_model_218epochs.tar", format="torch", sha256=None
            ),
            "fullsubnet_sb": DeploymentArtifact(
                path="assets/cum_fullsubnet_best_model_218epochs.manifest.json", format="json", sha256=None
            ),
        },
        execution=("fullsubnet_fb", "fullsubnet_sb"),
        bindings={
            "fullsubnet_fb": _torch_bindings("fullsubnet_fb"),
            "fullsubnet_sb": _torch_bindings("fullsubnet_sb"),
        },
    )


def _validated(root: Path, deployment: Deployment) -> ValidatedDeployment:
    return ValidatedDeployment(
        bundle_root=root,
        manifest_path=root / "inference_manifest.json",
        manifest=SimpleNamespace(),
        deployment_name="torch_cpu",
        deployment=deployment,
        top_level_identity=SimpleNamespace(interface="tensor_model", model_type="fullsubnet", operation="enhance"),
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


def _context(root: Path, deployment: Deployment, runtime_options=None) -> RuntimeContext:
    return RuntimeContext(_validated(root, deployment), runtime_options or {}, role="fullsubnet_fb")


@pytest.fixture()
def fake_executor(monkeypatch):
    instances: list[_FakeExecutor] = []

    def factory(checkpoint, state_contract, *, device, timing_enabled):
        executor = _FakeExecutor(checkpoint, state_contract, device=device, timing_enabled=timing_enabled)
        instances.append(executor)
        return executor

    monkeypatch.setattr(torch_executor_module, "StatefulTorchFullSubNetExecutor", factory)
    return instances


def test_builder_loads_and_executes_fb_sb_roles(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment())
    session = build_fullsubnet_torch_session(context)
    assert isinstance(session, FullSubNetTorchSession)
    session.load(context)

    assert len(fake_executor) == 1
    assert fake_executor[0].device == "cpu"
    assert fake_executor[0].checkpoint.endswith("cum_fullsubnet_best_model_218epochs.tar")

    request = ModelRequest({}, {})
    execution = ExecutionContext("fullsubnet-test")
    fb = session.execute_role(
        "fullsubnet_fb",
        {"host.fullsubnet.fb_spectrum": np.zeros((4, 2, 257), dtype=np.float32)},
        request,
        execution,
    )
    assert fb["host.fullsubnet.fb_features"].shape == (4, 2, 257)
    sb = session.execute_role(
        "fullsubnet_sb",
        {"host.fullsubnet.sb_features": np.zeros((1028, 2, 32), dtype=np.float32)},
        request,
        execution,
    )
    assert sb["host.fullsubnet.sb_mask"].shape == (1028, 2, 2)

    session.reset()
    assert fake_executor[0].resets == 1
    session.close()
    assert fake_executor[0].closed is True


def test_role_execution_rejects_invalid_shapes(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment())
    session = build_fullsubnet_torch_session(context)
    session.load(context)

    with pytest.raises(BackendInferenceError) as invalid:
        session.execute_role(
            "fullsubnet_fb",
            {"host.fullsubnet.fb_spectrum": np.zeros((1, 2, 257), dtype=np.float32)},
            ModelRequest({}, {}),
            ExecutionContext("bad-shape"),
        )
    assert invalid.value.code == "role_input_shape_mismatch"

    with pytest.raises(BackendInferenceError) as missing:
        session.execute_role(
            "fullsubnet_fb",
            {"other": np.zeros((4, 2, 257), dtype=np.float32)},
            ModelRequest({}, {}),
            ExecutionContext("bad-semantic"),
        )
    assert missing.value.code in {"missing_semantic_input", "role_fullsubnet_fb_input_semantic_mismatch"}


def test_request_execution_requires_host_orchestration(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment())
    session = build_fullsubnet_torch_session(context)
    session.load(context)

    with pytest.raises(BackendInferenceError) as exc:
        session.execute(ModelRequest({}, {}), ExecutionContext("request"))
    assert exc.value.code == "host_orchestration_required"


def test_unknown_runtime_options_are_rejected(tmp_path, fake_executor) -> None:
    root = _bundle(tmp_path)
    context = _context(root, _deployment(), {"device": "cuda"})
    session = build_fullsubnet_torch_session(context)
    with pytest.raises(BackendLoadError) as exc:
        session.load(context)
    assert exc.value.code == "invalid_runtime_options"


def test_builder_rejects_non_torch_identity(tmp_path) -> None:
    root = _bundle(tmp_path)
    deployment = _deployment()
    validated = _validated(root, deployment)
    object.__setattr__(
        validated,
        "top_level_identity",
        SimpleNamespace(interface="tensor_model", model_type="silero_vad", operation="vad"),
    )
    context = RuntimeContext(validated, {}, role="fullsubnet_fb")
    with pytest.raises(BackendLoadError) as exc:
        build_fullsubnet_torch_session(context)
    assert exc.value.code == "invalid_deployment"

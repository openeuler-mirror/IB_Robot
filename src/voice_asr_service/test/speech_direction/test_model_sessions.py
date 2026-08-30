"""Tests for speech-direction ModelSession protocol adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2]
_WORKSPACE_SRC = _SRC.parent
for package_root in (_SRC, _WORKSPACE_SRC / "inference_manifest", _WORKSPACE_SRC / "inference_service"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from inference_manifest import load_inference_manifest  # noqa: E402
from inference_service.backends import RuntimeContext  # noqa: E402
from inference_service.model_sessions import StatefulAscendOmModelSession  # noqa: E402
from inference_service.runtime_composition import build_runtime_dependencies  # noqa: E402
from voice_asr_service.model_session_builders import register_speech_direction_session_builder  # noqa: E402
from voice_asr_service.speech_direction.config import FullSubNetConfig, VadConfig  # noqa: E402
from voice_asr_service.speech_direction.model_sessions import SpeechDirectionRoleRunner  # noqa: E402
from voice_asr_service.speech_direction.speech_gate import SileroVadEngine  # noqa: E402


class _Execution:
    def __init__(self) -> None:
        self.calls = []

    def invoke(self, role, values):
        self.calls.append((role, values))
        if role == "fullsubnet_fb":
            return {"host.fullsubnet.fb_features": np.zeros((4, 2, 257), dtype=np.float32)}
        return {"host.fullsubnet.sb_mask": np.zeros((1028, 2, 2), dtype=np.float32)}


class _Session:
    def __init__(self) -> None:
        self.execute_role_calls = []

    def execute_role(self, role, values, request, context):
        self.execute_role_calls.append((role, values, request, context))
        if role == "fullsubnet_fb":
            return {"host.fullsubnet.fb_features": np.zeros((4, 2, 257), dtype=np.float32)}
        if role == "fullsubnet_sb":
            return {"host.fullsubnet.sb_mask": np.zeros((1028, 2, 2), dtype=np.float32)}
        return {"host.silero.prob": np.array([[0.75]], dtype=np.float32)}


class _VadRunner:
    def __init__(self) -> None:
        self.inputs = []
        self.reset_calls = 0
        self.close_calls = 0

    def infer(self, audio: np.ndarray) -> float:
        self.inputs.append(audio)
        return 0.5

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_voice_asr_uses_canonical_backend_names_without_runtime_acl_options() -> None:
    assert FullSubNetConfig().backend == "ascend"
    assert VadConfig().backend == "ascend"
    assert not hasattr(FullSubNetConfig(), "acl_config_path")


def test_stateful_ascend_maps_recurrent_bindings_by_semantic_suffix() -> None:
    bindings = type(
        "Bindings",
        (),
        {
            "inputs": (
                type("Binding", (), {"semantic": "host.fullsubnet.fb_spectrum", "index": 0})(),
                type("Binding", (), {"semantic": "host.fullsubnet.fb_hidden_in", "index": 1})(),
                type("Binding", (), {"semantic": "host.fullsubnet.fb_cell_in", "index": 2})(),
            ),
            "outputs": (
                type("Binding", (), {"semantic": "host.fullsubnet.fb_features", "index": 0})(),
                type("Binding", (), {"semantic": "host.fullsubnet.fb_hidden_out", "index": 1})(),
                type("Binding", (), {"semantic": "host.fullsubnet.fb_cell_out", "index": 2})(),
            ),
        },
    )()

    hidden_link = type("Link", (), {"state_name": "hidden"})()
    cell_link = type("Link", (), {"state_name": "cell"})()
    assert StatefulAscendOmModelSession._state_indices_for_role(bindings, hidden_link, "fullsubnet_fb") == ((1, 1),)
    assert StatefulAscendOmModelSession._state_indices_for_role(bindings, cell_link, "fullsubnet_fb") == ((2, 2),)


def test_silero_engine_reuses_runner_and_adds_context(tmp_path) -> None:
    model_path = tmp_path / "silero.om"
    model_path.write_bytes(b"mock-om")
    runner = _VadRunner()
    engine = SileroVadEngine(str(model_path), acl_runner=runner)

    assert engine.inference(np.zeros(512, dtype=np.float32)) == pytest.approx(0.5)
    assert runner.inputs[0].shape == (1, 576)

    engine.reset_state()
    engine.close()
    assert runner.reset_calls == 1
    assert runner.close_calls == 1


def test_fullsubnet_roles_share_one_session_execution_scope() -> None:
    session = _Session()
    runner = SpeechDirectionRoleRunner(session, context=None)

    with runner.execution_scope():
        fb = runner.run_fb(np.zeros((4, 2, 257), dtype=np.float32))
        sb = runner.run_sb(np.zeros((1028, 2, 32), dtype=np.float32))

    assert fb.shape == (4, 2, 257)
    assert sb.shape == (1028, 2, 2)
    assert [call[0] for call in session.execute_role_calls] == ["fullsubnet_fb", "fullsubnet_sb"]
    assert len({id(call[3]) for call in session.execute_role_calls}) == 1


def test_role_runner_rejects_nested_execution_scopes() -> None:
    runner = SpeechDirectionRoleRunner(_Session(), context=None)

    with runner.execution_scope(), pytest.raises(RuntimeError, match="already active"), runner.execution_scope():
        pass


def test_silero_inference_uses_standalone_session_execution() -> None:
    session = _Session()
    runner = SpeechDirectionRoleRunner(session, context=None)

    probability = runner.infer(np.zeros((1, 576), dtype=np.float32))

    assert probability == pytest.approx(0.75)
    assert [call[0] for call in session.execute_role_calls] == ["silero_vad"]


@pytest.mark.parametrize(
    "bundle_rel,deployment_name,role",
    [
        ("fullsubnet", "ascend_310p", "fullsubnet_fb"),
        ("silero-vad", "ascend_310p", "model"),
    ],
)
def test_checked_in_speech_manifest_selects_generic_stateful_session(
    tmp_path, bundle_rel, deployment_name, role
) -> None:
    config_root = _WORKSPACE_SRC.parent / "models" / bundle_rel
    manifest = json.loads((config_root / "inference_manifest.json").read_text(encoding="utf-8"))
    for deployment in manifest["deployments"].values():
        for artifact in deployment.get("artifacts", {}).values():
            artifact.pop("sha256", None)
    (tmp_path / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "README.md").write_text("test bundle\n", encoding="utf-8")
    for deployment in manifest["deployments"].values():
        for artifact in deployment.get("artifacts", {}).values():
            path = tmp_path / artifact["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mock-om")
    for entry in manifest["bundle"]["files"]:
        path = tmp_path / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mock-file")

    context = RuntimeContext(load_inference_manifest(tmp_path, deployment_name), {"device_id": 0}, role=role)
    assert context.target_runtime == "acl"
    assert context.runtime_abi == "cann-8.1.RC1"
    dependencies = build_runtime_dependencies(
        lambda session_registry, _assembler_registry: register_speech_direction_session_builder(session_registry)
    )
    try:
        session = dependencies.registry_set.session_builder_registry.create(
            context,
            backend_registry=dependencies.registry_set.backend_registry,
            providers=dependencies.providers,
        )
    finally:
        dependencies.providers.close()

    assert isinstance(session, StatefulAscendOmModelSession)

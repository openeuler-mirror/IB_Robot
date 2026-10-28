from __future__ import annotations

from types import SimpleNamespace

from inference_manifest import CompiledDeployment
from inference_service.model_sessions import AscendOmModelSession, StatefulAscendOmModelSession
from inference_service.unified_runtime import LoadRollback


def _stateful_deployment() -> CompiledDeployment:
    return CompiledDeployment.model_validate(
        {
            "uuid": "123e4567-e89b-42d3-a456-426614174001",
            "revision": 1,
            "execution_contract": {
                "state_scope": "stream",
                "execution_structure": "direct",
                "cancellation_granularity": "checkpoint",
                "stateful": True,
                "state_links": [
                    {
                        "role": "model",
                        "state_name": "hidden",
                        "owner": "session",
                        "source": "state.in",
                        "target": "state.out",
                        "scope": "runtime",
                        "state_bank": "model.bank",
                    }
                ],
                "state_bank_mode": "runtime_exclusive",
                "max_open_streams": 1,
            },
            "runtime_profile": {
                "backend": "ascend",
                "target": {"soc": "Ascend310P1", "runtime": "acl", "runtime_abi": "cann-8.1.RC1"},
                "profile": {"device_id": 0},
            },
            "artifacts": {"model": {"path": "artifacts/model.om", "format": "om"}},
            "execution": ["model"],
            "bindings": {
                "model": {
                    "inputs": [{"semantic": "host.model_hidden_in", "index": 0, "dtype": "float32", "shape": [1, 4]}],
                    "outputs": [{"semantic": "host.model_hidden_out", "index": 0, "dtype": "float32", "shape": [1, 4]}],
                }
            },
        }
    )


def test_stateful_ascend_session_publishes_reset_capability(monkeypatch) -> None:
    session = StatefulAscendOmModelSession(runtime_manager=object())
    session._loading = True
    monkeypatch.setattr(AscendOmModelSession, "_load", lambda *_args: None)

    session._load(SimpleNamespace(deployment=_stateful_deployment()), LoadRollback())

    assert session.capabilities.stateful is True
    assert session.capabilities.resettable is True
    session.reset()

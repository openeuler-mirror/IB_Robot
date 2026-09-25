"""ProfileRegistry admission against real robot_config scheduler identities.

``test_scheduler_profiles.py`` covers ProfileRegistry with synthetic fingerprint
strings, and robot_config covers that its own fingerprints change when runtime
options change. These two tests cover the seam between them: that the real
fingerprint values robot_config computes are the ones ProfileRegistry compares,
so a profile calibrated under different runtime options is actually rejected.

They live here rather than in robot_config because robot_config cannot declare a
test dependency on inference_service - inference_service already depends on
robot_config, and colcon refuses to order a workspace containing a cycle.

The bundle and robot-config builders below are duplicated from
``robot_config/test/test_inference_scheduler_config.py``. That duplication is
deliberate: the alternative is a shared test-fixture package, which neither
package currently has.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from inference_manifest import BundleFile, canonical_bundle_digest
from inference_service.scheduler.profiles import ProfileRegistry
from robot_config import parse_inference_config

_BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
_DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _create_bundle(root: Path, deployment: str = "cpu") -> Path:
    root.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "type": "act",
            "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "policy_preprocessor", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "policy_postprocessor", "steps": []})
    (root / "model.safetensors").write_bytes(b"test-policy-weights")

    bundle_paths = ("config.json", "model.safetensors", "policy_postprocessor.json", "policy_preprocessor.json")
    entries = [BundleFile(path=path) for path in bundle_paths]
    deployment_value = {
        "uuid": _DEPLOYMENT_UUID,
        "revision": 1,
        "execution_contract": {
            "state_scope": "request",
            "execution_structure": "direct",
            "cancellation_granularity": "request_boundary",
        },
        "runtime_profile": {
            "backend": "torch",
            "target": {"runtime": "torch"},
            "profile": {"device": "cpu"},
        },
    }
    _write_json(
        root / "inference_manifest.json",
        {
            "schema_version": 3,
            "bundle": {
                "uuid": _BUNDLE_UUID,
                "revision": 1,
                "name": root.name,
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(_BUNDLE_UUID, 1, root.name, entries),
                },
            },
            "model": {
                "interface": "policy",
                "model_type": "act",
                "operation": "predict",
                "inputs": [{"semantic": "observation.state", "dtype": "float32", "shape": [6]}],
                "outputs": [{"semantic": "action", "dtype": "float32", "shape": [6]}],
            },
            "deployments": {deployment: deployment_value},
        },
    )
    return root


def _profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text("closure_profiles: []\n", encoding="utf-8")
    return path


def _pipeline(bundle: Path, *, profile: Path, compatibility_group: str = "so101_action") -> dict[str, Any]:
    return {
        "model_path": str(bundle),
        "deployment": "cpu",
        "execution_mode": "monolithic",
        "transport": {
            "open_session": "/inference/policy/session/open",
            "dispatch": "/inference/policy/scheduled_dispatch",
            "close_session": "/inference/policy/session/close",
            "serving_status": "/inference/policy/serving_status",
        },
        "required": True,
        "compatibility_group": compatibility_group,
        "hardware_resource_id": "ascend:0",
        "hardware_profile_fingerprint": "a" * 64,
        "profile_path": str(profile),
        "public_capacity": {
            "session_control": {"max_in_flight": 1},
            "action_generation": {"max_in_flight": 1},
        },
    }


def _scheduler_block() -> dict[str, Any]:
    return {
        "enable": True,
        "global_endpoints": {
            "readiness": "/inference/scheduler/ready",
            "open_session": "/inference/session/open",
            "dispatch": "/inference/dispatch",
            "close_session": "/inference/session/close",
        },
        "profile_min_samples": 10000,
    }


def _robot_config(pipelines: dict[str, Any], *, scheduler: dict[str, Any] | None) -> dict[str, Any]:
    inference: dict[str, Any] = {"enabled": True, "pipelines": pipelines}
    if scheduler is not None:
        inference["scheduler"] = scheduler
    return {
        "control_modes": {
            "model_inference": {
                "inference": inference,
                "executor": {
                    "inference_pipeline": "policy",
                    "inference_fallback_chain": [],
                    "inference_priority": 0,
                    "inference_retry": {
                        "max_not_started_attempts": 3,
                        "initial_backoff_ms": 50,
                        "max_backoff_ms": 500,
                    },
                },
            }
        }
    }


def test_profile_registry_rejects_profiles_calibrated_with_other_runtime_options(tmp_path: Path) -> None:
    """Toggling the collection switch invalidates previously valid profiles.

    Priority-0 admission must fail closed (no p99 estimate) instead of
    reusing measurements taken with a different execution configuration.
    """
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    calibrated_pipeline = _pipeline(bundle, profile=profile)
    calibrated = parse_inference_config(
        _robot_config({"policy": calibrated_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]

    now_ns = time.time_ns()
    _write_json(
        profile,
        {
            "closure_profiles": [
                {
                    "deployment_fingerprint": calibrated.validated_manifest.fingerprint,
                    "hardware_fingerprint": calibrated.hardware_profile_fingerprint,
                    "profile_compatibility_fingerprint": calibrated.profile_compatibility_fingerprint,
                    "scope": "global_proxy",
                    "work_class": 2,
                    "closure_key": "full_infer",
                    "hardware_priority": 0,
                    "input_contract_fingerprint": "c" * 64,
                    "prompt_bytes_max": 4096,
                    "goal_acceptance_p999_ms": 1.0,
                    "latency_p99_ms": 50.0,
                    "profiled_at_ns": now_ns,
                    "sample_count": 10000,
                }
            ]
        },
    )

    reconfigured_pipeline = _pipeline(bundle, profile=profile)
    reconfigured_pipeline["runtime_options"] = {"auto_horizon_enabled": True}
    reconfigured = parse_inference_config(
        _robot_config({"policy": reconfigured_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]

    registry = ProfileRegistry(
        profile_path=str(reconfigured.profile_path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint=reconfigured.validated_manifest.fingerprint,
        hardware_fingerprint=str(reconfigured.hardware_profile_fingerprint),
        profile_compatibility_fingerprint=str(reconfigured.profile_compatibility_fingerprint),
        now_ns=lambda: now_ns,
    )
    registry.load()

    assert registry.profile_count == 0
    assert (
        registry.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=0,
        )
        is None
    )


def test_config_identity_loads_matching_open_and_dispatch_profiles(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    robot = _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block())
    first = parse_inference_config(robot, "model_inference").pipelines["policy"]
    assert first.runtime_policy_fingerprint is not None
    assert first.profile_compatibility_fingerprint is not None
    now_ns = time.time_ns()
    common = {
        "deployment_fingerprint": first.validated_manifest.fingerprint,
        "hardware_fingerprint": first.hardware_profile_fingerprint,
        "profile_compatibility_fingerprint": first.profile_compatibility_fingerprint,
        "scope": "global_proxy",
        "hardware_priority": 0,
        "goal_acceptance_p999_ms": 1.0,
        "profiled_at_ns": now_ns,
        "sample_count": 10000,
    }
    _write_json(
        profile,
        {
            "closure_profiles": [
                {
                    **common,
                    "work_class": 1,
                    "closure_key": "session_open",
                    "input_contract_fingerprint": "",
                    "prompt_bytes_max": 0,
                    "latency_p99_ms": 10.0,
                },
                {
                    **common,
                    "work_class": 2,
                    "closure_key": "full_infer",
                    "input_contract_fingerprint": "c" * 64,
                    "prompt_bytes_max": 4096,
                    "latency_p99_ms": 50.0,
                },
            ]
        },
    )

    second = parse_inference_config(robot, "model_inference").pipelines["policy"]
    assert second.runtime_policy_fingerprint == first.runtime_policy_fingerprint
    registry = ProfileRegistry(
        profile_path=str(second.profile_path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint=second.validated_manifest.fingerprint,
        hardware_fingerprint=str(second.hardware_profile_fingerprint),
        profile_compatibility_fingerprint=str(second.profile_compatibility_fingerprint),
        now_ns=lambda: now_ns,
    )
    registry.load()

    assert (
        registry.closure_p99_ms(
            work_class=1,
            closure_key="session_open",
            hardware_priority=0,
            input_contract_fingerprint="",
            prompt_bytes=0,
        )
        == 10.0
    )
    assert (
        registry.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=0,
        )
        == 50.0
    )

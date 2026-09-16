"""Typed validation tests for the scheduler feature switch and its policy block.

Covers the whole-graph scheduler control plane: the single `scheduler.enable` switch, bounded policy
fields, public_capacity, scheduled transport endpoints, the runtime_policy
fingerprint, and fail-closed semantics. The false/absent path is asserted unchanged
relative to the legacy dispatcher behavior.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from inference_manifest import BundleFile, canonical_bundle_digest
from inference_service.scheduler.profiles import ProfileRegistry
from robot_config import InferenceConfigError, parse_inference_config
from robot_config.inference_config import (
    StageSchedulingConfig,
    _validate_stage_priority_range,
    scheduler_enabled_from_raw_config,
)

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


def _create_compiled_bundle(root: Path, *, artifact_sha256: str | None) -> Path:
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
    (root / "policy.om").write_bytes(b"compiled-policy")
    bundle_paths = ("config.json", "policy_postprocessor.json", "policy_preprocessor.json")
    entries = [BundleFile(path=path) for path in bundle_paths]
    artifact: dict[str, Any] = {"path": "policy.om", "format": "om"}
    if artifact_sha256 is not None:
        artifact["sha256"] = artifact_sha256
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
            "deployments": {
                "npu": {
                    "uuid": _DEPLOYMENT_UUID,
                    "revision": 1,
                    "execution_contract": {
                        "state_scope": "request",
                        "execution_structure": "direct",
                        "cancellation_granularity": "request_boundary",
                    },
                    "runtime_profile": {
                        "backend": "ascend",
                        "target": {"soc": "ascend310", "runtime": "acl"},
                        "profile": {"device_id": 0},
                    },
                    "artifacts": {"policy": artifact},
                    "execution": ["policy"],
                    "bindings": {
                        "policy": {
                            "inputs": [
                                {
                                    "semantic": "observation.state",
                                    "index": 0,
                                    "dtype": "float32",
                                    "shape": [6],
                                }
                            ],
                            "outputs": [
                                {
                                    "semantic": "action",
                                    "index": 0,
                                    "dtype": "float32",
                                    "shape": [6],
                                }
                            ],
                        }
                    },
                }
            },
        },
    )
    return root


def _profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text("closure_profiles: []\n", encoding="utf-8")
    return path


def _create_independent_bundle(root: Path) -> Path:
    """Two-role request-scoped Ascend bundle."""

    root.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "type": "pi05",
            "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "policy_preprocessor", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "policy_postprocessor", "steps": []})
    roles = ("vlm", "action_expert")
    (root / "vlm.om").write_bytes(b"compiled-vlm")
    (root / "action_expert.om").write_bytes(b"compiled-action_expert")
    bundle_paths = ("config.json", "policy_postprocessor.json", "policy_preprocessor.json")
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
            "backend": "ascend",
            "target": {"soc": "ascend310", "runtime": "acl"},
            "profile": {"device_id": 0},
        },
        "artifacts": {
            role: {
                "path": f"{role}.om",
                "format": "om",
                "sha256": hashlib.sha256(f"compiled-{role}".encode()).hexdigest(),
            }
            for role in roles
        },
        "execution": list(roles),
        "bindings": {
            role: {
                "inputs": [{"semantic": "observation.state", "index": 0, "dtype": "float32", "shape": [6]}],
                "outputs": [{"semantic": "action", "index": 0, "dtype": "float32", "shape": [6]}],
            }
            for role in roles
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
                "model_type": "pi05",
                "operation": "predict",
                "inputs": [{"semantic": "observation.state", "dtype": "float32", "shape": [6]}],
                "outputs": [{"semantic": "action", "dtype": "float32", "shape": [6]}],
            },
            "deployments": {"npu": deployment_value},
        },
    )
    return root


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


def _legacy_pipeline(bundle: Path) -> dict[str, Any]:
    return {
        "model_path": str(bundle),
        "deployment": "cpu",
        "execution_mode": "monolithic",
    }


def test_raw_scheduler_switch_reader_does_not_parse_pipeline_artifacts() -> None:
    invalid_pipeline = {"model_path": "/missing/bundle", "profile_path": "/missing/profile"}
    robot = {
        "control_modes": {
            "model_inference": {
                "inference": {
                    "enabled": True,
                    "scheduler": {"enable": True},
                    "pipelines": {"policy": invalid_pipeline},
                }
            }
        }
    }

    assert scheduler_enabled_from_raw_config(robot, "model_inference") is True
    robot["control_modes"]["model_inference"]["inference"]["scheduler"]["enable"] = False
    assert scheduler_enabled_from_raw_config(robot, "model_inference") is False
    assert scheduler_enabled_from_raw_config({}, "model_inference") is False


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


# ---------------------------------------------------------------------------
# Default and false path regression: no scheduler effect.
# ---------------------------------------------------------------------------


def test_no_scheduler_block_leaves_config_unchanged(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    cfg = parse_inference_config(
        _robot_config({"policy": _legacy_pipeline(bundle)}, scheduler=None),
        "model_inference",
    )
    assert cfg.scheduler is None
    assert cfg.inference_pipeline is None
    assert cfg.inference_fallback_chain == ()
    assert cfg.inference_priority == 0
    pipeline = cfg.pipelines["policy"]
    # Scheduled fields must not be populated when scheduler is absent.
    assert pipeline.compatibility_group is None
    assert pipeline.hardware_resource_id is None
    assert pipeline.profile_path is None
    assert pipeline.runtime_policy_fingerprint is None
    assert dict(pipeline.public_capacity) == {}


def test_scheduler_enable_false_does_not_materialize_scheduler(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    block = _scheduler_block()
    block["enable"] = False
    cfg = parse_inference_config(
        _robot_config({"policy": _legacy_pipeline(bundle)}, scheduler=block),
        "model_inference",
    )
    # false branch: structurally accepted, no SchedulerConfig, no scheduled fields.
    assert cfg.scheduler is None
    pipeline = cfg.pipelines["policy"]
    assert pipeline.compatibility_group is None
    assert pipeline.runtime_policy_fingerprint is None


def test_scheduler_false_does_not_require_compiled_artifact_sha256(tmp_path: Path) -> None:
    bundle = _create_compiled_bundle(tmp_path / "bundle", artifact_sha256=None)
    block = _scheduler_block()
    block["enable"] = False
    pipeline = _legacy_pipeline(bundle)
    pipeline["deployment"] = "npu"
    cfg = parse_inference_config(
        _robot_config({"policy": pipeline}, scheduler=block),
        "model_inference",
    )
    assert cfg.scheduler is None


def test_scheduler_true_requires_compiled_artifact_sha256(tmp_path: Path) -> None:
    bundle = _create_compiled_bundle(tmp_path / "bundle", artifact_sha256=None)
    pipeline = _pipeline(bundle, profile=_profile_file(tmp_path))
    pipeline["deployment"] = "npu"
    with pytest.raises(InferenceConfigError, match="must declare a content sha256"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


def test_scheduler_true_rejects_compiled_artifact_sha256_mismatch(tmp_path: Path) -> None:
    bundle = _create_compiled_bundle(tmp_path / "bundle", artifact_sha256="0" * 64)
    pipeline = _pipeline(bundle, profile=_profile_file(tmp_path))
    pipeline["deployment"] = "npu"
    with pytest.raises(InferenceConfigError, match="content sha256 mismatch"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


def test_scheduler_true_accepts_matching_compiled_artifact_sha256(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"compiled-policy").hexdigest()
    bundle = _create_compiled_bundle(tmp_path / "bundle", artifact_sha256=digest)
    pipeline = _pipeline(bundle, profile=_profile_file(tmp_path))
    pipeline["deployment"] = "npu"
    cfg = parse_inference_config(
        _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    )
    assert cfg.scheduler is not None


@pytest.mark.parametrize("value", (0, 1, "false", "true", None))
def test_scheduler_enable_requires_boolean(tmp_path: Path, value: object) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    block = _scheduler_block()
    block["enable"] = value
    with pytest.raises(InferenceConfigError, match="must be a boolean"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, profile=_profile_file(tmp_path))}, scheduler=block),
            "model_inference",
        )


def test_scheduler_enable_true_rejects_distributed_pipeline(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline = _pipeline(bundle, profile=_profile_file(tmp_path))
    pipeline["execution_mode"] = "distributed"

    with pytest.raises(InferenceConfigError, match="must be 'monolithic'"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


@pytest.mark.parametrize(
    "endpoint",
    ("open_session", "dispatch", "close_session", "serving_status"),
)
def test_scheduler_enable_true_requires_pipeline_transport_endpoint(tmp_path: Path, endpoint: str) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline = _pipeline(bundle, profile=_profile_file(tmp_path))
    del pipeline["transport"][endpoint]

    with pytest.raises(InferenceConfigError, match=endpoint):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


# ---------------------------------------------------------------------------
# Hardware-independent static validation and fingerprint.
# ---------------------------------------------------------------------------


def test_scheduler_enable_true_materializes_full_policy(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    cfg = parse_inference_config(
        _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block()),
        "model_inference",
    )
    assert cfg.scheduler is not None and cfg.scheduler.enable is True
    pipeline = cfg.pipelines["policy"]
    assert pipeline.compatibility_group == "so101_action"
    assert pipeline.hardware_resource_id == "ascend:0"
    assert pipeline.profile_path == profile.resolve()
    assert pipeline.runtime_policy_fingerprint is not None
    assert len(pipeline.runtime_policy_fingerprint) == 64  # sha256 hex
    # executor fields parsed only on the enabled branch.
    assert cfg.inference_pipeline == "policy"
    assert cfg.inference_fallback_chain == ()
    assert cfg.inference_priority == 0
    assert cfg.inference_retry["max_not_started_attempts"] == 3
    # nanosecond conversion of a timeout.
    assert cfg.scheduler.default_open_timeout_ns == 10_000_000_000
    assert cfg.scheduler.session_idle_timeout_ns == 30_000_000_000


def test_scheduler_profile_path_is_optional(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline_config = _pipeline(bundle, profile=_profile_file(tmp_path))
    del pipeline_config["profile_path"]

    cfg = parse_inference_config(
        _robot_config({"policy": pipeline_config}, scheduler=_scheduler_block()),
        "model_inference",
    )

    assert cfg.pipelines["policy"].profile_path is None


def test_scheduler_rejects_an_explicit_empty_profile_path(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline_config = _pipeline(bundle, profile=_profile_file(tmp_path))
    pipeline_config["profile_path"] = ""

    with pytest.raises(InferenceConfigError, match="profile_path must be a non-empty string"):
        parse_inference_config(
            _robot_config({"policy": pipeline_config}, scheduler=_scheduler_block()),
            "model_inference",
        )


def test_runtime_policy_fingerprint_stable_and_sensitive(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    cfg_a = parse_inference_config(
        _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block()),
        "model_inference",
    )
    cfg_b = parse_inference_config(
        _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block()),
        "model_inference",
    )
    fp_a = cfg_a.pipelines["policy"].runtime_policy_fingerprint
    fp_b = cfg_b.pipelines["policy"].runtime_policy_fingerprint
    assert fp_a == fp_b  # canonical -> stable across runs

    # Mutating hardware_resource_id changes the fingerprint.
    pipeline2 = _pipeline(bundle, profile=profile)
    pipeline2["hardware_resource_id"] = "ascend:1"
    cfg_c = parse_inference_config(
        _robot_config({"policy": pipeline2}, scheduler=_scheduler_block()),
        "model_inference",
    )
    assert cfg_c.pipelines["policy"].runtime_policy_fingerprint != fp_a


def test_runtime_policy_fingerprint_does_not_change_with_profile_evidence(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    robot = _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block())
    before = parse_inference_config(robot, "model_inference").pipelines["policy"].runtime_policy_fingerprint

    profile.write_text('{"closure_profiles": []}\n', encoding="utf-8")
    after = parse_inference_config(robot, "model_inference").pipelines["policy"].runtime_policy_fingerprint

    assert before == after


def test_profile_compatibility_fingerprint_ignores_endpoint_names_but_tracks_capacity(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    base_pipeline = _pipeline(bundle, profile=profile)
    base = parse_inference_config(
        _robot_config({"policy": base_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]

    renamed_pipeline = _pipeline(bundle, profile=profile)
    renamed_pipeline["transport"]["dispatch"] = "/renamed/pipeline/dispatch"
    renamed = parse_inference_config(
        _robot_config({"policy": renamed_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]
    assert renamed.runtime_policy_fingerprint != base.runtime_policy_fingerprint
    assert renamed.profile_compatibility_fingerprint == base.profile_compatibility_fingerprint

    capacity_pipeline = _pipeline(bundle, profile=profile)
    capacity_pipeline["public_capacity"]["action_generation"] = {"max_in_flight": 2}
    capacity = parse_inference_config(
        _robot_config({"policy": capacity_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]
    assert capacity.profile_compatibility_fingerprint != base.profile_compatibility_fingerprint


def test_profile_compatibility_fingerprint_tracks_runtime_options(tmp_path: Path) -> None:
    """Latency-relevant runtime options participate in the profile identity.

    Enabling AutoHorizon attention collection changes the Torch execution
    path (per-layer GPU->CPU attention copies), so profiles calibrated with
    collection disabled must not be reused.
    """
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    base_pipeline = _pipeline(bundle, profile=profile)
    base = parse_inference_config(
        _robot_config({"policy": base_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]

    enabled_pipeline = _pipeline(bundle, profile=profile)
    enabled_pipeline["runtime_options"] = {"auto_horizon_enabled": True}
    enabled = parse_inference_config(
        _robot_config({"policy": enabled_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]
    assert enabled.profile_compatibility_fingerprint != base.profile_compatibility_fingerprint

    tuned_pipeline = _pipeline(bundle, profile=profile)
    tuned_pipeline["runtime_options"] = {"auto_horizon_enabled": True, "auto_horizon_sampling_step": 5}
    tuned = parse_inference_config(
        _robot_config({"policy": tuned_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]
    assert tuned.profile_compatibility_fingerprint != enabled.profile_compatibility_fingerprint

    # Spelling out the effective defaults keeps the identity of the omitted
    # form: identical execution configuration, identical calibration scope.
    explicit_pipeline = _pipeline(bundle, profile=profile)
    explicit_pipeline["runtime_options"] = {
        "model_dtype": "native",
        "auto_horizon_enabled": False,
        "auto_horizon_hold_threshold": 0.3,
        "auto_horizon_entropy_quantile": 0.9,
        "auto_horizon_run_length": 1,
        "auto_horizon_sampling_step": 3,
    }
    explicit = parse_inference_config(
        _robot_config({"policy": explicit_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]
    assert explicit.profile_compatibility_fingerprint == base.profile_compatibility_fingerprint


def test_runtime_policy_fingerprint_tracks_runtime_options(tmp_path: Path) -> None:
    """The runtime-policy identity covers the normalized effective options.

    The pipeline node validates its actually-executed runtime options against
    this identity, so a node parameter override without an SSOT update is
    rejected at startup instead of serving under a stale policy identity.
    """
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    base = parse_inference_config(
        _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]

    enabled_pipeline = _pipeline(bundle, profile=profile)
    enabled_pipeline["runtime_options"] = {"auto_horizon_enabled": True}
    enabled = parse_inference_config(
        _robot_config({"policy": enabled_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]
    assert enabled.runtime_policy_fingerprint != base.runtime_policy_fingerprint

    # Explicitly spelled-out effective defaults keep the omitted-form identity.
    explicit_pipeline = _pipeline(bundle, profile=profile)
    explicit_pipeline["runtime_options"] = {
        "model_dtype": "native",
        "auto_horizon_enabled": False,
        "auto_horizon_hold_threshold": 0.3,
        "auto_horizon_entropy_quantile": 0.9,
        "auto_horizon_run_length": 1,
        "auto_horizon_sampling_step": 3,
    }
    explicit = parse_inference_config(
        _robot_config({"policy": explicit_pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    ).pipelines["policy"]
    assert explicit.runtime_policy_fingerprint == base.runtime_policy_fingerprint
    assert (
        json.loads(explicit.runtime_policy_json or "")["runtime_options"]
        == json.loads(base.runtime_policy_json or "")["runtime_options"]
    )


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


# ---------------------------------------------------------------------------
# Missing required scheduled fields reject startup.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("missing", "field"),
    [
        ("compatibility_group", "compatibility_group"),
        ("hardware_resource_id", "hardware_resource_id"),
        ("hardware_profile_fingerprint", "hardware_profile_fingerprint"),
    ],
)
def test_missing_required_scheduled_field_fails_closed(tmp_path: Path, missing: str, field: str) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    pipeline = _pipeline(bundle, profile=profile)
    del pipeline[missing]
    with pytest.raises(InferenceConfigError) as exc:
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )
    assert field in str(exc.value)


def test_scheduler_false_accepts_dormant_scheduled_pipeline_fields(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    pipeline = _pipeline(bundle, profile=profile)
    # Dormant values are not scheduler-runtime validated until enable=true.
    pipeline["hardware_profile_fingerprint"] = "not-a-sha256"
    pipeline["profile_path"] = str(tmp_path / "missing-profile.yaml")

    parsed = parse_inference_config(
        _robot_config({"policy": pipeline}, scheduler={"enable": False}),
        "model_inference",
    )

    assert parsed.scheduler is None
    assert parsed.inference_pipeline is None
    assert parsed.pipelines["policy"].hardware_profile_fingerprint is None
    assert parsed.pipelines["policy"].profile_path is None
    assert parsed.pipelines["policy"].runtime_policy_json is None


def test_scheduler_false_accepts_dormant_scheduled_transport_fields(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline = _legacy_pipeline(bundle)
    pipeline["transport"] = {"open_session": "/inference/policy/session/open"}

    parsed = parse_inference_config(
        _robot_config({"policy": pipeline}, scheduler={"enable": False}),
        "model_inference",
    )

    assert parsed.scheduler is None
    assert parsed.pipelines["policy"].transport.open_session is None


def test_scheduled_fields_still_require_an_explicit_scheduler_block(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline = _legacy_pipeline(bundle)
    pipeline["compatibility_group"] = "group"

    with pytest.raises(InferenceConfigError, match="unsupported fields"):
        parse_inference_config(_robot_config({"policy": pipeline}, scheduler=None), "model_inference")


def test_action_generation_capacity_must_be_positive(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    pipeline = _pipeline(bundle, profile=_profile_file(tmp_path))
    pipeline["public_capacity"]["action_generation"] = {"max_in_flight": 0}

    with pytest.raises(InferenceConfigError, match="positive integer"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


def test_session_control_capacity_must_be_one(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    pipeline = _pipeline(bundle, profile=profile)
    pipeline["public_capacity"]["session_control"] = {"max_in_flight": 2}
    with pytest.raises(InferenceConfigError, match="session_control"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


def test_missing_action_generation_capacity_fails_closed(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    pipeline = _pipeline(bundle, profile=profile)
    del pipeline["public_capacity"]["action_generation"]
    with pytest.raises(InferenceConfigError, match="action_generation"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


def test_action_generation_capacity_above_one_is_preserved_for_capable_backends(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    pipeline = _pipeline(bundle, profile=profile)
    pipeline["public_capacity"]["action_generation"] = {"max_in_flight": 2}

    cfg = parse_inference_config(
        _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
        "model_inference",
    )

    assert cfg.pipelines["policy"].public_capacity["action_generation"].max_in_flight == 2


def test_missing_global_endpoint_fails_closed(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    block = _scheduler_block()
    del block["global_endpoints"]["open_session"]
    with pytest.raises(InferenceConfigError, match="open_session"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=block),
            "model_inference",
        )


def test_negative_priority_rejected(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    rc = _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block())
    rc["control_modes"]["model_inference"]["executor"]["inference_priority"] = -1
    with pytest.raises(InferenceConfigError, match="inference_priority"):
        parse_inference_config(rc, "model_inference")


def test_priority_must_fit_action_int32(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    rc = _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block())
    rc["control_modes"]["model_inference"]["executor"]["inference_priority"] = 2_147_483_648
    with pytest.raises(InferenceConfigError, match="int32"):
        parse_inference_config(rc, "model_inference")


def test_dispatch_safety_margin_must_be_below_idle_timeout(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    block = _scheduler_block()
    block["dispatch_safety_margin_ms"] = 60_000  # >= session_idle_timeout*1000 (30s)
    with pytest.raises(InferenceConfigError, match="dispatch_safety_margin_ms"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=block),
            "model_inference",
        )


def test_edf_is_parsed_without_profile_deadline_admission(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    block = _scheduler_block()
    block["global_policy"] = "edf"

    cfg = parse_inference_config(
        _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=block),
        "model_inference",
    )

    assert cfg.scheduler is not None
    assert cfg.scheduler.global_policy == "edf"
    assert not cfg.scheduler.priority_zero_deadline_admission.enable


@pytest.mark.parametrize(
    "enabled,policy,admission",
    [(True, "fifo", False), (True, "fifo", True), (True, "edf", False), (False, "fifo", False)],
)
def test_default_fallback_requires_supported_policy_only_when_enabled(tmp_path, enabled, policy, admission):
    bundle = _create_bundle(tmp_path / "bundle")
    block = _scheduler_block()
    block.update(enable=enabled, global_policy=policy, priority_zero_deadline_admission={"enable": admission})
    pipeline = _pipeline(bundle, profile=_profile_file(tmp_path))
    backup = {
        **pipeline,
        "transport": {key: value.replace("policy", "backup") for key, value in pipeline["transport"].items()},
    }
    rc = _robot_config({"policy": pipeline, "backup": backup}, scheduler=block)
    rc["control_modes"]["model_inference"]["executor"]["inference_fallback_chain"] = ["backup"]
    if enabled and policy == "fifo" and not admission:
        with pytest.raises(InferenceConfigError, match="inference_fallback_chain requires"):
            parse_inference_config(rc, "model_inference")
    else:
        parse_inference_config(rc, "model_inference")


def test_edf_rejects_profile_deadline_admission(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    block = _scheduler_block()
    block["global_policy"] = "edf"
    block["priority_zero_deadline_admission"] = {"enable": True}

    with pytest.raises(InferenceConfigError, match="edf requires priority-zero deadline admission to be disabled"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=block),
            "model_inference",
        )


def _independent_pipeline(bundle: Path, profile: Path, *, pipeline_id: str = "policy") -> dict[str, Any]:
    pipeline = _pipeline(bundle, profile=profile)
    pipeline["deployment"] = "npu"
    pipeline["scheduling"] = {"stage_policy": "independent", "stages": {}}
    pipeline["transport"] = {key: value.replace("policy", pipeline_id) for key, value in pipeline["transport"].items()}
    return pipeline


@pytest.mark.parametrize(
    "policy,admission",
    [("edf", False), ("fifo", True)],
)
def test_deadline_driven_priority_zero_rejects_independent_target_or_fallback(tmp_path, policy, admission):
    independent_bundle = _create_independent_bundle(tmp_path / "independent")
    sequential = _pipeline(_create_bundle(tmp_path / "sequential"), profile=_profile_file(tmp_path))
    block = _scheduler_block()
    block["global_policy"] = policy
    block["priority_zero_deadline_admission"] = {"enable": admission}

    # Target selecting an independent pipeline is rejected.
    with pytest.raises(InferenceConfigError, match="cannot select independent-stage pipelines: \\['policy'\\]"):
        parse_inference_config(
            _robot_config(
                {"policy": _independent_pipeline(independent_bundle, _profile_file(tmp_path))}, scheduler=block
            ),
            "model_inference",
        )

    # Independent entry in the fallback chain is equally rejected.
    rc = _robot_config(
        {
            "policy": sequential,
            "backup": _independent_pipeline(independent_bundle, _profile_file(tmp_path), pipeline_id="backup"),
        },
        scheduler=block,
    )
    rc["control_modes"]["model_inference"]["executor"]["inference_fallback_chain"] = ["backup"]
    with pytest.raises(InferenceConfigError, match="cannot select independent-stage pipelines: \\['backup'\\]"):
        parse_inference_config(rc, "model_inference")


@pytest.mark.parametrize("max_age_ms", [0, -1, True, 1.5, "50"])
def test_snapshot_age_requires_positive_integer(tmp_path, max_age_ms):
    independent = _independent_pipeline(_create_independent_bundle(tmp_path / "independent"), _profile_file(tmp_path))
    independent["scheduling"]["stages"] = {"vlm": {"max_snapshot_age_ms": max_age_ms}}
    with pytest.raises(InferenceConfigError, match="max_snapshot_age_ms"):
        parse_inference_config(_robot_config({"policy": independent}, scheduler=_scheduler_block()), "model_inference")


@pytest.mark.parametrize("stage_policy,role", [("sequential", "vlm"), ("independent", "action_expert")])
def test_snapshot_age_rejects_stages_without_snapshot_production(tmp_path, stage_policy, role):
    independent = _independent_pipeline(_create_independent_bundle(tmp_path / "independent"), _profile_file(tmp_path))
    independent["scheduling"] = {"stage_policy": stage_policy, "stages": {role: {"max_snapshot_age_ms": 50}}}
    with pytest.raises(InferenceConfigError, match="requires an independent producer stage"):
        parse_inference_config(_robot_config({"policy": independent}, scheduler=_scheduler_block()), "model_inference")


def test_snapshot_age_flows_into_runtime_policy_and_fingerprint(tmp_path):
    independent = _independent_pipeline(_create_independent_bundle(tmp_path / "independent"), _profile_file(tmp_path))
    block = _scheduler_block()
    block["priority_zero_deadline_admission"] = {"enable": False}
    rc = _robot_config({"policy": independent}, scheduler=block)
    original = parse_inference_config(rc, "model_inference").pipelines["policy"]
    assert original.scheduling.stages["vlm"].max_snapshot_age_ms == 5000
    independent["scheduling"]["stages"] = {"vlm": {"max_snapshot_age_ms": 50}}
    updated = parse_inference_config(rc, "model_inference").pipelines["policy"]
    assert updated.scheduling.stages["vlm"].max_snapshot_age_ms == 50
    assert json.loads(updated.runtime_policy_json)["scheduling"]["stages"]["vlm"]["max_snapshot_age_ms"] == 50
    assert updated.runtime_policy_fingerprint != original.runtime_policy_fingerprint


def test_independent_pipeline_accepts_priority_zero_under_plain_fifo(tmp_path):
    """Plain FIFO dispatches priority-0 without ordering, so independent stays legal."""

    independent = _independent_pipeline(_create_independent_bundle(tmp_path / "independent"), _profile_file(tmp_path))
    block = _scheduler_block()
    block["global_policy"] = "fifo"
    block["priority_zero_deadline_admission"] = {"enable": False}

    cfg = parse_inference_config(
        _robot_config({"policy": independent}, scheduler=block),
        "model_inference",
    )
    assert cfg.pipelines["policy"].scheduling is not None
    assert cfg.pipelines["policy"].scheduling.stage_policy == "independent"


@pytest.mark.parametrize("policy,admission", [("edf", False), ("fifo", True)])
def test_independent_pipeline_accepts_positive_priority_under_deadline_driven_policy(tmp_path, policy, admission):
    """Deadline-driven ordering applies to priority-0 only."""

    independent = _independent_pipeline(_create_independent_bundle(tmp_path / "independent"), _profile_file(tmp_path))
    block = _scheduler_block()
    block["global_policy"] = policy
    block["priority_zero_deadline_admission"] = {"enable": admission}
    rc = _robot_config({"policy": independent}, scheduler=block)
    rc["control_modes"]["model_inference"]["executor"]["inference_priority"] = 1

    cfg = parse_inference_config(rc, "model_inference")
    assert cfg.inference_priority == 1


def test_lower_priority_dispatch_contexts_must_leave_priority_zero_reserve(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    block = _scheduler_block()
    block["dispatch_goal_contexts"] = 4
    block["lower_priority_dispatch_goal_contexts"] = 4

    with pytest.raises(InferenceConfigError, match="lower_priority_dispatch_goal_contexts"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=block),
            "model_inference",
        )


def test_scheduler_enable_true_requires_inference_enabled(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    rc = _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block())
    rc["control_modes"]["model_inference"]["inference"]["enabled"] = False
    with pytest.raises(InferenceConfigError, match="requires"):
        parse_inference_config(rc, "model_inference")


def test_fallback_chain_cannot_contain_target(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    rc = _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=_scheduler_block())
    rc["control_modes"]["model_inference"]["executor"]["inference_fallback_chain"] = ["policy"]
    with pytest.raises(InferenceConfigError, match="target"):
        parse_inference_config(rc, "model_inference")


def test_unknown_scheduler_field_rejected(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    block = _scheduler_block()
    block["another_enable"] = True
    with pytest.raises(InferenceConfigError, match="unsupported fields"):
        parse_inference_config(
            _robot_config({"policy": _pipeline(bundle, profile=profile)}, scheduler=block),
            "model_inference",
        )


# ---------------------------------------------------------------------------
# Only one scheduling feature switch exists.
# ---------------------------------------------------------------------------


def test_only_one_scheduling_switch_exists() -> None:
    """Scan the typed config module: `scheduler.enable` is the only switch."""
    import re

    from robot_config import inference_config as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = [r"with_scheduler", r"SCHEDULER_ENABLE_ENV"]
    for pattern in forbidden:
        assert not re.search(pattern, source), f"forbidden switch pattern {pattern!r} present"
    # `enable` parsed exactly under the scheduler block.
    assert "_parse_scheduler" in source


def test_stage_instance_count_cannot_claim_unloaded_replicas(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    pipeline = _pipeline(bundle, profile=profile)
    pipeline["scheduling"] = {
        "stage_policy": "sequential",
        "stages": {"policy": {"instance_count": 2}},
    }

    with pytest.raises(InferenceConfigError, match="one stage artifact owns one serial runtime instance"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


@pytest.mark.parametrize(
    "obsolete",
    [
        {"triggers": {"action_generation": {"queue_watermark": 20}}},
        {"stages": {"policy": {"trigger": "dispatch"}}},
    ],
)
def test_scheduler_rejects_obsolete_pipeline_trigger_configuration(tmp_path: Path, obsolete: dict[str, Any]) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    pipeline = _pipeline(bundle, profile=profile)
    pipeline["scheduling"] = {"stage_policy": "sequential", **obsolete}

    with pytest.raises(InferenceConfigError, match="unsupported fields"):
        parse_inference_config(
            _robot_config({"policy": pipeline}, scheduler=_scheduler_block()),
            "model_inference",
        )


def test_frame_base_priority_and_stage_offset_must_fit_ascend_range() -> None:
    stages = {"encoder": StageSchedulingConfig("encoder", 4, 1)}

    with pytest.raises(InferenceConfigError, match="priorities must fit"):
        _validate_stage_priority_range(stages, 4, "pipeline.scheduling")


def test_sequential_rejects_ineffective_stage_priority_offset(tmp_path):
    pipeline = _pipeline(_create_bundle(tmp_path / "bundle"), profile=_profile_file(tmp_path))
    pipeline["scheduling"] = {"stage_policy": "sequential", "stages": {"policy": {"priority_offset": 1}}}
    with pytest.raises(InferenceConfigError, match="requires independent"):
        parse_inference_config(_robot_config({"policy": pipeline}, scheduler=_scheduler_block()), "model_inference")

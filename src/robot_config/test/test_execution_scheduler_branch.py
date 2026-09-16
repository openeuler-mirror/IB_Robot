"""Launch-graph branch tests for the scheduler.enable and executor.enabled switches.

Asserts the false/absent branch produces the legacy `action_dispatcher_node` set
byte-for-byte unchanged, and the true branch produces the scheduled topology
(`pipeline_policy_node` + `global_inference_scheduler_node` +
`scheduled_action_dispatcher_node`). The two dispatchers never coexist.

Also asserts `executor.enabled: false` drops the dispatcher while keeping the
pipeline node, which is what a recording-only deployment needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from launch_ros.actions import Node

from inference_manifest import BundleFile, canonical_bundle_digest
from robot_config.dispatch_strategies import DispatchStrategyError
from robot_config.inference_config import InferenceConfigError
from robot_config.launch_builders.execution import generate_execution_nodes
from robot_config.runtime_target import RuntimeTarget

_BUNDLE_UUID = "123e4567-e89b-42d3-a456-426614174000"
_DEPLOYMENT_UUID = "123e4567-e89b-42d3-a456-426614174001"
_LEGACY_BASELINE_SHA = "63d80599bc8e"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _create_bundle(root: Path) -> Path:
    root.mkdir(parents=True)
    _write_json(
        root / "config.json",
        {
            "type": "act",
            "input_features": {"observation.state": {"type": "STATE", "shape": [6]}},
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(root / "policy_preprocessor.json", {"name": "pre", "steps": []})
    _write_json(root / "policy_postprocessor.json", {"name": "post", "steps": []})
    (root / "model.safetensors").write_bytes(b"test-weights")
    paths = ("config.json", "model.safetensors", "policy_postprocessor.json", "policy_preprocessor.json")
    entries = [BundleFile(path=path) for path in paths]
    deployment_values = {
        "cpu": {
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
            "deployments": deployment_values,
        },
    )
    return root


def _profile_file(tmp_path: Path) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text("closure_profiles: []\n", encoding="utf-8")
    return path


def _text(substitutions):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in substitutions)


def _node_parameters(node: Node) -> dict[str, Any]:
    def decode_parameter(value):
        if not isinstance(value, tuple):
            return value
        if all(isinstance(item, list) for item in value):
            return [decode_parameter(tuple(item)) for item in value]
        text = _text(value)
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return text.strip()

    raw = node._Node__parameters[0]  # type: ignore[attr-defined]
    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        name = _text(key)
        parsed[name] = decode_parameter(value)
    return parsed


def _nodes(actions) -> list[Node]:
    return [action for action in actions if isinstance(action, Node)]


def _legacy_robot_config(config_path: Path, bundle: Path) -> dict[str, Any]:
    return {
        "_config_path": str(config_path),
        "name": "test_robot",
        "joints": {"all": ["1", "2", "3", "4", "5", "6"]},
        "control_modes": {
            "model_inference": {
                "inference": {
                    "enabled": True,
                    "pipelines": {
                        "policy": {
                            "model_path": str(bundle),
                            "deployment": "cpu",
                            "execution_mode": "monolithic",
                        }
                    },
                },
                "executor": {"type": "topic", "mode": "model_inference"},
            }
        },
    }


def _scheduled_robot_config(config_path: Path, bundle: Path, profile: Path) -> dict[str, Any]:
    rc = _legacy_robot_config(config_path, bundle)
    rc["control_modes"]["model_inference"]["inference"]["scheduler"] = {
        "enable": True,
        "global_endpoints": {
            "readiness": "/inference/scheduler/ready",
            "open_session": "/inference/session/open",
            "dispatch": "/inference/dispatch",
            "close_session": "/inference/session/close",
        },
        "profile_min_samples": 10000,
    }
    pipeline = rc["control_modes"]["model_inference"]["inference"]["pipelines"]["policy"]
    pipeline.update(
        {
            "transport": {
                "open_session": "/inference/policy/session/open",
                "dispatch": "/inference/policy/scheduled_dispatch",
                "close_session": "/inference/policy/session/close",
                "serving_status": "/inference/policy/serving_status",
            },
            "required": True,
            "compatibility_group": "so101_action",
            "hardware_resource_id": "ascend:0",
            "hardware_profile_fingerprint": "a" * 64,
            "profile_path": str(profile),
            "public_capacity": {
                "session_control": {"max_in_flight": 1},
                "action_generation": {"max_in_flight": 1},
            },
        }
    )
    rc["control_modes"]["model_inference"]["executor"]["inference_pipeline"] = "policy"
    rc["control_modes"]["model_inference"]["executor"]["inference_fallback_chain"] = []
    rc["control_modes"]["model_inference"]["executor"]["inference_priority"] = 0
    rc["control_modes"]["model_inference"]["executor"]["inference_retry"] = {
        "max_not_started_attempts": 3,
        "initial_backoff_ms": 50,
        "max_backoff_ms": 500,
    }
    return rc


# ---------------------------------------------------------------------------
# False and absent produce the legacy launch graph.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_sim", [True, "true", False])
def test_simulation_rejects_enabled_inference_scheduler(tmp_path: Path, use_sim) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, _profile_file(tmp_path))

    with pytest.raises(InferenceConfigError, match="simulation does not support"):
        generate_execution_nodes(config, use_sim=use_sim, runtime_target=RuntimeTarget.SIMULATION)


def test_legacy_use_sim_argument_rejects_enabled_inference_scheduler(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, _profile_file(tmp_path))

    with pytest.raises(InferenceConfigError, match="simulation does not support"):
        generate_execution_nodes(config, use_sim="true")


@pytest.mark.parametrize("explicit_false", [False, True])
def test_simulation_uses_legacy_inference_without_scheduler(tmp_path: Path, explicit_false) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)
    if explicit_false:
        config["control_modes"]["model_inference"]["inference"]["scheduler"] = {"enable": False}

    nodes = generate_execution_nodes(config, use_sim=True, runtime_target=RuntimeTarget.SIMULATION)

    assert [node.node_executable for node in nodes] == ["pipeline_policy_node", "action_dispatcher_node"]


def test_absent_scheduler_produces_legacy_dispatcher_only(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    robot_config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)

    nodes = generate_execution_nodes(robot_config, "model_inference")

    executables = [node.node_executable for node in nodes]
    assert executables == ["pipeline_policy_node", "action_dispatcher_node"]
    pipeline = next(node for node in nodes if node.node_executable == "pipeline_policy_node")
    assert "scheduler_enabled" not in _node_parameters(pipeline)
    assert "scheduled_action_dispatcher_node" not in executables
    assert "global_inference_scheduler_node" not in executables


def test_scheduler_enable_false_produces_legacy_dispatcher_only(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    robot_config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)
    robot_config["control_modes"]["model_inference"]["inference"]["scheduler"] = {"enable": False}

    nodes = generate_execution_nodes(robot_config, "model_inference")

    executables = [node.node_executable for node in nodes]
    assert executables == ["pipeline_policy_node", "action_dispatcher_node"]


def test_executor_enabled_false_drops_the_dispatcher_but_keeps_the_pipeline(tmp_path: Path) -> None:
    """A recording-only deployment streams video but must never request inference.

    The cloud peer runs no inference backend, so a dispatched request can only
    time out; that timeout invalidates the distributed session and the edge
    stops sending video altogether. `executor.enabled: false` is how a config
    declares "stream, do not dispatch" — the pipeline node must survive because
    it owns the RTP sender.
    """
    bundle = _create_bundle(tmp_path / "bundle")
    robot_config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)
    robot_config["control_modes"]["model_inference"]["executor"]["enabled"] = False

    nodes = generate_execution_nodes(robot_config, "model_inference")

    assert [node.node_executable for node in nodes] == ["pipeline_policy_node"]


def test_executor_enabled_absent_keeps_the_dispatcher(tmp_path: Path) -> None:
    """Absent means enabled: the switch must not change any existing config."""
    bundle = _create_bundle(tmp_path / "bundle")
    robot_config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)
    assert "enabled" not in robot_config["control_modes"]["model_inference"]["executor"]

    nodes = generate_execution_nodes(robot_config, "model_inference")

    assert [node.node_executable for node in nodes] == ["pipeline_policy_node", "action_dispatcher_node"]


def test_executor_disabled_with_scheduler_enabled_is_rejected(tmp_path: Path) -> None:
    """The scheduled topology exists to feed a dispatcher, so the pair is incoherent.

    Honouring the switch on only one branch is how a config silently means two
    different things, so refuse the combination instead of ignoring it.
    """
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    robot_config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, profile)
    robot_config["control_modes"]["model_inference"]["executor"]["enabled"] = False

    with pytest.raises(ValueError, match="executor.enabled"):
        generate_execution_nodes(robot_config, "model_inference")


def test_scheduler_enable_false_matches_absent_scheduler_launch_graph(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    absent_config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)
    false_config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)
    false_config["control_modes"]["model_inference"]["inference"]["scheduler"] = {"enable": False}

    absent_nodes = generate_execution_nodes(absent_config, "model_inference")
    false_nodes = generate_execution_nodes(false_config, "model_inference")

    assert all(isinstance(action, Node) for action in false_nodes)
    assert [node.node_executable for node in false_nodes] == [node.node_executable for node in absent_nodes]
    assert [_node_parameters(node) for node in false_nodes] == [_node_parameters(node) for node in absent_nodes]
    pipeline_params = _node_parameters(false_nodes[0])
    assert "scheduler_enabled" not in pipeline_params
    assert not any(key.startswith("scheduled_") for key in pipeline_params)


def test_complete_scheduled_config_needs_only_enable_false_for_legacy_launch(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    config_path = tmp_path / "robot.yaml"
    legacy_config = _legacy_robot_config(config_path, bundle)
    disabled_config = _scheduled_robot_config(config_path, bundle, profile)
    disabled_config["control_modes"]["model_inference"]["inference"]["scheduler"]["enable"] = False

    legacy_nodes = generate_execution_nodes(legacy_config, "model_inference")
    disabled_nodes = generate_execution_nodes(disabled_config, "model_inference")

    assert [node.node_executable for node in disabled_nodes] == [node.node_executable for node in legacy_nodes]
    assert [_node_parameters(node) for node in disabled_nodes] == [_node_parameters(node) for node in legacy_nodes]
    assert [node.node_executable for node in disabled_nodes] == ["pipeline_policy_node", "action_dispatcher_node"]

    pipeline_params = _node_parameters(disabled_nodes[0])
    assert not any(key.startswith("scheduled_") for key in pipeline_params)
    for key in (
        "runtime_policy_json",
        "runtime_policy_fingerprint",
        "hardware_resource_id",
        "public_capacity_json",
        "session_idle_timeout_ns",
        "max_session_records",
        "terminal_result_cache_entries",
        "max_duplicate_waiters_per_request",
        "terminal_session_retention_ns",
    ):
        assert key not in pipeline_params


def test_edf_policy_is_forwarded_to_global_scheduler(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    config_path = tmp_path / "robot.yaml"
    robot_config = _scheduled_robot_config(config_path, bundle, profile)
    robot_config["control_modes"]["model_inference"]["inference"]["scheduler"]["global_policy"] = "edf"

    nodes = generate_execution_nodes(robot_config, "model_inference")
    scheduler = next(node for node in nodes if node.node_executable == "global_inference_scheduler_node")
    parameters = _node_parameters(scheduler)

    assert parameters["global_policy"] == "edf"
    assert parameters["priority_zero_deadline_admission_enabled"] is False


def test_scheduler_disabled_matches_63d80599_legacy_launch_contract(tmp_path: Path) -> None:
    """Golden public launch contract captured from the pre-scheduler baseline."""

    bundle = _create_bundle(tmp_path / "bundle")
    config_path = tmp_path / "robot.yaml"
    robot_config = _legacy_robot_config(config_path, bundle)
    robot_config["control_modes"]["model_inference"]["inference"]["scheduler"] = {"enable": False}

    nodes = generate_execution_nodes(robot_config, "model_inference")

    assert [node.node_executable for node in nodes] == [
        "pipeline_policy_node",
        "action_dispatcher_node",
    ], f"legacy launch topology captured from {_LEGACY_BASELINE_SHA} changed"
    pipeline_params = _node_parameters(nodes[0])
    dispatcher_params = _node_parameters(nodes[1])
    assert set(pipeline_params) == {
        "pipeline_id",
        "model_path",
        "deployment",
        "execution_mode",
        "request_timeout",
        "default_task",
        "runtime_options_json",
        "robot_config_path",
        "use_sim",
        "use_sim_time",
        "node_name",
        "action_server",
        "reset_service",
        "health_topic",
        "action_topic",
        "request_topic",
        "result_topic",
        "heartbeat_topic",
        "video_descriptor_topic",
        "video_status_topic",
    }
    assert {
        key: pipeline_params[key]
        for key in (
            "pipeline_id",
            "deployment",
            "execution_mode",
            "request_timeout",
            "action_server",
            "reset_service",
            "health_topic",
            "action_topic",
        )
    } == {
        "pipeline_id": "policy",
        "deployment": "cpu",
        "execution_mode": "monolithic",
        "request_timeout": 5.0,
        "action_server": "/inference/policy/dispatch",
        "reset_service": "/inference/policy/reset",
        "health_topic": "/inference/policy/health",
        "action_topic": "/actions/policy",
    }
    assert set(dispatcher_params) == {
        "enable_dual_mode",
        "executor_mode",
        "robot_name",
        "joint_names",
        "queue_size",
        "watermark_threshold",
        "min_queue_size",
        "control_frequency",
        "temporal_ensemble_coeff",
        "chunk_size",
        "smoothing_device",
        "chunking_strategy",
        "blending_strategy",
        "control_mode",
        "interpolation_enabled",
        "interpolation_step",
        "max_interpolation_time",
        "on_inference_failure",
        "on_queue_exhausted",
        "max_inference_timeout",
        "max_retry_attempts",
        "retry_backoff_base",
        "stale_obs_threshold_ms",
        "exhaustion_timeout",
        "joint_state_topic",
        "dispatch_action_topic",
        "robot_config_path",
        "inference_action_server",
        "inference_reset_service",
        "inference_timeout_sec",
        "policy_reset_timeout_sec",
        "inference_prompt",
        "navigation_mode",
        "use_sim_time",
    }
    assert dispatcher_params["inference_action_server"] == "/inference/policy/dispatch"
    assert dispatcher_params["inference_reset_service"] == "/inference/policy/reset"
    assert dispatcher_params["inference_timeout_sec"] == 5.0


def test_legacy_dispatcher_parameters_unchanged(tmp_path: Path) -> None:
    """The false-branch dispatcher carries exactly the legacy parameter set."""
    bundle = _create_bundle(tmp_path / "bundle")
    robot_config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)

    nodes = generate_execution_nodes(robot_config, "model_inference")
    dispatcher = next(n for n in nodes if n.node_executable == "action_dispatcher_node")
    params = _node_parameters(dispatcher)
    # legacy transport endpoints only — no global/scheduled endpoints leaked.
    assert params["inference_action_server"] == "/inference/policy/dispatch"
    assert params["inference_reset_service"] == "/inference/policy/reset"
    for scheduled_key in (
        "scheduler_readiness_endpoint",
        "open_session_endpoint",
        "dispatch_endpoint",
        "inference_retry_json",
    ):
        assert scheduled_key not in params, f"{scheduled_key} leaked into legacy dispatcher"


# ---------------------------------------------------------------------------
# True produces the scheduled topology; dispatchers never coexist.
# ---------------------------------------------------------------------------


def test_scheduler_enable_true_produces_scheduled_topology(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    robot_config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, profile)

    actions = generate_execution_nodes(robot_config, "model_inference")

    nodes = _nodes(actions)
    handlers = [action for action in actions if type(action).__name__ == "RegisterEventHandler"]
    assert len(nodes) == 3
    assert len(handlers) == 3
    executables = [node.node_executable for node in nodes]
    assert executables == [
        "pipeline_policy_node",
        "global_inference_scheduler_node",
        "scheduled_action_dispatcher_node",
    ]
    # The two dispatchers must never coexist.
    assert not (executables.count("action_dispatcher_node") and executables.count("scheduled_action_dispatcher_node"))
    dispatcher = next(node for node in nodes if node.node_executable == "scheduled_action_dispatcher_node")
    assert _node_parameters(dispatcher)["inference_pipeline"] == "policy"
    scheduler = next(node for node in nodes if node.node_executable == "global_inference_scheduler_node")
    assert _node_parameters(scheduler)["readiness_endpoint"] == "/inference/scheduler/ready"


def test_optional_pipeline_exit_is_not_registered_for_global_shutdown(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    robot_config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, profile)
    robot_config["control_modes"]["model_inference"]["inference"]["pipelines"]["policy"]["required"] = False

    actions = generate_execution_nodes(robot_config, "model_inference")

    nodes = _nodes(actions)
    handlers = [action for action in actions if type(action).__name__ == "RegisterEventHandler"]
    assert len(nodes) == 3
    assert len(handlers) == 2


def test_scheduled_dispatcher_receives_global_endpoints_and_runtime_policy(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    robot_config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, profile)

    actions = generate_execution_nodes(robot_config, "model_inference")
    dispatcher = next(node for node in _nodes(actions) if node.node_executable == "scheduled_action_dispatcher_node")
    params = _node_parameters(dispatcher)
    assert params["scheduler_readiness_endpoint"] == "/inference/scheduler/ready"
    assert params["open_session_endpoint"] == "/inference/session/open"
    assert params["dispatch_endpoint"] == "/inference/dispatch"
    assert params["inference_pipeline"] == "policy"
    assert json.loads(params["inference_retry_json"])["max_not_started_attempts"] == 3
    pipeline = next(node for node in _nodes(actions) if node.node_executable == "pipeline_policy_node")
    pipeline_params = _node_parameters(pipeline)
    assert "scheduler_enabled" not in pipeline_params
    assert pipeline_params["runtime_policy_json"]
    assert json.loads(pipeline_params["pipeline_scheduling_json"])["stages"]["policy"]["max_snapshot_age_ms"] == 5000
    assert pipeline_params["max_prompt_bytes"] == 4096
    assert pipeline_params["max_error_message_bytes"] == 1024
    assert pipeline_params["max_error_details_bytes"] == 8192


def test_scheduler_node_receives_pipeline_registry_and_endpoints(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "bundle")
    profile = _profile_file(tmp_path)
    robot_config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, profile)

    actions = generate_execution_nodes(robot_config, "model_inference")
    scheduler = next(node for node in _nodes(actions) if node.node_executable == "global_inference_scheduler_node")
    params = _node_parameters(scheduler)
    assert params["readiness_endpoint"] == "/inference/scheduler/ready"
    assert params["default_target_pipeline_id"] == "policy"
    pipelines = json.loads(params["pipelines_json"])
    assert len(pipelines) == 1
    assert pipelines[0]["pipeline_id"] == "policy"
    assert pipelines[0]["serving_status"] == "/inference/policy/serving_status"
    assert pipelines[0]["hardware_profile_fingerprint"] == "a" * 64
    assert pipelines[0]["runtime_policy_fingerprint"]  # non-empty sha256
    assert params["session_idle_timeout_ns"] == 30_000_000_000


@pytest.fixture(params=[False, True], ids=["legacy", "scheduled"])
def strategy_launch_config(request, tmp_path):
    bundle = _create_bundle(tmp_path / "bundle")
    if request.param:
        return _scheduled_robot_config(tmp_path / "robot.yaml", bundle, _profile_file(tmp_path))
    return _legacy_robot_config(tmp_path / "robot.yaml", bundle)


@pytest.mark.parametrize("field", ["scheduler", "chunking", "blending", "type"])
@pytest.mark.parametrize("value", [False, 0, [], {}, "unknown_strategy"])
def test_launch_rejects_invalid_strategy_names(strategy_launch_config, field, value):
    mode = strategy_launch_config["control_modes"]["model_inference"]
    section = mode["executor"] if field == "type" else mode.setdefault("dispatch", {})
    section[field] = value
    with pytest.raises(DispatchStrategyError, match="unknown"):
        generate_execution_nodes(strategy_launch_config, "model_inference")


@pytest.mark.parametrize("value", [False, True, None, "", "false", "true", 0, 1, [], {}])
def test_launch_rejects_removed_smoothing(strategy_launch_config, value):
    strategy_launch_config["control_modes"]["model_inference"]["executor"]["temporal_smoothing_enabled"] = value
    with pytest.raises(DispatchStrategyError, match="has been removed.*dispatch.blending"):
        generate_execution_nodes(strategy_launch_config, "model_inference")


@pytest.mark.parametrize("enabled", [False, True])
def test_launch_blending_parameters(strategy_launch_config, enabled):
    mode = strategy_launch_config["control_modes"]["model_inference"]
    blending = "temporal_ensemble" if enabled else "none"
    mode["dispatch"] = {"blending": blending}
    nodes = _nodes(generate_execution_nodes(strategy_launch_config, "model_inference"))
    dispatcher = next(node for node in nodes if "action_dispatcher_node" in node.node_executable)
    params = _node_parameters(dispatcher)
    assert params["chunking_strategy"] == "full_chunk"
    assert params["blending_strategy"] == blending
    assert "temporal_smoothing_enabled" not in params


@pytest.mark.parametrize("value", [None, ""])
def test_launch_name_defaults(strategy_launch_config, value):
    mode = strategy_launch_config["control_modes"]["model_inference"]
    mode["executor"]["type"] = value
    mode["dispatch"] = dict.fromkeys(("scheduler", "chunking", "blending"), value)
    nodes = _nodes(generate_execution_nodes(strategy_launch_config, "model_inference"))
    dispatcher = next(node for node in nodes if "action_dispatcher_node" in node.node_executable)
    params = _node_parameters(dispatcher)
    assert params["chunking_strategy"] == "full_chunk"
    assert params["blending_strategy"] == "none"
    assert "temporal_smoothing_enabled" not in params


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("blending", ["none", "temporal_ensemble"])
def test_launch_rejects_removed_flag_even_with_explicit_blending(strategy_launch_config, enabled, blending):
    mode = strategy_launch_config["control_modes"]["model_inference"]
    mode["executor"]["temporal_smoothing_enabled"] = enabled
    mode["dispatch"] = {"blending": blending}
    with pytest.raises(DispatchStrategyError, match="has been removed"):
        generate_execution_nodes(strategy_launch_config, "model_inference")


def test_launch_preserves_legacy_action_alias(strategy_launch_config):
    strategy_launch_config["control_modes"]["model_inference"]["executor"]["type"] = "action"
    nodes = _nodes(generate_execution_nodes(strategy_launch_config, "model_inference"))
    assert sum("action_dispatcher_node" in node.node_executable for node in nodes) == 1


@pytest.mark.parametrize(
    "executor,scheduler",
    [("benchmark", "wait_for_feedback"), ("topic", "wait_for_feedback"), ("benchmark", "continuous")],
)
def test_scheduled_rejects_unsupported_explicit_selection(tmp_path, executor, scheduler):
    bundle = _create_bundle(tmp_path / "bundle")
    config = _scheduled_robot_config(tmp_path / "robot.yaml", bundle, _profile_file(tmp_path))
    mode = config["control_modes"]["model_inference"]
    mode["executor"]["type"] = executor
    mode["dispatch"] = {"scheduler": scheduler}
    with pytest.raises(DispatchStrategyError, match="scheduled entrypoint requires"):
        generate_execution_nodes(config, "model_inference")


def test_launch_rejects_benchmark_auto_horizon_combination(strategy_launch_config):
    mode = strategy_launch_config["control_modes"]["model_inference"]
    mode["executor"]["type"] = "benchmark"
    mode["dispatch"] = {"scheduler": "wait_for_feedback", "chunking": "auto_horizon"}
    with pytest.raises(DispatchStrategyError, match="requires executor type 'topic'"):
        generate_execution_nodes(strategy_launch_config, "model_inference")


def test_launch_accepts_auto_horizon_for_topic_executor(tmp_path):
    bundle = _create_bundle(tmp_path / "bundle")
    config = _legacy_robot_config(tmp_path / "robot.yaml", bundle)
    config["control_modes"]["model_inference"]["dispatch"] = {"chunking": "auto_horizon"}
    nodes = _nodes(generate_execution_nodes(config, "model_inference"))
    dispatcher = next(node for node in nodes if "action_dispatcher_node" in node.node_executable)
    assert _node_parameters(dispatcher)["chunking_strategy"] == "auto_horizon"

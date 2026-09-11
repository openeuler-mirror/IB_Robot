"""Dispatch strategy SSOT validation tests (pairing, chunking, blending)."""

from __future__ import annotations

import pytest

from robot_config.dispatch_strategies import (
    DEFAULT_CHUNKING,
    DispatchStrategyError,
    resolve_dispatch_strategies,
    validate_blending_strategy,
    validate_chunking_strategy,
    validate_dispatch_strategies,
    validate_executor_scheduler_pairing,
)


@pytest.mark.parametrize(
    ("executor_type", "scheduler_mode"),
    [
        ("topic", "continuous"),
        ("benchmark", "wait_for_feedback"),
        # Unknown strings pass through so the registries fail-fast themselves.
        ("action", "continuous"),
        ("topic", "unknown_scheduler"),
        ("unknown_executor", "continuous"),
    ],
)
def test_legal_pairings_pass_through(executor_type, scheduler_mode):
    validate_executor_scheduler_pairing(executor_type, scheduler_mode)


@pytest.mark.parametrize(
    ("executor_type", "scheduler_mode", "message"),
    [
        ("benchmark", "continuous", "requires scheduler_mode 'wait_for_feedback'"),
        ("topic", "wait_for_feedback", "requires executor type 'benchmark'"),
    ],
)
def test_illegal_pairings_fail_fast(executor_type, scheduler_mode, message):
    with pytest.raises(DispatchStrategyError, match=message):
        validate_executor_scheduler_pairing(executor_type, scheduler_mode)


def test_chunking_strategy_exact_match():
    validate_chunking_strategy(DEFAULT_CHUNKING)
    with pytest.raises(DispatchStrategyError, match="unknown chunking strategy"):
        validate_chunking_strategy("auto_horizon")
    with pytest.raises(DispatchStrategyError, match="unknown chunking strategy"):
        validate_chunking_strategy("FULL_CHUNK")


def test_blending_strategy_exact_match():
    validate_blending_strategy("none")
    validate_blending_strategy("temporal_ensemble")
    with pytest.raises(DispatchStrategyError, match="unknown blending strategy"):
        validate_blending_strategy("rtc")
    with pytest.raises(DispatchStrategyError, match="unknown blending strategy"):
        validate_blending_strategy("Temporal_Ensemble")


def test_validate_dispatch_strategies_combines_all_rules():
    validate_dispatch_strategies(
        executor_type="topic",
        scheduler_mode="continuous",
        chunking="full_chunk",
        blending="temporal_ensemble",
    )
    with pytest.raises(DispatchStrategyError, match="wait_for_feedback"):
        validate_dispatch_strategies(
            executor_type="benchmark",
            scheduler_mode="continuous",
            chunking="full_chunk",
            blending="none",
        )
    with pytest.raises(DispatchStrategyError, match="unknown chunking strategy"):
        validate_dispatch_strategies(
            executor_type="topic",
            scheduler_mode="continuous",
            chunking="rtc",
            blending="none",
        )
    with pytest.raises(DispatchStrategyError, match="unknown blending strategy"):
        validate_dispatch_strategies(
            executor_type="topic",
            scheduler_mode="continuous",
            chunking="full_chunk",
            blending="rtc",
        )


# ---------------------------------------------------------------------------
# Launch-builder strategy resolution.
# ---------------------------------------------------------------------------


def _resolve_strategies(executor_config, chunking_yaml, blending_yaml):
    from robot_config.launch_builders.execution import _resolve_dispatch_strategies

    return _resolve_dispatch_strategies(executor_config, chunking_yaml, blending_yaml)


def test_launch_resolution_defaults_without_strategy_fields():
    assert _resolve_strategies({}, None, None) == ("full_chunk", "none")


def test_launch_resolution_explicit_blending_drives_smoothing_when_flag_absent():
    assert _resolve_strategies({}, None, "temporal_ensemble") == ("full_chunk", "temporal_ensemble")
    assert _resolve_strategies({}, None, "none") == ("full_chunk", "none")


@pytest.mark.parametrize("blending", [None, "none", "temporal_ensemble"])
@pytest.mark.parametrize("enabled", [False, True])
def test_launch_resolution_rejects_removed_flag(blending, enabled):
    with pytest.raises(DispatchStrategyError, match="has been removed.*dispatch.blending"):
        _resolve_strategies({"temporal_smoothing_enabled": enabled}, None, blending)


def test_launch_resolution_rejects_unknown_strategy_names():
    with pytest.raises(DispatchStrategyError, match="unknown chunking strategy"):
        _resolve_strategies({}, "rtc", None)
    with pytest.raises(DispatchStrategyError, match="unknown blending strategy"):
        _resolve_strategies({}, None, "rtc")


@pytest.mark.parametrize("field", ["executor_type", "scheduler_mode", "chunking", "blending"])
@pytest.mark.parametrize("value", [False, True, 0, 1, [], {}, ["full_chunk"], {"name": "none"}])
def test_resolver_rejects_nonstring_names(field, value):
    with pytest.raises(DispatchStrategyError, match="unknown"):
        resolve_dispatch_strategies(**{field: value})


@pytest.mark.parametrize("value", [False, True, None, "", "false", "true", 0, 1, [], {}])
def test_resolver_rejects_removed_smoothing_keyword(value):
    with pytest.raises(TypeError, match="unexpected keyword argument 'temporal_smoothing_enabled'"):
        resolve_dispatch_strategies(temporal_smoothing_enabled=value)
    with pytest.raises(DispatchStrategyError, match="has been removed"):
        _resolve_strategies({"temporal_smoothing_enabled": value}, None, None)


@pytest.mark.parametrize("value", [None, ""])
def test_resolver_only_documented_name_defaults(value):
    selection = resolve_dispatch_strategies(executor_type=value, scheduler_mode=value, chunking=value, blending=value)
    assert (selection.executor_type, selection.scheduler_mode, selection.chunking, selection.blending) == (
        "topic",
        "continuous",
        "full_chunk",
        "none",
    )
    assert not hasattr(selection, "temporal_smoothing_enabled")


@pytest.mark.parametrize("entrypoint", ["legacy", "scheduled"])
@pytest.mark.parametrize("blending", ["none", "temporal_ensemble"])
def test_resolver_blending_is_sole_selection(entrypoint, blending):
    selection = resolve_dispatch_strategies(entrypoint=entrypoint, blending=blending)
    assert selection.blending == blending
    assert not hasattr(selection, "temporal_smoothing_enabled")


@pytest.mark.parametrize("field", ["chunking", "blending"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_combination_validator_does_not_skip_falsy_names(field, value):
    with pytest.raises(DispatchStrategyError, match="unknown"):
        validate_dispatch_strategies(executor_type="topic", scheduler_mode="continuous", **{field: value})


@pytest.mark.parametrize("blending", ["none", "temporal_ensemble"])
def test_resolver_preserves_benchmark_combination(blending):
    selection = resolve_dispatch_strategies(
        executor_type="benchmark", scheduler_mode="wait_for_feedback", blending=blending
    )
    assert selection.executor_type == "benchmark"
    assert selection.scheduler_mode == "wait_for_feedback"
    assert selection.blending == blending


@pytest.mark.parametrize("loader_name", ["load_robot_config_dict", "load_robot_config"])
@pytest.mark.parametrize("value", [False, True, None, "false", 0, [], {}])
@pytest.mark.parametrize("inherited", [False, True])
def test_config_loaders_reject_removed_smoothing(tmp_path, loader_name, value, inherited):
    import yaml

    from robot_config import loader

    config = {
        "robot": {
            "name": "test",
            "control_modes": {
                "model_inference": {
                    "executor": {"temporal_smoothing_enabled": value},
                    "dispatch": {"blending": "none"},
                }
            },
        }
    }
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(config))
    if inherited:
        path = tmp_path / "overlay.yaml"
        path.write_text(yaml.safe_dump({"robot": {"base_config": "robot", "name": "overlay"}}))
    with pytest.raises(ValueError, match=r"executor.temporal_smoothing_enabled has been removed.*dispatch.blending"):
        getattr(loader, loader_name)(path)


@pytest.mark.parametrize("blending", ["none", "temporal_ensemble"])
def test_config_loader_preserves_blending_and_executor_settings(tmp_path, blending):
    import yaml

    from robot_config.loader import load_robot_config_dict

    mode = {
        "executor": {"queue_size": 23, "temporal_ensemble_coeff": 0.04},
        "dispatch": {"blending": blending, "scheduler": "continuous", "execution_timeout_sec": 17.0},
    }
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": {"name": "test", "control_modes": {"model_inference": mode}}}))
    assert load_robot_config_dict(path)["control_modes"]["model_inference"] == mode

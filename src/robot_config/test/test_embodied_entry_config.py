from pathlib import Path

import pytest
import yaml

from robot_config.loader import (
    _validate_sound_orientation_config,
    load_robot_config,
    load_robot_config_dict,
    validate_config,
    validate_embodied_launch_dict,
)
from robot_config.timeout_policy import resolve_embodied_timeout_policy
from robot_skill_cli.catalog import compile_local_snapshot

GRIPPER_TRAJECTORY_DURATION_SEC = 1.0


def _snapshot(config_path: Path):
    return compile_local_snapshot(load_robot_config_dict(config_path), config_path)


def _sorting_hat_policy(*, enabled: bool, announce: bool = False) -> dict:
    return {
        "enabled": enabled,
        "announce": announce,
        "handler": "sorting_hat_v1",
        "summary": "Choose a Hogwarts house.",
    }


def _voice_tts(**overrides) -> dict:
    return {
        "enabled": True,
        "bundle_path": "models/zipvoice",
        "deployment": "test_deployment",
        "service_name": "/voice_tts/synthesize",
        **overrides,
    }


@pytest.mark.parametrize(
    "config_name",
    ["so101_single_arm"],
)
def test_compiled_profile_includes_dance_basic(config_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / f"{config_name}.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    skill_templates = _snapshot(config_path).templates

    assert "dance_basic" in skill_templates
    primitive_sequence = skill_templates["dance_basic"]["primitive_sequence"]
    assert primitive_sequence
    trajectory_step = next(
        step for step in primitive_sequence if step["primitive_name"] == "move_through_joint_positions"
    )
    assert trajectory_step["joint_waypoints"]


def test_default_loader_uses_robot_config_environment_path(monkeypatch):
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    monkeypatch.setenv("ROBOT_CONFIG", str(config_path))

    config = load_robot_config_dict()

    assert config["_config_path"] == str(config_path.resolve())


@pytest.mark.parametrize("required_control_mode", ["unknown_mode", 1])
def test_loader_rejects_invalid_global_skill_required_control_mode(tmp_path, required_control_mode):
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "name": "test_robot",
                    "control_modes": {"moveit_planning": {}},
                    "skill_required_control_mode": required_control_mode,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="skill_required_control_mode"):
        load_robot_config_dict(config_path)


def test_so101_skill_gateway_control_mode_is_global_and_safety_has_no_motion_authorization():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]
    embodied = raw_config["embodied"]

    assert raw_config["skill_required_control_mode"] == "moveit_planning"
    assert raw_config["skill_required_control_mode"] in raw_config["control_modes"]
    assert "motion_authorized" not in embodied["safety"]
    assert "skill_templates" not in embodied

    typed_config = load_robot_config(config_path)
    assert not hasattr(typed_config.embodied, "motion_authorized")


def test_lekiwi_sound_orientation_is_configured_but_default_disabled():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "lekiwi_nav_grasp.yaml"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]

    sound_orientation = config["embodied"]["idle_behaviors"]["sound_orientation"]
    assert sound_orientation["enabled"] is False
    assert sound_orientation["skill_name"] == "nav_turn"
    assert sound_orientation["trigger_phrases"] == ["转向我"]


def test_sound_orientation_requires_voice_and_navigation_inputs(tmp_path):
    source_path = Path(__file__).parent.parent / "config" / "robots" / "lekiwi_nav_grasp.yaml"
    copied_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    embodied = copied_config["robot"]["embodied"]
    embodied["enabled"] = True
    embodied["idle_behaviors"] = {"sound_orientation": {"enabled": True}}
    copied_config["robot"]["voice_asr"]["enabled"] = False
    copied_config["robot"]["speech_direction"]["enabled"] = False
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(copied_config), encoding="utf-8")

    errors = _validate_sound_orientation_config(copied_config["robot"])

    assert any("speech_direction.enabled" in error for error in errors)
    assert any("voice_asr.enabled" in error for error in errors)


@pytest.mark.parametrize(
    ("value", "field"),
    [
        ("false", "enabled"),
        ({"enabled": True, "trigger_phrases": ["!!!"]}, "trigger_phrases"),
        ({"enabled": True, "status_retry_sec": 0.0}, "status_retry_sec"),
    ],
)
def test_sound_orientation_rejects_invalid_behavior_config(value, field):
    config = {
        "embodied": {
            "enabled": True,
            "idle_behaviors": {"sound_orientation": value if isinstance(value, dict) else {"enabled": value}},
        },
        "voice_asr": {"enabled": True},
        "speech_direction": {"enabled": True},
        "control_modes": {"base_navigation": {}},
        "navigation": {
            "enabled": True,
            "command_server": {"enabled": True, "action_name": "/navigation/execute"},
        },
    }

    errors = _validate_sound_orientation_config(config)

    assert any(field in error for error in errors)


def test_compiled_skills_match_profile_enabled_set():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    profile_path = config_path.parents[3] / "skill_catalog" / "config" / "profiles" / "so101_single_arm.yaml"
    expected = {entry["name"] for entry in yaml.safe_load(profile_path.read_text(encoding="utf-8"))["enabled_skills"]}

    assert set(_snapshot(config_path).enabled_skill_names) == expected


def test_production_robot_yaml_has_no_inline_skill_catalog():
    source_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    embodied = yaml.safe_load(source_path.read_text(encoding="utf-8"))["robot"]["embodied"]
    assert "skill_templates" not in embodied
    assert embodied["skill_catalog_profile"] == "so101_single_arm"


def test_embodied_visual_games_are_typed_without_asr_routing_config():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    games = config.embodied.visual_games
    sorting_hat = games["sorting_hat"]
    assert "enabled" in sorting_hat
    assert sorting_hat["handler"] == "sorting_hat_v1"
    assert sorting_hat["summary"]
    assert "trigger_mode" not in sorting_hat
    assert not hasattr(config.embodied, "entry")
    assert config.embodied.start_visual_game_service == "/embodied/start_visual_game"
    assert config.embodied.get_visual_game_result_service == "/embodied/get_visual_game_result"


def test_enabled_game_requires_perception_enabled():
    """An enabled visual game with perception disabled must fail validation."""
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    config.embodied.enabled = True
    config.embodied.perception = {**config.embodied.perception, "enabled": False}
    config.embodied.visual_games = {"sorting_hat": _sorting_hat_policy(enabled=True)}

    errors = validate_config(config)
    assert any("visual_games" in error for error in errors)


def test_disabled_games_do_not_require_perception():
    """All games disabled: perception may stay off without a validation error."""
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    config.embodied.enabled = True
    config.embodied.perception = {**config.embodied.perception, "enabled": False}
    config.embodied.visual_games = {"sorting_hat": _sorting_hat_policy(enabled=False)}

    errors = validate_config(config)
    assert not any("visual_games" in error for error in errors)


def test_launch_dict_enabled_game_without_perception_is_rejected():
    """The raw-dict launch gate rejects a game enabled + perception disabled."""
    config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": False},
            "visual_games": {"sorting_hat": _sorting_hat_policy(enabled=True, announce=True)},
        }
    }
    errors = validate_embodied_launch_dict(config)
    assert any("visual_games" in error for error in errors)


def test_raw_loader_rejects_enabled_game_without_perception(tmp_path):
    source_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    copied_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    copied_config["robot"]["embodied"]["enabled"] = True
    copied_config["robot"]["embodied"]["perception"]["enabled"] = False
    copied_config["robot"]["embodied"]["visual_games"]["sorting_hat"]["enabled"] = True
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(copied_config), encoding="utf-8")

    with pytest.raises(ValueError, match="embodied.perception.enabled"):
        load_robot_config_dict(config_path)


def test_launch_dict_enabled_game_with_perception_is_accepted_in_hermes_mode():
    """Visual games coexist with the Hermes runtime via an independent gateway control plane."""
    config = {
        "voice_tts": _voice_tts(),
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": _sorting_hat_policy(enabled=True, announce=True)},
        },
    }
    errors = validate_embodied_launch_dict(config)
    assert errors == []


def test_launch_dict_accepts_incubating_agent_entry():
    config = {
        "embodied": {
            "enabled": True,
            "entry_mode": "agent",
            "agent": {
                "enabled": True,
                "incubation": True,
                "execution_enabled": False,
                "test_allowlist": ["wave_hello"],
                "ledger_path": "/tmp/ibrobot-agent-test.sqlite3",
                "conversation_path": "/tmp/ibrobot-agent-conversation-test.sqlite3",
                "deployment_lock_path": "/tmp/ibrobot-agent-test.lock",
            },
        }
    }

    assert validate_embodied_launch_dict(config) == []


def test_launch_dict_rejects_agent_execution_without_allowlist():
    config = {
        "embodied": {
            "enabled": True,
            "entry_mode": "agent",
            "agent": {
                "enabled": True,
                "incubation": True,
                "execution_enabled": True,
                "test_allowlist": [],
                "ledger_path": "/tmp/ibrobot-agent-test.sqlite3",
                "conversation_path": "/tmp/ibrobot-agent-conversation-test.sqlite3",
                "deployment_lock_path": "/tmp/ibrobot-agent-test.lock",
            },
        }
    }

    errors = validate_embodied_launch_dict(config)
    assert "embodied.agent.test_allowlist must be non-empty when execution is enabled" in errors


def test_launch_dict_rejects_agent_without_incubation_marker():
    config = {
        "embodied": {
            "enabled": True,
            "entry_mode": "agent",
            "agent": {
                "enabled": True,
                "execution_enabled": False,
                "ledger_path": "/tmp/ibrobot-agent-test.sqlite3",
                "conversation_path": "/tmp/ibrobot-agent-conversation-test.sqlite3",
                "deployment_lock_path": "/tmp/ibrobot-agent-test.lock",
            },
        }
    }

    errors = validate_embodied_launch_dict(config)
    assert "embodied.agent.incubation must be true for the incubating Agent entry" in errors


def test_launch_dict_rejects_literal_agent_planner_api_key():
    config = {
        "embodied": {
            "enabled": True,
            "entry_mode": "agent",
            "agent": {
                "enabled": True,
                "incubation": True,
                "execution_enabled": False,
                "test_allowlist": ["wave_hello"],
                "ledger_path": "/tmp/ibrobot-agent-test.sqlite3",
                "conversation_path": "/tmp/ibrobot-agent-conversation-test.sqlite3",
                "deployment_lock_path": "/tmp/ibrobot-agent-test.lock",
                "planner": {
                    "mode": "vlm",
                    "provider": "openai_compatible",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "sk-literal-secret",
                    "model": "test-model",
                },
            },
        }
    }

    errors = validate_embodied_launch_dict(config)
    assert "embodied.agent.planner must not contain a literal api_key; use api_key_env instead" in errors


def test_launch_dict_rejects_removed_visual_game_trigger_mode():
    config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": {**_sorting_hat_policy(enabled=True), "trigger_mode": "asr"}},
        },
    }
    errors = validate_embodied_launch_dict(config)
    assert any("trigger_mode" in error and "no longer supports" in error for error in errors)


@pytest.mark.parametrize("entry", [{}, {"visual_game_aliases": {"sorting_hat": ["分院帽"]}}])
def test_launch_dict_rejects_removed_visual_game_entry(entry):
    config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": _sorting_hat_policy(enabled=True)},
            "entry": entry,
        },
    }

    errors = validate_embodied_launch_dict(config)
    assert any("embodied.entry is no longer supported" in error for error in errors)


def test_launch_dict_rejects_colliding_visual_game_service_names():
    """Launch overrides that collide start/result services are caught by the raw-dict gate."""
    config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": _sorting_hat_policy(enabled=True)},
            "start_visual_game_service": "/embodied/same",
            "get_visual_game_result_service": "/embodied/same",
        },
    }
    errors = validate_embodied_launch_dict(config)
    assert any("must be different" in error for error in errors)


def test_launch_dict_rejects_non_positive_visual_game_result_capacity():
    config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": _sorting_hat_policy(enabled=True)},
            "visual_game_result_capacity": 0,
        },
    }
    errors = validate_embodied_launch_dict(config)
    assert any("result_capacity must be a positive integer" in error for error in errors)


def test_launch_dict_enabled_game_does_not_require_tts_runtime():
    config = {
        "voice_tts": {"enabled": False},
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": _sorting_hat_policy(enabled=True, announce=True)},
        },
    }

    assert validate_embodied_launch_dict(config) == []


def test_typed_enabled_game_does_not_require_complete_tts_config():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)
    config.embodied.enabled = True
    config.embodied.perception = {**config.embodied.perception, "enabled": True}
    config.embodied.visual_games = {"sorting_hat": _sorting_hat_policy(enabled=True, announce=True)}
    config.voice_tts.enabled = True
    config.voice_tts.deployment = ""

    errors = validate_config(config)
    assert "voice_tts.deployment is required when voice_tts.enabled is true" in errors
    assert not any("when visual games" in error for error in errors)


def test_visual_game_timeout_must_exceed_model_idle_timeout():
    with pytest.raises(ValueError, match="visual_game_timeout_sec.*greater than model_idle_timeout_sec"):
        resolve_embodied_timeout_policy(
            {
                "timeouts": {
                    "model_idle_timeout_sec": 30.0,
                    "visual_game_timeout_sec": 30.0,
                }
            }
        )


def test_launch_dict_skips_when_embodied_disabled():
    """A disabled embodied stack is never gated on game/perception consistency."""
    config = {
        "embodied": {
            "enabled": False,
            "perception": {"enabled": False},
            "visual_games": {"sorting_hat": _sorting_hat_policy(enabled=True, announce=True)},
        }
    }
    assert validate_embodied_launch_dict(config) == []


def test_launch_dict_unknown_visual_game_handler_is_rejected():
    policy = _sorting_hat_policy(enabled=False)
    policy["handler"] = "missing_v1"
    errors = validate_embodied_launch_dict(
        {"embodied": {"enabled": True, "perception": {"enabled": False}, "visual_games": {"bad": policy}}}
    )

    assert any("unsupported visual game handler" in error for error in errors)


def test_raw_loader_rejects_unknown_visual_game_handler_when_embodied_disabled(tmp_path):
    source_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    copied_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    copied_config["robot"]["embodied"]["enabled"] = False
    copied_config["robot"]["embodied"]["visual_games"]["sorting_hat"]["handler"] = "missing_v1"
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(copied_config), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported visual game handler"):
        load_robot_config_dict(config_path)


def test_raw_loader_rejects_removed_entry_scoped_visual_games(tmp_path):
    source_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    copied_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    embodied = copied_config["robot"]["embodied"]
    embodied["entry"] = {"visual_games": {"sorting_hat": {"enabled": False, "trigger_aliases": ["分院帽"]}}}
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(copied_config), encoding="utf-8")

    with pytest.raises(ValueError, match="embodied.entry is no longer supported"):
        load_robot_config_dict(config_path)


def test_visual_game_service_names_must_be_distinct():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)
    config.embodied.get_visual_game_result_service = config.embodied.start_visual_game_service

    assert any("start and result services must be different" in error for error in validate_config(config))


def test_raw_loader_rejects_duplicate_visual_game_service_names(tmp_path):
    source_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    copied_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    embodied = copied_config["robot"]["embodied"]
    embodied["get_visual_game_result_service"] = embodied["start_visual_game_service"]
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(copied_config), encoding="utf-8")

    with pytest.raises(ValueError, match="start and result services must be different"):
        load_robot_config_dict(config_path)


def test_embodied_config_keeps_only_supported_direct_skills():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    skill_templates = _snapshot(config_path).templates

    assert "dance_basic" in skill_templates
    assert "pick_named_target" not in skill_templates
    assert "place_named_pose" not in skill_templates
    assert "observe_target_area" not in skill_templates


@pytest.mark.parametrize(
    ("skill_name", "pose_name"),
    [
        ("recover_safe_pose", "home"),
        ("inspect_scene", "observe_table"),
        ("recover_zero_pose", "zero"),
    ],
)
def test_embodied_named_pose_skills_map_to_configured_poses(skill_name, pose_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config_dict(config_path)
    skill_templates = _snapshot(config_path).templates

    assert pose_name in config["embodied"]["named_poses"]
    step = skill_templates[skill_name]["primitive_sequence"][0]
    assert dict(step) == {"primitive_name": "move_to_named_pose", "pose_name": pose_name}


def test_enabled_embodied_config_uses_configured_default_place_pose():
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config(config_path)

    config.embodied.enabled = True

    assert validate_config(config) == []
    assert config.embodied.default_place_name in config.embodied.named_poses


@pytest.mark.parametrize(
    "skill_name",
    ["wave_hello", "nod_yes", "shake_no", "act_cute", "happy_spin_upright"],
)
def test_social_gesture_duration_estimate_covers_configured_motion(skill_name):
    config_path = Path(__file__).parent.parent / "config" / "robots" / "so101_single_arm.yaml"
    skill = _snapshot(config_path).templates[skill_name]
    manifest_path = config_path.parents[3] / "skill_catalog" / "config" / "skills" / skill_name / "manifest.yaml"
    description = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["description"]

    configured_duration = 0.0
    if skill.get("initial_gripper_state") in {"open", "closed"}:
        configured_duration += GRIPPER_TRAJECTORY_DURATION_SEC
    for step in skill["primitive_sequence"]:
        primitive_name = step["primitive_name"]
        if primitive_name == "move_to_joint_positions":
            configured_duration += float(step.get("duration_sec", 0.4))
        elif primitive_name == "move_through_joint_positions":
            configured_duration += len(step["joint_waypoints"]) * float(step["waypoint_duration_sec"])
        elif primitive_name in {"open_gripper", "close_gripper"}:
            configured_duration += GRIPPER_TRAJECTORY_DURATION_SEC

    assert float(description["duration_sec_estimate"]) >= configured_duration

import json
from pathlib import Path

import pytest
from launch.actions import DeclareLaunchArgument, EmitEvent, SetEnvironmentVariable
from launch_ros.actions import Node

from embodied_bringup.launch_builders.embodied import _resolve_development_source_root, generate_embodied_nodes
from robot_config.launch_builders.perception_models import generate_perception_model_nodes
from robot_config.loader import load_robot_config_dict


def _sorting_hat_game(*, enabled: bool = True, announce: bool = False) -> dict:
    return {
        "enabled": enabled,
        "announce": announce,
        "handler": "sorting_hat_v1",
        "summary": "Sort",
    }


def _voice_tts(**overrides) -> dict:
    return {
        "enabled": True,
        "bundle_path": "models/zipvoice",
        "deployment": "test_deployment",
        "service_name": "/voice_tts/synthesize",
        "playback_service_name": "/voice_tts/play",
        "playback_timeout_sec": 300.0,
        **overrides,
    }


def _normalize_launch_param_mapping(raw_params):
    normalized = {}
    for raw_key, raw_value in raw_params.items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            value = "".join(getattr(item, "text", str(item)) for item in raw_value)
        else:
            value = raw_value
        normalized[key] = value
    return normalized


def _decode_launch_json_string(raw_value: str):
    normalized = raw_value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    return json.loads(normalized)


def _decode_launch_string(raw_value: str):
    normalized = raw_value.strip()
    if normalized.endswith("\n..."):
        normalized = normalized[:-4].rstrip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    return normalized


def _skill_executor_params(nodes):
    skill_executor = next(
        node
        for node in nodes
        if node.__dict__.get("_Node__package") == "skill_library"
        and node.__dict__.get("_Node__node_executable") == "skill_executor_node"
    )
    return _normalize_launch_param_mapping(skill_executor._Node__parameters[0])


def _normalize_launch_environment(node):
    return {
        "".join(getattr(item, "text", str(item)) for item in key): "".join(
            getattr(item, "text", str(item)) for item in value
        )
        for key, value in node.additional_env
    }


def test_hermes_entry_does_not_launch_voice_routing_node():
    robot_config = {
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "execution": {},
            "entry": {},
            "named_poses": {},
            "named_targets": {},
            "skill_templates": {},
            "safety": {},
            "planner": {},
            "perception": {},
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="moveit_planning")
    node_names = {vars(node)["_Node__node_name"] for node in nodes}
    assert "task_entry_node" not in node_names
    assert "task_executor_node" not in node_names


def test_launch_does_not_inject_legacy_skill_templates():
    embodied = {
        "enabled": True,
        "entry_mode": "hermes",
        "execution": {},
        "entry": {},
        "named_poses": {},
        "named_targets": {},
        "safety": {},
        "planner": {},
        "perception": {},
    }
    nodes = generate_embodied_nodes({"embodied": embodied}, active_control_mode="moveit_planning")
    params = _skill_executor_params(nodes)
    assert "skill_templates_json" not in params


@pytest.mark.parametrize("config_name", ["so101_single_arm"])
def test_generate_embodied_nodes_passes_arm_joint_metadata(config_name):
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / f"{config_name}.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config_dict(config_path)
    config["embodied"]["enabled"] = True
    nodes = generate_embodied_nodes(config, "moveit_planning")

    params = _skill_executor_params(nodes)

    assert _decode_launch_json_string(params["arm_joint_names_json"]) == ["1", "2", "3", "4", "5"]
    assert set(_decode_launch_json_string(params["joint_limits_json"]).keys()) >= {"1", "2", "3", "4", "5"}


def test_launch_injects_gateway_startup_params_from_runtime_and_ssot():
    robot_config = {
        "name": "test_robot",
        "default_control_mode": "model_inference",
        "skill_required_control_mode": "moveit_planning",
        "embodied": {
            "enabled": True,
            "skill_gateway_status_service": "/test/gateway_status",
            "timeouts": {
                "task_budget_sec": 90.0,
                "default_skill_timeout_sec": 12.0,
                "robot_state_freshness_sec": 0.25,
                "scene_freshness_sec": 0.75,
                "model_idle_timeout_sec": 45.0,
                "rpc_timeout_sec": 2.0,
                "gripper_settle_sec": 0.8,
            },
        },
    }

    nodes = generate_embodied_nodes(
        robot_config,
        active_control_mode="moveit_runtime_override",
        motion_authorized=True,
    )
    params = _skill_executor_params(nodes)

    assert params["motion_authorized"] is True
    assert _decode_launch_string(params["active_control_mode"]) == "moveit_runtime_override"
    assert _decode_launch_string(params["skill_required_control_mode"]) == "moveit_planning"
    assert _decode_launch_string(params["skill_gateway_status_service"]) == "/test/gateway_status"
    assert _decode_launch_string(params["robot_name"]) == "test_robot"
    assert params.get("context_schema_version") == 1
    assert "navigation_action_name" not in params
    assert params["default_skill_timeout_sec"] == 12.0
    assert params["task_budget_sec"] == 90.0
    assert params["robot_state_freshness_sec"] == 0.25
    assert params["scene_freshness_sec"] == 0.75
    assert params["model_idle_timeout_sec"] == 45.0
    assert params["rpc_timeout_sec"] == 2.0
    assert params["gripper_settle_sec"] == 0.8


def test_navigation_profile_uses_base_navigation_mode_and_action_endpoint():
    robot_config = {
        "name": "lekiwi_lidar",
        "default_control_mode": "base_navigation",
        "skill_required_control_mode": "base_navigation",
        "control_modes": {"base_navigation": {"controllers": ["base_velocity_controller"]}},
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "skill_catalog_profile": "lekiwi_lidar",
        },
        "navigation": {
            "enabled": True,
            "command_server": {
                "enabled": True,
                "action_name": "/robot/navigation/execute",
            },
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="base_navigation")
    params = _skill_executor_params(nodes)

    assert _decode_launch_string(params["active_control_mode"]) == "base_navigation"
    assert _decode_launch_string(params["skill_required_control_mode"]) == "base_navigation"
    assert _decode_launch_string(params["navigation_action_name"]) == "/robot/navigation/execute"
    assert params.get("context_schema_version") == 2
    assert all(vars(node).get("_Node__package") != "robot_moveit" for node in nodes)


def test_sound_orientation_node_is_projected_from_robot_config():
    robot_config = {
        "name": "lekiwi_test",
        "default_control_mode": "base_navigation",
        "skill_required_control_mode": "base_navigation",
        "control_modes": {"base_navigation": {"controllers": ["base_velocity_controller"]}},
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "skill_catalog_profile": "lekiwi_test",
            "idle_behaviors": {
                "sound_orientation": {
                    "enabled": True,
                    "trigger_phrases": ["转向我"],
                    "deadband_deg": 20.0,
                }
            },
        },
        "navigation": {
            "enabled": True,
            "command_server": {"enabled": True, "action_name": "/navigation/execute"},
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="base_navigation")
    node = next(node for node in nodes if vars(node).get("_Node__node_name") == "sound_orientation_node")
    params = _normalize_launch_param_mapping(node._Node__parameters[0])

    assert "trigger_phrases" in params
    assert params["deadband_deg"] == 20.0
    assert _decode_launch_string(params["skill_name"]) == "nav_turn"
    assert _decode_launch_string(params["skill_action_name"]) == "/embodied/execute_skill"


def test_hybrid_profile_projects_runtime_control_mode_switching_parameters():
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "lekiwi_nav_grasp.yaml"
    config = load_robot_config_dict(config_path)

    params = _skill_executor_params(generate_embodied_nodes(config, active_control_mode="moveit_planning"))

    assert params["context_schema_version"] == 3
    assert _decode_launch_json_string(params["supported_control_modes_json"]) == [
        "moveit_planning",
        "base_navigation",
    ]
    assert _decode_launch_string(params["motion_mode_service"]) == "motion_mode/set_navigation_enabled"
    assert _decode_launch_string(params["semantic_map_target_service"]) == "/semantic_mapping/resolve_target"
    assert params["semantic_map_stand_off_distance_m"] == 0.3


def test_navigation_endpoint_missing_action_name_raises():
    import pytest

    robot_config = {
        "name": "lekiwi_lidar",
        "default_control_mode": "base_navigation",
        "skill_required_control_mode": "base_navigation",
        "control_modes": {"base_navigation": {"controllers": ["base_velocity_controller"]}},
        "embodied": {"enabled": True, "entry_mode": "hermes", "skill_catalog_profile": "lekiwi_lidar"},
        "navigation": {
            "enabled": True,
            "command_server": {"enabled": True, "action_name": ""},
        },
    }
    with pytest.raises(ValueError, match="action_name"):
        generate_embodied_nodes(robot_config, active_control_mode="base_navigation")


def test_navigation_endpoint_relative_name_raises():
    import pytest

    robot_config = {
        "name": "lekiwi_lidar",
        "default_control_mode": "base_navigation",
        "skill_required_control_mode": "base_navigation",
        "control_modes": {"base_navigation": {"controllers": ["base_velocity_controller"]}},
        "embodied": {"enabled": True, "entry_mode": "hermes", "skill_catalog_profile": "lekiwi_lidar"},
        "navigation": {
            "enabled": True,
            "command_server": {"enabled": True, "action_name": "navigation/execute"},
        },
    }
    with pytest.raises(ValueError, match="absolute ROS name"):
        generate_embodied_nodes(robot_config, active_control_mode="base_navigation")


def test_legacy_navigation_endpoint_field_raises():
    import pytest

    robot_config = {
        "name": "lekiwi_lidar",
        "default_control_mode": "base_navigation",
        "skill_required_control_mode": "base_navigation",
        "control_modes": {"base_navigation": {"controllers": ["base_velocity_controller"]}},
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "skill_catalog_profile": "lekiwi_lidar",
            "execution": {"navigation_action_name": "/robot/navigation/execute"},
        },
        "navigation": {
            "enabled": True,
            "command_server": {"enabled": True, "action_name": "/robot/navigation/execute"},
        },
    }
    with pytest.raises(ValueError, match="navigation_action_name is retired"):
        generate_embodied_nodes(robot_config, active_control_mode="base_navigation")


def test_disabled_command_server_uses_v1_context_without_navigation_endpoint():
    robot_config = {
        "name": "lekiwi_lidar",
        "default_control_mode": "base_navigation",
        "skill_required_control_mode": "base_navigation",
        "control_modes": {"base_navigation": {"controllers": ["base_velocity_controller"]}},
        "embodied": {"enabled": True, "entry_mode": "hermes", "skill_catalog_profile": "lekiwi_lidar"},
        "navigation": {"enabled": True, "command_server": {"enabled": False}},
    }

    params = _skill_executor_params(generate_embodied_nodes(robot_config, active_control_mode="base_navigation"))

    assert params.get("context_schema_version") == 1
    assert "navigation_action_name" not in params


def test_installed_profile_falls_back_to_ament_skill_catalog_source():
    robot_config = {
        "_config_path": "/opt/ibrobot/install/robot_config/share/robot_config/config/robots/robot.yaml",
        "embodied": {
            "enabled": True,
            "skill_catalog_source_mode": "development",
            "skill_catalog_source_root": "src/skill_catalog",
        },
    }

    params = _skill_executor_params(generate_embodied_nodes(robot_config, "moveit_planning"))

    assert _decode_launch_string(params["skill_catalog_source_mode"]) == "installed"
    assert _decode_launch_string(params["skill_catalog_source_root"]) == ""


def test_direct_builder_defaults_gateway_motion_authorization_to_false():
    nodes = generate_embodied_nodes(
        {"embodied": {"enabled": True, "safety": {"motion_authorized": True}}},
        "moveit_planning",
    )

    params = _skill_executor_params(nodes)

    assert params["motion_authorized"] is False


def test_no_interaction_skills_node_is_generated():
    robot_config = {
        "voice_tts": _voice_tts(),
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "execution": {},
            "visual_games": {"sorting_hat": _sorting_hat_game(announce=True)},
            "perception": {"enabled": True},
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="moveit_planning")

    node_names = [vars(node)["_Node__node_name"] for node in nodes]
    assert "interaction_skills_node" not in node_names
    assert "visual_game_gateway_node" in node_names

    node_names = {vars(node)["_Node__node_name"] for node in nodes}
    assert "task_entry_node" not in node_names
    assert "perception_service_node" in node_names


def test_hermes_entry_mode_omits_voice_pipeline_nodes():
    nodes = generate_embodied_nodes(
        {"embodied": {"enabled": True, "entry_mode": "hermes"}},
        active_control_mode="moveit_planning",
    )

    node_names = {vars(node)["_Node__node_name"] for node in nodes}
    assert {"safety_guard_node", "skill_executor_node", "agent_plan_node"} <= node_names
    assert {"task_entry_node", "task_executor_node"}.isdisjoint(node_names)


def test_development_catalog_resolves_from_source_module_when_config_is_installed():
    installed_config = Path(__file__).parents[3] / "install/share/robot_config/config/robots/robot.yaml"

    resolved = _resolve_development_source_root(installed_config, "src/skill_catalog")

    assert resolved == (Path(__file__).parents[3] / "src/skill_catalog").resolve()


def test_non_hermes_entry_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="entry_mode must be hermes"):
        generate_embodied_nodes(
            {"embodied": {"enabled": True, "entry_mode": "voice"}},
            active_control_mode="moveit_planning",
        )


def test_non_moveit_game_launches_only_gateway_and_perception():
    robot_config = {
        "name": "test_robot",
        "voice_tts": _voice_tts(),
        "embodied": {
            "enabled": True,
            "visual_games": {"sorting_hat": _sorting_hat_game()},
            "perception": {"enabled": True},
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="model_inference")

    assert {vars(node)["_Node__node_name"] for node in nodes} == {
        "visual_game_gateway_node",
        "perception_service_node",
    }


def test_tts_enabled_game_adds_announcer_using_existing_service_config():
    robot_config = {
        "name": "test_robot",
        "voice_tts": _voice_tts(
            service_name="/custom/tts",
            playback_service_name="/custom/play",
            tts_timeout_sec=8.0,
            playback_timeout_sec=9.0,
        ),
        "embodied": {
            "enabled": True,
            "visual_games": {"sorting_hat": _sorting_hat_game(announce=True)},
            "perception": {"enabled": True},
            "visual_game_event_topic": "/custom/game_events",
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="model_inference")
    announcer = next(node for node in nodes if vars(node)["_Node__node_name"] == "visual_game_announcer_node")
    params = _normalize_launch_param_mapping(announcer._Node__parameters[0])

    assert _decode_launch_string(params["event_topic"]) == "/custom/game_events"
    assert _decode_launch_string(params["tts_service"]) == "/custom/tts"
    assert _decode_launch_string(params["playback_service"]) == "/custom/play"
    assert float(params["tts_timeout_sec"]) == 8.0
    assert float(params["playback_timeout_sec"]) == 9.0


def test_tts_enabled_game_adds_shared_announcer():
    robot_config = {
        "name": "test_robot",
        "voice_tts": _voice_tts(),
        "embodied": {
            "enabled": True,
            "visual_games": {"sorting_hat": _sorting_hat_game(announce=True)},
            "perception": {"enabled": True},
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="model_inference")

    assert "visual_game_announcer_node" in {vars(node)["_Node__node_name"] for node in nodes}


@pytest.mark.parametrize(
    "voice_tts",
    [
        {"enabled": False},
        {
            "enabled": True,
            "bundle_path": "models/zipvoice",
            "deployment": "",
            "service_name": "/voice_tts/synthesize",
            "playback_service_name": "/voice_tts/play",
        },
        {
            "enabled": True,
            "bundle_path": "models/zipvoice",
            "deployment": "test_deployment",
            "service_name": "/voice_tts/synthesize",
            "playback_service_name": "",
        },
    ],
)
def test_enabled_visual_game_without_complete_tts_config_skips_announcer(voice_tts):
    robot_config = {
        "voice_tts": voice_tts,
        "embodied": {
            "enabled": True,
            "visual_games": {"sorting_hat": _sorting_hat_game(announce=True)},
            "perception": {"enabled": True},
        },
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="model_inference")
    assert "visual_game_announcer_node" not in {vars(node)["_Node__node_name"] for node in nodes}


def test_non_moveit_without_enabled_game_preserves_motion_mode_error():
    robot_config = {
        "embodied": {
            "enabled": True,
            "visual_games": {"sorting_hat": _sorting_hat_game(enabled=False)},
        }
    }

    with pytest.raises(ValueError, match="MoveIt-compatible"):
        generate_embodied_nodes(robot_config, active_control_mode="model_inference")


class _FakeLaunchContext:
    def __init__(self, launch_configurations):
        self.launch_configurations = launch_configurations


def _load_launch_module():
    """Load the ``.launch.py`` file by path (not an importable package module)."""
    import importlib.util

    launch_path = Path(__file__).parents[1] / "launch" / "embodied_pipeline.launch.py"
    spec = importlib.util.spec_from_file_location("embodied_pipeline_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authorize_motion_launch_argument_defaults_to_false():
    module = _load_launch_module()
    description = module.generate_launch_description()
    authorize_motion = next(
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == "authorize_motion"
    )

    assert authorize_motion.default_value[0].text == "false"


def test_navigation_stage_launch_argument_is_declared_and_forwarded(monkeypatch, tmp_path):
    module = _load_launch_module()
    description = module.generate_launch_description()
    nav_stage = next(
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == "nav_stage"
    )
    assert nav_stage.default_value[0].text == ""
    control_mode = next(
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == "control_mode"
    )
    assert control_mode.default_value[0].text == ""

    captured = {}

    def load_config(robot_name, config_path, selected_stage):
        captured["load"] = (robot_name, config_path, selected_stage)
        return {
            "default_control_mode": "base_navigation",
            "control_modes": {"base_navigation": {"controllers": []}},
            "embodied": {"enabled": False, "perception": {"enabled": False}},
        }

    monkeypatch.setattr(module, "_load_config", load_config)
    monkeypatch.setattr(module, "get_package_share_directory", lambda _package: str(tmp_path))
    actions = module.launch_setup(
        _FakeLaunchContext(
            {
                "robot_config": "lekiwi_lidar",
                "nav_stage": "navigation",
                "control_mode": control_mode.default_value[0].text,
                "with_embodied": "false",
            }
        )
    )

    assert captured["load"] == ("lekiwi_lidar", "", "navigation")
    base_launch_arguments = dict(actions[0]._IncludeLaunchDescription__launch_arguments)
    assert base_launch_arguments["nav_stage"] == "navigation"
    assert base_launch_arguments["control_mode"] == "base_navigation"


def test_hybrid_navigation_entry_starts_moveit_for_complete_runtime(monkeypatch, tmp_path):
    module = _load_launch_module()
    monkeypatch.setattr(
        module,
        "_load_config",
        lambda *_args: {
            "name": "lekiwi_handeye_realsense_grasp_lidar",
            "nav_stage": "hybrid",
            "default_control_mode": "base_navigation",
            "skill_required_control_mode": "moveit_planning",
            "control_modes": {"base_navigation": {"controllers": []}},
            "navigation": {"enabled": True},
            "grasp_execution": {"enabled": True},
            "embodied": {"enabled": False, "perception": {"enabled": False}},
        },
    )
    monkeypatch.setattr(module, "get_package_share_directory", lambda _package: str(tmp_path))

    actions = module.launch_setup(
        _FakeLaunchContext(
            {
                "robot_config": "lekiwi_nav_grasp",
                "with_embodied": "false",
            }
        )
    )

    base_launch_arguments = dict(actions[0]._IncludeLaunchDescription__launch_arguments)
    assert base_launch_arguments["with_moveit"] == "true"


def test_hybrid_teleop_override_does_not_auto_start_moveit(monkeypatch, tmp_path):
    module = _load_launch_module()
    monkeypatch.setattr(
        module,
        "_load_config",
        lambda *_args: {
            "name": "lekiwi_handeye_realsense_grasp_lidar",
            "nav_stage": "hybrid",
            "default_control_mode": "base_navigation",
            "skill_required_control_mode": "moveit_planning",
            "control_modes": {"teleop": {"controllers": []}},
            "navigation": {"enabled": True},
            "grasp_execution": {"enabled": True},
            "embodied": {"enabled": False, "perception": {"enabled": False}},
        },
    )
    monkeypatch.setattr(module, "get_package_share_directory", lambda _package: str(tmp_path))

    actions = module.launch_setup(
        _FakeLaunchContext(
            {
                "robot_config": "lekiwi_nav_grasp",
                "control_mode": "teleop",
                "with_embodied": "false",
            }
        )
    )

    base_launch_arguments = dict(actions[0]._IncludeLaunchDescription__launch_arguments)
    assert base_launch_arguments["with_moveit"] == ""


def test_mapping_stage_defaults_embodied_runtime_off(monkeypatch, tmp_path):
    module = _load_launch_module()
    description = module.generate_launch_description()
    with_embodied = next(
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == "with_embodied"
    )
    generated = []
    monkeypatch.setattr(
        module,
        "_load_config",
        lambda *_args: {
            "default_control_mode": "teleop",
            "control_modes": {"teleop": {"controllers": []}},
            "embodied": {"enabled": False, "perception": {"enabled": False}},
        },
    )
    monkeypatch.setattr(module, "get_package_share_directory", lambda _package: str(tmp_path))
    monkeypatch.setattr(module, "generate_embodied_nodes", lambda *_args, **_kwargs: generated.append(True))

    actions = module.launch_setup(
        _FakeLaunchContext(
            {
                "robot_config": "lekiwi_lidar",
                "nav_stage": "mapping",
                "with_embodied": with_embodied.default_value[0].text,
            }
        )
    )

    assert len(actions) == 1
    assert generated == []


def test_pipeline_forces_udp_transport_for_fastdds_service_discovery():
    module = _load_launch_module()
    description = module.generate_launch_description()
    transport = next(entity for entity in description.entities if isinstance(entity, SetEnvironmentVariable))
    name = "".join(getattr(item, "text", str(item)) for item in transport.name)
    value = "".join(getattr(item, "text", str(item)) for item in transport.value)
    assert name == "FASTDDS_BUILTIN_TRANSPORTS"
    assert value == "UDPv4"


def test_launch_setup_aborts_when_game_enabled_but_perception_disabled():
    """with_perception:=false while a game is enabled must fail the launch, not
    start a node graph that routes the game to a dead topic. The validation gate
    runs before any base-launch include, so this raises without needing ROS
    share dirs."""
    module = _load_launch_module()

    fake_config = {
        "voice_tts": _voice_tts(),
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": _sorting_hat_game()},
        },
    }
    module._load_config = lambda *args, **kwargs: fake_config

    context = _FakeLaunchContext(
        {
            "robot_config": "so101_single_arm",
            "with_embodied": "true",
            "with_perception": "false",
        }
    )

    with pytest.raises(RuntimeError, match="visual_games"):
        module.launch_setup(context)


@pytest.mark.parametrize(("authorize_motion", "expected"), [(None, False), ("true", True)])
def test_launch_setup_passes_operator_motion_authorization(monkeypatch, authorize_motion, expected):
    module = _load_launch_module()
    config = {
        "default_control_mode": "model_inference",
        "embodied": {"enabled": True, "perception": {"enabled": False}},
    }
    module._load_config = lambda *_args, **_kwargs: config
    monkeypatch.setattr(module, "get_package_share_directory", lambda package: f"/tmp/{package}")
    generated = []

    def capture_nodes(
        robot_config,
        active_control_mode,
        *,
        motion_authorized=False,
        include_motion=True,
        include_visual_games=True,
        include_perception=True,
    ):
        generated.append(
            {
                "robot_config": robot_config,
                "active_control_mode": active_control_mode,
                "motion_authorized": motion_authorized,
                "include_motion": include_motion,
                "include_visual_games": include_visual_games,
                "include_perception": include_perception,
            }
        )
        return []

    monkeypatch.setattr(module, "generate_embodied_nodes", capture_nodes)
    launch_configurations = {
        "robot_config": "so101_single_arm",
        "control_mode": "moveit_runtime_override",
        "with_embodied": "true",
    }
    if authorize_motion is not None:
        launch_configurations["authorize_motion"] = authorize_motion

    module.launch_setup(_FakeLaunchContext(launch_configurations))

    assert len(generated) == 2
    assert all(item["active_control_mode"] == "moveit_runtime_override" for item in generated)
    assert all(item["motion_authorized"] is expected for item in generated)


def test_moveit_visual_closure_starts_perception_before_controller_readiness(monkeypatch):
    module = _load_launch_module()
    config = {
        "default_control_mode": "moveit_planning",
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "visual_games": {"sorting_hat": _sorting_hat_game()},
        },
    }
    module._load_config = lambda *_args, **_kwargs: config
    monkeypatch.setattr(module, "get_package_share_directory", lambda package: f"/tmp/{package}")
    generated = []

    def capture_nodes(
        _robot_config,
        _active_control_mode,
        *,
        motion_authorized=False,
        include_motion=True,
        include_visual_games=True,
        include_perception=True,
    ):
        generated.append(
            {
                "include_motion": include_motion,
                "include_visual_games": include_visual_games,
                "include_perception": include_perception,
            }
        )
        return []

    monkeypatch.setattr(module, "generate_embodied_nodes", capture_nodes)
    module.launch_setup(
        _FakeLaunchContext(
            {
                "robot_config": "so101_single_arm",
                "control_mode": "moveit_planning",
                "with_embodied": "true",
            }
        )
    )

    assert generated == [
        {"include_motion": False, "include_visual_games": True, "include_perception": True},
        {"include_motion": True, "include_visual_games": False, "include_perception": False},
    ]


def test_handeye_grasp_config_launches_pick_and_place_pipelines():
    config_path = (
        Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "lekiwi_handeye_realsense_grasp.yaml"
    )
    config = load_robot_config_dict(config_path)
    assert config["grasp_execution"]["planner_node"]["enable_source_gripper_tabletop_sweep"] is False
    assert config["grasp_execution"]["ik"]["worker_count"] == 4
    config["embodied"]["enabled"] = True
    nodes = generate_embodied_nodes(config, "moveit_planning")
    executables = {(node.__dict__.get("_Node__package"), node.__dict__.get("_Node__node_executable")) for node in nodes}
    assert ("manipulation_execution", "pick_executor_node") in executables
    assert ("manipulation_execution", "place_executor_node") in executables
    assert ("manipulation_service", "grasp_planner_node") in executables
    assert ("manipulation_service", "grasp_verifier_node") in executables
    assert ("perception_service", "grounded_sam2_node") not in executables

    model_nodes = generate_perception_model_nodes(config)
    assert {vars(node).get("_Node__node_name") for node in model_nodes} == {
        "grasp_grounding_detect",
        "grasp_segment_detections",
    }
    model_params = {
        vars(node).get("_Node__node_name"): _normalize_launch_param_mapping(node._Node__parameters[0])
        for node in model_nodes
    }
    assert _decode_launch_string(str(model_params["grasp_grounding_detect"]["service_type"])) == (
        "ibrobot_msgs/srv/GroundingDetect"
    )
    assert _decode_launch_string(str(model_params["grasp_grounding_detect"]["service_endpoint"])) == (
        "/perception/grasp/grounding_detect"
    )
    assert _decode_launch_string(str(model_params["grasp_segment_detections"]["service_type"])) == (
        "ibrobot_msgs/srv/SegmentDetections"
    )
    assert _decode_launch_string(str(model_params["grasp_segment_detections"]["service_endpoint"])) == (
        "/perception/grasp/segment_detections"
    )

    planner = next(node for node in nodes if vars(node).get("_Node__node_name") == "grasp_planner")
    planner_params = _normalize_launch_param_mapping(planner._Node__parameters[0])
    assert _decode_launch_string(str(planner_params["model_dir"])).endswith("/models/graspgen")
    assert str(planner_params["inference_backend"]).splitlines()[0] == "ascend_local"
    assert _decode_launch_string(str(planner_params["ascend_local_manifest_path"])).endswith("/models/graspgen")
    assert planner_params["startup_warmup"] is True
    assert _decode_launch_string(str(planner_params["legacy_detect_service"])) == ("/grasp_planner/detect_and_segment")
    assert "remote_310p_host" not in planner_params
    assert "host_runtime" not in planner_params
    assert _normalize_launch_environment(planner) == {
        "GOMP_SPINCOUNT": "0",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "4",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": "1",
    }

    pick_executor = next(node for node in nodes if vars(node).get("_Node__node_name") == "pick_executor_node")
    pick_executor_params = _normalize_launch_param_mapping(pick_executor._Node__parameters[0])
    home_joint_positions = _decode_launch_json_string(str(pick_executor_params["home_joint_positions_json"]))
    assert home_joint_positions["5"] == 0.0
    grasp_execution_json = _decode_launch_json_string(str(pick_executor_params["grasp_execution_json"]))
    assert grasp_execution_json["post_grasp_motion"] == {
        "pose_name": "place_container",
        "joint_names": ["1", "2", "3", "4", "5"],
        "joint_positions": {
            "1": -0.047553,
            "2": -0.073631,
            "3": -0.840621,
            "4": 1.497165,
            "5": -1.570790,
        },
        "duration_sec": 5.0,
        "velocity_scaling": 0.08,
    }
    place_executor = next(node for node in nodes if vars(node).get("_Node__node_name") == "placement_executor_node")
    place_params = _normalize_launch_param_mapping(place_executor._Node__parameters[0])
    placement_json = _decode_launch_json_string(str(place_params["placement_execution_json"]))
    assert placement_json["action_name"] == "/manipulation/execute_place"
    assert placement_json["executor"] == "placement_pipeline"
    assert placement_json["motion"]["place_pose"] == "place_container"
    assert placement_json["motion"]["place_joint_names"] == ["1", "2", "3", "4", "5"]
    assert placement_json["motion"]["place_joint_positions"] == {
        "1": -0.047553,
        "2": -0.073631,
        "3": -0.840621,
        "4": 1.497165,
        "5": -1.570790,
    }
    assert placement_json["motion"]["place_duration_sec"] == 5.0
    assert placement_json["motion"]["post_release"] == {
        "verify_joint_name": "3",
        "verify_joint_position": -0.533825,
        "verify_duration_sec": 2.0,
        "return_duration_sec": 2.0,
    }
    assert "observe_pose" not in placement_json["motion"]
    assert "plan_service" not in placement_json
    assert "arm_joint_names_json" not in place_params
    assert "named_poses_json" not in place_params
    assert _decode_launch_string(str(place_params["gripper_joint_name"])) == "6"

    skill_params = _skill_executor_params(nodes)
    assert _decode_launch_string(str(skill_params["place_action_name"])) == "/manipulation/execute_place"


def test_pc_handeye_grasp_launch_uses_cuda_catalog_and_place_pipeline():
    config_path = (
        Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "lekiwi_handeye_realsense_grasp_pc.yaml"
    )
    config = load_robot_config_dict(config_path)
    config["embodied"]["enabled"] = True

    nodes = generate_embodied_nodes(config, "moveit_planning")
    executables = {(vars(node).get("_Node__package"), vars(node).get("_Node__node_executable")) for node in nodes}
    assert ("manipulation_execution", "pick_executor_node") in executables
    assert ("manipulation_execution", "place_executor_node") in executables
    planner = next(node for node in nodes if vars(node).get("_Node__node_name") == "grasp_planner")
    planner_params = _normalize_launch_param_mapping(planner._Node__parameters[0])
    assert _decode_launch_string(str(planner_params["inference_backend"])) == "local_cuda"
    local_manifest_path = _decode_launch_string(str(planner_params["local_manifest_path"]))
    assert Path(local_manifest_path).is_absolute()
    assert local_manifest_path.endswith("/models/graspgen")
    assert "$(env " not in local_manifest_path
    planner_model_dir = _decode_launch_string(str(planner_params["model_dir"]))
    assert Path(planner_model_dir).is_absolute()
    assert planner_model_dir.endswith("/models/graspgen")
    assert "$(env " not in planner_model_dir
    skill_params = _skill_executor_params(nodes)
    assert _decode_launch_string(str(skill_params["skill_catalog_profile"])) == ("lekiwi_handeye_realsense_grasp_pc")
    assert _decode_launch_string(str(skill_params["place_action_name"])) == "/manipulation/execute_place"


def test_handeye_grasp_launch_auto_starts_parallel_ik_workers(monkeypatch, tmp_path):
    module = _load_launch_module()
    monkeypatch.setattr(module, "get_package_share_directory", lambda _package: str(tmp_path))
    config = {
        "moveit": {"joint_state_topic": "/arm_joint_state_broadcaster/joint_states"},
        "grasp_execution": {
            "enabled": True,
            "auto_start_dependencies": True,
            "ik": {
                "worker_count": 4,
                "worker_namespace_prefix": "/ik_worker",
                "auto_start_workers": True,
            },
        },
    }

    action = module._parallel_ik_worker_action(config, "false")

    assert action is not None
    assert action.__class__.__name__ == "IncludeLaunchDescription"
    arguments = dict(action._IncludeLaunchDescription__launch_arguments)
    assert arguments["joint_state_topic"] == "/arm_joint_state_broadcaster/joint_states"


def test_source_workspace_profile_uses_absolute_development_catalog_root():
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config_dict(config_path)
    config["embodied"]["enabled"] = True

    params = _skill_executor_params(generate_embodied_nodes(config, active_control_mode="moveit_planning"))

    assert _decode_launch_string(params["skill_catalog_source_mode"]) == "development"
    source_root = Path(_decode_launch_string(params["skill_catalog_source_root"]))
    assert source_root.is_absolute()
    assert source_root == Path(__file__).parents[3] / "src" / "skill_catalog"
    assert _decode_launch_string(params["skill_catalog_profile"]) == "so101_single_arm"


def test_embodied_runtime_waits_for_required_controllers():
    module = _load_launch_module()
    config = {
        "controller_startup_timeout": {"hardware": 30.0},
        "control_modes": {
            "moveit_planning": {
                "controllers": [
                    "joint_state_broadcaster",
                    "arm_trajectory_controller",
                    "gripper_trajectory_controller",
                ]
            }
        },
    }

    waiter = module._controller_ready_waiter(config, "moveit_planning", use_sim=False, auto_start=True)

    assert isinstance(waiter, Node)
    assert waiter.node_package == "robot_config"
    assert waiter.node_executable == "wait_for_controllers"
    assert "arm_trajectory_controller" in [_decode_launch_string(str(arg)) for arg in waiter._Node__arguments]


def test_embodied_runtime_readiness_handler_starts_only_after_success():
    module = _load_launch_module()
    runtime_actions = [object(), object()]
    handler = module._start_runtime_after_controller_readiness(runtime_actions)

    assert handler(type("Event", (), {"returncode": 0})(), None) == runtime_actions
    failure_actions = handler(type("Event", (), {"returncode": 1})(), None)
    assert len(failure_actions) == 1
    assert isinstance(failure_actions[0], EmitEvent)

import argparse
import builtins
import copy
import importlib
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from embodied_common.skill_request import canonical_skill_payload, skill_payload_hash
from embodied_common.visual_game_contracts import build_visual_game_capability_view
from robot_config.loader import load_robot_config_dict
from robot_config.timeout_policy import resolve_embodied_timeout_policy

CONFIG_PATH = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"


def _enabled_game_view() -> dict:
    return build_visual_game_capability_view(
        "test_robot",
        {
            "sorting_hat": {
                "enabled": True,
                "handler": "sorting_hat_v1",
                "summary": "Choose a Hogwarts house.",
            }
        },
        timeout_sec=130.0,
        result_retention_sec=300.0,
        result_capacity=128,
        start_service="/embodied/start_visual_game",
        result_service="/embodied/get_visual_game_result",
    )


def test_lidar_navigation_stage_compiles_navigation_profile(monkeypatch):
    from robot_config import loader
    from robot_skill_cli.catalog import compile_local_snapshot

    root = Path(__file__).resolve().parents[3]
    config_path = root / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    mount_path = root / "src/robot_config/config/hardware/lekiwi_mid360_mount.yaml"
    original_resolver = loader.resolve_ros_path

    def resolve_path(value):
        if value == "$(find robot_config)/config/hardware/lekiwi_mid360_mount.yaml":
            return str(mount_path)
        return original_resolver(value)

    monkeypatch.setattr(loader, "resolve_ros_path", resolve_path)
    navigation = loader.load_robot_config_dict(config_path, nav_stage="navigation")

    snapshot = compile_local_snapshot(navigation, config_path)

    assert snapshot.robot_context.context_schema_version == 2
    assert snapshot.robot_context.execution_endpoints["navigation_action"] == "/navigation/execute"
    assert set(snapshot.enabled_skill_names) == {
        "nav_abs_coordinate",
        "nav_straight",
        "nav_turn",
    }


def test_hybrid_lekiwi_profile_compiles_manipulation_and_navigation_domains():
    from robot_skill_cli.catalog import compile_local_snapshot

    root = Path(__file__).resolve().parents[3]
    config_path = root / "src/robot_config/config/robots/lekiwi_nav_grasp.yaml"
    config = load_robot_config_dict(config_path)

    snapshot = compile_local_snapshot(config, config_path)

    assert snapshot.robot_context.context_schema_version == 3
    assert snapshot.robot_context.supported_control_modes == ("moveit_planning", "base_navigation")
    control_modes = {
        skill_name: snapshot.capability_view[skill_name]["required_control_mode"]
        for skill_name in snapshot.enabled_skill_names
    }
    assert {name for name, mode in control_modes.items() if mode == "base_navigation"} == {
        "nav_abs_coordinate",
        "nav_straight",
        "nav_turn",
    }
    assert {name for name, mode in control_modes.items() if mode == "moveit_planning"} >= {
        "pick_object",
        "place_in_container",
        "open_gripper_skill",
    }


def test_non_navigation_profile_keeps_v1_robot_context():
    from robot_skill_cli.catalog import compile_local_snapshot

    config = load_robot_config_dict(CONFIG_PATH)
    snapshot = compile_local_snapshot(config, CONFIG_PATH)

    assert snapshot.robot_context.context_schema_version == 1
    assert "navigation_action" not in snapshot.robot_context.execution_endpoints


@pytest.mark.parametrize("stage", ["navigation", "hybrid"])
def test_sound_following_overlay_compiles_with_matching_runtime_executor(stage):
    from embodied_bringup.launch_builders.embodied import generate_embodied_nodes
    from robot_config.loader import load_voice_asr_config
    from robot_skill_cli.catalog import compile_local_snapshot
    from skill_library.skill_executor_node import SkillExecutorNode

    path = CONFIG_PATH.parent / "lekiwi_nav_grasp_sound_real.yaml"
    config = load_robot_config_dict(path, nav_stage=stage)
    voice = load_voice_asr_config(config["voice_asr"])
    assert voice.enabled
    assert voice.bundle_path.endswith("sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23")
    assert voice.deployment == "torch_cpu"
    assert voice.vad_input_channel == 1

    nodes = generate_embodied_nodes(config, active_control_mode="base_navigation")
    executor_node = next(node for node in nodes if vars(node).get("_Node__node_name") == "skill_executor_node")
    parameters = {key[0].text: value for key, value in executor_node._Node__parameters[0].items()}
    assert parameters["sound_following_enabled"] is True

    snapshot = compile_local_snapshot(config, path)
    assert set(snapshot.enabled_skill_names) == {"nav_turn", "nav_straight", "nav_abs_coordinate", "sound_following"}
    runtime = object.__new__(SkillExecutorNode)
    # Project the real launch parameters instead of hand-picking flags: the
    # runtime/catalog consistency check must catch a stage that inherits an
    # executor (e.g. imitate_human_motion from the base profile) that its
    # navigation catalog profile does not expose.
    runtime._imitate_human_motion_enabled = parameters["imitate_human_motion_enabled"]
    runtime._sound_following_enabled = parameters["sound_following_enabled"]
    runtime._sound_following_service = "/sound_orientation_node/set_following"
    runtime._skill_templates = {}
    runtime._pick_action_name = ""
    runtime._place_action_name = ""
    runtime._grasp_execution = {}
    runtime._placement_execution = {}
    runtime._semantic_map_target_service = ""
    runtime._validate_hri_runtime_catalog_consistency(snapshot)
    assert (
        snapshot.delegated_executors["sound_following"] == runtime._delegated_executor_descriptors()["sound_following"]
    )


def test_lekiwi_nav_grasp_hybrid_compiles_sound_following_with_full_skill_set(tmp_path):
    from embodied_bringup.launch_builders.embodied import generate_embodied_nodes
    from robot_skill_cli.catalog import compile_local_snapshot
    from skill_library.skill_executor_node import SkillExecutorNode

    # Hermetic copy: the checkout shares the model bundles but has no approved
    # camera calibration (clearing the artifact keeps the inline transform),
    # and the catalog source root must stay absolute for a tmp path.
    copied = yaml.safe_load((CONFIG_PATH.parent / "lekiwi_nav_grasp.yaml").read_text(encoding="utf-8"))
    robot = copied["robot"]
    robot["sensor_calibration"]["artifacts"]["base_to_front_camera"] = ""
    robot["embodied"]["skill_catalog_source_root"] = str(Path(__file__).resolve().parents[2] / "skill_catalog")
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(copied), encoding="utf-8")

    config = load_robot_config_dict(config_path, nav_stage="hybrid")

    nodes = generate_embodied_nodes(config, active_control_mode="base_navigation")
    executor_node = next(node for node in nodes if vars(node).get("_Node__node_name") == "skill_executor_node")
    parameters = {key[0].text: value for key, value in executor_node._Node__parameters[0].items()}
    assert parameters["sound_following_enabled"] is True
    assert parameters["imitate_human_motion_enabled"] is True

    snapshot = compile_local_snapshot(config, config_path)
    skill_names = set(snapshot.enabled_skill_names)
    # The hybrid profile keeps the full mobile-manipulator set and adds the
    # session toggle; the runtime flags must match the catalog exposure.
    for skill in (
        "sound_following",
        "imitate_human_motion",
        "pick_object",
        "place_in_container",
        "open_gripper_skill",
        "nav_turn",
        "nav_straight",
        "nav_abs_coordinate",
    ):
        assert skill in skill_names

    runtime = object.__new__(SkillExecutorNode)
    runtime._imitate_human_motion_enabled = parameters["imitate_human_motion_enabled"]
    runtime._sound_following_enabled = parameters["sound_following_enabled"]
    runtime._sound_following_service = "/sound_orientation_node/set_following"
    runtime._skill_templates = {}
    runtime._pick_action_name = ""
    runtime._place_action_name = ""
    runtime._grasp_execution = {}
    runtime._placement_execution = {}
    runtime._semantic_map_target_service = ""
    runtime._validate_hri_runtime_catalog_consistency(snapshot)
    assert (
        snapshot.delegated_executors["sound_following"] == runtime._delegated_executor_descriptors()["sound_following"]
    )


def test_hybrid_switch_test_overlay_disables_inherited_sound_following():
    from robot_skill_cli.catalog import compile_local_snapshot

    path = CONFIG_PATH.parent / "lekiwi_hybrid_switch_test.yaml"
    config = load_robot_config_dict(path, nav_stage="hybrid")

    assert config["embodied"]["idle_behaviors"]["sound_orientation"]["enabled"] is False
    assert "sound_following" not in compile_local_snapshot(config, path).enabled_skill_names


def test_catalog_import_does_not_load_rclpy(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rclpy" or name.startswith("rclpy."):
            raise AssertionError("catalog-only commands must not import rclpy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("robot_skill_cli.catalog")
    importlib.reload(module)


def test_status_preflight_timeout_has_hardware_discovery_floor():
    from robot_skill_cli.cli import _plan_rpc_timeout, _status_preflight_timeout

    context = type("Context", (), {"view": {"timeout_policy": {"rpc_timeout_sec": 5.0}}})()
    assert _plan_rpc_timeout(context) == 30.0
    assert _status_preflight_timeout(context) == 30.0


def test_agent_control_timeout_preserves_larger_configured_value():
    from robot_skill_cli.cli import _plan_rpc_timeout

    context = type("Context", (), {"view": {"timeout_policy": {"rpc_timeout_sec": 40.0}}})()
    assert _plan_rpc_timeout(context) == 40.0


@pytest.mark.parametrize(
    "workflow_step",
    [
        {
            "schema_version": 2,
            "skill_name": "nav_abs_coordinate",
            "has_x": False,
            "x": 0.0,
        },
        {
            "schema_version": 2,
            "skill_name": "open_gripper_skill",
        },
    ],
)
def test_plan_workflow_forwards_explicit_step_version_without_domain_inference(workflow_step):
    from robot_skill_cli.cli import _run_plan_workflow

    context = SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 2.0}})
    bridge = Mock()
    bridge.plan_agent_command.return_value = {"success": True}
    args = SimpleNamespace(
        request_id="versioned-workflow",
        raw_command="submit an explicit workflow",
        workflow_json=json.dumps([workflow_step]),
    )

    result = _run_plan_workflow(args, context, bridge)

    assert result == {"success": True}
    bridge.plan_agent_command.assert_called_once_with(
        request_id="versioned-workflow",
        raw_command="submit an explicit workflow",
        workflow_steps=[workflow_step],
        timeout_sec=30.0,
    )


def test_plan_workflow_rejects_unversioned_typed_step_instead_of_domain_rewriting():
    from robot_skill_cli.cli import _CliArgumentError, _run_plan_workflow

    context = SimpleNamespace(
        view={
            "timeout_policy": {"rpc_timeout_sec": 2.0},
            "skills": [
                {
                    "name": "nav_abs_coordinate",
                    "schema_version": 2,
                    "domain": "navigation",
                }
            ],
        }
    )
    bridge = Mock()
    args = SimpleNamespace(
        request_id="unversioned-workflow",
        raw_command="submit an unversioned workflow",
        workflow_json=json.dumps([{"skill_name": "nav_abs_coordinate", "x": 0.0}]),
    )

    with pytest.raises(_CliArgumentError, match="schema_version"):
        _run_plan_workflow(args, context, bridge)

    bridge.plan_agent_command.assert_not_called()


def test_load_catalog_uses_exported_config_resolver(monkeypatch):
    from robot_skill_cli import catalog

    calls = []
    monkeypatch.setattr(
        catalog,
        "resolve_robot_config_path",
        lambda *, config_name, config_path: calls.append((config_name, config_path)) or CONFIG_PATH,
    )

    view = catalog.load_capability_catalog(config_name="selected", config_path=None)

    assert calls == [("selected", None)]
    assert view["robot_name"] == "so101_single_arm"


def test_pc_grasp_catalog_registers_enabled_delegated_executors(monkeypatch):
    from robot_skill_cli.catalog import load_capability_catalog

    pc_config = CONFIG_PATH.parents[2] / "config" / "robots" / "lekiwi_handeye_realsense_grasp_pc.yaml"
    monkeypatch.setenv("WORKSPACE", str(pc_config.parents[4]))

    view = load_capability_catalog(config_path=pc_config)

    assert view["robot_name"] == "lekiwi_handeye_realsense_grasp_pc"
    assert load_robot_config_dict(pc_config)["embodied"]["skill_catalog_profile"] == (
        "lekiwi_handeye_realsense_grasp_pc"
    )
    assert {skill["name"] for skill in view["skills"]} >= {"pick_object", "place_in_container"}


def test_list_skills_uses_every_profile_enabled_entry_and_only_public_fields():
    from robot_skill_cli.catalog import list_skills, load_capability_catalog

    data = list_skills(load_capability_catalog(config_path=CONFIG_PATH))

    assert len(data["skills"]) == 16
    assert [skill["name"] for skill in data["skills"]] == sorted(skill["name"] for skill in data["skills"])
    assert all(
        set(skill)
        == {
            "name",
            "contract_schema_version",
            "summary",
            "domain",
            "moves_robot",
            "required_control_mode",
        }
        for skill in data["skills"]
    )
    encoded = json.dumps(data)
    for forbidden in (
        "primitive_sequence",
        "primitive_name",
        "rule_entry",
        "target_pose_key",
        "place_name_from_request",
        "motion_direction_from_request",
        "motion_distance_from_request",
        "joint_positions",
        "description",
    ):
        assert forbidden not in encoded


def test_list_games_exposes_only_enabled_control_plane_metadata():
    from robot_skill_cli.catalog import list_games

    game_view = _enabled_game_view()
    data = list_games(game_view)

    assert data == {
        "robot_name": "test_robot",
        "config_digest": game_view["config_digest"],
        "games": [
            {
                "name": "sorting_hat",
                "summary": "Choose a Hogwarts house.",
                "result_field": "scene_summary",
            }
        ],
    }


def test_visual_game_context_does_not_compile_motion_catalog(monkeypatch):
    from robot_skill_cli import catalog

    def fail_if_compiled(*_args, **_kwargs):
        raise AssertionError("visual-game commands must not compile the motion Skill catalog")

    monkeypatch.setattr(catalog, "compile_local_snapshot", fail_if_compiled)

    context = catalog.load_visual_game_context(config_path=CONFIG_PATH)
    runtime_context, transport = catalog.load_visual_game_runtime_context(config_path=CONFIG_PATH)

    assert context.game_view["robot_name"] == "so101_single_arm"
    assert runtime_context.game_view == context.game_view
    assert transport.start_visual_game_service == "/embodied/start_visual_game"


def test_visual_game_context_hides_games_when_embodied_is_disabled(monkeypatch):
    from robot_skill_cli import catalog
    from robot_skill_cli.catalog import list_games

    config = load_robot_config_dict(CONFIG_PATH)
    config["embodied"]["enabled"] = False
    config["embodied"]["visual_games"]["sorting_hat"]["enabled"] = True
    monkeypatch.setattr(catalog, "load_robot_config_dict", lambda _path: copy.deepcopy(config))

    context = catalog.load_visual_game_context(config_path=CONFIG_PATH)

    assert list_games(context.game_view)["games"] == []


def test_list_games_rejects_enabled_game_without_perception(tmp_path, capsys):
    from robot_skill_cli.cli import main

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["robot"]["embodied"]["enabled"] = True
    config["robot"]["embodied"]["perception"]["enabled"] = False
    config["robot"]["embodied"]["visual_games"]["sorting_hat"]["enabled"] = True
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert main(["--config-path", str(config_path), "list-games"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["error"]["code"] == "INVALID_ARGUMENT"
    assert "embodied.perception.enabled" in result["error"]["message"]


def test_start_game_rejects_disabled_embodied_before_starting_ros(monkeypatch, capsys):
    from robot_skill_cli import cli

    def fail_if_bridge_created(_transport):
        raise AssertionError("disabled visual game must fail before ROS bridge creation")

    monkeypatch.setattr(cli, "_create_bridge", fail_if_bridge_created)

    assert (
        cli.main(
            [
                "--config-path",
                str(CONFIG_PATH),
                "start-game",
                "sorting_hat",
                "--request-id",
                "disabled-game-test",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert result["error"]["code"] == "UNKNOWN_GAME"


def test_visual_game_timeouts_do_not_change_motion_catalog_digest():
    from robot_skill_cli.catalog import compile_local_snapshot

    original = load_robot_config_dict(CONFIG_PATH)
    changed = copy.deepcopy(original)
    changed["embodied"].setdefault("timeouts", {})["visual_game_timeout_sec"] = 142.0
    changed["embodied"]["timeouts"]["visual_game_result_retention_sec"] = 84.0

    assert (
        compile_local_snapshot(original, CONFIG_PATH).capability_digest
        == compile_local_snapshot(changed, CONFIG_PATH).capability_digest
    )


def test_game_catalog_commands_use_game_only_context(monkeypatch, capsys):
    from robot_skill_cli import catalog
    from robot_skill_cli.cli import main

    enabled_config = load_robot_config_dict(CONFIG_PATH)
    enabled_config["embodied"]["enabled"] = True
    enabled_config["embodied"]["perception"]["enabled"] = True
    enabled_config["embodied"]["visual_games"]["sorting_hat"]["enabled"] = True
    monkeypatch.setattr(catalog, "load_robot_config_dict", lambda _path: copy.deepcopy(enabled_config))
    monkeypatch.setattr(
        catalog,
        "compile_local_snapshot",
        lambda *_args, **_kwargs: pytest.fail("game catalog command compiled motion Skills"),
    )

    for arguments in (["list-games"], ["describe-game", "sorting_hat"]):
        assert main(["--config-path", str(CONFIG_PATH), *arguments]) == 0
        assert json.loads(capsys.readouterr().out)["ok"] is True


@pytest.mark.parametrize("command", ["start-game", "game-result"])
def test_game_runtime_commands_use_game_only_context(monkeypatch, capsys, command):
    from robot_skill_cli import catalog, cli

    enabled_config = load_robot_config_dict(CONFIG_PATH)
    enabled_config["embodied"]["enabled"] = True
    enabled_config["embodied"]["perception"]["enabled"] = True
    enabled_config["embodied"]["visual_games"]["sorting_hat"]["enabled"] = True
    monkeypatch.setattr(catalog, "load_robot_config_dict", lambda _path: copy.deepcopy(enabled_config))
    monkeypatch.setattr(
        catalog,
        "compile_local_snapshot",
        lambda *_args, **_kwargs: pytest.fail("game runtime command compiled motion Skills"),
    )
    bridge = Mock()
    bridge.start.return_value = True
    bridge.start_visual_game.return_value = {
        "accepted": True,
        "duplicate": False,
        "request_id": "game-test-1",
        "config_digest": catalog.load_visual_game_context(config_path=CONFIG_PATH).game_view["config_digest"],
        "error_code": "",
        "message": "accepted",
    }
    bridge.get_visual_game_result.return_value = {
        "found": True,
        "terminal": False,
        "success": False,
        "game_name": "sorting_hat",
        "scene_summary": "",
        "result_json": "",
        "config_digest": bridge.start_visual_game.return_value["config_digest"],
        "error_code": "",
        "message": "running",
    }
    monkeypatch.setattr(cli, "_create_bridge", lambda _transport: bridge)
    arguments = (
        ["start-game", "sorting_hat", "--request-id", "game-test-1"]
        if command == "start-game"
        else ["game-result", "--request-id", "game-test-1"]
    )

    assert cli.main(["--config-path", str(CONFIG_PATH), *arguments]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    bridge.close.assert_called_once_with()


def test_start_game_returns_request_id_without_waiting_for_result():
    from robot_skill_cli.cli import _run_start_game

    context = SimpleNamespace(
        game_view=_enabled_game_view(),
        view={"timeout_policy": {"rpc_timeout_sec": 2.0}},
    )
    bridge = Mock()
    bridge.start_visual_game.return_value = {
        "accepted": True,
        "duplicate": False,
        "request_id": "game-test-1",
        "config_digest": context.game_view["config_digest"],
        "error_code": "",
        "message": "visual game request accepted",
    }

    result = _run_start_game(SimpleNamespace(game="sorting_hat", request_id="game-test-1"), context, bridge)

    assert result["request_id"] == "game-test-1"
    bridge.start_visual_game.assert_called_once_with(
        "sorting_hat",
        request_id="game-test-1",
        expected_config_digest=context.game_view["config_digest"],
        timeout_sec=2.0,
    )
    assert "scene_summary" not in result


def test_game_result_exposes_structured_house_result():
    from robot_skill_cli.cli import _run_game_result

    context = SimpleNamespace(
        game_view=_enabled_game_view(),
        view={"timeout_policy": {"rpc_timeout_sec": 2.0}},
    )
    bridge = Mock()
    bridge.get_visual_game_result.return_value = {
        "found": True,
        "terminal": True,
        "success": True,
        "game_name": "sorting_hat",
        "scene_summary": "格兰芬多",
        "config_digest": context.game_view["config_digest"],
        "error_code": "",
        "message": "completed",
    }

    result = _run_game_result(SimpleNamespace(request_id="game-test-1"), context, bridge)

    assert result["request_id"] == "game-test-1"
    assert result["scene_summary"] == "格兰芬多"
    bridge.get_visual_game_result.assert_called_once_with("game-test-1", timeout_sec=2.0)


def test_game_result_checks_digest_before_not_found():
    from robot_skill_cli.cli import _CommandError, _run_game_result

    context = SimpleNamespace(
        game_view=_enabled_game_view(),
        view={"timeout_policy": {"rpc_timeout_sec": 2.0}},
    )
    bridge = Mock()
    bridge.get_visual_game_result.return_value = {
        "found": False,
        "config_digest": "stale-digest",
        "error_code": "GAME_REQUEST_NOT_FOUND",
        "message": "not found",
    }

    with pytest.raises(_CommandError, match="configuration does not match") as exc_info:
        _run_game_result(SimpleNamespace(request_id="game-test-1"), context, bridge)

    assert exc_info.value.code == "CONFIG_MISMATCH"


def test_describe_game_exposes_result_contract_timeout_and_digest():
    from robot_skill_cli.catalog import describe_game

    game_view = _enabled_game_view()
    described = describe_game(game_view, "sorting_hat")

    assert described["required_inputs"] == ["primary_image"]
    assert described["result_schema"]["field"] == "scene_summary"
    assert described["timeout_sec"] == 130.0
    assert described["result_capacity"] == 128
    assert described["config_digest"] == game_view["config_digest"]


def test_catalog_list_and_describe_project_explicit_capability_fields():
    from robot_skill_cli.catalog import describe_skill, list_skills, load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    listed = list_skills(view)
    described = describe_skill(view, "move_relative_ee")

    assert set(listed["skills"][0]) == {
        "name",
        "contract_schema_version",
        "summary",
        "domain",
        "moves_robot",
        "required_control_mode",
    }
    assert described["summary"] == "Move the end effector in a requested direction by a requested distance."
    assert described["domain"] == "manipulation"
    assert described["moves_robot"] is True
    assert described["required_control_mode"] == "moveit_planning"
    assert described["parameters"]["properties"]["motion_distance"]["unit"] == "meters"
    assert described["recovery_policy"] == "ask_user"
    assert "description" not in described


def test_describe_exposes_derived_schema_timeout_policy_and_digest():
    from robot_skill_cli.catalog import describe_skill, load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    relative = describe_skill(view, "move_relative_ee")
    rotation = describe_skill(view, "rotate_gripper_cw")

    assert relative["parameters"]["required"] == ["motion_direction", "motion_distance"]
    assert relative["parameters"]["properties"]["motion_direction"]["enum"] == [
        "forward",
        "backward",
        "left",
        "right",
        "up",
        "down",
    ]
    assert relative["parameters"]["properties"]["motion_distance"] == {
        "exclusiveMinimum": 0,
        "type": "number",
        "unit": "meters",
    }
    assert rotation["parameters"]["properties"]["motion_distance"]["unit"] == "degrees"
    resolved_timeouts = resolve_embodied_timeout_policy(load_robot_config_dict(CONFIG_PATH)["embodied"])
    assert relative["timeout_policy"] == {
        name: value for name, value in resolved_timeouts.items() if not name.startswith("visual_game_")
    }
    assert relative["timeout_sec"] == 120.0
    assert relative["config_digest"] == view["capability_digest"]


def test_catalog_uses_capability_metadata_without_legacy_descriptions():
    from embodied_common.capability_view import build_capability_view
    from robot_skill_cli.catalog import describe_skill, list_skills

    view = build_capability_view(
        {
            "name": "fallback_robot",
            "embodied": {
                "skill_templates": {
                    "fallback": {
                        "capability": {
                            "schema_version": 1,
                            "summary": "Open the fallback gripper.",
                            "domain": "manipulation",
                            "moves_robot": True,
                            "required_control_mode": "moveit_planning",
                            "parameters": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {},
                                "required": [],
                            },
                            "recovery_policy": "never_retry",
                        },
                        "description": {"summary": "Private fallback description."},
                        "primitive_sequence": [{"primitive_name": "open_gripper"}],
                    }
                }
            },
        },
        timeout_policy={"default_skill_timeout_sec": 1.0, "task_budget_sec": 2.0},
    )

    listed = list_skills(view)["skills"][0]
    described = describe_skill(view, "fallback")

    assert listed == {
        "name": "fallback",
        "contract_schema_version": 1,
        "summary": "Open the fallback gripper.",
        "domain": "manipulation",
        "moves_robot": True,
        "required_control_mode": "moveit_planning",
    }
    assert "description" not in described


def test_catalog_requires_an_explicit_profile(tmp_path):
    from robot_skill_cli.catalog import load_capability_catalog

    config_path = tmp_path / "robot.yaml"
    config_path.write_text("robot:\n  name: catalog_robot\n", encoding="utf-8")

    with pytest.raises(KeyError, match="embodied"):
        load_capability_catalog(config_path=config_path)


def test_unknown_skill_has_stable_error_and_exit_code(capsys):
    from robot_skill_cli.cli import main

    exit_code = main(["--config-path", str(CONFIG_PATH), "describe", "missing"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert captured.err == ""
    assert payload["error"]["code"] == "UNKNOWN_SKILL"


def test_catalog_commands_emit_one_json_document(capsys):
    from robot_skill_cli.cli import main

    for arguments in (["list-skills"], ["list-games"], ["describe", "move_relative_ee"], ["list-poses"]):
        exit_code = main(["--config-path", str(CONFIG_PATH), *arguments])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert captured.err == ""
        assert len(captured.out.strip().splitlines()) == 1
        payload = json.loads(captured.out)
        assert payload["schema_version"] == 1
        assert payload["ok"] is True


def test_agent_plan_parser_uses_typed_workflow_option_and_lifecycle_commands():
    from robot_skill_cli.cli import _build_parser

    parser = _build_parser()
    plan = parser.parse_args(
        [
            "plan-workflow",
            "--request-id",
            "request-1",
            "--text",
            "打开夹爪",
            "--workflow-json",
            '[{"skill_name":"open_gripper_skill"}]',
        ]
    )
    confirm = parser.parse_args(
        [
            "confirm-plan",
            "--plan-token",
            "plan-token",
            "--plan-digest",
            "digest",
            "--task-id",
            "task-1",
        ]
    )

    assert plan.raw_command == "打开夹爪"
    assert plan.workflow_json == '[{"skill_name":"open_gripper_skill"}]'
    assert plan.request_id == "request-1"
    assert confirm.command == "confirm-plan"


def test_run_workflow_parser_accepts_one_typed_workflow():
    from robot_skill_cli.cli import _build_parser

    parser = _build_parser()
    run = parser.parse_args(
        [
            "run-workflow",
            "--text",
            "打开夹爪",
            "--workflow-json",
            '[{"schema_version":1,"skill_name":"open_gripper_skill"}]',
        ]
    )

    assert run.command == "run-workflow"
    assert run.raw_command == "打开夹爪"


def test_run_workflow_parser_accepts_request_id():
    from robot_skill_cli.cli import _build_parser

    run = _build_parser().parse_args(
        [
            "run-workflow",
            "--request-id",
            "request-1",
            "--text",
            "打开夹爪",
            "--workflow-json",
            '[{"schema_version":1,"skill_name":"open_gripper_skill"}]',
        ]
    )

    assert run.request_id == "request-1"


def test_run_workflow_delegates_lifecycle_to_controller(monkeypatch, capsys):
    from types import SimpleNamespace

    from robot_skill_cli import interactive_control
    from robot_skill_cli.cli import _run_workflow

    calls = []

    class _Controller:
        def __init__(self, bridge, *, timeout_policy, execution_mode):
            calls.append((bridge, timeout_policy, execution_mode))

        def run(
            self,
            raw_command,
            workflow_steps,
            *,
            presentation_callback,
            authorization_callback,
            stop_event,
            **kwargs,
        ):
            calls.append((raw_command, workflow_steps, stop_event))
            presentation_callback({"task_id": "task-1"})
            authorization_callback({"task_id": "task-1"})
            return {"state": "succeeded", "task_id": "task-1", "result": {"success": True}}

    monkeypatch.setattr(interactive_control, "InteractiveController", _Controller)
    args = SimpleNamespace(
        raw_command="打开夹爪",
        workflow_json='[{"schema_version":1,"skill_name":"open_gripper_skill"}]',
    )
    context = SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 5.0}})

    result = _run_workflow(args, context, object())

    assert result.exit_code == 0
    assert calls[0][2] == "immediate_after_presentation"
    assert calls[1][0:2] == (
        "打开夹爪",
        [{"schema_version": 1, "skill_name": "open_gripper_skill"}],
    )
    assert isinstance(calls[1][2], threading.Event)
    assert [json.loads(line)["event"] for line in capsys.readouterr().out.strip().splitlines()] == [
        "workflow_presentation",
        "workflow_terminal",
    ]


def test_run_workflow_signal_requests_controller_stop_and_restores_handlers(monkeypatch, capsys):
    from types import SimpleNamespace

    from robot_skill_cli import cli, interactive_control
    from robot_skill_cli.cli import _run_workflow

    calls = []
    installed = {}

    class _Controller:
        def __init__(self, _bridge, *, timeout_policy, execution_mode):
            calls.append((timeout_policy, execution_mode))

        def request_stop(self):
            calls.append("request_stop")

        def run(
            self,
            _raw_command,
            _workflow_steps,
            *,
            presentation_callback,
            authorization_callback,
            stop_event,
            **kwargs,
        ):
            calls.append(stop_event)
            installed[cli.signal.SIGINT](cli.signal.SIGINT, None)
            assert stop_event.is_set()
            presentation_callback({"task_id": "task-stop"})
            return {
                "state": "stopped",
                "task_id": "task-stop",
                "error_code": "SKILL_CANCELLED",
                "result": {"success": False, "error_code": "SKILL_CANCELLED"},
            }

    def install(signum, handler):
        previous = installed.get(signum, cli.signal.SIG_DFL)
        installed[signum] = handler
        return previous

    monkeypatch.setattr(interactive_control, "InteractiveController", _Controller)
    monkeypatch.setattr(cli.signal, "signal", install)
    args = SimpleNamespace(
        raw_command="打开夹爪",
        workflow_json='[{"schema_version":1,"skill_name":"open_gripper_skill"}]',
    )
    context = SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 5.0}})

    result = _run_workflow(args, context, object())

    assert result.exit_code == 13
    assert "request_stop" in calls
    assert calls[0][1] == "immediate_after_presentation"
    assert installed[cli.signal.SIGINT] is cli.signal.SIG_DFL
    assert installed[cli.signal.SIGTERM] is cli.signal.SIG_DFL
    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert [event["event"] for event in events] == ["workflow_presentation", "workflow_terminal"]


def test_run_workflow_handles_local_terminal_without_nested_result(monkeypatch, capsys):
    from types import SimpleNamespace

    from robot_skill_cli import interactive_control
    from robot_skill_cli.cli import _run_workflow

    class _Controller:
        def __init__(self, _bridge, *, timeout_policy, execution_mode):
            pass

        def run(self, *_args, **_kwargs):
            return {
                "state": "unknown",
                "task_id": "task-unknown",
                "error_code": "SKILL_CANCEL_TIMEOUT",
                "message": "robot stop state is unknown",
                "result": {},
            }

    monkeypatch.setattr(interactive_control, "InteractiveController", _Controller)
    args = SimpleNamespace(raw_command="stop", workflow_json='[{"schema_version":1,"skill_name":"wave_hello"}]')
    context = SimpleNamespace(view={"timeout_policy": {"rpc_timeout_sec": 5.0}})

    result = _run_workflow(args, context, object())

    assert result.exit_code == 15
    assert json.loads(capsys.readouterr().out.strip())["event"] == "workflow_terminal"


@pytest.mark.parametrize(
    ("skill_name", "arguments", "properties", "expected"),
    [
        (
            "nav_straight",
            ["--direction", "left", "--distance", "1.25"],
            {
                "direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]},
                "distance": {"type": "number", "exclusiveMinimum": 0},
            },
            {"direction": "left", "distance": 1.25},
        ),
        (
            "nav_turn",
            ["--direction", "right", "--degree", "450"],
            {
                "direction": {"type": "string", "enum": ["left", "right"]},
                "degree": {"type": "number", "exclusiveMinimum": 0},
            },
            {"direction": "right", "degree": 450.0},
        ),
        (
            "nav_abs_coordinate",
            ["--x", "0", "--y", "-2.5", "--yaw", "0"],
            {"x": {"type": "number"}, "y": {"type": "number"}, "yaw": {"type": "number"}},
            {"x": 0.0, "y": -2.5, "yaw": 0.0},
        ),
        (
            "resolve_object_pose",
            ["--target-name", "Paper Ball", "--stand-off-distance-m", "0.3"],
            {
                "target_name": {"type": "string", "enum": ["banana", "paper ball"]},
                "stand_off_distance_m": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
            {"target_name": "paper ball", "stand_off_distance_m": 0.3},
        ),
    ],
)
def test_navigation_cli_parser_validates_all_public_parameters(skill_name, arguments, properties, expected):
    from robot_skill_cli.cli import _build_parser, _validate_schema

    args = _build_parser().parse_args(["validate", skill_name, *arguments])
    skill = {
        "name": skill_name,
        "parameters": {
            "properties": properties,
            "required": list(properties),
        },
    }

    _validate_schema(skill, args)

    for field_name, value in expected.items():
        actual = getattr(args, field_name)
        assert actual == pytest.approx(value) if isinstance(value, float) else actual == value
    assert args.motion_direction is None
    assert args.motion_distance is None


@pytest.mark.parametrize(
    ("error_code", "exit_code"),
    [
        ("SKILL_PACKAGE_NOT_FOUND", 10),
        ("SKILL_SCHEMA_INVALID", 11),
        ("SKILL_WORKFLOW_LEASE_MISMATCH", 13),
        ("SERVER_UNAVAILABLE", 14),
        ("SKILL_TASK_DEADLINE_EXPIRED", 15),
        ("SKILL_CANCEL_TIMEOUT", 15),
    ],
)
def test_agent_plan_error_codes_use_v1_exit_groups(error_code: str, exit_code: int) -> None:
    from robot_skill_cli.cli import _agent_error_exit_code

    assert _agent_error_exit_code(error_code) == exit_code


def test_runtime_capability_view_rejects_tampered_snapshot() -> None:
    from robot_skill_cli.catalog import capability_view_from_snapshot

    bridge = _FakeBridge(_ready_status("legacy-digest"))
    status = bridge.get_status()
    snapshot = bridge.get_skill_snapshot()
    snapshot["snapshot_json"] += " "

    with pytest.raises(ValueError, match="SKILL_SNAPSHOT_DIGEST_MISMATCH"):
        capability_view_from_snapshot(snapshot, status)


def test_skill_payload_normalization_and_default_timeout_identity():
    omitted = canonical_skill_payload(
        " move_relative_ee ",
        schema_version=1,
        motion_direction=" FoRwArD ",
        motion_distance=-0.0,
        default_timeout_sec=30.0,
    )
    explicit = canonical_skill_payload(
        "move_relative_ee",
        schema_version=1,
        motion_direction="forward",
        motion_distance=0.0,
        timeout_sec=30,
        default_timeout_sec=30.0,
    )

    assert omitted == explicit
    assert omitted["motion_distance"] == 0.0
    assert math.copysign(1.0, omitted["motion_distance"]) == 1.0
    assert skill_payload_hash(omitted) == skill_payload_hash(explicit)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skill_name": ""},
        {"skill_name": "demo", "motion_distance": float("nan")},
        {"skill_name": "demo", "timeout_sec": float("inf")},
    ],
)
def test_skill_payload_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        canonical_skill_payload(schema_version=1, default_timeout_sec=30.0, **kwargs)


def test_schema_allows_omitted_optional_declared_parameter():
    from robot_skill_cli.cli import _validate_schema

    skill = {
        "name": "optional_relative_motion",
        "parameters": {
            "properties": {
                "motion_direction": {"type": "string", "enum": ["forward"]},
                "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
            "required": ["motion_direction"],
        },
    }
    args = argparse.Namespace(
        target_name=None,
        container_name=None,
        place_name=None,
        motion_direction="forward",
        motion_distance=None,
    )

    _validate_schema(skill, args)


def test_schema_rejects_omitted_declared_required_parameter():
    from robot_skill_cli.cli import _CliArgumentError, _validate_schema

    skill = {
        "name": "optional_relative_motion",
        "parameters": {
            "properties": {
                "motion_direction": {"type": "string", "enum": ["forward"]},
                "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
            "required": ["motion_direction"],
        },
    }
    args = argparse.Namespace(
        target_name=None,
        container_name=None,
        place_name=None,
        motion_direction=None,
        motion_distance=None,
    )

    with pytest.raises(_CliArgumentError, match="motion_direction is required for skill optional_relative_motion"):
        _validate_schema(skill, args)


def test_schema_validates_supplied_optional_parameter():
    from robot_skill_cli.cli import _CliArgumentError, _validate_schema

    skill = {
        "name": "optional_relative_motion",
        "parameters": {
            "properties": {
                "motion_distance": {"type": "number", "exclusiveMinimum": 0, "unit": "meters"},
            },
            "required": [],
        },
    }
    args = argparse.Namespace(
        target_name=None,
        container_name=None,
        place_name=None,
        motion_direction=None,
        motion_distance=0.0,
    )

    with pytest.raises(_CliArgumentError, match="motion_distance must be a finite number greater than zero"):
        _validate_schema(skill, args)


class _FakeBridge:
    def __init__(self, status):
        self.status = status
        self.calls = []
        self.validate_payloads = []
        self.status_timeouts = []
        self.reload_requests = []
        self._snapshot = None

    def start(self):
        self.calls.append("start")
        return True

    def get_status(self, **kwargs):
        self.calls.append("status")
        self.status_timeouts.append(kwargs.get("timeout_sec"))
        if self._snapshot is None:
            self._snapshot = self._build_snapshot()
            self.status.update(
                registry_epoch="epoch-1",
                registry_generation=1,
                registry_digest=self._snapshot.registry_digest,
                capability_digest=self._snapshot.capability_digest,
            )
        return self.status

    def validate_skill(self, payload, **kwargs):
        self.calls.append("validate")
        self.validate_payloads.append(payload)
        return {"allowed": True, "reason": "allowed"}

    def reload_skill_catalog(self, **kwargs):
        self.reload_requests.append(kwargs)
        return {
            "success": True,
            "registry_epoch": "epoch-2",
            "old_generation": 1,
            "generation": 2,
            "registry_digest": "registry-2",
            "capability_digest": "capability-2",
            "source_release_digest": "source-2",
            "provenance_digest": "provenance-2",
            "error_code": "",
            "message": "reloaded",
            "changed_skills": ["nod_yes"],
            "diagnostics": [],
        }

    def get_skill_snapshot(self, **_kwargs):
        snapshot = self._snapshot or self._build_snapshot()
        return {
            "success": True,
            "registry_epoch": "epoch-1",
            "generation": 1,
            "registry_digest": snapshot.registry_digest,
            "capability_digest": snapshot.capability_digest,
            "provenance_digest": snapshot.provenance_digest,
            "snapshot_json": snapshot.snapshot_json,
        }

    @staticmethod
    def _build_snapshot():
        from robot_config.loader import load_robot_config_dict
        from robot_skill_cli.catalog import compile_local_snapshot

        config = load_robot_config_dict(CONFIG_PATH)
        return compile_local_snapshot(config, CONFIG_PATH)

    def close(self):
        self.calls.append("close")


def _ready_status(config_digest, *, ready=True, reason=""):
    return {
        "schema_version": 1,
        "robot_name": "so101_single_arm",
        "motion_authorized": ready,
        "active_control_mode": "moveit_planning",
        "busy": False,
        "active_task_id": "",
        "default_skill_timeout_sec": 30.0,
        "task_budget_sec": 180.0,
        "rpc_timeout_sec": 5.0,
        "config_digest": config_digest,
        "registry_epoch": "epoch-1",
        "registry_generation": 1,
        "registry_digest": "registry-digest",
        "request_state": "",
        "request_error_code": "",
        "capabilities": [
            {
                "name": "move_relative_ee",
                "ready": ready,
                "reason": reason,
                "required_control_mode": "moveit_planning",
            },
            {
                "name": "open_gripper_skill",
                "ready": ready,
                "reason": reason,
                "required_control_mode": "moveit_planning",
            },
        ],
    }


def test_status_command_uses_hardware_discovery_timeout_floor(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _FakeBridge(_ready_status("legacy-digest"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _transport: bridge)

    assert cli.main(["--config-path", str(CONFIG_PATH), "status"]) == 0

    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert bridge.status_timeouts == [30.0]


def test_reload_catalog_command_uses_bound_gateway_source(monkeypatch, capsys):
    from robot_skill_cli import cli

    bridge = _FakeBridge(_ready_status("legacy-digest"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _transport: bridge)

    assert (
        cli.main(
            [
                "--config-path",
                str(CONFIG_PATH),
                "reload-catalog",
                "--request-id",
                "reload-1",
                "--force",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out)["data"]["changed_skills"] == ["nod_yes"]
    assert bridge.reload_requests == [{"request_id": "reload-1", "force": True, "timeout_sec": 60.0}]


def test_validate_uses_verified_snapshot_instead_of_legacy_config_digest(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status("different-digest"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "1.0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert view["capability_digest"] != bridge.status["config_digest"]
    assert exit_code == 0
    assert payload["ok"] is True
    assert bridge.calls == ["start", "status", "validate", "close"]


def test_validate_checks_schema_before_readiness_or_safety(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="MOTION_NOT_AUTHORIZED"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "open_gripper_skill",
            "--motion-distance",
            "1.0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert bridge.calls == ["start", "status", "close"]


def test_validate_checks_runtime_readiness_before_safety(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="MOTION_NOT_AUTHORIZED"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"]["code"] == "MOTION_NOT_AUTHORIZED"
    assert payload["error"]["message"] == "MOTION_NOT_AUTHORIZED"
    assert bridge.calls == ["start", "status", "close"]


@pytest.mark.parametrize(
    "reason",
    [
        "MOTION_NOT_AUTHORIZED: operator authorization is disabled",
        "CONTROL_MODE_MISMATCH: requires moveit_planning, active mode is teleop",
        "SKILL_BUSY: another root execution is active",
        "CAPABILITY_NOT_READY: validate skill service unavailable",
        "CAPABILITY_NOT_READY: task executor action unavailable",
        "CAPABILITY_NOT_READY: arm trajectory action unavailable",
        "CAPABILITY_NOT_READY: ee pose unavailable or stale",
    ],
)
def test_validate_preserves_detailed_gateway_readiness_reason(monkeypatch, capsys, reason):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason=reason))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"] == {"code": reason.split(":", 1)[0], "message": reason}
    assert bridge.calls == ["start", "status", "close"]


def test_validate_preserves_legacy_bare_gateway_reason(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="CAPABILITY_NOT_READY"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"] == {"code": "CAPABILITY_NOT_READY", "message": "CAPABILITY_NOT_READY"}
    assert bridge.calls == ["start", "status", "close"]


def test_validate_splits_only_first_gateway_reason_colon(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    reason = "CAPABILITY_NOT_READY: ee pose unavailable or stale: gateway snapshot retained"
    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason=reason))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["error"] == {"code": "CAPABILITY_NOT_READY", "message": reason}


def test_validate_checks_timeout_schema_before_runtime_readiness(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"], ready=False, reason="MOTION_NOT_AUTHORIZED"))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    exit_code = cli.main(
        [
            "--config-path",
            str(CONFIG_PATH),
            "validate",
            "move_relative_ee",
            "--motion-direction",
            "forward",
            "--motion-distance",
            "0.03",
            "--timeout-sec",
            "999.0",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error"]["code"] == "INVALID_ARGUMENT"
    assert bridge.calls == ["start", "status", "close"]


def test_validate_uses_status_default_timeout_and_shared_payload_hash(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"]))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)
    base_arguments = [
        "--config-path",
        str(CONFIG_PATH),
        "validate",
        "move_relative_ee",
        "--motion-direction",
        "forward",
        "--motion-distance",
        "0.03",
    ]

    assert cli.main(base_arguments) == 0
    omitted = json.loads(capsys.readouterr().out)
    assert cli.main([*base_arguments, "--timeout-sec", "30.0"]) == 0
    explicit = json.loads(capsys.readouterr().out)

    assert omitted["data"]["payload"] == explicit["data"]["payload"]
    assert omitted["data"]["payload_hash"] == explicit["data"]["payload_hash"]
    assert bridge.validate_payloads[0] == bridge.validate_payloads[1]


def test_omitted_timeout_uses_skill_implementation_limit():
    from robot_skill_cli.catalog import describe_skill, load_capability_catalog
    from robot_skill_cli.cli import _validate_timeout

    view = load_capability_catalog(config_path=CONFIG_PATH)
    skill = describe_skill(view, "move_relative_ee")
    status = {"default_skill_timeout_sec": 300.0, "task_budget_sec": 360.0}

    assert _validate_timeout(status, None, skill_timeout_cap=skill["timeout_sec"]) == 120.0


def test_validate_uses_action_wire_zero_for_omitted_motion_distance(monkeypatch, capsys):
    from robot_skill_cli import cli
    from robot_skill_cli.catalog import load_capability_catalog

    view = load_capability_catalog(config_path=CONFIG_PATH)
    bridge = _FakeBridge(_ready_status(view["capability_digest"]))
    monkeypatch.setattr(cli, "_create_bridge", lambda _context: bridge)

    assert cli.main(["--config-path", str(CONFIG_PATH), "validate", "open_gripper_skill"]) == 0

    response = json.loads(capsys.readouterr().out)
    assert response["data"]["payload"]["motion_distance"] == 0.0
    assert response["data"]["payload_hash"] == skill_payload_hash(response["data"]["payload"])
    assert bridge.validate_payloads == [response["data"]["payload"]]

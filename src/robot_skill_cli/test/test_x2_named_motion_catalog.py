"""The X2 named-motion catalog compiles offline and matches the runtime it targets."""

from pathlib import Path

import yaml

from embodied_common.dispatch_binding import delegated_executor_identity
from robot_config.loader import load_robot_config_dict

SRC = Path(__file__).resolve().parents[2]
CONFIG_PATH = SRC / "robot_config" / "config" / "robots" / "aimdk_x2_skills.yaml"
RUNTIME_PROFILE = SRC / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra.yaml"


def _snapshot():
    from robot_skill_cli.catalog import compile_local_snapshot

    config = load_robot_config_dict(CONFIG_PATH, defer_interface_binding=True)
    return compile_local_snapshot(config, CONFIG_PATH)


def test_x2_profile_exposes_only_named_motion_skills():
    snapshot = _snapshot()

    assert snapshot.robot_context.context_schema_version == 1
    assert snapshot.robot_context.required_control_mode == "named_motion"
    assert snapshot.enabled_skill_names == ("blow_kiss", "clap_hands", "handshake", "raise_hand", "wave_hand")
    assert snapshot.planner_visible_skill_names == snapshot.enabled_skill_names
    for name in snapshot.enabled_skill_names:
        capability = snapshot.capability_view[name]
        assert capability["required_control_mode"] == "named_motion"
        assert list(capability["required_capabilities"]) == ["motion.named"]
        assert capability["moves_robot"] is True
        assert snapshot.templates[name]["executor"] == "runtime_named_motion"


def test_every_bound_motion_is_advertised_by_the_x2_runtime_profile():
    """Static guard against catalog/runtime drift; the executor re-checks the live status."""
    profile = yaml.safe_load(RUNTIME_PROFILE.read_text(encoding="utf-8"))
    advertised = set(profile["capabilities"]["motion.named"]["names"])
    snapshot = _snapshot()

    bound = {snapshot.templates[name]["binding"]["motion"] for name in snapshot.enabled_skill_names}

    assert bound <= advertised
    assert bound == {"wave", "handshake", "raise_hand", "blow_kiss", "clap"}


def test_offline_executor_identity_matches_the_skill_executor():
    from skill_library.skill_executor_node import SkillExecutorNode

    config = load_robot_config_dict(CONFIG_PATH, defer_interface_binding=True)
    config["runtime"]["interface_description"] = {
        "robot": {"runtime_name": "aimdk_robot"},
        "interfaces": {"motion.named": {"endpoint": "/motion/execute_named"}},
    }
    from robot_skill_cli.catalog import compile_local_snapshot

    offline = compile_local_snapshot(config, CONFIG_PATH).delegated_executors["runtime_named_motion"]
    runtime = object.__new__(SkillExecutorNode)
    runtime._skill_templates = {}
    runtime._grasp_execution = {}
    runtime._placement_execution = {}
    runtime._pick_action_name = "/manipulation/execute_pick"
    runtime._place_action_name = "/manipulation/execute_place"
    runtime._semantic_map_target_service = ""
    runtime._runtime_named_motion_enabled = True
    runtime._runtime_named_motion_action = "/motion/execute_named"
    runtime._runtime_name = "aimdk_robot"

    online = runtime._delegated_executor_descriptors()["runtime_named_motion"]

    assert offline == online
    assert (
        offline.configuration_digest
        == (
            delegated_executor_identity(
                name="runtime_named_motion",
                endpoint_name="/motion/execute_named",
                endpoint_kind="ros_action",
                configuration={"runtime_name": "aimdk_robot"},
            )["configuration_digest"]
        )
    )

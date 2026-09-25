from pathlib import Path

import pytest
from ament_index_python.packages import PackageNotFoundError

from robot_config.launch_builders import task_execution as task_execution_builder
from robot_config.launch_builders.cmd_vel import generate_cmd_vel_nodes
from robot_config.launch_builders.task_execution import generate_task_executor_node
from robot_config.loader import load_robot_config_dict

CONFIG = Path(__file__).parents[1] / "config" / "robots" / "lekiwi_handeye_realsense_grasp.yaml"


def _launch_parameter_value(node, name: str):
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if key != name:
            continue
        if not isinstance(raw_value, tuple):
            return raw_value
        text = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
        if text.endswith("\n..."):
            text = text[:-4].rstrip()
        if text in {"True", "False"}:
            return text == "True"
        try:
            return float(text)
        except ValueError:
            return text
    raise KeyError(name)


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_310p_profile_uses_feedback_and_tf_completion_barrier() -> None:
    # Motion services come from the robot runtime; the generic YAML only keeps
    # the frames/feedback contract that consumers address.
    config = load_robot_config_dict(CONFIG)

    assert config["moveit"]["motion_status_hold_s"] == 0.0
    assert config["moveit"]["motion_feedback_timeout_s"] == 0.3
    assert config["moveit"]["motion_feedback_tolerance_rad"] == 0.12
    assert config["moveit"]["motion_require_tf_sync"] is True
    # LeKiwiSystemHardware does not publish the typed hardware heartbeat.
    assert config["moveit"]["motion_hardware_feedback_topic"] == ""
    assert config["moveit"]["joint_state_topic"] == "/arm_joint_state_broadcaster/joint_states"
    assert config["motion_mode"]["enabled"] is True
    assert config["motion_mode"]["navigation_enabled_on_startup"] is False
    assert config["grasp_execution"]["contact_realign"]["settle_sec"] == 0.0
    assert config["grasp_execution"]["pose_diagnostics"]["settle_sec"] == 0.0


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_310p_profile_does_not_skip_the_initial_gripper_open() -> None:
    config = load_robot_config_dict(CONFIG)

    node = generate_task_executor_node(config, "moveit_planning")

    assert _launch_parameter_value(node, "skip_redundant_gripper_open") is False
    assert _launch_parameter_value(node, "gripper_open_position") == 1.0
    assert _launch_parameter_value(node, "gripper_position_tolerance") == 0.05
    assert _launch_parameter_value(node, "joint_state_max_age_s") == 0.25


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_310p_profile_keeps_base_bridge_online_but_command_gated() -> None:
    config = load_robot_config_dict(CONFIG)

    node = generate_cmd_vel_nodes(
        config["navigation"],
        motion_mode_config=config["motion_mode"],
    )[0]

    assert node.node_package == "robot_navigation"
    assert node.node_executable == "cmd_vel_bridge_node"
    assert _launch_parameter_value(node, "motion_mode_enabled") is True
    assert _launch_parameter_value(node, "navigation_enabled_on_startup") is False
    assert _launch_parameter_value(node, "navigation_enabled_topic") == "motion_mode/navigation_enabled"
    assert _launch_parameter_value(node, "navigation_mode_ack_topic") == "motion_mode/base_navigation_enabled"


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_sim_profile_keeps_motion_mode_bridge_online() -> None:
    config = load_robot_config_dict(CONFIG)

    node = generate_cmd_vel_nodes(
        config["navigation"],
        motion_mode_config=config["motion_mode"],
        use_sim=True,
    )[0]

    assert _launch_parameter_value(node, "use_sim_time") is True
    assert _launch_parameter_value(node, "navigation_enabled_on_startup") is False


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_moveit_task_executor_fails_before_launch_when_package_is_missing(monkeypatch) -> None:
    config = load_robot_config_dict(CONFIG)

    def raise_package_not_found(_package_name: str):
        raise PackageNotFoundError("task_dispatch")

    monkeypatch.setattr(task_execution_builder, "get_package_prefix", raise_package_not_found)

    with pytest.raises(RuntimeError, match="task_dispatch.*not present"):
        generate_task_executor_node(config, "moveit_planning")

"""Pipeline-acceleration profile tests for the lekiwi hand-eye config.

The task-executor node tests moved to task_dispatch with the
``task_executor`` entry (openspec change
``migrate-business-launch-builders``); the base-bridge launch assertion
moved to robot_navigation with the navigation builder (same change), and
the cmd_vel bridge param tests went away with the bridge itself (openspec
lekiwi-application-migration task 2.2). This file keeps the moveit profile
assertions for the provider-less hand-eye configuration.
"""

from pathlib import Path

import pytest

from robot_config.loader import load_robot_config_dict

CONFIG = Path(__file__).parents[1] / "config" / "robots" / "lekiwi_handeye_realsense_grasp.yaml"


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

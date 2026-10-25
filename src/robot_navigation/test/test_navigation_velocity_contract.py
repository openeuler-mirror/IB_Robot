"""Navigation velocity-contract tests (robot_navigation's contribution).

Migrated from ``robot_config/test/test_navigation_velocity_contract.py``
(openspec change ``migrate-business-launch-builders``, task B.7): the
rejection cases follow the navigation module (and its private validator) to
its owning package. The YAML-shape cases stayed in robot_config with the
robot configs; the cmd_vel builder cases went away with the bridge (openspec
lekiwi-application-migration task 2.2 deleted the builder and the node).
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _lidar_config() -> dict:
    path = ROOT / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["robot"]


def test_dynamic_navigation_rejects_a_stop_topic_off_the_admitted_channel():
    from robot_navigation.launch_builders.navigation import _validate_navigation_velocity_contract

    config = _lidar_config()
    navigation = dict(config["nav_stages"]["navigation"]["navigation"])
    navigation["command_server"] = dict(navigation["command_server"])
    navigation["command_server"]["stop_velocity_topic"] = "/cmd_vel_safe"

    with pytest.raises(ValueError, match="stop_velocity_topic must be /cmd_vel"):
        _validate_navigation_velocity_contract(config, navigation)


def test_dynamic_navigation_rejects_a_restated_bridge_block():
    from robot_navigation.launch_builders.navigation import _validate_navigation_velocity_contract

    config = _lidar_config()
    navigation = dict(config["nav_stages"]["navigation"]["navigation"])
    navigation["cmd_vel_bridge"] = {"enabled": True, "cmd_vel_topic": "/cmd_vel_safe"}

    with pytest.raises(ValueError, match="cmd_vel_bridge was removed"):
        _validate_navigation_velocity_contract(config, navigation)


def test_non_dynamic_navigation_accepts_the_contract_channel():
    from robot_navigation.launch_builders.navigation import _validate_navigation_velocity_contract

    config = _lidar_config()
    mapping = config["nav_stages"]["mapping"]["navigation"]

    # No dyn_avoid -> no velocity-contract requirements beyond the channel
    # itself; a missing command_server block is stage-validated elsewhere.
    _validate_navigation_velocity_contract(config, mapping)

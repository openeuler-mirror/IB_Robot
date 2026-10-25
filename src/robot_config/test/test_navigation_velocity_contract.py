"""Navigation velocity-channel shape tests (robot_config's share).

Upstream ``migrate-business-launch-builders`` split the former suite: the
validator rejection cases follow the navigation module to its owning package
(``robot_navigation/test/test_navigation_velocity_contract.py``), while the
YAML-shape cases stay here with the robot configs. The bridge-topic cases
were reworked by openspec lekiwi-application-migration tasks 2.2/3.2: the
bridge and its topic indirection are gone, so every producer publishes onto
the runtime's admitted /cmd_vel channel.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def _lidar_config() -> dict:
    path = ROOT / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["robot"]


def test_lidar_stages_admit_base_velocity_on_the_contract_channel():
    config = _lidar_config()

    mapping = config["nav_stages"]["mapping"]["navigation"]
    navigation = config["nav_stages"]["navigation"]["navigation"]

    # The bridge and its topic indirection are gone: every producer (planner,
    # collision monitor, command-server stops, teleop) publishes onto the
    # runtime's admitted /cmd_vel channel (openspec
    # lekiwi-application-migration tasks 2.2/3.2).
    assert "cmd_vel_bridge" not in mapping
    assert "cmd_vel_bridge" not in navigation
    assert navigation["command_server"]["stop_velocity_topic"] == "/cmd_vel"

# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base gamepad under a runtime provider (lekiwi-teleoperation, tasks 4.1-4.3).

Moved from robot_config's test suite with the teleop builder (upstream
``migrate-business-launch-builders``): the gamepad publishes the runtime's
public base channel and is checked against the base.cmd_vel interface the
runtime binds -- topic, type, deadman, scales within the published limits,
and service commands limited to the stop contract.
"""

import copy
from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory

from robot_teleop.launch_builders.teleop import (
    ConfigError,
    generate_teleop_nodes,
    validate_provider_base_drive,
)

GAMEPAD = Path(get_package_share_directory("robot_config")) / "config" / "lekiwi" / "lekiwi_gamepad_base.yaml"
CONFIGS = Path(get_package_share_directory("robot_config")) / "config" / "robots"
# What the LeKiwi runtime binds for base.cmd_vel (profiles lekiwi_lidar and
# lekiwi_nav_grasp publish the same envelope).
BASE = {
    "endpoint": "/cmd_vel",
    "message_type": "geometry_msgs/msg/Twist",
    "limits": {"max_vx": 0.26, "max_vy": 0.23, "max_wz": 1.5},
}


def _gamepad() -> dict:
    return yaml.safe_load(GAMEPAD.read_text())


def _drive(config: dict) -> dict:
    return config["joy_teleop"]["ros__parameters"]["base_drive"]


def test_shipped_gamepad_fits_the_runtime_base_channel():
    assert validate_provider_base_drive(_gamepad(), BASE) == []


def test_stop_button_calls_the_runtime_stop_contract():
    stop = _gamepad()["joy_teleop"]["ros__parameters"]["runtime_stop"]
    assert stop["service_name"] == "/runtime/stop"
    assert stop["interface_type"] == "ibrobot_msgs/srv/StopRuntime"
    assert stop["service_request"] == {"policy": "HOLD"}
    assert set(stop["buttons"]).isdisjoint(_drive(_gamepad())["deadman_buttons"])


@pytest.mark.parametrize("axis,limit", (("linear-x", "max_vx"), ("linear-y", "max_vy"), ("angular-z", "max_wz")))
def test_scale_beyond_the_published_limit_is_rejected_naming_it(axis, limit):
    config = _gamepad()
    _drive(config)["axis_mappings"][axis]["scale"] = BASE["limits"][limit] + 0.01
    errors = validate_provider_base_drive(config, BASE)
    assert any(axis in e and limit in e for e in errors), errors


def test_wrong_topic_and_missing_deadman_are_rejected():
    config = _gamepad()
    _drive(config)["topic_name"] = "/base_controller/cmd_vel"
    del _drive(config)["deadman_buttons"]
    errors = validate_provider_base_drive(config, BASE)
    assert any("/cmd_vel" in e for e in errors) and any("deadman" in e for e in errors), errors


def test_service_other_than_runtime_stop_is_rejected():
    config = _gamepad()
    config["joy_teleop"]["ros__parameters"]["estop"] = {
        "type": "service",
        "interface_type": "std_srvs/srv/SetBool",
        "service_name": "/emergency_stop",
        "service_request": {"data": True},
        "buttons": [0],
    }
    assert any("/runtime/stop" in e for e in validate_provider_base_drive(config, BASE))


@pytest.mark.parametrize("name", ("lekiwi_lidar", "lekiwi_nav_grasp"))
def test_migrated_configs_launch_the_gamepad_on_the_base_channel(name):
    robot = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())["robot"]
    teleop = robot["teleoperation"]
    assert teleop["active_device"] == "", "the gamepad is opt-in per host"
    robot = copy.deepcopy(robot)
    robot["teleoperation"]["active_device"] = "lekiwi_gamepad"
    robot["runtime"]["interface_description"] = {"interfaces": {"base.cmd_vel": BASE}}
    nodes = generate_teleop_nodes(robot, {})
    assert [(n.node_package, n.node_executable) for n in nodes] == [("joy", "joy_node"), ("joy_teleop", "joy_teleop")]


def test_provider_gamepad_requires_a_bound_base_channel():
    robot = yaml.safe_load((CONFIGS / "lekiwi_lidar.yaml").read_text())["robot"]
    robot["teleoperation"]["active_device"] = "lekiwi_gamepad"
    robot["runtime"]["interface_description"] = {"interfaces": {}}
    with pytest.raises(ConfigError, match="base.cmd_vel"):
        generate_teleop_nodes(robot, {})

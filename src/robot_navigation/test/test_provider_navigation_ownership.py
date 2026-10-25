"""Under a runtime provider, the runtime alone owns lidar odometry and the base.

(Migrated from ``robot_config/test/test_provider_navigation_ownership.py``
— openspec change ``migrate-business-launch-builders``, task B.7: the tests
follow the navigation module to its owning package; assertions are
verbatim, the robot config loads through robot_config's installed share
directory and only the module import path changed.)

Regressions for binding ``lekiwi_nav_grasp`` to ``runtime.provider: lekiwi_robot``:

- ``launch_builders.runtime`` handed each stage's ``navigation.fast_lio`` to the
  robot runtime while the generic navigation builder kept launching FAST-LIO and
  its odom bridge as well -- two lidar odometry chains and two
  ``odom -> base_link`` publishers.
- The generic ``cmd_vel_bridge`` was removed entirely (openspec
  lekiwi-application-migration task 2.2): the runtime's base node owns cmd_vel
  projection, so no navigation stage may launch a second driver of the wheels.
- Stages that drive the base from boot declare ``navigation.open_gate_on_startup``:
  the builder starts the launch-time admission client (mode + contract gate),
  wired to shut the launch down when admission is rejected (design A3/A5).
"""

from copy import deepcopy
from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory

from robot_config.launch_builders.runtime import runtime_provider
from robot_config.loader import load_robot_config_dict
from robot_navigation.launch_builders.navigation import generate_navigation_nodes

CONFIG_PATH = Path(get_package_share_directory("robot_config")) / "config/robots/lekiwi_nav_grasp.yaml"
FAST_LIO_STAGES = ("mapping", "navigation", "hybrid")


def _executables(actions) -> list[str]:
    names = []
    for action in actions:
        if hasattr(action, "node_package"):
            names.append(f"{action.node_package}/{action.node_executable}")
        # GroupActions/conditions hide their nodes; nothing we assert on uses
        # them on this path, so a shallow scan is sufficient here.
    return names


def _stage(nav_stage: str) -> dict:
    return load_robot_config_dict(CONFIG_PATH, nav_stage=nav_stage, defer_interface_binding=True)


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("yolox_x_640")
@pytest.mark.parametrize("nav_stage", FAST_LIO_STAGES)
def test_provider_stage_leaves_fast_lio_and_the_base_to_the_runtime(nav_stage):
    config = _stage(nav_stage)
    assert runtime_provider(config) == "lekiwi_robot"
    assert config["navigation"]["fast_lio"]["enabled"] is True
    assert config["navigation"]["open_gate_on_startup"] is True
    assert "cmd_vel_bridge" not in config["navigation"]

    launched = _executables(generate_navigation_nodes(config, use_sim=False, force_enable=True))

    assert not [name for name in launched if "fast_lio" in name or "fastlio" in name], launched
    # The generic bridge is deleted; the only base driver is the runtime's.
    assert not [name for name in launched if "cmd_vel_bridge" in name], launched


@pytest.mark.model_bundle("yolox_x_640")
@pytest.mark.parametrize("nav_stage", ("navigation", "hybrid"))
def test_provider_stage_with_collision_monitor_launches_the_admission_client(nav_stage):
    """dyn_avoid stages launch under the provider since the 3.2 reroute.

    The collision monitor now publishes onto the admitted ``/cmd_vel`` channel
    (controller -> ``/cmd_vel_nav`` -> monitor -> ``/cmd_vel``), so the #457
    refusal is gone. Base-driving stages additionally start the admission
    client that selects the runtime mode and opens the gate, and a failure
    shuts the launch down.
    """
    config = _stage(nav_stage)
    assert config["navigation"]["nav2_bringup"]["dyn_avoid_enabled"] is True

    actions = generate_navigation_nodes(config, use_sim=False, force_enable=True)
    launched = _executables(actions)
    assert "robot_navigation/navigation_admission" in launched, launched
    assert any(type(action).__name__ == "RegisterEventHandler" for action in actions), actions


@pytest.mark.parametrize("nav_stage", ("mapping", "navigation"))
def test_provider_stage_without_gate_intent_launches_no_admission_client(nav_stage):
    config = _stage(nav_stage)
    config["navigation"]["open_gate_on_startup"] = False

    launched = _executables(generate_navigation_nodes(config, use_sim=False, force_enable=True))

    assert "robot_navigation/navigation_admission" not in launched, launched


def test_provider_stage_requires_a_runtime_mode_for_gate_intent():
    config = _stage("navigation")
    config["control_modes"][config["default_control_mode"]].pop("runtime_mode")

    with pytest.raises(ValueError, match="runtime_mode"):
        generate_navigation_nodes(config, use_sim=False, force_enable=True)


def test_provider_less_stage_still_launches_fast_lio_and_no_bridge(nav_stage=FAST_LIO_STAGES[0]):
    config = deepcopy(_stage(nav_stage))
    config.pop("runtime")
    config["navigation"].pop("open_gate_on_startup", None)

    launched = _executables(generate_navigation_nodes(config, use_sim=False, force_enable=True))

    assert launched.count("fast_lio/fastlio_mapping") == 1, launched
    assert launched.count("robot_navigation/fast_lio_odom_bridge") == 1, launched
    # The generic bridge is gone for provider-less configurations too.
    assert not [name for name in launched if "cmd_vel_bridge" in name], launched


@pytest.mark.model_bundle("yolox_x_640")
@pytest.mark.parametrize("nav_stage", FAST_LIO_STAGES[1:])
def test_provider_less_fast_lio_stages_still_launch_their_odometry_chain(nav_stage):
    config = deepcopy(_stage(nav_stage))
    config.pop("runtime")
    config["navigation"].pop("open_gate_on_startup", None)

    launched = _executables(generate_navigation_nodes(config, use_sim=False, force_enable=True))

    assert launched.count("fast_lio/fastlio_mapping") == 1, launched
    assert launched.count("robot_navigation/fast_lio_odom_bridge") == 1, launched
    assert not [name for name in launched if "cmd_vel_bridge" in name], launched

"""Launch-builder tests for the navigation entry (robot_navigation's contribution).

Migrated from ``robot_config/test/test_launch_readiness.py`` (the
sim-style navigation node-generation cases) — openspec change
``migrate-business-launch-builders``, task B.7: the tests follow the
navigation module to its owning package. Assertions are verbatim; the EKF
config loads through robot_config's installed share directory.
"""

from pathlib import Path

import pytest
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_config.loader import load_robot_config_dict
from robot_navigation.launch_builders.navigation import generate_navigation_nodes

ekf_config = str(Path(get_package_share_directory("robot_config")) / "config" / "lekiwi" / "navigation" / "ekf.yaml")


def test_generate_navigation_nodes_for_lekiwi_mode():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": True,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        }
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


def test_generate_navigation_nodes_honors_force_enable_override():
    ekf_config = str(Path(__file__).resolve().parents[1] / "config" / "lekiwi" / "navigation" / "ekf.yaml")
    nodes = generate_navigation_nodes(
        {
            "navigation": {
                "enabled": False,
                "default_mode": "full",
                "modes": {"full": {"config": ekf_config}},
            }
        },
        force_enable=True,
    )

    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_310p_profile_launches_no_base_bridge() -> None:
    """The generic cmd_vel bridge is gone for provider-less stages too.

    Moved from robot_config's pipeline-acceleration suite with the navigation
    builder (upstream ``migrate-business-launch-builders``). Coverage note
    (openspec lekiwi-application-migration task 2.2): the old bridge param
    wiring tests were removed with the node. The projection's numerical
    equivalence was recorded in the change's baseline replay; the mode/gate
    handshake is covered by the admission-client and loader rejection tests.
    """
    config_path = (
        Path(get_package_share_directory("robot_config")) / "config" / "robots" / "lekiwi_handeye_realsense_grasp.yaml"
    )
    config = load_robot_config_dict(config_path)

    launched = [
        f"{node.node_package}/{node.node_executable}"
        for node in generate_navigation_nodes(config, use_sim=False, force_enable=True)
        if hasattr(node, "node_package")
    ]

    assert not [name for name in launched if "cmd_vel_bridge" in name], launched

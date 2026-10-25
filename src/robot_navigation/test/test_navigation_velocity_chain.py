"""Velocity chain wiring after the collision-monitor reroute (task 3.2, R2-1).

The pre-migration chain was controller -> /cmd_vel_nav -> velocity_smoother ->
/cmd_vel -> collision_monitor -> /cmd_vel_safe -> bridge. After the migration
the base is driven only through the runtime's admitted /cmd_vel channel, and
the velocity smoother must stay in the chain (no silent loss of accel/velocity
smoothing):

- collision monitor ON:  controller -> /cmd_vel_nav -> velocity_smoother ->
  /cmd_vel_smoothed -> collision_monitor -> /cmd_vel
- collision monitor OFF (stock): controller -> /cmd_vel_nav ->
  velocity_smoother -> /cmd_vel

In both modes exactly one node publishes /cmd_vel: the monitor when it runs,
the smoother otherwise. These tests introspect the vendored
navigation_launch.py directly (its OpaqueFunction builds nodes from
launch_configurations, so no LaunchService is needed) and pin the collision
monitor topics in both nav2 param profiles.
"""

from pathlib import Path

import yaml
from launch.launch_context import LaunchContext

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "config"
PARAM_PROFILES = [CONFIG_ROOT / "nav2_params.yaml", CONFIG_ROOT / "nav2_params_dwb.yaml"]


def _load_navigation_launch():
    """Import the vendored launch file; launch/ is not a Python package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "robot_navigation_vendored_navigation_launch", PACKAGE_ROOT / "launch" / "navigation_launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VENDORED = _load_navigation_launch()
_navigation_nodes = _VENDORED._navigation_nodes


def _nodes(collision_monitor: bool):
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "namespace": "",
            "use_sim_time": "false",
            "autostart": "true",
            "params_file": str(CONFIG_ROOT / "nav2_params.yaml"),
            "use_composition": "False",
            "use_respawn": "False",
            "container_name": "nav2_container",
            "log_level": "info",
            "use_collision_monitor": "true" if collision_monitor else "false",
        }
    )
    nodes = {}
    for action in _navigation_nodes(context):
        for node in getattr(action, "_GroupAction__actions", None) or []:
            nodes[str(node._Node__node_name)] = node
    return nodes


def _remappings(node) -> dict[str, str]:
    resolved = {}

    def _text(side) -> str:
        return "".join(getattr(part, "text", str(part)) for part in side)

    for source, target in node._Node__remappings:
        resolved[_text(source)] = _text(target)
    return resolved


def test_smoother_stays_in_the_chain_with_the_collision_monitor():
    nodes = _nodes(collision_monitor=True)

    assert set(nodes) == {
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
        "lifecycle_manager_navigation",
    }
    assert _remappings(nodes["velocity_smoother"]) == {
        "/tf": "tf",
        "/tf_static": "tf_static",
        "cmd_vel": "cmd_vel_nav",
        "cmd_vel_smoothed": "cmd_vel_smoothed",
    }
    # The lifecycle manager still manages all seven navigation nodes, smoother
    # included. launch_ros normalizes a list value inside a dict that also
    # carries substitutions into per-element TextSubstitution objects whose
    # text embeds a repr artifact ("\n...\n"); split it back off.
    parameters = {}
    for entry in nodes["lifecycle_manager_navigation"]._Node__parameters:
        for key, value in entry.items():
            name = "".join(getattr(part, "text", str(part)) for part in key)
            parameters[name] = value
    node_names = [
        "".join(getattr(part, "text", str(part)) for part in group).split("\n")[0] for group in parameters["node_names"]
    ]
    assert node_names == [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]


def test_without_collision_monitor_the_stock_chain_is_unchanged():
    nodes = _nodes(collision_monitor=False)

    assert _remappings(nodes["controller_server"])["cmd_vel"] == "cmd_vel_nav"
    assert _remappings(nodes["velocity_smoother"]) == {
        "/tf": "tf",
        "/tf_static": "tf_static",
        "cmd_vel": "cmd_vel_nav",
        "cmd_vel_smoothed": "cmd_vel",
    }


def test_exactly_one_publisher_on_the_admitted_channel_in_each_mode():
    """Count nodes whose resolved remappings put an output on cmd_vel.

    With the monitor on, the monitor (launched by nav2_bringup.launch.py, not
    this file) is the only cmd_vel publisher: the controller is on
    cmd_vel_nav and the smoother on cmd_vel_smoothed. With the monitor off,
    the smoother is the only cmd_vel publisher.
    """
    for collision_monitor in (True, False):
        nodes = _nodes(collision_monitor=collision_monitor)
        publishers = []
        for name, node in nodes.items():
            remap = _remappings(node)
            outputs = [key for key, value in remap.items() if value == "cmd_vel" and key.startswith("cmd_vel")]
            if outputs:
                publishers.append((name, outputs))
        if collision_monitor:
            assert publishers == [], publishers
        else:
            assert publishers == [("velocity_smoother", ["cmd_vel_smoothed"])], publishers


def test_nav2_profiles_route_the_collision_monitor_onto_the_admitted_channel():
    for path in PARAM_PROFILES:
        params = yaml.safe_load(path.read_text(encoding="utf-8"))
        monitor = params["collision_monitor"]["ros__parameters"]
        assert monitor["cmd_vel_in_topic"] == "/cmd_vel_smoothed", path
        assert monitor["cmd_vel_out_topic"] == "/cmd_vel", path


def test_vendored_launch_keeps_the_upstream_license_header():
    content = (PACKAGE_ROOT / "launch" / "navigation_launch.py").read_text(encoding="utf-8")
    assert "Copyright (c) 2018 Intel Corporation" in content
    assert "Apache License, Version 2.0" in content
    assert "http://www.apache.org/licenses/LICENSE-2.0" in content
    assert "Modifications Copyright 2026 IB_Robot Contributors" in content

"""LeKiwi runtime launch peripheral fragment application (runtime-deployment-peripherals D7).

Renders the shipped ``runtime.launch.py`` against the real ``lekiwi_nav_grasp``
profile: a deployment fragment disabling the declared ``front`` RealSense
camera removes it from the live description and from the physical and
synthetic peripheral actions alike.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import launch
import yaml
from launch_ros.actions import Node

_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_nav_grasp.yaml"
_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py"


def _runtime_launch_module():
    spec = importlib.util.spec_from_file_location("lekiwi_robot_runtime_launch_peripherals", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(simulated: str, peripherals_file: str):
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(_PROFILE),
            "simulated": simulated,
            "use_sim": "",
            "sim_platform": "",
            "gazebo_world": "",
            "mujoco_scene": "",
            "spawn_x": "0.0",
            "spawn_y": "0.0",
            "spawn_z": "0.05",
            "spawn_yaw": "0.0",
            "display": "false",
            "instance_id": "",
            "peripherals": peripherals_file,
        }
    )
    return _runtime_launch_module()._launch_setup(context)


def _node_params(node):
    from launch import LaunchContext
    from launch_ros.utilities import evaluate_parameters

    params = {}
    for entry in evaluate_parameters(LaunchContext(), node._Node__parameters):
        params.update(entry)
    return params


def _fragment(tmp_path, peripherals):
    path = tmp_path / "deployment_peripherals.yaml"
    path.write_text(yaml.safe_dump({"peripherals": peripherals}), encoding="utf-8")
    return str(path)


def test_fragment_disabling_a_camera_removes_it_from_description_and_actions(tmp_path):
    fragment = _fragment(tmp_path, [{"type": "camera", "name": "front", "disabled": True}])

    # physical: the facade keeps the fragment path and applies it at node
    # initialisation (D9); the description it will serve — computed exactly
    # the way the facade does — omits the camera, and no realsense driver
    # chain exists for it
    actions = _render("false", fragment)
    nodes = [action for action in actions if isinstance(action, Node)]
    facade = next(node for node in nodes if node._Node__node_executable == "runtime_facade")
    assert _node_params(facade)["peripherals"] == fragment
    from robot_runtime.interface_description import build_description
    from robot_runtime.peripherals import load_peripherals_file, merge_peripherals
    from robot_runtime.profile import load_profile

    profile = load_profile(_PROFILE)
    merged = merge_peripherals(profile.get("peripherals"), load_peripherals_file(fragment).get("peripherals"))
    description = build_description({**profile, "peripherals": merged}, use_sim=False)
    assert not any(name.startswith("camera.front.") for name in description["interfaces"])

    def node_ids(node):
        # (executable, node name, namespace): a driver chain is identified by
        # its executable AND its node name — the realsense driver node is
        # named after the camera, its relays and static TFs after it too.
        return (node._Node__node_executable, str(node._Node__node_name), str(node._Node__node_namespace))

    composed = {node_ids(node) for node in nodes}
    assert not any("front" in name for _exe, name, _ns in composed), (
        "the disabled front camera must compose no driver, relay or TF action"
    )
    assert not any(exe == "realsense2_camera_node" and "front" in name for exe, name, _ns in composed)
    assert any(exe == "realsense2_camera_node" and name == "wrist_camera" for exe, name, _ns in composed), (
        "the wrist camera keeps its driver"
    )
    # the other declared peripherals stay (the wrist camera keeps its surface)
    assert "camera.wrist.color" in description["interfaces"]

    # sdk: the synthetic perception description omits the camera too
    actions = _render("true", fragment)
    synthetic = next(
        node for node in actions if isinstance(node, Node) and node._Node__node_executable == "synthetic_perception"
    )
    synthetic_description = json.loads(_node_params(synthetic)["description_json"])
    assert not any(name.startswith("camera.front.") for name in synthetic_description["interfaces"])
    assert "camera.wrist.color" in synthetic_description["interfaces"]


def test_launch_declares_the_peripherals_argument():
    """D7 (m3): the provider's real launch description declares the
    peripherals argument every provider contract requires."""
    from launch.actions import DeclareLaunchArgument

    def text(value):
        return "".join(part.text if hasattr(part, "text") else str(part) for part in value)

    description = _runtime_launch_module().generate_launch_description()
    names = {
        text(action._DeclareLaunchArgument__name)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert "peripherals" in names

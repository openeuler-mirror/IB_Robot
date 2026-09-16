"""Construct the real ROS executor from runtime composition, without device I/O."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import rclpy
import yaml
from so101_robot import teleoperation

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import render_robot_description


def test_runtime_helper_constructs_real_ros_node(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "173")
    root = Path(__file__).resolve().parents[5]
    profile_path = root / "src/robots/so101/so101_robot/profiles/so101_single_arm.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    xml = render_robot_description(profile, profile_path, True)
    descriptor = build_description(profile, simulated=True, robot_description=xml)
    monkeypatch.setattr(teleoperation, "Node", lambda **kwargs: kwargs)
    action = teleoperation.generate_teleoperation_nodes(profile, descriptor, {"robot_description": xml})[0]
    params = action["parameters"][0]
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}))
    source = root / "src/robots/so101/so101_motion/scripts/so101_placo_servo_node.py"
    spec = importlib.util.spec_from_file_location("managed_teleop_construction", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    captured = {}
    monkeypatch.setattr(module, "_require_placo", lambda _logger: None)
    monkeypatch.setattr(
        module, "SO101PlacoDiffIK", lambda **kwargs: captured.update(kwargs) or SimpleNamespace(close=lambda: None)
    )
    rclpy.init(args=["--ros-args", "--params-file", str(path)])
    node = None
    try:
        node = module.SO101PlacoServoNode()
        assert node.managed_teleop
        assert node.input_mode == "auto"
        assert not node._enabled and not node._managed_owner
        assert captured["urdf_xml"] == xml
        assert node.joint_intent_topic == "/motion/arm/joints"
        assert node.gripper_joint_names == ["6"]
        assert node._on_start_srv(None, module.Trigger.Response()).success is False
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()

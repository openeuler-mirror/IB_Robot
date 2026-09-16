# Copyright 2026 IB_Robot Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from xml.etree import ElementTree

import xacro


def test_simulated_transport_model_is_single_arm_and_has_no_startup_home():
    package = Path(__file__).resolve().parents[1]
    document = xacro.process_file(
        str(package / "urdf/lerobot/so101/so101.urdf.xacro"),
        mappings={"use_sim": "false", "simulated": "true"},
    )
    robot = ElementTree.fromstring(document.toxml())
    controls = robot.findall("ros2_control")
    assert len(controls) == 1
    hardware = controls[0].find("hardware")
    assert hardware.findtext("plugin") == "so101_hardware/SO101SystemHardware"
    params = {node.get("name"): node.text or "" for node in hardware.findall("param")}
    assert params["simulated"] == "true"
    assert not params["reset_positions"]
    assert "home_positions" not in params
    joints = controls[0].findall("joint")
    assert [joint.get("name") for joint in joints] == [str(i) for i in range(1, 7)]
    for joint in joints:
        name = joint.get("name")
        assert joint.findtext("param[@name='id']") == name
        limit = robot.find(f"joint[@name='{name}']/limit")
        command = joint.find("command_interface[@name='position']")
        assert float(command.findtext("param[@name='min']")) == float(limit.get("lower"))
        assert float(command.findtext("param[@name='max']")) == float(limit.get("upper"))
    for mesh in robot.findall(".//mesh"):
        prefix = "package://so101_description/"
        assert mesh.get("filename").startswith(prefix)
        assert (package / mesh.get("filename").removeprefix(prefix)).is_file()

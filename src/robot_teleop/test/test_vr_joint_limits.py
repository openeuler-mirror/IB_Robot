"""Exercise managed VR limit validation and both outputs with real ROS messages."""

import json
import time

import numpy as np
import pytest
import rclpy
import yaml
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from robot_teleop.vr_teleop import VRDualArmTcpServer, VRTeleopNode


@pytest.fixture
def make_node(tmp_path):
    nodes = []

    def make(**overrides):
        params = {
            "managed_teleop": True,
            "output_profile": "so101",
            "so101_input_mode": "pose",
            "host": "127.0.0.1",
            "port": 0,
            "gripper_joint_name": "jaw",
            "so101_gripper_topic": "/test_vr/joints",
            "command_lease_topic": "/test_vr/lease",
            "so101_gripper_closed": -0.6,
            "so101_gripper_open": 1.6,
            "joint_limits": json.dumps({"arm": {"min": -1.0, "max": 1.0}, "jaw": {"min": -0.2, "max": 1.1}}),
            **overrides,
        }
        path = tmp_path / "vr.yaml"
        path.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}))
        rclpy.init(args=["--ros-args", "--params-file", str(path)])
        node = VRTeleopNode.__new__(VRTeleopNode)
        try:
            node.__init__()
        except Exception:
            Node.destroy_node(node)
            raise
        nodes.append(node)
        node._timer.cancel()
        return node

    yield make
    for node in nodes:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


@pytest.mark.parametrize("mode", ["pose", "velocity"])
@pytest.mark.parametrize("reversed_endpoints", [False, True])
def test_joint_intent_clamped_on_both_publication_paths(make_node, mode, reversed_endpoints):
    endpoints = {"so101_gripper_closed": 1.6, "so101_gripper_open": -0.6} if reversed_endpoints else {}
    node = make_node(so101_input_mode=mode, **endpoints)
    peer = Node("vr_limit_observer", use_global_arguments=False)
    received = []
    peer.create_subscription(JointState, "/test_vr/joints", received.append, 10)
    try:
        deadline = time.monotonic() + 3.0
        while node._so101_gripper_pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(peer, timeout_sec=0.02)
        assert node._so101_gripper_pub.get_subscription_count() == 1
        node._so101_started = True
        for count, grip in enumerate((0.0, 1.0, 0.5), start=1):
            if mode == "pose":
                node._publish_so101_pose(np.zeros(3), Rotation.identity(), grip)
            else:
                node._publish_so101(np.zeros(3), np.zeros(3), grip)
            deadline = time.monotonic() + 3.0
            while len(received) < count and time.monotonic() < deadline:
                rclpy.spin_once(peer, timeout_sec=0.02)
            assert len(received) == count
        expected = [-0.2, 1.1, 0.5] if reversed_endpoints else [1.1, -0.2, 0.5]
        assert [msg.position[0] for msg in received] == pytest.approx(expected)
        assert all(msg.name == ["jaw"] for msg in received)
        assert all(msg.header.stamp.sec > 0 for msg in received)
        node._publish_so101_gripper(float("nan"))
        node._so101_started = False
        node._publish_so101_gripper(0.0)
        rclpy.spin_once(peer, timeout_sec=0.1)
        assert len(received) == 3
    finally:
        peer.destroy_node()


@pytest.mark.parametrize(
    "limits",
    [
        "not json",
        "[]",
        "null",
        "{}",
        '{"jaw": null}',
        '{"jaw": {"min": 0}}',
        '{"jaw": {"min": true, "max": 1}}',
        '{"jaw": {"min": "0", "max": 1}}',
        '{"jaw": {"min": NaN, "max": 1}}',
        '{"jaw": {"min": 0, "max": Infinity}}',
        '{"jaw": {"min": 1, "max": 1}}',
        '{"jaw": {"min": 2, "max": 1}}',
        '{"jaw": {"min": 0, "max": 1}, "arm": {"min": 2, "max": 1}}',
    ],
)
def test_malformed_limits_fail_before_tcp_start(make_node, monkeypatch, limits):
    started = []
    monkeypatch.setattr(VRDualArmTcpServer, "start", lambda self: started.append(True))
    with pytest.raises(ValueError):
        make_node(joint_limits=limits)
    assert not started


def test_managed_humanoid_cannot_bypass_joint_limits(make_node):
    with pytest.raises(ValueError, match="output_profile=so101"):
        make_node(output_profile="humanoid")

"""Construct the real ROS executor from runtime composition, without device I/O."""

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
import yaml
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import SetRuntimeMode, StopRuntime
from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import render_robot_description
from so101_robot import teleoperation


@pytest.fixture
def runtime_node(tmp_path, monkeypatch, request):
    monkeypatch.setenv("ROS_DOMAIN_ID", "182")
    root = Path(__file__).resolve().parents[5]
    profile_path = root / "src/robots/so101/so101_robot/profiles/so101_single_arm.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    xml = render_robot_description(profile, profile_path, True)
    descriptor = build_description(profile, simulated=True, robot_description=xml)
    monkeypatch.setattr(teleoperation, "Node", lambda **kwargs: kwargs)
    action = teleoperation.generate_teleoperation_nodes(profile, descriptor, {"robot_description": xml})[0]
    params = action["parameters"][0]
    params.update(getattr(request, "param", {}))
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
        module,
        "SO101PlacoDiffIK",
        lambda **kwargs: (
            captured.update(kwargs)
            or SimpleNamespace(close=lambda: None, ee_position=lambda q: q[:3], ee_rotation=lambda q: np.eye(3))
        ),
    )
    rclpy.init(args=["--ros-args", "--params-file", str(path)])
    node = None
    try:
        node = module.SO101PlacoServoNode()
        yield node, module, captured, xml
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_runtime_helper_constructs_real_ros_node(runtime_node):
    node, module, captured, xml = runtime_node
    assert node.managed_teleop
    assert node.input_mode == "auto"
    assert not node._enabled and not node._managed_owner
    assert captured["urdf_xml"] == xml
    assert node.joint_intent_topic == "/motion/arm/joints"
    assert node.gripper_joint_names == ["6"]
    assert node.input_timeout == node.command_lease_timeout_s == node.home_joint_state_stale_s == 0.5
    # The status heartbeat budget is independent of the command budget.
    assert node.runtime_status_stale_s == 2.5
    assert node._begin_start(module.Trigger.Response()).success is False


@pytest.mark.parametrize("fresh_feedback,interrupt", [(True, False), (False, False), (True, True), ("after", False)])
def test_slow_admission_keeps_feedback_and_stop_callbacks_live(runtime_node, fresh_feedback, interrupt):
    node, _module, _captured, _xml = runtime_node
    # Keep a stricter deployment deadline to exercise a switch longer than it.
    node.home_joint_state_stale_s = 0.2
    peer = Node("admission_test_peer")
    executor = MultiThreadedExecutor(num_threads=4)
    state = SimpleNamespace(mode="idle", switching_at=None, stop_future=None, held=False)
    joints = peer.create_publisher(JointState, "/joint_states", 1)
    status_pub = peer.create_publisher(RuntimeStatus, node.runtime_status_topic, 1)
    start = peer.create_client(Trigger, node.start_srv_name)
    stop = peer.create_client(Trigger, node.stop_srv_name)

    def set_mode(request, response):
        assert request.mode == "stream"
        state.switching_at = time.monotonic()
        time.sleep(0.35)  # longer than the unchanged 0.2 s feedback deadline
        state.mode = "stream"
        response.success = True
        return response

    def hold(_request, response):
        state.mode = "idle"
        state.held = True
        response.success = True
        return response

    def publish_feedback():
        stamp = peer.get_clock().now().to_msg()
        if (
            fresh_feedback is True
            or state.switching_at is None
            or (fresh_feedback == "after" and time.monotonic() - state.switching_at > 0.42)
        ):
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = node.arm_joint_names + node.gripper_joint_names
            msg.position = [0.3 if state.switching_at is not None else 0.1] * len(msg.name)
            joints.publish(msg)
        status = RuntimeStatus()
        status.stamp = stamp
        status.lifecycle = "ACTIVE"
        status.active_mode = state.mode
        status.stop_latched = state.held
        status.stop_policy = "HOLD" if state.held else ""
        status_pub.publish(status)
        if interrupt and state.switching_at is not None and state.stop_future is None:
            state.stop_future = stop.call_async(Trigger.Request())

    peer.create_service(SetRuntimeMode, node.runtime_mode_service, set_mode)
    peer.create_service(StopRuntime, node.runtime_stop_service, hold)
    peer.create_timer(0.01, publish_feedback, callback_group=ReentrantCallbackGroup())
    executor.add_node(node)
    executor.add_node(peer)
    try:
        assert start.wait_for_service(timeout_sec=2.0)
        deadline = time.monotonic() + 3.0
        while (node._latest_js is None or node._runtime_status is None) and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.02)
        assert node._latest_js is not None and node._runtime_status is not None
        future = start.call_async(Trigger.Request())
        executor.spin_until_future_complete(future, timeout_sec=3.0)
        assert future.done()
        response = future.result()
        assert response.success == (bool(fresh_feedback) and not interrupt), response.message
        if response.success:
            assert node._managed_owner and node._enabled
            np.testing.assert_allclose(node._last_cmd, [0.3] * len(node.arm_joint_names))
        else:
            deadline = time.monotonic() + 2.0
            while node._start_request is not None and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
            assert node._start_request is None
            assert not node._managed_owner and not node._enabled
    finally:
        node.prepare_shutdown()
        executor.shutdown(timeout_sec=2.0)
        peer.destroy_node()


@pytest.mark.parametrize("runtime_node", [{"managed_teleop": False}], indirect=True)
@pytest.mark.parametrize("available", [False, True])
def test_nonmanaged_stop_keeps_estop_and_feedback_live(runtime_node, available):
    node, _module, _captured, _xml = runtime_node
    peer = Node("nonmanaged_stop_test_peer")
    executor = MultiThreadedExecutor(num_threads=4)
    stop = peer.create_client(Trigger, node.stop_srv_name)
    joints = peer.create_publisher(JointState, "/joint_states", 1)
    estop = peer.create_publisher(Bool, node.estop_topic, 1)
    state = SimpleNamespace(released=False)

    def idle(request, response):
        assert request.mode == "idle"
        time.sleep(0.7)
        state.released = True
        response.success = True
        return response

    if available:
        peer.create_service(SetRuntimeMode, node.runtime_mode_service, idle)

    def feedback():
        msg = JointState()
        msg.name = node.arm_joint_names
        msg.position = [0.1] * len(msg.name)
        joints.publish(msg)
        estop.publish(Bool(data=True))

    executor.add_node(node)
    executor.add_node(peer)
    try:
        assert stop.wait_for_service(timeout_sec=2.0)
        if available:
            assert node._runtime_mode_client.wait_for_service(timeout_sec=2.0)
        started = time.monotonic()
        future = stop.call_async(Trigger.Request())
        executor.spin_until_future_complete(future, timeout_sec=0.4)
        assert future.done() and time.monotonic() - started < 0.5
        assert future.result().success and "release in progress" in future.result().message
        assert not node._enabled
        generation = node._joint_state_generation
        peer.create_timer(0.01, feedback, callback_group=ReentrantCallbackGroup())
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and not (node._estop_active and node._joint_state_generation > generation):
            executor.spin_once(timeout_sec=0.01)
        assert node._estop_active and node._joint_state_generation > generation
        assert not state.released
        if available:
            deadline = time.monotonic() + 2.0
            while not node._stop_confirmed and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
            assert node._stop_confirmed
            response = node._on_stop_srv(None, Trigger.Response())
            assert response.success and "runtime idle release confirmed" in response.message
    finally:
        # Drain queued callbacks before destroying their ROS handles.
        for timer in (*peer.timers, *node.timers):
            timer.cancel()
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        executor.shutdown(timeout_sec=2.0)
        peer.destroy_node()

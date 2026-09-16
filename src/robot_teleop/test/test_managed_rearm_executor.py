"""Real single-threaded ROS executor coverage; no hardware or runtime process."""

import json
import time

import pytest
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from ibrobot_msgs.msg import RuntimeStatus
from robot_teleop.teleop_node import TeleopNode


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "174")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    # Legacy VR unit tests install an action stub during collection.
    import ibrobot_msgs.action
    from ibrobot_msgs.action._arm_return_home import ArmReturnHome

    monkeypatch.setattr(ibrobot_msgs.action, "ArmReturnHome", ArmReturnHome)
    config = {
        "type": "leader_topic",
        "managed_teleop": True,
        "joint_mapping": {"1": "1", "6": "6"},
        "input_gripper_joint": "6",
        "input_stale_s": 0.15,
        "source_topic": "/test_rearm/input",
        "base_link_name": "base",
        "tool_frame": "tool",
        "gripper_closed": -0.5,
        "gripper_open": 1.5,
        "cartesian_backend_config": {
            "start_srv": "/test_rearm/start",
            "stop_srv": "/test_rearm/stop",
            "runtime_status_topic": "/test_rearm/status",
        },
    }
    params = {
        "device_config": json.dumps(config),
        "joint_limits": json.dumps({"1": {"min": -1.0, "max": 1.0}, "6": {"min": -0.5, "max": 1.5}}),
        "managed_joint_topic": "/test_rearm/intent",
        "estop_topic": "/test_rearm/estop",
        "rearm_timeout_s": 0.8,
    }
    args = ["--ros-args"]
    for name, value in params.items():
        args.extend(["-p", f"{name}:={json.dumps(value)}"])
    rclpy.init(args=args)
    node = TeleopNode()
    peer = Node("test_rearm_peer", use_global_arguments=False)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(peer)
    state = {"starts": 0, "stops": 0, "samples": 0, "send_input": True, "reply": None}

    async def start(_request, response):
        state["starts"] += 1
        state["reply"] = Future()
        success, message = await state["reply"]
        response.success = success
        response.message = message
        return response

    def stop(_request, response):
        state["stops"] += 1
        response.success = True
        return response

    peer.create_service(Trigger, "/test_rearm/start", start, callback_group=ReentrantCallbackGroup())
    peer.create_service(Trigger, "/test_rearm/stop", stop)
    publisher = peer.create_publisher(JointState, "/test_rearm/input", 1)
    status_pub = peer.create_publisher(RuntimeStatus, "/test_rearm/status", 1)
    estop_pub = peer.create_publisher(Bool, "/test_rearm/estop", 1)

    def sample():
        if state["send_input"]:
            msg = JointState()
            msg.header.stamp = peer.get_clock().now().to_msg()
            msg.header.frame_id = "leader_radians_gripper_ratio_v1"
            msg.name, msg.position = ["1", "6"], [0.25, 0.5]
            publisher.publish(msg)
            state["samples"] += 1

    peer.create_timer(0.01, sample)
    client = peer.create_client(Trigger, "/robot_teleop_node/rearm")

    def until(predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert predicate(), "executor did not make progress before deadline"

    try:
        until(lambda: client.service_is_ready() and node._managed_backend._start_cli.service_is_ready())
        until(lambda: node.device.sample is not None)
        yield node, peer, state, client, until, status_pub, estop_pub
    finally:
        node.destroy_node()
        peer.destroy_node()
        executor.shutdown(timeout_sec=1.0)
        rclpy.shutdown()


def test_slow_admission_keeps_source_and_replies_live(admission):
    node, _peer, state, client, until, _status, _estop = admission
    first = client.call_async(Trigger.Request())
    until(lambda: state["starts"] == 1)
    stamp = node.device.stamp
    second = client.call_async(Trigger.Request())
    until(second.done)
    assert not second.result().success
    assert "pending" in second.result().message
    until(lambda: node.device.stamp > stamp + 0.25)
    assert not first.done()
    assert state["stops"] == 0
    state["reply"].set_result((True, "final runtime admission"))
    until(first.done)
    assert first.result().success
    assert first.result().message == "final runtime admission"
    assert node._managed_backend.is_enabled
    assert state["stops"] == 0


def test_final_rejection_is_returned(admission):
    _node, _peer, state, client, until, _status, _estop = admission
    result = client.call_async(Trigger.Request())
    until(lambda: state["reply"] is not None)
    state["reply"].set_result((False, "follower feedback unavailable"))
    until(result.done)
    assert not result.result().success
    assert result.result().message == "follower feedback unavailable"


@pytest.mark.parametrize("interrupt", ["timeout", "estop", "runtime_stop", "source_loss"])
def test_admission_failure_and_late_success_cannot_enable(admission, interrupt):
    node, _peer, state, client, until, status, estop = admission
    result = client.call_async(Trigger.Request())
    until(lambda: state["reply"] is not None)
    if interrupt == "estop":
        estop.publish(Bool(data=True))
    elif interrupt == "runtime_stop":
        msg = RuntimeStatus()
        msg.stop_latched = True
        status.publish(msg)
    elif interrupt == "source_loss":
        state["send_input"] = False
        until(lambda: time.monotonic() - node.device.received > node.device.timeout)
        state["reply"].set_result((True, "admitted"))
    until(result.done)
    assert not result.result().success
    assert not node._managed_backend.is_enabled
    until(lambda: state["stops"] >= 1)
    stops = state["stops"]
    if interrupt != "source_loss":
        state["reply"].set_result((True, "late admission"))
        until(lambda: state["stops"] > stops)
        assert not node._managed_backend.is_enabled


def test_rearm_requires_new_source_sample_before_start(admission):
    node, _peer, state, client, until, _status, _estop = admission
    state["send_input"] = False
    result = client.call_async(Trigger.Request())
    until(lambda: node._rearm_pending)
    assert node.device.sample is None
    until(result.done)
    assert not result.result().success
    assert "fresh leader input" in result.result().message
    assert state["starts"] == 0

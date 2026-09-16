"""Facade tests against a fake controller_manager: mode switching, stop ordering, latch, hardware lifecycle."""

from __future__ import annotations

import os
import threading
import time

import pytest
import rclpy
import yaml
from action_msgs.srv import CancelGoal
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SetHardwareComponentState, SwitchController
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode, StopRuntime
from robot_runtime import contract as C

ACTION = "/arm_trajectory_controller/follow_joint_trajectory"


class FakeControllerManager(Node):
    """Records every request the facade makes, in order."""

    def __init__(self):
        super().__init__("controller_manager")
        cb = ReentrantCallbackGroup()
        self.active: set[str] = set()
        self.hardware: dict[str, str] = {"so101_system": "active"}
        self.log: list[tuple] = []
        self.cancel_code = CancelGoal.Response.ERROR_NONE
        self.switch_entered = threading.Event()
        self.switch_release = threading.Event()
        self.switch_release.set()
        self.ignore_deactivation = False
        self.create_service(SwitchController, "controller_manager/switch_controller", self._switch, callback_group=cb)
        self.create_service(ListControllers, "controller_manager/list_controllers", self._list, callback_group=cb)
        self.create_service(
            SetHardwareComponentState, "controller_manager/set_hardware_component_state", self._hw, callback_group=cb
        )
        self.cancel_service = self.create_service(
            CancelGoal, f"{ACTION}/_action/cancel_goal", self._cancel, callback_group=cb
        )

    def _switch(self, request, response):
        self.log.append(
            ("switch", tuple(request.activate_controllers), tuple(request.deactivate_controllers), time.monotonic())
        )
        if request.activate_controllers:
            self.switch_entered.set()
            if not self.switch_release.wait(5.0):
                response.ok = False
                return response
        if not self.ignore_deactivation:
            self.active -= set(request.deactivate_controllers)
        self.active |= set(request.activate_controllers)
        response.ok = True
        return response

    def _list(self, _request, response):
        for name in ("arm_position_controller", "arm_trajectory_controller"):
            state = ControllerState()
            state.name = name
            state.state = "active" if name in self.active else "inactive"
            response.controller.append(state)
        return response

    def _hw(self, request, response):
        self.log.append(("hardware", request.name, request.target_state.label, time.monotonic()))
        self.hardware[request.name] = request.target_state.label
        response.ok = True
        return response

    def _cancel(self, _request, response):
        self.log.append(("cancel", ACTION, time.monotonic()))
        response.return_code = self.cancel_code
        return response


PROFILE = {
    "runtime": {"name": "facade_test", "version": "0.0.1"},
    "controller_manager": "controller_manager",
    "modes": {
        "initial": "idle",
        "idle": {"controllers": [], "transitions": ["stream", "trajectory"]},
        "stream": {
            "controllers": ["arm_position_controller"],
            "allows_stream": True,
            "transitions": ["idle", "trajectory"],
        },
        "trajectory": {
            "controllers": ["arm_trajectory_controller"],
            "allows_trajectory": True,
            "transitions": ["idle", "stream"],
        },
    },
    "capabilities": {
        "joint.state": {"joint_count": 1, "rate_hz": 50.0},
        "joint.trajectory": {"joint_count": 1},
        "runtime.stop": {"cancel_bound_s": 2.0, "idle_bound_s": 3.0, "torque_off_bound_s": 4.0},
    },
    "joints": ["1"],
    "trajectory_actions": [ACTION],
    "hardware_components": ["so101_system"],
}


class Caller(Node):
    def __init__(self):
        super().__init__("facade_test_caller")
        self.mode = self.create_client(SetRuntimeMode, C.SET_MODE_SERVICE)
        self.status = self.create_client(GetRuntimeStatus, C.GET_STATUS_SERVICE)
        self.stop = self.create_client(StopRuntime, C.STOP_SERVICE)

    def call(self, client, request):
        assert client.wait_for_service(timeout_sec=5.0), client.srv_name
        future = client.call_async(request)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        assert future.done(), f"{client.srv_name} timed out"
        return future.result()

    def set_mode(self, mode):
        request = SetRuntimeMode.Request()
        request.mode = mode
        return self.call(self.mode, request)

    def get_status(self):
        return self.call(self.status, GetRuntimeStatus.Request()).status

    def stop_with(self, policy):
        request = StopRuntime.Request()
        request.policy = policy
        return self.call(self.stop, request)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    os.environ.setdefault("ROS_DOMAIN_ID", "48")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
    profile_path = tmp_path_factory.mktemp("profile") / "runtime.yaml"
    profile_path.write_text(yaml.safe_dump(PROFILE), encoding="utf-8")
    rclpy.init()
    from robot_runtime.facade_node import RuntimeFacade

    cm = FakeControllerManager()
    facade = RuntimeFacade(profile_path=str(profile_path))
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(cm)
    executor.add_node(facade)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    caller = Caller()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and caller.get_status().lifecycle != C.LIFECYCLE_ACTIVE:
        time.sleep(0.2)
    assert caller.get_status().lifecycle == C.LIFECYCLE_ACTIVE, (
        "facade never reached ACTIVE against the fake controller_manager"
    )
    yield cm, caller
    executor.shutdown(timeout_sec=2.0)
    caller.destroy_node()
    rclpy.shutdown()


def test_set_mode_maps_to_strict_switch(stack):
    cm, caller = stack
    cm.log.clear()
    assert caller.set_mode("trajectory").success
    assert cm.log[0][:3] == ("switch", ("arm_trajectory_controller",), ())
    assert caller.get_status().active_controllers == ["arm_trajectory_controller"]
    assert caller.set_mode("stream").success
    assert cm.log[-1][:3] == ("switch", ("arm_position_controller",), ("arm_trajectory_controller",))
    assert caller.set_mode("idle").success
    assert cm.active == set()


def test_invalid_transition_is_rejected_without_switch(stack):
    cm, caller = stack
    caller.set_mode("idle")
    cm.log.clear()
    response = caller.set_mode("nope")
    assert not response.success and "not declared" in response.message
    assert response.valid_transitions == ["stream", "trajectory"]
    assert cm.log == []


def test_hold_stop_orders_cancel_then_idle_and_latches(stack):
    cm, caller = stack
    assert caller.set_mode("trajectory").success
    cm.log.clear()
    response = caller.stop_with(C.STOP_HOLD)
    assert response.success, response.message
    kinds = [entry[0] for entry in cm.log]
    assert kinds == ["cancel", "switch"], f"[Stop / HOLD ordering] observed {kinds}"
    assert cm.log[1][2] == ("arm_trajectory_controller",), "idle switch must deactivate the trajectory controller"
    assert 0 <= response.cancel_latency_s <= response.idle_latency_s
    assert response.torque_off_latency_s == -1.0
    status = caller.get_status()
    assert status.lifecycle == C.LIFECYCLE_STOPPED and status.stop_latched and status.active_mode == "idle"
    rejected = caller.set_mode("stream")
    assert not rejected.success and rejected.valid_transitions == ["idle"]
    assert caller.set_mode("idle").success
    assert not caller.get_status().stop_latched


def test_torque_off_stop_deactivates_hardware_and_idle_reactivates(stack):
    cm, caller = stack
    assert caller.set_mode("stream").success
    cm.log.clear()
    response = caller.stop_with(C.STOP_TORQUE_OFF)
    assert response.success, response.message
    kinds = [entry[0] for entry in cm.log]
    assert kinds == ["switch", "hardware"], f"[Stop / TORQUE_OFF ordering] observed {kinds}"
    assert cm.log[1][1:3] == ("so101_system", "inactive")
    assert response.torque_off_latency_s >= response.idle_latency_s >= response.cancel_latency_s >= 0
    assert cm.hardware["so101_system"] == "inactive"
    cm.log.clear()
    assert caller.set_mode("idle").success
    assert cm.log[0][:3] == ("hardware", "so101_system", "active"), (
        "clearing a TORQUE_OFF stop must reactivate hardware first"
    )
    assert cm.hardware["so101_system"] == "active"
    assert caller.get_status().lifecycle == C.LIFECYCLE_ACTIVE


def test_unknown_stop_policy_rejected(stack):
    _cm, caller = stack
    response = caller.stop_with("EXPLODE")
    assert not response.success and "unknown stop policy" in response.message
    assert not caller.get_status().stop_latched


@pytest.mark.parametrize(
    "code, success",
    [
        (CancelGoal.Response.ERROR_NONE, True),
        (CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID, True),
        (CancelGoal.Response.ERROR_REJECTED, False),
        (CancelGoal.Response.ERROR_GOAL_TERMINATED, False),
    ],
)
def test_stop_checks_cancel_return_code(stack, code, success):
    cm, caller = stack
    assert caller.set_mode("idle").success
    assert caller.set_mode("trajectory").success
    cm.cancel_code = code
    try:
        response = caller.stop_with(C.STOP_HOLD)
        assert response.success == success, response.message
        status = caller.get_status()
        assert status.stop_latched
        assert status.lifecycle == (C.LIFECYCLE_STOPPED if success else C.LIFECYCLE_FAULTED)
        if not success:
            assert f"return_code={code}" in response.message
            assert any(f"return_code={code}" in fault for fault in status.faults)
    finally:
        cm.cancel_code = CancelGoal.Response.ERROR_NONE
        assert caller.set_mode("idle").success


@pytest.mark.parametrize("mode", ["idle", "stream", "trajectory"])
def test_unreachable_cancel_service_only_fails_for_active_trajectory(stack, mode):
    cm, caller = stack
    assert caller.set_mode("idle").success
    assert caller.set_mode(mode).success
    cm.destroy_service(cm.cancel_service)
    try:
        response = caller.stop_with(C.STOP_HOLD)
        assert response.success == (mode != "trajectory"), response.message
        status = caller.get_status()
        assert status.stop_latched
        if mode == "trajectory":
            assert "unreachable" in response.message and ACTION in response.message
            assert status.lifecycle == C.LIFECYCLE_FAULTED
        else:
            assert status.lifecycle == C.LIFECYCLE_STOPPED
    finally:
        cm.cancel_service = cm.create_service(
            CancelGoal, f"{ACTION}/_action/cancel_goal", cm._cancel, callback_group=ReentrantCallbackGroup()
        )
        assert caller.set_mode("idle").success


def test_stop_verifies_controller_deactivation(stack):
    cm, caller = stack
    assert caller.set_mode("stream").success
    cm.ignore_deactivation = True
    try:
        response = caller.stop_with(C.STOP_HOLD)
        assert not response.success
        assert "command controllers still active" in response.message
        status = caller.get_status()
        assert status.stop_latched and status.lifecycle == C.LIFECYCLE_FAULTED
        assert status.active_controllers == ["arm_position_controller"]
    finally:
        cm.ignore_deactivation = False
        cm.active.clear()
        assert caller.set_mode("idle").success


@pytest.mark.parametrize("policy", [C.STOP_HOLD, C.STOP_TORQUE_OFF])
def test_stop_fences_in_flight_mode_switch(stack, policy):
    cm, caller = stack
    assert caller.set_mode("idle").success
    cm.switch_entered.clear()
    cm.switch_release.clear()
    pending = caller.mode.call_async(SetRuntimeMode.Request(mode="stream"))
    try:
        assert cm.switch_entered.wait(3.0), "mode switch never reached controller_manager"
        stopped = caller.stop_with(policy)
        assert stopped.success, stopped.message
        assert not cm.switch_release.is_set(), "stop must finish without waiting for the mode switch"
    finally:
        cm.switch_release.set()
    deadline = time.monotonic() + 5.0
    while not pending.done() and time.monotonic() < deadline:
        rclpy.spin_once(caller, timeout_sec=0.05)
    assert pending.done(), "fenced mode switch never returned"
    response = pending.result()
    assert not response.success and response.valid_transitions == [C.IDLE_MODE]
    status = caller.get_status()
    assert status.stop_latched and status.lifecycle == C.LIFECYCLE_STOPPED
    assert status.active_mode == C.IDLE_MODE and not status.active_controllers
    assert not cm.active
    assert any("stop engaged during the controller switch" in fault for fault in status.faults)
    assert caller.set_mode("idle").success

"""Facade tests for the D3.1 per-platform controller override (MuJoCo stack).

A separate module from ``test_facade``: both stacks run real ROS nodes with
the same absolute service names, so they must not share a DDS domain — this
module's fixtures reserve their own domain and initialize rclpy for the
module's lifetime, after ``test_facade``'s module has torn itself down.

The profile under test is the LeKiwi-shaped declaration: on MuJoCo the base
controller is never released (a MuJoCo velocity actuator keeps its last
target after its controller is deactivated), so ``idle`` carries it through
``controllers_by_platform`` and the base node is the gate on that platform.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

import pytest
import rclpy
import yaml
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from ibrobot_msgs.srv import GetRuntimeStatus, SetRuntimeMode, StopRuntime
from robot_runtime import contract as C

PLATFORM_PROFILE = {
    "runtime": {"name": "facade_platform_test", "version": "0.0.1"},
    "controller_manager": "controller_manager",
    "use_sim": True,
    "sim_platforms": ["sdk", "mujoco"],
    "sim_platform": "mujoco",
    "modes": {
        "initial": "idle",
        "idle": {
            "controllers": [],
            "controllers_by_platform": {"mujoco": ["base_velocity_controller"]},
            "transitions": ["base_navigation"],
        },
        "base_navigation": {
            "controllers": ["base_velocity_controller"],
            "allows_base": True,
            "transitions": ["idle"],
        },
    },
    "capabilities": {
        "joint.state": {"joint_count": 3, "rate_hz": 100.0},
        "runtime.stop": {"cancel_bound_s": 2.0, "idle_bound_s": 3.0, "torque_off_bound_s": 4.0},
    },
    "joints": ["7", "8", "9"],
}


class FakeControllerManager(Node):
    """The fake the mujoco facade talks to: records switches, lists state."""

    def __init__(self):
        super().__init__("controller_manager")
        cb = ReentrantCallbackGroup()
        self.active: set[str] = {"base_velocity_controller"}  # what the spawners start
        self.log: list[tuple] = []
        self.create_service(SwitchController, "controller_manager/switch_controller", self._switch, callback_group=cb)
        self.create_service(ListControllers, "controller_manager/list_controllers", self._list, callback_group=cb)

    def _switch(self, request, response):
        self.log.append(
            ("switch", tuple(request.activate_controllers), tuple(request.deactivate_controllers), time.monotonic())
        )
        self.active -= set(request.deactivate_controllers)
        self.active |= set(request.activate_controllers)
        response.ok = True
        return response

    def _list(self, _request, response):
        for name in ("base_velocity_controller",):
            state = ControllerState()
            state.name = name
            state.state = "active" if name in self.active else "inactive"
            response.controller.append(state)
        return response


class Caller(Node):
    def __init__(self):
        super().__init__("facade_platform_test_caller")
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
def platform_stack(tmp_path_factory):
    previous_domain = os.environ.get("ROS_DOMAIN_ID")
    reservation = contextlib.ExitStack()
    try:
        from domain_coordinator import domain_id as reserve_domain_id
    except ImportError:
        pass
    else:
        os.environ["ROS_DOMAIN_ID"] = str(reservation.enter_context(reserve_domain_id()))
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    profile_path = tmp_path_factory.mktemp("profile_mujoco") / "runtime.yaml"
    profile_path.write_text(yaml.safe_dump(PLATFORM_PROFILE), encoding="utf-8")
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
        "the mujoco facade never reached ACTIVE with its idle controller active"
    )
    yield cm, caller
    executor.shutdown(timeout_sec=2.0)
    caller.destroy_node()
    rclpy.shutdown()
    reservation.close()
    if previous_domain is None:
        os.environ.pop("ROS_DOMAIN_ID", None)
    else:
        os.environ["ROS_DOMAIN_ID"] = previous_domain


def test_platform_mode_switch_never_releases_the_override_controller(platform_stack):
    cm, caller = platform_stack
    cm.log.clear()
    assert caller.set_mode("base_navigation").success
    # both modes resolve to the same controller set: the switch commands
    # nothing (the controller was spawned active and stays that way)
    assert all(entry[0] != "switch" or (not entry[1] and not entry[2]) for entry in cm.log)
    assert cm.active == {"base_velocity_controller"}
    assert caller.set_mode("idle").success
    assert cm.active == {"base_velocity_controller"}, "idle on mujoco keeps the base controller active"
    # the health check agrees: idle's expected set is active, lifecycle stays ACTIVE
    assert caller.get_status().lifecycle == C.LIFECYCLE_ACTIVE
    assert caller.get_status().active_controllers == ["base_velocity_controller"]


def test_platform_stop_treats_the_override_controller_as_non_command(platform_stack):
    cm, caller = platform_stack
    cm.log.clear()
    assert caller.set_mode("base_navigation").success
    stopped = caller.stop_with(C.STOP_HOLD)
    assert stopped.success, stopped.message
    # the stop switched back to idle — which on mujoco keeps the controller;
    # the command-controller check must not flag it
    assert cm.active == {"base_velocity_controller"}
    assert caller.get_status().stop_latched and caller.get_status().lifecycle == C.LIFECYCLE_STOPPED
    # re-entering the base mode does not resume motion: no switch happened
    assert not any(entry[0] == "switch" and (entry[1] or entry[2]) for entry in cm.log)
    assert caller.set_mode("idle").success
    assert caller.get_status().lifecycle == C.LIFECYCLE_ACTIVE


# ---------------------------------------------------------------------------
# D5.4: RuntimeStatus capabilities from the platform-filtered inventory
# ---------------------------------------------------------------------------

LIDAR_PERIPHERALS = [
    {
        "type": "lidar",
        "name": "mid360",
        "driver": "livox_mid360",
        "xfer_format": 1,
        "frame_id": "livox_frame",
        "pointcloud_topic": "/livox/lidar",
        "imu_topic": "/livox/imu",
    },
    {
        "type": "lidar",
        "name": "ld06",
        "driver": "ldlidar",
        "scan_topic": "/ld06/scan",
        "frame_id": "ld06_frame",
    },
]


def _facade_capabilities(profile: dict, tmp_path):
    """Construct the real facade for ``profile`` and read its capability set.

    The capabilities the RuntimeStatus will carry are fixed in ``__init__``
    (platform resolution then the platform-filtered projection), so the node
    is constructed and inspected directly — no service graph needed.
    """
    if not rclpy.ok():
        rclpy.init()
    from robot_runtime.facade_node import RuntimeFacade

    profile_path = tmp_path / "runtime.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    facade = RuntimeFacade(profile_path=str(profile_path))
    try:
        return facade._state.capabilities
    finally:
        facade.destroy_node()


def test_mujoco_capabilities_match_the_platform_filtered_description(tmp_path):
    from robot_runtime.interface_description import build_description

    controllers = tmp_path / "controllers.yaml"
    controllers.write_text("controller_manager:\n  ros__parameters:\n    update_rate: 100\n", encoding="utf-8")
    profile = {
        **PLATFORM_PROFILE,
        "ros2_control": {"controllers_config": str(controllers)},
        "peripherals": LIDAR_PERIPHERALS,
    }
    capabilities = _facade_capabilities(profile, tmp_path)
    # the MID-360's endpoints only: the ldlidar is not simulated on MuJoCo
    assert capabilities["perception.lidar"]["topics"] == ["/livox/lidar", "/livox/imu"]
    # the same projection the interface description builds from the same data
    interfaces = build_description(
        {
            "runtime": {"name": "facade_platform_test", "version": "0.0.1"},
            "capabilities": {},
            "joints": ["7", "8", "9"],
            "use_sim": True,
            "sim_platform": "mujoco",
            "peripherals": LIDAR_PERIPHERALS,
            "ros2_control": {"controllers_config": str(controllers)},
        },
        use_sim=True,
        sim_platform="mujoco",
    )["interfaces"]
    lidar_endpoints = sorted(
        spec["endpoint"] for spec in interfaces.values() if spec.get("capability") == "perception.lidar"
    )
    assert lidar_endpoints == sorted(capabilities["perception.lidar"]["topics"])
    # the unsupported driver appears nowhere in the description
    assert not any("ld06" in key for key in interfaces)


def test_an_unsupported_only_lidar_claims_nothing_on_mujoco(tmp_path):
    profile = {
        **PLATFORM_PROFILE,
        "peripherals": [LIDAR_PERIPHERALS[1]],  # the ldlidar only
    }
    capabilities = _facade_capabilities(profile, tmp_path)
    assert "perception.lidar" not in capabilities


def test_gazebo_and_sdk_capabilities_are_unchanged(tmp_path):
    profile = {
        **PLATFORM_PROFILE,
        "sim_platforms": ["sdk", "gazebo", "mujoco"],
        "peripherals": LIDAR_PERIPHERALS,
    }
    for platform in ("gazebo", "sdk"):
        selected = {**profile, "sim_platform": platform}
        capabilities = _facade_capabilities(selected, tmp_path)
        assert set(capabilities["perception.lidar"]["topics"]) == {"/livox/lidar", "/livox/imu", "/ld06/scan"}, platform

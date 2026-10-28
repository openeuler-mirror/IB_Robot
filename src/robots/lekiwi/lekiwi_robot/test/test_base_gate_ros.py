# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mode, navigation gate and stop latch, through the running base node.

``test_base_gating.py`` covers the admission predicate directly. This drives
the real chain instead -- ``RuntimeStatus`` in, the gate service, ``/cmd_vel``
in, wheel commands out -- so a wiring mistake between them (a callback that
forgets the gate, a service that does not stop the base) is caught too.

It runs a live ROS graph in-process on a private ``ROS_DOMAIN_ID``, so a
runtime left running by another test or worktree cannot answer for it.
"""

import os
import threading
import time

import pytest
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import RuntimeStatus
from lekiwi_robot.base_node import LeKiwiBaseNode
from robot_runtime import contract as C

WHEELS = "/base_velocity_controller/commands"
PARAMS = {
    "wheel_joints": "['7','8','9']",
    "wheel_radius": "0.05",
    "base_radius": "0.125",
    "max_wheel_radps": "4.602",
    "control_frequency": "50.0",
    "staleness_s": "0.5",
    # the published cmd_vel envelope (D3.2): no defaults, the node fails
    # closed without it
    "max_vx": "0.26",
    "max_vy": "0.23",
    "max_wz": "1.5",
    "odom_frame": "odom",
    "base_frame": "base_link",
    "publish_tf": "false",
}


class Driver(Node):
    def __init__(self):
        super().__init__("lekiwi_base_gate_driver")
        self.wheels: list[tuple[float, tuple[float, ...]]] = []
        self.acks: list[bool] = []
        self._status = self.create_publisher(RuntimeStatus, C.STATUS_TOPIC, 10)
        self._cmd = self.create_publisher(Twist, C.CMD_VEL_TOPIC, 10)
        self.create_subscription(Float64MultiArray, WHEELS, self._on_wheels, 50)
        self.create_subscription(Bool, C.NAVIGATION_ACK_TOPIC, lambda m: self.acks.append(bool(m.data)), 10)
        self._gate = self.create_client(SetBool, C.NAVIGATION_ENABLE_SERVICE)

    def _on_wheels(self, msg):
        self.wheels.append((time.monotonic(), tuple(msg.data)))

    def status(self, mode: str, stop_latched: bool = False) -> None:
        self._status.publish(RuntimeStatus(active_mode=mode, stop_latched=stop_latched))

    def gate(self, enabled: bool) -> bool:
        assert self._gate.wait_for_service(timeout_sec=5.0)
        future = self._gate.call_async(SetBool.Request(data=enabled))
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert future.done(), "navigation gate service did not answer"
        return future.result().success

    def drive_for(self, seconds: float, vx: float = 0.2) -> bool:
        """Stream /cmd_vel; True when any non-zero wheel command came out."""
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            msg = Twist()
            msg.linear.x = vx
            self._cmd.publish(msg)
            time.sleep(0.02)
        time.sleep(0.1)
        return any(t >= start and any(v != 0.0 for v in data) for t, data in self.wheels)


@pytest.fixture(scope="module")
def graph():
    # The workspace-root conftest reserves a per-process ROS domain
    # (IBROBOT_TEST_ROS_DOMAIN_ID) before any fixture runs; this module
    # must NOT roll its own unreserved random domain — a leftover value
    # would leak into every later test in the process (review round 2).
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), (
        "the workspace-root conftest's ROS domain isolation is not active "
        "(IBROBOT_TEST_ROS_DOMAIN_ID unset); refusing to run on an unreserved domain"
    )
    args = ["--ros-args", *[a for k, v in PARAMS.items() for a in ("-p", f"{k}:={v}")]]
    rclpy.init(args=args)
    base = None
    driver = None
    executor = None
    spinner = None
    try:
        base = LeKiwiBaseNode()
        driver = Driver()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(base)
        executor.add_node(driver)
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        time.sleep(0.5)  # discovery
        yield driver
    finally:
        if executor is not None:
            executor.shutdown()
        if spinner is not None:
            spinner.join(timeout=5.0)
        if base is not None:
            base.destroy_node()
        if driver is not None:
            driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def settle(driver: Driver, mode: str, stop_latched: bool = False) -> None:
    for _ in range(5):
        driver.status(mode, stop_latched)
        time.sleep(0.02)


def test_gate_is_closed_on_startup(graph):
    settle(graph, "base_navigation")
    assert not graph.drive_for(0.3), "the base moved before the navigation gate was opened"


def test_mode_active_gate_closed_does_not_move(graph):
    settle(graph, "base_navigation")
    assert graph.gate(False)
    assert not graph.drive_for(0.3)


def test_gate_open_mode_inactive_does_not_move(graph):
    settle(graph, "teleop")
    assert graph.gate(True)
    assert not graph.drive_for(0.3)


def test_gate_closed_mode_inactive_does_not_move(graph):
    settle(graph, "teleop")
    assert graph.gate(False)
    assert not graph.drive_for(0.3)


def test_mode_active_gate_open_moves(graph):
    settle(graph, "base_navigation")
    assert graph.gate(True)
    assert graph.acks and graph.acks[-1] is True
    assert graph.drive_for(0.3), "mode active and gate open, but no wheel command reached the controller"


def test_stop_latch_overrides_an_open_gate(graph):
    settle(graph, "base_navigation")
    assert graph.gate(True)
    assert graph.drive_for(0.2)
    settle(graph, "base_navigation", stop_latched=True)
    # Re-opening the gate while latched must not re-admit commands: the latch
    # overrides the gate, never the reverse.
    graph.gate(True)
    assert not graph.drive_for(0.3), "the base moved while the runtime stop was latched"
    assert graph.wheels[-1][1] == (0.0, 0.0, 0.0), "the last wheel command after the latch is not zero"

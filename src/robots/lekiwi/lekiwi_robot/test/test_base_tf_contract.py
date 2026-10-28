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

"""The base node's TF output contract: exactly one odom -> base_link transform.

Pins the ownership-elimination leg of robot_config's navigation e2e
(lekiwi-mujoco-runtime-platform D15, review round 1): that test attributes
AMCL as the SOLE publisher of map -> odom by eliminating the other two /tf
publishers — robot_state_publisher (whose frames are the URDF's joint
pairs, checked there against the live /robot_description) and this base
node, whose TF sender must emit exactly one odom -> base_link transform per
odometry update and nothing else. Without this pin the elimination leaned on
untested source-reading.

It runs a live ROS graph in-process on a private ``ROS_DOMAIN_ID``, so a
runtime left running by another test or worktree cannot answer for it
(the ``test_base_gate_ros.py`` harness pattern).
"""

import os
import threading
import time

import pytest
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from ibrobot_msgs.msg import RuntimeStatus
from lekiwi_robot.base_node import LeKiwiBaseNode
from robot_runtime import contract as C

PARAMS = {
    "wheel_joints": "['7','8','9']",
    "wheel_radius": "0.05",
    "base_radius": "0.125",
    "max_wheel_radps": "4.602",
    "control_frequency": "50.0",
    "staleness_s": "0.5",
    "max_vx": "0.26",
    "max_vy": "0.23",
    "max_wz": "1.5",
    "odom_frame": "odom",
    "base_frame": "base_link",
    "publish_tf": "true",  # the TF contract under test
}


class Driver(Node):
    def __init__(self):
        super().__init__("lekiwi_base_tf_driver")
        self.tf: list[TFMessage] = []
        self._status = self.create_publisher(RuntimeStatus, C.STATUS_TOPIC, 10)
        self.create_subscription(TFMessage, "/tf", self.tf.append, 50)
        self.joints = self.create_publisher(JointState, "/joint_states", 50)

    def status(self, mode: str) -> None:
        self._status.publish(RuntimeStatus(active_mode=mode))

    def drive_wheel_feedback(self, seconds: float, wheel_radps: float = 2.0) -> None:
        """Stream advancing wheel positions so odometry integrates and TF flows."""
        positions = [0.0, 0.0, 0.0]
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            now = self.get_clock().now().seconds_nanoseconds()
            message = JointState()
            message.header.stamp = TimeMsg(sec=now[0], nanosec=now[1])
            message.name = ["7", "8", "9"]
            positions = [p + wheel_radps * 0.02 for p in positions]
            message.position = list(positions)
            message.velocity = [wheel_radps, wheel_radps, wheel_radps]
            self.joints.publish(message)
            self.status("base_navigation")
            time.sleep(0.02)


@pytest.fixture(scope="module")
def graph():
    # The workspace-root conftest reserves a per-process ROS domain
    # (IBROBOT_TEST_ROS_DOMAIN_ID) before any fixture runs; this module
    # must NOT roll its own unreserved random domain — a leftover value
    # would leak into every later test in the process, and the value is
    # never restored (review round 2, same defect as test_base_gate_ros).
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


def test_the_base_node_publishes_exactly_one_odom_to_base_link_transform(graph):
    """Every /tf message from the base node carries exactly one transform,
    ``odom`` -> ``base_link`` (the configured frames), one per odometry
    update — never a second pair such as map -> odom."""
    graph.tf.clear()
    graph.drive_wheel_feedback(1.0)
    assert graph.tf, "the base node published no /tf at all under wheel feedback"
    for message in graph.tf:
        assert len(message.transforms) == 1, (
            f"a /tf message carried {len(message.transforms)} transforms; the base node sends exactly one per update"
        )
        transform = message.transforms[0]
        assert transform.header.frame_id == "odom", f"unexpected parent frame {transform.header.frame_id!r}"
        assert transform.child_frame_id == "base_link", f"unexpected child frame {transform.child_frame_id!r}"

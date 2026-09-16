"""Parity of the runtime motion services with the planning framework they wrap.

robot-motion-services, "Recorded request replay matches": the pre-migration
pick pipeline called MoveIt's /compute_ik and /compute_fk directly. Replaying
a fixed set of requests against both the runtime-neutral services
(/motion/compute_ik, /motion/compute_fk) and the framework services in the
same launch graph must give the same solutions and poses.

Runs against so101_robot launched through its own entry in simulated
transport (like the conformance suite). Set MOTION_PARITY_LAUNCH to override
the launch command, or MOTION_PARITY_SKIP=1 to skip.
"""

from __future__ import annotations

import math
import os
import shlex
import signal
import subprocess
import time

import pytest

pytest.importorskip("moveit_msgs")
rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import PoseStamped  # noqa: E402
from moveit_msgs.srv import GetPositionFK, GetPositionIK  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from ibrobot_msgs.srv import ComputeFk, ComputeIk  # noqa: E402
from robot_runtime import contract as C  # noqa: E402

LAUNCH = os.environ.get(
    "MOTION_PARITY_LAUNCH", "ros2 launch so101_robot runtime.launch.py profile:=so101_single_arm simulated:=true"
)
ARM_JOINTS = ["1", "2", "3", "4", "5"]
EE_LINK = "gripper"
BASE_FRAME = "base"

# Recorded joint configurations from the pick pipeline's typical envelope.
RECORDED_CONFIGURATIONS = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, -1.2, 1.3, 0.6, 0.0],
    [-0.4, -0.9, 1.0, 0.9, 0.3],
    [0.6, -1.5, 1.5, 0.8, -0.5],
    [0.0, -0.6, 0.8, 1.2, 0.8],
]

pytestmark = pytest.mark.skipif(os.environ.get("MOTION_PARITY_SKIP") == "1", reason="MOTION_PARITY_SKIP=1")


class Caller(Node):
    def __init__(self):
        super().__init__("motion_parity_caller")
        self.runtime_fk = self.create_client(ComputeFk, C.COMPUTE_FK_SERVICE)
        self.runtime_ik = self.create_client(ComputeIk, C.COMPUTE_IK_SERVICE)
        self.moveit_fk = self.create_client(GetPositionFK, "/compute_fk")
        self.moveit_ik = self.create_client(GetPositionIK, "/compute_ik")

    def call(self, client, request, timeout: float = 10.0):
        assert client.wait_for_service(timeout_sec=30.0), f"{client.srv_name} unavailable"
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        assert future.done(), f"{client.srv_name} timed out"
        return future.result()


def _joint_state(positions):
    js = JointState()
    js.name = list(ARM_JOINTS)
    js.position = [float(p) for p in positions]
    return js


@pytest.fixture(scope="module")
def caller():
    os.environ.setdefault("ROS_DOMAIN_ID", "52")
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")
    process = subprocess.Popen(shlex.split(LAUNCH), start_new_session=True)
    rclpy.init()
    node = Caller()
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and not (
            node.runtime_ik.wait_for_service(timeout_sec=1.0) and node.moveit_ik.wait_for_service(timeout_sec=1.0)
        ):
            pass
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def _moveit_fk(caller: Caller, positions):
    request = GetPositionFK.Request()
    request.header.frame_id = BASE_FRAME
    request.fk_link_names = [EE_LINK]
    request.robot_state.joint_state = _joint_state(positions)
    response = caller.call(caller.moveit_fk, request)
    assert int(response.error_code.val) == 1, f"MoveIt FK failed: {response.error_code.val}"
    return response.pose_stamped[0].pose


def _runtime_fk(caller: Caller, positions):
    request = ComputeFk.Request()
    request.joint_state = _joint_state(positions)
    request.link_names = [EE_LINK]
    response = caller.call(caller.runtime_fk, request)
    assert response.success, response.message
    return response.poses[0].pose


def _pose_distance(a, b) -> float:
    return math.sqrt(
        (a.position.x - b.position.x) ** 2 + (a.position.y - b.position.y) ** 2 + (a.position.z - b.position.z) ** 2
    )


@pytest.mark.parametrize("positions", RECORDED_CONFIGURATIONS)
def test_fk_replay_matches_framework(caller: Caller, positions):
    runtime = _runtime_fk(caller, positions)
    framework = _moveit_fk(caller, positions)
    assert _pose_distance(runtime, framework) < 1e-9, (
        f"[FK replay] {positions}: runtime {runtime.position} vs framework {framework.position}"
    )
    dot = abs(
        runtime.orientation.x * framework.orientation.x
        + runtime.orientation.y * framework.orientation.y
        + runtime.orientation.z * framework.orientation.z
        + runtime.orientation.w * framework.orientation.w
    )
    assert dot > 1.0 - 1e-9, f"[FK replay] orientation mismatch for {positions}"


@pytest.mark.parametrize("positions", RECORDED_CONFIGURATIONS[1:])
def test_ik_replay_matches_framework(caller: Caller, positions):
    """Seeded IK on a pose the arm can reach (its own FK): both paths must land on the same solution."""
    target = PoseStamped()
    target.header.frame_id = BASE_FRAME
    target.pose = _moveit_fk(caller, positions)
    seed = [p + 0.05 for p in positions]

    moveit_request = GetPositionIK.Request()
    moveit_request.ik_request.group_name = "arm"
    moveit_request.ik_request.ik_link_name = EE_LINK
    moveit_request.ik_request.pose_stamped = target
    moveit_request.ik_request.robot_state.joint_state = _joint_state(seed)
    moveit_request.ik_request.timeout.nanosec = int(0.2 * 1e9)
    framework = caller.call(caller.moveit_ik, moveit_request)
    assert int(framework.error_code.val) == 1, f"MoveIt IK failed: {framework.error_code.val}"
    framework_solution = dict(
        zip(framework.solution.joint_state.name, framework.solution.joint_state.position, strict=False)
    )

    runtime_request = ComputeIk.Request()
    runtime_request.target = target
    runtime_request.seed = _joint_state(seed)
    runtime_request.orientation_tolerance = math.pi  # position-priority, like the pipeline
    runtime_request.timeout = 0.2
    runtime = caller.call(caller.runtime_ik, runtime_request)
    assert runtime.success, f"[IK replay] runtime failed: {runtime.code} {runtime.message}"
    runtime_solution = dict(zip(runtime.solution.name, runtime.solution.position, strict=False))

    # The seeded LMA solve is deterministic for the same seed and target; the
    # runtime must not alter the framework's solution. Compare through FK as
    # well so an equivalent branch would still be caught as a difference.
    for joint in ARM_JOINTS:
        assert abs(runtime_solution[joint] - framework_solution[joint]) < 1e-6, (
            f"[IK replay] joint {joint}: runtime {runtime_solution[joint]:.6f} vs framework {framework_solution[joint]:.6f}"
        )
    assert _pose_distance(_runtime_fk(caller, [runtime_solution[j] for j in ARM_JOINTS]), target.pose) < 2e-3

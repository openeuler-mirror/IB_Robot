"""Application E2E for the migrated LeKiwi configurations (task 5.1).

binding -> mode -> command against the real LeKiwi runtime in simulated
transport, per configuration profile:

1. the runtime reaches its ready lifecycle (the bound description's surface);
2. with the ``base_navigation`` mode active but the gate still closed, a
   Nav2-style ``/cmd_vel`` stream produces **no** wheel motion;
3. the real launch-time admission client (``robot_navigation/
   navigation_admission``) opens the gate through the contract service;
4. the same ``/cmd_vel`` stream now drives the wheels with the recorded
   projection (the collision monitor's output channel; the monitor itself is
   pinned by ``test_navigation_velocity_chain.py``);
5. stopping the stream zeroes the wheels within the runtime's staleness bound
   (the deadman-release robot-side condition, task 4.2);
6. for the mobile-manipulator profile, a motion service call (the bound
   ``motion.*`` surface) answers through the runtime contract.

Skipped when ``lekiwi_robot`` is not installed (e.g. core-only builds).
"""

import subprocess
import threading
import time

import pytest
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import SetBool

from ibrobot_msgs.msg import RuntimeStatus
from ibrobot_msgs.srv import ComputeFk, SetRuntimeMode, StopRuntime
from robot_runtime import contract as C

try:
    from ament_index_python.packages import PackageNotFoundError, get_package_prefix

    LEKIWI_INSTALLED = True
    try:
        get_package_prefix("lekiwi_robot")
    except PackageNotFoundError:
        LEKIWI_INSTALLED = False
except ImportError:
    LEKIWI_INSTALLED = False

pytestmark = pytest.mark.skipif(not LEKIWI_INSTALLED, reason="lekiwi_robot is not installed")

STARTUP_TIMEOUT_S = 120.0
STALENESS_MARGIN_S = 1.5
WHEEL_COMMAND_TOPIC = "/base_velocity_controller/commands"


class ApplicationHarness(Node):
    def __init__(self):
        super().__init__("lekiwi_application_e2e")
        self.status: RuntimeStatus | None = None
        self.wheel_commands: list[Float64MultiArray] = []
        self.joint_states: list[JointState] = []
        self.acks: list[bool] = []
        reliable = QoSProfile(depth=50)
        self.create_subscription(RuntimeStatus, C.STATUS_TOPIC, self._on_status, reliable)
        self.create_subscription(Float64MultiArray, WHEEL_COMMAND_TOPIC, self.wheel_commands.append, reliable)
        self.create_subscription(JointState, "/joint_states", self.joint_states.append, reliable)
        self.create_subscription(Bool, C.NAVIGATION_ACK_TOPIC, lambda m: self.acks.append(bool(m.data)), reliable)
        self.mode_client = self.create_client(SetRuntimeMode, C.SET_MODE_SERVICE)
        self.gate_client = self.create_client(SetBool, C.NAVIGATION_ENABLE_SERVICE)
        self.fk_client = self.create_client(ComputeFk, C.COMPUTE_FK_SERVICE)
        self.stop_client = self.create_client(StopRuntime, C.STOP_SERVICE)

    def _on_status(self, msg):
        self.status = msg

    def spin_until(self, predicate, timeout_s: float, what: str):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError(f"E2E timed out waiting for {what}")

    def wheel_velocity_magnitude(self) -> float:
        recent = [msg for msg in self.joint_states if set(msg.name) >= {"7", "8", "9"}][-5:]
        if not recent:
            return 0.0
        peak = 0.0
        for msg in recent:
            for name, velocity in zip(msg.name, msg.velocity, strict=False):
                if name in {"7", "8", "9"}:
                    peak = max(peak, abs(float(velocity)))
        return peak

    def last_wheel_command(self) -> list[float] | None:
        return list(self.wheel_commands[-1].data) if self.wheel_commands else None


@pytest.fixture(scope="module")
def rclpy_ctx():
    rclpy.init()
    yield
    rclpy.shutdown()


def _publish_cmd_vel(harness: ApplicationHarness, vx: float, vy: float, wz: float, duration_s: float):
    pub = harness.create_publisher(Twist, C.CMD_VEL_TOPIC, 10)
    time.sleep(0.3)  # discovery
    twist = Twist()
    twist.linear.x = vx
    twist.linear.y = vy
    twist.angular.z = wz
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        pub.publish(twist)
        rclpy.spin_once(harness, timeout_sec=0.05)
    harness.destroy_publisher(pub)


@pytest.mark.parametrize("profile", ["lekiwi_lidar", "lekiwi_nav_grasp"])
def test_application_admission_and_command_e2e(profile, rclpy_ctx):
    launch = subprocess.Popen(
        [
            "ros2",
            "launch",
            "lekiwi_robot",
            "runtime.launch.py",
            f"profile:={profile}",
            "simulated:=true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    harness = ApplicationHarness()
    try:
        # 1. binding surface: the runtime reaches ACTIVE and declares the
        #    base channel (its capabilities are the bound description).
        harness.spin_until(
            lambda: harness.status is not None and harness.status.lifecycle == "ACTIVE",
            STARTUP_TIMEOUT_S,
            "the runtime to become ACTIVE",
        )
        assert "base.cmd_vel" in harness.status.capabilities

        # Mode first (the admission client's order), gate still closed.
        request = SetRuntimeMode.Request()
        request.mode = "base_navigation"
        future = harness.mode_client.call_async(request)
        harness.spin_until(lambda: future.done(), 10.0, "the mode response")
        assert future.result().success, future.result()
        harness.spin_until(
            lambda: harness.status is not None and harness.status.active_mode == "base_navigation",
            10.0,
            "base_navigation to be active",
        )

        # 2. gate closed: a Nav2-style command stream must not move the wheels.
        _publish_cmd_vel(harness, 0.1, 0.0, 0.0, 1.5)
        rclpy.spin_once(harness, timeout_sec=0.3)
        assert (harness.last_wheel_command() or [1.0]) == [0.0, 0.0, 0.0] or not harness.wheel_commands
        assert harness.wheel_velocity_magnitude() == pytest.approx(0.0, abs=1e-6)

        # 3. the real launch-time admission client opens the gate.
        admission = subprocess.run(
            ["ros2", "run", "robot_navigation", "navigation_admission"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert admission.returncode == 0, admission.stderr
        harness.spin_until(lambda: any(harness.acks), 5.0, "the gate acknowledgment")

        # 4. the same command stream now drives the wheels with the recorded
        #    projection (0.1 m/s forward -> wheels 7/9 at +-1.732 rad/s).
        harness.wheel_commands.clear()
        _publish_cmd_vel(harness, 0.1, 0.0, 0.0, 1.5)
        commands = [cmd for cmd in harness.wheel_commands if any(abs(v) > 1e-9 for v in cmd.data)]
        assert commands, "no non-zero wheel command while the gate is open"
        latest = commands[-1].data
        assert latest[0] == pytest.approx(-0.1 * 0.8660254 / 0.05, abs=5e-3), latest
        assert latest[1] == pytest.approx(0.0, abs=5e-3), latest
        assert latest[2] == pytest.approx(0.1 * 0.8660254 / 0.05, abs=5e-3), latest
        harness.spin_until(lambda: harness.wheel_velocity_magnitude() > 0.05, 5.0, "the simulated wheels to turn")

        # 5. deadman release: the stream stops, the runtime zeroes the wheels
        #    (robot-side condition, not publisher silence).
        harness.wheel_commands.clear()
        time.sleep(0.5)
        rclpy.spin_once(harness, timeout_sec=0.5)
        deadline = time.monotonic() + STALENESS_MARGIN_S + 2.0
        zeroed = False
        while time.monotonic() < deadline:
            rclpy.spin_once(harness, timeout_sec=0.2)
            last = harness.last_wheel_command()
            if last is not None and last == [0.0, 0.0, 0.0] and harness.wheel_velocity_magnitude() < 1e-6:
                zeroed = True
                break
        assert zeroed, "wheels did not zero after the command stream stopped"

        # 6. two producers on /cmd_vel (navigation still streaming after the
        #    operator lets go): only the runtime stop holds the base. The latch
        #    overrides the open gate while the autonomous stream continues.
        harness.wheel_commands.clear()
        streaming = threading.Event()
        streaming.set()
        autonomy = harness.create_publisher(Twist, C.CMD_VEL_TOPIC, 10)

        def _autonomous_stream():
            twist = Twist()
            twist.linear.x = 0.1
            while streaming.is_set():
                autonomy.publish(twist)
                time.sleep(0.05)

        producer = threading.Thread(target=_autonomous_stream, daemon=True)
        producer.start()
        try:
            harness.spin_until(
                lambda: any(any(abs(v) > 1e-9 for v in cmd.data) for cmd in harness.wheel_commands),
                5.0,
                "the autonomous stream to drive the wheels",
            )
            stop = StopRuntime.Request()
            stop.policy = StopRuntime.Request.HOLD
            future = harness.stop_client.call_async(stop)
            harness.spin_until(lambda: future.done(), 15.0, "the stop response")
            assert future.result().success, future.result()
            harness.spin_until(
                lambda: harness.status is not None and harness.status.stop_latched, 5.0, "the stop latch"
            )
            harness.wheel_commands.clear()
            time.sleep(1.0)
            rclpy.spin_once(harness, timeout_sec=0.2)
            assert all(list(cmd.data) == [0.0, 0.0, 0.0] for cmd in harness.wheel_commands), (
                "the base moved while the runtime stop was latched"
            )
        finally:
            streaming.clear()
            producer.join(timeout=2.0)
            harness.destroy_publisher(autonomy)

        # 7. bound motion surface (mobile-manipulator profile): FK answers
        #    through the runtime contract.
        if profile == "lekiwi_nav_grasp":
            request = ComputeFk.Request()
            request.link_names = ["gripper"]
            request.joint_state.name = ["1", "2", "3", "4", "5", "6"]
            request.joint_state.position = [0.0, -1.5854, 1.5708, 0.7854, 0.0, 0.5]
            future = harness.fk_client.call_async(request)
            harness.spin_until(lambda: future.done(), 15.0, "the FK response")
            response = future.result()
            assert response is not None and response.success, response
            assert len(response.poses) == 1, response
    finally:
        harness.destroy_node()
        launch.send_signal(subprocess.signal.SIGINT)
        try:
            launch.wait(timeout=30)
        except subprocess.TimeoutExpired:
            launch.kill()
            launch.wait(timeout=10)

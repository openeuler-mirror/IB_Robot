"""Integration tests: the X2 bridge against the in-repo vendor mock.

The mock speaks the real vendor IDL, so these exercise the whole wrapper —
mode machine, arbitration, command projection, stop semantics, telemetry and
the interaction surface — without a robot. They skip, with a named reason, when
the AimDK overlay is not sourced; that is the same rule the runtime
independence gate applies.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("aimdk_msgs", reason="AimDK overlay not sourced; source the vendor workspace to run these")

from aimdk_robot import projection  # noqa: E402
from aimdk_robot.runtime_node import AimdkRuntimeNode  # noqa: E402
from aimdk_robot.vendor_mock_node import AimdkVendorMock  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64MultiArray  # noqa: E402
from std_srvs.srv import SetBool  # noqa: E402

from ibrobot_msgs.action import ExecuteNamedMotion  # noqa: E402
from ibrobot_msgs.msg import PowerState  # noqa: E402
from ibrobot_msgs.srv import (  # noqa: E402
    GetRuntimeStatus,
    PlayExpression,
    SetLedPattern,
    SetRuntimeMode,
    SpeakText,
    StopRuntime,
)

PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "x2_ultra.yaml"
OMNIHAND_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "x2_ultra_omnihand.yaml"
TIMEOUT_S = 15.0


class Harness:
    """A running bridge + vendor mock pair, with client helpers."""

    def __init__(self, profile: Path = PROFILE, **mock_params) -> None:
        overrides = [rclpy.parameter.Parameter(name, value=value) for name, value in mock_params.items()]
        self.mock = AimdkVendorMock(parameter_overrides=overrides)
        self.runtime = AimdkRuntimeNode(
            parameter_overrides=[
                rclpy.parameter.Parameter("profile", value=str(profile)),
                rclpy.parameter.Parameter("simulated", value=True),
                rclpy.parameter.Parameter("startup_timeout_s", value=8.0),
            ]
        )
        self.node = rclpy.create_node("aimdk_test_client")
        self.executor = MultiThreadedExecutor()
        for node in (self.mock, self.runtime, self.node):
            self.executor.add_node(node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self.executor.shutdown(timeout_sec=2.0)
        for node in (self.node, self.runtime, self.mock):
            node.destroy_node()

    # --- helpers ------------------------------------------------------------

    def call(self, service_type, endpoint, request, timeout_s: float = TIMEOUT_S):
        client = self.node.create_client(service_type, endpoint)
        assert client.wait_for_service(timeout_sec=timeout_s), f"{endpoint} never appeared"
        future = client.call_async(request)
        deadline = time.time() + timeout_s
        while time.time() < deadline and not future.done():
            time.sleep(0.02)
        assert future.done(), f"{endpoint} did not answer"
        return future.result()

    def status(self):
        return self.call(GetRuntimeStatus, "/runtime/get_status", GetRuntimeStatus.Request()).status

    def wait_for(self, predicate, what: str, timeout_s: float = TIMEOUT_S):
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            last = predicate()
            if last:
                return last
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what} (last value: {last!r})")

    def wait_active(self):
        return self.wait_for(lambda: self.status().lifecycle == "ACTIVE", "the runtime to reach ACTIVE")

    def enable_navigation(self, enable: bool = True):
        request = SetBool.Request()
        request.data = enable
        return self.call(SetBool, "/motion_mode/set_navigation_enabled", request)

    def set_mode(self, mode: str):
        request = SetRuntimeMode.Request()
        request.mode = mode
        return self.call(SetRuntimeMode, "/runtime/set_mode", request)

    def platform_action(self) -> str:
        return self.mock.current_action()

    def publish(self, topic: str, message, message_type=Float64MultiArray, count: int = 15):
        publisher = self.node.create_publisher(message_type, topic, 10)
        for _ in range(count):
            publisher.publish(message)
            time.sleep(0.02)


def _sensor_qos():
    """Best-effort, matching what the profile declares for sensor streams."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


@pytest.fixture(scope="module", autouse=True)
def ros_context():
    """Own the ROS context only if nothing else in the session already does."""
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    yield
    if owned and rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def harness():
    running = Harness()
    try:
        yield running
    finally:
        running.shutdown()


# --- lifecycle --------------------------------------------------------------


def test_runtime_reaches_active_against_the_mock(harness):
    harness.wait_active()
    status = harness.status()
    assert status.runtime_name == "aimdk_robot"
    assert "runtime.stop" in status.capabilities
    assert "interaction.tts" in status.capabilities


def test_joint_states_follow_the_public_joint_order(harness):
    received: list[JointState] = []
    harness.node.create_subscription(JointState, "/joint_states", received.append, 10)
    harness.wait_for(lambda: received, "a /joint_states message")
    names = received[-1].name
    assert names[:2] == ["left_shoulder_pitch_joint", "left_shoulder_roll_joint"]
    assert "left_hand" in names
    # Vendor-controlled joints are not public joint-group members.
    assert "left_knee_joint" not in names
    assert "waist_yaw_joint" not in names


def test_body_state_carries_the_vendor_controlled_joints(harness):
    received: list[JointState] = []
    harness.node.create_subscription(JointState, "/aimdk/body_joint_states", received.append, 10)
    harness.wait_for(lambda: received, "a body state message")
    names = received[-1].name
    assert "left_knee_joint" in names
    assert "waist_yaw_joint" in names


# --- modes ------------------------------------------------------------------


def test_mode_switch_drives_the_platform_action(harness):
    harness.wait_active()
    response = harness.set_mode("stream")
    assert response.success, response.message
    harness.wait_for(
        lambda: harness.platform_action() == "UPPERBODY_REMOTE_SPLIT",
        "the platform to report UPPERBODY_REMOTE_SPLIT",
    )
    assert harness.status().active_mode == "stream"


def test_platform_rejection_is_surfaced_and_mode_is_unchanged(harness):
    import rclpy

    harness.wait_active()
    harness.mock.set_parameters([rclpy.parameter.Parameter("reject_action_code", value=4)])
    response = harness.set_mode("stream")
    assert not response.success
    assert "INVALID_POSTURE" in response.message
    assert harness.status().active_mode == "idle"
    assert "stream" in response.valid_transitions


def test_undeclared_transition_lists_the_valid_ones(harness):
    harness.wait_active()
    response = harness.set_mode("no_such_mode")
    assert not response.success
    assert response.valid_transitions


# --- command projection -----------------------------------------------------


def test_arm_command_reaches_the_platform_in_vendor_order(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    message = Float64MultiArray()
    message.data = [0.01 * index for index in range(14)]
    harness.publish("/aimdk/arm/commands", message)
    received = harness.wait_for(lambda: harness.mock.last_upper_body, "an upper body command")
    assert list(received.arm_pos) == pytest.approx(message.data)
    assert received.hand_sub_mode == projection.HAND_SUB_MODE_CLAW
    assert received.source == "ibrobot.aimdk_robot"


def test_command_outside_its_mode_is_counted_not_forwarded(harness):
    harness.wait_active()
    before = harness.mock.upper_body_count
    message = Float64MultiArray()
    message.data = [0.0] * 14
    harness.publish("/aimdk/arm/commands", message)
    status = harness.wait_for(
        lambda: harness.status() if "arm_stream" in harness.status().rejected_channels else None,
        "a rejected arm command",
    )
    index = list(status.rejected_channels).index("arm_stream")
    assert status.rejected_counts[index] >= 1
    assert harness.mock.upper_body_count == before


def test_out_of_range_gripper_command_is_rejected_with_its_reason(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    message = Float64MultiArray()
    message.data = [5.0, 0.5]
    harness.publish("/aimdk/gripper/commands", message)
    status = harness.wait_for(
        lambda: (
            harness.status()
            if any("GRIPPER_COMMAND_RANGE" in channel for channel in harness.status().rejected_channels)
            else None
        ),
        "the out-of-range gripper rejection",
    )
    assert any("GRIPPER_COMMAND_RANGE" in channel for channel in status.rejected_channels)


def test_gripper_only_command_holds_the_measured_arm_pose(harness):
    """arm_pos is a target vector: an uncommanded arm must be told to stay put, not go to zero."""
    harness.wait_active()
    measured = {name: 0.1 * (index + 1) for index, name in enumerate(projection.VENDOR_ARM_ORDER)}
    harness.mock.set_positions(measured)
    # Wait until the bridge has seen the non-zero arm pose before commanding.
    received: list[JointState] = []
    harness.node.create_subscription(JointState, "/joint_states", received.append, 10)
    harness.wait_for(
        lambda: received and "left_elbow_joint" in received[-1].name and max(received[-1].position) > 0.5,
        "the measured arm pose to be reported",
    )
    assert harness.set_mode("stream").success
    message = Float64MultiArray()
    message.data = [0.3, 0.7]
    harness.publish("/aimdk/gripper/commands", message)
    sent = harness.wait_for(lambda: harness.mock.last_upper_body, "an upper body command")
    assert list(sent.hand_pos) == pytest.approx([0.3, 0.7])
    assert list(sent.arm_pos) == pytest.approx([measured[name] for name in projection.VENDOR_ARM_ORDER])
    assert not any(value == 0.0 for value in sent.arm_pos)


def test_gripper_traffic_does_not_keep_a_quiet_arm_target_alive():
    """Each channel expires on its own clock: an old arm target must yield to the measured hold."""
    harness = Harness(follow_commands=False)
    try:
        harness.wait_active()
        measured = {name: 0.2 for name in projection.VENDOR_ARM_ORDER}
        harness.mock.set_positions(measured)
        received: list[JointState] = []
        harness.node.create_subscription(JointState, "/joint_states", received.append, 10)
        harness.wait_for(
            lambda: received and "left_elbow_joint" in received[-1].name and max(received[-1].position) > 0.1,
            "the measured arm pose to be reported",
        )
        assert harness.set_mode("stream").success
        arm = Float64MultiArray()
        arm.data = [0.05] * 14
        harness.publish("/aimdk/arm/commands", arm, count=3)
        sent = harness.wait_for(
            lambda: harness.mock.last_upper_body if harness.mock.last_upper_body is not None else None,
            "the arm target to be forwarded",
        )
        assert list(sent.arm_pos) == pytest.approx([0.05] * 14)
        # Keep the gripper channel busy well past stream_staleness_s (0.5 s).
        gripper = Float64MultiArray()
        gripper.data = [0.4, 0.4]
        harness.publish("/aimdk/gripper/commands", gripper, count=40)  # ~0.8 s
        sent = harness.mock.last_upper_body
        assert list(sent.hand_pos) == pytest.approx([0.4, 0.4])
        assert list(sent.arm_pos) == pytest.approx([0.2] * 14), "the stale arm target was replayed"
    finally:
        harness.shutdown()


def test_upper_body_command_is_withheld_without_a_hold_measurement():
    """No fresh measurement for an uncommanded joint means no fixed-length target can be built honestly."""
    # The head feed is quiet (not a required group, so the runtime stays
    # ACTIVE), and head_pos is a fixed-length target field on this platform.
    harness = Harness(publish_joint_groups=["arm", "waist", "leg", "hand"])
    try:
        harness.wait_active()
        assert harness.set_mode("stream").success
        before = harness.mock.upper_body_count
        message = Float64MultiArray()
        message.data = [0.3, 0.7]
        harness.publish("/aimdk/gripper/commands", message)
        status = harness.wait_for(
            lambda: (
                harness.status() if "upper_body:HOLD_TARGET_UNAVAILABLE" in harness.status().rejected_channels else None
            ),
            "the withheld upper-body command to be counted",
        )
        assert "upper_body:HOLD_TARGET_UNAVAILABLE" in status.rejected_channels
        assert harness.mock.upper_body_count == before
    finally:
        harness.shutdown()


def test_locomotion_command_reaches_the_platform(harness):
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    assert harness.enable_navigation().success
    twist = Twist()
    twist.linear.x = 0.5
    twist.angular.z = 0.3
    harness.publish("/cmd_vel", twist, message_type=Twist)
    received = harness.wait_for(lambda: harness.mock.last_locomotion, "a locomotion command")
    assert received.forward_velocity == pytest.approx(0.5)
    assert received.angular_velocity == pytest.approx(0.3)


def test_below_minimum_velocity_is_rejected(harness):
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    assert harness.enable_navigation().success
    twist = Twist()
    twist.linear.x = 0.05  # above the deadband, below the platform minimum
    harness.publish("/cmd_vel", twist, message_type=Twist)
    status = harness.wait_for(
        lambda: (
            harness.status()
            if any("VELOCITY_BELOW_MIN" in channel for channel in harness.status().rejected_channels)
            else None
        ),
        "the below-minimum velocity rejection",
    )
    assert any("VELOCITY_BELOW_MIN" in channel for channel in status.rejected_channels)


# --- stop -------------------------------------------------------------------


def test_hold_stop_latches_and_only_idle_is_accepted(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    request = StopRuntime.Request()
    request.policy = "HOLD"
    response = harness.call(StopRuntime, "/runtime/stop", request)
    assert response.success, response.message
    assert response.idle_latency_s >= 0.0
    status = harness.status()
    assert status.stop_latched and status.stop_policy == "HOLD"
    assert not harness.set_mode("stream").success
    assert harness.set_mode("idle").success
    assert not harness.status().stop_latched


def test_torque_off_reports_the_platform_safe_equivalent(harness):
    harness.wait_active()
    request = StopRuntime.Request()
    request.policy = "TORQUE_OFF"
    response = harness.call(StopRuntime, "/runtime/stop", request)
    assert response.success, response.message
    # The platform's damping mode is what TORQUE_OFF becomes on a balancing
    # biped: registered, and permitted even under the safety mechanism.
    assert "DAMPING_DEFAULT" in response.message
    assert harness.platform_action() == "DAMPING_DEFAULT"


def test_hold_stop_switches_no_mode_and_still_closes_admission(harness):
    """This platform registers no hold action, so the latch is the whole stop."""
    harness.wait_active()
    assert harness.set_mode("stream").success
    message = Float64MultiArray()
    message.data = [0.01] * 14
    harness.publish("/aimdk/arm/commands", message)
    harness.wait_for(lambda: harness.mock.last_upper_body, "an upper body command")
    before_action = harness.platform_action()
    request = StopRuntime.Request()
    request.policy = "HOLD"
    response = harness.call(StopRuntime, "/runtime/stop", request)
    assert response.success, response.message
    assert "no registered hold action" in response.message
    # The platform stays in whatever mode it was in; nothing was switched.
    assert harness.platform_action() == before_action
    status = harness.status()
    assert status.stop_latched
    assert status.active_mode == "idle"
    before = harness.mock.upper_body_count
    harness.publish("/aimdk/arm/commands", message)
    time.sleep(0.3)
    assert harness.mock.upper_body_count == before


def test_a_stop_action_the_platform_does_not_register_is_refused_before_it_is_sent():
    """Code 3 is "not registered in the configuration", not "unknown message"."""
    with pytest.raises(projection.ProjectionError):
        projection.resolve_stop_plan(
            "HOLD", body_pose=projection.BODY_POSE_STAND, stop_config={"hold_action": "JOINT_FREEZE"}
        )


def test_stop_preempts_a_mode_switch_waiting_for_confirmation():
    """A stop must never queue behind a set_mode that is waiting on the platform."""
    harness = Harness(confirm_modes=False)
    try:
        harness.wait_active()
        mode_client = harness.node.create_client(SetRuntimeMode, "/runtime/set_mode")
        assert mode_client.wait_for_service(timeout_sec=TIMEOUT_S)
        mode_request = SetRuntimeMode.Request()
        mode_request.mode = "stream"
        mode_future = mode_client.call_async(mode_request)
        # The platform has accepted but will never confirm: set_mode is now
        # inside its 5 s confirmation wait.
        harness.wait_for(lambda: harness.platform_action() == "UPPERBODY_REMOTE_SPLIT", "the vendor switch")
        requested = time.monotonic()
        stop_request = StopRuntime.Request()
        stop_request.policy = "HOLD"
        response = harness.call(StopRuntime, "/runtime/stop", stop_request)
        answered = time.monotonic() - requested
        assert response.success, response.message
        # Measured from the client's request, not from callback entry: the
        # profile's cancel bound is 1.0 s and the confirmation wait is 5 s.
        assert answered < 1.0, f"stop answered after {answered:.2f}s"
        harness.wait_for(lambda: mode_future.done(), "the interrupted set_mode to answer")
        mode_response = mode_future.result()
        assert not mode_response.success
        assert "stop engaged" in mode_response.message
        status = harness.status()
        assert status.stop_latched and status.active_mode == "idle"
    finally:
        harness.shutdown()


def test_stop_keeps_admission_closed_when_the_platform_rejects_it(harness):
    """Whether the platform confirmed the stop and whether input is admitted are two things."""
    harness.wait_active()
    assert harness.set_mode("stream").success
    message = Float64MultiArray()
    message.data = [0.01] * 14
    harness.publish("/aimdk/arm/commands", message)
    harness.wait_for(lambda: harness.mock.last_upper_body, "an upper body command")
    harness.mock.set_parameters([rclpy.parameter.Parameter("reject_action_code", value=8)])
    request = StopRuntime.Request()
    # TORQUE_OFF is the policy that asks the platform for a mode; HOLD switches
    # none on this platform, so only this one has a refusal to surface.
    request.policy = "TORQUE_OFF"
    response = harness.call(StopRuntime, "/runtime/stop", request)
    assert not response.success
    assert "MOVING_BUSY" in response.message
    status = harness.status()
    assert status.stop_latched, "a platform refusal must not reopen local admission"
    assert status.active_mode == "idle"
    assert any("stop latched" in fault for fault in status.faults)
    # Nothing is published and nothing is admitted until idle is requested.
    before = harness.mock.upper_body_count
    harness.publish("/aimdk/arm/commands", message)
    time.sleep(0.3)
    assert harness.mock.upper_body_count == before
    assert not harness.set_mode("stream").success
    harness.mock.set_parameters([rclpy.parameter.Parameter("reject_action_code", value=0)])
    assert harness.set_mode("idle").success
    assert not harness.status().stop_latched


def test_idle_cannot_clear_a_stop_that_is_still_in_progress():
    """The latch belongs to the stop until the platform has answered; idle meanwhile is refused."""
    harness = Harness(action_hold_s=1.0)
    try:
        harness.wait_active()
        stop_client = harness.node.create_client(StopRuntime, "/runtime/stop")
        assert stop_client.wait_for_service(timeout_sec=TIMEOUT_S)
        stop_request = StopRuntime.Request()
        # TORQUE_OFF reaches the platform, so the call is still in flight while
        # the latch is already closed; HOLD switches no mode and returns at once.
        stop_request.policy = "TORQUE_OFF"
        stop_future = stop_client.call_async(stop_request)
        # Latched at once, while the vendor call is still pending.
        harness.wait_for(lambda: harness.status().stop_latched, "the latch to engage")
        assert not stop_future.done()
        response = harness.set_mode("idle")
        assert not response.success
        assert "in progress" in response.message
        assert harness.status().stop_latched, "idle must not have released the latch under the stop"
        harness.wait_for(lambda: stop_future.done(), "the stop to complete", timeout_s=5.0)
        # Only now may idle clear it.
        assert harness.set_mode("idle").success
        assert not harness.status().stop_latched
    finally:
        harness.shutdown()


def test_unknown_stop_policy_is_refused(harness):
    request = StopRuntime.Request()
    request.policy = "COAST"
    response = harness.call(StopRuntime, "/runtime/stop", request)
    assert not response.success
    assert "COAST" in response.message


# --- telemetry and interaction ----------------------------------------------


def test_power_state_is_republished_on_the_neutral_topic(harness):
    received: list[PowerState] = []
    harness.node.create_subscription(PowerState, "/power_state", received.append, 10)
    message = harness.wait_for(lambda: received[-1] if received else None, "a power state message")
    assert message.battery_percentage == pytest.approx(87.0)
    assert "orin" in message.rail_names


def test_speak_maps_onto_the_platform_priority_enumeration(harness):
    """Vendor TTS priorities are named layers, not a numeric band.

    Every output must be one of the vendor's defined levels, low neutral
    priorities must land on the low layers, and SAFETY_L10 (the platform's
    life-safety layer) must be unreachable from the contract.
    """
    from aimdk_msgs.msg import TtsPriorityLevel

    harness.wait_active()
    expected = {
        0: TtsPriorityLevel.BACKGROUND_L1,
        30: TtsPriorityLevel.SERVICE_L2,
        50: TtsPriorityLevel.MISSION_L4,
        70: TtsPriorityLevel.INTERACTION_L6,
        85: TtsPriorityLevel.SYSTEM_L7,
        100: TtsPriorityLevel.WARNING_L8,
    }
    legal = {
        TtsPriorityLevel.BACKGROUND_L1,
        TtsPriorityLevel.SERVICE_L2,
        TtsPriorityLevel.MISSION_L4,
        TtsPriorityLevel.INTERACTION_L6,
        TtsPriorityLevel.SYSTEM_L7,
        TtsPriorityLevel.WARNING_L8,
        TtsPriorityLevel.SAFETY_L10,
    }
    for neutral, level in expected.items():
        harness.mock.last_tts = None
        request = SpeakText.Request()
        request.text = f"hello {neutral}"
        request.priority = neutral
        request.interrupt = True
        response = harness.call(SpeakText, "/speech/speak", request)
        assert response.success, response.message
        sent = harness.wait_for(lambda: harness.mock.last_tts, "a TTS request")
        assert sent.tts_req.text == f"hello {neutral}"
        assert sent.tts_req.is_interrupted
        assert sent.tts_req.priority_level.value == level
        assert sent.tts_req.priority_level.value in legal
        assert sent.tts_req.priority_level.value != TtsPriorityLevel.SAFETY_L10


def test_empty_speech_is_refused(harness):
    request = SpeakText.Request()
    request.text = "   "
    response = harness.call(SpeakText, "/speech/speak", request)
    assert not response.success
    assert response.error_code == "INVALID_REQUEST"


def test_unknown_expression_is_refused_without_substitution(harness):
    request = PlayExpression.Request()
    request.expression = "smug"
    response = harness.call(PlayExpression, "/expression/play", request)
    assert not response.success
    assert response.error_code == "UNKNOWN_EXPRESSION"
    assert harness.mock.last_emoji is None


def test_known_expression_is_mapped_to_the_platform_id_and_play_mode(harness):
    """Contract MODE_ONCE is 0, the vendor's is 1: the value cannot pass through."""
    from aimdk_msgs.srv import PlayEmoji

    request = PlayExpression.Request()
    request.expression = "happy"
    request.mode = PlayExpression.Request.MODE_ONCE
    response = harness.call(PlayExpression, "/expression/play", request)
    assert response.success, response.message
    sent = harness.wait_for(lambda: harness.mock.last_emoji, "an emoji request")
    assert sent.emotion_id == 90
    assert sent.mode == PlayEmoji.Request.EMOTION_MODE_ONCE


def test_looping_expression_is_mapped_to_the_vendor_loop_mode(harness):
    from aimdk_msgs.srv import PlayEmoji

    request = PlayExpression.Request()
    request.expression = "happy"
    request.mode = PlayExpression.Request.MODE_LOOP
    assert harness.call(PlayExpression, "/expression/play", request).success
    sent = harness.wait_for(lambda: harness.mock.last_emoji, "an emoji request")
    assert sent.mode == PlayEmoji.Request.EMOTION_MODE_LOOP


def test_undefined_expression_mode_is_refused(harness):
    request = PlayExpression.Request()
    request.expression = "happy"
    request.mode = 7
    response = harness.call(PlayExpression, "/expression/play", request)
    assert not response.success
    assert response.error_code == "INVALID_REQUEST"
    assert harness.mock.last_emoji is None


def test_led_pattern_is_forwarded(harness):
    request = SetLedPattern.Request()
    request.pattern = SetLedPattern.Request.PATTERN_BREATHE
    request.r, request.g, request.b = 10, 20, 30
    response = harness.call(SetLedPattern, "/led/set_pattern", request)
    assert response.success, response.message
    sent = harness.wait_for(lambda: harness.mock.last_led, "an LED request")
    assert (sent.r, sent.g, sent.b) == (10, 20, 30)
    assert sent.led_strip_mode == 1


def test_unsupported_led_pattern_is_refused(harness):
    request = SetLedPattern.Request()
    request.pattern = 9
    response = harness.call(SetLedPattern, "/led/set_pattern", request)
    assert not response.success
    assert response.error_code == "UNSUPPORTED_PATTERN"


# --- named motions ----------------------------------------------------------


def test_named_preset_motion_is_dispatched(harness):
    from rclpy.action import ActionClient

    harness.wait_active()
    client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
    assert client.wait_for_server(timeout_sec=TIMEOUT_S)
    goal = ExecuteNamedMotion.Goal()
    goal.name = "wave"
    goal.target = "right"
    handle_future = client.send_goal_async(goal)
    harness.wait_for(lambda: handle_future.done(), "the goal to be accepted")
    handle = handle_future.result()
    assert handle.accepted
    result_future = handle.get_result_async()
    harness.wait_for(lambda: result_future.done(), "the named motion result")
    result = result_future.result().result
    assert result.success, result.message
    sent = harness.wait_for(lambda: harness.mock.last_preset, "a preset motion request")
    assert sent.motion.value == 1002


def test_named_preset_motion_completes_only_when_the_platform_reports_success():
    """Dispatch acknowledgement is not completion: the result waits for the platform's terminal state."""
    from rclpy.action import ActionClient

    harness = Harness(preset_duration_s=0.8)
    try:
        harness.wait_active()
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        goal = ExecuteNamedMotion.Goal()
        goal.name = "wave"
        started = time.monotonic()
        handle_future = client.send_goal_async(goal)
        harness.wait_for(lambda: handle_future.done(), "the goal to be accepted")
        assert handle_future.result().accepted
        result_future = handle_future.result().get_result_async()
        # While the platform reports RUNNING the goal is still open.
        time.sleep(0.4)
        assert not result_future.done()
        assert harness.mock.preset_running()
        harness.wait_for(lambda: result_future.done(), "the named motion result")
        elapsed = time.monotonic() - started
        result = result_future.result().result
        assert result.success, result.message
        assert "completed" in result.message
        assert elapsed >= 0.8
        assert harness.mock.preset_state_queries >= 2
    finally:
        harness.shutdown()


def test_named_motion_is_refused_when_the_runtime_does_not_hold_arbitration():
    """SetMcPresetMotion has no priority protection; the runtime's admission is the only guard."""
    from rclpy.action import ActionClient

    harness = Harness()
    try:
        harness.wait_active()
        harness.mock.set_parameters([rclpy.parameter.Parameter("arbitration_holder", value="rc")])
        harness.wait_for(lambda: harness.status().lifecycle == "DEGRADED", "lost arbitration")
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        goal = ExecuteNamedMotion.Goal()
        goal.name = "wave"
        goal.interrupt = True
        handle_future = client.send_goal_async(goal)
        harness.wait_for(lambda: handle_future.done(), "the goal decision")
        result_future = handle_future.result().get_result_async()
        harness.wait_for(lambda: result_future.done(), "the named motion result")
        result = result_future.result().result
        assert not result.success
        assert result.error_code == ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE
        assert harness.mock.last_preset is None
    finally:
        harness.shutdown()


def test_named_motion_is_refused_outside_its_admitted_modes(harness):
    from rclpy.action import ActionClient

    harness.wait_active()
    assert harness.set_mode("stream").success
    client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
    assert client.wait_for_server(timeout_sec=TIMEOUT_S)
    goal = ExecuteNamedMotion.Goal()
    goal.name = "wave"
    handle_future = client.send_goal_async(goal)
    harness.wait_for(lambda: handle_future.done(), "the goal decision")
    result_future = handle_future.result().get_result_async()
    harness.wait_for(lambda: result_future.done(), "the named motion result")
    result = result_future.result().result
    assert not result.success
    assert result.error_code == ExecuteNamedMotion.Result.MODE_NOT_ALLOWED
    assert harness.mock.last_preset is None


def test_named_motion_cancel_is_refused_not_ignored():
    """The platform cannot cancel a motion, so a cancel is refused instead of accepted and dropped."""
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient

    harness = Harness(preset_duration_s=1.5)
    try:
        harness.wait_active()
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        goal = ExecuteNamedMotion.Goal()
        goal.name = "wave"
        handle_future = client.send_goal_async(goal)
        harness.wait_for(lambda: handle_future.done(), "the goal to be accepted")
        handle = handle_future.result()
        harness.wait_for(lambda: harness.mock.preset_running(), "the motion to start")
        cancel_future = handle.cancel_goal_async()
        harness.wait_for(lambda: cancel_future.done(), "the cancel decision")
        assert not cancel_future.result().goals_canceling
        result_future = handle.get_result_async()
        harness.wait_for(lambda: result_future.done(), "the named motion result")
        assert result_future.result().status == GoalStatus.STATUS_SUCCEEDED
    finally:
        harness.shutdown()


def test_stop_during_a_named_motion_ends_it_with_the_stop_code():
    from rclpy.action import ActionClient

    harness = Harness(preset_duration_s=3.0)
    try:
        harness.wait_active()
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        goal = ExecuteNamedMotion.Goal()
        goal.name = "wave"
        handle_future = client.send_goal_async(goal)
        harness.wait_for(lambda: handle_future.done(), "the goal to be accepted")
        harness.wait_for(lambda: harness.mock.preset_running(), "the motion to start")
        request = StopRuntime.Request()
        request.policy = "HOLD"
        assert harness.call(StopRuntime, "/runtime/stop", request).success
        result_future = handle_future.result().get_result_async()
        harness.wait_for(lambda: result_future.done(), "the named motion result", timeout_s=3.0)
        result = result_future.result().result
        assert not result.success
        assert result.error_code == ExecuteNamedMotion.Result.STOP_LATCHED
    finally:
        harness.shutdown()


def _send_named(harness, client, name: str, target: str = "", interrupt: bool = False):
    goal = ExecuteNamedMotion.Goal()
    goal.name, goal.target, goal.interrupt = name, target, interrupt
    handle_future = client.send_goal_async(goal)
    harness.wait_for(lambda: handle_future.done(), "the goal decision")
    return handle_future.result()


def test_failed_replacement_leaves_the_running_motion_tracked():
    """Ownership moves only when the platform accepts the replacement."""
    from action_msgs.msg import GoalStatus
    from rclpy.action import ActionClient

    harness = Harness(preset_duration_s=1.5)
    try:
        harness.wait_active()
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        first = _send_named(harness, client, "wave")
        assert first.accepted
        harness.wait_for(lambda: harness.mock.preset_running(), "the first motion to start")
        first_result = first.get_result_async()

        # (1) A replacement that fails validation never reaches the platform.
        bad_target = _send_named(harness, client, "wave", target="sideways", interrupt=True)
        assert bad_target.accepted
        bad_result = bad_target.get_result_async()
        harness.wait_for(lambda: bad_result.done(), "the invalid replacement to fail")
        assert bad_result.result().result.error_code == ExecuteNamedMotion.Result.INVALID_TARGET
        assert not first_result.done(), "the running motion was dropped by a replacement that never happened"

        # (2) A replacement the platform refuses leaves the old motion tracked too.
        harness.mock.set_parameters([rclpy.parameter.Parameter("reject_preset_code", value=3)])
        refused = _send_named(harness, client, "handshake", interrupt=True)
        assert refused.accepted
        refused_result = refused.get_result_async()
        harness.wait_for(lambda: refused_result.done(), "the refused replacement to fail")
        assert refused_result.result().result.error_code == ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM
        assert not first_result.done()
        harness.mock.set_parameters([rclpy.parameter.Parameter("reject_preset_code", value=0)])

        # The original goal runs to the platform's terminal state, not CANCELLED.
        harness.wait_for(lambda: first_result.done(), "the first motion to complete", timeout_s=5.0)
        assert first_result.result().status == GoalStatus.STATUS_SUCCEEDED
        assert first_result.result().result.success
    finally:
        harness.shutdown()


def test_accepted_replacement_interrupts_the_running_motion():
    from rclpy.action import ActionClient

    harness = Harness(preset_duration_s=1.5)
    try:
        harness.wait_active()
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        first = _send_named(harness, client, "wave")
        harness.wait_for(lambda: harness.mock.preset_running(), "the first motion to start")
        first_result = first.get_result_async()
        second = _send_named(harness, client, "handshake", interrupt=True)
        assert second.accepted
        harness.wait_for(lambda: first_result.done(), "the first motion to be interrupted", timeout_s=3.0)
        assert first_result.result().result.error_code == ExecuteNamedMotion.Result.CANCELLED
        assert "handshake" in first_result.result().result.message
        second_result = second.get_result_async()
        harness.wait_for(lambda: second_result.done(), "the replacement to complete", timeout_s=5.0)
        assert second_result.result().result.success
    finally:
        harness.shutdown()


def test_unknown_named_motion_is_rejected(harness):
    from rclpy.action import ActionClient

    client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
    assert client.wait_for_server(timeout_sec=TIMEOUT_S)
    goal = ExecuteNamedMotion.Goal()
    goal.name = "moonwalk"
    handle_future = client.send_goal_async(goal)
    harness.wait_for(lambda: handle_future.done(), "the goal decision")
    assert not handle_future.result().accepted


# --- failure modes ----------------------------------------------------------


def test_hand_type_mismatch_blocks_activation():
    """The declared end effector must match the installed one (design D2)."""
    harness = Harness(hand_type="dexterous")
    try:
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "FAULTED" else None,
            "the runtime to fault on a hand type mismatch",
        )
        assert any("hand type mismatch" in fault for fault in status.faults)
        assert any("claw" in fault for fault in status.faults)
    finally:
        harness.shutdown()


def test_vendor_audio_playback_is_published_reliable(harness):
    """The platform subscribes RELIABLE; a BEST_EFFORT publisher reaches nothing."""
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

    harness.wait_active()
    playback = harness.runtime._playback_pub.qos_profile
    assert playback.reliability == ReliabilityPolicy.RELIABLE
    assert playback.durability == DurabilityPolicy.VOLATILE
    # The vendor's command tier is BEST_EFFORT, and stays that way.
    assert harness.runtime._upper_body_pub.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
    assert harness.runtime._locomotion_pub.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT


def test_audio_play_reaches_the_platform(harness):
    """End to end over the QoS the platform really uses.

    This is the check that was missing when the runtime published playback
    BEST_EFFORT into the platform's RELIABLE subscription: the mismatch was
    invisible until the robot logged it, because nothing carried audio across.
    """
    from audio_common_msgs.msg import AudioDataStamped

    harness.wait_active()
    message = AudioDataStamped()
    message.audio.data = list(bytes(32))
    harness.publish("/audio/play", message, message_type=AudioDataStamped, count=5)
    sent = harness.wait_for(lambda: harness.mock.last_playback, "the platform to receive playback audio")
    assert len(sent.data.data) == 32
    assert sent.pkg_name == "ibrobot.aimdk_robot"
    assert sent.info.sample_rate == 16000


def test_platform_audio_capture_is_republished_on_the_contract_topic(harness):
    from audio_common_msgs.msg import AudioDataStamped

    received: list[AudioDataStamped] = []
    harness.node.create_subscription(AudioDataStamped, "/audio/capture_stamped", received.append, 10)
    captured = harness.wait_for(lambda: received[-1] if received else None, "republished capture audio")
    assert len(captured.audio.data) == 64


def test_audio_playback_takes_the_vendor_audio_focus(harness):
    """hal_audio does not take focus for its publishers; the bridge must."""
    from audio_common_msgs.msg import AudioDataStamped

    harness.wait_active()
    message = AudioDataStamped()
    message.audio.data = list(bytes(32))
    harness.publish("/audio/play", message, message_type=AudioDataStamped, count=5)
    harness.wait_for(lambda: harness.mock.last_playback, "focused playback to reach the platform")
    assert harness.mock.focus_holder == "ibrobot.aimdk_robot"
    # Nothing was pushed at the speaker before the platform handed it over.
    assert harness.mock.unfocused_playback_count == 0


def test_audio_focus_is_released_once_the_stream_goes_quiet(harness):
    """Holding focus while idle would keep preempting the platform's own voice."""
    from audio_common_msgs.msg import AudioDataStamped

    harness.wait_active()
    message = AudioDataStamped()
    message.audio.data = list(bytes(32))
    harness.publish("/audio/play", message, message_type=AudioDataStamped, count=5)
    harness.wait_for(lambda: harness.mock.last_playback, "playback to reach the platform")
    harness.wait_for(lambda: harness.mock.focus_holder == "", "audio focus to be released when idle")
    assert harness.mock.focus_releases >= 1


def test_audio_is_withheld_when_the_platform_refuses_focus():
    """A refused request still answers SUCCESS; only focus_gain says it was granted."""
    from audio_common_msgs.msg import AudioDataStamped

    harness = Harness(grant_audio_focus=False)
    try:
        harness.wait_active()
        message = AudioDataStamped()
        message.audio.data = list(bytes(32))
        harness.publish("/audio/play", message, message_type=AudioDataStamped, count=5)
        harness.wait_for(lambda: harness.mock.focus_requests >= 1, "the bridge to ask for audio focus")
        time.sleep(0.5)
        # Publishing anyway would contend with whatever owns the speaker.
        assert harness.mock.playback_count == 0
        assert harness.mock.unfocused_playback_count == 0
    finally:
        harness.shutdown()


def test_audio_playback_stops_when_focus_is_preempted(harness):
    """Focus loss arrives as an event; publishing past it fights the platform."""
    from audio_common_msgs.msg import AudioDataStamped

    harness.wait_active()
    message = AudioDataStamped()
    message.audio.data = list(bytes(32))
    harness.publish("/audio/play", message, message_type=AudioDataStamped, count=5)
    harness.wait_for(lambda: harness.mock.last_playback, "playback to reach the platform")
    harness.mock.preempt_audio_focus()
    harness.wait_for(lambda: not harness.runtime._focus_held, "the bridge to notice the focus loss")
    before = harness.mock.unfocused_playback_count
    harness.publish("/audio/play", message, message_type=AudioDataStamped, count=5)
    assert harness.mock.unfocused_playback_count == before


def test_unreported_hand_type_activates_unverified_and_confirms_later():
    """NONE means "no hand state yet", not "wrong hand": it must not fault the runtime."""
    harness = Harness(hand_type="none")
    try:
        # The platform reports nothing about its hands (as it does while they
        # are not enumerated), so the declaration stays unverified — and the
        # runtime is still usable.
        harness.wait_active()
        assert not harness.status().faults
        # Once the hands enumerate, the same check confirms the declaration.
        harness.mock.set_parameters([rclpy.parameter.Parameter("hand_type", value="claw")])
        harness.wait_for(
            lambda: harness.status().lifecycle == "ACTIVE" and not harness.status().faults,
            "the declaration to be confirmed",
        )
    finally:
        harness.shutdown()


def test_hand_state_stream_confirms_what_the_service_does_not_know():
    """Both sources carry the enumeration; a definite one outranks an empty answer."""
    harness = Harness(hand_type="claw", hand_type_service="none")
    try:
        harness.wait_active()
        assert not harness.status().faults
    finally:
        harness.shutdown()


def test_hand_subsystem_error_is_reported_without_claiming_a_mismatch():
    harness = Harness(hand_type="error")
    try:
        status = harness.wait_for(
            lambda: harness.status() if harness.status().faults else None,
            "the hand error to be reported",
        )
        assert any("hand subsystem error" in fault for fault in status.faults)
        assert not any("mismatch" in fault for fault in status.faults)
        assert status.lifecycle == "DEGRADED", "a hand fault is not a wrong declaration"
    finally:
        harness.shutdown()


def test_a_dexterous_family_variant_still_matches_the_dexterous_profile():
    """LEISAI/LITE_S are dexterous hands; only a different family is a mismatch."""
    harness = Harness(hand_type="leisai", profile=OMNIHAND_PROFILE)
    try:
        harness.wait_active()
        assert not harness.status().faults
    finally:
        harness.shutdown()


def test_losing_arbitration_is_reported_as_degraded():
    """A pre-empted command source is silent on the wire; it must not be silent here."""
    harness = Harness()
    try:
        harness.wait_active()
        harness.mock.set_parameters([rclpy.parameter.Parameter("arbitration_holder", value="rc")])
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "DEGRADED" else None,
            "the runtime to report lost arbitration",
        )
        assert any("rc" in fault for fault in status.faults)
    finally:
        harness.shutdown()


def test_blocking_system_mode_is_reported_as_faulted():
    """Develop_MC disables the very tier this runtime commands."""
    harness = Harness()
    try:
        harness.wait_active()
        harness.mock.set_parameters([rclpy.parameter.Parameter("system_state", value="Develop_MC")])
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "FAULTED" else None,
            "the runtime to fault on a blocking system mode",
        )
        assert any("Develop_MC" in fault for fault in status.faults)
    finally:
        harness.shutdown()


def test_unconfirmed_mode_switch_is_reported_as_failure():
    """A service that returns success is not evidence the platform switched."""
    harness = Harness(confirm_modes=False)
    try:
        harness.wait_active()
        response = harness.set_mode("stream")
        assert not response.success
        assert "did not confirm" in response.message
        assert harness.status().active_mode == "idle"
    finally:
        harness.shutdown()


def test_navigation_gate_blocks_velocity_until_enabled(harness):
    """The gate exists so a navigation stack cannot drive before it may."""
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    before = harness.mock.locomotion_count
    twist = Twist()
    twist.linear.x = 0.5
    harness.publish("/cmd_vel", twist, message_type=Twist)
    status = harness.wait_for(
        lambda: harness.status() if "base_velocity" in harness.status().rejected_channels else None,
        "the gated velocity rejection",
    )
    index = list(status.rejected_channels).index("base_velocity")
    assert status.rejected_counts[index] >= 1
    assert harness.mock.locomotion_count == before


def test_disabling_navigation_revokes_a_cached_velocity(harness):
    """The gate acts on the output: a cached velocity must not keep driving until it expires."""
    harness.wait_active()
    assert harness.set_mode("locomotion").success
    assert harness.enable_navigation().success
    twist = Twist()
    twist.linear.x = 0.5
    harness.publish("/cmd_vel", twist, message_type=Twist)
    harness.wait_for(
        lambda: harness.mock.last_locomotion is not None and harness.mock.last_locomotion.forward_velocity > 0.4,
        "the velocity to reach the platform",
    )
    assert harness.enable_navigation(False).success
    harness.wait_for(
        lambda: harness.mock.last_locomotion.forward_velocity == 0.0, "an explicit zero after the gate closed"
    )
    # Nothing non-zero follows: the cache was dropped, not left to expire.
    for _ in range(10):
        time.sleep(0.03)
        assert harness.mock.last_locomotion.forward_velocity == 0.0


def test_losing_arbitration_stops_publishing_and_requires_fresh_commands():
    """Cached targets were accepted under a claim that is gone; recovery needs new input."""
    harness = Harness()
    try:
        harness.wait_active()
        assert harness.set_mode("stream").success
        message = Float64MultiArray()
        message.data = [0.02] * 14
        harness.publish("/aimdk/arm/commands", message)
        harness.wait_for(lambda: harness.mock.last_upper_body, "an upper body command")
        harness.mock.set_parameters([rclpy.parameter.Parameter("arbitration_holder", value="rc")])
        harness.wait_for(lambda: harness.status().lifecycle == "DEGRADED", "lost arbitration")
        settled = harness.mock.upper_body_count
        time.sleep(0.3)
        assert harness.mock.upper_body_count == settled
        # Arbitration returns; the old target is not replayed.
        harness.mock.set_parameters([rclpy.parameter.Parameter("arbitration_holder", value="")])
        harness.wait_active()
        time.sleep(0.3)
        assert harness.mock.upper_body_count == settled
        harness.publish("/aimdk/arm/commands", message)
        harness.wait_for(lambda: harness.mock.upper_body_count > settled, "a fresh command to be forwarded")
    finally:
        harness.shutdown()


def test_quiet_feedback_source_drops_out_and_degrades_the_runtime():
    """Old measurements must not be re-stamped as new ones when their source goes quiet."""
    harness = Harness()
    try:
        harness.wait_active()
        received: list[JointState] = []
        harness.node.create_subscription(JointState, "/joint_states", received.append, 10)
        harness.wait_for(lambda: received and "left_elbow_joint" in received[-1].name, "arm feedback")
        harness.mock.set_parameters(
            [rclpy.parameter.Parameter("publish_joint_groups", value=["head", "waist", "leg", "hand"])]
        )
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "DEGRADED" else None,
            "the runtime to degrade on stale arm feedback",
        )
        assert any("joint feedback stale: arm" in fault for fault in status.faults)
        received.clear()
        harness.wait_for(lambda: received, "hand feedback to keep flowing")
        assert "left_elbow_joint" not in received[-1].name
        assert "left_hand" in received[-1].name
        # The source returns: the fault clears without any other intervention.
        harness.mock.set_parameters(
            [rclpy.parameter.Parameter("publish_joint_groups", value=["arm", "head", "waist", "leg", "hand"])]
        )
        harness.wait_active()
    finally:
        harness.shutdown()


def test_navigation_gate_refuses_outside_a_locomotion_mode(harness):
    harness.wait_active()
    assert harness.set_mode("stream").success
    response = harness.enable_navigation()
    assert not response.success
    assert "velocity" in response.message


# --- clock agreement --------------------------------------------------------


def test_clock_skew_holds_the_runtime_out_of_active_instead_of_failing_silently():
    """A skewed clock disables every stream; the platform never says so."""
    harness = Harness(clock_offset_s=12.0)
    try:
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "FAULTED" else None,
            "the runtime to fault on a clock disagreement",
        )
        assert any("clock disagrees with the platform" in fault for fault in status.faults)
        assert any("discarded by the platform" in fault for fault in status.faults)
        # Nothing may leave for a platform that would discard all of it.
        assert not harness.runtime.output_admitted()
    finally:
        harness.shutdown()


def test_synchronised_clocks_leave_no_clock_fault(harness):
    harness.wait_active()
    assert not [fault for fault in harness.status().faults if "clock" in fault]


# --- arbitration ------------------------------------------------------------


def test_empty_arbitration_holder_is_not_a_lost_claim(harness):
    """The platform reports no holder until a source sends a valid command.

    Treating that as a lost claim closes output, which guarantees the holder
    stays empty: the runtime deadlocks itself out of ever being able to drive.
    """
    harness.wait_active()
    # The mock only names a holder once a non-zero velocity has been claimed,
    # so the steady state right after startup is exactly the empty holder.
    assert harness.runtime.holds_arbitration()
    time.sleep(1.5)  # at least one arbitration poll
    assert harness.status().lifecycle == "ACTIVE"
    assert harness.runtime.output_admitted()


def test_upper_body_stream_reaches_the_platform_with_an_empty_holder(harness):
    """The claim is taken by streaming, so streaming cannot require the claim."""
    harness.wait_active()
    assert harness.set_mode("stream").success
    message = Float64MultiArray()
    message.data = [0.02] * 14
    harness.publish("/aimdk/arm/commands", message)
    harness.wait_for(lambda: harness.mock.last_upper_body, "an upper body command to reach the platform")
    assert harness.status().lifecycle == "ACTIVE"


def test_a_holder_this_runtime_outranks_does_not_degrade_it():
    """Streaming is how a higher-priority source takes over; it must be allowed to."""
    harness = Harness(arbitration_holder="debug_tool", arbitration_holder_priority=10)
    try:
        harness.wait_active()
        time.sleep(1.5)
        assert harness.status().lifecycle == "ACTIVE"
        assert harness.runtime.output_admitted()
    finally:
        harness.shutdown()


def test_a_holder_that_outranks_this_runtime_degrades_it_and_names_the_priority():
    harness = Harness(arbitration_holder="rc", arbitration_holder_priority=80)
    try:
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "DEGRADED" else None,
            "the runtime to degrade on a real preemption",
        )
        assert any("held by 'rc'" in fault and "priority 80" in fault for fault in status.faults)
        assert not harness.runtime.output_admitted()
    finally:
        harness.shutdown()


def test_registration_enables_the_source_not_only_adds_it():
    """A source that is added but never enabled has its commands discarded."""
    harness = Harness()
    try:
        harness.wait_active()
        assert "ibrobot.aimdk_robot" in harness.mock.enabled_sources()
    finally:
        harness.shutdown()


# --- status snapshot --------------------------------------------------------


def test_a_degraded_snapshot_always_names_its_reason():
    """Lifecycle and faults move together; neither is readable without the other."""
    harness = Harness(arbitration_holder="rc", arbitration_holder_priority=80)
    try:
        harness.wait_for(lambda: harness.status().lifecycle == "DEGRADED", "the runtime to degrade")
        for _ in range(40):
            status = harness.status()
            if status.lifecycle in ("DEGRADED", "FAULTED"):
                assert status.faults, "a non-ACTIVE snapshot with no reason is not actionable"
            time.sleep(0.02)
    finally:
        harness.shutdown()


# --- joint state rate -------------------------------------------------------


def test_joint_states_are_published_at_the_declared_rate_not_the_vendor_rate(harness):
    """The declared rate is what consumers are held to, so it drives publication.

    The vendor feeds every group at ~500 Hz. Re-aggregating per callback made
    the observed rate a function of spare CPU on the host, which is not a
    contract anyone can verify.
    """
    stamps: list[float] = []
    harness.node.create_subscription(JointState, "/joint_states", lambda m: stamps.append(time.monotonic()), 10)
    harness.wait_active()
    harness.wait_for(lambda: len(stamps) > 5, "joint states to flow")
    stamps.clear()
    time.sleep(2.0)
    declared = harness.runtime._joint_state_rate_hz
    observed = len(stamps) / 2.0
    assert declared * 0.7 <= observed <= declared * 1.3, f"declared {declared} Hz, observed {observed:.1f} Hz"


def test_an_interaction_service_call_does_not_look_like_a_clock_drift(harness):
    """A busy node must not fault as if its clock had moved.

    Calling an interaction service makes a vendor round trip while the joint
    feedback torrent keeps arriving. If clock skew were judged from a single
    stamp difference, the resulting callback latency would read as offset and
    fault the runtime — which is what happened on the robot.
    """
    harness.wait_active()
    for _ in range(5):
        request = PlayExpression.Request()
        request.expression = "happy"
        request.mode = 0
        request.priority = 50
        response = harness.call(PlayExpression, "/expression/play", request)
        assert response.success, response.message
    time.sleep(1.5)  # at least one clock check after the calls
    status = harness.status()
    assert status.lifecycle == "ACTIVE", status.faults
    assert not [fault for fault in status.faults if "clock" in fault]


def test_platform_state_is_read_off_the_feedback_torrents_callback_group(harness):
    """Low-rate state must not queue behind ~2500 joint callbacks a second."""
    harness.wait_active()
    telemetry = harness.runtime._telemetry_cb
    joint_publish = harness.runtime._joint_pub_cb
    assert telemetry is not joint_publish
    assert telemetry is not harness.runtime.default_callback_group
    assert joint_publish is not harness.runtime.default_callback_group


# --- named motion dispatch --------------------------------------------------


def test_a_dispatch_that_creates_no_task_is_a_refusal_not_a_completion():
    """Accepted with task_id 0 means nothing was dispatched.

    GetMcPresetMotionState distinguishes only "executing" from "completed", so
    polling a task the platform never created answers SUCCESS at once. On the
    robot that produced `success: true, "completed (task 0)"` with the robot
    standing still — the one answer a caller must never be given.
    """
    from rclpy.action import ActionClient

    harness = Harness(preset_creates_no_task=True)
    try:
        harness.wait_active()
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        goal = ExecuteNamedMotion.Goal()
        goal.name = "wave"
        goal.interrupt = True
        handle_future = client.send_goal_async(goal)
        harness.wait_for(lambda: handle_future.done(), "the goal decision")
        result_future = handle_future.result().get_result_async()
        harness.wait_for(lambda: result_future.done(), "the named motion result")
        result = result_future.result().result
        assert not result.success
        assert result.error_code == ExecuteNamedMotion.Result.REJECTED_BY_PLATFORM
        assert "created no task" in result.message
    finally:
        harness.shutdown()


def test_named_motion_reasserts_the_input_source_before_dispatching(harness):
    """The vendor's own preset client registers immediately before each request."""
    from rclpy.action import ActionClient

    harness.wait_active()
    before = harness.mock.input_source_calls
    client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
    assert client.wait_for_server(timeout_sec=TIMEOUT_S)
    goal = ExecuteNamedMotion.Goal()
    goal.name = "wave"
    goal.interrupt = True
    handle_future = client.send_goal_async(goal)
    harness.wait_for(lambda: handle_future.done(), "the goal decision")
    result_future = handle_future.result().get_result_async()
    harness.wait_for(lambda: result_future.done(), "the named motion result")
    assert result_future.result().result.success
    assert harness.mock.input_source_calls > before, "the claim was not re-asserted before dispatch"
    # And the request carried the registered identity, as the SDK example does.
    assert harness.mock.last_preset.input_source.name == "ibrobot.aimdk_robot"


def test_a_completed_motion_still_reports_its_real_task_id(harness):
    from rclpy.action import ActionClient

    harness.wait_active()
    client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
    assert client.wait_for_server(timeout_sec=TIMEOUT_S)
    goal = ExecuteNamedMotion.Goal()
    goal.name = "wave"
    goal.interrupt = True
    handle_future = client.send_goal_async(goal)
    harness.wait_for(lambda: handle_future.done(), "the goal decision")
    result_future = handle_future.result().get_result_async()
    harness.wait_for(lambda: result_future.done(), "the named motion result")
    result = result_future.result().result
    assert result.success, result.message
    assert "task 0)" not in result.message, "a real dispatch never carries task 0"


def test_preset_motion_asks_to_play_now_rather_than_at_this_hosts_clock(harness):
    """`header.stamp` on SetMcPresetMotion is the play time, not a message stamp.

    The SDK documents it as "stamp 用于指定播放时刻（UTC），为 0 时立即播放".
    Stamping it with this host's clock schedules the motion at a UTC instant,
    which is only correct if this host's clock matches the robot's — on a
    separate compute pack it asks for a play time in the past or the future.
    Zero is the documented "now" and needs no agreement about clocks at all.
    """
    from rclpy.action import ActionClient

    harness.wait_active()
    client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
    assert client.wait_for_server(timeout_sec=TIMEOUT_S)
    goal = ExecuteNamedMotion.Goal()
    goal.name = "wave"
    goal.interrupt = True
    handle_future = client.send_goal_async(goal)
    harness.wait_for(lambda: handle_future.done(), "the goal decision")
    result_future = handle_future.result().get_result_async()
    harness.wait_for(lambda: result_future.done(), "the named motion result")
    assert result_future.result().result.success
    stamp = harness.mock.last_preset.header.stamp
    assert (stamp.sec, stamp.nanosec) == (0, 0), "a play time from this host's clock is not 'now' on the robot"


def test_named_motion_is_refused_when_the_platform_is_not_in_stable_stand():
    """Every documented preset pair is annotated "稳定站立模式下执行".

    The runtime's own mode being idle is not the same as the platform being in
    STAND_DEFAULT — a robot on a gantry may never get there — so the platform's
    reported action is what decides, and a mismatch is named rather than sent.
    """
    from rclpy.action import ActionClient

    harness = Harness(initial_action="JOINT_DEFAULT")
    try:
        harness.wait_active()
        client = ActionClient(harness.node, ExecuteNamedMotion, "/motion/execute_named")
        assert client.wait_for_server(timeout_sec=TIMEOUT_S)
        goal = ExecuteNamedMotion.Goal()
        goal.name = "wave"
        goal.interrupt = True
        handle_future = client.send_goal_async(goal)
        harness.wait_for(lambda: handle_future.done(), "the goal decision")
        result_future = handle_future.result().get_result_async()
        harness.wait_for(lambda: result_future.done(), "the named motion result")
        result = result_future.result().result
        assert not result.success
        assert result.error_code == ExecuteNamedMotion.Result.RUNTIME_UNAVAILABLE
        assert "STAND_DEFAULT" in result.message
        assert "JOINT_DEFAULT" in result.message
        assert harness.mock.last_preset is None, "nothing may be sent when the precondition fails"
    finally:
        harness.shutdown()


def test_only_documented_control_areas_are_offered_as_targets():
    """Areas 4 and 8 exist in McControlArea but in no documented preset pair."""
    from aimdk_robot.named_motion import CONTROL_AREAS

    assert set(CONTROL_AREAS.values()) == {0, 1, 2, 3, 11}
    assert "head" not in CONTROL_AREAS
    assert "waist" not in CONTROL_AREAS


# --- platform state fields (already subscribed, previously discarded) -------


def test_balance_safe_state_degrades_the_runtime():
    """The platform entering its own SAFE state must not read as ACTIVE."""
    harness = Harness(fsm_state=projection.FSM_STATE_SAFE)
    try:
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "DEGRADED" else None,
            "the runtime to degrade on the platform's balance state",
        )
        assert any("SAFE" in fault for fault in status.faults)
        assert not harness.runtime.output_admitted()
    finally:
        harness.shutdown()


def test_a_preemption_is_seen_from_the_state_stream_without_waiting_for_a_poll():
    """The holder arrives ten times a second; waiting for the poll drops commands."""
    harness = Harness(arbitration_holder="rc", arbitration_holder_priority=80)
    try:
        # Faster than the 1 Hz arbitration poll: this can only come from the
        # state message, which carries the same fact.
        status = harness.wait_for(
            lambda: harness.status() if harness.status().lifecycle == "DEGRADED" else None,
            "the runtime to degrade from the state stream",
            timeout_s=0.9,
        )
        assert any("held by 'rc'" in fault for fault in status.faults)
    finally:
        harness.shutdown()


def test_velocity_is_checked_against_the_envelope_the_platform_reports_now():
    """A static ceiling either accepts what the platform refuses, or refuses what it allows."""
    harness = Harness(speed_max_linear=0.4, speed_max_angular=0.4)
    try:
        harness.wait_active()
        assert harness.set_mode("locomotion").success
        assert harness.enable_navigation(True).success
        harness.wait_for(lambda: harness.runtime._speed_envelope is not None, "a reported speed envelope")
        twist = Twist()
        twist.linear.x = 0.8  # inside the profile ceiling (1.0), outside the platform's 0.4
        harness.publish("/cmd_vel", twist, message_type=Twist, count=5)
        status = harness.wait_for(
            lambda: harness.status() if harness.status().rejected_channels else None,
            "the out-of-envelope velocity to be rejected",
        )
        assert any("VELOCITY_ABOVE_MAX" in channel for channel in status.rejected_channels)
        # Inside the platform's envelope it is accepted.
        twist.linear.x = 0.3
        harness.publish("/cmd_vel", twist, message_type=Twist, count=5)
        harness.wait_for(lambda: harness.mock.last_locomotion, "an accepted velocity")
    finally:
        harness.shutdown()


def test_head_targets_are_checked_against_the_platform_travel(harness):
    harness.wait_active()
    assert harness.set_mode("head").success
    message = Float64MultiArray()
    message.data = [0.9, 0.0]  # yaw far outside the platform's +/-0.38
    harness.publish("/aimdk/head/commands", message)
    status = harness.wait_for(
        lambda: harness.status() if harness.status().rejected_channels else None,
        "the out-of-travel head target to be rejected",
    )
    assert any("HEAD_COMMAND_RANGE" in channel for channel in status.rejected_channels)


def test_dexterous_hand_touch_is_republished_with_named_pads():
    """The densest contact sensor on the robot rides on a message already subscribed."""
    received: list[Float64MultiArray] = []
    harness = Harness(profile=OMNIHAND_PROFILE, hand_type="dexterous")
    try:
        # Best-effort, as the profile declares it: a 100 Hz tactile stream is a
        # sensor feed, and a reliable subscription receives nothing from it.
        harness.node.create_subscription(Float64MultiArray, "/aimdk/hand_touch", received.append, _sensor_qos())
        frame = harness.wait_for(lambda: received[-1] if received else None, "a hand touch frame")
        pads = {dim.label: dim.size for dim in frame.layout.dim}
        assert pads["left_palm"] == 36
        assert pads["right_back_of_hand"] == 36
        assert pads["left_thumb"] == 16
        # Every pad of both hands, and the data length matches the layout.
        assert len(pads) == 14
        assert len(frame.data) == sum(pads.values())
        assert set(frame.data) == {7.0}
    finally:
        harness.shutdown()


def test_a_claw_contributes_no_tactile_frames():
    """The gripper has no touch array; publishing its zeros would invent a sensor.

    The touch fields exist on every hand state message, a gripper's included,
    where they are all zeros. Those zeros are the absence of a sensor, not a
    measurement of no contact, so that hand contributes nothing.
    """
    from aimdk_msgs.msg import HandStateArray, HandType
    from aimdk_robot import vendor_gateway as vendor

    message = HandStateArray()
    message.left_hand_type = HandType(value=vendor.HAND_TYPE_CLAW)
    message.right_hand_type = HandType(value=vendor.HAND_TYPE_CLAW)
    message.left_touch_sensors.palm_touch_data = [5] * 36
    assert vendor.hand_touch_frames(message) == {}

    # One dexterous hand contributes its own pads and only its own.
    message.right_hand_type = HandType(value=vendor.HAND_TYPE_NIMBLE)
    message.right_touch_sensors.palm_touch_data = [5] * 36
    frames = vendor.hand_touch_frames(message)
    assert frames, "the dexterous hand must contribute its pads"
    assert all(pad.startswith("right_") for pad in frames)
    assert frames["right_palm"] == [5] * 36

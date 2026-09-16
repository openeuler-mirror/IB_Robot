"""In-repo mock of the AimDK vendor stack, for tests and `simulated: true`.

It speaks the real vendor message types, so the bridge is exercised against the
same IDL a robot uses; it does not simulate physics. Parameters expose the
failure modes the design has to handle — mode rejection, arbitration loss, a
body pose that forbids torque-off, a mismatched hand type — so those paths are
tested rather than assumed.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

import rclpy
from aimdk_msgs.msg import (
    AudioCapture,
    AudioInfo,
    AudioPlayback,
    FocusResponse,
    HandState,
    HandStateArray,
    HandType,
    JointState,
    JointStateArray,
    McCommonState,
    McLocomotionVelocity,
    MessageHeader,
    PmuState,
    SmSystemState,
    UpperBodyCommandArray,
)
from aimdk_msgs.srv import (
    AbandonAudioFocus,
    GetCurrentInputSource,
    GetHandType,
    GetMcPresetMotionState,
    PlayEmoji,
    PlayTts,
    RequestAudioFocus,
    SetMcAction,
    SetMcInputSource,
    SetMcPresetMotion,
    SetPmuLed,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from aimdk_robot import projection
from aimdk_robot import vendor_gateway as vg


def _qos(key: str) -> QoSProfile:
    """The QoS the platform uses for one endpoint, per the vendor's own tables.

    The mock speaks the vendor's QoS for the same reason it speaks the vendor's
    IDL: a bridge that publishes best-effort into a reliable platform
    subscription delivers nothing, and ROS 2 reports that once and then stays
    silent. Matching here is what turns such a mismatch into a test failure
    instead of a discovery on the robot.
    """
    reliability, durability, depth = projection.vendor_qos(None, key, **_VENDOR_QOS[key])
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


#: Documented platform QoS per endpoint (SDK interface tables).
_VENDOR_QOS: dict[str, dict[str, str]] = {
    "upper_body_command": {"reliability": "best_effort", "durability": "volatile"},
    "locomotion_command": {"reliability": "best_effort", "durability": "volatile"},
    "mc_state": {"reliability": "best_effort", "durability": "transient_local"},
    "system_state": {"reliability": "best_effort", "durability": "transient_local"},
    "joint_state": {"reliability": "best_effort", "durability": "transient_local"},
    "hand_state": {"reliability": "best_effort", "durability": "transient_local"},
    "power_state": {"reliability": "best_effort", "durability": "transient_local"},
    "audio_capture": {"reliability": "reliable", "durability": "volatile"},
    "audio_playback": {"reliability": "reliable", "durability": "volatile"},
    "audio_focus_response": {"reliability": "reliable", "durability": "transient_local"},
}

ARM_JOINTS = list(projection.VENDOR_ARM_ORDER)
HEAD_JOINTS = list(projection.VENDOR_HEAD_ORDER)
WAIST_JOINTS = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
LEG_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]


class AimdkVendorMock(Node):
    """Minimum vendor surface the X2 runtime bridge talks to."""

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("aimdk_vendor_mock", **node_kwargs)
        self.declare_parameter("hand_type", "claw")
        self.declare_parameter("body_pose", projection.BODY_POSE_STAND)
        self.declare_parameter("initial_action", "STAND_DEFAULT")
        self.declare_parameter("system_state", "Business")
        # Test knobs: the next SetMcAction fails with this vendor code, and the
        # arbitration query reports this holder (empty = the registered source).
        self.declare_parameter("reject_action_code", 0)
        self.declare_parameter("arbitration_holder", "")
        # The priority the forced holder claims. The platform's own sources all
        # outrank the 20-39 secondary-development band, so a preemption test
        # that leaves this at the default models a real one.
        self.declare_parameter("arbitration_holder_priority", 80)
        self.declare_parameter("confirm_modes", True)
        # A preset motion reports RUNNING for this long before SUCCESS, so the
        # bridge's completion observation is exercised, not just the dispatch.
        self.declare_parameter("preset_duration_s", 0.3)
        # The next SetMcPresetMotion fails with this vendor code (0 = accept).
        self.declare_parameter("reject_preset_code", 0)
        # Accept preset motions with code 0 but create no task (task_id 0), the
        # shape the robot produced for a source that was not enabled.
        self.declare_parameter("preset_creates_no_task", False)
        # SetMcAction answers only after this delay, to model a platform that
        # takes time to act on a stop or a mode switch.
        self.declare_parameter("action_hold_s", 0.0)
        # Whether commanded positions are reflected into joint state. Off, the
        # platform "stays where it is" so measured and commanded pose differ.
        self.declare_parameter("follow_commands", True)
        # GetHandType answers with this instead of hand_type when set.
        self.declare_parameter("hand_type_service", "")
        # Feedback sources that publish; dropping one models a quiet source.
        self.declare_parameter("publish_joint_groups", ["arm", "head", "waist", "leg", "hand"])
        # Refuse audio focus, as a higher-priority source already holding the
        # speaker would. The request still answers SUCCESS: only focus_gain says.
        self.declare_parameter("grant_audio_focus", True)
        # Seconds the platform's clock leads this host's, as a compute pack
        # whose clock was never synchronised would observe.
        self.declare_parameter("clock_offset_s", 0.0)
        # The rest of McCommonState, so the fields the bridge reads can be
        # driven from a test: balance FSM, speed envelope and motion player.
        self.declare_parameter("fsm_state", projection.FSM_STATE_STABLE)
        self.declare_parameter("speed_mode", 2)
        self.declare_parameter("speed_max_linear", 1.0)
        self.declare_parameter("speed_max_angular", 1.0)
        self.declare_parameter("player_state", projection.PLAYER_STATE_IDLE)
        # Touch cell value reported by every pad of the dexterous hands.
        self.declare_parameter("hand_touch_value", 7)

        self._lock = threading.Lock()
        self._action = str(self.get_parameter("initial_action").value)
        self._registered: dict[str, int] = {}
        self._enabled: set[str] = set()
        self._current_source = ""
        self.last_upper_body: Any = None
        self.last_locomotion: Any = None
        self.locomotion_applied_count = 0
        self.locomotion_discarded_count = 0
        self.last_tts: Any = None
        self.last_emoji: Any = None
        self.last_led: Any = None
        self.last_preset: Any = None
        self.upper_body_count = 0
        self.locomotion_count = 0
        self.last_playback: Any = None
        self.playback_count = 0
        # Audio focus, as hal_audio models it: playback published without it is
        # accepted by DDS and then not heard, so the mock counts it separately.
        self.unfocused_playback_count = 0
        self.focus_holder = ""
        self.focus_requests = 0
        self.focus_releases = 0
        self.input_source_calls = 0
        self.preset_state_queries = 0
        self._preset_task_id = 0
        self._preset_running_until = 0.0
        # Commanded positions, reflected into joint state. The platform is not
        # simulated physically; it is modelled as a follower that reaches what
        # it was told, which is what the contract's streaming tests observe.
        self._positions: dict[str, float] = {}

        group = ReentrantCallbackGroup()
        self.create_service(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction", self._set_action, callback_group=group)
        self.create_service(
            SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource", self._set_input_source, callback_group=group
        )
        self.create_service(
            GetCurrentInputSource,
            "/aimdk_5Fmsgs/srv/GetCurrentInputSource",
            self._get_input_source,
            callback_group=group,
        )
        self.create_service(GetHandType, "/aimdk_5Fmsgs/srv/GetHandType", self._get_hand_type, callback_group=group)
        self.create_service(
            SetMcPresetMotion, "/aimdk_5Fmsgs/srv/SetMcPresetMotion", self._preset_motion, callback_group=group
        )
        self.create_service(
            GetMcPresetMotionState,
            "/aimdk_5Fmsgs/srv/GetMcPresetMotionState",
            self._preset_motion_state,
            callback_group=group,
        )
        self.create_service(PlayTts, "/aimdk_5Fmsgs/srv/PlayTts", self._play_tts, callback_group=group)
        self.create_service(PlayEmoji, "/aimdk_5Fmsgs/srv/PlayEmoji", self._play_emoji, callback_group=group)
        self.create_service(SetPmuLed, "/aimdk_5Fmsgs/srv/SetPmuLed", self._set_led, callback_group=group)

        self.create_subscription(
            UpperBodyCommandArray, "/mc/upper_body_command", self._on_upper_body, _qos("upper_body_command")
        )
        self.create_subscription(
            McLocomotionVelocity, "/aima/mc/locomotion/velocity", self._on_locomotion, _qos("locomotion_command")
        )

        self._mc_pub = self.create_publisher(McCommonState, "/aima/mc/common/state", _qos("mc_state"))
        self._sm_pub = self.create_publisher(SmSystemState, "/aima/sm/system_state", _qos("system_state"))
        self._pmu_pub = self.create_publisher(PmuState, "/aima/hal/pmu/state", _qos("power_state"))
        joint_qos = _qos("joint_state")
        self._joint_pubs = {
            "arm": (self.create_publisher(JointStateArray, "/aima/hal/joint/arm/state", joint_qos), ARM_JOINTS),
            "head": (self.create_publisher(JointStateArray, "/aima/hal/joint/head/state", joint_qos), HEAD_JOINTS),
            "waist": (self.create_publisher(JointStateArray, "/aima/hal/joint/waist/state", joint_qos), WAIST_JOINTS),
            "leg": (self.create_publisher(JointStateArray, "/aima/hal/joint/leg/state", joint_qos), LEG_JOINTS),
        }
        self._hand_pub = self.create_publisher(HandStateArray, "/aima/hal/joint/hand/state", _qos("hand_state"))
        # Audio: the platform publishes capture and subscribes playback RELIABLE.
        self._audio_pub = self.create_publisher(AudioCapture, "/aima/hal/audio/capture", _qos("audio_capture"))
        self.create_subscription(
            AudioPlayback, "/aima/hal/audio/playback", self._on_audio_playback, _qos("audio_playback")
        )
        self._focus_pub = self.create_publisher(
            FocusResponse, "/aima/hal/audio/focus_response", _qos("audio_focus_response")
        )
        self.create_service(
            RequestAudioFocus, "/aimdk_5Fmsgs/srv/RequestAudioFocus", self._request_focus, callback_group=group
        )
        self.create_service(
            AbandonAudioFocus, "/aimdk_5Fmsgs/srv/AbandonAudioFocus", self._abandon_focus, callback_group=group
        )

        self.create_timer(0.1, self._publish_mc_state)
        self.create_timer(1.0, self._publish_system_state)
        self.create_timer(0.02, self._publish_joint_states)
        self.create_timer(1.0, self._publish_pmu)
        self.create_timer(0.03, self._publish_audio_capture)

    def current_action(self) -> str:
        """The MC action the platform currently reports (test inspection)."""
        with self._lock:
            return self._action

    def set_positions(self, positions: dict[str, float]) -> None:
        """Place joints without a command, as if the platform moved them itself."""
        with self._lock:
            self._positions.update({str(name): float(value) for name, value in positions.items()})

    def preset_running(self) -> bool:
        with self._lock:
            return time.monotonic() < self._preset_running_until

    def enabled_sources(self) -> set[str]:
        """Input sources that are both registered and enabled (test inspection)."""
        with self._lock:
            return {name for name in self._registered if name in self._enabled}

    # --- services -----------------------------------------------------------

    def _set_action(self, request, response):
        hold = float(self.get_parameter("action_hold_s").value)
        if hold > 0.0:
            time.sleep(hold)
        code = int(self.get_parameter("reject_action_code").value)
        if code:
            response.response.header.code = code
            response.response.status.value = 2
            response.response.message = f"mock rejection {code}"
            return response
        action = str(request.command.action_desc) or vg.MC_ACTION_BY_VALUE.get(int(request.command.action.value), "")
        if action not in projection.VENDOR_ACTIONS:
            # Code 3, "动作未在配置中登记". The firmware matches action_desc
            # against its registered configuration, which is narrower than the
            # McAction enumeration: a name that exists in the IDL and not in the
            # configuration is refused exactly like a typo.
            response.response.header.code = 3
            response.response.status.value = 2
            response.response.message = f"action {action} is not registered"
            return response
        with self._lock:
            self._action = action
        response.response.header.code = 0
        response.response.status.value = 1
        response.response.message = f"action {action} accepted"
        return response

    def _set_input_source(self, request, response):
        """ADD registers, ENABLE enables. Neither makes the source the holder.

        The platform's arbitration is a takeover rule: a registered source
        becomes the holder only by sending a valid (non-zero) command. Modelling
        registration as "you are now the holder" is what let a bridge that
        waited to be handed the claim look correct here and deadlock on the
        robot.
        """
        name = str(request.input_source.name)
        action = int(request.action.value)
        self.input_source_calls += 1
        with self._lock:
            if action == vg.INPUT_ACTION_ADD:
                if name in self._registered:
                    response.response.header.code = 1
                    return response
                self._registered[name] = int(request.input_source.priority)
            elif action == vg.INPUT_ACTION_ENABLE:
                self._registered.setdefault(name, int(request.input_source.priority))
                self._enabled.add(name)
            else:
                response.response.header.code = 2
                return response
        response.response.header.code = 0
        return response

    def _get_input_source(self, _request, response):
        holder = str(self.get_parameter("arbitration_holder").value)
        with self._lock:
            if holder:
                response.input_source.name = holder
                response.input_source.priority = int(self.get_parameter("arbitration_holder_priority").value)
            else:
                response.input_source.name = self._current_source
                response.input_source.priority = self._registered.get(self._current_source, 0)
        response.response.header.code = 0
        return response

    def _claim(self, source: str, *, moving: bool) -> bool:
        """Apply the documented takeover rule. Returns whether the command counts.

        Unregistered or not-enabled sources are discarded. A zero command is
        accepted but never takes the claim, so a runtime that only ever
        published safe zeros would never become the holder.
        """
        with self._lock:
            if source not in self._registered or source not in self._enabled:
                return False
            if not moving:
                return self._current_source in ("", source)
            holder_priority = self._registered.get(self._current_source, -1)
            if self._current_source in ("", source) or self._registered[source] >= holder_priority:
                self._current_source = source
            return self._current_source == source

    def _get_hand_type(self, _request, response):
        # The service may know less than the state stream (hand_type_service
        # overrides it, e.g. NONE from the service while the stream reports a
        # real hand type).
        override = str(self.get_parameter("hand_type_service").value)
        value = HAND_TYPE_VALUES.get(override or str(self.get_parameter("hand_type").value), vg.HAND_TYPE_CLAW)
        response.left_hands_type = HandType(value=value)
        response.right_hands_type = HandType(value=value)
        response.reponse.header.code = 0
        return response

    def _preset_motion(self, request, response):
        self.last_preset = request
        duration = float(self.get_parameter("preset_duration_s").value)
        reject = int(self.get_parameter("reject_preset_code").value)
        source = str(request.input_source.name)
        with self._lock:
            if reject:
                response.response.header.code = reject
                response.response.task_id = self._preset_task_id
                response.response.state.value = projection.TASK_STATE_FAILURE
                return response
            enabled = source in self._registered and source in self._enabled
            if not enabled or bool(self.get_parameter("preset_creates_no_task").value):
                # Accepted, but no task created — the shape the robot produced
                # for a source that was not enabled. Nothing executes, and the
                # caller has no task to observe.
                response.response.header.code = 0
                response.response.task_id = 0
                response.response.state.value = projection.TASK_STATE_SUCCESS
                return response
            running = time.monotonic() < self._preset_running_until
            if running and not bool(request.interrupt):
                # First come, first served (preset_motion.html): the reply
                # carries the running task and a non-zero code.
                response.response.header.code = 1
                response.response.task_id = self._preset_task_id
                response.response.state.value = projection.TASK_STATE_RUNNING
                return response
            self._preset_task_id += 1
            self._preset_running_until = time.monotonic() + duration
            response.response.header.code = 0
            response.response.task_id = self._preset_task_id
            response.response.state.value = projection.TASK_STATE_RUNNING
        return response

    def _preset_motion_state(self, request, response):
        self.preset_state_queries += 1
        with self._lock:
            queried = int(request.request.task_id)
            known = queried == self._preset_task_id
            running = time.monotonic() < self._preset_running_until
            response.response.task_id = self._preset_task_id
        response.response.header.code = 0
        if known and running:
            response.response.state.value = projection.TASK_STATE_RUNNING
        else:
            # The platform distinguishes only "executing" from "completed", so
            # a task id it never issued — including 0 — reads as completed.
            # That is precisely the trap a caller must not walk into.
            response.response.state.value = projection.TASK_STATE_SUCCESS
        return response

    def _play_tts(self, request, response):
        self.last_tts = request
        response.header.header.code = 0
        response.tts_resp.is_success = True
        response.tts_resp.trace_id = request.tts_req.trace_id
        return response

    def _play_emoji(self, request, response):
        self.last_emoji = request
        response.header.header.code = 0
        response.success = True
        return response

    def _set_led(self, request, response):
        self.last_led = request
        response.header.code = 0
        response.status_code = 0
        return response

    # --- command sinks ------------------------------------------------------

    def _on_upper_body(self, message) -> None:
        # Upper-body commands are not arbitration-gated on the platform: the
        # vendor's own upper_body_control example registers no input source at
        # all and still moves the arms. Only the source name is recorded.
        self.last_upper_body = message
        self.upper_body_count += 1
        if not bool(self.get_parameter("follow_commands").value):
            return
        with self._lock:
            for name, value in zip(ARM_JOINTS, list(message.arm_pos), strict=False):
                self._positions[name] = float(value)
            for name, value in zip(HEAD_JOINTS, list(message.head_pos), strict=False):
                self._positions[name] = float(value)
            hand_pos = list(message.hand_pos)
            if int(message.hand_sub_mode) == projection.HAND_SUB_MODE_CLAW and len(hand_pos) == 2:
                self._positions["left_hand"], self._positions["right_hand"] = hand_pos[0], hand_pos[1]

    def _on_locomotion(self, message) -> None:
        # Velocity is the arbitrated tier: a command from an unregistered or
        # outranked source is discarded, and a non-zero one from a registered
        # source is how the claim is taken in the first place.
        self.last_locomotion = message
        self.locomotion_count += 1
        moving = any(
            abs(float(value)) > 0.0
            for value in (message.forward_velocity, message.lateral_velocity, message.angular_velocity)
        )
        if self._claim(str(message.source), moving=moving):
            self.locomotion_applied_count += 1
        else:
            self.locomotion_discarded_count += 1

    def _on_audio_playback(self, message) -> None:
        # hal_audio does not take focus for a publisher: audio arriving without
        # it is simply not played, which is silent on the wire and audible only
        # as nothing coming out of the speaker.
        with self._lock:
            focused = self.focus_holder != "" and str(message.pkg_name) == self.focus_holder
        if not focused:
            self.unfocused_playback_count += 1
            return
        self.last_playback = message
        self.playback_count += 1

    def _request_focus(self, request, response):
        """Grant the speaker. The status is SUCCESS either way; focus_gain says."""
        requester = request.focus_requester
        self.focus_requests += 1
        granted = bool(self.get_parameter("grant_audio_focus").value)
        previous = ""
        with self._lock:
            if granted:
                previous = self.focus_holder
                self.focus_holder = str(requester.pkg_name)
        if granted and previous and previous != str(requester.pkg_name):
            self._publish_focus(previous, gain=False)
        response.focus_response.pkg_name = str(requester.pkg_name)
        response.focus_response.focus_gain = granted
        return response

    def _abandon_focus(self, request, response):
        """Release only for an exact requester match, as the platform does."""
        requester = request.focus_requester
        self.focus_releases += 1
        with self._lock:
            if self.focus_holder == str(requester.pkg_name):
                self.focus_holder = ""
        response.focus_response.pkg_name = str(requester.pkg_name)
        response.focus_response.focus_gain = False
        return response

    def preempt_audio_focus(self) -> None:
        """Take the speaker away, as a higher-priority source would."""
        with self._lock:
            holder, self.focus_holder = self.focus_holder, "preempting.source"
        if holder:
            self._publish_focus(holder, gain=False)

    def _publish_focus(self, pkg_name: str, *, gain: bool) -> None:
        event = FocusResponse()
        event.pkg_name = str(pkg_name)
        event.focus_gain = bool(gain)
        self._focus_pub.publish(event)

    # --- state publication --------------------------------------------------

    def _header(self) -> MessageHeader:
        header = MessageHeader()
        # clock_offset_s models a platform whose clock disagrees with this
        # host's, which is how a separate compute pack fails: the stamps look
        # ordinary and every command it sends falls outside the window.
        offset_ns = int(float(self.get_parameter("clock_offset_s").value) * 1e9)
        header.stamp = Time(nanoseconds=self.get_clock().now().nanoseconds + offset_ns).to_msg()
        return header

    def _publish_mc_state(self) -> None:
        message = McCommonState()
        message.header = self._header()
        with self._lock:
            action = self._action
            source = self._current_source
        message.action_info.action_desc = action
        message.action_info.current_action.value = vg.MC_ACTION_VALUES.get(action, 0)
        message.body_status.value = int(self.get_parameter("body_pose").value)
        holder = str(self.get_parameter("arbitration_holder").value)
        message.input_source.name = holder or source
        message.input_source.priority = (
            int(self.get_parameter("arbitration_holder_priority").value) if holder else self._registered.get(source, 0)
        )
        # The rest of the state message the platform pushes every tick. The
        # bridge reads all of it, so the mock has to produce all of it.
        message.fsm_state.current_state = int(self.get_parameter("fsm_state").value)
        message.speed_status.speed_mode.value = int(self.get_parameter("speed_mode").value)
        reach = float(self.get_parameter("speed_max_linear").value)
        turn = float(self.get_parameter("speed_max_angular").value)
        for bounds, limit in (
            (message.speed_status.forward_bounds, reach),
            (message.speed_status.lateral_bounds, reach),
            (message.speed_status.angular_bounds, turn),
        ):
            bounds.max_value, bounds.min_value = limit, -limit
        message.motion_status.player_state.value = int(self.get_parameter("player_state").value)
        hand_status = (
            projection.HAND_STATUS_OMNI_PICKER
            if str(self.get_parameter("hand_type").value) == "claw"
            else projection.HAND_STATUS_OMNI_HAND
        )
        message.runtime_model.left_hand_status.value = hand_status
        message.runtime_model.right_hand_status.value = hand_status
        message.runtime_model.waist_status = True
        if not bool(self.get_parameter("confirm_modes").value):
            message.action_info.action_desc = ""
        self._mc_pub.publish(message)

    def _publish_system_state(self) -> None:
        message = SmSystemState()
        message.header = self._header()
        message.cur_state = str(self.get_parameter("system_state").value)
        message.cur_status.value = 1
        self._sm_pub.publish(message)

    def _publish_joint_states(self) -> None:
        with self._lock:
            positions = dict(self._positions)
        publishing = {str(group) for group in self.get_parameter("publish_joint_groups").value}
        for group, (publisher, names) in self._joint_pubs.items():
            if group not in publishing:
                continue
            array = JointStateArray()
            array.header = self._header()
            array.joints = [
                JointState(name=name, position=positions.get(name, 0.0), velocity=0.0, effort=0.0) for name in names
            ]
            publisher.publish(array)
        if "hand" not in publishing:
            return
        hands = HandStateArray()
        hands.header = self._header()
        value = HAND_TYPE_VALUES.get(str(self.get_parameter("hand_type").value), vg.HAND_TYPE_CLAW)
        hands.left_hand_type = HandType(value=value)
        hands.right_hand_type = HandType(value=value)
        hands.left_hands = [
            HandState(name="left_hand", position=positions.get("left_hand", 0.0), velocity=0.0, effort=0.0)
        ]
        hands.right_hands = [
            HandState(name="right_hand", position=positions.get("right_hand", 0.0), velocity=0.0, effort=0.0)
        ]
        # Touch arrays ride on this message; a claw reports none.
        if value != vg.HAND_TYPE_CLAW:
            cell = int(self.get_parameter("hand_touch_value").value)
            for sensors in (hands.left_touch_sensors, hands.right_touch_sensors):
                sensors.palm_touch_data = [cell] * 36
                sensors.back_of_hand_touch_data = [cell] * 36
                for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "little_finger"):
                    setattr(sensors, f"{finger}_touch_data", [cell] * 16)
        self._hand_pub.publish(hands)

    def _publish_audio_capture(self) -> None:
        message = AudioCapture()
        message.stamps = self.get_clock().now().to_msg()
        # info.channels = mic_channels + ref_channels (AudioCapture.msg). The
        # built-in array is 4 microphones + 2 echo-reference = 6 total.
        message.mic_channels = 4
        message.ref_channels = 2
        message.info = AudioInfo(channels=6, sample_rate=16000, sample_format="S16LE", coding_format="pcm")
        message.data.data = bytes(64)
        self._audio_pub.publish(message)

    def _publish_pmu(self) -> None:
        message = PmuState()
        message.battery_remaining_capacity_percentage = 87
        message.battery_pack_voltage = 48.2
        message.battery_current = -3.1
        message.battery_temperature = 31.5
        message.battery_output_power = 149.0
        message.battery_cycle_count = 42
        message.pmu_temperature = 35.0
        message.fan_speed = 2400.0
        message.fan_pecentage = 40
        self._pmu_pub.publish(message)


HAND_TYPE_VALUES: dict[str, int] = {
    "claw": vg.HAND_TYPE_CLAW,
    "dexterous": vg.HAND_TYPE_NIMBLE,
    "leisai": vg.HAND_TYPE_LEISAI_NIMBLE,
    "lite_s": vg.HAND_TYPE_LITE_S,
    # The platform reports these while the hands are not enumerated, or on a
    # hand fault; neither names a hand family.
    "none": vg.HAND_TYPE_NONE,
    "error": vg.HAND_TYPE_ERROR,
}


def main(argv=None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    node = AimdkVendorMock()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

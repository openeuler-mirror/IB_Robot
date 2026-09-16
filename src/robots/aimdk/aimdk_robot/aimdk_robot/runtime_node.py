"""AgiBot X2 runtime: the public runtime contract served over the vendor MC tier.

This is the wrapper layer. Generic IB-Robot packages talk only to contract
names (``/runtime_status``, ``/runtime/set_mode``, ``/runtime/stop``,
``/joint_states``, the declared command channels, ``/cmd_vel``) and to the
neutral interaction/telemetry interfaces declared in the runtime profile; the
vendor's topics, services, message types, mode names and priority scales stay
behind this node.

What the node does NOT do (design D1): instantiate a controller manager, load a
ros2_control hardware component, publish on the unprotected low-level joint
command tier, or re-implement balance, gait or whole-body control.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from collections import deque
from typing import Any

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

from aimdk_robot import projection
from aimdk_robot import vendor_gateway as vg
from aimdk_robot.projection import CommandRejected, LocomotionLimits
from ibrobot_msgs.msg import PowerState, RuntimeStatus
from ibrobot_msgs.srv import (
    GetRuntimeStatus,
    PlayExpression,
    SetLedPattern,
    SetRuntimeMode,
    SpeakText,
    StartRelocalization,
    StopRuntime,
)
from robot_runtime.contract import (
    GET_STATUS_SERVICE,
    IDLE_MODE,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_CONNECTING,
    LIFECYCLE_DEGRADED,
    LIFECYCLE_FAULTED,
    NAVIGATION_ACK_TOPIC,
    NAVIGATION_ENABLE_SERVICE,
    SET_MODE_SERVICE,
    STATUS_TOPIC,
    STOP_POLICIES,
    STOP_SERVICE,
)
from robot_runtime.interface_description import build_description, validate_description
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.profile import load_profile
from robot_runtime.state import RuntimeState

#: Which vendor hand sub-mode a hand command channel speaks. The sub-mode
#: follows the channel the command arrived on, so a deployment can offer joint
#: control and gestures without a mode switch between them.
_CHANNEL_HAND_TYPES = {
    "gripper_stream": "claw",
    "hand_stream": "dexterous_joint",
    "hand_gesture_stream": "dexterous_gesture",
}

# Contract-side endpoints this runtime adds for vendor capabilities. They are
# vendor-neutral: another robot with speech or lights serves the same names.
SPEAK_SERVICE = "/speech/speak"
EXPRESSION_SERVICE = "/expression/play"
LED_SERVICE = "/led/set_pattern"
RELOCALIZE_SERVICE = "/localization/relocalize"
POWER_TOPIC = "/power_state"
DIAGNOSTICS_TOPIC = "/diagnostics"
BODY_STATE_TOPIC = "/aimdk/body_joint_states"

#: Fallback for a vendor endpoint the profile's `vendor.qos` does not name.
#: The per-endpoint values come from the vendor's interface tables instead; see
#: `projection.vendor_qos`.
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
_RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

#: How long to wait before asking for audio focus again after a refusal. A
#: refusal means another source holds the speaker, so retrying every tick would
#: only add service traffic to a queue that is already occupied.
_AUDIO_FOCUS_RETRY_S = 1.0

#: Window over which clock-offset samples are kept. Long enough that a burst of
#: callback latency cannot fill it, short enough to notice a clock being
#: stepped while the runtime is up.
_CLOCK_SKEW_WINDOW_S = 30.0


class AimdkRuntimeNode(Node):
    """Bridge between the public runtime contract and the AimDK MC tier."""

    def __init__(self, **node_kwargs: Any) -> None:
        super().__init__("aimdk_runtime", **node_kwargs)
        self.declare_parameter("profile", "")
        self.declare_parameter("simulated", False)
        self.declare_parameter("initial_mode", "")
        self.declare_parameter("instance_id", "")
        self.declare_parameter("startup_timeout_s", 20.0)
        # In simulated transport the launch entry supplies the description it
        # also feeds to the synthetic sensor publishers, so what the runtime
        # advertises and what it produces cannot drift apart.
        self.declare_parameter("interface_description_json", "")

        profile_path = self.get_parameter("profile").value
        if not profile_path:
            raise RuntimeError("aimdk_runtime requires the 'profile' parameter")
        self._profile = load_profile(profile_path)
        self._simulated = bool(self.get_parameter("simulated").value)
        instance_id = str(self.get_parameter("instance_id").value or "")
        if instance_id:
            self._profile["runtime"]["instance_id"] = instance_id
        initial_mode = str(self.get_parameter("initial_mode").value or "")
        if initial_mode:
            if initial_mode not in self._profile["modes"] or initial_mode == "initial":
                raise RuntimeError(f"unsupported initial_mode {initial_mode!r}")
            self._profile["modes"]["initial"] = initial_mode

        self._vendor = self._profile.get("vendor") or {}
        self._source = str((self._vendor.get("input_source") or {}).get("name", "ibrobot.aimdk_robot"))
        self._source_priority = int((self._vendor.get("input_source") or {}).get("priority", 30))
        self._command_rate_hz = float((self._vendor.get("command") or {}).get("rate_hz", 50.0))
        self._stamp_window_s = float((self._vendor.get("command") or {}).get("stamp_window_s", 0.2))
        # A command whose stamp falls outside the platform's window is dropped
        # by the platform without a word, so a clock that disagrees with the
        # robot's disables every stream at once. Hold the runtime out of ACTIVE
        # rather than let that look like a robot that ignores commands.
        self._clock_skew_limit_s = float((self._vendor.get("command") or {}).get("clock_skew_limit_s", 0.1))
        self._clock_samples: deque[tuple[float, float]] = deque(maxlen=256)
        # Facts the platform pushes with every state message. Reading them here
        # costs no extra traffic and removes the lag of asking over services.
        self._speed_envelope: projection.SpeedEnvelope | None = None
        self._player_state = projection.PLAYER_STATE_IDLE
        self._hand_statuses: list[int] = []
        self._head_limits = {
            str(joint): float(limit) for joint, limit in ((self._vendor.get("head") or {}).get("limits") or {}).items()
        }
        # The declared rate is what the conformance suite and every consumer
        # hold this runtime to, so it is what drives publication.
        self._joint_state_rate_hz = float(
            ((self._profile.get("capabilities") or {}).get("joint.state") or {}).get("rate_hz", 50.0)
        )
        self._locomotion = LocomotionLimits.from_profile(self._vendor.get("locomotion"))
        self._hand_type = str((self._vendor.get("hand") or {}).get("type", "claw"))
        self._default_hand_sub_mode = int(
            (self._vendor.get("hand") or {}).get("sub_mode", projection.HAND_SUB_MODE_CLAW)
        )

        # Raw audio playback is focus-gated by the platform; these stay inert
        # until a profile declares a playback endpoint.
        self._audio_lock = threading.Lock()
        self._playback_buffer: deque[list[int]] = deque(maxlen=1)
        self._focus_held = False
        self._focus_attempt_s = 0.0
        self._last_play_s = 0.0
        self._focus_priority = 6
        self._focus_weight = 0
        self._focus_release_idle_s = 1.0
        self._request_focus: Any = None
        self._abandon_focus: Any = None

        supplied = str(self.get_parameter("interface_description_json").value or "")
        if supplied:
            description = json.loads(supplied)
            validate_description(description)
        else:
            description = build_description(self._profile, simulated=self._simulated)
        modes = ModeModel(ModeModelConfig.from_profile(self._profile))
        self._state = RuntimeState(
            name=str(self._profile["runtime"]["name"]),
            version=str(self._profile["runtime"]["version"]),
            capabilities=self._profile["capabilities"],
            modes=modes,
            on_change=self._publish_status,
            interface_description=description,
        )
        self._modes = modes

        # Latest commands per channel, published at a fixed rate while their
        # mode is active. None means "nothing to send". Everything a command
        # publication reads is guarded by _lock, and the publication itself
        # runs under it, so a stop that clears the caches cannot interleave
        # with a copy already taken for publication.
        self._lock = threading.RLock()
        self._arm_cmd: list[float] | None = None
        self._hand_cmd: tuple[int, list[float]] | None = None
        self._head_cmd: list[float] | None = None
        # One receipt stamp per channel: a channel's target expires on its own
        # clock, and input on another channel must not extend it.
        self._arm_stamp = 0.0
        self._head_stamp = 0.0
        self._hand_stamp = 0.0
        self._twist: tuple[float, float, float] | None = None
        self._twist_stamp = 0.0
        self._sequence = 0
        self._arbitration_ok = False
        self._body_pose = projection.BODY_POSE_UNKNOWN
        self._navigation_enabled = False
        self._vendor_action = ""
        self._system_state = ""
        self._hold_target_missing = False
        self._hand_type_verified = False
        self._hand_type_reported: list[str] = []
        #: Hand families as reported by the hand state stream (HandStateArray
        #: carries the same enumeration as GetHandType).
        self._hand_state_types: list[int] = []
        # Per-source joint feedback with its receipt time: a source that goes
        # quiet must drop out of the aggregate instead of being re-stamped.
        self._joint_groups: dict[str, list[tuple[str, float, float, float]]] = {}
        self._feedback_received: dict[str, float] = {}
        self._feedback_clock: dict[str, Any] = {}
        feedback = self._vendor.get("feedback") or {}
        self._feedback_window_s = float(feedback.get("staleness_s", 0.5))
        self._required_feedback = [str(group) for group in feedback.get("required_groups", ["arm", "hand"])]
        self._stream_staleness_s = float((self._vendor.get("command") or {}).get("stream_staleness_s", 0.5))
        # Fault sources that decide the lifecycle between them: each holds at
        # most one fault, DEGRADED ones are recoverable, FAULTED ones persist
        # until the platform clears the condition.
        self._fault_sources: dict[str, tuple[str, str]] = {}

        self._client_cb = ReentrantCallbackGroup()
        self._service_cb = MutuallyExclusiveCallbackGroup()
        # The platform feeds five joint groups at ~500 Hz each. Left in the
        # node's default group, that torrent serialises ahead of every other
        # subscription: the low-rate state topics queue behind it, and their
        # stamps arrive late enough to be mistaken for a clock that disagrees.
        # Telemetry that must be timely gets its own group.
        self._telemetry_cb = MutuallyExclusiveCallbackGroup()
        self._joint_pub_cb = MutuallyExclusiveCallbackGroup()
        # Stop and status must never queue behind a mode switch that is
        # waiting for the platform's confirmation: they run in their own
        # reentrant group, and the stop path serialises itself.
        self._stop_cb = ReentrantCallbackGroup()
        self._stop_lock = threading.Lock()

        self._status_pub = self.create_publisher(RuntimeStatus, STATUS_TOPIC, 10)
        self._joint_pub = self.create_publisher(JointState, str(self._profile["joint_state_topic"]), _RELIABLE_QOS)
        self._body_pub = self.create_publisher(JointState, BODY_STATE_TOPIC, _RELIABLE_QOS)
        self._power_pub = self.create_publisher(PowerState, POWER_TOPIC, _RELIABLE_QOS)
        self._diag_pub = self.create_publisher(DiagnosticArray, DIAGNOSTICS_TOPIC, 10)
        self._localization_pub: Any = None

        vg.require_vendor()
        self._setup_vendor_io()
        self._setup_contract_surface()

        self.create_timer(1.0, self._publish_status, callback_group=self._stop_cb)
        self.create_timer(1.0 / max(self._command_rate_hz, 1.0), self._publish_commands, callback_group=self._client_cb)
        self.create_timer(1.0, self._poll_arbitration, callback_group=self._client_cb)
        self.create_timer(1.0, self._check_clock_skew, callback_group=self._client_cb)
        self.create_timer(0.2, self._check_feedback, callback_group=self._client_cb)
        self.create_timer(5.0, self._recheck_hand_type, callback_group=self._client_cb)
        # Joint feedback is published at the rate the profile declares, not at
        # the rate the platform happens to feed it: the vendor publishes every
        # group at ~500 Hz, and re-aggregating on each callback would make the
        # declared rate a function of this host's spare CPU.
        self.create_timer(
            1.0 / max(self._joint_state_rate_hz, 1.0), self._publish_joint_states, callback_group=self._joint_pub_cb
        )
        self._startup_thread = threading.Thread(target=self._startup_checks, daemon=True)
        self._startup_thread.start()

    # --- wiring -------------------------------------------------------------

    def _vendor_qos(self, key: str, **defaults: Any) -> QoSProfile:
        """QoS for one vendor endpoint, as the vendor documents it."""
        reliability, durability, depth = projection.vendor_qos(self._vendor.get("qos"), key, **defaults)
        return QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliability == "reliable" else ReliabilityPolicy.BEST_EFFORT,
            durability=(
                DurabilityPolicy.TRANSIENT_LOCAL if durability == "transient_local" else DurabilityPolicy.VOLATILE
            ),
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
        )

    def _setup_vendor_io(self) -> None:
        from aimdk_msgs.msg import (  # noqa: PLC0415 - vendor import stays local
            AlertCodeArray,
            AudioCapture,
            DiagnosticInfoArray,
            HandStateArray,
            JointStateArray,
            McCommonState,
            McLocomotionVelocity,
            PmuState,
            SmSystemState,
            UpperBodyCommandArray,
        )
        from aimdk_msgs.srv import (  # noqa: PLC0415
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

        topics = self._vendor.get("topics") or {}
        services = self._vendor.get("services") or {}

        self._upper_body_pub = self.create_publisher(
            UpperBodyCommandArray,
            str(topics.get("upper_body_command", "/mc/upper_body_command")),
            self._vendor_qos("upper_body_command"),
        )
        self._locomotion_pub = self.create_publisher(
            McLocomotionVelocity,
            str(topics.get("locomotion_command", "/aima/mc/locomotion/velocity")),
            self._vendor_qos("locomotion_command"),
        )

        def caller(service_type: Any, key: str, default: str) -> vg.ServiceCaller:
            endpoint = str(services.get(key, default))
            return vg.ServiceCaller(self, self.create_client(service_type, endpoint, callback_group=self._client_cb))

        self._set_action = caller(SetMcAction, "set_action", "/aimdk_5Fmsgs/srv/SetMcAction")
        self._set_input = caller(SetMcInputSource, "set_input_source", "/aimdk_5Fmsgs/srv/SetMcInputSource")
        self._get_input = caller(GetCurrentInputSource, "get_input_source", "/aimdk_5Fmsgs/srv/GetCurrentInputSource")
        self._get_hand_type = caller(GetHandType, "get_hand_type", "/aimdk_5Fmsgs/srv/GetHandType")
        self._preset_motion = caller(SetMcPresetMotion, "set_preset_motion", "/aimdk_5Fmsgs/srv/SetMcPresetMotion")
        self._preset_state = caller(
            GetMcPresetMotionState, "get_preset_motion_state", "/aimdk_5Fmsgs/srv/GetMcPresetMotionState"
        )
        self._play_tts = caller(PlayTts, "play_tts", "/aimdk_5Fmsgs/srv/PlayTts")
        self._play_emoji = caller(PlayEmoji, "play_emoji", "/aimdk_5Fmsgs/srv/PlayEmoji")
        self._set_led = caller(SetPmuLed, "set_led", "/aimdk_5Fmsgs/srv/SetPmuLed")

        state_topics = topics.get("joint_state") or {}
        joint_state_qos = self._vendor_qos("joint_state")
        for group, topic in state_topics.items():
            self.create_subscription(
                JointStateArray,
                str(topic),
                lambda message, group=str(group): self._on_joint_state(group, message),
                joint_state_qos,
            )
        if topics.get("hand_state"):
            self.create_subscription(
                HandStateArray, str(topics["hand_state"]), self._on_hand_state, self._vendor_qos("hand_state")
            )
            # The hands' touch arrays ride on the state message the line above
            # already subscribes to, so publishing them costs no extra traffic.
            touch_topic = str((self._vendor.get("hand") or {}).get("touch_topic", ""))
            if touch_topic:
                self._hand_touch_pub = self.create_publisher(Float64MultiArray, touch_topic, _SENSOR_QOS)
        if topics.get("mc_state"):
            self.create_subscription(
                McCommonState,
                str(topics["mc_state"]),
                self._on_mc_state,
                self._vendor_qos("mc_state"),
                callback_group=self._telemetry_cb,
            )
        if topics.get("system_state"):
            self.create_subscription(
                SmSystemState,
                str(topics["system_state"]),
                self._on_system_state,
                self._vendor_qos("system_state"),
                callback_group=self._telemetry_cb,
            )
        if self._state.has_capability("power.state") and topics.get("power_state"):
            self.create_subscription(
                PmuState, str(topics["power_state"]), self._on_power_state, self._vendor_qos("power_state")
            )
        if self._state.has_capability("diagnostics.codes"):
            if topics.get("diag_codes"):
                self.create_subscription(
                    DiagnosticInfoArray, str(topics["diag_codes"]), self._on_diagnostics, self._vendor_qos("diag_codes")
                )
            if topics.get("alert_codes"):
                self.create_subscription(
                    AlertCodeArray, str(topics["alert_codes"]), self._on_alerts, self._vendor_qos("alert_codes")
                )
        if self._state.has_capability("interaction.audio_capture") and topics.get("audio_capture"):
            self._setup_audio(AudioCapture, topics, caller, RequestAudioFocus, AbandonAudioFocus)
        if self._state.has_capability("localization.pose") and topics.get("localization_pose"):
            self._setup_localization(topics)

    def _setup_audio(
        self,
        audio_capture_type: Any,
        topics: dict[str, Any],
        caller: Any,
        request_focus_type: Any,
        abandon_focus_type: Any,
    ) -> None:
        """Republish vendor microphone audio on the project's audio contract."""
        from audio_common_msgs.msg import AudioDataStamped, AudioInfo  # noqa: PLC0415

        audio = self._vendor.get("audio") or {}
        # Reliable on the contract side too: the project's own ASR consumer
        # subscribes reliably, and a best-effort publisher would never reach it.
        self._audio_pub = self.create_publisher(
            AudioDataStamped, str(audio.get("capture_topic", "/audio/capture_stamped")), _RELIABLE_QOS
        )
        self._audio_info_pub = self.create_publisher(AudioInfo, str(audio.get("info_topic", "/audio/info")), 1)
        self._audio_stamped_type = AudioDataStamped
        self._audio_info_type = AudioInfo
        self.create_subscription(
            audio_capture_type,
            str(topics["audio_capture"]),
            self._on_audio_capture,
            self._vendor_qos("audio_capture", reliability="reliable"),
        )
        if topics.get("audio_playback"):
            from aimdk_msgs.msg import AudioPlayback, FocusResponse  # noqa: PLC0415

            # The platform subscribes RELIABLE here: a BEST_EFFORT publisher would
            # be silently dropped (observed on the robot, 2026-09-22).
            self._playback_pub = self.create_publisher(
                AudioPlayback, str(topics["audio_playback"]), self._vendor_qos("audio_playback", reliability="reliable")
            )
            self._playback_type = AudioPlayback
            priority, weight, release_idle_s, buffer_chunks = projection.audio_focus_request(audio)
            self._focus_priority = priority
            self._focus_weight = weight
            self._focus_release_idle_s = release_idle_s
            self._playback_buffer = deque(maxlen=buffer_chunks)
            self._request_focus = caller(
                request_focus_type, "request_audio_focus", "/aimdk_5Fmsgs/srv/RequestAudioFocus"
            )
            self._abandon_focus = caller(
                abandon_focus_type, "abandon_audio_focus", "/aimdk_5Fmsgs/srv/AbandonAudioFocus"
            )
            # Preemption is announced, not polled: a higher-priority source
            # taking the speaker arrives here, and publishing past it would
            # only fight whatever the platform decided should be heard.
            self.create_subscription(
                FocusResponse,
                str(topics.get("audio_focus_response", "/aima/hal/audio/focus_response")),
                self._on_audio_focus,
                self._vendor_qos("audio_focus_response", reliability="reliable", durability="transient_local"),
            )
            self.create_subscription(
                AudioDataStamped, str(audio.get("play_topic", "/audio/play")), self._on_audio_play, _SENSOR_QOS
            )
            self._focus_timer = self.create_timer(0.1, self._audio_focus_tick, callback_group=self._client_cb)

    def _setup_localization(self, topics: dict[str, Any]) -> None:
        from nav_msgs.msg import Odometry  # noqa: PLC0415

        params = self._state.capabilities.get("localization.pose", {})
        self._localization_frame = str(params.get("reference_frame", "map"))
        self._localization_pub = self.create_publisher(PoseWithCovarianceStamped, "/localization/pose", _RELIABLE_QOS)
        self.create_subscription(
            Odometry,
            str(topics["localization_pose"]),
            self._on_localization,
            self._vendor_qos("localization_pose", reliability="reliable"),
        )

    def _setup_contract_surface(self) -> None:
        self.create_service(SetRuntimeMode, SET_MODE_SERVICE, self._on_set_mode, callback_group=self._service_cb)
        self.create_service(GetRuntimeStatus, GET_STATUS_SERVICE, self._on_get_status, callback_group=self._stop_cb)
        self.create_service(StopRuntime, STOP_SERVICE, self._on_stop, callback_group=self._stop_cb)

        for channel in self._profile.get("command_channels", []):
            name = str(channel["channel"])
            topic = str(channel["topic"])
            if str(channel.get("type", "float64_array")).lower() == "twist":
                self.create_subscription(
                    Twist, topic, lambda message, name=name: self._on_twist(name, message), _SENSOR_QOS
                )
                continue
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda message, channel=channel: self._on_stream(channel, message),
                _SENSOR_QOS,
            )

        if self._state.has_capability("interaction.tts"):
            self.create_service(SpeakText, SPEAK_SERVICE, self._on_speak, callback_group=self._service_cb)
        if self._state.has_capability("interaction.expression"):
            self.create_service(
                PlayExpression, EXPRESSION_SERVICE, self._on_expression, callback_group=self._service_cb
            )
        if self._state.has_capability("interaction.led"):
            self.create_service(SetLedPattern, LED_SERVICE, self._on_led, callback_group=self._service_cb)
        if self._state.has_capability("base.navigation_gate"):
            from std_srvs.srv import SetBool  # noqa: PLC0415

            self._nav_ack_pub = self.create_publisher(Bool, NAVIGATION_ACK_TOPIC, 10)
            self.create_service(
                SetBool, NAVIGATION_ENABLE_SERVICE, self._on_set_navigation, callback_group=self._service_cb
            )
            self.create_timer(0.5, self._publish_navigation_ack, callback_group=self._service_cb)
        if self._state.has_capability("localization.map"):
            self.create_service(
                StartRelocalization, RELOCALIZE_SERVICE, self._on_relocalize, callback_group=self._service_cb
            )
        if self._state.has_capability("motion.named") or self._state.has_capability("motion.posture"):
            from aimdk_robot.named_motion import NamedMotionServer  # noqa: PLC0415

            self._named_motion = NamedMotionServer(self)

    def _on_set_navigation(self, request, response):
        """Gate externally sourced velocity, with the platform's own precondition.

        The platform only accepts velocity in a stable-stand/locomotion action,
        and only from the arbitration holder; acknowledging the gate before both
        hold would tell a navigation stack it may drive when it may not.
        """
        enable = bool(request.data)
        if not enable:
            # Revoking the gate revokes the output, not just future input: a
            # cached velocity must not keep driving until it expires.
            with self._lock:
                self._navigation_enabled = False
                self._twist = None
            self.publish_safe_command()
            self._publish_navigation_ack()
            response.success, response.message = True, "navigation disabled"
            return response
        if self._state.stop_latched:
            response.success, response.message = False, "stop latched"
            return response
        if self._state.lifecycle != LIFECYCLE_ACTIVE:
            response.success = False
            response.message = f"runtime is {self._state.lifecycle}; navigation is only admitted while ACTIVE"
            return response
        spec = self._modes.spec()
        if not spec.allows_base:
            response.success = False
            response.message = f"mode {self._modes.mode!r} does not accept velocity commands"
            return response
        with self._lock:
            holding = self._arbitration_ok
        if not holding:
            # Try to take the claim rather than refusing on a stale flag; only
            # a real failure to register blocks the gate.
            holding = self._register_input_source()
        if not holding:
            response.success, response.message = False, "runtime does not hold platform command arbitration"
            return response
        with self._lock:
            self._navigation_enabled = True
        self._publish_navigation_ack()
        response.success, response.message = True, "navigation enabled"
        return response

    def _publish_navigation_ack(self) -> None:
        publisher = getattr(self, "_nav_ack_pub", None)
        if publisher is None:
            return
        with self._lock:
            enabled = self._navigation_enabled
        publisher.publish(Bool(data=enabled))

    # --- startup ------------------------------------------------------------

    def _startup_checks(self) -> None:
        """Reach ACTIVE only when the platform actually answers what we declared."""
        deadline = time.time() + float(self.get_parameter("startup_timeout_s").value)
        required = [("SetMcAction", self._set_action), ("SetMcInputSource", self._set_input)]
        missing = []
        while time.time() < deadline:
            missing = [name for name, client in required if not client.ready(0.5)]
            if not missing:
                break
        if missing:
            self._set_fault(
                "startup", f"vendor services unavailable: {', '.join(missing)}", lifecycle=LIFECYCLE_FAULTED
            )
            self._state.set_lifecycle(LIFECYCLE_FAULTED)
            return
        if not self._verify_hand_type():
            return
        if not self._register_input_source():
            self._set_fault("arbitration", "vendor input source registration failed; commands would be discarded")
            self._state.set_lifecycle(LIFECYCLE_DEGRADED)
            return
        self._state.set_lifecycle(LIFECYCLE_ACTIVE)
        # Faults observed while CONNECTING (a quiet feedback source, a
        # blocking system mode) apply from the first ACTIVE moment.
        self._refresh_lifecycle()

    def _verify_hand_type(self) -> bool:
        """Check the declared end effector against the installed one (design D2).

        Returns False only when the platform positively reports a *different*
        hand family. Two answers are not mismatches and must not be treated as
        one: ``NONE`` means the platform has no hand state to report (the hands
        are not enumerated in this mode, or none are installed) and ``ERROR``
        is a platform-side hand fault. Both leave the declaration unverified,
        and ``_recheck_hand_type`` asks again — hands enumerate late.

        Two sources answer this: the ``GetHandType`` service and the hand state
        topic, which carries the same enumeration. Either is accepted, because
        a service that reports NONE while the state stream reports a real hand
        type has simply been asked too early.
        """
        declared = "claw" if self._hand_type == "claw" else "dexterous"
        reported: list[int] = []
        if self._get_hand_type.ready(2.0):
            result = self._get_hand_type.call(vg.hand_type_request(self))
            if result.ok and result.response is not None:
                reported = [
                    int(result.response.left_hands_type.value),
                    int(result.response.right_hands_type.value),
                ]
            else:
                self.get_logger().warning("GetHandType did not answer; declared hand type is unverified")
        else:
            self.get_logger().warning("GetHandType unavailable; declared hand type is unverified")
        with self._lock:
            from_state = list(self._hand_state_types)
        families = {vg.hand_type_family(value) for value in reported} if reported else {None}
        if None in families and from_state:
            # The service had nothing definite; the state stream might.
            state_families = {vg.hand_type_family(value) for value in from_state}
            if None not in state_families:
                reported, families = from_state, state_families

        if families == {declared}:
            with self._lock:
                self._hand_type_verified = True
            self._set_fault("hand", None)
            return True

        names = [vg.hand_type_name(value) for value in reported]
        if None in families:
            # Nothing to compare against: report it, keep the declaration
            # unverified, and let the recheck decide later.
            if vg.HAND_TYPE_ERROR in reported:
                self._set_fault(
                    "hand",
                    f"platform reports a hand subsystem error ({names}); "
                    f"the declared {declared!r} end effector is not usable",
                )
            else:
                self._set_fault("hand", None)
                if names != self._hand_type_reported:
                    self.get_logger().warning(
                        f"platform reports hand type {names or 'nothing'}; declared {declared!r} "
                        "is unverified (hands may not be enumerated in the current platform mode)"
                    )
            self._hand_type_reported = names
            return True

        self._set_fault(
            "hand",
            f"hand type mismatch: profile declares {declared!r}, platform reports {names}",
            lifecycle=LIFECYCLE_FAULTED,
        )
        self._state.set_lifecycle(LIFECYCLE_FAULTED)
        return False

    def _recheck_hand_type(self) -> None:
        """Ask again while the declaration is unverified; hands enumerate late."""
        with self._lock:
            verified = self._hand_type_verified
        if verified or self._state.lifecycle in (LIFECYCLE_CONNECTING, LIFECYCLE_FAULTED):
            return
        self._verify_hand_type()

    def _register_input_source(self) -> bool:
        """ADD, then ENABLE — the SDK's documented registration sequence.

        Both are sent, but only ENABLE decides: the platform discards commands
        from a source that is "未注册或未启用", so a source that was added and
        never enabled is ignored exactly like one that was never added. ADD is
        allowed to fail — it does, with "already registered", every time this
        runs again to re-assert a claim — and that is not a failure to register.
        """
        source = self._vendor.get("input_source") or {}

        def request(action: int) -> Any:
            return vg.input_source_request(
                self,
                name=self._source,
                priority=self._source_priority,
                timeout_ms=int(source.get("timeout_ms", 1000)),
                action_value=action,
            )

        self._set_input.call(request(vg.INPUT_ACTION_ADD))
        if not self._set_input.call(request(vg.INPUT_ACTION_ENABLE)).ok:
            return False
        with self._lock:
            self._arbitration_ok = True
        return True

    def _poll_arbitration(self) -> None:
        """Report only a claim we have actually lost to a source that outranks us.

        The platform's arbitration is a *takeover* rule, not a reservation: a
        registered source becomes the holder by sending a non-zero command when
        there is no holder, the holder has timed out, or it outranks the
        holder. Two consequences the runtime has to respect, or it deadlocks
        itself into never being able to claim anything:

        - An empty holder means "nothing has sent a valid command yet", which
          the SDK states outright. It is not a lost claim, and closing output
          on it guarantees the holder stays empty forever.
        - A holder we outrank is not a lost claim either: streaming is how the
          takeover happens. Only a holder that outranks us can keep our
          commands out, and that is the one worth degrading for.
        """
        if self._state.lifecycle == LIFECYCLE_CONNECTING:
            return
        if not self._get_input.ready(0.0):
            return
        result = self._get_input.call(vg.current_input_source_request(self))
        if not result.ok or result.response is None:
            return
        holder = str(result.response.input_source.name or "")
        holder_priority = int(getattr(result.response.input_source, "priority", 0) or 0)
        outranked = bool(holder) and holder != self._source and holder_priority >= self._source_priority
        with self._lock:
            registered = self._arbitration_ok
        if outranked:
            if registered:
                # Whatever was cached was accepted under a claim we no longer
                # hold; recovery requires fresh commands, not a replay.
                self._revoke_output()
            with self._lock:
                self._arbitration_ok = False
            self._set_fault(
                "arbitration",
                f"command arbitration held by {holder!r} at priority {holder_priority} "
                f"(this runtime registers {self._source_priority}); streamed commands are not applied",
            )
            return
        if not registered:
            # Take the claim back rather than waiting to be handed it: the
            # registration is ours to re-assert, and nothing else will.
            self._register_input_source()
        with self._lock:
            registered = self._arbitration_ok
        self._set_fault(
            "arbitration",
            None if registered else "vendor input source registration failed; commands would be discarded",
        )

    # --- faults and lifecycle ----------------------------------------------

    def _set_fault(self, source: str, detail: str | None, *, lifecycle: str = LIFECYCLE_DEGRADED) -> None:
        """Record (or clear, with ``detail=None``) one source's fault and re-derive the lifecycle."""
        with self._lock:
            if detail is None:
                if source not in self._fault_sources:
                    return
                del self._fault_sources[source]
            else:
                if self._fault_sources.get(source) == (lifecycle, detail):
                    return
                self._fault_sources[source] = (lifecycle, detail)
        self._refresh_lifecycle()

    def _refresh_lifecycle(self) -> None:
        """Lifecycle = the worst live fault source; ACTIVE when none remain.

        Never touches CONNECTING (startup decides when that ends) or STOPPED
        (the latch owns the lifecycle until it is cleared). Lifecycle and
        faults move together, so a reader never sees a degraded runtime that
        names no reason.
        """
        with self._lock:
            sources = dict(self._fault_sources)
        faults = [detail for _lifecycle, detail in sources.values()]
        if self._state.stop_latched or self._state.lifecycle == LIFECYCLE_CONNECTING:
            self._state.set_status(None, faults)
            return
        if any(lifecycle == LIFECYCLE_FAULTED for lifecycle, _detail in sources.values()):
            target = LIFECYCLE_FAULTED
        elif sources:
            target = LIFECYCLE_DEGRADED
        else:
            target = LIFECYCLE_ACTIVE
        self._state.set_status(target, faults)

    def holds_arbitration(self) -> bool:
        with self._lock:
            return self._arbitration_ok

    def output_admitted(self) -> bool:
        """Whether anything may leave for the platform right now.

        The same rule gates command input and command output: not stopped,
        ACTIVE (so neither DEGRADED by lost arbitration nor FAULTED by a
        blocking system mode), and holding the platform's command claim.
        """
        if self._state.stop_latched or self._state.lifecycle != LIFECYCLE_ACTIVE:
            return False
        return self.holds_arbitration()

    # --- contract services --------------------------------------------------

    def _on_set_mode(self, request, response):
        target = str(request.mode)
        epoch = self._state.stop_epoch
        if self._state.stop_latched and target != IDLE_MODE:
            response.success = False
            response.message = f"stop latched ({self._state.stop_policy}); only {IDLE_MODE!r} is accepted"
            response.valid_transitions = [IDLE_MODE]
            return response
        decision = self._modes.can_switch(target)
        if not decision.allowed:
            response.success = False
            response.message = decision.reason
            response.valid_transitions = sorted(self._modes.valid_transitions())
            return response
        if decision.mode == self._modes.mode and decision.reason == "already active":
            # A stop leaves the runtime in idle with the latch engaged, so
            # requesting idle again is how the latch is cleared: "already
            # active" must not short-circuit that.
            if self._state.stop_latched and target == IDLE_MODE:
                if not self._clear_stop_latch():
                    response.success = False
                    response.message = (
                        "stop still in progress (waiting for the platform); "
                        f"request {IDLE_MODE!r} again once it has completed"
                    )
                    response.valid_transitions = [IDLE_MODE]
                    return response
                response.success = True
                response.message = "stop latch cleared"
                return response
            response.success = True
            response.message = "already active"
            return response
        ok, detail = self._request_vendor_mode(target, epoch)
        if not ok:
            response.success = False
            response.message = detail
            response.valid_transitions = (
                [IDLE_MODE] if self._state.stop_latched else sorted(self._modes.valid_transitions())
            )
            return response
        if self._state.stop_epoch != epoch:
            # A stop engaged between the platform's confirmation and this
            # commit. The stop handler already put the runtime in idle and
            # latched it; the late success must not override that.
            response.success = False
            response.message = f"stop engaged while switching to {target!r}; request {IDLE_MODE!r} to clear the latch"
            response.valid_transitions = [IDLE_MODE]
            return response
        self._clear_commands()
        self._modes.commit(target)
        if self._state.stop_latched and target == IDLE_MODE and not self._clear_stop_latch():
            response.success = False
            response.message = f"stop still in progress; request {IDLE_MODE!r} again once it has completed"
            response.valid_transitions = [IDLE_MODE]
            return response
        self._publish_status()
        response.success = True
        response.message = detail
        return response

    def _clear_stop_latch(self) -> bool:
        """Clear the latch, unless a stop is still running. Returns whether it cleared.

        The stop handler holds ``_stop_lock`` from latching until the platform
        has answered, so the latch cannot be released underneath a stop that
        is still in progress: an idle request arriving meanwhile is refused
        and must be repeated once the stop has completed.
        """
        if not self._stop_lock.acquire(blocking=False):
            return False
        try:
            self._set_fault("stop", None)
            self._state.clear_stop()
            # A stop released the platform arbitration claim; re-arm it, or the
            # next command stream would be silently discarded.
            self._register_input_source()
            # clear_stop reports ACTIVE; a fault that is still live outranks that.
            self._refresh_lifecycle()
        finally:
            self._stop_lock.release()
        return True

    def _request_vendor_mode(self, mode: str, epoch: int) -> tuple[bool, str]:
        """Switch the vendor MC action and wait for its own state to confirm.

        The wait is interruptible: a stop engaged meanwhile (``epoch`` moves)
        ends it at once, so a stop never queues behind a confirmation.
        """
        action = projection.mode_action(self._profile, mode)
        result = self._set_action.call(vg.set_action_request(self, source=self._source, action_name=action))
        if not result.ok:
            reason = projection.mode_rejection_reason(result.code)
            return False, f"platform rejected {action}: {reason} ({result.message})".strip()
        deadline = time.time() + float((self._vendor.get("command") or {}).get("mode_confirm_timeout_s", 5.0))
        while time.time() < deadline:
            if self._state.stop_epoch != epoch:
                return False, f"stop engaged while waiting for the platform to confirm {action}"
            with self._lock:
                confirmed = self._vendor_action == action
            if confirmed:
                return True, f"platform confirmed {action}"
            time.sleep(0.05)
        return False, f"platform did not confirm {action} within the timeout"

    def _on_get_status(self, _request, response):
        response.status = self._state.to_msg(self.get_clock().now().to_msg())
        return response

    def _on_stop(self, request, response):
        policy = str(request.policy or STOP_POLICIES[0])
        response.cancel_latency_s = -1.0
        response.idle_latency_s = -1.0
        response.torque_off_latency_s = -1.0
        if policy not in STOP_POLICIES:
            response.success = False
            response.message = f"unknown stop policy {policy!r}; supported: {list(STOP_POLICIES)}"
            return response
        with self._lock:
            body_pose = self._body_pose
        try:
            plan = projection.resolve_stop_plan(policy, body_pose=body_pose, stop_config=self._vendor.get("stop"))
        except CommandRejected as rejected:
            # A humanoid cannot always honour TORQUE_OFF. Refusing is the honest
            # answer; silently holding instead and reporting success is not.
            response.success = False
            response.message = f"{rejected.reason}: {rejected.detail}"
            return response

        with self._stop_lock:
            started = time.monotonic()
            # (1) Close admission first, atomically: the latch and the idle
            #     mode stop both input callbacks and the output timer before
            #     anything else happens, and outlive a vendor failure below.
            self._state.engage_stop(policy)
            self._modes.commit(IDLE_MODE if IDLE_MODE in self._modes.declared_modes() else self._modes.mode)
            self._clear_commands()
            self.publish_safe_command()
            response.cancel_latency_s = time.monotonic() - started

            # (2) Ask the platform to hold (or release), then confirm. A plan
            #     with no vendor action is not an omission: this platform
            #     registers no mode that holds the robot from every posture, so
            #     closing admission is the hold and switching modes would be a
            #     change of behaviour rather than a stop.
            if plan.release_arbitration:
                self._release_input_source()
            if not plan.vendor_action:
                response.idle_latency_s = time.monotonic() - started
                self._publish_status()
                response.success = True
                response.message = (
                    "stopped: command admission closed and an explicit safe command published; "
                    "the platform holds its current motion mode (no registered hold action)"
                )
                return response
            result = self._set_action.call(
                vg.set_action_request(self, source=self._source, action_name=plan.vendor_action)
            )
            elapsed = time.monotonic() - started
            if policy == "TORQUE_OFF":
                response.torque_off_latency_s = elapsed
            response.idle_latency_s = elapsed
            if not result.ok:
                # The local guarantees stand: nothing is admitted or published
                # until idle is requested. Only the platform-side guarantee is
                # unconfirmed, and the answer says exactly that.
                detail = (
                    f"platform rejected stop action {plan.vendor_action}: "
                    f"{projection.mode_rejection_reason(result.code)}"
                )
                self._set_fault("stop", f"{detail}; command admission stays closed (stop latched)")
                response.success = False
                response.message = f"{detail}; stop latched locally, platform-side stop unconfirmed"
                return response
            self._publish_status()
            response.success = True
            response.message = f"stopped via {plan.vendor_action}" + (
                f" (requested {plan.downgraded_from}; platform-safe equivalent)" if plan.downgraded_from else ""
            )
            return response

    def _release_input_source(self) -> None:
        with self._lock:
            self._arbitration_ok = False

    def _revoke_output(self) -> None:
        """Drop every cached command and leave an explicit zero behind."""
        self._clear_commands()
        self.publish_safe_command()

    # --- interaction services ----------------------------------------------

    def _on_speak(self, request, response):
        response.utterance_id = ""
        if self._state.stop_latched:
            response.success, response.error_code = False, "STOP_LATCHED"
            response.message = "stop latched"
            return response
        try:
            vendor_request = vg.tts_request(
                self,
                text=str(request.text),
                priority=int(request.priority),
                interrupt=bool(request.interrupt),
                trace_id=str(request.trace_id),
                domain=str((self._vendor.get("tts") or {}).get("domain", "ibrobot")),
            )
        except CommandRejected as rejected:
            response.success, response.error_code, response.message = False, rejected.reason, rejected.detail
            return response
        if not str(request.text).strip():
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = "text is empty"
            return response
        result = self._play_tts.call(vendor_request)
        response.success = result.ok
        response.error_code = "" if result.ok else "REJECTED"
        response.message = result.message
        if result.ok and result.response is not None:
            response.utterance_id = str(getattr(result.response.tts_resp, "trace_id", "") or request.trace_id)
        return response

    def _on_expression(self, request, response):
        table = {str(k): int(v) for k, v in (self._vendor.get("expressions") or {}).items()}
        try:
            emotion_id = projection.resolve_named(str(request.expression), table, reason="UNKNOWN_EXPRESSION")
        except CommandRejected as rejected:
            response.success, response.error_code, response.message = False, rejected.reason, rejected.detail
            return response
        mode = int(request.mode)
        if mode not in (PlayExpression.Request.MODE_ONCE, PlayExpression.Request.MODE_LOOP):
            response.success, response.error_code = False, "INVALID_REQUEST"
            response.message = f"mode {mode} is neither MODE_ONCE nor MODE_LOOP"
            return response
        result = self._play_emoji.call(
            vg.emoji_request(
                self,
                emotion_id=emotion_id,
                loop=mode == PlayExpression.Request.MODE_LOOP,
                priority=int(request.priority),
            )
        )
        response.success = result.ok
        response.error_code = "" if result.ok else "REJECTED"
        response.message = result.message
        return response

    def _on_led(self, request, response):
        pattern = int(request.pattern)
        # Only a call that took effect may claim an effective priority: the
        # field means "priority effective after this call", so echoing the
        # request on a refusal would tell the caller it won when it lost.
        response.active_priority = 0
        if pattern not in vg.LED_PATTERNS:
            response.success, response.error_code = False, "UNSUPPORTED_PATTERN"
            response.message = f"pattern {pattern} is not supported"
            return response
        result = self._set_led.call(
            vg.led_request(
                self,
                pattern=vg.LED_PATTERNS[pattern],
                rgb=(int(request.r), int(request.g), int(request.b)),
                priority=int(request.priority),
                preempt=bool(request.preempt),
                trace_id=str(request.trace_id),
            )
        )
        status_code = int(getattr(result.response, "status_code", 0)) if result.response is not None else -1
        # 0x1024: the platform's own light manager outranked this request.
        if status_code == 0x1024:
            response.success, response.error_code = False, "INSUFFICIENT_PRIORITY"
            response.message = (
                "platform light manager holds a higher priority; it does not report the "
                "effective value in this response (it is published on the vendor's led_state topic)"
            )
            return response
        response.success = result.ok and status_code == 0
        response.error_code = "" if response.success else "REJECTED"
        response.message = result.message
        if response.success:
            response.active_priority = int(request.priority)
        return response

    def _on_relocalize(self, request, response):
        from aimdk_robot.localization import relocalize  # noqa: PLC0415

        return relocalize(self, request, response)

    # --- command channels ---------------------------------------------------

    def _on_stream(self, channel: dict[str, Any], message: Float64MultiArray) -> None:
        name = str(channel["channel"])
        if not self._channel_active(channel):
            self._modes.note_rejected(name)
            self._publish_status()
            return
        joints = [str(joint) for joint in channel.get("joints", [])]
        try:
            if name == "arm_stream":
                payload = projection.arm_payload(message.data, channel_joints=joints)
                with self._lock:
                    self._arm_cmd = payload
                    self._arm_stamp = time.monotonic()
            elif name == "head_stream":
                payload = projection.head_payload(message.data, channel_joints=joints, limits=self._head_limits)
                with self._lock:
                    self._head_cmd = payload
                    self._head_stamp = time.monotonic()
            elif name in ("gripper_stream", "hand_stream", "hand_gesture_stream"):
                hand = self._vendor.get("hand") or {}
                sub_mode, payload = projection.hand_payload(
                    message.data,
                    hand_type=_CHANNEL_HAND_TYPES[name],
                    command_min=float(hand.get("command_min", 0.0)),
                    command_max=float(hand.get("command_max", 1.0)),
                    gesture_ids=hand.get("gesture_ids"),
                )
                with self._lock:
                    self._hand_cmd = (sub_mode, payload)
                    self._hand_stamp = time.monotonic()
            else:
                self._modes.note_rejected(name)
                return
        except CommandRejected as rejected:
            self._modes.note_rejected(f"{name}:{rejected.reason}")
            self.get_logger().warning(f"rejected {name} command: {rejected}")
            self._publish_status()

    def _on_twist(self, name: str, message: Twist) -> None:
        channel = next(
            (c for c in self._profile.get("command_channels", []) if str(c.get("channel")) == name),
            {},
        )
        if not self._channel_active(channel):
            self._modes.note_rejected(name)
            self._publish_status()
            return
        try:
            # The platform narrows its own speed envelope with battery, load
            # and terrain and reports it in every state message; a static table
            # would either accept what it will refuse or refuse what it allows.
            with self._lock:
                envelope = self._speed_envelope
            triple = projection.twist_to_locomotion(
                message.linear.x,
                message.linear.y,
                message.angular.z,
                self._locomotion.with_platform_envelope(envelope),
            )
        except CommandRejected as rejected:
            self._modes.note_rejected(f"{name}:{rejected.reason}")
            self.get_logger().warning(f"rejected {name} command: {rejected}")
            self._publish_status()
            return
        with self._lock:
            self._twist = triple
            self._twist_stamp = time.monotonic()

    def _channel_active(self, channel: dict[str, Any]) -> bool:
        if not self.output_admitted():
            return False
        modes = [str(mode) for mode in channel.get("modes", [])]
        if not modes or self._modes.mode not in modes:
            return False
        if self._is_twist_channel(channel) and self._state.has_capability("base.navigation_gate"):
            with self._lock:
                return self._navigation_enabled
        return True

    @staticmethod
    def _is_twist_channel(channel: dict[str, Any]) -> bool:
        return str(channel.get("type", "float64_array")).lower() == "twist"

    def _clear_commands(self) -> None:
        with self._lock:
            self._arm_cmd = None
            self._hand_cmd = None
            self._head_cmd = None
            self._twist = None

    # --- fixed-rate command publication -------------------------------------

    def _publish_commands(self) -> None:
        """Publish at the vendor's recommended rate while a command mode is active.

        The vendor's low-level tier has no fail-safe and its MC tier drops stale
        commands, so this runs on a timer rather than echoing subscriber
        callbacks. An absent upstream publisher becomes an explicit zero
        velocity for the base, and silence for the upper body: the platform
        then holds, and a frozen last target is never replayed.

        Admission is re-checked here, on the output, not only when a command
        arrived: a stop, a revoked navigation gate, lost arbitration or a
        blocking platform mode all take effect on the very next tick.
        """
        mode = self._modes.mode
        spec = self._modes.spec()
        with self._lock:
            if not spec.allows_stream or not self.output_admitted():
                return
            now = time.monotonic()
            self._sequence += 1
            sequence = self._sequence
            locomotion_mode = str((self._vendor.get("mc_actions") or {}).get(mode, "")) in (
                "STAND_DEFAULT",
                "LOCOMOTION_DEFAULT",
            )
            if locomotion_mode:
                if self._twist is None:
                    return
                gated = self._state.has_capability("base.navigation_gate") and not self._navigation_enabled
                staleness = float(self._state.capabilities.get("base.cmd_vel", {}).get("staleness_s", 0.5))
                twist = self._twist
                if gated or now - self._twist_stamp > staleness:
                    twist = (0.0, 0.0, 0.0)
                    self._twist = None
                self._locomotion_pub.publish(
                    vg.locomotion_command(
                        self,
                        source=self._source,
                        sequence=sequence,
                        forward=twist[0],
                        lateral=twist[1],
                        angular=twist[2],
                    )
                )
                return
            # Each channel expires on its own clock: an upstream publisher
            # that went quiet stops being repeated (its joints fall back to the
            # measured hold target below), and traffic on another channel does
            # not keep its last target alive.
            if self._arm_cmd is not None and now - self._arm_stamp > self._stream_staleness_s:
                self._arm_cmd = None
            if self._head_cmd is not None and now - self._head_stamp > self._stream_staleness_s:
                self._head_cmd = None
            if self._hand_cmd is not None and now - self._hand_stamp > self._stream_staleness_s:
                self._hand_cmd = None
            if self._arm_cmd is None and self._head_cmd is None and self._hand_cmd is None:
                return
            # Every element of arm_pos/head_pos is a target on this platform.
            # Joints this tick does not command are told to stay where the
            # platform measures them; without that measurement nothing is
            # sent, because a zero-filled vector would be a real motion.
            arm = self._arm_cmd if self._arm_cmd is not None else self._hold_target_locked(projection.VENDOR_ARM_ORDER)
            head = (
                self._head_cmd if self._head_cmd is not None else self._hold_target_locked(projection.VENDOR_HEAD_ORDER)
            )
            if arm is None or head is None:
                if not self._hold_target_missing:
                    self._hold_target_missing = True
                    self._modes.note_rejected("upper_body:HOLD_TARGET_UNAVAILABLE")
                    self.get_logger().warning(
                        "upper-body command withheld: no fresh measurement to hold the uncommanded joints"
                    )
                return
            self._hold_target_missing = False
            hand_sub_mode, hand_pos = (
                self._hand_cmd if self._hand_cmd is not None else (self._default_hand_sub_mode, [])
            )
            self._upper_body_pub.publish(
                vg.upper_body_command(
                    self,
                    source=self._source,
                    sequence=sequence,
                    hand_sub_mode=hand_sub_mode,
                    head_pos=head,
                    arm_pos=arm,
                    hand_pos=hand_pos,
                )
            )

    def _hold_target_locked(self, order: tuple[str, ...]) -> list[float] | None:
        """Fresh measured positions in vendor order, or None. Caller holds _lock."""
        fresh, _stale = projection.partition_fresh(
            self._feedback_received, now_s=time.monotonic(), window_s=self._feedback_window_s
        )
        groups = {group: entries for group, entries in self._joint_groups.items() if group in fresh}
        return projection.hold_target(groups, order=order)

    def publish_safe_command(self) -> None:
        """Explicit zero before we stop publishing (the tier has no fail-safe)."""
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        self._locomotion_pub.publish(
            vg.locomotion_command(self, source=self._source, sequence=sequence, forward=0.0, lateral=0.0, angular=0.0)
        )

    # --- vendor state -------------------------------------------------------

    def _on_joint_state(self, group: str, message: Any) -> None:
        entries = [
            (str(joint.name), float(joint.position), float(joint.velocity), float(joint.effort))
            for joint in message.joints
        ]
        self._record_feedback(group, entries)

    def _on_hand_state(self, message: Any) -> None:
        with self._lock:
            self._hand_state_types = [int(message.left_hand_type.value), int(message.right_hand_type.value)]
        entries: list[tuple[str, float, float, float]] = []
        for side, hands in (("left", message.left_hands), ("right", message.right_hands)):
            for hand in hands:
                name = str(hand.name) or f"{side}_hand"
                entries.append((name, float(hand.position), float(hand.velocity), float(hand.effort)))
        self._record_feedback("hand", entries)
        self._publish_hand_touch(message)

    def _publish_hand_touch(self, message: Any) -> None:
        """Republish the hand touch arrays that arrive on this same message.

        The dexterous hands carry the densest contact sensing on the robot —
        palm, back and five fingertips per hand — and it rides along with the
        joint state the bridge already subscribes to. Dropping it left grasping
        and contact-aware interaction with no tactile source at all.

        The pads have different cell counts, so the layout names each one with
        its size rather than forcing them into a fixed-width matrix.
        """
        publisher = getattr(self, "_hand_touch_pub", None)
        if publisher is None:
            return
        frames = vg.hand_touch_frames(message)
        if not frames:
            return
        from std_msgs.msg import MultiArrayDimension  # noqa: PLC0415

        array = Float64MultiArray()
        offset = sum(len(cells) for cells in frames.values())
        for pad in sorted(frames):
            cells = frames[pad]
            dimension = MultiArrayDimension()
            dimension.label = pad
            dimension.size = len(cells)
            dimension.stride = offset
            offset -= len(cells)
            array.layout.dim.append(dimension)
            array.data.extend(float(cell) for cell in cells)
        publisher.publish(array)

    def _record_feedback(self, group: str, entries: list[tuple[str, float, float, float]]) -> None:
        with self._lock:
            self._joint_groups[group] = entries
            self._feedback_received[group] = time.monotonic()
            self._feedback_clock[group] = self.get_clock().now()

    def _fresh_feedback_locked(self) -> tuple[dict[str, list[tuple[str, float, float, float]]], Any]:
        """Fresh groups and the receipt time of the oldest one. Caller holds _lock."""
        fresh, _stale = projection.partition_fresh(
            self._feedback_received, now_s=time.monotonic(), window_s=self._feedback_window_s
        )
        groups = {group: list(entries) for group, entries in self._joint_groups.items() if group in fresh}
        oldest = min((self._feedback_clock[group] for group in groups), default=None)
        return groups, oldest

    def _publish_joint_states(self) -> None:
        with self._lock:
            groups, oldest = self._fresh_feedback_locked()
        if not groups or oldest is None:
            return
        # The stamp is the oldest measurement in the aggregate, not "now": a
        # source that stopped reporting drops out above instead of being
        # re-stamped, and what remains is never newer than its stalest part.
        stamp = oldest.to_msg()
        public_order = [str(joint) for joint in self._profile["joints"]]
        names, positions, velocities, efforts = projection.aggregate_joint_state(groups, order=public_order)
        if names:
            message = JointState()
            message.header.stamp = stamp
            message.name, message.position, message.velocity, message.effort = names, positions, velocities, efforts
            self._joint_pub.publish(message)

        body_interface = (self._profile.get("interfaces") or {}).get("joint.body_state") or {}
        body_order = [str(joint) for joint in body_interface.get("joint_names", [])]
        if not body_order:
            return
        names, positions, velocities, efforts = projection.aggregate_joint_state(groups, order=body_order)
        if not names:
            return
        body = JointState()
        body.header.stamp = stamp
        body.name, body.position, body.velocity, body.effort = names, positions, velocities, efforts
        self._body_pub.publish(body)

    def _check_feedback(self) -> None:
        """A required feedback source that went quiet degrades the runtime."""
        with self._lock:
            received = dict(self._feedback_received)
            for group in self._required_feedback:
                received.setdefault(group, float("-inf"))
            _fresh, stale = projection.partition_fresh(
                received, now_s=time.monotonic(), window_s=self._feedback_window_s
            )
        missing = sorted(group for group in stale if group in self._required_feedback)
        if missing:
            self._set_fault("feedback", f"joint feedback stale: {', '.join(missing)}")
        else:
            self._set_fault("feedback", None)

    def _on_mc_state(self, message: Any) -> None:
        """Read the whole state message, not the two fields the bridge started with.

        Every field here arrives on a topic already subscribed, so using them
        costs no extra traffic and removes the lag of asking for the same facts
        over services.
        """
        action = str(message.action_info.action_desc or "")
        if not action:
            action = vg.MC_ACTION_BY_VALUE.get(int(message.action_info.current_action.value), "")
        stamp = message.header.stamp
        remote_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        skew_s = self.get_clock().now().nanoseconds * 1e-9 - remote_s if remote_s > 0.0 else None
        holder = str(getattr(message.input_source, "name", "") or "")
        holder_priority = int(getattr(message.input_source, "priority", 0) or 0)
        fsm_state = int(getattr(message.fsm_state, "current_state", projection.FSM_STATE_UNKNOWN))
        envelope = vg.speed_envelope(message.speed_status)
        player_state = int(getattr(getattr(message.motion_status, "player_state", None), "value", 0))
        hand_statuses = vg.hand_statuses(message.runtime_model)
        with self._lock:
            self._vendor_action = action
            self._body_pose = int(message.body_status.value)
            self._speed_envelope = envelope
            self._player_state = player_state
            self._hand_statuses = hand_statuses
            if skew_s is not None:
                # One sample is offset + however long this callback waited, so
                # it is kept with its arrival time and judged against its peers
                # rather than on its own (see projection.clock_skew_estimate).
                self._clock_samples.append((time.monotonic(), skew_s))
        self._note_arbitration_holder(holder, holder_priority)
        self._set_fault("balance", projection.fsm_fault(fsm_state))

    def _note_arbitration_holder(self, holder: str, holder_priority: int) -> None:
        """Apply the holder the platform reports with every state message.

        The same fact the arbitration poll asks for arrives here ten times a
        second, so a preemption is seen at once instead of up to a poll period
        later — and every command published in that window would have been
        discarded.
        """
        if not holder or holder == self._source:
            return
        if holder_priority < self._source_priority:
            return
        with self._lock:
            already_lost = not self._arbitration_ok
            self._arbitration_ok = False
        if not already_lost:
            self._revoke_output()
        self._set_fault(
            "arbitration",
            f"command arbitration held by {holder!r} at priority {holder_priority} "
            f"(this runtime registers {self._source_priority}); streamed commands are not applied",
        )

    def _check_clock_skew(self) -> None:
        """Compare this host's clock with the platform's own state stamps."""
        horizon = time.monotonic() - _CLOCK_SKEW_WINDOW_S
        with self._lock:
            samples = [skew for received, skew in self._clock_samples if received >= horizon]
        estimate = projection.clock_skew_estimate(samples)
        if estimate is None:
            # No recent platform stamps. Silence is a feedback problem, which
            # _check_feedback reports; it is not evidence about the clock.
            return
        detail = projection.clock_skew_fault(estimate, limit_s=self._clock_skew_limit_s, window_s=self._stamp_window_s)
        self._set_fault("clock", detail, lifecycle=LIFECYCLE_FAULTED)

    def _on_system_state(self, message: Any) -> None:
        state = str(message.cur_state or "")
        with self._lock:
            previous, self._system_state = self._system_state, state
        if state == previous:
            return
        blocking = set(
            self._vendor.get("blocking_system_states") or ["Develop_MC", "EStop", "OTA", "Poweroff", "Reboot"]
        )
        if state in blocking:
            self._revoke_output()
            self._set_fault(
                "system",
                f"platform system mode {state!r} disables the interfaces this runtime depends on",
                lifecycle=LIFECYCLE_DEGRADED if state == "EStop" else LIFECYCLE_FAULTED,
            )
        else:
            self._set_fault("system", None)

    def _on_power_state(self, message: Any) -> None:
        power = PowerState()
        power.header.stamp = self.get_clock().now().to_msg()
        power.battery_present = True
        power.battery_percentage = float(message.battery_remaining_capacity_percentage)
        power.battery_voltage = float(message.battery_pack_voltage)
        power.battery_current = float(message.battery_current)
        power.battery_temperature = float(message.battery_temperature)
        power.battery_power = float(message.battery_output_power)
        power.battery_cycle_count = int(message.battery_cycle_count)
        power.charging = float(message.battery_current) > 0.0
        power.temperature = float(message.pmu_temperature)
        power.fan_speed_rpm = float(message.fan_speed)
        power.fan_percentage = float(message.fan_pecentage)
        power.rail_names = ["48v_bus", "12v_output", "orin", "rk3588"]
        power.rail_voltages = [
            float(message.bus_48v_voltage),
            float(message.output_12v_voltage),
            float(message.orin_voltage),
            float(message.rk3588_voltage),
        ]
        power.rail_currents = [
            float(message.bus_48v_current),
            float(message.output_12v_current),
            float(message.orin_current),
            float(message.rk3588_current),
        ]
        self._power_pub.publish(power)

    def _on_diagnostics(self, message: Any) -> None:
        self._publish_diagnostics(
            [(str(entry.info), f"{int(entry.diag_code)}", DiagnosticStatus.OK) for entry in message.diagnostics],
            prefix="aimdk/diagnostic",
        )

    def _on_alerts(self, message: Any) -> None:
        self._publish_diagnostics(
            [("platform alert", str(entry.code), DiagnosticStatus.ERROR) for entry in message.list],
            prefix="aimdk/alert",
        )

    def _publish_diagnostics(self, entries: list[tuple[str, str, int]], *, prefix: str) -> None:
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        for detail, code, level in entries:
            status = DiagnosticStatus()
            status.name = f"{prefix}/{code}"
            status.hardware_id = self._state.name
            status.level = bytes([level])
            status.message = detail
            status.values = [KeyValue(key="code", value=code)]
            array.status.append(status)
        self._diag_pub.publish(array)

    def _on_audio_capture(self, message: Any) -> None:
        stamped = self._audio_stamped_type()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.audio.data = list(message.data.data)
        self._audio_pub.publish(stamped)
        info = self._audio_info_type()
        info.channels = int(message.info.channels)
        info.sample_rate = int(message.info.sample_rate)
        info.sample_format = str(message.info.sample_format)
        info.coding_format = str(message.info.coding_format)
        self._audio_info_pub.publish(info)

    def _on_audio_play(self, message: Any) -> None:
        chunk = list(message.audio.data)
        with self._audio_lock:
            self._last_play_s = time.monotonic()
            if self._focus_held:
                publish = True
            else:
                # Hold the head of the stream rather than drop it: focus takes
                # a service round trip, and an utterance that starts mid-word
                # is worse than one that starts a fraction of a second late.
                self._playback_buffer.append(chunk)
                publish = False
        if publish:
            self._publish_playback(chunk)

    def _publish_playback(self, chunk: list[int]) -> None:
        audio = self._vendor.get("audio") or {}
        playback = self._playback_type()
        playback.stamps = self.get_clock().now().to_msg()
        playback.info.channels = int(audio.get("playback_channels", 1))
        playback.info.sample_rate = int(audio.get("playback_sample_rate", 16000))
        playback.info.sample_format = str(audio.get("playback_sample_format", "S16LE"))
        playback.info.coding_format = str(audio.get("playback_coding_format", "pcm"))
        playback.data.data = chunk
        # pkg_name identifies the playback source; a change flushes the
        # platform's buffer, so every chunk of one stream carries the same one.
        playback.pkg_name = self._source
        self._playback_pub.publish(playback)

    def _audio_focus_tick(self) -> None:
        """Hold audio focus for exactly as long as there is audio to play.

        ``hal_audio`` never requests focus on a publisher's behalf, so nothing
        published here is guaranteed to be heard until this succeeds. Focus is
        released once the stream goes quiet: holding it idle would keep
        preempting whatever else the platform wants to say.
        """
        now = time.monotonic()
        with self._audio_lock:
            wants_focus = bool(self._playback_buffer) or (now - self._last_play_s) < self._focus_release_idle_s
            held = self._focus_held
            retry_due = (now - self._focus_attempt_s) >= _AUDIO_FOCUS_RETRY_S
        if wants_focus and not held and retry_due:
            self._acquire_audio_focus()
        elif held and not wants_focus:
            self.release_audio_focus()

    def _acquire_audio_focus(self) -> None:
        self._focus_attempt_s = time.monotonic()
        result = self._request_focus.call(
            vg.audio_focus_request(
                self, pkg_name=self._source, priority=self._focus_priority, weight=self._focus_weight
            )
        )
        if not (result.ok and vg.focus_granted(result.response)):
            # The status code says only that the request was processed, so the
            # grant flag is the answer. Refused means something with a higher
            # priority owns the speaker; buffered audio is already stale.
            with self._audio_lock:
                self._playback_buffer.clear()
            self.get_logger().warning(
                f"audio focus refused (priority {self._focus_priority}): playback is not reaching the speaker",
                throttle_duration_sec=10.0,
            )
            return
        with self._audio_lock:
            self._focus_held = True
            pending = list(self._playback_buffer)
            self._playback_buffer.clear()
        for chunk in pending:
            self._publish_playback(chunk)

    def release_audio_focus(self) -> None:
        """Give the speaker back. All three requester fields must match."""
        if self._abandon_focus is None:
            return
        with self._audio_lock:
            if not self._focus_held:
                return
            self._focus_held = False
            self._playback_buffer.clear()
        self._abandon_focus.call(
            vg.audio_focus_release(
                self, pkg_name=self._source, priority=self._focus_priority, weight=self._focus_weight
            )
        )

    def _on_audio_focus(self, message: Any) -> None:
        if str(message.pkg_name) != self._source:
            return
        gained = bool(message.focus_gain)
        with self._audio_lock:
            if self._focus_held == gained:
                return
            self._focus_held = gained
            if not gained:
                self._playback_buffer.clear()
        if not gained:
            self.get_logger().warning("audio focus lost to a higher-priority source; playback stopped")

    def _on_localization(self, message: Any) -> None:
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = str(message.header.frame_id or self._localization_frame)
        pose.pose.pose = message.pose.pose
        pose.pose.covariance = message.pose.covariance
        self._localization_pub.publish(pose)

    # --- status -------------------------------------------------------------

    def _publish_status(self) -> None:
        self._status_pub.publish(self._state.to_msg(self.get_clock().now().to_msg()))

    # --- accessors used by helper modules ----------------------------------

    @property
    def profile(self) -> dict[str, Any]:
        return self._profile

    @property
    def runtime_state(self) -> RuntimeState:
        return self._state

    @property
    def vendor_config(self) -> dict[str, Any]:
        return self._vendor

    @property
    def preset_motion_caller(self) -> vg.ServiceCaller:
        return self._preset_motion

    @property
    def preset_state_caller(self) -> vg.ServiceCaller:
        return self._preset_state

    @property
    def action_caller(self) -> vg.ServiceCaller:
        return self._set_action

    @property
    def service_callback_group(self) -> Any:
        return self._service_cb

    @property
    def action_callback_group(self) -> Any:
        """Long-running action execution must not block the mode/stop services."""
        return self._client_cb

    @property
    def input_source(self) -> str:
        return self._source

    def ensure_input_source(self) -> bool:
        """Re-assert the vendor input source claim, as the SDK's clients do.

        A registration made at startup is not permanent — a stop releases it —
        and the platform discards commands from a source that is not currently
        enabled. Taking the claim again costs one service round trip and is
        what the vendor's own preset-motion client does before every dispatch.
        """
        return self._register_input_source()

    def current_body_pose(self) -> int:
        with self._lock:
            return self._body_pose

    def platform_action(self) -> str:
        """The MC action the platform itself reports, not the mode we committed."""
        with self._lock:
            return self._vendor_action

    def current_mode(self) -> str:
        return self._modes.mode


def main(argv=None) -> int:
    rclpy.init(args=argv if argv is not None else sys.argv)
    try:
        node = AimdkRuntimeNode()
    except Exception as exc:  # noqa: BLE001 - startup failure must be legible
        print(f"aimdk_runtime failed to start: {exc}", file=sys.stderr)
        rclpy.shutdown()
        return 1
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # The vendor command tier has no fail-safe: leave an explicit zero
        # behind even when shutdown is already going wrong.
        with contextlib.suppress(Exception):
            node.publish_safe_command()
        with contextlib.suppress(Exception):
            node.release_audio_focus()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

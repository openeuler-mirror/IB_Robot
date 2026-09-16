"""The only module that imports vendor (``aimdk_msgs``) types.

Keeping the vendor dependency here means the projection logic, the profile and
the tests that cover them run without the AimDK SDK installed, and that no
generic IB-Robot package ever links a vendor type. Builders are pure functions
of vendor message classes, so they can be exercised against the real generated
types without a robot.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from aimdk_robot import projection

try:  # pragma: no cover - import guard exercised by the missing-SDK test
    from aimdk_msgs.msg import (  # type: ignore[import-not-found]
        McActionCommand,
        McInputAction,
        McInputSource,
        McLocomotionVelocity,
        MessageHeader,
        TtsPriorityLevel,
        UpperBodyCommandArray,
    )
    from aimdk_msgs.srv import (  # type: ignore[import-not-found]
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

    VENDOR_AVAILABLE = True
    VENDOR_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - depends on the environment
    VENDOR_AVAILABLE = False
    VENDOR_IMPORT_ERROR = str(exc)


class VendorUnavailableError(RuntimeError):
    """Raised when the AimDK overlay is not on the ROS 2 path."""

    def __init__(self) -> None:
        super().__init__(
            "aimdk_msgs is not available: source the AimDK overlay before launching the X2 runtime "
            f"(see aimdk_robot/README.md). Import error: {VENDOR_IMPORT_ERROR}"
        )


def require_vendor() -> None:
    if not VENDOR_AVAILABLE:
        raise VendorUnavailableError()


# --- vendor enum values (aimdk_msgs/interface/robot/mc) ---------------------

MC_ACTION_VALUES: dict[str, int] = {
    "PASSIVE_DEFAULT": 1,
    "SOFT_EMERGENCY_STOP": 2,
    "DAMPING_DEFAULT": 3,
    "ZERO_TORQUE_DEFAULT": 4,
    "JOINT_DEFAULT": 100,
    "JOINT_FREEZE": 101,
    "STAND_DEFAULT": 200,
    "STAND_BODY_CONTROL": 201,
    "LOCOMOTION_DEFAULT": 300,
    "RUN_DEFAULT": 301,
    "LOCOMOTION_STEP": 302,
    "VR_REMOTE_CONTROLLER": 400,
    "SIT_DOWN_DEFAULT": 2000,
    "CROUCH_DOWN_DEFAULT": 2002,
    "LIE_DOWN_DEFAULT": 2004,
    "STAND_UP_DEFAULT": 2005,
    "ASCEND_STAIRS": 2006,
    "DESCEND_STAIRS": 2008,
}
# HEAD_ONLY and UPPERBODY_REMOTE_SPLIT are documented by name only; the SDK
# examples switch modes with action_desc, which is the supported path.
MC_ACTION_BY_VALUE: dict[int, str] = {value: name for name, value in MC_ACTION_VALUES.items()}

INPUT_ACTION_ADD = 1001
INPUT_ACTION_ENABLE = 2001

# HandType values (hal/msg/HandType.msg). NONE means "no hand state reported"
# (the hands are not enumerated yet, or none are installed) and ERROR is a
# platform-side hand fault: neither is a hand family, so neither can confirm
# or contradict what a profile declares.
HAND_TYPE_NONE = 0x00
HAND_TYPE_NIMBLE = 0x01
HAND_TYPE_CLAW = 0x02
HAND_TYPE_LEISAI_NIMBLE = 0x03
HAND_TYPE_LITE_S = 0x04
HAND_TYPE_ERROR = 0xFF
HAND_TYPE_NAMES: dict[int, str] = {
    HAND_TYPE_NONE: "none",
    HAND_TYPE_NIMBLE: "nimble_hands",
    HAND_TYPE_CLAW: "claw",
    HAND_TYPE_LEISAI_NIMBLE: "leisai_nimble_hands",
    HAND_TYPE_LITE_S: "lite_s_hands",
    HAND_TYPE_ERROR: "error",
}
#: Which end-effector family a reported hand type belongs to. Values absent
#: from this map (NONE, ERROR, anything a newer firmware adds) carry no family.
HAND_TYPE_FAMILIES: dict[int, str] = {
    HAND_TYPE_NIMBLE: "dexterous",
    HAND_TYPE_CLAW: "claw",
    HAND_TYPE_LEISAI_NIMBLE: "dexterous",
    HAND_TYPE_LITE_S: "dexterous",
}


def hand_type_family(value: int) -> str | None:
    """The end-effector family of a vendor hand type, or None if it names none."""
    return HAND_TYPE_FAMILIES.get(int(value))


def hand_type_name(value: int) -> str:
    return HAND_TYPE_NAMES.get(int(value), f"0x{int(value):02x}")


# LED strip modes (hal/srv/SetPmuLed.srv).
LED_PATTERNS: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3}


def _tts_priority_levels() -> tuple[tuple[int, int], ...]:
    """Neutral 0..100 thresholds -> vendor TTS levels, from the vendor's own constants.

    ``TtsPriorityLevel`` (interaction/msg/TtsPriorityLevel.msg) is an
    enumeration of scheduling layers, not a numeric range. The contract
    priority steps through the layers an application may use, in the
    vendor's own order. ``SAFETY_L10`` is the platform's life-safety layer and
    is deliberately unreachable from the contract: no neutral priority maps
    onto it.
    """
    if not VENDOR_AVAILABLE:  # pragma: no cover - the builders need the SDK anyway
        return ()
    return (
        (0, int(TtsPriorityLevel.BACKGROUND_L1)),
        (20, int(TtsPriorityLevel.SERVICE_L2)),
        (40, int(TtsPriorityLevel.MISSION_L4)),
        (60, int(TtsPriorityLevel.INTERACTION_L6)),
        (80, int(TtsPriorityLevel.SYSTEM_L7)),
        (95, int(TtsPriorityLevel.WARNING_L8)),
    )


TTS_PRIORITY_LEVELS: tuple[tuple[int, int], ...] = _tts_priority_levels()
#: Names of the levels above, in the same order, for the interaction.tts
#: capability parameters.
TTS_PRIORITY_LEVEL_NAMES: tuple[str, ...] = ("background", "service", "mission", "interaction", "system", "warning")

#: Neutral 0..100 thresholds -> screen (emoji/video) priorities. This scale is
#: not 0..100: the platform's own modules display fault indications at 8-10
#: (over-temperature 8, damped-fall / disabled-arm 10) and the documentation
#: states that anything above 10 overrides them — "会覆盖系统模块触发的故障表情
#: 提示（如摔倒保护、过温告警等）, 导致用户无法通过屏幕感知这些故障". Passing a
#: neutral priority through unmapped therefore hides fault indications for most
#: of the scale. Ordinary content stays below the fault layer; only the top of
#: the contract's range asks to be seen over it, and that is a deliberate act.
SCREEN_PRIORITY_LEVELS: tuple[tuple[int, int], ...] = ((0, 3), (40, 5), (80, 7), (95, 11))
SCREEN_PRIORITY_LEVEL_NAMES: tuple[str, ...] = ("background", "normal", "foreground", "over_fault")

#: Neutral 0..100 thresholds -> LED priorities. The platform keeps a *threshold*
#: that each accepted request raises ("新请求的 priority 大于等于门槛时才会被
#: 接受, 并将门槛更新为该值"), so a raw neutral priority ratchets the threshold
#: up and locks the caller out of its own later, lower-priority requests. A
#: small band keeps that headroom; `reset_priority` is the documented escape.
LED_PRIORITY_LEVELS: tuple[tuple[int, int], ...] = ((0, 0), (40, 2), (80, 5))
LED_PRIORITY_LEVEL_NAMES: tuple[str, ...] = ("background", "normal", "foreground")


def _stamp(node: Any) -> Any:
    return node.get_clock().now().to_msg()


def message_header(node: Any, *, frame_id: str, sequence: int) -> Any:
    header = MessageHeader()
    header.stamp = _stamp(node)
    header.frame_id = frame_id
    header.sequence = int(sequence) & 0xFFFFFFFF
    return header


def upper_body_command(
    node: Any,
    *,
    source: str,
    sequence: int,
    hand_sub_mode: int,
    head_pos: list[float],
    arm_pos: list[float],
    hand_pos: list[float],
) -> Any:
    """Build ``/mc/upper_body_command``.

    The vendor drops commands whose stamp is outside its acceptance window, so
    the stamp is always taken now, at publication time.
    """
    message = UpperBodyCommandArray()
    message.header = message_header(node, frame_id="mc_upper_body", sequence=sequence)
    message.source = source
    message.hand_sub_mode = int(hand_sub_mode)
    message.head_pos = [float(value) for value in head_pos]
    message.arm_pos = [float(value) for value in arm_pos]
    message.hand_pos = [float(value) for value in hand_pos]
    return message


def locomotion_command(node: Any, *, source: str, sequence: int, forward: float, lateral: float, angular: float) -> Any:
    """Build ``/aima/mc/locomotion/velocity``."""
    message = McLocomotionVelocity()
    message.header = message_header(node, frame_id="mc_locomotion", sequence=sequence)
    message.source = source
    message.forward_velocity = float(forward)
    message.lateral_velocity = float(lateral)
    message.angular_velocity = float(angular)
    return message


def set_action_request(node: Any, *, source: str, action_name: str) -> Any:
    """Build a ``SetMcAction`` request.

    ``action_desc`` carries the documented action name, which is the only path
    the platform still uses: the numeric ``action`` field is "v0.8.2开始不再使
    用". It is filled in when a legacy value exists, purely for an older build
    — and note that two of the registered modes (``HEAD_ONLY``,
    ``UPPERBODY_REMOTE_SPLIT``) have no enum value at all, so the numeric field
    is not a complete substitute for the name on any build.
    """
    request = SetMcAction.Request()
    request.header.stamp = _stamp(node)
    request.source = source
    command = McActionCommand()
    command.action_desc = action_name
    value = MC_ACTION_VALUES.get(action_name)
    if value is not None:
        command.action.value = int(value)
    request.command = command
    return request


def input_source_request(node: Any, *, name: str, priority: int, timeout_ms: int, action_value: int) -> Any:
    """Build a ``SetMcInputSource`` request (ADD, then ENABLE on failure)."""
    request = SetMcInputSource.Request()
    request.request.header.stamp = _stamp(node)
    request.action = McInputAction(value=int(action_value))
    source = McInputSource()
    source.name = name
    source.priority = int(priority)
    source.timeout = int(timeout_ms)
    request.input_source = source
    return request


def preset_motion_request(
    node: Any,
    *,
    motion_value: int,
    area_value: int,
    interrupt: bool,
    source: str = "",
    priority: int = 0,
    timeout_ms: int = 0,
) -> Any:
    """Build a ``SetMcPresetMotion`` request.

    ``header.stamp`` on this service is **not** a message timestamp: the vendor
    documents it as the play time — "stamp 用于指定播放时刻（UTC），为 0 时立即
    播放". Stamping it with this host's clock therefore schedules the motion at
    a UTC instant, which is only ever correct if this host's clock agrees with
    the robot's; on a separate compute pack it asks for a play time in the past
    (or the future), and the platform can accept the request and schedule
    nothing. Zero is the documented "play now" and is clock-independent, so
    that is what a "do this now" contract sends.

    ``input_source`` is documented as reserved, but the SDK example fills it
    with the registered source, so the same identity is carried here.
    """
    request = SetMcPresetMotion.Request()
    request.header.stamp.sec = 0
    request.header.stamp.nanosec = 0
    request.input_source.name = source
    request.input_source.priority = int(priority)
    request.input_source.timeout = int(timeout_ms)
    request.motion.value = int(motion_value)
    request.area.value = int(area_value)
    request.interrupt = bool(interrupt)
    return request


def preset_motion_state_request(node: Any, *, task_id: int) -> Any:
    """Build a ``GetMcPresetMotionState`` request for one dispatched task."""
    request = GetMcPresetMotionState.Request()
    request.request.header.stamp = _stamp(node)
    request.request.task_id = int(task_id)
    return request


def task_outcome(response: Any) -> tuple[int, int]:
    """``(task_id, state)`` from a ``CommonTaskResponse``-bearing response."""
    task = getattr(response, "response", None)
    if task is None:
        return 0, projection.TASK_STATE_UNKNOWN
    return int(getattr(task, "task_id", 0)), int(getattr(getattr(task, "state", None), "value", 0))


def speed_envelope(speed_status: Any) -> projection.SpeedEnvelope:
    """The platform's currently reported speed bounds.

    Published with every state message; the platform narrows it with battery,
    load and terrain, so it is the only envelope that is true right now.
    """
    mode = int(getattr(getattr(speed_status, "speed_mode", None), "value", 0) or 0)
    forward = getattr(speed_status, "forward_bounds", None)
    lateral = getattr(speed_status, "lateral_bounds", None)
    angular = getattr(speed_status, "angular_bounds", None)

    def reach(bounds: Any) -> float:
        if bounds is None:
            return 0.0
        return max(abs(float(getattr(bounds, "max_value", 0.0))), abs(float(getattr(bounds, "min_value", 0.0))))

    return projection.SpeedEnvelope(
        mode=mode,
        max_linear_mps=max(reach(forward), reach(lateral)),
        max_angular_radps=reach(angular),
    )


def hand_statuses(runtime_model: Any) -> list[int]:
    """``[left, right]`` hand status from the state stream (``McHandStatus``).

    A second, continuous source for what ``GetHandType`` answers once at
    startup — which matters because the hands enumerate later than the runtime
    starts.
    """
    return [
        int(getattr(getattr(runtime_model, "left_hand_status", None), "value", 0) or 0),
        int(getattr(getattr(runtime_model, "right_hand_status", None), "value", 0) or 0),
    ]


def hand_touch_frames(message: Any) -> dict[str, list[int]]:
    """Flatten the touch arrays of hands that actually have them, into named pads.

    ``HandTouchSensorData`` carries a 36-cell palm and back plus five 16-cell
    fingertips per hand. The arrays are fixed-length for the widest supported
    hand, so a narrower one (the Leisai hand has no back pad and 12-cell tips)
    simply leaves the tail zeroed.

    The fields exist on every hand state message, including a gripper's, where
    they are all zeros. A gripper has no tactile array, so those zeros are the
    absence of a sensor and not a measurement of no contact — publishing them
    would invent a sensor. Only a hand whose reported type belongs to the
    dexterous family contributes.
    """
    pads = ("palm", "back_of_hand", "thumb", "index_finger", "middle_finger", "ring_finger", "little_finger")
    frames: dict[str, list[int]] = {}
    sides = (
        ("left", "left_touch_sensors", "left_hand_type"),
        ("right", "right_touch_sensors", "right_hand_type"),
    )
    for side, sensor_attribute, type_attribute in sides:
        sensors = getattr(message, sensor_attribute, None)
        reported = int(getattr(getattr(message, type_attribute, None), "value", HAND_TYPE_NONE))
        if sensors is None or hand_type_family(reported) != "dexterous":
            continue
        for pad in pads:
            data = getattr(sensors, f"{pad}_touch_data", None)
            if data is not None:
                frames[f"{side}_{pad}"] = [int(cell) for cell in data]
    return frames


def tts_request(node: Any, *, text: str, priority: int, interrupt: bool, trace_id: str, domain: str) -> Any:
    request = PlayTts.Request()
    request.header.header.stamp = _stamp(node)
    request.tts_req.text = text
    request.tts_req.priority_level.value = projection.priority_level(priority, levels=TTS_PRIORITY_LEVELS)
    # priority_weight is documented "do not use unless a product requires it";
    # the contract does not, so it stays at the vendor default.
    request.tts_req.priority_weight = 0
    request.tts_req.domain = domain
    request.tts_req.trace_id = trace_id
    request.tts_req.is_interrupted = bool(interrupt)
    return request


def emoji_request(node: Any, *, emotion_id: int, loop: bool, priority: int) -> Any:
    """Build a ``PlayEmoji`` request.

    The vendor's play modes are ``EMOTION_MODE_ONCE`` / ``EMOTION_MODE_LOOP``
    (1 / 2); the contract's are 0 / 1, so the semantic ``loop`` flag is what
    crosses this boundary, never the raw enum value.
    """
    request = PlayEmoji.Request()
    request.header.header.stamp = _stamp(node)
    request.emotion_id = int(emotion_id)
    request.mode = int(PlayEmoji.Request.EMOTION_MODE_LOOP if loop else PlayEmoji.Request.EMOTION_MODE_ONCE)
    # Stepped, not passed through: the screen priority scale is not 0..100 and
    # the platform's fault indications live at 8-10 (see SCREEN_PRIORITY_LEVELS).
    request.priority = projection.priority_level(priority, levels=SCREEN_PRIORITY_LEVELS)
    return request


def audio_focus_request(node: Any, *, pkg_name: str, priority: int, weight: int) -> Any:
    """Build a ``RequestAudioFocus`` request for raw stream playback.

    ``hal_audio``'s playback subscription does not take focus for its clients,
    so a bridge that publishes audio without asking contends with whatever is
    already playing rather than being mixed with it.
    """
    request = RequestAudioFocus.Request()
    request.request.header.stamp = _stamp(node)
    request.focus_requester = _focus_requester(pkg_name, priority, weight)
    return request


def audio_focus_release(node: Any, *, pkg_name: str, priority: int, weight: int) -> Any:
    """Build an ``AbandonAudioFocus`` request.

    All three requester fields must repeat the values the matching
    ``RequestAudioFocus`` carried, or the platform cannot match the holder and
    the focus is never released.
    """
    request = AbandonAudioFocus.Request()
    request.request.header.stamp = _stamp(node)
    request.focus_requester = _focus_requester(pkg_name, priority, weight)
    return request


def _focus_requester(pkg_name: str, priority: int, weight: int) -> Any:
    requester = RequestAudioFocus.Request().focus_requester
    requester.pkg_name = str(pkg_name)
    requester.priority = int(priority)
    requester.priority_weight = int(weight)
    return requester


def focus_granted(response: Any) -> bool:
    """Whether a focus call actually handed over the speaker.

    The response status is ``SUCCESS`` whether or not focus was granted — it
    only reports that the request was processed — so the answer lives in
    ``focus_response.focus_gain`` and nowhere else.
    """
    focus = getattr(response, "focus_response", None)
    return bool(getattr(focus, "focus_gain", False))


def led_request(node: Any, *, pattern: int, rgb: tuple[int, int, int], priority: int, preempt: bool, trace_id: str):
    request = SetPmuLed.Request()
    request.request.header.stamp = _stamp(node)
    request.trace_id = trace_id
    request.led_strip_mode = int(pattern)
    request.r, request.g, request.b = (int(channel) for channel in rgb)
    # The platform ratchets a priority threshold upward on every accepted
    # request, so the neutral range is stepped into a small band rather than
    # passed through (see LED_PRIORITY_LEVELS).
    request.priority = projection.priority_level(priority, levels=LED_PRIORITY_LEVELS)
    request.reset_priority = bool(preempt)
    return request


def hand_type_request(node: Any) -> Any:
    request = GetHandType.Request()
    request.request.header.stamp = _stamp(node)
    return request


def current_input_source_request(node: Any) -> Any:
    request = GetCurrentInputSource.Request()
    request.request.header.stamp = _stamp(node)
    return request


# --- response helpers -------------------------------------------------------


def response_code(response: Any) -> int:
    """Vendor responses nest their code differently per service family."""
    for attribute in ("response", "header", "reponse"):  # 'reponse' is the SDK's own typo
        nested = getattr(response, attribute, None)
        if nested is None:
            continue
        header = getattr(nested, "header", None)
        if header is not None and hasattr(header, "code"):
            return int(header.code)
        if hasattr(nested, "code"):
            return int(nested.code)
    raise AttributeError(f"no response code in {type(response).__name__}")


def response_message(response: Any) -> str:
    for attribute in ("response", "header", "reponse"):
        nested = getattr(response, attribute, None)
        if nested is not None and hasattr(nested, "message"):
            return str(nested.message)
    return ""


@dataclass(frozen=True, slots=True)
class VendorCallResult:
    ok: bool
    code: int
    message: str
    response: Any = None


class ServiceCaller:
    """Call a vendor service with the SDK's documented robustness pattern.

    The vendor states plainly that "standard ROS does not handle cross-host
    services well" and every SDK example retries. A single failed call is
    therefore not evidence that the platform refused anything.
    """

    def __init__(self, node: Any, client: Any, *, attempts: int = 8, timeout_s: float = 0.25) -> None:
        self._node = node
        self._client = client
        self._attempts = int(attempts)
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()

    def ready(self, timeout_s: float = 0.0) -> bool:
        return bool(self._client.wait_for_service(timeout_sec=timeout_s))

    def call(self, request: Any) -> VendorCallResult:
        with self._lock:
            for _ in range(self._attempts):
                future = self._client.call_async(request)
                if _spin_until_done(self._node, future, self._timeout_s) and future.done():
                    response = future.result()
                    if response is None:
                        continue
                    try:
                        code = response_code(response)
                    except AttributeError:
                        code = 0
                    return VendorCallResult(
                        ok=code == 0, code=code, message=response_message(response), response=response
                    )
        return VendorCallResult(ok=False, code=-1, message="vendor service did not answer", response=None)


def _spin_until_done(node: Any, future: Any, timeout_s: float) -> bool:
    """Wait for a future without touching the executor.

    Vendor calls are made from timers, service callbacks and the startup
    thread while an executor is already spinning this node. Calling
    ``spin_until_future_complete`` from any of those re-enters the executor
    from a second thread and deadlocks, so the future is simply polled: the
    spinning executor is what completes it.
    """
    import time as _time

    deadline = _time.monotonic() + max(timeout_s, 0.0)
    while _time.monotonic() < deadline:
        if future.done():
            return True
        _time.sleep(0.005)
    return future.done()

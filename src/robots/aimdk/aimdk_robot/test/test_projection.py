"""Unit tests for the X2 projection rules.

These are the rules that decide what the robot is told to do, so they are
tested without ROS, without the vendor SDK and without a robot.
"""

from __future__ import annotations

import pytest
from aimdk_robot import projection
from aimdk_robot.projection import CommandRejected, LocomotionLimits, ProjectionError

ARM = list(projection.VENDOR_ARM_ORDER)
HEAD = list(projection.VENDOR_HEAD_ORDER)


# --- arm / head -------------------------------------------------------------


def test_arm_payload_reorders_into_vendor_index_order():
    """A profile may declare its own order; the wire order is the vendor's."""
    reversed_order = list(reversed(ARM))
    values = [float(index) for index in range(14)]
    payload = projection.arm_payload(values, channel_joints=reversed_order)
    assert payload == [float(13 - index) for index in range(14)]


def test_arm_payload_identity_when_orders_match():
    values = [0.1 * index for index in range(14)]
    assert projection.arm_payload(values, channel_joints=ARM) == pytest.approx(values)


def test_arm_payload_rejects_wrong_length():
    with pytest.raises(CommandRejected) as excinfo:
        projection.arm_payload([0.0] * 13, channel_joints=ARM)
    assert excinfo.value.reason == "ARM_COMMAND_LENGTH"


def test_arm_payload_rejects_unknown_joint_names():
    bad = ["not_a_joint"] + ARM[1:]
    with pytest.raises(CommandRejected) as excinfo:
        projection.arm_payload([0.0] * 14, channel_joints=bad)
    assert excinfo.value.reason == "ARM_COMMAND_JOINTS"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_commands_are_rejected(value):
    with pytest.raises(CommandRejected) as excinfo:
        projection.arm_payload([value] + [0.0] * 13, channel_joints=ARM)
    assert excinfo.value.reason == "NON_FINITE_COMMAND"


def test_head_payload_uses_yaw_then_pitch():
    payload = projection.head_payload([0.2, 0.1], channel_joints=["head_pitch_joint", "head_yaw_joint"])
    assert payload == [0.1, 0.2]


# --- end effectors ----------------------------------------------------------


def test_claw_payload_passes_in_range_values():
    sub_mode, payload = projection.hand_payload([0.0, 1.0], hand_type="claw")
    assert sub_mode == projection.HAND_SUB_MODE_CLAW
    assert payload == [0.0, 1.0]


def test_claw_payload_rejects_out_of_range_instead_of_clamping():
    """Clamping would execute a grasp the caller did not ask for."""
    with pytest.raises(CommandRejected) as excinfo:
        projection.claw_payload([1.5, 0.5])
    assert excinfo.value.reason == "GRIPPER_COMMAND_RANGE"
    assert "left" in excinfo.value.detail


def test_dexterous_joint_payload_requires_both_hands():
    sub_mode, payload = projection.hand_payload([0.0] * 20, hand_type="dexterous_joint")
    assert sub_mode == projection.HAND_SUB_MODE_DEXTEROUS_JOINT
    assert len(payload) == 20
    with pytest.raises(CommandRejected) as excinfo:
        projection.dexterous_joint_payload([0.0] * 10)
    assert excinfo.value.reason == "HAND_COMMAND_LENGTH"


def test_gesture_payload_validates_ids_and_openness():
    sub_mode, payload = projection.hand_payload(
        [1.0, 0.5, 2.0, 1.0], hand_type="dexterous_gesture", gesture_ids=[0, 1, 2, 3]
    )
    assert sub_mode == projection.HAND_SUB_MODE_DEXTEROUS_GESTURE
    assert payload == [1.0, 0.5, 2.0, 1.0]
    with pytest.raises(CommandRejected) as excinfo:
        projection.gesture_payload([9.0, 0.5, 1.0, 0.5], gesture_ids=[0, 1])
    assert excinfo.value.reason == "HAND_GESTURE_ID"
    with pytest.raises(CommandRejected) as excinfo:
        projection.gesture_payload([1.0, 1.5, 1.0, 0.5])
    assert excinfo.value.reason == "HAND_GESTURE_RANGE"


def test_unknown_hand_type_is_a_configuration_error():
    with pytest.raises(ProjectionError):
        projection.hand_payload([0.0, 0.0], hand_type="magnetic")


# --- locomotion -------------------------------------------------------------


def test_twist_below_deadband_becomes_exact_zero():
    limits = LocomotionLimits()
    assert projection.twist_to_locomotion(0.001, -0.002, 0.0, limits) == (0.0, 0.0, 0.0)


def test_twist_below_platform_minimum_is_rejected_not_raised_to_the_minimum():
    limits = LocomotionLimits()
    with pytest.raises(CommandRejected) as excinfo:
        projection.twist_to_locomotion(0.1, 0.0, 0.0, limits)
    assert excinfo.value.reason == "VELOCITY_BELOW_MIN"
    assert "would not move the robot" in excinfo.value.detail


def test_twist_above_maximum_is_rejected():
    limits = LocomotionLimits()
    with pytest.raises(CommandRejected) as excinfo:
        projection.twist_to_locomotion(0.0, 0.0, 2.0, limits)
    assert excinfo.value.reason == "VELOCITY_ABOVE_MAX"


def test_twist_in_band_passes_through():
    limits = LocomotionLimits()
    assert projection.twist_to_locomotion(0.5, -0.3, 0.4, limits) == (0.5, -0.3, 0.4)


def test_locomotion_limits_from_profile():
    limits = LocomotionLimits.from_profile({"min_linear_mps": 0.3, "max_linear_mps": 0.9})
    assert limits.min_linear_mps == 0.3
    assert limits.max_linear_mps == 0.9
    assert limits.min_angular_radps == 0.1


# --- freshness --------------------------------------------------------------


def test_stamp_outside_the_window_is_stale():
    assert projection.is_fresh(100.0, 100.1, 0.2)
    assert not projection.is_fresh(100.0, 100.5, 0.2)
    # Stamp 0 means "use receive time" on this platform.
    assert projection.is_fresh(0.0, 100.0, 0.2)


# --- state aggregation ------------------------------------------------------


def test_aggregate_joint_state_follows_the_public_order():
    groups = {
        "arm": [("right_elbow_joint", 1.0, 0.1, 0.2), ("left_elbow_joint", 2.0, 0.0, 0.0)],
        "hand": [("left_hand", 0.5, 0.0, 0.0)],
    }
    names, positions, _velocities, _efforts = projection.aggregate_joint_state(
        groups, order=["left_elbow_joint", "left_hand", "right_elbow_joint"]
    )
    assert names == ["left_elbow_joint", "left_hand", "right_elbow_joint"]
    assert positions == [2.0, 0.5, 1.0]


def test_aggregate_joint_state_omits_absent_joints_instead_of_zero_filling():
    """A fabricated zero reads downstream as a real measurement."""
    names, positions, _v, _e = projection.aggregate_joint_state(
        {"arm": [("left_elbow_joint", 1.0, 0.0, 0.0)]}, order=["left_elbow_joint", "missing_joint"]
    )
    assert names == ["left_elbow_joint"]
    assert positions == [1.0]


def test_partition_fresh_separates_quiet_sources_and_treats_silence_as_stale():
    received = {"arm": 10.0, "hand": 9.4, "never": float("-inf")}
    fresh, stale = projection.partition_fresh(received, now_s=10.0, window_s=0.5)
    assert fresh == {"arm"}
    assert stale == {"hand", "never"}


def test_hold_target_requires_every_joint_measured():
    """A partial measurement is no basis for a fixed-length target vector."""
    order = ("a", "b")
    assert projection.hold_target({"g": [("a", 0.1, 0, 0), ("b", 0.2, 0, 0)]}, order=order) == [0.1, 0.2]
    assert projection.hold_target({"g": [("a", 0.1, 0, 0)]}, order=order) is None
    assert projection.hold_target({}, order=order) is None


# --- modes ------------------------------------------------------------------


def test_mode_action_resolves_from_profile():
    profile = {"vendor": {"mc_actions": {"stream": "UPPERBODY_REMOTE_SPLIT"}}}
    assert projection.mode_action(profile, "stream") == "UPPERBODY_REMOTE_SPLIT"


def test_mode_action_rejects_unknown_mode_and_unknown_action():
    with pytest.raises(ProjectionError):
        projection.mode_action({"vendor": {"mc_actions": {}}}, "stream")
    with pytest.raises(ProjectionError):
        projection.mode_action({"vendor": {"mc_actions": {"stream": "NOT_AN_ACTION"}}}, "stream")


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        (0, ""),
        (4, "INVALID_POSTURE"),
        (6, "SAFETY_FORBIDS_TRANSITION"),
        (8, "MOVING_BUSY"),
        (10, "NO_TRANSITION_PATH"),
    ],
)
def test_platform_rejection_codes_are_surfaced_verbatim(code, reason):
    assert projection.mode_rejection_reason(code) == reason


def test_unknown_rejection_code_is_reported_not_swallowed():
    assert projection.mode_rejection_reason(77) == "PLATFORM_CODE_77"


# --- stop -------------------------------------------------------------------


def test_hold_stop_switches_no_mode_when_the_platform_registers_no_hold_action():
    """Closing admission is the hold; no registered mode holds every posture."""
    plan = projection.resolve_stop_plan("HOLD", body_pose=projection.BODY_POSE_STAND, stop_config={})
    assert plan.vendor_action == ""
    assert plan.latch and plan.release_arbitration


def test_hold_stop_uses_a_configured_action_when_one_is_registered():
    plan = projection.resolve_stop_plan(
        "HOLD", body_pose=projection.BODY_POSE_STAND, stop_config={"hold_action": "JOINT_DEFAULT"}
    )
    assert plan.vendor_action == "JOINT_DEFAULT"


def test_stop_actions_the_firmware_does_not_register_are_refused():
    """The McAction enum is wider than the firmware's registered mode set.

    JOINT_FREEZE, SOFT_EMERGENCY_STOP and ZERO_TORQUE_DEFAULT are all in the
    IDL and none are registered; the platform answers code 3 for them. A
    profile naming one must fail here, not on the robot.
    """
    for name in ("JOINT_FREEZE", "SOFT_EMERGENCY_STOP", "ZERO_TORQUE_DEFAULT", "STAND_BODY_CONTROL"):
        assert name not in projection.VENDOR_MOTION_MODES
        with pytest.raises(ProjectionError):
            projection.resolve_stop_plan(
                "HOLD", body_pose=projection.BODY_POSE_STAND, stop_config={"hold_action": name}
            )
        with pytest.raises(ProjectionError):
            projection.resolve_stop_plan(
                "TORQUE_OFF", body_pose=projection.BODY_POSE_STAND, stop_config={"soft_estop_action": name}
            )


def test_torque_off_defaults_to_the_platforms_damping_mode_and_says_so():
    """DAMPING_DEFAULT is registered and "always permitted"; the enum's soft
    e-stop name is neither."""
    plan = projection.resolve_stop_plan("TORQUE_OFF", body_pose=projection.BODY_POSE_STAND, stop_config={})
    assert plan.vendor_action == "DAMPING_DEFAULT"
    assert plan.downgraded_from == "TORQUE_OFF"


def test_torque_off_while_standing_is_refused_when_zero_torque_is_configured():
    """Removing torque from a balancing biped drops it."""
    with pytest.raises(CommandRejected) as excinfo:
        projection.resolve_stop_plan(
            "TORQUE_OFF",
            body_pose=projection.BODY_POSE_STAND,
            stop_config={"torque_off_policy": "zero_torque_when_seated"},
        )
    assert excinfo.value.reason == "TORQUE_OFF_UNSAFE_POSTURE"
    assert "stand" in excinfo.value.detail


@pytest.mark.parametrize(
    "pose",
    [projection.BODY_POSE_SIT, projection.BODY_POSE_SQUAT, projection.BODY_POSE_LIE_FACE_UP],
)
def test_torque_off_is_allowed_in_a_supported_posture(pose):
    plan = projection.resolve_stop_plan(
        "TORQUE_OFF", body_pose=pose, stop_config={"torque_off_policy": "zero_torque_when_seated"}
    )
    assert plan.vendor_action == "PASSIVE_DEFAULT"
    assert plan.downgraded_from == ""


def test_unknown_stop_policy_is_a_configuration_error():
    with pytest.raises(ProjectionError):
        projection.resolve_stop_plan("COAST", body_pose=projection.BODY_POSE_STAND, stop_config={})


# --- interaction helpers ----------------------------------------------------


LEVELS = ((0, 1), (20, 2), (40, 4), (60, 6), (80, 7), (95, 8))


def test_priority_maps_onto_the_platform_levels_as_a_step_function():
    """Platform priorities are an enumeration: only listed values may come out."""
    assert projection.priority_level(0, levels=LEVELS) == 1
    assert projection.priority_level(19, levels=LEVELS) == 1
    assert projection.priority_level(20, levels=LEVELS) == 2
    assert projection.priority_level(50, levels=LEVELS) == 4
    assert projection.priority_level(79, levels=LEVELS) == 6
    assert projection.priority_level(94, levels=LEVELS) == 7
    assert projection.priority_level(100, levels=LEVELS) == 8
    legal = {value for _threshold, value in LEVELS}
    assert all(projection.priority_level(n, levels=LEVELS) in legal for n in range(101))
    # Order is preserved, and the top of the neutral range is the last listed
    # level: nothing above it (the platform's own safety layer) is reachable.
    outputs = [projection.priority_level(n, levels=LEVELS) for n in range(101)]
    assert outputs == sorted(outputs)
    assert max(outputs) == 8


def test_priority_outside_the_neutral_range_is_rejected():
    with pytest.raises(CommandRejected) as excinfo:
        projection.priority_level(120, levels=LEVELS)
    assert excinfo.value.reason == "PRIORITY_RANGE"


def test_priority_levels_must_be_ordered_and_start_at_zero():
    with pytest.raises(ProjectionError):
        projection.priority_level(10, levels=((5, 1), (20, 2)))
    with pytest.raises(ProjectionError):
        projection.priority_level(10, levels=((0, 1), (20, 2), (20, 4)))


def test_resolve_named_refuses_unknown_names_and_lists_the_known_ones():
    with pytest.raises(CommandRejected) as excinfo:
        projection.resolve_named("moonwalk", {"wave": 1002}, reason="UNKNOWN_MOTION")
    assert excinfo.value.reason == "UNKNOWN_MOTION"
    assert "wave" in excinfo.value.detail


# --- vendor hand types ------------------------------------------------------


def test_hand_type_families_cover_the_vendor_enumeration():
    """HandType.msg has six values; only four name an end-effector family."""
    from aimdk_robot import vendor_gateway as vg

    assert vg.hand_type_family(vg.HAND_TYPE_CLAW) == "claw"
    assert vg.hand_type_family(vg.HAND_TYPE_NIMBLE) == "dexterous"
    assert vg.hand_type_family(vg.HAND_TYPE_LEISAI_NIMBLE) == "dexterous"
    assert vg.hand_type_family(vg.HAND_TYPE_LITE_S) == "dexterous"
    # Absence of hand state and a hand fault are not families: they can
    # neither confirm nor contradict what a profile declares.
    assert vg.hand_type_family(vg.HAND_TYPE_NONE) is None
    assert vg.hand_type_family(vg.HAND_TYPE_ERROR) is None
    # A value a newer firmware adds is unknown, not a mismatch.
    assert vg.hand_type_family(0x7E) is None
    assert vg.hand_type_name(vg.HAND_TYPE_NONE) == "none"
    assert vg.hand_type_name(0x7E) == "0x7e"


# --- vendor QoS -------------------------------------------------------------


def test_vendor_qos_takes_the_profile_value_then_the_default():
    config = {"audio_playback": {"reliability": "reliable"}, "mc_state": {"durability": "transient_local", "depth": 5}}
    assert projection.vendor_qos(config, "audio_playback") == ("reliable", "volatile", 10)
    assert projection.vendor_qos(config, "mc_state") == ("best_effort", "transient_local", 5)
    # An endpoint the profile does not name falls back to the caller's default.
    assert projection.vendor_qos(config, "other", reliability="reliable") == ("reliable", "volatile", 10)
    assert projection.vendor_qos(None, "other") == ("best_effort", "volatile", 10)


def test_vendor_qos_rejects_values_that_would_silently_mismatch():
    """A typo must fail loudly: a wrong QoS drops every message with no error."""
    for bad in ({"a": {"reliability": "RELIABLE"}}, {"a": {"durability": "latched"}}, {"a": {"depth": 0}}):
        with pytest.raises(ProjectionError):
            projection.vendor_qos(bad, "a")


# --- audio focus ------------------------------------------------------------


def test_audio_focus_defaults_to_the_platform_priority_band():
    assert projection.audio_focus_request(None) == (6, 0, 1.0, 50)
    assert projection.audio_focus_request({"focus_priority": 8, "focus_release_idle_s": 0.2})[0] == 8
    assert projection.audio_focus_request({"focus_release_idle_s": 0.2})[2] == 0.2


def test_audio_focus_rejects_priorities_the_platform_cannot_express():
    for bad in ({"focus_priority": 0}, {"focus_priority": 11}, {"focus_priority_weight": 100}):
        with pytest.raises(ProjectionError):
            projection.audio_focus_request(bad)
    for bad in ({"focus_release_idle_s": 0.0}, {"focus_buffer_chunks": 0}):
        with pytest.raises(ProjectionError):
            projection.audio_focus_request(bad)


# --- clock agreement --------------------------------------------------------


def test_clock_skew_estimate_is_the_minimum_because_delay_only_adds():
    """A late callback inflates its own sample; it must not move the estimate.

    Each sample is offset + transport/queueing delay, and that delay is always
    positive. One callback stuck behind a burst of joint feedback therefore
    reports a large difference while the clocks are fine — which is how a
    service call on a busy node looked like a clock that had drifted.
    """
    assert projection.clock_skew_estimate([]) is None
    # Clocks agree; one callback waited 1.2 s behind other work.
    assert projection.clock_skew_estimate([0.004, 0.006, 1.2, 0.005]) == pytest.approx(0.004)
    # A real offset survives, because every sample carries it.
    assert projection.clock_skew_estimate([12.4, 12.9, 13.6]) == pytest.approx(12.4)
    # A board that lags reads negative and is just as real.
    assert projection.clock_skew_estimate([-12.4, -12.1, -11.0]) == pytest.approx(-12.4)


def test_a_delayed_callback_alone_never_raises_a_clock_fault():
    """The whole point: busy != skewed."""
    estimate = projection.clock_skew_estimate([0.003, 0.9, 2.5, 0.004])
    assert projection.clock_skew_fault(estimate, limit_s=0.1, window_s=0.2) is None


def test_clock_skew_within_the_limit_is_not_a_fault():
    assert projection.clock_skew_fault(None, limit_s=0.1, window_s=0.2) is None
    assert projection.clock_skew_fault(0.05, limit_s=0.1, window_s=0.2) is None
    assert projection.clock_skew_fault(-0.05, limit_s=0.1, window_s=0.2) is None


def test_clock_skew_beyond_the_limit_names_the_silent_failure_it_causes():
    """The platform drops out-of-window commands without a word; say so."""
    detail = projection.clock_skew_fault(12.4, limit_s=0.1, window_s=0.2)
    assert detail is not None
    assert "+12.400s" in detail
    assert "discarded by the platform" in detail
    # A clock that lags is as fatal as one that leads.
    assert projection.clock_skew_fault(-12.4, limit_s=0.1, window_s=0.2) is not None


# --- interaction priority scales --------------------------------------------


def test_screen_priority_never_hides_a_fault_indication_below_the_top_band():
    """The platform shows its own fault indications at 8-10.

    A neutral priority passed through unmapped would sit above them for most of
    the contract's range and silently hide over-temperature and fall-protection
    indications from the operator. Only the explicit top band may do that.
    """
    vg = pytest.importorskip("aimdk_robot.vendor_gateway")
    levels = vg.SCREEN_PRIORITY_LEVELS
    for neutral in (0, 25, 50, 79, 90, 94):
        assert projection.priority_level(neutral, levels=levels) < 8, neutral
    # Only the deliberate top of the scale overrides a fault indication.
    assert projection.priority_level(95, levels=levels) > 10
    assert projection.priority_level(100, levels=levels) > 10


def test_led_priority_stays_in_a_band_that_leaves_headroom():
    """Each accepted request raises the platform's threshold; 100 must not pin it."""
    vg = pytest.importorskip("aimdk_robot.vendor_gateway")
    levels = vg.LED_PRIORITY_LEVELS
    values = [projection.priority_level(n, levels=levels) for n in (0, 50, 100)]
    assert values == sorted(values)
    assert max(values) <= 10, "a neutral 100 must not ratchet the threshold out of reach"


# --- platform state fields --------------------------------------------------


def test_balance_fsm_safe_state_is_a_fault_and_normal_states_are_not():
    """SAFE is the platform protecting itself; ACTIVE through it is a lie."""
    assert projection.fsm_fault(projection.FSM_STATE_STABLE) is None
    assert projection.fsm_fault(projection.FSM_STATE_MOVING) is None
    # A firmware that does not populate the field must not degrade the runtime.
    assert projection.fsm_fault(projection.FSM_STATE_UNKNOWN) is None
    safe = projection.fsm_fault(projection.FSM_STATE_SAFE)
    assert safe is not None and "SAFE" in safe
    assert projection.fsm_fault(projection.FSM_STATE_STARTING) is not None
    assert projection.fsm_fault(projection.FSM_STATE_TEST) is not None


def test_platform_speed_envelope_only_tightens_the_profile_ceiling():
    """The profile is the ceiling; the platform may narrow it, never widen it."""
    limits = projection.LocomotionLimits()
    narrow = projection.SpeedEnvelope(mode=1, max_linear_mps=0.4, max_angular_radps=0.3)
    tightened = limits.with_platform_envelope(narrow)
    assert tightened.max_linear_mps == 0.4
    assert tightened.max_angular_radps == 0.3
    # A wider platform envelope does not raise the profile's ceiling.
    wide = projection.SpeedEnvelope(mode=3, max_linear_mps=5.0, max_angular_radps=5.0)
    assert limits.with_platform_envelope(wide).max_linear_mps == limits.max_linear_mps
    # Minima and dead-band are the profile's alone.
    assert tightened.min_linear_mps == limits.min_linear_mps
    assert tightened.deadband == limits.deadband


def test_an_unreported_speed_envelope_is_ignored_not_enforced():
    """All-zero bounds mean "not reported"; enforcing them would refuse everything."""
    limits = projection.LocomotionLimits()
    assert limits.with_platform_envelope(None) == limits
    assert limits.with_platform_envelope(projection.SpeedEnvelope()) == limits
    assert not projection.SpeedEnvelope().usable


def test_head_targets_outside_the_platform_travel_are_refused_not_clamped():
    limits = {"head_yaw_joint": 0.38, "head_pitch_joint": 0.35}
    joints = ["head_yaw_joint", "head_pitch_joint"]
    assert projection.head_payload([0.3, -0.3], channel_joints=joints, limits=limits) == [0.3, -0.3]
    with pytest.raises(CommandRejected) as excinfo:
        projection.head_payload([0.5, 0.0], channel_joints=joints, limits=limits)
    assert excinfo.value.reason == "HEAD_COMMAND_RANGE"
    with pytest.raises(CommandRejected):
        projection.head_payload([0.0, -0.4], channel_joints=joints, limits=limits)
    # Without declared limits the behaviour is unchanged.
    assert projection.head_payload([2.0, 2.0], channel_joints=joints) == [2.0, 2.0]

"""Contract tests for the fixed-trigger sound-orientation policy."""

import math

import pytest

from embodied_agent.sound_orientation_policy import (
    DecisionKind,
    DirectionSample,
    GatewaySnapshot,
    OrientationPolicyConfig,
    OrientationState,
    SessionState,
    SoundOrientationPolicy,
)


def _direction(*, seq_id: int = 1, stamp_sec: float = 100.0, azimuth_rad: float = 0.5) -> DirectionSample:
    return DirectionSample(seq_id, stamp_sec, azimuth_rad)


def _voice_begin_direction(*, seq_id: int = 1, stamp_sec: float = 100.0, azimuth_rad: float = 0.5) -> DirectionSample:
    return DirectionSample(seq_id, stamp_sec, azimuth_rad, direction_type="voice_begin")


def _gateway(**overrides) -> GatewaySnapshot:
    values = {
        "control_plane_ready": True,
        "motion_authorized": True,
        "busy": False,
        "capability_ready": True,
        "active_control_mode": "base_navigation",
        "required_control_mode": "base_navigation",
        "control_mode_switching_enabled": True,
    }
    values.update(overrides)
    return GatewaySnapshot(**values)


def _ready_policy(**overrides) -> SoundOrientationPolicy:
    config_values = {
        "trigger_phrases": ("转向我",),
        "deadband_deg": 15.0,
        "max_direction_age_sec": 1.3,
        "direction_wait_sec": 0.5,
        "cooldown_sec": 1.5,
        "max_turn_deg": 180.0,
    }
    config_values.update(overrides)
    return SoundOrientationPolicy(OrientationPolicyConfig(**config_values))


def test_non_trigger_text_never_dispatches():
    policy = _ready_policy()

    decision = policy.handle_text("拿起红色方块", now_sec=100.1, gateway=_gateway())

    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "NON_EXACT_TRIGGER"
    assert policy.state is OrientationState.IDLE_LISTENING


@pytest.mark.parametrize("text", ["请转向那边", "帮我转一下", "左转", "转向我然后拿起红色方块"])
def test_keyword_mode_rejects_non_exact_phrases(text):
    policy = _ready_policy()
    policy.handle_direction(_direction(azimuth_rad=0.5), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text(text, now_sec=100.0, gateway=_gateway())

    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "NON_EXACT_TRIGGER"


def test_text_without_keyword_is_ignored():
    policy = _ready_policy()

    decision = policy.handle_text("拿起红色方块", now_sec=100.1, gateway=_gateway())

    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "NON_EXACT_TRIGGER"
    assert policy.state is OrientationState.IDLE_LISTENING


def test_keyword_mode_uses_latest_direction_estimate():
    policy = _ready_policy()
    early = DirectionSample(1, 100.0, 0.35, segment_id=7, direction_type="voice_begin")
    later = DirectionSample(2, 100.1, -0.7, segment_id=7, direction_type="seg_end")

    policy.handle_direction(early, now_sec=100.0, gateway=_gateway())
    policy.handle_direction(later, now_sec=100.1, gateway=_gateway())
    decision = policy.handle_text("转向我", now_sec=100.2, gateway=_gateway())

    assert decision.kind is DecisionKind.DISPATCH
    assert decision.request is not None
    assert decision.request.direction == "right"
    assert decision.request.direction_event_key == later.event_key


def test_delayed_asr_text_accepts_cached_voice_begin_direction():
    policy = _ready_policy(max_direction_age_sec=30.0)
    early = _voice_begin_direction(seq_id=5, stamp_sec=100.0, azimuth_rad=0.35)

    policy.handle_direction(early, now_sec=100.1, gateway=_gateway())
    decision = policy.handle_text("转向我", now_sec=120.0, gateway=_gateway())

    assert decision.kind is DecisionKind.DISPATCH
    assert decision.request is not None
    assert decision.request.direction_event_key == early.event_key


def test_periodic_mode_dispatches_new_segment_once_per_interval():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    assert policy.activate_following().kind is DecisionKind.STATE_CHANGED
    policy.handle_direction(
        DirectionSample(1, 100.0, 0.5, segment_id=7, direction_type="voice_begin"),
        now_sec=100.1,
    )

    assert policy.tick(now_sec=100.1).reason == "NO_TRANSITION"
    assert policy.periodic_tick(now_sec=110.0).reason == "WAITING_FOR_GATEWAY"
    decision = policy.periodic_tick(now_sec=110.0, gateway=_gateway())
    assert decision.kind is DecisionKind.DISPATCH
    assert decision.request is not None

    policy.mark_action_submitted()
    policy.complete_action(now_sec=110.2, terminal_known=True)
    assert policy.tick(now_sec=112.0).reason == "COOLDOWN_COMPLETE"
    assert policy.periodic_tick(now_sec=120.0, gateway=_gateway()).reason == "NO_FRESH_DIRECTION"


def test_periodic_mode_prefers_seg_end_and_dispatches_next_segment():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=7, direction_type="voice_begin"), now_sec=100.1)
    policy.handle_direction(DirectionSample(2, 100.1, -0.4, segment_id=7, direction_type="seg_end"), now_sec=100.2)
    first = policy.periodic_tick(now_sec=110.0, gateway=_gateway())
    assert first.request is not None
    assert first.request.direction == "right"

    policy.mark_action_submitted()
    policy.complete_action(now_sec=110.2, terminal_known=True)
    policy.tick(now_sec=112.0)
    policy.handle_direction(DirectionSample(3, 113.0, 0.3, segment_id=8, direction_type="voice_begin"), now_sec=113.1)
    second = policy.periodic_tick(now_sec=123.0, gateway=_gateway())
    assert second.kind is DecisionKind.DISPATCH
    assert second.request is not None
    assert second.request.direction_event_key == (3, 113.0)


def test_periodic_mode_recovers_after_known_action_failure():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.1)
    first = policy.periodic_tick(now_sec=110.0, gateway=_gateway())
    assert first.kind is DecisionKind.DISPATCH
    policy.mark_action_submitted()
    policy.complete_action(now_sec=110.2, terminal_known=True)

    policy.handle_direction(DirectionSample(2, 113.0, -0.4, segment_id=2, direction_type="seg_end"), now_sec=113.1)
    second = policy.periodic_tick(now_sec=120.0, gateway=_gateway())

    assert second.kind is DecisionKind.DISPATCH
    assert second.request is not None
    assert second.request.direction == "right"


def test_periodic_mode_blocks_dispatch_until_cooldown_completes():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.0)
    assert policy.periodic_tick(now_sec=100.0, gateway=_gateway()).kind is DecisionKind.DISPATCH
    policy.mark_action_submitted()
    policy.complete_action(now_sec=100.1, terminal_known=True)
    policy.handle_direction(DirectionSample(2, 100.2, -0.4, segment_id=2, direction_type="seg_end"), now_sec=100.2)

    assert policy.periodic_tick(now_sec=101.0).reason == "COOLDOWN"
    assert policy.periodic_tick(now_sec=110.0, gateway=_gateway()).kind is DecisionKind.DISPATCH


def test_periodic_gateway_rejection_consumes_segment_without_retry():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.0)

    assert policy.periodic_tick(now_sec=100.0, gateway=_gateway(busy=True)).reason == "SKILL_BUSY"
    assert policy.periodic_tick(now_sec=110.0, gateway=_gateway()).reason == "NO_FRESH_DIRECTION"


def test_periodic_mode_prunes_expired_segments():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=5.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.0)

    assert policy.periodic_tick(now_sec=106.0).reason == "NO_FRESH_DIRECTION"
    assert policy._periodic_latest_by_segment == {}  # noqa: SLF001
    assert policy._periodic_consumed_segments == set()  # noqa: SLF001


def test_long_utterance_is_not_replayed_after_direction_expires():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=1.0, max_direction_age_sec=2.0, cooldown_sec=0.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=7, direction_type="voice_begin"), now_sec=100.0)
    assert policy.periodic_tick(now_sec=100.0, gateway=_gateway()).kind is DecisionKind.DISPATCH
    policy.complete_action(now_sec=100.1, terminal_known=True)
    assert policy.periodic_tick(now_sec=110.0).reason == "NO_FRESH_DIRECTION"

    end = DirectionSample(2, 111.0, -0.5, segment_id=7, direction_type="seg_end")
    assert policy.handle_direction(end, now_sec=111.0).reason == "DUPLICATE_SEGMENT"
    assert policy.periodic_tick(now_sec=111.0, gateway=_gateway()).reason == "NO_FRESH_DIRECTION"


def test_resident_session_memory_stays_bounded():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=1.0, max_direction_age_sec=2.0, cooldown_sec=0.0)
    policy.activate_following()
    for segment_id in range(1, 1001):
        now = float(segment_id)
        policy.handle_direction(DirectionSample(segment_id, now, 0.5, segment_id=segment_id), now_sec=now)
        assert policy.periodic_tick(now_sec=now, gateway=_gateway()).kind is DecisionKind.DISPATCH
        policy.complete_action(now_sec=now, terminal_known=True)
    assert len(policy._periodic_latest_by_segment) <= 3
    assert len(policy._periodic_consumed_segments) <= 256
    assert len(policy._periodic_consumed_order) <= 256


def test_inactive_session_reclaims_cached_directions():
    policy = _ready_policy(mode="periodic")
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=7), now_sec=100.0)
    policy.periodic_tick(now_sec=103.0)
    assert not policy._periodic_latest_by_segment


def test_periodic_mode_ignores_asr_text():
    policy = _ready_policy(mode="periodic")
    decision = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())
    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "PERIODIC_MODE_IGNORES_ASR"


def test_session_inactive_blocks_periodic_dispatch():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.1)

    decision = policy.periodic_tick(now_sec=110.0, gateway=_gateway())

    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "SOUND_FOLLOWING_INACTIVE"


def test_activate_clears_stale_cached_directions():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.1)
    policy.activate_following()

    decision = policy.periodic_tick(now_sec=110.0, gateway=_gateway())

    assert decision.reason == "NO_FRESH_DIRECTION"


def test_deactivate_stops_dispatch_immediately():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.1)

    policy.deactivate_following()
    decision = policy.periodic_tick(now_sec=110.0, gateway=_gateway())

    assert decision.reason == "SOUND_FOLLOWING_INACTIVE"


def test_deactivate_during_turn_reaches_inactive_after_terminal():
    policy = _ready_policy(mode="periodic", periodic_interval_sec=10.0, max_direction_age_sec=30.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.1)
    decision = policy.periodic_tick(now_sec=110.0, gateway=_gateway())
    assert decision.kind is DecisionKind.DISPATCH
    policy.mark_action_submitted()

    shutdown = policy.deactivate_following()
    assert shutdown.reason == "SOUND_FOLLOWING_SHUTTING_DOWN"

    blocked = policy.periodic_tick(now_sec=115.0, gateway=_gateway())
    assert blocked.reason == "SOUND_FOLLOWING_INACTIVE"

    terminal = policy.complete_action(now_sec=116.0, terminal_known=True)
    assert terminal.reason == "SOUND_FOLLOWING_DEACTIVATED"
    assert policy.session_state.value == "inactive"


def test_deactivate_during_turn_reaches_inactive_after_unknown_terminal():
    policy = _ready_policy(mode="periodic", max_direction_age_sec=30.0)
    policy.activate_following()
    policy.handle_direction(DirectionSample(1, 100.0, 0.5, segment_id=1, direction_type="seg_end"), now_sec=100.1)
    assert policy.periodic_tick(now_sec=110.0, gateway=_gateway()).kind is DecisionKind.DISPATCH
    policy.mark_action_submitted()
    policy.deactivate_following()

    terminal = policy.complete_action(now_sec=111.0, terminal_known=False)

    assert terminal.reason == "FAULT_UNKNOWN"
    assert policy.session_state is SessionState.INACTIVE


def test_activate_and_deactivate_are_idempotent():
    policy = _ready_policy(mode="periodic")

    first = policy.activate_following()
    second = policy.activate_following()
    assert first.kind is DecisionKind.STATE_CHANGED
    assert second.reason == "ALREADY_ACTIVE"

    first_off = policy.deactivate_following()
    second_off = policy.deactivate_following()
    assert first_off.kind is DecisionKind.STATE_CHANGED
    assert second_off.reason == "ALREADY_INACTIVE"


def test_activate_rejected_in_keyword_mode():
    policy = _ready_policy(mode="keyword")

    decision = policy.activate_following()

    assert decision.kind is DecisionKind.DROPPED
    assert decision.reason == "NOT_PERIODIC_MODE"


def test_default_active_initializes_periodic_session():
    policy = _ready_policy(mode="periodic", default_active=True)

    assert policy.session_state is SessionState.ACTIVE


def test_trigger_requires_a_fresh_direction():
    policy = _ready_policy()
    policy.handle_direction(_direction(stamp_sec=90.0), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())

    assert decision.kind is DecisionKind.WAITING
    assert policy.state is OrientationState.WAITING_FOR_DIRECTION


def test_direction_after_trigger_completes_waiting_request():
    policy = _ready_policy()
    waiting = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())

    decision = policy.handle_direction(_direction(azimuth_rad=0.5), now_sec=100.1, gateway=_gateway())

    assert waiting.kind is DecisionKind.WAITING
    assert decision.kind is DecisionKind.DISPATCH
    assert decision.request is not None
    assert decision.request.direction == "left"
    assert decision.request.degree == pytest.approx(math.degrees(0.5))
    assert policy.state is OrientationState.DISPATCHING


def test_positive_and_negative_angles_map_to_left_and_right():
    left_policy = _ready_policy()
    left_policy.handle_direction(_direction(azimuth_rad=0.5), now_sec=100.0, gateway=_gateway())
    left = left_policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())

    right_policy = _ready_policy()
    right_policy.handle_direction(_direction(azimuth_rad=-0.5), now_sec=100.0, gateway=_gateway())
    right = right_policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())

    assert left.request is not None and left.request.direction == "left"
    assert right.request is not None and right.request.direction == "right"


def test_deadband_drops_small_turn_without_dispatch():
    policy = _ready_policy(deadband_deg=15.0)
    policy.handle_direction(_direction(azimuth_rad=math.radians(10.0)), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())

    assert decision.kind is DecisionKind.DROPPED
    assert decision.reason == "WITHIN_DEADBAND"
    assert policy.state is OrientationState.IDLE_LISTENING


@pytest.mark.parametrize(
    "snapshot, reason",
    [
        (_gateway(control_plane_ready=False), "CONTROL_PLANE_NOT_READY"),
        (_gateway(motion_authorized=False), "MOTION_NOT_AUTHORIZED"),
        (_gateway(busy=True), "SKILL_BUSY"),
        (_gateway(capability_ready=False), "CAPABILITY_NOT_READY"),
        (
            _gateway(control_mode_switching_enabled=False, active_control_mode="moveit_planning"),
            "CONTROL_MODE_MISMATCH",
        ),
    ],
)
def test_gateway_admission_conditions_drop_trigger(snapshot, reason):
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text("转向我", now_sec=100.0, gateway=snapshot)

    assert decision.kind is DecisionKind.DROPPED
    assert decision.reason == reason
    assert policy.state is OrientationState.IDLE_LISTENING


def test_busy_does_not_create_a_pending_queue():
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway(busy=True))
    later = policy.handle_direction(_direction(seq_id=2, stamp_sec=100.2), now_sec=100.2, gateway=_gateway())

    assert decision.reason == "SKILL_BUSY"
    assert later.reason == "DIRECTION_CACHED"
    assert policy.state is OrientationState.IDLE_LISTENING
    assert policy.active_request is None


def test_same_direction_event_is_consumed_only_once():
    policy = _ready_policy(max_direction_age_sec=3.0)
    sample = _direction(seq_id=7, stamp_sec=100.0)
    policy.handle_direction(sample, now_sec=100.0, gateway=_gateway())
    first = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())
    policy.mark_action_submitted()
    policy.complete_action(now_sec=100.2, terminal_known=True)
    policy.tick(now_sec=101.8)

    policy.handle_direction(sample, now_sec=101.8, gateway=_gateway())
    second = policy.handle_text("转向我", now_sec=101.8, gateway=_gateway())

    assert first.kind is DecisionKind.DISPATCH
    assert second.kind is DecisionKind.WAITING


def test_turning_and_cooldown_drop_new_directions():
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())
    decision = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())
    policy.mark_action_submitted()

    during_turn = policy.handle_direction(_direction(seq_id=2, stamp_sec=100.1), now_sec=100.1, gateway=_gateway())
    policy.complete_action(now_sec=100.2, terminal_known=True)
    during_cooldown = policy.handle_direction(_direction(seq_id=3, stamp_sec=100.3), now_sec=100.3, gateway=_gateway())

    assert decision.kind is DecisionKind.DISPATCH
    assert during_turn.reason == "NOT_LISTENING"
    assert during_cooldown.reason == "NOT_LISTENING"
    assert policy.state is OrientationState.COOLDOWN


def test_known_skill_busy_result_is_not_retried():
    policy = _ready_policy(cooldown_sec=0.0)
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())
    policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())
    policy.mark_action_submitted()

    terminal = policy.complete_action(now_sec=100.1, terminal_known=True)
    policy.tick(now_sec=100.1)
    next_trigger = policy.handle_text("转向我", now_sec=100.1, gateway=_gateway())

    assert terminal.reason == "ACTION_TERMINAL"
    assert next_trigger.kind is DecisionKind.WAITING


def test_unknown_action_result_fails_closed_until_reset():
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())
    policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())
    policy.mark_action_submitted()

    terminal = policy.complete_action(now_sec=100.1, terminal_known=False)
    blocked = policy.handle_text("转向我", now_sec=100.2, gateway=_gateway())
    policy.reset_fault()
    recovered = policy.handle_text("转向我", now_sec=100.2, gateway=_gateway())

    assert terminal.reason == "FAULT_UNKNOWN"
    assert blocked.reason == "FAULT_UNKNOWN"
    assert recovered.kind is DecisionKind.WAITING
    assert policy.active_request is None


@pytest.mark.parametrize(
    "sample",
    [
        DirectionSample(1, 100.0, float("nan")),
        DirectionSample(2, 100.0, float("inf")),
        DirectionSample(3, 100.0, 0.5, frame_id="map"),
        DirectionSample(4, 98.0, 0.5),
        DirectionSample(5, 100.0, math.pi + 0.01),
    ],
)
def test_invalid_or_stale_direction_is_ignored(sample):
    policy = _ready_policy()

    decision = policy.handle_direction(sample, now_sec=100.0, gateway=_gateway())

    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "INVALID_OR_STALE_DIRECTION"
    assert policy.state is OrientationState.IDLE_LISTENING


def test_keyword_mode_rejects_multi_intent_command():
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text("转向我然后拿起红色方块", now_sec=100.0, gateway=_gateway())

    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "NON_EXACT_TRIGGER"


def test_trigger_punctuation_and_whitespace_are_normalized_without_substring_matching():
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text(" 转 向 我！ ", now_sec=100.0, gateway=_gateway())

    assert decision.kind is DecisionKind.DISPATCH


def test_trigger_waits_for_gateway_status_before_dispatch():
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0)

    decision = policy.handle_text("转向我", now_sec=100.0)
    status_decision = policy.try_dispatch(now_sec=100.1, gateway=_gateway())

    assert decision.reason == "WAITING_FOR_GATEWAY"
    assert status_decision.kind is DecisionKind.DISPATCH


def test_trigger_accepts_direction_after_realistic_asr_doa_delay():
    policy = _ready_policy(direction_wait_sec=8.0, max_direction_age_sec=8.0)

    waiting = policy.handle_text("转向我", now_sec=100.0, gateway=_gateway())
    decision = policy.handle_direction(_direction(stamp_sec=106.0), now_sec=106.0, gateway=_gateway())

    assert waiting.reason == "WAITING_FOR_DIRECTION"
    assert decision.kind is DecisionKind.DISPATCH
    assert decision.request is not None

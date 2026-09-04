"""Contract tests for the fixed-trigger sound-orientation policy."""

import math

import pytest

from embodied_agent.sound_orientation_policy import (
    DecisionKind,
    DirectionSample,
    GatewaySnapshot,
    OrientationPolicyConfig,
    OrientationState,
    SoundOrientationPolicy,
)


def _direction(*, seq_id: int = 1, stamp_sec: float = 100.0, azimuth_rad: float = 0.5) -> DirectionSample:
    return DirectionSample(seq_id, stamp_sec, azimuth_rad)


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


def test_exact_trigger_matching_does_not_split_multi_intent_command():
    policy = _ready_policy()
    policy.handle_direction(_direction(), now_sec=100.0, gateway=_gateway())

    decision = policy.handle_text("转向我然后拿起红色方块", now_sec=100.0, gateway=_gateway())

    assert decision.kind is DecisionKind.IGNORED
    assert decision.reason == "NON_EXACT_TRIGGER"
    assert policy.active_request is None


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

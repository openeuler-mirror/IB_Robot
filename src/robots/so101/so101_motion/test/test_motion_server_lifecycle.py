#!/usr/bin/env python3
"""Unit tests for the SO-101 motion server: motion ownership, cancellation, runtime gate, outcomes."""

from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import Pose

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from motion_server import MotionServer, MoveIt2State  # noqa: E402

from robot_runtime import contract as C  # noqa: E402


class FakeLogger:
    def __init__(self):
        self.messages = []

    def _record(self, level, message):
        self.messages.append((level, message))

    def debug(self, message):
        self._record("debug", message)

    def info(self, message):
        self._record("info", message)

    def warning(self, message):
        self._record("warning", message)

    warn = warning

    def error(self, message):
        self._record("error", message)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message.data)


class DoneFuture:
    def __init__(self, result):
        self._result = result

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        return self._result


class FakeMoveIt2:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = MoveIt2State.IDLE
        self.motion_suceeded = False
        self.max_velocity = 0.8
        self.cancel_calls = 0
        self.move_to_configuration_calls = []
        self.cancel_called = threading.Event()
        self.dispatch = True

    def query_state(self):
        with self._lock:
            return self._state

    def set_state(self, state):
        with self._lock:
            self._state = state

    def cancel_execution(self):
        self.cancel_calls += 1
        self.cancel_called.set()

    def get_last_execution_error_code(self):
        return None

    def clear_goal_constraints(self):
        pass

    def move_to_configuration(self, joint_positions):
        self.move_to_configuration_calls.append(list(joint_positions))
        if self.dispatch:
            self.set_state(MoveIt2State.REQUESTING)


class FakeTfBuffer:
    def __init__(self):
        self.stamp_ns = 0

    def set_stamp(self, stamp_ns: int):
        self.stamp_ns = int(stamp_ns)

    def lookup_transform(self, *_args, **_kwargs):
        return SimpleNamespace(header=_header(self.stamp_ns))


def _header(stamp_ns: int):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=int(stamp_ns) // 1_000_000_000,
            nanosec=int(stamp_ns) % 1_000_000_000,
        )
    )


def _joint_state(stamp_ns: int, positions=(0.1, 0.2)):
    return SimpleNamespace(header=_header(stamp_ns), name=["1", "2"], position=list(positions))


def _hardware_feedback(stamp_ns: int):
    return SimpleNamespace(header=_header(stamp_ns))


def make_gateway(*, status_seen=True):
    gateway = MotionServer.__new__(MotionServer)
    gateway._initialize_motion_coordinator()
    gateway._trajectory_modes = {"trajectory"}
    gateway._runtime_status_stale_s = 0.5
    if status_seen:
        gateway._runtime_status_callback(runtime_status())
    gateway._motion_start_timeout_s = 0.05
    gateway._motion_execution_timeout_s = 0.0
    gateway._motion_cancel_timeout_s = 0.2
    gateway._motion_status_hold_s = 0.0
    gateway._motion_feedback_timeout_s = 0.05
    gateway._motion_feedback_tolerance_rad = 0.05
    gateway._motion_require_tf_sync = True
    gateway._motion_hardware_feedback_topic = "/so101_follower/joint_currents"
    gateway._motion_status = "idle"
    gateway.motion_status_pub = FakePublisher()
    gateway.moveit2 = FakeMoveIt2()
    gateway.joint_names = ["1", "2"]
    gateway._joint_state_lock = threading.Lock()
    gateway._joint_state_sequence = 0
    gateway._hardware_feedback_sequence = 0
    gateway._latest_hardware_feedback_stamp_ns = 0
    gateway.latest_joint_state = None
    gateway.base_link = "base"
    gateway.ee_link = "gripper"
    gateway.shoulder_link = "shoulder"
    gateway.tf_buffer = FakeTfBuffer()
    logger = FakeLogger()
    gateway.get_logger = lambda: logger
    return gateway


def make_configuration_request():
    return SimpleNamespace(
        target_joint_state=SimpleNamespace(name=["1", "2"], position=[0.1, 0.2]),
        velocity_scaling=0.0,
    )


def make_response():
    return SimpleNamespace(
        success=None, message="", execution_time_s=0.0, outcome="", position_error_m=0.0, orientation_error_rad=0.0
    )


def runtime_status(*, stop_latched: bool = False, active_mode: str = "trajectory"):
    return SimpleNamespace(stop_latched=stop_latched, active_mode=active_mode)


def test_concurrent_service_request_returns_busy_without_dispatch():
    gateway = make_gateway()
    entered = threading.Event()
    release = threading.Event()
    dispatch_count = 0

    def blocking_move(_joint_positions):
        nonlocal dispatch_count
        dispatch_count += 1
        entered.set()
        assert release.wait(timeout=1.0)
        return False

    gateway.move_to_joint = blocking_move
    first_response = make_response()
    first_thread = threading.Thread(
        target=gateway._move_to_configuration_service_cb,
        args=(make_configuration_request(), first_response),
    )
    first_thread.start()
    assert entered.wait(timeout=1.0)
    statuses_before = list(gateway.motion_status_pub.messages)

    second_response = gateway._move_to_configuration_service_cb(make_configuration_request(), make_response())

    assert second_response.success is False
    assert second_response.message == "motion server is busy"
    assert dispatch_count == 1
    assert gateway.motion_status_pub.messages == statuses_before
    release.set()
    first_thread.join(timeout=1.0)
    assert not first_thread.is_alive()


def test_service_is_rejected_while_cmd_pose_owns_motion():
    gateway = make_gateway()
    entered = threading.Event()
    release = threading.Event()

    def blocking_strategy(_position, _orientation):
        entered.set()
        assert release.wait(timeout=1.0)
        gateway.moveit2.set_state(MoveIt2State.EXECUTING)
        return True

    gateway._move_with_strategies = blocking_strategy
    cmd_thread = threading.Thread(target=gateway.cmd_pose_callback, args=(Pose(),))
    cmd_thread.start()
    assert entered.wait(timeout=1.0)
    statuses_before = list(gateway.motion_status_pub.messages)

    response = gateway._move_to_configuration_service_cb(make_configuration_request(), make_response())

    assert response.success is False
    assert response.message == "motion server is busy"
    assert gateway.motion_status_pub.messages == statuses_before
    release.set()
    cmd_thread.join(timeout=1.0)
    assert not cmd_thread.is_alive()

    gateway.moveit2.motion_suceeded = True
    gateway.moveit2.set_state(MoveIt2State.IDLE)
    gateway._motion_watchdog_callback()
    assert gateway._active_motion_token is None


def test_timeout_restores_velocity_and_publishes_idle_only_after_cancel_is_confirmed():
    gateway = make_gateway()
    token = gateway._claim_motion("MoveToConfiguration")
    gateway._prepare_motion(token)
    gateway._set_motion_velocity(token, 0.2)
    gateway.moveit2.set_state(MoveIt2State.EXECUTING)
    result = None

    def wait_for_motion():
        nonlocal result
        result = gateway._wait_for_motion_completion(token, "MoveToConfiguration")

    wait_thread = threading.Thread(target=wait_for_motion)
    wait_thread.start()
    assert gateway.moveit2.cancel_called.wait(timeout=1.0)

    assert gateway.motion_status_pub.messages == ["executing"]
    assert gateway.moveit2.max_velocity == 0.2
    assert gateway._active_motion_token == token

    gateway.moveit2.set_state(MoveIt2State.IDLE)
    wait_thread.join(timeout=1.0)
    assert not wait_thread.is_alive()
    assert result[0] is False
    assert result[2] is True

    gateway._finalize_motion(token, success=False)
    assert gateway.moveit2.max_velocity == 0.8
    assert gateway.motion_status_pub.messages == ["executing", "failed", "idle"]
    assert gateway._active_motion_token is None


def test_unconfirmed_cancellation_keeps_gateway_busy_until_watchdog_sees_idle():
    gateway = make_gateway()
    gateway._motion_cancel_timeout_s = 0.0
    token = gateway._claim_motion("MoveToConfiguration")
    gateway._prepare_motion(token)
    gateway._set_motion_velocity(token, 0.2)
    gateway.moveit2.set_state(MoveIt2State.EXECUTING)

    success, message, terminal_confirmed = gateway._wait_for_motion_completion(token, "MoveToConfiguration")

    assert success is False
    assert "cancellation is still pending" in message
    assert terminal_confirmed is False
    assert gateway._claim_motion("new request") is None
    assert gateway.motion_status_pub.messages == ["executing"]
    assert gateway.moveit2.max_velocity == 0.2

    gateway._motion_watchdog_callback()
    assert gateway._active_motion_token == token
    gateway.moveit2.set_state(MoveIt2State.IDLE)
    gateway._motion_watchdog_callback()
    assert gateway.moveit2.max_velocity == 0.8
    assert gateway.motion_status_pub.messages == ["executing", "failed", "idle"]


def test_old_token_cannot_finalize_a_new_motion():
    gateway = make_gateway()
    old_token = gateway._claim_motion("old")
    gateway._prepare_motion(old_token)
    assert gateway._finalize_motion(old_token, success=True)

    new_token = gateway._claim_motion("new")
    gateway._prepare_motion(new_token)
    statuses_before = list(gateway.motion_status_pub.messages)

    assert gateway._finalize_motion(old_token, success=False) is False
    assert gateway._active_motion_token == new_token
    assert gateway.motion_status_pub.messages == statuses_before


def test_skipped_moveit_dispatch_is_not_reported_as_successful():
    gateway = make_gateway()
    gateway.moveit2.dispatch = False

    assert gateway.move_to_joint([0.1, 0.2]) is False

    class DoneFuture:
        @staticmethod
        def done():
            return True

    gateway.moveit2.compute_ik_async = lambda **_kwargs: DoneFuture()
    gateway.moveit2.get_compute_ik_result = lambda _future: SimpleNamespace(
        name=["1", "2"],
        position=[0.1, 0.2],
    )
    assert gateway.solve_and_move(Pose()) is False


def test_post_motion_feedback_waits_for_a_fresh_converged_sample():
    gateway = make_gateway()
    result = None

    def wait_for_feedback():
        nonlocal result
        result = gateway._wait_for_post_motion_feedback({"1": 0.1, "2": 0.2})

    wait_thread = threading.Thread(target=wait_for_feedback)
    wait_thread.start()
    time.sleep(0.01)
    gateway.tf_buffer.set_stamp(2_000_000_000)
    gateway.hardware_feedback_callback(_hardware_feedback(2_000_000_000))
    gateway.joint_state_callback(_joint_state(2_000_000_000, positions=(0.11, 0.18)))
    wait_thread.join(timeout=1.0)

    assert not wait_thread.is_alive()
    assert result is True


def test_post_motion_feedback_does_not_accept_a_stale_sample():
    gateway = make_gateway()
    gateway._motion_feedback_timeout_s = 0.02
    gateway.tf_buffer.set_stamp(2_000_000_000)
    gateway.hardware_feedback_callback(_hardware_feedback(2_000_000_000))
    gateway.joint_state_callback(_joint_state(2_000_000_000))

    assert gateway._wait_for_post_motion_feedback({"1": 0.1, "2": 0.2}) is False


def test_post_motion_feedback_requires_a_new_hardware_read():
    gateway = make_gateway()
    gateway._motion_feedback_timeout_s = 0.03
    result = None

    def wait_for_feedback():
        nonlocal result
        result = gateway._wait_for_post_motion_feedback({"1": 0.1, "2": 0.2})

    wait_thread = threading.Thread(target=wait_for_feedback)
    wait_thread.start()
    time.sleep(0.01)
    gateway.tf_buffer.set_stamp(3_000_000_000)
    gateway.joint_state_callback(_joint_state(3_000_000_000))
    wait_thread.join(timeout=1.0)

    assert not wait_thread.is_alive()
    assert result is False


def test_post_motion_feedback_requires_tf_for_the_accepted_joint_sample():
    gateway = make_gateway()
    gateway._motion_feedback_timeout_s = 0.03
    result = None

    def wait_for_feedback():
        nonlocal result
        result = gateway._wait_for_post_motion_feedback({"1": 0.1, "2": 0.2})

    wait_thread = threading.Thread(target=wait_for_feedback)
    wait_thread.start()
    time.sleep(0.01)
    gateway.tf_buffer.set_stamp(3_000_000_000)
    gateway.hardware_feedback_callback(_hardware_feedback(4_000_000_000))
    gateway.joint_state_callback(_joint_state(4_000_000_000))
    wait_thread.join(timeout=1.0)

    assert not wait_thread.is_alive()
    assert result is False


def test_post_motion_feedback_rejects_a_queued_hardware_sample_older_than_joint_state():
    gateway = make_gateway()
    gateway._motion_feedback_timeout_s = 0.03
    result = None

    def wait_for_feedback():
        nonlocal result
        result = gateway._wait_for_post_motion_feedback({"1": 0.1, "2": 0.2})

    wait_thread = threading.Thread(target=wait_for_feedback)
    wait_thread.start()
    time.sleep(0.01)
    gateway.tf_buffer.set_stamp(5_000_000_000)
    gateway.hardware_feedback_callback(_hardware_feedback(4_000_000_000))
    gateway.joint_state_callback(_joint_state(5_000_000_000))
    wait_thread.join(timeout=1.0)

    assert not wait_thread.is_alive()
    assert result is False


def test_post_motion_feedback_keeps_the_first_fresh_converged_joint_sample_as_sync_target():
    gateway = make_gateway()
    gateway._motion_feedback_timeout_s = 0.2
    result = None

    def wait_for_feedback():
        nonlocal result
        result = gateway._wait_for_post_motion_feedback({"1": 0.1, "2": 0.2})

    wait_thread = threading.Thread(target=wait_for_feedback)
    wait_thread.start()
    time.sleep(0.01)

    gateway.tf_buffer.set_stamp(1_900_000_000)
    gateway.hardware_feedback_callback(_hardware_feedback(1_900_000_000))
    gateway.joint_state_callback(_joint_state(2_000_000_000))
    time.sleep(0.03)

    gateway.joint_state_callback(_joint_state(3_000_000_000))
    gateway.tf_buffer.set_stamp(2_000_000_000)
    gateway.hardware_feedback_callback(_hardware_feedback(2_000_000_000))
    wait_thread.join(timeout=1.0)

    assert not wait_thread.is_alive()
    assert result is True


def test_move_rejected_while_runtime_stopped_reports_rejected_outcome():
    gateway = make_gateway()
    gateway._runtime_status_callback(runtime_status(stop_latched=True, active_mode="idle"))

    response = gateway._move_to_configuration_service_cb(make_configuration_request(), make_response())

    assert response.success is False
    assert response.outcome == C.OUTCOME_REJECTED
    assert "STOPPED" in response.message
    assert gateway.moveit2.move_to_configuration_calls == []


def test_move_rejected_when_mode_does_not_permit_trajectory():
    gateway = make_gateway()
    gateway._runtime_status_callback(runtime_status(active_mode="stream"))

    response = gateway._move_to_configuration_service_cb(make_configuration_request(), make_response())

    assert response.success is False
    assert response.outcome == C.OUTCOME_REJECTED
    assert "'stream'" in response.message


def test_moves_blocked_before_any_runtime_status_is_seen():
    gateway = make_gateway(status_seen=False)
    assert "not yet observed" in gateway._runtime_block_reason()
    assert gateway._claim_motion("test") is None


def test_runtime_status_freshness_gates_trajectory(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    gateway = make_gateway()
    assert gateway._runtime_block_reason() == ""
    monkeypatch.setattr(time, "monotonic", lambda: 100.51)
    assert "stale" in gateway._runtime_block_reason()
    assert gateway._claim_motion("test") is None
    gateway._runtime_status_callback(runtime_status())
    assert gateway._runtime_block_reason() == ""
    assert gateway._claim_motion("test") is not None


@pytest.mark.parametrize("budget", [None, 0.1, 1.0, 1.01, 2.5, 4.0, 0.0, -0.1, float("nan"), float("inf")])
def test_motion_status_budget_validation(monkeypatch, budget):
    import math

    from rclpy.node import Node

    params = {} if budget is None else {"runtime_status_stale_s": budget}
    monkeypatch.setattr(Node, "__init__", lambda self, *args: None)
    monkeypatch.setattr(Node, "declare_parameter", lambda self, name, value=None: params.setdefault(name, value))
    monkeypatch.setattr(Node, "get_parameter", lambda self, name: SimpleNamespace(value=params[name]))

    class Validated(Exception):
        pass

    def stop_after_validation(self):
        assert self._runtime_status_stale_s == (2.5 if budget is None else budget)
        raise Validated

    monkeypatch.setattr(MotionServer, "_initialize_motion_coordinator", stop_after_validation)
    if budget is None or (math.isfinite(budget) and budget > 0):
        with pytest.raises(Validated):
            MotionServer()
    else:
        with pytest.raises(ValueError, match="runtime_status_stale_s"):
            MotionServer()


def test_motion_default_status_budget_tolerates_heartbeat_gap(monkeypatch):
    # RuntimeStatus is a 1 Hz heartbeat; the 2.5 s default must keep
    # admitting between beats and only block after a missed beat plus margin.
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    gateway = make_gateway()
    gateway._runtime_status_stale_s = 2.5
    monkeypatch.setattr(time, "monotonic", lambda: 101.4)
    assert gateway._runtime_block_reason() == ""
    assert gateway._claim_motion("test") is not None
    monkeypatch.setattr(time, "monotonic", lambda: 102.6)
    assert "stale" in gateway._runtime_block_reason()
    assert gateway._claim_motion("test") is None


def test_stop_latch_during_motion_cancels_and_reports_cancelled():
    gateway = make_gateway()
    gateway._runtime_status_callback(runtime_status(active_mode="trajectory"))
    token = gateway._claim_motion("MoveToConfiguration")
    assert token is not None
    gateway.moveit2.set_state(MoveIt2State.EXECUTING)

    gateway._runtime_status_callback(runtime_status(stop_latched=True, active_mode="idle"))

    assert gateway.moveit2.cancel_calls == 1
    assert gateway._cancelled_by_stop is True
    # A second latched status must not cancel again.
    gateway._runtime_status_callback(runtime_status(stop_latched=True, active_mode="idle"))
    assert gateway.moveit2.cancel_calls == 1

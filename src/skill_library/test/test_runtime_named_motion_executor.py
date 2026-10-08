"""Behaviour of the runtime_named_motion delegated executor, without a ROS graph."""

import json
from threading import RLock
from types import SimpleNamespace

import pytest

from ibrobot_msgs.action import ExecuteNamedMotion
from ibrobot_msgs.msg import RuntimeStatus
from skill_library import runtime_named_motion as named_motion
from skill_library import skill_executor_node as executor_module
from skill_library.skill_executor_node import PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, SkillExecutorNode

ADVERTISED = {"motion.named": {"names": ["wave", "handshake"]}, "motion.posture": {"postures": ["sit_down"]}}
TEMPLATE = {"executor": "runtime_named_motion", "binding": {"motion": "handshake"}, "timeout_sec": 5.0}
RESULT = ExecuteNamedMotion.Result


class _Future:
    """A future that becomes done after a number of polls, or when completed."""

    def __init__(self, value=None, *, done_after_polls: int | None = 0):
        self._value = value
        self._polls_left = done_after_polls
        self._callbacks = []

    def done(self) -> bool:
        if self._polls_left is None:
            return False
        if self._polls_left > 0:
            self._polls_left -= 1
            return False
        return True

    def result(self):
        return self._value

    def add_done_callback(self, callback) -> None:
        if self._polls_left == 0:
            callback(self)
        else:
            self._callbacks.append(callback)

    def complete(self, value) -> None:
        self._value = value
        self._polls_left = 0
        for callback in self._callbacks:
            callback(self)


class _MotionHandle:
    def __init__(self, result_future, *, accepted=True):
        self.accepted = accepted
        self.result_future = result_future
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1
        raise AssertionError("named motions must never be cancelled through the action")


class _Client:
    def __init__(self, send_future, *, server_ready=True):
        self.send_future = send_future
        self.server_ready = server_ready
        self.goals = []

    def wait_for_server(self, timeout_sec):
        return self.server_ready

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        return self.send_future


class _GoalHandle:
    def __init__(self, *, cancel_requested=False, skill_name="handshake"):
        self.request = SimpleNamespace(skill_name=skill_name, timeout_sec=0.0, dispatch_binding=object())
        self.is_cancel_requested = cancel_requested
        self.terminal = None
        self.feedback = []

    def succeed(self):
        self.terminal = "succeeded"

    def abort(self):
        self.terminal = "aborted"

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)


def _motion_result(*, success=True, error_code=RESULT.NONE, message="done"):
    return SimpleNamespace(result=SimpleNamespace(success=success, error_code=error_code, message=message))


def _status(capabilities=ADVERTISED):
    status = RuntimeStatus()
    status.capabilities_json = json.dumps(capabilities)
    return status


def _node(client, *, status=None, required=("motion.named",)):
    node = object.__new__(SkillExecutorNode)
    node._runtime_named_motion_client = client
    node._runtime_named_motion_action = "/motion/execute_named"
    node._rpc_timeout = 0.1
    node._active_skill_admission = object()
    node.registered = []
    node.confirmed = []
    node.audits = []
    node._runtime_status_for_admission = lambda: (status or _status(), "")
    node._required_capabilities_for_skill = lambda _name: list(required)
    node._set_result_catalog_identity = lambda _result: None
    node._wait_for_future = lambda future, timeout_sec=None, **_kwargs: future.done()
    node._register_delegated_dispatch = lambda nonce, admission, binding: node.registered.append(nonce) or b"key"
    node._confirm_delegated_terminal = lambda admission, nonce, key: node.confirmed.append(nonce)
    node._audit = lambda event, **_fields: node.audits.append(event)
    node._public_audit_context_copy = lambda: {}
    node.get_logger = lambda: SimpleNamespace(warning=lambda _m: None, error=lambda _m: None)
    return node


@pytest.fixture(autouse=True)
def _rclpy_running(monkeypatch):
    monkeypatch.setattr(executor_module.rclpy, "ok", lambda: True)


def _run(node, handle, template=TEMPLATE, timeout=5.0):
    return node._execute_runtime_named_motion_skill(handle, template, effective_timeout_sec=timeout)


def test_bound_motion_is_sent_with_profile_target_and_without_preemption():
    handle_future = _Future(_motion_result(message="preset motion handshake completed"))
    client = _Client(_Future(_MotionHandle(handle_future)))
    node = _node(client)
    goal_handle = _GoalHandle()

    result = _run(node, goal_handle)

    assert result.success is True
    assert goal_handle.terminal == "succeeded"
    assert result.executed_primitives == ["runtime_named_motion:handshake"]
    sent = client.goals[0]
    assert (sent.name, sent.target, sent.interrupt) == ("handshake", "", False)
    assert node.confirmed == node.registered


def test_disabled_executor_fails_closed():
    node = _node(None)

    result = _run(node, _GoalHandle())

    assert (result.success, result.error_code) == (False, named_motion.RUNTIME_UNAVAILABLE)


def test_template_without_binding_is_rejected_before_dispatch():
    client = _Client(_Future(None))
    node = _node(client)

    result = _run(node, _GoalHandle(), template={"executor": "runtime_named_motion"})

    assert result.error_code == named_motion.INVALID_BINDING
    assert client.goals == []


def test_name_the_runtime_does_not_advertise_is_rejected_before_dispatch():
    client = _Client(_Future(None))
    node = _node(client)

    result = _run(node, _GoalHandle(), template={**TEMPLATE, "binding": {"motion": "bow"}})

    assert result.error_code == named_motion.UNKNOWN
    assert client.goals == []


def test_posture_names_only_count_for_skills_requiring_the_posture_capability():
    client = _Client(_Future(_MotionHandle(_Future(_motion_result()))))
    template = {**TEMPLATE, "binding": {"motion": "sit_down"}}

    assert _run(_node(client), _GoalHandle(), template=template).error_code == named_motion.UNKNOWN
    assert _run(_node(client, required=("motion.posture",)), _GoalHandle(), template=template).success is True


def test_runtime_without_named_motion_parameters_is_not_ready():
    node = _node(_Client(_Future(None)), status=_status({"motion.named": {}}))

    result = _run(node, _GoalHandle())

    assert result.error_code == named_motion.PLATFORM_NOT_READY


def test_unavailable_action_server_is_reported():
    node = _node(_Client(_Future(None), server_ready=False))

    assert _run(node, _GoalHandle()).error_code == named_motion.RUNTIME_UNAVAILABLE


def test_refused_goal_is_reported_and_releases_the_dispatch():
    client = _Client(_Future(_MotionHandle(_Future(None), accepted=False)))
    node = _node(client)

    result = _run(node, _GoalHandle())

    assert result.error_code == named_motion.REJECTED
    assert node.confirmed == node.registered


def _result_codes() -> dict[str, int]:
    """Every result-code constant the ExecuteNamedMotion action contract declares."""
    return {
        name: value
        for name, value in vars(RESULT).items()
        if name.isupper() and not name.startswith("_") and isinstance(value, int) and not isinstance(value, bool)
    }


def test_every_runtime_failure_code_has_an_explicit_public_mapping():
    """A result code added to the action must not fall through to the default silently."""
    codes = _result_codes()
    failures = {name: value for name, value in codes.items() if value != RESULT.NONE}

    assert "INTERNAL_ERROR" in failures  # the reflection really reads the contract
    unmapped = sorted(name for name, value in failures.items() if value not in named_motion.RUNTIME_ERROR_CODES)
    assert unmapped == []
    assert RESULT.NONE not in named_motion.RUNTIME_ERROR_CODES
    assert set(named_motion.RUNTIME_ERROR_CODES) <= set(codes.values())


@pytest.mark.parametrize(
    ("runtime_code", "public_code"),
    [
        (RESULT.UNKNOWN_MOTION, named_motion.UNKNOWN),
        (RESULT.INVALID_TARGET, named_motion.UNKNOWN),
        (RESULT.BUSY, named_motion.BUSY),
        (RESULT.MODE_NOT_ALLOWED, "CONTROL_MODE_MISMATCH"),
        (RESULT.STOP_LATCHED, named_motion.STOP_LATCHED),
        (RESULT.RUNTIME_UNAVAILABLE, named_motion.PLATFORM_NOT_READY),
        (RESULT.REJECTED_BY_PLATFORM, named_motion.REJECTED),
        (RESULT.UNSAFE_POSTURE, named_motion.REJECTED),
        (RESULT.TIMEOUT, named_motion.TIMEOUT),
        (RESULT.CANCELLED, named_motion.PREEMPTED),
        (RESULT.INTERNAL_ERROR, named_motion.FAILED),
    ],
)
def test_runtime_failure_codes_map_to_public_codes(runtime_code, public_code):
    failure = _motion_result(success=False, error_code=runtime_code, message="platform is in SIT_DOWN_DEFAULT")
    node = _node(_Client(_Future(_MotionHandle(_Future(failure)))))
    goal_handle = _GoalHandle()

    result = _run(node, goal_handle)

    assert (result.success, result.error_code) == (False, public_code)
    assert result.message == "platform is in SIT_DOWN_DEFAULT"
    assert goal_handle.terminal == "aborted"


def test_cancel_is_never_forwarded_and_the_real_outcome_is_reported():
    motion = _MotionHandle(_Future(_motion_result(message="preset motion handshake completed"), done_after_polls=3))
    node = _node(_Client(_Future(motion)))
    goal_handle = _GoalHandle(cancel_requested=True)

    result = _run(node, goal_handle)

    assert motion.cancel_calls == 0
    assert node.audits == ["cancel_not_supported"]
    assert result.success is True
    assert result.message == "cancel not supported; preset motion handshake completed"
    assert node.confirmed == node.registered


def test_deadline_keeps_the_admission_until_the_runtime_reports_a_terminal_state():
    result_future = _Future(None, done_after_polls=None)
    motion = _MotionHandle(result_future)
    node = _node(_Client(_Future(motion)))

    result = _run(node, _GoalHandle(), timeout=0.1)

    assert result.error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
    assert motion.cancel_calls == 0
    assert node.confirmed == []
    result_future.complete(_motion_result())
    assert node.confirmed == node.registered


def test_goal_response_timeout_awaits_a_late_goal_without_cancelling_it():
    send_future = _Future(None, done_after_polls=None)
    node = _node(_Client(send_future))

    result = _run(node, _GoalHandle())

    assert result.error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
    assert node.confirmed == []
    late_result = _Future(None, done_after_polls=None)
    late_motion = _MotionHandle(late_result)
    send_future.complete(late_motion)
    assert node.confirmed == []
    late_result.complete(_motion_result())
    assert late_motion.cancel_calls == 0
    assert node.confirmed == node.registered


def _descriptor_node(enabled, action="/motion/execute_named"):
    node = object.__new__(SkillExecutorNode)
    node._skill_templates = {}
    node._grasp_execution = {}
    node._placement_execution = {}
    node._pick_action_name = "/manipulation/execute_pick"
    node._place_action_name = "/manipulation/execute_place"
    node._semantic_map_target_service = ""
    node._runtime_named_motion_enabled = enabled
    node._runtime_named_motion_action = action
    node._runtime_name = "aimdk_robot"
    return node


def test_enabled_executor_registers_an_action_descriptor_bound_to_the_runtime():
    descriptor = _descriptor_node(True)._delegated_executor_descriptors()["runtime_named_motion"]
    other_runtime = _descriptor_node(True)
    other_runtime._runtime_name = "other_runtime"

    assert descriptor.endpoint_kind == "ros_action"
    assert descriptor.endpoint_name == "/motion/execute_named"
    assert (
        other_runtime._delegated_executor_descriptors()["runtime_named_motion"].configuration_digest
        != descriptor.configuration_digest
    )


def test_disabled_executor_registers_no_descriptor():
    assert "runtime_named_motion" not in _descriptor_node(False)._delegated_executor_descriptors()


@pytest.mark.parametrize(("enabled", "in_catalog"), [(True, False), (False, True)])
def test_runtime_and_catalog_must_agree_on_the_executor(enabled, in_catalog):
    node = object.__new__(SkillExecutorNode)
    node._runtime_named_motion_enabled = enabled
    templates = {"handshake": dict(TEMPLATE)} if in_catalog else {}
    snapshot = SimpleNamespace(templates=templates, enabled_skill_names=tuple(templates))

    with pytest.raises(ValueError, match="runtime_named_motion configuration mismatch"):
        node._validate_hri_runtime_catalog_consistency(snapshot)


def test_idle_mapped_skill_mode_never_requests_a_runtime_mode_switch():
    """Requesting idle would clear an operator stop latch, so it must never be sent."""
    node = object.__new__(SkillExecutorNode)
    node._runtime_enabled = True
    node._motion_mode_switch_lock = RLock()
    node._runtime_mode_map = {"named_motion": "idle"}
    status = SimpleNamespace(active_mode="idle", declared_modes=["idle", "stream"], stop_latched=True)
    node._runtime_status_for_admission = lambda: (status, "")
    node._required_control_mode_for_skill = lambda _name: "named_motion"
    node._ensure_skill_capabilities = lambda _name: (True, "")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("set_mode must not be requested")

    node._request_runtime_mode = forbidden

    assert node._ensure_skill_control_mode("handshake", SimpleNamespace(is_cancel_requested=False)) == (True, "")

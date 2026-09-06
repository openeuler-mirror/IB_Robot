"""Node-boundary regressions for scheduler certainty, deadlines, and wire bounds."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from std_srvs.srv import Trigger

from ibrobot_msgs.action import CloseInferenceSession, OpenInferenceSession, ScheduledDispatchInfer
from ibrobot_msgs.msg import InferenceOutcome, InferenceServingStatus, InferenceWorkCapacity, ScheduledInferenceError
from inference_service import pipeline_policy_node as pipeline_policy_module
from inference_service.backends import BackendCapabilities
from inference_service.global_inference_scheduler_node import (
    GlobalInferenceSchedulerNode,
    _DownstreamCall,
)
from inference_service.pipeline_policy_node import PipelinePolicyNode
from inference_service.scheduler.action_idempotency import replay_terminal
from inference_service.scheduler.global_scheduler_core import SchedulerError
from inference_service.scheduler.goal_slots import GoalSlotPool
from inference_service.scheduler.ledger import IdempotencyLedger, LedgerAction
from inference_service.scheduler.time_domains import monotonic_expiry_to_ros_ns
from inference_service.scheduler.wire_bounds import set_scheduled_error, utf8_size
from inference_service.scheduler.work_classes import WorkClass, work_class_name
from robot_config.contract_utils import ActionSpec, Contract, ObservationSpec, iter_specs
from robot_config.inference_runtime_options import effective_latency_runtime_options

SESSION_ID = "00112233-4455-4677-8899-aabbccddeeff"
REQUEST_ID = "11112233-4455-4677-8899-aabbccddeeff"
BOOT_ID = "22222222-2222-4222-8222-222222222222"
NEW_BOOT_ID = "33333333-3333-4333-8333-333333333333"


class _Feature:
    def __init__(self, feature_type: str, shape: tuple[int, ...]) -> None:
        self.feature_type = feature_type
        self.shape = shape

    def model_dump(self, *, mode: str):
        assert mode == "json"
        return {"type": self.feature_type, "shape": list(self.shape)}


class _GoalHandle:
    def __init__(self, request) -> None:
        self.request = request
        self.aborted = False
        self.succeeded = False
        self.canceled_status = False
        self.is_cancel_requested = False

    def abort(self) -> None:
        self.aborted = True

    def succeed(self) -> None:
        self.succeeded = True

    def canceled(self) -> None:
        self.canceled_status = True


def _compatibility_node(*, state_shape: tuple[int, ...]) -> PipelinePolicyNode:
    node = object.__new__(PipelinePolicyNode)
    node._contract = Contract(
        name="test",
        version=1,
        rate_hz=50.0,
        max_duration_s=10.0,
        observations=[
            ObservationSpec(
                key="observation.state",
                topic="/joint_states",
                type="sensor_msgs/msg/JointState",
                selector={"names": ["1", "2"]},
            )
        ],
        actions=[
            ActionSpec(
                key="action",
                publish_topic="/arm",
                type="std_msgs/msg/Float64MultiArray",
                selector={"names": ["action.0", "action.1"]},
                safety_behavior="hold",
            )
        ],
        tasks=[],
        recording={},
    )
    specs = list(iter_specs(node._contract))
    node._obs_specs = [spec for spec in specs if not spec.is_action]
    node._state_specs = node._obs_specs
    node._frequency = 50.0
    node._n_obs_steps = 1
    node._manifest = SimpleNamespace(
        policy=SimpleNamespace(
            policy_type="act",
            input_features={"observation.state": _Feature("STATE", state_shape)},
            output_features={"action": _Feature("ACTION", (2,))},
            max_action_dimension=2,
        )
    )
    node._config = SimpleNamespace(default_task="pick")
    return node


def test_pipeline_compatibility_fingerprint_covers_model_input_abi():
    compatible_a = _compatibility_node(state_shape=(2,))
    compatible_b = _compatibility_node(state_shape=(2,))
    incompatible = _compatibility_node(state_shape=(3,))

    first = PipelinePolicyNode._build_pipeline_compatibility_fingerprint(compatible_a)
    second = PipelinePolicyNode._build_pipeline_compatibility_fingerprint(compatible_b)
    different = PipelinePolicyNode._build_pipeline_compatibility_fingerprint(incompatible)

    assert first == second
    assert different != first


def _scheduler_stub(**overrides):
    values = {
        "_goal_acceptance_timeout_ns": 100_000_000,
        "_max_prompt_bytes": 4,
        "_max_error_message_bytes": 32,
        "_max_error_details_bytes": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_expired_deadline_is_not_sent_downstream():
    client = SimpleNamespace(
        wait_calls=0,
        send_calls=0,
    )

    def wait_for_server(*, timeout_sec):
        client.wait_calls += 1
        return True

    def send_goal_async(_goal):
        client.send_calls += 1
        raise AssertionError("expired goal must not be sent")

    client.wait_for_server = wait_for_server
    client.send_goal_async = send_goal_async

    call = GlobalInferenceSchedulerNode._call_downstream(
        _scheduler_stub(), client, object(), deadline_monotonic_ns=time.monotonic_ns() - 1
    )

    assert call.certainty == "not_started"
    assert call.reason == "deadline_exceeded"
    assert client.wait_calls == 0
    assert client.send_calls == 0


def test_late_goal_acceptance_runs_cleanup_after_acceptance_timeout():
    class _DeferredFuture:
        def __init__(self) -> None:
            self.callback = None
            self.value = None

        def add_done_callback(self, callback):
            self.callback = callback

        def result(self):
            return self.value

        def complete(self, value) -> None:
            self.value = value
            assert self.callback is not None
            self.callback(self)

    sent = _DeferredFuture()
    client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: sent,
    )
    node = _scheduler_stub(_goal_acceptance_timeout_ns=1_000_000)
    late_acceptances: list[object] = []

    call = GlobalInferenceSchedulerNode._call_downstream(
        node,
        client,
        object(),
        deadline_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        late_acceptance_callback=late_acceptances.append,
    )

    assert call.certainty == "unknown"
    assert call.reason == "goal_acceptance_timeout"
    late_goal_handle = SimpleNamespace(accepted=True)
    sent.complete(late_goal_handle)
    assert late_acceptances == [late_goal_handle]


def test_global_rejects_prompt_above_configured_byte_limit_before_ledger():
    goal = ScheduledDispatchInfer.Goal()
    goal.prompt = "12345"
    goal.session_id = "session"
    goal.request_id = "request"
    goal_handle = _GoalHandle(goal)
    node = _scheduler_stub()
    node._set_error = GlobalInferenceSchedulerNode._set_error.__get__(node)
    node._idempotency_failure = GlobalInferenceSchedulerNode._idempotency_failure.__get__(node)

    result = GlobalInferenceSchedulerNode._execute_idempotent(
        node,
        goal_handle=goal_handle,
        action=LedgerAction.DISPATCH,
        key=("dispatch",),
        payload={},
        deadline=goal.deadline,
        is_open=False,
        execute=lambda _goal_handle: (_ for _ in ()).throw(AssertionError("must not execute")),
        request_id=goal.request_id,
    )

    assert goal_handle.aborted
    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "prompt_too_large"


def test_global_cached_canceled_terminal_replays_canceled_ros_status():
    goal_handle = _GoalHandle(ScheduledDispatchInfer.Goal())
    result = object()

    replayed = replay_terminal(
        goal_handle,
        SimpleNamespace(ros_status=5, result=result),
    )

    assert replayed is result
    assert goal_handle.canceled_status is True
    assert goal_handle.aborted is False
    assert goal_handle.succeeded is False


def test_global_goal_slots_accept_two_and_reject_third_until_release():
    slots = GoalSlotPool(("dispatch",))

    assert slots.try_acquire("dispatch")
    assert slots.try_acquire("dispatch")
    assert not slots.try_acquire("dispatch")
    assert slots.run("dispatch", lambda goal_handle: goal_handle, "result") == "result"
    assert slots.try_acquire("dispatch")
    slots.run("dispatch", lambda _goal_handle: None, object())
    slots.run("dispatch", lambda _goal_handle: None, object())


def test_lower_priority_goal_slots_cannot_consume_priority_zero_reserve():
    slots = GoalSlotPool(("dispatch",), capacity=4, protected_capacity=2)

    assert slots.try_acquire("dispatch")
    assert slots.try_acquire("dispatch")
    assert not slots.try_acquire("dispatch")
    assert slots.try_acquire("dispatch", protected=True)
    assert slots.try_acquire("dispatch", protected=True)
    assert not slots.try_acquire("dispatch", protected=True)

    slots.run("dispatch", lambda _goal_handle: None, object())
    slots.run("dispatch", lambda _goal_handle: None, object())
    slots.run("dispatch", lambda _goal_handle: None, object(), protected=True)
    slots.run("dispatch", lambda _goal_handle: None, object(), protected=True)


def _idempotent_node() -> GlobalInferenceSchedulerNode:
    node = object.__new__(GlobalInferenceSchedulerNode)
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=time.time_ns()))
    node._default_open_timeout_ns = 2_000_000_000
    node._default_request_timeout_ns = 2_000_000_000
    node._max_prompt_bytes = 4096
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._ingress_ledger = IdempotencyLedger(
        max_session_records=4,
        max_duplicate_waiters_per_request=2,
        terminal_session_retention_ns=10_000_000_000,
        now_ns=time.monotonic_ns,
    )
    return node


def _dispatch_goal(*, priority: int, deadline_ns: int, fallback_chain: list[str]) -> ScheduledDispatchInfer.Goal:
    goal = ScheduledDispatchInfer.Goal()
    goal.session_id = SESSION_ID
    goal.session_generation = 1
    goal.request_id = REQUEST_ID
    goal.target_pipeline_id = "policy"
    goal.fallback_chain = fallback_chain
    goal.priority = priority
    goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
    return goal


def test_global_open_is_logical_only_and_deadline_changes_replay_same_result():
    node = _idempotent_node()
    open_calls: list[str] = []

    class _Core:
        @staticmethod
        def open_session(*, session_id):
            open_calls.append(session_id)
            return SimpleNamespace(
                session_generation=1,
                lease_expires_at_ns=time.monotonic_ns() + 30_000_000_000,
            )

    node._core = _Core()
    first_goal = OpenInferenceSession.Goal()
    first_goal.session_id = SESSION_ID
    first_deadline_ns = time.time_ns() + 1_000_000_000
    first_goal.deadline.sec, first_goal.deadline.nanosec = divmod(first_deadline_ns, 1_000_000_000)
    second_goal = OpenInferenceSession.Goal()
    second_goal.session_id = SESSION_ID
    second_deadline_ns = time.time_ns() + 9_000_000_000
    second_goal.deadline.sec, second_goal.deadline.nanosec = divmod(second_deadline_ns, 1_000_000_000)
    first = _GoalHandle(first_goal)
    second = _GoalHandle(second_goal)

    first_result = GlobalInferenceSchedulerNode._open_endpoint(node, first)
    second_result = GlobalInferenceSchedulerNode._open_endpoint(node, second)

    assert first_result.success and second_result.success
    assert first_result.session_generation == second_result.session_generation == 1
    assert first_result.actual_pipeline_id == second_result.actual_pipeline_id == ""
    assert first_result.deployment_fingerprint == second_result.deployment_fingerprint == ""
    assert open_calls == [SESSION_ID]


def test_global_expired_open_does_not_create_logical_session():
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._core = SimpleNamespace(
        open_session=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("expired Open must not create a logical session")
        )
    )
    goal = OpenInferenceSession.Goal()
    goal.session_id = SESSION_ID
    expired_ns = time.time_ns() - 1_000_000
    goal.deadline.sec, goal.deadline.nanosec = divmod(expired_ns, 1_000_000_000)
    goal_handle = _GoalHandle(goal)

    result = GlobalInferenceSchedulerNode._open_once(node, goal_handle, None)

    assert goal_handle.aborted
    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "deadline_exceeded"


def test_global_close_without_pipeline_bindings_completes_locally():
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._default_request_timeout_ns = 2_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._core = SimpleNamespace(
        begin_close=lambda **_kwargs: None,
        session_record=lambda _session_id: SimpleNamespace(session_generation=1),
        wait_for_bindings_to_settle=lambda *_args: True,
        close_bindings=lambda _session_id: [],
        record_close_complete=lambda _session_id, **_kwargs: 2,
    )
    goal = CloseInferenceSession.Goal()
    goal.session_id = SESSION_ID
    goal.session_generation = 1
    deadline_ns = time.time_ns() + 1_000_000_000
    goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
    goal_handle = _GoalHandle(goal)

    result = GlobalInferenceSchedulerNode._close_once(node, goal_handle, None)

    assert result.success
    assert result.pipeline_id == ""
    assert result.closed_session_generation == 1
    assert result.drained_generation == 2


def test_global_close_scheduler_error_uses_clean_wire_code():
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._default_request_timeout_ns = 2_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._core = SimpleNamespace(
        begin_close=lambda **_kwargs: (_ for _ in ()).throw(SchedulerError("close_in_progress"))
    )
    goal = CloseInferenceSession.Goal()
    goal.session_id = SESSION_ID
    goal.session_generation = 1
    deadline_ns = time.time_ns() + 1_000_000_000
    goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
    goal_handle = _GoalHandle(goal)

    result = GlobalInferenceSchedulerNode._close_once(node, goal_handle, None)

    assert goal_handle.aborted
    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "close_in_progress"


def test_global_close_explicit_not_started_restores_session_for_retry():
    restored: list[str] = []
    failures: list[str] = []
    downstream = CloseInferenceSession.Result()
    downstream.session_id = SESSION_ID
    downstream.pipeline_id = "policy"
    downstream.success = False
    downstream.outcome.value = InferenceOutcome.NOT_STARTED
    downstream.error.code = "no_session_capacity"
    downstream.error.recoverable = True

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._default_request_timeout_ns = 2_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._core = SimpleNamespace(
        begin_close=lambda **_kwargs: None,
        session_record=lambda _session_id: SimpleNamespace(session_generation=1),
        wait_for_bindings_to_settle=lambda *_args: True,
        close_bindings=lambda _session_id: [SimpleNamespace(pipeline_id="policy", pipeline_generation=7)],
        record_close_not_started=restored.append,
        mark_session_failed=lambda session_id, **_kwargs: failures.append(session_id),
    )
    node._pipeline_clients = {"policy": {"close": object()}}
    node._candidate_by_id = {
        "policy": SimpleNamespace(
            pipeline_id="policy",
            deployment_fingerprint="deployment",
            runtime_policy_fingerprint="runtime",
        )
    }
    node._call_downstream = lambda *_args, **_kwargs: _DownstreamCall(
        "not_started", result=downstream, reason="no_session_capacity"
    )
    goal = CloseInferenceSession.Goal()
    goal.session_id = SESSION_ID
    goal.session_generation = 1
    deadline_ns = time.time_ns() + 1_000_000_000
    goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
    goal_handle = _GoalHandle(goal)

    result = GlobalInferenceSchedulerNode._close_once(node, goal_handle, None)

    assert goal_handle.aborted
    assert not result.success
    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "no_session_capacity"
    assert result.error.recoverable
    assert restored == [SESSION_ID]
    assert failures == []


def test_work_class_source_matches_wire_contract():
    assert int(WorkClass.SESSION_CONTROL) == InferenceWorkCapacity.SESSION_CONTROL
    assert int(WorkClass.ACTION_GENERATION) == InferenceWorkCapacity.ACTION_GENERATION
    assert work_class_name(WorkClass.SESSION_CONTROL) == "session_control"
    assert work_class_name(WorkClass.ACTION_GENERATION) == "action_generation"


def test_positive_priority_ignores_caller_deadline_and_fallback_for_dispatch_identity():
    node = _idempotent_node()
    executions: list[int] = []

    def execute(goal_handle, _entry):
        executions.append(goal_handle.request.deadline.sec)
        result = ScheduledDispatchInfer.Result()
        result.success = True
        result.outcome.value = InferenceOutcome.COMPLETED
        goal_handle.succeed()
        return result

    node._dispatch_once = execute
    first = _GoalHandle(
        _dispatch_goal(
            priority=3,
            deadline_ns=time.time_ns() - 1_000_000_000,
            fallback_chain=["missing_a"],
        )
    )
    second = _GoalHandle(
        _dispatch_goal(
            priority=3,
            deadline_ns=time.time_ns() + 99_000_000_000,
            fallback_chain=["missing_b"],
        )
    )

    first_result = GlobalInferenceSchedulerNode._dispatch_endpoint(node, first)
    second_result = GlobalInferenceSchedulerNode._dispatch_endpoint(node, second)

    assert first_result.success and second_result.success
    assert first.succeeded and second.succeeded
    assert len(executions) == 1
    first_deadline = first.request.deadline.sec * 1_000_000_000 + first.request.deadline.nanosec
    second_deadline = second.request.deadline.sec * 1_000_000_000 + second.request.deadline.nanosec
    assert first_deadline == second_deadline
    assert first_deadline > time.time_ns()


def test_priority_zero_resolves_zero_deadline_once_for_replay():
    node = _idempotent_node()
    executions: list[int] = []

    def execute(goal_handle, _entry):
        deadline_ns = goal_handle.request.deadline.sec * 1_000_000_000 + goal_handle.request.deadline.nanosec
        executions.append(deadline_ns)
        result = ScheduledDispatchInfer.Result()
        result.success = True
        result.outcome.value = InferenceOutcome.COMPLETED
        goal_handle.succeed()
        return result

    node._dispatch_once = execute
    first = _GoalHandle(_dispatch_goal(priority=0, deadline_ns=0, fallback_chain=["backup"]))
    second = _GoalHandle(_dispatch_goal(priority=0, deadline_ns=0, fallback_chain=["backup"]))

    GlobalInferenceSchedulerNode._dispatch_endpoint(node, first)
    GlobalInferenceSchedulerNode._dispatch_endpoint(node, second)

    assert len(executions) == 1
    assert executions[0] > time.time_ns()
    replay_deadline = second.request.deadline.sec * 1_000_000_000 + second.request.deadline.nanosec
    assert replay_deadline == executions[0]


def test_priority_zero_estimate_includes_lazy_open_and_dispatch_profiles():
    calls: list[tuple[str, int, str]] = []

    class _Registry:
        @staticmethod
        def closure_p99_ms(*, work_class, closure_key, **_kwargs):
            calls.append(("closure", work_class, closure_key))
            return 10.0 if work_class == 1 else 20.0

        @staticmethod
        def goal_acceptance_p999_ms(*, work_class, **_kwargs):
            calls.append(("acceptance", work_class, ""))
            return 1.0 if work_class == 1 else 2.0

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._profile_registries = {"policy": _Registry()}
    node._profile_errors = {}
    node._goal_acceptance_safety_margin_ms = 3
    node._dispatch_safety_margin_ms = 4
    node._status_lock = threading.Lock()
    status = InferenceServingStatus()
    status.pipeline_compatibility_fingerprint = "c" * 64
    node._serving_status = {"policy": SimpleNamespace(message=status)}

    estimate_ms, reason = GlobalInferenceSchedulerNode._priority_zero_estimate_ms(
        node,
        pipeline_id="policy",
        requires_open=True,
        prompt_bytes=128,
    )

    assert reason == ""
    assert estimate_ms == 43.0
    assert calls == [
        ("closure", 1, "session_open"),
        ("acceptance", 1, ""),
        ("closure", 2, "full_infer"),
        ("acceptance", 2, ""),
    ]


def test_missing_profile_fails_closed_only_when_priority_zero_estimate_is_requested():
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._profile_registries = {}
    node._profile_errors = {"policy": "priority_zero_profile_unavailable:profile_path_missing"}

    estimate_ms, reason = GlobalInferenceSchedulerNode._priority_zero_estimate_ms(
        node,
        pipeline_id="policy",
        requires_open=True,
        prompt_bytes=0,
    )

    assert estimate_ms is None
    assert reason == "priority_zero_profile_unavailable:profile_path_missing"


def test_priority_zero_readiness_does_not_require_profiles():
    candidate = SimpleNamespace(pipeline_id="policy", required=True, compatibility_group="group")
    status = InferenceServingStatus()
    status.pipeline_compatibility_fingerprint = "compatible"
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._candidates = [candidate]
    node._candidate_by_id = {"policy": candidate}
    node._default_target_pipeline_id = "policy"
    node._default_priority = 0
    node._status_reason = lambda _candidate, **_kwargs: ""
    node._status_lock = threading.RLock()
    node._serving_status = {"policy": SimpleNamespace(message=status)}

    response = GlobalInferenceSchedulerNode._readiness_callback(node, Trigger.Request(), Trigger.Response())

    assert response.success


def test_capacity_accepting_uses_the_requested_work_class():
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._status_lock = threading.Lock()
    status = InferenceServingStatus()
    session_capacity = InferenceWorkCapacity()
    session_capacity.work_class = 1
    session_capacity.accepting_requests = True
    status.capacities.append(session_capacity)
    action_capacity = InferenceWorkCapacity()
    action_capacity.work_class = 2
    action_capacity.accepting_requests = False
    status.capacities.append(action_capacity)
    node._serving_status = {"policy": SimpleNamespace(message=status)}

    assert GlobalInferenceSchedulerNode._capacity_accepting(node, "policy", 1)
    assert not GlobalInferenceSchedulerNode._capacity_accepting(node, "policy", 2)


def test_nonzero_default_priority_readiness_checks_only_the_default_target():
    target = SimpleNamespace(pipeline_id="ascend", required=False, compatibility_group="group")
    other = SimpleNamespace(pipeline_id="cpu", required=True, compatibility_group="group")
    calls = []

    def status_reason(candidate, **kwargs):
        calls.append((candidate.pipeline_id, kwargs.get("required_priority")))
        if candidate.pipeline_id == "ascend" and kwargs.get("required_priority") == 3:
            return "unsupported_default_priority"
        return ""

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._candidates = [target, other]
    node._candidate_by_id = {candidate.pipeline_id: candidate for candidate in node._candidates}
    node._default_target_pipeline_id = "ascend"
    node._default_priority = 3
    node._status_reason = status_reason

    response = GlobalInferenceSchedulerNode._readiness_callback(node, Trigger.Request(), Trigger.Response())

    assert not response.success
    assert response.message == '{"ascend": "unsupported_default_priority"}'
    assert calls == [("cpu", None), ("ascend", 3)]


def test_downstream_not_started_recoverability_is_preserved():
    result = ScheduledDispatchInfer.Result()
    result.outcome.value = InferenceOutcome.NOT_STARTED
    result.error.recoverable = False

    assert GlobalInferenceSchedulerNode._call_recoverable(_DownstreamCall("not_started", result=result)) is False


def test_global_not_started_can_be_marked_nonrecoverable():
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    goal_handle = _GoalHandle(_dispatch_goal(priority=8, deadline_ns=0, fallback_chain=[]))

    result = GlobalInferenceSchedulerNode._dispatch_not_started(
        node,
        goal_handle,
        goal_handle.request,
        "policy",
        "unsupported_priority",
        "",
        recoverable=False,
    )

    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.recoverable is False


def _routing_node(candidate_ids: tuple[str, ...]):
    terminal_sessions: list[str] = []

    class _Core:
        @staticmethod
        def resolve_dispatch_plan(**_kwargs):
            return SimpleNamespace(candidate_ids=candidate_ids)

        @staticmethod
        def prepare_dispatch_candidate(*, pipeline_id, **_kwargs):
            return SimpleNamespace(
                pipeline_id=pipeline_id,
                pipeline_generation=7,
                needs_open=False,
            )

        @staticmethod
        def record_request_terminal(session_id):
            terminal_sessions.append(session_id)

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = _Core()
    node._candidate_by_id = {
        pipeline_id: SimpleNamespace(pipeline_id=pipeline_id, hardware_resource_id="ascend:0")
        for pipeline_id in candidate_ids
    }
    node._default_request_timeout_ns = 2_000_000_000
    node._status_reason = lambda _candidate: ""
    node._compatibility_reason = lambda _target, _candidate: ""
    node._capacity_accepting = lambda _pipeline_id, _work_class: True
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._release_reservation = lambda _reservation: None
    node._mark_reservation_unknown = lambda _reservation: None
    node._deadline_reservations = SimpleNamespace(wait_for_turn=lambda *_args, **_kwargs: "ready")
    return node, terminal_sessions


def test_priority_zero_checks_each_candidate_and_dispatches_first_feasible_fallback():
    node, terminal_sessions = _routing_node(("policy", "backup"))
    checked: list[tuple[str, bool]] = []
    dispatched: list[str] = []

    def reserve(*, pipeline_id, requires_open, **_kwargs):
        checked.append((pipeline_id, requires_open))
        reservation = SimpleNamespace() if pipeline_id == "backup" else None
        return reservation, "profile unavailable" if pipeline_id == "policy" else ""

    def dispatch(_goal_handle, goal, candidate, **_kwargs):
        dispatched.append(candidate.pipeline_id)
        result = ScheduledDispatchInfer.Result()
        result.success = True
        result.session_generation = 7
        result.outcome.value = InferenceOutcome.COMPLETED
        return _DownstreamCall("completed", result=result), goal

    node._reserve_priority_zero = reserve
    node._dispatch_bound_pipeline = dispatch
    goal = _dispatch_goal(
        priority=0,
        deadline_ns=time.time_ns() + 2_000_000_000,
        fallback_chain=["backup"],
    )
    goal_handle = _GoalHandle(goal)

    result = GlobalInferenceSchedulerNode._dispatch_once(node, goal_handle, None)

    assert result.success
    assert checked == [("policy", False), ("backup", False)]
    assert dispatched == ["backup"]
    assert terminal_sessions == [SESSION_ID]


def test_priority_zero_returns_error_without_dispatch_when_all_candidates_miss_deadline():
    node, _terminal_sessions = _routing_node(("policy", "backup"))
    node._reserve_priority_zero = lambda **_kwargs: (None, "measured closure exceeds deadline")
    node._dispatch_bound_pipeline = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("infeasible priority-0 request must not be dispatched")
    )
    goal_handle = _GoalHandle(
        _dispatch_goal(
            priority=0,
            deadline_ns=time.time_ns() + 1_000_000,
            fallback_chain=["backup"],
        )
    )

    result = GlobalInferenceSchedulerNode._dispatch_once(node, goal_handle, None)

    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "no_feasible_deadline"
    assert result.error.message == "measured closure exceeds deadline"


def test_first_dispatch_lazily_opens_selected_pipeline_binding():
    terminal_sessions: list[str] = []
    binding_opens: list[str] = []
    binding_generations: list[int] = []

    class _Core:
        @staticmethod
        def resolve_dispatch_plan(**_kwargs):
            return SimpleNamespace(candidate_ids=("policy",))

        @staticmethod
        def prepare_dispatch_candidate(**_kwargs):
            return SimpleNamespace(pipeline_id="policy", pipeline_generation=0, needs_open=True)

        @staticmethod
        def record_binding_open_success(*, pipeline_id, pipeline_generation, **_kwargs):
            binding_opens.append(pipeline_id)
            binding_generations.append(pipeline_generation)
            return False

        @staticmethod
        def record_request_terminal(session_id):
            terminal_sessions.append(session_id)

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = _Core()
    node._candidate_by_id = {"policy": SimpleNamespace(pipeline_id="policy", hardware_resource_id="ascend:0")}
    node._default_request_timeout_ns = 2_000_000_000
    node._status_reason = lambda _candidate: ""
    node._compatibility_reason = lambda _target, _candidate: ""
    node._capacity_accepting = lambda _pipeline_id, _work_class: True
    node._reserve_priority_zero = lambda **_kwargs: (SimpleNamespace(), "")
    node._release_reservation = lambda _reservation: None
    node._mark_reservation_unknown = lambda _reservation: None
    node._deadline_reservations = SimpleNamespace(wait_for_turn=lambda *_args, **_kwargs: "ready")
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    open_result = OpenInferenceSession.Result()
    open_result.success = True
    open_result.session_generation = 9
    node._open_dispatch_binding = lambda *_args, **_kwargs: _DownstreamCall("completed", result=open_result)

    def dispatch(_goal_handle, goal, _candidate, *, pipeline_generation, **_kwargs):
        result = ScheduledDispatchInfer.Result()
        result.success = True
        result.session_generation = pipeline_generation
        result.outcome.value = InferenceOutcome.COMPLETED
        return _DownstreamCall("completed", result=result), goal

    node._dispatch_bound_pipeline = dispatch
    goal_handle = _GoalHandle(
        _dispatch_goal(
            priority=0,
            deadline_ns=time.time_ns() + 1_000_000_000,
            fallback_chain=[],
        )
    )

    result = GlobalInferenceSchedulerNode._dispatch_once(node, goal_handle, None)

    assert result.success
    assert result.session_generation == 1
    assert binding_opens == ["policy"]
    assert binding_generations == [9]
    assert terminal_sessions == [SESSION_ID]


def test_positive_priority_dispatches_target_without_profile_admission():
    node, _terminal_sessions = _routing_node(("policy",))
    node._reserve_priority_zero = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("positive priority must not query deadline profiles")
    )
    dispatched: list[str] = []

    def dispatch(_goal_handle, goal, candidate, **_kwargs):
        dispatched.append(candidate.pipeline_id)
        result = ScheduledDispatchInfer.Result()
        result.success = True
        result.session_generation = 7
        result.outcome.value = InferenceOutcome.COMPLETED
        return _DownstreamCall("completed", result=result), goal

    node._dispatch_bound_pipeline = dispatch
    goal_handle = _GoalHandle(
        _dispatch_goal(
            priority=5,
            deadline_ns=time.time_ns() - 1_000_000_000,
            fallback_chain=["ignored"],
        )
    )

    result = GlobalInferenceSchedulerNode._dispatch_once(node, goal_handle, None)

    assert result.success
    assert dispatched == ["policy"]


def test_pipeline_executor_preserves_legacy_thread_count(monkeypatch):
    thread_counts = []
    configs = iter(
        (
            (SimpleNamespace(scheduler_enabled=False), "legacy"),
            (SimpleNamespace(scheduler_enabled=True), "scheduled"),
        )
    )

    class _Node:
        @staticmethod
        def destroy_node():
            return None

    class _Executor:
        def __init__(self, *, num_threads):
            thread_counts.append(num_threads)

        @staticmethod
        def add_node(_node):
            return None

        @staticmethod
        def spin():
            return None

    monkeypatch.setattr(pipeline_policy_module.rclpy, "init", lambda **_kwargs: None)
    monkeypatch.setattr(pipeline_policy_module.rclpy, "ok", lambda: False)
    monkeypatch.setattr(pipeline_policy_module, "_read_config", lambda: next(configs))
    monkeypatch.setattr(pipeline_policy_module, "PipelinePolicyNode", lambda _config, node_name: _Node())
    monkeypatch.setattr(pipeline_policy_module, "MultiThreadedExecutor", _Executor)

    pipeline_policy_module.main()
    pipeline_policy_module.main()

    assert thread_counts == [4, 8]


def test_disabled_pipeline_config_has_no_scheduled_runtime_state():
    config = pipeline_policy_module.PipelineNodeConfig(
        pipeline_id="policy",
        model_path="/model",
        deployment="cpu",
        execution_mode="monolithic",
        request_timeout=5.0,
        default_task="",
        runtime_options_json="{}",
        robot_config_path="/robot.yaml",
        use_sim=False,
        action_server="/inference/policy/dispatch",
        reset_service="/inference/policy/reset",
        health_topic="/inference/policy/health",
        action_topic="/actions/policy",
        request_topic="",
        result_topic="",
        heartbeat_topic="",
    )

    assert config.scheduler_enabled is False
    assert config.runtime_policy_json == ""
    assert config.runtime_policy_fingerprint == ""
    assert config.hardware_resource_id == ""
    assert config.scheduled_open_session == ""
    assert config.scheduled_dispatch == ""
    assert config.scheduled_close_session == ""
    assert config.scheduled_serving_status == ""
    assert pipeline_policy_module._pipeline_executor_threads(config) == 4


def test_scheduled_pipeline_executor_scales_with_public_capacity():
    config = SimpleNamespace(
        scheduler_enabled=True,
        public_capacity_json='{"action_generation":{"max_in_flight":12}}',
    )

    assert pipeline_policy_module._pipeline_executor_threads(config) == 16


def _runtime_policy_node(runtime_options_json: str) -> PipelinePolicyNode:
    node = object.__new__(PipelinePolicyNode)
    node._config = SimpleNamespace(
        pipeline_id="policy",
        execution_mode="monolithic",
        hardware_resource_id="ascend:0",
        public_capacity_json='{"session_control": {"max_in_flight": 1}}',
        runtime_options_json=runtime_options_json,
        scheduled_open_session="/inference/policy/session/open",
        scheduled_dispatch="/inference/policy/scheduled_dispatch",
        scheduled_close_session="/inference/policy/session/close",
        scheduled_serving_status="/inference/policy/serving_status",
        health_topic="/inference/policy/health",
    )
    node._manifest = SimpleNamespace(fingerprint="d" * 64)
    return node


def _runtime_policy(runtime_options: dict) -> dict:
    return {
        "pipeline_id": "policy",
        "execution_mode": "monolithic",
        "hardware_resource_id": "ascend:0",
        "deployment_fingerprint": "d" * 64,
        "public_capacity": {"session_control": {"max_in_flight": 1}},
        "runtime_options": effective_latency_runtime_options(runtime_options),
        "transport": {
            "open_session": "/inference/policy/session/open",
            "dispatch": "/inference/policy/scheduled_dispatch",
            "close_session": "/inference/policy/session/close",
            "serving_status": "/inference/policy/serving_status",
            "health_topic": "/inference/policy/health",
        },
    }


def test_scheduled_runtime_policy_rejects_node_side_runtime_option_override():
    """SSOT unchanged, only the node parameter enables collection: refuse startup.

    The runtime policy (and the latency profiles calibrated against it) is
    declared in the SSOT; a runtime_options_json override that changes the
    actually-executed options must not silently serve under that identity.
    """
    node = _runtime_policy_node('{"auto_horizon_enabled": true}')
    policy = _runtime_policy({})  # SSOT declares collection disabled

    with pytest.raises(RuntimeError, match="runtime_options mismatch"):
        PipelinePolicyNode._validate_runtime_policy(node, policy)


def test_scheduled_runtime_policy_accepts_matching_and_normalized_options():
    node = _runtime_policy_node('{"auto_horizon_enabled": true, "auto_horizon_sampling_step": 5}')
    policy = _runtime_policy({"auto_horizon_enabled": True, "auto_horizon_sampling_step": 5})
    PipelinePolicyNode._validate_runtime_policy(node, policy)

    # Explicit effective defaults stay identical to the omitted form.
    default_node = _runtime_policy_node("{}")
    explicit = _runtime_policy(
        {
            "model_dtype": "native",
            "auto_horizon_enabled": False,
            "auto_horizon_hold_threshold": 0.3,
            "auto_horizon_entropy_quantile": 0.9,
            "auto_horizon_run_length": 1,
            "auto_horizon_sampling_step": 3,
        }
    )
    PipelinePolicyNode._validate_runtime_policy(default_node, explicit)


def test_scheduled_runtime_policy_rejects_policy_without_options_identity():
    node = _runtime_policy_node("{}")
    policy = _runtime_policy({})
    del policy["runtime_options"]

    with pytest.raises(RuntimeError, match="runtime_options mismatch"):
        PipelinePolicyNode._validate_runtime_policy(node, policy)


def test_scheduled_close_drain_acquires_every_execution_slot_before_reset():
    node = object.__new__(PipelinePolicyNode)
    node._scheduled_operation_slots = threading.BoundedSemaphore(2)
    node._scheduled_operation_capacity = 2
    assert node._scheduled_operation_slots.acquire(blocking=False)
    released = threading.Event()

    def finish_inference() -> None:
        time.sleep(0.02)
        node._scheduled_operation_slots.release()
        released.set()

    thread = threading.Thread(target=finish_inference)
    thread.start()
    acquired = PipelinePolicyNode._acquire_scheduled_drain_slots(
        node,
        datetime.now(timezone.utc) + timedelta(seconds=1),
    )
    thread.join(timeout=1)

    assert released.is_set()
    assert acquired == 2
    PipelinePolicyNode._release_scheduled_drain_slots(node, acquired)


def test_scheduled_close_drain_timeout_releases_partially_acquired_slots():
    node = object.__new__(PipelinePolicyNode)
    node._scheduled_operation_slots = threading.BoundedSemaphore(2)
    node._scheduled_operation_capacity = 2
    assert node._scheduled_operation_slots.acquire(blocking=False)

    with pytest.raises(RuntimeError, match="timed out draining"):
        PipelinePolicyNode._acquire_scheduled_drain_slots(
            node,
            datetime.now(timezone.utc) + timedelta(milliseconds=10),
        )

    node._scheduled_operation_slots.release()
    assert node._scheduled_operation_slots.acquire(blocking=False)
    assert node._scheduled_operation_slots.acquire(blocking=False)
    assert not node._scheduled_operation_slots.acquire(blocking=False)
    node._scheduled_operation_slots.release()
    node._scheduled_operation_slots.release()


def test_mismatched_pipeline_id_cannot_reconcile_a_new_boot():
    reconciled = []
    deadline_reconciled = []
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._status_lock = threading.RLock()
    node._serving_status = {}
    node._trusted_status_cursors = {}
    node._candidate_by_id = {
        "policy": SimpleNamespace(
            pipeline_id="policy",
            deployment_fingerprint="deployment",
            runtime_policy_fingerprint="runtime",
            hardware_resource_id="resource",
            public_capacity={},
        )
    }
    now_ns = time.time_ns()
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns))
    node._clock_skew_tolerance_ns = 1_000_000_000
    node._status_stale_timeout_ns = 5_000_000_000
    node._core = SimpleNamespace(reconcile_pipeline_boot=reconciled.append)
    node._deadline_reservations = SimpleNamespace(reconcile_pipeline=deadline_reconciled.append)
    callback = node._make_status_callback("policy")

    previous = InferenceServingStatus()
    previous.pipeline_id = "policy"
    previous.boot_id = BOOT_ID
    previous.sequence = 1
    previous.header.stamp.sec = now_ns // 1_000_000_000
    previous.header.stamp.nanosec = now_ns % 1_000_000_000
    previous.state = InferenceServingStatus.IDLE
    previous.deployment_fingerprint = "deployment"
    previous.runtime_policy_fingerprint = "runtime"
    previous.configured_hardware_resource_id = "resource"
    previous.runtime_hardware_resource_id = "resource"
    previous.hardware_priority_levels = 1
    callback(previous)

    mismatched = InferenceServingStatus()
    mismatched.pipeline_id = "other"
    mismatched.boot_id = NEW_BOOT_ID
    mismatched.sequence = 1
    callback(mismatched)

    assert reconciled == []
    assert deadline_reconciled == []
    assert node._serving_status["policy"].invalid_reason == "pipeline_id_mismatch"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda status: setattr(status, "boot_id", "not-a-uuid"), "invalid_boot_or_sequence"),
        (
            lambda status: (
                setattr(status.header.stamp, "sec", 0),
                setattr(status.header.stamp, "nanosec", 0),
            ),
            "invalid_status_timestamp",
        ),
        (
            lambda status: setattr(status, "state", InferenceServingStatus.FAILED),
            f"state_{InferenceServingStatus.FAILED}",
        ),
        (
            lambda status: setattr(status, "deployment_fingerprint", "wrong"),
            "deployment_fingerprint_mismatch",
        ),
    ],
)
def test_invalid_new_boot_status_cannot_reconcile_quarantine(mutation, expected_reason):
    reconciled = []
    deadline_reconciled = []
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._status_lock = threading.RLock()
    node._serving_status = {}
    node._trusted_status_cursors = {"policy": (BOOT_ID, 7)}
    node._candidate_by_id = {
        "policy": SimpleNamespace(
            pipeline_id="policy",
            deployment_fingerprint="deployment",
            runtime_policy_fingerprint="runtime",
            hardware_resource_id="resource",
            public_capacity={},
        )
    }
    now_ns = time.time_ns()
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns))
    node._clock_skew_tolerance_ns = 1_000_000_000
    node._status_stale_timeout_ns = 5_000_000_000
    node._core = SimpleNamespace(reconcile_pipeline_boot=reconciled.append)
    node._deadline_reservations = SimpleNamespace(reconcile_pipeline=deadline_reconciled.append)
    callback = node._make_status_callback("policy")
    status = InferenceServingStatus()
    status.pipeline_id = "policy"
    status.boot_id = NEW_BOOT_ID
    status.sequence = 1
    status.header.stamp.sec = now_ns // 1_000_000_000
    status.header.stamp.nanosec = now_ns % 1_000_000_000
    status.state = InferenceServingStatus.IDLE
    status.deployment_fingerprint = "deployment"
    status.runtime_policy_fingerprint = "runtime"
    status.configured_hardware_resource_id = "resource"
    status.runtime_hardware_resource_id = "resource"
    status.hardware_priority_levels = 1
    mutation(status)

    callback(status)

    assert reconciled == []
    assert deadline_reconciled == []
    assert node._trusted_status_cursors["policy"] == (BOOT_ID, 7)
    assert node._serving_status["policy"].invalid_reason == expected_reason


def test_fully_valid_new_boot_reconciles_quarantine_once():
    reconciled = []
    deadline_reconciled = []
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._status_lock = threading.RLock()
    node._serving_status = {}
    node._trusted_status_cursors = {"policy": (BOOT_ID, 7)}
    candidate = SimpleNamespace(
        pipeline_id="policy",
        deployment_fingerprint="deployment",
        runtime_policy_fingerprint="runtime",
        hardware_resource_id="resource",
        public_capacity={},
    )
    node._candidate_by_id = {"policy": candidate}
    now_ns = time.time_ns()
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns))
    node._clock_skew_tolerance_ns = 1_000_000_000
    node._status_stale_timeout_ns = 5_000_000_000
    node._core = SimpleNamespace(reconcile_pipeline_boot=reconciled.append)
    node._deadline_reservations = SimpleNamespace(reconcile_pipeline=deadline_reconciled.append)
    status = InferenceServingStatus()
    status.pipeline_id = "policy"
    status.boot_id = NEW_BOOT_ID
    status.sequence = 1
    status.header.stamp.sec = now_ns // 1_000_000_000
    status.header.stamp.nanosec = now_ns % 1_000_000_000
    status.state = InferenceServingStatus.IDLE
    status.deployment_fingerprint = "deployment"
    status.runtime_policy_fingerprint = "runtime"
    status.configured_hardware_resource_id = "resource"
    status.runtime_hardware_resource_id = "resource"
    status.hardware_priority_levels = 1

    callback = node._make_status_callback("policy")
    callback(status)
    status.sequence = 2
    callback(status)

    assert reconciled == ["policy"]
    assert deadline_reconciled == ["policy"]
    assert node._trusted_status_cursors["policy"] == (NEW_BOOT_ID, 2)


def test_fallback_compatibility_does_not_require_primary_status():
    target = SimpleNamespace(pipeline_id="primary", compatibility_group="group")
    fallback = SimpleNamespace(pipeline_id="fallback", compatibility_group="group")
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._status_lock = threading.RLock()
    node._serving_status = {
        "fallback": SimpleNamespace(message=SimpleNamespace(pipeline_compatibility_fingerprint="fallback-contract"))
    }
    node._status_reason = lambda candidate: "missing" if candidate.pipeline_id == "primary" else ""

    assert GlobalInferenceSchedulerNode._compatibility_reason(node, target, fallback) == ""


def test_fallback_compatibility_still_compares_a_healthy_primary():
    target = SimpleNamespace(pipeline_id="primary", compatibility_group="group")
    fallback = SimpleNamespace(pipeline_id="fallback", compatibility_group="group")
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._status_lock = threading.RLock()
    node._serving_status = {
        "primary": SimpleNamespace(message=SimpleNamespace(pipeline_compatibility_fingerprint="primary-contract")),
        "fallback": SimpleNamespace(message=SimpleNamespace(pipeline_compatibility_fingerprint="fallback-contract")),
    }
    node._status_reason = lambda _candidate: ""

    assert (
        GlobalInferenceSchedulerNode._compatibility_reason(node, target, fallback) == "pipeline_compatibility_mismatch"
    )


def test_idle_close_identity_mismatch_marks_session_failed():
    session_id = "00112233-4455-4677-8899-aabbccddeeff"
    failures = []
    close_successes = []
    result = CloseInferenceSession.Result()
    result.success = True
    result.outcome.value = InferenceOutcome.COMPLETED
    result.session_id = session_id
    result.pipeline_id = "other"
    result.closed_session_generation = 3
    result.drained_generation = 4

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = SimpleNamespace(
        expired_sessions=lambda: [session_id],
        session_record=lambda _session_id: SimpleNamespace(session_generation=3),
        begin_close=lambda **_kwargs: None,
        wait_for_bindings_to_settle=lambda *_args: True,
        close_bindings=lambda _session_id: [SimpleNamespace(pipeline_id="policy", pipeline_generation=3)],
        record_binding_close_success=lambda *args: close_successes.append(args),
        record_close_complete=lambda *args, **_kwargs: 4,
        record_close_not_started=lambda _session_id: None,
        mark_session_failed=lambda session_id, **_kwargs: failures.append(session_id),
    )
    node._pipeline_clients = {"policy": {"close": object()}}
    node._candidate_by_id = {
        "policy": SimpleNamespace(
            pipeline_id="policy",
            deployment_fingerprint="deployment",
            runtime_policy_fingerprint="runtime",
        )
    }
    node._default_request_timeout_ns = 1_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._call_downstream = lambda *_args, **_kwargs: _DownstreamCall("completed", result=result)

    node._idle_sweep()

    assert failures == [session_id]
    assert close_successes == []


def test_pipeline_rejects_expired_open_before_session_admission():
    class _Controller:
        def begin_open(self, _session_id):
            raise AssertionError("expired Open must not reach session admission")

    goal = OpenInferenceSession.Goal()
    goal.session_id = "00112233-4455-4677-8899-aabbccddeeff"
    expired_ns = time.time_ns() - 1_000_000
    goal.deadline.sec = expired_ns // 1_000_000_000
    goal.deadline.nanosec = expired_ns % 1_000_000_000
    goal_handle = _GoalHandle(goal)
    node = object.__new__(PipelinePolicyNode)
    node._session_controller = _Controller()
    node._config = SimpleNamespace(
        pipeline_id="policy",
        request_timeout=1.0,
        max_prompt_bytes=4096,
        max_error_message_bytes=1024,
        max_error_details_bytes=8192,
    )

    result = PipelinePolicyNode._scheduled_open_once(node, goal_handle)

    assert goal_handle.aborted
    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "deadline_exceeded"


def test_pipeline_rejects_priority_not_supported_by_single_priority_backend():
    goal = ScheduledDispatchInfer.Goal()
    goal.request_id = REQUEST_ID
    goal.session_id = SESSION_ID
    goal.session_generation = 1
    goal.target_pipeline_id = "policy"
    goal.priority = 1
    deadline_ns = time.time_ns() + 1_000_000_000
    goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
    goal_handle = _GoalHandle(goal)
    node = object.__new__(PipelinePolicyNode)
    node._session_controller = SimpleNamespace()
    node._config = SimpleNamespace(
        pipeline_id="policy",
        request_timeout=1.0,
        max_prompt_bytes=4096,
        max_error_message_bytes=1024,
        max_error_details_bytes=8192,
        runtime_policy_fingerprint="runtime",
    )
    node._manifest = SimpleNamespace(fingerprint="deployment")
    node._require_manager = lambda: SimpleNamespace(capabilities=lambda _pipeline_id: BackendCapabilities())

    result = PipelinePolicyNode._scheduled_dispatch_once(node, goal_handle)

    assert goal_handle.aborted
    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "unsupported_priority"
    assert result.error.recoverable is False


def test_pipeline_post_inference_failure_is_completed_not_not_started():
    class _Controller:
        @staticmethod
        def admit(*_args, **_kwargs):
            return SimpleNamespace(accepted=True)

        @staticmethod
        def record_product_activity():
            return None

        @staticmethod
        def is_stale_generation(_generation):
            return True

        @staticmethod
        def release_in_flight(_work_class):
            return None

    goal = ScheduledDispatchInfer.Goal()
    goal.request_id = REQUEST_ID
    goal.session_id = SESSION_ID
    goal.session_generation = 1
    goal.target_pipeline_id = "policy"
    goal.priority = 0
    goal.obs_timestamp.sec = 1
    deadline_ns = time.time_ns() + 1_000_000_000
    goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
    goal_handle = _GoalHandle(goal)
    manager = SimpleNamespace(
        capabilities=lambda _pipeline_id: BackendCapabilities(),
        infer=lambda *_args: SimpleNamespace(
            action=[[0.0]],
            actual_chunk_size=1,
            backend_latency_ms=1.0,
            total_latency_ms=2.0,
        ),
    )
    node = object.__new__(PipelinePolicyNode)
    node._session_controller = _Controller()
    node._scheduled_operation_slots = threading.BoundedSemaphore(1)
    node._config = SimpleNamespace(
        pipeline_id="policy",
        request_timeout=1.0,
        max_prompt_bytes=4096,
        max_error_message_bytes=1024,
        max_error_details_bytes=8192,
        runtime_policy_fingerprint="runtime",
    )
    node._manifest = SimpleNamespace(fingerprint="deployment")
    node._require_manager = lambda: manager
    node._scheduled_deadline = lambda _deadline: datetime.now(timezone.utc) + timedelta(seconds=1)
    node._raise_if_deadline_expired = lambda *_args: None
    node._goal_cancel_requested = lambda _goal_handle: False
    node._sample_observations = lambda _sample_time: {"observation.state": [0.0]}
    node._rad_to_lerobot = lambda value: value
    node._to_policy_inputs = lambda observations: observations
    node._last_error = ""

    result = PipelinePolicyNode._scheduled_dispatch_once(node, goal_handle)

    assert goal_handle.aborted
    assert result.outcome.value == InferenceOutcome.COMPLETED
    assert result.error.recoverable is False


def test_monotonic_lease_expiry_is_converted_to_ros_clock_domain():
    assert (
        monotonic_expiry_to_ros_ns(
            50_000,
            monotonic_now_ns=20_000,
            ros_now_ns=1_000_000,
        )
        == 1_030_000
    )
    assert (
        monotonic_expiry_to_ros_ns(
            10_000,
            monotonic_now_ns=20_000,
            ros_now_ns=1_000_000,
        )
        == 1_000_000
    )


def test_wire_error_truncation_preserves_utf8_and_valid_details_json():
    error = ScheduledInferenceError()

    set_scheduled_error(
        error,
        code="x" * 100,
        message="测" * 20,
        details={"payload": "y" * 100},
        max_message_bytes=10,
        max_details_bytes=20,
    )

    assert utf8_size(error.code) == 64
    assert utf8_size(error.message) <= 10
    assert error.details_json == '{"truncated":true}'

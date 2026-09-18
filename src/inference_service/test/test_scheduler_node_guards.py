"""Node-boundary regressions for scheduler certainty, deadlines, and wire bounds."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from std_srvs.srv import Trigger

from ibrobot_msgs.action import (
    CloseInferenceSession,
    ClosePipelineBinding,
    DispatchPipelineBinding,
    OpenInferenceSession,
    OpenPipelineBinding,
    ScheduledDispatchInfer,
)
from ibrobot_msgs.msg import InferenceOutcome, InferenceServingStatus, InferenceWorkCapacity, ScheduledInferenceError
from inference_service import pipeline_policy_node as pipeline_policy_module
from inference_service.backends import BackendCapabilities
from inference_service.backends.errors import BackendInferenceError
from inference_service.global_inference_scheduler_node import (
    GlobalInferenceSchedulerNode,
    _DownstreamCall,
)
from inference_service.pipeline_policy_node import PipelinePolicyNode
from inference_service.scheduler.action_idempotency import replay_terminal
from inference_service.scheduler.deadline_reservations import DeadlineReservationTable
from inference_service.scheduler.global_scheduler_core import (
    GlobalSchedulerCore,
    GlobalSessionState,
    PipelineCandidate,
    SchedulerError,
)
from inference_service.scheduler.goal_slots import GoalSlotPool
from inference_service.scheduler.ledger import IdempotencyLedger, LedgerAction
from inference_service.scheduler.operations import Certainty, OperationIdentity, OperationKind, OperationRegistry
from inference_service.scheduler.time_domains import monotonic_expiry_to_ros_ns
from inference_service.scheduler.wire_bounds import set_scheduled_error, utf8_size
from inference_service.scheduler.work_classes import WorkClass, work_class_name
from robot_config.contract_utils import ActionSpec, Contract, ObservationSpec, iter_specs
from robot_config.inference_runtime_options import effective_latency_runtime_options

SESSION_ID = "00112233-4455-4677-8899-aabbccddeeff"
REQUEST_ID = "11112233-4455-4677-8899-aabbccddeeff"
BOOT_ID = "22222222-2222-4222-8222-222222222222"
NEW_BOOT_ID = "33333333-3333-4333-8333-333333333333"
BINDING_ID = "44444444-4444-4444-8444-444444444444"


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


def _close_protocol_node():
    from inference_service.scheduler.session_controller import ProductSessionController, WorkClassCapacity

    node = object.__new__(PipelinePolicyNode)
    node._boot_id = BOOT_ID
    node._config = SimpleNamespace(pipeline_id="policy", max_error_message_bytes=1024, max_error_details_bytes=8192)
    node._pipeline_ledger = IdempotencyLedger(
        max_session_records=4,
        max_duplicate_waiters_per_request=4,
        terminal_session_retention_ns=10**12,
        now_ns=time.monotonic_ns,
        max_entries=32,
    )
    ctrl = node._session_controller = ProductSessionController(
        boot_id=BOOT_ID,
        capacities={
            WorkClass.SESSION_CONTROL: WorkClassCapacity(WorkClass.SESSION_CONTROL, 1),
            WorkClass.ACTION_GENERATION: WorkClassCapacity(WorkClass.ACTION_GENERATION, 1),
        },
        session_idle_timeout_ns=10**12,
        now_ns=time.monotonic_ns,
    )
    opened = ctrl.begin_open(SESSION_ID)
    ctrl.finish_open(success=True)
    node._scheduled_binding_identity = (SESSION_ID, 1, BINDING_ID, 1, BOOT_ID, opened.fence_generation)
    node._scheduled_operation_slots = threading.BoundedSemaphore(1)
    node._scheduled_operation_capacity = 1
    node._acquire_scheduled_drain_slots = lambda _deadline: 1
    node._release_scheduled_drain_slots = lambda _count: None
    node._publish_serving_status = lambda: None
    node.get_logger = lambda: SimpleNamespace(exception=lambda message: pytest.fail(message))

    def goal(operation_id=None):
        value = ClosePipelineBinding.Goal()
        value.session_id, value.logical_generation = SESSION_ID, 1
        value.binding_id, value.binding_incarnation = BINDING_ID, 1
        value.expected_boot_id = BOOT_ID
        value.expected_pipeline_generation = opened.fence_generation
        value.operation_id = operation_id or str(uuid4())
        value.deadline.sec, value.deadline.nanosec = divmod(time.time_ns() + 10**10, 10**9)
        return value

    return node, goal


@pytest.mark.parametrize("first_outcome", [InferenceOutcome.COMPLETED, InferenceOutcome.UNKNOWN])
def test_private_close_failed_drain_new_attempt_and_replays(first_outcome):
    node, goal = _close_protocol_node()
    resets = []

    def reset(response, _deadline):
        resets.append(1)
        response.success = len(resets) > 1
        response.message = "drain result"
        return response

    node._reset_with_deadline = reset
    execute_close = node._scheduled_close_once

    def execute(goal_handle):
        result = execute_close(goal_handle)
        if not result.success:
            result.outcome.value = first_outcome
        return result

    node._scheduled_close_once = execute
    first_goal = goal()
    first = node._scheduled_close_callback(_GoalHandle(first_goal))
    assert not first.success and first.error.code == "close_drain_failed"
    assert node._scheduled_close_callback(_GoalHandle(first_goal)).error.code == "close_drain_failed"
    assert len(resets) == 1
    second = node._scheduled_close_callback(_GoalHandle(goal()))
    assert second.success
    assert second.drained_generation > second.closed_pipeline_generation > 0
    assert len(resets) == 2
    # A caller that lost the successful reply can confirm the drained identity.
    replay = node._scheduled_close_callback(_GoalHandle(goal()))
    assert replay.success and replay.drained_generation == second.drained_generation
    assert len(resets) == 2


def test_private_close_new_attempt_cannot_overlap_active_drain():
    node, goal = _close_protocol_node()
    entered, release = threading.Event(), threading.Event()
    results = []

    def reset(response, _deadline):
        entered.set()
        assert release.wait(5)
        response.success = True
        return response

    node._reset_with_deadline = reset
    worker = threading.Thread(target=lambda: results.append(node._scheduled_close_callback(_GoalHandle(goal()))))
    worker.start()
    try:
        assert entered.wait(5)
        retry = node._scheduled_close_callback(_GoalHandle(goal()))
        assert not retry.success and retry.outcome.value == InferenceOutcome.NOT_STARTED
        assert retry.error.code in {"close_in_progress", "no_session_capacity"}
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and results[0].success


@pytest.mark.parametrize(
    "expected,closed,drained,valid",
    [
        (3, 3, 4, True),
        (3, 3, 3, False),
        (3, 3, 0, False),
        (3, 2, 4, False),
        (3, 0, 0, False),
        (0, 0, 0, True),
        (0, 3, 4, True),
        (0, 0, 4, False),
    ],
)
def test_private_close_validates_successful_drain_generation(expected, closed, drained, valid):
    node, goal_factory = _close_protocol_node()
    goal = goal_factory()
    goal.expected_pipeline_generation = expected
    result = ClosePipelineBinding.Result()
    for field in ("session_id", "logical_generation", "binding_id", "binding_incarnation", "operation_id"):
        setattr(result, field, getattr(goal, field))
    result.boot_id, result.pipeline_id = BOOT_ID, "policy"
    result.success, result.outcome.value = True, InferenceOutcome.COMPLETED
    result.closed_pipeline_generation, result.drained_generation = closed, drained
    reason = GlobalInferenceSchedulerNode._validate_downstream_result(
        "close", goal, result, SimpleNamespace(pipeline_id="policy")
    )
    assert (reason == "") is valid


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
        "_downstream_operations": OperationRegistry(max_records=8, max_waiters_per_operation=2),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("tracing", [False, True])
def test_private_dispatch_trace_uses_existing_operation_and_preserves_callbacks(monkeypatch, tracing):
    import json
    import logging

    from ibrobot_tracing import TraceEmitter
    from inference_service import global_inference_scheduler_node as module

    records = []
    logger = logging.Logger("binding-trace", logging.INFO)
    handler = logging.Handler()
    handler.emit = lambda record: records.append(json.loads(record.getMessage().split(" ", 1)[1]))
    logger.addHandler(handler)
    monkeypatch.setattr(module, "trace", TraceEmitter(logger, enabled=tracing))
    goal = DispatchPipelineBinding.Goal()
    goal.session_id, goal.request_id = SESSION_ID, REQUEST_ID
    goal.logical_generation, goal.binding_incarnation = 1, 1
    goal.binding_id, goal.expected_boot_id, goal.operation_id = BINDING_ID, BOOT_ID, NEW_BOOT_ID
    goal.expected_pipeline_generation = 1
    result = DispatchPipelineBinding.Result()
    result.pipeline_id, result.success = "policy", True
    result.outcome.value = InferenceOutcome.COMPLETED
    counts = []

    class Future:
        def __init__(self, value, kind):
            self.value, self.kind = value, kind

        def result(self):
            return self.value

        def add_done_callback(self, callback):
            counts.append(self.kind)
            callback(self)

    def result_future():
        counts.append("get_result")
        return Future(SimpleNamespace(result=result), "result_callback")

    gh = SimpleNamespace(accepted=True, get_result_async=result_future)

    def send(sent_goal):
        assert sent_goal is goal and goal.operation_id == NEW_BOOT_ID
        counts.append("send")
        return Future(gh, "accepted_callback")

    node = _scheduler_stub(_operation_result=GlobalInferenceSchedulerNode._operation_result)
    call = GlobalInferenceSchedulerNode._call_downstream(
        node,
        SimpleNamespace(wait_for_server=lambda **kwargs: True, send_goal_async=send),
        goal,
        operation_kind=OperationKind.DISPATCH,
        deadline_monotonic_ns=time.monotonic_ns() + 10**9,
    )
    assert call.certainty == "completed" and call.result is result
    assert counts == ["send", "accepted_callback", "get_result", "result_callback"]
    assert len(node._downstream_operations) == 0
    if tracing:
        assert {(r["event"], r["fields"]["edge_id"]) for r in records} == {
            ("flow_send", "scheduler_to_pipeline_dispatch"),
            ("flow_receive", "pipeline_result_to_scheduler"),
        }
        assert all(r["fields"]["flow_id"] == NEW_BOOT_ID and r["fields"]["trace_id"] == REQUEST_ID for r in records)
    else:
        assert not records


def test_pipeline_binding_guard_rejects_wrong_boot_and_stale_binding():
    node = object.__new__(PipelinePolicyNode)
    node._boot_id = BOOT_ID
    node._scheduled_binding_identity = (SESSION_ID, 7, BINDING_ID, 2, BOOT_ID, 11)
    goal = DispatchPipelineBinding.Goal()
    goal.session_id = SESSION_ID
    goal.logical_generation = 7
    goal.binding_id = BINDING_ID
    goal.binding_incarnation = 2
    goal.expected_boot_id = NEW_BOOT_ID
    goal.expected_pipeline_generation = 11

    assert PipelinePolicyNode._validate_scheduled_binding_goal(node, goal) == "boot_mismatch"
    goal.expected_boot_id = BOOT_ID
    goal.binding_incarnation = 3
    assert PipelinePolicyNode._validate_scheduled_binding_goal(node, goal) == "binding_identity_mismatch"


def test_pipeline_binding_guard_accepts_exact_identity_and_generation_zero_cleanup():
    node = object.__new__(PipelinePolicyNode)
    node._boot_id = BOOT_ID
    node._scheduled_binding_identity = (SESSION_ID, 7, BINDING_ID, 2, BOOT_ID, 11)
    goal = DispatchPipelineBinding.Goal()
    goal.session_id = SESSION_ID
    goal.logical_generation = 7
    goal.binding_id = BINDING_ID
    goal.binding_incarnation = 2
    goal.expected_boot_id = BOOT_ID
    goal.expected_pipeline_generation = 11
    assert PipelinePolicyNode._validate_scheduled_binding_goal(node, goal) == ""
    goal.logical_generation = 0
    goal.expected_pipeline_generation = 0
    assert PipelinePolicyNode._validate_scheduled_binding_goal(node, goal) == ""


def test_global_validates_private_result_operation_and_boot_identity():
    goal = DispatchPipelineBinding.Goal()
    goal.session_id = SESSION_ID
    goal.logical_generation = 7
    goal.request_id = REQUEST_ID
    goal.binding_id = BINDING_ID
    goal.binding_incarnation = 2
    goal.operation_id = "55555555-5555-4555-8555-555555555555"
    goal.expected_boot_id = BOOT_ID
    goal.expected_pipeline_generation = 11

    result = DispatchPipelineBinding.Result()
    result.session_id = goal.session_id
    result.logical_generation = goal.logical_generation
    result.request_id = goal.request_id
    result.binding_id = goal.binding_id
    result.binding_incarnation = goal.binding_incarnation
    result.operation_id = goal.operation_id
    result.boot_id = goal.expected_boot_id
    result.pipeline_id = "policy"
    result.pipeline_generation = goal.expected_pipeline_generation
    result.deployment_fingerprint = "d" * 64
    result.runtime_policy_fingerprint = "r" * 64

    candidate = SimpleNamespace(
        pipeline_id="policy",
        deployment_fingerprint="d" * 64,
        runtime_policy_fingerprint="r" * 64,
    )
    assert not GlobalInferenceSchedulerNode._validate_downstream_result("dispatch", goal, result, candidate)

    result.operation_id = "66666666-6666-4666-8666-666666666666"
    assert (
        GlobalInferenceSchedulerNode._validate_downstream_result("dispatch", goal, result, candidate)
        == "dispatch_operation_id_mismatch"
    )
    result.operation_id = goal.operation_id
    result.boot_id = NEW_BOOT_ID
    assert (
        GlobalInferenceSchedulerNode._validate_downstream_result("dispatch", goal, result, candidate)
        == "dispatch_boot_id_mismatch"
    )


def test_pipeline_param_reader_preserves_legacy_parameters_without_new_staged_options(monkeypatch):
    class _Reader:
        def __init__(self, overrides=None):
            self.overrides = dict(overrides or {})
            self.declared = set()

        def declare_parameter(self, name, default):
            self.declared.add(name)

        def get_parameter(self, name):
            return SimpleNamespace(
                value=self.overrides.get(name, "")
                if name == "runtime_policy_json"
                else {
                    "node_name": "inference_policy",
                }.get(name)
            )

        def undeclare_parameter(self, name):
            self.declared.remove(name)

        def destroy_node(self):
            return None

    reader = _Reader()
    monkeypatch.setattr(pipeline_policy_module, "Node", lambda _name: reader)

    config, node_name = pipeline_policy_module._read_config()

    assert node_name == "inference_policy"
    assert config.scheduler_enabled is False
    assert "runtime_policy_json" in reader.declared
    assert reader.declared.issuperset(
        {
            "scheduled_open_session",
            "scheduled_dispatch",
            "scheduled_close_session",
            "scheduled_serving_status",
            "runtime_policy_fingerprint",
            "public_capacity_json",
        }
    )
    assert not reader.declared.intersection(
        {"pipeline_stage_policy", "pipeline_scheduling_json", "max_supported_public_priority"}
    )


@pytest.mark.parametrize("runtime_policy", ["", "{}"])
@pytest.mark.parametrize("external", [False, True])
def test_parameter_reader_declares_common_video_parameters_once(monkeypatch, runtime_policy, external):
    values = {}
    overrides = {"runtime_policy_json": runtime_policy, "external_video_producer": external}

    class Reader:
        def __init__(self, name):
            pass

        def declare_parameter(self, name, default):
            assert name not in values, f"duplicate parameter: {name}"
            values[name] = overrides.get(name, default)

        def get_parameter(self, name):
            return SimpleNamespace(value=values[name])

        def undeclare_parameter(self, name):
            del values[name]

        def destroy_node(self):
            pass

    monkeypatch.setattr(pipeline_policy_module, "Node", Reader)
    config, _ = pipeline_policy_module._read_config()
    assert config.external_video_producer is external
    assert config.scheduler_enabled is bool(runtime_policy)
    assert values["video_descriptor_topic"] == config.video_descriptor_topic
    assert values["video_status_topic"] == config.video_status_topic


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
        _scheduler_stub(),
        client,
        object(),
        operation_kind=OperationKind.DISPATCH,
        deadline_monotonic_ns=time.monotonic_ns() - 1,
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

    goal = OpenPipelineBinding.Goal()
    goal.session_id = SESSION_ID
    goal.logical_generation = 1
    goal.binding_id = REQUEST_ID
    goal.binding_incarnation = 1
    goal.operation_id = NEW_BOOT_ID
    goal.expected_boot_id = BOOT_ID
    call = GlobalInferenceSchedulerNode._call_downstream(
        node,
        client,
        goal,
        operation_kind=OperationKind.OPEN,
        deadline_monotonic_ns=time.monotonic_ns() + 1_000_000_000,
        late_acceptance_callback=late_acceptances.append,
    )

    assert call.certainty == "unknown"
    assert call.reason == "goal_acceptance_timeout"
    late_goal_handle = SimpleNamespace(accepted=True, get_result_async=lambda: _DeferredFuture())
    sent.complete(late_goal_handle)
    assert late_acceptances == [late_goal_handle]


def test_late_known_result_after_waiter_timeout_reclaims_operation():
    class _DeferredFuture:
        def __init__(self) -> None:
            self.callback = None
            self.value = None

        def add_done_callback(self, callback):
            self.callback = callback
            if self.value is not None:
                callback(self)

        def result(self):
            return self.value

        def complete(self, value) -> None:
            self.value = value
            if self.callback is not None:
                self.callback(self)

    result_future = _DeferredFuture()
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: result_future,
        cancel_goal_async=lambda: None,
    )
    sent = _DeferredFuture()
    sent.complete(goal_handle)
    node = _scheduler_stub(_goal_acceptance_timeout_ns=100_000_000)
    goal = OpenPipelineBinding.Goal()
    goal.session_id = SESSION_ID
    goal.logical_generation = 1
    goal.binding_id = REQUEST_ID
    goal.binding_incarnation = 1
    goal.operation_id = NEW_BOOT_ID
    goal.expected_boot_id = BOOT_ID

    call = GlobalInferenceSchedulerNode._call_downstream(
        node,
        SimpleNamespace(
            wait_for_server=lambda **_kwargs: True,
            send_goal_async=lambda _goal: sent,
        ),
        goal,
        operation_kind=OperationKind.OPEN,
        deadline_monotonic_ns=time.monotonic_ns() + 50_000_000,
    )
    assert call.certainty == "unknown"
    assert call.reason == "downstream_result_timeout"
    assert len(node._downstream_operations) == 1

    result = OpenPipelineBinding.Result()
    result.success = True
    result.outcome.value = InferenceOutcome.COMPLETED
    result_future.complete(SimpleNamespace(result=result))

    assert len(node._downstream_operations) == 0


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


@pytest.mark.parametrize("reply", ["success", "failed", "unknown", "wrong_boot", "wrong_binding", "no_drain"])
@pytest.mark.parametrize("path", ["public_close", "late_open_cleanup"])
def test_global_close_releases_unknown_reservation_only_after_validated_drain(reply, path):
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._default_request_timeout_ns = 2_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._global_policy = "edf"
    node._priority_zero_deadline_admission_enabled = False
    table = node._deadline_reservations = DeadlineReservationTable(policy="edf")
    owner = dict(session_id=SESSION_ID, binding_id=BINDING_ID, binding_incarnation=1, expected_boot_id=BOOT_ID)
    operations = node._downstream_operations = OperationRegistry(max_records=1, max_waiters_per_operation=1)
    operation, _ = operations.create_or_get(
        kind=OperationKind.DISPATCH,
        idempotency_key=("dispatch", "old"),
        identity=OperationIdentity(logical_generation=1, **owner),
        deadline_mono_ns=10,
        waiter_id="caller",
    )
    operation.claim_send()
    operations.finish(operation.operation_id, certainty=Certainty.UNKNOWN)
    operations.detach_waiter(operation.operation_id, "caller")
    reservation, _ = node._reserve_priority_zero(
        pipeline_id="policy",
        hardware_resource_id="ascend:0",
        requires_open=False,
        prompt_bytes=0,
        deadline_monotonic_ns=time.monotonic_ns() + 10**9,
        **owner,
    )
    assert reservation is not None
    table.mark_unknown(reservation)
    binding = SimpleNamespace(pipeline_id="policy", pipeline_generation=7, close_operation_id="close-op", **owner)
    closed = []
    node._core = SimpleNamespace(
        begin_close=lambda **_kwargs: None,
        session_record=lambda _session_id: SimpleNamespace(session_generation=1),
        wait_for_bindings_to_settle=lambda *_args: True,
        close_bindings=lambda _session_id: [binding],
        mark_session_failed=lambda *_args, **_kwargs: None,
        record_binding_close_success=lambda *args: closed.append(args),
        record_close_complete=lambda *_args, **_kwargs: 2,
    )
    node._candidate_by_id = {"policy": SimpleNamespace(pipeline_id="policy")}
    node._pipeline_clients = {"policy": {"close": object()}}

    def close(_client, goal, **_kwargs):
        if reply == "unknown":
            return _DownstreamCall("unknown")
        result = ClosePipelineBinding.Result()
        for field in ("session_id", "logical_generation", "binding_id", "binding_incarnation", "operation_id"):
            setattr(result, field, getattr(goal, field))
        result.boot_id = NEW_BOOT_ID if reply == "wrong_boot" else BOOT_ID
        if reply == "wrong_binding":
            result.binding_id = str(uuid4())
        result.pipeline_id = "policy"
        result.success = reply != "failed"
        result.outcome.value = InferenceOutcome.COMPLETED
        result.closed_pipeline_generation = 7
        result.drained_generation = 7 if reply == "no_drain" else 8
        return _DownstreamCall("completed", result=result)

    node._call_downstream = close
    goal = CloseInferenceSession.Goal()
    goal.session_id, goal.session_generation = SESSION_ID, 1
    if path == "public_close":
        result = node._close_once(_GoalHandle(goal), None)
        assert result.success is (reply == "success")
    else:
        from concurrent.futures import Future

        def done(value):
            future = Future()
            future.set_result(value)
            return future

        def send_close(close_goal):
            call = close(None, close_goal)
            return done(
                SimpleNamespace(accepted=True, get_result_async=lambda: done(SimpleNamespace(result=call.result)))
            )

        node._pipeline_clients["policy"]["close"] = SimpleNamespace(send_goal_async=send_close)
        late_open = SimpleNamespace(cancel_goal_async=lambda: None, get_result_async=lambda: done(None))
        node._cleanup_late_open("policy", SimpleNamespace(**owner), late_open)
    next_reservation = table.try_reserve(
        pipeline_id="backup",
        hardware_resource_id="ascend:0",
        now_ns=0,
        deadline_ns=10,
    )
    assert bool(closed) is (reply == "success")
    assert (next_reservation is not None) is (reply == "success")
    assert len(operations) == (0 if reply == "success" else 1)


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


@pytest.mark.parametrize("recoverable", [True, False])
def test_global_close_explicit_not_started_keeps_close_irreversible(recoverable):
    failures: list[str] = []
    completed: list[tuple[object, ...]] = []
    pending: list[str] = []
    downstream = ClosePipelineBinding.Result()
    downstream.session_id = SESSION_ID
    downstream.logical_generation = 1
    downstream.binding_id = ""
    downstream.binding_incarnation = 1
    downstream.operation_id = "close-op"
    downstream.boot_id = ""
    downstream.pipeline_id = "policy"
    downstream.success = False
    downstream.outcome.value = InferenceOutcome.NOT_STARTED
    downstream.error.code = "no_session_capacity"
    downstream.error.recoverable = recoverable

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._default_request_timeout_ns = 2_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._core = SimpleNamespace(
        begin_close=lambda **_kwargs: None,
        session_record=lambda _session_id: SimpleNamespace(session_generation=1),
        wait_for_bindings_to_settle=lambda *_args: True,
        close_bindings=lambda _session_id: [
            SimpleNamespace(
                pipeline_id="policy",
                pipeline_generation=7,
                binding_id="",
                binding_incarnation=1,
                close_operation_id="close-op",
                expected_boot_id="",
            )
        ],
        record_close_complete=lambda *args, **kwargs: completed.append((args, kwargs)) or 1,
        record_close_not_started=lambda session_id: pending.append(session_id),
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
    assert result.error.recoverable is recoverable
    assert not completed
    assert not failures
    assert pending == [SESSION_ID]


@pytest.mark.parametrize("other_outcome", ["success", "unknown", "failed"])
@pytest.mark.parametrize("not_started_reason", ["downstream_unavailable", "goal_rejected"])
def test_global_close_partial_drain_and_not_started_retry(other_outcome, not_started_reason):
    node = _idempotent_node()
    node._downstream_operations = OperationRegistry(max_records=8, max_waiters_per_operation=2)
    candidates = [
        PipelineCandidate(pid, "g", "ascend:0", "h", "d", "r", "/open", "/dispatch", "/close", "/status", "")
        for pid in ("a", "b")
    ]
    core = node._core = GlobalSchedulerCore(
        candidates=candidates,
        max_session_records=4,
        max_product_requests_per_session=4,
        terminal_session_retention_ns=10**9,
        session_idle_timeout_ns=30 * 10**9,
        max_fallback_pipelines=4,
        now_ns=time.monotonic_ns,
    )
    core.open_session(session_id=SESSION_ID)
    for candidate in candidates:
        core.prepare_dispatch_candidate(session_id=SESSION_ID, session_generation=1, pipeline_id=candidate.pipeline_id)
        core.record_binding_open_success(
            session_id=SESSION_ID,
            pipeline_id=candidate.pipeline_id,
            pipeline_generation=7,
            hardware_resource_id="ascend:0",
        )
    node._candidate_by_id = {candidate.pipeline_id: candidate for candidate in candidates}
    node._pipeline_clients = {candidate.pipeline_id: {"close": candidate.pipeline_id} for candidate in candidates}
    node._deadline_reservations = DeadlineReservationTable(policy="fifo")
    calls = []
    retry = False

    def close(pid, goal, **_kwargs):
        calls.append((pid, goal.operation_id))
        if pid == "b" and not retry:
            return _DownstreamCall("not_started", reason=not_started_reason)
        if other_outcome == "unknown":
            return _DownstreamCall("unknown", reason="result_timeout")
        result = ClosePipelineBinding.Result()
        for field in ("session_id", "logical_generation", "binding_id", "binding_incarnation", "operation_id"):
            setattr(result, field, getattr(goal, field))
        result.boot_id = goal.expected_boot_id
        result.pipeline_id = pid
        result.success = other_outcome == "success"
        result.outcome.value = InferenceOutcome.COMPLETED
        result.closed_pipeline_generation = 7
        result.drained_generation = 8 if result.success else 0
        result.error.code = "" if result.success else "close_failed"
        return _DownstreamCall("completed", result=result)

    node._call_downstream = close
    goal = CloseInferenceSession.Goal()
    goal.session_id, goal.session_generation = SESSION_ID, 1
    first = node._close_endpoint(_GoalHandle(goal))
    record = core.session_record(SESSION_ID)
    assert [pid for pid, _ in calls] == ["a", "b"]
    assert not record.bindings["b"].quarantine
    with pytest.raises(SchedulerError, match="session_not_active" if other_outcome == "success" else "session_failed"):
        core.prepare_dispatch_candidate(session_id=SESSION_ID, session_generation=1, pipeline_id="b")
    if other_outcome != "success":
        assert first.outcome.value == (
            InferenceOutcome.UNKNOWN if other_outcome == "unknown" else InferenceOutcome.COMPLETED
        )
        assert record.quarantine and record.bindings["a"].quarantine
        return
    assert first.outcome.value == InferenceOutcome.NOT_STARTED and first.error.recoverable
    assert record.state == GlobalSessionState.CLOSING and not record.quarantine
    assert set(record.bindings) == {"b"}
    retry = True
    second = node._close_endpoint(_GoalHandle(goal))
    assert second.success and second.drained_generation == 2
    assert [pid for pid, _ in calls] == ["a", "b", "b"]
    assert calls[1][1] != calls[2][1]
    assert core.session_state(SESSION_ID) == GlobalSessionState.CLOSED


@pytest.mark.parametrize("ros_now_ns", [10**9, 1_800_000_000 * 10**9])
def test_observation_age_uses_ros_clock_then_monotonic_anchor(monkeypatch, ros_now_ns):
    node = object.__new__(PipelinePolicyNode)
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=ros_now_ns))
    monkeypatch.setattr(time, "monotonic_ns", lambda: 20 * 10**9)
    metadata = node._observation_time_metadata(ros_now_ns - 200_000_000)
    assert metadata["observation_timestamp_ns"] == ros_now_ns - 200_000_000
    assert metadata["observation_monotonic_ns"] == 19_800_000_000


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
    node._priority_zero_deadline_admission_safety_margin_ms = 5
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
    assert estimate_ms == 48.0
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
                binding_id=BINDING_ID,
                binding_incarnation=1,
                expected_boot_id=BOOT_ID,
            )

        @staticmethod
        def record_request_terminal(session_id, *, request_id=None, session_generation=None):
            terminal_sessions.append(session_id)

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = _Core()
    node._candidate_by_id = {
        pipeline_id: SimpleNamespace(pipeline_id=pipeline_id, hardware_resource_id="ascend:0")
        for pipeline_id in candidate_ids
    }
    node._default_request_timeout_ns = 2_000_000_000
    node._status_reason = lambda _candidate, **_kwargs: ""
    node._compatibility_reason = lambda _target, _candidate: ""
    node._capacity_accepting = lambda _pipeline_id, _work_class: True
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._release_reservation = lambda _reservation: None
    node._mark_reservation_unknown = lambda _reservation, **_kwargs: None
    node._deadline_reservations = SimpleNamespace(wait_for_turn=lambda *_args, **_kwargs: "ready")
    node._priority_zero_deadline_admission_enabled = False
    return node, terminal_sessions


def test_dispatch_waits_for_an_opening_binding_before_sending_generation_zero():
    terminal_sessions: list[str] = []
    wait_calls: list[str] = []
    dispatch_generations: list[int] = []
    binding_active = False

    class _Core:
        @staticmethod
        def resolve_dispatch_plan(**_kwargs):
            return SimpleNamespace(candidate_ids=("policy",))

        @staticmethod
        def prepare_dispatch_candidate(*, pipeline_id, **_kwargs):
            return SimpleNamespace(
                pipeline_id=pipeline_id,
                reason="" if binding_active else "binding_open_in_progress",
                pipeline_generation=7 if binding_active else 0,
                needs_open=False,
            )

        @staticmethod
        def wait_for_binding_open(**_kwargs):
            nonlocal binding_active
            wait_calls.append("policy")
            binding_active = True
            return True

        @staticmethod
        def record_request_terminal(session_id, **_kwargs):
            terminal_sessions.append(session_id)

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = _Core()
    node._candidate_by_id = {"policy": SimpleNamespace(pipeline_id="policy", hardware_resource_id="ascend:0")}
    node._default_request_timeout_ns = 2_000_000_000
    node._status_reason = lambda _candidate, **_kwargs: ""
    node._compatibility_reason = lambda _target, _candidate: ""
    node._capacity_accepting = lambda _pipeline_id, _work_class: True
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._serving_status = {}
    node._priority_zero_deadline_admission_enabled = False

    def dispatch(_goal_handle, goal, _candidate, *, pipeline_generation, **_kwargs):
        dispatch_generations.append(pipeline_generation)
        result = ScheduledDispatchInfer.Result()
        result.request_id = goal.request_id
        result.session_id = goal.session_id
        result.pipeline_id = "policy"
        result.session_generation = goal.session_generation
        result.success = True
        result.outcome.value = InferenceOutcome.COMPLETED
        return _DownstreamCall("completed", result=result), goal

    node._dispatch_bound_pipeline = dispatch
    result = GlobalInferenceSchedulerNode._dispatch_once(
        node,
        _GoalHandle(_dispatch_goal(priority=1, deadline_ns=0, fallback_chain=[])),
        None,
    )

    assert result.success
    assert wait_calls == ["policy"]
    assert dispatch_generations == [7]
    assert terminal_sessions == [SESSION_ID]


def test_visual_unknown_quarantines_pipeline_session_controller():
    quarantined: list[bool] = []
    published: list[bool] = []

    class _FailedFuture:
        @staticmethod
        def result():
            raise BackendInferenceError(
                "async completion is uncertain",
                code="async_execution_uncertain",
                operation_started=True,
                outcome_known=False,
            )

    node = object.__new__(PipelinePolicyNode)
    node._frame_trigger_lock = threading.Lock()
    node._frame_trigger_future = _FailedFuture()
    node._visual_trigger_epoch = 0
    node._last_error = ""
    node._session_controller = SimpleNamespace(mark_failed_quarantine=lambda: quarantined.append(True))
    node.get_logger = lambda: SimpleNamespace(error=lambda *_args, **_kwargs: None)
    node._publish_serving_status = lambda: published.append(True)

    PipelinePolicyNode._visual_frame_completed(node, node._frame_trigger_future, 0)

    assert quarantined == [True]
    assert published == [True]


@pytest.mark.parametrize("invalidate_during_result", [False, True])
def test_old_visual_failure_cannot_quarantine_new_epoch(invalidate_during_result):
    node = object.__new__(PipelinePolicyNode)
    node._frame_trigger_lock = threading.RLock()
    node._visual_trigger_epoch = 0
    node._last_error = ""
    quarantined = []
    node._session_controller = SimpleNamespace(mark_failed_quarantine=lambda: quarantined.append(True))

    class FailedFuture:
        def result(self):
            if invalidate_during_result:
                node._invalidate_visual_trigger()
            raise BackendInferenceError("uncertain", operation_started=True, outcome_known=False)

    future = FailedFuture()
    node._frame_trigger_future = future
    if not invalidate_during_result:
        node._invalidate_visual_trigger()
    node._visual_frame_completed(future, 0)
    assert quarantined == []
    assert node._last_error == ""


def test_visual_sampling_is_reserved_and_fenced_before_submission():
    from concurrent.futures import ThreadPoolExecutor

    from inference_service.scheduler.session_controller import ServingState

    entered, release = threading.Event(), threading.Event()
    sampled, submitted = [], []
    node = object.__new__(PipelinePolicyNode)
    node._config = SimpleNamespace(
        scheduler_enabled=True, pipeline_stage_policy="independent", pipeline_id="policy", default_task="task"
    )
    node._frame_trigger_lock = threading.RLock()
    node._frame_trigger_future = None
    node._pending_visual_sample_time_ns = None
    node._visual_trigger_epoch = 0
    node._visual_trigger_enabled = True
    node._frame_trigger_priority = 1
    node._scheduled_binding_identity = (SESSION_ID, 1, BINDING_ID, 1, BOOT_ID, 1)
    node._session_controller = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            state=ServingState.ACTIVE, product_session_id=SESSION_ID, product_session_generation=1
        )
    )
    node._manager = SimpleNamespace(submit_frame=lambda *args: submitted.append(args))
    node._to_policy_inputs = lambda value: value
    node._current_task = "task"
    node._last_error = ""
    node.get_logger = lambda: SimpleNamespace(debug=lambda *args, **kwargs: None)

    def sample(timestamp):
        sampled.append(timestamp)
        entered.set()
        assert release.wait(2)
        return {}

    node._sample_observations = sample
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(node._request_visual_frame, 10)
        try:
            assert entered.wait(1)
            node._request_visual_frame(20)
            assert sampled == [10]
            node._invalidate_visual_trigger()
            node._visual_trigger_enabled = True
            release.set()
            first.result(timeout=2)
            assert submitted == []
            assert node._frame_trigger_future is None
            assert node._last_error == ""
        finally:
            release.set()


def test_priority_zero_checks_each_candidate_and_dispatches_first_feasible_fallback():
    node, terminal_sessions = _routing_node(("policy", "backup"))
    node._priority_zero_deadline_admission_enabled = True
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
    node._priority_zero_deadline_admission_enabled = True
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


def test_edf_priority_zero_uses_unprofiled_reservation_and_fallback() -> None:
    node, terminal_sessions = _routing_node(("policy", "backup"))
    node._global_policy = "edf"
    node._priority_zero_deadline_admission_enabled = False
    node._serving_status = {}
    reserved: list[tuple[str, int | None]] = []

    class _Reservations:
        @staticmethod
        def try_reserve(*, pipeline_id, estimate_ns=None, **_kwargs):
            reserved.append((pipeline_id, estimate_ns))
            return None if pipeline_id == "policy" else SimpleNamespace()

        @staticmethod
        def wait_for_turn(*_args, **_kwargs):
            return "ready"

    node._deadline_reservations = _Reservations()
    node._priority_zero_estimate_ms = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("EDF must not require a profile estimate")
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
            priority=0,
            deadline_ns=time.time_ns() + 1_000_000_000,
            fallback_chain=["backup"],
        )
    )

    result = GlobalInferenceSchedulerNode._dispatch_once(node, goal_handle, None)

    assert result.success
    assert reserved == [("policy", None), ("backup", None)]
    assert dispatched == ["backup"]
    assert terminal_sessions == [SESSION_ID]


def test_edf_global_dispatch_orders_concurrent_requests_by_absolute_deadline() -> None:
    node, terminal_sessions = _routing_node(("policy",))
    node._global_policy = "edf"
    node._priority_zero_deadline_admission_enabled = False
    node._serving_status = {}
    both_waiting = threading.Barrier(2)

    class _SynchronizedReservations(DeadlineReservationTable):
        def wait_for_turn(self, reservation, **kwargs):
            both_waiting.wait(timeout=1.0)
            return super().wait_for_turn(reservation, **kwargs)

    reservations = _SynchronizedReservations(policy="edf")
    node._deadline_reservations = reservations
    node._release_reservation = reservations.release
    node._mark_reservation_unknown = lambda reservation, **_kwargs: reservations.mark_unknown(reservation)
    dispatch_order: list[str] = []

    def dispatch(_goal_handle, goal, candidate, **_kwargs):
        dispatch_order.append(goal.request_id)
        result = ScheduledDispatchInfer.Result()
        result.success = True
        result.session_generation = goal.session_generation
        result.request_id = goal.request_id
        result.pipeline_id = candidate.pipeline_id
        result.outcome.value = InferenceOutcome.COMPLETED
        return _DownstreamCall("completed", result=result), goal

    node._dispatch_bound_pipeline = dispatch
    later = _dispatch_goal(priority=0, deadline_ns=time.time_ns() + 1_500_000_000, fallback_chain=[])
    earlier = _dispatch_goal(priority=0, deadline_ns=time.time_ns() + 1_000_000_000, fallback_chain=[])
    earlier.request_id = NEW_BOOT_ID
    results: list[ScheduledDispatchInfer.Result] = []

    def run(goal):
        results.append(GlobalInferenceSchedulerNode._dispatch_once(node, _GoalHandle(goal), None))

    later_thread = threading.Thread(target=run, args=(later,))
    earlier_thread = threading.Thread(target=run, args=(earlier,))
    later_thread.start()
    earlier_thread.start()
    later_thread.join(timeout=2.0)
    earlier_thread.join(timeout=2.0)

    assert not later_thread.is_alive() and not earlier_thread.is_alive()
    assert all(result.success for result in results)
    assert dispatch_order == [NEW_BOOT_ID, REQUEST_ID]
    assert terminal_sessions == [SESSION_ID, SESSION_ID]


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
        def record_request_terminal(session_id, *, request_id=None, session_generation=None):
            terminal_sessions.append(session_id)

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = _Core()
    node._candidate_by_id = {"policy": SimpleNamespace(pipeline_id="policy", hardware_resource_id="ascend:0")}
    node._default_request_timeout_ns = 2_000_000_000
    node._status_reason = lambda _candidate, **_kwargs: ""
    node._compatibility_reason = lambda _target, _candidate: ""
    node._capacity_accepting = lambda _pipeline_id, _work_class: True
    node._reserve_priority_zero = lambda **_kwargs: (SimpleNamespace(), "")
    node._release_reservation = lambda _reservation: None
    node._mark_reservation_unknown = lambda _reservation, **_kwargs: None
    node._deadline_reservations = SimpleNamespace(wait_for_turn=lambda *_args, **_kwargs: "ready")
    node._priority_zero_deadline_admission_enabled = False
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


@pytest.mark.parametrize("open_certainty", ["completed", "not_started"])
def test_close_racing_real_lazy_open_settles_binding_and_releases_reservation(open_certainty):
    candidate = PipelineCandidate(
        pipeline_id="policy",
        compatibility_group="g",
        hardware_resource_id="ascend:0",
        hardware_profile_fingerprint="a" * 64,
        deployment_fingerprint="d" * 64,
        runtime_policy_fingerprint="r" * 64,
        endpoint_open="/open",
        endpoint_dispatch="/dispatch",
        endpoint_close="/close",
        endpoint_serving_status="/status",
        profile_path="",
    )
    core = GlobalSchedulerCore(
        candidates=[candidate],
        max_session_records=4,
        max_product_requests_per_session=4,
        terminal_session_retention_ns=1_000_000_000,
        session_idle_timeout_ns=30_000_000_000,
        max_fallback_pipelines=4,
        now_ns=time.monotonic_ns,
    )
    core.open_session(session_id=SESSION_ID)
    node, _ = _routing_node(("policy",))
    node._core = core
    node._candidate_by_id = {"policy": candidate}
    node._global_policy = "edf"
    node._serving_status = {
        "policy": SimpleNamespace(
            message=SimpleNamespace(supports_priority_zero_deadline_admission=True, boot_id=BOOT_ID)
        )
    }
    node._status_lock = threading.RLock()
    node._trusted_status_cursors = {"policy": (BOOT_ID, 1)}
    node._pipeline_clients = {"policy": {"open": object()}}
    table = node._deadline_reservations = DeadlineReservationTable(policy="edf")
    node._release_reservation = table.release
    del node._mark_reservation_unknown
    sent = []

    def reserve(**kwargs):
        # Deterministically linearize Close after prepare, before the real Open helper.
        core.begin_close(session_id=SESSION_ID, session_generation=1)
        return table.try_reserve(
            pipeline_id="policy",
            hardware_resource_id="ascend:0",
            now_ns=time.monotonic_ns(),
            deadline_ns=kwargs["deadline_monotonic_ns"],
        ), ""

    node._reserve_priority_zero = reserve

    def send(_client, goal, **_kwargs):
        sent.append(goal)
        assert core.session_state(SESSION_ID) == GlobalSessionState.CLOSING
        if open_certainty == "not_started":
            return _DownstreamCall("not_started", reason="downstream_rejected")
        result = OpenPipelineBinding.Result()
        for field in ("session_id", "logical_generation", "binding_id", "binding_incarnation", "operation_id"):
            setattr(result, field, getattr(goal, field))
        result.boot_id = goal.expected_boot_id
        result.pipeline_id = candidate.pipeline_id
        result.pipeline_generation = 2
        result.deployment_fingerprint = candidate.deployment_fingerprint
        result.runtime_policy_fingerprint = candidate.runtime_policy_fingerprint
        result.success = True
        result.outcome.value = InferenceOutcome.COMPLETED
        return _DownstreamCall("completed", result=result)

    node._call_downstream = send
    result = node._dispatch_once(
        _GoalHandle(_dispatch_goal(priority=0, deadline_ns=time.time_ns() + 1_000_000_000, fallback_chain=[])), None
    )

    assert len(sent) == 1
    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert core.wait_for_bindings_to_settle(SESSION_ID, time.monotonic_ns() + 100_000_000)
    assert core.session_state(SESSION_ID) == GlobalSessionState.CLOSING
    assert (
        table.try_reserve(
            pipeline_id="policy",
            hardware_resource_id="ascend:0",
            now_ns=time.monotonic_ns(),
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
        )
        is not None
    )
    if open_certainty == "not_started":
        assert core.session_record(SESSION_ID).bindings == {}
    else:
        assert core.session_record(SESSION_ID).bindings["policy"].pipeline_generation == 2


@pytest.mark.parametrize("fence_order", ["before_unknown", "after_unknown", "no_fence"])
def test_late_unknown_reservation_respects_trusted_boot_fence(fence_order):
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._status_lock = threading.RLock()
    node._trusted_status_cursors = {"policy": (BOOT_ID, 1)}
    table = node._deadline_reservations = DeadlineReservationTable()
    now = time.monotonic_ns()
    reservation = table.try_reserve(
        pipeline_id="policy", hardware_resource_id="ascend:0", now_ns=now, deadline_ns=now + 10**9
    )

    def fence():
        with node._status_lock:
            table.reconcile_pipeline("policy")
            node._trusted_status_cursors["policy"] = (NEW_BOOT_ID, 1)

    if fence_order == "before_unknown":
        fence()
    node._mark_reservation_unknown(reservation, expected_boot_id=BOOT_ID)
    if fence_order == "after_unknown":
        fence()
    replacement = table.try_reserve(
        pipeline_id="policy", hardware_resource_id="ascend:0", now_ns=now, deadline_ns=now + 10**9
    )
    if fence_order == "no_fence":
        assert replacement is None
    else:
        assert replacement is not None
        assert table.wait_for_turn(replacement, deadline_ns=now + 10**9) == "ready"


def test_unsupported_profile_admission_releases_unsent_binding():
    candidate = PipelineCandidate(
        pipeline_id="policy",
        compatibility_group="g",
        hardware_resource_id="ascend:0",
        hardware_profile_fingerprint="a" * 64,
        deployment_fingerprint="d" * 64,
        runtime_policy_fingerprint="r" * 64,
        endpoint_open="/open",
        endpoint_dispatch="/dispatch",
        endpoint_close="/close",
        endpoint_serving_status="/status",
        profile_path="",
    )
    core = GlobalSchedulerCore(
        candidates=[candidate],
        max_session_records=4,
        max_product_requests_per_session=4,
        terminal_session_retention_ns=1_000_000_000,
        session_idle_timeout_ns=30_000_000_000,
        max_fallback_pipelines=4,
        now_ns=time.monotonic_ns,
    )
    core.open_session(session_id=SESSION_ID)
    node, _ = _routing_node(("policy",))
    node._core = core
    node._candidate_by_id = {"policy": candidate}
    node._global_policy = "fifo"
    node._priority_zero_deadline_admission_enabled = True
    node._serving_status = {
        "policy": SimpleNamespace(
            message=SimpleNamespace(supports_priority_zero_deadline_admission=False, boot_id=BOOT_ID)
        )
    }
    result = node._dispatch_once(
        _GoalHandle(_dispatch_goal(priority=0, deadline_ns=time.time_ns() + 1_000_000_000, fallback_chain=[])), None
    )
    assert not result.success
    assert result.error.code == "priority_zero_deadline_admission_not_supported"
    assert core.session_record(SESSION_ID).bindings == {}
    next_binding = core.prepare_dispatch_candidate(session_id=SESSION_ID, session_generation=1, pipeline_id="policy")
    assert next_binding.needs_open


def test_edf_priority_zero_skips_independent_stage_candidates():
    """EDF is deadline-driven ordering: independent stages cannot honor it."""

    node, terminal_sessions = _routing_node(("policy", "backup"))
    node._global_policy = "edf"
    node._priority_zero_deadline_admission_enabled = False
    node._serving_status = {
        "policy": SimpleNamespace(
            message=SimpleNamespace(supports_priority_zero_deadline_admission=False, boot_id=BOOT_ID)
        ),
        "backup": SimpleNamespace(
            message=SimpleNamespace(supports_priority_zero_deadline_admission=True, boot_id=BOOT_ID)
        ),
    }
    reserved: list[str] = []

    def reserve(*, pipeline_id, **_kwargs):
        reserved.append(pipeline_id)
        return SimpleNamespace(), ""

    node._reserve_priority_zero = reserve
    dispatched: list[str] = []

    def dispatch(_goal_handle, goal, candidate, **_kwargs):
        dispatched.append(candidate.pipeline_id)
        result = ScheduledDispatchInfer.Result()
        result.success = True
        result.session_generation = 7
        result.outcome.value = InferenceOutcome.COMPLETED
        return _DownstreamCall("completed", result=result), goal

    node._dispatch_bound_pipeline = dispatch
    result = GlobalInferenceSchedulerNode._dispatch_once(
        node,
        _GoalHandle(_dispatch_goal(priority=0, deadline_ns=time.time_ns() + 1_000_000_000, fallback_chain=["backup"])),
        None,
    )

    assert result.success
    assert reserved == ["backup"]
    assert dispatched == ["backup"]
    assert terminal_sessions == [SESSION_ID]


def test_edf_priority_zero_reports_no_feasible_candidate_when_all_independent():
    node, _terminal_sessions = _routing_node(("policy", "backup"))
    node._global_policy = "edf"
    node._priority_zero_deadline_admission_enabled = False
    node._serving_status = {
        pipeline_id: SimpleNamespace(
            message=SimpleNamespace(supports_priority_zero_deadline_admission=False, boot_id=BOOT_ID)
        )
        for pipeline_id in ("policy", "backup")
    }
    node._dispatch_bound_pipeline = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("deadline-driven priority-0 must not dispatch to independent stages")
    )
    result = GlobalInferenceSchedulerNode._dispatch_once(
        node,
        _GoalHandle(_dispatch_goal(priority=0, deadline_ns=time.time_ns() + 1_000_000_000, fallback_chain=["backup"])),
        None,
    )

    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "priority_zero_deadline_admission_not_supported"


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


def test_dispatch_rejects_priority_missing_from_public_status_mask():
    node, terminal_sessions = _routing_node(("policy",))
    checked: list[int | None] = []

    def status_reason(_candidate, **kwargs):
        checked.append(kwargs.get("required_priority"))
        return "unsupported_public_priority"

    node._status_reason = status_reason
    node._dispatch_bound_pipeline = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("priority rejected by Global must not reach downstream")
    )
    result = GlobalInferenceSchedulerNode._dispatch_once(
        node,
        _GoalHandle(
            _dispatch_goal(
                priority=3,
                deadline_ns=time.time_ns() + 1_000_000_000,
                fallback_chain=[],
            )
        ),
        None,
    )

    assert result.outcome.value == InferenceOutcome.NOT_STARTED
    assert result.error.code == "unsupported_public_priority"
    assert checked == [3]
    assert terminal_sessions == [SESSION_ID]


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
    monkeypatch.setattr(
        pipeline_policy_module,
        "PipelinePolicyNode",
        lambda _config, node_name, registry_set, providers: _Node(),
    )
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
        pipeline_stage_policy="sequential",
        pipeline_scheduling_json='{"stage_policy":"sequential","frame_base_priority":0,"stages":{}}',
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
        "scheduling": {"stage_policy": "sequential", "frame_base_priority": 0, "stages": {}},
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


@pytest.mark.parametrize(
    "override",
    [
        {"pipeline_stage_policy": "independent"},
        {"pipeline_scheduling_json": '{"stage_policy":"independent"}'},
        {"pipeline_scheduling_json": '{"frame_base_priority":3}'},
        {"pipeline_scheduling_json": '{"stages":{"encoder":{"priority_offset":2,"instance_count":1}}}'},
    ],
)
def test_scheduled_runtime_policy_rejects_scheduling_parameter_overrides(override):
    node = _runtime_policy_node("{}")
    for key, value in override.items():
        setattr(node._config, key, value)
    with pytest.raises(RuntimeError, match="scheduling mismatch"):
        node._validate_runtime_policy(_runtime_policy({}))


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
    node._core = SimpleNamespace(reconcile_pipeline_boot=lambda pipeline_id, boot_id: reconciled.append(pipeline_id))
    node._deadline_reservations = SimpleNamespace(reconcile_pipeline=deadline_reconciled.append)
    node._downstream_operations = SimpleNamespace(fence_boot=lambda _boot_id: None)
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
    node._core = SimpleNamespace(reconcile_pipeline_boot=lambda pipeline_id, boot_id: reconciled.append(pipeline_id))
    node._deadline_reservations = SimpleNamespace(reconcile_pipeline=deadline_reconciled.append)
    node._downstream_operations = SimpleNamespace(fence_boot=lambda _boot_id: None)
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


@pytest.mark.parametrize("schema_version", [0, 1, 2])
def test_only_supported_schema_new_boot_reconciles_quarantine_once(schema_version):
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
    node._core = SimpleNamespace(reconcile_pipeline_boot=lambda pipeline_id, boot_id: reconciled.append(pipeline_id))
    node._deadline_reservations = SimpleNamespace(reconcile_pipeline=deadline_reconciled.append)
    node._downstream_operations = SimpleNamespace(fence_boot=lambda _boot_id: None)
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

    status.scheduling_capability_schema_version = schema_version
    status.stage_policy = "SEQUENTIAL"

    callback = node._make_status_callback("policy")
    callback(status)
    status.sequence = 2
    callback(status)

    if schema_version == 1:
        assert reconciled == ["policy"]
        assert deadline_reconciled == ["policy"]
        assert node._trusted_status_cursors["policy"] == (NEW_BOOT_ID, 2)
    else:
        assert reconciled == []
        assert deadline_reconciled == []
        assert node._trusted_status_cursors["policy"] == (BOOT_ID, 7)
        assert node._serving_status["policy"].invalid_reason == "scheduling_capability_schema_mismatch"


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
    node._status_reason = lambda _candidate, **_kwargs: ""

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
        session_record=lambda _session_id: SimpleNamespace(
            session_generation=3,
            state=GlobalSessionState.ACTIVE,
            in_flight_requests=0,
        ),
        begin_close=lambda **_kwargs: None,
        wait_for_bindings_to_settle=lambda *_args: True,
        close_bindings=lambda _session_id: [SimpleNamespace(pipeline_id="policy", pipeline_generation=3)],
        record_binding_close_success=lambda *args: close_successes.append(args),
        record_close_complete=lambda *args, **_kwargs: 4,
        mark_session_failed=lambda session_id, **_kwargs: failures.append(session_id),
        aged_quarantined_sessions=lambda _age_ns: [],
    )
    node._close_retry_exhausted = set()
    node._session_idle_timeout_ns = 30_000_000_000
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
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_args: None, error=lambda *_args: None)

    node._idle_sweep()

    assert failures == [session_id]
    assert close_successes == []


def test_pipeline_rejects_expired_open_before_session_admission():
    class _Controller:
        def begin_open(self, _session_id):
            raise AssertionError("expired Open must not reach session admission")

    goal = OpenPipelineBinding.Goal()
    goal.session_id = "00112233-4455-4677-8899-aabbccddeeff"
    goal.logical_generation = 1
    goal.binding_id = BINDING_ID
    goal.binding_incarnation = 1
    goal.operation_id = REQUEST_ID
    goal.expected_boot_id = BOOT_ID
    expired_ns = time.time_ns() - 1_000_000
    goal.deadline.sec = expired_ns // 1_000_000_000
    goal.deadline.nanosec = expired_ns % 1_000_000_000
    goal_handle = _GoalHandle(goal)
    node = object.__new__(PipelinePolicyNode)
    node._session_controller = _Controller()
    node._boot_id = BOOT_ID
    node._scheduled_binding_identity = None
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
    goal = DispatchPipelineBinding.Goal()
    goal.request_id = REQUEST_ID
    goal.session_id = SESSION_ID
    goal.logical_generation = 1
    goal.binding_id = BINDING_ID
    goal.binding_incarnation = 1
    goal.operation_id = NEW_BOOT_ID
    goal.expected_boot_id = BOOT_ID
    goal.expected_pipeline_generation = 1
    goal.priority = 1
    deadline_ns = time.time_ns() + 1_000_000_000
    goal.deadline.sec, goal.deadline.nanosec = divmod(deadline_ns, 1_000_000_000)
    goal_handle = _GoalHandle(goal)
    node = object.__new__(PipelinePolicyNode)
    node._session_controller = SimpleNamespace()
    node._boot_id = BOOT_ID
    node._scheduled_binding_identity = (SESSION_ID, 1, BINDING_ID, 1, BOOT_ID, 1)
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

    goal = DispatchPipelineBinding.Goal()
    goal.request_id = REQUEST_ID
    goal.session_id = SESSION_ID
    goal.logical_generation = 1
    goal.binding_id = BINDING_ID
    goal.binding_incarnation = 1
    goal.operation_id = NEW_BOOT_ID
    goal.expected_boot_id = BOOT_ID
    goal.expected_pipeline_generation = 1
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
    node._boot_id = BOOT_ID
    node._scheduled_binding_identity = (SESSION_ID, 1, BINDING_ID, 1, BOOT_ID, 1)
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
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))
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


def _capability_status(*, stage_policy: str, deadline_admission: bool) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_id="policy",
        boot_id=BOOT_ID,
        sequence=1,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
        state=InferenceServingStatus.IDLE,
        error=SimpleNamespace(code=""),
        deployment_fingerprint="d" * 64,
        runtime_policy_fingerprint="r" * 64,
        pipeline_compatibility_fingerprint="c" * 64,
        configured_hardware_resource_id="ascend:0",
        runtime_hardware_resource_id="ascend:0",
        hardware_priority_levels=8,
        scheduling_capability_schema_version=1,
        stage_policy=stage_policy,
        supports_priority_zero_deadline_admission=deadline_admission,
        max_supported_public_priority=7,
        capacities=[],
    )


def test_independent_status_claiming_deadline_admission_is_rejected():
    candidate = SimpleNamespace(
        pipeline_id="policy",
        deployment_fingerprint="d" * 64,
        runtime_policy_fingerprint="r" * 64,
        hardware_resource_id="ascend:0",
        public_capacity={},
    )
    node = object.__new__(GlobalInferenceSchedulerNode)
    node._clock_skew_tolerance_ns = 0
    node._status_stale_timeout_ns = 10**30  # bypass stale checks with a synthetic clock
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))
    message = _capability_status(stage_policy="INDEPENDENT", deadline_admission=True)

    reason = GlobalInferenceSchedulerNode._status_message_reason(node, candidate, message)

    assert reason == "priority_zero_deadline_admission_capability_invalid"
    consistent = _capability_status(stage_policy="INDEPENDENT", deadline_admission=False)
    assert GlobalInferenceSchedulerNode._status_message_reason(node, candidate, consistent) == ""
    sequential = _capability_status(stage_policy="SEQUENTIAL", deadline_admission=True)
    assert GlobalInferenceSchedulerNode._status_message_reason(node, candidate, sequential) == ""


def test_public_dispatch_result_fields_cover_private_result():
    """The Global node copies private dispatch results onto the public result
    field-by-field (see _dispatch_scheduled in global_inference_scheduler_node).
    This guard fails when either message definition drifts so a field is
    silently dropped or added without updating the copy list."""
    private_fields = set(DispatchPipelineBinding.Result.get_fields_and_field_types())
    public_fields = set(ScheduledDispatchInfer.Result.get_fields_and_field_types())
    # Every public result field must exist on the private result so the copy
    # loop's getattr never fails at runtime.
    missing_on_private = public_fields - private_fields - {"session_generation"}
    assert not missing_on_private, f"public fields missing on private result: {sorted(missing_on_private)}"
    # session_generation is copied from the public goal, not the private result.
    assert "session_generation" in public_fields
    # Fields the private result carries beyond the public contract must stay
    # private-only; extend this set only when the private protocol adds
    # identity bookkeeping the public API deliberately hides.
    private_only = {
        "logical_generation",
        "binding_id",
        "binding_incarnation",
        "operation_id",
        "boot_id",
        "pipeline_generation",
    }
    unexpected_private_only = private_fields - public_fields - private_only
    assert not unexpected_private_only, (
        f"private result fields missing from the public contract or the private-only allowlist: "
        f"{sorted(unexpected_private_only)}"
    )


def test_call_downstream_unknown_cache_is_not_replayed_but_resent():
    """A cached UNKNOWN downstream outcome must never be replayed (P0-2):
    the quarantined record is fenced and the goal re-sent under a fresh
    operation id, so a Close retry actually reaches the pipeline again."""
    from inference_service.scheduler.operations import OperationRegistry

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._downstream_operations = OperationRegistry(max_records=4, max_waiters_per_operation=2)
    node._goal_acceptance_timeout_ns = 100_000_000

    class _Client:
        def wait_for_server(self, timeout_sec=None):
            return True

        sent_goals = []

        def send_goal_async(self, goal):
            _Client.sent_goals.append(goal)

            class _Future:
                def add_done_callback(self, callback):
                    class _GoalHandle:
                        accepted = True

                        def get_result_async(self):
                            class _ResultFuture:
                                def add_done_callback(self, result_callback):
                                    result = ClosePipelineBinding.Result()
                                    result.session_id = goal.session_id
                                    result.logical_generation = goal.logical_generation
                                    result.binding_id = goal.binding_id
                                    result.binding_incarnation = goal.binding_incarnation
                                    result.operation_id = goal.operation_id
                                    result.boot_id = goal.expected_boot_id
                                    result.pipeline_id = "policy"
                                    result.success = True
                                    result.outcome.value = InferenceOutcome.COMPLETED

                                    class _Done:
                                        def result(self):
                                            return SimpleNamespace(result=result)

                                    result_callback(_Done())
                                    return self

                            return _ResultFuture()

                    class _SentDone:
                        def result(self):
                            return _GoalHandle()

                    callback(_SentDone())
                    return self

            return _Future()

    downstream_goal = ClosePipelineBinding.Goal()
    downstream_goal.session_id = SESSION_ID
    downstream_goal.logical_generation = 1
    downstream_goal.binding_id = str(uuid4())
    downstream_goal.binding_incarnation = 1
    downstream_goal.operation_id = "close-op-unknown"
    downstream_goal.expected_boot_id = str(uuid4())

    # Seed the cache with a quarantined UNKNOWN record for the first attempt.
    seeded, created = node._downstream_operations.create_or_get(
        kind=OperationKind.CLOSE,
        idempotency_key=(OperationKind.CLOSE.value, "close-op-unknown"),
        identity=OperationIdentity(SESSION_ID, 1, downstream_goal.binding_id, 1, downstream_goal.expected_boot_id),
        deadline_mono_ns=time.monotonic_ns() + 5_000_000_000,
        waiter_id="seed",
    )
    assert created
    seeded.claim_send()
    node._downstream_operations.finish(seeded.operation_id, certainty=Certainty.UNKNOWN, error="timeout")
    node._downstream_operations.detach_waiter(seeded.operation_id, "seed")

    call = GlobalInferenceSchedulerNode._call_downstream(
        node,
        _Client(),
        downstream_goal,
        operation_kind=OperationKind.CLOSE,
        deadline_monotonic_ns=time.monotonic_ns() + 5_000_000_000,
    )

    assert call.certainty == "completed"
    assert downstream_goal.operation_id != "close-op-unknown"
    assert _Client.sent_goals[0].operation_id == downstream_goal.operation_id
    assert node._downstream_operations.find((OperationKind.CLOSE.value, "close-op-unknown")) is None
    assert node._downstream_operations.find((OperationKind.CLOSE.value, downstream_goal.operation_id)) is None


def test_call_downstream_completed_cache_replays_without_resend():
    """A known terminal outcome still replays for concurrent waiters without
    re-sending; the detached-record case simply creates a fresh send."""
    from inference_service.scheduler.operations import OperationRegistry

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._downstream_operations = OperationRegistry(max_records=4, max_waiters_per_operation=4)
    node._goal_acceptance_timeout_ns = 100_000_000

    class _Client:
        def wait_for_server(self, timeout_sec=None):
            return True

        def send_goal_async(self, goal):
            raise AssertionError("cached COMPLETED outcome must not re-send")

    downstream_goal = ClosePipelineBinding.Goal()
    downstream_goal.session_id = SESSION_ID
    downstream_goal.logical_generation = 1
    downstream_goal.binding_id = str(uuid4())
    downstream_goal.binding_incarnation = 1
    downstream_goal.operation_id = "close-op-done"
    downstream_goal.expected_boot_id = str(uuid4())

    seeded, created = node._downstream_operations.create_or_get(
        kind=OperationKind.CLOSE,
        idempotency_key=(OperationKind.CLOSE.value, "close-op-done"),
        identity=OperationIdentity(SESSION_ID, 1, downstream_goal.binding_id, 1, downstream_goal.expected_boot_id),
        deadline_mono_ns=time.monotonic_ns() + 5_000_000_000,
        waiter_id="seed",
    )
    assert created
    seeded.claim_send()
    node._downstream_operations.finish(seeded.operation_id, certainty=Certainty.COMPLETED, result="cached-result")
    # The seed waiter stays attached, so the terminal record is retained and
    # a concurrent caller with the same idempotency key replays it.
    seeded.attach_waiter("seed2", max_waiters=4)
    assert seeded.waiter_count == 2

    call = GlobalInferenceSchedulerNode._call_downstream(
        node,
        _Client(),
        downstream_goal,
        operation_kind=OperationKind.CLOSE,
        deadline_monotonic_ns=time.monotonic_ns() + 5_000_000_000,
    )

    assert call.certainty == "completed"
    assert call.result == "cached-result"
    assert downstream_goal.operation_id == "close-op-done"
    # Cleanup: dropping the last two waiters reclaims the terminal record.
    node._downstream_operations.detach_waiter(seeded.operation_id, "seed2")
    node._downstream_operations.detach_waiter(seeded.operation_id, "seed")
    assert node._downstream_operations.get(seeded.operation_id) is None


def test_idle_sweep_retries_quarantined_close_once_then_retains_ownership():
    """Exhaustion must leave cleanup ownership available to a later Close."""
    sweep_calls: list[str] = []
    record = SimpleNamespace(
        session_generation=2,
        state=GlobalSessionState.QUARANTINED,
        in_flight_requests=0,
        unresolved_cleanup=True,
        bindings={"policy": BINDING_ID},
    )

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = SimpleNamespace(
        expired_sessions=lambda: [],
        session_record=lambda _session_id: record,
        aged_quarantined_sessions=lambda _age_ns: [SESSION_ID] if len(sweep_calls) < 3 else [],
        begin_close=lambda **_kwargs: None,
        wait_for_bindings_to_settle=lambda *_args: True,
        close_bindings=lambda _session_id: [],
        record_close_complete=lambda *args, **kwargs: 2,
    )
    node._close_retry_exhausted = set()
    node._session_idle_timeout_ns = 1
    node._default_request_timeout_ns = 1_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node._pipeline_clients = {}
    node._candidate_by_id = {}
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_args: None, error=lambda *_args: None)

    def _close_once(goal_handle, _ledger):
        sweep_calls.append(goal_handle.request.session_id)
        result = CloseInferenceSession.Result()
        result.outcome.value = InferenceOutcome.UNKNOWN
        return result

    node._close_once = _close_once

    # First sweep: the aged quarantined session is retried once.
    node._idle_sweep()
    assert sweep_calls == [SESSION_ID]
    assert SESSION_ID in node._close_retry_exhausted

    # Further sweeps cannot forget ownership without a drain or boot fence.
    node._idle_sweep()
    node._idle_sweep()
    assert sweep_calls == [SESSION_ID]
    assert record.bindings == {"policy": BINDING_ID}
    assert record.state is GlobalSessionState.QUARANTINED
    assert record.unresolved_cleanup
    assert SESSION_ID in node._close_retry_exhausted


@pytest.mark.parametrize("wrapped_in_failure", [False, True])
def test_superseded_visual_frame_is_debug_not_failure(wrapped_in_failure):
    """P2: a superseded Visual frame is normal coalescing. It must not set
    _last_error, log at ERROR, or quarantine the session controller — whether
    the raw VisualFrameSupersededError or its ExecutionFailure-wrapped form
    (with the original as ``cause``) arrives."""
    from inference_service.pipeline import VisualFrameSupersededError
    from inference_service.unified_runtime import ExecutionFailure, OutcomeEvidence

    quarantined: list[bool] = []
    logged: list[tuple[str, object]] = []

    class _SupersededFuture:
        @staticmethod
        def result():
            raise VisualFrameSupersededError("pending Visual frame was replaced by a newer frame")

    node = object.__new__(PipelinePolicyNode)
    node._frame_trigger_lock = threading.Lock()
    node._frame_trigger_future = _SupersededFuture()
    node._visual_trigger_epoch = 0
    node._last_error = "previous"
    node._session_controller = SimpleNamespace(mark_failed_quarantine=lambda: quarantined.append(True))

    class _Logger:
        def debug(self, *_args, **_kwargs):
            logged.append(("debug", None))

        def error(self, message, *_args, **_kwargs):
            logged.append(("error", message))

    node.get_logger = lambda: _Logger()
    node._publish_serving_status = lambda: logged.append(("publish", None))

    if wrapped_in_failure:

        class _WrappedFuture:
            @staticmethod
            def result():
                raise ExecutionFailure(
                    "execution_failed",
                    "pending Visual frame was replaced by a newer frame",
                    evidence=OutcomeEvidence.not_started("backend"),
                    cause=VisualFrameSupersededError("superseded"),
                )

        PipelinePolicyNode._visual_frame_completed(node, _WrappedFuture(), 0)
    else:
        PipelinePolicyNode._visual_frame_completed(node, node._frame_trigger_future, 0)

    assert quarantined == []
    assert node._last_error == "previous"
    assert [level for level, _ in logged] == ["debug"]


def test_idle_sweep_preserves_retry_slot_when_close_already_in_progress():
    """A concurrently running Close must not consume the bounded retry slot:
    begin_close's close_in_progress rejection dispatches nothing, so the next
    sweep still gets its one real retry."""
    from inference_service.scheduler.global_scheduler_core import SchedulerError as _SchedulerError

    node = object.__new__(GlobalInferenceSchedulerNode)
    node._core = SimpleNamespace(
        expired_sessions=lambda: [],
        session_record=lambda _session_id: SimpleNamespace(
            session_generation=2,
            state=GlobalSessionState.QUARANTINED,
            in_flight_requests=0,
        ),
        aged_quarantined_sessions=lambda _age_ns: [SESSION_ID],
        begin_close=lambda **_kwargs: (_ for _ in ()).throw(_SchedulerError("close_in_progress")),
    )
    node._close_retry_exhausted = set()
    node._session_idle_timeout_ns = 1
    node._default_request_timeout_ns = 1_000_000_000
    node._max_error_message_bytes = 1024
    node._max_error_details_bytes = 8192
    node.get_logger = lambda: SimpleNamespace(warning=lambda *_args: None, error=lambda *_args: None)

    for _ in range(3):
        node._idle_sweep()

    # The slot is never consumed: no release ever fires.
    assert node._close_retry_exhausted == set()

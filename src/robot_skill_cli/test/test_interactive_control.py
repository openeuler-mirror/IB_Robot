"""Unit tests for the interactive closed-loop controller (no ROS, fake bridge)."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from robot_skill_cli import interactive_control as ic


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_hook = None

    def monotonic(self) -> float:
        return self.now

    def sleep(self, sec: float) -> None:
        self.now += float(sec)
        if self.sleep_hook is not None:
            hook, self.sleep_hook = self.sleep_hook, None
            hook()


class FakeFuture:
    def __init__(self, result_value: Any, *, done: bool = True, error: Exception | None = None) -> None:
        self._result = result_value
        self._done = done
        self._error = error

    def done(self) -> bool:
        return self._done

    def result(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._result


class FakeGoalHandle:
    def __init__(self, *, accepted: bool = True, result_future: FakeFuture | None = None) -> None:
        self.accepted = accepted
        self._result_future = result_future or FakeFuture(None, done=False)

    def get_result_async(self) -> FakeFuture:
        return self._result_future


def _capability_view() -> dict[str, Any]:
    def skill(name: str, *, visible: bool, level: str = "skill") -> dict[str, Any]:
        return {
            "name": name,
            "semantic_level": level,
            "planner_visible": visible,
            "summary": "",
            "domain": "",
            "moves_robot": True,
            "required_control_mode": "moveit_planning",
            "parameters": {},
            "recovery_policy": {},
        }

    return {
        "robot_name": "test_robot",
        "skills": [
            skill("nod_yes", visible=True),
            skill("wave_hand", visible=True),
            skill("internal_only", visible=False, level="atomic_operator"),
        ],
        "pose_names": [],
        "timeout_policy": {"rpc_timeout_sec": 30.0},
        "capability_digest": "capdig",
        "profile_name": "test",
    }


class FakeBridge:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.status = {
            "control_plane_ready": True,
            "registry_epoch": "epoch-1",
            "registry_generation": 1,
            "registry_digest": "regdig",
            "task_budget_sec": 90.0,
            "rpc_timeout_sec": 2.0,
            "capabilities": [
                {
                    "name": "nod_yes",
                    "ready": True,
                    "reason": "",
                    "semantic_level": "skill",
                    "planner_visible": True,
                    "required_control_mode": "moveit_planning",
                },
                {
                    "name": "wave_hand",
                    "ready": True,
                    "reason": "",
                    "semantic_level": "skill",
                    "planner_visible": True,
                    "required_control_mode": "moveit_planning",
                },
            ],
        }
        self.snapshot = {"success": True, "snapshot_json": ""}
        self.plan_result_status = 4
        self.plan_result = {
            "success": True,
            "plan_id": "",
            "plan_digest": "pdig",
            "completed_step_count": 1,
            "error_code": "",
            "message": "ok",
            "actual_registry_epoch": "epoch-1",
            "actual_registry_generation": 1,
            "actual_registry_digest": "regdig",
        }
        self.server_ready = True
        self.result_future = FakeFuture(None, done=False)
        self.goal_future = FakeFuture(FakeGoalHandle(result_future=self.result_future), done=True)
        self.validate_hook = None
        self.confirm_hook = None
        self.send_hook = None
        self.status_hook = None
        self.wait_future_hook = None

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, dict(kwargs)))

    def get_status(self, *, task_id, payload_hash, timeout_sec):
        self._record("get_status", {"task_id": task_id, "payload_hash": payload_hash})
        if self.status_hook is not None:
            self.status_hook()
        return dict(self.status)

    def get_skill_snapshot(self, *, registry_epoch, generation, timeout_sec):
        self._record("get_skill_snapshot", {"registry_epoch": registry_epoch, "generation": generation})
        return dict(self.snapshot)

    def plan_agent_command(
        self, *, request_id, raw_command, workflow_steps, timeout_sec, execution_mode="interactive_confirmation"
    ):
        self._record(
            "plan_agent_command",
            {"request_id": request_id, "raw_command": raw_command, "execution_mode": execution_mode},
        )
        self.plan_result["plan_id"] = "pid-" + request_id
        self.plan_result["completed_step_count"] = len(workflow_steps)
        return {
            "success": True,
            "plan": {
                "plan_token": "ptok-" + request_id,
                "plan_digest": "pdig",
                "plan_id": "pid-" + request_id,
                "plan_kind": 1 if len(workflow_steps) == 1 else 2,
                "raw_command": raw_command,
                "workflow_steps": [dict(step) for step in workflow_steps],
                "registry_epoch": "epoch-1",
                "registry_generation": 1,
                "registry_digest": "regdig",
                "execution_mode": execution_mode,
            },
            "error_code": "",
            "message": "",
        }

    def validate_agent_plan(self, *, plan_token, timeout_sec):
        self._record("validate_agent_plan", {"plan_token": plan_token})
        if self.validate_hook is not None:
            self.validate_hook()
        return {
            "allowed": True,
            "plan_id": self.plan_result["plan_id"],
            "plan_digest": "pdig",
            "error_code": "",
            "message": "",
        }

    def confirm_agent_plan(
        self,
        *,
        plan_token,
        plan_digest,
        task_id,
        status,
        task_budget_sec,
        timeout_sec,
        execution_mode="interactive_confirmation",
    ):
        self._record(
            "confirm_agent_plan",
            {
                "plan_token": plan_token,
                "plan_digest": plan_digest,
                "task_id": task_id,
                "execution_mode": execution_mode,
            },
        )
        if self.confirm_hook is not None:
            self.confirm_hook()
        return {
            "confirmed": True,
            "confirmation_token": "ctok-" + task_id,
            "confirmed_task_budget_sec": float(task_budget_sec),
            "error_code": "",
            "message": "",
        }

    def wait_for_execute_plan_server(self, *, timeout_sec):
        return self.server_ready

    def send_agent_plan_goal(self, *, plan_token, confirmation_token, task_id, timeout_sec, feedback_callback):
        self._record(
            "send_agent_plan_goal",
            {"plan_token": plan_token, "confirmation_token": confirmation_token, "task_id": task_id},
        )
        if self.send_hook is not None:
            self.send_hook()
        return self.goal_future

    def wait_future(self, future, *, timeout_sec, interrupt_event=None):
        if self.wait_future_hook is not None:
            self.wait_future_hook()
        return future.done() and (interrupt_event is None or not interrupt_event.is_set())

    def cancel_agent_plan(self, task_id, *, timeout_sec):
        self._record("cancel_agent_plan", {"task_id": task_id})
        return {"accepted": True, "return_code": 0}

    def get_agent_plan_result(self, task_id, *, timeout_sec):
        self._record("get_agent_plan_result", {"task_id": task_id})
        return {"status": self.plan_result_status, "result": dict(self.plan_result)}


@pytest.fixture
def rig(monkeypatch):
    clock = FakeClock()
    bridge = FakeBridge(clock)
    counter = iter(range(1, 100))
    controller = ic.InteractiveController(
        bridge,
        timeout_policy={"rpc_timeout_sec": 30.0},
        id_factory=lambda: f"id-{next(counter)}",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        view_resolver=lambda snapshot, status: _capability_view(),
    )
    return controller, bridge


def _step(skill_name: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_name": skill_name,
        "target_name": "",
        "place_name": "",
        "motion_direction": "",
        "motion_distance": 0.0,
        "timeout_sec": 0.0,
    }


def test_classify_closed_grammar():
    assert ic.classify("确认") == ic.INTENT_CONFIRM
    assert ic.classify(" 别动 ") == ic.INTENT_STOP
    assert ic.classify("继续吧") == ic.INTENT_CONTINUE
    assert ic.classify("行不行") == ic.INTENT_UNKNOWN
    assert ic.classify("可以不要执行") == ic.INTENT_UNKNOWN
    assert ic.classify("继续吧，不要执行") == ic.INTENT_UNKNOWN
    assert ic.classify("") == ic.INTENT_UNKNOWN


def test_discover_is_readonly_and_queries_catalog(rig):
    controller, bridge = rig
    result = controller.discover()

    methods = [call[0] for call in bridge.calls]
    assert methods == ["get_status", "get_skill_snapshot"]
    assert controller.state == ic.DISCOVERED
    assert result["robot_name"] == "test_robot"
    assert "nod_yes" in result["planner_visible_names"]
    assert "internal_only" not in result["planner_visible_names"]
    # Feature 1: no plan/confirm/execute motion primitives were invoked.
    assert "plan_agent_command" not in methods
    assert "send_agent_plan_goal" not in methods


def test_reject_out_of_catalog_blocks_planning(rig):
    controller, bridge = rig
    controller.discover()

    with pytest.raises(ic.OutOfCatalogError) as exc_info:
        controller.prepare_workflow("ghost motion", [_step("ghost_skill")])
    assert exc_info.value.code == "SKILL_REFERENCE_MISSING"

    # A planner-invisible (internal) skill is also rejected.
    with pytest.raises(ic.OutOfCatalogError):
        controller.prepare_workflow("internal", [_step("internal_only")])

    methods = [call[0] for call in bridge.calls]
    assert "plan_agent_command" not in methods


def test_prepare_then_confirm_presentation_and_nl_grammar(rig):
    controller, bridge = rig
    controller.discover()

    presentation = controller.prepare_workflow("点个头", [_step("nod_yes")])

    assert controller.state == ic.PREPARED
    assert presentation["steps"][0]["skill_name"] == "nod_yes"
    assert presentation["plan_digest"] == "pdig"
    assert presentation["task_id"] == "id-2"
    assert presentation["execution_mode"] == "interactive_confirmation"
    assert presentation["plan_kind"] == 1
    assert presentation["proposed_task_budget_sec"] == 90.0

    # Open grammar must not confirm.
    with pytest.raises(ic.NotConfirmedError):
        controller.confirm("行不行吧")

    confirmed = controller.confirm("确认执行当前计划")
    assert controller.state == ic.CONFIRMED
    assert confirmed["task_id"] == "id-2"
    confirm_call = next(c for m, c in bridge.calls if m == "confirm_agent_plan")
    assert confirm_call["plan_token"] == "ptok-id-1"
    assert confirm_call["plan_digest"] == "pdig"
    assert confirm_call["task_id"] == "id-2"


def test_execute_success_reaches_succeeded_terminal(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    bridge.plan_result_status = 4
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")

    result = controller.execute()

    assert controller.state == ic.SUCCEEDED
    assert result["state"] == ic.SUCCEEDED
    assert result["error_code"] == ""
    assert "send_agent_plan_goal" in [m for m, _ in bridge.calls]


def test_confirm_plan_runs_without_grammar_gate(rig):
    controller, bridge = rig
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])

    confirmed = controller.confirm_plan()

    assert controller.state == ic.CONFIRMED
    assert confirmed["task_id"] == "id-2"
    methods = [m for m, _ in bridge.calls]
    assert "validate_agent_plan" in methods
    assert "confirm_agent_plan" in methods


def test_run_executes_immediately_after_required_presentation(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    bridge.plan_result_status = 4

    result = controller.run("点个头", [_step("nod_yes")], presentation_callback=lambda _presentation: None)

    assert controller.state == ic.SUCCEEDED
    assert result["state"] == ic.SUCCEEDED
    methods = [m for m, _ in bridge.calls]
    # No user 确认 gate: validate+confirm happen automatically, then execute.
    assert "validate_agent_plan" in methods
    assert "confirm_agent_plan" in methods
    assert "send_agent_plan_goal" in methods
    assert "get_agent_plan_result" in methods


def test_run_rejects_changed_expected_registry_identity_before_planning(rig):
    controller, bridge = rig

    with pytest.raises(ic.InteractiveControlError, match="registry identities differ"):
        controller.run(
            "点个头",
            [_step("nod_yes")],
            expected_registry_identity=("stale-epoch", 1, "stale-digest"),
            presentation_callback=lambda _presentation: None,
        )

    methods = [method for method, _ in bridge.calls]
    assert "plan_agent_command" not in methods
    assert "send_agent_plan_goal" not in methods
    assert controller.state == ic.DISCOVERED


def test_submission_callback_failure_blocks_goal_submission(rig):
    _controller, bridge = rig
    controller = ic.InteractiveController(
        bridge,
        timeout_policy={"rpc_timeout_sec": 5.0},
        submission_callback=lambda _detail: (_ for _ in ()).throw(OSError("disk full")),
        view_resolver=lambda _snapshot, _status: _capability_view(),
    )
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)

    result = controller.run("点个头", [_step("nod_yes")], presentation_callback=lambda _presentation: None)

    assert result["state"] == ic.FAILED
    assert result["error_code"] == "SUBMISSION_PERSISTENCE_FAILED"
    assert "send_agent_plan_goal" not in [method for method, _ in bridge.calls]


def test_run_uses_explicit_immediate_execution_mode(rig):
    controller, bridge = rig
    controller = ic.InteractiveController(
        bridge,
        timeout_policy={"rpc_timeout_sec": 5.0},
        execution_mode="immediate_after_presentation",
        view_resolver=lambda _snapshot, _status: {
            "robot_name": "test",
            "capability_digest": "capdig",
            "skills": bridge.status["capabilities"],
        },
    )
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)

    controller.run("点个头", [_step("nod_yes")], presentation_callback=lambda _presentation: None)

    plan_call = next(call for method, call in bridge.calls if method == "plan_agent_command")
    confirm_call = next(call for method, call in bridge.calls if method == "confirm_agent_plan")
    assert plan_call.get("execution_mode") == "immediate_after_presentation"
    assert confirm_call.get("execution_mode") == "immediate_after_presentation"


def test_run_notifies_authorization_before_goal_submission(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    events = []

    controller.run(
        "点个头",
        [_step("nod_yes")],
        presentation_callback=lambda _presentation: None,
        authorization_callback=lambda _confirmation: events.append([method for method, _ in bridge.calls]),
    )

    assert events
    assert "confirm_agent_plan" in events[0]
    assert "send_agent_plan_goal" not in events[0]


def test_run_presents_before_internal_confirm_and_goal_send(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    methods_at_presentation = []

    controller.run(
        "点个头",
        [_step("nod_yes")],
        presentation_callback=lambda _presentation: methods_at_presentation.extend(m for m, _ in bridge.calls),
    )

    assert "plan_agent_command" in methods_at_presentation
    assert "validate_agent_plan" not in methods_at_presentation
    assert "confirm_agent_plan" not in methods_at_presentation
    assert "send_agent_plan_goal" not in methods_at_presentation


def test_run_does_not_confirm_when_presentation_requests_stop(rig):
    controller, bridge = rig

    result = controller.run(
        "点个头",
        [_step("nod_yes")],
        presentation_callback=lambda _presentation: controller.request_stop(),
    )

    assert result["state"] == ic.STOPPED
    assert "confirm_agent_plan" not in [method for method, _ in bridge.calls]


def test_run_no_gate_stop_interrupts(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5  # canceled -> STOPPED
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled", completed_step_count=0)
    stop_event = threading.Event()
    stop_event.set()

    result = controller.run(
        "点个头",
        [_step("nod_yes")],
        presentation_callback=lambda _presentation: None,
        stop_event=stop_event,
    )

    assert controller.state == ic.STOPPED
    assert result["state"] == ic.STOPPED
    assert "send_agent_plan_goal" not in [m for m, _ in bridge.calls]
    assert "cancel_agent_plan" not in [m for m, _ in bridge.calls]
    with pytest.raises(ic.IllegalStateError):
        controller.confirm_plan()


def test_run_latches_stop_during_discovery(rig):
    controller, bridge = rig
    bridge.status_hook = controller.request_stop

    result = controller.run(
        "点个头",
        [_step("nod_yes")],
        presentation_callback=lambda _presentation: None,
    )

    assert result["state"] == ic.STOPPED
    assert "send_agent_plan_goal" not in [m for m, _ in bridge.calls]


def test_run_latches_stop_from_presentation_callback(rig):
    controller, bridge = rig

    result = controller.run(
        "点个头",
        [_step("nod_yes")],
        presentation_callback=lambda _presentation: controller.request_stop(),
    )

    assert result["state"] == ic.STOPPED
    assert "confirm_agent_plan" not in [m for m, _ in bridge.calls]
    assert "send_agent_plan_goal" not in [m for m, _ in bridge.calls]


def test_run_rejects_concurrent_operation(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    status_started = threading.Event()
    release_status = threading.Event()

    def block_status():
        status_started.set()
        assert release_status.wait(timeout=5.0)

    bridge.status_hook = block_status
    result_holder = {}
    run_thread = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result",
            controller.run("点个头", [_step("nod_yes")], presentation_callback=lambda _presentation: None),
        )
    )
    run_thread.start()
    assert status_started.wait(timeout=5.0)

    with pytest.raises(ic.IllegalStateError):
        controller.run("挥手", [_step("wave_hand")], presentation_callback=lambda _presentation: None)

    release_status.set()
    run_thread.join(timeout=5.0)
    assert not run_thread.is_alive()
    assert result_holder["result"]["state"] == ic.SUCCEEDED
    assert [m for m, _ in bridge.calls].count("send_agent_plan_goal") == 1


def test_run_releases_starting_state_after_prepare_failure(rig):
    controller, bridge = rig
    bridge.plan_agent_command = lambda **_kwargs: {
        "success": False,
        "error_code": "SKILL_SCHEMA_INVALID",
        "message": "invalid",
    }

    with pytest.raises(ic.InteractiveControlError):
        controller.run("点个头", [_step("nod_yes")], presentation_callback=lambda _presentation: None)

    assert controller.state == ic.DISCOVERED


def test_execute_releases_ownership_when_server_is_unavailable(rig):
    controller, bridge = rig
    bridge.server_ready = False
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()

    result = controller.execute()

    assert result["state"] == ic.FAILED
    assert result["error_code"] == "SERVER_UNAVAILABLE"
    assert controller.state == ic.FAILED


def test_execute_acceptance_transport_failure_converges(rig):
    controller, bridge = rig
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()
    bridge.wait_future_hook = lambda: (_ for _ in ()).throw(RuntimeError("transport failed"))
    bridge.plan_result_status = 5
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled")

    result = controller.execute()

    assert result["state"] == ic.STOPPED
    assert result["error_code"] == "SKILL_CANCELLED"
    assert [m for m, _ in bridge.calls].count("cancel_agent_plan") == 1


def test_confirm_response_loss_invalidates_operation(rig):
    controller, bridge = rig
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    bridge.confirm_hook = lambda: (_ for _ in ()).throw(RuntimeError("response lost"))

    with pytest.raises(ic.InteractiveControlError) as exc_info:
        controller.confirm_plan()

    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"
    assert controller.state == ic.UNKNOWN
    with pytest.raises(ic.IllegalStateError):
        controller.execute()


def test_presentation_failure_invalidates_pending_plan(rig):
    controller, bridge = rig

    def fail_presentation(_presentation):
        raise OSError("output transport failed")

    with pytest.raises(ic.InteractiveControlError) as exc_info:
        controller.run("点个头", [_step("nod_yes")], presentation_callback=fail_presentation)

    assert exc_info.value.code == "PRESENTATION_FAILED"
    assert controller.state == ic.FAILED
    with pytest.raises(ic.IllegalStateError):
        controller.confirm_plan()
    assert "confirm_agent_plan" not in [m for m, _ in bridge.calls]
    assert "send_agent_plan_goal" not in [m for m, _ in bridge.calls]


def test_execute_stop_reaches_definite_stopped_terminal(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5  # GoalStatus.CANCELED
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled", completed_step_count=0)
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")
    bridge.clock.sleep_hook = controller.request_stop

    result = controller.execute()

    assert controller.state == ic.STOPPED
    assert result["state"] == ic.STOPPED
    methods = [m for m, _ in bridge.calls]
    assert "cancel_agent_plan" in methods
    assert "get_agent_plan_result" in methods


def test_execute_stop_unknown_refuses_continue(rig):
    controller, bridge = rig
    bridge.plan_result_status = 0  # never converges to a terminal GoalStatus
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")
    bridge.clock.sleep_hook = controller.request_stop

    result = controller.execute()

    assert controller.state == ic.UNKNOWN
    assert result["state"] == ic.UNKNOWN
    assert result["error_code"] == "SKILL_CANCEL_TIMEOUT"

    # Feature 5 gate: unknown stop state must refuse a continuation.
    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("再点一次", [_step("nod_yes")])
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"


def test_continue_uses_fresh_state_and_new_ids(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.plan_result_status = 5
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled", completed_step_count=0)
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")
    bridge.clock.sleep_hook = controller.request_stop
    controller.execute()
    assert controller.state == ic.STOPPED

    first_plan_calls = [c for m, c in bridge.calls if m == "plan_agent_command"]
    first_send_calls = [c for m, c in bridge.calls if m == "send_agent_plan_goal"]
    assert len(first_plan_calls) == 1 and len(first_send_calls) == 1
    first_task_id = first_send_calls[0]["task_id"]

    presentation = controller.continue_workflow("再点一次", [_step("wave_hand")])

    assert controller.state == ic.PREPARED
    assert presentation["continues_from"]["state"] == ic.STOPPED
    # Fresh discover happened again (second status + snapshot after the first batch).
    method_counts = {m: sum(1 for mm, _ in bridge.calls if mm == m) for m in ("get_status", "get_skill_snapshot")}
    assert method_counts["get_status"] >= 2
    assert method_counts["get_skill_snapshot"] >= 2
    # New task id, never reused.
    assert presentation["task_id"] != first_task_id
    second_plan_calls = [c for m, c in bridge.calls if m == "plan_agent_command"]
    assert len(second_plan_calls) == 2
    assert second_plan_calls[1]["request_id"] != second_plan_calls[0]["request_id"]


def test_continue_preserves_canceled_proof_after_replanning_failure(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.plan_result_status = 5
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled", completed_step_count=0)
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()
    bridge.clock.sleep_hook = controller.request_stop
    controller.execute()
    prior_terminal = dict(controller._terminal)
    bridge.plan_agent_command = lambda **_kwargs: {
        "success": False,
        "error_code": "SKILL_SCHEMA_INVALID",
        "message": "invalid",
    }

    with pytest.raises(ic.InteractiveControlError):
        controller.continue_workflow("继续", [_step("wave_hand")])

    assert controller.state == ic.STOPPED
    assert controller._terminal == prior_terminal


def test_continue_requires_definite_terminal(rig):
    controller, bridge = rig
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")
    assert controller.state == ic.CONFIRMED

    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("继续", [_step("nod_yes")])
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"


def test_continue_rejects_unproven_stopped_state(rig):
    controller, _bridge = rig
    controller._state = ic.STOPPED
    controller._terminal = {"state": ic.STOPPED, "task_id": "task-1", "result": {}}

    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("继续", [_step("nod_yes")])

    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"


def test_request_stop_from_another_thread_reaches_stopped(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled", completed_step_count=0)
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm("确认")

    result_holder: dict[str, Any] = {}

    def run_execute():
        result_holder["result"] = controller.execute()

    thread = threading.Thread(target=run_execute)
    thread.start()
    # Wait for the execute loop to enter EXECUTING, then trigger stop externally.
    for _ in range(200):
        if controller.state == ic.EXECUTING:
            break
        import time

        time.sleep(0.001)
    controller.request_stop()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert controller.state == ic.STOPPED
    assert result_holder["result"]["state"] == ic.STOPPED


def test_request_stop_does_not_block_on_goal_submission(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled", completed_step_count=0)
    bridge.result_future = FakeFuture(None, done=False)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    send_started = threading.Event()
    release_send = threading.Event()

    def block_send():
        send_started.set()
        assert release_send.wait(timeout=5.0)

    bridge.send_hook = block_send
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()
    result_holder = {}
    execute_thread = threading.Thread(target=lambda: result_holder.setdefault("result", controller.execute()))
    execute_thread.start()
    assert send_started.wait(timeout=5.0)

    stop_thread = threading.Thread(target=controller.request_stop)
    stop_thread.start()
    stop_thread.join(timeout=0.5)
    assert not stop_thread.is_alive()
    release_send.set()
    execute_thread.join(timeout=5.0)

    assert not execute_thread.is_alive()
    assert result_holder["result"]["state"] == ic.STOPPED
    assert [m for m, _ in bridge.calls].count("cancel_agent_plan") == 1


def test_execute_rejects_concurrent_caller(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    send_started = threading.Event()
    release_send = threading.Event()

    def block_send():
        send_started.set()
        assert release_send.wait(timeout=5.0)

    bridge.send_hook = block_send
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()
    result_holder = {}
    first = threading.Thread(target=lambda: result_holder.setdefault("result", controller.execute()))
    first.start()
    assert send_started.wait(timeout=5.0)

    with pytest.raises(ic.IllegalStateError):
        controller.execute()

    release_send.set()
    first.join(timeout=5.0)
    assert not first.is_alive()
    assert result_holder["result"]["state"] == ic.SUCCEEDED
    assert [m for m, _ in bridge.calls].count("send_agent_plan_goal") == 1


def test_presentation_uses_authoritative_planned_steps(rig):
    controller, bridge = rig
    original_plan = bridge.plan_agent_command

    def canonical_plan(**kwargs):
        result = original_plan(**kwargs)
        result["plan"]["workflow_steps"][0]["motion_distance"] = 0.10000000149
        return result

    bridge.plan_agent_command = canonical_plan
    requested = _step("nod_yes")
    requested["motion_distance"] = 0.1
    controller.discover()

    presentation = controller.prepare_workflow("点个头", [requested])

    assert presentation["steps"][0]["motion_distance"] == 0.10000000149


def test_prepare_rejects_noncanonical_float_difference(rig):
    controller, bridge = rig
    original_plan = bridge.plan_agent_command

    def noncanonical_plan(**kwargs):
        result = original_plan(**kwargs)
        result["plan"]["workflow_steps"][0]["motion_distance"] = 0.10000005
        return result

    bridge.plan_agent_command = noncanonical_plan
    requested = _step("nod_yes")
    requested["motion_distance"] = 0.1
    controller.discover()

    with pytest.raises(ic.InteractiveControlError) as exc_info:
        controller.prepare_workflow("点个头", [requested])

    assert exc_info.value.code == "SKILL_SNAPSHOT_DIGEST_MISMATCH"


def test_prepare_rejects_server_plan_identity_mismatch(rig):
    controller, bridge = rig
    original_plan = bridge.plan_agent_command

    def wrong_identity(**kwargs):
        result = original_plan(**kwargs)
        result["plan"]["registry_digest"] = "other-registry"
        return result

    bridge.plan_agent_command = wrong_identity
    controller.discover()

    with pytest.raises(ic.InteractiveControlError) as exc_info:
        controller.prepare_workflow("点个头", [_step("nod_yes")])

    assert exc_info.value.code == "SKILL_SNAPSHOT_DIGEST_MISMATCH"


@pytest.mark.parametrize(
    ("status", "updates", "expected_error"),
    [
        (4, {"success": True, "error_code": "", "completed_step_count": 0}, "SKILL_CANCEL_TIMEOUT"),
        (6, {"success": False, "error_code": "SKILL_CANCELLED", "completed_step_count": 0}, "SKILL_CANCEL_TIMEOUT"),
        (
            6,
            {"success": False, "error_code": "SKILL_CANCEL_TIMEOUT", "completed_step_count": 0},
            "SKILL_CANCEL_TIMEOUT",
        ),
        (
            6,
            {"success": False, "error_code": "GATEWAY_FINALIZATION_FAILED", "completed_step_count": 0},
            "GATEWAY_FINALIZATION_FAILED",
        ),
        (
            6,
            {"success": False, "error_code": "SKILL_EXECUTION_BUSY", "completed_step_count": 0},
            "SKILL_EXECUTION_BUSY",
        ),
        (
            6,
            {"success": False, "error_code": "NEW_CLEANUP_STATE_UNKNOWN", "completed_step_count": 0},
            "SKILL_CANCEL_TIMEOUT",
        ),
        (4, {"success": True, "error_code": "", "actual_registry_generation": "invalid"}, "SKILL_CANCEL_TIMEOUT"),
    ],
)
def test_terminal_proof_mismatch_is_unknown(rig, status, updates, expected_error):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    bridge.plan_result_status = status
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()
    bridge.plan_result.update(updates)

    result = controller.execute()

    assert result["state"] == ic.UNKNOWN
    assert result["error_code"] == expected_error


def test_nonterminal_result_triggers_single_cancel(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    bridge.plan_result_status = 2
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()

    result = controller.execute()

    assert result["state"] == ic.UNKNOWN
    assert [m for m, _ in bridge.calls].count("cancel_agent_plan") == 1


def test_stable_aborted_result_is_failed(rig):
    controller, bridge = rig
    bridge.result_future = FakeFuture(None, done=True)
    bridge.goal_future = FakeFuture(FakeGoalHandle(result_future=bridge.result_future), done=True)
    bridge.plan_result_status = 6
    bridge.plan_result.update(success=False, error_code="CAPABILITY_NOT_READY", completed_step_count=0)
    controller.discover()
    controller.prepare_workflow("点个头", [_step("nod_yes")])
    controller.confirm_plan()

    result = controller.execute()

    assert result["state"] == ic.FAILED
    assert result["error_code"] == "CAPABILITY_NOT_READY"


def test_continue_resume_is_rejected_without_server_admission(rig):
    controller, bridge = rig
    bridge.plan_result_status = 5  # GoalStatus.CANCELED -> STOPPED
    bridge.plan_result.update(success=False, error_code="SKILL_CANCELLED", message="cancelled", completed_step_count=2)
    plan_steps = [_step("nod_yes"), _step("wave_hand"), _step("nod_yes"), _step("wave_hand")]
    controller.discover()
    controller.prepare_workflow("点头挥手四步", plan_steps)
    controller.confirm("确认")
    first_task_id = controller._pending["task_id"]
    bridge.clock.sleep_hook = controller.request_stop
    first_terminal = controller.execute()

    assert first_terminal["state"] == ic.STOPPED

    with pytest.raises(ic.IllegalStateError) as exc_info:
        controller.continue_workflow("继续", resume=True)
    assert exc_info.value.code == "SKILL_CONTINUATION_UNAVAILABLE"
    assert controller._terminal["task_id"] == first_task_id


def test_continue_resume_requires_definite_terminal(rig):
    controller, bridge = rig
    controller.discover()
    controller.prepare_workflow("点头", [_step("nod_yes")])
    controller.confirm("确认")
    assert controller.state == ic.CONFIRMED

    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("继续", resume=True)
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"


def test_continue_resume_unknown_refused(rig):
    controller, bridge = rig
    bridge.plan_result_status = 0  # never converges -> UNKNOWN
    bridge.plan_result["completed_step_count"] = 2
    controller.discover()
    controller.prepare_workflow("四步", [_step("nod_yes"), _step("wave_hand"), _step("nod_yes"), _step("wave_hand")])
    controller.confirm("确认")
    bridge.clock.sleep_hook = controller.request_stop
    controller.execute()
    assert controller.state == ic.UNKNOWN

    with pytest.raises(ic.UnknownStopError) as exc_info:
        controller.continue_workflow("继续", resume=True)
    assert exc_info.value.code == "SKILL_CANCEL_TIMEOUT"

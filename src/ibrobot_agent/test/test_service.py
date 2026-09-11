from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import AgentRequest, PlannerIdentity
from ibrobot_agent.planner import RulePlanner
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService


class FakePlanner:
    def __init__(self, outcome):
        self.identity = PlannerIdentity("fake", "fake", "json", "1", "1", "fake-config")
        self.outcome = outcome
        self.calls = 0

    def plan(self, request, context, catalog, cancel_token):
        self.calls += 1
        return self.outcome


class BlockingPlanner(FakePlanner):
    def __init__(self, outcome):
        super().__init__(outcome)
        self.started = threading.Event()
        self.release = threading.Event()

    def plan(self, request, context, catalog, cancel_token):
        self.calls += 1
        self.started.set()
        self.release.wait(2)
        return self.outcome


class FakeCatalog:
    def get_status(self):
        return {
            "robot_name": "so101_single_arm",
            "active_control_mode": "moveit_planning",
            "control_plane_ready": True,
            "motion_authorized": False,
            "registry_epoch": "epoch",
            "registry_generation": 1,
            "registry_digest": "digest",
        }

    def get_catalog(self, status):
        return {
            "robot_name": "so101_single_arm",
            "skills": [{"name": "wave_hello", "planner_visible": True, "semantic_level": "skill"}],
        }


class UnavailableCatalog:
    def get_status(self):
        raise RuntimeError("offline")

    def get_catalog(self, status):
        raise RuntimeError("offline")


class Events:
    def __init__(self):
        self.items = []
        self.ready = threading.Event()

    def publish(self, event):
        self.items.append(event)
        self.ready.set()


class FakeExecution:
    def __init__(self, *, status="succeeded"):
        self.status = status
        self.calls = 0
        self.stop_calls = 0

    def execute(
        self,
        proposal,
        *,
        expected_registry_identity,
        presentation_callback,
        submission_callback,
        stop_event,
    ):
        from ibrobot_agent.contracts import ExecutionResult, Presentation, TaskRef

        self.calls += 1
        task_ref = TaskRef("task-1", "plan-1", "plan-digest", "epoch", 1, "digest", 1)
        presentation_callback(
            Presentation(
                task_ref=task_ref,
                plan_kind=1,
                steps=proposal.outcome.steps,
                execution_mode="immediate_after_presentation",
                proposed_task_budget_sec=90.0,
                summary=proposal.outcome.summary,
            )
        )
        submission_callback(task_ref, {"confirmed": True})
        return ExecutionResult(self.status, task_ref, "" if self.status == "succeeded" else "FAILED", "done", {})

    def request_stop(self, request_key):
        self.stop_calls += 1


def _request(request_id: str = "r1", *, text: str = "hello", reply_to_request_id: str | None = None) -> AgentRequest:
    return AgentRequest(
        schema_version=1,
        request_id=request_id,
        session_id="s1",
        channel_id="test",
        principal_id="p1",
        robot_scope="so101_single_arm",
        text=text,
        received_at=datetime.now(timezone.utc),
        reply_to_request_id=reply_to_request_id,
    )


def _service(tmp_path, outcome):
    events = Events()
    planner = FakePlanner(outcome)
    service = AgentService(
        planner=planner,
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
    )
    return service, planner, events


def test_conversation_is_answered_without_execution(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, planner, events = _service(tmp_path, PlannerOutcome(kind="conversation", user_message="你好"))
    try:
        accepted = service.send_message(_request())
        assert accepted.accepted
        assert events.ready.wait(2)
        for _ in range(100):
            if service._store.get_request(_request().to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert planner.calls == 1
        for _ in range(100):
            if service._store.get_request(_request().to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service._store.get_request(_request().to_key()).state == "ANSWERED"
    finally:
        service.close()


def test_conversation_remains_available_when_gateway_is_offline(tmp_path):
    service = AgentService(
        planner=RulePlanner(),
        catalog=UnavailableCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        request = _request(text="你好")
        service.send_message(request)
        for _ in range(100):
            if service.get_request(request.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service.get_request(request.to_key()).state == "ANSWERED"
    finally:
        service.close()


def test_llm_conversation_can_answer_when_gateway_catalog_is_offline(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service = AgentService(
        planner=FakePlanner(PlannerOutcome(kind="conversation", user_message="我是小智。")),
        catalog=UnavailableCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        request = _request(text="你的名字是什么？")
        service.send_message(request)
        for _ in range(100):
            if service.get_request(request.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        record = service.get_request(request.to_key())
        assert record.state == "ANSWERED"
        assert record.terminal.message == "我是小智。"
    finally:
        service.close()


def test_dry_run_keeps_proposal_ready_and_never_calls_execution(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, _, events = _service(
        tmp_path,
        PlannerOutcome(
            kind="workflow",
            user_message="准备挥手",
            summary="挥手",
            steps=(CanonicalWorkflowStep(1, "wave_hello"),),
        ),
    )
    try:
        service.send_message(_request())
        assert events.ready.wait(2)
        assert service._store.get_request(_request().to_key()).state == "ANSWERED"
        assert any(event.event_type == "proposal_ready" for event in events.items)
    finally:
        service.close()


def test_duplicate_request_is_idempotent(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, planner, events = _service(tmp_path, PlannerOutcome(kind="conversation", user_message="ok"))
    try:
        request = _request()
        service.send_message(request)
        assert events.ready.wait(2)
        service.send_message(request)
        time.sleep(0.05)
        assert planner.calls == 1
    finally:
        service.close()


def test_second_motion_request_is_rejected_while_planning(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    planner = BlockingPlanner(PlannerOutcome(kind="conversation", user_message="ok"))
    service = AgentService(
        planner=planner,
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        assert service.send_message(_request("r1")).accepted
        assert planner.started.wait(2)
        second = service.send_message(_request("r2"))
        assert not second.accepted
        assert second.reason_code == "BUSY"
    finally:
        planner.release.set()
        service.close()


def test_execution_requires_non_empty_allowlist(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    planner = FakePlanner(PlannerOutcome(kind="conversation", user_message="ok"))
    with pytest.raises(ValueError, match="non-empty incubation allowlist"):
        AgentService(
            planner=planner,
            catalog=FakeCatalog(),
            store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
            execution_enabled=True,
        )


def test_execution_requires_execution_port(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    with pytest.raises(ValueError, match="requires an execution port"):
        AgentService(
            planner=FakePlanner(PlannerOutcome(kind="conversation", user_message="ok")),
            catalog=FakeCatalog(),
            store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
            execution_enabled=True,
            allowed_skills={"wave_hello"},
        )


def test_read_only_status_returns_gateway_facts(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, _, events = _service(
        tmp_path,
        PlannerOutcome(kind="read_only", user_message="checking", query_kind="status"),
    )
    try:
        request = _request()
        service.send_message(request)
        for _ in range(100):
            if service.get_request(request.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        for _ in range(100):
            if any(event.event_type == "read_only" for event in events.items):
                break
            time.sleep(0.01)
        assert "so101_single_arm" in service.get_request(request.to_key()).terminal.message
        assert "运动未授权" in service.get_request(request.to_key()).terminal.message
        assert any(event.event_type == "read_only" for event in events.items)
    finally:
        service.close()


def test_skill_query_uses_deterministic_path_and_bounds_message(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    class LargeCatalog(FakeCatalog):
        def get_catalog(self, status):
            return {
                "skills": [
                    {"name": f"skill_{index:03d}", "planner_visible": True, "semantic_level": "skill"}
                    for index in range(80)
                ]
            }

    planner = FakePlanner(
        PlannerOutcome(
            kind="workflow",
            user_message="不应调用",
            summary="不应调用",
            steps=(CanonicalWorkflowStep(1, "wave_hello"),),
        )
    )
    events = Events()
    service = AgentService(
        planner=planner,
        catalog=LargeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
    )
    try:
        request = _request(text="当前有哪些技能")
        service.send_message(request)
        for _ in range(100):
            record = service.get_request(request.to_key())
            if record.state == "ANSWERED":
                break
            time.sleep(0.01)
        assert record.state == "ANSWERED"
        assert len(record.terminal.message) <= 150
        assert "另有" in record.terminal.message
        assert planner.calls == 0
    finally:
        service.close()


def test_execution_persists_presentation_submission_and_terminal(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    events = Events()
    execution = FakeExecution()
    service = AgentService(
        planner=FakePlanner(
            PlannerOutcome(
                kind="workflow",
                user_message="准备挥手",
                summary="挥手",
                steps=(CanonicalWorkflowStep(1, "wave_hello"),),
            )
        ),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=execution,
        event_sink=events,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    try:
        request = _request()
        service.send_message(request)
        for _ in range(100):
            record = service.get_request(request.to_key())
            if record.state == "SUCCEEDED":
                break
            time.sleep(0.01)
        assert record.state == "SUCCEEDED"
        assert record.may_have_submitted
        assert record.task_ref is not None
        assert execution.calls == 1
    finally:
        service.close()


def test_unknown_execution_quarantines_robot_scope(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    execution = FakeExecution(status="unknown")
    service = AgentService(
        planner=FakePlanner(
            PlannerOutcome(
                kind="workflow",
                user_message="准备挥手",
                summary="挥手",
                steps=(CanonicalWorkflowStep(1, "wave_hello"),),
            )
        ),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=execution,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    try:
        first = _request("r1")
        service.send_message(first)
        for _ in range(100):
            if service.get_request(first.to_key()).state == "UNKNOWN":
                break
            time.sleep(0.01)
        second = service.send_message(_request("r2"))
        assert not second.accepted
        assert second.reason_code == "ROBOT_QUARANTINED"
    finally:
        service.close()


def test_clarification_reply_is_bound_and_consumed(tmp_path):
    service = AgentService(
        planner=RulePlanner(),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        clarification = _request("clarify-1", text="帮我做一个动作")
        service.send_message(clarification)
        for _ in range(100):
            if service.get_request(clarification.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        answer = _request("answer-1", text="挥手", reply_to_request_id="clarify-1")
        service.send_message(answer)
        for _ in range(100):
            if service.get_request(answer.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service.get_request(answer.to_key()).terminal.error_code == "DRY_RUN_ONLY"

        duplicate = _request("answer-2", text="挥手", reply_to_request_id="clarify-1")
        service.send_message(duplicate)
        for _ in range(100):
            if service.get_request(duplicate.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service.get_request(duplicate.to_key()).terminal.error_code == "CLARIFICATION_EXPIRED"
    finally:
        service.close()

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import AgentRequest, ExecutionResult, Presentation, TaskRef, request_hash_text
from ibrobot_agent.planner import RulePlanner
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService


def request(request_id="old", robot_scope="so101_single_arm"):
    return AgentRequest(1, request_id, "session", "test", "operator", robot_scope, "hello", datetime.now(timezone.utc))


def seed(store, value, state, *, submitted=False):
    key = value.to_key()
    record = store.admit(key, input_hash=request_hash_text(value), session_id=value.session_id)
    if state == "RECEIVED":
        return record
    record = store.begin_planning(key, expected_generation=0)
    generation = record.planning_generation
    if state == "PLANNING":
        return record
    record = store.mark_proposal_ready(key, expected_generation=generation)
    if state == "PROPOSAL_READY":
        return record
    record = store.mark_preparing(key, expected_generation=generation)
    if state == "PREPARING":
        return record
    task = TaskRef("task-" + value.request_id, "plan", "pdig", "epoch", 1, "rdig", 1)
    presentation = Presentation(
        task, 1, (CanonicalWorkflowStep(1, "wave_hello"),), "immediate_after_presentation", 30.0, "wave"
    )
    record = store.bind_task(
        key, expected_generation=generation, task_ref=task, presentation=presentation, proposal_json="{}"
    )
    if submitted:
        record = store.mark_submitted(key, expected_generation=generation, task_ref=task)
    if state == "STOPPING":
        record = store.mark_stop(key, expected_generation=generation)
    return record


class NoCatalogCalls:
    def get_status(self):
        pytest.fail("recovery must not infer completion from Gateway status")

    def get_catalog(self, status):
        pytest.fail("recovery must not plan another goal")


@pytest.mark.parametrize("state", ["RUNNING", "STOPPING"])
def test_restart_quarantines_uncertain_submission_before_admission(tmp_path, state):
    path = tmp_path / "requests.sqlite3"
    old = request()
    store = SQLiteRequestStore(path)
    previous = seed(store, old, state, submitted=True)
    store.close()
    for _ in range(2):
        store = SQLiteRequestStore(path)
        service = AgentService(planner=RulePlanner(), catalog=NoCatalogCalls(), store=store)
        try:
            recovered = service.get_request(old.to_key())
            assert recovered.state == "UNKNOWN"
            assert recovered.task_ref == previous.task_ref
            assert recovered.may_have_submitted
            assert not service.healthy
            assert not service.send_message(old).accepted
            admission = service.send_message(request("new"))
            assert not admission.accepted
            assert admission.reason_code == "ROBOT_QUARANTINED"
        finally:
            service.close()


@pytest.mark.parametrize("state", ["RECEIVED", "PLANNING", "PROPOSAL_READY", "PREPARING", "MAY_EXECUTE", "STOPPING"])
def test_restart_converges_unsubmitted_requests_without_replay(tmp_path, state):
    path = tmp_path / "requests.sqlite3"
    old = request()
    store = SQLiteRequestStore(path)
    seed(store, old, state)
    store.close()
    store = SQLiteRequestStore(path)
    service = AgentService(planner=RulePlanner(), catalog=NoCatalogCalls(), store=store)
    try:
        result = service.get_request(old.to_key())
        assert result.state == ("CANCELLED_BEFORE_EXECUTION" if state == "STOPPING" else "FAILED")
        assert result.terminal.detail["submitted"] is False
        assert service.healthy
        assert service.send_message(old).state == result.state
        assert store.recover_interrupted_requests(old.robot_scope) == 0
    finally:
        service.close()


def test_recovery_preserves_terminal_and_other_robot_records(tmp_path):
    store = SQLiteRequestStore(tmp_path / "requests.sqlite3")
    completed = request("completed")
    store.admit(completed.to_key(), input_hash=request_hash_text(completed), session_id=completed.session_id)
    terminal = ExecutionResult("succeeded", None, "", "answered", {})
    store.finish(completed.to_key(), expected_generation=0, result=terminal)
    other = request("other", "another_robot")
    seed(store, other, "RUNNING", submitted=True)
    try:
        assert store.recover_interrupted_requests(completed.robot_scope) == 0
        assert store.get_request(completed.to_key()).terminal == terminal
        assert store.get_request(other.to_key()).state == "RUNNING"
    finally:
        store.close()


def test_recovery_is_atomic_and_startup_fails_closed_on_write_error(tmp_path, monkeypatch):
    store = SQLiteRequestStore(tmp_path / "requests.sqlite3")
    first, second = request("first"), request("second")
    seed(store, first, "PLANNING")
    seed(store, second, "RUNNING", submitted=True)
    original = store._append_event_locked

    def fail_second(key, **kwargs):
        if key == second.to_key():
            raise RuntimeError("storage unavailable")
        return original(key, **kwargs)

    monkeypatch.setattr(store, "_append_event_locked", fail_second)
    try:
        with pytest.raises(RuntimeError, match="storage unavailable"):
            AgentService(planner=RulePlanner(), catalog=NoCatalogCalls(), store=store)
        assert store.get_request(first.to_key()).state == "PLANNING"
        assert store.get_request(second.to_key()).state == "RUNNING"
    finally:
        store.close()

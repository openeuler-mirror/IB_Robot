from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import rclpy
from rclpy.context import Context
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import (
    MAX_REQUEST_JSON_BYTES,
    MAX_REQUEST_TEXT_LENGTH,
    AgentRequest,
    PlannerIdentity,
    PlannerOutcome,
    PlanProposal,
    RegistryIdentity,
    RequestKey,
)
from ibrobot_agent.node import IncubationAgentNode, _RosExecutionAdapter


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
        return {"robot_name": "so101_single_arm", "skills": [], "pose_names": ["home", "zero"]}


class _ReadyFuture:
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value


class _ResultGoalHandle:
    accepted = True

    def __init__(self, result):
        self._result = result

    def get_result_async(self):
        return _ReadyFuture(self._result)


class _DirectExecutionBridge:
    def __init__(self):
        self.calls = []
        self.goal_trace_id = ""

    def get_status(self, **kwargs):
        self.calls.append("status")
        return {
            "registry_epoch": "epoch",
            "registry_generation": 1,
            "registry_digest": "digest",
            "task_budget_sec": 30.0,
        }

    def prepare_agent_plan(self, **kwargs):
        self.calls.append("prepare")
        return {
            "success": True,
            "plan": {
                "plan_token": "plan-token",
                "plan_digest": "plan-digest",
                "plan_id": "plan-id",
                "plan_kind": 1,
                "workflow_steps": [
                    {
                        "schema_version": 1,
                        "skill_name": "wave_hello",
                        "target_name": "",
                        "place_name": "",
                        "motion_direction": "",
                        "motion_distance": 0.0,
                        "timeout_sec": 0.0,
                    }
                ],
                "registry_epoch": "epoch",
                "registry_generation": 1,
                "registry_digest": "digest",
            },
        }

    def validate_agent_plan(self, **kwargs):
        self.calls.append("validate")
        return {"allowed": True, "plan_id": "plan-id", "plan_digest": "plan-digest"}

    def confirm_agent_plan(self, **kwargs):
        self.calls.append("confirm")
        return {"confirmed": True, "confirmation_token": "confirmation-token", "confirmed_task_budget_sec": 30.0}

    def wait_for_execute_plan_server(self, **kwargs):
        self.calls.append("wait_server")
        return True

    def send_agent_plan_goal(self, **kwargs):
        self.calls.append("submit")
        self.goal_trace_id = kwargs["trace_id"]
        result = type(
            "Result",
            (),
            {
                "status": 4,
                "result": type(
                    "AgentResult",
                    (),
                    {
                        "success": True,
                        "plan_id": "plan-id",
                        "plan_digest": "plan-digest",
                        "workflow_digest": "",
                        "completed_step_count": 1,
                        "error_code": "",
                        "message": "skill completed",
                        "actual_registry_epoch": "epoch",
                        "actual_registry_generation": 1,
                        "actual_registry_digest": "digest",
                    },
                )(),
            },
        )()
        return _ReadyFuture(_ResultGoalHandle(result))


def _proposal():
    request = AgentRequest(
        schema_version=1,
        request_id="request-wave",
        session_id="session-1",
        channel_id="channel-1",
        principal_id="principal-1",
        robot_scope="so101_single_arm",
        text="请挥手",
        received_at=datetime.now(timezone.utc),
    )
    return PlanProposal(
        request=request,
        planning_generation=1,
        catalog_identity=RegistryIdentity("epoch", 1, "digest"),
        outcome=PlannerOutcome(
            kind="workflow",
            user_message="准备挥手",
            summary="wave_hello",
            steps=(CanonicalWorkflowStep(1, "wave_hello"),),
        ),
        planner_identity=PlannerIdentity("rule", "rule", "local", "none", "1", "config"),
    )


def test_direct_execution_adapter_keeps_natural_language_skill_order():
    bridge = _DirectExecutionBridge()
    adapter = _RosExecutionAdapter(bridge, timeout_policy={"rpc_timeout_sec": 1.0, "task_budget_sec": 30.0})
    proposal = _proposal()
    presentations = []
    submissions = []
    result = adapter.execute(
        proposal,
        expected_registry_identity=proposal.catalog_identity,
        presentation_callback=lambda value: (presentations.append(value), bridge.calls.append("presentation")),
        submission_callback=lambda task_ref, detail: submissions.append((task_ref, detail)),
        stop_event=threading.Event(),
    )

    assert result.status == "succeeded"
    assert presentations[0].task_ref.plan_id == "plan-id"
    assert submissions[0][0].task_id
    assert bridge.goal_trace_id == "request-wave"
    assert bridge.calls == ["prepare", "presentation", "status", "confirm", "wait_server", "submit"]


@pytest.mark.parametrize("stop_stage", ["get_status", "wait_for_execute_plan_server", "submission_callback"])
def test_stop_during_pre_submission_wait_never_sends_goal(monkeypatch, stop_stage):
    bridge = _DirectExecutionBridge()
    adapter = _RosExecutionAdapter(bridge, timeout_policy={"rpc_timeout_sec": 1.0, "task_budget_sec": 30.0})
    stop_event = threading.Event()
    proposal = _proposal()
    if stop_stage != "submission_callback":
        original = getattr(bridge, stop_stage)

        def stop_after_wait(**kwargs):
            result = original(**kwargs)
            stop_event.set()
            return result

        monkeypatch.setattr(bridge, stop_stage, stop_after_wait)

    def submission_callback(*args):
        if stop_stage == "submission_callback":
            stop_event.set()

    result = adapter.execute(
        proposal,
        expected_registry_identity=proposal.catalog_identity,
        presentation_callback=lambda _: None,
        submission_callback=submission_callback,
        stop_event=stop_event,
    )
    assert result.status == "cancelled"
    assert result.detail["submitted"] is False
    assert "submit" not in bridge.calls
    if stop_stage == "get_status":
        assert "confirm" not in bridge.calls


def test_submission_callback_exception_propagates_without_sending_goal():
    bridge = _DirectExecutionBridge()
    adapter = _RosExecutionAdapter(bridge, timeout_policy={"rpc_timeout_sec": 1.0, "task_budget_sec": 30.0})
    proposal = _proposal()

    def fail_submission(*args):
        raise RuntimeError("ledger unavailable")

    with pytest.raises(RuntimeError, match="ledger unavailable"):
        adapter.execute(
            proposal,
            expected_registry_identity=proposal.catalog_identity,
            presentation_callback=lambda _: None,
            submission_callback=fail_submission,
            stop_event=threading.Event(),
        )
    assert "submit" not in bridge.calls


def test_stop_during_presentation_is_canceled_before_confirmation():
    bridge = _DirectExecutionBridge()
    adapter = _RosExecutionAdapter(bridge, timeout_policy={"rpc_timeout_sec": 1.0, "task_budget_sec": 30.0})
    proposal = _proposal()
    stop = threading.Event()

    def interrupted(_):
        stop.set()
        raise RuntimeError("stopped during rendering")

    result = adapter.execute(
        proposal,
        expected_registry_identity=proposal.catalog_identity,
        presentation_callback=interrupted,
        submission_callback=lambda *_: None,
        stop_event=stop,
    )
    assert result.status == "cancelled"
    assert "confirm" not in bridge.calls
    assert "submit" not in bridge.calls


def test_ros_presentation_receipt_unblocks_only_the_displayed_plan(tmp_path):
    from rclpy.executors import SingleThreadedExecutor

    from ibrobot_agent.planner import RulePlanner

    class Catalog(FakeCatalog):
        def get_catalog(self, status):
            return {"skills": [{"name": "wave_hello", "planner_visible": True, "semantic_level": "skill"}]}

    context = Context()
    rclpy.init(context=context)
    bridge = _DirectExecutionBridge()
    node = IncubationAgentNode(
        context=context,
        planner=RulePlanner(),
        catalog_port=Catalog(),
        execution_port=_RosExecutionAdapter(bridge, timeout_policy={"rpc_timeout_sec": 1.0, "task_budget_sec": 30.0}),
        parameter_overrides=[
            Parameter("ledger_path", value=str(tmp_path / "requests.sqlite3")),
            Parameter("conversation_path", value=str(tmp_path / "conversation.sqlite3")),
            Parameter("deployment_lock_path", value=str(tmp_path / "agent.lock")),
            Parameter("execution_enabled", value=True),
            Parameter("allowed_skills_json", value='["wave_hello"]'),
        ],
    )
    client = rclpy.create_node("presentation_test_client", context=context)
    events, responses = [], []
    subscriptions = [
        client.create_subscription(String, "/agent/event", lambda m: events.append(json.loads(m.data)), 10),
        client.create_subscription(String, "/agent/response", lambda m: responses.append(json.loads(m.data)), 10),
    ]
    control = client.create_publisher(String, "/agent/control", 10)
    requests = client.create_publisher(String, "/agent/request", 10)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    executor.add_node(client)

    def wait(predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert predicate()

    try:
        wait(lambda: requests.get_subscription_count() > 0 and control.get_subscription_count() > 0)
        requests.publish(
            String(
                data=json.dumps(
                    {"schema_version": 1, "request_id": "wire-receipt", "session_id": "session", "text": "请挥手"}
                )
            )
        )
        wait(lambda: any(e["event_type"] == "presentation" for e in events))
        event = next(e for e in events if e["event_type"] == "presentation")
        assert "confirm" not in bridge.calls and "submit" not in bridge.calls
        detail = event["detail"]
        receipt = {
            "operation": "presentation_rendered",
            "request_id": "wire-receipt",
            "request_key": event["request_key"],
            "presentation_digest": detail["presentation_digest"],
            "receipt_token": "wrong",
        }
        control.publish(String(data=json.dumps(receipt)))
        wait(lambda: any(r.get("presentation_received") is False for r in responses))
        assert "confirm" not in bridge.calls and "submit" not in bridge.calls
        receipt["receipt_token"] = detail["receipt_token"]
        control.publish(String(data=json.dumps(receipt)))
        wait(lambda: any(e.get("state") == "SUCCEEDED" for e in events))
        assert bridge.calls.count("confirm") == 1
        assert bridge.calls.count("submit") == 1
    finally:
        node.close()
        executor.shutdown()
        node.destroy_node()
        client.destroy_node()
        del subscriptions
        rclpy.shutdown(context=context)


@pytest.mark.parametrize("oversize", ["text", "json_bytes"])
def test_node_rejects_oversize_request_before_service_admission(tmp_path, monkeypatch, oversize):
    context = Context()
    rclpy.init(context=context)
    node = IncubationAgentNode(
        catalog_port=FakeCatalog(),
        context=context,
        parameter_overrides=[
            Parameter("ledger_path", value=str(tmp_path / "requests.sqlite3")),
            Parameter("conversation_path", value=str(tmp_path / "conversation.sqlite3")),
            Parameter("deployment_lock_path", value=str(tmp_path / "agent.lock")),
        ],
    )
    responses = []
    monkeypatch.setattr(node, "_publish_response", responses.append)
    monkeypatch.setattr(node._service, "send_message", lambda _: pytest.fail("oversize request was admitted"))
    payload = replace(_proposal().request, text="hello").to_dict()
    if oversize == "text":
        payload["text"] = "挥" * (MAX_REQUEST_TEXT_LENGTH + 1)
    else:
        payload["padding"] = "挥" * (MAX_REQUEST_JSON_BYTES // 3)
    message = String(data=json.dumps(payload, ensure_ascii=False))
    try:
        node._request_callback(message)
        assert responses[0]["accepted"] is False
        assert responses[0]["reason_code"] == "REQUEST_SCHEMA_INVALID"
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown(context=context)


def test_node_binds_runtime_identity_and_answers_read_only(tmp_path):
    context = Context()
    rclpy.init(context=context)
    node = IncubationAgentNode(
        catalog_port=FakeCatalog(),
        context=context,
        parameter_overrides=[
            Parameter("ledger_path", value=str(tmp_path / "requests.sqlite3")),
            Parameter("conversation_path", value=str(tmp_path / "conversation.sqlite3")),
            Parameter("deployment_lock_path", value=str(tmp_path / "agent.lock")),
            Parameter("robot_scope", value="so101_single_arm"),
            Parameter("channel_id", value="bound-channel"),
            Parameter("principal_id", value="bound-principal"),
        ],
    )
    try:
        ready = node._ready_callback(Trigger.Request(), Trigger.Response())
        assert ready.success

        node._service._quarantined = True
        ready = node._ready_callback(Trigger.Request(), Trigger.Response())
        assert not ready.success
        node._service._quarantined = False

        message = String()
        message.data = json.dumps(
            {
                "schema_version": 1,
                "request_id": "node-request-1",
                "session_id": "node-session-1",
                "channel_id": "spoofed-channel",
                "principal_id": "spoofed-principal",
                "robot_scope": "spoofed-robot",
                "text": "当前状态",
            }
        )
        node._request_callback(message)
        key = RequestKey("so101_single_arm", "bound-channel", "bound-principal", "node-request-1")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and node._service.get_request(key).state != "ANSWERED":
            time.sleep(0.01)
        record = node._service.get_request(key)
        assert record.state == "ANSWERED"
        assert "运动未授权" in record.terminal.message
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown(context=context)

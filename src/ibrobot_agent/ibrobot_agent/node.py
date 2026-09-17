"""ROS 2 composition root for the incubating Agent service.

The node owns transport and lifecycle only.  AgentService remains ROS-free and
is intentionally injected with ports so the same core can later be published
from an independent repository.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from embodied_common.agent_terminal_contract import classify_agent_terminal, stable_agent_execution_error_code
from embodied_common.canon import sha256_text, to_canonical_json
from embodied_common.tracing import create_trace_logger, trace_scope, trace_stage
from embodied_common.workflow_contracts import normalize_workflow_steps
from ibrobot_agent.contracts import (
    MAX_REQUEST_JSON_BYTES,
    AgentRequest,
    EventSink,
    ExecutionResult,
    Presentation,
    RequestKey,
    TaskRef,
)
from ibrobot_agent.conversation_store import SQLiteConversationStore
from ibrobot_agent.deployment_lock import DeploymentLock
from ibrobot_agent.planner import PlannerAdapter, RulePlanner, build_planner_messages
from ibrobot_agent.presentation import PresentationGate
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService

_trace = create_trace_logger("ib_trace.agent")


class _RosEventSink(EventSink):
    def __init__(self, publisher) -> None:
        self._publisher = publisher
        self._lock = threading.Lock()

    def publish(self, event) -> None:
        message = String()
        message.data = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._publisher.publish(message)


class _UnavailableCatalog:
    def get_status(self):
        raise RuntimeError("Agent catalog adapter is not configured")

    def get_catalog(self, status):
        raise RuntimeError("Agent catalog adapter is not configured")


class _RosBridgeCatalog:
    """Expose the existing Gateway read path through the core catalog port."""

    def __init__(self, bridge, *, rpc_timeout_sec: float) -> None:
        self._bridge = bridge
        self._rpc_timeout_sec = rpc_timeout_sec

    def get_status(self):
        return self._bridge.get_status(task_id="", payload_hash="", timeout_sec=self._rpc_timeout_sec)

    def get_catalog(self, status):
        from robot_skill_cli.catalog import capability_view_from_snapshot

        snapshot = self._bridge.get_skill_snapshot(
            registry_epoch=status["registry_epoch"],
            generation=status["registry_generation"],
            timeout_sec=status.get("rpc_timeout_sec", self._rpc_timeout_sec),
        )
        view = capability_view_from_snapshot(snapshot, status)
        view["control_plane_ready"] = bool(status.get("control_plane_ready"))
        view["active_control_mode"] = str(status.get("active_control_mode", ""))
        view["motion_authorized"] = bool(status.get("motion_authorized"))
        view["registry_epoch"] = str(status.get("registry_epoch", ""))
        view["registry_generation"] = int(status.get("registry_generation", 0))
        view["registry_digest"] = str(status.get("registry_digest", ""))
        return view


class _RosExecutionAdapter:
    """Execute typed Agent plans without duplicating the CLI controller state machine."""

    def __init__(self, bridge, *, timeout_policy: dict[str, float]) -> None:
        self._bridge = bridge
        self._rpc_timeout_sec = max(float(timeout_policy.get("rpc_timeout_sec", 5.0)), 0.1)
        self._task_budget_sec = float(timeout_policy.get("task_budget_sec", 120.0))
        self._active_lock = threading.Lock()
        self._active_stop_event: threading.Event | None = None
        self._active_task_id: str | None = None

    def execute(self, proposal, *, expected_registry_identity, presentation_callback, submission_callback, stop_event):
        task_id = uuid.uuid4().hex
        with self._active_lock:
            self._active_stop_event = stop_event
            self._active_task_id = task_id
        try:
            return self._execute_direct(
                proposal,
                task_id=task_id,
                expected_registry_identity=expected_registry_identity,
                presentation_callback=presentation_callback,
                submission_callback=submission_callback,
                stop_event=stop_event,
            )
        finally:
            with self._active_lock:
                self._active_stop_event = None
                self._active_task_id = None

    def request_stop(self, request_key) -> None:
        with self._active_lock:
            if self._active_stop_event is not None:
                self._active_stop_event.set()

    def _execute_direct(
        self,
        proposal,
        *,
        task_id: str,
        expected_registry_identity,
        presentation_callback,
        submission_callback,
        stop_event: threading.Event,
    ) -> ExecutionResult:
        expected_identity = (
            expected_registry_identity.epoch,
            expected_registry_identity.generation,
            expected_registry_identity.digest,
        )
        if stop_event.is_set():
            return ExecutionResult(
                "cancelled", None, "SKILL_CANCELLED", "stopped before planning", {"submitted": False}
            )

        prepare_agent_plan = getattr(self._bridge, "prepare_agent_plan", None)
        if prepare_agent_plan is None:
            prepare_agent_plan = self._legacy_prepare_agent_plan
        try:
            plan_response = prepare_agent_plan(
                request_id=proposal.request.request_id,
                raw_command=proposal.request.text,
                workflow_steps=[step.to_dict() for step in proposal.outcome.steps],
                timeout_sec=self._rpc_timeout_sec,
                execution_mode="immediate_after_presentation",
                trace_id=proposal.request.request_id,
            )
        except Exception as exc:
            return self._failed_result("CAPABILITY_NOT_READY", f"agent plan preparation failed: {exc}")
        if not plan_response.get("success") or not plan_response.get("allowed", True):
            return self._failed_result(
                str(plan_response.get("error_code") or "SKILL_SCHEMA_INVALID"),
                str(plan_response.get("message") or "agent plan preparation failed"),
            )

        plan = plan_response["plan"]
        try:
            planned_steps = tuple(normalize_workflow_steps(plan.get("workflow_steps", [])))
        except (TypeError, ValueError) as exc:
            return self._failed_result("SKILL_SCHEMA_INVALID", str(exc))
        if planned_steps != tuple(proposal.outcome.steps):
            return self._failed_result("SKILL_SNAPSHOT_DIGEST_MISMATCH", "planned workflow differs from proposal")
        plan_identity = (
            str(plan.get("registry_epoch", "")),
            int(plan.get("registry_generation", 0)),
            str(plan.get("registry_digest", "")),
        )
        if plan_identity != expected_identity:
            return self._failed_result("SKILL_SNAPSHOT_DIGEST_MISMATCH", "planned workflow uses a different registry")

        presentation = _presentation_from_mapping(
            {
                "task_id": task_id,
                "plan_id": plan["plan_id"],
                "plan_digest": plan["plan_digest"],
                "plan_kind": int(plan.get("plan_kind", 0)),
                "steps": [step.to_dict() for step in planned_steps],
                "execution_mode": "immediate_after_presentation",
                "proposed_task_budget_sec": self._task_budget_sec,
                "summary": proposal.outcome.summary,
            },
            proposal,
        )
        try:
            presentation_callback(presentation)
        except Exception as exc:
            if stop_event.is_set():
                return ExecutionResult(
                    "cancelled", None, "SKILL_CANCELLED", "stopped before presentation completed", {"submitted": False}
                )
            return self._failed_result("PRESENTATION_FAILED", f"plan presentation failed: {exc}", presentation)
        if stop_event.is_set():
            return ExecutionResult(
                "cancelled", None, "SKILL_CANCELLED", "stopped after presentation", {"submitted": False}
            )

        try:
            status = self._bridge.get_status(task_id="", payload_hash="", timeout_sec=self._rpc_timeout_sec)
            if stop_event.is_set():
                return ExecutionResult(
                    "cancelled", None, "SKILL_CANCELLED", "stopped before confirmation", {"submitted": False}
                )
            task_budget_sec = min(float(status["task_budget_sec"]), self._task_budget_sec)
            confirmation = self._bridge.confirm_agent_plan(
                plan_token=plan["plan_token"],
                plan_digest=plan["plan_digest"],
                task_id=task_id,
                status=status,
                task_budget_sec=task_budget_sec,
                timeout_sec=self._rpc_timeout_sec,
                execution_mode="immediate_after_presentation",
            )
        except Exception:
            return self._unknown_result("confirmation response was unavailable", presentation)
        if not confirmation.get("confirmed"):
            return self._failed_result(
                str(confirmation.get("error_code") or "SKILL_REQUEST_ID_CONFLICT"),
                str(confirmation.get("message") or "plan confirmation failed"),
                presentation,
            )
        if stop_event.is_set():
            return ExecutionResult(
                "cancelled", None, "SKILL_CANCELLED", "stopped before goal submission", {"submitted": False}
            )
        if not self._bridge.wait_for_execute_plan_server(timeout_sec=self._rpc_timeout_sec):
            return self._failed_result("CAPABILITY_NOT_READY", "agent plan action server unavailable", presentation)

        if stop_event.is_set():
            return ExecutionResult(
                "cancelled", None, "SKILL_CANCELLED", "stopped before submission persistence", {"submitted": False}
            )
        # Let AgentService inspect durable submission state if persistence raises.
        submission_callback(
            presentation.task_ref,
            {
                "confirmed": True,
                "task_id": task_id,
                "plan_id": presentation.task_ref.plan_id,
                "plan_digest": presentation.task_ref.plan_digest,
                "registry_epoch": presentation.task_ref.registry_epoch,
                "registry_generation": presentation.task_ref.registry_generation,
                "registry_digest": presentation.task_ref.registry_digest,
                "expected_step_count": presentation.task_ref.expected_step_count,
            },
        )
        if stop_event.is_set():
            return ExecutionResult(
                "cancelled", None, "SKILL_CANCELLED", "stopped before goal submission", {"submitted": False}
            )
        try:
            goal_future = self._bridge.send_agent_plan_goal(
                plan_token=plan["plan_token"],
                confirmation_token=confirmation["confirmation_token"],
                task_id=task_id,
                timeout_sec=float(confirmation.get("confirmed_task_budget_sec", task_budget_sec)),
                trace_id=proposal.request.request_id,
                feedback_callback=None,
            )
        except Exception as exc:
            return self._converge_after_submission(task_id, presentation, f"goal submission failed: {exc}")

        if not self._wait_future(goal_future, self._rpc_timeout_sec, stop_event):
            return self._converge_after_submission(task_id, presentation, "goal acceptance was not confirmed")
        try:
            goal_handle = goal_future.result()
        except Exception as exc:
            return self._converge_after_submission(task_id, presentation, f"goal response unavailable: {exc}")
        if goal_handle is None or not goal_handle.accepted:
            if stop_event.is_set():
                return self._converge_after_submission(task_id, presentation, "goal was rejected during stop")
            return self._failed_result("GOAL_REJECTED", "agent plan goal was rejected", presentation)

        try:
            result_future = goal_handle.get_result_async()
        except Exception as exc:
            return self._converge_after_submission(task_id, presentation, f"result request unavailable: {exc}")
        timeout_sec = float(confirmation.get("confirmed_task_budget_sec", task_budget_sec))
        deadline = time.monotonic() + timeout_sec + self._rpc_timeout_sec
        while not result_future.done() and time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.02)
        if not result_future.done() or stop_event.is_set():
            return self._converge_after_submission(
                task_id,
                presentation,
                "execution was stopped" if stop_event.is_set() else "agent plan result timed out",
            )
        try:
            wrapped_result = result_future.result()
            terminal = {
                "status": int(wrapped_result.status),
                "result": _agent_plan_result_from_message(wrapped_result.result),
            }
        except Exception as exc:
            return self._converge_after_submission(task_id, presentation, f"terminal result unavailable: {exc}")
        return self._execution_result_from_terminal(terminal, presentation)

    def _legacy_prepare_agent_plan(self, **kwargs):
        """Compatibility path for test bridges and pre-prepare ROS bridges."""
        response = self._bridge.plan_agent_command(
            request_id=kwargs["request_id"],
            raw_command=kwargs["raw_command"],
            workflow_steps=kwargs["workflow_steps"],
            timeout_sec=kwargs["timeout_sec"],
            execution_mode=kwargs["execution_mode"],
        )
        if response.get("success"):
            response["allowed"] = True
        return response

    @staticmethod
    def _wait_future(future, timeout_sec: float, stop_event: threading.Event) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.01)
        return future.done()

    def _converge_after_submission(self, task_id: str, presentation: Presentation, detail: str) -> ExecutionResult:
        with contextlib.suppress(Exception):
            self._bridge.cancel_agent_plan(task_id, timeout_sec=self._rpc_timeout_sec)
        deadline = time.monotonic() + self._rpc_timeout_sec
        while time.monotonic() < deadline:
            try:
                terminal = self._bridge.get_agent_plan_result(task_id, timeout_sec=self._rpc_timeout_sec)
                if int(terminal.get("status", 0)) in {4, 5, 6}:
                    return self._execution_result_from_terminal(terminal, presentation)
            except Exception:
                pass
            time.sleep(0.02)
        return self._unknown_result(detail, presentation)

    @staticmethod
    def _execution_result_from_terminal(terminal: dict, presentation: Presentation) -> ExecutionResult:
        result = terminal.get("result", {}) or {}
        expectation = {
            "plan_id": presentation.task_ref.plan_id,
            "plan_digest": presentation.task_ref.plan_digest,
            "registry_epoch": presentation.task_ref.registry_epoch,
            "registry_generation": presentation.task_ref.registry_generation,
            "registry_digest": presentation.task_ref.registry_digest,
            "step_count": presentation.task_ref.expected_step_count,
        }
        classification = classify_agent_terminal(terminal.get("status", 0), result, expectation)
        if classification == "succeeded":
            return ExecutionResult("succeeded", presentation.task_ref, "", str(result.get("message", "")), result)
        if classification == "stopped":
            return ExecutionResult(
                "cancelled", presentation.task_ref, "SKILL_CANCELLED", str(result.get("message", "")), result
            )
        if classification == "failed":
            return ExecutionResult(
                "failed",
                presentation.task_ref,
                stable_agent_execution_error_code(str(result.get("error_code", ""))),
                str(result.get("message", "")),
                result,
            )
        return ExecutionResult(
            "unknown",
            presentation.task_ref,
            str(result.get("error_code") or "SKILL_CANCEL_TIMEOUT"),
            str(result.get("message") or "robot stop state is unknown"),
            result,
        )

    @staticmethod
    def _failed_result(error_code: str, message: str, presentation: Presentation | None = None) -> ExecutionResult:
        return ExecutionResult(
            "failed", presentation.task_ref if presentation else None, error_code[:64], message[:300], {}
        )

    @staticmethod
    def _unknown_result(message: str, presentation: Presentation | None = None) -> ExecutionResult:
        return ExecutionResult(
            "unknown",
            presentation.task_ref if presentation else None,
            "SKILL_CANCEL_TIMEOUT",
            message[:300],
            {},
        )


def _presentation_from_mapping(value: dict, proposal) -> Presentation:
    identity = proposal.catalog_identity
    steps = tuple(normalize_workflow_steps(value.get("steps", [])))
    task_ref = TaskRef(
        task_id=str(value["task_id"]),
        plan_id=str(value["plan_id"]),
        plan_digest=str(value["plan_digest"]),
        registry_epoch=identity.epoch,
        registry_generation=identity.generation,
        registry_digest=identity.digest,
        expected_step_count=len(steps),
    )
    return Presentation(
        task_ref=task_ref,
        plan_kind=int(value["plan_kind"]),
        steps=steps,
        execution_mode=str(value.get("execution_mode", "immediate_after_presentation")),
        proposed_task_budget_sec=float(value["proposed_task_budget_sec"]),
        summary=str(value.get("summary") or "；".join(step.skill_name for step in steps)),
    )


def _agent_plan_result_from_message(result) -> dict:
    return {
        "success": bool(result.success),
        "plan_id": str(result.plan_id),
        "plan_digest": str(result.plan_digest),
        "workflow_digest": str(result.workflow_digest),
        "completed_step_count": int(result.completed_step_count),
        "error_code": str(result.error_code),
        "message": str(result.message),
        "actual_registry_epoch": str(result.actual_registry_epoch),
        "actual_registry_generation": int(result.actual_registry_generation),
        "actual_registry_digest": str(result.actual_registry_digest),
    }


class IncubationAgentNode(Node):
    """Minimal ROS node; production adapters are supplied in IB-Robot."""

    def __init__(
        self,
        *,
        planner=None,
        bridge_factory: Callable[..., object] | None = None,
        catalog_port=None,
        execution_port=None,
        parameter_overrides=None,
        context=None,
    ) -> None:
        super().__init__("ibrobot_agent_node", parameter_overrides=parameter_overrides, context=context)
        self.declare_parameter("request_topic", "/agent/request")
        self.declare_parameter("event_topic", "/agent/event")
        self.declare_parameter("response_topic", "/agent/response")
        self.declare_parameter("control_topic", "/agent/control")
        self.declare_parameter("ledger_path", "")
        self.declare_parameter("conversation_path", "")
        self.declare_parameter("deployment_lock_path", "")
        self.declare_parameter("execution_enabled", False)
        self.declare_parameter("max_session_turns", 12)
        self.declare_parameter("clarification_ttl_sec", 300.0)
        self.declare_parameter("event_queue_size", 128)
        self.declare_parameter("presentation_timeout_sec", 30.0)
        self.declare_parameter("robot_scope", "so101_single_arm")
        self.declare_parameter("channel_id", "agent_incubation")
        self.declare_parameter("principal_id", "local_operator")
        self.declare_parameter("allowed_skills_json", "[]")
        self.declare_parameter("simulation_mode", False)
        self.declare_parameter("rpc_timeout_sec", 5.0)
        self.declare_parameter("task_budget_sec", 120.0)
        self.declare_parameter("planner_config_json", '{"mode":"rule"}')
        self.declare_parameter("gateway_status_service", "/embodied/get_skill_gateway_status")
        self.declare_parameter("gateway_validate_skill_service", "/embodied/validate_skill")
        self.declare_parameter("gateway_skill_action", "/embodied/execute_skill")
        self.declare_parameter("gateway_plan_service", "/embodied/plan_agent_command")
        self.declare_parameter("gateway_prepare_plan_service", "/embodied/prepare_agent_plan")
        self.declare_parameter("gateway_validate_plan_service", "/embodied/validate_agent_plan")
        self.declare_parameter("gateway_confirm_plan_service", "/embodied/confirm_agent_plan")
        self.declare_parameter("gateway_execute_plan_action", "/embodied/execute_agent_plan")
        ledger_path = str(self.get_parameter("ledger_path").value).strip()
        if not ledger_path:
            ledger_path = str(Path.home() / ".ros" / "ibrobot_agent" / "requests.sqlite3")
        conversation_path = str(self.get_parameter("conversation_path").value).strip()
        if not conversation_path:
            conversation_path = str(Path.home() / ".ros" / "ibrobot_agent" / "conversation.sqlite3")
        deployment_lock_path = str(self.get_parameter("deployment_lock_path").value).strip()
        if not deployment_lock_path:
            deployment_lock_path = str(Path.home() / ".ros" / "ibrobot_agent" / "agent.lock")
        self._deployment_lock = DeploymentLock(deployment_lock_path)
        event_queue_size = int(self.get_parameter("event_queue_size").value)
        event_publisher = self.create_publisher(String, str(self.get_parameter("event_topic").value), event_queue_size)
        self._response_publisher = self.create_publisher(
            String, str(self.get_parameter("response_topic").value), event_queue_size
        )
        self._store = SQLiteRequestStore(ledger_path)
        self._presentation_gate = PresentationGate(float(self.get_parameter("presentation_timeout_sec").value))
        self._bridge = None
        catalog = catalog_port or _UnavailableCatalog()
        execution = execution_port
        active_planner = planner or _planner_from_config(str(self.get_parameter("planner_config_json").value))
        self._injected_ready = catalog_port is not None
        self._gateway_ready = self._injected_ready
        if catalog_port is None:
            try:
                if bridge_factory is None:
                    from robot_skill_cli.ros_bridge import RosBridge

                    bridge_factory = RosBridge

                self._bridge = bridge_factory(
                    status_service=str(self.get_parameter("gateway_status_service").value),
                    validate_skill_service=str(self.get_parameter("gateway_validate_skill_service").value),
                    skill_action=str(self.get_parameter("gateway_skill_action").value),
                    plan_service=str(self.get_parameter("gateway_plan_service").value),
                    prepare_plan_service=str(self.get_parameter("gateway_prepare_plan_service").value),
                    validate_plan_service=str(self.get_parameter("gateway_validate_plan_service").value),
                    confirm_plan_service=str(self.get_parameter("gateway_confirm_plan_service").value),
                    execute_plan_action=str(self.get_parameter("gateway_execute_plan_action").value),
                )
                if self._bridge.start():
                    rpc_timeout_sec = float(self.get_parameter("rpc_timeout_sec").value)
                    catalog = _RosBridgeCatalog(self._bridge, rpc_timeout_sec=rpc_timeout_sec)
                    # Refresh asynchronously after all Pipeline nodes have
                    # had a chance to create their service/action clients.
                    self._gateway_ready = False
                    if bool(self.get_parameter("execution_enabled").value):
                        if isinstance(active_planner, RulePlanner) and not bool(
                            self.get_parameter("simulation_mode").value
                        ):
                            raise ValueError("RulePlanner execution is restricted to simulation mode")
                        execution = _RosExecutionAdapter(
                            self._bridge,
                            timeout_policy={
                                "rpc_timeout_sec": rpc_timeout_sec,
                                "task_budget_sec": float(self.get_parameter("task_budget_sec").value),
                            },
                        )
            except Exception as exc:
                self.get_logger().error(f"Gateway adapter unavailable: {exc}")
        self._planner_ready = active_planner is not None
        try:
            decoded_allowlist = json.loads(str(self.get_parameter("allowed_skills_json").value))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("allowed_skills_json must be a JSON array") from exc
        if not isinstance(decoded_allowlist, list) or any(
            not isinstance(item, str) or not item.strip() for item in decoded_allowlist
        ):
            raise ValueError("allowed_skills_json must be a JSON array of non-empty strings")
        allowed_skills = set(decoded_allowlist)
        self._service = AgentService(
            planner=active_planner,
            catalog=catalog,
            store=self._store,
            execution=execution,
            presentation=self._presentation_gate,
            event_sink=_RosEventSink(event_publisher),
            execution_enabled=bool(self.get_parameter("execution_enabled").value),
            allowed_skills=allowed_skills,
            conversation=SQLiteConversationStore(
                conversation_path,
                max_turns=int(self.get_parameter("max_session_turns").value),
                clarification_ttl_sec=float(self.get_parameter("clarification_ttl_sec").value),
            ),
            robot_scope=str(self.get_parameter("robot_scope").value),
        )
        self._request_subscription = self.create_subscription(
            String,
            str(self.get_parameter("request_topic").value),
            self._request_callback,
            10,
        )
        self._control_subscription = self.create_subscription(
            String,
            str(self.get_parameter("control_topic").value),
            self._control_callback,
            10,
        )
        self._health_service = self.create_service(Trigger, "~/health", self._health_callback)
        self._ready_service = self.create_service(Trigger, "~/ready", self._ready_callback)
        self._ready_timer = self.create_timer(1.0, self._refresh_gateway_ready)
        self.get_logger().info(
            f"ibrobot_agent_node started; gateway_ready={self._gateway_ready}; "
            f"execution_enabled={bool(self.get_parameter('execution_enabled').value)}"
        )

    @property
    def agent_service(self):
        return self._service

    def _request_callback(self, message: String) -> None:
        request_id = ""
        try:
            if len(message.data) > MAX_REQUEST_JSON_BYTES or len(message.data.encode("utf-8")) > MAX_REQUEST_JSON_BYTES:
                raise ValueError(f"request JSON must be at most {MAX_REQUEST_JSON_BYTES} UTF-8 bytes")
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            request_id = str(payload.get("request_id", ""))
            request = AgentRequest(
                schema_version=payload["schema_version"],
                request_id=payload["request_id"],
                session_id=payload["session_id"],
                channel_id=str(self.get_parameter("channel_id").value),
                principal_id=str(self.get_parameter("principal_id").value),
                robot_scope=str(self.get_parameter("robot_scope").value),
                text=payload["text"],
                received_at=datetime.now(timezone.utc),
                reply_to_request_id=payload.get("reply_to_request_id"),
            )
            with trace_scope(request.request_id), trace_stage(_trace, "agent.admit", request_id=request.request_id):
                response = self._service.send_message(request)
            self._publish_response(response.__dict__)
            self.get_logger().info(json.dumps(response.__dict__, sort_keys=True))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._publish_response(
                {
                    "accepted": False,
                    "request_id": request_id,
                    "state": "FAILED",
                    "reason_code": "REQUEST_SCHEMA_INVALID",
                    "message": str(exc),
                }
            )
            self.get_logger().error(f"invalid Agent request: {exc}")

    def _control_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict) or payload.get("operation") not in {"stop", "presentation_rendered"}:
                raise ValueError("control request must be stop or presentation_rendered")
            key = RequestKey(
                robot_scope=str(self.get_parameter("robot_scope").value),
                channel_id=str(self.get_parameter("channel_id").value),
                principal_id=str(self.get_parameter("principal_id").value),
                request_id=payload["request_id"],
            )
            if payload["operation"] == "presentation_rendered":
                if payload.get("request_key") != key.to_dict():
                    raise ValueError("presentation receipt identity does not match this Agent")
                accepted = self._presentation_gate.acknowledge(
                    key,
                    digest=payload["presentation_digest"],
                    token=payload["receipt_token"],
                )
                self._publish_response({"request_id": key.request_id, "presentation_received": accepted})
                return
            receipt = self._service.stop_request(key)
            self._publish_response(
                {
                    "requested": receipt.requested,
                    "request_id": receipt.record.key.request_id,
                    "state": receipt.record.state,
                    "task_ref": None if receipt.record.task_ref is None else receipt.record.task_ref.to_dict(),
                    "reason_code": receipt.reason_code,
                    "message": receipt.message,
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._publish_response(
                {
                    "requested": False,
                    "reason_code": "REQUEST_SCHEMA_INVALID",
                    "message": str(exc),
                }
            )
            self.get_logger().error(f"invalid Agent control request: {exc}")

    def _publish_response(self, payload: dict) -> None:
        response = String()
        response.data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._response_publisher.publish(response)

    def _health_callback(self, _request, response):
        response.success = True
        response.message = "ibrobot_agent_node is running"
        return response

    def _ready_callback(self, _request, response):
        response.success = self._gateway_ready and self._planner_ready and self._service.healthy
        response.message = (
            "Gateway adapter is ready"
            if response.success
            else "Gateway adapter, planner, or Agent ledger is unavailable"
        )
        return response

    def _refresh_gateway_ready(self) -> None:
        if self._injected_ready:
            self._gateway_ready = True
            return
        if self._bridge is None:
            self._gateway_ready = False
            return
        try:
            if not self._bridge.wait_for_agent_plan_interfaces(timeout_sec=0.1, include_prepare=True):
                self._gateway_ready = False
                return
            status = self._bridge.get_status(task_id="", payload_hash="", timeout_sec=0.2)
            self._gateway_ready = bool(
                status.get("control_plane_ready")
                and status.get("registry_epoch")
                and int(status.get("registry_generation", 0)) >= 0
                and status.get("registry_digest")
            )
        except Exception:
            self._gateway_ready = False

    def close(self) -> None:
        try:
            try:
                self._service.close()
            finally:
                if self._bridge is not None:
                    self._bridge.close()
        finally:
            self._deployment_lock.close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = IncubationAgentNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


__all__ = ["IncubationAgentNode", "main"]


def _planner_from_config(raw: str):
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("planner_config_json must contain a JSON object") from exc
    if not isinstance(config, dict):
        raise ValueError("planner_config_json must contain a JSON object")
    mode = config.get("mode", "rule")
    if mode == "rule":
        return RulePlanner()
    if mode != "vlm":
        raise ValueError("Agent planner mode must be rule or vlm")
    if str(config.get("api_key", "")).strip():
        raise ValueError("VLM Agent planner must not carry a literal api_key; use api_key_env")
    required = ("provider", "base_url", "model")
    if any(not isinstance(config.get(field), str) or not config[field].strip() for field in required):
        raise ValueError("VLM Agent planner requires provider, base_url and model")
    from embodied_common.vlm_api_client import VLMAPIClient

    client = VLMAPIClient(
        provider=config["provider"],
        base_url=config["base_url"],
        api_key_env=str(config.get("api_key_env", "")),
        model=config["model"],
        timeout_sec=float(config.get("timeout_sec", 120.0)),
        temperature=float(config.get("temperature", 0.0)),
        multimodal=False,
        api_protocol=str(config.get("api_protocol", "chat_completions")),
    )
    from ibrobot_agent.contracts import PlannerIdentity

    return PlannerAdapter(
        client,
        prompt_builder=build_planner_messages,
        planner_identity=PlannerIdentity(
            route="vlm",
            returned_model=config["model"],
            protocol=str(config.get("api_protocol", "chat_completions")),
            prompt_version="agent-planner-v1",
            schema_version="1",
            config_digest=sha256_text(to_canonical_json(config)),
        ),
    )

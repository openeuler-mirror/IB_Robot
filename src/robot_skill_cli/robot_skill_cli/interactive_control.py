"""Interactive closed-loop controller for the IB-Robot Capability Gateway.

This module is a pure-Python, ROS-free reference implementation of the interactive
closed loop (catalog discovery, out-of-catalog rejection, in-process workflow
prepare/execute, ``stop`` to a definite terminal, and safe fresh-state continuation).
It depends only on an injected ``RosBridge``-like object and the shared
catalog/workflow helpers, so it is unit-testable without a ROS stack.

The controller never parses free natural language into structured steps; it only
recognizes a closed vocabulary (confirm / stop / continue) and drives the existing
Gateway primitives. Structured ``workflow_steps`` remain the caller's responsibility.
"""

from __future__ import annotations

import contextlib
import struct
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from embodied_common.agent_execution_contract import (
    INTERACTIVE_CONFIRMATION,
    validate_agent_execution_mode,
)
from embodied_common.agent_terminal_contract import GOAL_CANCELED, TERMINAL_GOAL_STATUSES, classify_agent_terminal
from embodied_common.workflow_contracts import normalize_workflow_steps

# Closed natural-language grammars. Motion-enabling intents require an exact match
# after normalization; unknown text never acts.
_CONFIRM_PHRASES = (
    "确认执行当前计划",
    "确认然后挥手",
    "确认一下计划内容",
    "确认执行",
    "确认",
    "执行吧",
    "好的",
    "好",
    "可以",
    "是的",
    "confirm",
)
_STOP_PHRASES = ("别动了", "别动", "停止", "停下", "停", "stop", "halt")
_CONTINUE_PHRASES = ("继续吧", "继续", "接着来", "go on", "continue")

INTENT_CONFIRM = "confirm"
INTENT_STOP = "stop"
INTENT_CONTINUE = "continue"
INTENT_UNKNOWN = "unknown"

# Controller states.
IDLE = "idle"
STARTING = "starting"
DISCOVERED = "discovered"
PREPARED = "prepared"
CONFIRMED = "confirmed"
EXECUTING = "executing"
STOPPING = "stopping"
STOPPED = "stopped"
SUCCEEDED = "succeeded"
FAILED = "failed"
UNKNOWN = "unknown"

_RPC_TIMEOUT_FLOOR_SEC = 30.0
_STATUS_PREFLIGHT_FLOOR_SEC = 15.0
_CANCEL_POLL_INTERVAL_SEC = 0.02
_EXECUTE_POLL_INTERVAL_SEC = 0.02


def _float32(value: Any) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _default_view_resolver(snapshot: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Verify a Gateway snapshot and return its public catalog view.

    Imported lazily so this module stays ROS-free at import time; in production the
    bridge is already started (rclpy + ibrobot_msgs available) when this runs.
    """
    from robot_skill_cli.catalog import capability_view_from_snapshot

    return capability_view_from_snapshot(snapshot, status)


class InteractiveControlError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class OutOfCatalogError(InteractiveControlError):
    pass


class NotConfirmedError(InteractiveControlError):
    pass


class UnknownStopError(InteractiveControlError):
    pass


class IllegalStateError(InteractiveControlError):
    pass


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def classify(text: str) -> str:
    """Classify a free-form string against the closed confirm/stop/continue grammar."""
    normalized = _normalize_text(text)
    if not normalized:
        return INTENT_UNKNOWN
    if normalized in _STOP_PHRASES:
        return INTENT_STOP
    if normalized in _CONTINUE_PHRASES:
        return INTENT_CONTINUE
    if normalized in _CONFIRM_PHRASES:
        return INTENT_CONFIRM
    return INTENT_UNKNOWN


def planner_visible_names(view: dict[str, Any]) -> set[str]:
    return {skill["name"] for skill in view.get("skills", []) if skill.get("planner_visible") is True}


class InteractiveController:
    """Drive the five-feature closed loop over an injected Gateway bridge.

    The bridge is duck-typed: it must expose the ``RosBridge`` methods this
    controller calls (``get_status``, ``get_skill_snapshot``, ``plan_agent_command``,
    ``validate_agent_plan``, ``confirm_agent_plan``, ``wait_for_execute_plan_server``,
    ``send_agent_plan_goal``, ``wait_future``, ``cancel_agent_plan``,
    ``get_agent_plan_result``). The module itself never imports ``rclpy``.
    """

    def __init__(
        self,
        bridge: Any,
        *,
        timeout_policy: dict[str, Any],
        id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        view_resolver: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
        execution_mode: str = INTERACTIVE_CONFIRMATION,
    ) -> None:
        self._bridge = bridge
        self._rpc_timeout_sec = max(
            float(timeout_policy.get("rpc_timeout_sec", _RPC_TIMEOUT_FLOOR_SEC)), _RPC_TIMEOUT_FLOOR_SEC
        )
        self._status_timeout_sec = max(self._rpc_timeout_sec, _STATUS_PREFLIGHT_FLOOR_SEC)
        self._id_factory = id_factory
        self._monotonic = monotonic
        self._sleep = sleep
        self._resolve_view = view_resolver or _default_view_resolver
        self._execution_mode = validate_agent_execution_mode(execution_mode)
        self._fresh_status: dict[str, Any] | None = None
        self._fresh_view: dict[str, Any] | None = None
        self._fresh_identity: tuple[str, int, str] | None = None
        self._pending: dict[str, Any] | None = None
        self._confirmed: dict[str, Any] | None = None
        self._terminal: dict[str, Any] | None = None
        self._state = IDLE
        self._active_task_id: str | None = None
        self._stop_event: threading.Event | None = None
        self._stop_requested = threading.Event()
        self._state_lock = threading.RLock()
        self._cancel_started = False
        self._cancel_accepted = False
        self._goal_future = None
        self._goal_handle = None
        self._result_future = None
        self._submission_started = False

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def _rpc(self) -> float:
        return self._rpc_timeout_sec

    def discover(self) -> dict[str, Any]:
        """Feature 1: read-only query of the current runtime catalog and identity."""
        status = self._bridge.get_status(task_id="", payload_hash="", timeout_sec=self._status_timeout_sec)
        if not status.get("control_plane_ready"):
            raise InteractiveControlError("SERVER_UNAVAILABLE", "Gateway control plane is not ready")
        identity = (
            status.get("registry_epoch", ""),
            int(status.get("registry_generation", 0)),
            status.get("registry_digest", ""),
        )
        if not (identity[0] and identity[2] and identity[1] > 0):
            raise InteractiveControlError("SKILL_REGISTRY_NOT_READY", "Gateway registry identity is not ready")
        snapshot = self._bridge.get_skill_snapshot(
            registry_epoch=identity[0],
            generation=identity[1],
            timeout_sec=status.get("rpc_timeout_sec", self._rpc_timeout_sec),
        )
        view = self._resolve_view(snapshot, status)
        with self._state_lock:
            self._fresh_status = status
            self._fresh_view = view
            self._fresh_identity = identity
            if self._stop_requested_now():
                self._state = STOPPING
            elif self._state != STARTING:
                self._state = DISCOVERED
        return {
            "robot_name": view["robot_name"],
            "registry_identity": {
                "registry_epoch": identity[0],
                "registry_generation": identity[1],
                "registry_digest": identity[2],
            },
            "capability_digest": view["capability_digest"],
            "planner_visible_names": sorted(planner_visible_names(view)),
            "capabilities": [
                {
                    "name": skill["name"],
                    "semantic_level": skill.get("semantic_level"),
                    "planner_visible": bool(skill.get("planner_visible")),
                    "ready": self._capability_ready(status, skill["name"]),
                }
                for skill in view["skills"]
            ],
        }

    @staticmethod
    def _capability_ready(status: dict[str, Any], skill_name: str) -> bool:
        for capability in status.get("capabilities", []):
            if capability.get("name") == skill_name:
                return bool(capability.get("ready"))
        return False

    def reject_out_of_catalog(self, steps: list[dict[str, Any]], view: dict[str, Any] | None = None) -> None:
        """Feature 2: reject any step whose skill is not planner-visible in the catalog."""
        catalog_view = view if view is not None else self._fresh_view
        if catalog_view is None:
            raise IllegalStateError("ILLEGAL_STATE", "discover() must run before catalog rejection")
        visible = planner_visible_names(catalog_view)
        for step in steps:
            skill_name = str(step.get("skill_name", "")).strip()
            capability = next((s for s in catalog_view["skills"] if s["name"] == skill_name), None)
            if capability is None or skill_name not in visible:
                raise OutOfCatalogError("SKILL_REFERENCE_MISSING", f"skill is not planner-visible: {skill_name}")
            if capability.get("semantic_level") not in {"atomic_operator", "skill"}:
                raise OutOfCatalogError(
                    "SKILL_REFERENCE_MISSING", f"skill semantic level is not plannable: {skill_name}"
                )

    def prepare_workflow(self, raw_command: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
        """Feature 3 (create + present): plan one workflow and bind the pending plan in-session."""
        if self._fresh_view is None:
            raise IllegalStateError("ILLEGAL_STATE", "discover() must run before prepare_workflow()")
        self.reject_out_of_catalog(steps)
        normalized = [step.to_dict() for step in normalize_workflow_steps(steps)]
        request_id = self._id_factory()
        task_id = self._id_factory()
        result = self._bridge.plan_agent_command(
            request_id=request_id,
            raw_command=raw_command,
            workflow_steps=normalized,
            timeout_sec=self._rpc(),
            execution_mode=self._execution_mode,
        )
        if not result.get("success"):
            raise InteractiveControlError(
                str(result.get("error_code") or "SKILL_SCHEMA_INVALID"),
                str(result.get("message") or "plan_agent_command failed"),
            )
        plan = result["plan"]
        plan_steps = [step.to_dict() for step in normalize_workflow_steps(plan.get("workflow_steps", []))]
        if not self._workflow_steps_match(normalized, plan_steps):
            raise InteractiveControlError(
                "SKILL_SNAPSHOT_DIGEST_MISMATCH",
                "planned workflow differs from the requested workflow",
            )
        plan_identity = (
            str(plan.get("registry_epoch", "")),
            int(plan.get("registry_generation", 0)),
            str(plan.get("registry_digest", "")),
        )
        if plan_identity != self._fresh_identity:
            raise InteractiveControlError(
                "SKILL_SNAPSHOT_DIGEST_MISMATCH",
                "planned workflow uses a different registry identity",
            )
        pending = {
            "plan_token": plan["plan_token"],
            "plan_digest": plan["plan_digest"],
            "plan_id": plan["plan_id"],
            "plan_kind": int(plan.get("plan_kind", 0)),
            "raw_command": raw_command,
            "steps": plan_steps,
            "registry_identity": plan_identity,
            "task_id": task_id,
        }
        with self._state_lock:
            self._pending = pending
            self._confirmed = None
            self._terminal = None
            self._state = STOPPING if self._stop_requested_now() else PREPARED
        return self._presentation()

    @staticmethod
    def _workflow_steps_match(requested: list[dict[str, Any]], planned: list[dict[str, Any]]) -> bool:
        if len(requested) != len(planned):
            return False
        text_fields = (
            "skill_name",
            "target_name",
            "container_name",
            "place_name",
            "motion_direction",
            "arm_side",
            "direction",
        )
        for expected, actual in zip(requested, planned, strict=True):
            if int(actual.get("schema_version", 0)) != int(expected.get("schema_version", 0)):
                return False
            if any(str(actual.get(field, "")) != str(expected.get(field, "")) for field in text_fields):
                return False
            for field in ("motion_distance", "imitation_duration_sec", "timeout_sec"):
                if _float32(actual.get(field, 0.0)) != _float32(expected.get(field, 0.0)):
                    return False
            for field in ("distance", "degree"):
                if float(actual.get(field, 0.0)) != float(expected.get(field, 0.0)):
                    return False
            for field in ("x", "y", "yaw"):
                has_field = f"has_{field}"
                if bool(actual.get(has_field, False)) != bool(expected.get(has_field, False)):
                    return False
                if bool(expected.get(has_field, False)) and float(actual.get(field, 0.0)) != float(
                    expected.get(field, 0.0)
                ):
                    return False
        return True

    def _presentation(self) -> dict[str, Any]:
        assert self._pending is not None
        registry_identity = self._pending["registry_identity"]
        return {
            "state": self._state,
            "steps": list(self._pending["steps"]),
            "plan_digest": self._pending["plan_digest"],
            "plan_id": self._pending["plan_id"],
            "registry_identity": {
                "registry_epoch": registry_identity[0],
                "registry_generation": registry_identity[1],
                "registry_digest": registry_identity[2],
            },
            "task_id": self._pending["task_id"],
            "execution_mode": self._execution_mode,
        }

    def confirm_plan(self) -> dict[str, Any]:
        """Internal validate + confirm (no user gate).

        Runs ``validate_agent_plan`` + ``confirm_agent_plan`` on the in-session
        pending plan so the Gateway can execute it. The user-facing ``确认`` gate
        is removed: this is called automatically right after presentation so
        execution starts immediately; the user aborts a wrong workflow with
        ``别动`` during execution instead of confirming beforehand.
        """
        if self._pending is None or self._state not in {PREPARED, STOPPING}:
            raise IllegalStateError("ILLEGAL_STATE", "no pending workflow to confirm")
        if self._stop_requested_now():
            self._state = STOPPING
            return {
                "state": self._state,
                "task_id": self._pending["task_id"],
                "confirmation_token": "",
                "stopped_before_execution": True,
            }
        try:
            validation = self._bridge.validate_agent_plan(
                plan_token=self._pending["plan_token"],
                timeout_sec=self._status_timeout_sec,
            )
        except Exception as exc:
            with self._state_lock:
                self._clear_operation()
                self._state = FAILED
            raise InteractiveControlError("SERVER_UNAVAILABLE", "plan validation response was unavailable") from exc
        if not validation.get("allowed"):
            raise InteractiveControlError(
                str(validation.get("error_code") or "SKILL_VALIDATION_FAILED"),
                str(validation.get("message") or "validate_agent_plan failed"),
            )
        if self._stop_requested_now():
            self._state = STOPPING
            return {
                "state": self._state,
                "task_id": self._pending["task_id"],
                "confirmation_token": "",
                "stopped_before_execution": True,
            }
        expected_plan = (self._pending["plan_id"], self._pending["plan_digest"])
        actual_plan = (str(validation.get("plan_id", "")), str(validation.get("plan_digest", "")))
        if actual_plan != expected_plan:
            raise InteractiveControlError(
                "SKILL_SNAPSHOT_DIGEST_MISMATCH",
                "validate_agent_plan returned a different plan identity",
            )
        task_budget_sec = float(self._fresh_status["task_budget_sec"])
        try:
            result = self._bridge.confirm_agent_plan(
                plan_token=self._pending["plan_token"],
                plan_digest=self._pending["plan_digest"],
                task_id=self._pending["task_id"],
                status=self._fresh_status,
                task_budget_sec=task_budget_sec,
                timeout_sec=self._status_timeout_sec,
                execution_mode=self._execution_mode,
            )
        except Exception as exc:
            terminal = self._record_unknown(
                self._pending["task_id"],
                failure_detail="plan confirmation response was unavailable",
            )
            raise InteractiveControlError(terminal["error_code"], terminal["message"]) from exc
        if not result.get("confirmed"):
            raise InteractiveControlError(
                str(result.get("error_code") or "SKILL_REQUEST_ID_CONFLICT"),
                str(result.get("message") or "confirm_agent_plan failed"),
            )
        if self._stop_requested_now():
            self._state = STOPPING
            return {
                "state": self._state,
                "task_id": self._pending["task_id"],
                "confirmation_token": result["confirmation_token"],
                "stopped_before_execution": True,
            }
        self._confirmed = {
            "confirmation_token": result["confirmation_token"],
            "task_id": self._pending["task_id"],
            "task_budget_sec": float(result.get("confirmed_task_budget_sec", task_budget_sec)),
        }
        self._state = CONFIRMED
        return {
            "state": self._state,
            "task_id": self._confirmed["task_id"],
            "confirmed_task_budget_sec": self._confirmed["task_budget_sec"],
        }

    def confirm(self, confirmation_text: str) -> dict[str, Any]:
        """Optional gated confirm: bind the pending plan via the closed NL ``确认`` grammar, then confirm_plan()."""
        if self._state != PREPARED or self._pending is None:
            raise IllegalStateError("ILLEGAL_STATE", "no pending workflow to confirm")
        if classify(confirmation_text) != INTENT_CONFIRM:
            raise NotConfirmedError("NOT_CONFIRMED", f"not a confirmation phrase: {confirmation_text!r}")
        return self.confirm_plan()

    def run(
        self,
        raw_command: str,
        steps: list[dict[str, Any]],
        *,
        presentation_callback: Callable[[dict[str, Any]], None],
        authorization_callback: Callable[[dict[str, Any]], None] | None = None,
        stop_event: threading.Event | None = None,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """No-gate one-shot flow: discover → prepare → auto confirm_plan → execute.

        Removes the user-facing ``确认`` gate: the plan is validated and confirmed
        internally so execution starts immediately after presentation. The user
        interrupts a wrong workflow with ``别动`` (``stop_event`` / ``request_stop``)
        during execution. Callable from ``IDLE`` or any definite terminal.
        """
        with self._state_lock:
            if self._state not in {IDLE, DISCOVERED, STOPPED, SUCCEEDED, FAILED}:
                raise IllegalStateError("ILLEGAL_STATE", f"cannot run from state {self._state}")
            reuse_discovery = self._state == DISCOVERED
            if not reuse_discovery:
                self._fresh_status = None
                self._fresh_view = None
                self._fresh_identity = None
            self._pending = None
            self._confirmed = None
            self._terminal = None
            self._active_task_id = None
            self._cancel_started = False
            self._cancel_accepted = False
            self._goal_future = None
            self._goal_handle = None
            self._result_future = None
            self._submission_started = False
            self._stop_requested.clear()
            self._stop_event = stop_event
            self._state = STARTING
            if stop_event is not None and stop_event.is_set():
                self._stop_requested.set()
                self._state = STOPPING
        if self._fresh_view is None:
            try:
                self.discover()
            except Exception:
                with self._state_lock:
                    self._clear_operation()
                    self._state = IDLE
                raise
        try:
            presentation = self.prepare_workflow(raw_command, steps)
        except Exception:
            with self._state_lock:
                self._clear_operation()
                self._state = DISCOVERED if self._fresh_view is not None else IDLE
            raise
        try:
            presentation_callback(presentation)
        except Exception as exc:
            task_id = self._pending["task_id"] if self._pending is not None else ""
            self._record_terminal(task_id, FAILED, "PRESENTATION_FAILED", "plan presentation failed")
            raise InteractiveControlError("PRESENTATION_FAILED", "plan presentation failed") from exc
        with self._state_lock:
            if self._stop_requested_now():
                return self._record_local_stop(self._pending["task_id"], "stopped after plan presentation")
            if self._state != PREPARED or self._pending is None:
                raise IllegalStateError("ILLEGAL_STATE", "workflow presentation did not complete")
        confirmation = self.confirm_plan()
        if authorization_callback is not None and self.state == CONFIRMED:
            authorization_callback(confirmation)
        return self.execute(stop_event=stop_event, feedback_callback=feedback_callback)

    def execute(
        self,
        *,
        stop_event: threading.Event | None = None,
        feedback_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Feature 4: execute the confirmed plan; interruptible via ``stop_event``."""
        with self._state_lock:
            if self._state == STOPPING and self._confirmed is None and self._pending is not None:
                return self._record_local_stop(self._pending["task_id"], "stopped before goal admission")
            if self._state != CONFIRMED or self._confirmed is None or self._pending is None:
                raise IllegalStateError("ILLEGAL_STATE", "no confirmed workflow to execute")
            task_id = self._confirmed["task_id"]
            timeout_sec = self._confirmed["task_budget_sec"]
            plan_token = self._pending["plan_token"]
            confirmation_token = self._confirmed["confirmation_token"]
            self._stop_event = stop_event or self._stop_event
            if self._stop_requested_now():
                return self._record_local_stop(task_id, "stopped before goal admission")
            self._active_task_id = task_id
            self._state = EXECUTING

        def guarded_feedback(feedback: dict[str, Any]) -> None:
            if feedback_callback is None:
                return
            with self._state_lock:
                if self._active_task_id != task_id or self._state not in {EXECUTING, STOPPING}:
                    return
                feedback_callback(feedback)

        try:
            if not self._wait_for_server_interruptibly():
                if self._stop_requested_now():
                    return self._record_local_stop(task_id, "stopped before goal admission")
                return self._record_terminal(
                    task_id,
                    FAILED,
                    "SERVER_UNAVAILABLE",
                    "agent plan action server unavailable",
                )
        except InteractiveControlError:
            raise
        except Exception as exc:
            return self._converge_after_failure(task_id, "agent plan action server unavailable", exc)
        if self._stop_requested_now():
            return self._record_local_stop(task_id, "stopped before goal admission")
        with self._state_lock:
            if self._state not in {EXECUTING, STOPPING} or self._active_task_id != task_id:
                raise IllegalStateError("ILLEGAL_STATE", "execution ownership was lost")
            if self._stop_requested_now():
                return self._record_local_stop(task_id, "stopped before goal admission")
            self._submission_started = True
        try:
            goal_future = self._bridge.send_agent_plan_goal(
                plan_token=plan_token,
                confirmation_token=confirmation_token,
                task_id=task_id,
                timeout_sec=timeout_sec,
                feedback_callback=guarded_feedback,
            )
        except Exception as exc:
            submit_error = exc
            goal_future = None
        else:
            submit_error = None
            with self._state_lock:
                self._goal_future = goal_future
        if submit_error is not None:
            return self._converge_after_failure(task_id, "goal submission failed", submit_error)
        try:
            goal_ready = self._wait_future_interruptibly(goal_future, self._rpc())
        except Exception as exc:
            return self._converge_after_failure(task_id, "goal response unavailable", exc)
        if not goal_ready:
            if self._stop_requested_now():
                return self._stop_and_converge(task_id)
            return self._converge_after_failure(task_id, "goal response timed out")
        try:
            goal_handle = goal_future.result()
        except Exception as exc:
            return self._converge_after_failure(task_id, "goal response unavailable", exc)
        with self._state_lock:
            self._goal_handle = goal_handle
        if goal_handle is None or not getattr(goal_handle, "accepted", False):
            if self._stop_requested_now():
                return self._record_local_stop(task_id, "stopped before goal acceptance")
            return self._record_terminal(task_id, FAILED, "GOAL_REJECTED", "agent plan goal was rejected")
        try:
            result_future = goal_handle.get_result_async()
        except Exception as exc:
            return self._converge_after_failure(task_id, "result request unavailable", exc)
        with self._state_lock:
            self._result_future = result_future
        deadline = self._monotonic() + timeout_sec + self._rpc_timeout_sec
        while not result_future.done():
            if self._stop_requested_now():
                return self._stop_and_converge(task_id)
            if self._monotonic() >= deadline:
                return self._stop_and_converge(task_id, deadline_expired=True)
            self._sleep(_EXECUTE_POLL_INTERVAL_SEC)
        return self._read_terminal(task_id, cancel_on_nonterminal=True)

    def request_stop(self) -> None:
        """Feature 4: thread-safe stop trigger from outside the execute loop."""
        with self._state_lock:
            self._stop_requested.set()
            if self._stop_event is not None:
                self._stop_event.set()
            if self._state not in {STOPPED, SUCCEEDED, FAILED, UNKNOWN}:
                self._state = STOPPING

    def _stop_and_converge(self, task_id: str, *, deadline_expired: bool = False) -> dict[str, Any]:
        self._state = STOPPING
        self._cancel_once(task_id)
        return self._read_terminal(task_id, deadline_expired=deadline_expired, cancel_on_nonterminal=True)

    def _converge_after_failure(self, task_id: str, detail: str, _error: Exception | None = None) -> dict[str, Any]:
        if not self._submission_started:
            return self._record_terminal(task_id, FAILED, "SERVER_UNAVAILABLE", detail)
        self._cancel_once(task_id)
        return self._read_terminal(task_id, failure_detail=detail, cancel_on_nonterminal=True)

    def _stop_requested_now(self) -> bool:
        if self._stop_event is not None and self._stop_event.is_set():
            self._stop_requested.set()
        return self._stop_requested.is_set()

    def _wait_future_interruptibly(self, future: Any, timeout_sec: float) -> bool:
        deadline = self._monotonic() + timeout_sec
        while not future.done():
            if self._stop_requested_now():
                return False
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                return False
            wait_timeout = min(remaining, _EXECUTE_POLL_INTERVAL_SEC)
            ready = self._bridge.wait_future(future, timeout_sec=wait_timeout, interrupt_event=self._stop_requested)
            if ready and future.done():
                return True
        return True

    def _wait_for_server_interruptibly(self) -> bool:
        deadline = self._monotonic() + self._rpc()
        while not self._stop_requested_now():
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                return False
            if self._bridge.wait_for_execute_plan_server(timeout_sec=min(remaining, _EXECUTE_POLL_INTERVAL_SEC)):
                return True
            self._sleep(min(remaining, _EXECUTE_POLL_INTERVAL_SEC))
        return False

    def _cancel_once(self, task_id: str, *, retry: bool = False) -> None:
        with self._state_lock:
            if self._cancel_accepted or (self._cancel_started and not retry):
                return
            self._cancel_started = True
        with contextlib.suppress(Exception):
            response = self._bridge.cancel_agent_plan(task_id, timeout_sec=self._rpc())
            with self._state_lock:
                self._cancel_accepted = bool(response.get("accepted"))

    def _read_terminal(
        self,
        task_id: str,
        *,
        deadline_expired: bool = False,
        failure_detail: str = "",
        cancel_on_nonterminal: bool = False,
    ) -> dict[str, Any]:
        deadline = self._monotonic() + self._rpc_timeout_sec
        terminal = None
        while self._monotonic() < deadline:
            with contextlib.suppress(Exception):
                terminal = self._bridge.get_agent_plan_result(task_id, timeout_sec=self._rpc())
            try:
                status = int(terminal.get("status", 0)) if terminal else 0
            except (TypeError, ValueError):
                status = 0
            if status in TERMINAL_GOAL_STATUSES:
                return self._record_terminal_result(task_id, terminal)
            if cancel_on_nonterminal:
                self._cancel_once(task_id, retry=True)
            self._sleep(_CANCEL_POLL_INTERVAL_SEC)
        return self._record_unknown(task_id, deadline_expired=deadline_expired, failure_detail=failure_detail)

    def _record_terminal_result(self, task_id: str, terminal: dict[str, Any]) -> dict[str, Any]:
        status = int(terminal.get("status", 0))
        result = terminal.get("result", {}) or {}
        classification = self._classify_terminal(status, result)
        if classification is None:
            return self._record_unknown(task_id, failure_detail="terminal result failed identity or state validation")
        state, error_code = classification
        return self._record_terminal(
            task_id,
            state,
            error_code,
            str(result.get("message", "")),
            result=result,
            goal_status=status,
        )

    def _classify_terminal(self, status: int, result: dict[str, Any]) -> tuple[str, str] | None:
        if self._pending is None:
            return None
        registry_identity = self._pending["registry_identity"]
        expectation = {
            "plan_id": self._pending["plan_id"],
            "plan_digest": self._pending["plan_digest"],
            "registry_epoch": registry_identity[0],
            "registry_generation": registry_identity[1],
            "registry_digest": registry_identity[2],
            "step_count": len(self._pending["steps"]),
        }
        classification = classify_agent_terminal(status, result, expectation)
        if classification == "succeeded":
            return SUCCEEDED, ""
        if classification == "stopped":
            return STOPPED, "SKILL_CANCELLED"
        if classification == "failed":
            return FAILED, str(result["error_code"])
        if classification == "unknown":
            return UNKNOWN, str(result["error_code"])
        return None

    def _record_terminal(
        self,
        task_id: str,
        state: str,
        error_code: str,
        message: str,
        *,
        result: dict[str, Any] | None = None,
        goal_status: int | None = None,
    ) -> dict[str, Any]:
        terminal = {
            "state": state,
            "task_id": task_id,
            "error_code": error_code,
            "message": message,
            "result": result or {},
        }
        if goal_status is not None:
            terminal["goal_status"] = goal_status
        with self._state_lock:
            self._terminal = terminal
            self._state = state
            self._clear_operation()
            self._stop_requested.clear()
        return dict(terminal)

    def _record_local_stop(self, task_id: str, message: str) -> dict[str, Any]:
        terminal = {
            "state": STOPPED,
            "task_id": task_id,
            "error_code": "SKILL_CANCELLED",
            "message": message,
            "result": {},
            "stopped_before_execution": True,
        }
        with self._state_lock:
            self._terminal = terminal
            self._state = STOPPED
            self._clear_operation()
            self._stop_requested.clear()
        return dict(terminal)

    def _record_unknown(
        self, task_id: str, *, deadline_expired: bool = False, failure_detail: str = ""
    ) -> dict[str, Any]:
        message = "robot stop state is unknown"
        if failure_detail:
            message = f"{message}: {failure_detail}"
        terminal = {
            "state": UNKNOWN,
            "task_id": task_id,
            "error_code": "SKILL_CANCEL_TIMEOUT",
            "message": message,
            "result": {},
        }
        with self._state_lock:
            self._terminal = terminal
            self._state = UNKNOWN
            self._clear_operation()
        return dict(terminal)

    def _clear_operation(self) -> None:
        self._pending = None
        self._confirmed = None
        self._active_task_id = None
        self._stop_event = None
        self._cancel_started = False
        self._cancel_accepted = False
        self._goal_future = None
        self._goal_handle = None
        self._result_future = None
        self._submission_started = False

    def continue_workflow(
        self,
        raw_command: str,
        steps: list[dict[str, Any]] | None = None,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        """Feature 5: continue on fresh state.

        Fresh mode (``resume=False``, default): plan the caller-provided ``steps``
        as a brand-new continuation with a new ``request_id`` / ``task_id``.

        Resume mode (``resume=True``) is unavailable in this baseline because it
        has no server-owned continuation admission contract. Client-side slicing
        of ``completed_step_count`` is telemetry-based and therefore fails closed.

        Both modes require a definitely canceled prior plan. Success, failure, and
        unknown-stop states do not authorize continuation.
        """
        with self._state_lock:
            if self._state != STOPPED:
                raise UnknownStopError("SKILL_CANCEL_TIMEOUT", "robot stop state is unknown; refuse to continue")
            previous_terminal = dict(self._terminal or {})
            result = previous_terminal.get("result") or {}
            canceled_terminal = (
                previous_terminal.get("goal_status") == GOAL_CANCELED
                and result.get("success") is False
                and result.get("error_code") == "SKILL_CANCELLED"
            )
            if not canceled_terminal:
                raise UnknownStopError("SKILL_CANCEL_TIMEOUT", "robot stop state is unknown; refuse to continue")
            if resume:
                raise IllegalStateError(
                    "SKILL_CONTINUATION_UNAVAILABLE",
                    "resume requires a server-owned continuation admission contract",
                )
            if not steps:
                raise IllegalStateError("ILLEGAL_STATE", "non-resume continuation requires steps")
            steps_to_plan = steps
            self._fresh_status = None
            self._fresh_view = None
            self._fresh_identity = None
            self._pending = None
            self._confirmed = None
            self._stop_requested.clear()
            self._state = STARTING
        try:
            self.discover()
            presentation = self.prepare_workflow(raw_command, steps_to_plan)
        except Exception:
            with self._state_lock:
                self._clear_operation()
                self._state = STOPPED
            raise
        presentation["continues_from"] = previous_terminal
        presentation["resume"] = resume
        return presentation

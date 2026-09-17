"""Lazy ROS 2 clients for the Capability Gateway runtime surface."""

from __future__ import annotations

import contextlib
import os
import threading
import time
import uuid
from typing import Any

from embodied_common.agent_execution_contract import INTERACTIVE_CONFIRMATION
from embodied_common.skill_request import skill_goal_uuid, validate_request_schema_version
from robot_skill_cli.output import EXIT_ROS_UNAVAILABLE, EXIT_TIMEOUT

_NAVIGATION_WORKFLOW_FIELDS = frozenset(
    {"direction", "distance", "degree", "has_x", "x", "has_y", "y", "has_yaw", "yaw"}
)


class BridgeError(RuntimeError):
    def __init__(self, code: str, message: str, *, exit_code: int) -> None:
        self.code = code
        self.exit_code = exit_code
        super().__init__(message)


class RosBridge:
    """Own Gateway clients while keeping rclpy imports out of catalog commands."""

    def __init__(
        self,
        *,
        status_service: str,
        snapshot_service: str = "/embodied/get_skill_snapshot",
        reload_service: str = "/embodied/reload_skill_catalog",
        validate_skill_service: str,
        skill_action: str,
        primitive_action: str = "/embodied/execute_primitive",
        validate_primitive_service: str = "/embodied/validate_primitive",
        plan_service: str = "/embodied/plan_agent_command",
        validate_plan_service: str = "/embodied/validate_agent_plan",
        confirm_plan_service: str = "/embodied/confirm_agent_plan",
        prepare_plan_service: str = "/embodied/prepare_agent_plan",
        execute_plan_action: str = "/embodied/execute_agent_plan",
        start_visual_game_service: str = "/embodied/start_visual_game",
        get_visual_game_result_service: str = "/embodied/get_visual_game_result",
    ) -> None:
        self._status_service = status_service
        self._snapshot_service = snapshot_service
        self._reload_service = reload_service
        self._validate_skill_service = validate_skill_service
        self._skill_action = skill_action.rstrip("/")
        self._primitive_action = primitive_action.rstrip("/")
        self._validate_primitive_service = validate_primitive_service
        self._plan_service = plan_service
        self._validate_plan_service = validate_plan_service
        self._confirm_plan_service = confirm_plan_service
        self._prepare_plan_service = prepare_plan_service
        self._execute_plan_action = execute_plan_action.rstrip("/")
        self._start_visual_game_service = start_visual_game_service
        self._get_visual_game_result_service = get_visual_game_result_service
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._context = None
        self._status_client = None
        self._snapshot_client = None
        self._reload_client = None
        self._validate_client = None
        self._skill_client = None
        self._validate_primitive_client = None
        self._primitive_client = None
        self._plan_client = None
        self._validate_plan_client = None
        self._confirm_plan_client = None
        self._prepare_plan_client = None
        self._execute_plan_client = None
        self._cancel_client = None
        self._cancel_plan_client = None
        self._plan_result_client = None
        self._start_visual_game_client = None
        self._get_visual_game_result_client = None
        self._GetSkillGatewayStatus = None
        self._GetSkillSnapshot = None
        self._ReloadSkillCatalog = None
        self._ValidateSkill = None
        self._SkillCommand = None
        self._PlanAgentCommand = None
        self._ValidateAgentPlan = None
        self._ConfirmAgentPlan = None
        self._PrepareAgentPlan = None
        self._ExecuteAgentPlan = None
        self._CancelGoal = None
        self._WorkflowStep = None
        self._StartVisualGame = None
        self._GetVisualGameResult = None
        self.start_error = ""

    def start(self) -> bool:
        try:
            import rclpy
            from action_msgs.srv import CancelGoal
            from rclpy.action import ActionClient
            from rclpy.callback_groups import ReentrantCallbackGroup
            from rclpy.context import Context
            from rclpy.executors import MultiThreadedExecutor

            from embodied_common.wire_contracts import validate_public_request_wire_contracts
            from ibrobot_msgs.action import ExecuteAgentPlan, PrimitiveCommand, SkillCommand
            from ibrobot_msgs.msg import WorkflowStep
            from ibrobot_msgs.srv import (
                ConfirmAgentPlan,
                GetSkillGatewayStatus,
                GetSkillSnapshot,
                GetVisualGameResult,
                PlanAgentCommand,
                PrepareAgentPlan,
                ReloadSkillCatalog,
                StartVisualGame,
                ValidateAgentPlan,
                ValidatePrimitive,
                ValidateSkill,
            )

            validate_public_request_wire_contracts()
        except Exception as exc:
            self.start_error = f"{type(exc).__name__}: {exc}"
            return False

        try:
            self._context = Context()
            rclpy.init(context=self._context)
            suffix = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            self._node = rclpy.create_node(
                f"robot_skill_cli_{suffix}",
                context=self._context,
                use_global_arguments=False,
            )
            callback_group = ReentrantCallbackGroup()
            self._status_client = self._node.create_client(
                GetSkillGatewayStatus,
                self._status_service,
                callback_group=callback_group,
            )
            self._snapshot_client = self._node.create_client(
                GetSkillSnapshot, self._snapshot_service, callback_group=callback_group
            )
            self._reload_client = self._node.create_client(
                ReloadSkillCatalog, self._reload_service, callback_group=callback_group
            )
            self._validate_client = self._node.create_client(
                ValidateSkill,
                self._validate_skill_service,
                callback_group=callback_group,
            )
            self._skill_client = ActionClient(
                self._node,
                SkillCommand,
                self._skill_action,
                callback_group=callback_group,
            )
            self._validate_primitive_client = self._node.create_client(
                ValidatePrimitive,
                self._validate_primitive_service,
                callback_group=callback_group,
            )
            self._primitive_client = ActionClient(
                self._node,
                PrimitiveCommand,
                self._primitive_action,
                callback_group=callback_group,
            )
            self._plan_client = self._node.create_client(
                PlanAgentCommand, self._plan_service, callback_group=callback_group
            )
            self._validate_plan_client = self._node.create_client(
                ValidateAgentPlan, self._validate_plan_service, callback_group=callback_group
            )
            self._confirm_plan_client = self._node.create_client(
                ConfirmAgentPlan, self._confirm_plan_service, callback_group=callback_group
            )
            self._prepare_plan_client = self._node.create_client(
                PrepareAgentPlan, self._prepare_plan_service, callback_group=callback_group
            )
            self._execute_plan_client = ActionClient(
                self._node,
                ExecuteAgentPlan,
                self._execute_plan_action,
                callback_group=callback_group,
            )
            self._cancel_client = self._node.create_client(
                CancelGoal,
                f"{self._skill_action}/_action/cancel_goal",
                callback_group=callback_group,
            )
            self._cancel_plan_client = self._node.create_client(
                CancelGoal,
                f"{self._execute_plan_action}/_action/cancel_goal",
                callback_group=callback_group,
            )
            self._plan_result_client = self._node.create_client(
                ExecuteAgentPlan.Impl.GetResultService,
                f"{self._execute_plan_action}/_action/get_result",
                callback_group=callback_group,
            )
            self._start_visual_game_client = self._node.create_client(
                StartVisualGame,
                self._start_visual_game_service,
                callback_group=callback_group,
            )
            self._get_visual_game_result_client = self._node.create_client(
                GetVisualGameResult,
                self._get_visual_game_result_service,
                callback_group=callback_group,
            )
            self._GetSkillGatewayStatus = GetSkillGatewayStatus
            self._GetSkillSnapshot = GetSkillSnapshot
            self._ReloadSkillCatalog = ReloadSkillCatalog
            self._ValidateSkill = ValidateSkill
            self._SkillCommand = SkillCommand
            self._PlanAgentCommand = PlanAgentCommand
            self._ValidateAgentPlan = ValidateAgentPlan
            self._ConfirmAgentPlan = ConfirmAgentPlan
            self._PrepareAgentPlan = PrepareAgentPlan
            self._ExecuteAgentPlan = ExecuteAgentPlan
            self._CancelGoal = CancelGoal
            self._WorkflowStep = WorkflowStep
            self._StartVisualGame = StartVisualGame
            self._GetVisualGameResult = GetVisualGameResult
            self._executor = MultiThreadedExecutor(num_threads=2, context=self._context)
            self._executor.add_node(self._node)
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()
        except Exception as exc:
            self.start_error = f"{type(exc).__name__}: {exc}"
            self.close()
            return False
        return True

    @staticmethod
    def _wait_future(future, timeout_sec: float, interrupt_event: threading.Event | None = None) -> bool:
        deadline = time.monotonic() + timeout_sec
        while True:
            if future.done():
                return True
            if interrupt_event is not None and interrupt_event.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(0.01, remaining))

    def wait_future(self, future, *, timeout_sec: float, interrupt_event: threading.Event | None = None) -> bool:
        return self._wait_future(future, timeout_sec, interrupt_event)

    def _call_service(self, client, request, *, service_name: str, timeout_sec: float):
        try:
            service_ready = client is not None and client.wait_for_service(timeout_sec=timeout_sec)
        except Exception as exc:
            raise BridgeError(
                "ROS_UNAVAILABLE",
                f"service unavailable: {service_name}: {exc}",
                exit_code=EXIT_ROS_UNAVAILABLE,
            ) from exc
        if not service_ready:
            raise BridgeError(
                "SERVER_UNAVAILABLE",
                f"service unavailable: {service_name}",
                exit_code=EXIT_ROS_UNAVAILABLE,
            )
        try:
            future = client.call_async(request)
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc
        if not self._wait_future(future, timeout_sec):
            raise BridgeError(
                "RESULT_TIMEOUT",
                f"service response timed out: {service_name}",
                exit_code=EXIT_TIMEOUT,
            )
        try:
            return future.result()
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc

    def get_status(
        self,
        *,
        task_id: str = "",
        payload_hash: str = "",
        timeout_sec: float,
    ) -> dict[str, Any]:
        if self._GetSkillGatewayStatus is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._GetSkillGatewayStatus.Request()
        request.schema_version = 1
        request.task_id = task_id
        request.payload_hash = payload_hash
        response = self._call_service(
            self._status_client,
            request,
            service_name=self._status_service,
            timeout_sec=timeout_sec,
        )
        return {
            "schema_version": int(response.schema_version),
            "robot_name": response.robot_name,
            "motion_authorized": bool(response.motion_authorized),
            "active_control_mode": response.active_control_mode,
            "busy": bool(response.busy),
            "active_task_id": response.active_task_id,
            "default_skill_timeout_sec": float(response.default_skill_timeout_sec),
            "task_budget_sec": float(response.task_budget_sec),
            "rpc_timeout_sec": float(response.rpc_timeout_sec),
            "config_digest": response.config_digest,
            "capability_digest": response.capability_digest,
            "registry_epoch": response.registry_epoch,
            "registry_generation": int(response.registry_generation),
            "registry_digest": response.registry_digest,
            "primitive_contract_digest": response.primitive_contract_digest,
            "source_release_digest": response.source_release_digest,
            "provenance_digest": response.provenance_digest,
            "control_plane_ready": bool(response.control_plane_ready),
            "control_plane_state": response.control_plane_state,
            "control_plane_error_code": response.control_plane_error_code,
            "request_state": response.request_state,
            "request_error_code": response.request_error_code,
            "capabilities": [
                {
                    "name": capability.name,
                    "semantic_level": capability.semantic_level,
                    "planner_visible": bool(capability.planner_visible),
                    "ready": bool(capability.ready),
                    "reason": capability.reason,
                    "required_control_mode": capability.required_control_mode,
                }
                for capability in response.capabilities
            ],
        }

    def validate_skill(
        self,
        payload: dict[str, Any],
        *,
        timeout_sec: float,
        registry_identity: tuple[str, int, str] | None = None,
    ) -> dict[str, Any]:
        if self._ValidateSkill is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._ValidateSkill.Request()
        request.schema_version = self._payload_schema_version(payload)
        request.dispatch_binding.schema_version = 1
        request.dispatch_binding.task_id = str(payload.get("_task_id", ""))
        request.dispatch_binding.root_task_id = str(payload.get("_root_task_id", request.dispatch_binding.task_id))
        identity = registry_identity or (
            str(payload.get("_registry_epoch", "")),
            int(payload.get("_registry_generation", 0)),
            str(payload.get("_registry_digest", "")),
        )
        request.dispatch_binding.expected_registry_epoch = identity[0]
        request.dispatch_binding.expected_registry_generation = identity[1]
        request.dispatch_binding.expected_registry_digest = identity[2]
        request.skill_name = payload["skill_name"]
        request.target_name = payload["target_name"]
        request.container_name = str(payload.get("container_name", ""))
        request.place_name = payload["place_name"]
        request.motion_direction = payload["motion_direction"]
        request.motion_distance = float(payload["motion_distance"] or 0.0)
        request.arm_side = str(payload.get("arm_side", ""))
        request.imitation_duration_sec = float(payload.get("imitation_duration_sec", 0.0) or 0.0)
        self._copy_navigation_fields(request, payload)
        response = self._call_service(
            self._validate_client,
            request,
            service_name=self._validate_skill_service,
            timeout_sec=timeout_sec,
        )
        return {
            "allowed": bool(response.allowed),
            "reason": response.reason,
            "error_code": response.error_code,
            "actual_registry_epoch": response.actual_registry_epoch,
            "actual_registry_generation": int(response.actual_registry_generation),
            "actual_registry_digest": response.actual_registry_digest,
            "diagnostics": [
                {
                    "schema_version": int(diagnostic.schema_version),
                    "severity": int(diagnostic.severity),
                    "error_code": diagnostic.error_code,
                    "source_relative_path": diagnostic.source_relative_path,
                    "field_path": diagnostic.field_path,
                    "message": diagnostic.message,
                }
                for diagnostic in response.diagnostics
            ],
        }

    def get_skill_snapshot(
        self,
        *,
        registry_epoch: str,
        generation: int,
        timeout_sec: float,
    ) -> dict[str, Any]:
        if self._GetSkillSnapshot is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._GetSkillSnapshot.Request()
        request.schema_version = 1
        request.registry_epoch = registry_epoch
        request.generation = int(generation)
        response = self._call_service(
            self._snapshot_client,
            request,
            service_name=self._snapshot_service,
            timeout_sec=timeout_sec,
        )
        return {
            "success": bool(response.success),
            "registry_epoch": str(response.registry_epoch),
            "generation": int(response.generation),
            "registry_digest": str(response.registry_digest),
            "capability_digest": str(response.capability_digest),
            "source_release_digest": str(response.source_release_digest),
            "provenance_digest": str(response.provenance_digest),
            "profile_name": str(response.profile_name),
            "snapshot_json": str(response.snapshot_json),
            "error_code": str(response.error_code),
            "message": str(response.message),
        }

    def reload_skill_catalog(self, *, request_id: str, force: bool, timeout_sec: float) -> dict[str, Any]:
        if self._ReloadSkillCatalog is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._ReloadSkillCatalog.Request()
        request.schema_version = 1
        request.request_id = request_id
        request.force = force
        response = self._call_service(
            self._reload_client,
            request,
            service_name=self._reload_service,
            timeout_sec=timeout_sec,
        )
        return {
            "success": bool(response.success),
            "registry_epoch": str(response.registry_epoch),
            "old_generation": int(response.old_generation),
            "generation": int(response.generation),
            "registry_digest": str(response.registry_digest),
            "capability_digest": str(response.capability_digest),
            "source_release_digest": str(response.source_release_digest),
            "provenance_digest": str(response.provenance_digest),
            "error_code": str(response.error_code),
            "message": str(response.message),
            "changed_skills": list(response.changed_skills),
            "diagnostics": [
                {
                    "schema_version": int(diagnostic.schema_version),
                    "severity": int(diagnostic.severity),
                    "error_code": diagnostic.error_code,
                    "source_relative_path": diagnostic.source_relative_path,
                    "field_path": diagnostic.field_path,
                    "message": diagnostic.message,
                }
                for diagnostic in response.diagnostics
            ],
        }

    @staticmethod
    def _workflow_step_dict(step) -> dict[str, Any]:
        result = {
            "schema_version": int(step.schema_version),
            "skill_name": str(step.skill_name),
            "target_name": str(step.target_name),
            "container_name": str(step.container_name),
            "place_name": str(step.place_name),
            "motion_direction": str(step.motion_direction),
            "motion_distance": float(step.motion_distance),
            "arm_side": str(step.arm_side),
            "imitation_duration_sec": float(step.imitation_duration_sec),
            "timeout_sec": float(step.timeout_sec),
        }
        if int(step.schema_version) == 2:
            result.update(
                {
                    "direction": str(step.direction),
                    "distance": float(step.distance),
                    "degree": float(step.degree),
                    "has_x": bool(step.has_x),
                    "x": float(step.x),
                    "has_y": bool(step.has_y),
                    "y": float(step.y),
                    "has_yaw": bool(step.has_yaw),
                    "yaw": float(step.yaw),
                }
            )
        return result

    @classmethod
    def _agent_plan_dict(cls, plan) -> dict[str, Any]:
        return {
            "schema_version": int(plan.schema_version),
            "plan_id": str(plan.plan_id),
            "plan_token": str(plan.plan_token),
            "plan_kind": int(plan.plan_kind),
            "raw_command": str(plan.raw_command),
            "workflow_steps": [cls._workflow_step_dict(step) for step in plan.workflow_steps],
            "plan_digest": str(plan.plan_digest),
            "registry_epoch": str(plan.registry_epoch),
            "registry_generation": int(plan.registry_generation),
            "registry_digest": str(plan.registry_digest),
            "execution_mode": str(plan.execution_mode),
            "expires_at": {
                "sec": int(plan.expires_at.sec),
                "nanosec": int(plan.expires_at.nanosec),
            },
        }

    def plan_agent_command(
        self,
        *,
        request_id: str,
        raw_command: str,
        workflow_steps: list[dict[str, Any]],
        timeout_sec: float,
        execution_mode: str = INTERACTIVE_CONFIRMATION,
    ) -> dict[str, Any]:
        if self._PlanAgentCommand is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._PlanAgentCommand.Request()
        request.schema_version = 1
        request.request_id = request_id
        request.raw_command = raw_command
        request.workflow_steps = [self._workflow_step_message(step) for step in workflow_steps]
        request.execution_mode = execution_mode
        response = self._call_service(
            self._plan_client,
            request,
            service_name=self._plan_service,
            timeout_sec=timeout_sec,
        )
        return {
            "success": bool(response.success),
            "plan": self._agent_plan_dict(response.plan),
            "error_code": str(response.error_code),
            "message": str(response.message),
            "diagnostics": self._diagnostics(response.diagnostics),
        }

    def prepare_agent_plan(
        self,
        *,
        request_id: str,
        raw_command: str,
        workflow_steps: list[dict[str, Any]],
        timeout_sec: float,
        execution_mode: str = INTERACTIVE_CONFIRMATION,
        trace_id: str = "",
    ) -> dict[str, Any]:
        if self._PrepareAgentPlan is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._PrepareAgentPlan.Request()
        request.schema_version = 1
        request.request_id = request_id
        request.raw_command = raw_command
        request.workflow_steps = [self._workflow_step_message(step) for step in workflow_steps]
        request.execution_mode = execution_mode
        request.trace_id = str(trace_id).strip()
        response = self._call_service(
            self._prepare_plan_client,
            request,
            service_name=self._prepare_plan_service,
            timeout_sec=timeout_sec,
        )
        return {
            "success": bool(response.success),
            "allowed": bool(response.allowed),
            "plan": self._agent_plan_dict(response.plan),
            "error_code": str(response.error_code),
            "message": str(response.message),
            "diagnostics": self._diagnostics(response.diagnostics),
        }

    def _workflow_step_message(self, step: dict[str, Any]):
        if self._WorkflowStep is None:
            raise BridgeError(
                "ROS_UNAVAILABLE", "WorkflowStep interface is unavailable", exit_code=EXIT_ROS_UNAVAILABLE
            )
        message = self._WorkflowStep()
        message.schema_version = self._payload_schema_version(step)
        message.skill_name = str(step.get("skill_name", ""))
        message.target_name = str(step.get("target_name", ""))
        message.container_name = str(step.get("container_name", ""))
        message.place_name = str(step.get("place_name", ""))
        message.motion_direction = str(step.get("motion_direction", ""))
        message.motion_distance = float(step.get("motion_distance", 0.0))
        message.arm_side = str(step.get("arm_side", ""))
        message.imitation_duration_sec = float(step.get("imitation_duration_sec", 0.0) or 0.0)
        self._copy_navigation_fields(message, step)
        message.timeout_sec = float(step.get("timeout_sec", 0.0))
        return message

    @staticmethod
    def _payload_schema_version(values: dict[str, Any]) -> int:
        if "schema_version" in values:
            try:
                return validate_request_schema_version(values["schema_version"])
            except ValueError as exc:
                raise BridgeError("SKILL_SCHEMA_INVALID", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc
        if _NAVIGATION_WORKFLOW_FIELDS.intersection(values):
            raise BridgeError(
                "SKILL_SCHEMA_INVALID",
                "navigation fields require an explicit schema_version 2",
                exit_code=EXIT_ROS_UNAVAILABLE,
            )
        return 1

    @staticmethod
    def _copy_navigation_fields(message, values: dict[str, Any]) -> None:
        message.direction = str(values.get("direction", ""))
        message.distance = float(values.get("distance", 0.0) or 0.0)
        message.degree = float(values.get("degree", 0.0) or 0.0)
        for field_name in ("x", "y", "yaw"):
            has_field = f"has_{field_name}"
            presence = values.get(has_field)
            if presence is None:
                presence = field_name in values and values[field_name] is not None
            setattr(message, has_field, bool(presence))
            setattr(message, field_name, float(values.get(field_name, 0.0) or 0.0))

    def validate_agent_plan(self, *, plan_token: str, timeout_sec: float) -> dict[str, Any]:
        if self._ValidateAgentPlan is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._ValidateAgentPlan.Request()
        request.schema_version = 1
        request.plan_token = plan_token
        response = self._call_service(
            self._validate_plan_client,
            request,
            service_name=self._validate_plan_service,
            timeout_sec=timeout_sec,
        )
        return {
            "allowed": bool(response.allowed),
            "plan_id": str(response.plan_id),
            "plan_digest": str(response.plan_digest),
            "error_code": str(response.error_code),
            "message": str(response.message),
            "diagnostics": self._diagnostics(response.diagnostics),
        }

    def confirm_agent_plan(
        self,
        *,
        plan_token: str,
        plan_digest: str,
        task_id: str,
        status: dict[str, Any],
        task_budget_sec: float,
        timeout_sec: float,
        execution_mode: str = INTERACTIVE_CONFIRMATION,
    ) -> dict[str, Any]:
        if self._ConfirmAgentPlan is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._ConfirmAgentPlan.Request()
        request.schema_version = 1
        request.plan_token = plan_token
        request.plan_digest = plan_digest
        request.task_id = task_id
        request.registry_epoch = status["registry_epoch"]
        request.registry_generation = int(status["registry_generation"])
        request.registry_digest = status["registry_digest"]
        request.task_budget_sec = float(task_budget_sec)
        request.execution_mode = execution_mode
        response = self._call_service(
            self._confirm_plan_client,
            request,
            service_name=self._confirm_plan_service,
            timeout_sec=timeout_sec,
        )
        return {
            "confirmed": bool(response.confirmed),
            "confirmation_token": str(response.confirmation_token),
            "confirmed_task_budget_sec": float(response.confirmed_task_budget_sec),
            "task_budget_started_at": {
                "sec": int(response.task_budget_started_at.sec),
                "nanosec": int(response.task_budget_started_at.nanosec),
            },
            "task_budget_deadline": {
                "sec": int(response.task_budget_deadline.sec),
                "nanosec": int(response.task_budget_deadline.nanosec),
            },
            "error_code": str(response.error_code),
            "message": str(response.message),
            "diagnostics": self._diagnostics(response.diagnostics),
        }

    @staticmethod
    def _diagnostics(diagnostics) -> list[dict[str, Any]]:
        return [
            {
                "schema_version": int(item.schema_version),
                "severity": int(item.severity),
                "error_code": str(item.error_code),
                "source_relative_path": str(item.source_relative_path),
                "field_path": str(item.field_path),
                "message": str(item.message),
            }
            for item in diagnostics
        ]

    def wait_for_execute_plan_server(self, *, timeout_sec: float) -> bool:
        if self._execute_plan_client is None:
            return False
        return self._execute_plan_client.wait_for_server(timeout_sec=timeout_sec)

    def wait_for_agent_plan_interfaces(self, *, timeout_sec: float, include_prepare: bool = False) -> bool:
        """Return whether every Hermes-facing plan service/action is discoverable."""
        deadline = time.monotonic() + timeout_sec
        clients = [self._plan_client, self._validate_plan_client, self._confirm_plan_client]
        if include_prepare:
            clients.insert(1, self._prepare_plan_client)
        for client in clients:
            remaining = deadline - time.monotonic()
            if client is None or remaining <= 0.0 or not client.wait_for_service(timeout_sec=remaining):
                return False
        remaining = deadline - time.monotonic()
        return (
            self._execute_plan_client is not None
            and remaining > 0.0
            and self._execute_plan_client.wait_for_server(timeout_sec=remaining)
        )

    def wait_for_public_request_interfaces(self, *, timeout_sec: float) -> bool:
        """Return whether all versioned Skill and Primitive public endpoints are discoverable."""
        deadline = time.monotonic() + timeout_sec
        for client in (self._validate_client, self._validate_primitive_client):
            remaining = deadline - time.monotonic()
            if client is None or remaining <= 0.0 or not client.wait_for_service(timeout_sec=remaining):
                return False
        for client in (self._skill_client, self._primitive_client):
            remaining = deadline - time.monotonic()
            if client is None or remaining <= 0.0 or not client.wait_for_server(timeout_sec=remaining):
                return False
        return True

    def wait_for_visual_game_interfaces(self, *, timeout_sec: float) -> bool:
        """Return whether the independent visual-game start/query services are discoverable."""
        deadline = time.monotonic() + timeout_sec
        for client in (self._start_visual_game_client, self._get_visual_game_result_client):
            remaining = deadline - time.monotonic()
            if client is None or remaining <= 0.0 or not client.wait_for_service(timeout_sec=remaining):
                return False
        return True

    def send_agent_plan_goal(
        self,
        *,
        plan_token: str,
        confirmation_token: str,
        task_id: str,
        timeout_sec: float,
        trace_id: str = "",
        feedback_callback=None,
    ):
        if self._ExecuteAgentPlan is None or self._execute_plan_client is None:
            raise BridgeError(
                "ROS_UNAVAILABLE", "agent plan action client is not started", exit_code=EXIT_ROS_UNAVAILABLE
            )
        goal = self._ExecuteAgentPlan.Goal()
        goal.schema_version = 1
        goal.plan_token = plan_token
        goal.confirmation_token = confirmation_token
        goal.task_id = task_id
        goal.timeout_sec = float(timeout_sec)
        goal.trace_id = str(trace_id).strip()

        def on_feedback(feedback) -> None:
            if feedback_callback is None:
                return
            message = getattr(feedback, "feedback", feedback)
            feedback_callback(
                {
                    "state": str(message.state),
                    "detail": str(message.detail),
                    "current_skill": str(message.current_skill),
                    "workflow_step_index": int(message.workflow_step_index),
                }
            )

        from unique_identifier_msgs.msg import UUID

        goal_uuid = UUID(uuid=list(skill_goal_uuid(task_id).bytes))
        try:
            return self._execute_plan_client.send_goal_async(goal, feedback_callback=on_feedback, goal_uuid=goal_uuid)
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc

    def cancel_agent_plan(self, task_id: str, *, timeout_sec: float) -> dict[str, Any]:
        if self._cancel_plan_client is None or self._CancelGoal is None:
            raise BridgeError(
                "ROS_UNAVAILABLE", "agent plan cancel client is not started", exit_code=EXIT_ROS_UNAVAILABLE
            )
        request = self._CancelGoal.Request()
        request.goal_info.goal_id.uuid = list(skill_goal_uuid(task_id).bytes)
        service_name = f"{self._execute_plan_action}/_action/cancel_goal"
        response = self._call_service(
            self._cancel_plan_client,
            request,
            service_name=service_name,
            timeout_sec=timeout_sec,
        )
        return {
            "accepted": int(response.return_code) == 0 and bool(response.goals_canceling),
            "return_code": int(response.return_code),
        }

    @staticmethod
    def _agent_plan_result_dict(result) -> dict[str, Any]:
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

    def get_agent_plan_result(self, task_id: str, *, timeout_sec: float) -> dict[str, Any]:
        if self._plan_result_client is None or self._ExecuteAgentPlan is None:
            raise BridgeError(
                "ROS_UNAVAILABLE", "agent plan result client is not started", exit_code=EXIT_ROS_UNAVAILABLE
            )
        request = self._ExecuteAgentPlan.Impl.GetResultService.Request()
        request.goal_id.uuid = list(skill_goal_uuid(task_id).bytes)
        response = self._call_service(
            self._plan_result_client,
            request,
            service_name=f"{self._execute_plan_action}/_action/get_result",
            timeout_sec=timeout_sec,
        )
        return {
            "status": int(response.status),
            "result": self._agent_plan_result_dict(response.result),
        }

    def wait_for_skill_server(self, *, timeout_sec: float) -> bool:
        if self._skill_client is None:
            return False
        try:
            return self._skill_client.wait_for_server(timeout_sec=timeout_sec)
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc

    def send_skill_goal(self, payload: dict[str, Any], *, task_id: str, feedback_callback=None):
        from unique_identifier_msgs.msg import UUID

        if self._skill_client is None or self._SkillCommand is None:
            raise BridgeError("ROS_UNAVAILABLE", "skill action client is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        goal = self._SkillCommand.Goal()
        goal.schema_version = self._payload_schema_version(payload)
        goal.dispatch_binding.schema_version = 1
        goal.dispatch_binding.task_id = task_id
        goal.dispatch_binding.root_task_id = task_id
        goal.dispatch_binding.expected_registry_epoch = str(payload.get("_registry_epoch", ""))
        goal.dispatch_binding.expected_registry_generation = int(payload.get("_registry_generation", 0))
        goal.dispatch_binding.expected_registry_digest = str(payload.get("_registry_digest", ""))
        goal.skill_name = payload["skill_name"]
        goal.target_name = payload["target_name"]
        goal.container_name = str(payload.get("container_name", ""))
        goal.place_name = payload["place_name"]
        goal.motion_direction = payload["motion_direction"]
        goal.motion_distance = float(payload["motion_distance"] or 0.0)
        goal.arm_side = str(payload.get("arm_side", ""))
        goal.imitation_duration_sec = float(payload.get("imitation_duration_sec", 0.0) or 0.0)
        self._copy_navigation_fields(goal, payload)
        goal.timeout_sec = float(payload["timeout_sec"])

        def on_feedback(feedback) -> None:
            if feedback_callback is None:
                return
            message = getattr(feedback, "feedback", feedback)
            feedback_callback({"state": str(message.state), "detail": str(message.detail)})

        goal_uuid = UUID(uuid=list(skill_goal_uuid(task_id).bytes))
        try:
            return self._skill_client.send_goal_async(goal, feedback_callback=on_feedback, goal_uuid=goal_uuid)
        except Exception as exc:
            raise BridgeError("ROS_UNAVAILABLE", str(exc), exit_code=EXIT_ROS_UNAVAILABLE) from exc

    def cancel_task(self, task_id: str, *, timeout_sec: float) -> dict[str, Any]:
        if self._cancel_client is None or self._CancelGoal is None:
            raise BridgeError("ROS_UNAVAILABLE", "cancel client is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._CancelGoal.Request()
        request.goal_info.goal_id.uuid = list(skill_goal_uuid(task_id).bytes)
        response = self._call_service(
            self._cancel_client,
            request,
            service_name=f"{self._skill_action}/_action/cancel_goal",
            timeout_sec=timeout_sec,
        )
        return {
            "accepted": int(response.return_code) == 0 and bool(response.goals_canceling),
            "return_code": int(response.return_code),
        }

    def start_visual_game(
        self,
        game_name: str,
        *,
        request_id: str,
        expected_config_digest: str,
        timeout_sec: float,
    ) -> dict[str, Any]:
        if self._StartVisualGame is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._StartVisualGame.Request()
        request.request_id = request_id
        request.game_name = game_name
        request.expected_config_digest = expected_config_digest
        response = self._call_service(
            self._start_visual_game_client,
            request,
            service_name=self._start_visual_game_service,
            timeout_sec=timeout_sec,
        )
        return {
            "accepted": bool(response.accepted),
            "duplicate": bool(response.duplicate),
            "request_id": response.request_id,
            "config_digest": response.config_digest,
            "error_code": response.error_code,
            "message": response.message,
        }

    def get_visual_game_result(self, request_id: str, *, timeout_sec: float) -> dict[str, Any]:
        if self._GetVisualGameResult is None:
            raise BridgeError("ROS_UNAVAILABLE", "ROS bridge is not started", exit_code=EXIT_ROS_UNAVAILABLE)
        request = self._GetVisualGameResult.Request()
        request.request_id = request_id
        response = self._call_service(
            self._get_visual_game_result_client,
            request,
            service_name=self._get_visual_game_result_service,
            timeout_sec=timeout_sec,
        )
        return {
            "found": bool(response.found),
            "terminal": bool(response.terminal),
            "success": bool(response.success),
            "game_name": response.game_name,
            "scene_summary": response.scene_summary,
            "result_json": response.result_json,
            "config_digest": response.config_digest,
            "error_code": response.error_code,
            "message": response.message,
        }

    def cancel_goal(self, goal_handle, result_future, *, timeout_sec: float) -> bool:
        if result_future.done():
            return True
        try:
            cancel_future = goal_handle.cancel_goal_async()
        except Exception:
            return False
        if not self._wait_future(cancel_future, timeout_sec):
            return False
        try:
            cancel_future.result()
        except Exception:
            return False
        return self._wait_future(result_future, timeout_sec)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._executor is not None:
                self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            if self._node is not None:
                self._node.destroy_node()
        with contextlib.suppress(Exception):
            if self._context is not None and self._context.ok():
                self._context.shutdown()
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._context = None

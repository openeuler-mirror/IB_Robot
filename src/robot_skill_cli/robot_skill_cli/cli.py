"""Argument parsing and command dispatch for robot-skill."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import threading
import time
from collections.abc import Sequence
from typing import Any

from embodied_common.agent_execution_contract import IMMEDIATE_AFTER_PRESENTATION
from embodied_common.agent_terminal_contract import TERMINAL_GOAL_STATUSES, classify_agent_terminal
from robot_skill_cli import __version__
from robot_skill_cli.output import (
    EXIT_GATEWAY_REJECTED,
    EXIT_INVALID_INPUT,
    EXIT_ROS_UNAVAILABLE,
    EXIT_SIGINT,
    EXIT_SIGTERM,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    error_envelope,
    feedback_event,
    json_dumps,
    result_event,
    success_envelope,
)

_STATUS_PREFLIGHT_TIMEOUT_FLOOR_SEC = 15.0
_CATALOG_RELOAD_TIMEOUT_FLOOR_SEC = 60.0
_AGENT_CONTROL_TIMEOUT_FLOOR_SEC = 30.0

_AGENT_NOT_FOUND_CODES = {
    "SKILL_PROFILE_NOT_FOUND",
    "SKILL_PACKAGE_NOT_FOUND",
    "SKILL_IMPLEMENTATION_NOT_FOUND",
}
_AGENT_SCHEMA_CODES = {
    "SKILL_SCHEMA_INVALID",
    "SKILL_REFERENCE_MISSING",
    "SKILL_UNKNOWN_PRIMITIVE",
    "SKILL_UNKNOWN_EXECUTOR",
    "SKILL_LIMIT_VIOLATION",
}
_AGENT_TIMEOUT_CODES = {
    "TIMEOUT_EXCEEDS_POLICY",
    "SKILL_TASK_BUDGET_MISMATCH",
    "SKILL_TASK_DEADLINE_EXPIRED",
    "SKILL_CANCEL_TIMEOUT",
}
_NAVIGATION_WORKFLOW_FIELDS = frozenset(
    {"direction", "distance", "degree", "has_x", "x", "has_y", "y", "has_yaw", "yaw"}
)


def _agent_error_exit_code(error_code: str) -> int:
    if error_code in _AGENT_NOT_FOUND_CODES:
        return 10
    if error_code in _AGENT_SCHEMA_CODES:
        return 11
    if error_code in _AGENT_TIMEOUT_CODES or "TIMEOUT" in error_code:
        return 15
    if error_code in {"SERVER_UNAVAILABLE", "ROS_TRANSPORT_ERROR"}:
        return 14
    return 13


class _CliArgumentError(ValueError):
    pass


class _CommandError(RuntimeError):
    def __init__(self, code: str, message: str, *, exit_code: int) -> None:
        self.code = code
        self.exit_code = exit_code
        super().__init__(message)


class _CommandExit:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CliArgumentError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="robot-skill", description="Inspect and execute IB-Robot capabilities.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config-name", help="robot_config name")
    config_group.add_argument("--config-path", help="explicit robot_config YAML path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-skills", help="list enabled high-level skills")
    subparsers.add_parser("list-games", help="list enabled visual games")
    describe_game_parser = subparsers.add_parser("describe-game", help="describe one enabled visual game")
    describe_game_parser.add_argument("game")
    describe_parser = subparsers.add_parser("describe", help="describe one enabled skill")
    describe_parser.add_argument("skill")
    subparsers.add_parser("list-poses", help="list configured named poses")
    subparsers.add_parser("status", help="query Capability Gateway status")
    validate_parser = subparsers.add_parser("validate", help="validate one skill without executing it")
    validate_parser.add_argument("skill")
    _add_skill_parameters(validate_parser)
    execute_parser = subparsers.add_parser("execute", help="execute one high-level skill")
    execute_parser.add_argument("skill")
    execute_parser.add_argument("--task-id", required=True)
    _add_skill_parameters(execute_parser)
    cancel_parser = subparsers.add_parser("cancel", help="cancel one active task by task ID")
    cancel_parser.add_argument("--task-id", required=True)
    reload_parser = subparsers.add_parser("reload-catalog", help="reload the configured runtime skill catalog")
    reload_parser.add_argument("--request-id", required=True)
    reload_parser.add_argument("--force", action="store_true")
    plan_parser = subparsers.add_parser("plan-workflow", help="plan one typed Agent workflow")
    plan_parser.add_argument("--text", dest="raw_command", required=True, help="audit text for this workflow")
    plan_parser.add_argument("--request-id", required=True)
    plan_parser.add_argument("--workflow-json", required=True)
    run_workflow_parser = subparsers.add_parser(
        "run-workflow", help="run one typed workflow through the complete Gateway lifecycle"
    )
    run_workflow_parser.add_argument("--text", dest="raw_command", required=True, help="audit text for this workflow")
    run_workflow_parser.add_argument("--workflow-json", required=True)
    run_workflow_parser.add_argument("--request-id", help="caller-owned idempotency key")
    validate_plan_parser = subparsers.add_parser("validate-plan", help="preflight an Agent plan")
    validate_plan_parser.add_argument("--plan-token", required=True)
    confirm_plan_parser = subparsers.add_parser("confirm-plan", help="confirm one exact Agent plan")
    confirm_plan_parser.add_argument("--plan-token", required=True)
    confirm_plan_parser.add_argument("--plan-digest", required=True)
    confirm_plan_parser.add_argument("--task-id", required=True)
    confirm_plan_parser.add_argument("--timeout-sec", type=float)
    execute_plan_parser = subparsers.add_parser("execute-plan", help="execute one confirmed Agent plan")
    execute_plan_parser.add_argument("--plan-token", required=True)
    execute_plan_parser.add_argument("--confirmation-token", required=True)
    execute_plan_parser.add_argument("--task-id", required=True)
    execute_plan_parser.add_argument("--timeout-sec", type=float)
    _add_agent_terminal_expectation(execute_plan_parser)
    cancel_plan_parser = subparsers.add_parser("cancel-plan", help="cancel an active Agent plan by task ID")
    cancel_plan_parser.add_argument("--task-id", required=True)
    _add_agent_terminal_expectation(cancel_plan_parser)
    start_game_parser = subparsers.add_parser("start-game", help="start one visual game")
    start_game_parser.add_argument("game")
    start_game_parser.add_argument("--request-id", required=True)
    game_result_parser = subparsers.add_parser("game-result", help="query one visual game request")
    game_result_parser.add_argument("--request-id", required=True)
    return parser


def _add_skill_parameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-name")
    parser.add_argument("--stand-off-distance-m", dest="stand_off_distance_m", type=float)
    parser.add_argument("--container-name")
    parser.add_argument("--place-name")
    parser.add_argument("--motion-direction")
    parser.add_argument("--motion-distance", type=float)
    parser.add_argument("--arm-side")
    parser.add_argument("--imitation-duration-sec", type=float)
    parser.add_argument("--direction")
    parser.add_argument("--distance", type=float)
    parser.add_argument("--degree", type=float)
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--yaw", type=float)
    parser.add_argument("--timeout-sec", type=float)


def _add_agent_terminal_expectation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--plan-digest", required=True)
    parser.add_argument("--registry-epoch", required=True)
    parser.add_argument("--registry-generation", required=True, type=int)
    parser.add_argument("--registry-digest", required=True)
    parser.add_argument("--expected-step-count", required=True, type=int)


def _run_catalog_command(args: argparse.Namespace, context) -> dict:
    from robot_skill_cli.catalog import describe_skill, list_poses, list_skills

    if args.command == "list-skills":
        return list_skills(context.view)
    if args.command == "list-games":
        from robot_skill_cli.catalog import list_games

        return list_games(context.game_view)
    if args.command == "describe-game":
        from robot_skill_cli.catalog import describe_game

        return describe_game(context.game_view, args.game)
    if args.command == "describe":
        return describe_skill(context.view, args.skill)
    if args.command == "list-poses":
        return list_poses(context.view)
    raise _CliArgumentError(f"unsupported command: {args.command}")


def _create_bridge(transport):
    from robot_skill_cli.ros_bridge import RosBridge

    return RosBridge(
        status_service=transport.status_service,
        snapshot_service=transport.snapshot_service,
        reload_service=transport.reload_service,
        validate_skill_service=transport.validate_skill_service,
        skill_action=transport.skill_action_name,
        primitive_action=transport.primitive_action_name,
        validate_primitive_service=transport.validate_primitive_service,
        plan_service=transport.plan_service,
        validate_plan_service=transport.validate_plan_service,
        confirm_plan_service=transport.confirm_plan_service,
        execute_plan_action=transport.execute_plan_action,
        start_visual_game_service=transport.start_visual_game_service,
        get_visual_game_result_service=transport.get_visual_game_result_service,
    )


def _validate_schema(skill: dict[str, Any], args: argparse.Namespace) -> None:
    parameters = skill["parameters"]
    properties = parameters["properties"]
    required = set(parameters["required"])
    values = {
        "target_name": getattr(args, "target_name", None),
        "stand_off_distance_m": getattr(args, "stand_off_distance_m", None),
        "container_name": getattr(args, "container_name", None),
        "place_name": getattr(args, "place_name", None),
        "motion_direction": getattr(args, "motion_direction", None),
        "motion_distance": getattr(args, "motion_distance", None),
        "arm_side": getattr(args, "arm_side", None),
        "imitation_duration_sec": getattr(args, "imitation_duration_sec", None),
        "direction": getattr(args, "direction", None),
        "distance": getattr(args, "distance", None),
        "degree": getattr(args, "degree", None),
        "x": getattr(args, "x", None),
        "y": getattr(args, "y", None),
        "yaw": getattr(args, "yaw", None),
    }
    for name, value in values.items():
        if name not in properties:
            if value not in (None, "", 0, 0.0):
                raise _CliArgumentError(f"{name} is not accepted by skill {skill['name']}")
            continue
        schema = properties[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            if name in required:
                raise _CliArgumentError(f"{name} is required for skill {skill['name']}")
            if value is None:
                continue
        if schema["type"] == "string" and "enum" in schema:
            normalized_value = value.strip().lower()
            if normalized_value not in schema["enum"]:
                raise _CliArgumentError(f"{name} must be one of: {', '.join(schema['enum'])}")
            setattr(args, name, normalized_value)
        if schema["type"] == "number" and (
            isinstance(value, bool) or not math.isfinite(value) or value <= schema.get("exclusiveMinimum", -math.inf)
        ):
            raise _CliArgumentError(f"{name} must be a finite number greater than zero")


def _contract_schema_version(skill: dict[str, Any]) -> int:
    version = skill.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version not in {1, 2}:
        raise _CliArgumentError("skill contract schema_version must be 1 or 2")
    return version


def _workflow_steps_with_schema_versions(workflow_steps: list[dict[str, Any]], context) -> list[dict[str, Any]]:
    normalized = []
    for step in workflow_steps:
        if not isinstance(step, dict):
            raise _CliArgumentError("each workflow step must be an object")
        common_fields = {
            "schema_version",
            "skill_name",
            "target_name",
            "container_name",
            "place_name",
            "motion_direction",
            "motion_distance",
            "timeout_sec",
        }
        allowed_fields = common_fields | _NAVIGATION_WORKFLOW_FIELDS
        unknown_fields = set(step) - allowed_fields
        if unknown_fields:
            raise _CliArgumentError(f"workflow step contains unknown fields: {', '.join(sorted(unknown_fields))}")
        if "schema_version" in step:
            # The Agent plan boundary compares explicit versions against its
            # snapshot. Do not rewrite a submitted mismatch at the CLI edge.
            normalized.append(step)
            continue
        if _NAVIGATION_WORKFLOW_FIELDS.intersection(step):
            raise _CliArgumentError("navigation typed workflow steps require explicit schema_version")
        normalized.append({**step, "schema_version": 1})
    return normalized


def _validate_timeout(
    status: dict[str, Any], timeout_sec: float | None, *, skill_timeout_cap: float | None = None
) -> float:
    default_timeout = status["default_skill_timeout_sec"]
    if skill_timeout_cap is not None:
        if not math.isfinite(skill_timeout_cap) or skill_timeout_cap <= 0.0:
            raise _CliArgumentError("skill timeout cap must be finite and positive")
        default_timeout = min(default_timeout, skill_timeout_cap)
    effective_timeout = default_timeout if timeout_sec is None else timeout_sec
    if not math.isfinite(effective_timeout) or effective_timeout <= 0.0:
        raise _CliArgumentError("timeout_sec must be a finite number greater than zero")
    if effective_timeout > status["task_budget_sec"]:
        raise _CliArgumentError("timeout_sec must not exceed the Gateway task budget")
    if skill_timeout_cap is not None and effective_timeout > skill_timeout_cap:
        raise _CliArgumentError("timeout_sec must not exceed the skill timeout cap")
    return effective_timeout


def _prepare_request(
    args: argparse.Namespace, context, bridge
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    from embodied_common.skill_request import canonical_skill_payload, skill_payload_hash
    from robot_skill_cli.catalog import capability_view_from_snapshot, describe_skill

    status = bridge.get_status(
        task_id="",
        payload_hash="",
        timeout_sec=_status_preflight_timeout(context),
    )
    snapshot = bridge.get_skill_snapshot(
        registry_epoch=status["registry_epoch"],
        generation=status["registry_generation"],
        timeout_sec=status["rpc_timeout_sec"],
    )
    try:
        runtime_view = capability_view_from_snapshot(snapshot, status)
    except ValueError as exc:
        error_code, _separator, message = str(exc).partition(":")
        raise _CommandError(
            error_code or "SKILL_SNAPSHOT_DIGEST_MISMATCH",
            message.strip() or str(exc),
            exit_code=EXIT_GATEWAY_REJECTED,
        ) from exc
    skill = describe_skill(runtime_view, args.skill)
    _validate_schema(skill, args)
    skill_timeout_cap = skill.get("timeout_sec")
    effective_timeout = _validate_timeout(
        status,
        args.timeout_sec,
        skill_timeout_cap=float(skill_timeout_cap) if skill_timeout_cap is not None else None,
    )
    capability = next((item for item in status["capabilities"] if item["name"] == skill["name"]), None)
    if capability is None or not capability["ready"]:
        reason = capability["reason"] if capability is not None and capability["reason"] else "CAPABILITY_NOT_READY"
        reason = str(reason)
        error_code, separator, _message = reason.partition(":")
        error_code = error_code.strip()
        if not error_code:
            error_code = "CAPABILITY_NOT_READY"
            reason = error_code
        elif not separator:
            reason = error_code
        raise _CommandError(error_code, reason, exit_code=EXIT_GATEWAY_REJECTED)
    payload = canonical_skill_payload(
        skill["name"],
        schema_version=_contract_schema_version(skill),
        target_name=args.target_name,
        container_name=args.container_name,
        place_name=args.place_name,
        motion_direction=args.motion_direction,
        motion_distance=0.0 if args.motion_distance is None else args.motion_distance,
        arm_side=getattr(args, "arm_side", None),
        imitation_duration_sec=getattr(args, "imitation_duration_sec", None),
        direction=args.direction,
        distance=(
            float(getattr(args, "stand_off_distance_m", None))
            if getattr(args, "stand_off_distance_m", None) is not None
            else (0.0 if args.distance is None else args.distance)
        ),
        degree=0.0 if args.degree is None else args.degree,
        x=args.x,
        y=args.y,
        yaw=args.yaw,
        timeout_sec=effective_timeout,
        default_timeout_sec=status["default_skill_timeout_sec"],
    )
    validation = bridge.validate_skill(
        payload,
        registry_identity=(
            status.get("registry_epoch", ""),
            status.get("registry_generation", 0),
            status.get("registry_digest", ""),
        ),
        timeout_sec=status["rpc_timeout_sec"],
    )
    actual_identity = (
        validation.get("actual_registry_epoch", ""),
        validation.get("actual_registry_generation", 0),
        validation.get("actual_registry_digest", ""),
    )
    if any(actual_identity) and actual_identity != (
        status.get("registry_epoch", ""),
        status.get("registry_generation", 0),
        status.get("registry_digest", ""),
    ):
        raise _CommandError(
            "SKILL_REGISTRY_VERSION_MISMATCH",
            "Gateway validation used a different registry identity",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    if not validation["allowed"]:
        raise _CommandError(
            validation.get("error_code") or "SAFETY_REJECTED",
            validation["reason"],
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    return payload, skill_payload_hash(payload), status, validation


def _run_validate(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    payload, payload_hash, _status, validation = _prepare_request(args, context, bridge)
    return {
        "allowed": True,
        "reason": validation["reason"],
        "payload": payload,
        "payload_hash": payload_hash,
    }


def _run_reload_catalog(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    request_id = args.request_id.strip()
    if not request_id:
        raise _CliArgumentError("request_id must be non-empty")
    result = bridge.reload_skill_catalog(
        request_id=request_id,
        force=bool(args.force),
        timeout_sec=max(_plan_rpc_timeout(context), _CATALOG_RELOAD_TIMEOUT_FLOOR_SEC),
    )
    if not result["success"]:
        raise _CommandError(
            result["error_code"] or "SKILL_SCHEMA_INVALID",
            result["message"] or "skill catalog reload failed",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    return result


def _plan_rpc_timeout(context) -> float:
    timeout_sec = context.view["timeout_policy"]["rpc_timeout_sec"]
    if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
        raise _CliArgumentError("configured rpc_timeout_sec must be a finite number greater than zero")
    # Agent plan services compose multiple internal RPCs (Gateway status,
    # snapshot/Safety validation). The outer client deadline must not expire
    # before those bounded inner calls can converge.
    return max(timeout_sec, _AGENT_CONTROL_TIMEOUT_FLOOR_SEC)


def _status_preflight_timeout(context) -> float:
    return max(_plan_rpc_timeout(context), _STATUS_PREFLIGHT_TIMEOUT_FLOOR_SEC)


def _raise_plan_error(result: dict[str, Any], *, default_code: str, exit_code: int) -> None:
    if result.get("success", result.get("allowed", result.get("confirmed", False))):
        return
    raise _CommandError(
        str(result.get("error_code") or default_code),
        str(result.get("message") or result.get("reason") or default_code),
        exit_code=exit_code,
    )


def _run_plan_workflow(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    request_id = args.request_id.strip()
    raw_command = args.raw_command.strip()
    if not request_id:
        raise _CliArgumentError("request_id must be non-empty")
    if not raw_command:
        raise _CliArgumentError("raw_command must be non-empty")
    try:
        workflow_steps = json.loads(args.workflow_json)
    except json.JSONDecodeError as exc:
        raise _CliArgumentError("workflow_json must be valid JSON") from exc
    if not isinstance(workflow_steps, list) or not 1 <= len(workflow_steps) <= 16:
        raise _CliArgumentError("workflow_json must contain between 1 and 16 steps")
    workflow_steps = _workflow_steps_with_schema_versions(workflow_steps, context)
    result = bridge.plan_agent_command(
        request_id=request_id,
        raw_command=raw_command,
        workflow_steps=workflow_steps,
        timeout_sec=_plan_rpc_timeout(context),
    )
    code = str(result.get("error_code") or "SKILL_SCHEMA_INVALID")
    _raise_plan_error(result, default_code=code, exit_code=_agent_error_exit_code(code))
    return result


def _run_validate_plan(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    result = bridge.validate_agent_plan(plan_token=args.plan_token, timeout_sec=_plan_rpc_timeout(context))
    code = str(result.get("error_code") or "SKILL_SCHEMA_INVALID")
    _raise_plan_error(result, default_code=code, exit_code=_agent_error_exit_code(code))
    return result


def _run_confirm_plan(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    task_id = args.task_id.strip()
    if not task_id:
        raise _CliArgumentError("task_id must be non-empty")
    timeout_sec = _status_preflight_timeout(context)
    status = bridge.get_status(task_id="", payload_hash="", timeout_sec=timeout_sec)
    task_budget_sec = status["task_budget_sec"] if args.timeout_sec is None else args.timeout_sec
    if not math.isfinite(task_budget_sec) or task_budget_sec <= 0.0 or task_budget_sec > status["task_budget_sec"]:
        raise _CliArgumentError("timeout_sec must be finite, positive, and within the Gateway task budget")
    result = bridge.confirm_agent_plan(
        plan_token=args.plan_token,
        plan_digest=args.plan_digest,
        task_id=task_id,
        status=status,
        task_budget_sec=task_budget_sec,
        timeout_sec=timeout_sec,
    )
    code = str(result.get("error_code") or "SKILL_REQUEST_ID_CONFLICT")
    _raise_plan_error(result, default_code=code, exit_code=_agent_error_exit_code(code))
    return {"task_id": task_id, "registry": status, **result}


def _public_agent_plan_result(result) -> dict[str, Any]:
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


def _agent_plan_payload_hash(args: argparse.Namespace, timeout_sec: float) -> str:
    payload = {
        "schema_version": 1,
        "plan_token": args.plan_token,
        "confirmation_token": args.confirmation_token,
        "task_id": args.task_id.strip(),
        "timeout_sec": float(timeout_sec),
        "plan_id": args.plan_id,
        "plan_digest": args.plan_digest,
        "registry_epoch": args.registry_epoch,
        "registry_generation": int(args.registry_generation),
        "registry_digest": args.registry_digest,
        "expected_step_count": int(args.expected_step_count),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _agent_feedback_event(task_id: str, payload_hash: str, feedback: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event": "feedback",
        "task_id": task_id,
        "payload_hash": payload_hash,
        "data": {
            "state": feedback["state"],
            "detail": feedback["detail"],
            "current_skill": feedback["current_skill"],
            "workflow_step_index": feedback["workflow_step_index"],
        },
    }


def _agent_result_event(task_id: str, payload_hash: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "event": "result", "task_id": task_id, "payload_hash": payload_hash, "data": data}


def _converge_unknown_agent_goal(bridge, task_id: str, rpc_timeout: float) -> dict[str, Any] | None:
    cancel_needed = True

    def cancel() -> None:
        nonlocal cancel_needed
        try:
            response = bridge.cancel_agent_plan(task_id, timeout_sec=rpc_timeout)
            cancel_needed = not bool(response.get("accepted"))
        except Exception:
            cancel_needed = True

    cancel()
    deadline = time.monotonic() + rpc_timeout
    while time.monotonic() < deadline:
        try:
            terminal = bridge.get_agent_plan_result(task_id, timeout_sec=rpc_timeout)
        except Exception:
            terminal = None
        try:
            status = int(terminal.get("status", 0)) if terminal is not None else 0
        except (TypeError, ValueError):
            status = 0
        # action_msgs/GoalStatus terminal values: succeeded=4, canceled=5, aborted=6.
        if status in TERMINAL_GOAL_STATUSES:
            return terminal
        if cancel_needed:
            cancel()
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(min(0.02, remaining))
    return None


def _unknown_agent_result() -> dict[str, Any]:
    return {
        "success": False,
        "plan_id": "",
        "plan_digest": "",
        "workflow_digest": "",
        "completed_step_count": 0,
        "error_code": "SKILL_CANCEL_TIMEOUT",
        "message": "robot stop state is unknown",
        "actual_registry_epoch": "",
        "actual_registry_generation": 0,
        "actual_registry_digest": "",
    }


def _rejected_agent_result() -> dict[str, Any]:
    return {
        "success": False,
        "plan_id": "",
        "plan_digest": "",
        "workflow_digest": "",
        "completed_step_count": 0,
        "error_code": "GOAL_REJECTED",
        "message": "agent plan goal was rejected; the server never created this goal",
        "actual_registry_epoch": "",
        "actual_registry_generation": 0,
        "actual_registry_digest": "",
    }


def _agent_terminal_expectation(args: argparse.Namespace) -> dict[str, Any]:
    expectation = {
        "plan_id": str(args.plan_id).strip(),
        "plan_digest": str(args.plan_digest).strip(),
        "registry_epoch": str(args.registry_epoch).strip(),
        "registry_generation": int(args.registry_generation),
        "registry_digest": str(args.registry_digest).strip(),
        "step_count": int(args.expected_step_count),
    }
    if not all(expectation[key] for key in ("plan_id", "plan_digest", "registry_epoch", "registry_digest")):
        raise _CliArgumentError("plan and registry identity fields must be non-empty")
    if expectation["registry_generation"] <= 0:
        raise _CliArgumentError("registry_generation must be positive")
    if not 1 <= expectation["step_count"] <= 16:
        raise _CliArgumentError("expected_step_count must be between 1 and 16")
    return expectation


def _agent_terminal_exit_code(terminal: dict[str, Any]) -> int:
    status = int(terminal.get("status", 0))
    data = terminal.get("result", {}) or {}
    success = bool(data.get("success"))
    error_code = str(data.get("error_code", ""))
    if status == 4 and success and not error_code:
        return EXIT_SUCCESS
    if status == 5 and not success and error_code == "SKILL_CANCELLED":
        return _agent_error_exit_code(error_code)
    if status == 6 and not success and error_code:
        return _agent_error_exit_code(error_code)
    return 15


def _validated_agent_terminal(terminal: dict[str, Any], expectation: dict[str, Any]) -> dict[str, Any] | None:
    data = terminal.get("result", {}) or {}
    return dict(data) if classify_agent_terminal(terminal.get("status", 0), data, expectation) is not None else None


def _run_execute_plan(args: argparse.Namespace, context, bridge) -> _CommandExit:
    task_id = args.task_id.strip()
    if not task_id:
        raise _CliArgumentError("task_id must be non-empty")
    expectation = _agent_terminal_expectation(args)
    rpc_timeout = _plan_rpc_timeout(context)
    timeout_sec = args.timeout_sec
    payload_hash = ""
    output_lock = threading.Lock()
    terminal_written = False

    def emit_feedback(feedback: dict[str, Any]) -> None:
        nonlocal terminal_written
        with output_lock:
            if not terminal_written:
                print(json_dumps(_agent_feedback_event(task_id, payload_hash, feedback)), flush=True)

    def emit_result(data: dict[str, Any]) -> None:
        nonlocal terminal_written
        with output_lock:
            if terminal_written:
                return
            terminal_written = True
            print(json_dumps(_agent_result_event(task_id, payload_hash, data)), flush=True)

    interrupted = {"signal": None}
    interrupt_event = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        if interrupted["signal"] is None:
            interrupted["signal"] = signum
        interrupt_event.set()

    def converge_or_unknown() -> _CommandExit:
        terminal = _converge_unknown_agent_goal(bridge, task_id, rpc_timeout)
        data = _validated_agent_terminal(terminal, expectation) if terminal is not None else None
        if data is None:
            emit_result(_unknown_agent_result())
            return _CommandExit(15)
        emit_result(data)
        if classify_agent_terminal(terminal["status"], data, expectation) == "unknown":
            return _CommandExit(15)
        return _CommandExit(_agent_terminal_exit_code(terminal))

    previous_handlers: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, _handle_signal)
        try:
            status = bridge.get_status(task_id=task_id, payload_hash="", timeout_sec=_status_preflight_timeout(context))
        except Exception:
            return converge_or_unknown()
        timeout_sec = status["task_budget_sec"] if timeout_sec is None else timeout_sec
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0 or timeout_sec > status["task_budget_sec"]:
            raise _CliArgumentError("timeout_sec must be finite, positive, and within the Gateway task budget")
        payload_hash = _agent_plan_payload_hash(args, timeout_sec)
        if interrupt_event.is_set():
            return converge_or_unknown()
        deadline = time.monotonic() + rpc_timeout
        server_ready = False
        try:
            while not interrupt_event.is_set() and time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                server_ready = bridge.wait_for_execute_plan_server(timeout_sec=min(0.02, remaining))
                if server_ready:
                    break
                time.sleep(min(0.01, remaining))
        except Exception:
            server_ready = False
        if interrupt_event.is_set():
            return converge_or_unknown()
        if not server_ready:
            return converge_or_unknown()
        if interrupt_event.is_set():
            return converge_or_unknown()
        try:
            goal_future = bridge.send_agent_plan_goal(
                plan_token=args.plan_token,
                confirmation_token=args.confirmation_token,
                task_id=task_id,
                timeout_sec=timeout_sec,
                feedback_callback=emit_feedback,
            )
        except Exception:
            return converge_or_unknown()
        try:
            goal_ready = bridge.wait_future(
                goal_future,
                timeout_sec=rpc_timeout,
                interrupt_event=interrupt_event,
            )
        except Exception:
            return converge_or_unknown()
        if not goal_ready:
            return converge_or_unknown()
        try:
            goal_handle = goal_future.result()
        except Exception:
            return converge_or_unknown()
        if goal_handle is None or not goal_handle.accepted:
            if not interrupt_event.is_set():
                emit_result(_rejected_agent_result())
                return _CommandExit(13)
            return converge_or_unknown()
        try:
            result_future = goal_handle.get_result_async()
        except Exception:
            return converge_or_unknown()
        deadline = time.monotonic() + timeout_sec + rpc_timeout
        while not result_future.done():
            if interrupt_event.is_set():
                return converge_or_unknown()
            if time.monotonic() >= deadline:
                return converge_or_unknown()
            time.sleep(0.02)
        try:
            result = result_future.result()
            data = _public_agent_plan_result(result.result)
            terminal = {"status": int(result.status), "result": data}
        except Exception:
            return converge_or_unknown()
        if int(terminal.get("status", 0)) not in TERMINAL_GOAL_STATUSES:
            return converge_or_unknown()
        validated = _validated_agent_terminal(terminal, expectation)
        if validated is None:
            emit_result(_unknown_agent_result())
            return _CommandExit(15)
        emit_result(validated)
        if classify_agent_terminal(terminal["status"], validated, expectation) == "unknown":
            return _CommandExit(15)
        return _CommandExit(_agent_terminal_exit_code(terminal))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _run_cancel_plan(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    task_id = args.task_id.strip()
    if not task_id:
        raise _CliArgumentError("task_id must be non-empty")
    expectation = _agent_terminal_expectation(args)
    timeout_sec = _plan_rpc_timeout(context)
    cancel_failed = False
    cancel = {"accepted": False}

    def cancel_once() -> None:
        nonlocal cancel, cancel_failed
        try:
            cancel = bridge.cancel_agent_plan(task_id, timeout_sec=timeout_sec)
            cancel_failed = False
        except Exception:
            cancel = {"accepted": False}
            cancel_failed = True

    cancel_once()
    result_observed = False
    terminal = None
    deadline = time.monotonic() + context.view["timeout_policy"]["task_budget_sec"] + timeout_sec
    while time.monotonic() < deadline:
        try:
            terminal = bridge.get_agent_plan_result(task_id, timeout_sec=timeout_sec)
            result_observed = True
        except Exception:
            terminal = None
        try:
            status = int(terminal.get("status", 0)) if terminal is not None else 0
        except (TypeError, ValueError):
            status = 0
        if status in TERMINAL_GOAL_STATUSES:
            break
        if not cancel.get("accepted"):
            cancel_once()
        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(min(0.02, remaining))
    if terminal is not None:
        validated = _validated_agent_terminal(terminal, expectation)
        if validated is not None:
            classification = classify_agent_terminal(terminal["status"], validated, expectation)
            if classification == "unknown":
                raise _CommandError(
                    str(validated["error_code"]),
                    str(validated.get("message") or "robot stop state is unknown"),
                    exit_code=15,
                )
            return {
                "task_id": task_id,
                "accepted": bool(cancel["accepted"]),
                "terminal": True,
                "goal_status": int(terminal["status"]),
                "result": validated,
            }
    if result_observed:
        raise _CommandError("SKILL_CANCEL_TIMEOUT", "robot stop state is unknown", exit_code=15)
    if cancel_failed:
        raise _CommandError("SKILL_CANCEL_TIMEOUT", "robot stop state is unknown", exit_code=15)
    if not cancel["accepted"]:
        raise _CommandError(
            "GOAL_NOT_FOUND", "agent plan goal was not active and no terminal proof exists", exit_code=13
        )
    raise _CommandError("SKILL_CANCEL_TIMEOUT", "robot stop state is unknown", exit_code=15)


def _identity_status(bridge, task_id: str, payload_hash: str, timeout_sec: float) -> dict[str, Any]:
    return bridge.get_status(task_id=task_id, payload_hash=payload_hash, timeout_sec=timeout_sec)


def _public_result(result) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "error_code": str(result.error_code),
        "message": str(result.message),
        "executed_step_count": len(result.executed_primitives),
    }


def _print_result(task_id: str, payload_hash: str, data: dict[str, Any]) -> None:
    print(json_dumps(result_event(task_id, payload_hash, **data)), flush=True)


def _result_exit_code(data: dict[str, Any], *, agent: bool = False) -> int:
    if data.get("success") is True:
        return EXIT_SUCCESS
    if agent:
        return _agent_error_exit_code(str(data.get("error_code") or "CAPABILITY_NOT_READY"))
    if "TIMEOUT" in data["error_code"]:
        return EXIT_TIMEOUT
    return EXIT_GATEWAY_REJECTED


def _cancel_pending_task(bridge, task_id: str, rpc_timeout: float) -> bool:
    deadline = time.monotonic() + rpc_timeout
    cancel_confirmed = False
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        try:
            status = bridge.get_status(task_id=task_id, payload_hash="", timeout_sec=remaining)
            if status["request_error_code"]:
                return False
            if status["request_state"] == "terminal":
                return True
            if status["request_state"] == "active" or not cancel_confirmed:
                cancel = bridge.cancel_task(task_id, timeout_sec=remaining)
                cancel_confirmed = bool(cancel["accepted"])
        except Exception as exc:
            import sys

            sys.stderr.write(f"cancel polling failed: {exc}\n")
            return False
        time.sleep(0.02)
    return False


def _run_execute(args: argparse.Namespace, context, bridge) -> _CommandExit:
    task_id = args.task_id.strip()
    if not task_id:
        raise _CliArgumentError("task_id must be non-empty")
    payload, payload_hash, status, _validation = _prepare_request(args, context, bridge)
    payload["_registry_epoch"] = status.get("registry_epoch", "")
    payload["_registry_generation"] = status.get("registry_generation", 0)
    payload["_registry_digest"] = status.get("registry_digest", "")
    rpc_timeout = status["rpc_timeout_sec"]
    output_lock = threading.Lock()
    terminal_output = {"written": False}

    def emit_feedback(feedback: dict[str, str]) -> None:
        with output_lock:
            if terminal_output["written"]:
                return
            print(
                json_dumps(
                    feedback_event(
                        task_id,
                        payload_hash,
                        feedback["state"],
                        feedback["detail"],
                    )
                ),
                flush=True,
            )

    def emit_result(data: dict[str, Any]) -> None:
        with output_lock:
            if terminal_output["written"]:
                return
            terminal_output["written"] = True
            _print_result(task_id, payload_hash, data)

    def transport_failure(message: str) -> _CommandExit:
        if _cancel_pending_task(bridge, task_id, rpc_timeout):
            data = {
                "success": False,
                "error_code": "ROS_UNAVAILABLE",
                "message": message,
                "executed_step_count": 0,
            }
            exit_code = EXIT_ROS_UNAVAILABLE
        else:
            data = {
                "success": False,
                "error_code": "SKILL_CANCEL_TIMEOUT",
                "message": "robot stop state is unknown",
                "executed_step_count": 0,
            }
            exit_code = EXIT_TIMEOUT
        emit_result(data)
        return _CommandExit(exit_code)

    def read_terminal_result(result_future) -> dict[str, Any] | None:
        try:
            wrapped_result = result_future.result()
            if wrapped_result is None or wrapped_result.result is None:
                return None
            return _public_result(wrapped_result.result)
        except Exception:
            return None

    identity_status = _identity_status(bridge, task_id, payload_hash, rpc_timeout)
    if identity_status["request_error_code"]:
        error_code = identity_status["request_error_code"]
        emit_result(
            {
                "success": False,
                "error_code": error_code,
                "message": error_code,
                "executed_step_count": 0,
            },
        )
        return _CommandExit(EXIT_GATEWAY_REJECTED)
    if not bridge.wait_for_skill_server(timeout_sec=rpc_timeout):
        raise _CommandError("SERVER_UNAVAILABLE", "skill action server unavailable", exit_code=EXIT_ROS_UNAVAILABLE)

    interrupted = {"signal": None}

    def handle_signal(signum, _frame) -> None:
        interrupted["signal"] = signum

    previous_handlers = {
        signal.SIGINT: signal.signal(signal.SIGINT, handle_signal),
        signal.SIGTERM: signal.signal(signal.SIGTERM, handle_signal),
    }
    try:
        goal_future = bridge.send_skill_goal(
            payload,
            task_id=task_id,
            feedback_callback=emit_feedback,
        )
        if not bridge.wait_future(goal_future, timeout_sec=rpc_timeout):
            converged = _cancel_pending_task(bridge, task_id, rpc_timeout)
            if converged and interrupted["signal"] is not None:
                data = {
                    "success": False,
                    "error_code": "SKILL_CANCELLED",
                    "message": "task reached terminal after signal cancellation",
                    "executed_step_count": 0,
                }
                exit_code = EXIT_SIGINT if interrupted["signal"] == signal.SIGINT else EXIT_SIGTERM
            elif converged:
                data = {
                    "success": False,
                    "error_code": "RESULT_TIMEOUT",
                    "message": "task reached terminal after goal response timeout",
                    "executed_step_count": 0,
                }
                exit_code = EXIT_TIMEOUT
            else:
                data = {
                    "success": False,
                    "error_code": "SKILL_CANCEL_TIMEOUT",
                    "message": "robot stop state is unknown",
                    "executed_step_count": 0,
                }
                exit_code = EXIT_TIMEOUT
            emit_result(data)
            return _CommandExit(exit_code)
        try:
            goal_handle = goal_future.result()
        except Exception:
            return transport_failure("goal response unavailable")
        if goal_handle is None or not goal_handle.accepted:
            rejected_status = bridge.get_status(
                task_id=task_id,
                payload_hash=payload_hash,
                timeout_sec=rpc_timeout,
            )
            error_code = rejected_status["request_error_code"] or "GOAL_REJECTED"
            emit_result(
                {
                    "success": False,
                    "error_code": error_code,
                    "message": error_code,
                    "executed_step_count": 0,
                },
            )
            return _CommandExit(EXIT_GATEWAY_REJECTED)

        try:
            result_future = goal_handle.get_result_async()
        except Exception:
            return transport_failure("result request unavailable")
        deadline = time.monotonic() + payload["timeout_sec"] + rpc_timeout
        while not result_future.done():
            if interrupted["signal"] is not None:
                signal_number = interrupted["signal"]
                if not bridge.cancel_goal(goal_handle, result_future, timeout_sec=rpc_timeout):
                    data = {
                        "success": False,
                        "error_code": "SKILL_CANCEL_TIMEOUT",
                        "message": "robot stop state is unknown",
                        "executed_step_count": 0,
                    }
                    emit_result(data)
                    return _CommandExit(EXIT_TIMEOUT)
                data = read_terminal_result(result_future)
                if data is None:
                    return transport_failure("terminal result unavailable")
                emit_result(data)
                exit_code = EXIT_SIGINT if signal_number == signal.SIGINT else EXIT_SIGTERM
                return _CommandExit(exit_code)
            if time.monotonic() >= deadline:
                if bridge.cancel_goal(goal_handle, result_future, timeout_sec=rpc_timeout):
                    data = read_terminal_result(result_future)
                    if data is None:
                        return transport_failure("terminal result unavailable")
                else:
                    data = {
                        "success": False,
                        "error_code": "SKILL_CANCEL_TIMEOUT",
                        "message": "robot stop state is unknown",
                        "executed_step_count": 0,
                    }
                emit_result(data)
                return _CommandExit(EXIT_TIMEOUT)
            time.sleep(0.02)

        data = read_terminal_result(result_future)
        if data is None:
            return transport_failure("terminal result unavailable")
        emit_result(data)
        return _CommandExit(_result_exit_code(data))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _run_workflow(args: argparse.Namespace, context, bridge) -> _CommandExit:
    """Run the complete Agent workflow without returning to an LLM between phases."""
    try:
        workflow_steps = json.loads(args.workflow_json)
    except json.JSONDecodeError as exc:
        raise _CliArgumentError(f"workflow-json must be valid JSON: {exc.msg}") from exc
    if not isinstance(workflow_steps, list) or not workflow_steps:
        raise _CliArgumentError("workflow-json must be a non-empty array")

    from robot_skill_cli.interactive_control import InteractiveControlError, InteractiveController

    controller = InteractiveController(
        bridge,
        timeout_policy=context.view["timeout_policy"],
        execution_mode=IMMEDIATE_AFTER_PRESENTATION,
    )
    interrupt_event = threading.Event()

    def _handle_signal(signum, _frame) -> None:
        interrupt_event.set()
        controller.request_stop()

    def _notify_authorized(confirmation: dict[str, Any]) -> None:
        if os.environ.get("IBROBOT_HERMES_LIFECYCLE_SPEECH") != "1":
            return
        try:
            from robot_skill_cli.hermes_lifecycle_speech import notify_plan_authorized

            notify_plan_authorized(session_id=str(confirmation["task_id"]))
        except (ImportError, KeyError, OSError, ValueError):
            pass

    previous_handlers: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, _handle_signal)
        run_kwargs = {
            "presentation_callback": lambda presentation: print(
                json_dumps(
                    {
                        "schema_version": 1,
                        "command": "run-workflow",
                        "event": "workflow_presentation",
                        "task_id": presentation["task_id"],
                        "data": presentation,
                    }
                ),
                flush=True,
            ),
            "authorization_callback": _notify_authorized,
            "stop_event": interrupt_event,
        }
        request_id = getattr(args, "request_id", None)
        if request_id is not None:
            run_kwargs["request_id"] = request_id
        terminal = controller.run(args.raw_command, workflow_steps, **run_kwargs)
        print(
            json_dumps(
                {
                    "schema_version": 1,
                    "command": "run-workflow",
                    "event": "workflow_terminal",
                    "task_id": terminal["task_id"],
                    "data": terminal,
                }
            ),
            flush=True,
        )
    except InteractiveControlError as exc:
        raise _CommandError(exc.code, str(exc), exit_code=_agent_error_exit_code(exc.code)) from exc
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    result = terminal.get("result") or terminal
    return _CommandExit(_result_exit_code(result, agent=True))


def _task_status(bridge, task_id: str, timeout_sec: float) -> dict[str, Any]:
    status = bridge.get_status(task_id=task_id, payload_hash="", timeout_sec=timeout_sec)
    if status["request_error_code"] in {"DUPLICATE_TASK_ID", "TASK_ID_CONFLICT"}:
        raise _CommandError(
            status["request_error_code"],
            "task-ID-only lookup returned an identity error",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    return status


def _run_cancel(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    task_id = args.task_id.strip()
    if not task_id:
        raise _CliArgumentError("task_id must be non-empty")
    initial_timeout = context.view["timeout_policy"]["rpc_timeout_sec"]
    status = _task_status(bridge, task_id, initial_timeout)
    if status["request_state"] == "terminal":
        return {"task_id": task_id, "already_terminal": True, "status": status}
    if status["request_state"] != "active":
        raise _CommandError("GOAL_NOT_FOUND", f"active task not found: {task_id}", exit_code=EXIT_GATEWAY_REJECTED)

    rpc_timeout = status["rpc_timeout_sec"]
    cancel_result = bridge.cancel_task(task_id, timeout_sec=rpc_timeout)
    deadline = time.monotonic() + rpc_timeout
    while time.monotonic() < deadline:
        status = _task_status(bridge, task_id, rpc_timeout)
        if status["request_state"] == "terminal":
            return {
                "task_id": task_id,
                "already_terminal": False,
                "cancel": cancel_result,
                "status": status,
            }
        if status["request_state"] != "active":
            raise _CommandError(
                "GOAL_NOT_FOUND", f"task record disappeared: {task_id}", exit_code=EXIT_GATEWAY_REJECTED
            )
        time.sleep(0.02)
    raise _CommandError(
        "SKILL_CANCEL_TIMEOUT",
        "robot stop state is unknown",
        exit_code=EXIT_TIMEOUT,
    )


def _run_start_game(args: argparse.Namespace, context, bridge, *, game: dict[str, Any] | None = None) -> dict[str, Any]:
    from robot_skill_cli.catalog import require_enabled_game

    if game is None:
        game = require_enabled_game(context.game_view, args.game)
    request_id = args.request_id.strip()
    if not request_id:
        raise _CliArgumentError("request_id must be non-empty")
    timeout_sec = context.view["timeout_policy"]["rpc_timeout_sec"]
    result = bridge.start_visual_game(
        game["name"],
        request_id=request_id,
        expected_config_digest=context.game_view["config_digest"],
        timeout_sec=timeout_sec,
    )
    if not result["accepted"]:
        raise _CommandError(
            result["error_code"] or "GAME_REJECTED",
            result["message"] or "visual game request rejected",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    if result["config_digest"] != context.game_view["config_digest"]:
        raise _CommandError(
            "CONFIG_MISMATCH",
            "local visual game configuration does not match the running gateway",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    return result


def _run_game_result(args: argparse.Namespace, context, bridge) -> dict[str, Any]:
    request_id = args.request_id.strip()
    if not request_id:
        raise _CliArgumentError("request_id must be non-empty")
    timeout_sec = context.view["timeout_policy"]["rpc_timeout_sec"]
    result = bridge.get_visual_game_result(request_id, timeout_sec=timeout_sec)
    if result["config_digest"] != context.game_view["config_digest"]:
        raise _CommandError(
            "CONFIG_MISMATCH",
            "local visual game configuration does not match the running gateway",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    if not result["found"]:
        raise _CommandError(
            result["error_code"] or "GAME_REQUEST_NOT_FOUND",
            result["message"] or f"visual game request not found: {request_id}",
            exit_code=EXIT_GATEWAY_REJECTED,
        )
    return {"request_id": request_id, **result}


def _run_runtime_command(args: argparse.Namespace, context, transport) -> dict[str, Any] | _CommandExit:
    from robot_skill_cli.ros_bridge import BridgeError

    game = None
    if args.command == "start-game":
        from robot_skill_cli.catalog import require_enabled_game

        game = require_enabled_game(context.game_view, args.game)
    bridge = _create_bridge(transport)
    if not bridge.start():
        raise BridgeError("ROS_UNAVAILABLE", "failed to initialize ROS bridge", exit_code=EXIT_ROS_UNAVAILABLE)
    try:
        if args.command == "status":
            return bridge.get_status(timeout_sec=_status_preflight_timeout(context))
        if args.command == "reload-catalog":
            return _run_reload_catalog(args, context, bridge)
        if args.command == "validate":
            return _run_validate(args, context, bridge)
        if args.command == "execute":
            return _run_execute(args, context, bridge)
        if args.command == "cancel":
            return _run_cancel(args, context, bridge)
        if args.command == "plan-workflow":
            return _run_plan_workflow(args, context, bridge)
        if args.command == "run-workflow":
            return _run_workflow(args, context, bridge)
        if args.command == "validate-plan":
            return _run_validate_plan(args, context, bridge)
        if args.command == "confirm-plan":
            return _run_confirm_plan(args, context, bridge)
        if args.command == "execute-plan":
            return _run_execute_plan(args, context, bridge)
        if args.command == "cancel-plan":
            return _run_cancel_plan(args, context, bridge)
        if args.command == "start-game":
            return _run_start_game(args, context, bridge, game=game)
        if args.command == "game-result":
            return _run_game_result(args, context, bridge)
        raise _CliArgumentError(f"unsupported command: {args.command}")
    finally:
        bridge.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    command = "unknown"
    try:
        args = parser.parse_args(argv)
        command = args.command
        from robot_skill_cli.catalog import (
            load_catalog_context,
            load_runtime_context,
            load_visual_game_context,
            load_visual_game_runtime_context,
        )

        if command in {"list-games", "describe-game"}:
            context = load_visual_game_context(config_name=args.config_name, config_path=args.config_path)
            data = _run_catalog_command(args, context)
        elif command in {"start-game", "game-result"}:
            context, transport = load_visual_game_runtime_context(
                config_name=args.config_name,
                config_path=args.config_path,
            )
            data = _run_runtime_command(args, context, transport)
        elif command in {"list-skills", "describe", "list-poses"}:
            context = load_catalog_context(config_name=args.config_name, config_path=args.config_path)
            data = _run_catalog_command(args, context)
        else:
            context, transport = load_runtime_context(config_name=args.config_name, config_path=args.config_path)
            data = _run_runtime_command(args, context, transport)
    except _CommandError as exc:
        print(json_dumps(error_envelope(command, exc.code, str(exc))))
        return exc.exit_code
    except (_CliArgumentError, FileNotFoundError, ValueError) as exc:
        print(json_dumps(error_envelope(command, getattr(exc, "code", "INVALID_ARGUMENT"), str(exc))))
        return EXIT_INVALID_INPUT
    except Exception as exc:
        from robot_skill_cli.ros_bridge import BridgeError

        if isinstance(exc, BridgeError):
            print(json_dumps(error_envelope(command, exc.code, str(exc))))
            return exc.exit_code
        raise

    if isinstance(data, _CommandExit):
        return data.exit_code
    print(json_dumps(success_envelope(command, data)))
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())

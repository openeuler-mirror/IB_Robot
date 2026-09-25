"""ROS lifecycle tests for the Agent plan boundary."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from embodied_agent.agent_plan_node import AgentPlanNode
from embodied_agent.agent_plan_store import AgentPlanError
from embodied_common.dispatch_binding import new_binding
from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_msgs.action import ExecuteAgentPlan, SkillCommand
from ibrobot_msgs.msg import SkillCapabilityStatus, WorkflowStep
from ibrobot_msgs.srv import (
    ConfirmAgentPlan,
    GetSkillGatewayStatus,
    GetSkillSnapshot,
    PlanAgentCommand,
    ValidateAgentPlan,
    ValidateSkill,
)


def _future_result(future, timeout_sec: float = 3.0):
    deadline = time.monotonic() + timeout_sec
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.done()
    return future.result()


def _workflow_request(request, *skill_names: str) -> None:
    request.workflow_steps = []
    for skill_name in skill_names:
        step = WorkflowStep()
        step.schema_version = 1
        step.skill_name = skill_name
        request.workflow_steps.append(step)


def test_navigation_steps_normalize_to_v2_and_validate_with_presence():
    node = object.__new__(AgentPlanNode)
    source = WorkflowStep()
    source.schema_version = 2
    source.skill_name = "nav_abs_coordinate"
    source.has_x = True
    source.x = 0.0
    source.has_y = True
    source.y = -2.0
    source.has_yaw = True
    source.yaw = 0.0
    catalog = SimpleNamespace(
        planner_visible_names={"nav_abs_coordinate"},
        capability_view={
            "nav_abs_coordinate": {
                "semantic_level": "atomic_operator",
                "domain": "navigation",
                "schema_version": 2,
            }
        },
    )

    steps = node._normalize_steps([source], SimpleNamespace(), catalog)
    step = steps[0]

    assert step.schema_version == 2
    assert step.x == pytest.approx(0.0)
    assert step.y == pytest.approx(-2.0)
    assert step.yaw == pytest.approx(0.0)
    message = node._to_workflow_step_message(step)
    assert message.schema_version == 2
    assert message.has_x is True and message.x == pytest.approx(0.0)

    captured = {}
    node._validate_skill_client = object()
    node._validate_skill_service = "/validate"
    node._call_service = lambda _client, request, _service: captured.setdefault("request", request)
    plan = SimpleNamespace(
        plan_id="plan-1",
        registry_epoch="epoch",
        registry_generation=1,
        registry_digest="digest",
    )
    node._validate_step(plan, step)
    validation = captured["request"]
    assert validation.schema_version == 2
    assert validation.has_x is True and validation.x == pytest.approx(0.0)
    assert validation.has_y is True and validation.y == pytest.approx(-2.0)
    assert validation.has_yaw is True and validation.yaw == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("submitted_version", "capability_version", "skill_name", "domain"),
    [
        (1, 2, "nav_abs_coordinate", "navigation"),
        (2, 1, "open_gripper_skill", "manipulation"),
    ],
)
def test_plan_rejects_step_capability_version_mismatch_without_validation_or_goal_side_effects(
    plan_rig, monkeypatch, submitted_version, capability_version, skill_name, domain
):
    monkeypatch.setattr(
        plan_rig.plan_node,
        "_catalog_view",
        lambda _status: SimpleNamespace(
            planner_visible_names={skill_name},
            capability_view={
                skill_name: {
                    "semantic_level": "skill",
                    "domain": domain,
                    "schema_version": capability_version,
                }
            },
        ),
    )
    request = PlanAgentCommand.Request()
    request.schema_version = 1
    request.request_id = f"version-mismatch-{submitted_version}-{capability_version}"
    request.raw_command = "submit a versioned workflow"
    step = WorkflowStep()
    step.schema_version = submitted_version
    step.skill_name = skill_name
    request.workflow_steps = [step]

    planned = _future_result(plan_rig.plan_client.call_async(request))

    assert planned.success is False
    assert planned.error_code == "SKILL_SCHEMA_INVALID"
    assert planned.plan.plan_token == ""
    assert plan_rig.validation_requests == []
    assert plan_rig.executed_goals == []


def test_validate_step_propagates_v1_schema_version():
    captured = {}
    node = object.__new__(AgentPlanNode)
    node._validate_skill_client = object()
    node._validate_skill_service = "/validate"
    node._call_service = lambda _client, request, _service: captured.setdefault("request", request)
    plan = SimpleNamespace(
        plan_id="plan-1",
        registry_epoch="epoch",
        registry_generation=1,
        registry_digest="digest",
    )

    node._validate_step(plan, CanonicalWorkflowStep(1, "open_gripper_skill"))

    assert captured["request"].schema_version == 1


@pytest.fixture
def plan_rig(request):
    if rclpy.ok():
        rclpy.shutdown()
    rclpy.init()
    suffix = uuid.uuid4().hex
    names = SimpleNamespace(
        action=f"/agent_{suffix}/skill",
        confirm=f"/agent_{suffix}/confirm",
        plan=f"/agent_{suffix}/plan",
        status=f"/agent_{suffix}/status",
        snapshot=f"/agent_{suffix}/snapshot",
        validate=f"/agent_{suffix}/validate",
        validate_skill=f"/agent_{suffix}/validate_skill",
    )
    mock_node = rclpy.create_node(f"agent_mock_{suffix}")
    client_node = rclpy.create_node(f"agent_client_{suffix}")
    from robot_config.loader import load_robot_config_dict
    from robot_skill_cli.catalog import compile_local_snapshot

    # Offline-complete variant of the provider-bound so101_single_arm config:
    # catalog compilation needs real joint groups without a live description.
    config_name = getattr(request, "param", "so101_single_arm_legacy")
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / f"{config_name}.yaml"
    snapshot = compile_local_snapshot(load_robot_config_dict(config_path), config_path)
    registry = ("epoch-test", 7, snapshot.registry_digest)

    executed_goals = []

    def status_callback(_request, response):
        response.schema_version = 1
        response.registry_epoch, response.registry_generation, response.registry_digest = registry
        response.default_skill_timeout_sec = 2.0
        response.task_budget_sec = 10.0
        response.rpc_timeout_sec = 1.0
        response.control_plane_ready = True
        response.control_plane_state = "READY"
        response.capabilities = []
        for skill_name in snapshot.planner_visible_skill_names:
            capability = SkillCapabilityStatus()
            capability.schema_version = 1
            capability.name = skill_name
            capability.semantic_level = snapshot.semantic_levels[skill_name]
            capability.planner_visible = True
            capability.ready = True
            response.capabilities.append(capability)
        return response

    validation_control = {"registry": registry, "allowed": True}
    validation_requests = []

    def validate_callback(request, response):
        validation_requests.append(request)
        response.allowed = validation_control["allowed"]
        response.actual_registry_epoch, response.actual_registry_generation, response.actual_registry_digest = (
            validation_control["registry"]
        )
        return response

    def snapshot_callback(_request, response):
        response.success = True
        response.registry_epoch, response.generation, response.registry_digest = registry
        response.capability_digest = snapshot.capability_digest
        response.provenance_digest = snapshot.provenance_digest
        response.source_release_digest = str(snapshot.provenance["source_release_digest"])
        response.profile_name = snapshot.profile_name
        response.snapshot_json = snapshot.snapshot_json
        return response

    def execute_callback(goal_handle):
        executed_goals.append(goal_handle.request)
        result = SkillCommand.Result()
        result.success = True
        result.message = "skill completed"
        result.actual_registry_epoch, result.actual_registry_generation, result.actual_registry_digest = registry
        result.executed_primitives = []
        goal_handle.succeed()
        return result

    mock_node.create_service(GetSkillGatewayStatus, names.status, status_callback)
    mock_node.create_service(GetSkillSnapshot, names.snapshot, snapshot_callback)
    mock_node.create_service(ValidateSkill, names.validate_skill, validate_callback)
    skill_server = ActionServer(mock_node, SkillCommand, names.action, execute_callback)
    plan_node = AgentPlanNode(
        parameter_overrides=[
            Parameter(name, value=value)
            for name, value in {
                "gateway_status_service": names.status,
                "validate_skill_service": names.validate_skill,
                "skill_catalog_snapshot_service": names.snapshot,
                "skill_action_name": names.action,
                "plan_service": names.plan,
                "validate_plan_service": names.validate,
                "confirm_plan_service": names.confirm,
                "execute_plan_action": names.action + "_agent",
                "rpc_timeout_sec": 1.0,
                "default_target_name": "demo_object",
                "default_place_name": "home",
            }.items()
        ]
    )
    executor = MultiThreadedExecutor(num_threads=6)
    for node in (mock_node, client_node, plan_node):
        executor.add_node(node)
    import threading

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    plan_client = client_node.create_client(PlanAgentCommand, names.plan)
    validate_client = client_node.create_client(ValidateAgentPlan, names.validate)
    confirm_client = client_node.create_client(ConfirmAgentPlan, names.confirm)
    execute_client = ActionClient(
        client_node,
        ExecuteAgentPlan,
        names.action + "_agent",
    )
    for client in (plan_client, validate_client, confirm_client):
        assert client.wait_for_service(timeout_sec=2.0)
    assert execute_client.wait_for_server(timeout_sec=2.0)
    try:
        yield SimpleNamespace(
            confirm_client=confirm_client,
            execute_client=execute_client,
            node=plan_node,
            plan_client=plan_client,
            plan_node=plan_node,
            registry=registry,
            validate_client=validate_client,
            validation_control=validation_control,
            validation_requests=validation_requests,
            executed_goals=executed_goals,
        )
    finally:
        executor.shutdown(timeout_sec=1.0)
        thread.join(timeout=1.0)
        skill_server.destroy()
        for node in (plan_node, client_node, mock_node):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_plan_validate_confirm_execute_and_terminal_replay(plan_rig):
    plan_request = PlanAgentCommand.Request()
    plan_request.schema_version = 1
    plan_request.request_id = "request-1"
    plan_request.raw_command = "打开夹爪"
    _workflow_request(plan_request, "open_gripper_skill")
    planned = _future_result(plan_rig.plan_client.call_async(plan_request))
    assert planned.success is True
    assert planned.plan.plan_token
    assert planned.plan.workflow_steps[0].skill_name == "open_gripper_skill"

    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    validated = _future_result(plan_rig.validate_client.call_async(validate_request))
    assert validated.allowed is True

    confirm_request = ConfirmAgentPlan.Request()
    confirm_request.schema_version = 1
    confirm_request.plan_token = planned.plan.plan_token
    confirm_request.plan_digest = planned.plan.plan_digest
    confirm_request.task_id = "agent-task-1"
    confirm_request.registry_epoch, confirm_request.registry_generation, confirm_request.registry_digest = (
        plan_rig.registry
    )
    confirm_request.task_budget_sec = 5.0
    confirmed = _future_result(plan_rig.confirm_client.call_async(confirm_request))
    assert confirmed.confirmed is True

    def execute(timeout_sec=5.0):
        goal = ExecuteAgentPlan.Goal()
        goal.schema_version = 1
        goal.plan_token = planned.plan.plan_token
        goal.confirmation_token = confirmed.confirmation_token
        goal.task_id = "agent-task-1"
        goal.timeout_sec = timeout_sec
        handle = _future_result(plan_rig.execute_client.send_goal_async(goal))
        assert handle.accepted
        return _future_result(handle.get_result_async()).result

    first = execute()
    replay = execute()
    assert first.success is True
    assert replay.success is True
    assert replay.completed_step_count == 1


def test_canceled_terminal_replay_preserves_canceled_goal_status(plan_rig):
    plan_request = PlanAgentCommand.Request()
    plan_request.schema_version = 1
    plan_request.request_id = "request-canceled-replay"
    plan_request.raw_command = "打开夹爪"
    _workflow_request(plan_request, "open_gripper_skill")
    planned = _future_result(plan_rig.plan_client.call_async(plan_request))

    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    assert _future_result(plan_rig.validate_client.call_async(validate_request)).allowed is True

    confirm_request = ConfirmAgentPlan.Request()
    confirm_request.schema_version = 1
    confirm_request.plan_token = planned.plan.plan_token
    confirm_request.plan_digest = planned.plan.plan_digest
    confirm_request.task_id = "agent-task-canceled-replay"
    confirm_request.registry_epoch, confirm_request.registry_generation, confirm_request.registry_digest = (
        plan_rig.registry
    )
    confirm_request.task_budget_sec = 5.0
    confirmed = _future_result(plan_rig.confirm_client.call_async(confirm_request))

    with plan_rig.node._store_lock:  # noqa: SLF001
        execution = plan_rig.node._store.accept_execution(  # noqa: SLF001
            plan_token=planned.plan.plan_token,
            confirmation_token=confirmed.confirmation_token,
            task_id=confirm_request.task_id,
            registry_epoch=planned.plan.registry_epoch,
            registry_generation=planned.plan.registry_generation,
            registry_digest=planned.plan.registry_digest,
            task_budget_sec=5.0,
        )
        plan_rig.node._store.mark_terminal(  # noqa: SLF001
            plan_token=planned.plan.plan_token,
            task_id=confirm_request.task_id,
            execution_token=execution.execution_token,
            terminal_code="SKILL_CANCELLED",
            terminal_message="canceled",
        )

    goal = ExecuteAgentPlan.Goal()
    goal.schema_version = 1
    goal.plan_token = planned.plan.plan_token
    goal.confirmation_token = confirmed.confirmation_token
    goal.task_id = confirm_request.task_id
    goal.timeout_sec = 5.0
    handle = _future_result(plan_rig.execute_client.send_goal_async(goal))
    replay = _future_result(handle.get_result_async())

    assert replay.status == 5
    assert replay.result.error_code == "SKILL_CANCELLED"


def test_execute_rejects_budget_changed_after_confirmation(plan_rig):
    plan_request = PlanAgentCommand.Request()
    plan_request.schema_version = 1
    plan_request.request_id = "request-budget-mismatch"
    plan_request.raw_command = "打开夹爪"
    _workflow_request(plan_request, "open_gripper_skill")
    planned = _future_result(plan_rig.plan_client.call_async(plan_request))

    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    assert _future_result(plan_rig.validate_client.call_async(validate_request)).allowed is True

    confirm_request = ConfirmAgentPlan.Request()
    confirm_request.schema_version = 1
    confirm_request.plan_token = planned.plan.plan_token
    confirm_request.plan_digest = planned.plan.plan_digest
    confirm_request.task_id = "agent-task-budget-mismatch"
    confirm_request.registry_epoch, confirm_request.registry_generation, confirm_request.registry_digest = (
        plan_rig.registry
    )
    confirm_request.task_budget_sec = 5.0
    confirmed = _future_result(plan_rig.confirm_client.call_async(confirm_request))

    goal = ExecuteAgentPlan.Goal()
    goal.schema_version = 1
    goal.plan_token = planned.plan.plan_token
    goal.confirmation_token = confirmed.confirmation_token
    goal.task_id = confirm_request.task_id
    goal.timeout_sec = 4.0
    handle = _future_result(plan_rig.execute_client.send_goal_async(goal))
    result = _future_result(handle.get_result_async()).result

    assert result.success is False
    assert result.error_code == "SKILL_REQUEST_ID_CONFLICT"
    assert result.plan_id == planned.plan.plan_id
    assert result.plan_digest == planned.plan.plan_digest
    assert (result.actual_registry_epoch, result.actual_registry_generation, result.actual_registry_digest) == (
        planned.plan.registry_epoch,
        planned.plan.registry_generation,
        planned.plan.registry_digest,
    )


def test_plan_service_returns_one_typed_ordered_multi_skill_workflow(plan_rig):
    request = PlanAgentCommand.Request()
    request.schema_version = 1
    request.request_id = "request-workflow"
    request.raw_command = "先点头，然后挥手"
    _workflow_request(request, "nod_yes", "wave_hello")

    planned = _future_result(plan_rig.plan_client.call_async(request))

    assert planned.success is True
    assert [step.skill_name for step in planned.plan.workflow_steps] == ["nod_yes", "wave_hello"]
    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    validated = _future_result(plan_rig.validate_client.call_async(validate_request))
    assert validated.allowed is True


def test_validate_plan_unexpected_error_returns_stable_failure_and_service_survives(plan_rig, monkeypatch):
    plan_request = PlanAgentCommand.Request()
    plan_request.schema_version = 1
    plan_request.request_id = "request-unexpected-validation-error"
    plan_request.raw_command = "打开夹爪"
    _workflow_request(plan_request, "open_gripper_skill")
    planned = _future_result(plan_rig.plan_client.call_async(plan_request))
    assert planned.success is True

    original_validate_step = plan_rig.plan_node._validate_step

    def fail_validation(*_args, **_kwargs):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(plan_rig.plan_node, "_validate_step", fail_validation)
    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    failed = _future_result(plan_rig.validate_client.call_async(validate_request))

    assert failed.allowed is False
    assert failed.error_code == "CAPABILITY_NOT_READY"
    assert failed.message == "agent plan validation is temporarily unavailable"

    monkeypatch.setattr(plan_rig.plan_node, "_validate_step", original_validate_step)
    recovered = _future_result(plan_rig.validate_client.call_async(validate_request))
    assert recovered.allowed is True


@pytest.mark.parametrize("plan_rig", ["lekiwi_handeye_realsense_grasp_pc"], indirect=True)
# Loads a robot config whose perception services reference a gitignored
# model bundle; see the root conftest.py.
@pytest.mark.model_bundle("grounded_sam2_swint_ogc")
def test_marker_outputs_replay_uses_hermes_pick_contract_without_hardware(plan_rig):
    """Replay the saved marker planning evidence through the Agent plan boundary only."""
    repository_root = Path(__file__).parents[3]
    evidence_path = (
        repository_root / "outputs" / "success_cloud_replay_20260806" / "current_phases_after_wait_future_fix.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["fixture"].endswith("20260717_103241_marker_1784255561600731369")
    assert evidence["pr259"]["final_candidate_id"] == 82

    request = PlanAgentCommand.Request()
    request.schema_version = 1
    request.request_id = "hermes-marker-outputs-offline"
    request.raw_command = "用 outputs 的 marker 规划结果验证 Hermes 抓取通路（不启动真机）"
    step = WorkflowStep()
    step.schema_version = 1
    step.skill_name = "pick_object"
    step.target_name = "marker"
    request.workflow_steps = [step]

    planned = _future_result(plan_rig.plan_client.call_async(request))
    assert planned.success is True
    assert planned.plan.workflow_steps[0].skill_name == "pick_object"
    assert planned.plan.workflow_steps[0].target_name == "marker"

    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    validated = _future_result(plan_rig.validate_client.call_async(validate_request))
    assert validated.allowed is True

    confirm_request = ConfirmAgentPlan.Request()
    confirm_request.schema_version = 1
    confirm_request.plan_token = planned.plan.plan_token
    confirm_request.plan_digest = planned.plan.plan_digest
    confirm_request.task_id = "hermes-marker-outputs-task"
    confirm_request.registry_epoch, confirm_request.registry_generation, confirm_request.registry_digest = (
        plan_rig.registry
    )
    confirm_request.task_budget_sec = 5.0
    confirmed = _future_result(plan_rig.confirm_client.call_async(confirm_request))
    assert confirmed.confirmed is True

    goal = ExecuteAgentPlan.Goal()
    goal.schema_version = 1
    goal.plan_token = planned.plan.plan_token
    goal.confirmation_token = confirmed.confirmation_token
    goal.task_id = confirm_request.task_id
    goal.timeout_sec = 5.0
    handle = _future_result(plan_rig.execute_client.send_goal_async(goal))
    assert handle.accepted
    result = _future_result(handle.get_result_async()).result

    assert result.success is True
    assert result.completed_step_count == 1
    assert len(plan_rig.executed_goals) == 1
    child = plan_rig.executed_goals[0]
    assert child.skill_name == "pick_object"
    assert child.target_name == "marker"
    assert child.timeout_sec == pytest.approx(0.0)
    assert child.dispatch_binding.expected_registry_digest == plan_rig.registry[2]


# Replays a capture from a recorded run. outputs/ is gitignored, so the
# file is absent on any machine that did not produce it.
@pytest.mark.repo_asset("outputs/success_cloud_replay_20260806/current_phases_after_wait_future_fix.json")
def test_robot_skill_marker_outputs_replay_reaches_real_gateway_without_hardware(tmp_path):
    """Run the Hermes-bound CLI lifecycle through the real Gateway and a replay-only pick server."""
    allocated_domain = os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID", "")
    assert allocated_domain.isdecimal() and os.environ.get("ROS_DOMAIN_ID") == allocated_domain
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    robot_skill = shutil.which("robot-skill")
    assert robot_skill is not None
    if not rclpy.ok():
        rclpy.init()

    repository_root = Path(__file__).parents[3]
    config_path = (
        repository_root / "src" / "robot_config" / "config" / "robots" / "lekiwi_handeye_realsense_grasp_pc.yaml"
    )
    evidence_path = (
        repository_root / "outputs" / "success_cloud_replay_20260806" / "current_phases_after_wait_future_fix.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    candidate_id = int(evidence["pr259"]["final_candidate_id"])
    assert candidate_id == 82

    from embodied_common.dispatch_binding import fill_delegated_executor_identity
    from ibrobot_msgs.action import PickObject
    from robot_config.loader import load_robot_config_dict, robot_config_digest
    from robot_config.timeout_policy import resolve_embodied_timeout_policy
    from skill_library.skill_executor_node import SkillExecutorNode

    config = load_robot_config_dict(config_path)
    embodied = config["embodied"]
    execution = embodied["execution"]
    timeouts = resolve_embodied_timeout_policy(embodied)
    catalog_root = repository_root / embodied["skill_catalog_source_root"]
    mock_node = rclpy.create_node(f"hermes_marker_replay_{uuid.uuid4().hex}")
    pick_goals = []

    def validate_skill(request, response):
        response.allowed = request.skill_name == "pick_object" and request.target_name == "marker"
        response.reason = "offline replay allowed" if response.allowed else "unexpected replay request"
        response.actual_registry_epoch = request.dispatch_binding.expected_registry_epoch
        response.actual_registry_generation = request.dispatch_binding.expected_registry_generation
        response.actual_registry_digest = request.dispatch_binding.expected_registry_digest
        return response

    mock_node.create_service(ValidateSkill, embodied["validate_skill_service"], validate_skill)

    def execute_pick(goal_handle):
        goal = goal_handle.request
        pick_goals.append(goal)
        result = PickObject.Result()
        result.success = goal.target_query == "marker" and candidate_id == 82
        result.message = f"offline marker planning replay selected candidate {candidate_id}"
        result.attempts = 1
        result.verification_status = PickObject.Result.VERIFICATION_NOT_RUN
        result.debug_output_dir = evidence["fixture"]
        result.completed_phases = ["planning", "selecting"]
        result.candidate_index = candidate_id
        fill_delegated_executor_identity(
            result.actual_executor,
            {
                "name": goal.expected_executor.name,
                "contract_version": goal.expected_executor.contract_version,
                "endpoint_kind": goal.expected_executor.endpoint_kind,
                "endpoint_name": goal.expected_executor.endpoint_name,
                "configuration_digest": goal.expected_executor.configuration_digest,
                "model_deployment_name": goal.expected_executor.model_deployment_name,
                "model_fingerprint": goal.expected_executor.model_fingerprint,
                "model_bundle_digest": goal.expected_executor.model_bundle_digest,
            },
        )
        feedback = PickObject.Feedback()
        feedback.phase = "planning"
        feedback.detail = f"replaying {evidence_path}"
        goal_handle.publish_feedback(feedback)
        goal_handle.succeed()
        return result

    pick_server = ActionServer(mock_node, PickObject, config["grasp_execution"]["action_name"], execute_pick)
    params = {
        "active_control_mode": config["skill_required_control_mode"],
        "arm_joint_names_json": json.dumps(config["joints"]["arm"]),
        "config_digest": robot_config_digest(config),
        "default_skill_timeout_sec": timeouts["default_skill_timeout_sec"],
        "grasp_execution_json": json.dumps(config["grasp_execution"]),
        "gripper_closed_position": execution["gripper_closed_position"],
        "gripper_open_position": execution["gripper_open_position"],
        "gripper_settle_sec": timeouts["gripper_settle_sec"],
        "joint_limits_json": json.dumps(config["teleoperation"]["safety"]["joint_limits"]),
        "model_idle_timeout_sec": timeouts["model_idle_timeout_sec"],
        "motion_authorized": True,
        "named_poses_json": json.dumps(embodied["named_poses"]),
        "named_targets_json": json.dumps(embodied.get("named_targets", {})),
        "pick_action_name": config["grasp_execution"]["action_name"],
        "placement_execution_json": json.dumps(config["placement_execution"]),
        "relative_motion_direction_mapping_json": json.dumps(execution["relative_motion_direction_mapping"]),
        "relative_motion_reference_frame": execution["relative_motion_reference_frame"],
        "relative_motion_step_m": execution["relative_motion_step_m"],
        "robot_name": config["name"],
        "robot_state_freshness_sec": timeouts["robot_state_freshness_sec"],
        "rpc_timeout_sec": 1.0,
        "scene_freshness_sec": timeouts["scene_freshness_sec"],
        "skill_catalog_profile": embodied["skill_catalog_profile"],
        "skill_catalog_source_mode": "development",
        "skill_catalog_source_root": str(catalog_root),
        "skill_required_control_mode": config["skill_required_control_mode"],
        "task_budget_sec": timeouts["task_budget_sec"],
        "workspace_json": json.dumps(embodied["safety"]["workspace"]),
    }
    gateway_node = SkillExecutorNode(
        parameter_overrides=[Parameter(name, value=value) for name, value in params.items()],
        node_name=f"hermes_marker_gateway_{uuid.uuid4().hex}",
    )
    agent_node = AgentPlanNode(parameter_overrides=[Parameter("rpc_timeout_sec", value=1.0)])
    executor = MultiThreadedExecutor(num_threads=8)
    for node in (mock_node, gateway_node, agent_node):
        executor.add_node(node)
    import threading

    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    from robot_skill_cli.hermes_launcher import _prepare_robot_skill_wrapper

    wrapper_dir = _prepare_robot_skill_wrapper(
        tmp_path / "hermes-workspace",
        robot_skill,
        os.environ.get("PYTHONPATH", ""),
        config_path,
        os.environ.copy(),
    )
    bound_robot_skill = wrapper_dir / "robot-skill"

    def run_cli(*arguments: str, jsonl: bool = False):
        completed = subprocess.run(
            [bound_robot_skill, *arguments],
            check=False,
            capture_output=True,
            env=os.environ.copy(),
            text=True,
            timeout=20.0,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        lines = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        return lines if jsonl else lines[-1]

    try:
        rejected_override = subprocess.run(
            [bound_robot_skill, "--config-name", "so101_single_arm", "status"],
            check=False,
            capture_output=True,
            env=os.environ.copy(),
            text=True,
            timeout=20.0,
        )
        assert rejected_override.returncode == 2
        assert "configuration is bound by hermes-robot" in rejected_override.stderr

        status = run_cli("status")
        assert status["data"]["control_plane_ready"] is True
        assert status["data"]["motion_authorized"] is True

        listing = run_cli("list-skills")
        assert "pick_object" in {skill["name"] for skill in listing["data"]["skills"]}

        plan = run_cli(
            "plan-workflow",
            "--request-id",
            "hermes-marker-outputs-cli",
            "--text",
            "用之前的 outputs 验证 Hermes marker 抓取通路，不启动真机",
            "--workflow-json",
            json.dumps([{"skill_name": "pick_object", "target_name": "marker"}]),
        )
        planned = plan["data"]["plan"]
        assert planned["workflow_steps"][0]["target_name"] == "marker"

        description = run_cli("describe", "pick_object")
        assert description["data"]["parameters"]["required"] == ["target_name"]

        validated = run_cli("validate-plan", "--plan-token", planned["plan_token"])
        assert validated["data"]["allowed"] is True
        task_id = "hermes-marker-outputs-cli-task"
        confirmed = run_cli(
            "confirm-plan",
            "--plan-token",
            planned["plan_token"],
            "--plan-digest",
            planned["plan_digest"],
            "--task-id",
            task_id,
            "--timeout-sec",
            "240",
        )
        confirmation_token = confirmed["data"]["confirmation_token"]

        events = run_cli(
            "execute-plan",
            "--plan-token",
            planned["plan_token"],
            "--confirmation-token",
            confirmation_token,
            "--task-id",
            task_id,
            "--timeout-sec",
            "240",
            "--plan-id",
            planned["plan_id"],
            "--plan-digest",
            planned["plan_digest"],
            "--registry-epoch",
            planned["registry_epoch"],
            "--registry-generation",
            str(planned["registry_generation"]),
            "--registry-digest",
            planned["registry_digest"],
            "--expected-step-count",
            str(len(planned["workflow_steps"])),
            jsonl=True,
        )
        terminal = events[-1]
        assert terminal["event"] == "result"
        assert terminal["data"]["success"] is True
        assert terminal["data"]["completed_step_count"] == 1
        assert len(pick_goals) == 1
        assert pick_goals[0].target_query == "marker"
        assert pick_goals[0].mode == PickObject.Goal.MODE_EXECUTE
        assert pick_goals[0].release_after_success is False
        assert pick_goals[0].dispatch_binding.dispatch_nonce
        assert candidate_id == 82
    finally:
        executor.shutdown(timeout_sec=1.0)
        thread.join(timeout=1.0)
        pick_server.destroy()
        for node in (agent_node, gateway_node, mock_node):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_confirm_plan_rejects_plan_that_was_not_validated(plan_rig):
    plan_request = PlanAgentCommand.Request()
    plan_request.schema_version = 1
    plan_request.request_id = "request-unvalidated"
    plan_request.raw_command = "打开夹爪"
    _workflow_request(plan_request, "open_gripper_skill")
    planned = _future_result(plan_rig.plan_client.call_async(plan_request))
    assert planned.success is True

    confirm_request = ConfirmAgentPlan.Request()
    confirm_request.schema_version = 1
    confirm_request.plan_token = planned.plan.plan_token
    confirm_request.plan_digest = planned.plan.plan_digest
    confirm_request.task_id = "agent-task-unvalidated"
    confirm_request.registry_epoch, confirm_request.registry_generation, confirm_request.registry_digest = (
        plan_rig.registry
    )
    confirm_request.task_budget_sec = 5.0

    confirmed = _future_result(plan_rig.confirm_client.call_async(confirm_request))

    assert confirmed.confirmed is False
    assert confirmed.error_code == "SKILL_REQUEST_ID_CONFLICT"


def test_validate_plan_rejects_mismatched_safety_identity(plan_rig):
    request = PlanAgentCommand.Request()
    request.schema_version = 1
    request.request_id = "request-validation-identity"
    request.raw_command = "打开夹爪"
    _workflow_request(request, "open_gripper_skill")
    planned = _future_result(plan_rig.plan_client.call_async(request))
    plan_rig.validation_control["registry"] = ("epoch-test", 8, "different")

    validate_request = ValidateAgentPlan.Request()
    validate_request.schema_version = 1
    validate_request.plan_token = planned.plan.plan_token
    validated = _future_result(plan_rig.validate_client.call_async(validate_request))

    assert validated.allowed is False
    assert validated.error_code == "SKILL_REGISTRY_VERSION_MISMATCH"


class _PendingFuture:
    def done(self):
        return False


class _DoneFuture:
    def __init__(self, value):
        self._value = value

    def done(self):
        return True

    def result(self):
        return self._value


class _UnknownAcceptanceSkillClient:
    def wait_for_server(self, *, timeout_sec):
        return True

    def send_goal_async(self, _goal):
        return _PendingFuture()


class _CancelRejectedHandle:
    accepted = True

    def get_result_async(self):
        return _PendingFuture()

    def cancel_goal_async(self):
        return _DoneFuture(SimpleNamespace(goals_canceling=[]))


class _CancelRejectedSkillClient(_UnknownAcceptanceSkillClient):
    def send_goal_async(self, _goal):
        return _DoneFuture(_CancelRejectedHandle())


def _send_skill_node(client):
    node = object.__new__(AgentPlanNode)
    node._skill_client = client
    node._rpc_timeout = 0.01
    node._wait_future = lambda future, timeout_sec: future.done()
    return node


def test_send_skill_rejects_success_payload_from_aborted_action():
    node = _send_skill_node(
        SimpleNamespace(
            wait_for_server=lambda **_: True,
            send_goal_async=lambda _goal: _DoneFuture(
                SimpleNamespace(
                    accepted=True,
                    get_result_async=lambda: _DoneFuture(
                        SimpleNamespace(
                            status=6,
                            result=SimpleNamespace(
                                success=True,
                                actual_registry_epoch="epoch",
                                actual_registry_generation=1,
                                actual_registry_digest="digest",
                            ),
                        )
                    ),
                )
            ),
        )
    )

    with pytest.raises(AgentPlanError) as raised:
        node._send_skill(
            SimpleNamespace(is_cancel_requested=False),
            new_binding(task_id="task-1"),
            CanonicalWorkflowStep(1, "open_gripper_skill", timeout_sec=1.0),
            time.monotonic() + 1.0,
            ("epoch", 1, "digest"),
        )

    assert raised.value.code == "CAPABILITY_NOT_READY"


def test_send_skill_preserves_navigation_goal_presence():
    captured = {}

    def send_goal(goal):
        captured["goal"] = goal
        result = SimpleNamespace(
            success=True,
            actual_registry_epoch="epoch",
            actual_registry_generation=1,
            actual_registry_digest="digest",
        )
        return _DoneFuture(
            SimpleNamespace(
                accepted=True,
                get_result_async=lambda: _DoneFuture(SimpleNamespace(status=4, result=result)),
            )
        )

    node = _send_skill_node(SimpleNamespace(wait_for_server=lambda **_: True, send_goal_async=send_goal))
    step = CanonicalWorkflowStep(
        schema_version=2,
        skill_name="nav_abs_coordinate",
        x=0.0,
        y=-2.0,
        yaw=0.0,
        timeout_sec=1.0,
    )

    node._send_skill(
        SimpleNamespace(is_cancel_requested=False),
        new_binding(task_id="task-1"),
        step,
        time.monotonic() + 1.0,
        ("epoch", 1, "digest"),
    )

    goal = captured["goal"]
    assert goal.has_x is True and goal.x == pytest.approx(0.0)
    assert goal.has_y is True and goal.y == pytest.approx(-2.0)
    assert goal.has_yaw is True and goal.yaw == pytest.approx(0.0)


def test_send_skill_reports_unknown_when_goal_acceptance_times_out():
    node = _send_skill_node(_UnknownAcceptanceSkillClient())

    with pytest.raises(AgentPlanError) as raised:
        node._send_skill(
            SimpleNamespace(is_cancel_requested=False),
            new_binding(task_id="task-1"),
            CanonicalWorkflowStep(1, "open_gripper_skill", timeout_sec=1.0),
            time.monotonic() + 1.0,
            ("epoch", 1, "digest"),
        )

    assert raised.value.code == "SKILL_CANCEL_TIMEOUT"


def test_send_skill_requires_cancel_acceptance_and_terminal_result():
    node = _send_skill_node(_CancelRejectedSkillClient())

    with pytest.raises(AgentPlanError) as raised:
        node._send_skill(
            SimpleNamespace(is_cancel_requested=True),
            new_binding(task_id="task-1"),
            CanonicalWorkflowStep(1, "open_gripper_skill", timeout_sec=1.0),
            time.monotonic() + 1.0,
            ("epoch", 1, "digest"),
        )

    assert raised.value.code == "SKILL_CANCEL_TIMEOUT"

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from embodied_common.canon import sha256_text, to_canonical_json
from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import (
    AgentEvent,
    AgentRequest,
    ExecutionResult,
    PlannerIdentity,
    PlannerOutcome,
    PlanProposal,
    Presentation,
    RegistryIdentity,
    RequestKey,
    RequestRecord,
    TaskRef,
    planner_outcome_from_mapping,
    request_hash_preimage,
    request_hash_text,
)


def _request() -> AgentRequest:
    return AgentRequest(
        schema_version=1,
        request_id="request-1",
        session_id="session-1",
        channel_id="channel-1",
        principal_id="principal-1",
        robot_scope="robot-a",
        text="请挥手",
        received_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )


def _registry_identity() -> RegistryIdentity:
    return RegistryIdentity(epoch="epoch-1", generation=3, digest="digest-1")


def _planner_identity() -> PlannerIdentity:
    return PlannerIdentity(
        route="local",
        returned_model="dummy",
        protocol="json",
        prompt_version="1",
        schema_version="1",
        config_digest="config-1",
    )


def test_request_key_and_request_hash_are_canonical() -> None:
    request = _request()

    assert request.to_key() == RequestKey("robot-a", "channel-1", "principal-1", "request-1")
    assert request_hash_preimage(request)["reply_to_request_id"] is None
    assert request_hash_text(request) == sha256_text(to_canonical_json(request_hash_preimage(request)))


def test_planner_outcome_rejects_invalid_skill_name_for_read_only() -> None:
    with pytest.raises(ValueError, match="skill_name is only allowed"):
        PlannerOutcome(
            kind="read_only",
            user_message="查看状态",
            query_kind="status",
            skill_name="open_gripper_skill",
        )


def test_planner_outcome_from_mapping_normalizes_workflow_steps() -> None:
    outcome = planner_outcome_from_mapping(
        {
            "kind": "workflow",
            "user_message": "执行计划",
            "summary": "挥手",
            "steps": [{"schema_version": 1, "skill_name": "open_gripper_skill"}],
        }
    )

    assert outcome.kind == "workflow"
    assert outcome.steps == (CanonicalWorkflowStep(1, "open_gripper_skill"),)


def test_contract_objects_validate_identity_and_state() -> None:
    request = _request()
    task_ref = TaskRef(
        task_id="task-1",
        plan_id="plan-1",
        plan_digest="plan-digest",
        registry_epoch="epoch-1",
        registry_generation=3,
        registry_digest="digest-1",
        expected_step_count=1,
    )
    outcome = PlannerOutcome(
        kind="workflow",
        user_message="执行计划",
        steps=(CanonicalWorkflowStep(1, "open_gripper_skill"),),
        summary="挥手",
    )
    Presentation(
        task_ref=task_ref,
        plan_kind=1,
        steps=(CanonicalWorkflowStep(1, "open_gripper_skill"),),
        execution_mode="interactive_confirmation",
        proposed_task_budget_sec=3.0,
        summary="挥手",
    )
    proposal = PlanProposal(
        request=request,
        planning_generation=1,
        catalog_identity=_registry_identity(),
        outcome=outcome,
        planner_identity=_planner_identity(),
    )
    record = RequestRecord(
        key=request.to_key(),
        session_id=request.session_id,
        state="MAY_EXECUTE",
        planning_generation=1,
        stop_requested=False,
        may_have_submitted=False,
        input_hash="hash-1",
        task_ref=task_ref,
        proposal_json="{}",
    )
    result = ExecutionResult(
        status="succeeded",
        task_ref=task_ref,
        error_code="",
        message="ok",
        detail={"steps": 1},
    )
    event = AgentEvent(
        schema_version=1,
        request_key=request.to_key(),
        sequence=1,
        event_type="planned",
        state="MAY_EXECUTE",
        user_message="完成",
        detail={"step": 1},
    )

    assert proposal.request.to_key() == request.to_key()
    assert proposal.catalog_identity == _registry_identity()
    assert proposal.outcome == outcome
    assert proposal.planner_identity == _planner_identity()
    assert record.task_ref == task_ref
    assert result.task_ref == task_ref
    assert event.request_key == request.to_key()
    assert outcome.to_dict()["steps"][0]["skill_name"] == "open_gripper_skill"


def test_invalid_state_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="state is invalid"):
        RequestRecord(
            key=RequestKey("robot-a", "channel-1", "principal-1", "request-1"),
            session_id="session-1",
            state="NOT_A_STATE",  # type: ignore[arg-type]
            planning_generation=1,
            stop_requested=False,
            may_have_submitted=False,
            input_hash="hash-1",
        )


def test_plan_and_presentation_identity_are_typed() -> None:
    request = _request()
    plan = Presentation(
        task_ref=TaskRef(
            task_id="task-1",
            plan_id="plan-1",
            plan_digest="plan-digest",
            registry_epoch="epoch-1",
            registry_generation=3,
            registry_digest="digest-1",
            expected_step_count=1,
        ),
        plan_kind=1,
        steps=(CanonicalWorkflowStep(1, "open_gripper_skill"),),
        execution_mode="interactive_confirmation",
        proposed_task_budget_sec=3.0,
        summary="挥手",
    )

    assert plan.task_ref.task_id == "task-1"
    assert _registry_identity().to_dict()["generation"] == 3
    assert _planner_identity().to_dict()["route"] == "local"
    assert request_hash_text(request)

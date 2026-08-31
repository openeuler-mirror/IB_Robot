import pytest

from embodied_agent.agent_plan_store import AgentPlanError, AgentPlanStore, compute_plan_digest
from embodied_common.workflow_contracts import CanonicalWorkflowStep


def _store():
    current = [100.0]
    tokens = iter(("p" * 32, "c" * 32, "q" * 32))
    store = AgentPlanStore(
        clock=lambda: current[0],
        wall_clock=lambda: 1_000.0 + current[0],
        token_factory=lambda: next(tokens),
        plan_id_factory=lambda: "plan-1",
        ttl_sec=300.0,
    )
    return store, current


def _create(store):
    return store.create_plan(
        request_id="request-1",
        raw_command="open the gripper",
        workflow_steps=[CanonicalWorkflowStep(1, "open_gripper_skill")],
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
    )


def _validate(store, plan):
    return store.mark_validated(
        plan_token=plan.plan_token,
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
    )


def test_plan_digest_excludes_opaque_token_and_is_deterministic():
    step = CanonicalWorkflowStep(1, "open_gripper_skill")
    first = compute_plan_digest(
        raw_command="open",
        workflow_steps=[step],
        registry_epoch="epoch",
        registry_generation=1,
        registry_digest="digest",
    )
    second = compute_plan_digest(
        raw_command="open",
        workflow_steps=[step],
        registry_epoch="epoch",
        registry_generation=1,
        registry_digest="digest",
    )
    assert first == second
    assert len(first) == 64


def test_execution_mode_is_bound_to_plan():
    store, _ = _store()
    plan = store.create_plan(
        request_id="request-mode",
        raw_command="点头",
        workflow_steps=[CanonicalWorkflowStep(1, "open_gripper_skill")],
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        execution_mode="immediate_after_presentation",
    )
    _validate(store, plan)
    with pytest.raises(AgentPlanError, match="execution mode"):
        store.confirm(
            plan_token=plan.plan_token,
            plan_digest=plan.plan_digest,
            task_id="task-mode",
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="digest-1",
            task_budget_sec=10.0,
            execution_mode="interactive_confirmation",
        )


def test_plan_lifecycle_binds_identity_and_consumes_confirmation_once():
    store, _ = _store()
    plan = _create(store)
    assert (
        store.create_plan(
            request_id="request-1",
            raw_command="open the gripper",
            workflow_steps=[CanonicalWorkflowStep(1, "open_gripper_skill")],
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="digest-1",
        )
        == plan
    )
    _validate(store, plan)
    confirmation = store.confirm(
        plan_token=plan.plan_token,
        plan_digest=plan.plan_digest,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )
    accepted = store.accept_execution(
        plan_token=plan.plan_token,
        confirmation_token=confirmation.confirmation_token,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )
    assert accepted.state == "ACCEPTED"
    assert accepted.newly_accepted is True
    repeated = store.accept_execution(
        plan_token=plan.plan_token,
        confirmation_token=confirmation.confirmation_token,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )
    assert repeated.state == "ACCEPTED"
    assert repeated.newly_accepted is False
    repeated_after_reload = store.accept_execution(
        plan_token=plan.plan_token,
        confirmation_token=confirmation.confirmation_token,
        task_id="task-1",
        registry_epoch="epoch-2",
        registry_generation=2,
        registry_digest="digest-2",
        task_budget_sec=10.0,
    )
    assert repeated_after_reload.state == "ACCEPTED"
    assert repeated_after_reload.plan.registry_digest == "digest-1"
    with pytest.raises(AgentPlanError) as raised:
        store.accept_execution(
            plan_token=plan.plan_token,
            confirmation_token=confirmation.confirmation_token,
            task_id="task-2",
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="digest-1",
            task_budget_sec=10.0,
        )
    assert raised.value.code == "SKILL_REQUEST_ID_CONFLICT"
    store.mark_terminal(
        plan_token=plan.plan_token,
        task_id="task-1",
        execution_token=accepted.execution_token,
        terminal_message="completed",
        workflow_digest="workflow-digest",
        completed_step_count=1,
    )
    replay = store.accept_execution(
        plan_token=plan.plan_token,
        confirmation_token=confirmation.confirmation_token,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )
    assert replay.state == "TERMINAL"
    assert replay.newly_accepted is False
    assert replay.workflow_digest == "workflow-digest"
    assert replay.completed_step_count == 1
    replay_after_reload = store.accept_execution(
        plan_token=plan.plan_token,
        confirmation_token=confirmation.confirmation_token,
        task_id="task-1",
        registry_epoch="epoch-2",
        registry_generation=2,
        registry_digest="digest-2",
        task_budget_sec=10.0,
    )
    assert replay_after_reload.state == "TERMINAL"

    with pytest.raises(AgentPlanError) as raised:
        store.mark_terminal(
            plan_token=plan.plan_token,
            task_id="task-1",
            execution_token=accepted.execution_token,
            terminal_code="DIFFERENT_RESULT",
        )
    assert raised.value.code == "SKILL_REQUEST_ID_CONFLICT"


def test_only_execution_owner_can_mark_terminal():
    store, _ = _store()
    plan = _create(store)
    _validate(store, plan)
    confirmation = store.confirm(
        plan_token=plan.plan_token,
        plan_digest=plan.plan_digest,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )
    store.accept_execution(
        plan_token=plan.plan_token,
        confirmation_token=confirmation.confirmation_token,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )

    with pytest.raises(AgentPlanError) as raised:
        store.mark_terminal(
            plan_token=plan.plan_token,
            task_id="task-1",
            execution_token="x" * 32,
        )
    assert raised.value.code == "SKILL_REQUEST_ID_CONFLICT"


def test_expiration_does_not_extend_and_reload_fails_closed():
    store, current = _store()
    plan = _create(store)
    current[0] += 300.0
    with pytest.raises(AgentPlanError) as raised:
        store.get(plan.plan_token)
    assert raised.value.code == "SKILL_AGENT_PLAN_EXPIRED"

    store, _ = _store()
    plan = _create(store)
    with pytest.raises(AgentPlanError) as raised:
        store.validate(
            plan_token=plan.plan_token,
            registry_epoch="epoch-2",
            registry_generation=2,
            registry_digest="digest-2",
        )
    assert raised.value.code == "SKILL_REGISTRY_VERSION_MISMATCH"


def test_expired_token_history_is_bounded():
    current = [0.0]
    index = [0]

    def token_factory():
        index[0] += 1
        return f"token-{index[0]:026d}"

    store = AgentPlanStore(
        clock=lambda: current[0],
        wall_clock=lambda: current[0],
        token_factory=token_factory,
        max_records=2,
        ttl_sec=1.0,
    )
    for plan_index in range(3):
        store.create_plan(
            request_id=f"request-{plan_index}",
            raw_command="open",
            workflow_steps=[CanonicalWorkflowStep(1, "open_gripper_skill")],
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="digest-1",
        )
        current[0] += 2.0

    store._purge()  # noqa: SLF001
    assert len(store._expired_tokens) == 2  # noqa: SLF001


def test_terminal_replay_expires_without_blocking_new_plans():
    store, current = _store()
    plan = _create(store)
    _validate(store, plan)
    confirmation = store.confirm(
        plan_token=plan.plan_token,
        plan_digest=plan.plan_digest,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )
    store.accept_execution(
        plan_token=plan.plan_token,
        confirmation_token=confirmation.confirmation_token,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    )
    store.mark_terminal(
        plan_token=plan.plan_token,
        task_id="task-1",
        execution_token=store._records[plan.plan_token].execution_token,  # noqa: SLF001
    )
    current[0] += 301.0

    with pytest.raises(AgentPlanError) as raised:
        store.accept_execution(
            plan_token=plan.plan_token,
            confirmation_token=confirmation.confirmation_token,
            task_id="task-1",
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="digest-1",
            task_budget_sec=10.0,
        )

    assert raised.value.code == "SKILL_AGENT_PLAN_EXPIRED"


def test_confirm_requires_successful_validation_state():
    store, _ = _store()
    plan = _create(store)

    with pytest.raises(AgentPlanError) as raised:
        store.confirm(
            plan_token=plan.plan_token,
            plan_digest=plan.plan_digest,
            task_id="task-1",
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="digest-1",
            task_budget_sec=10.0,
        )

    assert raised.value.code == "SKILL_REQUEST_ID_CONFLICT"
    _validate(store, plan)
    assert store.confirm(
        plan_token=plan.plan_token,
        plan_digest=plan.plan_digest,
        task_id="task-1",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=10.0,
    ).confirmed


def test_execution_budget_is_frozen_by_confirmation():
    store, _ = _store()
    plan = _create(store)
    _validate(store, plan)
    confirmation = store.confirm(
        plan_token=plan.plan_token,
        plan_digest=plan.plan_digest,
        task_id="task-budget",
        registry_epoch="epoch-1",
        registry_generation=1,
        registry_digest="digest-1",
        task_budget_sec=12.5,
    )

    with pytest.raises(AgentPlanError) as raised:
        store.accept_execution(
            plan_token=plan.plan_token,
            confirmation_token=confirmation.confirmation_token,
            task_id="task-budget",
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="digest-1",
            task_budget_sec=10.0,
        )

    assert raised.value.code == "SKILL_REQUEST_ID_CONFLICT"

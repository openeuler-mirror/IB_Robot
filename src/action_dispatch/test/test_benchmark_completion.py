"""Pure completion/reset contracts, without ROS, Torch or LIBERO.

The scheduler does not own the queue: these tests cover permission to commit,
not the dispatcher's actual queue pop, plan generation or Future callbacks.
"""

import importlib.machinery
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def modules():
    # Load real source under a private namespace, bypassing ROS-heavy __init__
    # files without replacing business logic or polluting production imports.
    root = Path(__file__).resolve().parents[1] / "action_dispatch"
    namespace = "_benchmark_completion_test"
    with pytest.MonkeyPatch.context() as patch:
        for suffix in ("", ".executors", ".schedulers"):
            spec = importlib.machinery.ModuleSpec(namespace + suffix, loader=None, is_package=True)
            package = importlib.util.module_from_spec(spec)
            package.__path__ = [str(root / suffix.lstrip("."))]
            patch.setitem(sys.modules, spec.name, package)

        loaded = {}
        for name in ("episode", "executors.completion", "schedulers.base", "schedulers.step_barrier"):
            spec = importlib.util.spec_from_file_location(
                namespace + "." + name, root / (name.replace(".", "/") + ".py")
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, spec.name, module)
            spec.loader.exec_module(module)
            loaded[name.rsplit(".", 1)[-1]] = module
        yield SimpleNamespace(**loaded)


@pytest.fixture
def pending(modules):
    scheduler = modules.step_barrier.StepBarrierScheduler(watermark=1, execution_timeout_sec=2.0)
    scheduler.set_observation_timestamp(100)
    snapshot = modules.base.SchedulerSnapshot(
        plan_length=1,
        watermark=1,
        inference_in_progress=False,
        policy_reset_in_progress=False,
    )
    context = modules.completion.ExecutionContext("step-0", episode_id=7, expected_step_id=0)
    completion = modules.completion.ExecutionCompletion(
        context.correlation_id,
        modules.completion.CompletionStatus.COMPLETED,
        observation_timestamp_ns=200,
        episode_id=7,
        step_id=0,
    )
    assert scheduler.choose_action(snapshot) is modules.base.ActionDecision.TAKE_NEXT
    assert scheduler.should_request_inference(snapshot)
    receipt = modules.completion.ExecutionReceipt(context.correlation_id, accepted=True)
    assert receipt.immediate_completion is None
    assert scheduler.on_submission(context, receipt, submitted_monotonic_ns=1_000_000_000) is None
    return scheduler, context, completion, snapshot


def _start_episode(modules, machine):
    assert machine.try_begin_preparing()
    spec = modules.episode.EpisodeGoalSpec(
        preparation_id=machine.complete_prepare(),
        episode_id=7,
        initial_step_id=0,
        initial_observation_timestamp_ns=100,
        max_actions=10,
        max_duration_sec=60.0,
        startup_timeout_sec=2.0,
        prompt="test action",
    )
    assert machine.try_accept_goal(spec, monotonic_now_ns=1_000_000_000) is None
    generation = machine.goal_generation
    assert machine.try_bind_goal_for_handle(generation, b"test-goal", monotonic_now_ns=1_000_000_000)
    assert machine.try_on_inference_success(generation)
    return generation


@pytest.fixture
def running_episode(modules):
    machine = modules.episode.EpisodeStateMachine()
    _start_episode(modules, machine)
    data = modules.episode.StepCompletionData(
        episode_id=7,
        step_id=0,
        observation_timestamp_ns=200,
        has_reward=True,
        reward=0.25,
        terminated=False,
        truncated=False,
        has_success=True,
        success=False,
        standard_metrics_json='{"score": 0.25}',
        native_metrics_json='{"native": true}',
        info_json='{"source": "test"}',
        round_trip_latency_ms=1.5,
        message="completed",
    )
    return machine, data


def test_submission_waits_and_matching_completion_commits_only_once(modules, pending):
    scheduler, context, completion, snapshot = pending
    assert scheduler.inflight_correlation_id == context.correlation_id
    assert not scheduler.can_accept_next
    assert scheduler.choose_action(snapshot) is modules.base.ActionDecision.WAIT
    assert scheduler.choose_action(replace(snapshot, plan_length=0)) is modules.base.ActionDecision.WAIT
    assert not scheduler.should_request_inference(snapshot)
    assert scheduler.on_tick(1_000_000_001).decision is modules.base.CompletionDecision.IGNORE
    assert scheduler.observation_timestamp_for_inference() == 100

    transition = scheduler.on_completion(completion)

    assert transition.decision is modules.base.CompletionDecision.COMMIT
    assert transition.correlation_id == context.correlation_id
    assert scheduler.inflight_correlation_id is None
    assert scheduler.fault_status is None
    assert scheduler.can_accept_next
    assert scheduler.observation_timestamp_for_inference() == 200
    assert scheduler.choose_action(snapshot) is modules.base.ActionDecision.TAKE_NEXT
    assert scheduler.should_request_inference(snapshot)
    assert scheduler.on_completion(completion).decision is modules.base.CompletionDecision.IGNORE
    assert scheduler.observation_timestamp_for_inference() == 200
    assert scheduler.can_accept_next
    assert scheduler.fault_status is None


def test_unrelated_completion_preserves_pending_submission(modules, pending):
    scheduler, context, completion, snapshot = pending

    transition = scheduler.on_completion(replace(completion, correlation_id="unrelated"))

    assert transition.decision is modules.base.CompletionDecision.IGNORE
    assert scheduler.inflight_correlation_id == context.correlation_id
    assert scheduler.observation_timestamp_for_inference() == 100
    assert scheduler.fault_status is None
    assert not scheduler.can_accept_next
    assert scheduler.choose_action(snapshot) is modules.base.ActionDecision.WAIT
    assert not scheduler.should_request_inference(snapshot)
    assert scheduler.on_completion(completion).decision is modules.base.CompletionDecision.COMMIT


@pytest.mark.parametrize("field,value", [("episode_id", 8), ("episode_id", None), ("step_id", 1), ("step_id", None)])
def test_scheduler_identity_mismatch_fails_closed_until_reset(modules, pending, field, value):
    scheduler, _, completion, snapshot = pending

    transition = scheduler.on_completion(replace(completion, **{field: value}))

    assert transition.decision is modules.base.CompletionDecision.FAIL_CLOSED
    assert transition.fault_status == scheduler.fault_status == f"{field}_mismatch"
    assert scheduler.inflight_correlation_id is None
    assert scheduler.observation_timestamp_for_inference() == 100
    assert not scheduler.can_accept_next
    assert scheduler.choose_action(snapshot) is modules.base.ActionDecision.WAIT
    assert not scheduler.should_request_inference(snapshot)
    assert scheduler.on_completion(completion).decision is modules.base.CompletionDecision.IGNORE
    assert scheduler.fault_status == f"{field}_mismatch"

    scheduler.reset()

    assert scheduler.fault_status is None
    assert scheduler.can_accept_next
    assert scheduler.observation_timestamp_for_inference() is None
    assert not scheduler.should_request_inference(snapshot)


def test_scheduler_reset_ignores_old_completion_before_and_after_new_submission(modules, pending):
    scheduler, context, completion, snapshot = pending
    scheduler.reset()

    assert scheduler.inflight_correlation_id is None
    assert scheduler.fault_status is None
    assert scheduler.can_accept_next
    assert scheduler.on_completion(completion).decision is modules.base.CompletionDecision.IGNORE
    assert scheduler.observation_timestamp_for_inference() is None
    assert not scheduler.should_request_inference(snapshot)

    scheduler.set_observation_timestamp(300)
    new_context = replace(context, correlation_id="after-reset")
    receipt = modules.completion.ExecutionReceipt(new_context.correlation_id, accepted=True)
    scheduler.on_submission(new_context, receipt, submitted_monotonic_ns=2_000_000_000)

    assert scheduler.on_completion(completion).decision is modules.base.CompletionDecision.IGNORE
    assert scheduler.inflight_correlation_id == new_context.correlation_id
    assert scheduler.observation_timestamp_for_inference() == 300
    assert not scheduler.can_accept_next
    assert not scheduler.should_request_inference(snapshot)
    new_completion = replace(completion, correlation_id=new_context.correlation_id, observation_timestamp_ns=400)
    assert scheduler.on_completion(new_completion).decision is modules.base.CompletionDecision.COMMIT
    assert scheduler.observation_timestamp_for_inference() == 400


def test_episode_matching_completion_advances_count_step_and_timestamp_once(modules, running_episode):
    machine, data = running_episode
    generation = machine.goal_generation
    assert machine.committed_count == 0
    assert machine.expected_step_id == 0
    assert machine.get_last_snapshot() is None
    assert machine.get_inference_timestamp() == 100

    outcome = machine.try_commit_step(generation, data, monotonic_now_ns=1_000_000_001)

    assert outcome == modules.episode.StepCommitOutcome(True, None, False)
    assert machine.committed_count == 1
    assert machine.expected_step_id == 1
    assert machine.get_last_snapshot() is data
    assert machine.get_inference_timestamp() == 200
    assert machine.phase is modules.episode.EpisodePhase.RUNNING
    duplicate = machine.try_commit_step(generation, data, monotonic_now_ns=1_000_000_002)
    assert duplicate == modules.episode.StepCommitOutcome(False, None, True)
    assert machine.committed_count == 1
    assert machine.expected_step_id == 1
    assert machine.get_last_snapshot() is data
    assert machine.get_inference_timestamp() == 200


@pytest.mark.parametrize("field,value", [("episode_id", 8), ("episode_id", None), ("step_id", 1), ("step_id", None)])
def test_episode_identity_mismatch_does_not_commit(modules, running_episode, field, value):
    machine, data = running_episode

    outcome = machine.try_commit_step(
        machine.goal_generation, replace(data, **{field: value}), monotonic_now_ns=1_000_000_001
    )

    assert outcome == modules.episode.StepCommitOutcome(False, None, True)
    assert machine.committed_count == 0
    assert machine.expected_step_id == 0
    assert machine.get_last_snapshot() is None
    assert machine.get_inference_timestamp() == 100


def test_episode_reset_rejects_late_completion_and_old_goal_generation(modules, running_episode):
    machine, data = running_episode
    old_generation = machine.goal_generation
    assert machine.try_external_reset(old_generation)
    assert machine.phase is modules.episode.EpisodePhase.FAULTED
    assert machine.termination_reason == modules.episode.TERMINATION_EXTERNAL_RESET
    assert not machine.is_gate_open

    ignored = modules.episode.StepCommitOutcome(False, None, False)
    assert machine.try_commit_step(old_generation, data, monotonic_now_ns=1_000_000_001) == ignored
    assert machine.committed_count == 0
    assert machine.expected_step_id is None
    assert machine.get_last_snapshot() is None
    assert machine.get_inference_timestamp() is None

    # Reuse environment identity so generation, not identity mismatch, must guard.
    new_generation = _start_episode(modules, machine)
    assert new_generation > old_generation
    assert machine.try_commit_step(old_generation, data, monotonic_now_ns=1_000_000_002) == ignored
    assert machine.committed_count == 0
    assert machine.expected_step_id == 0
    assert machine.get_last_snapshot() is None
    assert machine.get_inference_timestamp() == 100
    assert machine.try_commit_step(new_generation, data, monotonic_now_ns=1_000_000_003).applied
    assert machine.committed_count == 1
    assert machine.expected_step_id == 1
    assert machine.get_last_snapshot() is data
    assert machine.get_inference_timestamp() == 200

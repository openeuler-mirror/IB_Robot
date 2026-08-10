"""Pure serial benchmark evaluator supervisor and finalization progression."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from benchmark_runtime.evaluator_classifier import EpisodeOutcome
from benchmark_runtime.finalization import FinalizationIdentity, FinalizationPayload
from benchmark_runtime.plan import BenchmarkPlan


class SupervisorError(RuntimeError):
    """Raised on invalid serial supervisor transitions."""


class Phase(str, Enum):
    WAITING_READY = "waiting_ready"
    RESET = "reset"
    PREPARE = "prepare"
    RUN = "run"
    FINALIZE_EPISODE = "finalize_episode"
    FINALIZE_TASK = "finalize_task"
    FINALIZE_RUN = "finalize_run"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ResetCommand:
    suite: str
    task_id: int
    seed: int
    init_state_id: int
    use_init_state_id: bool = True


@dataclass(frozen=True, slots=True)
class PrepareCommand:
    pass


@dataclass(frozen=True, slots=True)
class RunPolicyCommand:
    prompt: str
    preparation_id: int
    episode_id: int
    initial_step_id: int
    initial_observation_timestamp_ns: int
    max_actions: int
    max_duration_sec: float
    startup_timeout_sec: float


@dataclass(frozen=True, slots=True)
class FinalizeCommand:
    scope: str
    payload: FinalizationPayload
    partial: bool


class SerialSupervisor:
    """Yield exactly one command at a time for a resolved benchmark plan."""

    def __init__(self, plan: BenchmarkPlan, *, run_id: str, ready: bool = True) -> None:
        self.plan = plan
        self.run_id = run_id
        self._phase = Phase.RESET if ready else Phase.WAITING_READY
        self._task_position = 0
        self._episode_index = 0
        self._episode_id = 0
        self._timestamp_ns = 0
        self._prompt = ""
        self._preparation_id = 0
        self._outcome: EpisodeOutcome | None = None
        self._fatal = False

    @property
    def completed(self) -> bool:
        return self._phase == Phase.COMPLETED

    @property
    def current_task_id(self) -> int | None:
        if self._task_position >= len(self.plan.selected_task_ids):
            return None
        return self.plan.selected_task_ids[self._task_position]

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def mark_ready(self) -> None:
        if self._phase != Phase.WAITING_READY:
            raise SupervisorError("readiness can only complete once")
        self._phase = Phase.RESET

    def next_command(self, remaining_startup_sec: float) -> Any | None:
        if self._phase == Phase.WAITING_READY or self._phase == Phase.COMPLETED:
            return None
        task_id = (
            self.plan.selected_task_ids[self._task_position]
            if self._task_position < len(self.plan.selected_task_ids)
            else None
        )
        if self._phase == Phase.RESET:
            if task_id is None:
                raise SupervisorError("reset requested after all tasks were accounted for")
            return ResetCommand(
                self.plan.suite,
                task_id,
                self.plan.seed,
                self._episode_index,
                self.plan.init_state_policy.use_init_state_id,
            )
        if self._phase == Phase.PREPARE:
            return PrepareCommand()
        if self._phase == Phase.RUN:
            return RunPolicyCommand(
                self._prompt,
                self._preparation_id,
                self._episode_id,
                0,
                self._timestamp_ns,
                self.plan.max_steps,
                self.plan.timeouts.max_duration_sec,
                remaining_startup_sec,
            )
        if self._phase == Phase.FINALIZE_EPISODE:
            assert task_id is not None
            assert self._outcome is not None
            result = self._outcome.result
            payload = FinalizationPayload(
                scope="episode",
                identity=FinalizationIdentity(
                    self.run_id, self.plan.suite, task_id, self._episode_id, self._episode_index
                ),
                result={
                    "protocol_success": result.protocol_success,
                    "has_success": result.has_success,
                    "success": result.success,
                    "published_actions": result.published_actions,
                    "has_final_step": result.has_final_step,
                    "final_step_id": result.final_step_id,
                    "has_reward": result.has_reward,
                    "reward": result.reward,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "standard_metrics_json": result.standard_metrics_json,
                    "native_metrics_json": result.native_metrics_json,
                    "info_json": result.info_json,
                    "round_trip_latency_ms": result.round_trip_latency_ms,
                },
                termination_reason=self._outcome.termination_reason,
                error_category=self._outcome.error_category,
                partial=self._outcome.partial,
                artifact_refs=(),
            )
            return FinalizeCommand("episode", payload, payload.partial)
        if self._phase == Phase.FINALIZE_TASK:
            if task_id is None:
                raise SupervisorError("task finalization requested after all tasks were accounted for")
            payload = FinalizationPayload(
                scope="task",
                identity=FinalizationIdentity(self.run_id, self.plan.suite, task_id),
                result={},
                termination_reason="fatal_error" if self._fatal else "completed",
                error_category="infrastructure" if self._fatal else "",
                partial=self._fatal,
                artifact_refs=(),
            )
            return FinalizeCommand("task", payload, payload.partial)
        payload = FinalizationPayload(
            scope="run",
            identity=FinalizationIdentity(self.run_id),
            result={},
            termination_reason="fatal_error" if self._fatal else "completed",
            error_category="infrastructure" if self._fatal else "",
            partial=self._fatal,
            artifact_refs=(),
        )
        return FinalizeCommand("run", payload, payload.partial)

    def accept_reset(
        self,
        success: bool,
        episode_id: int,
        step_id: int,
        prompt: str,
        timestamp_ns: int,
        message: str = "",
    ) -> None:
        if self._phase != Phase.RESET:
            raise SupervisorError("reset arrived out of order")
        if not success:
            self._fatal = True
            self._phase = Phase.FINALIZE_TASK
            return
        task = self.plan.tasks[self.plan.selected_task_ids[self._task_position]]
        if episode_id <= 0 or step_id != 0 or timestamp_ns <= 0 or prompt != task.prompt:
            raise SupervisorError("reset identity/timestamp/prompt mismatch")
        self._episode_id = episode_id
        self._timestamp_ns = timestamp_ns
        self._prompt = prompt
        self._phase = Phase.PREPARE

    def accept_prepare(self, success: bool, preparation_id: int, message: str = "") -> None:
        if self._phase != Phase.PREPARE:
            raise SupervisorError("prepare arrived out of order")
        if not success or preparation_id <= 0:
            from benchmark_runtime.evaluator_classifier import (
                RunPolicyResultData,
                classify_run_policy,
            )

            data = RunPolicyResultData(
                protocol_success=False,
                message=message or "PreparePolicyEpisode failed",
                has_success=False,
                success=False,
                termination_reason="prepare_failed",
                episode_id=self._episode_id,
                has_final_step=False,
                final_step_id=0,
                published_actions=0,
                has_reward=False,
                reward=0.0,
                terminated=False,
                truncated=False,
                standard_metrics_json="{}",
                native_metrics_json="{}",
                info_json="{}",
                round_trip_latency_ms=0.0,
            )
            self._outcome = classify_run_policy("aborted", data)
            self._fatal = True
            self._phase = Phase.FINALIZE_EPISODE
            return
        self._preparation_id = preparation_id
        self._phase = Phase.RUN

    def accept_run(self, outcome: EpisodeOutcome) -> None:
        if self._phase != Phase.RUN:
            raise SupervisorError("RunPolicy result arrived out of order")
        if outcome.result.episode_id != self._episode_id:
            raise SupervisorError("RunPolicy episode identity mismatch")
        self._outcome = outcome
        self._fatal = outcome.infrastructure_error or outcome.canceled
        self._phase = Phase.FINALIZE_EPISODE

    def accept_finalize(self, success: bool, scope: str) -> None:
        expected = {
            Phase.FINALIZE_EPISODE: "episode",
            Phase.FINALIZE_TASK: "task",
            Phase.FINALIZE_RUN: "run",
        }.get(self._phase)
        if not success or scope != expected:
            raise SupervisorError("finalization failed or arrived out of order")
        if scope == "episode":
            if self._fatal:
                self._phase = Phase.FINALIZE_TASK
            else:
                self._episode_index += 1
                self._phase = Phase.RESET if self._episode_index < self.plan.episodes_per_task else Phase.FINALIZE_TASK
        elif scope == "task":
            self._task_position += 1
            self._episode_index = 0
            if self._fatal:
                self._phase = (
                    Phase.FINALIZE_TASK
                    if self._task_position < len(self.plan.selected_task_ids)
                    else Phase.FINALIZE_RUN
                )
            else:
                self._phase = (
                    Phase.RESET if self._task_position < len(self.plan.selected_task_ids) else Phase.FINALIZE_RUN
                )
        else:
            self._phase = Phase.COMPLETED

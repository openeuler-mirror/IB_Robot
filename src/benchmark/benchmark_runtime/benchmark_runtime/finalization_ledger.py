"""Pure ordered and idempotent finalization ledger for serial evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from benchmark_runtime.finalization import FinalizationPayload, finalization_payload_to_wire
from benchmark_runtime.plan import BenchmarkPlan


class FinalizationLedgerError(ValueError):
    """Raised when reset/finalization order or identity is invalid."""


@dataclass(frozen=True, slots=True)
class FinalizationCommitResult:
    committed_scope: str
    artifact_refs: tuple[str, ...]
    already_committed: bool


class FinalizationLedger:
    """Track one serial plan with strict episode -> task -> run progression."""

    def __init__(
        self,
        plan: BenchmarkPlan,
        *,
        run_id: str | None = None,
        on_task_committed: Callable[[str, int], None] | None = None,
    ) -> None:
        self._plan = plan
        self._run_id = run_id
        self._on_task_committed = on_task_committed
        self._task_position = 0
        self._episode_index = 0
        self._pending_episode_id: int | None = None
        self._reset_authorized = False
        self._task_committed: set[int] = set()
        self._run_committed = False
        self._commits: dict[tuple[object, ...], tuple[str, ...]] = {}

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def current_task_id(self) -> int | None:
        if self._task_position >= len(self._plan.selected_task_ids):
            return None
        return self._plan.selected_task_ids[self._task_position]

    def authorize_reset(self, suite: str, task_id: int, episode_index: int) -> None:
        if self._run_committed:
            raise FinalizationLedgerError("run is already committed")
        if self._reset_authorized or self._pending_episode_id is not None:
            raise FinalizationLedgerError("previous episode is not finalized")
        expected_task = self.current_task_id
        if expected_task is None:
            raise FinalizationLedgerError("all selected tasks are already accounted for")
        if suite != self._plan.suite:
            raise FinalizationLedgerError(f"suite mismatch: got {suite!r}, expected {self._plan.suite!r}")
        if task_id != expected_task:
            relation = (
                "future task" if task_id in self._plan.selected_task_ids[self._task_position + 1 :] else "stale task"
            )
            raise FinalizationLedgerError(f"{relation}: got task {task_id}, expected task {expected_task}")
        if episode_index != self._episode_index:
            relation = "future" if episode_index > self._episode_index else "stale"
            raise FinalizationLedgerError(
                f"{relation} episode index: got {episode_index}, expected {self._episode_index}"
            )
        self._reset_authorized = True

    def record_reset(self, episode_id: int) -> None:
        if not self._reset_authorized:
            raise FinalizationLedgerError("record_reset without an authorized reset")
        if not isinstance(episode_id, int) or isinstance(episode_id, bool) or episode_id <= 0:
            raise FinalizationLedgerError("episode_id must be a positive int")
        self._pending_episode_id = episode_id
        self._reset_authorized = False

    def abort_reset(self) -> None:
        self._reset_authorized = False
        self._pending_episode_id = None

    def commit(self, payload: FinalizationPayload) -> FinalizationCommitResult:
        self._validate_run_id(payload.identity.run_id)
        if payload.artifact_refs:
            raise FinalizationLedgerError("incoming finalization payload artifact_refs must be empty")
        key = self._identity_key(payload)
        fingerprint = self._fingerprint(payload)
        previous = self._commits.get(key)
        if previous is not None:
            if previous != fingerprint:
                raise FinalizationLedgerError(f"conflicting replay for committed {payload.scope} identity")
            return FinalizationCommitResult(payload.scope, (), True)

        if payload.scope == "episode":
            self._commit_episode(payload, key, fingerprint)
        elif payload.scope == "task":
            self._commit_task(payload, key, fingerprint)
        else:
            self._commit_run(payload, key, fingerprint)
        if self._run_id is None:
            self._run_id = payload.identity.run_id
        return FinalizationCommitResult(payload.scope, (), False)

    def _validate_run_id(self, run_id: str) -> None:
        if self._run_id is not None and run_id != self._run_id:
            raise FinalizationLedgerError(f"run_id mismatch: got {run_id!r}, expected {self._run_id!r}")

    @staticmethod
    def _fingerprint(payload: FinalizationPayload) -> tuple[str, ...]:
        wire = finalization_payload_to_wire(payload)
        return (
            wire.scope,
            wire.identity_json,
            wire.result_json,
            wire.termination_reason,
            wire.error_category,
            str(wire.partial),
            wire.artifact_refs_json,
        )

    @staticmethod
    def _identity_key(payload: FinalizationPayload) -> tuple[object, ...]:
        identity = payload.identity
        if payload.scope == "episode":
            return (payload.scope, identity.run_id, identity.suite, identity.task_id, identity.episode_index)
        if payload.scope == "task":
            return (payload.scope, identity.run_id, identity.suite, identity.task_id)
        return (payload.scope, identity.run_id)

    def _commit_episode(
        self, payload: FinalizationPayload, key: tuple[object, ...], fingerprint: tuple[str, ...]
    ) -> None:
        expected_task = self.current_task_id
        identity = payload.identity
        if expected_task is None or identity.task_id != expected_task:
            relation = (
                "future task"
                if identity.task_id in self._plan.selected_task_ids[self._task_position + 1 :]
                else "stale task"
            )
            raise FinalizationLedgerError(f"{relation}: got task {identity.task_id}, expected task {expected_task}")
        if identity.suite != self._plan.suite:
            raise FinalizationLedgerError("suite mismatch")
        if identity.episode_index != self._episode_index:
            relation = (
                "future"
                if identity.episode_index is not None and identity.episode_index > self._episode_index
                else "stale"
            )
            raise FinalizationLedgerError(
                f"{relation} episode index: got {identity.episode_index}, expected {self._episode_index}"
            )
        if self._pending_episode_id is None:
            raise FinalizationLedgerError("episode finalization has no successful reset identity")
        if identity.episode_id != self._pending_episode_id:
            raise FinalizationLedgerError(
                f"episode_id mismatch: got {identity.episode_id}, expected {self._pending_episode_id}"
            )
        self._commits[key] = fingerprint
        self._pending_episode_id = None
        self._episode_index += 1

    def _commit_task(self, payload: FinalizationPayload, key: tuple[object, ...], fingerprint: tuple[str, ...]) -> None:
        expected_task = self.current_task_id
        identity = payload.identity
        if expected_task is None or identity.task_id != expected_task:
            raise FinalizationLedgerError(f"task identity mismatch: got {identity.task_id}, expected {expected_task}")
        if identity.suite != self._plan.suite:
            raise FinalizationLedgerError("suite mismatch")
        if self._pending_episode_id is not None or self._reset_authorized:
            raise FinalizationLedgerError("task cannot finalize while an episode is uncommitted")
        if self._episode_index < self._plan.episodes_per_task and not payload.partial:
            raise FinalizationLedgerError(f"episode {self._episode_index} is not committed for task {expected_task}")
        prior_task_position = self._task_position
        prior_episode_index = self._episode_index
        self._commits[key] = fingerprint
        self._task_committed.add(expected_task)
        self._task_position += 1
        self._episode_index = 0
        if self._on_task_committed is not None:
            try:
                self._on_task_committed(self._plan.suite, expected_task)
            except Exception:
                self._task_position = prior_task_position
                self._episode_index = prior_episode_index
                self._task_committed.remove(expected_task)
                del self._commits[key]
                raise

    def _commit_run(self, payload: FinalizationPayload, key: tuple[object, ...], fingerprint: tuple[str, ...]) -> None:
        if self._pending_episode_id is not None or self._reset_authorized:
            raise FinalizationLedgerError("run cannot finalize while an episode is uncommitted")
        expected_tasks = set(self._plan.selected_task_ids)
        missing = sorted(expected_tasks - self._task_committed)
        if missing:
            raise FinalizationLedgerError(f"task {missing[0]} is not committed")
        self._commits[key] = fingerprint
        self._run_committed = True

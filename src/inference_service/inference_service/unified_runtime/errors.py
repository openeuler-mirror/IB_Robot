"""Normalized failures and recovery requirements for unified runtimes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .contracts import OutcomeEvidence, OutcomeState, _freeze_json_mapping, json_safe_data


class RecoveryScope(str, Enum):
    REQUEST = "request"
    STREAM = "stream"
    RUNTIME = "runtime"


class RecoveryAction(str, Enum):
    NONE = "none"
    RETRY_REQUEST = "retry_request"
    RESET_STREAM = "reset_stream"
    CLOSE_STREAM = "close_stream"
    RESET_RUNTIME = "reset_runtime"
    RELOAD = "reload"


RECOVERY_MATRIX: dict[RecoveryScope, frozenset[RecoveryAction]] = {
    RecoveryScope.REQUEST: frozenset(
        {
            RecoveryAction.NONE,
            RecoveryAction.RETRY_REQUEST,
            RecoveryAction.RESET_RUNTIME,
            RecoveryAction.RELOAD,
        }
    ),
    RecoveryScope.STREAM: frozenset(
        {
            RecoveryAction.NONE,
            RecoveryAction.RESET_STREAM,
            RecoveryAction.CLOSE_STREAM,
            RecoveryAction.RELOAD,
        }
    ),
    RecoveryScope.RUNTIME: frozenset(
        {
            RecoveryAction.NONE,
            RecoveryAction.RESET_RUNTIME,
            RecoveryAction.RELOAD,
        }
    ),
}
VALID_RECOVERY_REQUIREMENTS = frozenset(
    (scope.value, action.value) for scope, actions in RECOVERY_MATRIX.items() for action in actions
)


class InvalidRecoveryRequirement(ValueError):
    """Raised when a scope/action pair is outside the finite matrix."""

    code = "invalid_recovery_requirement"


def _coerce_scope(value: RecoveryScope | str) -> RecoveryScope:
    if isinstance(value, RecoveryScope):
        return value
    try:
        return RecoveryScope(str(value))
    except ValueError:
        try:
            return RecoveryScope[str(value)]
        except KeyError as exc:
            raise ValueError(f"invalid recovery scope: {value!r}") from exc


def _coerce_action(value: RecoveryAction | str) -> RecoveryAction:
    if isinstance(value, RecoveryAction):
        return value
    try:
        return RecoveryAction(str(value))
    except ValueError:
        try:
            return RecoveryAction[str(value)]
        except KeyError as exc:
            raise ValueError(f"invalid recovery action: {value!r}") from exc


@dataclass(frozen=True)
class RecoveryRequirement:
    """Explicit scope and action for recovering one failed operation."""

    scope: RecoveryScope | str
    action: RecoveryAction | str

    def __post_init__(self) -> None:
        try:
            scope = _coerce_scope(self.scope)
            action = _coerce_action(self.action)
        except (TypeError, ValueError) as exc:
            raise InvalidRecoveryRequirement(f"invalid recovery requirement: {self.scope!r}/{self.action!r}") from exc
        if action not in RECOVERY_MATRIX[scope]:
            raise InvalidRecoveryRequirement(
                f"invalid recovery requirement: scope={scope.value!r} does not allow action={action.value!r}"
            )
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "action", action)

    @property
    def recoverable_action(self) -> bool:
        return self.action is not RecoveryAction.NONE

    @property
    def recoverable(self) -> bool:
        return self.recoverable_action

    def to_dict(self) -> dict[str, str]:
        return {"scope": self.scope.value, "action": self.action.value}


class RuntimeControlError(RuntimeError):
    """Internal control exception converted at the public runtime boundary."""

    code = "runtime_control_error"

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        operation_started: bool = False,
        outcome_known: bool = True,
        state_mutated: bool = False,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.operation_started = operation_started
        self.outcome_known = outcome_known
        self.state_mutated = state_mutated
        self.details = dict(details or {})


class DeadlineExceeded(RuntimeControlError):
    code = "deadline_exceeded"

    def __init__(self, *, phase: str = "deadline", operation_started: bool = False) -> None:
        super().__init__(
            "execution deadline expired",
            phase=phase,
            operation_started=operation_started,
            outcome_known=not operation_started,
            state_mutated=operation_started,
        )


class CancellationRequested(RuntimeControlError):
    code = "request_canceled"

    def __init__(
        self, *, phase: str = "cancellation", reason: str | None = None, operation_started: bool = False
    ) -> None:
        details = {"reason": reason} if reason is not None else {}
        super().__init__(
            "execution was canceled",
            phase=phase,
            operation_started=operation_started,
            outcome_known=not operation_started,
            state_mutated=operation_started,
            details=details,
        )


class OutputValidationError(ValueError):
    code = "output_validation_failed"


class ExecutionFailure(RuntimeError):
    """Exception-type public failure boundary for model execution."""

    def __init__(
        self,
        code: str,
        message: str,
        recoverable: bool = False,
        recovery: RecoveryRequirement | None = None,
        evidence: OutcomeEvidence | None = None,
        cause: BaseException | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("ExecutionFailure.code must be non-empty")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("ExecutionFailure.message must be non-empty")
        if recovery is None:
            recovery = RecoveryRequirement(RecoveryScope.REQUEST, RecoveryAction.NONE)
        if not isinstance(recovery, RecoveryRequirement):
            raise TypeError("ExecutionFailure.recovery must be RecoveryRequirement")
        if recovery.action is RecoveryAction.NONE and recoverable:
            raise ValueError("recoverable cannot be true when recovery action is none")
        if not isinstance(recoverable, bool):
            raise TypeError("ExecutionFailure.recoverable must be bool")
        if evidence is None:
            evidence = OutcomeEvidence.not_started()
        if not isinstance(evidence, OutcomeEvidence):
            raise TypeError("ExecutionFailure.evidence must be OutcomeEvidence")
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable
        self.recovery = recovery
        self.evidence = evidence
        self.cause = cause
        self.details = _freeze_json_mapping(details, path="failure.details")

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    @property
    def close_pending(self) -> bool:
        return self.details.get("close_pending", False) is True

    @property
    def recovery_scope(self) -> RecoveryScope:
        return self.recovery.scope

    @property
    def recovery_action(self) -> RecoveryAction:
        return self.recovery.action

    @property
    def outcome_evidence(self) -> OutcomeEvidence:
        return self.evidence

    def to_dict(self) -> dict[str, object]:
        """Serialize the stable boundary; local ``cause`` is intentionally omitted."""

        return {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "recovery": self.recovery.to_dict(),
            "evidence": self.evidence.to_dict(),
            "details": json_safe_data(self.details),
        }

    as_dict = to_dict
    serialize = to_dict

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class ExecutionFailureFactory:
    """Create normalized failures while enforcing the recovery matrix."""

    def validate_recovery(
        self,
        recovery: RecoveryRequirement | None = None,
        *,
        scope: RecoveryScope | str = RecoveryScope.REQUEST,
        action: RecoveryAction | str = RecoveryAction.NONE,
    ) -> RecoveryRequirement:
        if recovery is not None:
            if not isinstance(recovery, RecoveryRequirement):
                raise InvalidRecoveryRequirement("recovery must be a RecoveryRequirement")
            return recovery
        try:
            return RecoveryRequirement(scope, action)
        except InvalidRecoveryRequirement:
            raise
        except (TypeError, ValueError) as exc:
            raise InvalidRecoveryRequirement("invalid recovery requirement") from exc

    def create(
        self,
        code: str,
        message: str,
        *,
        evidence: OutcomeEvidence | None = None,
        recovery: RecoveryRequirement | None = None,
        scope: RecoveryScope | str = RecoveryScope.REQUEST,
        action: RecoveryAction | str = RecoveryAction.NONE,
        recoverable: bool = False,
        recovery_available: bool | None = None,
        available: bool | None = None,
        cause: BaseException | None = None,
        details: Mapping[str, object] | None = None,
    ) -> ExecutionFailure:
        selected = self.validate_recovery(recovery, scope=scope, action=action)
        if recovery_available is not None and available is not None and recovery_available != available:
            raise ValueError("recovery_available and available disagree")
        if recovery_available is not None:
            recoverable = recovery_available
        elif available is not None:
            recoverable = available
        if selected.action is RecoveryAction.NONE and recoverable:
            raise InvalidRecoveryRequirement("recoverable recovery requires a non-none action")
        return ExecutionFailure(
            code,
            message,
            recoverable=recoverable,
            recovery=selected,
            evidence=evidence,
            cause=cause,
            details=details,
        )

    __call__ = create

    def from_exception(
        self,
        error: BaseException,
        *,
        code: str | None = None,
        message: str | None = None,
        evidence: OutcomeEvidence | None = None,
        recovery: RecoveryRequirement | None = None,
        scope: RecoveryScope | str = RecoveryScope.REQUEST,
        action: RecoveryAction | str = RecoveryAction.NONE,
        recoverable: bool | None = None,
        recovery_available: bool | None = None,
        available: bool | None = None,
        details: Mapping[str, object] | None = None,
    ) -> ExecutionFailure:
        if (
            isinstance(error, ExecutionFailure)
            and code is None
            and message is None
            and evidence is None
            and recovery is None
        ):
            return error
        selected_code = code or str(getattr(error, "code", "execution_failed"))
        selected_message = message or str(error) or selected_code
        selected_evidence = evidence or getattr(error, "evidence", None)
        selected_recovery = recovery or getattr(error, "recovery", None)
        selected_recoverable = getattr(error, "recoverable", False) if recoverable is None else recoverable
        if selected_evidence is None:
            started = bool(getattr(error, "operation_started", False))
            phase_by_code = {
                "deadline_exceeded": "deadline",
                "request_canceled": "cancellation",
                "output_validation_failed": "output_validation",
                "adaptation_failed": "adaptation",
                "reset_failed": "reset",
                "transport_failed": "transport",
                "backend_async_failure": "acl_async",
            }
            selected_evidence = OutcomeEvidence(
                OutcomeState.STARTED if started else OutcomeState.NOT_STARTED,
                bool(getattr(error, "outcome_known", not started)),
                bool(getattr(error, "state_mutated", started)),
                str(getattr(error, "phase", phase_by_code.get(selected_code, "backend"))),
            )
        merged_details: dict[str, object] = {}
        error_details = getattr(error, "details", None)
        if isinstance(error_details, Mapping):
            merged_details.update(error_details)
        if details:
            merged_details.update(details)
        return self.create(
            selected_code,
            selected_message,
            evidence=selected_evidence,
            recovery=selected_recovery,
            scope=scope,
            action=action,
            recoverable=selected_recoverable,
            recovery_available=recovery_available,
            available=available,
            cause=error,
            details=merged_details,
        )

    build = create
    from_error = from_exception


def serialize_execution_failure(error: ExecutionFailure) -> dict[str, object]:
    if not isinstance(error, ExecutionFailure):
        raise TypeError("serialize_execution_failure requires ExecutionFailure")
    return error.to_dict()


__all__ = [
    "CancellationRequested",
    "DeadlineExceeded",
    "ExecutionFailure",
    "ExecutionFailureFactory",
    "InvalidRecoveryRequirement",
    "OutputValidationError",
    "RECOVERY_MATRIX",
    "RecoveryAction",
    "RecoveryRequirement",
    "RecoveryScope",
    "RuntimeControlError",
    "serialize_execution_failure",
    "VALID_RECOVERY_REQUIREMENTS",
]

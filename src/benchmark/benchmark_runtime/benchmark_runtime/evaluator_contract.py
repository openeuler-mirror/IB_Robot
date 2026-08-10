"""Pure provider-independent evaluator contract compatibility validation."""

from __future__ import annotations

from collections.abc import Iterable


class ContractCompatibilityError(ValueError):
    """Raised when robot, model, and benchmark contracts are incompatible."""


def _normalize_features(
    features: Iterable[tuple[str, str, tuple[int, ...]]],
    *,
    label: str,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    normalized: dict[str, tuple[str, tuple[int, ...]]] = {}
    for key, kind, shape in features:
        if not isinstance(key, str) or not key.strip():
            raise ContractCompatibilityError(f"{label} feature key must be a non-empty string")
        if key in normalized:
            raise ContractCompatibilityError(f"{label} contains duplicate feature key {key!r}")
        if not isinstance(kind, str) or not kind.strip():
            raise ContractCompatibilityError(f"{label} feature {key!r} kind must be a non-empty string")
        dimensions = tuple(shape)
        if not dimensions or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in dimensions
        ):
            raise ContractCompatibilityError(f"{label} feature {key!r} must have a non-empty positive shape")
        normalized[key] = (kind, dimensions)
    if not normalized:
        raise ContractCompatibilityError(f"{label} must contain at least one feature")
    return normalized


def _normalize_actions(actions: Iterable[tuple[str, tuple[int, ...]]]) -> dict[str, tuple[int, ...]]:
    normalized: dict[str, tuple[int, ...]] = {}
    for key, shape in actions:
        if not isinstance(key, str) or not key.strip():
            raise ContractCompatibilityError("action feature key must be a non-empty string")
        if key in normalized:
            raise ContractCompatibilityError(f"actions contains duplicate feature key {key!r}")
        dimensions = tuple(shape)
        if not dimensions or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in dimensions
        ):
            raise ContractCompatibilityError(f"action feature {key!r} must have a non-empty positive shape")
        normalized[key] = dimensions
    if not normalized:
        raise ContractCompatibilityError("actions must contain at least one feature")
    return normalized


def validate_contract_compatibility(
    *,
    observations: tuple[tuple[str, str, tuple[int, ...]], ...],
    actions: tuple[tuple[str, tuple[int, ...]], ...],
    control_mode: str,
    prompt_passthrough: bool,
    expected_observations: tuple[tuple[str, str, tuple[int, ...]], ...] | None = None,
    expected_actions: tuple[tuple[str, tuple[int, ...]], ...] | None = None,
) -> None:
    """Validate arbitrary feature counts and dimensions without provider constants."""

    actual_observations = _normalize_features(observations, label="model observations")
    actual_actions = _normalize_actions(actions)
    if expected_observations is not None:
        expected = _normalize_features(expected_observations, label="robot observations")
        if actual_observations != expected:
            raise ContractCompatibilityError(
                f"robot/model observation contract mismatch: expected {expected}, got {actual_observations}"
            )
    if expected_actions is not None:
        expected_output = _normalize_actions(expected_actions)
        if actual_actions != expected_output:
            raise ContractCompatibilityError(
                f"robot/model action contract mismatch: expected {expected_output}, got {actual_actions}"
            )
    if not isinstance(control_mode, str) or not control_mode.strip():
        raise ContractCompatibilityError("benchmark control mode must be a non-empty string")
    if prompt_passthrough is not True:
        raise ContractCompatibilityError("verbatim prompt passthrough must be enabled")

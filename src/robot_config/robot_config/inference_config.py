"""Typed validation for named inference pipelines in robot configuration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from inference_manifest import (
    MANIFEST_FILENAME,
    CompiledDeployment,
    ManifestError,
    ValidatedManifest,
    load_inference_manifest,
    load_inference_manifest_metadata,
    resolve_bundle_file,
)

from .inference_runtime_options import effective_latency_runtime_options

PIPELINE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_INT32_MAX = 2_147_483_647

_NODE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENDPOINT_PATTERN = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_INFERENCE_FIELDS = frozenset({"enabled", "pipelines", "scheduler"})
_SCHEDULED_PIPELINE_FIELDS = frozenset(
    {
        "required",
        "compatibility_group",
        "hardware_resource_id",
        "hardware_profile_fingerprint",
        "profile_path",
        "public_capacity",
    }
)
_LEGACY_PIPELINE_FIELDS = frozenset(
    {
        "model_path",
        "deployment",
        "execution_mode",
        "request_timeout",
        "default_task",
        "runtime_options",
        "transport",
    }
)
_PIPELINE_FIELDS = _LEGACY_PIPELINE_FIELDS | _SCHEDULED_PIPELINE_FIELDS
_TRANSPORT_FIELDS = (
    "node_name",
    "cloud_node_name",
    "action_server",
    "reset_service",
    "health_topic",
    "action_topic",
    "request_topic",
    "result_topic",
    "heartbeat_topic",
    "video_descriptor_topic",
    "video_status_topic",
    "open_session",
    "dispatch",
    "close_session",
    "serving_status",
)
_DISTRIBUTED_TRANSPORT_FIELDS = frozenset(
    {
        "cloud_node_name",
        "request_topic",
        "result_topic",
        "heartbeat_topic",
        "video_descriptor_topic",
        "video_status_topic",
    }
)
_SCHEDULED_TRANSPORT_FIELDS = ("open_session", "dispatch", "close_session", "serving_status")
_NON_SCHEDULED_INTERFACE_FIELDS = (
    "action_server",
    "reset_service",
    "health_topic",
    "action_topic",
    "request_topic",
    "result_topic",
    "heartbeat_topic",
    "video_descriptor_topic",
    "video_status_topic",
)

# --- scheduler: the single scheduling feature switch and its policy block ---
# `scheduler` is the only addition to _INFERENCE_FIELDS;
# it is not itself a switch: `scheduler.enable` is.
_SCHEDULER_FIELDS = frozenset(
    {
        "enable",
        "global_endpoints",
        "default_open_timeout",
        "default_request_timeout",
        "startup_readiness_timeout",
        "status_stale_timeout",
        "clock_skew_tolerance",
        "goal_acceptance_timeout",
        "goal_acceptance_safety_margin_ms",
        "dispatch_safety_margin_ms",
        "dispatch_goal_contexts",
        "lower_priority_dispatch_goal_contexts",
        "session_idle_timeout",
        "profile_min_samples",
        "profile_max_age_days",
        "max_product_requests_per_session",
        "terminal_result_cache_entries",
        "max_duplicate_waiters_per_request",
        "max_prompt_bytes",
        "max_fallback_pipelines",
        "max_error_message_bytes",
        "max_error_details_bytes",
        "terminal_session_retention",
        "max_session_records",
    }
)
_GLOBAL_ENDPOINT_FIELDS = ("readiness", "open_session", "dispatch", "close_session")
_WORK_CLASSES = ("session_control", "action_generation")
_PUBLIC_CAPACITY_FIELDS = frozenset({"max_in_flight"})
_COMPATIBILITY_GROUP_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_RESOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_:./-]{0,126}$")
# Wire bounds are fixed by the scheduled ROS interfaces; robot_config may only tighten them.
_WIRE_PROMPT_BYTES = 4096
_WIRE_FALLBACK_PIPELINES = 32
_WIRE_ERROR_MESSAGE_BYTES = 1024
_WIRE_ERROR_DETAILS_BYTES = 8192


class InferenceConfigError(ValueError):
    """Inference pipeline configuration is invalid."""


@dataclass(frozen=True)
class InferenceTransportConfig:
    """Resolved ROS names for one pipeline."""

    node_name: str
    cloud_node_name: str | None
    action_server: str
    reset_service: str
    health_topic: str
    action_topic: str
    request_topic: str | None
    result_topic: str | None
    heartbeat_topic: str | None
    video_descriptor_topic: str | None
    video_status_topic: str | None
    open_session: str | None
    dispatch: str | None
    close_session: str | None
    serving_status: str | None


@dataclass(frozen=True)
class InferencePipelineConfig:
    """Validated configuration and bundle metadata for one pipeline."""

    pipeline_id: str
    model_path: Path
    deployment: str
    execution_mode: Literal["monolithic", "distributed"]
    request_timeout: float
    default_task: str
    runtime_options: Mapping[str, object]
    transport: InferenceTransportConfig
    validated_manifest: ValidatedManifest
    # Scheduled-only fields. Present only when scheduler.enable is true;
    # None otherwise.
    required: bool = True
    compatibility_group: str | None = None
    hardware_resource_id: str | None = None
    hardware_profile_fingerprint: str | None = None
    profile_path: Path | None = None
    public_capacity: Mapping[str, InferenceWorkCapacityConfig] = dataclass_field(
        default_factory=lambda: MappingProxyType({})
    )
    # Canonical runtime policy (session/ingress/transport identity) built by
    # _build_runtime_policy_json; None when scheduler disabled.
    runtime_policy_json: str | None = None
    runtime_policy_fingerprint: str | None = None
    profile_compatibility_fingerprint: str | None = None


@dataclass(frozen=True)
class InferenceWorkCapacityConfig:
    """Per work-class public capacity."""

    work_class: str
    max_in_flight: int


@dataclass(frozen=True)
class GlobalEndpoints:
    """Fully-resolved global ROS endpoints for the scheduled path."""

    readiness: str
    open_session: str
    dispatch: str
    close_session: str


@dataclass(frozen=True)
class SchedulerConfig:
    """Typed scheduler policy block.

    enable is the only scheduling feature switch. All timeouts and retentions
    are in nanoseconds; `_ms` suffixes were converted from milliseconds and
    `_days` from UTC day-age at parse time for runtime comparison.
    """

    enable: bool
    global_endpoints: GlobalEndpoints
    default_open_timeout_ns: int
    default_request_timeout_ns: int
    startup_readiness_timeout_ns: int
    status_stale_timeout_ns: int
    clock_skew_tolerance_ns: int
    goal_acceptance_timeout_ns: int
    goal_acceptance_safety_margin_ms: int
    dispatch_safety_margin_ms: int
    dispatch_goal_contexts: int
    lower_priority_dispatch_goal_contexts: int
    session_idle_timeout_ns: int
    profile_min_samples: int
    profile_max_age_days: int
    max_product_requests_per_session: int
    terminal_result_cache_entries: int
    max_duplicate_waiters_per_request: int
    max_prompt_bytes: int
    max_fallback_pipelines: int
    max_error_message_bytes: int
    max_error_details_bytes: int
    terminal_session_retention_ns: int
    max_session_records: int


@dataclass(frozen=True)
class ControlModeInferenceConfig:
    """Typed inference configuration for one robot control mode."""

    control_mode: str
    enabled: bool
    pipelines: Mapping[str, InferencePipelineConfig]
    scheduler: SchedulerConfig | None = None
    # Executor-level fields consumed only by ScheduledActionDispatcher.
    # `inference_pipeline` is the only pre-existing one; fallback_chain/priority/retry
    # are new. None when inference disabled.
    inference_pipeline: str | None = None
    inference_fallback_chain: tuple[str, ...] = ()
    inference_priority: int = 0
    inference_retry: Mapping[str, int] = dataclass_field(default_factory=lambda: MappingProxyType({}))


def scheduler_enabled_from_raw_config(robot_config: Mapping[str, Any], control_mode: str) -> bool:
    """Read only the opt-in switch for launch branches that do not start inference."""

    control_modes = robot_config.get("control_modes")
    if not isinstance(control_modes, Mapping):
        return False
    mode_config = control_modes.get(control_mode)
    if not isinstance(mode_config, Mapping):
        return False
    inference = mode_config.get("inference")
    if not isinstance(inference, Mapping) or inference.get("enabled") is not True:
        return False
    scheduler = inference.get("scheduler")
    return isinstance(scheduler, Mapping) and scheduler.get("enable") is True


def parse_inference_config(
    robot_config: Mapping[str, Any],
    control_mode: str,
) -> ControlModeInferenceConfig:
    """Parse and validate ``control_modes.<mode>.inference`` before launch construction."""

    if not isinstance(robot_config, Mapping):
        raise InferenceConfigError("robot configuration must be a mapping")
    if not isinstance(control_mode, str) or not control_mode:
        raise InferenceConfigError("control mode must be a non-empty string")

    control_modes = robot_config.get("control_modes", {})
    if not isinstance(control_modes, Mapping):
        raise InferenceConfigError("control_modes must be a mapping")
    if control_mode not in control_modes:
        raise InferenceConfigError(f"control mode {control_mode!r} is not configured")

    mode_path = f"control_modes.{control_mode}"
    mode_config = control_modes[control_mode]
    if not isinstance(mode_config, Mapping):
        raise InferenceConfigError(f"{mode_path} must be a mapping")

    inference_path = f"{mode_path}.inference"
    if "inference" not in mode_config:
        return ControlModeInferenceConfig(
            control_mode=control_mode,
            enabled=False,
            pipelines=MappingProxyType({}),
        )

    inference = mode_config["inference"]
    if not isinstance(inference, Mapping):
        raise InferenceConfigError(f"{inference_path} must be a mapping")
    if "model" in inference:
        raise InferenceConfigError(
            f"{inference_path}.model is a legacy field; configure named entries under {inference_path}.pipelines"
        )
    _reject_unknown_fields(inference, _INFERENCE_FIELDS, inference_path)

    enabled = inference.get("enabled", False)
    if not isinstance(enabled, bool):
        raise InferenceConfigError(f"{inference_path}.enabled must be a boolean")

    raw_pipelines = inference.get("pipelines", {})
    if not isinstance(raw_pipelines, Mapping):
        raise InferenceConfigError(f"{inference_path}.pipelines must be a mapping")
    if enabled and not raw_pipelines:
        raise InferenceConfigError(f"{inference_path}.pipelines must be non-empty when inference is enabled")

    pipelines: dict[str, InferencePipelineConfig] = {}
    scheduler_enabled = False
    allow_scheduled_config = "scheduler" in inference
    if "scheduler" in inference:
        # Determine activation before parsing pipelines. The presence of the
        # scheduler block permits dormant scheduled fields so enable alone is
        # a complete runtime rollback switch.
        scheduler_enabled = _peek_scheduler_enable(inference["scheduler"], enabled, inference_path)

    for pipeline_id, pipeline_value in raw_pipelines.items():
        validated_id = _validate_pipeline_id(pipeline_id, inference_path)
        pipelines[validated_id] = _parse_pipeline(
            validated_id,
            pipeline_value,
            inference_path,
            scheduler_enabled=scheduler_enabled,
            allow_scheduled_config=allow_scheduled_config,
        )

    _validate_endpoint_conflicts(pipelines, scheduler_enabled=scheduler_enabled)
    if scheduler_enabled:
        _validate_compatibility_groups(pipelines)

    # --- scheduler block (single switch). `scheduler` is now an allowed field. ---
    scheduler: SchedulerConfig | None = None
    inference_pipeline: str | None = None
    fallback_chain: tuple[str, ...] = ()
    inference_priority = 0
    inference_retry: dict[str, int] = {}
    if "scheduler" in inference:
        scheduler = _parse_scheduler(inference["scheduler"], enabled, pipelines, inference_path)
        # Only parse executor fields when the scheduled path is enabled; legacy
        # dispatcher never reads them and they must not change false-branch behavior.
        if scheduler is not None and scheduler.enable:
            mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
            executor_config = mode_config.get("executor", {}) or {}
            if not isinstance(executor_config, Mapping):
                raise InferenceConfigError(f"control_modes.{control_mode}.executor must be a mapping")
            (
                inference_pipeline,
                fallback_chain,
                inference_priority,
                inference_retry,
            ) = _parse_executor_fields(executor_config, pipelines, f"control_modes.{control_mode}.executor")
            _validate_global_endpoint_conflicts(pipelines, scheduler.global_endpoints)

    return ControlModeInferenceConfig(
        control_mode=control_mode,
        enabled=enabled,
        pipelines=MappingProxyType(pipelines),
        scheduler=scheduler,
        inference_pipeline=inference_pipeline,
        inference_fallback_chain=fallback_chain,
        inference_priority=inference_priority,
        inference_retry=MappingProxyType(inference_retry),
    )


def _validate_pipeline_id(pipeline_id: Any, inference_path: str) -> str:
    if not isinstance(pipeline_id, str) or not PIPELINE_ID_PATTERN.fullmatch(pipeline_id):
        raise InferenceConfigError(
            f"{inference_path}.pipelines contains invalid pipeline ID {pipeline_id!r}; "
            "expected ^[a-z][a-z0-9_]{0,62}$"
        )
    return pipeline_id


def _parse_pipeline(
    pipeline_id: str,
    value: Any,
    inference_path: str,
    *,
    scheduler_enabled: bool = False,
    allow_scheduled_config: bool = False,
) -> InferencePipelineConfig:
    pipeline_path = f"{inference_path}.pipelines.{pipeline_id}"
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{pipeline_path} must be a mapping")
    for legacy_field in ("device", "concurrency"):
        if legacy_field in value:
            raise InferenceConfigError(f"{pipeline_path}.{legacy_field} is not supported")
    _reject_unknown_fields(
        value,
        _PIPELINE_FIELDS if allow_scheduled_config else _LEGACY_PIPELINE_FIELDS,
        pipeline_path,
    )

    model_path_value = _require_string(value, "model_path", pipeline_path)
    deployment = _require_string(value, "deployment", pipeline_path)
    execution_mode = _require_string(value, "execution_mode", pipeline_path)
    if execution_mode not in {"monolithic", "distributed"}:
        raise InferenceConfigError(
            f"{pipeline_path}.execution_mode must be 'monolithic' or 'distributed', got {execution_mode!r}"
        )
    if scheduler_enabled and execution_mode != "monolithic":
        raise InferenceConfigError(
            f"{pipeline_path}.execution_mode must be 'monolithic' when scheduler.enable is true; "
            "distributed inference continues to use the legacy protocol"
        )

    request_timeout = value.get("request_timeout", 5.0)
    if isinstance(request_timeout, bool) or not isinstance(request_timeout, int | float):
        raise InferenceConfigError(f"{pipeline_path}.request_timeout must be a positive finite number")
    request_timeout = float(request_timeout)
    if not math.isfinite(request_timeout) or request_timeout <= 0.0:
        raise InferenceConfigError(f"{pipeline_path}.request_timeout must be a positive finite number")

    default_task = value.get("default_task", "")
    if not isinstance(default_task, str):
        raise InferenceConfigError(f"{pipeline_path}.default_task must be a string")
    runtime_options = _parse_runtime_options(value.get("runtime_options", {}), pipeline_path)

    transport = _parse_transport(
        pipeline_id,
        execution_mode,
        value.get("transport", {}),
        pipeline_path,
        scheduler_enabled=scheduler_enabled,
        allow_scheduled_config=allow_scheduled_config,
    )
    model_path = _resolve_model_path(model_path_value, pipeline_path)
    validated_manifest = _validate_model_bundle(model_path, deployment, execution_mode, pipeline_path)

    # Scheduled fields may remain in the SSOT while disabled, but are
    # materialized only when scheduler.enable=true. This keeps false runtime
    # behavior identical to legacy while making enable a one-line rollback.
    required = True
    compatibility_group: str | None = None
    hardware_resource_id: str | None = None
    hardware_profile_fingerprint: str | None = None
    profile_path: Path | None = None
    public_capacity: dict[str, InferenceWorkCapacityConfig] = {}
    runtime_policy_json: str | None = None
    runtime_policy_fingerprint: str | None = None
    profile_compatibility_fingerprint: str | None = None
    if scheduler_enabled:
        required_raw = value.get("required", True)
        if not isinstance(required_raw, bool):
            raise InferenceConfigError(f"{pipeline_path}.required must be a boolean")
        required = required_raw
        compatibility_group = _require_string(value, "compatibility_group", pipeline_path)
        if not _COMPATIBILITY_GROUP_PATTERN.fullmatch(compatibility_group):
            raise InferenceConfigError(f"{pipeline_path}.compatibility_group must match ^[a-z][a-z0-9_]{{0,62}}$")
        hardware_resource_id = _require_string(value, "hardware_resource_id", pipeline_path)
        if not _RESOURCE_ID_PATTERN.fullmatch(hardware_resource_id):
            raise InferenceConfigError(f"{pipeline_path}.hardware_resource_id contains invalid characters")
        hardware_profile_fingerprint = _require_string(value, "hardware_profile_fingerprint", pipeline_path)
        if not re.fullmatch(r"[0-9a-f]{64}", hardware_profile_fingerprint):
            raise InferenceConfigError(
                f"{pipeline_path}.hardware_profile_fingerprint must be a lowercase SHA-256 hex digest"
            )
        profile_path = _resolve_profile_path(value.get("profile_path"), pipeline_path)
        public_capacity = _parse_public_capacity(value.get("public_capacity", {}), pipeline_path)
        _validate_public_capacity_for_pipeline(public_capacity, pipeline_path)
        _validate_scheduled_artifact_integrity(validated_manifest, execution_mode, pipeline_path)
        runtime_policy_json = _build_runtime_policy_json(
            pipeline_id=pipeline_id,
            execution_mode=execution_mode,
            transport=transport,
            public_capacity=public_capacity,
            required=required,
            compatibility_group=compatibility_group,
            hardware_resource_id=hardware_resource_id,
            hardware_profile_fingerprint=hardware_profile_fingerprint,
            deployment_fingerprint=validated_manifest.fingerprint,
            runtime_options=runtime_options,
        )
        runtime_policy_fingerprint = _sha256_hex(runtime_policy_json)
        profile_compatibility_fingerprint = _sha256_hex(
            _build_profile_compatibility_json(
                execution_mode=execution_mode,
                public_capacity=public_capacity,
                runtime_options=runtime_options,
            )
        )
    return InferencePipelineConfig(
        pipeline_id=pipeline_id,
        model_path=model_path,
        deployment=deployment,
        execution_mode=execution_mode,
        request_timeout=request_timeout,
        default_task=default_task,
        runtime_options=runtime_options,
        transport=transport,
        validated_manifest=validated_manifest,
        required=required,
        compatibility_group=compatibility_group,
        hardware_resource_id=hardware_resource_id,
        hardware_profile_fingerprint=hardware_profile_fingerprint,
        profile_path=profile_path,
        public_capacity=MappingProxyType(public_capacity),
        runtime_policy_json=runtime_policy_json,
        runtime_policy_fingerprint=runtime_policy_fingerprint,
        profile_compatibility_fingerprint=profile_compatibility_fingerprint,
    )


def _require_string(value: Mapping[str, Any], field: str, path: str) -> str:
    if field not in value:
        raise InferenceConfigError(f"{path}.{field} is required")
    field_value = value[field]
    if not isinstance(field_value, str) or not field_value:
        raise InferenceConfigError(f"{path}.{field} must be a non-empty string")
    return field_value


def _parse_runtime_options(value: Any, pipeline_path: str) -> Mapping[str, object]:
    options_path = f"{pipeline_path}.runtime_options"
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{options_path} must be a mapping")
    invalid_keys = sorted(repr(key) for key in value if not isinstance(key, str) or not key)
    if invalid_keys:
        raise InferenceConfigError(f"{options_path} keys must be non-empty strings: {invalid_keys}")
    try:
        normalized = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InferenceConfigError(f"{options_path} must contain JSON-compatible finite values: {exc}") from exc
    return MappingProxyType(normalized)


def _parse_transport(
    pipeline_id: str,
    execution_mode: str,
    value: Any,
    pipeline_path: str,
    *,
    scheduler_enabled: bool,
    allow_scheduled_config: bool,
) -> InferenceTransportConfig:
    transport_path = f"{pipeline_path}.transport"
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{transport_path} must be a mapping")
    allowed_transport_fields = frozenset(_TRANSPORT_FIELDS)
    if not allow_scheduled_config:
        allowed_transport_fields -= frozenset(_SCHEDULED_TRANSPORT_FIELDS)
    _reject_unknown_fields(value, allowed_transport_fields, transport_path)
    if execution_mode == "monolithic":
        distributed_overrides = sorted(_DISTRIBUTED_TRANSPORT_FIELDS.intersection(value))
        if distributed_overrides:
            raise InferenceConfigError(
                f"{transport_path} cannot override distributed fields for a monolithic pipeline: "
                f"{distributed_overrides}"
            )

    defaults: dict[str, str | None] = {
        "node_name": f"inference_{pipeline_id}",
        "cloud_node_name": f"inference_{pipeline_id}_cloud" if execution_mode == "distributed" else None,
        "action_server": f"/inference/{pipeline_id}/dispatch",
        "reset_service": f"/inference/{pipeline_id}/reset",
        "health_topic": f"/inference/{pipeline_id}/health",
        "action_topic": f"/actions/{pipeline_id}",
        "request_topic": f"/inference/{pipeline_id}/request" if execution_mode == "distributed" else None,
        "result_topic": f"/inference/{pipeline_id}/result" if execution_mode == "distributed" else None,
        "heartbeat_topic": f"/inference/{pipeline_id}/heartbeat" if execution_mode == "distributed" else None,
        "video_descriptor_topic": (
            f"/inference/{pipeline_id}/video/descriptors" if execution_mode == "distributed" else None
        ),
        "video_status_topic": f"/inference/{pipeline_id}/video/status" if execution_mode == "distributed" else None,
        "open_session": None,
        "dispatch": None,
        "close_session": None,
        "serving_status": None,
    }
    for field, override in value.items():
        if not isinstance(override, str) or not override:
            raise InferenceConfigError(f"{transport_path}.{field} must be a non-empty string")
        if field in _SCHEDULED_TRANSPORT_FIELDS and not scheduler_enabled:
            # Accepted as dormant SSOT configuration, but deliberately not
            # materialized into the disabled pipeline config.
            if not _ENDPOINT_PATTERN.fullmatch(override):
                raise InferenceConfigError(
                    f"{transport_path}.{field} must be an absolute ROS name with valid slash-separated tokens"
                )
            continue
        defaults[field] = override

    if scheduler_enabled:
        missing = [field for field in _SCHEDULED_TRANSPORT_FIELDS if defaults[field] is None]
        if missing:
            raise InferenceConfigError(
                f"{transport_path} requires scheduled endpoints when scheduler is enabled: {missing}"
            )

    for field in ("node_name", "cloud_node_name"):
        name = defaults[field]
        if name is not None and not _NODE_NAME_PATTERN.fullmatch(name):
            raise InferenceConfigError(
                f"{transport_path}.{field} must be a valid ROS node name containing only letters, digits, and '_'"
            )
    for field in _TRANSPORT_FIELDS[2:]:
        endpoint = defaults[field]
        if endpoint is not None and not _ENDPOINT_PATTERN.fullmatch(endpoint):
            raise InferenceConfigError(
                f"{transport_path}.{field} must be an absolute ROS name with valid slash-separated tokens"
            )

    return InferenceTransportConfig(**defaults)


def _resolve_model_path(value: str, pipeline_path: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        candidate = expanded
    else:
        workspace_value = os.environ.get("WORKSPACE")
        if not workspace_value:
            raise InferenceConfigError(
                f"{pipeline_path}.model_path is relative but WORKSPACE is unset; "
                "no current-directory or repository fallback is allowed"
            )
        workspace = Path(workspace_value).expanduser()
        if not workspace.is_absolute():
            raise InferenceConfigError("WORKSPACE must be an absolute path for relative inference model paths")
        try:
            workspace = workspace.resolve(strict=True)
        except OSError as exc:
            raise InferenceConfigError(f"WORKSPACE cannot be resolved: {workspace}: {exc}") from exc
        candidate = workspace / expanded

    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise InferenceConfigError(f"{pipeline_path}.model_path does not exist: {candidate}: {exc}") from exc
    if not resolved.is_dir():
        raise InferenceConfigError(f"{pipeline_path}.model_path is not a directory: {resolved}")
    return resolved


def _validate_model_bundle(
    model_path: Path,
    deployment: str,
    execution_mode: str,
    pipeline_path: str,
) -> ValidatedManifest:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise InferenceConfigError(f"{pipeline_path} bundle is missing config.json: {config_path}")
    manifest_path = model_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise InferenceConfigError(f"{pipeline_path} bundle is missing {MANIFEST_FILENAME}: {manifest_path}")
    try:
        loader = load_inference_manifest_metadata if execution_mode == "distributed" else load_inference_manifest
        return loader(model_path, deployment)
    except ManifestError as exc:
        raise InferenceConfigError(
            f"{pipeline_path} failed bundle validation for deployment {deployment!r}: {exc}"
        ) from exc


def _validate_scheduled_artifact_integrity(
    validated_manifest: ValidatedManifest,
    execution_mode: str,
    pipeline_path: str,
) -> None:
    deployment = validated_manifest.deployment
    if not isinstance(deployment, CompiledDeployment):
        return
    for role, artifact in deployment.artifacts.items():
        if artifact.sha256 is None:
            raise InferenceConfigError(
                f"{pipeline_path}: scheduled compiled artifact {role!r} must declare a content sha256"
            )
        if execution_mode == "distributed":
            continue
        artifact_path = resolve_bundle_file(validated_manifest.bundle_root, artifact.path)
        digest = hashlib.sha256()
        with artifact_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise InferenceConfigError(f"{pipeline_path}: scheduled compiled artifact {role!r} content sha256 mismatch")


def _reject_unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InferenceConfigError(f"{path} contains unsupported fields: {unknown}")


def _validate_endpoint_conflicts(
    pipelines: Mapping[str, InferencePipelineConfig], *, scheduler_enabled: bool = False
) -> None:
    node_owners: dict[str, tuple[str, str]] = {}
    interface_owners: dict[str, tuple[str, str]] = {}
    for pipeline_id, pipeline in pipelines.items():
        transport = pipeline.transport
        local_interfaces: dict[str, str] = {}
        for field in ("node_name", "cloud_node_name"):
            _record_endpoint_owner(node_owners, pipeline_id, field, getattr(transport, field))
        active_fields = (
            (
                "health_topic",
                "request_topic",
                "result_topic",
                "heartbeat_topic",
                "video_descriptor_topic",
                "video_status_topic",
                *_SCHEDULED_TRANSPORT_FIELDS,
            )
            if scheduler_enabled
            else _NON_SCHEDULED_INTERFACE_FIELDS
        )
        for field in active_fields:
            endpoint = getattr(transport, field)
            if endpoint is not None:
                previous_field = local_interfaces.get(endpoint)
                if previous_field is not None:
                    raise InferenceConfigError(
                        f"Inference endpoint conflict: pipeline {pipeline_id!r} {field} and {previous_field} "
                        f"both use {endpoint!r}"
                    )
                local_interfaces[endpoint] = field
            _record_endpoint_owner(interface_owners, pipeline_id, field, endpoint)


def _validate_global_endpoint_conflicts(
    pipelines: Mapping[str, InferencePipelineConfig], global_endpoints: GlobalEndpoints
) -> None:
    pipeline_endpoints = {
        endpoint
        for pipeline in pipelines.values()
        for field in (
            "health_topic",
            "request_topic",
            "result_topic",
            "heartbeat_topic",
            "video_descriptor_topic",
            "video_status_topic",
            *_SCHEDULED_TRANSPORT_FIELDS,
        )
        if (endpoint := getattr(pipeline.transport, field)) is not None
    }
    for field in _GLOBAL_ENDPOINT_FIELDS:
        endpoint = getattr(global_endpoints, field)
        if endpoint in pipeline_endpoints:
            raise InferenceConfigError(
                f"scheduler global endpoint {field} conflicts with pipeline endpoint {endpoint!r}"
            )


def _record_endpoint_owner(
    owners: dict[str, tuple[str, str]],
    pipeline_id: str,
    field: str,
    endpoint: str | None,
) -> None:
    if endpoint is None:
        return
    previous = owners.get(endpoint)
    if previous is not None and previous[0] != pipeline_id:
        previous_pipeline, previous_field = previous
        raise InferenceConfigError(
            f"Inference endpoint conflict: pipeline {pipeline_id!r} {field} and pipeline "
            f"{previous_pipeline!r} {previous_field} both use {endpoint!r}"
        )
    owners[endpoint] = (pipeline_id, field)


# ---------------------------------------------------------------------------
# Scheduler helpers: single switch, bounded policy, runtime fingerprint
# ---------------------------------------------------------------------------

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000
_NS_PER_DAY = 86_400 * _NS_PER_S


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _peek_scheduler_enable(scheduler_value: Any, inference_enabled: bool, inference_path: str) -> bool:
    """Read scheduler.enable before parsing pipelines so scheduled fields parse under it."""
    if not isinstance(scheduler_value, Mapping):
        raise InferenceConfigError(f"{inference_path}.scheduler must be a mapping")
    enable_raw = scheduler_value.get("enable", False)
    if not isinstance(enable_raw, bool):
        raise InferenceConfigError(f"{inference_path}.scheduler.enable must be a boolean")
    enable = enable_raw
    if enable and not inference_enabled:
        raise InferenceConfigError(f"{inference_path}.scheduler.enable=true requires {inference_path}.enabled=true")
    return enable


def _parse_global_endpoints(value: Any, inference_path: str) -> GlobalEndpoints:
    path = f"{inference_path}.scheduler.global_endpoints"
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{path} must be a mapping")
    _reject_unknown_fields(value, frozenset(_GLOBAL_ENDPOINT_FIELDS), path)
    resolved: dict[str, str] = {}
    for field_name in _GLOBAL_ENDPOINT_FIELDS:
        if field_name not in value:
            raise InferenceConfigError(f"{path}.{field_name} is required when scheduler is enabled")
        endpoint = value[field_name]
        if not isinstance(endpoint, str) or not endpoint:
            raise InferenceConfigError(f"{path}.{field_name} must be a non-empty string")
        if not _ENDPOINT_PATTERN.fullmatch(endpoint):
            raise InferenceConfigError(f"{path}.{field_name} must be an absolute ROS name")
        resolved[field_name] = endpoint
    # No two global endpoints may collide.
    seen: dict[str, str] = {}
    for field_name, endpoint in resolved.items():
        prev = seen.get(endpoint)
        if prev is not None:
            raise InferenceConfigError(f"{path}.{field_name} and {prev} both resolve to {endpoint!r}")
        seen[endpoint] = field_name
    return GlobalEndpoints(**resolved)


def _parse_public_capacity(value: Any, pipeline_path: str) -> dict[str, InferenceWorkCapacityConfig]:
    path = f"{pipeline_path}.public_capacity"
    if not isinstance(value, Mapping) or not value:
        raise InferenceConfigError(f"{path} must be a non-empty mapping of work class -> capacity")
    result: dict[str, InferenceWorkCapacityConfig] = {}
    for work_class, cap_value in value.items():
        if work_class not in _WORK_CLASSES:
            raise InferenceConfigError(f"{path} has unknown work class {work_class!r}; allowed: {list(_WORK_CLASSES)}")
        if not isinstance(cap_value, Mapping):
            raise InferenceConfigError(f"{path}.{work_class} must be a mapping")
        _reject_unknown_fields(cap_value, _PUBLIC_CAPACITY_FIELDS, f"{path}.{work_class}")
        max_in_flight_raw = cap_value.get("max_in_flight")
        if max_in_flight_raw is None:
            raise InferenceConfigError(f"{path}.{work_class}.max_in_flight is required")
        max_in_flight = _require_pos_int(max_in_flight_raw, f"{path}.{work_class}.max_in_flight")
        result[work_class] = InferenceWorkCapacityConfig(work_class=work_class, max_in_flight=max_in_flight)
    return result


def _validate_public_capacity_for_pipeline(
    public_capacity: Mapping[str, InferenceWorkCapacityConfig],
    pipeline_path: str,
) -> None:
    if "session_control" not in public_capacity:
        raise InferenceConfigError(f"{pipeline_path}.public_capacity.session_control is required")
    if "action_generation" not in public_capacity:
        raise InferenceConfigError(f"{pipeline_path}.public_capacity.action_generation is required")
    sc = public_capacity["session_control"]
    if sc.max_in_flight != 1:
        raise InferenceConfigError(f"{pipeline_path}.public_capacity.session_control.max_in_flight must be 1")


def _resolve_profile_path(value: Any, pipeline_path: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise InferenceConfigError(f"{pipeline_path}.profile_path must be a non-empty string")
    expanded = Path(value).expanduser()
    candidate = expanded if expanded.is_absolute() else Path(os.environ.get("WORKSPACE", "")) / expanded
    if not candidate.is_file():
        raise InferenceConfigError(f"{pipeline_path}.profile_path does not exist: {candidate}")
    return candidate.resolve(strict=True)


def _build_runtime_policy_json(
    *,
    pipeline_id: str,
    execution_mode: str,
    transport: InferenceTransportConfig,
    public_capacity: Mapping[str, InferenceWorkCapacityConfig],
    required: bool,
    compatibility_group: str,
    hardware_resource_id: str,
    hardware_profile_fingerprint: str,
    deployment_fingerprint: str,
    runtime_options: Mapping[str, object],
) -> str:
    """Canonical JSON of the per-pipeline runtime policy, hashed into the fingerprint.

    Covers session/ingress limits, transport identity, and the normalized
    effective runtime options. Runtime options change model execution latency,
    so the pipeline node validates its actually-executed options against this
    identity at startup and refuses to serve under a policy declared with
    different options. Profile evidence has its own digest and lifecycle, so
    it must not participate in this fingerprint.
    """
    public_capacity_payload = {
        wc.work_class: {
            "max_in_flight": wc.max_in_flight,
        }
        for wc in sorted(public_capacity.values(), key=lambda c: c.work_class)
    }
    transport_payload = {
        "node_name": transport.node_name,
        "health_topic": transport.health_topic,
        "open_session": transport.open_session,
        "dispatch": transport.dispatch,
        "close_session": transport.close_session,
        "serving_status": transport.serving_status,
    }
    payload = {
        "pipeline_id": pipeline_id,
        "execution_mode": execution_mode,
        "required": required,
        "compatibility_group": compatibility_group,
        "hardware_resource_id": hardware_resource_id,
        "hardware_profile_fingerprint": hardware_profile_fingerprint,
        "deployment_fingerprint": deployment_fingerprint,
        "transport": transport_payload,
        "public_capacity": public_capacity_payload,
        "runtime_options": effective_latency_runtime_options(runtime_options),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_profile_compatibility_json(
    *,
    execution_mode: str,
    public_capacity: Mapping[str, InferenceWorkCapacityConfig],
    runtime_options: Mapping[str, object],
) -> str:
    """Canonical timing-policy identity used by offline p99 profiles.

    Endpoint names, routing membership, and required/optional status do not
    affect measured closure latency and therefore intentionally stay out of
    this fingerprint. Deployment and hardware identities remain separate
    mandatory fields in each profile entry. Runtime options change model
    execution latency (for example AutoHorizon attention collection), so
    their normalized effective values participate: profiles calibrated
    under different options are rejected instead of reused.
    """

    payload = {
        "execution_mode": execution_mode,
        "public_capacity": {
            capacity.work_class: {"max_in_flight": capacity.max_in_flight}
            for capacity in sorted(public_capacity.values(), key=lambda item: item.work_class)
        },
        "runtime_options": effective_latency_runtime_options(runtime_options),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_compatibility_groups(pipelines: Mapping[str, InferencePipelineConfig]) -> None:
    """Pipelines sharing a compatibility group must be operationally interchangeable.

    The loader validates group membership. Each pipeline builds its full compatibility
    fingerprint from the synthesized contract at runtime, and Global rejects a fallback
    whose fingerprint differs from the target.
    """
    groups: dict[str, list[str]] = {}
    for pipeline in pipelines.values():
        if pipeline.compatibility_group is None:
            continue
        groups.setdefault(pipeline.compatibility_group, []).append(pipeline.pipeline_id)
    for group, members in groups.items():
        if not members:
            continue
        # Within a group, deployment/runtime_policy fingerprints must be comparable:
        # distinct deployment fingerprints are allowed (fallback target may differ),
        # but runtime_policy_fingerprint must exist for every scheduled pipeline.
        for pid in members:
            pipeline = pipelines[pid]
            if pipeline.runtime_policy_fingerprint is None:
                raise InferenceConfigError(
                    f"pipeline {pid!r} in compatibility group {group!r} has no runtime_policy_fingerprint"
                )


def _parse_scheduler(
    scheduler_value: Any,
    inference_enabled: bool,
    pipelines: Mapping[str, InferencePipelineConfig],
    inference_path: str,
) -> SchedulerConfig | None:
    """Parse the scheduler policy block. Returns None only when enable is false.

    When enable is false the block is structurally validated but no SchedulerConfig
    is materialized, so the false branch creates no endpoints or sessions.
    """
    path = f"{inference_path}.scheduler"
    if not isinstance(scheduler_value, Mapping):
        raise InferenceConfigError(f"{path} must be a mapping")
    _reject_unknown_fields(scheduler_value, _SCHEDULER_FIELDS, path)

    enable_raw = scheduler_value.get("enable", False)
    if not isinstance(enable_raw, bool):
        raise InferenceConfigError(f"{path}.enable must be a boolean")
    enable = enable_raw
    if enable and not inference_enabled:
        raise InferenceConfigError(f"{path}.enable=true requires {inference_path}.enabled=true")

    if not enable:
        # The false branch validates known retained fields but does not
        # materialize a SchedulerConfig or create scheduled runtime state.
        return None

    global_endpoints = _parse_global_endpoints(scheduler_value.get("global_endpoints"), inference_path)

    default_open_timeout = _require_pos_seconds(
        scheduler_value.get("default_open_timeout", 10.0), f"{path}.default_open_timeout"
    )
    default_request_timeout = _require_pos_seconds(
        scheduler_value.get("default_request_timeout", 5.0), f"{path}.default_request_timeout"
    )
    startup_readiness_timeout = _require_pos_seconds(
        scheduler_value.get("startup_readiness_timeout", 60.0), f"{path}.startup_readiness_timeout"
    )
    status_stale_timeout = _require_pos_seconds(
        scheduler_value.get("status_stale_timeout", 2.0), f"{path}.status_stale_timeout"
    )
    clock_skew_tolerance = _require_nonneg_seconds(
        scheduler_value.get("clock_skew_tolerance", 0.1), f"{path}.clock_skew_tolerance"
    )
    goal_acceptance_timeout = _require_pos_seconds(
        scheduler_value.get("goal_acceptance_timeout", 1.0), f"{path}.goal_acceptance_timeout"
    )
    goal_acceptance_safety_margin_ms = _require_nonneg_int(
        scheduler_value.get("goal_acceptance_safety_margin_ms", 100), f"{path}.goal_acceptance_safety_margin_ms"
    )
    dispatch_safety_margin_ms = _require_nonneg_int(
        scheduler_value.get("dispatch_safety_margin_ms", 10), f"{path}.dispatch_safety_margin_ms"
    )
    dispatch_goal_contexts = _require_pos_int(
        scheduler_value.get("dispatch_goal_contexts", 4), f"{path}.dispatch_goal_contexts"
    )
    lower_priority_dispatch_goal_contexts = _require_pos_int(
        scheduler_value.get("lower_priority_dispatch_goal_contexts", 2),
        f"{path}.lower_priority_dispatch_goal_contexts",
    )
    if lower_priority_dispatch_goal_contexts >= dispatch_goal_contexts:
        raise InferenceConfigError(
            f"{path}.lower_priority_dispatch_goal_contexts must be less than dispatch_goal_contexts"
        )
    session_idle_timeout = _require_pos_seconds(
        scheduler_value.get("session_idle_timeout", 30.0), f"{path}.session_idle_timeout"
    )
    profile_min_samples = _require_pos_int(
        scheduler_value.get("profile_min_samples", 10000), f"{path}.profile_min_samples"
    )
    if profile_min_samples < 10000:
        raise InferenceConfigError(f"{path}.profile_min_samples must be >= 10000")
    profile_max_age_days = _require_pos_int(
        scheduler_value.get("profile_max_age_days", 30), f"{path}.profile_max_age_days"
    )
    max_product_requests = _require_pos_int(
        scheduler_value.get("max_product_requests_per_session", 100000),
        f"{path}.max_product_requests_per_session",
    )
    terminal_result_cache = _require_pos_int(
        scheduler_value.get("terminal_result_cache_entries", 256), f"{path}.terminal_result_cache_entries"
    )
    max_duplicate_waiters = _require_pos_int(
        scheduler_value.get("max_duplicate_waiters_per_request", 4),
        f"{path}.max_duplicate_waiters_per_request",
    )
    max_prompt_bytes = _require_pos_int(scheduler_value.get("max_prompt_bytes", 4096), f"{path}.max_prompt_bytes")
    if max_prompt_bytes > _WIRE_PROMPT_BYTES:
        raise InferenceConfigError(
            f"{path}.max_prompt_bytes={max_prompt_bytes} exceeds wire bound {_WIRE_PROMPT_BYTES}"
        )
    max_fallback_pipelines = _require_pos_int(
        scheduler_value.get("max_fallback_pipelines", 16), f"{path}.max_fallback_pipelines"
    )
    if max_fallback_pipelines > _WIRE_FALLBACK_PIPELINES:
        raise InferenceConfigError(
            f"{path}.max_fallback_pipelines={max_fallback_pipelines} exceeds wire bound {_WIRE_FALLBACK_PIPELINES}"
        )
    max_error_message_bytes = _require_pos_int(
        scheduler_value.get("max_error_message_bytes", 1024), f"{path}.max_error_message_bytes"
    )
    if max_error_message_bytes > _WIRE_ERROR_MESSAGE_BYTES:
        raise InferenceConfigError(
            f"{path}.max_error_message_bytes={max_error_message_bytes} exceeds wire bound {_WIRE_ERROR_MESSAGE_BYTES}"
        )
    max_error_details_bytes = _require_pos_int(
        scheduler_value.get("max_error_details_bytes", 8192), f"{path}.max_error_details_bytes"
    )
    if max_error_details_bytes > _WIRE_ERROR_DETAILS_BYTES:
        raise InferenceConfigError(
            f"{path}.max_error_details_bytes={max_error_details_bytes} exceeds wire bound {_WIRE_ERROR_DETAILS_BYTES}"
        )
    terminal_session_retention = _require_pos_seconds(
        scheduler_value.get("terminal_session_retention", 300.0), f"{path}.terminal_session_retention"
    )
    max_session_records = _require_pos_int(
        scheduler_value.get("max_session_records", 256), f"{path}.max_session_records"
    )

    # dispatch_safety_margin_ms must be non-negative and less than the session idle timeout.
    if dispatch_safety_margin_ms >= int(session_idle_timeout * 1000):
        raise InferenceConfigError(
            f"{path}.dispatch_safety_margin_ms={dispatch_safety_margin_ms} must be < "
            f"session_idle_timeout*1000={int(session_idle_timeout * 1000)}"
        )

    return SchedulerConfig(
        enable=True,
        global_endpoints=global_endpoints,
        default_open_timeout_ns=int(default_open_timeout * _NS_PER_S),
        default_request_timeout_ns=int(default_request_timeout * _NS_PER_S),
        startup_readiness_timeout_ns=int(startup_readiness_timeout * _NS_PER_S),
        status_stale_timeout_ns=int(status_stale_timeout * _NS_PER_S),
        clock_skew_tolerance_ns=int(clock_skew_tolerance * _NS_PER_S),
        goal_acceptance_timeout_ns=int(goal_acceptance_timeout * _NS_PER_S),
        goal_acceptance_safety_margin_ms=goal_acceptance_safety_margin_ms,
        dispatch_safety_margin_ms=dispatch_safety_margin_ms,
        dispatch_goal_contexts=dispatch_goal_contexts,
        lower_priority_dispatch_goal_contexts=lower_priority_dispatch_goal_contexts,
        session_idle_timeout_ns=int(session_idle_timeout * _NS_PER_S),
        profile_min_samples=profile_min_samples,
        profile_max_age_days=profile_max_age_days,
        max_product_requests_per_session=max_product_requests,
        terminal_result_cache_entries=terminal_result_cache,
        max_duplicate_waiters_per_request=max_duplicate_waiters,
        max_prompt_bytes=max_prompt_bytes,
        max_fallback_pipelines=max_fallback_pipelines,
        max_error_message_bytes=max_error_message_bytes,
        max_error_details_bytes=max_error_details_bytes,
        terminal_session_retention_ns=int(terminal_session_retention * _NS_PER_S),
        max_session_records=max_session_records,
    )


def _parse_executor_fields(
    executor_config: Mapping[str, Any],
    pipelines: Mapping[str, InferencePipelineConfig],
    executor_path: str,
) -> tuple[str, tuple[str, ...], int, dict[str, int]]:
    """Parse executor.inference_* fields consumed only by ScheduledActionDispatcher."""
    selection = executor_config.get("inference_pipeline")
    if not isinstance(selection, str) or not selection:
        raise InferenceConfigError(
            f"{executor_path}.inference_pipeline must be a non-empty pipeline ID when scheduler is enabled"
        )
    if selection not in pipelines:
        raise InferenceConfigError(
            f"{executor_path}.inference_pipeline selects unknown pipeline {selection!r}; available: {list(pipelines)}"
        )
    # The target must not appear in fallback; negative priority is rejected at startup.
    fallback_raw = executor_config.get("inference_fallback_chain", []) or []
    if not isinstance(fallback_raw, list | tuple):
        raise InferenceConfigError(f"{executor_path}.inference_fallback_chain must be a list")
    # The wire bound (<=32) is enforced here; the configured max_fallback_pipelines
    # (which may be tighter) is enforced at Scheduler runtime admission.
    if len(fallback_raw) > _WIRE_FALLBACK_PIPELINES:
        raise InferenceConfigError(
            f"{executor_path}.inference_fallback_chain has {len(fallback_raw)} entries; "
            f"wire bound is {_WIRE_FALLBACK_PIPELINES}"
        )
    fallback: list[str] = []
    for item in fallback_raw:
        if not isinstance(item, str) or not item:
            raise InferenceConfigError(f"{executor_path}.inference_fallback_chain entries must be non-empty strings")
        if item == selection:
            raise InferenceConfigError(f"{executor_path}.inference_fallback_chain must not contain the target pipeline")
        if item not in pipelines:
            raise InferenceConfigError(f"{executor_path}.inference_fallback_chain references unknown pipeline {item!r}")
        if item in fallback:
            raise InferenceConfigError(f"{executor_path}.inference_fallback_chain must not repeat pipeline {item!r}")
        fallback.append(item)

    priority_raw = executor_config.get("inference_priority", 0)
    priority = _require_nonneg_int(priority_raw, f"{executor_path}.inference_priority")
    if priority > _INT32_MAX:
        raise InferenceConfigError(f"{executor_path}.inference_priority must fit the int32 action field")

    retry_raw = executor_config.get("inference_retry", {}) or {}
    if not isinstance(retry_raw, Mapping):
        raise InferenceConfigError(f"{executor_path}.inference_retry must be a mapping")
    _reject_unknown_fields(
        retry_raw,
        frozenset({"max_not_started_attempts", "initial_backoff_ms", "max_backoff_ms"}),
        f"{executor_path}.inference_retry",
    )
    max_attempts = _require_nonneg_int(
        retry_raw.get("max_not_started_attempts", 3), f"{executor_path}.inference_retry.max_not_started_attempts"
    )
    initial_backoff = _require_nonneg_int(
        retry_raw.get("initial_backoff_ms", 50), f"{executor_path}.inference_retry.initial_backoff_ms"
    )
    max_backoff = _require_nonneg_int(
        retry_raw.get("max_backoff_ms", 500), f"{executor_path}.inference_retry.max_backoff_ms"
    )
    if max_backoff < initial_backoff:
        raise InferenceConfigError(
            f"{executor_path}.inference_retry.max_backoff_ms={max_backoff} must be >= "
            f"initial_backoff_ms={initial_backoff}"
        )
    retry = {
        "max_not_started_attempts": max_attempts,
        "initial_backoff_ms": initial_backoff,
        "max_backoff_ms": max_backoff,
    }
    return selection, tuple(fallback), priority, retry


def _require_nonneg_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InferenceConfigError(f"{path} must be a non-negative integer")
    if value < 0:
        raise InferenceConfigError(f"{path} must be a non-negative integer")
    return value


def _require_pos_int(value: Any, path: str) -> int:
    result = _require_nonneg_int(value, path)
    if result <= 0:
        raise InferenceConfigError(f"{path} must be a positive integer")
    return result


def _require_pos_seconds(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InferenceConfigError(f"{path} must be a positive finite number (seconds)")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise InferenceConfigError(f"{path} must be a positive finite number (seconds)")
    return value


def _require_nonneg_seconds(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InferenceConfigError(f"{path} must be a finite non-negative number (seconds)")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise InferenceConfigError(f"{path} must be a finite non-negative number (seconds)")
    return value


__all__ = [
    "ControlModeInferenceConfig",
    "GlobalEndpoints",
    "InferenceConfigError",
    "InferencePipelineConfig",
    "InferenceTransportConfig",
    "InferenceWorkCapacityConfig",
    "PIPELINE_ID_PATTERN",
    "SchedulerConfig",
    "parse_inference_config",
]

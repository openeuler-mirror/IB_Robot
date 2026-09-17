"""Small constructors for the typed ROS dispatch envelope."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from embodied_common.canon import sha256_text, to_canonical_json
from ibrobot_msgs.msg import DelegatedExecutorIdentity, DispatchBinding, WorkflowStep


def new_binding(*, task_id: str = "", root_task_id: str = "", trace_id: str = "") -> DispatchBinding:
    binding = DispatchBinding()
    binding.schema_version = 1
    binding.task_id = task_id
    binding.root_task_id = root_task_id or task_id
    binding.workflow_step_index = 0
    binding.trace_id = trace_id
    return binding


def copy_binding(source: DispatchBinding) -> DispatchBinding:
    target = DispatchBinding()
    target.schema_version = source.schema_version
    target.task_id = source.task_id
    target.root_task_id = source.root_task_id
    target.task_budget = source.task_budget
    target.expected_registry_epoch = source.expected_registry_epoch
    target.expected_registry_generation = source.expected_registry_generation
    target.expected_registry_digest = source.expected_registry_digest
    target.workflow_digest = source.workflow_digest
    target.workflow_step_index = source.workflow_step_index
    target.root_lease_nonce = source.root_lease_nonce
    target.dispatch_nonce = source.dispatch_nonce
    target.trace_id = source.trace_id
    return target


def delegated_executor_identity(
    *,
    name: str,
    endpoint_name: str,
    contract_version: str = "1",
    endpoint_kind: str = "ros_action",
    configuration: Any = None,
    model_deployment_name: str = "",
    model_fingerprint: str = "",
    model_bundle_digest: str = "",
) -> dict[str, str]:
    configuration_digest = sha256_text(
        to_canonical_json(
            {
                "endpoint_kind": endpoint_kind,
                "endpoint_name": endpoint_name,
                "configuration": configuration if configuration is not None else {},
            }
        )
    )
    model_fields = (model_deployment_name, model_fingerprint, model_bundle_digest)
    if any(model_fields) and not all(model_fields):
        raise ValueError("model identity fields must be all present or all empty")
    return {
        "name": name,
        "contract_version": contract_version,
        "endpoint_kind": endpoint_kind,
        "endpoint_name": endpoint_name,
        "configuration_digest": configuration_digest,
        "model_deployment_name": model_deployment_name,
        "model_fingerprint": model_fingerprint,
        "model_bundle_digest": model_bundle_digest,
    }


def load_delegated_model_identity(configuration: Any) -> dict[str, str]:
    """Load the selected model deployment identity from a strict manifest."""
    if not isinstance(configuration, dict):
        return {"model_deployment_name": "", "model_fingerprint": "", "model_bundle_digest": ""}
    bundle_path = str(configuration.get("model_bundle_path", "")).strip()
    deployment = str(configuration.get("model_deployment", "")).strip()
    if not bundle_path or not deployment:
        return {"model_deployment_name": "", "model_fingerprint": "", "model_bundle_digest": ""}

    def replace_env(match: re.Match[str]) -> str:
        name = match.group(1)
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"environment variable is not set: {name}")
        return value

    bundle_path = re.sub(r"\$\(env\s+(\w+)\)", replace_env, bundle_path)
    try:
        from inference_manifest import load_inference_manifest

        validated = load_inference_manifest(Path(bundle_path), deployment)
    except Exception as exc:
        raise ValueError(f"delegated model manifest is invalid: {exc}") from exc
    return {
        "model_deployment_name": validated.deployment_name,
        "model_fingerprint": validated.fingerprint,
        "model_bundle_digest": validated.manifest.bundle.digest.value,
    }


def fill_delegated_executor_identity(message: DelegatedExecutorIdentity, identity: dict[str, str]) -> None:
    message.schema_version = 1
    for field_name, value in identity.items():
        setattr(message, field_name, value)


def delegated_executor_identity_matches(message: DelegatedExecutorIdentity, identity: dict[str, str]) -> bool:
    return message.schema_version == 1 and all(
        getattr(message, field_name) == value for field_name, value in identity.items()
    )


def workflow_step(
    *,
    schema_version: int = 1,
    skill_name: str,
    target_name: str = "",
    container_name: str = "",
    place_name: str = "",
    motion_direction: str = "",
    motion_distance: float = 0.0,
    arm_side: str = "",
    imitation_duration_sec: float = 0.0,
    timeout_sec: float = 0.0,
    direction: str = "",
    distance: float = 0.0,
    degree: float = 0.0,
    x: float | None = None,
    y: float | None = None,
    yaw: float | None = None,
) -> WorkflowStep:
    step = WorkflowStep()
    step.schema_version = schema_version
    step.skill_name = skill_name
    step.target_name = target_name
    step.container_name = container_name
    step.place_name = place_name
    step.motion_direction = motion_direction
    step.motion_distance = float(motion_distance)
    step.arm_side = str(arm_side)
    step.imitation_duration_sec = float(imitation_duration_sec)
    step.timeout_sec = float(timeout_sec)
    step.direction = direction
    step.distance = float(distance)
    step.degree = float(degree)
    step.has_x = x is not None
    step.x = 0.0 if x is None else float(x)
    step.has_y = y is not None
    step.y = 0.0 if y is None else float(y)
    step.has_yaw = yaw is not None
    step.yaw = 0.0 if yaw is None else float(yaw)
    return step


def binding_task_id(value: Any) -> str:
    return str(value.dispatch_binding.task_id).strip()

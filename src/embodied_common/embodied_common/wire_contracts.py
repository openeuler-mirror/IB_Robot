"""Startup checks for generated public request wire contracts."""

from __future__ import annotations


def validate_public_request_wire_contracts() -> None:
    """Reject stale generated interfaces that do not expose the version prefix."""
    from ibrobot_msgs.action import PrimitiveCommand, SkillCommand
    from ibrobot_msgs.srv import ValidatePrimitive, ValidateSkill

    request_types = (
        ("SkillCommand.Goal", SkillCommand.Goal),
        ("PrimitiveCommand.Goal", PrimitiveCommand.Goal),
        ("ValidateSkill.Request", ValidateSkill.Request),
        ("ValidatePrimitive.Request", ValidatePrimitive.Request),
    )
    expected_prefix = ("schema_version", "uint32")
    for type_name, request_type in request_types:
        try:
            fields = list(request_type.get_fields_and_field_types().items())
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"public request wire contract mismatch for {type_name}: first field must be uint32 schema_version"
            ) from exc
        if not fields or fields[0] != expected_prefix:
            raise RuntimeError(
                f"public request wire contract mismatch for {type_name}: first field must be uint32 schema_version"
            )
        if type_name in {"SkillCommand.Goal", "ValidateSkill.Request"}:
            field_types = dict(fields)
            if field_types.get("arm_side") != "string" or field_types.get("imitation_duration_sec") != "float":
                raise RuntimeError(
                    f"public request wire contract mismatch for {type_name}: schema_version contract must include "
                    "string arm_side and float32 imitation_duration_sec"
                )

    try:
        from ibrobot_msgs.msg import AgentPlan
        from ibrobot_msgs.srv import ConfirmAgentPlan, PlanAgentCommand
    except ModuleNotFoundError:
        # Keep stale-overlay tests focused on the shared public request prefix.
        # Full environments validate the AgentPlan execution-mode fields below.
        return

    for type_name, request_type, required_field in (
        ("PlanAgentCommand.Request", PlanAgentCommand.Request, "execution_mode"),
        ("ConfirmAgentPlan.Request", ConfirmAgentPlan.Request, "execution_mode"),
        ("AgentPlan", AgentPlan, "execution_mode"),
    ):
        try:
            fields = request_type.get_fields_and_field_types()
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(f"public request wire contract mismatch for {type_name}") from exc
        if required_field not in fields:
            raise RuntimeError(f"public request wire contract mismatch for {type_name}: missing {required_field}")

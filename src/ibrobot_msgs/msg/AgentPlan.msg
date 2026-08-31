# AgentPlan.msg - Short-lived immutable planned instance stored in the
# embodied_agent Agent plan store for the Hermes natural-language flow.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.6.
#
# plan_digest is the SHA-256 over (schema_version, raw_command, ordered
# workflow_steps, registry_epoch, registry_generation, registry_digest) and is
# NOT the final workflow_digest (which also binds root task id + TaskBudget).
# plan_token is an opaque >=128-bit random lookup token; it does not enter the
# digest, is not motion authorization and cannot be constructed by Hermes.
# A v1 plan contains at most 16 typed steps. plan_kind is SINGLE_SKILL (1) or
# WORKFLOW (2). execution_mode is captured from the plan request and is part of
# the confirmation contract. schema_version is 1 in v1.

uint32 schema_version
uint8 SINGLE_SKILL=1
uint8 WORKFLOW=2

string plan_id
string plan_token
uint8 plan_kind
string raw_command
WorkflowStep[] workflow_steps
string plan_digest
string registry_epoch
uint64 registry_generation
string registry_digest
builtin_interfaces/Time expires_at
string execution_mode

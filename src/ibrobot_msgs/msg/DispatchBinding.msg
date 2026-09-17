# DispatchBinding.msg - Strongly typed task/version envelope that every
# cross-process primitive, delegated, Skill, Validate and Workflow dispatch
# must carry. Converges the previous ad-hoc task_id fields into one identity.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.5.
#
# task_id identifies the current dispatch; root_task_id is stable across an
# entire Workflow. For root Task or direct root Skill the two are equal.
# dispatch_nonce is only for Gateway-admitted internal delegated/primitive
# dispatch and must be empty on root TaskCommand, ValidateSkill and
# SkillCommand. root_lease_nonce is only for Workflow child SkillCommand (and
# its read-only validation / internal dispatch) and must be empty on raw /
# planned TaskCommand, direct root SkillCommand and Begin/Finalize requests.
# Root-scope binding has no real step and uses workflow_step_index=0 as the
# canonical sentinel; it does not enter workflow_digest.

uint32 schema_version
string task_id
string root_task_id
TaskBudget task_budget
string expected_registry_epoch
uint64 expected_registry_generation
string expected_registry_digest
string workflow_digest
uint32 workflow_step_index
string root_lease_nonce
string dispatch_nonce
string trace_id

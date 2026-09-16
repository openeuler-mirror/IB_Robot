# SO-101 Consumer Migration Handoff

Implementation target: `/home/xqw/Research/IB_Robot-so101-refactor`.
Read-only reuse source: `/home/xqw/Research/IB_Robot-robot-runtime`, compared
against `d770a5c08556726651f329f743fad06fb9ca3f50`. Source implementation and
untracked D13 tests were selectively reused, not whole-branch cherry-picked.

## Behavior

- Provider-bound dispatch consumes public endpoints and actual QoS. Policy
  admission acquires `control_modes.model_inference.runtime_mode` from observed
  idle, not from a shared stream mode. Runtime revocation clears pending work
  and requires explicit restart. Acquisition and stale status are bounded.
- Providerless dispatch/task execution keeps its concrete legacy path.
- Pick execution uses neutral ComputeIk/ComputeFk. Legacy MoveIt definitions,
  SO-101 geometry, and wrist guards load only through explicit `legacy_moveit`.
  New providers use declared suite modules and a bound public description.
- Skills opt into runtime admission explicitly, reject missing/stale/unhealthy
  status and missing capabilities, and confirm mode changes from fresh status.
  They cannot take over an active policy/teleop stream without explicit idle.
- Embodied launch delegates provider startup and binding to robot.launch once.
  Consumers receive one atomically published effective config. Providerless
  configurations retain controller barriers and optional legacy IK workers.
- `embodied_common/trajectory_templates.py`: the explicit legacy `so101_arm_v1`
  lookup resolves against `so101_description` (the `robot_description` and
  `robot_moveit` packages were removed; generic launch ships no planner).

## Parent Integration Hooks

1. Public trajectory descriptions must identify `target_group: arm` and
   `target_group: gripper`, preferably with their ordered `joint_names`.
   Current generated `joint.trajectory_N` entries do not carry that identity.
   Embodied launch deliberately rejects ambiguous/unlabelled arm actions.
2. Set `task_dispatch.gripper_trajectory_interface` to the logical public
   gripper action ID. Task execution does not guess from controller names or
   trajectory array index. Optional overrides are `move_to_pose_interface`
   (default `motion.move_to_pose`) and `joint_state_interface` (`joint.state`).
3. Retain descriptor.model -> config.robot_model and joint_groups -> joints.
   Embodied launch projects model home, limits, runtime mode mapping, named arm
   action, `motion.ee_pose`, `joint.state`, and neutral motion endpoints.
4. Runtime policy/teleop start idle. Policy requests `policy_stream`; teleop
   requests `stream`. The facade must serialize transitions and require idle
   between these distinct modes. No generic multi-writer arbiter was added.
5. Suite metadata stays parent-owned: declare geometry/wrist providers and
   mesh metadata for enabled new-path grasp features. Do not copy legacy IK
   worker/group fields into migrated grasp config.
6. Five reported inference_service failures are outside this ownership and
   were not edited or claimed resolved. The parent's prior runtime E2E result
   is distinct from the focused consumer evidence recorded here.

## Verification

All commands ran from the target root using the dedicated local environment:

```bash
env -i HOME="$HOME" USER=xqw TERM=xterm \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash --noprofile --norc -c 'source /tmp/opencode/so101-env.sh && <command>'
```

| Command after environment initialization | Result |
| --- | --- |
| `ROS_DOMAIN_ID=219 ROS_LOCALHOST_ONLY=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/action_dispatch/test -q --tb=short` | 449 passed |
| `ROS_DOMAIN_ID=220 ROS_LOCALHOST_ONLY=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -o pythonpath="src/task_dispatch" src/task_dispatch/test -q --tb=short` | 23 passed |
| `ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/manipulation_execution/test -q --tb=short` | 216 passed |
| `ROS_DOMAIN_ID=217 IBROBOT_TEST_ROS_DOMAIN_ID=217 ROS_LOCALHOST_ONLY=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/skill_library/test -q --tb=short` | 269 passed |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/skill_catalog/test -q --tb=short` | 78 passed |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/embodied_bringup/test -q --tb=short` | 59 passed, one upstream xacro deprecation warning |

Total: 1094 passing tests. Ruff check and formatting were limited to modified
and new Python files in the six owned packages; check passed. Scoped
`git diff --check` passed. Gateway tests require their documented isolated ROS
domain variables; omitting them causes 43 fixture errors, not runtime failures.
No installs, builds, Docker, physical operations, commits, or pushes were run.
Full embodied launch on the SO-101 snapshot remains gated by the parent hooks.

## Exact Changed Files

Paths are relative to the target root. This list excludes parent-owned edits.
New helpers/tests not present in the source prototype are local extensions;
the D13 binding/startup tests include source untracked test reuse.

```text
src/action_dispatch/action_dispatch/action_dispatcher_node.py
src/action_dispatch/action_dispatch/contract_binding.py
src/action_dispatch/action_dispatch/policy_admission.py
src/action_dispatch/action_dispatch/executors/topic.py
src/action_dispatch/action_dispatch/scheduled_action_dispatcher_node.py
src/action_dispatch/package.xml
src/action_dispatch/test/test_blending_parameter_roundtrip.py
src/action_dispatch/test/test_legacy_contract_binding.py
src/action_dispatch/test/test_policy_admission.py
src/action_dispatch/test/test_scheduled_dispatcher_lifecycle.py
src/action_dispatch/test/test_topic_executor.py
src/action_dispatch/test/test_topic_executor_bindings.py
src/embodied_bringup/SO101_CONSUMER_MIGRATION.md
src/embodied_bringup/embodied_bringup/launch_builders/embodied.py
src/embodied_bringup/launch/embodied_pipeline.launch.py
src/embodied_bringup/package.xml
src/embodied_bringup/test/conftest.py
src/embodied_bringup/test/test_embodied_launch_builder.py
src/embodied_bringup/test/test_interface_startup.py
src/manipulation_execution/manipulation_execution/geometry.py
src/manipulation_execution/manipulation_execution/providers.py
src/manipulation_execution/manipulation_execution/phases/execution.py
src/manipulation_execution/manipulation_execution/phases/flow.py
src/manipulation_execution/manipulation_execution/phases/planning.py
src/manipulation_execution/manipulation_execution/phases/preparation.py
src/manipulation_execution/manipulation_execution/pick_executor_helpers.py
src/manipulation_execution/manipulation_execution/pick_executor_models.py
src/manipulation_execution/manipulation_execution/pick_executor_node.py
src/manipulation_execution/manipulation_execution/placement_executor_node.py
src/manipulation_execution/package.xml
src/manipulation_execution/test/test_geometry.py
src/manipulation_execution/test/test_pick_executor_orientation_guard.py
src/manipulation_execution/test/test_pick_kinematics_boundary.py
src/manipulation_execution/test/test_providers.py
src/skill_catalog/skill_catalog/compiler.py
src/skill_catalog/skill_catalog/validator.py
src/skill_catalog/test/test_compiler.py
src/skill_catalog/test/test_migrated_profiles.py
src/skill_catalog/test/test_navigation_catalog.py
src/skill_library/README.md
src/skill_library/package.xml
src/skill_library/skill_library/skill_executor_node.py
src/skill_library/test/test_capability_admission.py
src/skill_library/test/test_skill_executor_node.py
src/task_dispatch/package.xml
src/task_dispatch/task_dispatch/task_executor_node.py
src/task_dispatch/test/test_task_executor_bindings.py
```

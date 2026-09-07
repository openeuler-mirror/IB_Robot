---
name: ibrobot-control
description: "Use when a user asks Hermes or an Agent to discover, validate, execute, cancel, stop, or query IB-Robot capabilities and visual games through the `robot-skill` CLI and ROS Capability Gateway. Covers 'run a robot skill', 'execute robot action', 'cancel motion', 'stop robot', 'play a visual game', 'query a game result', '执行机器人动作', '取消动作', '停止机器人', '玩视觉游戏', '查询游戏结果', 'nod', 'wave', 'celebrate', 'look around', or interact with existing high-level robot skills. Requires the exact motion plan to be presented and flushed before physical motion; the user aborts a wrong plan with 别动 during execution. Never bypass the Gateway or call raw ros2 / MoveIt / controller commands directly."
---

# IB-Robot Control

## Overview

Use only `robot-skill` and the ROS Capability Gateway. Discover before acting, present the exact plan before execution,
and report only what the CLI proves.

## Skill Composition

Treat Skills as typed operators, not as a list of user phrases:

- `resolve_object_pose` provides a map pose `[x, y, yaw_degrees]`; it does not move or manipulate.
- `nav_abs_coordinate` consumes that pose as literal `x`, `y`, `yaw`; it does not resolve names.
- `nav_straight` consumes direction and distance; `nav_turn` consumes direction and angle.
- `pick_object` consumes an object name and performs visual grasping at the current base pose.
- `place_in_container` consumes a held object and container name and performs release/verification at the current base pose.
- `recover_safe_pose` returns the arm to home and is the final step after successful placement.

Use this generic routing:

- Need an object's semantic position: `resolve_object_pose(target_name, stand_off_distance_m=0.0)`.
- Need the base to approach an object for grasping: `resolve_object_pose(..., stand_off_distance_m=0.30)` ->
  `nav_abs_coordinate` -> `pick_object`.
- Need to approach a container for placement: `resolve_object_pose(..., stand_off_distance_m=0.30)` ->
  `nav_abs_coordinate` -> `place_in_container`.
- Need to transport an object: resolve the source and destination poses before motion -> navigate to source -> pick ->
  navigate to destination -> place -> recover safe pose.

Semantic queries are read-only single-skill calls performed before the motion plan. Their successful JSON pose results
must be parsed into finite numeric literals before creating one typed AgentPlan. The current workflow has no
`$previous.x`, `output_of`, or other runtime reference syntax; never put a placeholder or query step into the motion
plan. Compose new tasks by matching each Skill's input, output, precondition, and postcondition; do not search source or
invent a special-case recipe while handling a task.

When launched by `hermes-robot`, the `robot-skill` executable on `PATH` is already bound to the preflighted robot config
and ROS domain. Invoke that exact executable directly. Never source `.shrc_local` or another setup script, inspect or
modify ROS/Python environment variables, search for robot configs or repositories, load `ibrobot-env`, use an absolute
`robot-skill` path, or add `--config-name`/`--config-path`. On any nonzero exit, report the exact CLI error and stop; a
failed command never proves that a status check completed.

## Composite Workflow Entry

For every normal natural-language motion request, use the deterministic composite entry:

```text
robot-skill run-workflow [--request-id REQUEST_ID] --text TEXT --workflow-json JSON
```

Hermes produces the complete typed workflow once. `run-workflow` then performs discovery, catalog
visibility checks, planning, validation, presentation, confirmation, execution, and terminal-result
collection internally in one controlled call. Do not call the lifecycle commands separately for the
same normal request. They remain available for diagnostics and protocol tests.

For a single-step request, use one flat typed step, for example:

```bash
robot-skill run-workflow --text "打开夹爪" \
  --workflow-json '[{"schema_version":1,"skill_name":"open_gripper_skill"}]'
```

## Diagnostic Plan Workflow

When explicitly testing the lifecycle commands directly, resolve required semantic values before freezing the motion
plan and never put placeholders into `workflow-json`.

1. Query the Gateway: `robot-skill status`.
2. Discover capabilities: `robot-skill list-skills`.
3. Construct a fresh request ID directly in the conversation, such as `agent-request-YYYYMMDD-NNN`, then generate a
   typed plan: `robot-skill plan-workflow --request-id REQUEST_ID --text TEXT --workflow-json JSON`.
4. Read every selected contract with `robot-skill describe SKILL`, then run
   `robot-skill validate-plan --plan-token TOKEN`.
5. Show the exact ordered steps, parameters, plan digest, registry identity, and a fresh task ID, then flush the
   presentation to the user.
6. Immediately bind that exact tuple once with
   `robot-skill confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID --timeout-sec SEC`.
7. Execute only the returned confirmation token with
   `robot-skill execute-plan --plan-token TOKEN --confirmation-token CONFIRMATION_TOKEN --task-id ID --timeout-sec SEC
   --plan-id PLAN_ID --plan-digest DIGEST --registry-epoch EPOCH --registry-generation GENERATION
   --registry-digest REGISTRY_DIGEST --expected-step-count COUNT`.

By default, omit `--timeout-sec` from both commands so both use the current Gateway task budget. Do not derive a plan
budget from `default_skill_timeout_sec`. Only when the user explicitly requests a smaller budget, append the same
`--timeout-sec SEC` value to both commands. Never change the budget after confirmation.

`workflow-json` is an array of flat `WorkflowStep` objects. Every object must include an explicit `schema_version`.
Use `schema_version: 1` for legacy non-navigation contracts and `schema_version: 2` for navigation contracts. Skill
arguments are top-level fields, for example
`[{"schema_version":1,"skill_name":"pick_object","target_name":"marker"}]`. Never use `skill`, a nested
`parameters` object, or a bare object. Never infer or rewrite `WorkflowStep.schema_version` from the skill domain. Invoke
`plan-workflow` once with all three required options; do not probe it with an incomplete command.

Construct request IDs and task IDs directly in the conversation and `robot-skill` arguments. Do not call Python,
`uuidgen`, `date`, a shell, or any other helper tool to generate them. A command approval, including session-wide
approval, authorizes only that command and is not motion authorization. The displayed plan/task tuple is bound internally
by `confirm-plan` immediately after the presentation flush.

Natural-language single-Skill and Workflow requests both use `run-workflow` through the composite entry. The internal `confirm-plan` call is
the Gateway's technical binding for the exact plan/task tuple, not a second user confirmation gate. For an explicitly
selected single skill, the direct `describe -> validate -> execute` path remains valid.

Stop on any failure, unavailable/not-ready Gateway, unauthorized motion, or rejected validation.
Do not invent parameters absent from `describe`.
For an ordered multi-Skill request, call `run-workflow` exactly once with the user's original
wording and typed steps. Read-only semantic queries needed to obtain literal coordinates happen before this call and
are not workflow steps. The returned single plan must contain all ordered motion `workflow_steps`. If planning omits,
reorders, or rejects a requested step, report that exact result and stop; do not retry alternate phrasings and do not
split the motion plan into separately confirmed plans.

## Catalog Reload

When the operator asks to activate edited robot Skill YAML without restarting the robot, run exactly one
`robot-skill reload-catalog --request-id REQUEST_ID --force`. This reloads only the Gateway's configured catalog source;
it does not accept another path and does not reload visual-game configuration or handlers. Report `old_generation`,
`generation`, `changed_skills`, and diagnostics. Then run
`robot-skill status` and require its registry identity to match the reload result before creating any new plan. A reload
timeout has an unknown activation result: stop and ask for a new user request before checking or retrying.

This is separate from Hermes `/reload-skills`, which only reloads Agent instruction files such as this `SKILL.md`.
Changing `nod_yes` implementation or manifest YAML requires `reload-catalog`, not `/reload-skills`.

## Cancellation

Stop the right goal with the right command. The two executors are different actions; the wrong command cannot
reach the active goal and will time out as `SKILL_CANCEL_TIMEOUT` while the robot keeps moving.

- When no process in this session owns the executing **Agent plan**, issue `robot-skill cancel-plan --task-id ID
  --plan-id PLAN_ID --plan-digest DIGEST --registry-epoch EPOCH --registry-generation GENERATION
  --registry-digest REGISTRY_DIGEST --expected-step-count COUNT`, then wait for the strictly validated terminal result.
  This is the only external command that cancels the `/embodied/execute_agent_plan` goal. **Never** use `cancel` for an
  Agent plan — it targets the wrong action.
- `robot-skill cancel --task-id ID` cancels **only** a legacy single-skill `execute` goal
  (`/embodied/execute_skill`). Do not use it for `execute-plan`.
- SIGINT/SIGTERM on the running `execute-plan` process requests root cancellation and waits for its terminal result.
  Outside that process, use `cancel-plan` with the same task ID.
- **Cancellation requested is not robot stopped.** `SKILL_CANCEL_TIMEOUT`, transport failure, or an unknown result means
  the stop state is unknown; dispatch no further motion.

## Interactive Closed Loop: 别动 / 继续

Within one Hermes session the catalog query, plan/confirm/execute, stop, and continue steps form one closed loop.
Drive them with the closed vocabulary only; free text outside the grammar never acts.

1. **Discover (read-only).** `robot-skill status` then `robot-skill list-skills`. Do not plan or move.
2. **Reject out-of-catalog.** Any requested `skill_name` not in the current `planner_visible_names` is refused with
   `SKILL_REFERENCE_MISSING` before planning. Do not substitute or invent a skill.
3. **Prepare + present + execute immediately (no confirmation gate).** `plan-workflow` once, show and flush the ordered steps,
   plan digest, registry identity, and a fresh task ID, then run `validate-plan` + `confirm-plan` + `execute-plan`
   back-to-back without waiting for a `确认` reply — execution starts right after presentation. The user catches a
   wrong workflow with `别动` (step 4) during execution instead of confirming beforehand. The `确认` grammar is not
   needed for this flow. The session binds the sole unexpired pending plan; never accept a pending ID from the user or model.
4. **别动 → definite terminal.** On a stop phrase (`别动` / `停` / `停止` / `停下` / `stop` / `halt`), latch stop across
   validation, internal confirmation, action-server waiting, goal submission, and goal acceptance. If this session owns
   a running `execute-plan` process, signal only that process and let it cancel and converge. Use external `cancel-plan`
   only when no local process owns that task. Never use both for one task. Suppress any new goal after a pre-submission
   stop. A proven fresh in-process task needs no CancelGoal; if the deterministic task may be an idempotent retry, the
   owner must cancel/converge the existing goal instead of synthesizing a local terminal. GoalStatus, success/error,
   plan ID/digest, registry identity, and completed-step
   bounds must agree. Only status 5 + `success=false` + `SKILL_CANCELLED` + exact identity proves cancellation.
   `SKILL_CANCEL_TIMEOUT`, identity mismatch, or an unknown result means the stop state is unknown — report uncertainty
   and send no further motion.
5. **继续 → fresh user request only.** A continue phrase is permitted only after a definite canceled terminal
   (`GoalStatus=5` plus `SKILL_CANCELLED`). In this baseline `--resume` is rejected because `completed_step_count` is
   telemetry, not server-owned continuation admission. Do not slice the old plan or create a new task ID to bypass that
   boundary. After definite cancellation, a user may submit an independent fresh workflow, which re-queries the fresh
   registry and uses new plan/task tokens. Success, failure, and unknown states do not authorize automatic continuation.

A reference implementation of this loop (catalog discovery, out-of-catalog rejection, in-session prepare/confirm,
stop-to-definite-terminal, and fresh-state continuation) lives in
`robot_skill_cli.interactive_control.InteractiveController` and is unit-tested without a ROS stack.

## Perception Reads

For requests that need a runtime perception value (e.g., "转向我" needs the
user's azimuth), read it via the `ibrobot-perceive` wrapper **before** calling
`plan-workflow`, then inject the returned literal into `workflow_json`.

```
ibrobot-perceive --source voice_direction --field azimuth_rad
ibrobot-perceive --source arm_joint_position --field position [--config-name NAME]
```

- `ibrobot-perceive` is the **only** allowed path to read ROS topics. Never call `ros2 topic echo`,
  `ros2 topic list`, `ros2 param get`, or any other `ros2` subcommand directly — not even as a
  suggestion, fallback, or "alternative" when `ibrobot-perceive` rejects a source.
- If `ibrobot-perceive` rejects a source (e.g., `cmd_vel`) because it is not in
  the allowlist, report "该 source 未授权读取" and stop. Do **not** suggest `ros2 topic echo` or
  any other `ros2` command as a workaround; the rejection is a security boundary, not a missing
  feature.
- The wrapper's source/field allowlist is hard-coded in source; you cannot widen it by editing config.
  `--source` is a semantic alias, not a ROS topic name. The actual topic for config-backed sources
  (`arm_joint_position`) is resolved from `robot_config.moveit.joint_state_topic` at runtime
  (so101 -> `/joint_states`, lekiwi_handeye -> `/arm_joint_state_broadcaster/joint_states`); pass
  `--config-name` to match the robot the pipeline is running.
- `ros2 topic echo --once` returns the *next* published message, a single point-in-time sample, not a
  persistent snapshot. For volatile event sources such as `voice_direction` (published only on
  voice activity) the value may be absent within the timeout or already stale when consumed.
- The wrapper prints the value on stdout (e.g., `0.5236`) and errors on stderr. On any error,
  timeout, or missing field, report "无法感知" to the user and stop; do not fabricate a value.
- For requests asking for current motor or joint angles, run
  `ibrobot-perceive --source arm_joint_position --field position`. Return the raw position array in radians;
  do not invent joint names or reorder values because this minimal interface does not return the companion `name` field.
- The returned literal becomes a frozen plan parameter. It may be stale by execution time
  (open-loop, no correction); execution result is authoritative.

Example flow for "转向我" (requires a robot with a mobile base, e.g. lekiwi):

1. `ibrobot-perceive --source voice_direction --field azimuth_rad` -> `0.5236`
2. Convert azimuth_rad (radians, REP-103: 0=front, +π/2=left, -π/2=right) to
   direction and degree: positive => left, negative => right;
   degree = abs(azimuth_rad) * 180 / pi  (e.g. 0.5236 rad => 30.0 degrees).
3. `robot-skill run-workflow --text "转向我" --workflow-json '[{"schema_version":2,"skill_name":"nav_turn","direction":"left","degree":30.0}]'`

Do **not** map "转向我" to `rotate_gripper_cw`/`rotate_gripper_ccw` — those
rotate the wrist/gripper, not the robot base, and will not face the user.
For a single-arm robot without a mobile base (e.g. so101), "转向我" is not
available; report that the robot has no base rotation and ask the user for a
different request.

If the user says "left" or "right" with an explicit small nudge for the
end-effector (not base rotation), use `move_relative_ee` with **both**
`motion_direction` and `motion_distance` (the skill requires both):

```
[{"schema_version":1,"skill_name":"move_relative_ee","motion_direction":"left","motion_distance":0.05}]
```

## Visual Games

Visual games are non-motion capabilities with a separate asynchronous control surface:

1. Discover enabled games with `robot-skill list-games`.
2. Read the selected contract with `robot-skill describe-game GAME`.
3. Create a fresh caller-owned request ID, then run
   `robot-skill start-game GAME --request-id ID`.
4. Poll `robot-skill game-result --request-id ID` about once per second. Stop after the described
   `timeout_sec`; make one final query and report timeout or uncertainty if no terminal result is available.
5. Report the terminal structured result to the caller. Do not invoke an Agent TTS tool: when configured, the runtime
   announcer sends terminal text through `VisualGameEvent`, `/voice_tts/synthesize`, and the existing local
   `/voice_tts/play` service.

The CLI only starts and queries games. It must not play audio, wait indefinitely inside `start-game`, or retry a failed
game with a new request ID. If `start-game` loses its service response, querying or repeating the exact same game and
request ID within the advertised result-retention window is allowed: the Gateway treats that as idempotent recovery and
does not start a second request while the retained record exists. After retention expires the ID is no longer reserved,
so callers must not reuse it.

Game discovery exposes every enabled game. Visual games are started only through this Agent control surface; ASR and
task entry do not trigger them. `PERCEPTION_UNAVAILABLE` means the configured perception request topic has
no live subscriber; do not launch or restart infrastructure. `GAME_CAPACITY_EXHAUSTED` means the retained-result ledger
is full; records inside the advertised retention window are never evicted early, so wait for expiry or report the
capacity failure instead of retrying with a new request ID.

## Hard Boundaries

- The Agent **must not launch or restart the pipeline**.
- The Agent **must not enable motion authorization**; only the operator may set `authorize_motion`.
- The Agent **must not modify ROS parameters**.
- The Agent **must not source environment scripts, select a ROS domain, or discover another repository/config**.
- The Agent **must not call Python, `uuidgen`, `date`, a shell, or another helper tool to generate request/task IDs**.
- The Agent **must not call primitive, MoveIt, controller, or raw ros2 motion commands**.
- The Agent must not copy `docs/ib_robot_social_skill.md` as a control Skill.
- The Agent **must not automatically retry after failure, timeout, or unknown result** with a new task/request ID.

## Quick Reference

| Situation | Response |
|---|---|
| Catalog-only request | Use catalog commands; no motion confirmation is needed. |
| Runtime unavailable/unauthorized | Report the CLI error; do not start or alter infrastructure. |
| Visual-game perception unavailable | Report `PERCEPTION_UNAVAILABLE`; do not launch or restart perception. |
| Visual-game result ledger full | Report `GAME_CAPACITY_EXHAUSTED`; retained terminal results are not evicted early. |
| No clearly visible person | Report `NO_PERSON`; do not announce or invent a game result. |
| Terminal result | Report only public status/error fields. |
| Stop state unknown | Report uncertainty and send no new motion. |

## Rationalization Check

| Thought | Reality |
|---|---|
| "raw ROS is faster." | It bypasses the required control surface. |
| "More parameters make it safer." | Invented frame, orientation, velocity, or range arguments violate `describe`. |
| "It probably stopped; use a new ID." | Unknown is not stopped, and a new ID is still an automatic retry. |

## Red Flags

Prior permission, demo pressure, assumed stop, bypass commands, or an invented schema all mean: stop and restart the
required workflow.

## Common Mistakes

| Mistake | Correction |
|---|---|
| Treating accepted cancellation as physical stop | Wait for a terminal result or task ledger terminal state. |
| Treating command approval as motion authorization | Command approval authorizes only that command; motion still requires operator `authorize_motion` and a flushed plan presentation. |
| Retrying after checking idle | Require a new user request and repeat the entire workflow; never retry automatically. |
| Continuing after an unknown stop | `继续` needs a definite canceled terminal (5 + `SKILL_CANCELLED`). Otherwise refuse continuation and send no motion. |
| Slicing completed steps on `继续` | Reject breakpoint resume until the Gateway provides server-owned continuation admission. |

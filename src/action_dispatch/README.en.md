# Action Dispatch

A pull-based action distribution layer between inference models and ros2_control.

## Overview

This package distributes action chunks from embodied AI models such as ACT and Diffusion Policy
to robot controllers. Temporal blending can soften changes between overlapping predictions,
but guarantees neither continuous physical motion nor a gap-free supply of inference results.

Two mutually exclusive executables are selected solely by
`control_modes.<mode>.inference.scheduler.enable`:

- Absent or `false`: `action_dispatcher_node` uses the named
  `executor.inference_pipeline` through `DispatchInfer /dispatch` and `/reset`.
- `true`: `scheduled_action_dispatcher_node` waits for Global readiness, then uses
  `OpenInferenceSession`, `ScheduledDispatchInfer` and `CloseInferenceSession`.
  Open establishes only a logical session, with no model or fallback binding. The dispatcher
  owns one product session and validates every result identity. Terminal failure or `UNKNOWN`
  clears queue/smoother storage, publishes safe-stop, then closes and enters `FAILED`.
  Each Dispatch carries a target, priority and fresh absolute deadline; priority-0 also carries
  a fallback chain. Only retryable recoverable `NOT_STARTED` results receive bounded retries
  with a new request UUID and the original deadline. Infeasible deadlines, full capacity and
  ingress rejection safe-stop/Close immediately. Global reserves priority-0 ingress capacity
  that lower-priority requests cannot exhaust. Stop/Restart waits for pending Open and closes
  using the actual generation. SIGINT/SIGTERM keeps the executor and ROS context alive until
  safe-stop/Close completes or times out. Returned tensor steps must match result `chunk_size`
  before alignment by actions consumed during inference. Temporary exhaustion holds the last
  action. Safe-stop joint snapshot freshness uses local monotonic receive time, not ROS/sim
  time or header stamps.

Both nodes use `/action_dispatcher` and expose start/stop/status/toggle-smoothing interfaces;
the launch graph never runs them together. Scheduled consumes Global's whole-graph action chunk.

## System Architecture

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IB Robot System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         ┌─────────────┐ │
│  │  Inference       │         │  Action          │         │  ros2_      │ │
│  │  Service         │         │  Dispatch        │         │  control    │ │
│  │                  │         │                  │         │             │ │
│  │ ┌──────────────┐ │         │ ┌──────────────┐ │         │ ┌─────────┐ │ │
│  │ │ Model        │ │         │ │ Action       │ │         │ │ Joint   │ │ │
│  │ │ (ACT/Diff)   │ │         │ │ Dispatcher   │ │         │ │ State   │ │ │
│  │ └──────────────┘ │         │ │   Node       │ │         │ │ Pub/Sub │ │ │
│  │                  │         │ └──────────────┘ │         │ └─────────┘ │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Temporal     │ │         │             │ │
│  │                  │         │ │ Smoother     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Topic        │───────────▶│ Controllers│ │
│  │                  │         │ │ Executor     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  └──────────────────┘         └──────────────────┘         └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Communication Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS2 Communication                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Inference Service │                              │ Action Dispatch  │     │
│  │                   │                              │                  │     │
│  │                   │    DispatchInfer Action      │                  │     │
│  │                   │◀─────────────────────────────│                  │     │
│  │                   │    (ibrobot_msgs/action)     │                  │     │
│  │                   │                              │                  │     │
│  │                   │    VariantsList (Result)     │                  │     │
│  │                   │─────────────────────────────▶│                  │     │
│  │                   │    (action chunk tensor)     │                  │     │
│  └──────────────────┘                              └────────┬─────────┘     │
│                                                             │                │
│                                                             │                │
│  ┌──────────────────┐                              ┌────────▼─────────┐     │
│  │ ros2_control     │                              │ TopicExecutor    │     │
│  │                  │◀─────────────────────────────│                  │     │
│  │ /joint_commands  │   Float64MultiArray /        │                  │     │
│  │ /arm_commands    │   JointTrajectory            │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Sensor Layer     │                              │ Action Dispatch  │     │
│  │                  │                              │                  │     │
│  │ /joint_states    │─────────────────────────────▶│ (subscription)   │     │
│  │ (JointState)     │   optional                   │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Internal Data Flow

`continuous` consumes one available action per control tick when lifecycle gates allow,
without waiting for per-step execution feedback. It does not mean nonstop inference.
Replenishment requires `remaining <= watermark_threshold`, no in-flight inference, and
lifecycle permission (legacy is running with no policy reset; the scheduled session is ACTIVE).
`full_chunk` skips the new chunk prefix corresponding to actions consumed during the request,
then selects the entire remainder. It neither waits for the old chunk to be exhausted nor
limits the new chunk to its overlapping region.

```text
Control tick -> lifecycle gate -> check remaining/watermark -> record plan length, request asynchronously
                              -> ActivePlan consumes one (or hold/empty) -> executor

Result -> validate identity/decode -> count consumption during request -> FullChunkPlanner selects interval
       -> ActivePlan accepts atomically: none replaces; temporal_ensemble blends overlap and appends new tail
```

`TopicExecutor` routes `Float64MultiArray` or `JointTrajectory` according to the contract;
benchmark commits consumption through feedback instead. The communication diagram above
shows the legacy topic path; scheduled endpoints are listed under
[Topics and Services](#topics-and-services). Storage need not pass through a smoother;
the boundaries are detailed below.

## Strategy Layers And State Ownership

action_dispatch responsibilities are split into six boundaries. The
mutually exclusive legacy (`action_dispatcher_node`) and scheduled
(`scheduled_action_dispatcher_node`) product paths share the same boundary
contracts:

| Layer | Module | Owns | Never owns |
|-------|--------|------|-----------|
| Session lifecycle | per-node lifecycle state machines | session open/close, retry, result dedup, safe-stop, fail-and-close; gates per-tick dispatch | chunk algorithms, blending weights |
| Per-tick scheduler | `schedulers/` (registry + `continuous` / `wait_for_feedback`) | per-tick inference-request and action-submission decisions (`should_replenish_plan` is the shared watermark rule) | lifecycle state, chunk contents |
| Chunk planning | `chunk_planning.py` (`FullChunkPlanner` / `AutoHorizonPlanner`) | describes original-chunk interval `[start, stop)` and optional replenishment watermark (AutoHorizon truncates at the result-level `execution_horizon` and drops the watermark to 0) | mutating the accepted plan, publishing, requesting inference |
| Active plan | `active_plan.py` (`ActivePlan`) | atomically owns accepted storage, source, position, watermark and revision; direct consumption or reservation/commit; hold/empty selection | ROS I/O, episode/session transitions |
| Action blending | `temporal_smoother.py` | prepares overlapping weighted actions/counts before committing references; existing coefficient semantics | inference timing, session state |
| Executor | `executors/` (registry + `topic` / `benchmark`) | the final output channel | any scheduling decision |

All three paths use planning and owner acceptance. Benchmark consumption remains
completion-driven: reserve/submit does not consume; matching episode-approved
feedback commits once. It does not use continuous direct consumption.

| Product | Executor / scheduler | Storage and blending | Capacity / consumption |
|---------|----------------------|----------------------|------------------------|
| Legacy continuous | `topic` / `continuous` | queue with `none`; existing smoother manager with `temporal_ensemble` or disabled passthrough | queue keeps newest actions within capacity; direct logical consumption |
| Legacy benchmark | `benchmark` / `wait_for_feedback` | same queue/manager choices | same queue clipping; revision-bound reservation and feedback commit |
| Scheduled | `topic` / `continuous` only | queue with `none`; separate smoother with `temporal_ensemble` | queue overflow rejects, safe-stops and closes; direct logical consumption |

The `auto_horizon` chunking strategy is available only with the `topic` executor;
`benchmark` + `auto_horizon` is rejected by the SSOT (benchmark episodes execute
full chunks).

Smoother storage does not inherit deque capacity limits. Legacy toggles retain an
existing manager and its plan; a node without a manager cannot enable one via the
toggle. Scheduled toggles retain the inactive queue and its metadata, but discard
the smoother plan when disabling. No cross-store action migration is performed.
Legacy continuous stop/start retains actions, source, position and watermark while
invalidating old in-flight requests. Reset and benchmark cleanup discard the plan;
scheduled stop/safe-stop/close/restart clear both stores, including inactive metadata.

### State ownership (existing per-path differences, preserved by the refactor)

| State / semantics | Owner |
|-------------------|-------|
| `policy_reset_in_progress` inference gate, request/generation accounting | legacy node |
| queue clipping, accepted source interval, revision and watermark | active-plan owner (path-selected overflow policy) |
| capacity overflow as failure (`ValueError` -> safe-stop + session close), `(session_id, generation, request_id)` result dedup | scheduled path |
| session state machine (WAITING_READY/.../FAILED) and retries | scheduled path (not moved into `DispatchScheduler`) |
| watermark replenishment rule | shared (`schedulers.continuous.should_replenish_plan`; the configured watermark by default, or the plan-level threshold carried by an auto_horizon plan) |

### Extension points

- **AutoHorizon (supported)**: attention estimation and model capability checks live in
  `inference_service` (native Torch PI0.5 + `predict_action_chunk`); results reach the
  dispatcher via `execution_horizon`. `executed_during_inference` is the original-chunk start
  (auto_horizon collapses it to `[H, H)` when the whole prefix expired); `execution_horizon`
  is the exclusive stop, not a count after skipping. `replenishment_watermark=None` uses the
  configured default; zero means replenish when empty. Both stores apply the interval once.
  Runtime options participate in the profile-compatibility identity (see the inference_service
  README). The benchmark combination is rejected by the SSOT.
- **RTC (planned extension, strategy name not yet open)**: `PlanSource` records request and applicable generation/session IDs.
  Direct single-source `PlanSnapshot.next_position` includes skipped/clipped prefixes.
  Revision only invalidates local reservations. Ensemble has no exact single-source coordinate;
  latest source is diagnostic. Local acceptance/topic progress is neither physical completion nor remote cache acknowledgment.
  Cross-request cache identity, routing/failure reconciliation, coordinate transforms/relative-action re-anchoring
  and wire forwarding remain gaps.

## Installation

After setting up the environment, build from the workspace root:

```bash
source .shrc_local
./scripts/build.sh --packages-up-to action_dispatch
source .shrc_local
```

## Configuration and Usage

### Launch Node

Use robot_config launch for a complete robot. The standalone legacy debug entrypoint below
still needs valid robot YAML, an inference service and controllers. Additional scheduled
configuration is described in the [scheduler control plane](../robot_config/README.md#推理调度控制面).

```bash
ros2 run action_dispatch action_dispatcher_node --ros-args -p robot_config_path:=/path/to/robot.yaml
```

### Strategy configuration and combination validation

The SSOT for strategy names and legal combinations lives in
`robot_config.dispatch_strategies`; launch builders (launch-time validation)
and both dispatcher nodes (init-time defensive validation) share one
resolver, with entrypoint-specific capability checks:

In robot YAML, `executor`, `dispatch` and `inference` are siblings under `robot.control_modes.<mode>`.
See the [robot_config dispatch strategy SSOT](../robot_config/README.md#动作分发策略-ssot) for the schema.

```yaml
control_modes:
  model_inference:
    executor:
      type: topic                 # topic | benchmark
    dispatch:
      scheduler: continuous       # continuous | wait_for_feedback
      chunking: full_chunk        # full_chunk (default) | auto_horizon
      blending: none              # none | temporal_ensemble
```

- Legacy legal pairings: `topic`+`continuous`, `benchmark`+`wait_for_feedback`.
  Scheduled accepts only `topic`+`continuous`; explicit unsupported selections
  are rejected, not discarded. The historical executor `action` alias is mapped
  to `topic` only at the robot_config launch boundary, not direct node startup.
- `dispatch.chunking: auto_horizon` requires the `topic` executor; the
  `benchmark`+`auto_horizon` combination is rejected by the shared SSOT
  (benchmark episodes execute full chunks), both at launch time and at node
  init.
- **Breaking interface change:** `temporal_smoothing_enabled` is removed, with
  no alias or compatibility path. Old robot YAML and direct ROS overrides
  (constructor, CLI or parameter file) are rejected even when consistent.
  Migrate false to `dispatch.blending: none`, true to `temporal_ensemble`;
  direct ROS callers use `blending_strategy` instead.
- Omitted `dispatch.chunking`/`dispatch.blending` resolve to `full_chunk`/`none`.
  Both legacy and scheduled standalone nodes default to `blending_strategy: none`.
- Missing, null or empty-string strategy names use defaults (`topic`,
  `continuous`, `full_chunk`, and `none`). False, zero, lists, mappings,
  unknown names and case variants are rejected.
- Node parameters are `executor_type`, `scheduler_mode`, `chunking_strategy`,
  and `blending_strategy`. ROS parameter typing
  still applies; YAML null defaults refer to configuration resolution, not a
  promise that a ROS CLI null override is accepted.
- Deliberate rejection changes: malformed/empty/non-finite raw chunks are not
  accepted. Legacy continuous retains its old plan and logs rejection; benchmark
  aborts as inference-failed before successful startup; scheduled safe-stops and
  closes. A valid nonempty chunk with an empty selected interval clears executable
  storage. Rejected replacements cannot change the accepted watermark.
- `~/toggle_smoothing` stays `std_srvs/srv/Empty` on both nodes. Validation occurs
  before mutation; rejection is logged (Empty has no error field). Both expose
  `~/start_evaluate`, `~/stop_evaluate`, `~/get_status` as `std_srvs/srv/Trigger`.
  Legacy has `~/reset` (`Empty`); scheduled has `~/restart_session` (`Trigger`),
  not the legacy reset service. Service calls can change robot state.

### AutoHorizon integration boundary (`chunking: auto_horizon`)

AutoHorizon attention analysis belongs in `inference_service`, where the policy can access
action self-attention. The inference result carries an explicit `execution_horizon`; the
dispatcher-side `AutoHorizonPlanner` consumes that field: it truncates the executable plan
and lowers that plan's replenishment threshold to 0 (re-request inference once the prefix
is consumed instead of watermark prefetching). The dispatcher validates and executes the
prefix while retaining queue alignment, smoothing, and safe-stop responsibilities; it never
re-runs the model or infers attention from action values.

- With `dispatch.chunking: full_chunk` (default) the result-level
  `execution_horizon` is ignored and full-chunk + watermark behavior is kept;
  a missing or zero value also falls back to the full chunk under auto_horizon.
- The `benchmark` + `auto_horizon` combination is rejected by the shared SSOT
  (see the strategy matrix above): benchmark episodes must execute complete
  chunks, otherwise the evaluated action sequences and inference cadence would
  be truncated. There is no silent "accept but ignore the field" exception.
- The plan-level replenishment threshold is fixed at 0 as a direct mapping of the
  paper's synchronous evaluation semantics (`sample -> execute H steps -> sample`):
  the next inference is requested only after the prefix has been fully executed, so
  the robot holds for one full inference latency between prefixes (visible as a
  sustained `queue_size 0` in the mock e2e). The smaller H is, the larger the idle
  share. The current implementation intentionally does not prefetch in order to stay
  faithful to the paper; just-in-time prefetching (threshold
  `max(0, min(H-1, prefetch_ticks))`, reusing the existing `[H, H)` expiry
  normalization) is a recorded follow-up direction and is not implemented yet.
- A fully expired prefix that can appear after watermark prefetch (actions
  consumed during inference >= horizon, S >= H) is not an invalid chunk: the
  planner normalizes it to the legal empty interval `[H, H)` carrying
  watermark 0. The owner atomically accepts the empty plan (clearing the old
  one) and the scheduler re-requests inference on the next tick; the scheduled
  path does not safe-stop and the legacy path does not reject the result.
- Plan-level watermark cleanup semantics differ per product: the legacy
  `~/reset` and benchmark episode cleanup (prepare/end) clear the plan and its
  plan-level threshold, returning to the configured watermark; legacy
  continuous stop/start (pause/resume) keeps the active plan and its
  plan-level threshold and only invalidates in-flight requests; scheduled
  stop/safe-stop/close/restart clear both stores and return to the configured
  watermark.

Enabling the native Torch PI0.5 experiment requires **both configuration parts to be
active** — the dispatcher side selects the strategy, the serving side turns on
collection and estimation:

```yaml
# (1) dispatcher side (robot YAML control_modes.<mode>.dispatch):
#     consume the result-level execution_horizon and truncate the executable plan
dispatch:
  scheduler: continuous
  chunking: auto_horizon        # topic executor only; the benchmark combination is rejected by the SSOT
  blending: none

# (2) serving side (the inference pipeline's runtime_options):
#     enable action-expert attention collection and horizon estimation
runtime_options:
  auto_horizon_enabled: true
  auto_horizon_sampling_step: 3
  auto_horizon_hold_threshold: 0.3
  auto_horizon_entropy_quantile: 0.9
  auto_horizon_run_length: 1
```

With (1) but not (2) the result carries no effective horizon (equivalent to
`full_chunk` behavior); with (2) but not (1) the `full_chunk` strategy ignores the
field. Both together form the end-to-end loop (option semantics and defaults see the
[inference_service README](../inference_service/README.md)).

The experiment forces the action expert to eager attention and samples weights at the
configured denoising step. Missing action-expert `self_attn` modules (for example after a
lerobot upgrade moved module paths) or a `sampling_step` beyond `num_inference_steps` fail
closed at session load; transient inference-time anomalies (a non-finite weight, for
instance) still fall back to `execution_horizon=0`, i.e. the full chunk. Option ranges, load timing, the profile
identity coupling and the distributed cloud-side configuration location are documented
in the [inference_service README](../inference_service/README.md).

### Parameters

These are standalone node defaults. robot_config launch overrides runtime parameters from
same-named `executor` fields and strategy fields from `dispatch`. Legacy endpoints do not
apply to scheduled; see the [scheduler control plane](../robot_config/README.md#推理调度控制面).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `executor_type` | string | `topic` | Output channel; pairing constraints above |
| `scheduler_mode` | string | `continuous` | Per-tick scheduling, not the inference product-entrypoint switch |
| `chunking_strategy` | string | `''` | Empty resolves to `full_chunk` |
| `blending_strategy` | string | `none` | `none` or `temporal_ensemble` |
| `queue_size` | int | 100 | Maximum queue length; does not limit smoother storage |
| `watermark_threshold` | int | 20 | Replenish when remaining is at or below this value; not a fixed execution interval |
| `control_frequency` | double | 100.0 | Control tick frequency (Hz), not inference frequency |
| `robot_config_path` | string | `''` | Robot YAML path containing the contract |
| `joint_state_topic` | string | `/joint_states` | Joint state topic |
| `navigation_mode` | bool | false | Wait for an external trigger at startup |
| `temporal_ensemble_coeff` | double | 0.01 | Exponential blending coefficient |
| `chunk_size` | int | 100 | Smoother weight-table size, not the actual model output length |
| `smoothing_device` | string | `''` | Empty uses the input tensor device, or CPU for NumPy input |
| `inference_action_server` | string | `/inference/policy/dispatch` | Legacy only; launch overrides from the named pipeline |
| `inference_reset_service` | string | `/inference/policy/reset` | Legacy only; called best-effort during reset |
| `policy_reset_timeout_sec` | double | 2.0 | Legacy only; maximum wait for policy reset |

### Inference Replenishment and Blending Examples

These are partial settings to merge into an existing robot YAML, not complete launch
configurations. `executor` and `dispatch` are siblings under
`robot.control_modes.model_inference`. Keep `inference.enabled`, valid
`inference.pipelines`, controllers and the contract configured.
`executor.inference_pipeline: policy` must reference a declared pipeline; the scheduled
entrypoint also needs its session/scheduler configuration and is not enabled by these snippets alone.

**Replenish after exhaustion, without blending:**

```yaml
robot:
  control_modes:
    model_inference:
      executor:
        type: topic
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 0
      dispatch:
        scheduler: continuous
        chunking: full_chunk
        blending: none
```

With watermark 0, the next chunk is requested only after the old plan is exhausted.
While awaiting the result there are no new planned actions; the last action, if available,
continues to be published (hold last). This is neither a physical stop nor safe-stop,
and does not guarantee uninterrupted motion. Hold does not consume planned steps and
therefore does not increase the new chunk's skipped prefix.

**Replenish early and blend the overlap:**

```yaml
robot:
  control_modes:
    model_inference:
      executor:
        type: topic
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 80
        chunk_size: 100
        temporal_ensemble_coeff: 0.01
      dispatch:
        scheduler: continuous
        chunking: full_chunk
        blending: temporal_ensemble
```

Assume A and B each actually return 100 steps, with no pause, toggle or failure in between.
Indices below are zero-based with exclusive upper bounds:

```text
A returns: 100 steps
  -> Consume 20 steps: A[20:100], 80 remaining, trigger request B
  -> Consume 30 steps during asynchronous inference B: A[50:100], 50 remaining
  -> B returns 100 steps: skip B[0:30], retain B[30:100], 70 steps

Alignment at acceptance:   Overlap: 50 steps                New tail: 20 steps
Old plan                   A[50:100]                       (none)
New plan                   B[30:80]                        B[80:100]
Output                     blend(A[50:100], B[30:80])     + B[80:100] = 70 steps
```

The skipped prefix is the **30 steps** consumed since B's observation/request baseline,
not the cumulative 50 steps consumed from A. The implementation aligns using the change
in plan length between request start and result arrival, not elapsed wall time, and adds
no compensation for sensor sample age before the request. Consumption is local logical
progress, not confirmation of physical completion. Since 70 is still below watermark 80,
C may be requested on the next tick once the request has finished and other gates allow.
Watermark 80 does not mean inference every fixed 20 steps. Early replenishment also cannot
guarantee that inference is fast enough to prevent exhaustion.

Watermark and blending are independent choices:

- With watermark 0 and omitted `dispatch.blending`, the current resolver, legacy/scheduled
  launch paths and standalone nodes all default to `none`. The Python launch example later
  in this README explicitly enables ensemble; it is not the default.
- With watermark 0 and explicit `temporal_ensemble`, the selection is not rewritten to
  `none`. Normal replenish-after-exhaustion flow has no old/new plan overlap, but still uses
  a smoother. Compute/storage costs, capacity, source coordinates and toggle behavior differ;
  it is not fully equivalent to `none`. See the storage and state-ownership discussion above.
- With watermark 80 and `none`, early requests and prefix skipping still apply, but B's
  70 retained steps directly replace A's remaining 50 steps without blending or waiting
  behind the old plan.

### Launch File Example

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='action_dispatch',
            executable='action_dispatcher_node',
            name='action_dispatcher',
            parameters=[{
                'queue_size': 100,
                'watermark_threshold': 80,
                'control_frequency': 100.0,
                'blending_strategy': 'temporal_ensemble',
                'temporal_ensemble_coeff': 0.01,
                'chunk_size': 100,
                'robot_config_path': '/path/to/robot.yaml',
            }]
        )
    ])
```

## Cross-Frame Temporal Smoothing

### Alignment and Overlap

Use the [asynchronous replenishment example](#inference-replenishment-and-blending-examples)
above as the timing reference. One inference returns one chunk containing N actions, not
N chunks. The planner selects `[skip:N]`; the owner applies that interval once. The smoother
receives already-aligned actions and does not skip the prefix again.

Overlap length is `min(old remaining length, new selected length)`. Only the new tail beyond
the overlap is appended; any old tail beyond the new selection is discarded. The resulting
length therefore equals the new selected length. The first aligned position in the example
is `A[50]` with `B[30]`, not equal indices within the two chunks.

### Smoothing Formula

```python
blended[i] = (old[i] * cumsum[count[i]-1] + new[i] * weight[count[i]]) / cumsum[count[i]]
```

Where:
- `old[i]`: The i-th action in the old action plan
- `new[i]`: The i-th action in the aligned new selection
- `count[i]`: Existing prediction contributions at that position, initially 1
- `weight[k]`: Weight for k-th contribution = exp(-coeff * k)
- `cumsum[k]`: Cumulative weight sum

For the first example position with `count=1`:
`(A[50] * 1 + B[30] * exp(-0.01)) / (1 + exp(-0.01))`, approximately
`0.5025 * A[50] + 0.4975 * B[30]`. New tail counts start at 1. Once a count reaches
`chunk_size`, that position freezes and accepts no further weighted contributions.
`chunk_size` sizes the weight table; it is not the model output length or queue capacity.

### Smoothing Coefficient

| Coefficient Value | Effect |
|-------------------|--------|
| `0.0` | Uniform weighting, no preference for old/new |
| `Positive` | More weight to older actions (stable, conservative) |
| `Negative` | More weight to newer actions (responsive, may cause jitter) |

The current default coefficient is `0.01`; blending weights do not determine inference timing.

### Runtime Toggle

`blending_strategy` accepts startup CLI/YAML overrides but rejects direct runtime parameter writes,
including valid values and no-op writes. Both nodes use `~/toggle_smoothing` (Empty): all parameter
veto callbacks must pass before plan state changes. A successful toggle updates both the authoritative
strategy and public parameter, preserving the selection through parameter export/restart.
Rejection is logged; Empty has no error field. Legacy startup with `none` has no manager and cannot
enable one through toggle; an existing manager retains its plan. Scheduled retains the inactive queue
and discards the smoother plan when disabling, without migrating actions across stores.
See [state ownership](#strategy-layers-and-state-ownership) for pause and cleanup differences.
The following calls can change robot state:

```bash
# Toggle smoothing on/off
ros2 service call /action_dispatcher/toggle_smoothing std_srvs/srv/Empty

# Legacy only: reset state; scheduled uses restart_session (Trigger)
ros2 service call /action_dispatcher/reset std_srvs/srv/Empty "{}"
```

## Navigation Mode

When `navigation_mode=true`, the system starts in a stopped state and waits for an external trigger to begin execution. This mode is used when nav2 reaches the destination, then triggers the ACT model to execute grasping tasks.

### Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Navigation Mode Workflow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. System Startup                                                           │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  At startup: _is_running = False                        │
│     │ [NAV] Mode  │  System ready, waiting for trigger                       │
│     └─────────────┘                                                          │
│                                                                              │
│  2. Nav2 Navigation                                                          │
│     ┌─────────────┐                                                          │
│     │   Nav2      │  Navigate to target position                             │
│     │ Navigating  │  Dispatcher does not execute actions                     │
│     └─────────────┘                                                          │
│                                                                              │
│  3. Arrival at Destination                                                   │
│     ┌─────────────┐                                                          │
│     │  Nav2 Done  │  Call /action_dispatcher/start_evaluate                  │
│     └─────────────┘                                                          │
│                                                                              │
│  4. ACT Execution                                                            │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  _is_running = True                                      │
│     │ Executing   │  Trigger inference, execute ACT action sequence          │
│     └─────────────┘                                                          │
│                                                                              │
│  5. Task Complete                                                            │
│     ┌─────────────┐                                                          │
│     │    Call     │  Call /action_dispatcher/stop_evaluate                   │
│     │stop_evaluate│  Stop execution, set base velocity to zero               │
│     └─────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Usage

```bash
# Launch system (when navigation_mode=true)
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_realsense_navigation control_mode:=navi

# After Nav2 reaches destination, start execution
ros2 service call /action_dispatcher/start_evaluate std_srvs/srv/Trigger

# After task completion, stop execution
ros2 service call /action_dispatcher/stop_evaluate std_srvs/srv/Trigger

# Query current status
ros2 service call /action_dispatcher/get_status std_srvs/srv/Trigger
```

### Configuration Example

Merge these partial settings under the `robot` block in robot YAML:

```yaml
control_modes:
  navi:
    executor:
      navigation_mode: true    # Enable navigation mode
      watermark_threshold: 20
      control_frequency: 30.0
```

## Topics and Services

### Communication with Inference Service

| Direction | Topic/Action | Message Type | Description |
|-----------|--------------|--------------|-------------|
| Legacy request | `/inference/policy/dispatch` | `ibrobot_msgs/action/DispatchInfer` | Request the selected `executor.inference_pipeline` when scheduler is absent/false |
| Scheduled session | `/inference/session/open`, `/inference/session/close` | `OpenInferenceSession`, `CloseInferenceSession` | Global endpoints only; Open does not route models |
| Scheduled request | `/inference/dispatch` | `ScheduledDispatchInfer` | Carries session/generation/request/target/priority/deadline; priority-0 adds fallback chain |
| Response | `result.action_chunk` | `ibrobot_msgs/msg/VariantsList` | Receive action chunk (Tensor) |

### Published Topics

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `~/queue_size` | `std_msgs/Int32` | Current queue length |
| `~/smoothing_enabled` | `std_msgs/Bool` | Whether smoothing is enabled |

### Subscribed Topics

| Topic | Message Type | Description |
|-------|--------------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Joint states (optional) |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `~/reset` | `std_srvs/Empty` | Legacy only: reset queue/state and best-effort call `inference_reset_service`; scheduled uses `~/restart_session` |
| `~/toggle_smoothing` | `std_srvs/Empty` | Toggle smoothing on/off |
| `~/start_evaluate` | `std_srvs/Trigger` | Resume dispatcher execution |
| `~/stop_evaluate` | `std_srvs/Trigger` | Pause dispatcher execution; also stop the base when `navigation_mode=true` |
| `~/get_status` | `std_srvs/Trigger` | Get running status; scheduled returns session state-machine status |
| `~/restart_session` | `std_srvs/Trigger` | Scheduled only: safe-stop, Close, clear local state, then Open with a new UUID |

On the scheduled path, `~/start_evaluate`, `~/stop_evaluate`, and `~/restart_session` return
`success=false` with `message="lifecycle operation in progress"` on lifecycle contention.
This means the requested operation was **not executed**, including Stop: a busy response
is not a successful stop. Callers must check the response and decide whether to retry based on the state.

### Communication with ros2_control

| Direction | Topic | Message Type | Description |
|-----------|-------|--------------|-------------|
| Publish | `/joint_commands` | `std_msgs/Float64MultiArray` | Joint position commands |
| Publish | `/arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | Trajectory commands |

## API Usage

### Using TemporalSmoother Directly

```python
from action_dispatch import TemporalSmoother, TemporalSmootherConfig

# Create configuration
config = TemporalSmootherConfig(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# Create smoother
smoother = TemporalSmoother(config)

# First inference
actions1 = model.inference(obs)  # shape: (100, action_dim)
smoother.update(actions1, actions_executed_during_inference=0)

# Consume 20 steps before requesting B
for _ in range(20):
    robot.execute(smoother.get_next_action())

# Sample/compute B at the request baseline; simulate delaying asynchronous delivery
actions2 = model.inference(obs)
for _ in range(30):
    action = smoother.get_next_action()
    robot.execute(action)

# Deliver B after 30 steps were consumed during the request; 70 remain after update
smoother.update(actions2, actions_executed_during_inference=30)

# Continue executing smoothed actions
while smoother.plan_length > 0:
    action = smoother.get_next_action()
    robot.execute(action)
```

### Using TemporalSmootherManager

```python
from action_dispatch import TemporalSmootherManager

manager = TemporalSmootherManager(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# Runtime toggle
manager.set_enabled(False)  # Disable smoothing
manager.set_enabled(True)   # Enable smoothing

# Check status
print(f"Plan length: {manager.plan_length}")
print(f"Smoothing enabled: {manager.is_enabled}")
```

## Dependencies

- ROS2 Humble
- Python 3.10+
- PyTorch
- NumPy
- ibrobot_msgs
- tensormsg

## License

Apache License 2.0

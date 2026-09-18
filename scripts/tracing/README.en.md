# IB-Robot Tracing (ros2_tracing + LTTng)

Low-overhead tracing for the inference chain, following ROS 2 best practices.

## Setup

```bash
./scripts/setup.sh
```

`./scripts/setup.sh` now installs the tracing stack as part of the normal
workspace setup (LTTng, ros2_tracing, babeltrace2, and `tracetools-analysis`).

If the workspace is already set up and you only want to add tracing tools later,
you can still run:

```bash
bash scripts/tracing/setup_tracing.sh
```

## Usage

### Option A: Integrated launch (recommended)

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm use_sim:=true \
    control_mode:=model_inference \
    enable_tracing:=true
```

This uses `robot_config.launch_builders.tracing` to create an LTTng session
during launch, capturing ROS 2 UST events (`ros2:*`) and Python business
tracepoints (`ib_trace.*`).

If the default session name `ib_robot_trace` is already in use, launch will
auto-suffix a timestamp instead of overwriting the existing trace. Explicit
`trace_session_name:=...` values are never overwritten.

`IB_TRACE_ENABLED=1` is passed through scoped launch/process environments to this
launch's business child processes, including nodes deferred until controller
readiness. It replaces the early assignment to the parent's `os.environ` with a
launch scope that restores the environment on exit, avoiding leakage into later
launches in the same process. Control modes, execution-node parameters and model loading are unchanged.

The original LTTng startup failure contract remains: session inspection, create,
enable-event or start failures abort launch. Cleanup only owns sessions successfully
created by this invocation; there is no new strict option. Topology sidecar export
is outside startup, so an independent metadata failure cannot prevent robot startup
or cancel a successful recording.

### Option B: Separate trace session

```bash
# Terminal 1
bash scripts/tracing/start_trace.sh

# Terminal 2
source .shrc_local
export IB_TRACE_ENABLED=1
ros2 launch robot_config robot.launch.py ...

# When done
bash scripts/tracing/stop_trace.sh
```

In manual mode, `start_trace.sh` also auto-suffixes the default session name on
collision. If you explicitly pass a colliding session name, it fails fast rather
than clobbering the old trace.
Set `IB_TRACE_ENABLED=1` in the terminal that launches the robot nodes; the script
in Terminal 1 cannot enable Python instrumentation in Terminal 2.

### Optional Declared Topology

Ordinary analysis requires no robot YAML, model files or topology sidecar. For an
optional configuration declaration, run separately:

```bash
source .shrc_local
ros2 run robot_config ibrobot-trace-topology \
    src/robot_config/config/robots/so101_single_arm.yaml \
    --control-mode model_inference > /tmp/ibrobot-topology.json
```

The CLI uses the existing `load_robot_section` to parse YAML and sibling
`base_config` inheritance, not the full business loader or model availability
checks. It loads no models and leaves business-startup validation unchanged.
Metadata is marked `provenance=declared`, `runtime_verified=false`; it describes
configuration, not actual started nodes, and excludes launch overrides (except
`--control-mode`), nav stages and runtime-derived settings. Launch does not create
this file automatically, and the CLI does not manage LTTng sessions.

### Analyze

New traces use `IBTRACE1` structured events as the primary metric source. The analyzer keeps the
legacy `[event] key=value` parser for persisted traces. During dual emission, structured metrics
take precedence and observation records are deduplicated.

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-trace summary ~/.ros/tracing/ib_robot_trace
```

`scripts/tracing/analyze_trace.py` remains as a compatibility entrypoint for the historical command:

```bash
python3 scripts/tracing/analyze_trace.py --trace-dir ~/.ros/tracing/ib_robot_trace
```

See [`src/ibrobot_tracing/README.md`](../../src/ibrobot_tracing/README.md) for the complete CLI,
query projections, Critical Path, Span Profile, and trace comparison behavior.

## How It Works

1. **`robot_config.launch_builders.tracing` owns LTTng session management** —
   `robot.launch.py` stays an orchestrator that wires launch arguments into the
   builder, which enables `ros2:*` UST events and the Python tracing domain
   `ib_trace.*` on startup and stops/destroys the session on shutdown.

2. **`ibrobot_tracing` owns structured instrumentation and offline analysis** —
   nodes use `TraceEmitter`, `trace_context()`, `span()`, `event()`, and Flow APIs to
   create `IBTRACE1` records. The `ib_trace.*` logger is the LTTng Python-domain
   transport contract, while `component_id` is the analysis identity.

3. **`ibrobot-trace` is the primary offline analyzer** — `AnalysisService` reads CTF
   or structured logs and provides the same Query/Projection behavior to CLI and Web API.

4. **`ibrobot_tracing_web` and the Vue workspace run on Ubuntu only** — openEuler keeps
   trace capture and Core analysis, then transfers trace files to Ubuntu for browser analysis.

5. **Standard ROS 2/LTTng tools remain available** — `ros2 trace`, `lttng`,
   `babeltrace2`, and Trace Compass still support session management and low-level inspection.

## Business Tracepoints

`BUILTIN_TRACEPOINT_REGISTRY` is the code SSOT for Events and Spans; `topology.py` is the
code SSOT for Flow edges. Every built-in identity has `origin=built-in`.

### Events

| Component ID | Name | Logger | Purpose |
|---|---|---|---|
| `action_dispatcher.request` | `dispatch_request` | `ib_trace.dispatch` | Start a policy request |
| `action_dispatcher.decode` | `dispatch_result` | `ib_trace.dispatch` | Record result/decode status |
| `action_dispatcher.queue` | `queue_refill` | `ib_trace.dispatch` | Record queue state after refill |
| `action_dispatcher.execute` | `action_execute` | `ib_trace.dispatch` | Record sampled action publication |
| `action_dispatcher.execute` | `first_action_execute` | `ib_trace.dispatch` | Record first non-hold action publication |
| `action_dispatcher.execute` | `action_topic_publish` | `ib_trace.execute` | Record control-topic publication |
| `action_dispatcher.execute` | `safe_stop_topic_publish` | `ib_trace.execute` | Record scheduled safe-stop publication |
| `policy` | `dispatch_result` | `ib_trace.policy` | Record policy result status |
| `policy.observation` | `obs_receive` | `ib_trace.policy` | Record observation ingress latency |
| `policy.observation` | `obs_sample` | `ib_trace.policy` | Record sample freshness |
| `policy.observation` | `obs_frame` | `ib_trace.policy` | Record frame completeness |

### Spans

| Component ID | Name | Logger | Purpose |
|---|---|---|---|
| `action_dispatcher.decode` | `dispatch_decode` | `ib_trace.dispatch` | Decode the action chunk |
| `action_dispatcher.queue` | `queue_refill` | `ib_trace.dispatch` | Update the action queue |
| `action_dispatcher.execute` | `action_execute` | `ib_trace.dispatch` | Publish a sampled action |
| `action_dispatcher.execute` | `first_action_execute` | `ib_trace.dispatch` | Publish the first non-hold action |
| `policy` | `policy_pipeline` | `ib_trace.policy` | Process the complete policy request |
| `policy` | `policy_total` | `ib_trace.policy` | Preprocess, infer, and postprocess |
| `policy` | `cloud_roundtrip` | - | Registered distributed identity; currently derived from edge events |
| `policy.observation` | `observation_sampling` | `ib_trace.policy` | Assemble the observation frame |
| `policy.preprocess` | `preprocess` | `ib_trace.policy` | Produce model inputs |
| `policy.inference` | `model_call` | `ib_trace.policy` | Execute local model inference |
| `cloud_inference` | `model_call` | `ib_trace.policy` | Execute cloud model inference |
| `global_scheduler` | `scheduler_dispatch` | `ib_trace.scheduler` | Admit, route, and call downstream for a scheduled request |
| `policy.postprocess` | `postprocess` | `ib_trace.policy` | Convert model outputs |
| `policy.postprocess` | `action_chunk_publish` | `ib_trace.policy` | Package and publish the action chunk |
| `policy.postprocess` | `result_encoding` | `ib_trace.policy` | Encode a scheduled action result |

An Event and Span can share a name while expressing different semantics. For example, the
`queue_refill` Span measures elapsed time, while the Event records the resulting queue state.

### Flows

| Edge ID | Source Component | Target Component | Purpose |
|---|---|---|---|
| `dispatch_to_observation` | `action_dispatcher.request` | `policy.observation` | DispatchInfer request transfer |
| `observation_to_preprocess` | `policy.observation` | `policy.preprocess` | Observation enters preprocessing |
| `preprocess_to_inference` | `policy.preprocess` | `policy.inference` / `cloud_inference` | Model input enters inference |
| `inference_to_postprocess` | `policy.inference` / `cloud_inference` | `policy.postprocess` | Model output enters postprocessing |
| `result_to_decode` | `policy.postprocess` | `action_dispatcher.decode` | Result returns to dispatch |
| `decode_to_queue` | `action_dispatcher.decode` | `action_dispatcher.queue` | Decoded chunk enters the queue |
| `queue_to_execute` | `action_dispatcher.queue` | `action_dispatcher.execute` | Queued action enters execution |
| `scheduled_dispatch_to_scheduler` | `action_dispatcher.request` | `global_scheduler` | Scheduled request enters Global |
| `scheduler_to_pipeline_dispatch` | `global_scheduler` | `policy.observation` | Global calls the selected pipeline |
| `pipeline_result_to_scheduler` | `policy.postprocess` | `global_scheduler` | Pipeline result returns to Global |
| `scheduler_result_to_dispatcher` | `global_scheduler` | `action_dispatcher.decode` | Global result returns to dispatch |

The original seven legacy edges apply when the scheduler is absent or disabled. The four scheduler edges combine
with the internal processor and queue edges when scheduling is enabled.
Scheduler transport flow IDs include the ROS action goal UUID, so idempotent replays remain separate flow pairs.
Multi-pipeline scheduler topology uses one logical `Policy Pipelines` path; `pipeline_ids`, `pipeline_nodes`, and each
event's `pipeline_id` preserve the concrete primary or fallback identity without labeling fallback work as primary.

## Files

```
robot.launch.py          ← Launch orchestrator (wires enable_tracing:=true into the tracing builder)
src/robot_config/robot_config/launch_builders/tracing.py
                         ← Starts/stops the LTTng session
src/ibrobot_tracing/     ← Structured instrumentation, parsing, offline analysis, and CLI
tools/ibrobot_tracing_web/ ← opt-in FastAPI service
web/ibrobot_tracing_ui/  ← Vue workspace; Ubuntu users build the untracked production dist locally
scripts/tracing/
├── setup_tracing.sh     ← Retrofit tracing tools into an existing workspace
├── start_trace.sh       ← Manual session start
├── stop_trace.sh        ← Manual session stop
├── analyze_trace.py     ← Historical analysis-command compatibility entrypoint
├── README.md            ← Chinese documentation
└── README.en.md         ← English documentation
```

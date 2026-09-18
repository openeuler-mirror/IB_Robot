# inference_service

`inference_service` is IB-Robot's unified inference runtime. It selects one model bundle and one named deployment,
then runs the model through Torch, Ascend, Hisilicon, RKNN, or HMM. Policy models additionally use a stable pipeline
ID and support monolithic or distributed edge/cloud execution.

There is no compatibility layer for the removed runtime architecture. A backend is not selected with a launch
`device` argument. The runtime does not load per-backend sidecar manifests, scan directories for conventional
artifact names, or use environment variables to override artifacts.

Local requests use `ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)` and return `ModelResult`.
Independent producer frames use `ModelRuntimeHandle.submit_frame(ModelRequest, ExecutionContext)` and return a
future. They share request registration, cancellation, failure reporting and control draining with ordinary execution.
Frame admission is bounded separately so a running producer can overlap an admitted Dispatch; diagnostics count both.
`ModelRuntimeFactory` is a registered construction surface, not the request boundary. A policy reaches the handle
through its `InferencePipeline` facade. A typed plugin constructs a `ModelSession`, places it in a
`RuntimeAssembly`, and transfers the assembly to a handle. The handle owns public lifecycle, admission, deadlines,
cancellation, health, and close draining; the session resource owns and releases vendor model objects, device
leases, buffers, and workers.

## Core Concepts

### Policy Bundle

Every deployable policy directory contains the LeRobot semantic files and exactly one
`inference_manifest.json`:

```text
policy_bundle/
├── config.json
├── model.safetensors                         # when required by a Torch deployment
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── tokenizer/                                # when required by PI0.5 or SmolVLA
├── artifacts/
│   ├── ascend/<deployment>/...
│   ├── hisilicon/<deployment>/...
│   ├── rknn/<deployment>/...
│   └── hmm/<deployment>/...
└── inference_manifest.json
```

LeRobot owns `config.json`, processor JSON, processor state, tokenizer assets, and native weights. IB-Robot reads
those files without adding fields, removing fields, rewriting devices, or materializing a temporary policy
directory. All deployment metadata belongs in `inference_manifest.json`.

### Deployment

One manifest may declare multiple named deployments for the same policy, such as `cpu`, `cuda`, `rk3588`,
`ascend_310p3`, or `lq50`. A pipeline selects a deployment name, not a backend name.

A deployment uses an explicit v3 runtime profile for the backend, target, and instance fields:

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 1,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "torch",
    "target": {"runtime": "torch"},
    "profile": {"device": "cpu"}
  }
}
```

A compiled deployment declares its target, artifacts, execution order, and complete runtime ABI bindings:

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 3,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "rknn",
    "target": {
      "soc": "rk3588",
      "runtime": "rknn-lite2"
    },
    "profile": {
      "target_name": "rk3588",
      "core_mask": 7,
      "device_id": 0
    }
  },
  "artifacts": {
    "policy": {
      "path": "artifacts/rknn/rk3588/generations/<uuid>/policy.rknn",
      "format": "rknn"
    }
  },
  "execution": ["policy"],
  "bindings": {
    "policy": {
      "inputs": [
        {
          "semantic": "observation.state",
          "runtime_name": "observation.state",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 6]
        },
        {
          "semantic": "observation.images.top",
          "runtime_name": "observation.images.top",
          "index": 1,
          "dtype": "float32",
          "shape": [1, 480, 640, 3],
          "layout": "NHWC"
        }
      ],
      "outputs": [
        {
          "semantic": "action",
          "runtime_name": "action",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 100, 6]
        }
      ]
    }
  }
}
```

Every role in `execution` must have an artifact and a non-empty binding group. Image bindings must explicitly
declare `NCHW` or `NHWC`; non-image tensors are never transposed from rank alone. Multi-module deployments use
matching `internal.*` semantics or `device_links` that declare producer, consumer, device-pointer ownership, and
inference lifetime.

### Pipeline

The pipeline ID is the stable model-instance and ROS-routing identity. It must match
`^[a-z][a-z0-9_]{0,62}$`. Each pipeline independently owns:

- its policy bundle and named deployment
- its LeRobot preprocessor and postprocessor
- its policy codec and binding execution plan
- its `ModelRuntimeHandle`, admission state, and lifecycle
- its action, reset, health, action-output, and distributed transport endpoints

`InferencePipeline` is the policy facade that adapts the existing policy contract to `ModelRuntimeHandle` while
adding LeRobot processors, policy codecs, and action adaptation. The handle owns lifecycle, admission, deadlines,
cancellation, health, and diagnostics, and loads then releases the components transferred in its `RuntimeAssembly`.
Compiled models run through sequential or staged executors sharing `ComponentModelExecutor` helpers; iterative families
use `IterativeStage` to invoke roles. One `ExecutionContext` carries the request ID, deadline, and cancellation token
through every stage and `ModelSession` resource, while the session owns vendor resources shared across those roles.
Both strategies use `ModelSession.execute_role()` for role validation. Ascend selects isolated datasets internally,
while reusing semantic binding and diagnostic capture. The PI0.5 policy adapter validates the request-scoped
`vlm` / `action_expert` prefix-cache ABI, excluding state, iterative and reverse-link dependencies from the producer.
The manifest describes the model ABI; loaded runtime capabilities establish asynchronous execution and isolation.
The loaded backend must expose isolated async role execution and priority streams.
The executor owns triggers, immutable snapshots, prompt/freshness checks and generation fencing.
Private binding actions/status serve policy/VLA only. Deadline admission support comes from the actual executor,
not hardware, and is not a finish guarantee. Public priority uses one continuous upper bound.
Private Close retries use `(session_id, operation_id)` and still pass lifecycle/drain admission; successful drain
must advance the pipeline generation, except for explicit generation-zero cleanup requiring no work.
Perception, Echo, TTS, and other typed plugins use the direct
`ModelServicePlugin -> RuntimeAssembly/ModelRuntimeHandle -> ModelSession resource` composition path.

Default endpoints:

| Interface | Default |
| --- | --- |
| local node | `inference_<pipeline_id>` |
| cloud node | `inference_<pipeline_id>_cloud` |
| action server | `/inference/<pipeline_id>/dispatch` |
| reset service | `/inference/<pipeline_id>/reset` |
| health topic | `/inference/<pipeline_id>/health` |
| action output | `/actions/<pipeline_id>` |
| distributed request | `/inference/<pipeline_id>/request` |
| distributed result | `/inference/<pipeline_id>/result` |
| distributed heartbeat | `/inference/<pipeline_id>/heartbeat` |

### Scheduled Control Plane

`control_modes.<mode>.inference.scheduler.enable` is the only scheduler switch and defaults to `false`. When enabled,
the launch graph uses Global Open/Dispatch/Close actions and the scheduled action dispatcher. Only monolithic pipelines
are supported; they use the whole-graph sequential executor by default, while a supported functional manifest may opt
into independent staged execution.

When the switch is `false`, a complete scheduled configuration may remain in the SSOT as dormant configuration; the
launch graph, node parameters, endpoints, executor sizing, and backend execution still use the legacy path. When the
entire `scheduler` block is absent, scheduled fields remain unknown so legacy configurations keep strict typo checking.

A downstream Close with known `NOT_STARTED` preserves its outcome and recoverability. The session stays `CLOSING`
without quarantining the pipeline solely for that rejection. A retry uses new operation IDs and only visits bindings
that have not drained; Dispatch stays blocked. Mixed `UNKNOWN` results take precedence and remain quarantined;
executed failures still return unsuccessful `COMPLETED`. An identity-validated successful Close reclaims the drained
binding's reservations and operations. Operations with live waiters are reclaimed after detachment;
late results cannot restore released ownership.

Independent producers accept `scheduling.stages.<producer>.max_snapshot_age_ms` (positive integer, default `5000`).
Only the first manifest role may configure it; sequential and terminal stages reject the option. Snapshots must match
the generation and prompt and remain within the observation age limit. The node converts ROS capture age to a
monotonic anchor, including producer queueing and execution time. Dispatch refresh preserves the same capture time;
direct runtime calls without capture metadata age from initial admission. A refresh that is still stale fails instead
of repeatedly recomputing the same expired observation.

Four lifecycle fixes apply only when scheduling is enabled: repeated manager Close retries retained `close_pending` pipelines;
reset skips explicitly stateless, non-resettable non-executor components; stateful policy late/completed failures or
failures after execution started make subsequent `infer()` calls report not-ready until a successful reset; and reset support discovered
during LeRobot Torch session loading enables actual `policy.reset()`. With the scheduler disabled or absent,
the legacy topology, synchronous execution and lifecycle semantics are preserved. Tests cover launch parameters,
late results, reset and Close behavior.

Priority `0` is highest. Its resource ordering and fallback depend on the scheduler policy:

| `global_policy` | `priority_zero_deadline_admission.enable` | Behavior |
| --- | --- | --- |
| `fifo` (default) | `false` (default) | Dispatch the target directly without a resource queue; reject nonempty fallback chains |
| `fifo` | `true` | Profile-based finish admission, per-resource FIFO, and fallback; sequential pipelines only |
| `edf` | `false` | Per-resource deadline ordering, FIFO ties, and fallback; sequential pipelines only, no profile-based finish prediction |

EDF never preempts active work and does not guarantee completion before the deadline. Deadline-driven priority-0
(EDF or FIFO profile admission) is a sequential-only contract: independent-stage pipelines split one request across
overlapping workers, so per-request deadline ordering can be neither evaluated nor honored. robot_config rejects
such targets and fallback entries at configuration load, and Global skips independent candidates by serving-status
capability. Unsupported default FIFO/fallback combinations fail configuration loading when the scheduler is enabled.
Missing or invalid profiles affect
only FIFO with profile admission enabled, not readiness, EDF, or nonzero priorities. Only `NOT_STARTED` permits fallback;
`UNKNOWN` remains quarantined and must not be retried. Nonzero priorities dispatch only the target and are exempt from
the sequential-only constraint.

Offline measurements define a p99 admission SLA rather than an absolute completion guarantee. Action-generation
profiles must match the pipeline input-contract fingerprint and declare `prompt_bytes_max`; Global selects the smallest
profile bucket that covers the current prompt, adds the configured safety margins, and fails closed outside calibrated
coverage. Profile identity is independent of endpoint names and routing membership.

Serving readiness requires every required pipeline to report at least one generic priority level. If the configured
default priority is greater than zero, readiness additionally requires the configured default target pipeline to be
online and expose that priority. Other pipelines are not required to support multiple priorities. Backends without an
explicit generic-to-native mapping accept priority `0` only; Ascend currently maps generic priorities `[0, 7]` one to
one to ACL stream priorities.

## Robot Configuration

Inference is configured directly under `control_modes.<mode>.inference.pipelines`:

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/so101_act
          deployment: rk3588
          execution_mode: monolithic
          request_timeout: 5.0
          default_task: pick up the banana
          runtime_options: {}
```

A relative `model_path` is resolved only against the `WORKSPACE` environment variable. If `WORKSPACE` is unset,
configuration fails without falling back to the current directory, YAML directory, or source tree. Source the
project environment before project or ROS commands:

```bash
source .shrc_local
```

Multiple models are multiple pipelines, not a generic `concurrency` value:

```yaml
pipelines:
  action_policy:
    model_path: models/so101_act
    deployment: rk3588
    execution_mode: monolithic
  auxiliary_policy:
    model_path: models/auxiliary_smolvla
    deployment: cpu
    execution_mode: monolithic
```

YAML remains the default configuration source. For development, override one explicitly named pipeline:

```bash
ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/robot.yaml \
    control_mode:=model_inference \
    inference_pipeline:=policy \
    inference_execution_mode:=distributed
```

An empty `inference_execution_mode` preserves YAML configuration. A non-empty override requires
`inference_pipeline`, preventing accidental global changes in multi-pipeline configurations.

Each pipeline may override endpoints in a typed `transport` mapping. Node names, actions, services, and topics
must remain unique across pipelines. A monolithic pipeline cannot configure cloud-node, request, result, or
heartbeat overrides.

## Execution Modes

### Monolithic

`pipeline_policy_node` executes the complete path in one process:

```text
ROS observations
  -> contract adapter
  -> InferencePipeline policy facade
  -> ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)
  -> SequentialModelExecutor stages
       -> LeRobot preprocessor + policy codec
       -> ModelSession execute / role execution
       -> action decode + LeRobot postprocessor
  -> DispatchInfer result and action topic
```

Before processors run, the node checks every observation required by the policy `input_features`. A buffered
sample must exist at or before the requested timestamp and satisfy its `align.strategy` (`hold`, `asof`, or `drop`).
When the contract configures `max_age_ms > 0` for that observation, it is an additional maximum live sample age,
separate from `tol_ms` used by `asof` alignment. Live age uses the node's local receipt clock, which request
timestamps cannot rewind; request timestamps only select aligned historical samples. Missing,
future-dated, or stale samples return the recoverable `observation_not_ready` error instead of silently running the model with zero
padding. The pipeline reset service resets the policy and LeRobot preprocessor/postprocessor, then clears
observation buffers so the next inference waits for inputs from the new episode. Inference, reset, and distributed
cancellation use the pipeline `request_timeout` as a cooperative deadline: lock and admission waits exit on time, while
backend/processor hook overruns are detected when the hook returns. An uncertain reset or cancellation outcome fails
the edge closed to prevent cloud and edge episode state from diverging.

Normal robot startup creates pipelines from robot YAML through the `robot_config` launch builder. To evaluate one
pipeline directly:

```bash
source .shrc_local
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:="$WORKSPACE/src/robot_config/config/robots/so101_single_arm.yaml" \
    model_path:="$WORKSPACE/models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515" \
    deployment:=cpu \
    pipeline_id:=policy \
    action_server:=/inference/policy/dispatch \
    reset_service:=/inference/policy/reset
```

Trigger one request:

```bash
ros2 action send_goal /inference/policy/dispatch \
    ibrobot_msgs/action/DispatchInfer \
    "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'test-001'}"
```

### Distributed

For a distributed pipeline, the edge `pipeline_policy_node` retains observation sampling, robot-state unit
conversion, non-image tensor serialization, the action `TemporalSmoother`, and final robot-unit conversion. The
cloud `pure_inference_node` assembles the complete raw observation and runs the LeRobot preprocessor, selected
backend, and postprocessor in one process so processor state is not split across hosts. Before requests are
accepted, both sides must match:

- pipeline ID
- manifest schema version
- bundle digest
- deployment name
- selected deployment fingerprint
- policy input/output summary
- cloud backend `READY` state

Image observations may retain explicit `mode: dds` or use an H.264 RTP/UDP data plane. In RTP mode DDS carries
only descriptors, status/timestamp mappings, requests/results, and heartbeats; H.264 payloads never enter
`VariantsList`. Each camera needs a unique stream ID, SSRC, and even UDP port, with `port + 1` reserved during
collision validation. The cloud accepts requests only after every descriptor matches the protocol, session
generation, contract fingerprint, and deployment fingerprint and each stream has a keyframe and a fresh
RTP-to-capture timestamp mapping.

`encoder_backend` accepts `software`, `nvidia`, `ascend`, or `auto`; `nvidia` is currently encode-only and cannot
be selected as a decoder. The software backend uses PyAV 15 and probes its FFmpeg build for `libx264` and an H.264
decoder. The NVIDIA backend opens a real `h264_nvenc` session and uses ultra-low-latency, zero-delay, no-B-frame
H.264 with repeated SPS/PPS. RGB/BGR-to-NV12 conversion still occurs through FFmpeg and is not CUDA zero-copy.
The optional Ascend backend lazily discovers a
private FFmpeg `h264_ascend` installation through `IBROBOT_ASCEND_FFMPEG` or
`IBROBOT_ASCEND_FFMPEG_PREFIX`; the standard RPM entry point `/usr/bin/ffmpeg-ascend` is also detected. It neither
replaces system FFmpeg nor adds ACL/DVPP Python dependencies. Startup
logs (the "Video stream startup" lines) report configured and selected backends, endpoints, fingerprints,
lifecycle, and readiness.

RPM-installed ffmpeg-ascend runtimes (`/usr/bin/ffmpeg-ascend`, `/usr/local/bin/ffmpeg-ascend`, and the
versioned `/usr/local/ffmpeg-ascend-*/bin/ffmpeg` payload they dispatch to) start with an isolated
environment by default: the child process drops installation-path variables inherited from an unrelated
CANN toolkit (`ASCEND_TOOLKIT_HOME`, `ASCEND_HOME_PATH`, `ASCEND_AICPU_PATH`, `ASCEND_OPP_PATH`,
`ASCEND_NNRT_HOME`, `ASCEND_NNAE_HOME`, `TOOLCHAIN_HOME`) while preserving device runtime variables
(such as `ASCEND_RT_VISIBLE_DEVICES` and `ASCEND_DEVICE_ID`) so multi-NPU pinning keeps working.
Private builds are not isolated by default; `IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV=1` forces isolation for
any binary and `=0` opts out of the RPM default. Probing the RPM payload without isolation is known to
hang on the first frame.

Ascend DVPP VENC channels are a per-device hardware resource; `DeviceVideoStreamManager`
assigns dense 1..N IDs (≤128) to Ascend streams by sorted observation key. `device_id`
is currently fixed at 0 (single-NPU scope); multi-NPU support needs a resource allocation
contract at the pipeline/resource layer, not a code change in the manager.

`auto` probes `ascend`, then `nvidia`, then `software`. Ascend boards retain DVPP priority, NVIDIA hosts select
NVENC when a real session opens, and other Linux hosts fall back to software. Explicit backend failure never falls
back.

RTP/UDP provides no authentication, confidentiality, or integrity and is restricted to a trusted robot network.
An interrupted stream, descriptor mismatch, unavailable explicit backend, stale timestamp mapping, or excessive
camera skew fails closed without an RTP-to-DDS fallback. Rollback requires a matching `mode: dds` contract on both
hosts. rosbag/MCAP recording remains DDS-image based; RTP-aware recording and untrusted-network security are
separate follow-up work.

Cloud example:

```bash
source .shrc_local
ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda
```

To debug a distributed edge process directly:

```bash
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy \
    inference_execution_mode:=distributed
```

To start edge and cloud together on one host, replacing the old implicit local-cloud switch:

```bash
ros2 launch inference_service local_distributed_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy
```

The edge can be created from robot YAML with `execution_mode: distributed` or through the explicit launch
override above. A successful handshake binds a unique
session ID and generation. Heartbeat expiry, cloud restart, fingerprint change, or the runtime handle leaving `READY`
immediately revokes readiness, rejects new requests, and fails in-flight requests with a structured unavailable
error. Responses from old sessions are discarded, and recovery requires a new handshake.

A new handshake is not sufficient to recover a stateful runtime. When replacing an existing distributed session,
the cloud first stops request admission for that session and drains its `ModelRuntimeHandle` operations. A stateless
runtime may then create the new generation directly. A stateful runtime must first reset successfully through the
handle and return to `READY` before the cloud publishes a new generation. Reset failure remains fail-closed and
subsequent heartbeats cannot bypass this recovery barrier. If a stateful runtime declares `resettable: false`,
session rollover cannot recover through
handshaking; the cloud runtime must be restarted or rebuilt before it can serve requests again.

## Backends And Support Matrix

The only canonical backend names are:

| Backend | Responsibility |
| --- | --- |
| `torch` | native LeRobot on `cpu`, `cuda`, `mps`, or `npu` |
| `ascend` | Ascend ACL execution of OM artifacts |
| `hisilicon` | Hisilicon worker runtime, initially targeting SoC `sd3403` |
| `rknn` | RKNNLite execution of RKNN artifacts |
| `hmm` | Houmo TCIM execution of HMM multi-module artifacts |

Native Torch policies use the LeRobot factory by default. Repository-owned
models may register a stable `(model_type, backend, device)` provider; PI0.5
Ascend310P uses `(pi05, torch, npu)`. The generic runtime does not maintain
model-specific platform branches; each provider owns its configuration,
platform validation, and runtime preparation.

Compiled PI0.5 and SmolVLA loops are owned by shared executor stages, not model-session resources. They execute through
`InferencePipeline -> ModelRuntimeHandle -> SequentialModelExecutor -> InferenceStage -> ModelSession resource`.
The handle owns control-plane state, while each session owns only its vendor runtime and model resources. The
following matrix is normative and enforced at startup:

| Policy family | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | Runtime | Runtime | Runtime | Runtime | unsupported |
| Diffusion Policy | Runtime | unsupported | unsupported | unsupported | unsupported |
| PI0.5 | Runtime | Runtime | unsupported | unsupported | Runtime |
| SmolVLA | Runtime | unsupported | unsupported | Runtime | Runtime |

`Runtime` means execution through `ModelRequest` and `ExecutionContext`, with handle-owned lifecycle, admission,
health, cancellation, and recovery and session-owned vendor model resources. The registry-enforced perception
matrix is:

| Perception family | `torch` | `ascend` |
| --- | --- | --- |
| RAM++ | supported | supported |
| SAM2 | supported (automatic) | supported (prompt) |
| SigLIP2 | supported | supported |
| Grounding DINO | supported (combined) | supported (raw) |
| GraspGen | supported (CUDA) | supported |
| Dummy Echo | supported | unsupported |

The registry also requires matching `ConformanceEvidence` for every family/backend declaration; a family listed
without evidence still fails closed.

### PI0.5 Ascend Behavior

The optimized PI0.5 VLM combines all camera images into one temporary vision batch internally, then restores the
camera-major prefix before the handoff. This optimization does not change the external VLM ABI: runtime bindings,
per-camera observation semantics, raw image shapes, and ROS camera-topic contracts remain unchanged.

NPU export uses the accuracy-preserving `NPUGeglu` path for the Gemma text MLP by default. The explicit
`--fast-gelu-scope vision|vlm-text|ae|all` option limits approximate `NPUFastGelu` to one model region; the legacy
`--fast-gelu` option means global `all`. The approximate path may reduce action accuracy and must be validated
against an existing baseline.

A new Action Expert OM has a runtime output named `velocity` or `v_t`, while the Manifest still maps that tensor
to the policy `action` semantic. The policy runtime assembler reads strictly decreasing timesteps from the selected
deployment's `denoising_schedule` artifact, and the shared `IterativeStage` performs host-side Euler integration as
`x_next = x_t + (next_t - t) * velocity` before returning the final action. When export does not specify
`--schedule-file`, the exporter packages a uniform schedule derived from `config.num_inference_steps`; an explicit
file must be strict `pi05-denoising-schedule-v1` JSON.

`denoising_schedule` is a versioned, non-execution artifact. It is absent from `execution` and `bindings`, but its
artifact path and deployment revision are part of the deployment identity, so changing the schedule changes the selected deployment
fingerprint. Production runtime does not scan for a root `schedule.json` and does not accept schedule overrides.
`loss_compare` and the tuner inject a temporary schedule through an isolated diagnostic backend factory;
`curvature_log_path` only records diagnostics. The final schedule must be installed in the Manifest.

Compatibility is selected explicitly by the Action Expert runtime output. Existing legacy PI0.5 deployments with
an `action` output and no schedule artifact retain the old stepwise action-output behavior. A velocity deployment
without a schedule is rejected rather than given a guessed default. `hardware_mock` still validates only the raw
image/topic, joint, and action contracts and needs no PI0.5- or schedule-specific changes.

Native Torch Diffusion Policy samples observation history at the contract control rate according to the model's
`n_obs_steps`, and its nominal `predict_action_chunk()` length comes from `n_action_steps`. Missing startup history
is left-padded with each stream's first frame, while different sensor rates retain their configured `hold`, `asof`,
or `drop` alignment policy on a common time grid.

Optional SDKs are imported lazily. Importing the inference core does not require ACL, RKNNLite, TCIM, torch NPU,
or Hisilicon worker dependencies. A missing dependency fails only when its deployment is selected.

## Lifecycle, Health, And Capabilities

Public `ModelRuntimeHandle` states are `CREATED`, `LOADING`, `READY`, `RESET_REQUIRED`, `RESETTING`, `FAILED`,
`CLOSING`, and `CLOSED`. Only `READY` admits requests. The handle closes admission and drains active execution
before reset or close, records health and recovery requirements, and idempotently releases `RuntimeAssembly`
components in reverse ownership order.

`ModelSession` is a handle-owned resource, not the public lifecycle owner. It owns and releases contexts, vendor
model handles, device buffers, workers, tokenizers, and other model-specific assets, and implements
`execute(ModelRequest, ExecutionContext)` or role execution. The policy facade projects handle state into pipeline
diagnostics; distributed `HANDSHAKING` is a node-protocol state, not a model-resource lifecycle state.

Session and deployment capabilities let the handle determine:

- whether the backend is stateful, resettable, and thread-safe
- maximum in-flight requests per instance
- support for multiple instances
- shared resource-domain identity and limit
- attention and cancellation support

Defaults are conservative and serialized. A runtime may declare higher concurrency only after conformance tests
prove overlapping calls, output isolation, failure isolation, and deterministic cleanup. Different handles have
independent admission state, but a shared accelerator resource domain may still serialize them.

## Manifest Identity

Startup performs strict JSON/schema validation, deployment selection, UUID/revision and lightweight bundle-digest
validation, path-safety and regular-file checks, LeRobot metadata loading, and binding compatibility checks before
constructing the session resource, `RuntimeAssembly`, and `ModelRuntimeHandle`. Runtime does not read OM, RKNN,
HMM, or safetensors files to hash their contents.

`bundle.digest` is calculated as follows:

1. Normalize, deduplicate, and sort every `bundle.files` path.
2. Add the bundle UUID, revision, name, and a structure-format domain.
3. Serialize this small declaration as canonical UTF-8 JSON.
4. Calculate SHA-256 over the declaration bytes without reading the referenced files.

UUIDs, revisions, digests, and fingerprints provide version identity and distributed consistency, not tamper
protection. Production artifact updates must use the packager and publish a new revision; use signed read-only
images or verity when artifact authenticity is required.

The selected deployment fingerprint is SHA-256 over this canonical object:

```json
{
  "format": "ibrobot.deployment-structure-v3",
  "schema_version": 3,
  "bundle_digest": "...",
  "deployment_name": "rk3588",
  "deployment": {}
}
```

Paths cannot be absolute, use parent traversal, escape the bundle root through symlinks, or collide after
normalization.

### Identity Failures

Do not edit identities manually when startup reports:

- `Bundle digest mismatch`
- unsupported schema v1 (regeneration is required)
- missing or unexpected LeRobot semantic files
- an execution role missing an artifact or bindings
- runtime ABI incompatible with LeRobot feature shapes

Rerun the exporter or packaging workflow that owns the artifact. Exporters copy artifacts, read compiler/runtime
ABI metadata, generate bindings, update UUIDs/revisions and lightweight structural identities, and validate the
result through the production loader. Schema-v1 bundles and legacy artifacts are unsupported; regenerate a complete
schema-v3 bundle with the current exporter or packager.

## Exporter Entry Points

Generic compiled artifact packaging:

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment rk3588 \
    --backend rknn \
    --target-soc rk3588 \
    --target-runtime rknn-lite2 \
    --spec /path/to/compiler-package-spec.json
```

For PI0.5 and SmolVLA HMM packaging:

```bash
ros2 run model_utils package-hmm-deployment --help
```

ACT Ascend, ACT RKNN, Hisilicon, and policy-specific multi-module exporters live in `model_utils`. Every tool must
finish through the shared `inference_manifest` writer. Artifact paths, bindings, and digests are exporter-owned,
not hand-maintained configuration.

## Verification

When running from source, prefer source package paths and disable unrelated external pytest plugins:

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src/inference_manifest:src/inference_service \
pytest -q src/inference_service/tests
```

Check that removed identifiers have not re-entered active source, configuration, or tests:

```bash
source .shrc_local
python scripts/check_inference_legacy_identifiers.py
```

Run Ruff only on Python files changed by the current work. Always source `.shrc_local` before project or ROS
commands.

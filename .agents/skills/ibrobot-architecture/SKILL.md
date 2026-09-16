---
name: ibrobot-architecture
description: "Provides deep knowledge of IB-Robot's architecture. Use when user needs to 'understand architecture', 'explain design', 'check SSOT', 'modify robot_config', 'check contract', 'architecture', '架构', '设计说明', '配置加载', '数据流', '契约设计', 'unified inference runtime', '统一推理', 'model_service', '推理调度', '技能网关', 'capability gateway'. Triggers for 'how does it work?', '架构设计', '系统原理', or when modifying core robot parameters and single source of truth files."
---

# IB-Robot Architecture Skill

This skill provides comprehensive knowledge of IB-Robot's layered architecture, design principles, and core components. Content is aligned with the upstream master code (schema-v3 inference manifests, unified inference runtime, embodied skill stack).

**Reference Documentation**: https://deepwiki.com/wuxiaoqiang12/IB_Robot

## Core Design Principles

### 1. Single Source of Truth (robot YAML)

One YAML file per robot under `src/robot_config/config/robots/` is the **single authoritative source** for joints, controllers, peripherals, contracts, voice/perception/embodied subsystems, and launch behavior. Top-level key is `robot:`.

| Traditional Approach | IB-Robot Approach |
|---------------------|-------------------|
| Separate configs for ros2_control, cameras, ML contracts, voice, navigation | One YAML drives every subsystem via launch builders |
| Manual synchronization between systems | Fail-fast validation + config digest |
| Configuration drift over time | `base_config` overlay inheritance keeps variants thin |

Key mechanics:

- **Overlay inheritance**: `base_config: <sibling-name>` deep-merges a base robot YAML (same directory only, cycle-detected); lists append via the `__append__` key; provenance is reported in `_config_sources` (`robot_config/loader.py`).
- **Path resolution**: `config_path.py:resolve_robot_config_path()` precedence — explicit `config_path` arg → `config_name` → `ROBOT_CONFIG` env → `ROBOT_NAME` env → default `so101_single_arm`.
- **Runtime target**: `runtime_target.py` resolves `hardware | simulation | benchmark` (launch override → YAML `runtime.target` → legacy `use_sim`), with fail-fast consistency against `use_sim`. The simulation *platform* axis (`gazebo | mujoco | mock`) is separate, dispatched by `launch_builders/sim_backend/` adapters.
- **Digest**: `loader.robot_config_digest()` fingerprints the merged config for drift checks.

**Key Files**:
- `src/robot_config/config/robots/*.yaml` - Robot configurations (SSOT); `so101_single_arm.yaml` is the canonical arm example
- `src/robot_config/robot_config/loader.py` - Loading, deep merge, validation, digest (`load_robot_config`, `validate_config`)
- `src/robot_config/robot_config/config.py` - `RobotConfig` dataclass and `to_contract()`
- `src/robot_config/robot_config/config_path.py`, `runtime_target.py` - Path and runtime-target resolution

### 2. Contract-Driven Interface

A **Contract** is the typed observation/action interface between the robot and a policy, defined in `robot_config/contract_utils.py` as frozen dataclasses:

```python
@dataclass(frozen=True, slots=True)
class Contract:
    name: str
    version: int
    rate_hz: int
    max_duration_s: float
    observations: list[ObservationSpec]  # topic, type, selector, image, align, qos, transport
    actions: list[ActionSpec]            # publish_topic, from_tensor, publish_strategy, safety_behavior
    tasks: list[TaskSpec]
```

- Synthesis is **in-memory**: `RobotConfig.to_contract()` (typed path) and `loader.build_contract_from_robot_config_dict()` (dict path). There is no contract cache on disk.
- `decode_value()` / `encode_value()` bridge to `tensormsg.TensorMsgConverter` (registry-based ROS message ↔ tensor codec).
- `StreamBuffer` (same module) keeps capture-timestamp-ordered history per observation with `hold / asof / drop` alignment policies and live-age checks.

**Contract consumers** (identical processing for record / convert / infer):

1. `action_dispatch` - `action_dispatcher_node` / `scheduled_action_dispatcher_node` (action encoding)
2. `inference_service` - `pipeline_policy_node`, `pure_inference_node`, `recording_node`, plus the video-stream modules (`observation_sync`, `video_rtp`, `compute_video_streams`, `device_video_streams`)
3. `dataset_tools` - `episode_recorder`, `bag_to_lerobot`, `policy_eval`, `rerun_viewer`
4. `hardware_mock` - `contract_plan` (mock topic plan from contract)

### 3. Control Modes and Action Dispatch

Four control modes converge on the same `ros2_control` hardware interface; per-mode controller sets and execution settings live in the robot YAML under `control_modes:`:

| Mode | Typical controllers | Executor | Use Case |
|------|--------------------|----------|----------|
| `teleop` | `*_position_controller` | topic | Human teleoperation |
| `model_inference` | `*_position_controller` | topic | AI policy control |
| `moveit_planning` | `*_trajectory_controller` | action (via runtime `motion_server` / `task_dispatch`) | Motion planning, skills |
| `base_navigation` | `base_controller` / `base_velocity_controller` | topic (cmd_vel) | Mobile base (lekiwi), skill_catalog schema v2+ |

`action_dispatch` decouples "when to request/submit" from "where output goes":

- **Dispatchers**: `ActionDispatcherNode` (pull-based `DispatchInfer` action client) vs `ScheduledActionDispatcherNode` (session-based: `OpenInferenceSession` / `ScheduledDispatchInfer` / `CloseInferenceSession`, with `safe_stop` plan).
- **Schedulers** (registry): `continuous` (watermark refill) and `wait_for_feedback` (`StepBarrierScheduler`, single-in-flight, fail-closed).
- **Executors** (registry): `topic` (`Float64MultiArray` / `JointTrajectory` per contract action specs) and `benchmark` (`StepBenchmark` service).
- Pairing is guarded (`topic`+`continuous`, `benchmark`+`wait_for_feedback`); `TemporalSmoother` blends cross-frame action chunks for chunking policies (e.g. ACT).

### 4. Manifest-Driven Unified Inference Runtime

Model deployments are **bundles described by a schema-v3 `inference_manifest.json`** (package `inference_manifest`: strict loader/validator, `interface = policy | tensor_model`, semantic tensor bindings, per-deployment runtime profiles and artifact digests). Backend selection is manifest-driven, never hardcoded:

| Backend | Model types | Target runtime / SoC |
|---------|-------------|----------------------|
| `torch` | act/diffusion/pi05/smolvla policies; ram_plus/sam2/siglip2/grounding_dino/graspgen/zipvoice | cpu/cuda/mps/npu |
| `ascend` | act/pi05 policies; perception + voice models | ACL OM on ascend 310P/310B family |
| `hisilicon` | act policy | SD3403 worker |
| `rknn` | act/smolvla | RKNNLite on RK3588 |
| `hmm` | pi05/smolvla | TCIM on xh2/lq50/m50 |
| `onnx` | fullsubnet/silero_vad/speech_direction | onnxruntime |

Runtime layering (package `inference_service`):

- `model_sessions/*` - `ModelSession` ABC + per-backend sessions (`LeRobotTorchModelSession`, `AscendOmModelSession`, `RKNNModelSession`, `HMMModelSession`, `HisiliconModelSession`, `OnnxRuntimeModelSession`, `TorchModelSession`): native runtime state machine, semantic shape/dtype validation.
- `unified_runtime/` - `ModelRuntimeHandle` owns admission, deadlines, cancellation, recovery, lifecycle; `RuntimeAssembly` + three registries compose backend + session builder + assembler.
- Entry nodes: `pipeline_policy_node` (policy serving; successor of the removed `lerobot_policy_node`), `model_service_node` (generic typed-service host for perception/voice plugins), `global_inference_scheduler_node` (session lifecycle + admission: `GoalSlotPool`, deadline reservations, idempotency ledger), `pure_inference_node` (cloud endpoint for distributed mode).

### 5. Embodied Skill Stack

Agent-facing skills form a closed, safety-checked chain:

```
Agent/LLM → robot-skill CLI (robot_skill_cli)
  → Capability Gateway skill_executor_node (skill_library, /embodied/execute_skill)
    → preflight safety_guard_node (/embodied/validate_skill, read-only snapshot)
    → skill_catalog compiled manifests (exact snapshot: registry_epoch + generation + digest)
    → primitives → task_dispatch / runtime motion services / manipulation_execution / navigation
```

- `skill_library` owns gateway admission/idempotency and delegates to protected executors.
- `embodied_bringup` launches the minimum closure: `agent_plan_node` + `safety_guard_node` + `skill_executor_node` (+ optional perception, grasp stack, HRI, sound orientation).
- Manipulation skills use `manipulation_execution` (pick/place/imitate executors) over `manipulation_service` (GraspGen `PlanGrasp` / `VerifyGrasp`); navigation primitives delegate to `ExecuteNavigation` (`/navigation/execute`).

## Package Architecture

```
src/
├── robot_config/            # SSOT: robot YAML, contracts, launch orchestration
├── ibrobot_msgs/            # Interface definitions (actions / msgs / srvs)
├── tensormsg/               # ROS message <-> tensor codec registry (TensorMsgConverter)
├── inference_manifest/      # Schema-v3 model bundle manifest (loader, validator, writer)
├── inference_service/       # Unified inference runtime: pipeline/scheduler/model_service nodes, backends
├── perception_service/      # Perception model plugins (typed services) + VLM scene analysis node
├── observation_transport/   # H.264/RTP frame ingress-egress (FrameIngress, video codec registry)
├── model_utils/             # ONNX/OM/RKNN/HMM export and bundle packaging CLIs
├── action_dispatch/         # Pull-based action dispatch: schedulers, executors, smoothing
├── task_dispatch/           # Task plan execution (waypoints + gripper + waits)
├── skill_catalog/           # Skill manifest compiler + immutable catalog registry
├── skill_library/           # Capability gateway (skill_executor_node)
├── robot_skill_cli/         # Controlled CLI surface for LLM/Agent access
├── embodied_agent/          # Task entry, planning, visual game nodes
├── embodied_bringup/        # Embodied pipeline launch orchestration
├── embodied_common/         # Neutral shared helpers (contracts, base node)
├── safety_guard/            # Read-only skill/primitive validation preflight
├── manipulation_service/    # GraspGen grasp planning/verification services
├── manipulation_execution/  # Closed-loop pick/place/imitate executors
├── dataset_tools/           # Episode recording, bag_to_lerobot, policy_eval, rerun_viewer
├── benchmark/               # Benchmark runtime + LIBERO adapter (evaluation)
├── semantic_mapping/        # Persistent RGB-D 3D semantic mapping
├── object_tracker/          # Single-target RGB-D tracking + Nav2 following
├── robot_navigation/        # Nav2 client, navigation_command_server, chassis bridge
├── robot_runtime/           # Runtime contract: RuntimeStatus, capabilities, interface description, mock runtime
├── robot_teleop/            # Teleop bridges (50 Hz), glove/VR/mhandpro sources
├── voice_asr_service/       # sherpa-onnx ASR + speech direction nodes
├── voice_tts_service/       # Manifest-backed ZipVoice TTS typed service
├── robots/so101/            # SO-101 runtime suite: so101_sdk, so101_hardware, so101_description, so101_motion, so101_suite, so101_robot
├── robots/feetech/          # feetech_sdk (pinned Feetech servo SDK)
├── lekiwi_hardware/         # ros2_control hardware plugin (LeKiwi base + arm)
├── aero_hand_hardware/      # Aero Hand command/state bridge
├── hardware_mock/           # Contract-driven mock backend (no real hardware)
├── lekiwi_description/      # LeKiwi URDF/meshes
├── sim_models/              # Scene assets + scene compiler (Gazebo/MuJoCo)
├── robot_calibration/       # Sensor calibration capture/validate/activate workflows
├── attention_viz/           # ACT attention weight visualization
├── fast_lio/ fast_calib/ livox_ros_driver2/ omni_wheel_controller/  # (git submodules)
├── pymoveit2/ rosclaw/      # (git submodules, vendored)
└── workflows/               # CI gate definition (Jenkins), not a ROS package
```

For package responsibilities, README-as-contract rules, and the full Key Files Reference table, see `references/key-files.md`.

## Data Flow Overview

```
Policy (monolithic):  Camera/JointState → ROS Topic → tensormsg decode → StreamBuffer → PipelinePolicyNode
                      → ModelRuntimeHandle → ModelSession → VariantsList (/actions/<pipeline_id>)
                      → Action Dispatcher → TemporalSmoother → Scheduler → Executor → Controller → Hardware
Distributed (cloud-edge): Camera → FrameIngress → H.264 RTP → PureInferenceNode (cloud)
                      → DistributedResult → edge postprocess → VariantsList → (same as above)
Embodied skill:       Agent → robot-skill CLI → Gateway → safety preflight → catalog
                      → primitives → task/moveit/manipulation/navigation executors
```

For detailed code paths, inference execution modes (monolithic / distributed / scheduled), unified runtime layering, and temporal smoothing internals, see `references/data-flow.md`.

## Internal References

Read only the references needed for the current scenario:

| Purpose | Reference |
|---------|-----------|
| Observation/Action flows with key code paths, Inference Execution Modes (monolithic / distributed / scheduled), Unified Runtime layering + backend table, Model Service typed services, Temporal Smoothing | `references/data-flow.md` |
| Launch System (30 launch builders, sim backend adapters, all 25 launch arguments), Common Patterns (Launching, Adding New Robot, Debugging Contracts), Troubleshooting | `references/launch-and-troubleshooting.md` |
| Package Responsibilities by layer, README as Local Architecture Contract, Key Files Reference table | `references/key-files.md` |

Do not expose these references as separate skills.

## DeepWiki References

- [IB-Robot Overview](https://deepwiki.com/wuxiaoqiang12/IB_Robot/1-ib-robot-overview)
- [Core Concepts](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3-core-concepts)
- [Single Source of Truth Pattern](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.1-single-source-of-truth-pattern)
- [Contract System](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.2-contract-system)
- [Control Mode Architecture](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.3-control-mode-architecture)
- [System Architecture](https://deepwiki.com/wuxiaoqiang12/IB_Robot/4-system-architecture)
- [Configuration System](https://deepwiki.com/wuxiaoqiang12/IB_Robot/5-configuration-system-(robot_config))
- [Inference Pipeline](https://deepwiki.com/wuxiaoqiang12/IB_Robot/7-inference-pipeline)
- [Action Dispatch](https://deepwiki.com/wuxiaoqiang12/IB_Robot/8-action-dispatch)
- [Data Pipeline](https://deepwiki.com/wuxiaoqiang12/IB_Robot/9-data-pipeline)

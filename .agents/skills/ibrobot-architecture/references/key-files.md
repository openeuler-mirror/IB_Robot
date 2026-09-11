# Key Files and Package Responsibilities

## When to Read

- 需要快速定位某个职责对应的包或源文件（如推理、动作分发、契约合成、技能网关、记录转换等）
- 评估某项代码改动是否会影响包的职责边界、公开 API、配置键、跨包依赖或运行约束
- 进行架构审查时需要判断 package README 是否与代码行为存在漂移

## Package Responsibilities (by layer)

### Config & Contract

| Package | Primary Responsibility |
|---------|----------------------|
| `robot_config` | SSOT robot YAML (loading/overlay/validation/digest), contract synthesis, launch orchestration |
| `ibrobot_msgs` | Global interface definitions (15 actions, 48 msgs, 39 srvs) |
| `tensormsg` | ROS message ↔ tensor codec registry (`TensorMsgConverter.encode/decode`) |
| `inference_manifest` | Schema-v3 model bundle manifest: strict loader/validator, fingerprints, writer |

### Inference

| Package | Primary Responsibility |
|---------|----------------------|
| `inference_service` | Unified inference runtime: `pipeline_policy_node`, `model_service_node`, `global_inference_scheduler_node`, `pure_inference_node`, backends (torch/ascend/hisilicon/rknn/hmm/onnx), sessions, scheduler |
| `perception_service` | Perception model plugins for typed services (ram_plus/sam2/siglip2/grounding_dino/graspgen) + VLM scene analysis node |
| `observation_transport` | H.264/RTP video ingress-egress: `FrameIngress`/`NativeFrameIngress`, video codec registry (software/nvidia/ascend) |
| `model_utils` | ONNX/OM/RKNN/HMM export and bundle packaging CLIs (`models/_work` convention) |

### Action & Embodied Skills

| Package | Primary Responsibility |
|---------|----------------------|
| `action_dispatch` | Pull-based action dispatch: dispatchers, schedulers (continuous/wait_for_feedback), executors (topic/benchmark), temporal smoothing |
| `task_dispatch` | `task_executor` action server: sequential task plans (MOVE_TO_POSE / GRIPPER / WAIT) |
| `skill_catalog` | Skill manifest compiler + immutable catalog registry (SSOT for skill manifests/profiles) |
| `skill_library` | Capability gateway: `skill_executor_node`, admission/idempotency, primitive resolution |
| `robot_skill_cli` | Controlled CLI surface for LLM/Agent (`robot-skill`, `ibrobot-perceive`) |
| `embodied_agent` | Task entry, planning (`agent_plan_node`), visual game gateway nodes |
| `embodied_bringup` | Embodied pipeline launch orchestration (`embodied_pipeline.launch.py`) |
| `embodied_common` | ROS-independent shared helpers (contracts, base node, payload hashing) |
| `safety_guard` | Read-only skill/primitive validation preflight (`/embodied/validate_skill`) |
| `manipulation_service` | GraspGen grasp planning/verification services (`PlanGrasp`, `VerifyGrasp`) |
| `manipulation_execution` | Closed-loop pick/place/imitate executors with executor identity binding |

### Data & Evaluation

| Package | Primary Responsibility |
|---------|----------------------|
| `dataset_tools` | Episode recording (`recorder_server`), bag_to_lerobot (LeRobot v3.0 export), policy_eval, rerun_viewer |
| `benchmark` | Benchmark runtime + LIBERO adapter (evaluation, `StepBenchmark` service) |
| `attention_viz` | ACT attention weight heatmap visualization |
| `semantic_mapping` | Persistent RGB-D 3D semantic mapping (open-vocabulary) |
| `object_tracker` | Single-target RGB-D tracking + collision-aware Nav2 following |

### Hardware, Simulation & Motion

| Package | Primary Responsibility |
|---------|----------------------|
| `so101_hardware` | ros2_control hardware plugin: SO-101 arm via Feetech |
| `lekiwi_hardware` | ros2_control hardware plugin: LeKiwi arm + base via Feetech STS |
| `aero_hand_hardware` | Aero Hand command/state bridge (intentionally not ros2_control) |
| `hardware_mock` | Contract-driven mock backend for end-to-end inference pipelines |
| `robot_description` / `lekiwi_description` | URDF/xacro/meshes for SO-101 / LeKiwi |
| `sim_models` | Scene assets + scene compiler (Gazebo/MuJoCo) |
| `robot_moveit` | MoveIt 2 config, `moveit_gateway` (`MoveToPose`), PLACO servo, IK workers |
| `robot_teleop` | Teleop bridges (50 Hz serial-to-controller), glove/VR/mhandpro sources |
| `robot_navigation` | Nav2 client, `navigation_command_server` (`ExecuteNavigation`), cmd_vel bridge |
| `robot_calibration` | Sensor calibration capture/artifact/validation/activation workflows |
| `voice_asr_service` | sherpa-onnx ASR + speech direction nodes |
| `voice_tts_service` | Manifest-backed ZipVoice TTS typed service |

### Vendored / Submodules

| Path | Note |
|------|------|
| `fast_lio`, `fast_calib`, `livox_ros_driver2`, `omni_wheel_controller` | git submodules (LiDAR odometry / calibration / driver / wheel controller) |
| `pymoveit2`, `rosclaw` | git submodules (vendored) |
| `workflows` | CI gate definition (Jenkins shared library), **not** a ROS package |

## README as Local Architecture Contract

Each package-level `README.md` is treated as the package's local architecture contract. It should describe the package's responsibilities, public entry points, launch/configuration usage, data flow, dependency boundaries, and known constraints.

When code changes alter any of the following, the package README must be checked and updated if needed:

1. Package responsibilities or prohibited responsibilities
2. Public APIs, CLIs, launch arguments, topics, services, or actions
3. Configuration keys, defaults, or SSOT sources
4. Data flow, tensor/ROS message contracts, or control mode behavior
5. Cross-package dependencies or layer boundaries
6. Operational limitations, required hardware, or setup steps

Architecture reviews should flag README drift as an architecture issue when code behavior and documentation diverge. A stale README is not a minor documentation style problem; it invalidates the package contract and makes future architecture reviews unreliable.

## Key Files Reference

| File | Purpose |
|------|---------|
| `robot_config/config/robots/so101_single_arm.yaml` | Canonical arm robot configuration (SSOT) |
| `robot_config/robot_config/loader.py` | Config loading, overlay merge, validation, digest |
| `robot_config/robot_config/config.py` | `RobotConfig` dataclass, `to_contract()` |
| `robot_config/robot_config/contract_utils.py` | Contract/Spec dataclasses, `StreamBuffer`, `decode_value`/`encode_value` |
| `robot_config/robot_config/generators/contract.py` | Dict-path contract builder (`build_contract_from_robot_config_dict`) |
| `robot_config/robot_config/runtime_target.py` | hardware/simulation/benchmark target resolution |
| `robot_config/launch/robot.launch.py` | Main launch orchestrator (25 launch arguments) |
| `robot_config/robot_config/launch_builders/` | Modular launch builders (30 builders + `sim_backend/` adapters) |
| `inference_manifest/inference_manifest/loader.py` | Strict schema-v3 manifest loading/validation |
| `inference_service/inference_service/pipeline_policy_node.py` | Policy serving node (monolithic/distributed/scheduled) |
| `inference_service/inference_service/unified_runtime/handle.py` | `ModelRuntimeHandle` (admission/deadlines/lifecycle) |
| `inference_service/inference_service/model_sessions/` | Per-backend `ModelSession` implementations |
| `inference_service/inference_service/backends/registry.py` | Backend registry + conformance validation |
| `inference_service/inference_service/global_inference_scheduler_node.py` | Session lifecycle, routing, admission |
| `inference_service/inference_service/model_service_node.py` | Generic typed model-service host |
| `perception_service/perception_service/model_service_plugins.py` | Perception typed-service plugins |
| `observation_transport/observation_transport/native_frame_ingress.py` | Production H.264/RTP frame ingress |
| `action_dispatch/action_dispatch/action_dispatcher_node.py` | Pull-based action dispatcher |
| `action_dispatch/action_dispatch/scheduled_action_dispatcher_node.py` | Session-based dispatcher + safe stop |
| `action_dispatch/action_dispatch/temporal_smoother.py` | Cross-frame action chunk smoothing |
| `skill_library/skill_library/skill_executor_node.py` | Capability gateway (skill/primitive execution) |
| `task_dispatch/task_dispatch/task_executor_node.py` | Task plan action server |
| `dataset_tools/dataset_tools/episode_recorder.py` | Episodic rosbag recording |
| `dataset_tools/dataset_tools/bag_to_lerobot.py` | Rosbag → LeRobot v3.0 dataset conversion |

# Data Flow Architecture

## When to Read

- 需要追踪 Observation Flow 或 Action Flow 的具体代码路径与执行顺序
- 排查推理（inference）相关的问题：manifest/后端选择、ModelRuntimeHandle、Monolithic/Distributed/Scheduled 模式
- 排查动作抖动、chunk 对齐、temporal smoothing、scheduler/executor 选择相关问题
- 评估云边协同部署架构（H.264/RTP 视频通道、分布式推理协议），决定如何拆分 Edge / Cloud 节点
- 排查感知/语音 typed service（model_service）相关问题

## Observation Flow (Sensors → Inference)

```
Camera/JointState → ROS Topic → tensormsg decode → StreamBuffer → PipelinePolicyNode → ModelRuntimeHandle → ModelSession
```

**Key Code Paths**:
1. `tensormsg.TensorMsgConverter.decode(msg, spec)` in `src/tensormsg/tensormsg/converter.py`（registry 化的 ROS 消息 → tensor 解码；robot_config 的 `decode_value()` 是其薄封装）
2. `StreamBuffer.push/select` in `src/robot_config/robot_config/contract_utils.py`（按 capture 时间戳排序的历史缓存，`hold/asof/drop` 对齐 + live-age 检查）
3. `PipelinePolicyNode` in `src/inference_service/inference_service/pipeline_policy_node.py`（observation 订阅与同步；RTP 来源的 key 由 `observation_sync.select_synchronized_streams` 做时间戳同步）
4. `ModelRuntimeHandle.execute` in `src/inference_service/inference_service/unified_runtime/handle.py`（admission、deadline、cancellation、recovery 的唯一入口）

## Action Flow (Inference → Hardware)

```
Model → VariantsList (/actions/<pipeline_id>) → Action Dispatcher → TemporalSmoother
      → Scheduler decision → Executor submit → Controller Topic/Action → Hardware
```

**Key Code Paths**:
1. `ActionDispatcherNode`（pull 型：`DispatchInfer` action client）或 `ScheduledActionDispatcherNode`（session 型：`OpenInferenceSession` / `ScheduledDispatchInfer` / `CloseInferenceSession` + `safe_stop`）in `src/action_dispatch/action_dispatch/`
2. `TemporalSmoother.update()` in `src/action_dispatch/action_dispatch/temporal_smoother.py`（跨帧 chunk 指数平滑，两个 dispatcher 共用）
3. Scheduler registry: `continuous`（watermark 补充请求）/ `wait_for_feedback`（`StepBarrierScheduler`，单 in-flight、fail-closed），见 `schedulers/registry.py`
4. Executor registry: `topic`（按 contract action specs 发布 `Float64MultiArray` / `JointTrajectory`）/ `benchmark`（`StepBenchmark` service），见 `executors/registry.py`；pairing guard 强制 `topic+continuous`、`benchmark+wait_for_feedback`

## Inference Execution Modes

`PipelinePolicyNode` 的 `execution_mode` 决定推理拓扑；`scheduler_enabled` 时额外挂载 session 化控制面（仅 monolithic）：

### Monolithic Mode (Default)

单进程端到端：`InferencePipelineManager`（`pipeline/manager.py`，由 `pipeline/factory.py:create_pipeline_manager` 构造）在进程内驱动 `ModelRuntimeHandle` → 后端 `ModelSession`，同时服务 `DispatchInfer` action server 并发布 `VariantsList`。

### Distributed Mode (Cloud-Edge)

边缘做前/后处理与视频采集，云做模型推理：

```
Edge                                      Cloud
┌────────────────────────────┐           ┌────────────────────────────┐
│ DeviceVideoStreamManager   │  H.264    │ H264RtpReceiver            │
│  └ FrameIngress/           │  RTP/UDP  │  └ ComputeVideoStreamManager│
│    NativeFrameIngress      │ ────────► │     → observation_sync     │
│ EdgeProcessorRuntime       │           │ PureInferenceNode          │
│  (pre/postprocess)         │ ◄──────── │  └ DistributedCloudService │
│ EdgeSession.accept_result  │ Distributed│     └ CloudBackendRuntime │
└────────────────────────────┘ Result    └────────────────────────────┘
```

- 视频通道：`observation_transport` 的 `NativeFrameIngress` 做 admission → H.264 编码（codec registry：software/nvidia/ascend_ffmpeg）→ `H264RtpSender`；云侧 `H264RtpReceiver` 接收后同步。
- 控制消息：`DistributedInferenceRequest/Result`（ibrobot_msgs），tensor 经 `tensormsg` 编解码，协议版本 `distributed/types.py:PROTOCOL_VERSION`。
- 云端 launch：`launch/cloud_inference.launch.py` / `launch/local_distributed_inference.launch.py`（边缘侧运行 `pipeline_policy_node`）。

### Scheduled Mode (Control Plane)

客户端 → `GlobalInferenceSchedulerNode`（`/inference/session/open`、`/inference/dispatch`、`/inference/close`；admission 由 `GoalSlotPool` + `DeadlineReservationTable` + `IdempotencyLedger` 组成）→ 各 pipeline 的 `OpenInferenceSession` / `ScheduledDispatchInfer` / `CloseInferenceSession` action server（`PipelinePolicyNode` 内，由 `ProductSessionController` 管理两阶段 reset/close barrier 与容量租约）。调度器本身是 transport/control-plane proxy，不直接持有 runtime。

## Unified Runtime Layering

```
inference_manifest (schema v3, bundle artifacts + semantic bindings)
        │  manifest-driven selection
        ▼
backends/registry (torch │ ascend │ hisilicon │ rknn │ hmm │ onnx)   ← conformance validation
        ▼
ModelSession (model_sessions/*)        ← native runtime state machine
        ▼
ModelRuntimeHandle (unified_runtime/handle.py)  ← admission/deadlines/cancel/recovery
        ▼
RuntimeAssembly + RegistrySet (unified_runtime/registry.py)  ← composition root
```

- Backend 选择完全由所选 deployment 的 typed runtime profile 决定（`BackendRegistry.validate`）；`CANONICAL_BACKENDS = ("torch", "ascend", "hisilicon", "rknn", "hmm", "onnx")`。
- `PureInferenceEngine`（`core/pure_inference_engine.py`）是无 ROS 依赖的单 pipeline facade，仅用于 model_utils 离线/诊断工具，不是 ROS 云端节点。

## Model Service (Perception / Voice Typed Services)

`model_service_node`（inference_service）是通用 typed-service 宿主：从 bundle manifest 加载 `ModelServicePlugin`（adapter_class 注入），按参数化的 `service_type`（`ibrobot_msgs/srv/<Name>`）创建服务，响应统一携带 `ModelRuntimeInfo` / `inference_time_ms` / `success` / `message`。

| Plugin (perception_service) | Typed Service |
|---|---|
| `RAMPlusRecognizeTagsPlugin` | `RecognizeTags` |
| `SAM2GenerateMasksPlugin` / `SegmentDetectionsPlugin` | `GenerateMasks` / `SegmentDetections` |
| `SigLIP2EncodeEmbeddingsPlugin` / `SigLIP2EncodeTextPlugin` | `EncodeEmbeddings` / `EncodeText` |
| `GroundingDetectPlugin` / `GroundingDINORawDetectPlugin` | `GroundingDetect` |
| `GraspGenGenerateGraspsPlugin` | `GenerateGrasps` |
| voice_tts_service ZipVoice plugin | `SynthesizeSpeech` |

Perception/voice session builders 在 `runtime_composition.build_model_service_runtime_dependencies` 中注册进共享 `SessionBuilderRegistry`（`model_sessions` 复用同一套 runtime 层）。

## Temporal Smoothing

Cross-frame action chunk smoothing for seamless motion:

```
weight[k] = exp(-temporal_ensemble_coeff * k)
```

- Default `temporal_ensemble_coeff = 0.01` (from ACT paper)
- Precomputed weights for fast blending
- Aligns new chunk with `actions_executed` from previous chunk
- `TemporalSmootherManager` 管理多 pipeline 实例；两个 dispatcher 均可启用（`~/toggle_smoothing`）

**File**: `src/action_dispatch/action_dispatch/temporal_smoother.py`

## Naming Notes (legacy → current)

- `lerobot_policy_node` 已删除（脚本 `scripts/check_inference_legacy_identifiers.py` 将其列为禁用标识符）；后继者为 `pipeline_policy_node`。
- `frame_ingress` / `native_frame_ingress` 在 inference_service 下只是兼容别名，真实实现在 `observation_transport` 包。
- 契约合成为内存行为（`to_contract()` / `build_contract_from_robot_config_dict()`），没有 `/tmp` 契约缓存文件。

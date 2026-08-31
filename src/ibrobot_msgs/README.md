# ibrobot_msgs

`ibrobot_msgs` 是 IB-Robot 系统的**统一接口定义包**，包含所有 ROS 2 消息（msg）、动作（action）和服务（srv）的定义。

## 0. 公开请求类型的 schema 版本分离（v1 / v2）

四个公开请求类型——`SkillCommand.Goal`、`PrimitiveCommand.Goal`、`ValidateSkill.Request` 和
`ValidatePrimitive.Request`——现在都以 `uint32 schema_version` 作为**首字段**。该字段取值集合为
`{1, 2}`，其他值返回 `SKILL_SCHEMA_INVALID`。

- **v1**：仅承载 manipulation 字段（`motion_direction` / `motion_distance` 等）与原 3 种控制模式、10 个
  primitive；v1 请求**拒绝**任何导航字段（`direction` / `distance` / `degree` / `x` / `y` / `yaw`）。
- **v2**：在 v1 字段集合之上**并集**新增导航字段（`direction`、`distance`、`degree`、`has_x`、`x`、
  `has_y`、`y`、`has_yaw`、`yaw`），并允许 `base_navigation` 控制模式与 3 个导航 primitive
  （`nav_straight` / `nav_turn` / `nav_abs_coordinate`）。v2 请求**必须**显式设置 `schema_version=2`，
  不得依赖默认零值。

这是一个 **breaking IDL change**：修改同名 `.action` / `.srv` 的字段集合会改变 ROS type hash，
不提供旧二进制 wire compatibility，因此必须**原子重建并部署**全部 workspace producer / consumer
（`skill_library`、`safety_guard`、`embodied_agent`、`robot_skill_cli` 等），不得混跑 v1 与 v2 二进制。

`embodied_common.wire_contracts.validate_public_request_wire_contracts()` 在节点启动时对该契约执行
preflight 校验：四个请求类型必须以 `schema_version` 为首字段，且取值属于 `{1, 2}`；不匹配时节点不进入
ready，并以 `SKILL_SCHEMA_INVALID` 报告。该 preflight 与运行期 `primitive_contract_digest` 校验共同构成
schema / 契约二重门禁，防止 v1 与 v2 二进制混跑。

下游 `WorkflowStep.schema_version` 取值同样从固定 `1` 扩展为 `{1, 2}`，并与所属 capability 的
schema 版本一致（由 `robot_context.context_schema_version` 决定，详见
`docs/lightweight_skill_package_registry_design_zh.md` 的 Schema 版本演进一节）。

## 1. 消息定义（msg/）

### `MHandProFrame.msg` / `HumanHandState.msg`

`MHandProFrame` 是可选的 mHandPro 厂商原始数据边界，保存左右侧、帧序号、设备电量、
源频率、20 个 position/quaternion、5 个虚拟指尖、逐节点 sensor state、角速度、线加速度和
线速度。它用于显式启用的录制、诊断和后续算法迭代，不包含目标机械手语义；生产配置默认不发布。

`HumanHandState` 是厂商无关的手部契约：保留 landmarks/orientations/virtual tips，并用稳定字段名
发布尺度无关的解剖角度、置信度、张开分数和有效状态。机械手 retargeter 只依赖这个消息，因此
mHandPro 不感知 Aero Hand 的 7 个通道或 Amazing Hand 的 3 个通道。

### `TaskCommand.msg`

具身 AI 链路的主要任务数据载体，贯穿任务入口、规划和执行全流程。`task_id` 已移入
`dispatch_binding`，规划后的 `TaskCommand` 必须携带非空 `workflow_steps`、完整版本绑定和正确的
`workflow_digest`，且 `root_lease_nonce` / `dispatch_nonce` 仍为空。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `dispatch_binding` | `DispatchBinding` | 任务/版本信封（含 `task_id`、`root_task_id`、`task_budget` 与 exact registry identity） |
| `source` | `string` | 请求来源，如 `voice`、`text`、`vision`、`api` |
| `raw_command` | `string` | 原始自然语言命令 |
| `task_type` | `string` | 规则规划后的任务类型 |
| `workflow_steps` | `WorkflowStep[]` | 规划后的有序步骤；raw 请求可为空，planned 请求必须非空 |
| `target_name` | `string` | 顶层目标提示，仅作规划输入，task executor 必须按各 `WorkflowStep` 自身参数执行 |
| `container_name` | `string` | 顶层容器提示，仅作规划输入，task executor 必须按各 `WorkflowStep` 自身参数执行 |
| `place_name` | `string` | 顶层放置位提示 |
| `motion_direction` | `string` | 顶层相对运动方向提示 |
| `motion_distance` | `float32` | 顶层相对运动距离提示 |
| `priority` | `uint8` | 优先级：0=低、1=中、2=高、3=紧急 |
| `timeout_sec` | `float32` | 请求侧超时提示，实际 deadline 由 `dispatch_binding.task_budget` 决定 |
| `context_json` | `string` | 仅携带 planner/perception 注解；不得承载 registry identity、task budget 或 `skill_sequence` |

### `TaskStatus.msg`

任务执行状态报告，由 `task_entry_node`、`task_executor_node` 等节点发布到 `/embodied/task_status`。
所有 action result、validation response 和 status response 都返回实际使用或观测到的 registry identity。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，v1 固定为 `1` |
| `task_id` | `string` | 对应任务 ID |
| `state` | `string` | 当前状态（`planned` / `planning` / `rejected` / `executing` / `completed` / `failed`） |
| `success` | `bool` | 当前阶段是否成功 |
| `current_skill` | `string` | 正在执行的技能名 |
| `completed_skills` | `string[]` | 已完成技能列表 |
| `error_code` | `string` | 明确错误码 |
| `message` | `string` | 详细说明 |
| `recoverable` | `bool` | 是否可恢复 |
| `replan_requested` | `bool` | 是否建议重规划 |
| `actual_registry_epoch` | `string` | 实际使用的 registry epoch |
| `actual_registry_generation` | `uint64` | 实际使用的 registry generation |
| `actual_registry_digest` | `string` | 实际使用的 registry digest |
| `provenance_digest` | `string` | provenance digest，用于来源一致性校验 |

### `SkillCapabilityStatus.msg`

Gateway 对一个公开高层技能的非阻塞 readiness 快照。`semantic_level` 由 Gateway 从捕获的 snapshot 读取，
调用方不得自行声明级别。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，v1 固定为 `1` |
| `name` | `string` | 技能名 |
| `semantic_level` | `string` | `atomic_operator` 或 `skill`，由 snapshot 决定 |
| `planner_visible` | `bool` | 是否对 planner 可见 |
| `ready` | `bool` | 当前是否可接收该技能 |
| `reason` | `string` | 就绪时为空；拒绝时为稳定的 `CODE: detailed message` 格式 |
| `required_control_mode` | `string` | 该 Gateway 配置要求的控制模式 |

`reason` 的 Gateway admission 代码包括 `MOTION_NOT_AUTHORIZED`、`CONTROL_MODE_MISMATCH`、
`SKILL_BUSY`、`TIMEOUT_EXCEEDS_POLICY`、`CAPABILITY_NOT_READY` 和 `GATEWAY_FINALIZATION_FAILED`。
`GATEWAY_FINALIZATION_FAILED` 仅在 Gateway lease/ledger 终态化失败时出现，表示运动授权或 lease
状态可能不一致，调用方应停止派发新动作并交由操作员介入。可用性不足时，详细消息从以下
受控词表中取首个缺失项：`validate skill service unavailable`、`task executor action unavailable`、
`arm trajectory action unavailable`、`ee pose unavailable or stale`。例如：
`CAPABILITY_NOT_READY: ee pose unavailable or stale`。

### Capability Gateway 状态服务

`GetSkillGatewayStatus.srv` 是高层 Gateway 状态边界。请求固定为 `schema_version`、`task_id`、`payload_hash`；
响应固定为 `schema_version`、`robot_name`、`motion_authorized`、`active_control_mode`、`busy`、`active_task_id`、
`active_owner_kind`、`active_workflow_digest`、`active_workflow_step_index`、`control_plane_ready`、
`control_plane_state`、`control_plane_error_code`、`default_skill_timeout_sec`、`task_budget_sec`、`rpc_timeout_sec`、
`config_digest`（legacy 别名，字节级等于 `capability_digest`）、`capability_digest`、`registry_epoch`、
`registry_generation`、`registry_digest`、`primitive_contract_digest`、`source_release_digest`、
`provenance_digest`、`retained_generations`、`request_state`、`request_error_code` 和
`capabilities`（`SkillCapabilityStatus[]`）。

任务查询仅表达 identity/ledger 状态，不返回内部执行细节：空 `task_id` 和空 hash 不查询；已知 task ID
且 hash 为空返回 `active` 或 `terminal`；未知 task ID 返回空状态；task ID 与同一 hash 匹配时返回
`DUPLICATE_TASK_ID`，与不同 hash 匹配时返回 `TASK_ID_CONFLICT`，只有 hash 时返回 `INVALID_ARGUMENT`。
终态动作错误不是 `request_error_code` 的内容。

`control_plane_ready=true` 仅当：状态服务可达；当前 registry snapshot 与本地可重算的三套 digest 完全同步；
primitive contract digest、robot context 和 executor identity 检查通过；Agent plan store 的
plan/validate/confirm/execute 端点可达；且系统未处于初始编译、registry resync 或 reload 事务中。
`control_plane_state` 取值 `STARTING`、`SYNCING`、`READY`、`RELOADING`、`FAILED`；失败时
`control_plane_error_code` 为稳定代码，通常为 `SKILL_REGISTRY_NOT_READY`、`SKILL_RELOAD_IN_PROGRESS`、
`SKILL_SNAPSHOT_DIGEST_MISMATCH` 或 `SKILL_EXECUTOR_IDENTITY_MISMATCH`。`control_plane_ready` **不**包含
`motion_authorized`、单个 capability ready、busy、控制模式或工作空间安全条件，因此 catalog/plan/validate 可在
`control_plane_state=READY` 且 `motion_authorized=false` 时继续工作，运动授权在 action admission 时返回
`MOTION_NOT_AUTHORIZED`。状态服务永远不会暴露任一 nonce。

Agent plan 的 `execution_mode` 是显式控制模式契约：`interactive_confirmation` 为旧分阶段入口默认值，
`immediate_after_presentation` 仅由 `robot-skill run-workflow` 使用。模式在计划创建时捕获，写入
`AgentPlan`，并由 `ConfirmAgentPlan` 精确匹配；它不授予或修改 `authorize_motion`。立即模式仍必须完成
exact catalog、validation、计划展示并 flush、技术绑定、action admission、停止收敛和权威终态校验。

Gateway 的高层动作边界是 `SkillCommand.action`，dry-run 边界是 `ValidateSkill.srv`。状态服务不携带
执行器依赖、ROS transport 名称、配置路径、primitive sequence、坐标或底层控制器状态；这些都不是
`SkillCapabilityStatus` 或 `GetSkillGatewayStatus` 字段。

### `SceneAnalysisRequest.msg`

结构化场景理解请求，发布到 `perception_service_node`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `string` | 请求唯一 ID |
| `source` | `string` | 请求来源（如 `cli`、`vlm_planner`） |
| `session_id` | `string` | 会话 ID，用于多轮上下文 |
| `user_text` | `string` | 用户输入文本 |
| `context_json` | `string` | 附加上下文（关注目标、安全限制等） |
| `timeout_sec` | `float64` | 本次请求的模型超时（覆盖默认值） |

### `SceneAnalysisResult.msg`

结构化场景理解结果，由 `perception_service_node` 发布到 `/embodied/perception_result`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `string` | 对应请求 ID |
| `success` | `bool` | 理解是否成功 |
| `summary` | `string` | 场景文本摘要 |
| `objects_json` | `string` | 检测到的物体列表（JSON） |
| `error_message` | `string` | 失败时的错误信息 |

### 视觉游戏控制服务

`StartVisualGame.srv` 按调用方 `request_id`、`game_name` 和 `expected_config_digest` 幂等发起视觉游戏，
并返回 `accepted`、`duplicate`、实际 `config_digest` 和错误字段；
`GetVisualGameResult.srv` 按 `request_id` 返回 `found`、`terminal`、`success`、`game_name`、
通用 `result_json`、常用结果便捷字段 `scene_summary`、`config_digest` 和错误字段。两者是 Agent/CLI 的纯控制面，
不传图像、raw response 或音频，
也不负责 TTS。

`VisualGameEvent.msg` 是视觉游戏的异步事件通知边界，发布到
`/embodied/visual_game_events`。它包含调用方 `request_id`、每次真实准入生成的 `execution_id`、游戏名、版本化 handler、是否要求播报（`announce`）、
`accepted`/`succeeded`/`failed` 状态、结果摘要、错误码和配置摘要，供 TTS、UI 或日志消费者订阅。
执行 TTS 等外部副作用的消费者应使用 live-only 订阅，并按 `execution_id` 去重，不能在重启后回放旧事件；
同一 request ID 在 ledger retention 过期后重新准入时会得到新的 execution ID。

### `Detection2D.msg`

二维检测、mask 和由深度反投影得到的 3D 目标几何信息。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `header` | `std_msgs/Header` | 目标检测所在相机坐标系和时间戳 |
| `label` | `string` | 检测类别或文本 prompt 匹配标签 |
| `confidence` | `float32` | 检测置信度 |
| `bbox` | `float32[4]` | 像素坐标 `[x_min, y_min, x_max, y_max]` |
| `mask` | `sensor_msgs/Image` | `mono8` 二值分割 mask，与输入图像同尺寸 |
| `centroid_xyz` | `geometry_msgs/Point` | mask 内有效深度点的可见表面均值，单位米 |
| `volume_centroid_xyz` | `geometry_msgs/Point` | 凸包体积质心，退化或 SciPy 不可用时回退到 `centroid_xyz` 语义 |
| `volume_m3` | `float32` | 凸包体积，`0.0` 表示点数不足、几何退化或无法计算体积质心 |
| `point_count` | `int32` | mask 内有效深度点数量 |

### `DetectionArray.msg` 与抓取感知服务

抓取感知使用显式图像输入的强类型服务。`GroundingDetect` 返回 bbox、label 和 confidence；当所选 deployment
不内联分割时，`SegmentDetections` 接收同一图像与检测数组并补齐 `mono8` mask。PC 的组合 Torch deployment
可直接由 `GroundingDetect` 返回 mask，310P 则使用 `GroundingDetect -> SegmentDetections` 两个 named deployment。

这两个服务不订阅相机或深度 topic。抓取 planner 缓存 RGB-D 与 CameraInfo，用返回 mask 在同一帧深度上计算
表面质心、体积质心和点数，并通过 `PlanGrasp` 返回几何结果。

### 显式图像感知服务

| 服务 | 职责 |
| --- | --- |
| `GenerateMasks` | SAM2 无 prompt 盲扫，返回与输入同尺寸的 `mono8` masks |
| `RecognizeTags` | RAM++ 整图诊断标签；mask 请求按调用方 `excluded_labels` 过滤后返回 `max_mask_candidates` 个候选 |
| `EncodeEmbeddings` | 一张图像最多 8 个 masks 的 SigLIP2 批量编码与候选标签匹配 |
| `EncodeText` | 最多 16 条文本的 SigLIP2 查询时编码；不携带图像或持久 image embedding |
| `GroundingDetect` | 显式图像输入的低频目标确认，不参与建图主链 |
| `SegmentDetections` | 对显式图像和检测 bbox 执行 box-prompt 分割并补齐同尺寸 mask |

所有新服务返回 `ModelRuntimeInfo`。`EncodeText` 返回瞬时、归一化的查询文本向量，供语义地图内部与私有
image embedding 比较；持久对象 embedding 仍是语义地图私有数据，不进入对象快照消息。
`RecognizeTags` 在 masks 为空时维持整图识别语义，即使 `include_image=false`，以兼容原有整图调用方；
有 masks 时调用方可关闭整图识别，仅请求局部候选。

### `GraspCandidate.msg`

机器人无关的抓取候选，用于 manipulation service 与执行脚本之间传递 GraspGen 结果。

| 字段 | 说明 |
| --- | --- |
| `header` | 候选所在相机坐标系和时间戳 |
| `pose_matrix` | 4x4 row-major 抓取位姿矩阵 |
| `confidence` | GraspGen 候选置信度 |
| `collision_free` | GraspGen source gripper 碰撞过滤结果 |
| `target_width_m` | 从目标点云估计的抓取方向宽度，`0.0` 表示不可用 |
| `target_width_quality` | 宽度估计质量，范围通常为 `0.0` 到 `1.0` |
| `width_axis_camera` | 宽度估计轴在相机坐标系下的单位方向 |
| `target_width_min_offset_m` | 目标靠 `-width_axis_camera` 一侧的稳健边界，相对候选位姿原点 |
| `target_width_max_offset_m` | 目标靠 `+width_axis_camera` 一侧的稳健边界，相对候选位姿原点 |

### `SemanticObject3D.msg` / `SemanticObject3DArray.msg`

3D 语义地图的持久目标接口。`object_id` 跨观测和进程重启保持稳定；`pose` 和 `size` 位于配置的
持久固定坐标系，`first_seen`、`last_seen`、`observation_count` 和 `active` 描述目标生命周期。
SigLIP 特征向量属于关联内部状态，不通过公共消息发布。
`state`、`map_version`、`object_version` 和 readiness 字段用于 fail-closed 查询与动作准入；
`stale`、`missing`、`lost` 对象可诊断查询但不得直接驱动导航或抓取。

### `RobotStatus.msg`

机器人当前状态汇报，包含末端位姿、关节状态和控制模式信息。

### 分布式观测视频控制面

`VideoStreamDescriptor.msg` 通过可靠、transient-local DDS 控制面声明一条 RTP/H.264 观测流，包含
pipeline/session、observation key、stream ID、RTP endpoint/SSRC、媒体参数以及 contract/deployment
双 fingerprint。`VideoStreamStatus.msg` 周期发布生命周期、codec backend、RTP-to-capture 时间映射、
keyframe readiness、队列深度、收发/丢包计数和最后错误。

`DistributedInferenceRequest.msg` 保留 `VariantsList tensors`，非图像观测和显式 DDS 图像模式继续通过
该字段传输。RTP 图像只通过 `stream_observation_keys`/`stream_ids` 引用已协商 descriptor，并使用
`observation_timestamp` 指定统一采样时间；H.264 payload 不进入 `VariantsList`。协议版本 3 要求 DDS
和 RTP 模式都完成严格版本协商，旧版本或 fingerprint/descriptor 不匹配时 fail closed。

### `TaskStep.msg`

单个任务步骤描述，用于 `ExecuteTaskPlan.action` 中的步骤序列。

### `Variant.msg` / `VariantsList.msg`

通用变体类型，用于传递不同类型的配置或参数值。

### `DispatchBinding.msg`

强类型任务/版本信封，所有跨进程 primitive、delegated、Skill、Validate 和 Workflow dispatch 必须携带。
把以前的 ad-hoc `task_id` 字段收敛为单一 identity。`schema_version` v1 固定为 `1`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，v1 固定为 `1` |
| `task_id` | `string` | 当前 dispatch ID |
| `root_task_id` | `string` | 跨整个 Workflow 稳定的 root ID；root Task/direct root Skill 与 `task_id` 相等 |
| `task_budget` | `TaskBudget` | 不可变 task deadline 信封，由任务入口所有，下游原样传播 |
| `expected_registry_epoch` | `string` | 期望 registry epoch |
| `expected_registry_generation` | `uint64` | 期望 registry generation |
| `expected_registry_digest` | `string` | 期望 registry digest |
| `workflow_digest` | `string` | Workflow typed digest；direct root 为空 |
| `workflow_step_index` | `uint32` | root-scope 统一使用 `0` 作为 canonical sentinel，不进入 digest |
| `root_lease_nonce` | `string` | 仅 Workflow child `SkillCommand`（及其只读 validation/internal dispatch）携带；其余必须为空 |
| `dispatch_nonce` | `string` | 仅 Gateway 已准入的内部 delegated/primitive dispatch 携带；root Task/Validate/SkillCommand 必须为空 |

nonce 字段不写入日志、snapshot 或持久化文件，也不是网络认证的替代品；存在非受信 ROS participant 的部署
必须使用 SROS 2 policy 限制 Begin/Finalize service、Primitive action 和 delegated endpoint。

### `TaskBudget.msg`

不可变 task deadline 信封，同一 root Task/Workflow 内每次 dispatch 共享。由任务入口 boundary 创建，
planner、task executor、SkillCommand、delegated executor 和 primitive dispatch 必须原样传播，不得为每个
catalog entry 重置。`schema_version` v1 固定为 `1`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，v1 固定为 `1` |
| `started_at` | `builtin_interfaces/Time` | 任务起始时间 |
| `deadline` | `builtin_interfaces/Time` | 绝对截止时间 |

### `WorkflowStep.msg`

planned Workflow 或 Agent plan 的一条 typed step。`skill_name` 是兼容字段名，可指向
`semantic_level=atomic_operator` 或 `skill` 的 catalog entry。step 不自带 registry identity，而是继承所属
planned `TaskCommand.dispatch_binding` 的 exact snapshot。`schema_version` 取值集合为 `{1, 2}`，由所属
capability 的 schema 版本决定（v1 capability 产生 `schema_version=1` step，v2 capability 产生
`schema_version=2` step），不再固定为 `1`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，取值 `{1, 2}`；由所属 capability 的 schema 版本决定，不再固定为 `1` |
| `skill_name` | `string` | 兼容字段名，引用 atomic_operator 或 skill catalog entry |
| `target_name` | `string` | 命名目标 |
| `container_name` | `string` | 指定容器；对 `place_in_container` 是释放后的视觉检测 query |
| `place_name` | `string` | 命名放置位 |
| `motion_direction` | `string` | 相对运动方向（v1 manipulation 字段） |
| `motion_distance` | `float32` | 相对运动距离（v1 manipulation 字段） |
| `direction` | `string` | 导航方向（v2 新增）；v1 step 必须为空 |
| `distance` | `float32` | 导航距离（v2 新增）；v1 step 必须为 `0.0` |
| `degree` | `float32` | 导航旋转角度（v2 新增）；v1 step 必须为 `0.0` |
| `has_x` | `bool` | 是否显式提供 `x`（v2 新增）；用于区分“未提供”与“显式零” |
| `x` | `float32` | 导航目标 x 坐标（v2 新增）；`has_x=false` 时忽略 |
| `has_y` | `bool` | 是否显式提供 `y`（v2 新增）；用于区分“未提供”与“显式零” |
| `y` | `float32` | 导航目标 y 坐标（v2 新增）；`has_y=false` 时忽略 |
| `has_yaw` | `bool` | 是否显式提供 `yaw`（v2 新增）；用于区分“未提供”与“显式零” |
| `yaw` | `float32` | 导航目标 yaw（v2 新增）；`has_yaw=false` 时忽略 |
| `timeout_sec` | `float32` | `<=0` 使用 entry 默认；正值仍受 Skill cap 和共享 `TaskBudget` 约束 |

v1 step 不得携带任何导航字段（`direction` / `distance` / `degree` / `has_x` / `x` / `has_y` / `y` /
`has_yaw` / `yaw`），校验失败返回 `SKILL_SCHEMA_INVALID`。v2 step 的 `has_x` / `has_y` / `has_yaw` 三个
bool 字段用于区分**字段缺失**与**显式零值**：`has_*=false` 表示调用方未提供该坐标分量，executor 不得
将其当作合法零目标；`has_*=true` 时对应的 `x` / `y` / `yaw` 才是有效输入。该设计避免把“未设置”静默
当作“移动到 (0, 0, 0)”的导航目标。

### `DelegatedExecutorIdentity.msg`

delegated executor 的 identity，记录在 registry snapshot 中并在每次 delegated dispatch 时重申。model 字段
对非模型 executor 必须全空，对模型 executor 必须全非空。`schema_version` v1 固定为 `1`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，v1 固定为 `1` |
| `name` | `string` | executor 名称 |
| `contract_version` | `string` | 契约版本 |
| `endpoint_kind` | `string` | endpoint 类型 |
| `endpoint_name` | `string` | endpoint 名称 |
| `configuration_digest` | `string` | 配置 digest |
| `model_deployment_name` | `string` | 模型 deployment 名（非模型 executor 为空） |
| `model_fingerprint` | `string` | 模型 fingerprint（非模型 executor 为空） |
| `model_bundle_digest` | `string` | 模型 bundle digest（非模型 executor 为空） |

### `SkillDiagnostic.msg`

所有 catalog compile 入口（compiler、reload、validate、agent plan/validate/confirm）发出的结构化诊断。
按 `source_relative_path`、`error_code`、`field_path`、`message` 排序。`schema_version` v1 固定为 `1`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，v1 固定为 `1` |
| `severity` | `uint8` | `ERROR=1` 或 `WARNING=2` |
| `error_code` | `string` | 稳定错误码 |
| `source_relative_path` | `string` | 相对 release root 的源路径，无关联文件时为空 |
| `field_path` | `string` | 字段路径 |
| `message` | `string` | 诊断消息 |

### `SkillRegistryEvent.msg`

仅在成功 reload 后发布，告知晚加入的订阅者应查询哪个 epoch/generation。QoS 固定为
RELIABLE / TRANSIENT_LOCAL / KEEP_LAST / depth 1。`schema_version` v1 固定为 `1`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `uint32` | schema 版本，v1 固定为 `1` |
| `registry_epoch` | `string` | 当前 registry epoch |
| `old_generation` | `uint64` | reload 前 generation |
| `new_generation` | `uint64` | reload 后 generation |
| `registry_digest` | `string` | registry digest |
| `capability_digest` | `string` | capability digest |
| `source_release_digest` | `string` | source release digest |
| `provenance_digest` | `string` | provenance digest |
| `profile_name` | `string` | profile 名 |
| `changed_skills` | `string[]` | 变更技能列表；仅 profile/context/文档变化时可为空 |

---

## 2. 动作定义（action/）

### `SkillCommand.action`

技能执行动作接口，由 `skill_executor_node` 提供，路径 `/embodied/execute_skill`。v1 中 `SkillCommand`
同时承载 Atomic Operator 和 Skill 以限制 IDL 迁移面；admission 从 snapshot 读取 `semantic_level`，
调用方不得自行声明级别。`task_id` 已移入 `dispatch_binding`。planned `SkillCommand` 必须提供完整期望
identity；direct root `SkillCommand` 的 `root_lease_nonce` 与 `dispatch_nonce` 都必须为空，
`workflow_step_index=0`。

**Goal 字段**

| 字段 | 说明 |
| --- | --- |
| `schema_version` | schema 版本，取值 `{1, 2}`；首字段，v2 必须显式设置为 `2`，不得依赖默认零值 |
| `dispatch_binding` | 完整任务/版本信封（含 `task_id`、`task_budget` 与 exact registry identity） |
| `skill_name` | 技能名（如 `pick_object`、`move_relative_ee`） |
| `target_name` | 命名目标；对 `pick_object` 表示运行时视觉文本查询 |
| `container_name` | 指定容器；对 `place_in_container` 表示释放后的视觉检测 query，不参与运动规划 |
| `place_name` | 命名放置位 |
| `motion_direction` | 相对运动方向（`forward` / `backward` / `left` / `right` / `up` / `down`）；v1 manipulation 字段 |
| `motion_distance` | 相对运动距离（米）；v1 manipulation 字段 |
| `arm_side` | HRI 模仿来源侧别：`left`、`right` 或 `auto` |
| `imitation_duration_sec` | HRI 从 `start` 开始的采集/跟随时长，单位秒；必须为正数 |
| `direction` | 导航方向（v2 新增）；v1 请求必须为空 |
| `distance` | 导航距离（v2 新增）；v1 请求必须为 `0.0` |
| `degree` | 导航旋转角度（v2 新增）；v1 请求必须为 `0.0` |
| `has_x` | 是否显式提供 `x`（v2 新增）；区分“未提供”与“显式零” |
| `x` | 导航目标 x 坐标（v2 新增）；`has_x=false` 时忽略 |
| `has_y` | 是否显式提供 `y`（v2 新增）；区分“未提供”与“显式零” |
| `y` | 导航目标 y 坐标（v2 新增）；`has_y=false` 时忽略 |
| `has_yaw` | 是否显式提供 `yaw`（v2 新增）；区分“未提供”与“显式零” |
| `yaw` | 导航目标 yaw（v2 新增）；`has_yaw=false` 时忽略 |
| `timeout_sec` | per-catalog-entry 请求超时上限，受共享 `task_budget` 剩余预算约束 |

**Result 字段**

| 字段 | 说明 |
| --- | --- |
| `success` | 是否成功 |
| `error_code` | 错误码 |
| `message` | 详细说明 |
| `executed_primitives` | 已完成 primitive 名称列表 |
| `actual_registry_epoch` | 实际使用的 registry epoch |
| `actual_registry_generation` | 实际使用的 registry generation |
| `actual_registry_digest` | 实际使用的 registry digest |
| `source_release_digest` | source release digest |
| `provenance_digest` | provenance digest |
| `debug_output_dir` | delegated executor 产生的可重放证据目录；无调试输出时为空 |
| `diagnostics` | `SkillDiagnostic[]` 结构化诊断 |

**Feedback 字段**

| 字段 | 说明 |
| --- | --- |
| `state` | 当前执行状态 |
| `detail` | 当前进度说明 |
| `actual_registry_epoch` | 实际使用的 registry epoch |
| `actual_registry_generation` | 实际使用的 registry generation |

### `PrimitiveCommand.action`

原子动作接口，由 `skill_executor_node` 提供，路径 `/embodied/execute_primitive`。内部和 delegated
跨进程 dispatch 必须携带精确的 active `dispatch_binding`；对显式准入的外部 root Primitive，Gateway 在
取得排他 root lease 后才 canonicalize binding。Primitive server 逐项匹配 active child binding 的每个
字段，而非仅凭 task ID；child finalize/cancel 后 `dispatch_nonce` 立即失效。

**Goal 字段**

| 字段 | 说明 |
| --- | --- |
| `schema_version` | schema 版本，取值 `{1, 2}`；首字段，v2 必须显式设置为 `2`，不得依赖默认零值 |
| `dispatch_binding` | 完整任务/版本信封；外部 root Primitive 由 Gateway 在取得 root lease 后 canonicalize |
| `execution_token` | 父 `PickObject` goal UUID 派生的关联 token；授权始终以 `dispatch_binding` 为准 |
| `primitive_name` | 原子动作名。v1 白名单为 `move_to_named_pose`、`move_to_pose`、`move_to_configuration`、`move_relative_ee`、`move_to_joint_positions`、`move_through_joint_positions`、`open_gripper`、`close_gripper`、`rotate_gripper_cw` 和 `rotate_gripper_ccw`（共 10 个）。v2 在 v1 白名单之上**新增** `nav_straight`、`nav_turn`、`nav_abs_coordinate` 三个导航 primitive；v1 请求不得引用这三个名称，否则返回 `SKILL_UNKNOWN_PRIMITIVE` |
| `pose_name` | 命名位姿（`move_to_named_pose` 使用） |
| `target_pose` | 动态 base-frame 位姿（`move_to_pose` 使用） |
| `relative_dx/dy/dz` | 相对增量（`move_relative_ee` 使用，单位米） |
| `velocity_scaling` | MoveIt 速度比例，`0.0` 使用默认值 |
| `gripper_position` | 夹爪目标开合量（`[0.0, 1.0]`） |
| `joint_names` | 关节名列表，joint primitive 使用 |
| `joint_positions` | 单个关节目标位置，`move_to_joint_positions` 使用 |
| `primitive_duration_sec` | 单点关节轨迹持续时间 |
| `joint_waypoints` | 扁平化关节路点序列，按 `joint_names` 顺序展开 |
| `joint_waypoint_count` | `joint_waypoints` 中包含的路点数量 |
| `waypoint_duration_sec` | 相邻关节路点的时间间隔 |
| `navigation_command_type` | 导航 primitive 类型枚举（v2 新增）；仅 `nav_straight` / `nav_turn` / `nav_abs_coordinate` 使用，v1 请求必须为空 |
| `navigation_target_pose` | 导航目标位姿（v2 新增）；导航 primitive 的目标 `geometry_msgs/Pose`，v1 请求必须为默认值 |
| `navigation_value` | 导航标量参数（v2 新增）；`nav_straight` 的距离或 `nav_turn` 的角度，v1 请求必须为 `0.0` |
| `timeout_sec` | primitive 超时时间，受共享 `task_budget` 剩余预算约束 |

**Result 字段**

| 字段 | 说明 |
| --- | --- |
| `success` | 是否成功 |
| `error_code` | 错误码 |
| `message` | 详细说明 |
| `pose_name` | 实际使用的命名位姿 |
| `actual_registry_epoch` | 实际使用的 registry epoch（成功/失败/abort/cleanup timeout 各路径均携带） |
| `actual_registry_generation` | 实际使用的 registry generation |
| `actual_registry_digest` | 实际使用的 registry digest |

**Feedback 字段**

| 字段 | 说明 |
| --- | --- |
| `state` | 当前执行状态 |
| `detail` | 当前进度说明 |

### `PickObject.action`

抓取闭环接口，由 `manipulation_execution/pick_executor_node` 提供，默认路径
`/manipulation/execute_pick`。通常这是 delegated action：goal 必须携带 `dispatch_binding` 和
`expected_executor`，result 必须返回 `actual_executor`。delegated server 在接受请求前必须比对自身实际
identity，不匹配时 reject，且不得仅凭 endpoint 名判断。`task_id` 移入 `dispatch_binding`，抓取共享同一
root 的绝对 `task_budget`。唯一例外是人工真机 bring-up 的 `supervised_direct=true`：它要求空
`dispatch_nonce`，但仍必须携带从 live Gateway snapshot 取得的 exact registry/executor identity 和有效预算。
Hermes 与 catalog dispatch 必须保持该字段为 `false` 并使用 Gateway 生成的非空 nonce。

**Goal 字段**

| 字段 | 说明 |
| --- | --- |
| `dispatch_binding` | 完整任务/版本信封，携带共享 `task_budget` 与 exact registry identity |
| `target_query` | 运行时视觉文本查询，不是静态 `named_targets` 键 |
| `timeout_sec` | per-entry 超时上限，受 `task_budget` 剩余预算约束 |
| `expected_executor` | `DelegatedExecutorIdentity`，调用方声明的期望 executor identity |
| `supervised_direct` | 仅人工真机测试 client 为 `true`；Hermes/catalog 固定为 `false` |
| `mode` | `MODE_EXECUTE`、`MODE_PLAN_ONLY` 或 `MODE_OBSERVE_ONLY` |
| `release_after_success` | 验证成功并运输到 place container 后是否开爪 |

**Result 字段**

| 字段 | 说明 |
| --- | --- |
| `success` | 是否成功 |
| `error_code` | 错误码（共享预算过期返回 `TASK_TIMEOUT`） |
| `message` | 详细说明 |
| `attempts` | 物理执行尝试次数 |
| `verification_status` | 常量：`VERIFICATION_NOT_RUN=0`、`VERIFICATION_SUCCESS=1`、`VERIFICATION_FAILED=2`、`VERIFICATION_UNCERTAIN=3` |
| `verification_confidence` | 验证置信度 |
| `debug_output_dir` | 调试输出目录；未写文件时为空 |
| `completed_phases` | 已进入的抓取状态机阶段 |
| `actual_executor` | `DelegatedExecutorIdentity`，server 实际 identity |
| `candidate_index` | 最终规划或执行的候选索引，未选择时为 `-1` |
| `released_after_success` | 是否完成正式释放流程 |
| `pipeline_timings_json` | 正式 executor 返回的阶段耗时 JSON |

**Feedback 字段**

| 字段 | 说明 |
| --- | --- |
| `phase` | 当前抓取阶段 |
| `progress` | 进度 `0.0`-`1.0` |
| `attempt` | 当前尝试序号 |
| `detail` | 当前进度说明 |

### `ExecuteAgentPlan.action`

embodied_agent 对 Hermes 自然语言流程公开的高层 action，端点名固定为 `/embodied/execute_agent_plan`。
按 token 读取不可变 plan，重新校验 TTL、exact identity、task id 和 `ConfirmAgentPlan` 冻结的 task budget，
并在 action admission 时重新执行必要的 Gateway / safety 检查。

| 字段 | 说明 |
| --- | --- |
| `schema_version` | schema 版本，v1 固定为 `1` |
| `plan_token` | 不可变 plan 引用 token |
| `confirmation_token` | `ConfirmAgentPlan` 返回的一次性确认 token；首次 accepted 执行必须携带 |
| `task_id` | 绑定到该 token 的 task id |
| `timeout_sec` | 必须精确复用 `ConfirmAgentPlan` 冻结的 float32 task budget |
| `success` / `error_code` / `message` | 结果状态 |
| `plan_id` / `plan_digest` | plan 标识与 digest |
| `workflow_digest` | 单步为空；多步为 typed workflow digest |
| `completed_step_count` | 已完成步骤数 |
| `actual_registry_epoch/generation/digest` | 实际使用的 registry identity |

执行规则：单步使用 direct root `SkillCommand`（`workflow_digest` 为空）；两步及以上计算一个 typed
`workflow_digest` 并执行 Gateway Begin / ordered child `SkillCommand` / Finalize。Hermes 和 `robot-skill`
不接触 root lease nonce。首次 accepted 执行必须携带未消费的 `confirmation_token` 并绑定 task id；相同
token/task id 的重试可幂等返回既有 active/terminal 记录，不同 task id 返回 `SKILL_REQUEST_ID_CONFLICT`。
任一 step 失败、取消、timeout 或 unknown stop 状态后，不再执行下一步，也不自动创建新 token/task id。
`robot-skill cancel-plan` 通过标准 `CancelGoal` 接口取消本 action 的 root goal；CLI 同时要求 task ID、展示过的
plan/registry identity 和 expected step count，用于校验随后返回的终态。

### `ConfirmAgentPlan.srv`

显式确认边界。请求在 plan token/digest、task ID 和 exact registry identity 之外携带
`task_budget_sec`；该值必须为有限正数且不超过 Gateway task budget。成功响应返回
`confirmation_token`、规范化后的 `confirmed_task_budget_sec` 以及冻结的 `task_budget_started_at/deadline`。
执行 action 必须复用该精确绝对预算，确认到执行之间的等待会真实消耗预算。

### `ImitateHumanMotion.action`

HRI 的内部 delegated action，由 `manipulation_execution/imitate_human_motion_executor_node`
提供，默认路径为 `/hri/imitate_human_motion`。它不是 Agent 或 CLI 的公共入口；公共调用必须先进入
`SkillCommand`，再由 `skill_library` 通过现有 delegated Gateway 转发。goal 携带完整
`DispatchBinding`、`expected_executor`、`arm_side`、`imitation_duration_sec` 和独立的 `timeout_sec`。
内部 runtime 按 `prepare -> start -> mock_playback -> reset` 编排，并通过 `PrimitiveCommand` 执行动作和
`move_to_named_pose(home)` 恢复；请求时长超过 20 秒时只执行 20 秒，不循环。

| 字段 | 说明 |
| --- | --- |
| `arm_side` | `left`、`right` 或 `auto`，分别选择可区分的 Mock 动画 |
| `imitation_duration_sec` | 从 start 开始的采集/跟随时长；不等同于 action `timeout_sec` |
| `timeout_sec` | 整个内部 delegated 任务的执行预算，受 `DispatchBinding.task_budget` 约束 |
| `expected_executor` / `actual_executor` | Gateway 与 runtime 双向校验的 executor identity |
| `completed_phases` | 已进入的生命周期阶段；取消、失败或 reset 异常也会返回明确终态 |

### `PlaceObject.action`

已持物释放和视觉确认接口，由 `manipulation_execution/placement_executor_node` 提供，默认路径
`/manipulation/execute_place`。该 action 移动到配置的固定容器位，打开夹爪，将配置的验证关节（当前为 3 号）
移动到验证位进行视觉验证，最后返回固定释放位。

| 字段 | 说明 |
| --- | --- |
| `target_query` | 释放后用于视觉检测的物品名称，必填 |
| `container_query` | 释放后用于视觉检测的指定容器名称，必填；不改变固定释放位 |
| `release_status` | `NOT_RELEASED`、`RELEASED` 或 `UNKNOWN`；不代表已进入容器 |
| `verification_status` | 二维分割包含验证的 `SUCCESS`、`FAILED`、`UNCERTAIN` 或 `NOT_RUN` |
| `place_succeeded` | 仅夹爪确认打开且目标物品连续确认位于容器区域内时为 true |
| `completed_phases` | 包括 `move_to_place`、`release`、`move_to_verify`、`verify_place` 和 `return_to_place` |
| `debug_output_dir` | 当前固定放置证据目录；包含版本化 manifest、开爪 JointState、RGB、检测 mask、判定和最终结果，可由 `placement_replay` 离线重放 |

### `ArmReturnHome.action`

遥操作后端拥有的事务式回零接口。SO101 Placo 默认在
`/so101_placo_servo_node/return_home` 提供该 Action；Goal 使用 `target_name`
选择后端定义的目标，当前支持 `home`。执行结果以新鲜 JointState 的实测关节误差和
连续稳定时间为准，取消、急停、反馈超时或动作超时都会返回明确失败状态。

| 字段 | 说明 |
| --- | --- |
| `target_name` | 后端定义的回零目标名称 |
| `success` / `error_code` / `message` | 终态与可诊断失败原因 |
| `state` | 当前执行阶段反馈 |
| `max_joint_error_rad` | 当前最大关节误差反馈，单位弧度 |

### `ExecuteTaskPlan.action`

MoveIt 任务步骤执行接口，由 `task_executor_node` 提供，路径 `/task_executor/execute_task_plan`。

### `DispatchInfer.action`

推理派发动作接口，兼容 OpenClaw 社交控制链路。

### `RecordEpisode.action`

Episode 录制控制接口，由 `dataset_tools` 的录制服务提供。

### `RunPolicy.action`

策略执行动作接口，用于模型推理控制链路。

---

## 3. 服务定义（srv/）

### `GetSemanticObjects.srv`

查询持久 3D 语义目标。空 `object_ids` 和空 `label` 表示不按对应字段过滤；`query_text` 通过感知服务的
`EncodeText` 编码后，在语义地图边界内与私有 image embedding 比较。`states`、`min_confidence`、
`max_age_sec`、`region_center`/`region_radius_m` 和 `max_results` 提供结构化过滤；`include_inactive` 控制
是否返回 `stale`、`missing`、`lost` 等诊断对象。

### `PlanGrasp.srv`

GraspGen 抓取规划服务，通常由 `grasp_planner_node` 提供，路径 `/grasp_planner/plan_grasp`。

**请求**

| 字段 | 说明 |
| --- | --- |
| `text_prompt` | 目标文本 prompt，传给检测/分割服务 |
| `confidence_threshold` | 检测置信度阈值 |
| `grasp_threshold` | GraspGen discriminator 阈值 |
| `debug_output_mode` | `default`/空字符串沿用节点参数，`none` 不写文件，`diagnostic` 只写 JSON，`full` 写 JSON、PLY 和预览 |

**响应**

| 字段 | 说明 |
| --- | --- |
| `grasps` | 抓取候选数组，候选位姿位于相机坐标系 |
| `object_centroid_xyz` | 已选检测目标的可见表面质心，位于相机坐标系 |
| `object_volume_centroid_xyz` | 已选检测目标的凸包体积质心，位于相机坐标系 |
| `object_volume_m3` | 检测目标的凸包体积；大于 `0` 时体积质心可用于候选排序 |
| `object_point_count` | 检测目标掩码内的有效三维点数 |
| `table_plane_found` | 是否成功拟合执行侧可用的桌面平面 |
| `table_plane_normal` / `table_plane_offset` | 候选坐标系中的桌面方程 `normal·point + offset = 0` |
| `table_plane_inlier_ratio` | 桌面平面内点比例 |
| `object_top_xyz` | 沿桌面法向最高的目标点，用于安全 pregrasp 高度计算 |
| `execution_table_plane_found` | 是否从 completed scene cloud 获得执行侧桌面平面 |
| `execution_table_plane_normal` / `execution_table_plane_offset` | 与历史执行侧采样和 RANSAC 规则一致的桌面方程，目标夹爪 clearance 检查应优先使用 |
| `execution_table_plane_inlier_ratio` | completed-scene 执行桌面平面的内点比例 |
| `inference_time_ms` | 规划耗时，单位毫秒 |
| `success` | 是否成功生成可用候选 |
| `message` | 失败原因或成功摘要 |
| `debug_output_dir` | 本次请求写出的调试目录；未写文件时为空 |
| `diagnostic_details` | 稳定的 `key: value` 诊断行，包含 mask/depth、补全、collision/tabletop 等统计 |

### `VerifyGrasp.srv`

抓取后验证服务，通常由 `grasp_verifier_node` 提供。验证器在服务调用时采样当前传感器，融合夹爪位置、电流和腕部深度可见性证据。

**请求**

| 字段 | 说明 |
| --- | --- |
| `task_id` | 上层任务 ID，仅用于日志和编排关联 |
| `text_prompt` | 目标文本，仅用于日志和编排关联 |
| `grasp` | 规划候选；当前实现不做完整候选级几何验证，但会把 `grasp.target_width_m` 作为宽度 fallback |
| `expected_target_width_m` | planner 侧目标宽度；`0.0` 时回退到 `grasp.target_width_m` |
| `post_grasp_wait_s` | 采样前等待时间；`0.0` 时使用节点默认参数 |

**响应**

| 字段 | 说明 |
| --- | --- |
| `success` | 是否判定抓取成功 |
| `status` | `STATUS_FAILED` / `STATUS_SUCCESS` / `STATUS_UNCERTAIN` |
| `confidence` | 融合证据置信度 |
| `message` | 判定摘要，可能包含 gripper joint 配置提示 |
| `evidence` | 每条传感器证据和打分依据 |

### `ValidateSkill.srv`

只读 preflight 安全校验服务，路径 `/embodied/validate_skill`，由 `safety_guard_node` 提供。校验 exact
snapshot、entry 参数和当前机器人状态，但不查询或修改 coordinator，也不得把存在 `root_lease_nonce`
解释为执行授权。Workflow root、nonce、digest、期望 index 和完整 step payload 的权威校验只在后续
Gateway action admission 发生，因此 validation 通过不保证 admission 成功。`dispatch_binding` 在 planned
TaskCommand/SkillCommand validation 时必须提供完整期望 identity；Workflow child validation 可携带
`root_lease_nonce` 用于关联，但 safety_guard 不得将其视为授权。`dispatch_nonce` 必须为空。
`schema_version` 取值集合为 `{1, 2}`，作为请求首字段；v1 请求只携带 manipulation 字段，v2 请求
允许携带导航字段并必须显式设置为 `2`。

**请求**

| 字段 | 说明 |
| --- | --- |
| `schema_version` | schema 版本，取值 `{1, 2}`；首字段，v2 必须显式设置为 `2`，不得依赖默认零值 |
| `dispatch_binding` | `DispatchBinding`，提供完整期望 registry identity（`root_lease_nonce` 仅用于关联） |
| `skill_name` | 待执行技能名 |
| `target_name` | 命名目标 |
| `container_name` | 指定容器；对放置技能是释放后的视觉检测 query |
| `place_name` | 命名放置位 |
| `motion_direction` | `string`，相对运动方向（v1 manipulation 字段） |
| `motion_distance` | `float32`，相对运动距离（v1 manipulation 字段） |
| `direction` | `string`，导航方向（v2 新增）；v1 请求必须为空 |
| `distance` | `float32`，导航距离（v2 新增）；v1 请求必须为 `0.0` |
| `degree` | `float32`，导航旋转角度（v2 新增）；v1 请求必须为 `0.0` |
| `has_x` | `bool`，是否显式提供 `x`（v2 新增）；区分“未提供”与“显式零” |
| `x` | `float32`，导航目标 x 坐标（v2 新增）；`has_x=false` 时忽略 |
| `has_y` | `bool`，是否显式提供 `y`（v2 新增）；区分“未提供”与“显式零” |
| `y` | `float32`，导航目标 y 坐标（v2 新增）；`has_y=false` 时忽略 |
| `has_yaw` | `bool`，是否显式提供 `yaw`（v2 新增）；区分“未提供”与“显式零” |
| `yaw` | `float32`，导航目标 yaw（v2 新增）；`has_yaw=false` 时忽略 |

**响应**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许执行 |
| `reason` | 不允许时的原因 |
| `error_code` | 稳定错误码（如 `SKILL_SCHEMA_INVALID`、`SKILL_SNAPSHOT_NOT_RETAINED`、`SKILL_REGISTRY_VERSION_MISMATCH`、`SKILL_LIMIT_VIOLATION`） |
| `actual_registry_epoch` | 实际使用的 registry epoch |
| `actual_registry_generation` | 实际使用的 registry generation |
| `actual_registry_digest` | 实际使用的 registry digest |
| `diagnostics` | `SkillDiagnostic[]` 结构化诊断 |

该服务不携带 Gateway 授权、控制模式、lease、配置来源或下游 transport 信息。

### `ValidatePrimitive.srv`

原子动作安全校验服务，路径 `/embodied/validate_primitive`，由 `safety_guard_node` 提供。请求必须携带
exact registry identity 和非空 `dispatch_nonce`；server 在已验证的 exact snapshot 上做白名单 + 工作空间 +
关节限位静态校验。`schema_version` 取值集合为 `{1, 2}`，作为请求首字段；v1 请求只允许引用 v1 primitive
白名单，v2 请求允许引用 `nav_straight` / `nav_turn` / `nav_abs_coordinate` 三个导航 primitive 并必须显式
设置为 `2`。

**请求**

| 字段 | 说明 |
| --- | --- |
| `schema_version` | schema 版本，取值 `{1, 2}`；首字段，v2 必须显式设置为 `2`，不得依赖默认零值 |
| `dispatch_binding` | `DispatchBinding`，必须携带完整期望 registry identity 和非空 `dispatch_nonce` |
| `primitive_name` | 原子动作名；v1 白名单为 10 个 manipulation primitive，v2 额外允许 `nav_straight` / `nav_turn` / `nav_abs_coordinate` |
| `pose_name` | 命名位姿名 |
| `relative_dx/dy/dz` | 相对位移增量 |
| `target_x/y/z` | 执行层解析出的目标末端位置 |
| `target_qx/qy/qz/qw` | 执行层解析出的目标末端姿态四元数 |
| `velocity_scaling` | MoveIt 速度比例 |
| `gripper_position` | 夹爪目标开合量 |
| `joint_names` | 关节名列表，joint primitive 使用 |
| `joint_positions` | 单个关节目标位置 |
| `primitive_duration_sec` | 单点关节轨迹持续时间 |
| `joint_waypoints` | 扁平化关节路点序列 |
| `joint_waypoint_count` | 关节路点数量 |
| `waypoint_duration_sec` | 相邻关节路点的时间间隔 |
| `navigation_command_type` | 导航 primitive 类型枚举（v2 新增）；仅导航 primitive 使用，v1 请求必须为空 |
| `navigation_target_pose` | 导航目标位姿（v2 新增）；导航 primitive 的目标位姿，v1 请求必须为默认值 |
| `navigation_value` | 导航标量参数（v2 新增）；`nav_straight` 的距离或 `nav_turn` 的角度，v1 请求必须为 `0.0` |

**响应**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许执行 |
| `reason` | 不允许时的原因 |
| `error_code` | 稳定错误码（如 `SKILL_SCHEMA_INVALID`、`SKILL_SNAPSHOT_NOT_RETAINED`、`SKILL_REGISTRY_VERSION_MISMATCH`、`SKILL_LIMIT_VIOLATION`） |
| `actual_registry_epoch` | 实际使用的 registry epoch |
| `actual_registry_generation` | 实际使用的 registry generation |
| `actual_registry_digest` | 实际使用的 registry digest |
| `diagnostics` | `SkillDiagnostic[]` 结构化诊断 |

### `BeginWorkflowExecution.srv`

embodied_agent task executor 与 ExecutionCoordinator 之间的内部 execution-scope 服务。**不是**
Agent/CLI/user 可发现的 capability；部署策略必须限制调用方。请求必须满足 `task_id == root_task_id`、
非空 steps、完整期望 identity、非空 `workflow_digest`、`workflow_step_index=0` 且两个 nonce 都为空。
Coordinator 重算 digest、捕获当前 bundle 并原子取得 root lease。相同 root ID/digest/identity/budget 的
重试返回同一 active nonce；任一字段冲突返回 `SKILL_REQUEST_ID_CONFLICT` 且不得替换既有 root。

| 字段 | 说明 |
| --- | --- |
| `dispatch_binding` | `DispatchBinding`，root-scope 信封 |
| `workflow_steps` | `WorkflowStep[]` 有序步骤 |
| `success` / `error_code` / `message` | 结果状态 |
| `root_lease_nonce` | Begin 返回的 root lease nonce，仅受信 task executor 可用于 child dispatch |
| `workflow_digest` | Coordinator 确认的 typed workflow digest |
| `actual_registry_epoch/generation/digest` | 实际使用的 registry identity |

### `FinalizeWorkflowExecution.srv`

embodied_agent task executor 与 ExecutionCoordinator 之间的内部 execution-scope 服务。**不是**
Agent/CLI/user 可发现的 capability。幂等：相同 `(root_task_id, workflow_digest, root_lease_nonce)` 的重复
请求返回既有 terminal result；nonce 或 digest 冲突被拒绝且不得释放另一个 execution。请求必须满足
`task_id == root_task_id`、携带 Begin 返回的 `root_lease_nonce` 和相同 `workflow_digest`/identity/budget、
`workflow_step_index=0` 且 `dispatch_nonce` 为空。Coordinator ledger 的 completed index 是权威的，caller
count 仅用于冲突检测。不同 terminal state 或 count 返回 `SKILL_REQUEST_ID_CONFLICT`；不同 digest 或 nonce
返回对应 workflow digest / lease 错误；任一冲突都不得释放 root lease。

| 字段 | 说明 |
| --- | --- |
| `dispatch_binding` | `DispatchBinding`，root-scope 信封 |
| `terminal_state` | `SUCCEEDED=1` / `FAILED=2` / `CANCELED=3` |
| `completed_step_count` | caller 记录的已完成步骤数，仅用于冲突检测 |
| `success` / `error_code` / `message` | 结果状态 |
| `actual_terminal_state` | Coordinator 权威 terminal state |
| `actual_completed_step_count` | Coordinator 权威 completed step count |

### `GetSkillSnapshot.srv`

exact-version snapshot 查询。这是 runtime 节点之间的内部同步接口，不替代公开 Agent-facing capability
catalog。`schema_version` v1 固定为 `1`。查询语义：

1. `generation == 0` 且空 epoch：返回当前 bundle。
2. `generation == 0` 且非空 epoch：epoch 必须等于当前 epoch，否则 epoch mismatch。
3. `generation > 0`：必须精确匹配 `(epoch, generation)`，不静默升级到更新版本。
4. exact snapshot 已被回收：`SKILL_SNAPSHOT_NOT_RETAINED`。

`snapshot_json` 必须严格使用 `snapshot_payload_v1` 结构，不含 consumer-private 字段。

| 字段 | 说明 |
| --- | --- |
| `schema_version` | 请求 schema 版本，v1 固定为 `1` |
| `registry_epoch` | 期望 epoch |
| `generation` | 期望 generation |
| `success` | 是否成功 |
| `registry_epoch` / `generation` / `registry_digest` | 实际 snapshot identity |
| `capability_digest` / `source_release_digest` / `provenance_digest` | 关联 digest |
| `profile_name` | profile 名 |
| `snapshot_json` | `snapshot_payload_v1` canonical JSON |
| `error_code` / `message` | 失败原因 |

### `ReloadSkillCatalog.srv`

从配置的 source resolver 重新加载 skill catalog。`std_srvs/Trigger` 无法表达结构化结果。生产部署应禁用
该服务或通过 operator policy 限制调用方。该服务始终 reload 配置的 source root，绝不接受调用方提供的
任意路径。`request_id` 用于日志关联和短期幂等；相同 `request_id` 配不同请求字段返回
`SKILL_REQUEST_ID_CONFLICT`。`force=true` 仅强制重编译和重新校验：当 `registry_digest` 和
`provenance_digest` 都未变时，generation 不得人为递增。`schema_version` v1 固定为 `1`。成功的 no-op
返回 `success=true`、`old_generation == generation` 和空 `changed_skills`。

| 字段 | 说明 |
| --- | --- |
| `schema_version` | 请求 schema 版本，v1 固定为 `1` |
| `request_id` | 日志关联/短期幂等 ID |
| `force` | 是否强制重编译和重新校验 |
| `success` / `error_code` / `message` | 结果状态 |
| `registry_epoch` / `old_generation` / `generation` | reload 前后 generation |
| `registry_digest` / `capability_digest` / `source_release_digest` / `provenance_digest` | 关联 digest |
| `changed_skills` | 变更技能列表 |
| `diagnostics` | `SkillDiagnostic[]` 结构化诊断 |

### `MoveToPose.srv`

MoveIt 位姿移动服务，路径由 `moveit_gateway` 提供。

### `MoveToConfiguration.srv`

MoveIt 关节目标移动服务，路径为 `/moveit_gateway/move_to_configuration`。调用方传入已经通过
IK/FK 验证的 `sensor_msgs/JointState`，网关直接规划到同一组机械臂关节角，不会再次求解 IK。
该接口用于保证基于 FK 计算的动态 TCP/接触点补偿在执行时保持有效。

### `RecognizeFile.srv`

语音文件识别服务，由 `voice_asr_service` 提供。

### `SetHotwords.srv`

热词设置服务，由 `voice_asr_service` 提供。

### `SynthesizeSpeech.srv`

跨主机安全的语音合成服务，由 `voice_tts_service` 提供。请求携带文本和可选 WAV prompt 字节，
响应返回一个或多个完整 `SynthesizedAudio` WAV 段、稳定错误码、耗时和 `ModelRuntimeInfo`，不使用
服务端本地输入或输出路径。
请求级 prompt 的可用性由 named deployment 的真实能力决定；不支持时返回 `UNSUPPORTED_PROMPT`，不会
静默忽略 prompt。

### `PlayAudioFile.srv`

播放端本机 WAV 文件服务，由 `voice_tts_service` 提供。请求必须传入播放服务所在机器上的绝对路径；
响应返回 `success`、稳定 `error_code` 和可诊断消息。接口同步等待播放完成，不负责跨主机传输音频文件。

---

## 4. 推理调度接口

调度启用路径使用三个 product-session action；它们与 legacy `DispatchInfer.action` 并存，但不会由同一
launch graph 同时对产品调用方暴露：

| Action | 所有权与用途 |
| --- | --- |
| `OpenInferenceSession` | 公开 Open 只建立逻辑 session/generation；候选首次被 Dispatch 选中时，同一 action 才用于 pipeline 私有 Open/reset 和 generation fence |
| `ScheduledDispatchInfer` | 为本次请求指定 target、fallback、priority 和 deadline，在逻辑 session 内按需绑定 pipeline 并回显完整 identity |
| `CloseInferenceSession` | 停止新 admission，drain 后发布更高 generation；generation 0 只用于 uncertain Open cleanup |

`InferenceOutcome` 表示终态确定性，而不是业务成功：`NOT_STARTED` 保证没有执行副作用，`COMPLETED`
表示结果确定但允许 `success=false`，`UNKNOWN` 表示无法确认下游是否接受或完成。`UNKNOWN` 不允许 fallback
或重试，调用方必须进入 safe-stop，Global/pipeline 必须 quarantine 并通过 Close 或新 boot reconcile。

`InferenceServingStatus` 是 monolithic pipeline 对 Global 发布的 product-session 状态，包含 deployment/runtime-policy
fingerprint、运行时硬件资源、hardware priority levels 和公开容量。它不替代
`InferencePipelineStatus`；后者只负责 distributed edge/cloud transport handshake。

原有 `DistributedInferenceRequest`、`DistributedInferenceResult`、`InferencePipelineStatus` 保持 protocol v2
字段和 topic 不变。分布式推理当前不接入 scheduled product session 或优先级抢占；Open、ScheduledDispatch 和
Close 接口只用于 monolithic pipeline。

## 5. 许可证

Apache-2.0

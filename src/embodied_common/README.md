# embodied_common 架构契约

`embodied_common` 是具身管线的中立共享包，只承载无业务副作用的公共 helper 与默认 fallback 数据。
它用于消除 `embodied_agent`、`perception_service`、`safety_guard`、`skill_library` 之间的重复实现和反向依赖。

## 职责边界

允许放入本包的内容：

- 不启动 ROS 节点、不创建 action/service/client 的纯工具函数。
- 被多个具身包共同使用的轻量数据规整逻辑。
- Primitive canonical descriptor 和仅供兼容测试使用的默认 fallback 数据。
- 不依赖具体业务包内部实现的基础节点 helper。

不应放入本包的内容：

- 依赖感知、规划、安全或执行节点运行状态的业务逻辑。
- 机器人型号、相机 topic、命名位姿、工作空间等应由 `robot_config` 管理的 SSOT 配置。
- `ibrobot_msgs` 已经表达的 ROS msg/srv/action 契约。
- 对 `perception_service`、`safety_guard`、`skill_library`、`embodied_agent` 的反向 import。

## 依赖方向

`embodied_common` 位于具身业务包下方：

```text
embodied_agent
perception_service
safety_guard
skill_library
        ↓
embodied_common
        ↓
ibrobot_msgs / rclpy
```

本包不得依赖上述业务包，避免重新形成层级倒置或菱形耦合。

## 对外 API

当前稳定 API：

- `embodied_common.base_node.BaseTaskNode`
- `embodied_common.json_utils.extract_json_blob`
- `embodied_common.json_utils.load_json_mapping`
- `embodied_common.json_utils.load_json_list`
- `embodied_common.json_utils.string_list`
- `embodied_common.json_utils.parse_confidence`
- `embodied_common.skill_templates.SUPPORTED_PRIMITIVES`
- `embodied_common.skill_templates.DEFAULT_SKILL_TEMPLATES`（legacy fallback，不是运行时 SSOT）
- `embodied_common.skill_templates.DEFAULT_ALLOWED_SKILLS`（legacy fallback，不是运行时 SSOT）
- `embodied_common.skill_templates.DEFAULT_WAYPOINT_DURATION_SEC`
- `embodied_common.skill_templates.is_skill_disabled`
- `embodied_common.skill_templates.get_skill_templates`
- `embodied_common.capability_view.build_capability_view`
- `embodied_common.visual_game_contracts.normalize_visual_game_policies`
- `embodied_common.visual_game_contracts.load_visual_game_policies_json`
- `embodied_common.visual_game_contracts.get_visual_game_handler`
- `embodied_common.visual_game_contracts.get_default_visual_game_handler`
- `embodied_common.visual_game_contracts.get_visual_game_prompt`
- `embodied_common.visual_game_contracts.build_visual_game_capability_view`
- `embodied_common.perception_contracts.validate_result_schema`
- `embodied_common.visual_game_contracts.validate_visual_game_result`
- `embodied_common.visual_game_contracts.get_visual_game_terminal_error`
- `embodied_common.skill_request.canonical_skill_payload`
- `embodied_common.skill_request.skill_payload_hash`
- `embodied_common.skill_request.skill_goal_uuid`
- `embodied_common.skill_request.derive_skill_task_id`
- `embodied_common.perception_contracts.KNOWN_REQUIRED_INPUTS`
- `embodied_common.perception_contracts.SCENE_ANALYSIS_RESULT_FIELD_KINDS`
- `embodied_common.rgbd_snapshot.RGBDSnapshotBuffer.build_snapshot`
- `embodied_common.vlm_api_client.VLMAPIClient.analyze`（原有底层接口，返回 `(str, dict)`，向后兼容）
- `embodied_common.vlm_api_client.VLMAPIClient.complete`（底层扩展接口，返回结构化 dict）
- `embodied_common.vlm_api_client.VLMClient`（高层客户端，一行式多模型调用，自动路由 / 建图 / 上下文）
- `embodied_common.wire_contracts.validate_public_request_wire_contracts`（公开请求 wire contract 校验入口，按 schema version 分流 V1/V2 字段集）
- `embodied_common.wire_contracts.validate_request_schema_version`（公开 request wire 仍仅接受 `{1, 2}`；V3 hybrid context 按技能分别使用 V1 或 V2，与请求体顶层 `schema_version` 和 `dispatch_binding.schema_version` 一致性校验）
- `embodied_common.wire_contracts.PRIMITIVE_CONTRACT_V1`（V1 primitive contract digest 常量）
- `embodied_common.wire_contracts.PRIMITIVE_CONTRACT_V2`（V2 primitive contract digest 常量，含 nav_* primitive）
- `embodied_common.wire_contracts.primitive_contract_for_version(version)`（按 context_schema_version 选择 V1/V2/V3 digest）
- `embodied_common.wire_contracts.PrimitiveContractSet`（持有 V1/V2/V3 digest 与对应 primitive 名称集合，供工具与跨版本比对使用）
- `embodied_common.agent_execution_contract.INTERACTIVE_CONFIRMATION`
- `embodied_common.agent_execution_contract.IMMEDIATE_AFTER_PRESENTATION`
- `embodied_common.agent_execution_contract.validate_agent_execution_mode`
- `embodied_common.workflow_contracts.validate_workflow_steps`：原始 typed workflow 的共享校验入口。
- `embodied_common.tracing.trace_scope` / `trace_stage`：Agent 控制面的有界阶段耗时日志，主 trace ID
  与附加字段共用清洗规则，拒绝常见凭据标记和控制字符；该规则不替代通用密钥检测，不修改业务幂等键。
  边界：结构化 `IBTRACE1` 插桩与分析链路属于 `ibrobot_tracing` 包；本模块只产生 `[hop]` 阶段记录，
  其进程内 ContextVar 为 `embodied_stage_trace_id`，与 `ibrobot_tracing` 的 `ibrobot_trace_id`
  互不共享；跨子系统关联统一走 `DispatchBinding.trace_id` 消息字段。

Agent execution mode is captured when an Agent plan is created and must match
at confirmation. `immediate_after_presentation` only removes the second user
confirmation wait; it never grants motion authorization or bypasses Gateway
admission.

视觉游戏 JSON loader 供 `embodied_agent.visual_game_gateway_node` 解析 ROS 参数；
`get_default_visual_game_handler` 供
`embodied_agent.visual_games` 解析默认 handler；`perception_contracts.validate_result_schema` 供
`perception_service.perception_service_node` 与视觉游戏 Gateway 共享业务中立的结果 schema 校验；
`get_visual_game_terminal_error` 供 `embodied_agent.visual_game_gateway_node` 映射声明的终态错误。

## Capability Gateway 公开视图

`capability_view.build_capability_view()` 只保留为 legacy/config 单元测试兼容接口。生产 Gateway、CLI、Planner
和 Safety 的公开能力必须来自 `skill_catalog` 已验证 exact snapshot，不能从 inline robot YAML 重建第二份视图。
它只包含 `robot_name`、排序后的技能、排序后的命名位姿名称、完整的已解析 timeout policy
和 `capability_digest`，不启动节点，也不读取 ROS 运行状态。

- 每个技能公开 `name`、`summary`、`domain`、`moves_robot`、`required_control_mode`、
  `parameters` 与 `recovery_policy`。字段从 `capability` 深拷贝，调用方修改返回值不会改写配置。
- 共享 catalog 的 `list` 结果只使用 `name`、`summary`、`domain`、`moves_robot` 和
  `required_control_mode`；`describe` 才补充 `parameters`、`recovery_policy`、完整 timeout policy
  及 digest。
- `pose_names` 只公开名称，不公开位姿坐标。primitive sequence、目标绑定、关节/夹爪数值、
  named pose 坐标及 ROS service/action/topic 名称都不属于该视图。
- 缺省、显式空值或全部 `disabled: true` 的 `embodied.skill_templates` 都产生零个公开技能。
  `get_skill_templates()` 的禁用过滤在构建视图前生效。
- digest 是整个公开文档的 SHA-256：JSON 使用排序 key、紧凑分隔符、ASCII、
  `allow_nan=False`，且 `-0.0` 归一化为 `0.0`（由 `embodied_common.canon.to_canonical_json`
  统一实现，`robot_config`、`embodied_common`、`skill_catalog` 共用同一规范）。因此相同的
  公开能力、命名位姿名称或 timeout policy 才会得到相同 digest。

`skill_request` 为 Gateway 请求提供同一套规范化规则：技能名和可选字符串会去首尾空白，
`motion_direction` 会转为小写；`skill_name` 必须非空，`motion_distance`（给出时）必须为有限、
非负且不超过 float32 最大值，最终 `timeout_sec` 必须为有限正数且不超过 float32 最大值（与 ROS action
float32 字段对齐）。`canonical_skill_payload()` 在未给出 timeout 时使用
调用方传入的默认值，`skill_payload_hash()` 对该 payload 使用同样的规范 JSON 计算 SHA-256，
`skill_goal_uuid()` 则对去空白后的 task ID 计算 UUIDv5：
`uuid.NAMESPACE_URL` 下的 `ibrobot:{task_id}`。`derive_skill_task_id(parent_task_id, skill_index)`
派生 `<parent>/skill/<1-based index>` 形式的 child task ID，供 task_executor 与未来消费者共用。
这些 helper 不包含 primitive 或 ROS transport 数据。

V2 schema 之上的 navigation 字段集在 `skill_request` 中也保持同一套规范化规则：

- `navigation_direction` 转为小写并去首尾空白，必须命中允许枚举集合
  （`forward` / `backward` / `leftward` / `rightward` / `turn-left` / `turn-right`）。
- `navigation_distance`（给出时）必须为有限、正且不超过 float32 最大值；零或负数返回
  `SKILL_LIMIT_VIOLATION`。
- `navigation_degree` 必须为有限数，绝对值不超过 float32 最大值；允许负值表示反方向旋转的语义。
- `navigation_x` / `navigation_y` / `navigation_yaw` 与三个 presence flag
  `navigation_has_x` / `navigation_has_y` / `navigation_has_yaw` 配对使用：
  - 每个 `navigation_*` 数值字段都通过 `_optional_coordinate` helper 解析。调用方提供
    `has_*` 为 `True` 时，对应数值字段必须为有限数；提供 `False` 时，对应数值字段被忽略并归一化
    为缺省值（`0.0`），同时从 canonical payload 中剔除，避免 0.0 被误读为“显式 0 坐标”。
  - `has_x` / `has_y` 至少有一个为 `True`，否则视为空目标；`has_yaw` 可以单独为 `False`，
    表示不指定 yaw。
- 所有 V2 navigation 数值字段在下发到 ROS action 前都用 `struct.pack('<f', value)` /
  `struct.unpack('<f', ...)` 做 IEEE-754 binary32 归一化：先把 Python float 截断为 float32 表示，
  再用 `unpack` 恢复为 float。该步骤与 ROS msg 的 `float32` 字段对齐，确保
  `canonical_skill_payload()` 计算的 SHA-256 与下游节点接收到的 float32 字节级一致；
  未做归一化的 NaN / Inf / -0.0 / 双精度扩展位都会被 `validate_public_request_wire_contracts`
  以 `SKILL_SCHEMA_INVALID` 拒绝。
- `schema_version` 必须落在 `validate_request_schema_version` 接受集合 `{1, 2}` 内，且与
  `dispatch_binding.schema_version` 一致；V1 请求携带任何 navigation_* 字段返回
  `SKILL_SCHEMA_INVALID`，V2 请求缺失必填 navigation_* 字段（按对应 primitive 而定）同样拒绝。

视觉游戏 handler definition 同样保持 ROS 无关。部署策略只包含 `enabled`、`announce`、`handler`
与公开 `summary`；共享 registry 是 required inputs、结果 schema 和运行时 prompt 的唯一声明源。
公共 capability 只投影契约字段，不暴露 prompt；`embodied_agent` 通过 accessor 读取同一 definition 的 prompt，
不再维护第二份 handler/prompt 映射。
视觉游戏只通过 Agent 的 `robot-skill` 控制面触发，不声明或解析 ASR aliases。

## 输入前置条件（required_inputs）

`build_snapshot(required_inputs=...)` 提供通用的输入门控，供上层按每条请求声明"哪些输入缺失才阻塞"：

- 允许键为 `KNOWN_REQUIRED_INPUTS`（`primary_image` / `ee_pose` / `joint_state`）。
- 传入 `None`（默认）、缺失、或畸形值（非列表、空列表、列表内含非字符串项）时，回退到严格默认——
  primary_image + ee_pose + joint_state 全部要求在线。
- 传入合法子集时只门控该子集，未知/畸形键被安全忽略（不抛异常），使纯视觉请求可在 EE pose /
  joint state 离线时成功。

该词表业务中立，不含任何具体游戏/任务特判。

## VLMClient 使用说明

调用方一行创建实例即可随时调用云端模型，无需手动构造 messages 或指定 provider / key：

```python
from embodied_common.vlm_api_client import VLMClient

vlm = VLMClient()                              # 加载 vlm_models.yaml，一次初始化
vlm = VLMClient(system="你是一只小鸭子机器人")   # 预设 system prompt，每轮请求自动携带

vlm.chat("做任务规划")                          # 默认模型（defaults.model）
vlm.chat("分析图片", image=frame_bytes)         # 传入图片字节，自动构造多模态 message
vlm.chat("文本任务", model="my-llm")            # 按模型名路由到 vlm_models.yaml 中的条目
vlm.chat("快速回答", enable_thinking=False)     # 关闭思维链（支持的 provider 生效）
vlm.chat("新话题", clear_history=True)          # 清空上下文后再对话
```

同一个 `VLMClient` 实例的多次 `chat()` 自动累积历史（多轮记忆）；需要全新会话时新建实例。

构造时传入的 `system`（默认 `None`）是持久化的 system prompt：它被置于每轮请求最前，但**不写入对话历史**——因此对话变长不会把它挤出上下文，`clear_history=True` / `clear_history()` 也不会清掉它。

`chat()` / `complete()` 返回结构化 dict：

| 字段 | 含义 |
|------|------|
| `status` | `"ok"` / `"error"` |
| `model` | 实际调用的模型名 |
| `content` | 文本回复 |
| `reasoning` | 思维链内容（无则为空串） |
| `tool_calls` | function call 列表 |
| `usage` | token 用量 |
| `timing_ms` | 端到端耗时（毫秒） |
| `error` | 出错描述，成功时为 `None` |

### 模型路由配置（vlm_models.yaml）

位于 `embodied_common/vlm_models.yaml`。新增模型只需追加几行，`api_key_env` 只存环境变量名，
`multimodal: false` 时自动过滤图片。API key 通过环境变量注入，不写入任何配置文件：

```bash
export ALIYUN_API_KEY=sk-xxxxxx
export MY_LLM_API_KEY=sk-xxxxxx
```

`get_skill_templates` 会先过滤显式设置 `disabled: true` 的模板，再做深拷贝并就地展开
`trajectory_template` 为 `joint_waypoints`（调用 `expand_trajectory_template`）。只有字面量布尔值
`True` 表示禁用；`robot_config.loader` 会拒绝非布尔 `disabled`。因此 loader、规则入口、resolver、
safety guard 和 `robot-skill` catalog 共用同一份启用技能与模板展开语义。

## 与 SSOT 的关系

`DEFAULT_SKILL_TEMPLATES` 是最小闭环的默认 fallback，不是最终机器人级 SSOT。
机器人级 skill 名不在本包维护全局常量，而是从当前 `robot_config` YAML 的
`embodied.skill_templates` 键集合派生。

当前 `robot_config` 已提供 `embodied` 配置段，机器人级 skill templates、命名位姿、
命名目标、workspace 边界和感知相机 topic 均由 `robot_config` YAML 作为单一事实来源
管理。运行时由 `embodied_bringup` 从该 YAML 读取并注入到下游具身节点参数：

- skill templates
- named poses
- named targets
- workspace limits
- perception camera topics

`embodied_common` 只提供安全可运行的默认值，保证独立节点调试时不会因为缺少 launch
注入而完全不可用；一旦通过 `embodied_bringup` 启动，YAML 注入值会覆盖默认值。

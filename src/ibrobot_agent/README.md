# ibrobot_agent 架构契约

`ibrobot_agent` 是 IB-Robot 自然语言 Agent 的孵化运行时包。它把用户的自然语言请求转成
严格校验的 typed workflow，并把全部机器人执行委托给既有 Capability Gateway 链路
（Agent ExecutionPort → `agent_plan_node` → Safety Guard → Skill Executor）。
本包不拥有 Skill catalog、运动授权或物理执行权，也不实现感知。

本包与 `embodied_agent` 互补而非重复：`embodied_agent` 负责 Agent plan 的
plan/validate/confirm/execute 生命周期编排；`ibrobot_agent` 在其上层负责自然语言理解、
Planner 适配、请求状态机与持久化 ledger。Planner 是本包进程内的库适配器
（rule / vlm 两种模式），不是独立 ROS 节点，也不恢复历史上已删除的
`vlm_task_planner` 包路线。

## 当前 ROS 节点

| 节点 / 入口 | 主要职责 |
| --- | --- |
| `ibrobot_agent_node` | Agent 组合根：transport 与生命周期；ROS-free 的 `AgentService` 通过注入的 port 工作 |
| `chat_tui.py`（`ros2 run ibrobot_agent ibrobot_agent_chat` / `scripts/run_agent_chat.sh`） | 交互式终端 UI：输入历史、异步事件渲染、`/stop` 与停止词旁路；`--channel-id`/`--principal-id`/`--robot-scope` 覆盖会话身份以匹配 profile 的 `embodied.agent` 配置 |

## 调用链

```text
用户 / chat_tui
  -> /agent/request (std_msgs/String, JSON)
  -> ibrobot_agent_node
     -> Planner（rule / vlm，严格 JSON 契约）
     -> Agent ExecutionPort 复用 robot_skill_cli.RosBridge
        -> /embodied/prepare_agent_plan（capture + read-only validate）
        -> /embodied/confirm_agent_plan -> /embodied/execute_agent_plan
        -> embodied_agent.agent_plan_node -> Safety Guard -> Skill Executor
  <- /agent/response（同步应答） /agent/event（有序事件）
   -> /agent/control（request 粒度 stop / exact plan 展示完成回执）
```

## 请求状态机

```text
RECEIVED -> PLANNING -> PROPOSAL_READY -> PREPARING -> MAY_EXECUTE -> RUNNING
    -> SUCCEEDED / FAILED / CANCELLED / CANCELLED_BEFORE_EXECUTION
非运动应答: PLANNING -> ANSWERED
终态未知:   -> UNKNOWN（触发 robot-scope quarantine）
```

- 单 robot-scope 同一时间只接受一个活动请求，重复 request_id 幂等去重；BUSY 是终态，重试必须生成新的 request_id。
- stop 以 generation 防竞态；执行前后取消分别落到 `CANCELLED_BEFORE_EXECUTION` / `CANCELLED`。
- 请求、事件序列与会话记忆持久化在 SQLite（WAL + FULL sync）ledger 中。

## Topic / 服务契约（孵化期临时契约）

`/agent/request`、`/agent/response`、`/agent/event`、`/agent/control` 当前使用
`std_msgs/String` 承载 JSON。入站请求与控制消息由节点做构造器级 schema 校验
（`schema_version`、字段类型与非空检查）；重复 key、NaN/Infinity、围栏文本、
tool call 的严格拒绝只作用于 Planner 响应（`ibrobot_agent/planner.py`）。这是
**孵化期的有意决策**：契约稳定前不把不成熟接口固化进 `ibrobot_msgs`。计划在孵化期
结束后迁移为 `ibrobot_msgs` 的类型化消息；在此之前，外部集成方不应把这些 topic
当作稳定 API。Gateway 服务侧调用保持既有类型化契约不变。

节点另提供 `~/health` 与 `~/ready`（`std_srvs/Trigger`）。

`/agent/request` 的 JSON 在解析前限制为 64 KiB UTF-8 字节；`AgentRequest.text` 去除首尾空白后最多
4096 个字符（中文字符按一个字符计数）。超限返回 `REQUEST_SCHEMA_INVALID`，不会截断指令或进入规划队列。
限制由 `ibrobot_agent.contracts` 统一定义。状态、技能列表与姿态列表查询复用 Planner 层的
确定性只读解析，否定或假设请求不走该快捷路径；普通只读问句直接读取 Gateway 事实，不依赖模型调用。

## 安全设计

- 运动默认关闭：`execution_enabled=false` 时零 goal 提交，仅规划与应答。
- 执行必须同时满足：非空 `test_allowlist`、Gateway readiness（live catalog 身份一致）、
  操作员 `authorize_motion` launch 参数。
- 内置 RulePlanner 只允许在仿真模式执行；真机执行必须配置 vlm Planner。
- Planner 输出只接受单一严格 JSON 对象；未知字段、未知 skill、编造参数一律拒绝。
- 终态未知（UNKNOWN）进入 robot-scope quarantine，需人工介入。
- 启动时在接受请求前执行单事务恢复：未完成且可能已提交的请求转为持久化 UNKNOWN，保留 TaskRef
  供人工通过 Gateway 核实；确定未提交的中断请求转为 FAILED 或 CANCELLED_BEFORE_EXECUTION。
  不重放旧请求，不以 Gateway 空闲推断成功；已有终态和其他 robot scope 的记录保持原样。
- 部署锁（`deployment_lock_path`）防止双 Agent 实例并存。
- 提交前持久化回调异常由 `AgentService` 根据 ledger 判定：未提交时失败；可能已提交时
  UNKNOWN/quarantine；无法读取或写入 ledger 时保持隔离，不由 transport adapter 猜测终态。
- Planner 凭据统一按 Kimi 模式提供：`base_url` + `api_key_env`（环境变量名）；
  `embodied.agent.planner.api_key` 字面密钥会被 robot_config 校验直接拒绝，
  也不会进入 `config_digest` 的 preimage。

## 依赖方向

```text
ibrobot_agent
    -> robot_skill_cli   # RosBridge / catalog view（复用，不复制）
    -> embodied_common   # canon / workflow_contracts / VLMAPIClient
    -> robot_config      # SSOT 配置与 launch 校验（经 embodied_bringup 注入参数）
    -> rclpy / std_msgs / std_srvs
```

本包不得：直接发布运动命令或调用 controller/MoveIt、绕过 `skill_library` 与
`safety_guard`、维护机器人本体配置、感知模型或 Skill catalog。

## 配置入口

全部配置来自 `robot_config` SSOT YAML 的 `embodied.agent` 块（字段表与校验规则见
`robot_config` README 的具身 AI 流水线一节），由 `embodied_bringup` 在
`embodied.entry_mode: agent` 时注入为节点参数。仓库自带 SO-101 孵化 profile：
`so101_agent_manual`（真机手动无运动）、`so101_single_arm_agent_test`（rule Planner 测试）、
`so101_single_arm_agent_gazebo`（Gazebo 执行）、`so101_single_arm_agent_hardware*`（真机执行/停止）。

## 展示完成协议

`immediate_after_presentation` 仍要求完整 exact plan 展示并 flush 后才 technical confirm。
`presentation` 事件的 `detail` 包含完整 `presentation`（task/plan/registry tuple、有序步骤、预算和摘要）、
canonical `presentation_digest` 与一次性 `receipt_token`。TUI 在自己的事件循环中用底层 terminal output
打印并 flush 整个计划，再发送：

```json
{"operation":"presentation_rendered","request_id":"...","request_key":{"robot_scope":"...","channel_id":"...","principal_id":"...","request_id":"..."},"presentation_digest":"...","receipt_token":"..."}
```

服务端仅接受当前待展示计划的精确身份、摘要、token 和有效期；回执不代表用户二次批准，不授予运动权限。
`embodied.agent.presentation_timeout_sec` 默认 30 秒，必须为有限正数；无客户端、输出失败、事件传输失败、
超时、错误回执或停止请求均不能进入 confirm/goal 发送。旧的只订阅事件而不返回展示回执的客户端必须升级。
`/agent/control` 是受信本地客户端边界，回执是协议证据而非对任意恶意 ROS peer 的认证。

## 已知限制

- 孵化期包（`incubation: true` 强制标记），String JSON topic 契约尚未类型化。
- 会话记忆为有界滚动窗口，不做长期记忆检索。
- `chat_tui` 为单操作员本地终端，不提供多通道并发接入。
- UNKNOWN 恢复采用保守隔离，不自动重放或推断旧任务终态；需操作员依据保留的 TaskRef 完成核实。

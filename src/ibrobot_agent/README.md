# ibrobot_agent 架构契约

`ibrobot_agent` 是 IB-Robot 自然语言 Agent 的孵化运行时包。它把用户的自然语言请求转成
严格校验的 typed workflow，并把全部机器人执行委托给既有 Capability Gateway 链路
（`InteractiveController` → `agent_plan_node` → Safety Guard → Skill Executor）。
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
| `chat_tui.py`（`scripts/run_agent_chat.sh`） | 交互式终端 UI：输入历史、异步事件渲染、`/stop` 与停止词旁路 |

## 调用链

```text
用户 / chat_tui
  -> /agent/request (std_msgs/String, JSON)
  -> ibrobot_agent_node
     -> Planner（rule / vlm，严格 JSON 契约）
     -> 复用 robot_skill_cli.InteractiveController
        -> /embodied/plan_agent_command -> /embodied/validate_agent_plan
        -> /embodied/confirm_agent_plan -> /embodied/execute_agent_plan
        -> embodied_agent.agent_plan_node -> Safety Guard -> Skill Executor
  <- /agent/response（同步应答） /agent/event（有序事件）
  -> /agent/control（request 粒度 stop）
```

## 请求状态机

```text
RECEIVED -> PLANNING -> PROPOSAL_READY -> PREPARING -> MAY_EXECUTE -> RUNNING
    -> SUCCEEDED / FAILED / CANCELLED / CANCELLED_BEFORE_EXECUTION
非运动应答: PLANNING -> ANSWERED
终态未知:   -> UNKNOWN（触发 robot-scope quarantine）
```

- 单 robot-scope 同一时间只接受一个活动请求，重复 request_id 幂等去重。
- stop 以 generation 防竞态；执行前后取消分别落到 `CANCELLED_BEFORE_EXECUTION` / `CANCELLED`。
- 请求、事件序列与会话记忆持久化在 SQLite（WAL + FULL sync）ledger 中。

## Topic / 服务契约（孵化期临时契约）

`/agent/request`、`/agent/response`、`/agent/event`、`/agent/control` 当前使用
`std_msgs/String` 承载 JSON，由节点本地严格解析（重复 key、NaN/Infinity、围栏文本、
tool call 均拒绝）。这是**孵化期的有意决策**：契约稳定前不把不成熟接口固化进
`ibrobot_msgs`。计划在孵化期结束后迁移为 `ibrobot_msgs` 的类型化消息；在此之前，
外部集成方不应把这些 topic 当作稳定 API。Gateway 服务侧调用保持既有类型化契约不变。

节点另提供 `~/health` 与 `~/ready`（`std_srvs/Trigger`）。

## 安全设计

- 运动默认关闭：`execution_enabled=false` 时零 goal 提交，仅规划与应答。
- 执行必须同时满足：非空 `test_allowlist`、Gateway readiness（live catalog 身份一致）、
  操作员 `authorize_motion` launch 参数。
- 内置 RulePlanner 只允许在仿真模式执行；真机执行必须配置 vlm Planner。
- Planner 输出只接受单一严格 JSON 对象；未知字段、未知 skill、编造参数一律拒绝。
- 终态未知（UNKNOWN）进入 robot-scope quarantine，需人工介入。
- 部署锁（`deployment_lock_path`）防止双 Agent 实例并存。
- Planner 凭据统一按 Kimi 模式提供：`base_url` + `api_key_env`（环境变量名）；
  `embodied.agent.planner.api_key` 字面密钥会被 robot_config 校验直接拒绝，
  也不会进入 `config_digest` 的 preimage。

## 依赖方向

```text
ibrobot_agent
    -> robot_skill_cli   # InteractiveController / RosBridge / catalog view（复用，不复制）
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

## 已知限制

- 孵化期包（`incubation: true` 强制标记），String JSON topic 契约尚未类型化。
- 会话记忆为有界滚动窗口，不做长期记忆检索。
- `chat_tui` 为单操作员本地终端，不提供多通道并发接入。

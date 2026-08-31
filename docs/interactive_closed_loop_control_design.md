# 交互式闭环控制设计（展示后立即执行 / 别动 / 安全继续）

## 1. 目标

在同一个 Hermes 进程内提供以下闭环能力：

1. 只读查询当前 runtime Skill catalog；
2. 在规划前拒绝 catalog 外或不可见 Skill；
3. 创建并展示 exact Workflow，flush 后按显式 `execution_mode` 执行；`immediate_after_presentation` 不等待用户二次确认；
4. 在 validation、内部 confirm、goal 发送、goal acceptance 和执行期间可靠响应「别动」；
5. 只有确定取消后才允许独立的新 continuation 请求；当前基线不提供断点 resume。

实现必须继续使用现有 Capability Gateway、`validate-plan`、`confirm-plan`、`execute-plan`、
`cancel-plan` 和 `authorize_motion` 边界。`confirm-plan` 是 exact plan/task tuple 的内部技术绑定，
不是用户确认门，也不替代操作员的 motion authorization。执行模式由 `PlanAgentCommand` 捕获并绑定到
`AgentPlan`，`ConfirmAgentPlan` 必须精确匹配；历史分阶段入口默认使用 `interactive_confirmation`，
`run-workflow` 显式使用 `immediate_after_presentation`。

## 2. 状态机

```text
IDLE -> DISCOVERED -> PREPARED -> CONFIRMED -> EXECUTING
                       |            |             |
                       +------------+-------------+---- stop intent
                                                    |
                                                 STOPPING
                                                    |
                  +----------------+----------------+----------------+
                  |                |                                 |
             STOPPED          SUCCEEDED / FAILED                  UNKNOWN
       canceled + proof       authoritative result       missing/mismatched proof
                  |
        fresh user request only
```

Workflow 展示必须发生在 `confirm_agent_plan` 和 `send_agent_plan_goal` 之前。展示输出 flush 后，
控制器按 `validate_agent_plan -> confirm_agent_plan -> execute_agent_plan` 连续推进。

## 3. 停止不变量

- `_stop_requested` 是单调、跨阶段的 operation-scoped latch。`request_stop()` 只置位，不直接调用 ROS。
- latch 在内部 confirm 前后、action server 等待、goal 发送前、acceptance 等待和 result 等待中检查。
- 若 fresh in-process task 可证明从未提交，goal 发送前停止可记录本地 `STOPPED`，不发送 goal/cancel。
- 独立 `execute-plan` 进程不能仅凭 caller tuple 排除同 task/token 的幂等 retry；pre-send stop 必须抑制新 goal，
  并对可能已存在的 deterministic goal 执行 cancel + convergence，不能合成本地终态。
- 新 operation 的状态检查与 stop latch 初始化必须在同一锁临界区完成；任何 stop 线程不得等待 ROS 调用。
- goal 可能已发送后，只有执行线程调用一次 `cancel_agent_plan()`；外部线程不得与其并发取消。
- goal submission、acceptance、result request 或 transport 异常均进入 cancel + terminal convergence。
- cancellation accepted 不是机器人已停止；不能在未读取权威终态时报告 `STOPPED`。

## 4. 权威终态

控制器同时校验 Action GoalStatus、result payload 和计划身份：

| 控制器状态 | 必须同时满足 |
|---|---|
| `SUCCEEDED` | status=4、`success=true`、`error_code` 为空、全部步骤完成、plan ID/digest 和 registry identity 精确匹配 |
| `STOPPED` | status=5、`success=false`、`error_code=SKILL_CANCELLED`、plan ID/digest 和 registry identity 精确匹配 |
| `FAILED` | status=6、`success=false`、稳定失败码存在、身份精确匹配，且错误不表示执行/清理状态未知 |
| `UNKNOWN` | status/result 不一致、字段缺失、身份不匹配、`SKILL_CANCEL_TIMEOUT`、`GATEWAY_FINALIZATION_FAILED`、`SKILL_EXECUTION_BUSY` 或 convergence 超时 |

`UNKNOWN` 必须 fail closed，不得通过新 task ID 继续运动。若 Action 返回的是共享 contract 中已知的
uncertain-motion code，CLI/控制器保留该原始 code 并以 unknown 状态/退出码 15 报告；身份或结构证明无效时才
合成为 `SKILL_CANCEL_TIMEOUT`。

## 5. 继续语义

当前 `ExecuteAgentPlan.action` 只返回 `completed_step_count` 遥测，没有 prior-task identity、停止证明或
server-owned continuation admission。客户端直接执行
`prior_steps[completed_step_count:]` 会把遥测误当授权，并可通过新 task ID 绕过 Gateway。

因此当前契约为：

- `continue_workflow(..., resume=True)` 返回 `SKILL_CONTINUATION_UNAVAILABLE`，不规划、不确认、不发 goal；
- 只有 `GoalStatus=5 + SKILL_CANCELLED + exact identity` 的确定取消允许用户提出独立的新 Workflow；
- 新 Workflow 重新 discover、重新校验 catalog，并使用新 request/task/plan token；
- `SUCCEEDED`、`FAILED` 和 `UNKNOWN` 均不授权 continuation。

未来只有在 ROS 层提供 continuation planning/admission 和可验证停止证明后，才可实现“跳过完成步骤、
中断步骤从头执行”的断点语义。

## 6. CLI 信号与输出

`execute-plan` 对 SIGINT/SIGTERM 只在 signal handler 中置位 event。主执行线程立即进入
`cancel_agent_plan + get_agent_plan_result` convergence，并在 `finally` 中恢复原 handler。

`execute-plan` 和外部 `cancel-plan` 必须携带展示过的 plan ID/digest、registry epoch/generation/digest 和 expected
step count。CLI 只接受与该 tuple 精确匹配的 Action 终态；字段非空本身不是证明。

- goal acceptance wait 可被 event 打断；
- post-submission transport 异常最终输出恰好一条 JSONL `result`；
- exit code 由权威 Action status + result 决定，成功 result 不得与 130/143 并存；
- Agent plan 的未知停止统一退出码 15，不使用 direct execute 的 124；
- terminal result 输出后抑制 late feedback。

## 7. 公开入口与测试

`robot-skill-closed-loop` 是 live Gateway 验证入口。它按以下顺序输出：

1. `discover`
2. `prepare`（已 flush）
3. `auto_confirm`（内部 admission）
4. feedback 和唯一 `execute_terminal`

`--stop-after-sec` 模拟「别动」。`--resume` 用于验证当前 fail-closed 行为；确定取消后可使用
`--continue-skill` 提交独立的新 Workflow。

纯 Python FakeBridge 测试覆盖 catalog rejection、展示顺序、stop latch、单一取消、严格终态、
resume 拒绝、signal handler 恢复、goal acceptance 信号竞态和成功终态/退出码一致性。

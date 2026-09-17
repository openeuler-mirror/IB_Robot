# PR #431 启动恢复与展示屏障验证

日期：2026-09-18。工作区：`IB_Robot_agent_skill_refactor`，分支
`feature/ibrobot-agent-skill-refactor`，基线提交 `4700f487` 加本轮未提交修改。

## 修复范围

- Agent 在启动接受请求前，以单个 SQLite 事务收敛当前 robot scope 的中断记录。
  可能已提交的记录持久化为 UNKNOWN，保留 TaskRef 并隔离；确定未提交的记录进入失败或
  执行前取消终态。旧请求不会自动重放，其他机器人和已知终态不受影响。
- immediate 模式增加 exact-plan 展示回执。服务端绑定 RequestKey、完整 Presentation 摘要、
  一次性 token 和期限。客户端完整打印并 flush 后发送回执，随后才允许 technical confirm。
  回执不授予运动权限；Gateway 的授权和安全准入继续生效。
- TUI 渲染迁移到 prompt-toolkit 事件循环，使用底层 output.flush，不把仅入队的
  patch_stdout.flush 当作展示完成。旧硬件/仿真 probe 同步为新回执协议。
- 统一 ibrobot_msgs、Agent、plan node 和 robot_config 的文档与参数说明。

## 离线验证

在该 worktree 独立 venv 内，通过 `.shrc_local` 加载环境。测试使用隔离 domain 86、
`IBROBOT_TEST_ROS_DOMAIN_ID=86`、`ROS_LOCALHOST_ONLY=1`。

验证包括：

- RUNNING/STOPPING 的重启隔离、提交前各状态收敛、重复重启幂等、其他 robot scope 隔离、
  已有终态保留、恢复事务回滚和存储故障时启动失败。
- 无客户端、错身份/摘要/token、超时、停止优先、发布异常和输出异常均不放行动作。
- 实际 ROS 请求/事件/控制消息往返：错误回执没有 confirm/submit，正确回执只放行一次。
- 终端渲染顺序为完整计划写入、底层 flush、回执；不是在事件订阅回调中提前确认。
- 配置超时必须为有限正数，launch 按 SSOT 参数注入。

相关构建通过：`ibrobot_msgs embodied_common ibrobot_agent robot_config embodied_bringup
embodied_agent robot_skill_cli skill_library safety_guard`，统一通过 `scripts/build.sh`。

最终针对性回归：238 passed、2 deselected；新增 presentation timeout 和 Agent launch 注入：
8 passed。修改文件及新增 Python 文件的 Ruff 0.11.8 check / format 检查通过。

完整配置/launch 测试的额外运行曾出现 5 项失败：3 项缺少视觉模型 bundle（其中 hybrid
配置另有 camera parent_frame 校验失败），另 2 项涉及现有 Agent test profile observations
及 placement 配置断言。没有将这些用例计为通过，也没有为本次修复修改它们的预期。
后两项已用 `4700f487` 的原始 loader 重新运行并得到相同失败，确认不是本轮新增回归。
此前 marker outputs replay 的两项测试依赖本地模型/回放资源，本轮同样未计入通过。

## 本机真机验证

操作员已确认 SO-101 已连接、工作范围清空、急停可用，并授权使用现有 Gateway 测试。
本轮使用真实串口 `/dev/ttyACM0`、`use_sim=false`、ROS domain 52、`moveit_planning`、
`authorize_motion=true`、`moveit_display=true`；Planner 使用配置的 Kimi API。
相机和视觉感知关闭，测试 allowlist 仅有 `nod_yes` 和 `recover_safe_pose`。

测试 overlay 基于同目录的 `so101_agent_manual_hardware.yaml`，将 ledger、会话和锁放到
独立测试目录，展示超时设为 60 秒。原有运行 ledger 没有被清空或修改。
控制器、MoveIt Gateway、Safety Guard、Skill Executor、plan node 和 Agent ready 后才发送请求。

### 1. 展示屏障与点头

请求：`pr431-hardware-nod-20260918-02`。

- Kimi 返回仅含 `nod_yes` 的单步 typed plan。
- 延迟回执期间：请求为 MAY_EXECUTE，`may_have_submitted=0`，ledger 无 record_confirmation
  或 mark_submitted，确认没有 goal 提交。
- 完整计划展示并 flush 后发送 exact receipt。
- 终态 SUCCEEDED，`completed_step_count=1`，plan/registry identity 一致。
- Gateway 技能执行约 7.28 秒。
- Task：`a75f1beb69824b759fd906fdb7b35af8`。
- Plan：`720395e9-c617-4d06-9c6b-517811444da0`。
- Plan digest：`53de7626bf07d7254da195894c351b3d17a504dd5b0002a4d89805d466e66263`。

### 2. 回安全位

请求：`pr431-hardware-home-20260918-01`。

- 独立展示 `recover_safe_pose` exact plan，回执前同样保持零提交。
- 回执后执行成功，终态 SUCCEEDED，`completed_step_count=1`。
- Gateway 技能执行约 1.83 秒；MoveIt 和硬件控制器返回成功。
- Task：`c35f47f0e67c415fae89422760603105`。
- Plan：`df8e3fb8-9a47-49a8-96f8-1b1a1faed1a4`。
- Plan digest：`b411dcee28306f4737a56021b61cac7b0e1fac4abcbb5aa1a072bae50dd48f14`。

两次成功请求的 ledger 顺序均为：

```text
presentation -> presentation_rendered -> record_confirmation -> mark_submitted -> finish
```

共同 registry epoch：`02cc9642-e74b-4af2-8c9e-ba6092016d05`，generation：1，digest：
`9c89f0501ae7830b500af8097ebba80b0768a830d9bba8f0f2efbde8133992c4`。

### 3. 真实节点启动恢复检查（无运动）

同一 ROS 环境内启动独立命名、独立 topic、独立 ledger 的 Agent 节点；fixture 模拟旧请求
RUNNING 且 may_have_submitted=True，不向硬件发送任何对应任务。
实际结果：旧记录变为 UNKNOWN，TaskRef 完整保留，ready=false，新请求返回
`accepted=false / ROBOT_QUARANTINED`。这验证真实节点的启动/准入边界，并非在机械臂运动中
强制杀进程的测试。

### 4. 异常与关停记录

- 首次临时 overlay 用绝对 base_config，被 sibling-only 校验拒绝，尚未启动硬件；调整到
  正确配置目录后启动成功，未弱化 loader 校验。
- 首次辅助 probe 混用了独立 Context 和默认 executor，出现 `AttributeError: __enter__`。
  该请求最终为 CANCELLED_BEFORE_EXECUTION、may_have_submitted=0。修复辅助程序并取得
  操作员对新测试请求的授权后才进行上述点头验证，没有重放未知运动。
- RViz 出现 `so101_description` 网格资源缺失；无视觉输入时还出现 octomap/recognize_objects
  不可用日志。运动链路的 readiness 与权威成功终态均单独检查。
- SIGINT 关停时部分既有节点重复调用 rcl_shutdown，MoveIt 在析构阶段发生崩溃。
  硬件明确完成 deactivate/shutdown，启动的进程已全部退出，串口不再占用。
  因此本次结论是功能测试通过，不是整段系统日志无异常。

真机请求由协议 probe 复用 TUI 的 exact-plan 打印/回执实现，经过 Kimi、实际 Agent 节点和 Gateway；
完整交互式 TUI 的事件循环与底层 flush 顺序另由无运动回归覆盖。本轮没有在机械臂运动中进行进程崩溃注入。

本地原始证据位于 `/tmp/opencode/pr431-hardware-evidence/`，包括 launch.log、每个请求的
presentation/blocked/result JSON、SQLite ledger、recovery-probe/result.json 和测试 overlay 副本。
临时 overlay 已移出生产配置目录。本文不包含凭据，也不将本轮结果当作双平台 Docker 验证。

## 现场观察复验

同日按操作员请求再次运行。首个请求的展示交互超过 60 秒窗口，返回
PRESENTATION_FAILED，may_have_submitted=0，未提交运动。操作员重新授权后，以新请求和
独立测试 ledger 运行，测试用展示窗口设为 180 秒，生产默认值仍为 30 秒。

- `pr431-observe-nod-20260918-02`：SUCCEEDED，完成 1 步，技能执行约 7.39 秒。
- `pr431-observe-home-20260918-01`：SUCCEEDED，完成 1 步，技能执行约 1.72 秒。
- 两次请求都确认了回执前零提交，以及展示完成、确认、提交、终态的持久化顺序。
- 关停前 Gateway 空闲；硬件完成 deactivate/shutdown，进程全部退出、串口释放。
  关停仍复现重复 rcl_shutdown 和 MoveIt/RViz 析构异常，未计为无异常关停。

成功复验原始记录：`/tmp/opencode/pr431-hardware-observe-03/`。
此前超时拒绝记录：`/tmp/opencode/pr431-hardware-observe-02/`。

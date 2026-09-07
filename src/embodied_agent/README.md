# embodied_agent 架构契约

`embodied_agent` 是 Hermes-only 具身运行时中的 Agent plan 生命周期与 Workflow 编排包。
它不拥有 Skill catalog、运动授权或物理执行权。

## 当前 ROS 节点

| 节点 | 主要职责 |
| --- | --- |
| `agent_plan_node` | plan / validate / confirm / execute 生命周期，按顺序调用 Skill Gateway |
| `sound_orientation_node` | 固定触发词与新鲜声源方向的确定性路由；仅通过 Skill Gateway 调用 `nav_turn`，不经过 LLM/Agent plan |
| `visual_game_gateway_node` | 非运动视觉游戏（分院帽等）的异步 start/query 控制平面，复用 `perception_service`，自带有界 ledger、结果校验和 `VisualGameEvent` 事件发布 |
| `visual_game_announcer_node` | 视觉游戏终态的有界去重与 TTS 调用；不拥有声卡播放、不参与游戏准入或结果判定 |
| `task_entry_node` | legacy ASR task adapter；不参与视觉游戏路由，当前不由 Hermes-only bringup 启动 |
| `task_executor_node` | planned-task executor；当前不由 bringup 启动 |

规则 `task_planner_node` 已删除；`vlm_task_planner` 也已删除。运行时唯一公开入口模式为
`embodied.entry_mode: hermes`。

## 调用链

```text
Hermes / robot-skill
  -> /embodied/plan_agent_command
  -> /embodied/validate_agent_plan
  -> 展示 exact plan 后立即 /embodied/confirm_agent_plan
  -> /embodied/execute_agent_plan
  -> skill_library Gateway

视觉游戏（独立控制平面，不进入运动技能与安全执行链路）:
  robot-skill start-game -> StartVisualGame -> visual_game_gateway_node
     -> /embodied/perception_request -> perception_service_node -> /embodied/perception_result
     -> gateway ledger（不进 agent_plan_node / skill_executor）
     -> GetVisualGameResult 查询 pending/terminal
     -> terminal /embodied/visual_game_events
        -> visual_game_announcer_node -> /voice_tts/synthesize
        -> 本机临时 WAV -> /voice_tts/play -> 扬声器
```

## Agent plan 状态机

```text
PLANNED -> VALIDATED -> CONFIRMED -> ACCEPTED -> TERMINAL
```

- plan 捕获 exact catalog identity，并保存短时不可变 `AgentPlan`。
- validate 对每个步骤执行只读 Safety 预检。
- confirm 绑定 plan digest、task ID、registry identity 和绝对 task budget；这是内部技术绑定，不是用户二次确认门。
- execute 复用确认时冻结的预算，通过 Gateway 执行 Skill 或 Workflow。
- child 接受、取消或终态未知时保持 plan 为 `ACCEPTED`，不得自动重试或释放可能仍有效的 root lease。

Agent 必须通过 `robot-skill run-workflow` 提交结构化步骤；机器人运行时不解析自然语言，
`raw_command` 只作为审计文本和幂等请求摘要的一部分。

每个 `WorkflowStep` 必须显式携带 `schema_version`。非导航旧合同使用 v1，导航 typed step 使用 v2；CLI 拒绝缺少
版本的导航步骤，不会从 `domain` 推导或重写版本。IDL/生成接口、wire preflight、执行器、CLI 和 Agent skill 文档
按同一版本原子部署，避免新旧 public request wire 混装。`imitate_human_motion` 的 typed step 使用
v1，并携带 `arm_side` 与 `imitation_duration_sec`；warmup、prepare、start 和 reset 仍由 delegated runtime
内部编排，不扩展 Agent 状态机。

## 边界

- 本包不得调用 primitive、MoveIt 或 controller。
- `sound_orientation_node` 只能调用 `/embodied/execute_skill`，不得直接调用导航 Action 或发布 `/cmd_vel`。
- 声源转向是低优先级、非排队行为；Gateway busy 或未知终态时不得自动重试。
- `sound_orientation_node` 的 `~/reset_fault` 需要新鲜 Gateway status 且 `busy=false`；Action canceled 不代表安全停止。
- motion authorization 只能来自操作员 launch 参数。
- 所有执行必须经过 `skill_library` 和 `safety_guard`。
- `perception_service` 是独立服务，不由已停用的 voice adapter 自动路由。

## 启动

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  entry_mode:=hermes \
  control_mode:=moveit_planning \
  authorize_motion:=false
```

## 视觉游戏控制平面

`visual_game_gateway_node` 提供非运动视觉游戏（分院帽等）的独立控制平面，与 Hermes
Agent plan 运动链路并存但不交互：游戏请求不进入 `agent_plan_node` 或 `skill_executor_node`，
运动 Agent plan 也不触发游戏。

- **触发**：Agent 通过 `robot-skill start-game` 调用 `StartVisualGame`。`task_entry_node` 和
  `/voice_command` 不参与视觉游戏路由，调用方也不直接发布 perception request。
  `request_id` 去除首尾空白后必须非空且不超过 128 个字符。
- **执行**：gateway 构造 `SceneAnalysisRequest` 发布到 `/embodied/perception_request`，复用
  `perception_service`；订阅 `/embodied/perception_result` 后用 `embodied_common.visual_game_contracts`
  校验业务终态（如分院帽四学院 / `NO_PERSON`），落入有界 ledger。同一时刻只准入一个 pending 游戏，
  其他不同 request ID 返回 `GAME_BUSY`；相同 ID 仍按幂等语义恢复。
  请求携带 `model_idle_timeout_sec` 约束单次模型输出空闲时间，gateway 使用更长的
  `visual_game_timeout_sec` 管理端到端游戏 deadline，为感知结果回传和终态落账保留余量。
- **查询与事件**：`GetVisualGameResult` 在 advertised retention 窗口内以调用方 request ID 幂等
  查询 pending/terminal；`/embodied/visual_game_events` 以 reliable/transient-local QoS 发布 accepted/terminal
  事件，lifespan 与结果 retention 一致。日志/UI 等旁路消费者可恢复近期事件；announcer 使用 live-only
  订阅，重启后不会播报历史终态。Gateway 为每次真实准入生成 `execution_id`，announcer 按该执行代次
  去重；retention 过期后复用同一调用方 request ID 不会误抑制新结果。
- **播报**：启用 `announce: true` 的视觉游戏会依次调用 `robot.voice_tts.service_name` 和
  `robot.voice_tts.playback_service_name`；Announcer 与播放服务要求部署在同一主机。TTS 或播放服务未配置、
  未启动或调用失败时仅记录并跳过，不影响游戏控制面；
  `announce: false` 的游戏保持静默。声明播报的终态只经
  `VisualGameEvent -> visual_game_announcer_node -> /voice_tts/synthesize -> /voice_tts/play`，Agent 不得
  再调用第二套 TTS。播报规则来自 handler registry；Sorting Hat 成功只发送学院名，`NO_PERSON` 和可恢复的
  准入失败使用对应提示，未声明播报文本的失败保持静默。TTS 或播放服务暂时不可用时，announcer
  最多等待 3 秒，服务恢复后继续播报，超时则跳过；对已经发出的 TTS 请求做有界重试。单次 TTS future 超过 `robot.voice_tts.tts_timeout_sec` 时按失败重试，迟到响应
  被忽略。合成成功后 WAV 段写入 Announcer 私有临时目录，依次交给现有 `PlayAudioFile` 服务并及时清理；
  新一局 accepted 时，同一游戏尚未进入播放的旧播报和 TTS 重试会失效；已经开始的播放不强行中断。
  `GAME_RESULT_TIMEOUT` 保留为可查询终态但不播报，避免下一局期间出现上一局的迟到提示。播放失败或超过
  `playback_timeout_sec` 不重新合成或重放已完成段，以免重复出声。失败只记日志，不改变 gateway 终态。
- **配置**：游戏定义、handler、timeout、retention、ledger capacity 全部来自
  `embodied.visual_games` / `embodied.timeouts`，由
  `embodied_common.visual_game_contracts` 归一化并计算 game capability digest。

视觉游戏当前不支持热加载：上述配置和 handler registry 在节点启动时冻结。YAML 变更需要重启 pipeline，
Python handler 变更需要重新构建并重启；运动 catalog 的 `robot-skill reload-catalog` 不影响视觉游戏。

`announce: true` 才会为启用的游戏启动 announcer；未设置或为 `false` 时，游戏仍可
通过 `start-game` / `game-result` 使用，但不产生 TTS 播报。

启用某游戏需同时置 `embodied.perception.enabled: true` 与该游戏的 `enabled: true`；只有
`announce: true` 时若要启用 announcer 需提供完整的 `robot.voice_tts` 配置；缺少 TTS 时游戏仍可运行但保持静默。launch override 造成不一致时
`embodied_pipeline.launch.py` 会 fail-fast。

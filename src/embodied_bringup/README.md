# embodied_bringup 架构契约

`embodied_bringup` 是具身运行时的启动编排包，支持 `hermes`（默认）与 `agent`（孵化）两种
`embodied.entry_mode` 入口。它消费 `robot_config` SSOT YAML，启动 Agent plan、安全校验、
Skill Gateway 以及可选感知和抓取执行服务。

可选的 `sound_orientation_node` 由 `embodied.idle_behaviors.sound_orientation.enabled` 控制。它订阅最终 ASR 文本和 `SpeechDirection`，只通过 `/embodied/execute_skill` 调用 `nav_turn`；默认关闭，不进入视觉游戏的 controller-independent closure。正常自动启动 controller 时，它与 Gateway 节点共享 readiness barrier。

启用前必须同时具备 `voice_asr.enabled`、`speech_direction.enabled`、`base_navigation`、导航 command server 和包含 `nav_turn` 的 catalog profile。当前 Voice ASR 与 speech direction 是两个独立采集进程，部署侧未验证同一 ReSpeaker 可并发读取时必须保持该行为关闭。

## 职责边界

本包负责：

- 提供 `embodied_pipeline.launch.py` 公开入口。
- 从 `robot_config` 加载机器人配置并向下游注入参数。
- 启动 `agent_plan_node`、`safety_guard_node` 和 `skill_executor_node`。
- 当 `embodied.entry_mode: agent` 时，额外启动 `ibrobot_agent_node`（自然语言 Agent
  孵化入口），其全部参数来自 `embodied.agent` 配置块；运动仍经同一条 Gateway 链路授权。
- 按配置启动低优先级 `sound_orientation_node`，但不为其提供运动旁路、排队或自动重试。
- 按配置启动独立 `perception_service` 与抓取执行依赖。
- 按配置启动 launch-managed HRI runtime。
- 当 `robot.grasp_execution.enabled=true` 时，编排 Grounded-SAM2、GraspGen、抓取验证器和
  `manipulation_execution/pick_executor_node`。
- 将 `grasp_execution.perception_node/planner_node.host_runtime` 转换为对应节点的进程环境；该块不作为
  ROS 参数传给业务节点。
- 当 `robot.grasp_execution.ik.worker_count>0` 时，自动包含 `robot_moveit/so101_ik_workers.launch.py`，
  为 Hermes 启动与监督式抓取脚本相同的并行候选 IK/FK 池。
- 保持具身业务运行时依赖集中在 bringup 层，避免 `robot_config` 反向依赖业务包。

本包不负责：

- 自行维护机器人配置或 Skill catalog。
- 实现文本规划、VLM 规划、安全规则或物理执行。
- 绕过 `skill_library` / `safety_guard` 发布运动命令。

规则 Planner ROS 节点和 `vlm_task_planner` 包已经移除。普通 `/voice_command`
流水线不再启动；当前合法入口模式为 `embodied.entry_mode: hermes` 或 `agent`。
`agent` 模式启动的 `ibrobot_agent_node` 使用进程内 Planner 库适配器（rule/vlm，
见 `ibrobot_agent` README），不是独立 Planner ROS 节点，也不恢复 `vlm_task_planner`
路线；`entry_mode: agent` 要求 `embodied.agent.incubation: true`（孵化期强制标记）。

## 依赖方向

```text
embodied_bringup
    -> robot_config
    -> embodied_agent        # Agent plan 生命周期
    -> ibrobot_agent         # entry_mode=agent 时的自然语言 Agent 孵化入口
    -> safety_guard          # 只读校验
    -> skill_library         # 唯一物理执行 Gateway
    -> perception_service    # 可选独立感知服务
    -> manipulation_service  # 可选抓取感知/规划
    -> manipulation_execution
```

## 启动

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  entry_mode:=hermes \
  authorize_motion:=false
```

`authorize_motion` 默认关闭。只有操作员完成现场安全检查后才能在 launch 时显式开启；
Agent、CLI、YAML 和动态参数都不能代替操作员授权。

SO-101 真机手动验证必须使用 `ROS_DOMAIN_ID=52`、显式设置 `moveit_display:=true`，并通过 Hermes 完成
plan/validate/confirm/execute。完整启动、回原位和关停流程见
[`docs/hermes_so101_real_robot_manual_validation_zh.md`](../../docs/hermes_so101_real_robot_manual_validation_zh.md)。

`with_perception` 只控制独立感知服务，不恢复已删除的 voice/VLM Planner 流水线。

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `robot_config` | `so101_single_arm` | robot_config 中的机器人配置名 |
| `config_path` | 空 | 可选的 YAML 绝对路径覆盖 |
| `control_mode` | 空 | 默认继承所选 robot config/stage 的 `default_control_mode` |
| `nav_stage` | 空 | 配置声明的工作阶段；导航配置支持 `mapping`/`navigation`，移动抓取统一配置还支持 `grasp` |
| `use_sim` | `false` | 是否启动仿真路径 |
| `sim_platform` | 空 | 覆盖 robot YAML `simulation.platform` 的 CLI 透传参数 |
| `entry_mode` | 空 | 覆盖 `embodied.entry_mode`（hermes/agent）；agent 模式需 `embodied.agent` 块校验通过 |
| `with_moveit` | 空 | 传递给基础 robot launch 的 MoveIt 覆盖参数 |
| `moveit_display` | `false` | 是否启动 MoveIt RViz |
| `with_embodied` | 空 | 默认启动具身运行时；`nav_stage=mapping` 时默认关闭，可显式覆盖 |
| `with_perception` | 空 | 覆盖 `robot.embodied.perception.enabled` |
| `authorize_motion` | `false` | 操作员运动授权；唯一运行时授权来源 |

`entry_mode: agent` 的孵化启动示例（SO-101 手动无运动 profile）：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_agent_manual \
  control_mode:=moveit_planning \
  entry_mode:=agent \
  authorize_motion:=false
```

Agent 孵化 profile（`so101_agent_manual`、`so101_single_arm_agent_gazebo` 等）的
`embodied.agent` 字段表与校验规则见 `robot_config` README；真机执行类 profile 必须
由操作员现场显式开启 `authorize_motion` 并配置 vlm Planner（内置 RulePlanner 仅限
仿真执行）。交互终端用 `scripts/run_agent_chat.sh` 连接到已启动的 Agent 节点。

## 已知限制

- 运动技能闭环要求 `control_mode:=moveit_planning` 或名称中包含 `moveit` 的兼容控制模式。仅启用视觉游戏时
  可在其他控制模式启动 Gateway + perception，不启动 safety/skill/planner/executor 等运动节点。
- 机械臂 profile 要求 MoveIt 兼容控制模式；`lekiwi_lidar` 导航 profile 精确要求 `base_navigation`。
- 自然语言规则入口只支持观察、回位、夹爪开合、相对移动和夹爪旋转等最小闭环动作。
- 抓取、放置、目标物操作当前不由规则入口直接生成；应通过后续 VLM/显式技能链路完善。
- 视觉趣味游戏（分院帽等）的能力开关来自 `embodied.visual_games`；camera/VLM 由
  `embodied.perception` 管理，生命周期 timeout 由
  `embodied.timeouts` 管理。
- Gateway、perception 和 announcer 属于 controller-independent 视觉闭包，启动时不等待
  ros2_control controller readiness；运动节点与 IK worker 仍在 controller ready 后启动。
  视觉游戏只允许 Agent 通过 `robot-skill` 调用并列的 `visual_game_gateway_node` start/query 服务。
  节点接收统一的 visual-game timeout/retention、robot name 和服务名，
  并向 `embodied.visual_game_event_topic` 发布 accepted/terminal 事件；同时从同一配置计算覆盖 handler、transport、timeout、retention 和 ledger capacity 的 game capability digest。同一时刻只允许一个 pending 游戏，
  不同 request ID 的并发请求返回 `GAME_BUSY`，终态 ledger 仍按配置保留。
  启用 `announce: true` 的视觉游戏会在 TTS runtime 配置完整时启动
  `visual_game_announcer_node`；TTS 未配置时不启动 announcer，运行中的 TTS 或播放服务暂时不可用时
  最多等待 3 秒再跳过播报，游戏仍可使用 start/query 控制面。
  播报终态统一经事件、TTS 合成服务和本机播放服务处理；announcer 使用合成返回的 WAV 临时文件调用
  现有 `/voice_tts/play`，Agent 不再自行播报结果。
  启用某游戏需置 `embodied.perception.enabled: true` 与该游戏的
  `enabled: true`；若 launch
  override（如 `with_perception:=false`）造成不一致，`embodied_pipeline.launch.py` 会在生成节点前 fail-fast
  拒绝启动，避免生成不一致的运行时节点图。
- 只要启用任一视觉游戏，Gateway、perception 和 announcer（如需）就会作为同一个
  controller-independent 视觉闭包在 controller readiness 之前启动；MoveIt 混合拓扑也遵循这一规则。
  controller readiness 只延迟 safety、Skill Gateway、Agent plan、抓取执行和 IK worker 等运动闭包节点，
  不会让已经可发现的视觉游戏 start service 因 perception 尚未启动而暂时返回 `PERCEPTION_UNAVAILABLE`。
- `lekiwi_handeye_realsense_grasp` 可通过显式 `pick_object` 技能从 Hermes 调用完整抓取闭环。
- `lekiwi_nav_grasp` hybrid stage 启用 `embodied.imitate_human_motion` 时，bringup 会把 arm 关节顺序、
  reset positions 和 joint limits 从同一份 `robot_config` 注入 HRI runtime，并在启动时默认尝试一次 warmup。
- 真机端口、相机和手眼标定直接维护在该 robot YAML 中；本 launch 与 `robot-skill` 应使用同一个
  `robot_config` 名称，workspace 外部完整 YAML 才需要显式传 `config_path`。

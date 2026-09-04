# 固定触发词声源转向设计

状态：已接入 ROS launch，默认关闭；启用前必须在目标机器人配置中显式打开。

`VoiceASRNode` 与 `speech_direction_node` 当前各自打开音频设备，没有共享采集链路。目标机器人配置保持 `sound_orientation.enabled: false`；启用前必须先验证部署环境允许两个采集进程并发读取 ReSpeaker，或者先实现共享音频采集。未经验证不得在生产配置中默认开启。

本文定义“机器人收到固定触发词后，依据最近的新鲜声源方向执行一次底盘转向”的实现契约，供后续代码实现和大模型协作使用。

## 1. 目标

本功能满足以下行为：

- 机器人只在收到明确的固定触发词时执行声源转向。
- 普通说话或普通机器人 Skill 语句不会触发声源转向。
- 转向请求必须通过现有 Skill Gateway 和 `nav_turn` Skill。
- `nav_turn` 获得 root lease 后，转向期间状态为 busy，不能并行执行其他 root Skill。
- Gateway 已忙时，当前转向请求直接丢弃，不进入等待队列，不自动重试。
- `nav_turn` 终态未知时，停止自动派发，等待人工恢复。
- Action 取消不等于底盘已稳定停止；取消终态进入 `FAULT_UNKNOWN`，必须通过 reset service 和新鲜 Gateway 状态人工恢复。
- 一次触发最多执行一个转向动作，转向完成后进入 cooldown，避免重复微调。

默认固定触发词为：

```text
转向我
```

固定触发词必须来自机器人配置，不得硬编码在 ROS 节点中。配置可以扩展为多个等价短语，但第一版要求每条语音在归一化后与一个完整短语精确匹配，不允许使用宽泛的子串匹配。

## 2. 非目标

第一版不实现以下能力：

- 不经过 LLM、Hermes 或 `agent_plan_node` 做自然语言规划。
- 不把所有 `SpeechDirection` 事件自动转换为转向动作。
- 不创建跨 Skill 的通用等待队列。
- 不缓存转向期间产生的旧方向，转向完成后不执行旧方向。
- 不抢占正在执行的前台 Skill。
- 不直接调用 `/navigation/execute`、`/cmd_vel`、MoveIt、controller 或 `/task_executor/*`。
- 不实现唤醒词模型；“固定触发词”是 ASR 文本上的确定性路由规则。
- 不解决当前 ASR 文本和方向消息缺少共同 `speech_segment_id` 的结构性问题。

## 3. 当前架构定位

当前机器人运动控制链路为：

```text
voice_asr_node
  -> /voice_command (std_msgs/String)
  -> sound_orientation_node

speech_direction_node
  -> /voice/speech_direction (ibrobot_msgs/SpeechDirection)
  -> sound_orientation_node

sound_orientation_node
  -> /embodied/get_skill_gateway_status
  -> /embodied/execute_skill (SkillCommand: nav_turn)
  -> skill_executor_node
  -> safety_guard
  -> navigation_command_server
  -> Nav2 / base
```

该节点不经过：

```text
Hermes / LLM
agent_plan_node
plan-workflow
confirm-plan
```

但它必须经过：

```text
Skill Gateway
RootExecutionLease
motion_authorized
safety_guard
nav_turn catalog contract
```

这一区分很重要：本功能绕过的是 Agent 规划入口，不是运动安全控制面。

当前固定转向节点不得与旧的 `voice_control` 直接导航链路并行使用。旧链路面向 `/voice_asr/keyword_matched` 和 `nav2_goal_client`，不是本功能的标准执行入口。Hermes-only bringup 当前不启动旧 `task_entry_node`；如果部署组合手动启动了它，必须先增加路由隔离，避免同一条 `/voice_command` 同时被 Agent 和固定触发节点消费。

相关现有接口：

| 接口 | 文件 | 用途 |
| --- | --- | --- |
| `/voice/speech_direction` | `src/ibrobot_msgs/msg/SpeechDirection.msg` | 声源方向事件，`base_link` 下的 `azimuth_rad` |
| `/embodied/get_skill_gateway_status` | `src/ibrobot_msgs/srv/GetSkillGatewayStatus.srv` | Gateway 状态和 registry identity |
| `/embodied/execute_skill` | `src/ibrobot_msgs/action/SkillCommand.action` | 受保护的单 Skill 执行入口 |
| `nav_turn` | `src/skill_catalog/config/skills/nav_turn/manifest.yaml` | 底盘原地转向 Skill |

## 4. 触发和方向关联

当前 `SpeechDirection` 只有 `seq_id` 和方向产生时间，没有与 `/voice_command` 共享的语音段 ID。因此第一版采用受限的近似关联：

1. 节点持续缓存最近一条有效 `SpeechDirection`。
2. 节点收到固定触发词时，只接受 `max_direction_age_sec` 内的方向。
3. 没有新鲜方向时，进入短暂 `WAITING_FOR_DIRECTION`，等待触发词之后到达的新方向。
4. 等待超时后回到监听状态，本次触发丢弃。
5. 不使用超过新鲜度窗口的历史方向。
6. 同一 `(seq_id, header.stamp)` 只允许消费一次。

第一版不声称这能严格证明方向和文本来自同一语音段。后续若需要严格语义，应在 ASR 文本和方向消息中增加共同的 `speech_segment_id` 或 `interaction_id`，并将关联改为 ID 精确匹配。

方向转换契约：

```text
abs(azimuth_rad) < deadband_rad -> 不转向
azimuth_rad > 0                 -> direction=left
azimuth_rad < 0                 -> direction=right
```

消息必须满足：

- `header.frame_id == base_link`。
- `azimuth_rad` 是有限数。
- `azimuth_rad` 不超过允许的平面角范围，超出范围时拒绝本次事件。
- 消息产生时间不早于 `now - max_direction_age_sec`。

## 5. Gateway 和忙状态

当前 Gateway 没有高层 Skill 等待队列。

`BoundedRequestLedger` 只有 active 和 terminal 记录，职责是幂等、冲突检测和有限终态保留，不是 pending queue。`RootExecutionLease` 只允许同一时刻一个 root execution。`action_dispatcher` 的 deque 只缓冲模型 action chunk，与高层 `nav_turn` 无关。

因此执行语义是：

```text
Gateway idle
  -> nav_turn admission succeeds
  -> root lease acquired
  -> busy=true

another root Skill arrives
  -> SKILL_BUSY
  -> no implicit wait
  -> no automatic retry
```

节点在派发前应查询以下状态：

```text
control_plane_ready == true
motion_authorized == true
busy == false
nav_turn capability.ready == true
```

这些字段只是提前过滤。最终互斥必须依赖 `skill_executor_node` 内部的原子 Gateway admission，因为状态查询和 Action 提交之间存在竞态。

如果最终 Action 结果为 `SKILL_BUSY`，这是一次确定的失败终态，节点回到 cooldown 或监听状态，但不得使用新的 task ID 自动重试。

如果 Action 传输失败、取消超时、终态无法确认或 root lease 状态不确定，节点进入 `FAULT_UNKNOWN`，禁止继续自动派发。

节点提供 `~/reset_fault` 服务，但 reset 只有在最近的 Gateway status 新鲜且 `busy=false` 时成功；该服务不是自动恢复入口。

## 6. 状态机

```text
DISABLED
    --enabled--> IDLE_LISTENING

IDLE_LISTENING
    --非触发文本--> IDLE_LISTENING
    --触发词且无新鲜方向--> WAITING_FOR_DIRECTION
    --触发词且 Gateway 忙--> IDLE_LISTENING
    --触发词且条件满足--> DISPATCHING

WAITING_FOR_DIRECTION
    --新鲜方向且 Gateway 空闲--> DISPATCHING
    --超时--> IDLE_LISTENING
    --Gateway 忙/未授权/未就绪--> IDLE_LISTENING

DISPATCHING
    --Action 已提交--> TURNING
    --确定性拒绝--> COOLDOWN
    --提交结果未知--> FAULT_UNKNOWN

TURNING
    --确定终态--> COOLDOWN
    --取消或结果未知--> FAULT_UNKNOWN

COOLDOWN
    --cooldown 到期--> IDLE_LISTENING

FAULT_UNKNOWN
    --人工 reset--> IDLE_LISTENING
```

`TURNING` 和 `COOLDOWN` 期间到达的方向消息必须丢弃，不能更新下一次请求的缓存。这样可以避免在转向完成后重新使用转向期间的旧方向。

## 7. 推荐参数

参数名称应进入 `robot_config` 的 `embodied` 配置，由 loader 归一化并校验；ROS 节点只接收归一化参数。

| 参数 | 建议值 | 约束 |
| --- | ---: | --- |
| `enabled` | `false` | 默认关闭，需显式启用 |
| `trigger_phrases` | `["转向我"]` | 至少一个非空完整短语 |
| `direction_topic` | `/voice/speech_direction` | 使用现有消息契约 |
| `command_topic` | `/voice_command` | 使用现有 ASR 文本 |
| `gateway_status_service` | `/embodied/get_skill_gateway_status` | 只读状态接口 |
| `skill_action_name` | `/embodied/execute_skill` | 受保护 Skill Action |
| `skill_name` | `nav_turn` | 必须存在于当前 catalog |
| `deadband_deg` | `15.0` | 大于等于 0，小于 180 |
| `max_direction_age_sec` | `1.3` | 大于 0 |
| `direction_wait_sec` | `0.5` | 大于等于 0 |
| `cooldown_sec` | `1.5` | 大于等于 0 |
| `max_turn_deg` | `180.0` | 大于 0，不超过 catalog/安全上限 |
| `turn_timeout_sec` | `10.0` | 大于 0，受 Gateway task budget 限制 |
| `action_acceptance_timeout_sec` | `2.0` | Action goal acceptance 的 wall-clock watchdog |
| `busy_policy` | `drop` | 第一版只允许 `drop` |

第一版不支持通过参数把 `busy_policy` 改为 `queue` 或 `retry`。如果未来需要队列或前台抢占，应先扩展 Gateway 的 owner、优先级、取消和终态契约，不能在节点内部私自实现。

## 8. Skill 请求

节点生成的请求必须是 `SkillCommand` 的 direct root SkillCommand：

```text
schema_version = 2
dispatch_binding.schema_version = 1
dispatch_binding.task_id = task_id
dispatch_binding.root_task_id = task_id
dispatch_binding.root_lease_nonce = ""
dispatch_binding.dispatch_nonce = ""
skill_name = "nav_turn"
direction = "left" or "right"
degree = positive finite number
```

`DispatchBinding` 的 registry identity 必须来自最近一次 Gateway status：

```text
expected_registry_epoch
expected_registry_generation
expected_registry_digest
```

节点不得自行生成或修改 catalog digest，不得复用过期 binding。`task_id` 必须包含稳定的方向事件身份，使同一事件不会因为 ROS callback 重入而产生两个 root 请求。

## 9. 普通 Skill 的优先级

固定触发词只处理纯转向语句，例如：

```text
转向我
```

包含额外动作意图的语句不应被固定转向节点消费，例如：

```text
转向我，然后拿起红色方块
看向我并把香蕉拿过来
```

第一版应将这类文本交给统一 Agent 路由，固定转向节点返回 `NON_EXACT_TRIGGER`，不独立创建 `nav_turn`。否则会产生两个 root request 竞争同一个 lease。

如果系统未来要支持“先转向再执行 Skill”，必须由 Agent 生成一个统一 Workflow，不能由固定转向节点和 Agent 分别下发两个请求。

## 10. ROS launch 接入

节点已经由 `embodied_bringup` 的 `generate_embodied_nodes()` 接入。它与 `safety_guard_node`、`skill_executor_node` 和 `agent_plan_node` 使用相同的 controller readiness barrier：

```text
robot.launch.py
  -> controller readiness waiter
  -> embodied_pipeline.launch.py
  -> embodied_bringup.generate_embodied_nodes()
  -> sound_orientation_node
```

只有以下条件同时满足时才会启动：

- `embodied.enabled: true`。
- `embodied.idle_behaviors.sound_orientation.enabled: true`。
- 当前运行模式满足 embodied motion closure 的控制模式条件。
- controller readiness waiter 成功退出。

当 `auto_start_controllers:=false` 时，bringup 不创建 readiness waiter，运动节点会直接启动，但所有实际请求仍由 Gateway readiness 和 motion authorization fail-closed。该模式用于人工管理 controller 的调试部署，不代表已经完成自动 readiness 验证。

建议只在支持移动底盘、`base_navigation` 控制模式和 `nav_turn` catalog capability 的 robot profile 中启用：

```yaml
embodied:
  enabled: true
  idle_behaviors:
    sound_orientation:
      enabled: true
      trigger_phrases: ["转向我"]
      direction_topic: /voice/speech_direction
      command_topic: /voice_command
      skill_name: nav_turn
      direction_frame: base_link
      deadband_deg: 15.0
      max_direction_age_sec: 1.3
      direction_wait_sec: 0.5
      cooldown_sec: 1.5
      max_turn_deg: 180.0
      turn_timeout_sec: 10.0
      status_retry_sec: 0.5
```

配置还必须同时满足：

- `voice_asr.enabled: true`，因为固定触发词来自 `/voice_command`。
- `speech_direction.enabled: true`，因为转向角来自 `/voice/speech_direction`。
- 当前 profile 中存在 `control_modes.base_navigation`。
- 当前导航 stage 提供 `navigation.command_server.action_name`。
- 运行时启动时设置 `authorize_motion:=true` 才会实际允许运动；节点不能自行开启授权。
- 已验证 Voice ASR 和 speech direction 可以并发读取部署的麦克风设备；否则保持本行为关闭。

典型启动形式：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=lekiwi_nav_grasp \
  nav_stage:=hybrid \
  with_embodied:=true \
  authorize_motion:=true
```

如果需要先验证发现、状态和拒绝路径，使用 `authorize_motion:=false`。此时节点可以启动和监听，但 Gateway 对 `nav_turn` 返回 `MOTION_NOT_AUTHORIZED`，不会产生物理运动。

`with_embodied:=true` 只打开 embodied runtime 总开关，不会覆盖 `sound_orientation.enabled`。关闭该行为时只需将其设为 `false`，不需要删除节点代码或修改 launch 文件。

## 11. 实现拆分

建议后续实现按以下文件拆分：

| 文件 | 变更内容 |
| --- | --- |
| `src/embodied_agent/embodied_agent/sound_orientation_policy.py` | 无 ROS 依赖的状态机、输入校验、方向转换和决策 |
| `src/embodied_agent/embodied_agent/sound_orientation_node.py` | ROS topic/service/action 适配，不复制业务规则 |
| `src/embodied_agent/setup.py` | 注册 `sound_orientation_node` console script |
| `src/embodied_agent/package.xml` | 确认 `rclpy`、`ibrobot_msgs` 等依赖 |
| `src/embodied_bringup/embodied_bringup/launch_builders/embodied.py` | 按配置启动节点并复用 controller readiness barrier |
| `src/robot_config/robot_config/config.py` | 增加归一化配置结构 |
| `src/robot_config/robot_config/loader.py` | 增加配置和 catalog 关联校验 |
| `src/robot_config/config/robots/<robot>.yaml` | 仅在目标机器人启用该行为 |
| `src/embodied_agent/test/test_sound_orientation_policy.py` | 本文定义的纯策略契约测试 |
| `src/embodied_agent/test/test_sound_orientation_node.py` | ROS 适配和 Action 生命周期测试 |
| `src/embodied_bringup/test/test_embodied_launch_builder.py` | launch 节点和参数投影测试 |

不要把 `nav_turn` 的 primitive sequence 复制到 `embodied_agent`，不要修改 `skill_executor_node` 以增加本功能专用旁路。

## 12. 验收标准

- 未启用配置时不启动 `sound_orientation_node`。
- 普通文本不生成任何 `nav_turn` 请求。
- 只有完整匹配固定触发词的文本才进入方向等待或派发。
- 触发词后没有新鲜方向时不派发。
- 小于 deadband 的角度不派发。
- 正角度转换为 `left`，负角度转换为 `right`。
- 转向期间的新方向不生成第二个请求。
- cooldown 期间的新方向不生成请求。
- 同一 `(seq_id, stamp)` 不产生重复请求。
- Gateway busy、未授权、未就绪或 capability not ready 时不派发。
- Gateway 在查询后变 busy 时，最终 `SKILL_BUSY` 结果不会自动重试。
- 取消/传输/终态未知时进入 `FAULT_UNKNOWN`，不再自动派发；取消不能直接进入 cooldown。
- `~/reset_fault` 只在 fresh Gateway status 且 `busy=false` 时恢复，恢复时清理旧请求状态。
- direct `SkillCommand` 始终通过 `/embodied/execute_skill`，不直接调用导航或 controller 接口。
- 含“转向 + 其他 Skill”意图的句子不会被固定转向节点拆成两个 root request。

## 13. 给后续代码 Agent 的执行说明

实现前先阅读本文件、`src/ibrobot_msgs/msg/SpeechDirection.msg`、`src/ibrobot_msgs/srv/GetSkillGatewayStatus.srv`、`src/ibrobot_msgs/action/SkillCommand.action` 和 `src/skill_catalog/config/skills/nav_turn/manifest.yaml`。

实现顺序：

1. 先实现或保持 `sound_orientation_policy.py` 的纯函数/状态机契约，并使 `test_sound_orientation_policy.py` 全部通过。
2. 再实现 ROS 节点适配：订阅文本和方向、异步查询 Gateway、构造完整 `SkillCommand` binding、等待终态。
3. launch/config 接入已经存在；修改时必须保持 `sound_orientation.enabled` 默认关闭和 controller readiness barrier。
4. 添加或运行 ROS 集成测试，验证 Action accepted 不等于 admitted，`SKILL_BUSY` 和未知终态行为。

禁止：

- 使用 `robot-skill` CLI 子进程作为节点内部执行器。
- 在 subscription callback 内同步阻塞等待 Action 结果。
- 发现 busy 后通过新 task ID 重试。
- 在节点中直接发布 `/cmd_vel` 或调用 `/navigation/execute`。
- 把 `SpeechDirection` 的所有更新都当作动作触发。
- 为了解决同句多意图而在节点内增加模糊自然语言解析。

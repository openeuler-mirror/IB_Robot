# robot_skill_cli

`robot_skill_cli` 提供约束 LLM/Agent 对 ROS 访问面的命令行工具集，包含三条受控入口：

```text
运动控制:   Hermes -> ibrobot-control Skill -> robot-skill            -> ROS Capability Gateway
感知读取:   Hermes -> ibrobot-perceive (硬编码 allowlist)              -> ros2 topic echo --once
裸 ROS 拦截: Hermes pre_tool_call -> ibrobot-block-raw-ros hook        -> block ros2/rclpy/roslaunch
```

- `robot-skill`：运动技能请求的唯一受控入口，限定在 Capability Gateway 公开边界内，不直接调用
  primitive、MoveIt、controller 或裸 ROS 运动接口，也不修改 Gateway 的授权和安全策略。
- `ibrobot-perceive`：只读感知读取的唯一受信入口，硬编码 source/field allowlist + 审计日志，
  有界超时；不经 Gateway，不初始化 `rclpy`。
- `ibrobot-block-raw-ros`：Hermes `pre_tool_call` 防御 hook，拦截裸 `ros2`/`rclpy`/`roslaunch`
  调用，是 defense-in-depth，权威边界仍是 `authorize_motion` 与 Gateway plan validation。

运动控制始终走 Gateway；感知读取走受控 allowlist；裸 `ros2` 调用被 hook 拦截。

## 环境与配置

在工作区根目录加载环境：

```bash
source .shrc_local
```

`.shrc_local` 是 ROS+venv+install overlay 的 SSOT 入口（见 `AGENTS.md` 环境初始化章节），无需重复 `source install/setup.bash`。

`run-workflow` 会先完成 exact plan 的展示回调并 flush 输出，再执行内部技术 confirm；展示失败或收到停止请求时不会
进入运动执行。`immediate_after_presentation` 只取消二次用户确认等待，不绕过 Gateway validation、operator
`authorize_motion` 或 action admission。可选 `--request-id` 是调用方幂等键；timeout 或结果未知时不得自动重放。

`run-workflow` 底层的 `InteractiveController` 也是库级集成入口（如 `ibrobot_agent` 复用）。展示回调收到的
presentation dict 除步骤、plan digest、registry identity 与 task ID 外，还包含 `plan_kind` 和
`proposed_task_budget_sec`（fresh status 的任务预算快照）。库调用方可传入构造参数 `submission_callback`，在
goal 提交前收到展示 tuple（task/plan ID、plan digest、registry identity、expected step count）；回调抛错会以
`SUBMISSION_PERSISTENCE_FAILED` 终态阻断 goal 提交。`run(..., expected_registry_identity=...)` 会在 planning
前校验 live catalog 身份，与快照不一致时按 `SKILL_SNAPSHOT_DIGEST_MISMATCH` fail closed，不发送 goal。

每个命令可用 `--config-name NAME` 选择配置，或用 `--config-path PATH` 指向 YAML；两个 flag 在 CLI 中互斥。
配置解析完全复用 `robot_config.resolve_robot_config_path()`，CLI 不维护第二套路径优先级：底层选择顺序是
显式 path、显式 name、`ROBOT_CONFIG`、`ROBOT_NAME`、默认 `so101_single_arm`。按名称先查安装目录，再查
源码 `config/robots/`；显式 path 必须存在。

## 命令

| 命令 | ROS | 用途 |
|---|---:|---|
| `list-skills` | 否 | 列出所有启用的高层技能及公开描述 |
| `list-games` | 否 | 列出配置中已启用且允许 Agent 触发的视觉游戏 |
| `describe-game GAME` | 否 | 查看视觉游戏输入、结果 schema、timeout 和配置摘要 |
| `describe SKILL` | 否 | 查看参数 schema、单位、语义描述和 timeout policy |
| `list-poses` | 否 | 列出公开命名位姿 |
| `status` | 是 | 读取 Gateway 授权、控制模式、readiness、busy 和 ledger 状态 |
| `validate SKILL` | 是 | 本地 schema 检查后调用 Gateway 安全校验，不执行动作 |
| `execute SKILL --task-id ID` | 是 | 通过 `SkillCommand` 执行高层技能 |
| `cancel --task-id ID` | 是 | 以同一 deterministic goal UUID 请求取消并轮询 terminal |
| `reload-catalog --request-id ID --force` | 是 | 重新编译并原子激活 Gateway 已配置的 Skill catalog source |
| `plan-workflow --text TEXT --workflow-json JSON --request-id ID` | 是 | 提交一份短时 typed Agent plan |
| `run-workflow [--request-id ID] --text TEXT --workflow-json JSON` | 是 | 展示后按显式 immediate admission policy 自动执行 |
| `validate-plan --plan-token TOKEN` | 是 | 对 exact snapshot 计划做只读逐步预检 |
| `confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID [--timeout-sec SEC]` | 是 | 校验身份/摘要/task_id 并冻结 `task_budget_sec`，转入 `CONFIRMED` |
| `execute-plan ... --plan-id ID --plan-digest DIGEST --registry-* ... --expected-step-count N` | 是 | 执行已确认的 Agent plan，并以展示过的 tuple 校验终态 |
| `cancel-plan --task-id ID --plan-id ID --plan-digest DIGEST --registry-* ... --expected-step-count N` | 是 | 请求取消并以展示过的 tuple 校验终态 |
| `robot-skill-closed-loop ...` | 是 | 展示 Workflow 后立即执行，并验证「别动」和安全 continuation 门禁 |
| `start-game GAME --request-id ID` | 是 | 以调用方 ID 幂等发起视觉游戏 |
| `game-result --request-id ID` | 是 | 查询视觉游戏的 pending/terminal 结果 |
| `ibrobot-perceive --source SOURCE --field FIELD [--config-name NAME]` | 否 | 只读感知 source 读取（硬编码 allowlist + audit log），LLM 唯一受信入口 |

`ibrobot-perceive` 是独立的 console script，不经过 Gateway，也不初始化 `rclpy`。它以硬编码
source/field allowlist 限制可读取的感知量，通过 `ros2 topic echo --once` 读取 YAML 输出并打印请求
字段的裸字面量值（供 LLM 直接读取并注入 `workflow_json`），不遵循下文的 JSON envelope 输出契约；
任何非 allowlist 的 source/field 都会被拒绝并写入 `/tmp/hermes-perceive.log`。当前 allowlist 包含
`voice_direction`（topic `/voice/speech_direction`，字段 `azimuth_rad`、`seq_id`）和
`arm_joint_position`（字段 `position`），扩展必须修改源码，不接受 config.yaml 覆盖。

`--source` 是语义别名，不是 ROS topic 名；config-backed source 的实际 topic 在运行时从 `robot_config`
解析，复用 `resolve_robot_config_path()`，CLI 不维护第二套路径优先级：

- `voice_direction`：`voice_asr_service` 固定契约，topic 写死为 `/voice/speech_direction`，不读 robot_config。
- `arm_joint_position`：topic 取自 `robot_config` 的 `moveit.joint_state_topic`，缺失时回退 `/joint_states`。
  so101_single_arm 解析为 `/joint_states`；lekiwi_handeye_realsense_grasp 解析为
  `/arm_joint_state_broadcaster/joint_states`（手臂关节，不含底盘轮子）。`--config-name`/`--config-path`
  语义与 `robot-skill` 一致且互斥；省略时按 `ROBOT_CONFIG`/`ROBOT_NAME`/默认 `so101_single_arm` 解析。
  安全边界是硬编码的 *field* 集合，不是 topic 名——topic 名是机器人级配置，不是安全敏感面。

`arm_joint_position.position` 返回原始弧度数组，不包含关节名映射，不得编造关节名称、重排数值或把
数组索引解释为具体关节。

`ros2 topic echo --once` 返回下一条已发布消息的单次点时值，不是持久快照。对
`/voice/speech_direction` 这类事件型 topic，发布方不活跃时会在超时（5s）内取不到值；取到的值在
消费时可能已经过期。该字面量在 `plan-workflow` 时冻结进 plan digest，后续按 frozen plan 语义审计；
执行结果以真实运动为准，不自动重试或修正。

catalog-only 命令不初始化 `rclpy`，只读取本地归一化配置。runtime 命令只访问 Gateway status、
`ValidateSkill`、`SkillCommand`、`ValidatePrimitive`、`PrimitiveCommand`、Agent plan services/actions 和标准
`CancelGoal` 接口；视觉游戏 runtime 命令只访问 start/result 服务。Hermes 启动前会先验证四个版本化
Skill/Primitive 公共接口，再检查 Gateway/Agent 接口。

```bash
robot-skill --config-name so101_single_arm list-skills
robot-skill --config-name so101_single_arm describe move_relative_ee
robot-skill --config-name so101_single_arm list-poses
robot-skill --config-name so101_single_arm status
robot-skill --config-name so101_single_arm reload-catalog \
  --request-id reload-001 --force

robot-skill --config-name so101_single_arm validate move_relative_ee \
  --motion-direction forward --motion-distance 0.03

robot-skill --config-name so101_single_arm execute move_relative_ee \
  --task-id task-20260725-001 \
  --motion-direction forward --motion-distance 0.03

robot-skill --config-name so101_single_arm cancel \
  --task-id task-20260725-001

robot-skill --config-name so101_single_arm plan-workflow \
  --request-id plan-request-001 --text "打开夹爪" \
  --workflow-json '[{"schema_version":1,"skill_name":"open_gripper_skill"}]'
robot-skill --config-name so101_single_arm validate-plan \
  --plan-token PLAN_TOKEN
robot-skill --config-name so101_single_arm confirm-plan \
  --plan-token PLAN_TOKEN --plan-digest PLAN_DIGEST --task-id agent-task-001
# 可选 --timeout-sec 30 冻结一份小于等于 Gateway task budget 的预算
robot-skill --config-name so101_single_arm execute-plan \
  --plan-token PLAN_TOKEN --confirmation-token CONFIRMATION_TOKEN --task-id agent-task-001 \
  --plan-id PLAN_ID --plan-digest PLAN_DIGEST \
  --registry-epoch REGISTRY_EPOCH --registry-generation REGISTRY_GENERATION \
  --registry-digest REGISTRY_DIGEST --expected-step-count 1
# execute-plan 必须传 confirm 时使用的同一 --timeout-sec；省略时两端都默认 Gateway task budget
robot-skill --config-name so101_single_arm cancel-plan \
  --task-id agent-task-001 --plan-id PLAN_ID --plan-digest PLAN_DIGEST \
  --registry-epoch REGISTRY_EPOCH --registry-generation REGISTRY_GENERATION \
  --registry-digest REGISTRY_DIGEST --expected-step-count 1

robot-skill --config-name so101_single_arm list-games
robot-skill --config-name so101_single_arm describe-game sorting_hat
robot-skill --config-name so101_single_arm start-game sorting_hat --request-id game-20260803-001
robot-skill --config-name so101_single_arm game-result --request-id game-20260803-001
```

### Wire schema 与原子部署

`SkillCommand.Goal`、`PrimitiveCommand.Goal`、`ValidateSkill.Request` 和 `ValidatePrimitive.Request` 的首字段都是
`uint32 schema_version`，当前 v1 公共 wire 合同必须显式发送 `1`。`WorkflowStep` 也必须携带显式版本：旧的非导航
步骤使用 v1，导航步骤使用 v2；CLI 会拒绝缺少版本的导航 typed step，并且不会根据 `domain` 推导或改写版本。

IDL、生成的 ROS 接口、`embodied_common` wire preflight、执行器、CLI 和 Agent skill 文档必须作为一个版本化发布单元
原子部署。启动时发现生成接口不是同一版本会在创建 ROS client、subscription、action 或 readiness 之前失败；不要把
旧生成接口与新节点或新 CLI 混装。

可用 typed flags 包括 `--target-name`、`--place-name`、`--motion-direction`、`--motion-distance`、
`--arm-side`、`--imitation-duration-sec` 和 `--timeout-sec`。CLI 根据当前技能显式
`capability.parameters` schema 拒绝缺失参数和不属于该技能的参数。

`imitate_human_motion` 在 `lekiwi_handeye_realsense_grasp_lidar` profile 中公开，调用示例：

```bash
robot-skill --config-name lekiwi_nav_grasp validate imitate_human_motion \
  --arm-side auto --imitation-duration-sec 5
```

该命令只进入现有 Gateway；内部 HRI Action 不是 CLI 公共入口。

### 公开 catalog 字段

`list-skills` 输出 `robot_name`、`config_digest` 和技能数组；每个技能只含 `name`、`summary`、`domain`、
`moves_robot`、`required_control_mode`。`describe SKILL` 在这些字段之外输出该技能的 `parameters`、
`recovery_policy`、运动能力 `timeout_policy` 和 `config_digest`。`list-poses` 只输出命名位姿名称，不输出坐标。
primitive sequence、目标绑定、关节值和 ROS transport 名称不属于 CLI catalog。

`list-games` 只公开已启用游戏的 `name`、`summary`、`result_field` 和视觉游戏 `config_digest`；
四个视觉游戏命令使用独立的轻量配置上下文，不编译运动 Skill catalog，也不要求 MoveIt 或
`robot_description` 才能完成发现、启动和查询。
`describe-game` 进一步公开 required inputs、结果 schema、timeout、retention 与 ledger capacity。视觉游戏不属于运动 capability，因此不进入
`list-skills`、`ValidateSkill` 或 `SkillCommand`。

### 视觉游戏与 TTS

`start-game` 使用调用方提供的 request ID，通过视觉游戏控制服务幂等发起请求，不等待 VLM 结果；
在 advertised retention 窗口内，同 ID、同游戏的重复 start 返回原请求且不重复执行，同 ID、不同游戏被拒绝。
记录过期后 Gateway 不再保留该 ID，也不再保证检测重复；调用方应始终生成全局唯一 ID。`game-result` 按 ID
查询 Gateway 保存的 pending/terminal 结果；
pending 超过配置 deadline 会收敛为 `GAME_RESULT_TIMEOUT`。`sorting_hat` 成功终态中的 `scene_summary` 已由
perception response contract 约束为四学院之一；没有清晰可见的人时 Gateway 返回 `NO_PERSON` 失败终态，
不会向调用方暴露可播报的 `scene_summary`。Agent 仍应轮询至 `terminal=true` 获取结构化结果，但不得把结果
再次交给自身 TTS；运行时统一由
`VisualGameEvent -> visual_game_announcer_node -> /voice_tts/synthesize -> /voice_tts/play`
完成本机合成和播放。TTS 或播放服务不可用时跳过，CLI 不负责声卡播放。
`start-game` 的 accepted 仅表示 Gateway 已记账并把请求交给一个在线 subscriber；相机或 VLM 的运行时错误会在
后续 `game-result` 中作为失败终态返回。

CLI 与运行中 Gateway 必须使用相同的视觉游戏 `config_digest`。start service 响应丢失时，可以查询同一 ID，
或在 retention 窗口内用完全相同的 game/ID 重发 start 进行幂等恢复；不得换新 ID 自动重试。

## 调用顺序

传统单技能低层 CLI 流程按以下顺序工作：

1. `status`
2. `list-skills`
3. `describe SKILL`
4. `validate SKILL`
5. `execute SKILL --task-id ID`

以下分阶段顺序仅用于诊断、协议测试和明确的低层集成调试。正常 Hermes 自然语言运动请求不得逐个调用这些
命令，必须使用上文的 `run-workflow` 单一复合入口。

在明确进行分阶段协议测试时，顺序为：

1. `plan-workflow --text TEXT --workflow-json JSON --request-id ID`
2. `validate-plan --plan-token TOKEN`
3. 向用户展示步骤、参数、plan digest、registry identity 和 fresh task ID，并 flush 输出
4. 不等待二次确认，立即调用 `confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID [--timeout-sec SEC]`
5. 立即调用 `execute-plan`，除 token、task ID 和相同 timeout 外，传入刚展示的 plan ID/digest、registry
   epoch/generation/digest 和 expected step count

这里的 `confirm-plan` 是 Gateway 对 exact plan/task tuple 的内部技术绑定，不是用户二次确认。用户在展示后
说「别动」时，控制器必须把停止意图锁存到 goal 发送、goal acceptance 和执行阶段；goal 已可能提交时由唯一
执行线程发起一次 action cancellation 并等待权威终态；不要同时从另一个进程调用 `cancel-plan`。pre-send stop
必须抑制新 goal；只有能证明 fresh task 从未提交的 in-process controller 才不发送 CancelGoal，独立 CLI 必须
对可能存在的幂等 retry goal 执行 convergence。`GoalStatus` 本身不足以证明安全结果：result 的 success/error、plan
ID/digest、registry identity 和 completed step count 也必须与命令携带的展示 tuple 一致，否则按
`SKILL_CANCEL_TIMEOUT`/未知状态 fail closed。已知的 uncertain-motion error 保留原始 code 并退出 15；只有身份
或结构证明无效时才合成为 `SKILL_CANCEL_TIMEOUT`。

`robot-skill-closed-loop --resume` 在当前基线明确拒绝。当前 ROS 契约只有 `completed_step_count` 遥测，没有
server-owned continuation admission；客户端不得切片旧步骤并用新 task ID 绕过 Gateway。确定的
`CANCELED + SKILL_CANCELLED` 终态后，用户可以提出一个独立的新 continuation 请求；成功、失败和未知状态
都不授权自动继续。

`confirm-plan` 与 `execute-plan` 共享可选的 `--timeout-sec`：省略时两端都默认使用 Gateway 当前
`task_budget_sec`；显式给出时必须为有限正数且不超过 Gateway task budget。`confirm-plan` 把该值以 float32
冻结进计划，`execute-plan` 必须传入**同一**值（float32 严格相等），否则协调器以 `SKILL_REQUEST_ID_CONFLICT`
拒绝——这是为了防止执行阶段悄悄放大或缩小展示并绑定过的任务预算。

当前 `execute` 和 `execute-plan` 可用 SIGINT/SIGTERM 请求取消；另一个进程分别使用 `cancel` 或携带完整
展示 tuple 的 `cancel-plan`。同一 Agent plan 不得同时使用 signal 和外部 `cancel-plan`。失败、timeout 或
停止状态未知时不得自动重试。`SKILL_CANCEL_TIMEOUT` 表示机器人停止状态未知，不能表述为“已停止”。

`validate` 与 `execute` 先以规范 payload 请求 Gateway：字符串去空白、方向小写，未提供 timeout 时使用
Gateway 默认值；timeout 必须为有限正数且不超过 Gateway task budget。payload hash 是规范 JSON 的 SHA-256。
`execute` 的 ROS goal UUID，以及 `cancel --task-id` 使用的 CancelGoal UUID，都是同一 task ID 的
UUIDv5（`ibrobot:{task_id}`）。`cancel` 只对 ledger 中 `active` task 发取消并轮询 `terminal`；已终态 task
返回 `already_terminal`，未知 task 返回 `GOAL_NOT_FOUND`。

## 输出契约

除 `execute`、`execute-plan`、`run-workflow` 和 `ibrobot-perceive` 外，命令向 stdout 输出单行 JSON envelope：

```json
{"command":"status","data":{},"error":null,"ok":true,"schema_version":1}
```

`ibrobot-perceive` 是有意例外：它直接打印请求字段的裸字面量值（如 `0.5236`）供 LLM 读取并
注入 `workflow_json`，错误信息走 stderr；详见上文「命令」表格。

`execute` 与 `execute-plan` 输出 JSONL：零到多条 `feedback`，最后恰好一条 `result`。每行都包含 `task_id` 和
`payload_hash`；公开结果只提供 `executed_step_count`，不暴露 primitive、pose 或 joint 名称。

```json
{"data":{"detail":"step 1 of 1","state":"executing"},"event":"feedback","payload_hash":"...","schema_version":1,"task_id":"task-20260725-001"}
{"data":{"error_code":"","executed_step_count":1,"message":"skill completed","success":true},"event":"result","payload_hash":"...","schema_version":1,"task_id":"task-20260725-001"}
```

`run-workflow` 也使用版本化 JSONL：先输出一条 `workflow_presentation`，最后输出一条
`workflow_terminal`。两条记录都包含 `schema_version`、`command` 和 `task_id`；展示记录必须包含完整
步骤、plan digest、registry identity 和 execution mode。

普通命令错误也使用同一 JSON envelope 的 `error.code` 与 `error.message`。当 Gateway capability reason
为 `CODE: detailed message` 时，CLI 用第一个冒号前的文本作为 `error.code`；`error.message` 保留原始
reason（没有冒号时两者都是该 code）。因此调用方应按 `error.code` 做稳定分类，不要解析详细消息。

| 退出码 | 含义 |
|---:|---|
| `0` | 成功 |
| `2` | 参数、配置或 schema 错误 |
| `3` | Gateway、readiness 或 safety 拒绝 |
| `4` | ROS/Gateway 不可用 |
| `124` | timeout，可能包含停止状态未知 |
| `130` | direct `execute` 的 SIGINT 取消已收敛到 terminal |
| `143` | direct `execute` 的 SIGTERM 取消已收敛到 terminal |

上述 `3/4/124/130/143` 是 direct `validate/execute/cancel` 的兼容契约。Agent plan
命令（`plan-workflow`、`validate-plan`、`confirm-plan`、`execute-plan`、`cancel-plan`）使用
以下稳定分组：`10` 表示 package/profile/implementation 不存在，`11` 表示 schema/reference/limit
校验失败，`13` 表示 registry/workflow/admission/finalization 状态错误，`14` 表示 ROS transport
不可达，`15` 表示 timeout/budget/deadline 或停止状态未知。

## 授权边界

`authorize_motion` 是唯一运行时运动授权来源，launch 默认值为 `false`。CLI 和 Agent 不能启动或重启
pipeline、开启授权、修改 ROS 参数，或从 YAML 推导授权。只有操作员完成现场安全检查后，才能在启动
pipeline 时显式授权。未授权状态下仍可使用 catalog、`status` 和不执行动作的本地检查；Gateway 会拒绝运动。

`robot_mcp` 兼容层已移除。Hermes 推荐配置和示例统一通过 `robot-skill` 访问 Capability Gateway。

## Hermes 启动器

完整构建后可运行：

```bash
hermes-robot --config-name so101_single_arm
hermes-robot --config-name lekiwi_handeye_realsense_grasp -- --cli
hermes-robot --config-name lekiwi_handeye_realsense_grasp_pc -- --cli
hermes-robot --config-name so101_single_arm --mode visual-games
hermes-robot --config-name so101_single_arm --mode motion
hermes-robot --config-name so101_single_arm --mode both
```

启动器要求 Hermes Agent `0.16.0` 或更新版本，并在启动前验证 `hermes`、`robot-skill`、安装空间中的
`ibrobot-control`、目标 robot config 以及所选控制面的 ROS 接口。`--mode visual-games` 只预检独立的
`StartVisualGame` / `GetVisualGameResult` service，不要求 MoveIt、Skill Gateway 或 Agent plan interfaces；
`--mode motion` 预检 Gateway status 和全部 Agent plan service/action；`--mode both` 同时预检两套接口。
默认 `--mode auto` 按配置能力选择：仅有已启用视觉游戏时预检视觉游戏接口，仅有运动 catalog 时维持运动
预检，两者都有时预检两套接口。显式 `--mode` 可用于只检查其中一个控制面。启动时会将安装空间中的
`ibrobot-control` 幂等注册到当前 Hermes profile 的 `skills/` 目录；仅更新带有
`robot_skill_cli` 所有权标记的副本，遇到同名的用户自管 skill 时会以 `AGENT_SKILL_CONFLICT` 退出。
`motion_authorized=false` 不阻止 Hermes 启动，只会继续由 Gateway 拒绝运动。启动器仅设置精确
`ROBOT_CONFIG` 并预加载 `ibrobot-control`；它不会启动/重启 pipeline、修改 ROS 参数或开启运动授权。
进入 Hermes 会话后只调用启动器注入到 `PATH` 的 `robot-skill`，不得再传 `--config-name` 或
`--config-path`。自然语言抓取与其他 motion Skill 使用统一的 `run-workflow` 复合入口；该入口内部完成
`status -> catalog -> plan -> validate -> present/flush -> confirm -> execute` 生命周期；
抓取计划使用 `pick_object` 和必填的 `target_name`，Gateway 再将其委派给配置绑定的 `grasp_pipeline`。

运动 catalog 的 `reload-catalog` 是显式、受控的运行时 snapshot 切换，不是自动监听文件。视觉游戏当前
不支持热加载：YAML 配置变更需要重启 pipeline，Python handler 变更要重新构建并重启；
`reload-catalog` 不会更新视觉游戏。

## raw-ROS 拦截 hook

`resource/hermes/hooks/ibrobot-block-raw-ros` 是 Hermes `pre_tool_call` hook，从 stdin 解析 tool
payload 并用 `shlex` 分词，阻断任何裸 `ros2` 子命令和 `rclpy`/`roslaunch` 间接调用，强制 LLM 走
`ibrobot-perceive`（感知读取）或 `robot-skill`（运动控制）。它是 defense-in-depth，权威边界仍是
`authorize_motion` 和 Gateway plan validation。策略全文见
[`resource/hermes/POLICY.md`](resource/hermes/POLICY.md)。

## Hermes 集成同步

`hermes-robot-configure` 是 IB-Robot Hermes 集成的幂等 profile 同步命令，把 Skill、SOUL 策略和
speech hook 一次性写入当前 Hermes profile。它取代了手工维护的 `SOUL.md`、`POLICY.md`、
`ibrobot-control` Plugin 和 TTS hook 副本，作为 `robot_skill_cli/resource/hermes/` 唯一发布源的
落地入口。

```bash
hermes-robot-configure \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --restart-gateway
```

也可以使用安装到 `share/robot_skill_cli/hermes/sync_hermes.sh` 的 shell 入口（等价于
`hermes-robot-configure`），适合源码场景或无 `install/setup.bash` 时使用：

```bash
bash src/robot_skill_cli/resource/hermes/sync_hermes.sh \
  --config-name so101_single_arm --dry-run
```

同步会执行以下动作（`--dry-run` 仅预览不写文件）：

- 调用 `resolve_robot_config_path()` / `load_robot_config()` 从 `robot_config` SSOT 读取
  `voice_tts` 服务名、playback 服务名、synthesis 超时和 playback 超时；严禁硬编码 ROS 服务名。
  其中 `synthesis_timeout_sec` 是 hook 等待 `/voice_tts/synthesize` 返回合成结果的超时，
  `playback_timeout_sec` 是等待 `/voice_tts/play` 完成播放的超时；`tts_timeout_sec` 是
  `embodied_agent` 视觉游戏播报节点的内部 RPC 超时，三者语义不同，互不替代。
- 验证当前 `ROS_DOMAIN_ID`（0-232）和 workspace `.shrc_local`（ROS+venv+install overlay 的
  SSOT 入口，见 `AGENTS.md`）。
- 把 `resource/hermes/SOUL.md`（`--soul-mode replace`）或 `POLICY.md` 托管区块（`--soul-mode merge`）
  写入 profile/SOUL.md，`skip` 不修改 SOUL；`replace` 会先备份。
- 把 `ibrobot-control` Skill 幂等注册到 `profile/skills/`，仅替换带 `robot_skill_cli` 所有权标记
  的副本，遇到同名用户自管 skill 时以 `AGENT_SKILL_CONFLICT` 退出。
- 生成 `robot-skill`、最终回复 speech、生命周期 speech 和 interim speech wrapper，以及一个环境文件；
  `robot-skill`、最终回复 speech 和 interim wrapper source workspace `.shrc_local`；lifecycle wrapper
  依赖 Hermes 已初始化的 workspace 环境，不硬编码 `/opt/ros/humble/setup.bash` 或 venv 路径。
  interim wrapper 还绑定当前 Hermes profile 的 `ibrobot-speak` 路径。
- 安装 `post_llm_call`、生命周期和 `on_interim_message` speech hook，移除对应的旧 managed 副本；
  `--disable-speech` 移除全部受管 speech hook。
- 移除已退役的 `ibrobot-robot-control` Plugin 副本（即时执行改由 `robot-skill run-workflow` 的
  显式 `immediate_after_presentation` 模式承担），并清理 `external_dirs` 缓存目录下同名的 stale skill 副本。
  Plugin 目录只在带 `hermes-robot-configure` managed 标记时才会被 quarantine（重命名），
  不删除无标记或用户自管内容。
- `--accept-hooks` 先调用 `hermes hooks revoke` 清理旧 mtime（首次安装无既有审批时经
  `hermes hooks list` 确认未注册则跳过该失败），再用 `hermes --accept-hooks hooks doctor`
  重新批准 hook；`--restart-gateway` 调用 `hermes gateway restart`。

`hermes-robot-speak` 是 `post_llm_call` hook 的 Python 入口（由 `ibrobot-speak` shell wrapper 调用），
从 stdin 读取 Hermes hook payload，提取最终 assistant 回复，经 `sanitize_for_tts` 移除 ASCII
字母 run 和连续 7 位以上数字（避免播报技术标识符；ZipVoice 中文前端不发音英文），调用
`ibrobot_msgs.srv.SynthesizeSpeech` 合成 WAV
并通过 `PlayAudioFile` 服务播放。TTS 是系统自动功能，不是机器人 Skill，不进入 `workflow_json`。
诊断日志位于 `/tmp/hermes-speak.log`（受 `IBROBOT_HERMES_TTS_LOG` 覆盖）。

详细同步步骤、前置条件和升级流程见
[`resource/hermes/README.md`](resource/hermes/README.md)。

SO-101 真机的完整手动验证步骤见
[`docs/hermes_so101_real_robot_manual_validation_zh.md`](../../docs/hermes_so101_real_robot_manual_validation_zh.md)。

# skill_library 节点说明

`skill_library` 是当前具身执行链路里的**技能执行层**。  
它不直接做任务理解，而是负责把上层给出的技能请求拆成有限 primitive，或委托给配置声明的受保护执行器，
再桥接到底层机械臂和夹爪控制接口。

当前包内包含 1 个 ROS 2 节点：

| 节点 | 控制台入口 | 主要职责 |
| --- | --- | --- |
| `skill_executor_node` | `skill_executor_node = skill_library.skill_executor_node:main` | 提供技能/primitive action；位姿与夹爪动作交给 `task_dispatch`，关节轨迹发送到控制器 action |

## 1. 现在可以如何控制机械臂

当前有 2 种主要控制方式，都是通过 `skill_library` 最终落到真实执行：

| 控制方式 | 入口 | 适合场景 |
| --- | --- | --- |
| 技能级控制 | `/embodied/execute_skill` | 明确指定技能名，做稳定、可控的动作编排 |
| primitive 级控制 | `/embodied/execute_primitive` | 直接控制命名位姿、相对位移、关节轨迹、夹爪开合 |

整体链路如下：

```text
Hermes / robot-skill
  -> agent_plan_node
  -> /embodied/execute_skill
  -> skill_executor_node
  -> /embodied/execute_primitive
  -> pose/rotation/gripper: /task_executor/execute_task_plan
  -> joint trajectory: /arm_trajectory_controller/follow_joint_trajectory
```

## 2. 技能来源：versioned skill catalog snapshot

技能不再由 `robot_config` 的 `skill_templates` 内联注入。`skill_executor_node` 现在是
versioned skill catalog runtime owner：启动时通过 `SkillCatalogCompiler` 从配置的 source
（`skill_catalog_source_mode` = `installed` / `development` / `production`，root 由
`skill_catalog_source_root` 指定，profile 由 `skill_catalog_profile` 指定）编译出 immutable
`RuntimeSnapshot`，并由 `SkillRegistryOwner` 管理 epoch/generation 生命周期。

每个 snapshot 携带 exact registry identity 三元组 `(registry_epoch, generation, registry_digest)`，
以及 `capability_digest`、`source_release_digest`、`provenance_digest`、`primitive_contract_digest`
和 `robot_context`。`robot_context` 中的 `named_poses`、`named_targets`、`workspace_limits`、
`arm_joint_names`、`joint_limits`、`timeout_policy` 等字段在编译时由节点启动参数
（`named_poses_json`、`named_targets_json`、`workspace_json`、`arm_joint_names_json`、
`joint_limits_json`）作为 `SkillCompileContext` 烘焙进 snapshot，运行时校验直接读 snapshot，不再读
节点参数。`skill_templates_json` 参数已移除；`self._skill_templates` 由
`startup_bundle.snapshot.templates` 填充并在 reload 后替换。

启动后 runtime 通过以下边界同步 exact snapshot：

- `GetSkillSnapshot.srv`（`/embodied/get_skill_snapshot`）：exact-version 查询，`generation>0` 必须精确匹配，
  不静默升级；已被回收返回 `SKILL_SNAPSHOT_NOT_RETAINED`。
- `ReloadSkillCatalog.srv`（`/embodied/reload_skill_catalog`）：从配置 source 重新编译；`force=true` 仅强制
  重编译，digest 未变时 generation 不得人为递增。
- `SkillRegistryEvent.msg`（`/embodied/skill_registry_events`，RELIABLE/TRANSIENT_LOCAL/KEEP_LAST depth 1）：
  成功 reload 后发布，告知晚加入者查询哪个 epoch/generation。
- `GetSkillGatewayStatus.srv`（`/embodied/get_skill_gateway_status`）：状态边界，含 `control_plane_ready`、
  `registry_epoch/generation/digest`、`retained_generations` 等。

`embodied_common` 的 `get_skill_templates()` 仍存在，但 Gateway 不再使用它作为运行时模板来源；当前
机器人实际技能集合以编译后的 snapshot `templates` 为准，不应在本 README 另建固定白名单。

SO101 当前配置示例：

| 类别 | 技能 |
| --- | --- |
| 观察与恢复 | `inspect_scene`、`recover_safe_pose`、`recover_zero_pose` |
| 参数化动作 | `move_relative_ee`、`rotate_gripper_cw`、`rotate_gripper_ccw` |
| 夹爪 | `open_gripper_skill`、`close_gripper_skill` |
| 社交与娱乐 | `dance_basic`、`wave_hello`、`nod_yes`、`shake_no`、`celebrate`、`greet_observe_raise`、`act_cute`、`happy_spin_upright` |
| 抓取 | `pick_object`（仅抓取配置，要求显式传入 `target_name`） |
| 抓取与放置 | `pick_object`、`place_in_container`（固定位置释放，要求显式传入 `target_name` 和 `container_name` 做验证） |
| 人机交互模拟执行 | `imitate_human_motion`（要求 `arm_side` 和 `imitation_duration_sec`） |

## 3. 当前支持的 primitive

`skill_library` 只允许有限 primitive，避免上层直接下发任意危险动作：

| primitive | 作用 |
| --- | --- |
| `move_to_named_pose` | 移动到命名位姿 |
| `move_to_pose` | 移动到经过安全校验的动态 base-frame 位姿 |
| `move_to_configuration` | 经 MoveIt 移动到经过校验的完整机械臂关节配置 |
| `move_relative_ee` | 相对当前末端位姿做笛卡尔增量移动 |
| `move_to_joint_positions` | 按完整手臂关节顺序移动到单个关节目标 |
| `move_through_joint_positions` | 按完整手臂关节顺序执行多路点关节轨迹 |
| `open_gripper` | 张开夹爪 |
| `close_gripper` | 闭合夹爪 |
| `rotate_gripper_cw` | 绕当前末端局部 Z 轴顺时针旋转 |
| `rotate_gripper_ccw` | 绕当前末端局部 Z 轴逆时针旋转 |
| `nav_straight` | V2 request schema；V2/V3 context 可用；按方向枚举直行/横移指定距离，委托 `ExecuteNavigation` action |
| `nav_turn` | V2 request schema；V2/V3 context 可用；按 `turn-left` / `turn-right` 方向旋转指定角度，委托 `ExecuteNavigation` action |
| `nav_abs_coordinate` | V2 request schema；V2/V3 context 可用；导航到 `map` frame 绝对坐标 (x, y, yaw)，委托 `ExecuteNavigation` action |

`nav_*` 三个 primitive 委托到 `/navigation/execute`（`ibrobot_msgs/action/ExecuteNavigation`）。
它们只在 V2/V3 context 中可用；V1 请求携带 `nav_*` primitive 名直接返回
`SKILL_SCHEMA_INVALID`，不会进入 dispatch 阶段。详见
[导航 action dispatch](#9-导航-action-dispatch)。

## 4. 技能到 primitive 的映射方式

当前不是硬编码大分支，而是**模板驱动**：

- 模板来源：编译后的 `RuntimeSnapshot.templates`（由 `SkillCatalogCompiler` 从配置 source 生成）
- robot context：`named_poses` / `named_targets` / `workspace_limits` / `arm_joint_names` / `joint_limits`，编译时由节点启动参数烘焙进 snapshot
- 运行时校验直接读 snapshot 的 `robot_context`，不再读节点参数

### 4.1 夹爪归一化（initial_gripper_state）

模板可声明 `initial_gripper_state`（`open` / `closed` / `hold` / `none`），使技能
执行前显式归一化夹爪状态。例如 `wave_hello` 配置 `initial_gripper_state: closed`
后，实际 primitive 序列会自动在最前面插入一条 `close_gripper`。

- 纯夹爪技能（`open_gripper_skill` / `close_gripper_skill`）不应声明该字段，避免冗余。
- 非法值（如 `half_open`）会在 resolve 阶段抛出 `ValueError`。

例如：

| 技能 | primitive 序列 |
| --- | --- |
| `wave_hello` | `close_gripper` -> `move_to_joint_positions(entry)` -> `move_through_joint_positions(joint_waypoints)` -> `move_to_joint_positions(return)` |
| `recover_zero_pose` | `move_to_named_pose(zero)` |
| `move_relative_ee` | `move_relative_ee(direction, distance)` |
| `dance_basic` | `close_gripper` -> `move_to_joint_positions(entry)` -> `move_through_joint_positions(joint_waypoints)` |
| `celebrate` | `close_gripper` -> `move_to_named_pose(observe_table)` -> 多个 `move_relative_ee` -> `move_to_named_pose(observe_table)` |

`pick_object` 不展开静态 `named_targets` 位姿，而是委托给
`/manipulation/execute_pick`。GraspGen 在运行时生成动态 6-DOF 候选，执行器再通过安全 primitive
完成 approach、补偿下降、夹爪闭合和到 place container 位姿的运输。

`place_in_container` 同样只允许从 `/embodied/execute_skill` 进入。Gateway
将完整 `DispatchBinding` 和 `placement_pipeline` identity 传给
`/manipulation/execute_place`；放置执行器再把相同 binding 传给受保护的
`move_to_joint_positions`/`open_gripper` primitive。放置阶段先在 3 号电机 raw 1500 开爪，再移动到 raw 1700
（`verify_joint_position: -0.533825`）
验证，最后返回 raw 1500；视觉验证失败或不确定时，
结果会明确保留“已释放/释放状态未知”，不会把它误报为普通未执行失败。

## 5. 直接控制机械臂的几种用法

### 5.1 直接发技能 action

适合调试 skill 级执行，不经过自然语言解析。

动作接口：

- `/embodied/execute_skill`
- 类型：`ibrobot_msgs/action/SkillCommand`
- goal 携带 `dispatch_binding`（`DispatchBinding`）+ `skill_name` 等参数；`task_id` 在 binding 内

`SkillCommand` goal 必须提供完整期望 registry identity（`expected_registry_epoch/generation/digest`）。
手写 `ros2 action send_goal` 需要构造完整的 `dispatch_binding`（含 `task_budget`），既繁琐又容易与
当前 snapshot 不一致。推荐使用 `robot-skill execute`：CLI 会先查询 `GetSkillGatewayStatus` 取回当前
exact identity，再构造 direct root binding（`task_id == root_task_id`、零值 `task_budget`、
`workflow_step_index=0`、两个 nonce 为空），由 Gateway 在 admission 时 canonicalize 为
`task_budget_sec` 预算（canonical zero-budget direct root）。

常用 goal 字段：

| 字段 | 说明 |
| --- | --- |
| `dispatch_binding.task_id` / `root_task_id` | direct root 两者相等 |
| `dispatch_binding.task_budget` | direct root 可为零值，Gateway 在 admission 时 stamp canonical 预算 |
| `dispatch_binding.expected_registry_epoch/generation/digest` | exact snapshot identity |
| `dispatch_binding.schema_version` | wire contract schema version，仅接受 `{1, 2}`；由 `validate_request_schema_version` 校验，与请求体顶层 `schema_version` 必须一致 |
| `schema_version` | 请求体顶层 schema version，仅接受 `{1, 2}`；V1 不得携带 navigation_* 字段，V2 由导航技能下发；与 `dispatch_binding.schema_version` 必须一致 |
| `skill_name` | 技能名，如 `pick_object` |
| `target_name` | 目标引用；对 `pick_object` 表示运行时视觉文本查询，如 `banana` |
| `container_name` | 对 `place_in_container` 表示释放后用于视觉验收的容器文本查询，如 `black bowl`；不改变放置轨迹 |
| `place_name` | 命名放置位，如 `tray_right` |
| `motion_direction` | 相对运动方向 |
| `motion_distance` | 相对运动距离 |
| `timeout_sec` | per-entry 超时，受共享 `task_budget` 剩余预算约束 |
| `direction` | V2 导航方向枚举（`forward`/`backward`/`leftward`/`rightward`/`turn-left`/`turn-right`）；仅 `nav_*` 技能下发 |
| `distance` | V2 直行距离 (m)，仅 `nav_straight` 必填，必须为有限正数 |
| `degree` | V2 转向角度 (deg)，仅 `nav_turn` 必填，必须为有限数 |
| `x` | V2 绝对坐标 X (m)，仅 `nav_abs_coordinate` 使用 |
| `y` | V2 绝对坐标 Y (m)，仅 `nav_abs_coordinate` 使用 |
| `yaw` | V2 绝对坐标 yaw (rad)，仅 `nav_abs_coordinate` 使用 |
| `has_x` | V2 presence flag，标识 `x` 是否由调用方显式提供 |
| `has_y` | V2 presence flag，标识 `y` 是否由调用方显式提供 |
| `has_yaw` | V2 presence flag，标识 `yaw` 是否由调用方显式提供 |

例如挥手：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
robot-skill execute wave_hello --task-id demo-wave
```

例如让末端向前移动 3 cm：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
robot-skill execute move_relative_ee --task-id demo-move-forward \
  --motion-direction forward --motion-distance 0.03
```

result 会携带 `actual_registry_epoch/generation/digest`、`source_release_digest`、`provenance_digest`
和 `diagnostics`，无论成功或失败都反映实际使用的 snapshot identity。


### 5.3 直接发 primitive action

适合最低层调试。这是 external direct root primitive：goal 必须携带 `dispatch_binding`，提供完整
期望 registry identity；`task_budget` 可为零值（canonical zero-budget direct root），Gateway 在
取得排他 root lease 后 canonicalize binding（stamp `task_budget` 为 `task_budget_sec`、生成
`dispatch_nonce`）。identity 与当前 snapshot 不匹配返回 `SKILL_REGISTRY_VERSION_MISMATCH`。

动作接口：

- `/embodied/execute_primitive`
- 类型：`ibrobot_msgs/action/PrimitiveCommand`

`dispatch_binding` 公共部分（替换 `<epoch>`/`<generation>`/`<digest>` 为 `GetSkillGatewayStatus`
返回的当前 identity）：

```yaml
dispatch_binding:
  schema_version: 1
  task_id: "demo-home"        # 与 root_task_id 相等
  root_task_id: "demo-home"
  task_budget: {schema_version: 0}   # 零值，Gateway canonicalize
  expected_registry_epoch: "<epoch>"
  expected_registry_generation: <generation>
  expected_registry_digest: "<digest>"
  workflow_digest: ""
  workflow_step_index: 0
  root_lease_nonce: ""
  dispatch_nonce: ""           # Gateway 在 admission 后 stamp
```

例如直接去某个命名位姿：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  dispatch_binding: { schema_version: 1, task_id: "demo-home", root_task_id: "demo-home",
    task_budget: { schema_version: 0 },
    expected_registry_epoch: "<epoch>", expected_registry_generation: <generation>,
    expected_registry_digest: "<digest>", workflow_step_index: 0 },
  primitive_name: "move_to_named_pose",
  pose_name: "home",
  relative_dx: 0.0, relative_dy: 0.0, relative_dz: 0.0,
  gripper_position: 0.0, timeout_sec: 10.0
}'
```

例如让末端向前移动 3 cm：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  dispatch_binding: { schema_version: 1, task_id: "demo-forward", root_task_id: "demo-forward",
    task_budget: { schema_version: 0 },
    expected_registry_epoch: "<epoch>", expected_registry_generation: <generation>,
    expected_registry_digest: "<digest>", workflow_step_index: 0 },
  primitive_name: "move_relative_ee",
  relative_dx: 0.03, relative_dy: 0.0, relative_dz: 0.0,
  gripper_position: 0.0, timeout_sec: 10.0
}'
```

例如直接张开夹爪：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  dispatch_binding: { schema_version: 1, task_id: "demo-open", root_task_id: "demo-open",
    task_budget: { schema_version: 0 },
    expected_registry_epoch: "<epoch>", expected_registry_generation: <generation>,
    expected_registry_digest: "<digest>", workflow_step_index: 0 },
  primitive_name: "open_gripper",
  gripper_position: 1.0, timeout_sec: 5.0
}'
```

例如直接执行一段关节路点轨迹：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  dispatch_binding: { schema_version: 1, task_id: "demo-joint-waypoints", root_task_id: "demo-joint-waypoints",
    task_budget: { schema_version: 0 },
    expected_registry_epoch: "<epoch>", expected_registry_generation: <generation>,
    expected_registry_digest: "<digest>", workflow_step_index: 0 },
  primitive_name: "move_through_joint_positions",
  joint_names: ["1", "2", "3", "4", "5"],
  joint_waypoints: [0.02, 0.54, -0.82, -0.18, 0.02],
  joint_waypoint_count: 1,
  waypoint_duration_sec: 0.5,
  timeout_sec: 10.0
}'
```

primitive result 在成功、失败、abort、cancel cleanup timeout 和 lease finalization failure 各路径
都携带 `actual_registry_epoch/generation/digest`，反映实际使用的 snapshot identity。


## 6. primitive 如何桥接到底层

### 位姿、旋转与夹爪控制

以下 primitive 会被转换为单步 `ibrobot_msgs/action/ExecuteTaskPlan` goal，并发送到
`/task_executor/execute_task_plan`：

- `move_to_named_pose`
- `move_relative_ee`
- `rotate_gripper_cw`
- `rotate_gripper_ccw`
- `open_gripper`
- `close_gripper`

`task_dispatch` 负责继续路由到底层执行接口。`SkillExecutorNode` 当前不直接向
`/cmd_pose` 或夹爪控制 topic 发布这些动作。

### 手臂关节轨迹控制

`move_to_joint_positions` / `move_through_joint_positions` 最终都会发送 action 到：

- `/arm_trajectory_controller/follow_joint_trajectory`

消息类型：

- `control_msgs/action/FollowJointTrajectory`

关节名顺序来自 `robot_config` 注入的 `arm_joint_names_json`，关节限位由
`joint_limits_json` 传给安全层校验。

## 7. 相对移动语义

`move_relative_ee` 的方向语义由 `robot_config` 提供，当前默认是 base 坐标系：

| 中文语义 | 方向 | 默认增量方向 |
| --- | --- | --- |
| 前 | `forward` | `+x` |
| 后 | `backward` | `-x` |
| 左 | `left` | `+y` |
| 右 | `right` | `-y` |
| 上 | `up` | `+z` |
| 下 | `down` | `-z` |

规则解析默认步长来自：

- `embodied.execution.relative_motion_step_m`

当前默认值通常为：

- `0.03 m`

## 8. 安全联动

`skill_library` 不会绕过安全层，但两个 action 入口的校验路径不同：

- 高层 `SkillCommand` 先经过 Gateway 准入，再调用 `/embodied/validate_skill` 校验整个 skill；在父 admission
  有效时，解析出的每个内部 primitive 复用父 `SkillCommand` 的 borrowed lease，经
  `/embodied/execute_primitive`，并在下游 dispatch 前各自调用 `/embodied/validate_primitive`。
- `pick_object` 的抓取 pipeline 运行在独立 action server 中。Gateway 为该 `PickObject` action goal UUID
  注册一次性内部授权；pipeline 将它编码为不透明 `PrimitiveCommand.execution_token`，使观察位和后续
  primitive 借用同一个 root lease。仅复用 `task_id` 不会获得内部授权。
- 直接/外部 `PrimitiveCommand` 的 Gateway primitive 准入只检查运动授权、所需控制模式，以及 execution
  lease/busy 状态。对相对移动和夹爪旋转，随后才取得并检查新鲜的 EE state；
  `/embodied/validate_primitive` 再校验该 primitive 的最终 target/request。下游 action/server readiness 和
  已捕获 EE pose 的最终新鲜度在各自实际 send 边界前检查，而不是在校验前使用单一、通用的 readiness gate。
  直接/外部 primitive 不会额外调用 `/embodied/validate_skill`。

admission 还会校验 `dispatch_binding` 的 exact registry identity：`schema_version` 通过
`validate_request_schema_version` 校验、必须落在接受集合 `{1, 2}` 内且与请求体顶层 `schema_version`
一致、`task_id == root_task_id`、
`workflow_step_index=0`（root-scope canonical sentinel）、两个 nonce 为空、`expected_registry_epoch/generation/digest`
与当前 runtime bundle 完全匹配，否则返回 `SKILL_REGISTRY_VERSION_MISMATCH`（binding 结构非法返回
`SKILL_SCHEMA_INVALID`）。V1 请求携带任何 navigation_* 字段、或 V2 请求在
`context_schema_version=1` 的 snapshot 上执行 nav_* primitive，同样以 `SKILL_SCHEMA_INVALID`
拒绝。direct root `SkillCommand` 和外部 `PrimitiveCommand` 允许零值 `task_budget`：
Gateway 成为 owner，在 admission 时 stamp canonical `task_budget_sec` 预算（canonical zero-budget direct root）；
非零 budget 必须满足当前 policy，否则返回 `SKILL_SCHEMA_INVALID`。请求 `timeout_sec` 或 effective default
超过剩余 task budget 时返回 `TIMEOUT_EXCEEDS_POLICY`，不静默 clamp。

如果任一步被拒绝：

1. 当前 skill / primitive 立即停止
2. 不向下发送动作
3. 返回明确的 `error_code` 和 `message`

在调用安全校验之前，Gateway 还要求启动时只读参数 `motion_authorized=true`，并要求实际
`active_control_mode` 等于 SSOT 的 `skill_required_control_mode`。运动授权默认关闭，且只能由
`embodied_pipeline.launch.py` 的操作员 launch argument 注入；机器人 YAML、Agent、CLI 和动态参数
都不能开启或修改授权。未授权请求返回 `MOTION_NOT_AUTHORIZED`，控制模式不匹配返回
`CONTROL_MODE_MISMATCH`，两种情况都不会向 primitive 层发送动作。

### 8.1 Gateway 准入、readiness 与新鲜度

Gateway 先规范化 task ID 与公开 skill payload，再按固定顺序评估：操作员授权、控制模式、现有 root
执行 busy 状态、exact registry identity、task budget 和该技能的 runtime readiness。只有通过这些检查后，才会在本 Gateway
进程内原子地查询 ledger、取得 root execution lease，并创建 active ledger 记录；相同 task ID/hash 返回
`DUPLICATE_TASK_ID`，同一 task ID 的不同 hash 返回 `TASK_ID_CONFLICT`。lease 只协调本进程内的
root skill 或外部 primitive，不是 ROS graph 范围的锁。

`SkillCommand` 被拒绝时保留 admission 的原始 `error_code` 和 `message`，不会改写成笼统的安全错误。
典型返回为：`MOTION_NOT_AUTHORIZED: operator authorization is disabled`、
`CONTROL_MODE_MISMATCH: requires <required>, active mode is <active>`、
`SKILL_BUSY: another root execution is active`，以及 `CAPABILITY_NOT_READY` 加以下首个缺失原因：
`validate skill service unavailable`、`task executor action unavailable`、
`arm trajectory action unavailable`、`ee pose unavailable or stale`。

末端位姿回调在状态锁内同时记录消息和 monotonic receipt 时间；相对移动与夹爪旋转会从同一锁内深拷贝
一个原子 snapshot。若该 snapshot 在 primitive 安全校验前已经过期，会以
`CAPABILITY_NOT_READY: ee pose unavailable or stale` 返回，既不调用 `ValidatePrimitive`，也不向下游
发送动作。若它在校验或 task executor readiness 等待期间过期，发送前会再次检查，仍在下游 dispatch 前
失败。Gateway 不会对此类失败或任何动作失败自动重试。

### 8.2 UUID 维度

Gateway 涉及三类 UUID，分工不同，不要混淆：

- **task identifier（UUIDv5）**：`embodied_common.skill_request.skill_goal_uuid(task_id)` 派生
  `ibrobot:{task_id}` 的 UUIDv5，用于 ledger key、payload hash 关联，以及 `cancel --task-id` 的
  `CancelGoal.goal_id`。CLI 的 `execute` 与 `cancel` 共用同一 task ID 的 UUIDv5。
- **internal primitive ROS action goal_id（UUIDv4）**：`skill_executor_node` 在派发内部 primitive 时
  用 `uuid.uuid4()` 随机生成 ROS action `goal_id.uuid`，仅用于 rclpy 跟踪 `goal_handle`，不暴露给
  Agent，也不参与 ledger 关联。随机 UUID 避免 goal ID 冲突，是 rclpy 的实现细节。
- **pick handoff goal_id（UUIDv4）**：Gateway 派发 `PickObject` 时生成并注册。抓取 executor 将其编码到
  `PrimitiveCommand.execution_token`；Gateway 只有在 token、task ID 和当前 admission 全部匹配时才发放
  borrowed lease。正常终态或已确认的取消清理会撤销该授权。

因此 PR 自述的「确定性 goal UUID」专指 task identifier 维度；internal primitive 的 ROS action
goal_id 不在此约定范围内。

### 8.3 Workflow 执行：Begin / child SkillCommand / Finalize

多步 plan 不由 `SkillCommand` 直接展开，而是走 typed Workflow 协议。embodied_agent task executor
通过两个内部 execution-scope service 与 ExecutionCoordinator 交互（不是 Agent/CLI 可发现的 capability）：

- `BeginWorkflowExecution.srv`（`/embodied/begin_workflow_execution`）：请求必须满足
  `task_id == root_task_id`、非空 steps、完整期望 identity、非空 `workflow_digest`、
  `workflow_step_index=0`、两个 nonce 为空。Coordinator 重算 digest、捕获当前 bundle 并原子取得
  root lease，返回 `root_lease_nonce` 和 actual registry identity。相同 root ID/digest/identity/budget 的
  重试返回同一 active nonce；任一字段冲突返回 `SKILL_REQUEST_ID_CONFLICT` 且不替换既有 root。
- `FinalizeWorkflowExecution.srv`（`/embodied/finalize_workflow_execution`）：幂等终态化。请求携带
  Begin 返回的 `root_lease_nonce` 和相同 digest/identity/budget，`workflow_step_index=0`，`dispatch_nonce`
  为空。Coordinator ledger 的 completed index 是权威的，caller count 仅用于冲突检测。不同 terminal
  state/count 返回 `SKILL_REQUEST_ID_CONFLICT`；不同 digest/nonce 返回对应错误；任一冲突都不释放 root lease。

每个 Workflow child `SkillCommand` 携带 `root_lease_nonce`、实际零基 `workflow_step_index`、相同的
root/budget/identity 和 `workflow_digest`。Gateway 按严格零基顺序校验 child：重放、跳步、越界、参数篡改、
错误 digest、错误 root nonce 或该 step 已非 `pending` 都返回对应冲突错误。Hermes 和 `robot-skill` 不接触
`root_lease_nonce`。

Workflow step 状态为 `pending` / `active` / `succeeded` / `failed` / `canceled`。child 到达 Gateway admission
后先进入 `active`，terminal 时再进入对应终态。若 root 在 child 到达 Gateway 前失败或取消，step 仍为
`pending`，允许 root 收敛；一旦进入 `active`，Finalize 必须匹配 child 的真实终态。

- 成功：置 `succeeded`，`completed_step_count += 1`。
- 失败或取消：置 `failed`（`SKILL_CANCELLED` 置 `canceled`）。
- 已 terminal 的 step 不允许再次提交。

### 8.4 未知 child 清理与 root lease 保留

当 child `SkillCommand` 的终态为 `SKILL_CANCEL_TIMEOUT` 或 primitive 的 `CANCEL_CLEANUP_TIMEOUT`
（即 `cleanup_unknown`）时，Gateway **不**清除 `_active_skill_admission`、`_active_skill_owner`、
`_active_audit_context`、`_active_runtime_bundle` 和 `_retained_admission_cleanup`，即 root lease **不被**
释放，runtime bundle retention **不被**回收。此时该 step 仍被记录为 `failed`/`canceled`，但 Workflow 无法
继续 Finalize 成功，Coordinator 不释放 root lease，直到操作员介入或后续 Finalize 以 `FAILED`/`CANCELED`
幂等终态化（ledger terminal record 先写、再释放 root lease 与 bundle retention）。这避免在下游清理状态
未知时把 root lease 交给下一个请求，但要求显式终态化才能恢复。

deadline 超时的 Workflow 由后台 reaper 读取：先原子写 ledger terminal record（`SKILL_TASK_DEADLINE_EXPIRED`），
再释放 bundle retention 与 root lease；写 record 与释放的顺序保证 cleanup 后相同 root ID 的重试能命中
terminal record。

### 8.5 委托型抓取共享绝对预算

`pick_object` 技能把入口委托给 `/manipulation/execute_pick`（`PickObject` action）。Gateway 在 dispatch 时
把同一 root 的 `dispatch_binding`（含共享 `task_budget` 和 exact identity）和 `expected_executor`
传给 delegated executor。delegated server 在 goal acceptance 时校验 `dispatch_binding.schema_version` 落在 `validate_request_schema_version` 接受集合 `{1, 2}` 内、
非空 `task_id`/`root_task_id`、完整期望 identity、`task_budget.schema_version` 与请求体一致、deadline 未过期且
`timeout_sec` 不超过剩余 budget；执行时用 `min(timeout_sec, deadline - now)` 作为实际预算，预算已过期
返回 `TASK_TIMEOUT` 并 abort。Hermes/catalog goal 的 `supervised_direct` 固定为 `false` 且必须有非空
delegated nonce；只有人工 bring-up client 可使用 `supervised_direct=true` 和空 nonce。详见
`manipulation_execution` README。

`imitate_human_motion` 使用同一套 delegated binding、executor identity、共享绝对 budget、取消和
late-cleanup 规则，转发到内部 `/hri/imitate_human_motion` Action。内部 runtime 只通过
`PrimitiveCommand` 执行 Mock 动画和 `move_to_named_pose(home)` 恢复；Agent 不直接调用该 Action，
也不读取 warmup 状态。

### 8.6 导航 action dispatch

`nav_straight` / `nav_turn` / `nav_abs_coordinate` 三个 primitive（V2 request schema，V2/V3 context
可用）的 dispatch 入口是
`/navigation/execute`（`ibrobot_msgs/action/ExecuteNavigation`）。该 action name 由
`robot_config.navigation.command_server.action_name` SSOT 提供，经
`robot_execution_endpoints.navigation_action_name` 投影到 `SkillRobotContext`，并随
`context_schema_version` 进入 canonical execution-context digest。

Gateway 在 dispatch 一个 nav_* primitive 前会做下列按顺序的 admission：

1. **wire contract 校验**：请求体 `schema_version` 与 `dispatch_binding.schema_version` 都必须等于 `2`，
   由 `validate_request_schema_version` 校验；任一为 `1` 而请求携带 navigation_* 字段，或
   `context_schema_version=1` 的 snapshot 上执行 nav_* primitive，返回 `SKILL_SCHEMA_INVALID`。
2. **context version 选择**：snapshot 的 `context_schema_version` 必须为 `2` 或 `3`，否则即使 wire
   `schema_version=2` 也以 `SKILL_SCHEMA_INVALID` 拒绝（防止 V1 snapshot 上通过 wire 字段绕过 contract）。
3. **控制模式**：V2 snapshot 要求 `active_control_mode == base_navigation`；V3 hybrid snapshot 在下发
   前通过 `motion_mode/set_navigation_enabled` 切换到 `base_navigation`。切换失败返回
   `CONTROL_MODE_MISMATCH`，不发送 ExecuteNavigation goal。
4. **navigation_ready**：Gateway 在 dispatch 前等待 `navigation_ready=true`，即
   `navigation_command_server` 已声明 Action 可发现且 `/global_costmap/costmap` 非空。该 readiness
   门不同于 motion 授权或 EE pose 新鲜度，专门用于导航 primitive。
5. **costmap readiness**：`navigation_command_server` 自身在向 Nav2 发送 `NavigateToPose` 前会用
   `costmap_readiness_timeout`（默认 60s）轮询 `/global_costmap/costmap`，超时仍未拿到非空数据时直接
   以 `NAV2_UNAVAILABLE` 终态返回，不会取得 root lease。Gateway 不会重试。
6. **ExecuteNavigation goal**：Gateway 把同一 root 的 `dispatch_binding`、共享 `task_budget` 与
   exact identity 透传到 `ExecuteNavigation.Goal`；执行器负责把
   `direction` / `distance` / `degree` 或 `(x, y, yaw)` 翻译成 Nav2 target。

`ExecuteNavigation` 是单任务 action，Gateway 一次只允许一个 nav_* primitive 在途；需要切换目标时，
调用方必须先等待 `ExecuteNavigation` 进入终态或调用 `/navigation/cancel_current` 成功。cancel 路径
详见 [§9 取消终态契约](#取消终态契约)。

## 9. 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `skill_action_name` | `/embodied/execute_skill` | 技能 action 名 |
| `primitive_action_name` | `/embodied/execute_primitive` | primitive action 名 |
| `validate_skill_service` | `/embodied/validate_skill` | 技能校验服务 |
| `validate_primitive_service` | `/embodied/validate_primitive` | primitive 校验服务 |
| `skill_gateway_status_service` | `/embodied/get_skill_gateway_status` | 启动时固定；Gateway 状态服务名 |
| `skill_catalog_reload_service` | `/embodied/reload_skill_catalog` | catalog reload 服务名 |
| `skill_catalog_snapshot_service` | `/embodied/get_skill_snapshot` | exact-version snapshot 查询服务名 |
| `begin_workflow_service` | `/embodied/begin_workflow_execution` | 内部 Workflow Begin 服务名 |
| `finalize_workflow_service` | `/embodied/finalize_workflow_execution` | 内部 Workflow Finalize 服务名 |
| `skill_registry_event_topic` | `/embodied/skill_registry_events` | registry 事件 topic（RELIABLE/TRANSIENT_LOCAL/KEEP_LAST depth 1） |
| `skill_catalog_source_mode` | `installed` | catalog source 模式：`installed`/`development`/`production` |
| `skill_catalog_source_root` | 空字符串 | `development`/`production` 模式下的 source root |
| `skill_catalog_profile` | 空字符串 | catalog profile 名 |
| `robot_name` | `unknown` | 启动时固定；capability view 中的机器人名称 |
| `motion_authorized` | `false` | 启动时固定；唯一运动授权值，默认拒绝运动 |
| `active_control_mode` | 空字符串 | 启动时固定；launch override 生效后的实际控制模式 |
| `skill_required_control_mode` | 空字符串 | 启动时固定；SSOT 要求的技能控制模式 |
| `named_poses_json` | `{}` | 命名位姿字典，编译时烘焙进 snapshot `robot_context` |
| `named_targets_json` | `{}` | 命名目标字典，编译时烘焙进 snapshot `robot_context` |
| `relative_motion_reference_frame` | `base` | 相对运动参考系 |
| `relative_motion_step_m` | `0.03` | canonical robot context 中的默认相对移动步长 |
| `relative_motion_direction_mapping_json` | `{}` | 相对运动方向映射 |
| `rpc_timeout_sec` | `5.0` | 启动时固定；等待校验服务 / primitive action 的统一 RPC 超时，并参与 capability digest |
| `gripper_settle_sec` | `1.5` | 启动时固定；夹爪 goal 接受等待预算的组成部分，并参与 capability digest |
| `default_skill_timeout_sec` | `30.0` | catalog entry 未声明 timeout 时的 robot fallback；entry cap 优先，并参与 digest |
| `task_budget_sec` | `180.0` | 启动时固定；canonical root task 预算，zero-budget direct root 由 Gateway stamp，并参与 capability digest |
| `robot_state_freshness_sec` | `0.5` | 启动时固定；机器人状态新鲜度阈值，并参与 capability digest |
| `scene_freshness_sec` | `0.5` | 启动时固定；场景新鲜度阈值，并参与 capability digest |
| `model_idle_timeout_sec` | `120.0` | 启动时固定；模型空闲超时，并参与 capability digest |
| `config_digest` | 空字符串 | 启动时固定；legacy 别名，字节级等于 `capability_digest` |
| `ledger_terminal_capacity` | `100` | 普通 request ledger 的 terminal 历史容量；Workflow terminal record 在整个 registry epoch 内保留 |
| `gripper_open_position` | `1.0` | 张开值 |
| `gripper_closed_position` | `0.0` | 闭合值 |
| `arm_joint_names_json` | `[]` | 手臂关节名顺序，编译时烘焙进 snapshot `robot_context` |
| `joint_limits_json` | `{}` | 关节限位配置，编译时烘焙进 snapshot `robot_context` |
| `workspace_json` | `{}` | 工作空间边界，编译时烘焙进 snapshot `robot_context` |
| `arm_trajectory_action_name` | `/arm_trajectory_controller/follow_joint_trajectory` | 手臂轨迹 action 名 |
| `task_executor_action_name` | `/task_executor/execute_task_plan` | task_dispatch 执行动作名 |
| `pick_action_name` | `/manipulation/execute_pick` | 委托型抓取技能 action 名 |
| `move_configuration_service` | legacy: `/moveit_gateway/move_to_configuration`; runtime: `/motion/move_to_joint` | 精确关节配置服务；仅显式 runtime 使用中立默认端点 |
| `runtime_enabled` | `false` | 启动时固定；由 launch 根据显式 `runtime.provider` 设置，不根据收到的状态推断 |
| `runtime_name` | 空字符串 | runtime 启用时必填，必须匹配选定 provider 发布的 `RuntimeStatus.runtime_name` |
| `runtime_status_topic` | `/runtime_status` | reliable / volatile / keep-last 10，与公共 runtime 发布端一致 |
| `runtime_mode_service` | `/runtime/set_mode` | `SetRuntimeMode` 服务；不得退回 legacy motion-mode 服务 |
| `runtime_status_freshness_sec` | `3.0` | 正有限值；同时约束本机 monotonic 收包年龄和 ROS 消息时间戳年龄 |
| `runtime_mode_map_json` | `{"moveit_planning":"trajectory","teleop":"stream","model_inference":"stream"}` | 应用控制模式到 runtime 模式的映射；目标必须在 `declared_modes` 中 |
| `ee_pose_topic` | `/robot_status/ee_pose` | 末端位姿反馈 topic |
| `joint_state_topic` | `/joint_states` | 关节状态反馈 topic |
| `debug_tracing` | `false` | 是否输出调试日志 |

显式 runtime 路径在技能和外部 primitive 分发前检查新鲜状态、`ACTIVE` 生命周期、空 faults、
未锁存 stop、声明的模式以及技能 manifest 可选的 `capability.required_capabilities`。
`DEGRADED` 可能代表控制器不可用或硬件读取失败，不允许运动准入。模式 RPC 成功后仍需收到
请求之后的新鲜状态确认；服务等待、RPC 和状态确认分别受 `rpc_timeout_sec` 约束。
执行器不会自动请求 `idle` 清除 stop latch。runtime 与执行器必须使用一致的 ROS 时钟。

父 launch 负责从已验证的 `descriptor.model` 绑定到 `config.robot_model`，按选定命名 group 的
`joints` 顺序投影 `arm_joint_names_json`，从同一模型投影 `joint_limits_json`（应用限制只能收窄），
并投影 `arm_trajectory_action_name`、`joint_state_topic`、`ee_pose_topic`、`move_configuration_service`。
这里不读取 provider 私有 profile/calibration，也不自行选取第一个 group。无 provider 的旧配置
保留旧端点及控制模式逻辑，即使 DDS 中存在其他 runtime 的状态。

### 取消终态契约

父 `SkillCommand` 请求取消后，执行器会取消其直接 child primitive，并分别在最多
`rpc_timeout_sec` 内确认 child cancel response 和 child result 终态；即使 goal 在取消请求后才被接受，
也会继续取消并 drain。

- 仅当两项确认都完成时，SkillCommand 才返回 `SKILL_CANCELLED` 并进入 canceled 终态。
- send goal/result 状态未知、cancel response 或 terminal 超时、以及下游 primitive 清理状态未知时，
  SkillCommand 会 abort 并返回 `SKILL_CANCEL_TIMEOUT`。对应 Gateway ledger 终态和 audit terminal
  `error_code` 也保持为 `SKILL_CANCEL_TIMEOUT`。
- PrimitiveCommand 的底层清理信号仍可使用 `CANCEL_CLEANUP_TIMEOUT`，但不会作为 SkillCommand 的
  公共错误码暴露。
- `STOP_TIMEOUT`（ExecuteNavigation 速度未在 `cancel_cleanup_timeout_sec` 内稳定到零）与
  `INTERNAL_ERROR` 都是 fail-closed 终态：调用方与 Gateway **不得**释放 root lease，
  `_active_skill_admission` / `_active_skill_owner` / `_active_audit_context` /
  `_active_runtime_bundle` / `_retained_admission_cleanup` 全部保留，runtime bundle retention
  也不回收。Coordinator ledger 先写 terminal record，再由后续
  `FinalizeWorkflowExecution` 以匹配终态幂等收敛后才释放 root lease 与 bundle retention。
  这避免在底盘速度未归零时把 root lease 交给下一个请求；调用方必须显式终态化才能恢复。
- `nav_*` primitive 在 cancel/cleanup 路径上的 lease 保留语义与上面通用 primitive 一致；
  `ExecuteNavigation` 进入 `STOP_TIMEOUT` 时，对应 `SkillCommand` 终态保持 `SKILL_CANCEL_TIMEOUT`，
  root lease 同样不被释放。

### SkillCommand feedback

SkillCommand 每个 primitive 步骤都发布 `state="executing"`，并使用 `step <current> of <total>` 作为
`detail`，同时携带 `actual_registry_epoch` 和 `actual_registry_generation` 反馈当前 runtime bundle identity。
反馈不包含 primitive 名称、pose、关节或夹爪目标；PrimitiveCommand 自己的内部 feedback 不受此限制。

## 10. 当前限制

- `move_to_named_pose` / `move_relative_ee` 现在会等待 `/robot_status/ee_pose` 收敛到目标附近，而不是只做固定 sleep
- gripper primitive 的 goal 接受等待预算使用 `gripper_settle_sec + 3.0`，执行结果等待上限至少为 15 秒
- 关节 primitive 要求命令完整手臂关节列表，并依赖安全层做关节限位检查
- 技能集合仍是有限白名单，不支持任意自由组合动作
- 使用 `target_pose_key` 的模板依赖 `named_targets_json` 预先配置对应位姿键
- `pick_object` 已接入 GraspGen、IK/FK 接触补偿、候选重试和抓后验证，但仍不包含连续视觉伺服或力控。
- `move_to_configuration` 底层仍是同步服务，不能提供 action 级运动硬取消。

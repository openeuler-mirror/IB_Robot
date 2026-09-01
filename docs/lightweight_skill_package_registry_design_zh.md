# 轻量级 Skill 包与注册表设计

状态：修订草案
日期：2026-08-04

## 1. 修订摘要

本次修订根据上一版热加载方案的评审意见完成，修正了五个结构性问题：

1. 引入 `registry_epoch`，避免进程重启后重复使用同一个版本命名空间。
2. 将 `SkillRegistry` 与进程内执行协调器分离。
3. 快照同步使用精确版本标识，而不是仅依赖 `min_generation`。
4. Skill 请求必须绑定预期的注册表版本。
5. reload 使用暂存的不可变 catalog 和原子切换，不再直接读取可能被部分修改的实时目录树。
6. 增加 Hermes 本地 Agent 控制面：自然语言必须先生成 exact-snapshot plan，再经过只读校验、展示后内部 confirm 和
   direct Skill/typed Workflow 执行；Hermes 不拥有机器人启动、授权或底层运动权限。

同时修正文档中的 manifest 示例，使其与当前校验契约保持一致：

- `when_to_use` 保持列表类型。
- `motion_scope` 保持列表类型。
- `do_not_use` 保持对象列表类型。
- `rule_entry` 和 `requires_motion_params` 保持为显式字段。

本轮继续冻结 Primitive、Atomic Operator、Skill、Workflow 四层逻辑边界，但不将四层实现为四套运行系统：
Atomic Operator 与 Skill 共用 catalog、manifest、`SkillCommand` 和 Gateway；Workflow 由
`embodied_agent` 持有；Primitive 由单一静态 descriptor registry 管理且不能热加载。Manifest 必填
`semantic_level`，但该字段不是授权或安全边界。Workflow 使用 typed step，并在整个执行期间绑定同一个
catalog snapshot 和 root execution lease。

本文仅为设计文档，不修改运行时代码。

## 2. 背景

当前 IB-Robot 运行时仍将 Skill 定义内联在 `robot_config` YAML 中，并在启动时
以 JSON 参数形式注入 `skill_executor`、`safety_guard` 和各 planner。该方式将
Skill 与硬件事实集中在同一文件内，但也导致 Skill 迭代成本较高，简单的 YAML
修改也需要重启完整节点。

目标方向保持不变：

- `robot_config` 管理稳定的机器人硬件事实。
- 独立的 Skill catalog 管理 Skill 定义和 profile。
- 运行时节点消费不可变快照，而不是使用临时共享的全局状态。

以下修订设计保留该方向，同时修复上一版的版本管理和重启语义缺口。

## 3. 目标

1. 将 Skill 数据与机器人硬件数据拆分。
2. 在不修改控制模式或关节 SSOT 的前提下支持 Skill 定义 reload。
3. 确保所有 consumer 观察到相同的 catalog 版本标识。
4. reload、版本不匹配或快照状态未知时必须 fail closed。
5. reload 期间保留活动执行，同时防止新 goal 混用不同版本。
6. 使生产环境打包和发布具备原子性与确定性。

## 4. 非目标

1. 第三方 Skill 市场或在线安装机制。
2. 运行时加载新的 Python 或 C++ executor 代码。
3. 热加载 ROS message、service 或 action 定义。
4. 热加载机器人关节限位、控制器名称、命名位姿或工作空间边界。
5. 在进程重启后自动恢复执行中的 action。

## 5. 架构边界

修订后的边界如下：

```text
robot_config  ----->  skill_catalog  ----->  skill_library
      ^                    ^                      ^
      |                    |                      |
      |                    |                      +--> safety_guard / planners / CLI 消费快照
      |                    +--> 纯数据 compiler 和 registry
      +--> 硬件事实和机器人上下文
```

职责划分：

- `robot_config`：机器人名称、关节、控制模式、命名位姿、工作空间限位、
  timeout policy 和机器人相关执行上下文。
- `skill_catalog`：manifest/profile 加载、schema 校验、canonical digest、
  快照编译和数据源发现。
- `skill_library`：运行时执行协调、准入控制、bundle 切换和 action 分发。
- `safety_guard`、planner、CLI：仅作为快照 consumer。

`skill_catalog` 不得导入 `robot_config` 或 `skill_library`，只接收调用方传入的
普通 `SkillRobotContext` 对象。

### 5.1 四层能力模型

本设计把执行词汇、可发现能力和任务编排明确分成四层。它们是逻辑语义层，不是四个互相复制的
catalog 或四套 Gateway；首版只有 `skill_catalog` 一个公开能力数据源。Atomic Operator 与 Skill 是并列的
两类 `CatalogEntry`，不是 `Primitive -> Operator -> Skill` 的线性继承关系；Workflow 可以同时编排二者。

| 层 | 语义 | 权威归属 | 是否进入 catalog |
|---|---|---|---|
| Primitive | 固定的最小执行词汇，例如 `open_gripper`、`move_relative_ee`。它描述执行器能做什么，不描述用户目标。 | `embodied_common` 的 Primitive descriptor registry；`skill_library` 执行，`safety_guard` 校验 | 否，不能通过 reload 新增 |
| Atomic Operator | 面向调用方的单一直接操作，例如按请求移动末端、开合夹爪或旋转夹爪。它是可发现的 catalog entry，但不是 Primitive 名称的别名表。 | `skill_catalog` manifest/profile；运行时仍通过 `skill_library` | 是，`semantic_level: atomic_operator` |
| Skill | 面向目标和后置条件的能力，例如回到安全位、挥手或抓取。它可以使用一个或多个 Primitive，也可以委托给受约束的 executor。单 Primitive 不会自动成为 Operator。 | `skill_catalog` manifest/profile；运行时通过 `skill_library` 或 delegated executor | 是，`semantic_level: skill` |
| Workflow | 有序的任务步骤，引用 enabled 的 Atomic Operator 或 Skill。它负责理解、规划、预算和步骤编排，不定义运动实现。 | `embodied_agent` 的 `TaskCommand` / typed workflow steps | 否，首版不进入 `skill_catalog` |

规范依赖方向为：

```text
Workflow -> CatalogEntry(Atomic Operator | Skill)
Atomic Operator -> exactly one Primitive                 # schema v1
Skill -> one or more Primitives | Delegated Executor
```

展开后的关系为：

```text
                         +-> Atomic Operator -> exactly one Primitive
Workflow -> CatalogEntry |
                         +-> Skill -> Primitive sequence | Delegated Executor
```

典型示例：

- Primitive：executor 内部的 `move_relative_ee(dx, dy, dz)`。
- Atomic Operator：公开的 `move_relative_ee(direction=left, distance=0.05)`，调用方明确选择直接动作。
- 单 Primitive Skill：`recover_safe_pose`，调用方声明安全恢复目标，实现可以在后续版本中改变。
- 多 Primitive Skill：`wave_hello`，调用方只观察一个问候目标和结果，不编排内部轨迹。
- Workflow：`open_gripper_skill` Operator -> `pick_object` Skill -> `recover_safe_pose` Skill。

以下依赖禁止：Workflow 直接引用 Primitive、catalog entry 引用另一个 catalog entry、
Workflow 嵌套 Workflow，以及任何 consumer 自己维护一份 Primitive 白名单。

#### Atomic Operator 与 Skill 的准入规则

Manifest 顶层增加必填字段 `semantic_level`，只允许 `atomic_operator` 或 `skill`。
Compiler 必须执行以下结构门禁：

1. `atomic_operator` 必须选择 `kind: primitive_sequence`，规范化后恰好包含一个 Primitive。
2. `atomic_operator` 不得声明 `delegated_executor`、`initial_gripper_state`、
   `move_through_joint_positions` 或任何 catalog 引用；其参数只能描述该一次直接操作。
3. `skill` 必须选择 `primitive_sequence` 或 `delegated_executor`，不得同时缺失两者，且可以只有一个
   Primitive。Skill 的分类依据是公开目标/后置条件，不依据 primitive 数量；该后置条件首版通过
   `capability.summary`、参数约束和 `SKILL.md` verification 共同表达，不新增未经实现支持的 runtime 字段。
4. Compiler 不通过字符串语义猜测“目标型”或“直接操作型”；manifest review 和迁移测试必须证明
   `summary`、参数和 postcondition 与 `semantic_level` 一致。
5. `do_not_use.instead_use` 只是 planner 提示引用，不是执行依赖，不得用它实现 Skill 嵌套。

`semantic_level` 描述调用方表达的是“直接操作”还是“目标能力”，不得作为 ROS 访问控制、motion
authorization、control mode、lease、timeout 或 safety 的放宽条件。两类 entry 必须经过完全相同的 Gateway
准入链，caller 也不得在请求中自报层级；Gateway 只读取 captured snapshot 中的编译结果。

“Atomic Operator 规范化后恰好一个 Primitive”是 schema v1 的实现约束，用于控制首版执行面和验证复杂度，
不是长期语义定义。未来实现如果需要用多个底层控制指令保持同一个直接操作契约，必须升级 implementation
schema 并重新评审该门禁，不能通过伪造单 Primitive、改名为 Skill 或在 executor 中隐藏未声明步骤绕过。

首版不提供 `internal` catalog visibility。Profile 中所有 `enabled` entry 都是公开 Gateway 能力，可由任意
通过部署级 ROS access policy 的 direct API/CLI client 显式调用，并统一经过 motion authorization、control
mode、lease 和 safety 校验；`planner_visible` 只决定是否可以进入自动 planner，不是访问控制字段。不能公开
调用的实现不得作为 catalog entry 启用，必须保留在 Primitive 或 delegated executor 内部。Primitive action
本身不是公开 Agent/CLI 接口。这样不会把无法由 ROS transport 可靠认证的“内部 Skill”误当成安全边界。

Primitive 的 canonical descriptor registry 由 `embodied_common` 单独拥有；Stage 1 将当前
`skill_templates.py` 中的 `SUPPORTED_PRIMITIVES` 移到专用 `primitive_contracts.py`，其余包只能导入
它。该 registry 是随软件构建发布的静态、只读数据模块，不是第二个 ROS registry，不提供 reload service。
Descriptor schema v1 的字段精确为 `schema_version`、canonical `name`、`parameter_contract`、`required_runtime_capabilities` 和 `dispatch_kind`；callable、进程地址和运行时状态不得进入 descriptor 或
digest。`required_runtime_capabilities` 去重后按字典序排序，descriptor 按 name 排序。

registry 不再暴露单一全局 digest，而是以 **`PrimitiveContractSet`** 为单位组织版本化契约。一个
`PrimitiveContractSet` 由三元组 `(schema_version, descriptors, digest)` 组成：`schema_version` 标识
该契约集合的版本（`1` 或 `2`），`descriptors` 是该版本下的全部 canonical `PrimitiveDescriptor` 有序集合，
`digest` 是对该集合 canonical preimage 计算的 SHA-256。`embodied_common.primitive_contracts` 提供
`primitive_contract_for_version(version: int) -> PrimitiveContractSet` 选择器，按 `schema_version` 返回
对应契约集合；未知版本返回 `SKILL_SCHEMA_INVALID`。

- v1 契约集合的 `digest` 固定为 `537210f8...`（完整 SHA-256 见 golden vector），覆盖 10 个 manipulation
  primitive。
- v2 契约集合拥有**独立** `digest`（与 v1 不同），覆盖 13 个 primitive（v1 ∪
  `nav_straight` / `nav_turn` / `nav_abs_coordinate`）。v1 与 v2 digest 不混用，也不可互相回退。

registry preimage 写入的 `primitive_contract_digest` 由 `robot_context.context_schema_version` 选择：
compiler 调用 `primitive_contract_for_version(robot_context.context_schema_version)` 取对应
`PrimitiveContractSet`，将其 `digest` 写入 `registry_preimage.primitive_contract_digest`，并用其
`descriptors` 校验 implementation 的 primitive 白名单。compiler 不接受调用方临时覆盖或 merge
两个版本的 descriptor。

使用以下精确 preimage 计算 `PrimitiveContractSet.digest`：

```json
{
  "schema_version": 1,
  "primitives": [
    {
      "schema_version": 1,
      "name": "open_gripper",
      "parameter_contract": {
        "type": "object",
        "properties": {
          "primitive_name": {
            "type": "string",
            "const": "open_gripper"
          }
        },
        "required": ["primitive_name"],
        "additionalProperties": false
      },
      "required_runtime_capabilities": ["task_executor", "validate_skill"],
      "dispatch_kind": "task_executor_action"
    }
  ]
}
```

`parameter_contract` 是校验单个 source `primitive_sequence` step 的 canonical JSON Schema，并必须包含
`primitive_name: {type: string, const: <descriptor.name>}`。其 v1 schema keyword 封闭集合为：对象型 schema
节点只允许 `type`、
`properties`、`required`、`additionalProperties`、`allOf`、`oneOf`；property 层只允许 `type`、`const`、`enum`、
`minimum`、`exclusiveMinimum`、`minItems`、`items`。`type` 只允许 `object/array/string/number/boolean`，所有
`type=object` 的 schema 节点都必须显式 `additionalProperties`；默认值为 `false`，只有 `joint_positions`、`joint_position_offsets`
和其他由 descriptor 明确标注为 `robot_joint_mapping` 的 property 可以使用
`additionalProperties: {"type": "number"}`，且 compiler 必须再校验 key 属于 `SkillRobotContext.arm_joint_names`。
array `items` 只能使用同一 property 子集。未知 keyword、外部
`$ref`、默认值、format、正则、运行时代码或自定义 validator 全部拒绝。

`required_runtime_capabilities` 的 v1 元素封闭集合精确为 `validate_skill`、`task_executor`、
`arm_trajectory`、`fresh_ee_pose`、`move_configuration`。`dispatch_kind` 封闭集合及 endpoint role 映射为：

| `dispatch_kind` | `SkillRobotContext.execution_endpoints` role | 必需 readiness capability | v1 Primitive |
|---|---|---|---|
| `task_executor_action` | `task_executor_action` | `task_executor` | `move_to_named_pose`、内部动态 `move_to_pose`、`move_relative_ee`、`open_gripper`、`close_gripper`、`rotate_gripper_cw`、`rotate_gripper_ccw` |
| `arm_trajectory_action` | `arm_trajectory_action` | `arm_trajectory` | `move_to_joint_positions`、`move_through_joint_positions` |
| `move_configuration_service` | `move_configuration_service` | `move_configuration` | `move_to_configuration` |

表中 Primitive 必须与 canonical registry 一一对应，不能出现在两个 dispatch kind。Descriptor 的
`required_runtime_capabilities` 必须至少包含其 dispatch kind 对应的 readiness capability；其他 capability 只能
来自上述封闭集合。新增 Primitive 在保持同一 descriptor shape 时不升级 schema，但必须重新构建相关进程、改变
contract digest 并增加 golden vector；新增 schema keyword、capability 或 dispatch kind 必须升级 Primitive
descriptor schema 和 endpoint role 契约，不能由 consumer 私自扩展。

`skill_catalog`、`skill_library` 和 `safety_guard` 必须读取同一 canonical registry，不得复制名称列表、参数表或
默认值。Compiler 调用 `primitive_contract_for_version(robot_context.context_schema_version)` 选择对应
`PrimitiveContractSet`，将其 `digest` 写入 registry preimage 的 `primitive_contract_digest`；executor 和
safety 在接受快照前按同一 `context_schema_version` 重算本地 digest，不匹配时 fail closed，从而拒绝混跑
不同 schema 版本或不同 Primitive contract 的二进制。Primitive action 只接受
active `ExecutionCoordinator` 签发的内部 dispatch binding；外部能力入口只能使用 `SkillCommand`。

#### 当前配置的迁移分类

`so101_single_arm.yaml` 当前的 16 个 entry 按公开语义迁移，而不是按 primitive 数量机械分类：

| 分类 | 当前 entry | 迁移要求 |
|---|---|---|
| Atomic Operator | `move_relative_ee`、`open_gripper_skill`、`close_gripper_skill`、`rotate_gripper_cw`、`rotate_gripper_ccw` | 保持单 Primitive 直接操作契约；名称可以在后续 breaking release 中去掉 `_skill` 后缀，首版迁移不强制改名。 |
| Skill | `recover_safe_pose`、`recover_zero_pose`、`dance_basic`、`wave_hello`、`nod_yes`、`shake_no`、`celebrate`、`greet_observe_raise`、`act_cute`、`happy_spin_upright` | 保持目标/后置条件语义；复合轨迹继续通过 primitive sequence。 |
| Skill，需校准描述 | `inspect_scene` | 当前实现只移动到 `observe_table`，没有产生场景分析结果。迁移时必须把公开后置条件写成“到达观察位”，或增加真正的 perception/delegated executor；不得继续宣称已经完成场景理解。 |

`inspect_scene` 的语义校准是 Stage 2 门禁，不允许通过修改 digest 或 planner prompt 掩盖实现落差。

### 5.2 多领域 SSOT 与运行系统映射

本设计拆分的是不同领域的权威来源，不是复制一份全局 SSOT：

| 领域 | 唯一权威 | 禁止复制到 |
|---|---|---|
| 机器人事实：关节、硬限位、命名位姿、控制模式、endpoint role | `robot_config` | manifest、profile、Primitive descriptor |
| 内部执行词汇和参数契约 | `embodied_common.primitive_contracts` | catalog 私有白名单、executor/safety 平行常量 |
| Atomic Operator/Skill 定义、enablement、planner visibility | `skill_catalog` manifest/profile | robot YAML、planner 硬编码 allowlist |
| 已规划 Workflow 的步骤、顺序、逐步参数和任务预算 | `embodied_agent` typed `TaskCommand` | Skill implementation、`context_json`、`task_dispatch` |

`skill_library` 是统一执行 Gateway 和 coordinator，不拥有第二份 catalog source；它只持有 compiler 产生的
immutable bundle 和进程内执行状态。`safety_guard`、planner 和 CLI 只是 snapshot consumer。多个领域 SSOT
通过 compiler 输入和 digest 绑定组合，不允许通过复制文件内容获得“本地一致性”。

Workflow 首版不进入 `skill_catalog`，表示 catalog 不持久化可复用 Workflow 定义；不表示 Workflow 可以无
schema、无版本或散落在 planner 私有 JSON 中。Planned Workflow 必须是绑定 exact snapshot 的 typed
`WorkflowStep[]`。未来如果需要持久化、共享或版本化 Workflow template，应由 `embodied_agent` 增加独立
`WorkflowDefinition` schema 和 owner，不得把它伪装成 catalog entry 或 implementation。

### 5.3 Schema 版本演进（v1 / v2）

本节冻结 primitive contract 与公开请求类型的 schema 版本分离。schema_version 是契约层级的强类型
门禁，不是运行时可协商字段；新增版本必须升级 preimage、digest、IDL 和 golden vector，并在同一次
原子部署中切换全部 producer/consumer。

**v1（原始 manipulation 契约）**

- 公开请求类型（`SkillCommand.Goal`、`PrimitiveCommand.Goal`、`ValidateSkill.Request`、
  `ValidatePrimitive.Request`）和 `WorkflowStep` 都以 `uint32 schema_version` 为首字段，v1 取值为 `1`。
- 字段集合仅包含 manipulation 字段：`motion_direction`、`motion_distance`，以及 Primitive 联合字段
  （`pose_name`、`target_pose`、`relative_dx/dy/dz`、`joint_*` 等）。
- 控制模式封闭集合为 3 种（`teleop` / `model_inference` / `moveit_planning`）。
- Primitive 白名单为 10 个：`move_to_named_pose`、`move_to_pose`、`move_to_configuration`、
  `move_relative_ee`、`move_to_joint_positions`、`move_through_joint_positions`、`open_gripper`、
  `close_gripper`、`rotate_gripper_cw`、`rotate_gripper_ccw`。
- v1 请求**拒绝**任何导航字段（`direction` / `distance` / `degree` / `x` / `y` / `yaw`）：携带导航字段
  的 v1 请求返回 `SKILL_SCHEMA_INVALID`。
- v1 primitive contract digest 固定为 `537210f8...`（完整 SHA-256 见 `embodied_common.primitive_contracts`
  的 golden vector）。

**v2（v1 ∪ 导航扩展）**

- v2 在 v1 字段集合之上**并集**新增导航字段：`direction`、`distance`、`degree`、`has_x`、`x`、`has_y`、
  `y`、`has_yaw`、`yaw`。`has_*` 三个 bool 用于区分“字段缺失”与“显式零值”，避免把未设置静默当作
  “移动到 (0, 0, 0)”的导航目标。
- 控制模式封闭集合新增 `base_navigation`，共 4 种。
- Primitive 白名单新增 3 个导航 primitive：`nav_straight`、`nav_turn`、`nav_abs_coordinate`，共 13 个。
- v2 请求**必须**显式设置 `schema_version=2`，不得依赖默认零值；`schema_version=0` 或未知值返回
  `SKILL_SCHEMA_INVALID`。
- v2 拥有**独立**的 `primitive_contract_digest`（与 v1 不同），由 v2 descriptor 集合的 canonical preimage
  重新计算；v1 与 v2 digest 不混用。
- registry preimage 写入的 `primitive_contract_digest` 由 `robot_context.context_schema_version` 选择
  （`context_schema_version=1` 选 v1 digest，`=2` 选 v2 digest），compiler 不接受调用方临时覆盖。

**Breaking IDL change 与原子重建**

新增 / 重命名 schema 字段、primitive 或 dispatch kind 会改变 ROS type hash，不提供旧二进制 wire
compatibility。v1 → v2 升级是一次 **breaking IDL change**，必须**原子重建并部署**全部 workspace
producer / consumer（`skill_library`、`safety_guard`、`embodied_agent`、`robot_skill_cli`、
`ibrobot_msgs` 等），不得混跑 v1 与 v2 二进制。`embodied_common.wire_contracts.validate_public_request_wire_contracts()`
在节点启动时执行 preflight：四个公开请求类型必须以 `schema_version` 为首字段，且取值属于 `{1, 2}`；
不匹配时节点不进入 ready。该 preflight 与运行期 `primitive_contract_digest` 校验共同构成二重门禁。

## 6. 版本模型

### 6.1 标识字段

Catalog 的身份由以下元组表示：

```text
(registry_epoch, generation, registry_digest)
```

定义：

- `registry_epoch`：运行时 owner 激活 catalog 时生成的 UUID。
- `generation`：在单个 epoch 内单调递增的 `uint64`。
- `registry_digest`：不可变内部快照的哈希。
- `primitive_contract_digest`：静态 Primitive descriptor contract 的哈希，并进入 `registry_digest` preimage。
- `capability_digest`：仅包含公开 capability view 的哈希。
- `source_release_digest`：生成该快照的不可变 catalog release 哈希，仅用于 provenance，不替代
  `registry_digest`。
- `provenance_digest`：绑定 source release 和逐 Skill package digest 的快照 provenance 哈希。

Generation 在 candidate 的 `registry_digest` 或 `provenance_digest` 任一变化时递增。仅
`capability_digest` 不变不能判定为 no-op；例如只修改 `SKILL.md` 会保持执行内容和 capability 不变，但会
切换 source provenance，因此仍产生新 generation，且 `changed_skills` 可以为空。

### 6.2 为什么不能只使用 Generation

如果进程重启后 generation 再次从 `1` 开始，仍持有较高 generation 的 consumer
会永久将新快照判断为过期，从而使重启恢复无法完成。

引入 epoch 后：

- 进程重启会创建新的 `registry_epoch`。
- Consumer 会显式拒绝旧 epoch。
- 所有版本比较均为显式比较，不依赖隐含假设。

### 6.3 保留规则

只有当没有活动执行继续引用某个快照标识时，registry 才可以清理该历史快照。
如果运行时进程崩溃，进程内活动状态会丢失，系统重新启动后必须进入 fail-closed
状态。

进程崩溃后不保证静默恢复执行。

### 6.4 Canonical JSON 与 Digest Preimage

所有进程必须使用同一套 canonical JSON 规则：

```python
json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
```

规范化规则：

1. Mapping 的 key 必须是字符串，并按字典序排序。
2. tuple/list 在 preimage 中统一编码为 JSON array。
3. set/frozenset 必须先转换为排序后的 array。
4. `Path` 必须转换为 catalog release 内的相对 POSIX 路径。
5. 浮点值必须有限；拒绝 NaN 和 Infinity，并将 `-0.0` 规范化为 `0.0`。
6. Unicode 由 `ensure_ascii=True` 统一转义。
7. 运行时对象、memory address 和无序容器不得进入 preimage。

`registry_preimage_v1` 的顶层字段必须精确为以下集合，禁止实现自行增加或删除字段：

```json
{
  "schema_version": 1,
  "robot_name": "...",
  "profile_name": "...",
  "primitive_contract_digest": "...",
  "robot_context": {
    "context_schema_version": 1,
    "robot_config_digest": "...",
    "named_poses": {},
    "named_targets": {},
    "arm_joint_names": [],
    "joint_limits": {},
    "workspace_limits": {},
    "required_control_mode": "...",
    "timeout_policy": {},
    "relative_motion_reference_frame": "...",
    "relative_motion_step_m": 0.0,
    "relative_motion_direction_mapping": {},
    "gripper_open_position": 0.0,
    "gripper_closed_position": 0.0,
    "execution_endpoints": {}
  },
  "delegated_executors": [
    {
      "name": "grasp_pipeline",
      "contract_version": "1.0.0",
      "endpoint_kind": "ros_action",
      "endpoint_name": "/embodied/pick_object",
      "configuration_digest": "...",
      "model_deployment_name": "...",
      "model_fingerprint": "...",
      "model_bundle_digest": "..."
    }
  ],
  "skills": [
    {
      "name": "...",
      "version": "...",
      "semantic_level": "atomic_operator",
      "implementation_identity": "...",
      "template": {}
    }
  ],
  "aliases": {},
  "parameter_schemas": {},
  "requirements": {},
  "enabled_skill_names": [],
  "planner_visible_skill_names": []
}
```

`skills` 按 name 排序，所有 name list 和 `delegated_executors` 按稳定 identity 排序。
`primitive_contract_digest` 是 5.1 节 canonical Primitive descriptor 集合的 SHA-256；它绑定执行词汇、参数
字段和 required runtime capabilities，但不包含 callable 或运行时 endpoint 状态。任何 Primitive descriptor
变化都必须改变该 digest 和 `registry_digest`，即使当前 profile 没有引用发生变化的 Primitive。
`robot_config_digest` 是 `robot_config` 对上述完整、规范化 execution context 计算的 SHA-256，不是已有公开
`config_digest`；其 preimage 不包含 `robot_config_digest` 字段自身。Named pose/target 和 direction mapping
必须先由 robot_config loader 规范化为普通 mapping/array/有限数值。`registry_digest` 是该精确 payload 的
canonical JSON SHA-256。

`capability_preimage_v1` 的顶层字段必须精确为：

```json
{
  "schema_version": 1,
  "robot_name": "...",
  "profile_name": "...",
  "capability_view": {},
  "enabled_skill_names": [],
  "planner_visible_skill_names": [],
  "named_pose_names": [],
  "timeout_policy": {}
}
```

`capability_digest` 是该精确 payload 的 canonical JSON SHA-256。

`capability_view` 顶层是 entry name 到公开对象的 mapping；每个对象的字段必须精确为 `name`、`summary`、
`domain`、`semantic_level`、`planner_visible`、`moves_robot`、`required_control_mode`、`parameters` 和
`recovery_policy`。`planner_visible_skill_names` 继续作为完整的 planner allowlist；它必须与各 entry 的
`planner_visible` 字段完全一致。

`provenance_preimage_v1` 的顶层字段必须精确为：

```json
{
  "schema_version": 1,
  "source_release_digest": "...",
  "skill_package_digests": {}
}
```

`provenance_digest` 是该精确 payload 的 canonical JSON SHA-256。

跨进程 `snapshot_payload_v1` 直接封装三个可验证的完整 preimage，其顶层字段必须精确为：

```json
{
  "schema_version": 1,
  "registry_preimage": {},
  "capability_preimage": {},
  "provenance_preimage": {}
}
```

Consumer 必须从 `registry_preimage` 和 `capability_preimage` 分别重算 SHA-256，并与 service response 中的
digest 比较，并从 `provenance_preimage` 重算 `provenance_digest`；只比较 response 中的字符串不算完成同步。
前两个 preimage 中直接重复的 robot/profile、enabled set、planner-visible set 和 timeout policy 必须完全一致，
否则拒绝快照。`registry_preimage` 不重复存储 `capability_view`；consumer 必须使用共享的纯函数从其 normalized
`skills`、parameter schemas 和 profile visibility 确定性派生公开 view，并与
`capability_preimage.capability_view` 逐字节比较。派生结果不一致时拒绝快照。

以下内容不得进入内容 digest：

- `registry_epoch` 和 `generation`。
- 绝对 source path 或 install path。
- 文件 mtime、编译时间、进程 ID 和 request ID。
- primitive sequence、绝对关节值和内部 ROS transport 名称不得进入 `capability_digest`，但必须进入
  `registry_digest`。

仓库必须提供固定的 golden preimage 和 digest 测试向量，确保 compiler、runtime、CLI 和测试工具得到
完全相同的结果。

## 7. 包目录结构

该包保持为轻量级 catalog，而不是插件市场：

```text
src/skill_catalog/
├── package.xml
├── setup.py
├── skill_catalog/
│   ├── models.py
│   ├── source.py
│   ├── compiler.py
│   ├── validator.py
│   ├── digest.py
│   └── registry.py
├── config/
│   ├── schemas/
│   ├── profiles/
│   └── skills/
└── test/
```

运行时状态不得存放在该包中。

Atomic Operator 与 Skill 共用 `config/skills/<name>/` 的 package layout、manifest schema、profile、snapshot、
`SkillCommand` 和 Gateway，由 manifest 的 `semantic_level` 区分；首版不增加平行的 `config/operators/`。
Workflow 不属于该 source tree，也不得通过 `implementations/` 文件模拟 Workflow。

## 8. Catalog Entry 文件契约

### 8.1 `manifest.yaml`

Manifest 是单个 Atomic Operator 或 Skill package 的 SSOT。文档契约必须与当前 validator 保持一致，不能使用
为了说明方便而简化但不兼容的字段结构。

示例：

```yaml
schema_version: 1
name: wave_hello
version: 1.0.0
semantic_level: skill

description:
  summary: Perform a casual side-to-side greeting wave.
  category: social_greeting
  when_to_use:
    - greet a person
    - acknowledge a user
  aliases_zh: [挥手, 打招呼, 再见]
  aliases_en: [wave, say hello]
  motion_scope: [arm]
  intensity: moderate
  duration_sec_estimate: 3.0
  requires_motion_params: false
  rule_entry: true
  do_not_use:
    - condition: workspace is obstructed
      instead_use: inspect_scene

# 可选：仅当同一 package 的机器人 implementation 需要保留不同语义描述时使用。
# key 必须与 implementations key 一致，value 使用完整 description schema，不做字段 merge。
description_variants:
  so101_handeye_realsense_grasp:
    summary: Perform the hand-eye calibrated greeting wave.
    category: social_greeting
    when_to_use: [greet a person]
    aliases_zh: [挥手]
    aliases_en: [wave]
    motion_scope: [arm]
    intensity: moderate

capability:
  schema_version: 1
  summary: Perform a greeting wave.
  domain: social
  moves_robot: true
  required_control_mode: moveit_planning
  parameters:
    type: object
    additionalProperties: false
    properties: {}
    required: []
  recovery_policy: recover_safe_pose

implementations:
  so101_5dof_single_arm: implementations/so101_5dof_single_arm.yaml
```

规则：

1. `name` 必须与目录名一致。
2. `semantic_level` 必须显式声明，不从 implementation 步骤数推导。
3. `when_to_use` 必须是非空列表。
4. `motion_scope` 必须是列表。
5. `do_not_use` 必须是 `{condition, instead_use}` 对象列表。
6. `rule_entry` 属于 schema，因为规则解析器会使用该字段。
7. `SKILL.md` 仅作为文档，不得驱动运行时行为。
8. `description_variants` 是可选完整替换映射；选中 implementation 时使用同名 variant，否则使用 `description`。
   Variant 不允许 partial merge，必须通过同一 closed description schema 校验，防止 profile 间隐式继承别名。

### 8.2 Implementation 文件

Implementation 文件描述机器人运动学或执行变体：

```yaml
schema_version: 1
kind: primitive_sequence
robot: so101_5dof_single_arm

initial_gripper_state: closed
timeout_sec: 20.0

primitive_sequence:
  - primitive_name: move_to_joint_positions
    joint_positions:
      "1": 0.02
      "2": 0.54
      "3": -0.82
      "4": -0.18
      "5": 0.02
    duration_sec: 2.0
```

Skill implementation 中的 `workspace_limits` 仅属于局部前置约束，必须根据
`robot_config` 中的硬工作空间和关节边界进行校验。

### 8.3 Profile 文件

Profile 选择一个机器人变体启用的 catalog entry：

```yaml
schema_version: 1
name: so101_single_arm
robot_name: so101_single_arm

enabled_skills:
  - name: inspect_scene
    implementation: so101_5dof_single_arm
    planner_visible: true
  - name: wave_hello
    implementation: so101_5dof_single_arm
    planner_visible: true
```

Profile 不管理关节限位、控制器、相机 topic 或位姿数据。

Implementation 标识描述稳定的运动学或执行变体，而不是部署配置名称。例如，
`so101_single_arm` 和 `so101_rtp_distributed` 应复用
`so101_5dof_single_arm`，而不是复制相同的运动 YAML。

Profile 是 enabled catalog entry 和 planner-visible entry 集合的权威来源。
`planner_visible` 必须显式填写；对于可以通过 direct API/CLI 执行、但不应由 planner
生成的 Atomic Operator 或 Skill，将其设置为 `false`。

### 8.4 Manifest Schema v1 完整约束

| 字段 | 必需 | 约束 |
|---|---|---|
| `schema_version` | 是 | 首版必须等于 `1`。 |
| `name` | 是 | 与目录名一致，匹配 `^[a-z][a-z0-9_]*$`。 |
| `version` | 是 | 匹配 SemVer 2.0 基本格式；首版不解析版本依赖。 |
| `semantic_level` | 是 | 只能是 `atomic_operator` 或 `skill`。 |
| `description.summary` | 是 | 非空字符串，去除首尾空白后不超过 120 字符。 |
| `description.category` | 是 | 非空字符串。 |
| `description.when_to_use` | 是 | 非空字符串列表。 |
| `description.motion_scope` | 是 | 非空且无重复，元素只能为 `base/shoulder/elbow/wrist/gripper/arm`。 |
| `description.intensity` | 是 | `subtle`、`moderate` 或 `large`。 |
| `description.aliases_zh/en` | 否 | 同一语言和完整 profile 内不得冲突。 |
| `description.anchor_pose` | 否 | 必须是 `none` 或 robot context 中存在的命名位姿。 |
| `description.duration_sec_estimate` | 否 | 有限正数。 |
| `description.requires_motion_params` | 否 | Boolean，默认 `false`。 |
| `description.rule_entry` | 否 | Boolean，默认 `false`。 |
| `description.do_not_use` | 否 | `{condition, instead_use}` 对象列表。 |
| `capability` | 是 | 满足 capability schema v1。 |
| `implementations` | 是 | 非空 mapping，key 为稳定 implementation variant。 |

Manifest、description、capability 和 parameters schema 默认采用
`additionalProperties: false`。扩展字段必须先升级 schema version，不能由各 consumer 自行忽略。

Manifest 顶层字段精确为 `schema_version`、`name`、`version`、`semantic_level`、`description`、`capability`、
`implementations`。Description 字段精确为本节字段表列出的字段；缺省可选字段由 compiler 规范化后显式
写入 snapshot，consumer 不得各自应用默认值。

Implementation path 必须是 Skill package 目录内的相对 POSIX 路径，不允许绝对路径、`..`、内部 symlink
或跨 package 引用。

SemVer 使用以下正则作为首版语法门禁：

```text
^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$
```

Alias 去除首尾空白后必须非空，并在单个列表内无重复。`do_not_use` 子对象只允许 `condition` 和
`instead_use` 两个字段。

`do_not_use.instead_use` 必须引用同一 profile 中启用的其他 catalog entry，不得引用自身；如未来需要外部
fallback，必须通过显式、版本化字段表达。

`capability.parameters` 只支持当前受控 JSON Schema 子集：

- 根节点 `type` 必须为 `object`。
- 只允许 `properties`、`required`、`additionalProperties`。
- Property 名称只允许 `target_name`、`place_name`、`motion_direction`、`motion_distance`。
- `target_name/place_name/motion_direction` 的 `type` 必须为 `string`。
- `target_name` 可使用 `freeform: true`；否则 string property 必须提供非空 `enum`。
- `motion_direction.enum` 只能使用 `forward/backward/left/right/up/down`。
- `motion_distance.type` 必须为 `number`，`exclusiveMinimum` 必须等于 `0`，unit 只能为
  `meters` 或 `degrees`。
- `required` 必须引用存在的 property，且不得重复。
- String property 只允许 `type`、`enum`、`freeform`；number property 只允许 `type`、
  `exclusiveMinimum`、`unit`。

Capability 顶层字段精确为：

| 字段 | 约束 |
|---|---|
| `schema_version` | 必须等于 `1`。 |
| `summary` | 非空字符串。 |
| `domain` | 非空字符串。 |
| `moves_robot` | Boolean。 |
| `required_control_mode` | `teleop/model_inference/moveit_planning` 之一，并与 context 要求一致。 |
| `parameters` | 满足上述受控参数 schema。 |
| `recovery_policy` | `never_retry/ask_user/recover_safe_pose` 之一。 |

`recovery_policy=recover_safe_pose` 只表示返回给 task layer 的恢复建议，不授权 executor 在当前 catalog entry
内部嵌套调用 `recover_safe_pose`。是否追加恢复步骤由 `embodied_agent` 在同一 root budget、exact snapshot 和
Workflow 规则下决定。

Schema v1 的 capability parameter key 被刻意限制为
`target_name/place_name/motion_direction/motion_distance`，与 `WorkflowStep`、`ValidateSkill` 和
`SkillCommand` 的强类型字段一一对应。新增任意参数 key 必须协调升级 manifest schema 和全部相关 ROS IDL；
不得把执行参数塞回 `context_json`，也不得让 Operator 与 Skill 使用不同的私有参数通道。

### 8.5 Implementation Schema v1

Implementation 是 tagged union，`kind` 只能取以下值：

1. `primitive_sequence`
2. `delegated_executor`

`primitive_sequence` 约束：

- `primitive_sequence` 必须为非空列表。
- 每一步必须包含受支持的 `primitive_name`。
- `joint_positions` 必须覆盖该 primitive 所要求的完整关节集合。
- `trajectory_template` 与展开后的 `joint_waypoints` 不能同时作为输入。
- 展开结果必须满足关节限位、工作空间和绝对轨迹首尾连续性。
- `initial_gripper_state` 只能为 `open`、`closed`、`hold` 或 `none`。
- `timeout_sec` 和 duration 必须为有限正数。

Manifest 的 `semantic_level` 与 implementation kind 的组合还必须满足 5.1 节的四层规则。特别是，
`atomic_operator` 在 trajectory/template 展开后必须仍然只有一个 Primitive；compiler 不得把多个
Primitive 折叠成一个“看起来原子”的名称。`skill` 可以使用一个 Primitive，但其
`capability.summary`、参数约束和 `SKILL.md` verification 必须形成可验证的目标或后置条件。

`primitive_sequence` 顶层只允许：`schema_version`、`kind`、`robot`、`initial_gripper_state`、
`timeout_sec`、`primitive_sequence`。`delegated_executor` 顶层只允许：`schema_version`、`kind`、
`robot`、`executor`、`required_args`、`timeout_sec`。

每个 primitive step 默认 `additionalProperties: false`，source schema 使用以下字段表：

| Primitive | 允许和必需字段 |
|---|---|
| `move_to_named_pose` | `primitive_name`，以及 `pose_name`、`target_pose_key`、`place_name_from_request=true` 三者之一且仅一个。 |
| `move_relative_ee` | `primitive_name`；direction 使用合法 `motion_direction` 或 `motion_direction_from_request=true`；distance 使用正数 `motion_distance` 或 `motion_distance_from_request=true`。 |
| `open_gripper` / `close_gripper` | 只允许 `primitive_name`。 |
| `rotate_gripper_cw` / `rotate_gripper_ccw` | 可选正数 `motion_distance` 或 `motion_distance_from_request=true`；两者均缺省时使用 45 degrees。 |
| `move_to_configuration` / `move_to_joint_positions` | `primitive_name`；`joint_positions` 与 `joint_position_offsets` 必须二选一；可选正数 `duration_sec`。 |
| `move_through_joint_positions` | `primitive_name`；source 中 `trajectory_template` 与 `joint_waypoints` 二选一；展开后必须为非空完整 waypoint 和正数 `waypoint_duration_sec`。 |

`move_to_pose` 保留为 executor 内部动态 primitive，首版静态 catalog source 不允许直接声明，因为当前
Catalog command 没有完整 6-DOF pose 参数契约。若未来开放，必须升级 implementation schema。

`trajectory_template` 首版支持的 type 必须来自 `embodied_common.trajectory_templates` 的受控 registry；
每种 type 由独立 JSON Schema 定义参数和默认值。Compiler 只在 source schema 校验通过后展开，normalized
template 中删除 `trajectory_template` 并写入确定性的 `joint_waypoints` 和 `waypoint_duration_sec`。

`delegated_executor` 示例：

```yaml
schema_version: 1
kind: delegated_executor
robot: so101_handeye_grasp

executor: grasp_pipeline
required_args: [target_name]
timeout_sec: 180.0
```

`delegated_executor` 约束：

- 不允许同时包含 `primitive_sequence`。
- `executor` 必须存在于 compiler 输入的 `delegated_executors` descriptor mapping。
- `required_args` 必须是无重复字符串列表，并引用 capability parameter。
- Compiler 不要求 delegated Skill 含有非空 primitive sequence。
- `robot` 是稳定 implementation variant 标识，必须等于 manifest `implementations` 中选择该文件的 key；
  它不是 deployment profile 的 `robot_name`。

### 8.6 Profile Schema v1

Profile 必须满足：

1. `schema_version == 1`。
2. `name` 与 profile 文件名一致。
3. `robot_name == SkillRobotContext.robot_name`。
4. `enabled_skills` 中 catalog entry name 不得重复。
5. 每个 entry 必须存在 manifest。
6. 指定的 implementation 必须存在于该 manifest。
7. `planner_visible` 必须显式填写 Boolean，且只有 enabled entry 才能 planner-visible；首版不对新 entry
   隐式授予 planner 可见性。
8. Profile 不得定义 named pose、joint limit、controller、camera topic、workspace 或 primitive sequence。
9. 未知字段必须拒绝。

Profile 顶层字段精确为 `schema_version`、`name`、`robot_name`、`enabled_skills`；每个 enabled entry
精确包含 `name`、`implementation` 和必需的 `planner_visible`。

### 8.7 `SKILL.md`

`SKILL.md` 推荐包含：

```markdown
# Skill Name

## When to Use
## When Not to Use
## Parameters
## Examples
## Supported Robots
## Verification
```

运行时不得解析 `SKILL.md` 来决定 primitive、参数 schema、安全策略或 enabled 状态。

## 9. Compiler 与校验

### 9.1 Compiler 输入

Compiler 只接收普通数据：

```python
@dataclass(frozen=True)
class SkillRobotContext:
    robot_name: str
    context_schema_version: int
    robot_config_digest: str
    named_poses: Mapping[str, Mapping[str, Any]]
    named_targets: Mapping[str, Mapping[str, Any]]
    arm_joint_names: tuple[str, ...]
    joint_limits: Mapping[str, Mapping[str, float]]
    workspace_limits: Mapping[str, tuple[float, float]]
    required_control_mode: str
    timeout_policy: Mapping[str, float]
    relative_motion_reference_frame: str
    relative_motion_step_m: float
    relative_motion_direction_mapping: Mapping[str, tuple[float, float, float]]
    gripper_open_position: float
    gripper_closed_position: float
    execution_endpoints: Mapping[str, str]


@dataclass(frozen=True)
class PrimitiveDescriptor:
    schema_version: int
    name: str
    parameter_contract: Mapping[str, Any]
    required_runtime_capabilities: tuple[str, ...]
    dispatch_kind: str


@dataclass(frozen=True)
class DelegatedExecutorDescriptor:
    name: str
    contract_version: str
    endpoint_kind: str
    endpoint_name: str
    configuration_digest: str
    model_deployment_name: str
    model_fingerprint: str
    model_bundle_digest: str


@dataclass(frozen=True)
class SkillCompileContext:
    robot: SkillRobotContext
    primitive_contracts: Mapping[str, PrimitiveDescriptor]
    primitive_contract_digest: str
    delegated_executors: Mapping[str, DelegatedExecutorDescriptor]
```

`SkillRobotContext` 必须包含 primitive expansion 和 execution 会读取的全部 robot_config 值，而不只是名称。
字段新增必须升级 `context_schema_version`。`context_schema_version` 取值集合为 `{1, 2}`：v1 对应仅 manipulation
的 10-role `execution_endpoints` 与 v1 primitive contract；v2 对应新增 `navigation_action` role 的 11-role
mapping 与 v2 primitive contract。该版本由 `robot_config.loader` 的
`navigation_endpoint_projection()` 决定：当且仅当 `robot_config` 配置了 `navigation.command_server.action_name`
时，projection 输出 `context_schema_version=2` 并把该 action 名写入 `execution_endpoints.navigation_action`；
否则输出 `context_schema_version=1` 且不写入 `navigation_action` role。projection 是确定 context schema 版本
的**唯一**入口，caller 不得自行声明版本。

`context_schema_version` 同时决定下游 `WorkflowStep.schema_version`：planner 生成的 step schema_version
必须与所属 capability 的 schema 版本一致（v1 capability 产生 `schema_version=1` step，v2 capability 产生
`schema_version=2` step），**不再固定为 `1`**。`registry_preimage.primitive_contract_digest` 也由
`context_schema_version` 通过 `primitive_contract_for_version()` 选择对应 `PrimitiveContractSet`。

活动 goal 的 primitive dispatch、cancel 和 finalize 禁止回读节点的
current robot parameters，必须只使用其捕获 bundle 中冻结的 context。

`execution_endpoints` 是 role 到完全展开 ROS 名称的 canonical mapping。v1 key 精确为 10 个 role：
`skill_action`、`primitive_action`、`validate_skill_service`、`validate_primitive_service`、
`gateway_status_service`、`begin_workflow_service`、`finalize_workflow_service`、
`task_executor_action`、`arm_trajectory_action`、`move_configuration_service`，未知 key 拒绝。v2 在 v1 的
10 个 role 之上**新增** `navigation_action` role，共 11 个；v1 context 不得携带 `navigation_action`，
v2 context 必须携带。Delegated endpoint 只存在于其 descriptor，不能在这里重复。

`navigation_action` role 的**唯一权威**（sole owner）是 `robot_config` 中的
`navigation.command_server.action_name`；该 ROS action 名由 navigation command server 所有，compiler
从 `robot_config.navigation.command_server.action_name` 读取并写入 `execution_endpoints.navigation_action`。
legacy 参数 `embodied.execution.navigation_action_name` **已退役**（retired）：runtime 不再从该参数读取
endpoint，operator 必须改用 `robot_config.navigation.command_server.action_name` 配置；保留旧参数的部署
必须迁移，不得作为 fallback 静默继续使用。

Runtime dispatch 必须使用 captured mapping；新增或重命名 role 必须升级 context schema，并改变
`robot_config_digest` 和 registry digest。v1 → v2 新增 `navigation_action` role 属于 context schema
升级：`context_schema_version` 从 `1` 升级为 `2`，`robot_config_digest` 和 `registry_digest` 随之改变。

Delegated executor descriptor 的所有字符串字段都必须存在；不适用模型部署的 executor 将三个 model 字段
规范化为空字符串。Descriptor registry 由 runtime owner 根据实际启动的 executor、endpoint 和 deployment
manifest 构建，不属于 robot_config SSOT。

`primitive_contracts` 必须直接来自 `embodied_common.primitive_contracts` 的 immutable canonical mapping；
compiler 不接受 caller 临时覆盖或 merge。传入的 `primitive_contract_digest` 必须由同一 mapping 重算并校验，
不一致时拒绝编译。

Descriptor 约束：`name` 必须等于 mapping key；`contract_version` 使用 SemVer；`endpoint_kind` 首版只允许
`ros_action` 或 `ros_service`；`configuration_digest` 是 executor 对其行为配置计算的 64 字符小写 SHA-256。
三个 model 字段必须全空或全为非空，模型 executor 的值必须来自部署 `inference_manifest`。Descriptor set 在
单个 generation 内不可变；发现实际 endpoint/deployment identity 变化时，runtime owner 必须触发受控 reload
或进入 not-ready，不能继续以旧 generation 接受请求。

### 9.2 编译顺序

Compiler 按以下 pass 执行：

**Pass 1：Source 和结构校验**

1. 解析唯一的 immutable source root 并计算 source release digest。
2. 加载 profile，并执行 profile schema 校验。
3. 扫描 catalog entry 目录，检查目录名、manifest name 和重复 name。
4. 只选择 profile 显式启用的 Atomic Operator 或 Skill。
5. 检查 profile 中引用的 manifest 和 implementation 是否存在。

**Pass 2：Schema 和引用校验**

6. 分别校验 manifest 和 implementation schema。
7. 校验 `semantic_level`、implementation tagged union、Primitive descriptor、executor、required args、
   timeout 和参数 schema，并执行 Atomic Operator 单 Primitive 门禁。
8. 检查 alias 冲突、`instead_use` 悬空引用和自引用。
9. 派生 enabled set 和 planner-visible set。

**Pass 3：展开和 Robot Context 兼容性**

10. 展开 trajectory template，生成最终 primitive sequence。
11. 校验展开前后绝对轨迹的首尾连续性。
12. 校验 named pose/target、joint name、joint limit、workspace、control mode 和 delegated executor descriptor。
13. 校验所有 Skill 局部限制不超过 robot_config 硬限制。

**Pass 4：派生视图和冻结**

14. 构建带 `semantic_level` 和显式 `planner_visible` 的 normalized templates、aliases、parameter schemas、
    requirements 和 capability view。
15. 将校验后的 `primitive_contract_digest` 写入 registry preimage，构建 registry、capability、provenance
    三类 digest preimage，并生成 canonical JSON 和 digest。
16. 递归冻结数据并返回 `SkillSnapshot`。

Compiler 必须尽量收集完整诊断集合，而不是遇到首个 YAML 问题立即退出。诊断按
`source_relative_path`、`error_code`、`field_path` 确定性排序。单个文件无法解析时可以跳过该文件的
后续 pass，但仍应继续检查不依赖该文件的其他 catalog entry。

### 9.3 数据源读取规则

数据源读取必须具备事务语义：

- 生产环境从不可变 release 目录读取。
- 开发环境从 staging 根目录读取，并在提交前后验证 source release digest 未变化。
- 如果读取期间目录树发生变化，编译必须失败或重试，不得混合两个版本的文件。

### 9.4 权威编译规则

`robot_config.loader` 负责构造 `SkillRobotContext` 并解析选中的 profile 名称，
但不管理运行时 generation，也不得执行第二次权威编译。

运行时 owner 只编译一次权威 bundle。离线 CLI 校验可以调用同一个 compiler，
但离线结果不具备权威 epoch 或 generation。

### 9.5 核心 Snapshot 模型与不可变性

建议模型：

```python
@dataclass(frozen=True)
class CompiledSkill:
    name: str
    version: str
    semantic_level: str
    skill_package_digest: str
    source_relative_path: str
    implementation_identity: str
    implementation_relative_path: str
    template: Mapping[str, Any]


@dataclass(frozen=True)
class SkillSnapshot:
    robot_name: str
    profile_name: str
    primitive_contract_digest: str
    robot_context: SkillRobotContext
    delegated_executors: Mapping[str, DelegatedExecutorDescriptor]
    templates: Mapping[str, Mapping[str, Any]]
    semantic_levels: Mapping[str, str]
    aliases: Mapping[str, tuple[str, ...]]
    parameter_schemas: Mapping[str, Mapping[str, Any]]
    requirements: Mapping[str, frozenset[str]]
    provenance: Mapping[str, Any]
    enabled_skill_names: tuple[str, ...]
    planner_visible_skill_names: tuple[str, ...]
    capability_view: Mapping[str, Any]
    registry_digest: str
    capability_digest: str
    provenance_digest: str
    registry_preimage_json: str
    capability_preimage_json: str
    provenance_preimage_json: str
    snapshot_json: str
```

`frozen=True` 不能自动冻结内部 dict。实现必须递归执行：

- list 转 tuple。
- set 转 frozenset。
- dict 转只读 mapping。
- consumer 只能获得只读引用或深拷贝。

所有 `*_preimage_json` 和 `*_digest` 字段都是 compiler 由同一冻结对象计算的 `init=False` 派生值；构造函数
不得同时接受调用方传入的对象、JSON 和 digest 三套值。

`CompiledSkill` 保留现有名称是为了限制首版代码和 ROS IDL 的迁移面；它表示任意 catalog entry，
包括 Atomic Operator。公开 API 同理继续使用 `SkillCommand` 和 `list-skills` 作为兼容命名，但 response
必须返回 `semantic_level`，调用方不得再假设所有 entry 都是复合 Skill。

`SkillSnapshot` 是 Primitive contract identity、robot context、executor descriptors、templates、requirements、
capability view、provenance 和 digest preimage 的唯一权威对象。Timeout policy 通过
`snapshot.robot_context.timeout_policy` 访问，不在 snapshot 或 runtime bundle 中保存第二份副本。

`snapshot.provenance` 的精确结构就是 `provenance_preimage_v1`；`provenance_preimage_json` 只是它的 canonical
serialization，不是第二份可独立修改的数据。Source release 和逐 Skill package digest 不进入
registry/capability digest，但必须进入 `provenance_digest`。只修改 `SKILL.md` 因此可以保持
registry/capability digest 不变，但会改变 provenance digest，并因 source provenance 变化产生新 generation。

### 9.6 `SkillSource` 与路径解析

所有 compiler、runtime 和 CLI 必须复用同一个 source abstraction：

```python
class SkillSource(Protocol):
    def resolve_active_release(self) -> SkillReleaseLocation: ...
    def discover_packages(self, release: SkillReleaseLocation) -> Sequence[SkillPackageLocation]: ...
    def load_profile(self, release: SkillReleaseLocation, profile_name: str) -> Mapping[str, Any]: ...
    def compute_release_digest(self, release: SkillReleaseLocation) -> str: ...
```

首版实现：

- `DirectoryReleaseSkillSource`：从 package share 下不可变 release 加载。
- `DevelopmentStagingSkillSource`：从显式开发 staging 目录加载，并在 compile 前后校验 release digest。

默认路径必须通过 ament package share 解析，禁止通过 `Path(__file__).parents[...]` 猜测源码或安装布局。
`skill_catalog_root` 仅为开发参数，生产模式必须拒绝任意外部路径。

### 9.7 Loader、Launch 与 CLI 集成

Legacy/profile 决策表：

| `skill_profile` | 非空 inline templates | 行为 |
|---|---|---|
| 未配置 | 否 | 空 catalog；若 embodied 被要求启用则启动失败。 |
| 未配置 | 是 | Legacy 模式，并打印一次 deprecation warning。 |
| 已配置 | 否 | Profile 模式。 |
| 已配置 | 是 | 启动失败，禁止自动 merge 和双 SSOT。 |

Profile 模式下：

- `robot_config.loader` 只输出 profile 名和 `SkillRobotContext`；runtime owner 从 delegated executor
  registry 提供 descriptor mapping，两者共同构造 `SkillCompileContext`。
- `embodied_bringup` 不再将 `skill_templates_json` 或 `skill_aliases_json` 作为权威参数注入。
- skill executor 编译初始 bundle；编译失败时节点不进入 ready 状态。
- Offline CLI 使用同一 source resolver 和 compiler，但不生成权威 epoch/generation。
- Runtime CLI 先读取 gateway status，再将精确版本绑定到 validate 和 action 请求。

## 10. 运行时模型

### 10.1 Registry 与 Coordinator

`SkillRegistry` 只管理纯 catalog 状态，不管理：

- 活动执行 lease
- 执行中的 admission ledger
- 进程级 busy 状态
- executor cancel 清理流程

上述状态属于 `skill_library` 中的 `ExecutionCoordinator`。

该拆分非常重要，因为 reload 后的新 policy 不能忽略已经存在的根执行 lease。

### 10.2 Runtime Bundle

Runtime bundle 只包含不可变数据：

```python
@dataclass(frozen=True)
class SkillRuntimeBundle:
    registry_epoch: str
    generation: int
    snapshot: SkillSnapshot
    gateway_policy_view: GatewayPolicyView
```

`GatewayPolicyView` 只包含 generation-specific 的不可变准入规则、timeout policy 和 Skill requirement，
不得持有活动 lease、request ledger 或 busy 状态。它由 `SkillSnapshot` 构造，只保存指向 snapshot 冻结值的
只读索引，不复制 timeout policy 或 requirements。Capability view、requirements 和 parameter schemas 统一从
`bundle.snapshot` 读取。Bundle 不包含活动 lease 或请求 ledger。

### 10.3 Reload 事务

Reload 采用暂存后切换流程：

1. 获取 reload mutex，并设置 `_reload_in_progress`。
2. 从当前 catalog 数据源构建 candidate snapshot。
3. 完整校验 candidate。
4. 构建 candidate `GatewayPolicyView`、capability view 和其他 runtime 输入。
5. 比较 candidate snapshot 的 `registry_digest`、`provenance_digest` 与当前值。
6. 两者都没有变化时，保持 generation 不变并按 no-op 返回。
7. 如果发生变化，在提交前计算 `candidate_generation = current_generation + 1`。
8. 构建包含完整 epoch/generation 的 candidate bundle。
9. 在同一个 executor state lock 内同时提交 registry record 和 current bundle pointer。
10. 释放 state lock；在仍持有 reload mutex 时发布成功事件，保证连续 generation 的事件顺序。
11. 在 `finally` 中仍持有 `reload_mutex` 时重新获取 `executor_state_lock`，清除 `_reload_in_progress`，释放
    state lock，最后释放 reload mutex；任何读取或清理该 flag 的路径都必须持有 state lock。

候选构建或提交前校验失败时，当前 snapshot、bundle、generation 和 digest 均不得改变；失败只通过
service response、日志和 diagnostics 表达，不执行“写回旧版本”。

旧 bundle 在仍有活动 root Skill、Workflow 或延迟 cleanup 使用时继续保持有效，直到对应引用全部释放。

### 10.4 Active Execution 与历史保留

```python
@dataclass(frozen=True)
class CanonicalWorkflowStep:
    schema_version: int
    skill_name: str
    target_name: str
    place_name: str
    motion_direction: str
    motion_distance: float
    timeout_sec: float


@dataclass(frozen=True)
class ActiveRootExecution:
    root_task_id: str
    owner_kind: str
    bundle: SkillRuntimeBundle
    task_budget: TaskBudget
    workflow_digest: str
    root_lease_nonce: str
    lease_token: ExecutionLeaseToken
    workflow_steps: tuple[CanonicalWorkflowStep, ...]


@dataclass
class ActiveSkillExecution:
    root_execution: ActiveRootExecution
    bundle: SkillRuntimeBundle
    admission_token: AdmissionToken
    execution_lease: ExecutionLeaseToken | ExecutionLeaseBorrow
```

`ActiveRootExecution` 只保存 Begin 时 canonicalize 后的 immutable root metadata。Coordinator 另行维护可变的
`next_step_index`、每个 step 的 `PENDING/ACTIVE/SUCCEEDED/FAILED/CANCELED` 状态、当前 child、active primitive
dispatch 和 cleanup 引用；这些状态不得回写 root metadata，也不得从 current bundle 重新派生 Workflow steps。

整个 goal 生命周期中的参数校验、Safety 请求、primitive dispatch、cancel、late cleanup 和 finalize，
必须使用 admission 时捕获的 bundle 和 token。不得在 callback 中重新读取 `self._current_bundle` 或新
generation 的 policy view。

直接 root `SkillCommand` 在 admission 时创建 `ActiveRootExecution(owner_kind="catalog_entry")`，该 entry 完成后
释放 root lease。Workflow 则在第一步前创建 `ActiveRootExecution(owner_kind="workflow")`，所有 child catalog
execution 只持有 `ExecutionLeaseBorrow`，不得释放或替换 root lease。`workflow_digest` 对直接 entry 为空，对
Workflow 必须非空。

`ExecutionCoordinator` 在 reload 前后保持同一个实例，并持有全局 lease、ledger 和 busy 状态。历史
bundle 使用引用计数或等价的 lease tracking 保留；只有在活动引用和延迟 cleanup 都释放后才可清理。

Coordinator 另外持有当前 `registry_epoch` 内的 immutable `WorkflowTerminalRecord` ledger。每条记录保存
`(root_task_id, workflow_digest, exact identity, root_lease_nonce, canonical started_at, canonical deadline,
terminal_state, completed_step_count)`、实际 terminal response 和完成时间，但不再持有 bundle、lease 或 live
budget 对象。Finalize 的线性化点必须先在
coordinator lock 下写入该 record，再释放 root lease、bundle retention 和 root budget；因此 cleanup 后的相同
binding 仍能返回既有 terminal result。Terminal record 在 epoch 结束前完整保留，不按普通 ledger capacity 淘汰；
root task ID 在该 epoch 内不得复用；
重启会创建新 epoch，旧 binding 按 epoch mismatch 拒绝。

默认 retention policy 固定为：current bundle、所有仍被活动/cleanup 引用的 bundle，以及最近两个已完成
generation。运行参数 `snapshot_retention_generations` 默认 `2` 且不得小于 `1`；它只影响无活动引用的历史
缓存，不得回收活动 generation。Executor 发布 generation event 前必须已将新 bundle 纳入 retention 表。

Delegated dispatch 必须携带 captured descriptor 的全部 identity 字段。非模型 executor 的三个 model 字段
保持全空；模型 executor 必须携带来自 deployment manifest 的三个非空值。Delegated executor 在接受请求前
必须比对自身实际 identity，不匹配时返回稳定版本错误；不能只凭 endpoint 名称调用。

### 10.5 Workflow Execution Scope

Workflow owner 是 `embodied_agent`，但跨步骤 root lease 和 bundle retention 由 `ExecutionCoordinator` 管理。
`task_executor_node` 必须在第一个步骤前调用 `/embodied/begin_workflow_execution`，成功后才可发送 child
`SkillCommand`；在成功、失败或取消路径上都必须调用 `/embodied/finalize_workflow_execution`。

Begin 的线性化事务必须完成：

1. 校验 planned `TaskCommand` 的 schema、非空 typed steps、task budget 和 expected registry identity。
2. 对规范化后的完整 plan 计算 `workflow_digest`，并与 planner 填写值比较。
3. 在 `executor_state_lock` 下捕获 exact current bundle；普通 stale plan 不允许从 history 恢复。
4. 在 coordinator 中登记 root budget、获取唯一 root lease，并生成至少 128 bit 随机 `root_lease_nonce`。
5. 将 `(root_task_id, workflow_digest, exact version, budget, root_lease_nonce, ordered step payloads)` 记录为
   immutable `ActiveRootExecution`，并为捕获 bundle 增加 retention 引用。

每个 child `SkillCommand` 必须携带同一个 `root_task_id`、budget、exact version、`workflow_digest` 和
`root_lease_nonce`，以及对应的零基 `workflow_step_index`。Gateway 必须将 action goal 的 entry name 和参数
规范化后，与 Begin 时捕获的该 index step 逐字节比较；step 重放、跳步、越界、不同参数、不同版本或错误 nonce
全部 fail closed。首版按严格顺序执行，只有当前 step 成功终态后 coordinator 才接受下一 index。

每个 step 的 coordinator 状态机固定为：

```text
PENDING -> ACTIVE -> SUCCEEDED
                 +-> FAILED
                 +-> CANCELED
```

`ValidateSkill` 是 `safety_guard` 拥有的只读 preflight：它验证 exact snapshot、entry 参数和当前机器人状态，
但不查询或修改 coordinator，也不得把存在 `root_lease_nonce` 解释成已获执行授权。Workflow root、nonce、digest、
expected index 和完整 step payload 的权威校验只在后续 Gateway action admission 中执行；因此 preflight 成功不
保证 admission 成功。Validation 不得预留 step、改变 `PENDING`、写 child ledger 或消费重试机会。
`SkillCommand` action admission 是唯一允许执行 `PENDING -> ACTIVE` 的路径；相同 child task ID 和 payload 的
action retry 返回已有 active/terminal record，不得再次 dispatch。不同 payload 使用同一 child ID 返回 conflict。
成功完成执行
`ACTIVE -> SUCCEEDED` 并将 expected index 加一；失败或取消进入 terminal step 状态，Workflow 不得继续下一步，
必须先 cleanup 后 Finalize root。`FAILED/CANCELED` 的 step 不允许使用新 child ID 重试；首版重试或 replan 必须
Finalize 旧 Workflow，并创建新的 root task ID 和 `workflow_digest`。

Child admission 从 active root 捕获旧 bundle，而不是读取 current bundle。因此 reload 后尚未开始的后续步骤仍
使用 Workflow 开始时的 exact snapshot。非 Workflow root command 在 lease 存续期间返回 busy，不能插入步骤
之间。Child 完成只关闭自己的 ledger record 和 lease borrow；只有 Workflow finalize 才释放 root lease、bundle
retention 和 root budget record。

Finalize 必须幂等：相同 `(root_task_id, workflow_digest, root_lease_nonce)` 的重复请求返回既有 terminal result；
nonce 或 digest 冲突时拒绝，不能释放其他执行。Coordinator 必须强制执行以下门禁，而不能只依赖
`task_executor_node` 的调用顺序：

1. 仍有 ACTIVE child、Primitive/delegated dispatch、late cleanup 或未释放 lease borrow 时，Finalize 返回
   `SKILL_WORKFLOW_LEASE_MISMATCH`，不得释放 root lease。
2. `SUCCEEDED` 只在每个 canonical Workflow step 都是 `SUCCEEDED`、`next_step_index == len(workflow_steps)` 且没有
   active 引用时允许；caller 的 `completed_step_count` 不能将 partial Workflow 标记成功。
3. `FAILED`/`CANCELED` 只在对应 child 已进入相同 terminal state、没有活动引用且 root 处于可终止状态时允许。
   child 尚未到达 Gateway admission 时 step 仍为 `PENDING`，且不存在 active child；这种 pre-dispatch 失败或取消
   允许 root 直接终止。一旦 step 进入 `ACTIVE`，必须等待并匹配 child 的真实 terminal state。
4. 已存在 terminal record 时，相同 terminal payload 的重复 Finalize 返回既有结果；不同 terminal state 或 count
   返回 `SKILL_REQUEST_ID_CONFLICT`，不能释放其他执行。仍处于 active root 时，terminal state/count 的请求值
   只在后续 active-root 校验中判断。

Finalize 的检查优先级固定为，且每一项按列出的字段顺序 first-failure-wins。ROS transport 已先完成固定 IDL
字段存在性和类型检查；coordinator 使用以下应用层顺序：

1. `DispatchBinding.schema_version`、`TaskBudget.schema_version`、terminal enum、
   `workflow_step_index == 0` 和 `dispatch_nonce == ""`；schema/type 失败返回 `SKILL_SCHEMA_INVALID`，step
   sentinel 失败返回 `SKILL_WORKFLOW_STEP_MISMATCH`，dispatch sentinel 失败返回 `SKILL_DISPATCH_NOT_AUTHORIZED`；
2. `task_id == root_task_id`，然后按 `root_task_id` 查找 active 或 terminal root record；关系不等或 root record
   不存在返回 `SKILL_WORKFLOW_LEASE_MISMATCH`；
3. `expected_registry_epoch`、`expected_registry_generation`、`expected_registry_digest`；epoch 不一致返回
   `SKILL_REGISTRY_EPOCH_MISMATCH`，generation 或 registry digest 不一致返回
   `SKILL_REGISTRY_VERSION_MISMATCH`；
4. `workflow_digest`、`root_lease_nonce`、`TaskBudget.started_at`、`TaskBudget.deadline`；依次分别返回
   `SKILL_WORKFLOW_DIGEST_MISMATCH`、`SKILL_WORKFLOW_LEASE_MISMATCH`、`SKILL_TASK_BUDGET_MISMATCH`、
   `SKILL_TASK_BUDGET_MISMATCH`；`DispatchBinding` 没有未覆盖的 identity 字段；
5. 已完成的 `WorkflowTerminalRecord`：完全相同的 terminal payload 返回既有结果，不同 state/count 返回
   `SKILL_REQUEST_ID_CONFLICT`；
6. active child、Primitive/delegated dispatch、late cleanup 和 lease borrow 门禁；任一不为空返回
   `SKILL_WORKFLOW_LEASE_MISMATCH`；
7. active root 的 `terminal_state`、`completed_step_count` 和每个 step 的 terminal 状态；前两者与 coordinator
   当前可终止状态或实际 count 不一致返回 `SKILL_REQUEST_ID_CONFLICT`，step ledger 状态不一致返回
   `SKILL_WORKFLOW_STEP_MISMATCH`。

每个字段的失败使用错误表中对应的稳定错误码；因此 binding 错误优先于 terminal payload 冲突，active cleanup
门禁优先于 active root 的 terminal state/count 冲突，任何实现都不得改变该顺序。

`task_executor_node` 必须先 cancel/等待活动 child cleanup，再 Finalize。若 task executor 崩溃，coordinator 在 root
deadline 到期后进入受控 cancel/cleanup；所有 child 和 Primitive dispatch nonce 失效后才能释放 lease。不能仅因
begin client 断连就立即释放正在运动的执行。

`root_lease_nonce` 和 `dispatch_nonce` 职责不同：前者授权一个受信 task executor 在已登记 Workflow 中提交下一
个 catalog entry，后者授权 Gateway 在单个 admitted child 内调用 Primitive 或 delegated executor。二者均是
进程内 capability gate 的跨进程表示，不替代 SROS 2 身份认证；begin/finalize service、Primitive action 和
delegated endpoint 都必须受部署 policy 限制。

### 10.6 锁顺序与线性化点

全局锁顺序固定为：

```text
reload_mutex -> executor_state_lock -> coordinator_lock
```

任何路径不得反向获取。规则：

1. Reload 获取 `reload_mutex` 后，在短暂持有 `executor_state_lock` 时设置
   `_reload_in_progress=true`，随后释放 state lock 进行 candidate 编译。
2. Direct root goal admission 或 Workflow Begin 获取 `executor_state_lock`，检查 ready/reload 状态并捕获
   current bundle；在仍持有 state lock 时调用 coordinator，在 `coordinator_lock` 下创建 root execution 和
   lease token。
3. Workflow child admission 按同一锁顺序查找 active root，验证 nonce、digest、step index 和 payload，并借用
   root 已捕获的 bundle；不得因为 current generation 已变化而改绑或拒绝合法 child。
4. Root admission 的线性化点是 coordinator 成功登记 root lease 的时刻；child admission 的线性化点是
   coordinator 成功推进 expected step state 的时刻。
5. Reload 的线性化点是在 `executor_state_lock` 下同时更新 registry record 和 bundle pointer 的时刻。
6. Dispatch 阶段不得长期持有上述任一锁。
7. Child cancel/finalize 只使用捕获 token 获取 coordinator lock，不重新获取 state lock。Root Workflow
   finalize 和 deadline cleanup 按 `executor_state_lock -> coordinator_lock` 标记 terminal、释放 retention 和
   root lease，不得先持有 coordinator lock 再获取 state lock。
8. Reload 不获取 coordinator lock，也不替换 coordinator 实例。

`SkillRegistry` 不拥有独立 mutable lock。Current record、current bundle、历史 bundle 表、retention 引用和
reload `request_id` 幂等缓存全部只受 `executor_state_lock` 保护。Candidate 编译不持有该锁。Status、snapshot
查询和 history cleanup 在 state lock 内捕获只读引用后立即释放；序列化和 ROS response 构建在锁外执行。
Shutdown 只按 `reload_mutex -> executor_state_lock -> coordinator_lock` 顺序获取，禁止新增隐式 registry lock。

### 10.7 Timeout 解析

Timeout 的硬策略来自 `SkillRobotContext.timeout_policy`。Implementation 的 `timeout_sec` 表示该 Skill 的
默认值和最大允许执行时间，不是新的全局 SSOT。

解析规则：

1. `skill_timeout_cap = implementation.timeout_sec`；未配置时使用 robot policy 的
   `default_skill_timeout_sec`。
2. `skill_timeout_cap` 必须为有限正数，且不超过 robot `task_budget_sec`。
3. 请求 `timeout_sec <= 0` 时使用 `skill_timeout_cap`。
4. 正数请求 timeout 超过 `skill_timeout_cap` 或剩余 task budget 时拒绝，不静默 clamp。
5. `effective_timeout` 是校验后的请求值，并受剩余 task deadline 约束。
6. RPC wait timeout 使用 robot policy 的 `rpc_timeout_sec`，不得从 Skill implementation 覆盖。
7. 静态 resolved skill cap、task budget policy 和 RPC timeout 进入 registry digest；公开 timeout 能力进入
   capability digest。请求特定的 deadline、remaining budget 和 `effective_timeout` 不进入任何 catalog digest。

Primitive 和 delegated executor 使用同一解析算法。

Task deadline 由任务入口所有：

1. `task_entry_node` 或等价受信 API boundary 使用统一 ROS clock 创建 `TaskBudget(started_at, deadline)`，其中
   `deadline <= started_at + task_budget_sec`。
2. Planner、`TaskCommand`、task executor、`SkillCommand`、delegated executor 和 primitive dispatch 必须原样
   传播同一 budget，不得为每个 catalog entry 重置。
3. 直接发起 root `SkillCommand` 且 budget 为零值时，Gateway 成为 owner，在 admission 时创建 budget；非零
   budget 必须满足当前 policy。
4. `ExecutionCoordinator` 的 task ledger 按 `root_task_id` 记录首次接受的 budget；Workflow 子步骤使用不同
   `task_id` 但必须保留同一 root 和 budget，后续请求修改 started_at 或 deadline 时 fail closed。
5. 所有进程必须使用同一 ROS time domain；仿真使用 `/clock`。Deadline 已过时返回受控 timeout 错误。

### 10.8 重启语义

运行时 owner 重启时会创建新的 epoch 和新的 coordinator。系统不假设能够从
内存中恢复执行中的状态。

运行规则：

- 在新 bundle 加载完成前拒绝新的机器人运动。
- 持有旧状态的 consumer 必须同步到新 epoch。
- 崩溃恢复属于独立功能，不是本设计隐含提供的能力。

## 11. ROS 接口

文档使用稳定的 `/embodied/...` 接口名，不依赖节点私有 `~/...` 展开结果：

- `/embodied/reload_skills`
- `/embodied/get_skill_snapshot`
- `/embodied/skill_registry_event`
- `/embodied/get_skill_gateway_status`
- `/embodied/begin_workflow_execution`
- `/embodied/finalize_workflow_execution`

多机器人部署可以增加统一 ROS namespace，但 namespace 内的相对接口名必须保持稳定。
Begin/Finalize 是 `embodied_agent` task executor 与 `ExecutionCoordinator` 之间的内部执行范围接口，不是 Agent、
CLI 或用户可发现 capability；部署 policy 必须限制其 caller。

### 11.1 结构化诊断

所有 catalog 编译入口使用同一个诊断消息：

```text
# SkillDiagnostic.msg
uint8 ERROR=1
uint8 WARNING=2

uint32 schema_version
uint8 severity
string error_code
string source_relative_path
string field_path
string message
```

`schema_version` 首版为 `1`。`source_relative_path` 必须相对于 release root；没有对应文件或字段时使用空
字符串。Diagnostics 按 `source_relative_path`、`error_code`、`field_path`、`message` 排序。Service
response 的顶层 `error_code` 使用排序后第一个 ERROR；只有 WARNING 时仍可 `success=true`。

### 11.2 Reload Service

`std_srvs/Trigger` 无法表达足够的结构化结果，因此本设计使用自定义 service。

生产环境应默认禁用该 service，或通过 operator policy 限制调用权限。Service
始终 reload 已配置的数据源根目录，不得允许请求方传入任意路径。

建议 service：

```text
# ReloadSkillCatalog.srv
uint32 schema_version
string request_id
bool force
---
bool success
string registry_epoch
uint64 old_generation
uint64 generation
string registry_digest
string capability_digest
string source_release_digest
string provenance_digest
string error_code
string message
string[] changed_skills
SkillDiagnostic[] diagnostics
```

语义：

- `schema_version` 首版必须为 `1`。
- `request_id` 用于日志关联和短期幂等；同一 request ID 的重复调用应返回同一已完成结果。
- 同一 request ID 但 request 字段不同返回 `SKILL_REQUEST_ID_CONFLICT`。幂等缓存默认保留 600 秒、最多
  1024 条，按完成时间淘汰；缓存由 `executor_state_lock` 保护。
- `force=true` 只强制重新编译和校验；当 `registry_digest` 和 `provenance_digest` 都未变化时，不允许
  人为递增 generation。
- Reload 只能使用启动时配置的 source resolver，不得接受任意文件路径。
- 成功但无变化时返回 `success=true`、`old_generation == generation`、空 `changed_skills`。

### 11.3 Snapshot Service

快照请求必须支持精确版本查询。

```text
# GetSkillSnapshot.srv
uint32 schema_version
string registry_epoch
uint64 generation
---
bool success
string registry_epoch
uint64 generation
string registry_digest
string capability_digest
string source_release_digest
string provenance_digest
string profile_name
string snapshot_json
string error_code
string message
```

查询语义：

1. `generation == 0` 且 epoch 为空：返回当前 bundle。
2. `generation == 0` 且 epoch 非空：epoch 必须等于当前 epoch，否则返回 epoch mismatch。
3. `generation > 0`：必须精确匹配 `(epoch, generation)`，不允许自动升级到更新版本。
4. 精确快照已经回收时返回 `SKILL_SNAPSHOT_NOT_RETAINED`。

`snapshot_json` 必须严格使用 6.4 节的 `snapshot_payload_v1`，不得增加 consumer 私有字段。Runtime consumer
从完整 `registry_preimage` 派生 templates、aliases、parameter schemas、requirements、robot context 和
executor descriptors，并读取 `primitive_contract_digest`；从 `capability_preimage` 派生公开 capability view，
从 `provenance_preimage` 读取 source/package identity。执行型 consumer 必须将 snapshot 中的 Primitive digest
与本地 canonical descriptor registry 比较，不匹配时不得进入 ready。

该 service 是运行节点之间的内部同步接口，不替代面向 Agent 的公开 capability catalog。

### 11.4 Registry Event

```text
# SkillRegistryEvent.msg
uint32 schema_version
string registry_epoch
uint64 old_generation
uint64 new_generation
string registry_digest
string capability_digest
string source_release_digest
string provenance_digest
string profile_name
string[] changed_skills
```

事件只在成功后发布，并通知延迟加入的 subscriber 应查询哪个 epoch。

QoS 固定为：

- Reliability：`RELIABLE`。
- Durability：`TRANSIENT_LOCAL`。
- History：`KEEP_LAST`。
- Depth：`1`。

`changed_skills` 比较完整 Skill package 的 manifest、selected implementation、enabled 状态和
planner-visible 状态。只有 profile、robot context、source documentation 等非单 entry 执行内容变化时，
允许为空。

### 11.5 Status 与 Validation

Stage 0 冻结以下目标 IDL。它们保留当前业务字段语义，并将下游 dispatch 的 task/version envelope 收敛到
强类型 binding；不得由各 consumer 自行选择字段名。

```text
# TaskBudget.msg
uint32 schema_version
builtin_interfaces/Time started_at
builtin_interfaces/Time deadline
```

```text
# DispatchBinding.msg
uint32 schema_version
string task_id
string root_task_id
TaskBudget task_budget
string expected_registry_epoch
uint64 expected_registry_generation
string expected_registry_digest
string workflow_digest
uint32 workflow_step_index
string root_lease_nonce
string dispatch_nonce
```

`task_id` 标识当前 dispatch，`root_task_id` 在整个 Workflow 中保持不变。根 Task 或直接根 Skill 的两者
相等；`task_executor_node` 为每个步骤派生稳定 child `task_id`，但保留原 `root_task_id`。Task budget ledger
按 `root_task_id` 建账，请求幂等和 payload conflict 按当前 `task_id` 判断。

`dispatch_nonce` 只用于 Gateway 已准入后的内部 delegated/Primitive dispatch。根 `TaskCommand`、
`ValidateSkill` 和 `SkillCommand` 必须为空；`ExecutionCoordinator` 为内部 dispatch 生成至少 128 bit 随机
nonce，并将 `(root_task_id, task_id, version, root lease, nonce)` 绑定到 active lease。Primitive server 必须逐项
匹配 active binding，不能仅凭相同 task ID 放行；child finalize/cancel 后 nonce 立即失效。

`root_lease_nonce` 只用于已成功 Begin 的 Workflow child `SkillCommand`、其只读 validation 和 internal dispatch。Raw
或 planned `TaskCommand`、direct root `SkillCommand` 和 Begin request 中必须为空；planned `TaskCommand` 必须已
填写 `workflow_digest`，child 还必须填写对应 `workflow_step_index`。两个 nonce 都不得写入日志、snapshot 或
持久化文件，也都不是网络认证的替代品；存在非受信 ROS participant 的部署必须使用 SROS 2 policy 限制
begin/finalize service、Primitive action 和 delegated endpoint。

Root-scope binding 没有实际 step，统一使用 `workflow_step_index=0` 作为 canonical sentinel：planned
`TaskCommand`、Begin 和 Finalize request 都必须为 `0`；direct root `SkillCommand` 也必须为 `0`。该 sentinel
不进入 `workflow_digest`。只有 Workflow child `ValidateSkill`/`SkillCommand` 和其内部 dispatch 才按实际零基
step index 填写。任何其他组合均返回 `SKILL_WORKFLOW_STEP_MISMATCH`。

```text
# DelegatedExecutorIdentity.msg
uint32 schema_version
string name
string contract_version
string endpoint_kind
string endpoint_name
string configuration_digest
string model_deployment_name
string model_fingerprint
string model_bundle_digest
```

```text
# WorkflowStep.msg
uint32 schema_version
string skill_name
string target_name
string place_name
string motion_direction
float32 motion_distance
float32 timeout_sec
```

`WorkflowStep.skill_name` 是兼容字段名，实际可以引用 `semantic_level=atomic_operator` 或 `skill` 的 catalog
entry。Step 不携带独立 registry identity，必须继承所在 planned `TaskCommand.dispatch_binding` 的完整版本；
该精确 snapshot 已唯一确定 entry 的 manifest SemVer 和 implementation，不再复制 step-level version 字段。
`timeout_sec <= 0` 使用 entry 默认值，正值仍受 Skill cap 和共享 TaskBudget 限制。

`workflow_digest` 使用 6.4 节 canonical JSON 规则，对以下精确 preimage 计算 SHA-256；所有数值必须先按
`WorkflowStep`/ROS 字段类型规范化，并拒绝 NaN、Infinity 和负 timeout：

```json
{
  "schema_version": 1,
  "root_task_id": "...",
  "task_budget": {
    "started_at": {"sec": 0, "nanosec": 0},
    "deadline": {"sec": 0, "nanosec": 0}
  },
  "expected_registry_epoch": "...",
  "expected_registry_generation": 1,
  "expected_registry_digest": "...",
  "workflow_steps": []
}
```

`workflow_steps` 保持计划顺序，每个元素精确包含 `WorkflowStep` 的全部字段。Planner 和 coordinator 必须调用
`embodied_common` 的同一 digest helper，不得各自实现序列化。`root_lease_nonce`、`dispatch_nonce`、raw command、
planner annotations 和 task priority 不进入 preimage。

```text
# BeginWorkflowExecution.srv
DispatchBinding dispatch_binding
WorkflowStep[] workflow_steps
---
bool success
string root_lease_nonce
string workflow_digest
string actual_registry_epoch
uint64 actual_registry_generation
string actual_registry_digest
string error_code
string message
```

Begin request 必须满足 `task_id == root_task_id`、非空 steps、完整 expected identity、非空
`workflow_digest`、`workflow_step_index=0`，且两个 nonce 为空。Coordinator 重算 digest、捕获 current bundle
并原子取得 root lease。
同一 root ID、digest、identity 和 budget 的重试返回同一个 active nonce；任一字段冲突返回
`SKILL_REQUEST_ID_CONFLICT`，不得替换既有 root execution。

```text
# FinalizeWorkflowExecution.srv
uint8 SUCCEEDED=1
uint8 FAILED=2
uint8 CANCELED=3

DispatchBinding dispatch_binding
uint8 terminal_state
uint32 completed_step_count
---
bool success
uint8 actual_terminal_state
uint32 actual_completed_step_count
string error_code
string message
```

Finalize request 必须满足 `task_id == root_task_id`，携带 Begin 返回的 `root_lease_nonce` 和相同
`workflow_digest`/identity/budget，`workflow_step_index=0`，且 `dispatch_nonce` 为空。Coordinator 以自身 ledger
中的 completed index 为权威，caller 的 count 只用于冲突检测。相同 terminal payload 的重试幂等；不同
terminal state 或 count 返回 `SKILL_REQUEST_ID_CONFLICT`，不同 digest 或 nonce 返回对应的 Workflow digest/lease
错误，任何冲突都不得释放 lease。

所有跨进程 primitive 和 delegated dispatch 必须使用上述 binding。首版 `PrimitiveCommand.action` 目标 IDL
完整冻结为：

```text
# PrimitiveCommand.action goal
DispatchBinding dispatch_binding
string primitive_name
string pose_name
geometry_msgs/Pose target_pose
float32 relative_dx
float32 relative_dy
float32 relative_dz
float32 velocity_scaling
float32 gripper_position
string[] joint_names
float32[] joint_positions
float32 primitive_duration_sec
float32[] joint_waypoints
uint32 joint_waypoint_count
float32 waypoint_duration_sec
float32 timeout_sec
---
# result
bool success
string error_code
string message
string pose_name
string actual_registry_epoch
uint64 actual_registry_generation
string actual_registry_digest
---
# feedback
string state
string detail
```

当前 delegated `PickObject.action` 目标 IDL 完整冻结为：

```text
# PickObject.action goal
DispatchBinding dispatch_binding
string target_query
float32 timeout_sec
DelegatedExecutorIdentity expected_executor
---
# result
uint8 VERIFICATION_NOT_RUN=0
uint8 VERIFICATION_SUCCESS=1
uint8 VERIFICATION_FAILED=2
uint8 VERIFICATION_UNCERTAIN=3

bool success
string error_code
string message
uint32 attempts
uint8 verification_status
float32 verification_confidence
string debug_output_dir
string[] completed_phases
DelegatedExecutorIdentity actual_executor
---
# feedback
string phase
float32 progress
uint32 attempt
string detail
```

未来新增 delegated action/service 也必须在 goal/request 中包含 `DispatchBinding` 和
`DelegatedExecutorIdentity expected_executor`，并在 result/response 中返回 actual identity。

```text
# SkillCapabilityStatus.msg
uint32 schema_version
string name
string semantic_level
bool planner_visible
bool ready
string reason
string required_control_mode
```

```text
# GetSkillGatewayStatus.srv
uint32 schema_version
string task_id
string payload_hash
---
uint32 schema_version
string robot_name
bool motion_authorized
string active_control_mode
bool busy
string active_task_id
string active_owner_kind
string active_workflow_digest
int32 active_workflow_step_index
bool control_plane_ready
string control_plane_state
string control_plane_error_code
float32 default_skill_timeout_sec
float32 task_budget_sec
float32 rpc_timeout_sec
string config_digest
string capability_digest
string registry_epoch
uint64 registry_generation
string registry_digest
string primitive_contract_digest
string source_release_digest
string provenance_digest
uint64[] retained_generations
string request_state
string request_error_code
SkillCapabilityStatus[] capabilities
```

Status 不得暴露任一 nonce。`active_owner_kind` 只能为空、`catalog_entry` 或 `workflow`；没有活动 Workflow 时
`active_workflow_digest` 为空且 `active_workflow_step_index=-1`。

`control_plane_ready=true` 的唯一条件是：status response 可达；当前 registry snapshot 已完整同步且三类 digest
可本地重算；`primitive_contract_digest`、robot context 和 executor identity 校验通过；Agent plan store 的
plan/validate/confirm/execute endpoint 可达；当前不处于 initial compile、registry resync 或 reload transaction。
`control_plane_state` 取 `STARTING`、`SYNCING`、`READY`、`RELOADING` 或 `FAILED`；失败时必须填写稳定的
`control_plane_error_code`，通常为 `SKILL_REGISTRY_NOT_READY`、`SKILL_RELOAD_IN_PROGRESS`、
`SKILL_SNAPSHOT_DIGEST_MISMATCH` 或 `SKILL_EXECUTOR_IDENTITY_MISMATCH`。

`control_plane_ready` 不包含 `motion_authorized`、单个 capability 的 `ready`、busy、control mode 或 workspace
安全条件。这样 `hermes-robot` 可以在 `control_plane_state=READY` 且 `motion_authorized=false` 时保持运行，允许
catalog/plan/validate，并在实际 action admission 时由 Gateway 返回 `MOTION_NOT_AUTHORIZED`；`request_state` 和
`request_error_code` 仍只表达请求 task ledger 查询，不可被 launcher 当作 control-plane readiness。

```text
# ValidateSkill.srv
DispatchBinding dispatch_binding
string skill_name
string target_name
string place_name
string motion_direction
float32 motion_distance
---
bool allowed
string reason
string error_code
string actual_registry_epoch
uint64 actual_registry_generation
string actual_registry_digest
SkillDiagnostic[] diagnostics
```

```text
# SkillCommand.action goal
DispatchBinding dispatch_binding
string skill_name
string target_name
string place_name
string motion_direction
float32 motion_distance
float32 timeout_sec
---
# result
bool success
string error_code
string message
string[] executed_primitives
string actual_registry_epoch
uint64 actual_registry_generation
string actual_registry_digest
string source_release_digest
string provenance_digest
SkillDiagnostic[] diagnostics
---
# feedback
string state
string detail
string actual_registry_epoch
uint64 actual_registry_generation
```

```text
# TaskCommand.msg
DispatchBinding dispatch_binding
string source
string raw_command
string task_type
WorkflowStep[] workflow_steps
string target_name
string place_name
string motion_direction
float32 motion_distance
uint8 priority
float32 timeout_sec
string context_json
```

```text
# TaskStatus.msg
uint32 schema_version
string task_id
string state
bool success
string current_skill
string[] completed_skills
string error_code
string message
bool recoverable
bool replan_requested
string actual_registry_epoch
uint64 actual_registry_generation
string actual_registry_digest
string provenance_digest
```

兼容与校验规则：

1. `config_digest` 是现有公开语义的字段名，必须逐字节等于 `capability_digest`；更新后的 consumer 以
   `capability_digest` 为主，但可继续读取该别名。移除别名需要单独的 breaking IDL release。
2. Admission 看到的 registry 标识与 expected 标识不同时 fail closed，并返回
   `SKILL_REGISTRY_VERSION_MISMATCH`。
3. Action result、validation response 和 status response 返回实际使用或观察到的 identity。
4. `context_json` 不再承载 registry identity、task budget、`skill_sequence` 或其他强类型契约字段；它只保留
   非执行性的 planner/perception annotations。
5. 所有新增 request/binding 的 `schema_version` 首版为 `1`，其他值返回 `SKILL_SCHEMA_INVALID`。
6. Planned `TaskCommand`、`ValidateSkill` 和 `SkillCommand` 的 `dispatch_binding` 必须提供完整 expected
   identity；只有进入 planner 前的 raw `TaskCommand` 允许 identity 为空。Planner 输出时必须填充 identity 和
   `workflow_digest`，但 `root_lease_nonce`、`dispatch_nonce` 仍为空。
7. `TaskCommand.timeout_sec` 是任务入口请求的上限，budget owner 用它与 policy 取较小值后生成
   `TaskBudget`；`SkillCommand.timeout_sec` 是单 catalog entry 请求上限。
8. 修改同名 ROS `.srv/.msg/.action` 会改变 type identity，不提供旧二进制 wire compatibility。Stage 0 必须
   协调重编译并部署全部 workspace producer/consumer；所有重编译后的 caller 必须显式设置
   `schema_version=1`，不得依赖默认零值。
9. Planner 前的 raw `TaskCommand` 允许 `workflow_steps` 为空；planned `TaskCommand` 必须携带非空 typed
   steps、完整版本 binding、正确 `workflow_digest`，且每一步 `schema_version=1`。Task executor 禁止从
   `context_json` 回退读取旧 `skill_sequence`，也不得跳过 Workflow Begin 直接发送 child。
10. 自动 planner 只能生成当前 snapshot 中 `planner_visible=true` 的 entry；direct API 可以构造包含任意
    enabled entry 的单步 Task。`planner_visible=false` 不是授权拒绝，二者都不得引用 Primitive 名称。
11. Planned Task 顶层 `target_name/place_name/motion_*` 只保留 raw request hint 兼容语义，Task executor 必须
    使用每个 `WorkflowStep` 自己的参数，禁止用顶层值覆盖全部步骤。
12. `SkillCommand` 和既有 `ValidateSkill` 名称在 v1 同时承载 Atomic Operator 与 Skill，以控制 IDL 迁移面；
    admission 必须从 snapshot 读取 `semantic_level` 并验证 v1 结构不变量，不允许 caller 自报层级。
13. `semantic_level` 不得改变 Gateway 的认证、motion authorization、control mode、lease、timeout、safety 或
    version binding 路径；任何基于层级的 planner 展示或 lint 都不能成为执行绕过条件。
14. Schema v1 的四个参数字段必须同时存在于 `WorkflowStep`、`ValidateSkill` 和 `SkillCommand`。扩展参数集合
    属于协调 IDL/schema 升级，不能借用 `context_json` 或 consumer 私有 JSON。

### 11.6 Hermes Agent 自然语言计划接口

Hermes 是本地交互 Agent，不是新的运动执行器。自然语言入口必须先进入 `embodied_agent` 的受控计划边界，
不能让 Hermes 生成 Primitive、ROS endpoint、nonce、任意 Workflow JSON 或一串彼此独立的 root `execute`
命令。目标链路固定为：

```text
用户自然语言
  -> local Hermes Agent
  -> ibrobot-control Agent Skill
  -> robot-skill run-workflow（内部完成 status / catalog / plan / validate）
  -> 展示并 flush exact plan + fresh task ID -> 立即 confirm-plan（内部技术绑定）
  -> execute-plan；root cancel 单独使用 cancel-plan
  -> embodied_agent Agent plan store / task executor
  -> direct SkillCommand，或 Begin -> ordered child SkillCommand -> Finalize
  -> Capability Gateway -> safety_guard -> Primitive/delegated executor
```

`embodied_agent` 新增一个仅保存短期 immutable planned instance 的 Agent plan store。它不是 catalog、Workflow
template registry 或执行授权来源：catalog entry 和 aliases 仍来自 exact `SkillSnapshot`，Workflow 顺序和 typed
参数仍由 `embodied_agent` 拥有，真正运动授权仍来自 operator policy、Gateway admission、root lease 和 safety。

```text
# AgentPlan.msg
uint32 schema_version
uint8 SINGLE_SKILL=1
uint8 WORKFLOW=2

string plan_id
string plan_token
uint8 plan_kind
string raw_command
WorkflowStep[] workflow_steps
string plan_digest
string registry_epoch
uint64 registry_generation
string registry_digest
builtin_interfaces/Time expires_at
string execution_mode
```

`plan_digest` 使用 6.4 节 canonical JSON 规则，对
`(schema_version, raw_command, ordered workflow_steps, registry_epoch, registry_generation, registry_digest,
execution_mode)` 计算
SHA-256。它不是最终 `workflow_digest`：后者还绑定执行时才确定的 root task ID 和 `TaskBudget`。v1 单个 plan
最多包含 16 个 typed steps；超过上限必须在 plan 阶段返回 `SKILL_SCHEMA_INVALID`。`plan_token` 是至少 128 bit
随机的 opaque lookup token，不进入 digest、不属于 motion authorization，也不能由 Hermes 构造。

```text
# PlanAgentCommand.srv
uint32 schema_version
string request_id
string raw_command
WorkflowStep[] workflow_steps
string execution_mode
---
bool success
AgentPlan plan
string error_code
string message
SkillDiagnostic[] diagnostics
```

`PlanAgentCommand` 必须捕获 exact current snapshot，只能选择 `enabled && planner_visible` 的 Atomic Operator/Skill，
并输出一到多个 typed `WorkflowStep`。它不得输出 Primitive、嵌套 Workflow 或 `context_json` 执行字段。规则
resolver 只读取 snapshot 中的 alias/description；VLM resolver 使用同一 capability view 和结构化 response schema，
其输出仍由 deterministic validator 完整校验。未知、歧义、部分可解析、参数缺失或额外字段全部 fail closed，
不能执行已识别的前半段。

相同 `(request_id, raw_command, typed steps, execution_mode, exact identity)` 的重试返回同一 plan/token；同一 request ID 的不同 payload 返回
`SKILL_REQUEST_ID_CONFLICT`。Plan store 默认 TTL 为 300 秒、最多 1024 条；过期返回
`SKILL_AGENT_PLAN_EXPIRED`。Store 只受 `embodied_agent` 自身 state lock 保护，不获取 coordinator lock，也不持有
bundle lease。TTL 使用 process monotonic clock 判定，`expires_at` 只作为 ROS-facing 时间展示；clock jump 不能
延长 plan。已确认且已 accepted 的 plan 必须保留到 direct root/Workflow terminal record 写入后再回收，不能因
1024 条 store eviction 在活动执行中丢失；reload 后尚未执行的 plan 返回 `SKILL_REGISTRY_VERSION_MISMATCH`，要求
用户查看新计划并重新确认，不得自动升级 token。

```text
# ValidateAgentPlan.srv
uint32 schema_version
string plan_token
---
bool allowed
string plan_id
string plan_digest
string error_code
string message
SkillDiagnostic[] diagnostics
```

`ValidateAgentPlan` 对 plan 中每个 step 按顺序执行 exact-snapshot read-only preflight，返回完整 diagnostics，
但不创建 root lease、不预留 step、不刷新 token TTL。任一步失败则整体 `allowed=false`。它只支持给用户展示确认前
风险，不保证稍后的 action admission 成功。

```text
# ConfirmAgentPlan.srv
uint32 schema_version
string plan_token
string plan_digest
string task_id
string registry_epoch
uint64 registry_generation
string registry_digest
float32 task_budget_sec
string execution_mode
---
bool confirmed
string confirmation_token
float32 confirmed_task_budget_sec
builtin_interfaces/Time task_budget_started_at
builtin_interfaces/Time task_budget_deadline
string error_code
string message
SkillDiagnostic[] diagnostics
```

`ConfirmAgentPlan` 是展示后的受信 Agent 内部调用，不是用户二次确认门，也不是把口头确认变成
`authorize_motion`。Agent 必须先生成 fresh `task_id`，向用户展示包含 plan kind、ordered steps、typed
parameters、exact snapshot identity 和该 task ID 的计划并 flush 输出；随后 CLI 立即调用此 service。
Coordinator/plan store 必须原子校验
`(plan_token, plan_digest, task_id, registry_epoch, registry_generation, registry_digest, task_budget_sec, execution_mode)`，将 plan
从 `VALIDATED` 转为 `CONFIRMED`，并返回绑定同一 tuple 的单次 `confirmation_token`。`task_budget_sec` 必须为有限
正数且不超过 Gateway task budget，并以 float32 规范化冻结，同时冻结绝对 `started_at/deadline`；
`ExecuteAgentPlan.timeout_sec` 必须精确复用该值，执行时必须传播已冻结的绝对 deadline，不能重新计时。
不匹配、未完成 validate、重复确认、过期 plan 或
reload 后 identity 变化全部 fail closed。

`ConfirmAgentPlan` 和 `ExecuteAgentPlan` 只能被部署 policy 允许的 Agent/CLI caller 调用；ROS transport policy
是 caller boundary，不能把 Agent Skill 的文字规则当作安全边界。`confirmation_token` 不开启 motion authorization，
不持有 root lease，也不进入 digest；它只证明该 exact plan/task 已完成一次确认。

```text
# ExecuteAgentPlan.action goal
uint32 schema_version
string plan_token
string confirmation_token
string task_id
float32 timeout_sec
---
# result
bool success
string plan_id
string plan_digest
string workflow_digest
uint32 completed_step_count
string error_code
string message
string actual_registry_epoch
uint64 actual_registry_generation
string actual_registry_digest
---
# feedback
string state
string current_skill
uint32 workflow_step_index
string detail
```

`ExecuteAgentPlan` 是 `embodied_agent` 的公开高层 action。它按 token 读取 immutable plan，重新校验 TTL、exact
identity、task ID 和 timeout，并在 action admission 时重新执行必要的 Gateway/safety 校验：

v1 action 名称固定为 `/embodied/execute_agent_plan`。`robot-skill cancel-plan` 使用 task ID 和展示过的
plan/registry/step-count tuple，通过该 action 的标准 `CancelGoal` 取消 root goal，root executor 再向当前 direct
Skill 或 Workflow child 传播取消并轮询唯一 terminal result；现有 `robot-skill cancel --task-id ID` 仍只针对
`/embodied/execute_skill` 的 direct Skill goal。
Admission race、尚未绑定 child、active child cleanup、root Finalize 和 unknown stop state 都必须由 root ledger
记录并返回稳定结果，不能用 direct Skill 的 goal UUID 冒充 Agent plan root UUID。

1. 单个 step 使用 direct root `SkillCommand`，`workflow_digest` 为空。
2. 两个及以上 step 创建 typed planned `TaskCommand`，由 `task_executor_node` 计算最终 `workflow_digest` 并执行
   Begin/ordered child/Finalize；Hermes 和 `robot-skill` 不接触 root lease nonce。
3. Token 第一次 accepted execution 时必须携带尚未消费的 confirmation token 并绑定 task ID；相同 token/task ID
   的重试可按既有 active/terminal record 幂等返回，不同 task ID 返回 `SKILL_REQUEST_ID_CONFLICT`。确认 token
   只允许一次新的 admission，不能用它提交另一个 task。
4. Action cancel 必须向当前 direct Skill 或 Workflow child 传播，并等待 cleanup 和 terminal result；仅收到 cancel
   accepted 不能报告机器人已停止。
5. 任一步失败、取消、timeout 或停止状态未知时，不执行后续 step，不自动创建新 token/task ID 重试。

Hermes-facing CLI 新增以下稳定命令，且继续输出现有 JSON/JSONL envelope：

```text
robot-skill --config-name NAME plan-workflow --request-id ID --text TEXT
robot-skill --config-name NAME validate-plan --plan-token TOKEN
robot-skill --config-name NAME confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID
robot-skill --config-name NAME execute-plan --plan-token TOKEN --task-id ID --confirmation-token TOKEN --plan-id PLAN_ID --plan-digest DIGEST --registry-epoch EPOCH --registry-generation GENERATION --registry-digest REGISTRY_DIGEST --expected-step-count COUNT
robot-skill --config-name NAME cancel-plan --task-id ID --plan-id PLAN_ID --plan-digest DIGEST --registry-epoch EPOCH --registry-generation GENERATION --registry-digest REGISTRY_DIGEST --expected-step-count COUNT
```

目标命令的 CLI public projection 固定为：service response 的 `success/allowed/confirmed` 决定 envelope `ok`，
其余 public response 字段进入 `data`；`confirm-plan` 额外回显已规范化的 plan digest/task ID；action feedback/result
逐字段进入 JSONL `data`。Service 业务失败使用 `ok=false` 和 `error.code`，action 失败仍输出唯一 terminal
`result` event。`plan_token` 和 `confirmation_token` 只能出现在显式 CLI data 或受控进程内存中，不得进入
Hermes prompt、普通日志、snapshot 或 error message。以下是 v1 完整 success/canceled projection，不能省略
typed step 或 action result 的必填 public 字段：

```jsonl
{"command":"plan-workflow","data":{"diagnostics":[],"message":"","plan":{"expires_at":{"nanosec":0,"sec":1234567890},"plan_digest":"DIGEST","plan_id":"PLAN_ID","plan_kind":1,"plan_token":"PLAN_TOKEN","raw_command":"wave","registry_digest":"REGISTRY_DIGEST","registry_epoch":"EPOCH","registry_generation":1,"schema_version":1,"workflow_steps":[{"motion_direction":"","motion_distance":0.0,"place_name":"","schema_version":1,"skill_name":"wave_hello","target_name":"","timeout_sec":0.0}]}},"error":null,"ok":true,"schema_version":1}
{"command":"validate-plan","data":{"allowed":true,"diagnostics":[],"error_code":"","message":"","plan_digest":"DIGEST","plan_id":"PLAN_ID"},"error":null,"ok":true,"schema_version":1}
{"command":"confirm-plan","data":{"confirmation_token":"CONFIRMATION_TOKEN","confirmed":true,"diagnostics":[],"error_code":"","message":"","plan_digest":"DIGEST","task_id":"TASK_ID"},"error":null,"ok":true,"schema_version":1}
{"event":"feedback","data":{"current_skill":"wave_hello","detail":"step 1 of 1","state":"executing","workflow_step_index":0},"payload_hash":"PAYLOAD_HASH","schema_version":1,"task_id":"TASK_ID"}
{"event":"result","data":{"actual_registry_digest":"REGISTRY_DIGEST","actual_registry_epoch":"EPOCH","actual_registry_generation":1,"completed_step_count":1,"error_code":"","message":"plan completed","plan_digest":"DIGEST","plan_id":"PLAN_ID","success":true,"workflow_digest":""},"payload_hash":"PAYLOAD_HASH","schema_version":1,"task_id":"TASK_ID"}
{"command":"cancel-plan","data":{"accepted":true,"goal_status":5,"result":{"actual_registry_digest":"REGISTRY_DIGEST","actual_registry_epoch":"EPOCH","actual_registry_generation":1,"completed_step_count":0,"error_code":"SKILL_CANCELLED","message":"plan canceled","plan_digest":"DIGEST","plan_id":"PLAN_ID","success":false,"workflow_digest":""},"terminal":true},"error":null,"ok":true,"schema_version":1}
```

上述五个 Hermes-facing 命令和 `hermes-robot` 已在当前工作区实现；只有完成整套 package build、Gateway/Safety
snapshot 同步且 launcher prerequisite check 通过后，才能作为已部署能力使用。`cancel` 只取消 direct Skill
action，`cancel-plan` 取消 Agent plan root action，两者不能互换。

正常自然语言请求的 Agent workflow 固定为
`run-workflow -> 内部 status/catalog/plan/validate -> 生成并展示 fresh task ID 和 exact plan ->
confirm-plan -> execute-plan`。显式单 Skill 请求可继续使用现有 `validate/execute` 路径。展示必须发生在 plan 和
validation 之后，完整包含 plan kind、顺序、参数、snapshot identity 和 task ID，并在内部 `confirm-plan` 前
同步 flush；随后立即绑定并执行，不等待用户二次确认。`confirm-plan` 只是 exact tuple 的技术绑定，不修改
`authorize_motion`。Agent 不能启动/重启 pipeline、设置 `authorize_motion`、修改 ROS 参数或调用 raw ROS motion。

## 12. Consumer 行为

### 12.1 Safety Guard

safety_guard 必须维护本地快照；当请求携带的版本与其本地快照不匹配时，必须
拒绝校验。

为了支持跨 reload 的活动 goal，safety_guard 应维护按 `(epoch, generation)` 索引的只读快照缓存，而不是
只保存单一 current snapshot：

1. 启动时拉取 current snapshot。
2. Event 到达后拉取新的精确 snapshot 并设置为 current。
3. 原 current snapshot 按 status 的 `retained_generations` 全部保留，并额外保留最近两个 generation。
4. `ValidateSkill` callback 只能读取本地缓存，禁止在 callback 内同步调用 executor snapshot service。
5. 收到本地不存在的 generation 时立即 fail closed；后台 catch-up worker 可以在 callback 外拉取，但不能让
   当前 validation 等待形成 executor -> safety -> executor 调用环。
6. 快照不存在或 digest 不匹配时 fail closed。
7. 本地 `primitive_contract_digest` 与 snapshot 不一致时 fail closed，并报告软件 contract mismatch；不得通过
   重新拉取同一 snapshot 或复制 executor 的白名单绕过。

Safety 的周期性 status catch-up 必须先获取 `retained_generations`，再在后台补齐缺失快照，最后才清理已不在
该集合且超出最近两个 generation 的本地 view。

历史 safety view 只能在 executor 不再保留对应 bundle、且没有活动校验引用后清理。

### 12.2 Planner 与 CLI

Planner 和 CLI 应执行以下流程：

1. 读取当前 gateway status。
2. 拉取 snapshot payload，本地重算并校验 registry/capability/provenance digest。
3. Planner 将版本标识和 canonical `workflow_digest` 附加到最终 planned Task；direct CLI 将版本标识附加到
   单个 root `SkillCommand`，其 Workflow 字段保持为空。
4. 如果 preflight 和 admission 之间版本发生变化，拒绝执行。

Runtime `plan-workflow`、Agent `validate/confirm/execute` 和现有 runtime `validate/execute` 的 public capability view
必须以 Gateway exact snapshot 为权威；本地 install/source YAML 只允许用于 catalog-only 启动前发现，不能在 runtime
请求中作为 Gateway snapshot 的隐式 fallback。若 local catalog、profile、robot context 或 locally recomputed
digest 与 Gateway identity 不同，CLI/Agent 必须返回 `SKILL_REGISTRY_VERSION_MISMATCH` 或
`SKILL_SNAPSHOT_DIGEST_MISMATCH`，不得继续使用本地结果发送 action goal。Stage 3 的 catalog-only cache 也必须
标记为 non-authoritative，直到 snapshot identity 校验成功。

所有 planner 路径都必须根据快照中的 enabled set 和 planner-visible set 过滤 typed Workflow 输出。这不仅
包括动态加载的 alias，还包括规则解析器中硬编码的观察、恢复、夹爪和相对运动分支。Planner 必须把每一步
参数写入 `WorkflowStep`，不得再把不同步骤压缩到 `TaskCommand` 的一组全局参数。

Agent plan 入口每次切换快照时必须同时更新 planner-visible catalog entry 集合、
`semantic_level` 和 parameter schemas，不能只更新 aliases。

### 12.3 活动任务

活动直接 Skill 保留 admission 时捕获的 bundle，并使用同一个 bundle 和 root lease 完成 finalize。活动
Workflow 从 Begin 到 Finalize 保留同一个 bundle、root budget、`workflow_digest` 和 root lease；每个 child
只借用该 execution scope，不得独立释放或重新获取 root lease。

运行时可以在 reload 期间拒绝新 goal，但不得在活动 goal 执行过程中将其切换到
新的 bundle。

Reload 后 Workflow 的后续步骤继续使用 Begin 时捕获的旧 snapshot，不允许逐步自动升级。若 planner 必须基于
新 catalog 重新规划，应先取消活动 child、Finalize 旧 Workflow，再使用新的 root task ID、exact identity 和
`workflow_digest` 提交新 Task；不得复用旧 root ID 静默替换计划。

### 12.4 Consumer 同步状态机

所有 consumer 使用同一同步算法：

1. 先以规定 QoS 创建 event subscription，避免 status 查询与订阅之间丢失更新。
2. 查询 gateway status，读取 current epoch/generation/digest。
3. 请求该精确 snapshot。
4. 从 snapshot payload 中重算并校验 registry/capability/provenance digest，再校验 response identity。
5. 执行型 consumer 重算本地 `primitive_contract_digest` 并与 registry preimage 比较。
6. 在当前引用外构建全部派生状态。
7. 在单个本地 state lock 内原子替换 snapshot 和所有派生状态。
8. 处理同步期间缓存的 event；如果 event 指向更新版本，则继续追赶。

事件处理规则：

- 相同 epoch、相同 generation：幂等忽略。
- 相同 epoch、更低 generation：作为乱序事件忽略。
- 相同 epoch、更高 generation：请求事件指定的精确 snapshot。
- 新 epoch：丢弃 current 标记并重新执行完整同步。
- Event 丢失：通过周期性 status 检查或请求失败后的 status 重查恢复。
- Exact snapshot 未保留：重新查询 current status；planner 进入 not-ready，safety 对不匹配请求 fail closed。

同步失败时可以保留旧快照供已经绑定该版本的活动流程使用，但不得把旧快照报告为 current ready。
Retry 必须使用有界 exponential backoff，并发布可诊断的 last error code。

### 12.5 启动流程

```text
embodied_bringup
  -> load robot_config
  -> resolve skill_profile and SkillRobotContext
  -> start skill_executor with profile/context/source parameters
  -> skill_executor compiles epoch E, generation 1
  -> expose snapshot/reload/status interfaces
  -> publish transient-local event(E, 1)
  -> safety/planners fetch exact snapshot(E, 1)
```

启动规则：

1. Generation 1 编译失败时 executor 不进入 ready。
2. Executor 未 ready 时 Gateway 不接受 catalog entry goal。
3. safety_guard 未同步时 ValidateSkill fail closed。
4. planner 未同步时报告 not-ready，不生成过期计划。
5. Profile 模式下 launch 不注入大段 templates/aliases JSON 作为长期 SSOT。

## 13. 打包与安装目录

生产环境加载不得依赖可变的源码目录路径。

推荐目录结构：

```text
<install>/share/skill_catalog/releases/<source_release_digest>/...
<install>/share/skill_catalog/current -> releases/<source_release_digest>
```

`source_release_digest` 与 `registry_digest` 不同：

- `source_release_digest` 标识一套不可变的 catalog 源文件，可被多个 profile 和 robot context 编译。
- `registry_digest` 标识某个 profile 与 robot context 编译后的执行快照。

`source_release_digest` 使用 SHA-256，preimage 是按相对 POSIX path 排序的文件清单：

```json
{
  "schema_version": 1,
  "files": [
    {"path": "config/profiles/robot.yaml", "size": 123, "sha256": "..."}
  ]
}
```

规则：

- 纳入 release 下所有普通文件，包括 schema、profile、manifest、implementation 和 `SKILL.md`。
- 不纳入目录、mtime、permission、owner 和外部 `current` symlink。
- Release 内禁止 symlink、隐藏文件、编辑器临时文件和设备文件。
- `SkillSource.compute_release_digest()` 返回 `source_release_digest`。
- Release 目录名必须与重新计算的 digest 一致，否则返回 `SKILL_RELEASE_NOT_IMMUTABLE`。
- Compiler 使用同一文件清单算法对每个 `config/skills/<name>/` 子树计算 `skill_package_digest`。
- `skill_package_digests` mapping 按名称包含 release 中所有发现的 catalog package，包括 Atomic Operator 和
  Skill，而不只包含当前 profile
  enabled 集合；key 按字典序 canonicalize。
- 已发布的 `(skill name, SemVer)` 不得在后续 release 中对应不同 `skill_package_digest`；CI 与 release index
  检测到内容变化但 SemVer 未提升时必须失败。
- Source 和 release 目录激活后必须保持不可变，活动指针必须原子切换。
- `setup.py` 必须递归安装 YAML、JSON schema 和 `SKILL.md`；非源码 install space 必须能够加载相同内容。

在源码目录中新增或删除文件，并不等于完成生产环境热加载。只有在完整的不可变
release 部署完成、且活动指针完成原子切换后，生产环境才能观察到变化。

安装验证覆盖以下两种明确分离的模式：

- Production `DirectoryReleaseSkillSource` 允许 release 根目录外唯一的原子 `current` symlink，但其目标
  `releases/<digest>/` 子树必须全部是 materialize 后的普通文件；release 内部任何 symlink 都必须拒绝。
- `--symlink-install` 只用于 `DevelopmentStagingSkillSource`，不得冒充 production immutable release。
- Source workspace 的三份迁移 profile 使用 `skill_catalog_source_mode=development`，由 launch builder 根据
  robot config 的绝对路径解析 staging root；非 symlink release build 可切换到 ament `installed` source。
- 两种模式的 package share 都必须可发现全部 manifest、implementation、profile、schema 和 `SKILL.md`。
- Production compiler 不访问源码绝对路径；`current` 只能指向完整且 release digest 校验通过的目录。
- Release 激活后发现文件内容变化时，runtime 必须拒绝并报告 immutable violation。

## 14. 实现影响范围

| 区域 | 必需修改 |
|---|---|
| `src/skill_catalog/**` | 新增 compiler、schema、source、snapshot model 和测试。 |
| `src/embodied_common/embodied_common/primitive_contracts.py` | 新增唯一静态 Primitive descriptor registry 和 canonical digest helper；从 `skill_templates.py` 移出白名单、参数表和 runtime capability mapping，供 catalog、executor 和 safety 只读复用。 |
| `src/embodied_common/embodied_common/workflow_contracts.py` | 新增 typed plan 规范化和 canonical `workflow_digest` helper，供 planner、task executor 和 coordinator 复用；不持久化 Workflow。 |
| `src/robot_config/robot_config/loader.py` | 构建完整机器人执行上下文和 `robot_config_digest`，并校验 legacy/profile 互斥；不管理运行时 generation。 |
| `src/robot_config/robot_config/config.py` | 新增强类型 `skill_profile`；implementation 和 source-mode 不属于 robot_config。 |
| `so101_single_arm.yaml` | 迁移完成后删除内联 template，并选择 profile。 |
| `so101_handeye_realsense_grasp.yaml` | 迁移其委托抓取 Skill profile。 |
| `so101_rtp_distributed.yaml` | 选择共享 SO101 运动学 implementation，不再复制 template。 |
| `embodied_bringup` | 不再将启动期 template JSON 作为 profile 模式的权威来源。 |
| `embodied_bringup/launch/embodied_pipeline.launch.py` | 新增 source-mode、source、retention 和 watcher 运行参数。 |
| `skill_library` | 新增 bundle 切换、delegated executor descriptor registry、共享 `ExecutionCoordinator`、Workflow execution scope、root task budget ledger、root lease/dispatch nonce、精确版本准入和重启处理。 |
| `safety_guard` | 新增精确快照同步和 epoch/generation 校验。 |
| `safety_guard/rules.py` | 显式接收按版本索引的本地 validation view。 |
| `embodied_common.workflow_contracts` | 根据快照 enabled set 限制硬编码和 alias 输出。 |
| `embodied_agent` | 消费动态 enabled/planner-visible catalog entry，生成 typed `WorkflowStep[]` 和 `workflow_digest`；Agent plan executor 负责 Begin/child/Finalize 生命周期并传递精确版本。 |
| `embodied_agent` Agent plan store | 接收 Hermes 的自然语言 plan request，保存短期 immutable `AgentPlan`，提供 plan/validate/confirm/execute 高层接口；不拥有 catalog、Primitive、motion authorization 或 root lease。 |
| `task_dispatch` | 保持 MoveIt/GRIPPER/WAIT 的底层执行器职责，不接收或持久化用户级 Workflow。 |
| `robot_skill_cli` | 将 preflight、validation、confirmation 和 execution 绑定到同一个快照标识，并显示 entry 的 `semantic_level`；新增 `plan-workflow`、`validate-plan`、`confirm-plan`、`execute-plan`、`cancel-plan` 和 `hermes-robot` 启动器。 |
| `ibrobot_msgs` | 新增 `SkillDiagnostic`、`TaskBudget`、`DispatchBinding`、`WorkflowStep`、`DelegatedExecutorIdentity`、`AgentPlan`、reload/snapshot/event、Agent plan 和 Begin/Finalize Workflow 接口，并修改 task、capability、validation、status、`SkillCommand`、`PrimitiveCommand` 和 `PickObject` 契约。 |
| `ibrobot_msgs/CMakeLists.txt` | 注册新增和修改后的 msg/srv/action。 |
| `.agents/skills/ibrobot-control` | 明确自然语言 plan workflow、单 Skill/多步骤分流、确认、停止未知和禁止绕过规则；不维护机器人 alias 或 Primitive 白名单。 |
| `scripts/check_agent_skill.py` 与 launcher/CLI 测试 | 校验 Agent Skill 的自然语言触发、plan/validate/confirm/execute 顺序、Hermes 版本兼容和受控命令边界。 |
| 各相关 `package.xml` | 增加 `skill_catalog`、`builtin_interfaces`、消息和可选 watcher 依赖。 |
| 包 README 和测试 | 更新 SSOT、接口、安装目录和并发预期。 |

只有在目标基线中存在架构治理文件时，才更新对应治理规则。当前
`IB_Robot_0803` 工作区不包含文档中引用的 `.agents/architecture/` 规则目录，
因此迁移流程不得假设该 gate 已经可用。

## 15. 迁移计划

### Stage 0：治理与契约

工作：

- 更新 ADR；目标基线存在 architecture rules 时同步更新规则和 hash。
- 冻结带 `semantic_level` 的三类文件 schema v1、含 `primitive_contract_digest` 的 registry preimage v1、其余
  两类 digest preimage v1、typed `WorkflowStep`、`workflow_digest`、Begin/Finalize Workflow ROS IDL、Hermes
  `AgentPlan`/plan-validation/confirm-plan/execute-plan/cancel-plan IDL 和稳定错误码。
- 明确 `skill_catalog` 是 Atomic Operator/Skill SSOT，`embodied_agent` 是 Workflow owner，robot_config 是
  机器人硬件事实 SSOT。
- 冻结 `semantic_level` 不是授权边界、Atomic Operator 单 Primitive 只是 schema v1 implementation gate。
- 建立所有被修改 IDL 的 producer/consumer 清单和原子部署 manifest；Stage 0 只提交文档、schema、golden vector
  和未接入的接口草案，不安装新 type，不混跑旧二进制。

门禁：

- 设计评审通过。
- Schema 正反例和 golden digest vector 已纳入测试资源。
- 当前基线没有 architecture watch 时，使用人工架构检查记录替代，不伪造自动 gate。

回滚：仅文档和未消费 schema 变更，可直接删除，不改变运行行为。

### Stage 1：纯 Compiler

工作：

- 新增 `skill_catalog` 包、models、source、validator、compiler 和 digest。
- 将当前 loader 的纯 Skill 校验迁入 validator。
- 在 `embodied_common.primitive_contracts` 建立唯一 Primitive descriptor registry 和 digest helper，删除其他包的
  平行白名单、参数字段表和 runtime capability mapping。
- Compiler 临时接收 legacy inline templates，运行参数保持不变。

门禁：

- 现有 robot_config Skill 测试全部通过。
- 关键错误码和完整错误集合保持等价。
- 三个当前 Skill 配置的 normalized 输出不变。

回滚：删除尚未进入启动链路的 compiler 包，legacy 路径继续工作。

### Stage 2：Catalog 文件迁移

- 建立 manifest/implementation/profile 文件和 schema。
- 迁移 `so101_single_arm`、`so101_handeye_realsense_grasp` 和 `so101_rtp_distributed`。
- 按 5.1 节分类现有 entry，显式填写 `semantic_level` 和 `planner_visible`；校准 `inspect_scene` 的公开后置条件。
- 使用稳定 implementation variant 复用相同运动学实现。
- Legacy 与 profile 路径同时存在，但严格互斥。

门禁：

- Legacy 与 profile 编译后的执行 template canonical JSON 完全一致；新增 semantic metadata 单独比较。
- 除 `inspect_scene` 明确批准的描述校准外，Capability 行为和 primitive expansion 完全一致。
- Planner-visible set 与迁移前 allowed skills 等价。
- 所有文件可从普通 install share 加载，不依赖 source tree。

回滚：继续选择 legacy 模式；不得自动 merge 两套数据。

### Stage 3：启动路径和只读同步切换到 Registry

- Executor 启动时创建 epoch、Registry 和 generation 1 bundle。
- Profile 模式 launch 不再注入 templates/aliases JSON 作为权威输入。
- Gateway status 暴露 epoch/generation/digest，并提供只读 exact snapshot service。
- 发布 generation 1 的 transient-local event，consumer 使用统一状态机完成初始同步。
- Planner 和 task executor 切换到 `WorkflowStep[]`；planned Task 不再从 `context_json.skill_sequence` 回退。
- Planner 生成 canonical `workflow_digest`；task executor 通过 Begin/Finalize service 在所有 child 之间持有同一
  root lease、budget 和 generation 1 bundle。
- 在同一次协调部署中切换全部 `ibrobot_msgs` producer/consumer、planner、task executor、Gateway、safety、CLI
  和 installed Agent Skill；禁止只部署新的 IDL 或只切换其中一个 consumer。
- `embodied_agent` 开放 `PlanAgentCommand`、`ValidateAgentPlan`、`ConfirmAgentPlan` 和 `ExecuteAgentPlan`；Hermes 通过
  `robot-skill plan-workflow/validate-plan/confirm-plan/execute-plan` 使用这些高层接口，不直接发布 `/voice_command`
  或调用裸 ROS。
- `robot_skill_cli` 安装 `hermes-robot` 启动器：验证外部 Hermes 与 Agent Skill 可发现后预加载 `ibrobot-control`；
  启动器不得启动机器人 pipeline 或写入 `authorize_motion`。
- 本阶段不开放 reload service，catalog 在进程生命周期内保持 generation 1。

门禁：

- Generation 1 编译失败不产生部分可用 Gateway。
- Profile 模式完成 simulation smoke test。
- CLI list/describe/validate/execute 行为不回退；新增 plan/validate/confirm/execute/cancel-plan 命令只在整套
  新 IDL 和 producer/consumer 原子部署后开放。
- Executor、safety 和 planner 初始 digest 一致。
- Workflow 步骤之间无法插入 direct root Skill；错误 root nonce、step index 或 payload 均 fail closed。
- 本地 Hermes 进程能够从普通 install space 发现 `ibrobot-control`，并完成单 Skill 与两步 Workflow 的
  plan -> validate -> 展示 task ID/exact plan -> confirm-plan -> execute transcript；未确认或未确认 exact tuple
  时没有任何 action goal。
- Hermes 不能通过任何 prompt 启动/重启 pipeline、设置 `authorize_motion`、调用 Primitive/MoveIt/controller
  或在首步失败后自动执行下一步。

回滚：Launch 显式切回 legacy 模式，不保留双 owner。

### Stage 4：手动安全 Reload

- 开放 reload service，并在成功提交后通过既有 transient-local event 发布版本变化。
- Safety、planner、VLM 和 CLI 扩展状态机以处理 reload、新 epoch、乱序和 event loss。
- Task、validation 和 action 绑定预期版本。
- Active direct Skill 和完整 Workflow execution scope 捕获旧 bundle 和 coordinator token；reload 后 Workflow
  后续步骤继续使用 Begin 时的 exact snapshot。
- Primitive/delegated dispatch 使用 coordinator 签发的 active nonce，外部 Primitive goal fail closed。

门禁：

- 修改已有 Skill 后，新 goal 使用新版本。
- 新增/删除 Skill 后所有 consumer 最终一致。
- 非法候选 reload 失败，epoch/generation/digest/current bundle 不变。
- 活动 goal 使用旧 policy view 正常 finalize。
- 版本不匹配时不得下发机器人动作。
- Late consumer 和 event-loss 场景可恢复到 current snapshot。

回滚：禁用 reload service，保留最近一次有效 bundle；不得恢复到 live-tree 非原子读取。

### Stage 5：开发 Watcher

- 增加可选 watcher 依赖。
- Watcher 只观察 development staging fingerprint 或 production `current` 指针。
- 所有事件复用 Stage 4 的同一 reload coordinator。

门禁：

- Watcher 默认关闭。
- 单次逻辑编辑最多提交一个新 generation。
- 半写文件、临时文件和多文件中间状态不改变 current bundle。

回滚：关闭 watcher 参数，手动 reload 继续可用。

### Stage 6：移除 Legacy 内联 Skill SSOT

- 删除机器人 YAML 中的 `embodied.skill_templates`。
- 删除 unversioned authority、legacy fallback、`context_json.skill_sequence` 和固定 JSON consumer 路径。
- 更新 README、CMake、package dependencies 和测试。

门禁：

- 仓库扫描不存在生产内联 templates 和默认 fallback。
- Build、Ruff、单元、跨进程、安装和仿真测试通过。
- 所有公开命令只使用稳定接口。

回滚：回滚完整 release artifact，不在同一版本内重新启用双 SSOT。

## 16. 测试计划

### Compiler 测试

- Manifest name 与目录不一致、semver 非法和未知字段。
- Profile 引用不存在的 Skill 或 implementation。
- 重复 Skill 名称、alias 冲突、`instead_use` 悬空或自引用。
- Primitive 和 delegated executor 两种实现。
- Primitive descriptor name/schema/参数/runtime capability/dispatch kind 的正反例，以及 descriptor 中出现
  callable、未知字段或运行时对象时拒绝。
- Primitive descriptor 受控 JSON Schema keyword、runtime capability enum、dispatch kind enum、endpoint role
  映射及每个 v1 Primitive 唯一归属的 golden 正反例。
- `atomic_operator` 的 delegated、零/多 Primitive、`initial_gripper_state` 和
  `move_through_joint_positions` 反例全部拒绝；单 Primitive `skill` 正例允许。
- `semantic_level` 缺失/未知、`planner_visible` 缺失和 manifest/profile visibility 冲突。
- 不支持的 primitive、executor 或 required args。
- 非法关节、位姿、控制模式和局部 limit 越界。
- Trajectory template 展开和绝对轨迹连续性。
- Schema 示例与 validator 类型一致。
- Profile enabled 和 planner-visible 集合派生。
- 不同部署配置名称之间复用稳定 implementation。
- Delegated executor descriptor 或 model bundle identity 变化会改变 registry digest。
- Skill 内容变化但 SemVer 未提升时 release gate 失败。
- 完整诊断集合和确定性排序。

### Registry 与 Digest 测试

- Golden canonical JSON、source release digest、registry/capability/provenance digest vector。
- Dict 顺序、set 顺序和 source/install 路径不影响 digest。
- 执行内容变化必须改变 registry digest。
- Primitive descriptor 任一 canonical 字段变化必须改变 `primitive_contract_digest` 和 registry digest；仅内部
  contract 变化可以保持 capability digest 不变。
- `semantic_level` 变化必须同时改变 registry/capability digest；provenance preimage 仍保持原有精确顶层
  schema，只通过 source/package digest 反映文件内容变化。
- Named pose/target 值、方向映射、gripper 值或 executor descriptor 变化必须改变 registry digest。
- 仅内部轨迹变化不得改变 capability digest。
- Generation 1 初始化、有变化递增、无变化 no-op、失败不递增。
- 仅文档/provenance 变化时 registry/capability digest 可保持不变，但 provenance/source release digest 和
  generation 递增。
- 重启创建新 epoch，旧 epoch 被拒绝。
- 无效 candidate 保留旧 registry record 和 bundle。
- Consumer 无法修改当前 snapshot。
- `changed_skills` 计算正确。
- Executor 或 safety 本地 Primitive contract digest 与 snapshot 不一致时保持 not-ready/fail-closed。

### Executor 并发测试

- 并发 reload 调用必须串行化。
- reload 与活动 goal 不得发生死锁。
- coordinator lease 必须阻止两个根执行并发进入。
- Workflow Begin 对相同 root/digest/version/budget 幂等返回同一 active nonce，任一字段冲突时拒绝。
- Root-scope planned Task、Begin、Finalize 和 direct Skill 的 `workflow_step_index` 非零时拒绝。
- Begin 到 Finalize 期间 root lease 不在 child 边界释放，direct root Skill 不能插入两个 Workflow step 之间。
- Workflow child 必须按零基 index 严格顺序提交；重放、跳步、越界、参数篡改、错误 digest 或错误 root nonce
  全部 fail closed。
- 相同 child task ID 与相同 canonical payload 的重试必须复用既有 active/terminal record；相同 child task ID
  携带不同 payload 必须返回 `SKILL_REQUEST_ID_CONFLICT`，且不得再次 dispatch。
- Reload 后尚未执行的 Workflow child 使用 Begin 捕获的旧 bundle；不能回读 current 或自动升级。
- Workflow child finalize 只释放 borrow；正确且幂等的 root Finalize 才释放 bundle retention 和 root lease。
- Finalize 在仍有 active child、Primitive/delegated dispatch、late cleanup 或 lease borrow 时必须拒绝；
  `SUCCEEDED` 不能绕过未完成 step，partial Workflow 的 terminal count 必须 fail closed。
- 相同 root binding 的 Finalize terminal payload 重试必须幂等；不同 terminal state 或 completed count 必须返回
  `SKILL_REQUEST_ID_CONFLICT`，且不得释放 root lease。
- Finalize cleanup 后仍能从 `WorkflowTerminalRecord` 校验 canonical budget 并返回既有结果；修改 budget 的重试
  返回 `SKILL_TASK_BUDGET_MISMATCH`，同一 epoch 内复用 terminal root task ID 必须拒绝。
- 对同时包含多个 Finalize 缺陷的 golden request 按固定 first-failure 顺序返回唯一错误码，包括非空
  `dispatch_nonce`、错误 root task ID、错误 digest/nonce、active cleanup 和 terminal state/count 冲突组合。
- `ValidateSkill` 对 Workflow step 只读且不写 coordinator 状态；action admission 才执行
  `PENDING -> ACTIVE`，成功完成才推进 expected index，failed/canceled 后不能继续或换 child ID 重试。
- Task executor 崩溃后，coordinator 在 root deadline 执行受控 cancel/cleanup，nonce 失效前不提前放开 lease。
- 活动 goal 使用旧 template、robot context、executor descriptor、policy view 和 token finalize，且不回读 current 参数。
- Reload 期间新 root goal 返回受控错误。
- Reload 后新 goal 使用新 bundle。
- 新 goal 不能将旧 preflight 与新 admission 混用。
- Late cleanup 不得误用 current bundle。
- Registry current/history/idempotency 路径不引入第二把内部锁。
- 同一 root task ID 修改 task deadline 时 fail closed，Workflow 多步骤共享同一 deadline。
- Workflow child 使用稳定、互不冲突的 `task_id` 并保留同一 `root_task_id`。
- 外部 Primitive goal、空/错误/过期 dispatch nonce 和已经 finalize 的 nonce 全部 fail closed；合法 delegated
  executor 可以在 active lease 内执行多个受控 Primitive。

### 跨进程同步测试

- Consumer 启动时拉取 exact current snapshot。
- Late subscriber 收到 transient-local event。
- Event 重复、乱序或丢失后可恢复。
- 新 epoch 触发完整 resync。
- Snapshot 未保留时返回稳定错误。
- Safety mismatch fail closed，ValidateSkill callback 不发起反向 snapshot RPC。
- Safety 根据 status 的 `retained_generations` 保留跨越多个 reload 的活动旧 generation。
- Snapshot payload 可本地重算三类 digest；篡改任一 preimage 或 response digest 时拒绝。
- Validation callback 与 executor snapshot service 并发时不发生 callback 饥饿或调用环。
- VLM allowed skills、prompt 和 schema 随新增/删除更新。
- 规则解析器不能生成已从当前 profile 删除的 Skill。
- Planner 不能生成 `planner_visible=false` 的 Atomic Operator/Skill，也不能生成 Primitive 名称或嵌套 Workflow。
- Planned Task 使用 typed `WorkflowStep[]` 保留逐步参数和统一 registry binding；`context_json.skill_sequence`
  不再被 consumer 接受。
- Planner 与 coordinator 对同一 typed plan 生成相同 `workflow_digest`；篡改 step 顺序、参数、budget 或 expected
  identity 必须导致 Begin 失败。
- Begin/Finalize service 只允许受信 task executor policy；status 和日志不泄露 root/dispatch nonce。
- Consumer 同步失败时保留旧引用但不报告 current ready。

### Hermes Agent 与自然语言控制测试

这组测试必须验证实际的 Hermes 进程、仓库 Agent Skill、`robot-skill` 命令和 ROS Gateway 之间的行为契约，
不能只测试一个 mock parser。测试分为不需要模型/ROS 的 deterministic contract、使用本地假模型的 Hermes
process conformance、使用 fake Gateway 的 ROS integration 和需要仿真的 acceptance 四层；真机动作不进入 CI。

- Agent plan schema 正例：`"挥挥手"` 只产生一个 `wave_hello` step；`"先打开夹爪，然后回到安全位"` 产生按原顺序排列的 `open_gripper_skill`、`recover_safe_pose` 两个 step，每一步参数独立保存。
- CLI projection golden 测试：plan/validate/confirm/execute/cancel-plan 的 JSON/JSONL 必须包含 11.6 节定义的全部
  public 字段，typed `WorkflowStep` 不得缩减为只有 skill name，token 不得出现在 prompt/log/error。
- Agent plan schema 反例：未知 skill、disabled entry、`planner_visible=false`、Primitive 名称、嵌套 Workflow、缺少参数、未知字段、歧义 alias、NaN/Infinity 和 partial parse 全部拒绝，不执行已识别的前缀。
- Alias SSOT 测试：中文/英文触发词只从 exact snapshot 的 description/alias view 注入；删除或修改 alias 后 plan 必须随 registry digest 变化，consumer 不得保留旧 alias。
- Plan token 测试：request ID 幂等、不同 payload conflict、TTL 过期、registry reload 后 stale plan 拒绝、token 不可由 caller 自行构造、同一 token 只能绑定一个 accepted task ID。
- `ValidateAgentPlan` 测试：按 step 顺序调用只读 preflight；任何一步失败时整体拒绝；不创建 lease、不改变 step state、不刷新 TTL、不发 Primitive/action goal。
- `ConfirmAgentPlan` 测试：缺少/篡改 plan digest、task ID、registry identity、重复确认、过期 plan 和错误 caller
  全部拒绝；未生成 confirmation token 前没有 action goal，confirmation token 不能用于第二个 task。
- Direct Skill execution 测试：single plan 只产生一个 direct root `SkillCommand`，workflow 字段为空；action admission 前必须重新检查 exact identity、Gateway readiness、motion authorization 和 safety。
- Typed Workflow execution 测试：multi-step plan 必须执行 Begin、零基 child index、严格顺序和 Finalize；第一步失败/取消/timeout 后不得提交下一步，不得使用新 task ID 自动重试。
- Reload 测试：plan/validate 后 reload，execute 必须返回 `SKILL_REGISTRY_VERSION_MISMATCH` 并要求重新 plan；执行中 reload 时所有后续 child 继续使用 Begin 捕获的 bundle。
- Agent Skill 静态契约测试：`scripts/check_agent_skill.py` 必须检查自然语言触发、`status -> list/plan-workflow -> describe -> validate-plan -> 展示 task ID/exact plan -> confirm-plan -> execute-plan` 顺序、单 Skill/Workflow 分流、失败即停、`cancel-plan` 和停止未知说明，以及禁止 launch/authorize/raw ROS/Primitive/自动重试。
- Hermes 边界测试：由 `check_agent_skill.py`、launcher wrapper 测试、CLI plan lifecycle 测试和 Agent plan 节点测试共同覆盖禁止绕过、失败即停、确认绑定与单次执行；真实 Hermes provider 和真机 transcript 作为发布验收，不纳入默认 pytest。
- Root cancel 测试：携带 task ID 和完整展示 tuple 的 `cancel-plan` 只能取消
  `/embodied/execute_agent_plan` root goal，并验证 active child cleanup、Finalize、terminal polling、admission race
  和 unknown stop；现有 `cancel --task-id` 仍只覆盖 direct Skill。
- Compatibility error 测试：baseline direct CLI 保留 `MOTION_NOT_AUTHORIZED`、`CAPABILITY_NOT_READY`、`SKILL_BUSY`、
  `CONTROL_MODE_MISMATCH`、`TIMEOUT_EXCEEDS_POLICY`、`DUPLICATE_TASK_ID`、`TASK_ID_CONFLICT`、`INVALID_ARGUMENT`、
  `GATEWAY_FINALIZATION_FAILED`、`SKILL_CANCELLED`、`SKILL_CANCEL_TIMEOUT` 和 `GOAL_NOT_FOUND` 及其既有 exit
  contract；新 Agent plan code/exit mapping 只在原子 envelope/IDL migration 后启用。
- Launcher 测试：`hermes-robot --config-name NAME` 验证 Hermes binary、`robot-skill`、workspace Agent Skill 和 Gateway status，预加载 `ibrobot-control`，继承精确 `ROBOT_CONFIG`，不得启动/重启机器人或修改 `authorize_motion`。
- Launcher readiness 测试：control-plane snapshot/status ready 且 `motion_authorized=false` 时 launcher 保持运行并允许
  catalog/plan/validate；capability execution readiness 或 motion authorization 失败只能在 action admission 阶段拒绝。
- Normal install 测试：普通 install space 中启动 launcher 时仍能发现 Agent Skill、schema、catalog 和 CLI，不得依赖源码绝对路径或当前 checkout 的隐藏文件。

### 打包测试

- Package share 包含全部数据和 schema。
- Production 普通 install 能加载 materialized release，且 compiler 不访问源码绝对路径。
- `--symlink-install` 只能通过 development staging source 加载，production source 必须拒绝。
- 编译期间 source 变化时拒绝或重试。
- Release 指针切换具备原子性。
- 激活后的 release 内容变化触发 immutable violation。

### 仿真验收

1. 启动 epoch E、generation 1。
2. 手动 reload 修改后的 `wave_hello`，generation 递增。
3. 新请求使用新轨迹。
4. 非法 YAML reload 失败，当前版本不变。
5. Reload 期间活动 Skill 正常完成。
6. 删除 Skill 后 CLI、planner、safety 和 executor 均不再暴露或接受。
7. Executor 重启后新 epoch 能驱动全部 consumer 重新同步。
8. 同一 Workflow 中两个带不同参数的 entry 按各自 `WorkflowStep` 执行，不复用 Task 顶层 hint。
9. CLI/status 显示 Atomic Operator 与 Skill 的正确 `semantic_level`，planner 只看到显式可见子集。
10. Workflow 执行期间完成一次 reload，前后步骤仍使用 Begin 时的 generation，Workflow 完成后新 root 请求
    使用新 generation。
11. Workflow 中途失败或取消时先清理 active child，再幂等 Finalize root scope；后续 root 请求可以正常准入。
12. 在同一已启动的仿真 pipeline 上，执行 `hermes-robot` 后输入单 Skill 自然语言请求，完成
    `plan-workflow -> validate-plan -> 展示并 flush -> confirm-plan -> execute-plan -> terminal result`；未授权 launch 必须只返回
    Gateway 拒绝且不产生运动 goal。
13. 在同一 Agent session 中输入两步自然语言 Workflow，两个 typed steps 按原顺序执行，第二步参数不继承第一步
    的顶层 hint，第一步失败时没有第二步 action goal。
14. 操作员显式以 `authorize_motion:=true` 启动仿真并完成安全确认后，单 Skill 和两步 Workflow 均通过
    `SkillCommand`/Begin-child-Finalize 到达 fake MoveIt gateway；Hermes 仍没有任何底层 endpoint 工具。
15. plan 后 reload、Gateway 重启、cancel、timeout 和 unknown stop state 场景均按稳定错误码结束，不能自动
    replan/retry。

## 17. 错误模型与职责

建议异常层级：

```python
class SkillCatalogError(Exception):
    code: str
    source_relative_path: str | None
    field_path: str | None


class SkillSchemaError(SkillCatalogError): ...
class SkillProfileError(SkillCatalogError): ...
class SkillReferenceError(SkillCatalogError): ...
class SkillRobotCompatibilityError(SkillCatalogError): ...
class SkillRegistryError(SkillCatalogError): ...
```

本设计新增的 v1 稳定错误码和当前 baseline 必须保留的公开 runtime code 共同组成稳定词汇。未列入此表的
Python exception、ROS transport exception 或底层 provider 文本不得直接成为自动化接口：

```text
SKILL_PROFILE_NOT_FOUND
SKILL_PACKAGE_NOT_FOUND
SKILL_IMPLEMENTATION_NOT_FOUND
SKILL_SCHEMA_INVALID
SKILL_DUPLICATE_NAME
SKILL_ALIAS_CONFLICT
SKILL_REFERENCE_MISSING
SKILL_UNKNOWN_PRIMITIVE
SKILL_UNKNOWN_EXECUTOR
SKILL_UNKNOWN_POSE
SKILL_UNKNOWN_JOINT
SKILL_LIMIT_VIOLATION
CONTROL_MODE_MISMATCH
SKILL_CONTROL_MODE_MISMATCH
TIMEOUT_EXCEEDS_POLICY
SKILL_SOURCE_CHANGED_DURING_COMPILE
SKILL_RELEASE_NOT_IMMUTABLE
SKILL_RELOAD_DISABLED
SKILL_RELOAD_UNAUTHORIZED
SKILL_REQUEST_ID_CONFLICT
MOTION_NOT_AUTHORIZED
CAPABILITY_NOT_READY
SKILL_BUSY
SKILL_CANCELLED
SKILL_CANCEL_TIMEOUT
GOAL_NOT_FOUND
DUPLICATE_TASK_ID
TASK_ID_CONFLICT
INVALID_ARGUMENT
GATEWAY_FINALIZATION_FAILED
SKILL_AGENT_PLAN_NOT_FOUND
SKILL_AGENT_PLAN_EXPIRED
SKILL_REGISTRY_NOT_READY
SKILL_RELOAD_IN_PROGRESS
SKILL_REGISTRY_EPOCH_MISMATCH
SKILL_REGISTRY_VERSION_MISMATCH
SKILL_SNAPSHOT_NOT_RETAINED
SKILL_SNAPSHOT_DIGEST_MISMATCH
SKILL_PRIMITIVE_CONTRACT_MISMATCH
SKILL_EXECUTOR_IDENTITY_MISMATCH
SKILL_EXECUTION_BUSY
SKILL_WORKFLOW_DIGEST_MISMATCH
SKILL_WORKFLOW_LEASE_MISMATCH
SKILL_WORKFLOW_STEP_MISMATCH
SKILL_DISPATCH_NOT_AUTHORIZED
SKILL_TASK_BUDGET_MISMATCH
SKILL_TASK_DEADLINE_EXPIRED
```

Compiler exception、reload/snapshot response、action result 和 diagnostics 必须使用同一稳定错误码；CLI
按下表将具体错误码映射到稳定的 process exit code。用户消息可以变化，但自动化不得解析自由文本。

兼容规则：当前 direct `/embodied/execute_skill` 和 `/embodied/execute_skill/_action/cancel_goal` 继续输出
`MOTION_NOT_AUTHORIZED`、`CAPABILITY_NOT_READY`、`SKILL_BUSY`、`SKILL_CANCELLED`、`SKILL_CANCEL_TIMEOUT` 和
`GOAL_NOT_FOUND` 等既有 code；新 Agent root action 使用 `SKILL_EXECUTION_BUSY` 等 v1 新 code 时，CLI 不能在
同一 command/schema version 中静默改写旧 direct command 的 code。若需要统一别名，必须在 envelope/IDL version
中显式声明 old-to-new mapping。

直接命令与目标 Agent 命令的兼容矩阵固定如下：

| Current direct code | Stage 3 target handling |
|---|---|
| `CONTROL_MODE_MISMATCH` | Direct command 原样保留；新的 catalog diagnostic 可使用 `SKILL_CONTROL_MODE_MISMATCH`，二者映射必须在 schema/envelope 中声明。 |
| `TIMEOUT_EXCEEDS_POLICY` | Direct command 和新的 Agent plan admission 都原样使用；不增加第二个 prefixed alias，不改变 current direct exit contract。 |
| `DUPLICATE_TASK_ID` / `TASK_ID_CONFLICT` | Gateway status ledger query 原样保留；Agent plan 的 canonical payload/task binding conflict 使用 `SKILL_REQUEST_ID_CONFLICT`。 |
| `INVALID_ARGUMENT` | Direct CLI 输入原样映射 exit `2`；Agent plan schema/IDL 失败使用 `SKILL_SCHEMA_INVALID` 并保留 diagnostics。 |
| `GATEWAY_FINALIZATION_FAILED` | Direct 和 Agent root action 都原样保留；收到该 code 后停止继续派发并要求 operator 介入。 |
| `MOTION_NOT_AUTHORIZED` / `CAPABILITY_NOT_READY` | Direct 和 Agent action admission 都原样保留；launcher control-plane readiness 不得把二者吞掉。 |
| `SKILL_CANCELLED` / `SKILL_CANCEL_TIMEOUT` / `GOAL_NOT_FOUND` | Direct 和 Agent cancellation 都原样保留；`SKILL_CANCEL_TIMEOUT` 继续表示停止状态未知，不能映射为成功。 |

ROS 映射规则：

- Service 业务失败使用正常 response，`success=false` 并填写稳定 `error_code`；仅 transport/node 不可达属于
  ROS 调用失败。
- Action 在 admission 或执行失败时以 ABORTED 结束，并在 result 中填写稳定 `error_code` 和实际版本；用户
  取消使用 CANCELED，不伪装为 catalog 错误。
- Validation response 使用同一 `error_code`；存在多个 catalog 问题时附带完整 `SkillDiagnostic[]`。

`robot-skill` CLI 的粗粒度 process exit code 固定为：

| Exit code | 含义 |
|---:|---|
| `0` | 成功，包括 reload no-op。 |
| `2` | CLI 参数或本地输入格式错误，不对应 catalog error code。 |
| `10` | Profile/package/implementation 不存在。 |
| `11` | Schema、引用、robot compatibility 或 limit 校验失败。 |
| `12` | Source 在编译期间变化或 release immutable 校验失败。 |
| `13` | Registry、snapshot 或 delegated executor identity 状态错误。 |
| `14` | ROS transport、service 或 action server 不可达。 |
| `15` | Timeout policy、task budget 或 deadline 错误。 |

映射按上方 v1 错误码列表分组：前三个 not-found code 为 `10`；从 `SKILL_SCHEMA_INVALID` 到
`SKILL_CONTROL_MODE_MISMATCH` 为 `11`；两个 source/release code 为 `12`；reload/registry/snapshot/executor
identity/Primitive contract/execution busy/Agent plan/三个 Workflow code/dispatch authorization code 为 `13`；三个
timeout/budget/deadline code 为 `15`。Reload disabled、
unauthorized 和 request ID conflict 也映射为 `13`。

目标 Agent plan CLI 中，`MOTION_NOT_AUTHORIZED`、`CAPABILITY_NOT_READY`、`SKILL_BUSY`、`GOAL_NOT_FOUND` 和
`GATEWAY_FINALIZATION_FAILED` 属于 execution/admission group `13`；`SKILL_CANCEL_TIMEOUT` 属于 timeout group
`15`，必须继续表达停止状态未知；只有收到 terminal `SKILL_CANCELLED` 才能将取消报告为已收敛。当前 direct CLI
仍遵循 `robot_skill_cli/README.md` 的 exit `3/4/124/130/143` 契约，直到新的 envelope/IDL 版本原子切换。

错误码应用规则：

- 已有其他 root execution 时，direct Skill 或 Workflow Begin 返回 `SKILL_EXECUTION_BUSY`。
- 当前 baseline direct Skill 在同一条件下仍返回 `SKILL_BUSY`；Stage 3 原子迁移后，Agent root action 才使用
  `SKILL_EXECUTION_BUSY`，不得混用两个 producer/consumer 版本。
- 同一 Begin `root_task_id` 但 digest、identity、budget 或 step payload 不同返回
  `SKILL_REQUEST_ID_CONFLICT`。
- 同一 child task ID 的 canonical payload 不同，或同一 Finalize binding 的 terminal state/count 与 terminal record
  或 active root ledger 不同，返回 `SKILL_REQUEST_ID_CONFLICT`。
- 不存在的 plan token 返回 `SKILL_AGENT_PLAN_NOT_FOUND`；TTL 到期或 registry identity 过期返回
  `SKILL_AGENT_PLAN_EXPIRED` 或 `SKILL_REGISTRY_VERSION_MISMATCH`；plan token 绑定不同 task ID 返回
  `SKILL_REQUEST_ID_CONFLICT`。
- Finalize 的 binding/task budget schema 无效返回 `SKILL_SCHEMA_INVALID`；非空 `dispatch_nonce` 返回
  `SKILL_DISPATCH_NOT_AUTHORIZED`；`task_id != root_task_id` 或 root record 不存在返回
  `SKILL_WORKFLOW_LEASE_MISMATCH`。
- Digest 重算不一致返回 `SKILL_WORKFLOW_DIGEST_MISMATCH`；root nonce、active/terminal root identity 或 active
  cleanup/borrow 门禁不匹配返回 `SKILL_WORKFLOW_LEASE_MISMATCH`；step index、顺序、ledger 状态或 payload 不匹配
  返回 `SKILL_WORKFLOW_STEP_MISMATCH`。

上述执行期错误全部映射 CLI exit code `13`；`SKILL_SCHEMA_INVALID` 仍按通用 schema 分组映射 `11`。

CLI 的 JSON 输出必须同时保留具体稳定 `error_code` 和完整 diagnostics；自动化不得只依赖粗粒度 process
exit code 区分具体错误。

| 校验或职责 | 所属层 |
|---|---|
| Manifest/implementation/profile schema | skill_catalog |
| `semantic_level`、Atomic Operator schema v1 单 Primitive 和 visibility 门禁 | skill_catalog；不得作为执行授权 |
| Alias、instead_use 和 profile 全局引用 | skill_catalog |
| Primitive descriptor vocabulary、参数字段、runtime capability 和 contract digest SSOT | embodied_common |
| Primitive/executor 兼容性校验和 trajectory 展开 | skill_catalog，读取 canonical descriptor/executor registry |
| Named pose、joint 和局部 limit 兼容性 | skill_catalog，使用 robot context |
| Robot hardware limit 权威值 | robot_config |
| Workflow typed steps、digest、planner-visible 过滤和顺序编排 | embodied_agent |
| Gateway admission、Workflow root scope、lease、busy、nonce 和 timeout | skill_library/ExecutionCoordinator |
| 请求参数和当前机器人状态安全校验 | safety_guard |
| Primitive 与 delegated dispatch | skill_library；delegated implementation package |
| MoveIt/GRIPPER/WAIT 底层计划执行 | task_dispatch，不拥有 Workflow |
| 高频模型 action stream | action_dispatch |

## 18. Development Watcher

Watcher 是可选开发功能，默认关闭：

```text
enable_skill_file_watcher: false
skill_file_watcher_debounce_ms: 500
```

规则：

1. 开发模式监听 staging source 的 manifest、implementation 和 profile；生产模式最多监听 `current` 指针。
2. 合并 create、modify、move、delete 事件并 debounce。
3. 忽略隐藏文件、编辑器 swap、临时文件和未完成 release。
4. Watcher 只调用手动 service 使用的同一 reload coordinator。
5. Source release digest 在 debounce 窗口内继续变化时重新等待。
6. Parse 或 schema 失败时保留旧 generation。
7. 多文件协调修改优先使用完整 staging release 后原子激活。
8. Watcher 不绕过 operator policy，也不直接修改 bundle。

## 19. 风险

| 风险 | 缓解措施 |
|---|---|
| epoch 重启混淆 | 在所有事件和请求中携带 epoch。 |
| 过期 consumer 形成 split-brain | 使用精确 snapshot 查询并 fail closed。 |
| reload 干扰活动任务 | 将 coordinator 与 registry 分离。 |
| 部分文件更新 | 使用暂存的不可变 release 根目录。 |
| schema 漂移 | Manifest 示例必须与当前 validator 保持一致。 |
| 多进程短暂版本不一致 | Exact snapshot、版本绑定请求和 safety fail closed。 |
| 历史快照过早清理 | Active execution 引用计数和 snapshot retention。 |
| 安装遗漏 catalog 数据 | 递归安装和普通 install-space 测试。 |
| 新目录成为硬件第二 SSOT | 所有硬限位仅来自 `SkillRobotContext`。 |
| Primitive registry 退化成多份白名单 | `embodied_common` 只读 canonical descriptor + `primitive_contract_digest`；compiler、executor 和 safety 禁止复制名称、参数或默认值。 |
| Atomic Operator 与 Skill 只靠命名区分 | Manifest 必填 `semantic_level`，compiler 执行 schema v1 门禁，review 检查调用语义和公开后置条件。 |
| `semantic_level` 被误作权限边界 | Operator 与 Skill 共用 Gateway 全部准入链；权限只由部署 policy、enablement、lease 和 safety 决定。 |
| Workflow 继续藏在 `context_json` | 使用 typed `WorkflowStep[]`，planned consumer 拒绝 JSON fallback。 |
| Workflow 不进 catalog 后失去版本 | Planned Workflow 计算 `workflow_digest` 并绑定 exact registry identity；未来持久化定义由 `embodied_agent` 独立版本化。 |
| Reload 使 Workflow 前后步骤使用不同版本 | Begin 捕获 bundle 并保留到 Finalize；child 只从 active root 取 snapshot。 |
| Workflow 步骤间释放 lease 导致命令插入 | 显式 Begin/Finalize execution scope；child 只借用 root lease，deadline cleanup fail closed。 |
| 共用 `SkillCommand` 后参数私下扩展 | Schema v1 参数 key 与 ROS IDL 一一对应；新增 key 必须协调升级 schema/IDL，禁止 `context_json` fallback。 |
| 外部 client 绕过 Gateway 调 Primitive | Active dispatch nonce + coordinator lease 校验；非受信 ROS 图额外使用 SROS 2 policy。 |
| Hermes 不可用或版本漂移 | `hermes-robot` 启动前检查 binary/version/Agent Skill；CI 使用本地 fake model transcript，运行时输出稳定 prerequisite error，不静默降级为 raw ROS。 |
| Hermes LLM 生成错误 skill、参数或 Workflow | 只接受 `PlanAgentCommand` 的结构化 plan；exact snapshot、schema、enabled/planner-visible、digest 和 safety 全部由服务/Gateway 再校验，partial plan fail closed。 |
| Hermes 把一般聊天或“直接执行”当作机器人授权 | `ibrobot-control` 强制 plan/validate/展示 exact tuple/confirm-plan/execute；无确认 token 无 action goal，历史确认和通用权限不复用。 |
| Hermes 在多步骤之间独立执行多个 root Skill | 只允许 `ExecuteAgentPlan` 进入 `embodied_agent`；多步骤统一 Begin/child/Finalize，Agent 不持有或生成 root lease nonce。 |
| Agent 进程退出后机器人状态不明 | action cancel 与 task executor cleanup 负责收敛；收到 `SKILL_CANCEL_TIMEOUT` 或 unknown result 时停止后续计划并报告未知，禁止自动重试。 |

## 20. 运维接口目标

接口实现后的命令目标：

```bash
ros2 service call /embodied/reload_skills \
  ibrobot_msgs/srv/ReloadSkillCatalog \
  "{schema_version: 1, request_id: 'manual-001', force: false}"
```

```bash
ros2 service call /embodied/get_skill_snapshot \
  ibrobot_msgs/srv/GetSkillSnapshot \
  "{schema_version: 1, registry_epoch: '', generation: 0}"
```

```bash
ros2 topic echo /embodied/skill_registry_event
```

命令和 README 必须使用真实声明的稳定接口，不依赖节点私有名称展开。

### 20.1 本地 Hermes 启动与验收

外部 Hermes runtime 不纳入 ROS package，也不由 Agent 自动安装。安装/部署阶段必须验证兼容版本；当前开发机
验证命令为 `hermes --version`，设计实现必须在 compatibility matrix 中记录实际测试版本和 provider 配置，不把
API key、session 或用户配置提交到仓库。

下列 Stage 3 命令已在当前工作区实现。运行前仍必须完成构建并通过 `hermes-robot` 的 binary/version、installed
Agent Skill、Gateway status 和 Agent endpoint prerequisite check；检查失败时不能回退到 raw ROS。

机器人启动后，操作员在另一个终端执行：

```bash
source .shrc_local
source install/local_setup.bash
hermes-robot --config-name so101_single_arm
```

`hermes-robot` 必须：

1. 验证 `hermes`、`robot-skill`、install-space Agent Skill 和目标 `robot_config` 可发现。
2. 预加载 `ibrobot-control`，设置/传递精确 `ROBOT_CONFIG`，并以 workspace/installed Agent Skill 作为受控工作目录。
3. 可执行 catalog/status preflight，但不得启动或重启 `embodied_pipeline`，不得设置 `authorize_motion`，不得修改
   ROS 参数；只有 control-plane snapshot 同步、status service 或 Agent plan endpoint 不可用时才退出。`motion_authorized=false`
   或单个 capability `ready=false` 必须保留为可查询的受控状态，运动执行继续由 Gateway fail closed。
4. 不将 provider secret、plan token、root lease nonce 或 dispatch nonce 写入日志、prompt、snapshot 或错误消息。

安全验收默认先以 `authorize_motion:=false` 启动仿真，确认 Gateway 拒绝运动且没有 action goal；操作员完成现场
安全检查后，才允许用 `authorize_motion:=true` 重新启动并执行实际动作。Hermes 交互示例：

```text
用户：请挥挥手
Hermes：读取 status、生成/展示并 flush wave_hello plan、describe、validate-plan，生成 fresh task ID；随后立即
confirm-plan --plan-token <PLAN_TOKEN> --plan-digest <DIGEST> --task-id <fresh-id>；随后执行 execute-plan，并携带
<PLAN_TOKEN>、<CONFIRMATION_TOKEN>、<fresh-id>、展示过的 plan ID/digest、registry identity 和 step count，等待
唯一 terminal result

用户：先打开夹爪，然后回到安全位
Hermes：展示并 flush WORKFLOW 的两个 typed steps、每步参数、snapshot identity、预计顺序和 fresh task ID，完成整体 validate-plan 后立即
confirm-plan --plan-token <PLAN_TOKEN> --plan-digest <DIGEST> --task-id <fresh-id>；随后执行 execute-plan，并携带
token、fresh task ID、展示过的 plan/registry identity 和 expected step count；首步失败、取消或停止未知时不提交第二步
```

直接运行前必须满足：Gateway control-plane status ready、当前 control mode 满足 Skill、operator 已决定 motion authorization、
Hermes provider 已配置、`ibrobot-control` 版本通过 checker。自然语言解析、计划生成、validation 和 action result
都必须可从 JSON/JSONL transcript 重放；不能以“模型说已完成”作为验收结果。

## 21. Definition of Done

本方案完成的最低标准：

1. Robot YAML 不再承载大段生产 catalog template。
2. 每个 enabled Atomic Operator/Skill 都有 manifest、`semantic_level`、selected implementation 和 `SKILL.md`。
3. Profile 显式决定 enabled 和 planner-visible 集合，不存在 visibility 默认放行。
4. Profile 模式与迁移前 normalized runtime 行为等价。
5. Skill 修改后可手动 reload，无需重启 ROS 节点。
6. 无效 reload 不改变 epoch、generation、digest 或 current bundle。
7. 活动任务使用旧 bundle 和 coordinator token 正常结束。
8. Safety 对版本不一致请求 fail closed。
9. Planner、VLM、CLI、safety 和 executor 最终收敛到相同 current snapshot。
10. Late consumer 和 event-loss 场景可恢复。
11. 普通 install space 中可完整加载 Skill 数据。
12. Digest golden vector、跨进程、并发、安装和仿真测试通过。
13. README、CMake、package dependency 和目标基线存在的治理规则同步更新。
14. Snapshot consumer 能从 payload 本地重算 registry/capability/provenance digest。
15. 所有影响 primitive/delegated execution 的 robot context 和 executor identity 都绑定到 captured bundle。
16. Stage 0 冻结的全部 ROS IDL 已实现，`config_digest == capability_digest` 兼容不变量有测试覆盖。
17. Primitive descriptor 只有 `embodied_common` 一个 owner，compiler、executor 和 safety 不再复制白名单。
    Snapshot 绑定 `primitive_contract_digest`，不同 contract 二进制不能共同进入 ready。
18. Planned Task 只使用 typed `WorkflowStep[]` 并引用同一快照中的 enabled entry；自动 planner 生成的步骤
    还必须 planner-visible，所有路径都不得引用 Primitive 或嵌套 Workflow。
19. Atomic Operator schema v1 单 Primitive规则、Skill 单/多 Primitive 正例和 `inspect_scene` 语义校准均有
    测试；`semantic_level` 不影响授权、安全或版本门禁。
20. Primitive/delegated internal dispatch 绑定 active root lease 和 nonce，外部 Primitive goal fail closed。
21. Workflow 使用 canonical digest 和显式 Begin/Finalize execution scope；所有 child 严格按计划顺序借用同一
    root lease、budget 和 captured bundle，reload 不改变后续步骤版本。
22. 错误或过期 root nonce、step index、step payload、Finalize terminal payload 均 fail closed；task executor
    崩溃时 root deadline cleanup 不会提前放开仍在运动的 lease。
23. Hermes 单 Skill/Workflow 只通过 immutable AgentPlan token 进入 `embodied_agent`；单 plan 最多 16 个 step，
    token 默认 TTL 300 秒，未确认或已过期的 plan 不得产生 action goal。
24. Hermes 的每个运动 plan 都必须在内部 confirm 前展示并 flush exact plan；`confirm-plan` 只做 exact tuple
    技术绑定，不是用户二次确认。通用授权、`--yolo` 和自动重试均不能绕过该展示和 Gateway admission。
25. `hermes-robot`、`plan-workflow`、`validate-plan`、`confirm-plan`、`execute-plan` 和 `cancel-plan` 已实现并遵守
    exact snapshot、一次确认、opaque token 和 fail-closed prerequisite 契约。
26. 外部 Hermes provider/runtime 不属于 ROS package 或机器人执行 SSOT；兼容版本、provider 和 transcript 测试
    必须显式记录，secret/session/token 不进入仓库和日志。

## 22. 升级到完整 Robot Skill Package

未来升级到独立 Skill 软件包时，保持以下契约不变：

- 三类 schema。
- `semantic_level` 和四层引用方向。
- Profile 语义。
- Compiler/validator。
- Snapshot、epoch/generation/digest 和 reload 协议。
- Runtime bundle、coordinator 和 consumer 同步逻辑。

主要替换 `DirectoryReleaseSkillSource` 为 `AmentSkillSource` 或其他受控 source，并新增独立 package
metadata、版本依赖、安装/卸载、同名冲突、签名和 provenance。首版不得提前承诺第三方插件 ABI。

## 23. 首版冻结决策

1. 生产 reload service 默认禁用；只有显式运行操作员 policy 可启用，且请求不能指定任意 source path。
2. 历史快照保留 current、全部活动引用和最近两个已完成 generation。
3. 首版 catalog 数据源包名称固定为 `skill_catalog`。
4. `GetSkillSnapshot.srv` 首版固定使用 canonical `snapshot_payload_v1` JSON。未来结构化 ROS message 必须以
   新 schema/IDL 版本引入，不能改变 v1 字节语义。
5. Legacy inline compatibility window 固定为一个 release，下一 release 执行 Stage 6 移除。
6. Primitive 不进入 catalog，首版不能通过 reload 增加执行词汇。
7. Atomic Operator 与 Skill 共用 `config/skills/`、snapshot 和 `SkillCommand`，由必填 `semantic_level` 区分；
   首版不增加 `config/operators/` 或 `CapabilityCommand`。该层级不是授权边界，单 Primitive 是 schema v1
   implementation constraint，不是长期语义定义。
8. 首版 visibility 只有 enabled/public direct 与显式 `planner_visible` 两级，不引入不可执行的 `internal`
   catalog entry；`planner_visible` 不是访问控制字段。
9. Workflow 由 `embodied_agent` 通过 typed `WorkflowStep[]` 拥有，首版不持久化到 `skill_catalog`；planned
   instance 必须计算 `workflow_digest` 并通过 Begin/Finalize scope 在整个生命周期持有同一 root lease、budget
   和 captured bundle。
10. `registry_preimage_v1` 包含 `primitive_contract_digest`，`skills[]` 和 `capability_view` 包含
    `semantic_level`；provenance preimage 顶层 schema 保持不变。
11. Primitive descriptor registry 是静态软件 contract，不是第二个 ROS registry；descriptor 变化要求重新构建
     相关进程，并通过 contract digest 拒绝混跑。
12. Hermes Agent plan store、plan/validate/confirm/execute IDL、opaque token TTL/绑定、caller policy、root
    cancellation、单 Skill/Workflow 分流和 plan confirmation transcript 测试全部通过。
13. Hermes 进程无法启动/重启 pipeline、开启 `authorize_motion`、修改 ROS 参数、调用底层 endpoint 或在
    failure/timeout/unknown result 后自动重试；这些禁止行为有 process conformance 覆盖。
14. Stage 3 之前的文档、README 和验收脚本明确区分当前已实现 `robot-skill` 命令与目标 plan commands，不能
    把未实现接口报告为当前能力。

这些决策不得改变版本模型、coordinator 分离、精确 snapshot 协议或 SSOT 边界。

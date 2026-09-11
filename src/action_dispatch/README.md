# Action Dispatch

动作分发层，位于推理模型与 ros2_control 之间的拉取式动作分发系统。

## 概述

本包将具身智能模型输出的 action chunk 分发给机器人控制器，支持 ACT、Diffusion Policy 等模型。
跨帧融合可缓和重叠预测之间的变化，但不保证物理运动连续或推理供应无间隙。

本包包含两个互斥 executable，具体选择只来自
`control_modes.<mode>.inference.scheduler.enable`：

- 缺失或 `false`：启动 `action_dispatcher_node`，沿
  `executor.inference_pipeline -> DispatchInfer /dispatch + /reset` 工作，行为与原路径一致。
- `true`：启动 `scheduled_action_dispatcher_node`，等待 Global readiness 后依次使用
  `OpenInferenceSession`、`ScheduledDispatchInfer` 和 `CloseInferenceSession`。Open 只建立逻辑 session，
  不携带模型或 fallback；dispatcher 拥有单个 product session，
  校验所有结果 identity；terminal failure 或 `UNKNOWN` 会先清空队列/平滑器并发布 safe-stop，再 Close 并进入
  `FAILED`。每次 Dispatch 都显式携带 target、priority 和新的绝对 deadline；priority-0 还携带 fallback chain。
  只有适合重试的 recoverable `NOT_STARTED` 才使用新 request UUID 且复用原 deadline 做有界重试；deadline
  不可行、容量已满和 ingress 拒绝会直接 safe-stop/Close；Global 为 priority-0 保留独立进入容量，低优先级
  请求不能将其耗尽。Stop/Restart 会等待未决 Open 后用实际
  generation Close；SIGINT/SIGTERM 退出时保持 executor 和 ROS context 存活到 safe-stop/Close 完成或超时。
  新 action chunk 会校验 tensor step 数与 result `chunk_size` 一致，再按推理期间已执行的动作数对齐；队列暂时
  耗尽时持续发布最后动作保持控制输出。safe-stop 的 joint snapshot freshness 使用本进程 monotonic receive
  time，不依赖 ROS/sim time 或消息 header stamp。

两个节点都使用 `/action_dispatcher` 名称并保留 start/stop/status/toggle-smoothing 接口，launch graph 不会让它们
共存。Scheduled dispatcher 消费 Global 返回的 whole-graph action chunk。

## 系统架构

### 组件架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IB Robot System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         ┌─────────────┐ │
│  │  Inference       │         │  Action          │         │  ros2_      │ │
│  │  Service         │         │  Dispatch        │         │  control    │ │
│  │                  │         │                  │         │             │ │
│  │ ┌──────────────┐ │         │ ┌──────────────┐ │         │ ┌─────────┐ │ │
│  │ │ Model        │ │         │ │ Action       │ │         │ │ Joint   │ │ │
│  │ │ (ACT/Diff)   │ │         │ │ Dispatcher   │ │         │ │ State   │ │ │
│  │ └──────────────┘ │         │ │   Node       │ │         │ │ Pub/Sub │ │ │
│  │                  │         │ └──────────────┘ │         │ └─────────┘ │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Temporal     │ │         │             │ │
│  │                  │         │ │ Smoother     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Topic        │───────────▶│ Controllers│ │
│  │                  │         │ │ Executor     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  └──────────────────┘         └──────────────────┘         └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 通信架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS2 Communication                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Inference Service │                              │ Action Dispatch  │     │
│  │                   │                              │                  │     │
│  │                   │    DispatchInfer Action      │                  │     │
│  │                   │◀─────────────────────────────│                  │     │
│  │                   │    (ibrobot_msgs/action)     │                  │     │
│  │                   │                              │                  │     │
│  │                   │    VariantsList (Result)     │                  │     │
│  │                   │─────────────────────────────▶│                  │     │
│  │                   │    (action chunk tensor)     │                  │     │
│  └──────────────────┘                              └────────┬─────────┘     │
│                                                             │                │
│                                                             │                │
│  ┌──────────────────┐                              ┌────────▼─────────┐     │
│  │ ros2_control     │                              │ TopicExecutor    │     │
│  │                  │◀─────────────────────────────│                  │     │
│  │ /joint_commands  │   Float64MultiArray /        │                  │     │
│  │ /arm_commands    │   JointTrajectory            │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Sensor Layer     │                              │ Action Dispatch  │     │
│  │                  │                              │                  │     │
│  │ /joint_states    │─────────────────────────────▶│ (subscription)   │     │
│  │ (JointState)     │   optional                   │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 内部数据流

`continuous` 表示在生命周期允许时，每个控制 tick 消费一个可用动作，不等待逐步执行反馈；
它不表示不停发起推理。补货要求 `remaining <= watermark_threshold`、没有 in-flight 推理，
并通过生命周期门控（legacy 处于 running 且没有 policy reset；scheduled session 处于 ACTIVE）。
`full_chunk` 跳过请求期间已消费步数对应的新 chunk 前缀，再选择全部剩余动作；
它不要求先耗尽旧 chunk，也不把新 chunk 截成仅有重叠的部分。

```text
控制 tick -> 生命周期门控 -> 检查剩余量/水位线 -> 记录计划长度并异步请求推理
                         -> ActivePlan 消费一步（或 hold/empty）-> executor

推理返回 -> 校验身份/解码 -> 计算请求期间消费步数 -> FullChunkPlanner 选择区间
         -> ActivePlan 原子接纳：none 替换；temporal_ensemble 融合重叠并追加新尾部
```

`TopicExecutor` 按 contract 路由 `Float64MultiArray` 或 `JointTrajectory`；benchmark
则通过反馈提交消费。上方通信图展示 legacy topic 路径，scheduled endpoint 见
[话题和服务](#话题和服务)。存储不一定经过 smoother，具体边界如下。

## 调度策略分层与状态归属

action_dispatch 的职责划分为六个边界，legacy
（`action_dispatcher_node`）与 scheduled（`scheduled_action_dispatcher_node`）
两条互斥产品路径共享同一套边界契约：

| 层 | 模块 | 职责 | 不负责 |
|----|------|------|--------|
| Session lifecycle | 节点生命周期状态机 | 会话 open/close、retry、结果去重、safe-stop、fail-and-close；放行 per-tick 调度 | chunk 算法、融合权重 |
| Per-tick scheduler | `schedulers/`（registry + `continuous` / `wait_for_feedback`） | 每个 tick 是否请求推理、是否提交动作（`should_replenish_plan` 为共享水位线规则） | 生命周期状态、chunk 内容 |
| Chunk planning | `chunk_planning.py`（`FullChunkPlanner`） | 描述原始 chunk 的 `[start, stop)` 区间及可选补货阈值 | 修改活动计划、发布动作、请求推理 |
| Active plan | `active_plan.py`（`ActivePlan`） | 原子持有动作、来源、位置、水位线及 revision；直接消费或 reservation/commit；选择 hold/empty | ROS I/O、episode/session 迁移 |
| Action blending | `temporal_smoother.py` | 先准备重叠加权动作及 counts 再提交引用，保持既有系数语义 | 推理时机、会话状态 |
| Executor | `executors/`（registry + `topic` / `benchmark`） | 最终输出通道 | 任何调度决策 |

三条路径均经过 planning 和 owner 接纳。benchmark 仍由反馈驱动消费：reserve/submit
不消费，匹配且经 episode 接受的反馈只提交一次，不使用 continuous 的直接消费入口。

| 产品 | executor / scheduler | 存储与 blending | 容量 / 消费 |
|------|----------------------|-----------------|-------------|
| Legacy continuous | `topic` / `continuous` | `none` 使用 queue；已有 smoother manager 支持 `temporal_ensemble` 或关闭融合后的透传 | queue 保留容量内最新动作；逻辑直接消费 |
| Legacy benchmark | `benchmark` / `wait_for_feedback` | 同上 | 同样的 queue 裁剪；带 revision 的 reservation 和反馈提交 |
| Scheduled | 仅 `topic` / `continuous` | `none` 使用 queue；`temporal_ensemble` 使用独立 smoother | queue 超容拒绝并 safe-stop/close；逻辑直接消费 |

smoother 不继承 deque 容量限制。legacy toggle 保留已有 manager 及计划；没有 manager
的节点不能通过 toggle 创建它。scheduled toggle 保留非活动 queue 及其元数据，关闭
融合时丢弃 smoother 计划，不执行跨 store 动作迁移。legacy continuous stop/start
保留动作、来源、位置和水位线，同时使旧 in-flight 请求失效；reset 和 benchmark
清理丢弃计划。scheduled stop/safe-stop/close/restart 清理两个 store，包括非活动元数据。

### 状态归属（两条路径的既有差异，重构予以保留）

| 状态/语义 | 归属 |
|-----------|------|
| `policy_reset_in_progress` 推理门控、request/generation 记账 | legacy 节点 |
| queue 裁剪、已接纳来源区间、revision 和水位线 | active-plan owner（由产品选择超容策略） |
| 超容即失败（`ValueError` → safe-stop + session close）、`(session_id, generation, request_id)` 结果去重 | scheduled 路径 |
| session 状态机（WAITING_READY/…/FAILED）与重试 | scheduled 路径（不迁入 `DispatchScheduler`） |
| 水位线补货规则 | 共享（`schedulers.continuous.should_replenish_plan`） |

### 扩展点（后续 change）

- **#401 AutoHorizon**：`executed_during_inference` 为原始 chunk 起点，`execution_horizon` 为排他的终点，
  不是 skip 后的长度。`replenishment_watermark=None` 使用配置默认，0 表示耗尽才补货；
  两种存储只应用一次区间。attention 估计、模型能力校验及 runtime options 联动未实现。
- **#411 RTC**：`PlanSource` 记录 request 及适用的 generation/session 身份；直接单源计划的
  `PlanSnapshot.next_position` 包括 skip 和容量丢弃前缀。revision 只用于本地 reservation 失效。
  ensemble 没有精确单源坐标，最近来源仅用于诊断；本地接纳/topic 进度不是物理完成或远端缓存确认。
  跨请求缓存身份、路由/失败协调、坐标变换/相对动作重锚定及 wire 转发仍需补齐。
  AutoHorizon/RTC 均不是当前生产策略名；后续 PR 须独立重放并验证节点接线和协议。

## 安装

在工作区根目录完成环境搭建后构建本包：

```bash
source .shrc_local
./scripts/build.sh --packages-up-to action_dispatch
source .shrc_local
```

## 配置与使用

### 启动节点

完整机器人使用 robot_config launch；独立 legacy 节点调试入口如下，仍需有效机器人 YAML、
推理服务和控制器。scheduled 的额外配置见 [推理调度控制面](../robot_config/README.md#推理调度控制面)。

```bash
ros2 run action_dispatch action_dispatcher_node --ros-args -p robot_config_path:=/path/to/robot.yaml
```

### 策略配置与组合校验

策略名与合法组合的 SSOT 位于
`robot_config.dispatch_strategies`，launch builder（launch 期校验）与两个
dispatcher 节点（init 期防御性校验）共用同一解析器，并校验入口能力：

机器人 YAML 中 `executor`、`dispatch`、`inference` 同属 `robot.control_modes.<mode>`。
完整 schema 见 [robot_config 的动作分发策略 SSOT](../robot_config/README.md#动作分发策略-ssot)。

- legacy 合法配对：`topic`+`continuous`、`benchmark`+`wait_for_feedback`。
  scheduled 仅支持 `topic`+`continuous`，显式不支持的选择报错而非丢弃。
  历史 executor `action` 仅在 robot_config launch 边界映射为 `topic`，直接节点入口不接受该别名。
- **破坏性接口变更**：删除 `temporal_smoothing_enabled`，不提供别名或兼容路径。
  旧机器人 YAML 和直接 ROS override（构造参数、CLI、参数文件）即使值一致也会报错。
  原 false 迁移为 `dispatch.blending: none`，原 true 迁移为 `temporal_ensemble`；
  直接 ROS 调用使用 `blending_strategy`。
- 缺省 `dispatch.chunking`/`dispatch.blending` 解析为 `full_chunk`/`none`。
  legacy 和 scheduled 独立节点的默认参数均为 `blending_strategy: none`。
- 策略名缺失、null、空字符串采用默认值（`topic`、`continuous`、`full_chunk`，
  `none`）。false、0、list、dict、未知名及大小写变体拒绝。
- 对应节点参数为 `executor_type`、`scheduler_mode`、`chunking_strategy`、
  `blending_strategy`。ROS 参数仍受类型约束；配置层的
  YAML null 默认语义不代表 ROS CLI 可以传入 null override。
- 有意拒绝变化：错误形状、原始空 chunk、非有限动作不接纳；legacy continuous
  保留旧计划并记录拒绝，benchmark 在成功启动前以 inference-failed 终止，scheduled
  safe-stop/close。合法非空 chunk 的空选择区间会清空可执行存储。拒绝替换不改变已接纳水位线。

### 参数配置

下表为独立节点默认值；robot_config launch 可用 `executor` 中的同名字段覆盖运行参数，
策略字段则来自 `dispatch`。legacy 专用端点不用于 scheduled；其控制面配置见
[推理调度控制面](../robot_config/README.md#推理调度控制面)。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `executor_type` | string | `topic` | 输出通道，配对约束见上方 |
| `scheduler_mode` | string | `continuous` | 每 tick 调度，不是推理服务的产品入口开关 |
| `chunking_strategy` | string | `''` | 空值解析为 `full_chunk` |
| `blending_strategy` | string | `none` | `none` 或 `temporal_ensemble` |
| `queue_size` | int | 100 | queue 最大长度；不限制 smoother |
| `watermark_threshold` | int | 20 | 剩余量小于等于此值时可补货；不是固定执行间隔 |
| `control_frequency` | double | 100.0 | 控制 tick 频率（Hz），不是推理频率 |
| `robot_config_path` | string | `''` | 含 contract 的机器人 YAML 路径 |
| `joint_state_topic` | string | `/joint_states` | 关节状态话题 |
| `navigation_mode` | bool | false | 启动时等待外部触发 |
| `temporal_ensemble_coeff` | double | 0.01 | 指数融合系数 |
| `chunk_size` | int | 100 | smoother 权重表大小，不决定模型实际返回步数 |
| `smoothing_device` | string | `''` | 空值使用输入 tensor 设备，NumPy 输入为 CPU |
| `inference_action_server` | string | `/inference/policy/dispatch` | 仅 legacy；launch 根据命名 pipeline 覆盖 |
| `inference_reset_service` | string | `/inference/policy/reset` | 仅 legacy；reset 时 best-effort 调用 |
| `policy_reset_timeout_sec` | double | 2.0 | 仅 legacy；等待 policy 重置的最长时间 |

### 推理补货与融合配置示例

以下是合并到已有机器人 YAML 的局部配置，不是完整启动配置。`executor` 与 `dispatch`
同属 `robot.control_modes.model_inference`；仍需配置 `inference.enabled`、有效的
`inference.pipelines`、控制器和 contract。`executor.inference_pipeline: policy` 必须对应
已声明的 pipeline；scheduled 入口还需其 session/调度配置，不能仅靠下面片段启用。

**耗尽后补货，不融合：**

```yaml
robot:
  control_modes:
    model_inference:
      executor:
        type: topic
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 0
      dispatch:
        scheduler: continuous
        chunking: full_chunk
        blending: none
```

水位线为 0 时，旧计划耗尽后才请求下一个 chunk。等待结果期间没有新的计划动作，
若已有最后动作就继续发布它（hold last）；这不是物理停止或 safe-stop，也不保证运动无停顿。
hold 不消费计划步数，因此不会增加新 chunk 的跳过长度。

**提前补货，融合重叠区域：**

```yaml
robot:
  control_modes:
    model_inference:
      executor:
        type: topic
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 80
        chunk_size: 100
        temporal_ensemble_coeff: 0.01
      dispatch:
        scheduler: continuous
        chunking: full_chunk
        blending: temporal_ensemble
```

假设 A、B 每次实际返回 100 步，期间没有暂停、切换或失败；以下索引从 0 开始，右端不包含：

```text
A 返回：100 步
  -> 消费 20 步：A[20:100]，剩余 80，触发 B 请求
  -> B 异步推理期间消费 30 步：A[50:100]，剩余 50
  -> B 返回 100 步：跳过 B[0:30]，保留 B[30:100]，共 70 步

接纳时的对齐：       重叠 50 步                  新尾部 20 步
旧计划              A[50:100]                   （无）
新计划              B[30:80]                    B[80:100]
输出                blend(A[50:100], B[30:80]) + B[80:100] = 70 步
```

跳过的是 B 的观察/请求基准建立后实际消费的 **30 步**，不是 A 累计消费的 50 步。
当前实现按请求开始与返回时的计划长度差对齐，不按墙钟耗时换算步数，也不额外补偿
传感器样本在请求前的年龄。这里的消费是本地逻辑进度，不代表机器人已经物理完成动作。
70 步仍低于水位线 80，因此在请求结束且其他门控允许时，下一 tick 就可能请求 C；
水位线 80 不等于固定每执行 20 步推理一次。提前补货也不能保证推理足够快而永不耗尽。

水位线与融合是独立选择：

- 水位线为 0 且省略 `dispatch.blending` 时，当前 resolver、legacy/scheduled launch 和
  独立节点均默认 `none`；本 README 后面的 Python launch 示例是显式开启 ensemble，不是默认值。
- 水位线为 0 且显式选择 `temporal_ensemble` 时，不会自动改成 `none`。正常耗尽后补货没有
  新旧计划重叠，但仍使用 smoother，计算/存储开销、容量、来源坐标与 toggle 行为不同，
  不能视为与 `none` 完全等价，具体差异见上方存储与状态归属说明。
- 水位线为 80 且选择 `none` 时，仍提前请求、跳过同样的前缀，但用 B 的 70 步直接替换
  A 剩余的 50 步，不融合，也不排在旧计划之后等待执行。

### Launch 文件示例

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='action_dispatch',
            executable='action_dispatcher_node',
            name='action_dispatcher',
            parameters=[{
                'queue_size': 100,
                'watermark_threshold': 80,
                'control_frequency': 100.0,
                'blending_strategy': 'temporal_ensemble',
                'temporal_ensemble_coeff': 0.01,
                'chunk_size': 100,
                'robot_config_path': '/path/to/robot.yaml',
            }]
        )
    ])
```

## 跨帧平滑功能

### 对齐与重叠

时序统一使用上方[异步补货示例](#推理补货与融合配置示例)。一次推理返回的是一个包含 N 步动作的
chunk，不是 N 个 chunk。planner 先选择 `[skip:N]`，owner 只应用一次该区间；smoother
收到的是已对齐动作，不再次跳过前缀。

重叠长度为 `min(旧计划剩余长度, 新选择长度)`。融合后只追加新计划超出重叠的尾部，
不保留旧计划超出新选择长度的尾部，因此最终长度等于新选择长度。示例首个位置是
`A[50]` 对齐 `B[30]`，而不是按两个 chunk 的相同下标配对。

### 平滑公式

```python
blended[i] = (old[i] * cumsum[count[i]-1] + new[i] * weight[count[i]]) / cumsum[count[i]]
```

其中：
- `old[i]`: 旧动作规划中的第 i 个动作
- `new[i]`: 已对齐的新选择区间中的第 i 个动作
- `count[i]`: 该位置已有的预测贡献数，初始为 1
- `weight[k]`: 第 k 次贡献的权重 = exp(-coeff * k)
- `cumsum[k]`: 累积权重和

示例首个位置 `count=1`：`(A[50] * 1 + B[30] * exp(-0.01)) / (1 + exp(-0.01))`，
约为 `0.5025 * A[50] + 0.4975 * B[30]`。新尾部的 count 从 1 开始；当 count 达到
`chunk_size` 时该位置冻结，不再接纳新的加权贡献。`chunk_size` 是权重表大小，
不是模型输出长度或 queue 容量限制。

### 平滑系数说明

| 系数值 | 效果 |
|--------|------|
| `0.0` | 均匀权重，无新旧偏好 |
| `正数` | 更倾向于旧动作（稳定、保守） |
| `负数` | 更倾向于新动作（响应快，可能抖动） |

当前默认系数为 `0.01`；融合权重不决定推理时机。

### 运行时切换

`blending_strategy` 接受启动时 CLI/YAML 配置，但拒绝运行时直接参数写入，包括合法值和同值写入。
两节点只通过 `~/toggle_smoothing`（Empty）切换：所有参数 veto 回调通过后才修改计划状态，
成功后同步更新权威策略和公开参数，导出参数并重启后保持融合选择。拒绝只写日志，Empty 没有错误字段。
legacy 以 `none` 启动时没有 manager，不能通过 toggle 开启；已有 manager 的计划被保留。
scheduled 保留非活动 queue，关闭融合时丢弃 smoother 计划，不跨 store 迁移动作。
暂停与清理差异见[状态归属](#调度策略分层与状态归属)。以下调用可能改变机器人状态：

```bash
# 切换平滑开关
ros2 service call /action_dispatcher/toggle_smoothing std_srvs/srv/Empty

# 仅 legacy：重置状态；scheduled 使用 restart_session（Trigger）
ros2 service call /action_dispatcher/reset std_srvs/srv/Empty "{}"
```

## 导航模式 (Navigation Mode)

当 `navigation_mode=true` 时，系统启动后处于停止状态，等待外部触发开始执行。此模式用于 nav2 导航到达目的地后，触发 ACT 模型执行抓取任务。

### 工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Navigation Mode 工作流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 系统启动                                                                  │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  启动时 _is_running = False                              │
│     │ [NAV] 模式  │  系统就绪，等待触发                                        │
│     └─────────────┘                                                          │
│                                                                              │
│  2. Nav2 导航                                                                 │
│     ┌─────────────┐                                                          │
│     │   Nav2      │  导航到目标位置                                            │
│     │  导航中...  │  Dispatcher 不执行动作                                     │
│     └─────────────┘                                                          │
│                                                                              │
│  3. 到达目的地                                                                │
│     ┌─────────────┐                                                          │
│     │  Nav2 完成  │  调用 /action_dispatcher/start_evaluate                  │
│     └─────────────┘                                                          │
│                                                                              │
│  4. ACT 执行                                                                  │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  _is_running = True                                      │
│     │ 开始执行    │  触发推理，执行 ACT 动作序列                                │
│     └─────────────┘                                                          │
│                                                                              │
│  5. 任务完成                                                                  │
│     ┌─────────────┐                                                          │
│     │  调用       │  调用 /action_dispatcher/stop_evaluate                   │
│     │ stop_evaluate│  停止执行，底盘速度置零                                   │
│     └─────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 使用方法

```bash
# 启动系统（navigation_mode=true 时）
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_realsense_navigation control_mode:=navi

# Nav2 到达目的地后，开始执行
ros2 service call /action_dispatcher/start_evaluate std_srvs/srv/Trigger

# 任务完成后，停止执行
ros2 service call /action_dispatcher/stop_evaluate std_srvs/srv/Trigger

# 查询当前状态
ros2 service call /action_dispatcher/get_status std_srvs/srv/Trigger
```

### 配置示例

在机器人配置 YAML 的 `robot` 块中合并以下局部设置：

```yaml
control_modes:
  navi:
    executor:
      navigation_mode: true    # 启用导航模式
      watermark_threshold: 20
      control_frequency: 30.0
```

## 话题和服务

### 与 Inference Service 通信

| 方向 | 话题/Action | 消息类型 | 说明 |
|------|-------------|----------|------|
| legacy 请求 | `/inference/policy/dispatch` | `ibrobot_msgs/action/DispatchInfer` | scheduler 缺失/false 时向 `executor.inference_pipeline` 发送推理请求 |
| scheduled session | `/inference/session/open`、`/inference/session/close` | `OpenInferenceSession`、`CloseInferenceSession` | scheduler true 时只访问 Global endpoint；Open 不参与模型路由 |
| scheduled 请求 | `/inference/dispatch` | `ScheduledDispatchInfer` | scheduler true 时携带 session/generation/request/target/priority/deadline；priority-0 另带 fallback chain |
| 响应 | `result.action_chunk` | `ibrobot_msgs/msg/VariantsList` | 接收动作块 (Tensor) |

### 发布话题

| 话题 | 消息类型 | 说明 |
|------|----------|------|
| `~/queue_size` | `std_msgs/Int32` | 当前队列长度 |
| `~/smoothing_enabled` | `std_msgs/Bool` | 平滑是否启用 |

### 订阅话题

| 话题 | 消息类型 | 说明 |
|------|----------|------|
| `/joint_states` | `sensor_msgs/JointState` | 关节状态（可选） |

### 服务

| 服务 | 类型 | 说明 |
|------|------|------|
| `~/reset` | `std_srvs/Empty` | （仅 legacy/关闭分支）重置队列和状态；同时 best-effort 调用 `inference_reset_service` 重置推理侧 policy 状态。调度启用路径不使用本服务，改用 `~/restart_session` |
| `~/toggle_smoothing` | `std_srvs/Empty` | 切换平滑开关 |
| `~/start_evaluate` | `std_srvs/Trigger` | 恢复 dispatcher 执行 |
| `~/stop_evaluate` | `std_srvs/Trigger` | 暂停 dispatcher 执行；`navigation_mode=true` 时额外停止底盘 |
| `~/get_status` | `std_srvs/Trigger` | 获取运行状态；scheduled 路径返回 session state machine 状态 |
| `~/restart_session` | `std_srvs/Trigger` | 仅 scheduled executable：safe-stop、Close、清理本地状态并用新 UUID Open |

scheduled 路径的 `~/start_evaluate`、`~/stop_evaluate` 和 `~/restart_session` 在生命周期操作竞争时
返回 `success=false`、`message="lifecycle operation in progress"`，表示本次请求的操作**未执行**。
这同样适用于 Stop：busy 响应不代表停止成功。调用方必须检查响应，并根据状态决定是否重试。

### 与 ros2_control 通信

| 方向 | 话题 | 消息类型 | 说明 |
|------|------|----------|------|
| 发布 | `/joint_commands` | `std_msgs/Float64MultiArray` | 关节位置命令 |
| 发布 | `/arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 轨迹命令 |

## API 使用

### 直接使用 TemporalSmoother

```python
from action_dispatch import TemporalSmoother, TemporalSmootherConfig

# 创建配置
config = TemporalSmootherConfig(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# 创建平滑器
smoother = TemporalSmoother(config)

# 第一次推理
actions1 = model.inference(obs)  # shape: (100, action_dim)
smoother.update(actions1, actions_executed_during_inference=0)

# 请求 B 前先消费 20 步
for _ in range(20):
    robot.execute(smoother.get_next_action())

# 在请求基准采样并计算 B；模拟异步结果尚未交给 smoother
actions2 = model.inference(obs)
for _ in range(30):
    action = smoother.get_next_action()
    robot.execute(action)

# 交付 B：请求期间消费 30 步，更新后剩余 70 步
smoother.update(actions2, actions_executed_during_inference=30)

# 继续执行平滑后的动作
while smoother.plan_length > 0:
    action = smoother.get_next_action()
    robot.execute(action)
```

### 使用 TemporalSmootherManager

```python
from action_dispatch import TemporalSmootherManager

manager = TemporalSmootherManager(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# 运行时切换
manager.set_enabled(False)  # 禁用平滑
manager.set_enabled(True)   # 启用平滑

# 查看状态
print(f"Plan length: {manager.plan_length}")
print(f"Smoothing enabled: {manager.is_enabled}")
```

## 依赖

- ROS2 Humble
- Python 3.10+
- PyTorch
- NumPy
- ibrobot_msgs
- tensormsg

## 许可证

Apache License 2.0

# IB-Robot 链路追踪（ros2_tracing + LTTng）

基于 ROS 2 最佳实践的低开销推理链路追踪方案。

## 环境准备

```bash
./scripts/setup.sh
```

`./scripts/setup.sh` 现在会一并安装 tracing 依赖（LTTng、ros2_tracing、
babeltrace2、`tracetools-analysis`）。

如果工作区已经初始化完成，只是后补 tracing 工具，仍可单独执行：

```bash
bash scripts/tracing/setup_tracing.sh
```

## 使用方式

### 方式 A：在 launch 中集成开启（推荐）

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm use_sim:=true \
    control_mode:=model_inference \
    enable_tracing:=true
```

这会通过 `robot_config.launch_builders.tracing` 在 launch 期间创建一个 LTTng
session，同时采集 ROS 2 UST 事件（`ros2:*`）和 Python 业务追踪点
（`ib_trace.*`）。

如果默认 session 名 `ib_robot_trace` 已经被占用，launch 会自动追加时间戳后缀，
避免覆盖已有 trace；如果你显式传了 `trace_session_name:=...`，则不会帮你覆盖同名会话。

`IB_TRACE_ENABLED=1` 通过局部 launch/进程环境传给本次业务子进程，包括 controller
readiness 后启动的节点；不再提前赋值父进程 `os.environ`，launch scope 退出时恢复环境，
避免泄漏到同进程后续 launch。
control mode、执行节点参数和模型加载行为保持不变。

保留原有 LTTng 启动失败契约：会话检查、create、enable-event 或 start 失败仍中止
launch，清理仅作用于本次成功 create 的会话；不新增 strict 开关。Topology sidecar
不在启动链路中，独立元数据导出失败不会影响机器人启动或取消成功录制的会话。

### 方式 B：单独控制 trace 会话

```bash
# 终端 1
bash scripts/tracing/start_trace.sh

# 终端 2
source .shrc_local
export IB_TRACE_ENABLED=1
ros2 launch robot_config robot.launch.py ...

# 结束后
bash scripts/tracing/stop_trace.sh
```

手动模式下如果默认 session 名已存在，`start_trace.sh` 也会自动换成带时间戳的新名字；
如果你显式传入了同名 session，则脚本会直接报错，避免踩掉旧数据。
`IB_TRACE_ENABLED=1` 必须在启动机器人节点的终端中设置；终端 1 的脚本无法为终端 2
启用 Python 业务埋点。

### 可选声明拓扑

普通分析不需要机器人 YAML、模型文件或 topology sidecar。需要配置声明图时可独立运行：

```bash
source .shrc_local
ros2 run robot_config ibrobot-trace-topology \
    src/robot_config/config/robots/so101_single_arm.yaml \
    --control-mode model_inference > /tmp/ibrobot-topology.json
```

CLI 仅通过已有 `load_robot_section` 解析 YAML 和 sibling `base_config` 继承，不调用
完整业务 loader、不检查模型可用性、不加载模型；业务启动的校验保持不变。
元数据标注 `provenance=declared`、`runtime_verified=false`，只表达配置声明，不代表
实际 started nodes，也不包含 launch 覆盖（除 `--control-mode`）、nav stage 或运行时派生项。
Launch 不自动生成此文件，CLI 也不管理 LTTng 会话。

### 结果分析

新 trace 使用 `IBTRACE1` structured events 作为指标主来源；分析器仍保留旧
`[event] key=value` parser，以便读取历史 trace。迁移期间节点会双写，两种表示同时存在时
优先 structured 指标，并避免 observation 重复计数。

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-trace summary ~/.ros/tracing/ib_robot_trace
```

`scripts/tracing/analyze_trace.py` 保留为历史命令的兼容入口：

```bash
python3 scripts/tracing/analyze_trace.py --trace-dir ~/.ros/tracing/ib_robot_trace
```

完整 CLI、查询投影、Critical Path、Span Profile 和 trace 比较说明见
[`src/ibrobot_tracing/README.md`](../../src/ibrobot_tracing/README.md)。

## 工作原理

1. **`robot_config.launch_builders.tracing` 管理 LTTng session**
   `robot.launch.py` 只负责声明 launch 参数并组合 builder 输出；tracing builder
   启动时启用 `ros2:*` UST 事件和 Python tracing domain `ib_trace.*`，
   退出时自动 stop/destroy session。

2. **`ibrobot_tracing` 提供结构化埋点和离线分析**
   节点通过 `TraceEmitter`、`trace_context()`、`span()`、`event()` 和 Flow API 生成
   `IBTRACE1` 记录。`ib_trace.*` logger 是 LTTng Python domain 的传输契约，
   `component_id` 是分析层的业务身份。

3. **`ibrobot-trace` 是主要离线分析入口**
   `AnalysisService` 读取 CTF 或结构化日志，并向 CLI 和 Web API 提供同一套 Query/Projection。

4. **`ibrobot_tracing_web` 和 Vue 工作台仅在 Ubuntu 运行**
   openEuler 保留 trace 采集和 Core 分析能力；将 trace 文件传到 Ubuntu 后可使用浏览器工作台。

5. **ROS 2/LTTng 标准工具保持可用**
   `ros2 trace`、`lttng`、`babeltrace2` 和 Trace Compass 仍可用于 session 管理和底层检查。

## 业务追踪点

Event 和 Span 的代码 SSOT 是 `BUILTIN_TRACEPOINT_REGISTRY`；Flow edge 的代码 SSOT 是
`topology.py`。所有内置身份的 `origin` 均为 `built-in`。

### Event

| Component ID | Name | Logger | 说明 |
|---|---|---|---|
| `action_dispatcher.request` | `dispatch_request` | `ib_trace.dispatch` | 发起策略推理请求 |
| `action_dispatcher.decode` | `dispatch_result` | `ib_trace.dispatch` | 推理结果及解码状态 |
| `action_dispatcher.queue` | `queue_refill` | `ib_trace.dispatch` | 队列更新后的数量状态 |
| `action_dispatcher.execute` | `action_execute` | `ib_trace.dispatch` | 采样保留的动作发布结果 |
| `action_dispatcher.execute` | `first_action_execute` | `ib_trace.dispatch` | 首个非保持动作发布结果 |
| `action_dispatcher.execute` | `action_topic_publish` | `ib_trace.execute` | 控制话题发布结果 |
| `action_dispatcher.execute` | `safe_stop_topic_publish` | `ib_trace.execute` | scheduled safe-stop 安全命令发布结果 |
| `policy` | `dispatch_result` | `ib_trace.policy` | 策略节点推理结果状态 |
| `policy.observation` | `obs_receive` | `ib_trace.policy` | 单项观测接收及传输时延 |
| `policy.observation` | `obs_sample` | `ib_trace.policy` | 单项观测采样及新鲜度 |
| `policy.observation` | `obs_frame` | `ib_trace.policy` | 观测帧完整度 |

### Span

| Component ID | Name | Logger | 说明 |
|---|---|---|---|
| `action_dispatcher.decode` | `dispatch_decode` | `ib_trace.dispatch` | 解码动作块 |
| `action_dispatcher.queue` | `queue_refill` | `ib_trace.dispatch` | 更新动作队列 |
| `action_dispatcher.execute` | `action_execute` | `ib_trace.dispatch` | 发布采样保留的动作 |
| `action_dispatcher.execute` | `first_action_execute` | `ib_trace.dispatch` | 发布首个非保持动作 |
| `policy` | `policy_pipeline` | `ib_trace.policy` | 策略节点完整请求流水线 |
| `policy` | `policy_total` | `ib_trace.policy` | 预处理、推理和后处理总过程 |
| `policy` | `cloud_roundtrip` | - | 注册的分布式身份；当前时延由 edge send/receive 事件计算 |
| `policy.observation` | `observation_sampling` | `ib_trace.policy` | 组装观测帧 |
| `policy.preprocess` | `preprocess` | `ib_trace.policy` | 生成模型输入 |
| `policy.inference` | `model_call` | `ib_trace.policy` | 本地模型前向计算 |
| `cloud_inference` | `model_call` | `ib_trace.policy` | 云端模型前向计算 |
| `global_scheduler` | `scheduler_dispatch` | `ib_trace.scheduler` | scheduled 请求准入、选路和下游调用 |
| `policy.postprocess` | `postprocess` | `ib_trace.policy` | 模型输出转换 |
| `policy.postprocess` | `action_chunk_publish` | `ib_trace.policy` | 封装并发布动作块 |
| `policy.postprocess` | `result_encoding` | `ib_trace.policy` | scheduled 动作结果编码 |

同名 Event 和 Span 表达不同语义，例如 `queue_refill` Span 表示耗时，Event 表示完成后的队列状态。

### Flow

| Edge ID | Source Component | Target Component | 说明 |
|---|---|---|---|
| `dispatch_to_observation` | `action_dispatcher.request` | `policy.observation` | DispatchInfer 请求传递 |
| `observation_to_preprocess` | `policy.observation` | `policy.preprocess` | 观测帧进入预处理 |
| `preprocess_to_inference` | `policy.preprocess` | `policy.inference` / `cloud_inference` | 模型输入进入推理 |
| `inference_to_postprocess` | `policy.inference` / `cloud_inference` | `policy.postprocess` | 模型结果进入后处理 |
| `result_to_decode` | `policy.postprocess` | `action_dispatcher.decode` | 推理结果返回动作分发器 |
| `decode_to_queue` | `action_dispatcher.decode` | `action_dispatcher.queue` | 解码动作块进入队列 |
| `queue_to_execute` | `action_dispatcher.queue` | `action_dispatcher.execute` | 队列动作进入执行 |
| `scheduled_dispatch_to_scheduler` | `action_dispatcher.request` | `global_scheduler` | scheduled 请求进入 Global scheduler |
| `scheduler_to_pipeline_dispatch` | `global_scheduler` | `policy.observation` | Global scheduler 调用选中的 pipeline |
| `pipeline_result_to_scheduler` | `policy.postprocess` | `global_scheduler` | pipeline 结果返回 Global scheduler |
| `scheduler_result_to_dispatcher` | `global_scheduler` | `action_dispatcher.decode` | Global scheduler 结果返回分发器 |

前七条 legacy edge 在 scheduler 缺失或关闭时使用；四条 scheduler edge 与内部 processor、queue edge
共同组成 scheduler-enabled 路径。
scheduler transport flow ID 包含 ROS action goal UUID，因此幂等重放仍会形成相互独立的 flow pair。
多 pipeline scheduler 的 Topology 使用统一的 `Policy Pipelines` 逻辑路径，并通过 `pipeline_ids`、
`pipeline_nodes` 和事件中的 `pipeline_id` 保留 primary/fallback 的实际身份，避免将 fallback 标成 primary。

## 文件列表

```
robot.launch.py          ← launch 编排入口（enable_tracing:=true 时接入 tracing builder）
src/robot_config/robot_config/launch_builders/tracing.py
                         ← 启动/停止 LTTng session
src/ibrobot_tracing/     ← 结构化埋点、解析、离线分析和 CLI
tools/ibrobot_tracing_web/ ← 显式启用的 FastAPI 服务
web/ibrobot_tracing_ui/  ← Vue 工作台；生产 dist 由 Ubuntu 用户本地构建且不入库
scripts/tracing/
├── setup_tracing.sh     ← 给已初始化工作区补装 tracing 依赖（常规 setup 已包含）
├── start_trace.sh       ← 手动启动 tracing session
├── stop_trace.sh        ← 手动停止 tracing session
├── analyze_trace.py     ← 历史分析命令兼容入口
├── README.md            ← 中文文档
└── README.en.md         ← 英文文档
```

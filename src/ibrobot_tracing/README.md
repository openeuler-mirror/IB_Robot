# IB-Robot 性能追踪

`ibrobot_tracing` 提供 IB-Robot 的 Python 结构化埋点、LTTng CTF/日志解析、时延统计、
Timeline、Call Tree、Flow 和业务 Topology 分析。采集和分析相互解耦：机器人运行时只负责将
事件写入 trace，任务结束后由 `ibrobot-trace` 离线分析。

当前端到端性能边界是：

```text
Dispatch Request -> First Action Execute
```

该边界同时适用于 scheduler 关闭时的 legacy `DispatchInfer` 路径，以及 scheduler 开启时经过
`ScheduledActionDispatcher -> Global Scheduler -> Pipeline` 的 scheduled 路径。Topology 会根据
robot YAML 中的 `scheduler.enable` 记录实际 dispatch path。
多 pipeline scheduler 在图中聚合为 `Policy Pipelines` 逻辑路径，Topology metadata 的 `pipeline_ids`、
`pipeline_nodes` 与事件字段 `pipeline_id` 用于区分 primary/fallback；transport flow ID 还包含 ROS action
goal UUID，使同一 request 的幂等重放保持为独立 flow pair。

## 1. 环境准备

所有 ROS 2 和项目命令都应从 IB_Robot 根目录执行：

```bash
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=42
```

首次使用或代码更新后，构建相关包：

```bash
source .shrc_local
colcon build --symlink-install --merge-install \
    --packages-select ibrobot_tracing robot_config inference_service action_dispatch
source install/setup.bash
```

下文使用 `ros2 run ibrobot_tracing ibrobot-trace` 调用 CLI。如果 console script 已在 `PATH`
中，也可以简写为 `ibrobot-trace`。

## 2. 开启 Trace

### 2.1 推荐：通过统一 Launch 自动采集

在 `robot.launch.py` 中设置 `enable_tracing:=true`：

```bash
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=42

ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    config_path:=/path/to/so101_single_arm.yaml \
    use_sim:=true \
    sim_platform:=mock \
    control_mode:=model_inference \
    enable_tracing:=true \
    trace_session_name:=so101_trace
```

Launch 会自动完成：

1. 设置 `IB_TRACE_ENABLED=1`，启用 structured instrumentation。
2. 创建并启动 LTTng session。
3. 采集 `ros2:*` UST 事件和 `ib_trace.*` Python 事件。
4. 根据本次 robot YAML、control mode、execution mode 和 scheduler 开关自动生成 Topology。
5. Launch 退出时停止并销毁 LTTng session。

运行机器人并触发需要分析的业务流程，采集完成后按 `Ctrl+C`。默认输出目录为：

```text
~/.ros/tracing/<session-name>/
|-- ibrobot-topology.json
`-- ust/
```

启动日志中会显示实际路径：

```text
[tracing] Writing tracing session to: /home/user/.ros/tracing/so101_trace
[tracing] Wrote topology manifest: /home/user/.ros/tracing/so101_trace/ibrobot-topology.json
```

显式指定的 session 名称或目录已存在时，Launch 会拒绝覆盖。默认 session 名冲突时会自动附加
时间戳。

### 2.2 Action 执行采样

legacy 和 scheduled 控制循环中的每个 action 都埋点会显著增加 trace 体积。默认只完整记录每个
request 的首个非 hold action：

```text
IB_TRACE_ACTION_STEPS=first
```

如需记录全部 action：

```bash
export IB_TRACE_ACTION_STEPS=all
```

首动作必采、后续每 10 步采一次：

```bash
export IB_TRACE_ACTION_STEPS=sample:10
```

应在执行 `ros2 launch` 前设置该变量。

### 2.3 手动 LTTng Session

也可以使用 `scripts/tracing/start_trace.sh` 和 `stop_trace.sh`。手动启动时必须显式启用
structured instrumentation：

```bash
export IB_TRACE_ENABLED=1
bash scripts/tracing/start_trace.sh my_trace

# 启动并操作机器人

bash scripts/tracing/stop_trace.sh my_trace
```

手动 LTTng session 不知道当前 robot YAML，因此不会自动生成 Topology。可以补充生成：

```bash
source .shrc_local
ros2 run robot_config ibrobot-trace-topology \
    /path/to/so101_single_arm.yaml \
    --control-mode model_inference \
    > ~/.ros/tracing/my_trace/ibrobot-topology.json
```

robot YAML 由 `robot_config` 的规范 loader 解析，包括 `base_config` 继承；Core 只接收已展开的
配置 mapping 或 topology manifest，不另行解释机器人配置文件。

## 3. 自定义埋点

### 3.1 推荐初始化方式

每个模块创建并复用一个 emitter：

```python
from ibrobot_tracing import get_trace_emitter

trace = get_trace_emitter(
    "ib_trace.my_module",
    component_id="my_component",
)
```

### 3.1.1 Logger 名与 component_id

TraceEmitter 同时使用两个不同层次的身份：

- logger 名（例如 `ib_trace.dispatch`）是 LTTng Python domain 的传输契约。默认 session 只采集
  `ib_trace.*`；它也是旧 trace 或缺失 `component_id` 记录的分析 fallback。
- `component_id`（例如 `action_dispatcher.decode`）是结构化记录的主要业务身份，用于 Graph 绑定、
  组件聚合、Critical Path 归属和下钻。

同一 emitter 可以通过 `trace_context()`、`span()` 或 `event()` 切换 `component_id`。新结构化记录应始终
携带准确的 `component_id`；若输入为 API 构造的事件模型，分析器仍允许按声明的 logger/provider 映射补全组件身份。

### 3.2 Request Context

`trace_context()` 将同一次端到端请求中的事件关联起来：

```python
with trace.trace_context(request_id, component_id="policy"):
    process_request()
```

- `request_id`：一次业务请求的关联 ID。跨函数、跨组件和跨 ROS 节点应保持一致。
- `component_id`：当前上下文的默认业务组件，可省略。
- 上下文内的 span、event 和 flow 自动继承 `trace_id`、`request_id` 和 component。
- 离开 `with` 后自动恢复之前的上下文。

不属于特定 request 的事件，例如模型加载完成，可以不建立 request context。

跨 ROS 节点时，需要在 message/action 中传递 request ID，并在接收端重新建立 context。`asyncio`
task 通常会继承 `ContextVar`；手动创建的新线程目前需要显式传播 context。

### 3.3 Span：测量一段代码的耗时

`span()` 表示有开始和结束的处理区间，适合测量函数、算法或 I/O 操作：

```python
with trace.trace_context(request_id):
    with trace.span(
        "image_resize",
        component_id="policy.preprocess",
        input_shape=list(image.shape),
    ):
        image = resize_image(image)
```

框架自动产生 `span_begin` 和 `span_end`，并记录：

```text
trace_id / request_id
span_id / parent_span_id
component_id
origin
duration_ns
status / error_type
file / function / line
```

Span 默认 `origin="user"`，会进入 Summary 的 `Custom Span Summary`。框架内置 span 会显式使用
`origin="built-in"`，避免与用户统计重复。

Span 可以嵌套，形成 Call Tree：

```python
with trace.span("preprocess", component_id="policy.preprocess"):
    with trace.span("resize_image"):
        image = resize_image(image)
    with trace.span("normalize_tensor"):
        tensor = normalize(image)
```

对应：

```text
preprocess
|-- resize_image
`-- normalize_tensor
```

如果 span 内抛出异常，`span_end` 会记录 `status=error` 和异常类型，但不会吞掉异常。

同步或异步函数也可以使用装饰器：

```python
@trace.span_decorator("decode_image", component_id="policy.preprocess")
def decode_image(data):
    ...


@trace.span_decorator("fetch_observation", component_id="policy.observation")
async def fetch_observation():
    ...
```

### 3.4 Event：记录一个瞬间发生的事情

`event()` 表示瞬时事件，没有持续时间：

```python
with trace.span("build_model_batch", component_id="policy.preprocess"):
    batch = build_batch()
    trace.event(
        "model_batch_ready",
        origin="user",
        component_id="policy.preprocess",
        keys=sorted(batch),
    )
```

它表达“此时 batch 已准备完成”，而外层 span 表达“构建 batch 花了多久”。在 span 内调用时，
event 会自动携带当前 `span_id`，因此 Timeline 能将瞬时事件关联到所属 span。

Event 适合记录：

- 状态切换，例如 `model_loaded`、`fallback_enabled`。
- 关键阶段完成，例如 `tensor_ready`、`cache_ready`。
- 诊断值，例如 queue size、tensor shape、输入 keys。
- 不需要计算耗时的业务事实。

`origin` 用于区分来源：

```text
origin=user       用户自定义埋点，进入 Custom Marks
origin=built-in   IB-Robot 固定预埋点
```

省略 `origin` 时，`event()` 为兼容内置埋点按 `built-in` 解释，但不会主动把该字段写入 payload。
用户 event 应显式传 `origin="user"`；顶层 `mark()` 是用户 event 的快捷方式：

```python
from ibrobot_tracing import mark

mark("model_loaded", model="act")
```

使用模块 emitter 时优先采用 `trace.event()`，以便复用正确的 logger 和默认 component。

自定义字段必须保持精炼。可以记录 shape、数量、名称和状态，不要记录完整 Tensor 或大型对象：

```python
# 推荐
trace.event("batch_ready", origin="user", keys=sorted(batch), batch_size=1)

# 不推荐：可能产生巨大的字符串和 trace
trace.event("batch_ready", origin="user", batch=batch)
```

### 3.5 Flow：测量组件之间的数据传递

Flow 用一对 send/receive 事件描述数据从一个组件到另一个组件的耗时。

发送端：

```python
with trace.trace_context(request_id, component_id="policy.preprocess"):
    trace.flow_send("preprocess_to_inference", flow_id=request_id)
    publisher.publish(batch_msg)
```

接收端：

```python
with trace.trace_context(request_id, component_id="policy.inference"):
    trace.flow_receive("preprocess_to_inference", flow_id=request_id)
```

分析器使用以下组合配对：

```text
trace_id + edge_id + flow_id
```

- `edge_id`：稳定的业务边名称，通常应与 Topology edge ID 一致。
- `flow_id`：这条边上某次传输的实例 ID。每个并发传输必须可区分。
- `trace_id`：所属端到端 request，由 `trace_context()` 自动提供。

Flow 适合：

- ROS publisher 到 subscription callback。
- Action client request 到 Action server。
- Edge 到 Cloud 的传输。
- Policy result 到 Dispatcher decode。
- Queue refill 到首个 action execute。

Flow 测量的是 send 到 receive 的排队、调度和传输时间，不是发送函数自身耗时。发送或接收侧的
处理耗时应使用 span。

### 3.6 Span、Event 和 Flow 如何配合

```python
with trace.trace_context(request_id):
    with trace.span("build_model_batch", component_id="policy.preprocess"):
        batch = build_batch()
        trace.event(
            "model_batch_ready",
            origin="user",
            component_id="policy.preprocess",
            keys=sorted(batch),
        )

    trace.flow_send("preprocess_to_inference", request_id)
    publisher.publish(batch)
```

接收侧：

```python
with trace.trace_context(request_id, component_id="policy.inference"):
    trace.flow_receive("preprocess_to_inference", request_id)
    with trace.span("model_call"):
        result = model(batch)
```

可简单理解为：

```text
Span   一段代码执行了多久
Event  某个时刻发生了什么
Flow   数据从一个组件到另一个组件用了多久
```

### 3.7 Tracepoint 说明

Span 和 Event 可以声明一段稳定的中文说明。说明属于 tracepoint 定义，不会复制到每个
`TraceEvent` 或 `SpanRecord`：

```python
with trace.span(
    "image_resize",
    component_id="policy.preprocess",
    tracepoint_description="将输入图像缩放到模型要求的尺寸。",
):
    image = resize_image(image)

trace.event(
    "model_batch_ready",
    origin="user",
    component_id="policy.preprocess",
    tracepoint_description="模型输入批次已经组装完成。",
)
```

`tracepoint_description=""` 是默认值，适用于：

```text
TraceEmitter.event()
TraceEmitter.span()
TraceEmitter.async_span()
TraceEmitter.span_decorator()
ibrobot_tracing.mark()
ibrobot_tracing.span()
```

Tracepoint 的稳定身份是：

```text
(kind, component_id, name, origin)
```

其中 `kind` 为 `event` 或 `span`。同一个 emitter 只会为同一身份写入一次私有
`_tracepoint_definition` metadata record，而且仅在说明非空时写入。该 record 不继承
`trace_id`、`request_id` 或 `span_id`；解析器会把它移入 `TraceDataset.definitions`，不会作为普通
Event 出现在 Timeline、统计或查询结果中。关闭 `IB_TRACE_ENABLED` 时仍直接走 no-op 快速路径。

分析结果按以下优先级解析说明：

```text
runtime definition > topology manifest > 内置中文 registry > 空说明
```

同一来源或不同来源出现不同非空说明时，分析器使用稳定排序选择结果并写入确定性的 warning。
最终结果位于 `AnalysisResult.definitions`；原始 runtime 定义位于 `TraceDataset.definitions`；Topology
声明位于 `TraceTopology.definitions`。Topology manifest 继续使用 `schema_version: 2`，并以新增的
顶层 `tracepoints` 数组保存定义；不含该键的旧 v1/v2 manifest 仍可读取。

`ibrobot_tracing` 内置中文 registry 覆盖 action dispatch 和 inference 当前固定 span/event。自定义
说明用于用户埋点或显式覆盖。核心查询 API 提供 `TracepointQuery`、
`QueryService.tracepoints()` 和不受说明文字变化影响的 `stable_tracepoint_id()`。

## 4. 离线分析

以下命令中的 `TRACE_DIR` 是包含 `ust/` 的 trace 目录，例如：

```text
/home/user/.ros/tracing/so101_trace
```

`ibrobot-topology.json` 是可选 metadata，不是读取 events、spans 或 summary 的前提。缺失时按默认
coverage 规则分析，Graph 使用现有 observed topology 投影；不会猜测 robot YAML、scheduler 或部署模式。
自动发现或通过 `--topology` 指定的 manifest 若存在 JSON/UTF-8、schema 或 metadata 字段错误，分析器
忽略整个 manifest，并在 `AnalysisResult.warnings`、完整 JSON export 和 Explorer `warnings` 中记录
`Ignoring topology manifest`。可读的 trace 数据仍保留。该降级只处理 metadata 错误，不吞掉 source
读取、安全校验、文件权限/I/O 或意外程序异常；直接调用 `read_topology_manifest()` 仍会抛出 metadata 错误。

### 4.1 Summary：整体性能概览

```bash
ros2 run ibrobot_tracing ibrobot-trace summary TRACE_DIR
```

主要输出：

```text
Stage                         p50      p95      p99      max     mean    n
Model call                  10.7ms   22.1ms   29.0ms  245.6ms  15.0ms  103
>>>Dispatch->Execute        50.0ms  100.3ms  150.0ms  321.6ms  63.6ms  103
```

- `p50`：中位数，典型请求水平。
- `p95`：95% 请求不超过该值，常用于评估稳定性和尾延迟。
- `p99`：对极端长尾更敏感；样本很少时参考价值有限。
- `max`：最慢样本，常包含首次 CUDA warm-up 或异常调度。
- `mean`：算术平均，容易被少量长尾抬高。
- `n`：该指标可用于 request summary 的样本数；失败、歧义、缺失边界或仍在途的请求不计入该指标。

`>>>Dispatch->Execute` 是当前端到端指标。首次推理通常包含模型和 CUDA warm-up，分析稳态性能时
应结合 `requests` 找出首帧和长尾请求，不要只看 mean/max。

Request summary 不重建业务 retry 状态机，也不将各阶段的最早成功 occurrence 拼成一条成功链。
内置指标只接受唯一、完整且 `status=ok` 的 span；同一 request 下同一指标出现多个 occurrence
（包括 error、重试和未完成 span）时标记 `ambiguous`，不任选一个，也不进入该指标的 `stages` 聚合。
区间指标要求每个端点唯一；重复的边界 event 和 reported metric 同样不任意取首条。其他无歧义指标
仍可用，这不表示整个 request 成功，也不保证所有预期指标均有样本。

对应 span 不存在时，队列/首动作边界也接受唯一的 built-in event：
`action_dispatcher.queue/queue_refill` 和 `action_dispatcher.execute/first_action_execute`。已存在的
span 仍优先，包括失败、未完成或 ambiguous span，不能用 event 掩盖；用作 fallback 的重复 event
同样标记 `ambiguous`。同名 user event 或其他组件的 event 不参与这些内置指标。

Event 路径使用以下公式，所有 ns 差值最后换算为 ms：

```text
queue_refill_ms      = queue_refill.timestamp_ns - dispatch_request.timestamp_ns
refill_to_execute_ms = first_action_execute.publish_end_ns - queue_refill.timestamp_ns
total_ms            = first_action_execute.publish_end_ns - dispatch_request.timestamp_ns
execute_publish_ms  = first_action_execute.publish_ms
```

`publish_end_ns` 是锁内、原 `publish_ms` 计算之后单独捕获的 realtime 时间；额外 trace 时钟读取不计入
业务 `publish_ms`。Legacy 的结果 event timestamp 在 publish 之后，scheduled 的 timestamp 在 publish
之前，因此 Core 不用 event timestamp 或 `flow_receive.timestamp_ns + publish_ms` 猜测结束点。
缺失 `publish_end_ns` 时相关区间指标为 `incomplete`，类型无效或区间为负时为 `invalid`，不回退到
其他 fallback 值；独立有效的 `publish_ms` 仍可统计。分析使用捕获的时间和身份，不依赖锁外 flush 顺序。
这些记录仍是原始 event，不构造 span，也不进入 `span_summary`。

完整 JSON export 新增 `span_summary`，按 `(component_id, name, origin, status)` 分组，提供
`count/minimum/p50/p95/p99/maximum/mean`（ms）。它统计全部完整且耗时有限、非负的 span occurrence，
包括 built-in/user、重试、没有 request ID 的 span，以及 `status=error` 等失败耗时；不按请求挑选。
未完成或无效耗时不进入该数值统计，但所有原始 span 仍可通过 `QueryService.spans(SpanQuery(...))`、
Spans、Timeline 和 Span Profile 查询。现有 `custom_span_summary` 的字段和用途不变。

用户 `origin=user` 的 span 和 event 会分别显示在：

```text
Custom Span Summary
Custom Marks
```

```text
Structured spans: N
Correlated flows: N
```

表示 span 和 flow record 数量，其中可能包含失败或未完成记录，不是成功请求数。`Missing coverage`
表示某些预期指标没有可聚合样本，需要结合 warnings 检查歧义、失败、缺失边界或 execution mode。

### 4.2 Requests：定位慢请求

```bash
ros2 run ibrobot_tracing ibrobot-trace requests TRACE_DIR \
    --sort total_ms \
    --limit 20
```

每行是一个 request 的 JSON：

```json
{
  "request_id": "27ec1fcf",
  "inference_ms": 209.16,
  "inference_ms_source": "structured",
  "total_ms": 280.77,
  "total_ms_source": "structured"
}
```

`*_source` 表示指标来源：

```text
structured   新结构化 span/event
reported     业务结果消息中报告的字段
```

分析器只解析带有效正整数 epoch `timestamp_ns` 的 IBTRACE1 记录，不再读取旧 `[event] key=value`
或从 Babeltrace 的日内时钟猜测日期。旧记录会被警告并跳过；混合文件中保留合法结构化记录。
`structured` 只表示记录格式，不是 GPU 完成时间或更高可信度保证。取得慢请求的 `request_id` 后，继续查看
Timeline、Call Tree、Spans、Flows 或单请求 Graph。

失败或歧义的 structured 指标不会被其他 reported 值覆盖。被排除的指标省略数值键，但保留
`*_source`，并新增 `*_status`，例如：

```json
{
  "request_id": "retry-example",
  "inference_ms_source": "structured",
  "inference_ms_status": "ambiguous",
  "execute_publish_ms_source": "structured",
  "execute_publish_ms_status": "error"
}
```

`*_status` 只在排除指标时出现：`ambiguous` 表示多个 occurrence；`error`、`cancelled` 等保留原失败
状态；`incomplete`/`invalid` 表示边界未完成或耗时无效。一个区间的两个端点有不同非 ok 状态时，以
逗号连接并稳定排序，例如 `ambiguous,error`。缺失该字段不是整个请求成功的证明。
reported 字段为非有限、负数或非数值时标为 invalid，不混入百分位；后续采样动作不会冒充首次动作。
最多保留 1000 条分析诊断样例，其余显示抑制计数；原始事件和 Span 状态仍保留。
`QueryService.requests()` 原样返回这些附加字段，数值排序和 latency distribution 仍只使用存在的
数值指标。兼容 `legacy-json` 不输出 `*_source`、`*_status` 或新增的 `span_summary`。

分布视图和比较视图的默认指标优先使用实际存在的 `total_ms`，其次 `inference_ms`；比较仅能
选择双方都有的数据。`structured` 的执行发布值可能来自 span duration 或 event publish_ms，
两者都是主机侧观测，不以标签给出设备实测/自报的可信度等级。

`robot_config.tracing_utils` 仅保留旧 import 的薄转发；新代码直接导入 `ibrobot_tracing`。
其存在不代表 robot_config 拥有第二份埋点实现。

嵌入动态 Include/Timer/条件 action 的节点应在实际子动作执行边界应用 tracing 环境 scope；
不要把带显式 `env=` 的节点嵌套后假定其自动继承外层环境，也不要提前执行动态 launch 来探测节点。

### Independent staged 与私有 binding 协议

Global→Pipeline 的 `DispatchPipelineBinding` Flow 使用既有 `operation_id` 配对；公开的
Dispatcher→Global Flow 仍观测原有 action goal 身份。Tracing 不生成或覆写业务 UUID、binding、
generation 或 ledger payload。

`stage_policy=independent` 的两个 worker 分别记录 `policy.visual/visual_stage` 和
`policy.action/action_stage` 主机调用边界。请求身份在各 worker 内从 `ExecutionContext` 读取，
退出时恢复 tracing context；不把可变 span token 写入业务帧或复用快照。
`visual_snapshot_published` / `visual_snapshot_selected` 记录已有 generation/version。
快照可能被多个 Dispatch 复用，因此它们是观测事件，不伪造一对一 Flow。

`action_stage` 包含既有的快照等待，`visual_stage` 包含前缀准备和发布；它们都不是设备耗时。
旧的聚合 `model_call` 和模型两侧一对一 Flow 只用于 sequential 模式。Independent 模式可能没有
`inference_ms` 请求链投影，不能为覆盖率补事件、强制计算或改变业务策略。

### 4.3 Timeline：查看完整时序

```bash
ros2 run ibrobot_tracing ibrobot-trace timeline TRACE_DIR \
    --request-id REQUEST_ID
```

输出列：

```text
Relative      相对该结果集中首个事件的时间
Delta         与上一事件的间隔
Component     component_id 或 logger provider
Event         事件名称及字段
```

适合回答：

- 某个 mark 发生在什么时间？
- 两个 span 之间为什么出现空档？
- 请求在哪个组件等待最久？
- 用户 event 位于哪个内置 span 期间？

按组件过滤：

```bash
ros2 run ibrobot_tracing ibrobot-trace timeline TRACE_DIR \
    --request-id REQUEST_ID \
    --component policy.preprocess
```

### 4.4 Call Tree：查看 Span 父子关系

```bash
ros2 run ibrobot_tracing ibrobot-trace call-tree TRACE_DIR \
    --request-id REQUEST_ID
```

示例：

```text
policy_pipeline [built-in] 27.849ms (policy)
  observation_sampling [built-in] 0.701ms (policy.observation)
  policy_total [built-in] 26.237ms (policy)
    preprocess [built-in] 3.080ms (policy.preprocess)
      image_resize [user] 1.200ms (policy.preprocess)
    model_call [built-in] 23.375ms (policy.inference)
    postprocess [built-in] 0.261ms (policy.postprocess)
```

方括号表示 `origin`，括号表示 `component_id`。如果输出：

```text
No structured spans found.
```

说明 trace 中没有与当前 request/component 匹配且完整配对的 span；纯 Event 记录通常没有
Call Tree。

### 4.5 Span Profile：分析埋点 Wall Time

查看单个 request 的 span 区间图数据：

```bash
ros2 run ibrobot_tracing ibrobot-trace span-profile TRACE_DIR \
    --mode request \
    --request-id REQUEST_ID \
    --format text
```

输出机器可读 projection：

```bash
ros2 run ibrobot_tracing ibrobot-trace span-profile TRACE_DIR \
    --mode request \
    --request-id REQUEST_ID \
    --format json \
    --max-nodes 10000 \
    > request-span-profile.json
```

Span Profile 的 `measurement` 固定为 `instrumented_wall`。数据来源是结构化 `span_begin/span_end`
埋点，只覆盖被选中的已埋点区间，不能代表未埋点工作。所有数值都是经过的 wall interval，不是
线程执行量或 CPU 指标；projection 因此显式返回 `source`、`coverage` 和 `not_cpu=true`。

区间图使用以下时间语义：

- `duration_ns` 存在且可用时，它来自进程 monotonic clock；显示区间为
  `start_ns + duration_ns`。
- `observed_end_ns` 始终保留 trace 中实际观察到的结束时间；缺少 monotonic duration 时，使用
  `observed_end_ns - start_ns`。
- `uncovered_wall_ns` 是父 span wall interval 中未被任何直接子 span 区间覆盖的部分。多个直接子
  span 重叠时先计算区间并集，因此不会从父 span 中重复扣除。
- 非完整或负 duration 不参与权重，但保留节点和计数。

Request projection 保留所有逻辑根，不会根据时间包含关系或 flow 猜测父节点。每个节点包含稳定
`occurrence_id`、原始 `span_id`、父子 occurrence ID、`depth`、重叠 sibling 的 `track`、component、
name、origin、status、时间、diagnostics 和原始 fields。可使用以下过滤项：

```text
--component COMPONENT_ID
--start-ns START_NS
--end-ns END_NS
--origin ORIGIN
--status STATUS
```

跨 request 聚合：

```bash
ros2 run ibrobot_tracing ibrobot-trace span-profile TRACE_DIR \
    --mode aggregate \
    --format json \
    > aggregate-span-profile.json
```

Aggregate projection 先在每个 request 内独立解析父子关系，再把非零 `uncovered_wall_ns` 转为完整
路径权重。路径的每一层由 `(component_id, name, origin)` 标识；只有完整路径相同的节点才合并。
`self_value_ns` 是该完整路径自身的 uncovered wall 权重，`value_ns` 递归等于
`self_value_ns + children.value_ns`。节点同时报告 occurrence、request、error、incomplete、invalid
计数和占全部根权重的 percentage。

`concurrency_factor` 定义为：

```text
全部有效 selected occurrence 的 uncovered wall 权重之和
-----------------------------------------------------------
每个 request 内 selected span wall intervals 并集之和
```

顺序完整嵌套通常为 `1.0`；重叠 span 会使它大于 `1.0`；过滤或未覆盖区间可使它小于 `1.0`。
它只描述已埋点 wall intervals 的重叠/覆盖关系。

可检测的 diagnostics 包括：

```text
orphan
cycle
duplicate_span_id
incomplete
negative_duration
child_outside_parent
overlapping_siblings
filtered_parent
```

`max_nodes` 使用确定性的 breadth-first priority，并始终先保留 parent。发生截断时 projection 显式
返回 `total_nodes`、`returned_nodes`、`truncated=true` 和 `truncation_reason=max_nodes`，不会静默丢弃。

### 4.6 Critical Path：分析请求的关键 Wall 区间

查看一个请求从 dispatch 到首个 action execute 的 wall-time 归属：

```bash
ros2 run ibrobot_tracing ibrobot-trace critical-path TRACE_DIR \
    --request-id REQUEST_ID \
    --format text
```

输出 JSON，并限制组件、窗口和返回段数：

```bash
ros2 run ibrobot_tracing ibrobot-trace critical-path TRACE_DIR \
    --request-id REQUEST_ID \
    --component policy.inference \
    --start-ns START_NS \
    --end-ns END_NS \
    --max-segments 1000 \
    --format json \
    > critical-path.json
```

Critical Path 在首选的 `dispatch_request -> first_action_execute` 边界内生成不重叠、无空洞的 wall
partition。缺少首选端点时会显式回退到所选 record 的时间边界；请求窗口超出自然边界时会被 clamp，
diagnostics 会记录回退或 clamp 原因。零长度边界是合法的，返回空 segments 和零 totals。

每个区间只归属一个 owner，优先级固定为：

1. 有效且时钟可比较的 flow。
2. 当前最深的有效 span；同深度重叠使用确定性 tie-break。
3. 没有活动 record 时返回 `unattributed` gap。

Flow 默认参与归属；`--no-flows` 可只看 span。只有 `status=complete` 的 flow 会参与；存在原始 send/
receive 端点时，还要求相同的 allowlisted wall clock，且跨 host 必须有共享时钟保证。被排除的 span/flow
及原因都在 diagnostics 和 totals 中报告。

API 历史方法名 `deepest_active_wall_partition` 的“deepest”仅指 Span 间的优先级，Flow 仍优先。
包围较多计算的宽 Flow 会遮蔽其重叠 Span 的归属份额；这是一种固定 wall-time 分区策略，不是
证明该 Flow 构成业务因果瓶颈。需要看重叠计算时使用 `--no-flows` 对照，不从数据推断调度依赖。

该 projection 的 `measurement` 固定为 `instrumented_wall`，并显式返回 `not_cpu=true`。它不是 CPU
采样、线程运行时间或调度依赖 DAG，只说明已埋点 wall interval 的确定性归属。`max_segments` 默认
1000、最大 10000；超限时返回按时间排序的确定性前缀，同时 totals、bottlenecks、
`total_segments` 和 omitted metadata 仍描述完整 partition。

### 4.7 Graph：查看业务拓扑和耗时

ROS Node 视图：

```bash
ros2 run ibrobot_tracing ibrobot-trace graph TRACE_DIR \
    --metric p95 \
    --view nodes
```

组件视图：

```bash
ros2 run ibrobot_tracing ibrobot-trace graph TRACE_DIR \
    --metric p95 \
    --view components
```

Tracepoint 视图：

```bash
ros2 run ibrobot_tracing ibrobot-trace graph TRACE_DIR \
    --metric mean \
    --view tracepoints
```

支持的聚合指标：

```text
p50 p95 p99 minimum maximum mean
```

- `nodes`：显示 ROS Node 和根数据源/数据汇，折叠内部模块并聚合节点与跨节点 flow 指标。
- `components`：显示模块和已连接的根端点；隐藏没有规范数据流连接的纯父容器。
- `tracepoints`：显示完整组件层级及观测到的 span/event；非箭头 `Contains` 边表示所属关系。
- 名称后的 `*`：该 operation 从实际 trace 动态发现，而非静态 manifest 声明。
- `processing=...`：该组件或 span 自身的处理耗时。
- `A --edge 1.2ms--> B`：A 到 B 的 flow 耗时。

查看单个请求时不再聚合，显示该请求的实际值：

```bash
ros2 run ibrobot_tracing ibrobot-trace graph TRACE_DIR \
    --request-id REQUEST_ID \
    --view tracepoints
```

没有可用 manifest 时，Graph 从已观测 component/flow 生成投影，`topology_source=observed`；这不是
部署拓扑的声明。`components` 查询仍只列出 manifest 组件，缺少 manifest 时可为空，不影响其他查询。

### 4.8 Events、Spans、Flows、Components 和 Tracepoints

查看原始事件：

```bash
ros2 run ibrobot_tracing ibrobot-trace events TRACE_DIR \
    --request-id REQUEST_ID \
    --event model_batch_ready \
    --limit 100
```

还可使用：

```text
--component COMPONENT_ID
--provider ib_trace.policy
```

查看配对后的 span：

```bash
ros2 run ibrobot_tracing ibrobot-trace spans TRACE_DIR \
    --request-id REQUEST_ID \
    --component policy.preprocess
```

关键字段包括 `span_id`、`parent_span_id`、`duration_ms`、`status`、`origin` 和
`component_id`。

查看配对后的 flow：

```bash
ros2 run ibrobot_tracing ibrobot-trace flows TRACE_DIR \
    --request-id REQUEST_ID
```

关键字段包括 `edge_id`、`flow_id`、`send_ns`、`receive_ns` 和 `duration_ms`。没有结果通常表示
send/receive 未使用相同的 request、edge 和 flow ID，或其中一端未被采集。

列出 Topology component ID：

```bash
ros2 run ibrobot_tracing ibrobot-trace components TRACE_DIR
```

该命令可用于找到 `--component` 和 Explorer `use component` 所需的 ID。

列出解析后的 tracepoint 定义：

```bash
ros2 run ibrobot_tracing ibrobot-trace tracepoints TRACE_DIR \
    --kind span \
    --component policy.preprocess \
    --origin built-in \
    --limit 100
```

还可以使用 `--name NAME` 精确过滤。每行 JSON 包含稳定 `id`、`kind`、`component_id`、`name`、
`origin` 和 `description`。没有 runtime metadata 的旧 trace 仍可从 Topology manifest 或内置
registry 获得已有固定埋点的说明；未知埋点保留空说明。

### 4.9 Export：导出机器可读结果

导出完整 JSON：

```bash
ros2 run ibrobot_tracing ibrobot-trace export TRACE_DIR \
    --format json \
    > analysis.json
```

兼容旧 analyzer 的 JSON：

```bash
ros2 run ibrobot_tracing ibrobot-trace export TRACE_DIR \
    --format legacy-json \
    > legacy-analysis.json
```

导出 speedscope sampled profile：

```bash
ros2 run ibrobot_tracing ibrobot-trace export TRACE_DIR \
    --format speedscope \
    > span-wall.speedscope.json
```

每个 sample 是 aggregate profile 中的一条完整 span 路径，weight 是该路径的
`uncovered_wall_ns`。profile 名称显式标记为 `Instrumented Span Wall Time (not CPU)`。现有
`--format json` 和 `--format legacy-json` 的 schema 与行为保持不变。

旧脚本入口仍可使用：

```bash
python3 scripts/tracing/analyze_trace.py \
    --trace-dir TRACE_DIR \
    --format text
```

### 4.10 Explorer：交互式下钻

```bash
ros2 run ibrobot_tracing ibrobot-trace explore TRACE_DIR
```

常用操作：

```text
summary
requests 20
components
tracepoints
use request REQUEST_ID
timeline
call_tree
spans 50
flows 50
graph p95 tracepoints
use component policy.preprocess
timeline
spans
use all
warnings
quit
```

Explorer 内命令使用 `call_tree`，非交互子命令使用 `call-tree`。`quit`、`exit` 和终端
`Ctrl+D`（EOF）都会退出 Explorer，不会停止机器人或修改 trace。

### 4.11 Trace 对比投影

`TraceComparisonService` 直接比较两个 `AnalysisResult`，输出请求指标差值、共享边界 histogram，以及按完整
`component/name/origin` 路径聚合的 Span 墙钟差异。Core 不维护 baseline registry，也不提供 baseline 注册
CLI；Web UI 从同一受限 trace catalog 中选择另一份 trace 作为基线，并在比较前后校验其 catalog version。

`metrics` 保留两侧指标的并集。单侧缺失时，该侧 `count=0`、相关 delta 为 `null`，但不会清除其他可用
指标。coverage metadata 差异、未选指标缺失、样本数差异和 flame diff 节点截断仅作为 warning/diagnostic，
不会整体阻断可比的 latency metric；截断信息复用 `coverage_warnings`。

`metric` 选择 histogram 指标，并决定指标可用性方面的 `comparable`/`blocking_reasons`。未指定时，
优先从两侧共有的指标按既有优先级选择；所选指标单侧或双侧没有样本时返回不可比较及原因，而非抛出
“指标不可用”错误。即使所选指标缺失，其他可计算的指标和 flame diff 仍保留。`has_regression` 仍汇总
所有可计算指标的阈值判断，不限于 histogram 指标；`comparable=true` 不是完整覆盖或业务成功的保证。

Web 接入注意：request DTO 的字典形状不变，但表格需显示 `*_status`，区分歧义/失败与普通缺失；不能
因 `*_source` 存在就认为数值可用。Summary API/类型若要展示通用耗时统计，需要显式透传新增的
`span_summary`。Comparison projection 字段形状不变，UI 应保留可用 metric，并在 `comparable=true`
时也显示 coverage/count warnings 和 flame 截断诊断。上述展示继续复用现有 QueryService/Projection，
无需另建查询或 retry 关联框架。

## 5. 推荐分析工作流

1. 用 `summary` 判断主要瓶颈位于哪个 Stage，并观察 p50/p95/max 差异。
2. 用 `requests --sort total_ms` 找出慢请求和 warm-up 请求。
3. 用单请求 `graph --view tracepoints` 查看组件 processing 和 flow 等待。
4. 用 `call-tree` 查看内置 span 与用户 span 的包含关系。
5. 用 `span-profile --mode request` 查看区间重叠、uncovered wall 和诊断信息。
6. 用 `span-profile --mode aggregate` 或 speedscope 导出定位跨请求的主要完整路径。
7. 用 `timeline` 定位 span 之间的空档和 event 的准确位置。
8. 用 `spans`、`events`、`flows` 查看原始字段，确认配对 ID 和业务状态。
9. 需要自动化或二次处理时使用 `export --format json`。

## 6. 当前限制

- 自定义 instrumentation 目前只支持 Python。
- 分析是离线模式，尚不支持 LTTng Live。
- 手动创建的新线程需要显式传播 `ContextVar`。
- 跨主机 flow 需要时钟同步才能把 realtime 差值解释为可靠网络时延。
- 当前端到端边界止于首次 action publish，尚未覆盖 Hardware Write 或 Physical Actuation。
- Web UI 与 CLI 复用同一 Query/Projection 模型；Web 服务仅支持 Ubuntu，当前局域网服务尚未提供认证。
- openEuler 保留 `ibrobot_tracing` 的 trace 采集与离线分析能力，但不构建 `ibrobot_tracing_web`；可将
  trace 文件传到 Ubuntu 主机进行浏览器分析。

## 7. IB-Robot Tracing Web UI

Ubuntu 推理主机或其他 Ubuntu 主机可以运行独立的 `ibrobot_tracing_web` 服务；服务本身无需 CUDA，
并在浏览器中提供与 CLI 相同的 Query、Timeline、Call Tree 和 Graph projection。仓库不托管生产
`dist`；Web 不属于默认 setup/build，Ubuntu 用户需先按 `tools/ibrobot_tracing_web/README.md`
显式准备 Node.js 18+ 和 Web 依赖，再构建 UI 和两个 tracing ROS 包：

```bash
source .shrc_local
bash scripts/build_tracing_web.sh
source install/setup.bash

ros2 run ibrobot_tracing_web ibrobot-tracing-web \
    --host 0.0.0.0 \
    --port 8767 \
    --trace-root ~/.ros/tracing \
    --allowed-host 192.168.136.209 \
    --allow-unauthenticated-lan
```

同网段浏览器访问：

```text
http://192.168.136.209:8767
```

生成的 `dist` 保持 Git 忽略；Node.js 只用于构建，服务运行阶段不需要。openEuler trace 传输、
catalog refresh、健康检查和完整浏览器工作流见 `tools/ibrobot_tracing_web/README.md`。

Web 服务只允许读取配置的 trace root，不接受客户端文件路径、上传或 trace 删除。第一版局域网
模式没有认证，必须使用显式 `--allow-unauthenticated-lan`，且不应暴露到公网。详细 API 和部署
说明见 `tools/ibrobot_tracing_web/README.md`。

## 8. Roadmap

- 扩展到 Sensor Capture -> Hardware Write / Physical Actuation。
- 支持 C++ 和原生 UST 自定义埋点。
- 支持 LTTng Live 实时分析。
- 支持跨主机时钟同步和 Edge/Cloud trace 合并。
- 支持手动创建的新线程传播 trace context。
- 增加 Web API 认证和 HTTPS 部署指南。

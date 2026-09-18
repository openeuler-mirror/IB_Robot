# IB-Robot Tracing

面向 `ibrobot_tracing` 离线结果的 Vue 3 工作台。上方以 ELK.js 自动排布业务拓扑，下方从阶段汇总逐步下钻到请求、延迟分布、trace 基线比较、时间线、统一 Span 分析、请求关键墙钟路径以及原始 Event、Span、Flow 和分析警告。界面只提供中文，不包含模拟数据，也不把追踪信息伪装成通用管理后台。

## 本地运行

在 Ubuntu 上构建或开发 Web UI 需要 Node.js 18 或更高版本。仓库不托管 `dist`，每个部署环境必须
先从 Vue/TypeScript 源码生成生产资源。Web 不属于默认 setup/build，服务端位于
`tools/ibrobot_tracing_web`；Python 依赖准备和启动方式见
[服务端 README](../../tools/ibrobot_tracing_web/README.md)。以下 `npm ci` 是用户显式准备步骤，
构建脚本不会自动安装依赖。

```bash
cd web/ibrobot_tracing_ui
npm ci
npm run dev
```

Vite 默认监听 `http://localhost:4174`，并将 `/api` 代理到 `http://127.0.0.1:8000`。可在启动前通过 `IBROBOT_TRACING_API` 修改后端地址：

```bash
IBROBOT_TRACING_API=http://127.0.0.1:9000 npm run dev
```

验证命令：

```bash
npm test
npm run build
```

`dist/`、`node_modules/` 和覆盖率目录均保持忽略，不应提交构建产物。完整生产构建优先从仓库根目录
执行 `source .shrc_local && bash scripts/build_tracing_web.sh`，它会先检查已有依赖，再运行
`npm run build`，并通过统一 `scripts/build.sh` 显式扫描 `src` 和 `tools/ibrobot_tracing_web`，
仅构建 `ibrobot_tracing`、`ibrobot_tracing_web` 两个 ROS 包。

构建后，从仓库根目录运行服务端测试使用新路径（需已有 pytest 和 Web 测试依赖）：

```bash
source .shrc_local
python3 -m pytest tools/ibrobot_tracing_web/test
```

## API 契约

前端使用 `tools/ibrobot_tracing_web` 提供的以下 `/api/v1` 端点。分页响应使用 `{ "items": [...], "total": 0, "offset": 0, "limit": 1000, "next_offset": null }`；来源和警告列表使用各自的轻量响应。

| 方法 | 端点 | 用途 |
|---|---|---|
| `GET` | `/api/v1/capabilities` | 发现包括 `baseline_compare` 在内的服务能力和限制 |
| `GET` | `/api/v1/sources` | 发现可读取的 CTF 或日志追踪源 |
| `POST` | `/api/v1/sources/refresh` | 重新扫描受限来源目录 |
| `POST` | `/api/v1/analysis-jobs` | 以 `{ source_id }` 创建加载任务 |
| `GET` | `/api/v1/analysis-jobs/{job_id}` | 轮询 `queued/running/completed/failed/cancelled` 状态 |
| `GET` | `/api/v1/analyses/{id}/summary` | 元数据、阶段矩阵、自定义 Span/事件汇总与覆盖率 |
| `GET` | `/api/v1/analyses/{id}/requests` | 请求级延迟记录 |
| `GET` | `/api/v1/analyses/{id}/distribution` | 请求延迟直方图、桶内请求和高延迟请求；指标可省略并由后端选择 |
| `GET` | `/api/v1/analyses/{id}/tracepoints` | 分页埋点定义目录；用于检查器说明匹配 |
| `GET` | `/api/v1/analyses/{id}/graph` | 带处理/流转耗时的拓扑投影；`view` 接受 `nodes`、`components`、`tracepoints` |
| `GET` | `/api/v1/analyses/{id}/timeline` | Event、Span、Flow 的泳道投影 |
| `GET` | `/api/v1/analyses/{id}/call-tree` | 已计算父子、自身耗时和异常父链的调用树 |
| `GET` | `/api/v1/analyses/{id}/span-profile` | 单请求时间几何或跨请求聚合的 Span 墙钟剖析；`max_nodes` 默认 10000 |
| `GET` | `/api/v1/analyses/{id}/critical-path` | 单请求不重叠墙钟归因分区；`request_id` 必填，`max_segments` 默认 1000 |
| `GET` | `/api/v1/analyses/{id}/events` | 原始事件；接受 `request_id`、`component_id`、`limit` |
| `GET` | `/api/v1/analyses/{id}/spans` | 配对 Span；接受 `request_id`、`component_id`、`limit` |
| `GET` | `/api/v1/analyses/{id}/flows` | 配对 Flow；接受 `request_id`、`limit` |
| `GET` | `/api/v1/analyses/{id}/warnings` | 分析器警告 |
| `POST` | `/api/v1/comparison-jobs` | 以另一份 trace 的 source ID/version 和当前候选 analysis ID 创建比较任务 |
| `GET` | `/api/v1/comparison-jobs/{job_id}` | 轮询比较任务，不接收 trace 路径 |
| `GET` | `/api/v1/comparisons/{id}` | 读取指标差值、共享分桶和 Span 墙钟差异结果 |
| `DELETE` | `/api/v1/comparisons/{id}` | 删除临时比较结果 |

加载任务响应示例：

```json
{
  "id": "job-01",
  "source_id": "source-01",
  "source_version": "fingerprint-01",
  "status": "running",
  "created_at": "2026-07-15T10:00:00Z",
  "started_at": "2026-07-15T10:00:01Z",
  "finished_at": null,
  "analysis_id": null,
  "error": null,
  "cancel_requested": false,
  "queue_position": null,
  "deduplicated": false
}
```

任务完成时 `status` 为 `completed` 且 `analysis_id` 存在。服务端不提供虚假的阶段百分比，界面只区分排队和执行状态。比较请求固定使用当前已载入的 `candidate_analysis_id`，浏览器没有文件路径输入，也不会在请求或响应中传递基线源路径。

图投影中的节点和边通过 `metric_value_ms` 返回当前指标。选择单个请求时该值表示实际耗时，否则表示 `metric` 指定的聚合值。`tracepoints` 视图中的 `contains` 边是无向结构关系，界面以无箭头的灰色虚线连接组件和埋点；其余数据边保留标签和流向动画。绝对纳秒字段可由后端编码为十进制字符串，时间线使用安全的相对偏移数值。

## 交互

- 拖动中间分隔条调整拓扑高度；键盘聚焦分隔条后可用上下方向键调整。
- `/` 聚焦全局过滤框，`Esc` 关闭检查器或清除过滤内容。
- `Alt+1` 到 `Alt+9` 切换下方前九个可用分析视图。服务支持比较时，延迟分布为 `Alt+3`、基线比较为 `Alt+4`、Span 分析为 `Alt+6`、关键路径为 `Alt+7`；服务未声明 `baseline_compare` 时不显示比较标签，后续快捷键自动前移。
- 表格聚焦后可用 `J/K` 或上下方向键移动，回车检查当前行。
- 标签栏聚焦后可用左右方向键切换。
- 动画暂停按钮和系统“减少动态效果”偏好均会停止边上的流向示踪。
- 当前标签、请求、组件、拓扑指标、拓扑层级、搜索词、分布指标 `distribution_metric`，以及 Span 分析的 `mode=call-tree|request|aggregate` 保存在 URL 中，可直接分享并恢复分析范围。基线比较还保存 `baseline_source`、`statistic`、`relative_threshold`、`absolute_threshold_ms` 和 `comparison_metric`；无效或负数阈值恢复为 10% 与 1 ms。首次打开分布视图时可省略 `distribution_metric`，由后端选择指标后写回状态和 URL。默认省略 `mode=request`。旧 `tab=call-tree` 链接会映射到新的 Span 分析调用树模式。
- 延迟分布柱可悬停、单击或用 Tab 聚焦，并用回车/空格选择。选中桶和高延迟表中的请求可直接打开精确请求范围的时间线、Span 分析或关键路径；重复打开当前请求不会清除请求范围。
- 打开分析时会一次性分页读取完整埋点目录；检查器按记录身份显示“埋点说明”，没有说明时不显示该区域。
- Span 分析在同一入口提供调用树、请求 Span 区间和聚合 Span 剖析，并固定使用根在上的冰柱方向。冰柱图支持 `+`/`-` 缩放、水平滚动或拖动、`F` 适合宽度和 `0` 重置；聚合图双击节点聚焦子树，并通过面包屑返回父级或全部根。
- 基线比较必须点击“执行比较”，参数变化不会隐式启动昂贵任务。只有选定统计值的相对增幅和绝对增幅都严格超过阈值才标记回归；指标表始终同时显示文本状态与颜色。选择指标后再次执行，服务端会用同一组边界生成基线/候选叠加直方图。
- Instrumented Span Wall Diff 按完整 `component/name/origin` 父链接重建路径，以 `max(baseline, candidate)` 包含墙钟权重分区，并固定使用冰柱方向。它支持全局搜索和检查器；双击或 `Shift+Enter` 聚焦子树。该图包含等待、I/O 与异步暂停，明确不是 CPU profile。

## 实现边界

- 本目录只包含前端；对应服务端位于 `tools/ibrobot_tracing_web`。服务未启动时会显示明确的载入错误，不会回退到虚构数据。
- 基线比较直接复用受限 trace source catalog；当前候选 source 会从可选基线中排除。目录中不足两份 trace 时只显示提示，不提供浏览器路径输入。
- 比较轮询绑定分析 ID 和本次请求 generation。再次比较或切换分析会中止旧请求并清空 job、result、loading 与 error，迟到响应不会覆盖新状态。
- 原始 Event、Span、Flow 和请求表当前读取第一页，每页上限 1000 条；应使用请求和组件范围缩小结果。时间线和调用树使用后端完整投影，不受原始表单页限制。
- 延迟分布按标签首次打开时惰性读取；切换指标会中止旧请求，并按分析和指标缓存结果。桶内请求标识受后端 `bucket_request_limit` 限制，截断状态会在选中桶上明确显示。
- 埋点定义目录会跟随分析切换清空并重新完整加载，不随标签、请求或组件过滤重复请求。
- Span 冰柱图固定采用 **Instrumented Span Wall Time** 语义：数值包含等待、I/O 和异步暂停，不能解释为 CPU 使用率。单请求图按起始偏移和持续时间绘制，聚合图按包含墙钟权重分区；深色底条表示自身或未被子 Span 覆盖的墙钟权重。
- 单请求模式未选择请求时只显示明确提示，不调用要求 `request_id` 的 API。请求、组件或模式变化会中止旧请求；服务端截断和结构诊断会在图上方汇总。
- 关键路径同样要求明确请求，并按完整请求边界比例展示互不重叠的 Flow wait、最深 active Span 和未归因 Gap。它表示包含等待、I/O 与异步暂停的 Instrumented Wall Time，不是 CPU 时间或跨主机调度 DAG；截断时图表单独标出精确的省略尾部。
- 时间线已经展示 Event 点、Span 区间和 Flow 区间。调用树已展示总耗时，并在悬停时给出自身耗时。
- ELK 布局引擎在首次显示拓扑时按需加载，生产块体积较大但不会进入初始应用包。

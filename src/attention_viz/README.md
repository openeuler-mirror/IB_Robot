# attention_viz - ACT 注意力可视化工具

`attention_viz` 将 ACT 模型推理时产生的 cross-attention 权重转换为热力图，并叠加到相机图像上。它支持将热力图保存为文件、通过 ROS 话题发布热力图图像，也支持在推理首轮交互绘制 attention ignore mask。

## 概述

`attention_viz` 是一个面向算法调试的注意力可视化工具。它通过提取 ACT 模型在推理时产生的 cross-attention 权重，将其以热力图形式叠加到相机图像上，从而直观呈现模型在输出动作时关注的图像区域。需要验证遮挡区域影响时，也可在首个 action chunk 前通过 GUI 绘制忽略区域，转换为 ACT transformer mask 后参与本轮 episode 的后续推理。

该工具主要用于：

- **注意力分布分析**：验证模型是否聚焦于任务关键区域。
- **失败案例归因**：判断动作异常是否可能由注意力偏移引起，例如模型关注背景而非目标物体。
- **相机配置验证**：确认模型是否正确使用了 top、wrist、front 等相机输入。
- **训练效果对比**：比较不同 checkpoint 或不同训练策略下模型的关注模式差异。
- **交互式遮挡验证**：手动画出需要忽略的图像区域，验证 ACT 对特定视觉 token 的依赖。

> **注意**：本工具定位为离线/在线调试辅助工具，不适用于生产环境常驻运行，也不作为 Web 可视化平台或 RViz 轨迹可视化工具使用。

## 系统架构

```text
LeRobotPolicyNode                    AttentionVisualizationNode
┌─────────────────┐                  ┌─────────────────────┐
│ 推理 + Hook 捕获 │  /attention/     │  订阅注意力权重      │
│  attn_weights    │──weights──→      │  订阅相机图像        │
│                  │                  │  生成 JET 热力图     │
└─────────────────┘                  │  发布叠加结果        │
                                     └─────────────────────┘
                                              │
                                              ▼
                                    /visualization/heatmap/{camera_key}
```

运行时数据流如下：

```text
相机图像
  -> observation.images.*
  -> lerobot_policy_node 推理
  -> Hook 提取 ACT decoder cross-attention
  -> /attention/weights
  -> attention_visualization_node
  -> /visualization/heatmap/<camera_key>
     或保存到磁盘
```

模块职责：

- `inference_service`：在 monolithic 推理模式下安装 attention hook，发布 `AttentionWeights` 消息；启用 interactive masking 时，将用户绘制的像素 mask 转换为 ACT transformer mask，并在 episode 内持续应用。详见 [推理服务文档](../inference_service/README.md#注意力可视化)。
- `attention_viz`：订阅注意力权重和相机图像，生成热力图文件或热力图话题，并提供交互式 mask 绘制工具。
- `robot_config`：在机器人配置中决定是否启用注意力权重发布，并可自动拉起可视化节点。

主要实现位置：

- `src/attention_viz/attention_viz/attention_hook.py`
- `src/attention_viz/attention_viz/visualization_core.py`
- `src/attention_viz/attention_viz/attention_visualization_node.py`
- `src/inference_service/inference_service/lerobot_policy_node.py`
- `src/robot_config/config/robots/so101_single_arm.yaml`
- `src/robot_config/robot_config/launch_builders/execution.py`

## 快速上手

下面流程用于实验机验证。执行前请确认工作区已完成构建，当前终端能够找到 `robot_config`、`inference_service` 和 `attention_viz` 等 ROS 2 包。推理链路和可视化节点需要使用相同的 `ROS_DOMAIN_ID`，具体取值以现场环境为准。

### 前置条件

- 已准备 ACT 模型 checkpoint。
- 推理节点运行在 `execution_mode:=monolithic` 模式。
- 机器人配置中的相机输入与模型 `observation.images.*` key 对齐。
- 推理节点和可视化节点所在终端使用相同的 `ROS_DOMAIN_ID`。

### 步骤 1：配置节点化可视化

编辑 `src/robot_config/config/robots/so101_single_arm.yaml`，在 `control_modes.model_inference.inference` 中设置：

```yaml
control_modes:
  model_inference:
    inference:
      attention_viz_topic: /attention/weights
      attention_viz:
        enabled: true
        mode: file
        save_dir: /tmp/attention_viz
        update_frequency: 10.0
        interactive_masking: false
        mask_save_dir: /tmp/attention_masks
```

`attention_viz.enabled` 会自动启用推理侧 attention 发布，并启动 `attention_visualization` 节点。`mode: file` 会把热力图保存到 `save_dir`，同时发布 `/visualization/heatmap/<camera_key>` 图像话题。

### 步骤 2：启动推理链路

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=model_inference \
  execution_mode:=monolithic \
  use_sim:=true
```

预期日志：

| 日志内容 | 含义 |
|---------|------|
| `Attention visualization hook installed` | Hook 安装成功，推理节点会发布注意力权重 |
| `AttentionVisualizationNode started` | 可视化节点已启动 |
| `DispatchInfer Action Server ready` | 推理 Action 服务已就绪 |
| `lerobot_policy node ready (monolithic)` | 推理节点启动完成 |

若缺少 `Attention visualization hook installed`，请参见 [故障排除](#故障排除)。

注意力权重只会在推理节点完成一次有效推理后发布。启动节点后，需要让动作分发或评估流程触发一次 `DispatchInfer` 请求，随后可视化节点才能收到 `/attention/weights` 消息。

### 步骤 3：观察可视化输出

预期日志：

| 阶段 | 日志内容 |
|------|---------|
| 启动 | `Waiting for AttentionWeights on /attention/weights` |
| 收到权重 | `First AttentionWeights received` |
| 订阅相机 | `Subscribed to camera: ...` |
| 生成热力图 | `First heatmap saved: /tmp/attention_viz/.../step_XXXX_attn.jpg` |

注意：可视化节点只有在推理链路完成至少一次有效推理、并且收到匹配相机图像后，才会生成热力图。

### 步骤 4：验证运行状态

```bash
ros2 node list | grep -E 'act_inference_node|attention_visualization'
ros2 topic info /attention/weights
ros2 topic echo --once /attention/weights
ros2 topic list | grep /visualization/heatmap
find /tmp/attention_viz -name '*_attn.jpg' | head
```

预期结果：

- `/act_inference_node` 和 `/attention_visualization` 均在线。
- `/attention/weights` 的 `Publisher count` 为 1，并能 echo 到 `AttentionWeights` 消息。
- `/visualization/heatmap/<camera_key>` 在收到相机图像后出现。
- `/tmp/attention_viz` 下生成 `step_XXXX_attn.jpg`。

## 实时交互模式

`mode: interactive` 会打开 Matplotlib 实时窗口。窗口中显示各路相机画面，勾选 action query 后会把对应注意力热力图实时叠加到画面上。

```yaml
control_modes:
  model_inference:
    inference:
      attention_viz:
        enabled: true
        mode: interactive
        headless: false
```

实时交互模式需要图形显示环境。若运行环境没有图形显示，节点会退回文件模式，继续保存并发布热力图。

## 交互式 Mask

`interactive_masking` 用于复现原 `lerobot_ros2` 中的交互式 attention mask 能力。开启后，推理节点会在每个 episode 的首个 action chunk 前弹出 Matplotlib 窗口，用户可用鼠标拖拽标记需要忽略的图像区域。保存后，系统会将像素 mask 缩放到 ACT 特征图尺寸，并转换为 transformer attention mask。

```yaml
control_modes:
  model_inference:
    inference:
      attention_viz:
        enabled: true
        mode: interactive
        interactive_masking: true
        mask_save_dir: /tmp/attention_masks
```

窗口快捷键：

| 操作 | 含义 |
|------|------|
| 鼠标拖拽 | 标记忽略区域 |
| `s` / `Enter` | 保存并应用 mask |
| `r` | 清空当前绘制 |
| `q` / `Esc` | 跳过本轮 mask |

mask 在当前 episode 内持续生效；`/action_dispatcher/reset` 或 `/act_inference_node/reset_policy_state` 会清理已绘制 mask，下一个 episode 会重新提示。

## 单独启动可视化节点

如果推理节点已经单独启动，并且 `publish_attention` 已开启，可以独立启动可视化节点：

```bash
ros2 launch attention_viz attention_viz.launch.py \
  mode:=file \
  save_dir:=/tmp/attention_viz
```

## 相机话题配置

默认情况下，`attention_viz` 会根据模型 key 推导相机话题。例如 `observation.images.top` 默认对应 `/camera/top/image_raw`。

如果实验机相机话题不同，请通过参数文件覆写：

```yaml
/**:
  ros__parameters:
    camera_topics:
      - "observation.images.top:=/camera/color/image_raw"
      - "observation.images.wrist:=/camera/wrist/image_raw"
```

使用参数文件启动：

```bash
ros2 run attention_viz attention_visualization_node \
  --ros-args \
  --params-file /path/to/attention_viz_cameras.yaml \
  -p visualization_mode:=file \
  -p save_dir:=/tmp/attention_viz
```

当可视化节点收到 attention 但收不到相机图像时，会输出：

```text
Attention received, but required camera images are not cached yet.
Expected image topics: ...
```

请按日志中的 `Expected image topics` 检查对应图像话题是否存在并持续发布 `sensor_msgs/Image`。

## 参数说明

### 推理节点参数：节点化可视化（推荐）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `publish_attention` | bool | `false` | 启用注意力权重发布至 ROS 话题 |
| `attention_viz_topic` | string | `/attention/weights` | 注意力权重发布话题名 |
| `attention_viz.enabled` | bool | `false` | 自动启用 attention 发布并拉起可视化节点 |
| `attention_viz.mode` | string | `file` | `file`、`realtime` 或 `interactive` |
| `attention_viz.save_dir` | string | `attention_visualizations` | 文件模式保存目录 |
| `attention_viz.update_frequency` | float | `10.0` | 可视化节点最大更新频率，单位 Hz |
| `attention_viz.headless` | bool | `false` | 禁用 GUI，强制使用文件模式 |
| `attention_viz.interactive_masking` | bool | `false` | 首个 action chunk 前启用交互式 attention mask |
| `attention_viz.mask_save_dir` | string | `gui_interactions` | mask 交互截图保存目录 |

### 可视化节点参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `attention_topic` | string | `/attention/weights` | 订阅的注意力权重话题，默认与推理节点 `attention_viz_topic` 一致 |
| `visualization_mode` | string | `file` | `file`、`realtime` 或 `interactive`；`interactive` 是 `realtime` 的兼容别名 |
| `save_dir` | string | `attention_visualizations` | 热力图保存目录 |
| `queries_to_visualize` | int[] | `[0,20,40,60,80]` | 要可视化的 action query 索引 |
| `layer_idx` | int | `-1` | 注意力层索引，`-1` 表示最后一层 |
| `batch_idx` | int | `0` | 批次索引 |
| `average_heads` | bool | `true` | 是否平均所有注意力头 |
| `blend_alpha` | float | `0.4` | 热力图叠加透明度 |
| `update_frequency` | float | `10.0` | 最大更新频率，单位 Hz |
| `headless` | bool | `false` | 禁用 GUI，强制使用文件模式输出 |
| `heatmap_topic_prefix` | string | `/visualization/heatmap` | 热力图输出话题前缀 |
| `camera_topics` | string[] | `[]` | 相机话题覆写，格式为 `observation.images.top:=/camera/top/image_raw` |

## 配置文件

`config/visualization_params.yaml`：

```yaml
/**:
  ros__parameters:
    visualization_mode: 'file'
    attention_topic: '/attention/weights'
    save_dir: 'attention_visualizations'
    heatmap_topic_prefix: '/visualization/heatmap'
    queries_to_visualize: [0, 20, 40, 60, 80]
    layer_idx: -1
    batch_idx: 0
    average_heads: true
    blend_alpha: 0.4
    update_frequency: 10.0
```

带参数启动：

```bash
ros2 run attention_viz attention_visualization_node \
  --ros-args \
  --params-file /path/to/visualization_params.yaml
```

## 话题

| 话题 | 消息类型 | 方向 | 说明 |
|------|---------|------|------|
| `/attention/weights` | `ibrobot_msgs/msg/AttentionWeights` | 推理到可视化 | 扁平化注意力权重与元数据 |
| `/camera/{name}/image_raw` | `sensor_msgs/Image` | 外部到可视化 | 原始相机图像；默认按 `observation.images.<name>` 推导，也可由消息或参数覆写 |
| `/visualization/heatmap/{key}` | `sensor_msgs/Image` | 可视化到外部 | 热力图叠加结果，编码为 BGR8；前缀可由 `heatmap_topic_prefix` 覆写 |

## 性能影响

### 推理节点侧

- `publish_attention:=false`（默认）：不安装 hook，无额外开销。
- `publish_attention:=true`：推理节点会强制 ACT decoder cross-attention 返回权重，推理时间和显存占用会增加。
- `interactive_masking:=true`：首个 action chunk 前会等待 GUI 交互，保存后会在 ACT encoder / decoder attention 中屏蔽被标记的视觉 token。
- 注意力权重发布仅支持 `execution_mode:=monolithic`。

### 可视化节点侧

- 可视化节点以独立进程运行，不会阻塞推理节点的前处理、推理和后处理流程。
- 每次推理发布一次完整注意力权重消息，消息大小约为 `layers * batch * heads * query_len * key_len * 4` 字节。
- 文件模式下，每步会为每个 query 和每路相机保存一张 JPEG，磁盘写入量取决于 `queries_to_visualize` 和相机数量。
- 建议仅在调试、评估和失败分析时启用，不建议默认常开。

## 典型应用场景

建议在以下场景启用：

- 模型可以运行，但动作不稳定，需要判断是否关注了错误区域。
- 更换相机位置或相机话题后，需要确认模型仍在使用正确视角。
- 新数据集训练后，需要快速判断模型是否关注任务关键区域。
- 对比不同 checkpoint 的注意力分布差异。

不建议用于：

- Web 可视化监控平台。
- MoveIt 或 RViz 轨迹可视化替代品。
- 相机驱动调试工具。
- 面向最终用户的交互界面。

## 依赖与环境要求

- 仅构建 `attention_viz` 包时，需要标准 ROS 2 Humble 构建环境。
- 运行可视化节点时，需要 `cv_bridge`、`opencv`、`matplotlib` 和 `numpy`。
- 运行推理链路并发布 attention 时，还需要 `torch`、`torchvision`、`lerobot` 等模型推理依赖。

当前限制：

- 仅支持 ACT 注意力可视化链路，不是所有策略类型的通用 attention 框架。
- 节点化注意力权重发布仅支持 `monolithic` 推理模式。

## Policy Reset

推理节点提供状态重置服务，用于在 episode 切换时清理 policy 内部缓存和 attention hook 缓存：

```bash
ros2 service call /act_inference_node/reset_policy_state std_srvs/srv/Trigger "{}"
```

`/action_dispatcher/reset` 会先重置动作分发队列，再 best-effort 调用上述推理侧 reset 服务。`record_cli` 仅在模型推理录制场景下进入该链路：

```text
record_cli (model_inference) -> /action_dispatcher/reset -> /act_inference_node/reset_policy_state
```

详见 [推理服务文档](../inference_service/README.md#重置推理运行时状态) 和 [动作分发文档](../action_dispatch/README.md#话题和服务)。

## 故障排除

### 问题：`/attention/weights` 话题不存在

排查步骤：

1. 检查推理节点日志中是否出现 `Attention visualization hook installed`。
2. 若无此日志，确认 `execution_mode` 是否为 `monolithic`。
3. 确认 `publish_attention` 是否为 `true`。
4. 确认当前模型是否为 ACT，并且 policy 中存在 decoder cross-attention。

### 问题：收到注意力权重但未生成热力图

现象：`/attention/weights` 有消息，但没有热力图文件或热力图话题。

排查步骤：

1. 检查 `attention_viz` 日志中的 `Expected image topics`。
2. 确认日志中列出的相机话题存在并持续发布图像：

   ```bash
   ros2 topic hz /camera/top/image_raw
   ```

3. 若实际相机话题名与默认推导不同，通过 `camera_topics` 参数覆写。
4. 若文件已生成但话题不可见，确认 `attention_viz` 节点仍在线：

   ```bash
   ros2 node list | grep attention_visualization
   ```

### 问题：episode 切换后仍沿用旧动作或旧注意力状态

排查步骤：

1. 手动调用动作分发 reset 服务：

   ```bash
   ros2 service call /action_dispatcher/reset std_srvs/srv/Empty "{}"
   ```

2. 检查 `action_dispatcher` 日志中是否出现 policy reset 相关日志。
3. 如未启动 `action_dispatcher`，可直接调用推理侧 reset 服务：

   ```bash
   ros2 service call /act_inference_node/reset_policy_state std_srvs/srv/Trigger "{}"
   ```

## 文件清单

```text
src/attention_viz/
├── CMakeLists.txt
├── package.xml
├── attention_viz/
│   ├── __init__.py
│   ├── attention_hook.py                  # PyTorch hook，提取注意力权重
│   ├── attention_masking.py               # 交互式 mask 绘制与 token mask 转换
│   ├── attention_visualization_node.py    # ROS 可视化节点
│   ├── visualization_core.py              # 热力图渲染核心
│   └── utils.py                           # 消息转换工具
├── launch/
│   └── attention_viz.launch.py
├── config/
│   └── visualization_params.yaml
└── test/
    └── test_attention_utils.py
```

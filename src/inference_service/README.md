# Inference Service (推理服务)

[English](./README.en.md) | 简体中文

`inference_service` 是 IB-Robot 具身智能系统的核心 AI 执行引擎包。它为物理机器人的端到端机器学习策略（如 ACT, pi0 等）提供了一个标准化的运行框架，重点优化了高频控制下的**时间轴对齐**与**零拷贝延迟**。

## 架构：组合优于继承 (Composition over Inheritance)

整个推理管线被极致解耦为三个没有任何 ROS 依赖的纯 Python 核心组件（位于 `inference_service.core` 目录下）：
1. **TensorPreprocessor (前处理)**：负责将 ROS 2 订阅到的多模态传感器裸数据（相机图像、关节状态等）裁剪、归一化为标准的 PyTorch Tensors。
2. **PureInferenceEngine (纯推理引擎)**：一个绝对无状态、无 ROS 依赖的 GPU 算法执行引擎。
3. **TensorPostprocessor (后处理)**：将网络输出的动作 Tensors 反归一化为机器人底层可以直接执行的物理控制指令。

通过将纯粹的数学运算与 ROS 2 的通信层剥离，本功能包得以通过一个简单的 YAML 参数，支持两种完全不同的工业级部署模式。

---

## 🚀 部署模式 (Execution Modes)

### 模式 A：单机零拷贝模式 (Monolithic)
**适用场景**：机器人本体搭载了诸如 RTX 4060 等高性能板载 GPU。

在此模式下，端侧的 `lerobot_policy_node.py` 会实例化一个 `InferenceCoordinator`，在内部将前处理、推理、后处理三者串联。
* **数据流向**：传感器数据完全留在单个进程的内存/显存（RAM/VRAM）中，张量全程通过指针引用传递。
* **性能优势**：实现绝对意义上的**最低延迟**，彻底消除了跨进程的序列化/反序列化（Serialization）开销。
* **YAML 配置**：`execution_mode: "monolithic"`

### 模式 B：端-边-云分布式协同模式 (Device-Edge/Cloud Distributed)
**适用场景**：轻量级机器人（端侧）仅搭载了算力薄弱的 CPU（如树莓派、工控机），而庞大的多模态大模型运行在同一局域网下的高性能计算节点（边端）或云端服务器上。

为了保持对上层 `action_dispatch`（拉取式分发器）的兼容，同时**防止高帧率的视频流塞满局域网带宽**，端侧节点在此时会化身为一个**异步代理 (Asynchronous Proxy)**。
1. **端侧 (`lerobot_policy_node.py`)**：收到 Action Goal 后，按需抓取本地相机画面，在 CPU 上执行**前处理**，随后将轻量化的张量打包发往 `/preprocessed/batch` 话题。它会利用 `threading.Event` 将当前协程挂起，不占用额外资源。
2. **边/云端 (`pure_inference_node.py`)**：这是一个独立的节点，它订阅张量，死磕 GPU 算力进行推理，并将结果立刻发回 `/inference/action`。
3. **端侧**：监听到边/云端回传的结果，被瞬间唤醒，执行**后处理**闭环，最后将最终的物理指令提交给分发器。

* **性能优势**：完美的“算力卸载（Compute Offloading）”。端侧只有在需要推理的那一刻（例如 20Hz 下每 50ms 一次）才发送关键帧，极大地节约了网络带宽，且对上层应用完全透明。
* **YAML 配置**：`execution_mode: "distributed"`

```
端侧机器 (Robot / Sim)                    算力机器 (GPU Server)
┌──────────────────────────────┐          ┌──────────────────────────┐
│  action_dispatcher_node      │          │                          │
│       ↓                      │          │  pure_inference_node     │
│  lerobot_policy_node (Proxy) │          │  ├─ Subscribe            │
│  ├─ TensorPreprocessor (CPU) │          │  │  /preprocessed/batch  │
│  ├─ threading.Event          │          │  ├─ PureInferenceEngine  │
│  └─ TensorPostprocessor(CPU) │          │  │  (GPU)                │
│       ↓ Pub        ↑ Sub     │          │  └─ Publish              │
│  /preprocessed  /inference   │          │     /inference/action    │
│  /batch         /action      │          │                          │
└──────────┬──────────┬────────┘          └───────┬──────────┬───────┘
           │          │      LAN (同一 ROS_DOMAIN_ID)        │          │
           └──────────┴──────────────────────────┴──────────┘
```

---

## ⚙️ 配置与启动 (Configuration & Usage)

两种模式的切换极其丝滑，您完全不需要修改 Launch 启动文件，一切均由 `robot_config` 包中的 YAML 配置文件决定。

```yaml
# 位于: src/robot_config/config/robots/your_robot.yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      execution_mode: "distributed"  # 切换为 "monolithic" 即可秒切单机版
      model: so101_act
```

### 注意力可视化

推理节点支持节点化注意力可视化：

| 方式 | 适用模式 | 说明 | 详细文档 |
|------|---------|------|---------|
| 节点化可视化（推荐） | Monolithic | 通过 ROS 话题发布注意力权重，由独立 `attention_viz` 节点渲染热力图 | [attention_viz 文档](../attention_viz/README.md) |

#### 节点化可视化参数

```yaml
control_modes:
  model_inference:
    inference:
      attention_viz_topic: /attention/weights
      attention_viz:
        enabled: true
        mode: file
        save_dir: /tmp/attention_viz
        interactive_masking: false
        mask_save_dir: /tmp/attention_masks
```

`attention_viz.enabled` 会自动启用 `publish_attention` 并拉起独立可视化节点。`attention_viz.interactive_masking` 会在首个 action chunk 前弹出 mask 绘制窗口，将用户标记区域转换为 ACT transformer mask。可视化和交互式 mask 只在 `execution_mode: monolithic` 下生效；分布式模式下会在启动时输出警告。

### 重置推理运行时状态

推理节点提供显式状态重置服务，用于在 episode 切换时清理 policy 内部状态和 attention hook 缓存：

```bash
ros2 service call /act_inference_node/reset_policy_state std_srvs/srv/Trigger "{}"
```

重置范围：

| 组件 | 重置内容 |
|------|---------|
| Policy | 内部 queue、action chunk 和 episode 状态 |
| Attention Hook | 已缓存的注意力权重和交互式 attention mask |

`action_dispatcher` 在收到 reset 请求时会触发推理侧重置；`record_cli` 仅在 `control_mode:=model_inference` 或显式 `reset_before_episode:=true` 时调用该 reset 链路：

```text
record_cli (model_inference) -> /action_dispatcher/reset -> /act_inference_node/reset_policy_state
```

其中，`action_dispatcher` 的 reset 服务会先清理自身动作队列，再 best-effort 调用推理侧 reset 服务。若推理节点不在线，dispatcher reset 仍会完成自身状态清理。

### 启动命令

#### 场景一：跨机器分布式部署（推荐生产用法）

两台机器必须设置**相同的 `ROS_DOMAIN_ID`** 且在同一局域网内。

**步骤 1 — 在机器人本体（端侧 Device）上**：

端侧只启动 Edge 代理节点（前/后处理），不加载 GPU 模型：
```bash
export ROS_DOMAIN_ID=<你的域ID>
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true   # 仿真模式；真机去掉此参数
```

**步骤 2 — 在算力服务器（边端/云端 Edge/Cloud）上**：

```bash
export ROS_DOMAIN_ID=<与你的端侧一致的域ID>
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/models/pretrained_model \
    device:=cuda
```

如果 Cloud 节点运行在端侧开发板（openEuler / OpenHarmony）上的 Ascend NPU，可直接切换为：

```bash
export ROS_DOMAIN_ID=<与你的端侧一致的域ID>
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/models/pretrained_model \
    device:=npu
```

如果模型已经通过 ATC/SVP 工具链导出为 Ascend OM，可直接让
`inference_service` 承载从原 LeRobot 补丁迁移过来的 OM wrapper：

```bash
# 通用 Ascend ACL .om 推理后端
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/pretrained_model \
    device:=ascend_om

# SD3403 worker 二进制协议后端
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/pretrained_model \
    device:=ascend_om_3403
```

`device:=ascend_om` 会从 `policy_path/config.om.json` 的 `artifacts.policy` 解析
单 OM 模型。`device:=ascend_om_3403` 需要 `artifacts.policy` 和 `artifacts.worker`，
并要求 `execution` 为 `["policy", "worker"]`。两种模式的前/后处理、ROS 话题与
分布式通信仍沿用 `inference_service` 的现有管线。

如果 Cloud 节点运行在 RK3588 / OpenHarmony 板端并使用 RKNN Lite，可直接切换为：

```bash
export ROS_DOMAIN_ID=<与你的端侧一致的域ID>
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/models/pretrained_model \
    device:=rknn
```

`device:=rknn` 会继续复用 `policy_path/config.json` 中的 LeRobot 配置用于前后处理，
同时要求实际的 RKNN 模型文件随 policy 一起纳管到 `policy_path` 目录内，
默认文件名为 `model.rknn`。

### 编译模型后端的 Wrapper 设计

无论使用 PyTorch、Ascend OM 还是 RKNN，`inference_service` 对上层暴露的统一入口始终是
`PureInferenceEngine`。它内部通过 `PolicyWrapper` 抽象屏蔽不同运行时后端的差异，统一接口为：

1. `load(path, device)`：加载模型与运行时资源。
2. `infer(batch)`：对一批已经完成前处理的 Tensor 执行推理。
3. `get_chunk_size()`：返回动作 chunk 大小。
4. `policy_type`：返回当前策略类型标识（如 `"act"`、`"pi05"`）。
5. `backend_type`：返回运行时后端标识（如 `"cuda"`、`"ascend_om"`、`"rknn"`）。
6. `uses_action_chunking`：当前策略是否使用 action chunk 输出。

这意味着：

1. 前处理和后处理组件不需要感知底层是 PyTorch、ACL 还是 RKNN Lite。
2. 分布式与单机模式只改变部署位置，不改变 wrapper 设计。
3. 真正的后端差异被收敛到 `PolicyWrapper` 的实现中。

#### PyTorch 路径：LeRobotPolicyWrapper

常规 `device:=cuda/cpu/npu` 路径使用 `LeRobotPolicyWrapper`。

它的工作方式最接近上游 LeRobot：

1. 从 `policy_path/config.json` 识别 `type`。
2. 通过 `lerobot.policies.factory.get_policy_class()` 获取对应策略类。
3. 调用 `from_pretrained(path)` 加载权重。
4. 根据策略类型调用 `predict_action_chunk()` 或 `select_action()`。

在这条路径里，策略对象本身保留了输入输出语义，因此 wrapper 只是一个很薄的适配层。

#### 编译模型路径：CompiledPolicyWrapper + Adapter + RuntimeSession

`device:=ascend_om`、`device:=ascend_om_3403` 和 `device:=rknn` 均走编译模型路径。
它们共享同一个 `CompiledPolicyWrapper` 基类，通过 **组合模式** 将后端差异拆解为两个正交维度：

```
CompiledPolicyWrapper (PolicyWrapper)
  ├── CompiledModelAdapter (Protocol)  ← 模型语义适配：输入准备、输出解码
  └── RuntimeSession (Protocol)        ← 硬件运行时：加载、执行、释放
```

- **CompiledModelAdapter** 负责把 `dict[str, Tensor]` 转为后端可消费的输入格式，
  并把原始输出恢复为动作 Tensor。当前实现包括 `ACTCompiledAdapter`（ACT 家族）和
  `PI05CompiledAdapter`（Pi0.5），通过 `ADAPTER_REGISTRY` 按 `config.json` 的 `type` 字段派发。
- **RuntimeSession** 只负责硬件 artifact 的加载、执行与资源释放。当前实现包括
  `OMRuntimeSession`（通用 ACL OM）、`PI05OMRuntimeSession`（Pi0.5 VLM+AE 联合推理）、
  `SD3403RuntimeSession`（SD3403 板卡 worker）和 `RKNNRuntimeSession`（RKNN Lite），
  通过 `create_runtime_session()` 按 backend 名称和模型类型派发。

推理流程统一为三步：`adapter.prepare_inputs(batch)` → `session.execute(inputs)` →
`adapter.decode_outputs(outputs, device)`。

外层门面类只负责传入 backend 名称：

| 门面类 | backend | Adapter | RuntimeSession |
|--------|---------|---------|----------------|
| `AscendOMPolicyWrapper` | `ascend_om` | ACT: `ACTCompiledAdapter` / PI0.5: `PI05CompiledAdapter` | ACT: `OMRuntimeSession` / PI0.5: `PI05OMRuntimeSession` |
| `AscendOM3403PolicyWrapper` | `ascend_om_3403` | `ACTCompiledAdapter` | `SD3403RuntimeSession` |
| `AscendOMPi05PolicyWrapper` | `ascend_om` (Pi0.5) | `PI05CompiledAdapter` | `PI05OMRuntimeSession` |
| `RKNNPolicyWrapper` | `rknn` | `ACTCompiledAdapter` | `RKNNRuntimeSession` |

##### Ascend OM 路径

`device:=ascend_om` 下，实际执行的不再是 Python 中的 LeRobot policy 对象，
而是编译后的 `.om` 图。因此 OM 路径需要显式补回"模型语义层"：

- **ACT OM**：`ACTCompiledAdapter` 从 `config.json` 读取 `input_features`、`chunk_size`、
  `action_dim` 等元数据，负责输入顺序整理和输出 reshape。
  `OMRuntimeSession` 包装底层的 `OMmodel`（Ascend ACL runtime，包括
  `acl.init()`、`acl.mdl.load_from_file()`、Host/Device memcpy 与 `acl.mdl.execute()`）。
- **Pi0.5 OM**：`PI05CompiledAdapter` 处理 VLM 输入（图像 + token + mask）和动作输出。
  `PI05OMRuntimeSession` 包装底层的 `PI05OMModel`，后者加载 VLM 和 Action Expert
  两个 OM 模型，通过零拷贝 buffer 共享实现联合推理，并执行多步 denoising 循环。

##### Ascend OM SD3403 路径

`device:=ascend_om_3403` 使用 `ACTCompiledAdapter` + `SD3403RuntimeSession`：

- `SD3403RuntimeSession` 包装 `ACT3403Policy`，后者管理一个常驻 C++ worker 子进程。
- 通过自定义二进制协议（`PROTOCOL_MAGIC=0x53565031`）把输入写入 worker 的 `stdin`，
  worker 在板卡侧执行 OM 推理后通过 `stdout` 回传结果。
- `ACTCompiledAdapter` 负责输入准备和 SD3403 特有的输出 reshape（按 `sd3403_action_stride`
  裁剪到 `action_dim`）。

##### RKNN 路径

`device:=rknn` 使用 `RKNNPolicyWrapper(CompiledPolicyWrapper)`。

与 OM 路径类似使用 `ACTCompiledAdapter` + `RKNNRuntimeSession` 组合，但运行时更薄——
相当一部分"模型特定性"已经在导出阶段（`export_onnx_rknn.py`）固化：

1. 导出脚本直接从 `ACTPolicy` 导出 ONNX，用 `ACTONNXWrapper` 固定输入/输出接口，
   剥离多余输出，仅保留 `action` 输出。
2. `config.json` 中保留了 `input_features`、`chunk_size` 等元数据供运行时解释。
3. `RKNNRuntimeSession` 调用 `rknnlite.api.RKNNLite` 的 `load_rknn()` / `init_runtime()` /
   `inference()` 执行推理。
4. `ACTCompiledAdapter` 按 `input_features` 顺序从 batch 中抽取输入，将 Tensor 转为
   `numpy.float32`，并将输出恢复为动作 Tensor。

#### 编译产物配置：config.om.json

运行时设备以 launch/ROS 参数为准。若 LeRobot 导出的 `config.json` 内含
训练时设备记录（例如 `"device": "cuda"`），`inference_service` 会在本地临时
配置副本中将其覆盖为当前 runtime tensor device（OM/RKNN 等编译后端为 CPU），
不会修改原始模型目录，也不会让训练设备约束推理后端选择。

编译后端推荐在模型目录内生成独立的 `config.om.json` manifest，避免污染 LeRobot 原生
`config.json`。该文件描述后端 artifact 的路径和执行顺序：

```json
{
  "schema_version": 1,
  "policy_type": "act",
  "backend": "ascend_om",
  "artifact_dir": "om",
  "artifacts": {
    "policy": "act.om"
  },
  "execution": ["policy"]
}
```

各后端要求的 artifact 角色：

| 后端 | artifacts 键 | execution |
|------|-------------|-----------|
| `ascend_om` (ACT) | `policy`（单 `.om` 文件） | `["policy"]` |
| `ascend_om_3403` | `policy`（`.om`）+ `worker`（可执行二进制） | `["policy", "worker"]` |
| `ascend_om` (Pi0.5) | `vlm` + `action_expert`（各 `.om`） | `["vlm", "action_expert"]` |
| `rknn` | 从 `policy_path` 自动搜索 `model.rknn` 或 `*.rknn` | — |

`artifact_dir` 指定 artifact 的基础目录（相对于 manifest 文件解析）。`artifacts` 中
的值可以是字符串路径或 `{"path": "..."}` 对象。`execution` 显式声明通用串行 pipeline
的执行顺序。OM artifact 不再从 LeRobot `config.json`、环境变量或目录猜测读取；
转换工具必须生成 `config.om.json`。

#### 继承体系总览

```text
PolicyWrapper (ABC)
├── LeRobotPolicyWrapper          ← PyTorch / LeRobot 原生路径
└── CompiledPolicyWrapper         ← 所有编译后端的统一基类
      │   组合: CompiledModelAdapter + RuntimeSession
      ├── AscendOMPolicyWrapper       backend=ascend_om (ACT / Pi0.5)
      ├── AscendOM3403PolicyWrapper   backend=ascend_om_3403
      ├── AscendOMPi05PolicyWrapper   backend=ascend_om (Pi0.5 门面)
      └── RKNNPolicyWrapper           backend=rknn

CompiledModelAdapter (Protocol)
├── ACTCompiledAdapter             ← ACT 家族输入/输出语义适配
└── PI05CompiledAdapter            ← Pi0.5 VLM + Action Expert 适配

RuntimeSession (Protocol)
├── OMRuntimeSession               ← 包装 OMmodel (Ascend ACL)
├── PI05OMRuntimeSession           ← 包装 PI05OMModel (VLM+AE 零拷贝联合推理)
├── SD3403RuntimeSession           ← 包装 ACT3403Policy (板卡 worker 二进制协议)
└── RKNNRuntimeSession             ← 包装 RKNNLite (RK3588 NPU)
```

### 编译后端 vs PyTorch 推理流程对比

各路径的前处理与后处理保持一致，差异集中在中间的模型执行层。

#### PyTorch 推理

```text
Preprocessor -> Tensor batch -> LeRobotPolicyWrapper -> LeRobot policy object
           -> Action Tensor -> Postprocessor
```

特点：

1. 输入输出全程都是 `torch.Tensor`。
2. 策略对象自己知道输入输出语义。
3. 不需要 wrapper 手动管理运行时 buffer 与显式 memcpy。

#### 编译后端推理（Ascend OM / SD3403 / RKNN）

```text
Preprocessor -> Tensor batch -> CompiledPolicyWrapper
           -> Adapter.prepare_inputs() -> RuntimeSession.execute() -> Adapter.decode_outputs()
           -> Action Tensor -> Postprocessor
```

特点：

1. 运行时执行的是编译产物（`.om` 图 / worker 进程 / `.rknn` 模型），不是 Python policy 对象。
2. Adapter 负责输入顺序整理、输出 shape 恢复与 chunk 语义。
3. RuntimeSession 负责硬件加载、执行和资源释放。
4. 通用 ACL 路径需要显式执行 Host/Device 内存拷贝。
5. SD3403 路径通过 Python 与 worker 的二进制 IPC 协议通信。
6. Pi0.5 路径通过 VLM 和 Action Expert 的零拷贝 buffer 共享实现联合推理。
7. RKNN 路径的输出语义已在导出阶段固定，运行时恢复逻辑较薄。

### 设计总结

可以把三类后端理解为同一条推理管线中的不同"中间执行器"：

1. PyTorch 路径最原生，依赖完整的 LeRobot policy 类。
2. RKNN 路径把大量模型特定性前移到导出阶段，运行时 wrapper 更轻。
3. Ascend OM 路径的 runtime 更底层，因此需要在运行时显式补一层模型语义适配。
4. 所有编译后端共享 `CompiledPolicyWrapper` 的 Adapter + Session 组合模式，
   新增后端只需实现 `CompiledModelAdapter` 和 `RuntimeSession` 两个 Protocol。

这也是为什么 `inference_service` 选择通过 `PolicyWrapper` 统一抽象，而不是把不同后端
的逻辑直接散落到 `lerobot_policy_node.py` 或 ROS 节点实现中。

#### 场景二：单机调试（开发测试用）

在一台机器上同时运行 Edge + Cloud 节点，添加 `cloud_local:=true`：

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true \
    cloud_local:=true
```

### 验证分布式模式

```bash
# 1. 确认两个推理节点均在线
ros2 node list | grep -E 'act_inference|pure_inference'
# 预期输出：
#   /act_inference_node      ← Edge（前/后处理）
#   /pure_inference           ← Cloud（GPU 推理）

# 2. 确认分布式话题存在
ros2 topic list | grep -E 'preprocessed|inference/action'
# 预期输出：
#   /preprocessed/batch      ← Edge → Cloud
#   /inference/action         ← Cloud → Edge

# 3. 观察推理频率
ros2 topic hz /inference/action
```

### 日志说明

系统启动后，各节点会依次打印以下关键日志行，便于快速判断状态：

| 节点 | 日志示例 | 含义 |
|------|---------|------|
| `pure_inference` | `Waiting for preprocessed batches from edge node...` | Cloud 节点就绪，等待 Edge 发送数据 |
| `pure_inference` | `✓ First inference completed: latency=XXms` | 首次推理成功，确认端到端链路通畅 |
| `pure_inference` | `[stats] count=XX, avg=XXms, last=XXms` | 每 5 秒输出一次性能统计 |
| `act_inference_node` | `✓ First inference complete (distributed): total=XXms` | Edge 节点首次完成完整推理闭环 |
| `action_dispatcher` | `✓ First inference received: chunk=XX, latency=XXms` | 分发器首次收到可执行动作 |
| `action_dispatcher` | `[stats] inferences=XX, avg_latency=XXms, queue=XX, hold=XX` | 每 5 秒输出分发统计；`hold` 表示队列耗尽后保持末帧的次数 |

---

## 🧪 脱机测试 (Testing)
由于核心组件已经实现了零 ROS 依赖，模型工程师可以直接在没有任何 ROS 环境的机器上，使用 `pytest` 秒级验证张量维度的正确性：
```bash
pytest src/inference_service/tests/
```

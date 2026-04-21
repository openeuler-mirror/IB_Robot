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

### 启动命令

#### 场景一：跨机器分布式部署（推荐生产用法）

两台机器必须设置**相同的 `ROS_DOMAIN_ID`** 且在同一局域网内。

**步骤 1 — 在机器人本体（端侧 Device）上**：

端侧只启动 Edge 代理节点（前/后处理），不加载 GPU 模型：
```bash
export ROS_DOMAIN_ID=42
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true   # 仿真模式；真机去掉此参数
```

**步骤 2 — 在算力服务器（边端/云端 Edge/Cloud）上**：

```bash
export ROS_DOMAIN_ID=42   # 必须与端侧一致！
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/models/pretrained_model \
    device:=cuda
```

如果 Cloud 节点运行在端侧开发板（openEuler / OpenHarmony）上的 Ascend NPU，可直接切换为：

```bash
export ROS_DOMAIN_ID=42
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/models/pretrained_model \
    device:=npu
```

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

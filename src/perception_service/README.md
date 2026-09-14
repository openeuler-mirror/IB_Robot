# perception_service 节点说明

`perception_service` 是当前具身链路中的**连续场景理解包**。

它负责：

1. 持续监听相机、关节状态、末端位姿
2. 接收用户的连续文本提问或结构化请求
3. 将图像、机器人状态、用户补充信息一起发给大模型理解
4. 发布结构化理解结果和简短文本摘要
5. 可选运行 Grounding-DINO + SAM2 开放词汇检测与分割节点

## 1. 典型链路

```text
/camera/top/image_raw + /joint_states + /robot_status/ee_pose
  + 用户文本 / 用户 context_json
  -> perception_service_node
  -> /embodied/perception_result
  -> /embodied/perception_summary
```

如果启用了 RealSense / RGB-D，则当前链路也支持：

```text
/camera/front_camera/color/image_raw
  + /camera/front_camera/color/camera_info
  + /camera/front_camera/aligned_depth_to_color/image_raw
  + /camera/front_camera/depth/color/points
  + /joint_states
  + /robot_status/ee_pose
  -> perception_service_node
  -> 本地多模态模型
  -> /embodied/perception_result
```

## 2. 主要输入

### 2.1 当前支持的场景输入类型

现在 `perception_service_node` 不再只支持单张主相机图像，还支持：

| 输入 | 说明 |
| --- | --- |
| `primary_camera_topic` | 主视角 RGB 图像 |
| `wrist_camera_topic` | 末端 wrist 视角 RGB 图像，可选 |
| `primary_camera_info_topic` | 主视角内参，可选 |
| `primary_aligned_depth_topic` | 主视角对齐深度图，可选 |
| `primary_pointcloud_topic` | 主视角点云，可选 |
| `wrist_camera_info_topic` | wrist 视角内参，可选 |
| `wrist_aligned_depth_topic` | wrist 视角对齐深度图，可选 |
| `wrist_pointcloud_topic` | wrist 视角点云，可选 |

当只配置 `primary_camera_topic` 时，行为与旧版本保持一致。  
当额外配置 wrist / depth / pointcloud 时，节点会自动进入**多视角 + RGB-D**分析模式。

### 简单连续交互

```bash
ros2 topic pub --once /embodied/perception_text std_msgs/msg/String "{data: '看看桌面上有什么'}"
```

### 结构化请求

```bash
ros2 topic pub --once /embodied/perception_request ibrobot_msgs/msg/SceneAnalysisRequest \
  "{request_id: 'req-1', source: 'cli', session_id: 'demo', user_text: '判断红色物体是否适合抓取', context_json: '{\"focus_object\":\"red_block\",\"goal\":\"graspability\"}', timeout_sec: 120.0}"
```

### 视觉趣味游戏请求（分院帽等）

`perception_service_node` 作为通用视觉分析运行时，不感知具体游戏业务。视觉游戏 Gateway
（`embodied_agent/visual_game_gateway_node`）受理 Agent 请求后，会构造一条带角色
prompt 的 `SceneAnalysisRequest`（`source=game.<name>`，如 `game.sorting_hat`）发到本节点。
本节点按普通 scene-analysis 请求处理：抓场景 → 调 VLM → 发 `SceneAnalysisResult`。结果消费方
通过 `source` 识别业务类型，`scene_summary` 保存最终结果，失败时以 `error_code`/`message` 表达。
请求 `context_json.required_inputs` 声明该请求真正需要的输入（`primary_image` / `ee_pose` /
`joint_state`），本节点据此判定哪些缺失才阻塞：分院帽只声明 `primary_image`，故 EE pose / joint
state 离线时仍可成功；未声明或畸形的 `required_inputs` 维持严格默认（三者全需在线）。
`context_json.response_contract` 声明输出契约；当前支持 `kind=enum|string|number|string_array`，并与
视觉游戏 Gateway 复用同一个 validator。`enum` 会在发布成功结果前校验指定字段（如 `scene_summary`）
严格属于 `allowed_values`。契约声明畸形、`kind` 缺失/不受支持或字段值越界时，
本节点发布 `success=false`、`error_code=INVALID_RESPONSE_CONTRACT`，并保留 `raw_response` 便于诊断。

## 3. 主要输出

| topic | 类型 | 说明 |
| --- | --- | --- |
| `/embodied/perception_result` | `ibrobot_msgs/msg/SceneAnalysisResult` | 结构化场景理解结果 |
| `/embodied/perception_summary` | `std_msgs/msg/String` | 面向人类快速查看的摘要 |

说明：

- `SceneAnalysisResult` 的 ROS 消息字段本身没有因为 RGB-D 接入而破坏兼容。
- RGB-D / 多视角的中间上下文会进入节点内部 `scene_snapshot` / prompt，不会影响原有调用方。

## 4. 连续交互设计

- `session_id` 用于区分会话
- 节点会保留最近若干轮问答摘要，作为下一轮理解的上下文
- `context_json` 可携带用户附加信息，例如：
  - 关注目标
  - 关注区域
  - 安全限制
  - 任务目标

## 5. 当前默认模型配置

- provider: `openai_compatible`
- base URL: `http://localhost:8000/v1`
- model: `Qwen3.5-9B`
- API key env: 可留空；若本地服务需要鉴权，再额外配置环境变量
- 大模型输出超时：按**输出空闲超时**计算，默认来自 `embodied.timeouts.model_idle_timeout_sec`
- `SceneAnalysisRequest.timeout_sec` 现在会直接覆盖本次请求的模型 idle timeout，不再被强制抬高到 `120s`

仍然保留原远端 Kimicode 调用能力；只需把配置改回：

- provider: `kimicode`
- base URL: `https://api.kimi.com/coding/v1`
- api_key_env: `KIMICODE_API_KEY`
- model: `kimi-for-coding`

## 6. RealSense / RGB-D 接入说明

### 6.1 推荐的 `scene_sources` 配置

若要让 `perception_service_node` 直接消费 RealSense 数据，推荐在 `robot_config` 中配置：

```yaml
embodied:
  timeouts:
    task_budget_sec: 180.0
    scene_freshness_sec: 0.5
    model_idle_timeout_sec: 120.0
    rpc_timeout_sec: 5.0
    gripper_settle_sec: 1.5
  perception:
    scene_sources:
      primary_camera_topic: /camera/front_camera/color/image_raw
      primary_camera_info_topic: /camera/front_camera/color/camera_info
      primary_aligned_depth_topic: /camera/front_camera/aligned_depth_to_color/image_raw
      primary_pointcloud_topic: /camera/front_camera/depth/color/points
      ee_pose_topic: /robot_status/ee_pose
      joint_state_topic: /joint_states
      require_depth: true
      require_pointcloud: false
```

### 6.2 当前已落地的能力

当前 RealSense 改造已经支持：

1. 主相机 RGB 图像分析
2. wrist 图像与主图**联合分析**
3. `camera_info` 摘要注入
4. aligned depth 的结构化深度摘要注入
5. pointcloud 元信息摘要注入
6. 基于 RGB-D 的距离 / 遮挡 / 近距离障碍风险判断

注意：当前不是把原始深度数组或点云直接发给大模型，而是先在本地做摘要，再注入 prompt。

### 6.3 运行时注意事项

1. RealSense 图像/深度/点云订阅已改为**传感器 QoS**，否则容易因 QoS 不匹配而收不到数据。
2. 若 `require_depth=true`，但没有收到有效 depth topic，请求会直接失败，而不是静默退化。
3. 实际单独跑 `realsense2_camera_node` 时，设备默认常见原始 topic 是：

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/depth/color/points
```

如果希望统一走仓内推荐的 `/camera/front/...` 命名，应通过 `robot_config` launch 或显式 remap 做标准化。

## 7. 抓取检测与分割

抓取链使用 `inference_service.model_service_node` 承载强类型服务，不包含订阅相机快照的专用兼容节点。首次使用 PC Torch
deployment 前安装默认依赖并下载模型：

```bash
./scripts/setup.sh
./scripts/download_perception_models.sh
```

构建接口和 generic host：

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select ibrobot_msgs perception_service
```

正式运行由 robot-config `perception_services.services` 为每个 named deployment 启动一个 host。PC 配置选择组合
Grounded-SAM2 Torch deployment，通过 `GroundingDetect` 一次返回 bbox 和 mask；310P 配置分别选择
`grounding_dino` 与 `sam2` bundle，通过 `GroundingDetect -> SegmentDetections` 返回相同语义结果；
编译变体由 named deployment 和 bindings 表达。

### 7.1 310P 原始执行图的张量契约

`grounding-dino-raw-logits-cxcywh-v1` postprocessing 契约规定 `pred_logits` 的列布局：

| 维度 | 含义 |
| --- | --- |
| `pred_logits[0, q, :]` | decoder query `q` 的分类 logits |
| 列 `0 .. sequence_length-1` | 与 BERT prompt token 一一对应的相关性 logit |
| 列 `sequence_length .. 255` | 编译期填充，语义为空 |

`sequence_length` 不是常量，而由 manifest `model.inputs` 中 `input_ids` 的尾部维度决定
（310P 部署为 8）；adapter 从同一 `ModelDescriptor` 推导 tokenizer 窗口和 `encoder_tgt` 形状，
因此 manifest ABI 是唯一事实源。相关性打分只在前 `sequence_length` 列上取最大值，标签解码只在同
一窗口内按 token id 反查并剔除 `[CLS]`/`[SEP]`/`[PAD]`/`.` 等特殊 token；若 token 窗口宽于
`pred_logits` 的列数，adapter 直接抛错而不是截断，避免把填充列的噪声当成置信度。

`pred_boxes` 为归一化 `cxcywh`，由 adapter 换算到请求图像分辨率的 `xyxy` 并裁剪到图像边界；退化
框（宽或高非正）被丢弃。

### 7.2 检测数量与分割批次

原始 Grounding-DINO 每次输出 900 个 decoder query，而编译后的 SAM2 box-prompt decoder 只接受固定
数量的框（310P 部署为 4）。数量收敛由服务契约而非模型决定：

1. `model_contracts.MAX_DETECTIONS`（16）是 `GroundingDetect` 允许输出的检测上限。两个检测 plugin
   都用 `rank_detections()` 按置信度降序、原 query 序号稳定排序后截断，因此同一输入的输出顺序可复现。
2. `SegmentDetections` 先用 `validate_detection_batch()` 拒绝超过该上限的请求，再按 adapter 的
   `batch_size` 把检测切成多个编译批次逐批推理，并按原始检测顺序拼回 mask。检测数不是 `batch_size`
   的整数倍时，最后一批的空闲槽位用最后一个真实框填充，但只取前 `len(chunk)` 个 mask，不会多出检测。
3. 任一批次返回的 mask 数与该批检测数不符时服务直接失败，不做静默截断。

## 8. Generic Model Services

`inference_service.model_service_node` 是强类型模型服务的通用宿主。每个实例从 robot-config SSOT 读取 bundle、命名 deployment、
plugin class、具体 ROS service type、endpoint 和 runtime options。宿主不包含模型家族到 executable 的映射，也
不通过一个匿名 tensor service 暴露推理。

Plugin 必须实现 `ModelServicePlugin`：拥有具体 service request/response 映射，调用 adapter 和 shared
`ModelSession`，并投影统一的 runtime health。Adapter 拥有模型语义预处理和后处理；`inference_service` 拥有
设备生命周期、准入、tensor ABI、健康状态和资源回收。

每个响应中的 `ModelRuntimeInfo` 报告 instance ID、manifest fingerprint、deployment name/fingerprint、
backend、runtime state、readiness 和 failure reason。诊断身份来自节点配置和已验证 manifest，不允许 plugin
metadata 覆写，也不在启动时重新计算权重 SHA-256。

`perception_service.echo_adapter:EchoServicePlugin` 是唯一的通用层参考集成。它使用真实的
`ibrobot_msgs/srv/EchoModel` 契约以及无权重、无硬件 SDK 的
identity tensor session 验证 bundle -> runtime -> adapter -> typed response 链路，仅证明框架契约，不表示任何
生产模型或 accelerator deployment 已通过语义/硬件 conformance。生产服务类型、adapter、wrapper 和配置由各自
消费功能负责。

## 9. Semantic Mapping Model Services

新语义建图流水线使用五个独立的 generic `model_service_node` 进程：SAM2 mask、RAM++ tagging、SigLIP2
masked-image、SigLIP2 text 和可选 Grounding DINO confirmation。每个进程只承载一个 typed endpoint 和一个
named deployment；SigLIP2 image/text 不共享进程，但必须声明兼容的 embedding space。

每个 generic host 通过 response `ModelRuntimeInfo` 报告 readiness、semantic identity 和 deployment
provenance。Ascend OM 只能通过 schema-v3 manifest named deployment 进入 shared `AscendOmModelSession`；
wrapper 不接受旧的 backend alias。CUDA 不可用时必须在 SSOT 中选择另一个已经验证的 named
deployment，节点不会自动切换 backend。

感知模型与 ACT/PI0.5 使用相同的 bundle-first 结构。下载脚本直接在 `models/` 顶层生成四个 bundle
根目录：`models/sam2.1_hiera_tiny/`、`models/ram_plus_swin_large_14m/`、
`models/siglip2_so400m_patch14_384/` 和 `models/grounded_sam2_swint_ogc/`。每个目录包含
`inference_manifest.json`、`assets/adapter.json`、模型资产和 `torch_cpu`/`torch_cuda` named deployment。
Grounding DINO Swin-T OGC 的网络结构配置由 `perception_service.grounding_dino_config` 常量绑定到软件版本，
不作为模型资产复制进 bundle；bundle 仅保存 checkpoint、文本编码器和 SAM2 checkpoint 等运行资产。

SigLIP2 image/text 服务共享同一 bundle 与 embedding identity，但仍由独立进程加载独立 session。RAM++ runtime
由 `third_party/wheels/recognize-anything/` 中固定上游 commit 和补丁构建的 `ibrobot-ram` wheel 提供；Grounding-DINO
runtime 由 `third_party/wheels/groundingdino/` 中固定上游 commit 和补丁构建的 `ibrobot-groundingdino` 纯 Python wheel
提供（Transformers 5 兼容补丁已固化进源码，运行时不再 monkey-patch `transformers.BertModel`）；源码不是模型
资产。ONNX、OM 等未验证转换结果只能放在 bundle 外的
`models/_work/<bundle>/candidates/`，通过 conformance 与 promotion 后才可复制到不可变
`artifacts/<backend>/<deployment>/generations/<uuid>/` 并注册为 named deployment。

`RecognizeTags` 的 mask 输出按 mask 顺序扁平化，`mask_tag_counts` 描述每个 mask 对应的 slice。Adapter 固定过滤
RAM++ 颜色属性；调用方通过请求中的 `excluded_labels` 提供场景策略，服务在过滤后按 `max_mask_candidates`
截断置信度有序候选。generic host 不读取 robot_config，也不内置办公室或实验室业务 blacklist。整图 tags 仅用于
帧级诊断，不用于替代局部物体标签。

经 ABI 审核的 compiled-only 资产由 `perception_service.package_ascend_perception_bundles` 从
`models/_work/<bundle>/candidates/ascend_*` 打包。SAM2 和 SigLIP2 提供 `ascend_310p`、`ascend_310b` deployment；
Grounding DINO 当前只提供固定 720x1280、文本长度 8 的 `ascend_310p` deployment。多 OM pipeline 的中间 tensor
通过 manifest `device_links` 保持在设备侧，service adapter 只负责模型语义预处理与最终输出后处理。

### 检测结果质心

generic 感知服务不读取深度，因此其 `Detection2D` 只保证 bbox 与 mask。`grasp_planner_node` 使用同一 RGB
时间戳附近的 depth/CameraInfo 计算两种 3D 质心，并放入 `PlanGrasp` 响应（相机光学系，单位 m）：

| 字段 | 计算方式 | 适用场景 |
|------|---------|---------|
| `centroid_xyz` | mask 内可见表面点云的**算术平均** | 快速估计、小扁平物体 |
| `volume_centroid_xyz` | 点云**凸包体积质心**（四面体分解 + 体积加权） | 凸形物体的物理中心近似 |
| `volume_m3` | 凸包体积 | 物体尺寸估计 |

**差异**：`centroid_xyz` 只反映相机能看到的正面，深度方向偏前；`volume_centroid_xyz` 把凸包"虚构背面"补回，质心沿夹爪 −Z 方向后移约 7~12 mm（取决于物体厚度），更接近真实物理中心。

**已知局限**：
- 弯曲物体（banana）：凸包质心可能侧向偏出物体轮廓（投影法 ~47% 落在 mask 外）
- 小圆物体（strawberry）：凸包背面空腔大，质心离实际表面 11~15 mm
- 凸形物体（marker、cucumber）：两种质心差异小，体积质心可靠

若需更准确的体积质心，应使用多视角点云融合补全背面，再做网格体积积分。

### 9.1 Ascend 310P 抓取感知

310P 抓取感知使用两个 schema-v3 bundle 和 shared `AscendOmModelSession`，不进入 policy `AscendBackend`。
板端只依赖 NumPy、OpenCV、`inference_manifest`、`inference_service` 和 CANN ACL，不导入
GroundingDINO、SAM2、TorchVision 或对应 CUDA 扩展。用仓库打包器从已验证候选生成独立 bundle：

除非另有说明，本节所有命令均在 IB_Robot 仓库根目录执行；所有项目内路径均相对于仓库根目录。

```bash
source .shrc_local
ros2 run perception_service package_ascend_perception_bundles \
  --models-root models \
  --model-type grounding_dino
ros2 run perception_service package_ascend_perception_bundles \
  --models-root models \
  --model-type sam2
```

`--models-root` 必须与 robot config 的 bundle 根目录一致。打包器从 `models/_work/` 中读取已经验证的候选，
并把发布 bundle 写到 `models/grounding_dino_swint_seq8_1280x720/` 和
`models/sam2.1_hiera_tiny_prompt_ascend/`；它不执行 ONNX/OM 编译，也不会写回 `models/perception/` 旧布局。

SAM2 的 operation 是 v3 manifest 的 bundle 级模型身份，不是 deployment 属性：
`models/sam2.1_hiera_tiny/` 只承载 `sam2/automatic`，可以在该语义下提供 CPU、CUDA 或 Ascend named
deployment；`models/sam2.1_hiera_tiny_prompt_ascend/` 承载抓取所需的 `sam2/prompt`。两者可以复用同一
模型族的权重来源，但输入、输出和预后处理契约不同，不能合并到同一个 manifest。

Grounding-DINO bundle 记录并校验 12 个 OM、`encoder_tgt.npy`、bundle-local WordPiece vocab 和 D2D links；
SAM2 bundle 独立记录 encoder 与固定 batch-4 decoder。GroundingDINO 固定输入为
`1x3x720x1280`、文本长度为 8；提示词超过 8 个 BERT token 时会明确报错，不能静默截断。

完整抓取配置在顶层 `perception_services` 选择两个本地 Ascend named deployment：

```yaml
perception_services:
  services:
    - id: grasp_grounding
      bundle_path: models/grounding_dino_swint_seq8_1280x720_ascend
      deployment: ascend_310p
      service_type: ibrobot_msgs/srv/GroundingDetect
    - id: grasp_segmentation
      bundle_path: models/sam2.1_hiera_tiny_prompt_ascend
      deployment: ascend_310p
      service_type: ibrobot_msgs/srv/SegmentDetections
```

由统一 robot launch 启动后，RGB-D 相机、两个 model service、GraspGen、MoveIt 执行和抓取验证使用同一
ROS service/topic 契约；`grasp_planner_node` 通过配置的 endpoint 区分组合 PC 服务和 310P 两段服务。

## 10. GraspGen 6-DoF 抓取

GraspGen 是感知模型，不是 LeRobot policy：输入一朵已分割的物体点云，输出一组相机系 6-DoF
抓取位姿和置信度。它与 SAM2/RAM++/SigLIP2 走同一条链路——bundle-first、named deployment、
typed endpoint、`ModelSession`——所以既不出现在 `inference_service` 的 policy family 矩阵里，
也不通过匿名 tensor service 暴露。

### 10.1 Typed service

| 项 | 值 |
|---|---|
| Service type | `ibrobot_msgs/srv/GenerateGrasps` |
| Plugin | `perception_service.model_service_plugins:GraspGenGenerateGraspsPlugin` |
| Adapter | `perception_service.graspgen_adapter:GraspGenAdapter` |
| Model identity | `tensor_model/graspgen/generate_grasps` |

Request 是 `sensor_msgs/PointCloud2 object_points` 加 `max_grasps` / `min_confidence`；
Response 是 `GraspCandidateArray grasps` 加通用诊断字段 `model` / `inference_time_ms` /
`success` / `message`。点云按声明的 field offset 解码，因此带 RGB 通道的 RealSense 点云可以
直接送入。`GraspCandidate.pose_matrix` 是行主序展平的 4x4 相机系变换，与请求点云同 frame；
`target_width_m`、`width_axis_camera`、`collision_free` 保持默认值——GraspGen 只对位姿打分，
不测量夹爪开口也不做碰撞检查，这几项由 `manipulation_execution` 用自己的几何计算填充。

注意 `ibrobot_msgs/srv/PlanGrasp` 是 `manipulation_service` 的文本提示抓取管线服务，没有
`ModelRuntimeInfo model` 字段，过不了 `model_service_node` 的服务契约校验，两者不可互换。

### 10.2 Adapter 与 host orchestration

`GraspGenAdapter` 拥有模型语义的预处理与后处理：`prepare()` 剔除非有限点与统计离群点、
下采样到 `point_count`、按物体质心居中并乘 `kappa`，返回张量和质心；`postprocess()` 把
`grasp.poses` / `grasp.confidence` 还原回相机系（加回质心）、按置信度排序、先过
`min_confidence` 阈值再截断到 `max_grasps`。identity 为
`preprocessing=object-points-outlier-filtered-centered-kappa-scaled-v1`、
`postprocessing=grasp-pose-matrix-confidence-sorted-v1`、
`supported_deployments={ascend_310p, ascend_310b}`。

设备侧是八个 OM 子图（`inference_manifest.GRASPGEN_EXECUTION`）：generator 与 discriminator
各三个 PointNet++ 编码 stage，加一个 denoiser 和一个 discriminator head。子图之间的
PointNet++ 分组（FPS / ball query / grouping）、DDPM 去噪循环和 SO(3) 位姿转换都不在任何编译
图里，由主机计算，因此 `GraspGenAscendSession` 继承通用的 `AscendOmModelSession` 并只重写
`_execute`，逐角色驱动而不是直线执行 `execution`。

这些主机中间张量用 `host.` 语义命名空间声明（`host.graspgen.*`）。`host.` 是继 `internal.`
之后的第三类语义：既不属于对外契约（不参与 `ModelDescriptor` 的 1:1 校验），也不要求图内
producer。声明了任一 `host.` binding 的 deployment 即为 host-orchestrated；manifest 校验对它
放宽"声明的 semantic 必须都被绑定"，但仍然拒绝"绑定了未声明的外部 semantic"。两个 encoder
embedding 走 `internal.graspgen.*` 与 `device_links`，始终留在设备侧、不经主机拷贝。

对外契约只有三个 semantic：

```
inputs  = [observation.object_points  float32 [-1, 3]]
outputs = [grasp.poses      float32 [-1, 4, 4],
           grasp.confidence float32 [-1]]
```

`grasp.poses` 是主机积分出来的，不绑定到任何 OM 输出张量。

Runtime options 在通用 Ascend 的 `device_id` 之外多接受一个
`random_seed`，用于让去噪循环可复现；除此之外选项集合仍然是封闭的。统一 bundle 同时声明
`torch_cuda` 与 `ascend_310p`；两者共享 adapter、配置和 checkpoint 身份，但运行时必须显式选择
named deployment。Ascend 的八个 OM 与它们之间的主机数学仍是一个整体契约。

### 10.3 Bundle 与 promotion

Bundle 属于抓取模型域，不含任何 LeRobot 资产；通用运行时使用
`interface="tensor_model"`，不代表它应放在 `models/perception/`：

```
models/grasp/graspgen_robotiq_2f_140/
  inference_manifest.json          # schema v3, tensor_model/graspgen/generate_grasps
  assets/adapter.json              # model identity + kappa, diffusion_steps,
                                   # grasp_batch_size, point_count, geometry
  assets/graspgen_config.yml
  assets/generator_checkpoint.pth
  assets/discriminator_checkpoint.pth
  artifacts/ascend/ascend_310p/    # 八个 <role>.om
```

模型常量只落在 `assets/adapter.json`，由 `GraspGenAdapter.from_bundle()` 读回并逐项校验——
写错一个常量在设备上是一次沉默的 shape 失败，所以在加载时就拒绝。

ONNX → OM → ABI → bundle 的三步流程：前两步 `graspgen-export-onnx` 与
`graspgen-onnx-to-om` 是 `model_utils` 的导出工具（见
`src/model_utils/model_utils/README.md`）；第三步用本包的命令打包：

除非另有说明，以下命令在 IB_Robot 仓库根目录执行；所有项目内路径均相对于仓库根目录。

```bash
ros2 run perception_service package_graspgen_ascend_bundle \
    --bundle-root models/grasp/graspgen_robotiq_2f_140 \
    --onnx-manifest models/_work/graspgen_robotiq_2f_140/model_utils/onnx/graspgen.onnx.json \
    --om-dir models/_work/graspgen_robotiq_2f_140/model_utils/om \
    --om-abi-dir models/_work/graspgen_robotiq_2f_140/model_utils/abi \
    --soc-version Ascend310P3
```

八个 OM 的 binding 必须来自 ACL runtime introspection 得到的实际 ABI，不能用 ONNX tensor
name 代替；sidecar 缺失时 packager 默认用本机 Ascend device 生成，无 device 的打包主机需要预
先备齐并传 `--no-inspect-missing-abi`。全部八个角色解析通过后才写第一个字节，任一 OM 或 ABI
缺失都不会留下半成品 bundle。ONNX manifest 的 `contract_version` 与
`inference_manifest.GRASPGEN_CONTRACT_VERSION` 不一致时直接拒绝——角色顺序、语义命名或
PointNet++ 采样几何变更都会 bump 该版本号，避免旧图编出的 OM 被新 session 用错误的分组驱动。

### 10.4 SSOT 配置

与其它五个感知服务一样，由 robot-config 的 `perception_services.services[]` 声明；
`lekiwi_realsense_mapping.yaml` 里模板条目为 `id: semantic_graspgen_grasps`（默认 `enabled: false`）。
启用时补齐 `bundle_path`、`deployment`（如 `ascend_310p`）、`adapter_class`、`service_type`、
`endpoint`。GraspGen 不绑定到 `semantic_mapping.perception.semantic_roles` 的任何角色——
抓取由 manipulation 侧直接调用 `GenerateGrasps`，不参与建图。

### 10.5 Conformance

`perception_service.conformance` 提供 `grasp_pose_error()` 与 `evaluate_grasp_conformance()`，
按置信度排名逐位比较两个 backend 的抓取输出：平移误差、旋转误差（`R_ref.T @ R_cand` 的测地
角，能穿过 axis-angle 往返表示变换）、置信度差和抓取数量差，阈值见 `ConformanceThresholds`。
排名比较是刻意的——执行器只会尝试前几个，两个列表"元素集合相同但顺序不同"不算等价行为。

板端实机推理不在本仓库的自动化测试范围内；`test_graspgen_session.py` /
`test_graspgen_plugin.py` 用 fake `AclModel` 驱动八个角色，manifest 与 packager 保持真实，
因此 binding 写错会在这里失败而不是在设备上。

## 11. HRI 人体感知（YOLOX + PEAR）

HRI 链使用两个独立的 generic `model_service_node` 进程，与语义建图服务同构：一个 typed endpoint
对应一个 named deployment，不共享进程。

| id | service type | endpoint | plugin | bundle / deployment |
| --- | --- | --- | --- | --- |
| `hri_yolox_person` | `ibrobot_msgs/srv/YoloXDetect` | `/perception/hri/yolox_detect` | `model_service_plugins:YoloXPersonDetectPlugin` | `models/yolox_x_640` / `ascend_310p` |
| `hri_pear_parameters` | `ibrobot_msgs/srv/PearParameterPredict` | `/perception/hri/pear_parameters` | `model_service_plugins:PearParameterPredictPlugin` | `models/pear_parameter_network` / `ascend_310p` |

两个 bundle 都是仓库外的外部模型件（`models/` 下由下载获得），与 SAM2 / SigLIP2 一样通过 schema-v3
manifest 的 named deployment 进入 shared `AscendOmModelSession`。两条都是 `required: true`：bundle 缺失
或 deployment 不可用时服务在加载期直接失败，不进入"起来了但推不了"的状态。

### 11.1 预处理归属

**调用方传原始图像和框，不碰像素。** `YoloXDetect` 收整帧 `sensor_msgs/Image`，
`PearParameterPredict` 收同一张整帧加一个 `DetectionArray`；letterbox、通道顺序、裁剪、缩放、
归一化全部在 adapter 内。这与 §8 的分工一致（adapter 拥有模型语义预处理和后处理），也意味着
执行侧节点里不出现 `cv2`。

`YoloXAdapter`：

```text
源帧 RGB → BGR          （通道反转，在任何几何操作之前）
ratio = min(640/H, 640/W)，按 ratio 缩放，贴到 640×640 画布左上角，pad 114
```

通道反转是必须的：图是按上游 YOLOX demo 路径（`cv2.imread` 的 BGR 序）导出的。喂 RGB 不会报错，
但会导致检测框和后续人体参数产生系统性偏移。
输出为未解码的 head 输出（`[1,8400,85]`，无 grid/stride 解码），解码、NMS、按 `person` 过滤和
按 `ratio` 反映射回源图坐标都在 adapter 里；反映射只除 `ratio`，letterbox 没有居中偏移。

`PearParameterAdapter`：

```text
person bbox xyxy（源图坐标）
→ 中心 (cx, cy)，side = max(w, h) × 1.25
→ 方形仿射裁剪（INTER_LINEAR / BORDER_CONSTANT 0）→ 256×256 → BGR → NCHW float32 / 255
```

仿射的目标角是 **255 不是 256**：`cv2.getAffineTransform` 映射的是像素中心，写 256 会把每个 crop
整体缩放 256/255 并静默移动回归出的姿态。ImageNet 归一化和 `[:, :, :, 32:-32]` 宽度切片**在导出图
内部**，调用方再做一遍等于归一化两次，得到看起来合理但错误的姿态。

编译后的 PEAR 部署是 batch 1：一次请求只处理一个人体框，请求里给多于一个框会被拒绝。退化框
（宽或高非正、非有限）同样直接拒绝而不是裁剪修正——OpenCV 对退化框不报错，它返回一块均匀填充的
patch，网络会照样回归出一个姿态，形状/有限性/`success` 三道检查全都能通过。

### 11.2 输出语义

`PearParameterPredict` 按固定顺序返回八个张量：`smplx_pose_raw` [312]、`smplx_scale` [6]、
`smplx_shape` [200]、`smplx_expression` [50]、`flame_pose` [14]、`flame_shape` [300]、
`flame_expression` [50]、`camera_raw` [3]。`smplx_pose_raw` 切分为
`0:6` global_orient、`6:132` body_pose(21×6D)、`132:222` 左手、`222:312` 右手；6D 值不是欧拉角，
需要先解码成旋转矩阵。这些是 SMPL-X 局部关节旋转，**不是机器人电机角**；`camera_raw` 是相对/模型
坐标，没有真实内参和根节点深度时不能当作绝对相机 XYZ。EHM / SMPL-X LBS、网格生成与渲染都不在
bundle 内。

### 11.3 测试

`test/test_hri_perception_adapters.py` 覆盖 YOLOX 的通道与 letterbox/decode 契约、PEAR 的通道与裁剪几何、
输出尺寸以及无效输入校验。板端实机推理不在本包的自动化测试范围内。

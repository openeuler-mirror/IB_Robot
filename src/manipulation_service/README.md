# manipulation_service

`manipulation_service` 提供基于 GraspGen 的 6-DOF 抓取规划能力，以及抓取后成功/失败
验证服务。该包负责 ROS 2 服务封装、深度图和 mask 转换、GraspGen 调用、抓取结果发布、
抓取证据融合，以及在线/离线调试产物导出；不负责目标检测、机器人控制、MoveIt 执行、
策略推理或数据集转换。

## 入口与接口

节点/工具：

- `grasp_planner_node`：在线抓取规划节点。订阅对齐深度和 CameraInfo，
  调用 `perception_service` 获取目标 mask，运行 pip 安装的 GraspGen，提供
  `~/plan_grasp` 服务，并发布 `~/grasps`。
- `grasp_verifier_node`：抓取后验证节点。订阅 `/joint_states`、
  `/so101_follower/joint_currents` 和腕部深度图，提供 `~/verify_grasp` 服务，
  融合夹爪残余开度、夹爪电流和腕部 RealSense 可见性判断抓取是否成功。腕部相机抓后可能因
  夹爪/手腕自遮挡、目标离开视野、深度缺失或反光而不可靠；此时视觉结果
  只作为诊断/弱证据，不会单独判失败。深度订阅回调只保留最新消息；全帧有效率、近距比例和
  中位深度在 `VerifyGrasp` 请求时计算，避免空闲期间持续扫描 30 Hz 深度图。
- `test_graspgen.py`：离线 GraspGen 调试脚本。读取已有 RGB-D/mask fixture 目录，直接运行
  GraspGen，并保存 PLY、JSON 和可选 Open3D 视图。

ROS 接口：

- 服务：`ibrobot_msgs/srv/PlanGrasp`、`ibrobot_msgs/srv/VerifyGrasp`
- `PlanGrasp` 同时返回目标 surface/volume centroid、原始 scene 桌面方程、completed-scene 执行桌面
  方程和 object-top 点，供执行层做 contact-distance 排序、目标夹爪 mesh tabletop 检查和安全
  pregrasp 高度计算。
- `PlanGrasp.grasps.header` 的 `stamp` 是生成 3D 候选所用 depth frame 的采集时间；执行层必须用该
  时间查询 TF，不能用推理完成后的 latest transform。
- 话题：`ibrobot_msgs/msg/GraspCandidateArray`
- 单个抓取结果：`ibrobot_msgs/msg/GraspCandidate`
- 感知依赖：`ibrobot_msgs/srv/GroundingDetect`，可选串联 `ibrobot_msgs/srv/SegmentDetections`

`GraspCandidate` 只包含机器人无关的抓取候选信息。除了 GraspGen 位姿、置信度和碰撞标记，
节点还会基于分割目标点云估算每个候选沿源夹爪闭合轴方向的 `target_width_m`、
`target_width_quality`、`width_axis_camera` 和相对候选原点的稳健宽度区间。区间保留目标偏向
闭合轴哪一侧的信息，供非对称夹爪判断固定爪前缘间隙。这些字段用于下游机器人执行层做自己的夹爪几何适配；
`manipulation_service` 不读取任何 SO101 配置，也不输出 SO101 专用 offset。

默认在线话题遵循 SO101 wrist RealSense 抓取流水线约定：

- 对齐深度：`/camera/wrist/aligned_depth_to_color/image_raw`
- CameraInfo：`/camera/wrist/aligned_depth_to_color/camera_info`
- 检测服务：`/perception/grasp/grounding_detect`
- 分割服务（310P）：`/perception/grasp/segment_detections`

RealSense 顶置相机调试时常用的话题为：

- 对齐深度：`/camera/camera/aligned_depth_to_color/image_raw`
- CameraInfo：`/camera/camera/color/camera_info`

## 环境与依赖

CUDA 路径使用 pip 安装的 GraspGen：`manipulation_service` 在同一 Python 进程中导入
`grasp_gen`。Ascend 本地路径通过统一 `inference_manifest.json` 加载 GraspGen OM 子图。
两个后端都把 `num_grasps` 按每批最多 1000 拆分，并在所有批次合并后执行一次全局
`topk_num_grasps`；该值小于等于 0 时保留全部达到阈值的候选。Ascend 各批次连续消耗同一个随机流，
避免重复候选。GraspGen 不再放在
`libs/` 下；安装脚本会把固定上游源码作为 editable pip 依赖放到 workspace venv 的 `src/` 缓存中。

运行前需要满足：

- 使用 `local_cuda` 时当前环境需有可用 CUDA PyTorch；上游 `GraspGenSampler` 内部会把模型和点云移到 CUDA。
- 使用 `ascend_local` 时需有可加载的 Ascend GraspGen manifest/OM bundle 和 ACL 运行环境。
- 已通过 `./scripts/setup.sh` 默认安装 `grasp_gen` 和 `pointnet2_ops`（Ubuntu 默认安装；有 CUDA toolkit 时从源码编译，无 `nvcc` 时安装仓库内、绑定 Python 3.10 + Torch 2.7.1/cu126 + SM86 的 wheel；openEuler Embedded 因 CUDA 扩展限制跳过）。该 wheel 受 NVIDIA License 非商业研究/评估用途限制。
- GraspGen 运行 bundle 位于 `models/graspgen/`；其中
  `assets/adapter.json` 声明 bundle-relative 配置和权重路径。旧的
  `models/grasp/checkpoints/` 布局和 `GRASPGEN_MODEL_DIR` 仍作为兼容回退。
- 已构建 `manipulation_service` 和 `ibrobot_msgs`。

所有 ROS 调试命令都应在仓库根目录运行，并在同一条命令里完成环境初始化：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 service list
```

修改代码或首次运行前，先构建接口和本包：

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select ibrobot_msgs manipulation_service
```

## 调试 grasp_planner_node

### 1. 单独运行在线 GraspGen 抓取节点

运行前需先通过 robot-config 启动相机，以及抓取配置中每个 named deployment 对应的
`perception_service/model_service_node`。310P 配置会分别启动 Grounding-DINO 检测和 SAM2
分割实例；不要再启动已删除的 `grounded_sam2_node` 或 `grounded_sam2_snapshot`。默认使用
SO101 wrist camera 话题：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node
```

默认使用 GraspGen Robotiq 2F-140 作为 source gripper 生成候选，并用同一 source gripper
collision mesh 执行普通场景碰撞过滤。桌面平面拟合默认开启，但逐候选 source-gripper tabletop
sweep 默认关闭；目标执行层应使用自己的夹爪几何做 hard gate。

使用 RealSense 顶置相机，并开启调试输出：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info \
  -p save_debug_outputs:=true \
  -p debug_output_dir:=outputs/realsense_grasp_debug
```

目标宽度估计参数保持机器人无关，默认适配当前 GraspGen Robotiq 2F-140 配置：

```bash
-p target_width_axis_local:=1.0,0.0,0.0 \
-p target_width_percentile_low:=5.0 \
-p target_width_percentile_high:=95.0 \
-p target_width_min_m:=0.005 \
-p target_width_max_m:=0.14
```

如果更换 GraspGen 模型夹爪，只需把 `target_width_axis_local` 改为该源夹爪局部坐标系下的闭合轴，
不要在本包中加入机器人型号判断。

如果目标只是确认 GraspGen 是否能产生候选，可临时关闭几何过滤：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info \
  -p save_debug_outputs:=true \
  -p debug_output_dir:=outputs/realsense_grasp_debug \
  -p enable_collision_filter:=false \
  -p enable_tabletop_filter:=false \
  -p require_tabletop_filter:=false
```

### 2. 调用抓取规划服务

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp "{text_prompt: 'banana', confidence_threshold: 0.10, grasp_threshold: 0.5, debug_output_mode: 'default'}"
```

请求字段：

- `text_prompt`：目标名称，会转发给 `GroundingDetect`，例如 `banana`。
- `confidence_threshold`：感知检测阈值；大于 `0` 时覆盖 perception 节点默认值。
- `grasp_threshold`：GraspGen discriminator 置信度阈值；大于 `0` 时覆盖节点参数。
- `debug_output_mode`：单次请求调试输出模式。空字符串或 `default` 沿用节点
  `save_debug_outputs` 参数；`none` 不写文件；`diagnostic` 只写 `grasp_result.json`；
  `full` 写 `grasp_result.json`、点云 PLY、gripper mesh/line 和 PNG/HTML 预览。

响应重点看：

- `success` / `message`：是否规划成功和失败原因。
- `diagnostic_details`：规划诊断明细；没有生成抓取点时优先查看
  `failure_stage`、`failure_reason`、mask/depth 点数、GraspGen 原始候选数、
  collision/tabletop 过滤前后数量。
- `debug_output_dir`：本次请求实际写调试文件时返回输出目录；未写文件时为空。
- `grasps.grasps[*].pose_matrix`：每个抓取位姿的 4x4 row-major 矩阵。
- `grasps.header`：候选所在相机 frame，以及生成 3D 几何所用 depth frame 的采集时间。
- `grasps.grasps[*].confidence`：GraspGen 置信度。
- `grasps.grasps[*].collision_free`：当前过滤配置下是否认为无碰撞。

`manipulation_service` 返回的是机器人无关的 GraspGen 候选，排序主要来自 GraspGen
置信度和几何过滤结果。目标机器人专用的目标夹爪宽度补偿、IK/workspace 过滤和
接触点重对齐属于执行层后处理，统一由
`manipulation_execution/pick_executor_node` 完成。正式 executor 从
`robot_config.robot.grasp_execution` 读取服务名称、超时、速度、规划阈值、候选过滤、IK、接触补偿、
接触重对齐、位姿诊断、目标夹爪几何和执行评分；监督式客户端不能覆盖这些行为参数。因此 GraspGen
不需要绑定 SO101，新增机器人可在自己的 robot_config 中定义完整执行策略。

### 3. 通过正式执行层抓取

`manipulation_service` 不控制机械臂。完整抓取由 `manipulation_execution/pick_executor_node`
提供的 `/manipulation/execute_pick` delegated action 执行；它负责目标机器人专用的 IK/workspace、夹爪几何、
MoveIt 运动、恢复和抓后验证。推荐通过统一 robot-config bringup 启动 planner、verifier 和 executor，
不要使用历史调试脚本代替机器人执行层。外部调用必须通过 Capability Gateway 的 `pick_object` 入口，
不得直接构造 delegated action goal。

监督式执行示例：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
robot-skill --config-name lekiwi_handeye_realsense_grasp execute pick_object \
  --task-id pick-marker-001 --target-name marker --timeout-sec 240
```

只验证感知和规划而不产生运动时，保持 `authorize_motion:=false` 并调用规划服务：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp \
  "{text_prompt: 'marker', confidence_threshold: 0.1, grasp_threshold: 0.5, debug_output_mode: 'diagnostic'}"
```

SO101 执行侧 tabletop sweep 优先使用 `PlanGrasp.execution_table_plane_*`。该平面由服务端直接基于
completed scene cloud 按执行侧历史采样规则拟合，因此正常运行不再依赖 `scene_cloud.ply`，也不会为了
tabletop filter 强制提升到 `debug_output_mode=full`。显式 full-debug 请求仍会写 PLY；execution plane
不可用时，正式 executor 的目标夹爪 tabletop 门禁 fail closed。

常见 `diagnostic_details` 字段：

- `failure_stage` / `failure_reason`：失败阶段和具体原因；成功时通常不存在。
- `detection_confidence`：perception 返回的最佳目标置信度。
- `mask_pixel_count`：对齐到深度图后的目标 mask 像素数；为 `0` 表示 mask 为空。
- `valid_depth_in_mask_count` / `valid_depth_ratio_in_mask`：目标 mask 内有效深度点数和比例；
  为 `0` 时通常是 RGB/depth 未对齐、目标深度空洞、深度编码或尺度异常。
- `object_point_count` / `scene_point_count`：传入 GraspGen/过滤器的目标和场景点数。
- `scene_cloud_table_holes_enabled` / `scene_table_hole_added_count`：是否启用目标 footprint
  附近的 scene dense table patch，以及加入 scene/collision/tabletop filter 的补点数量。
- `raw_grasp_count`：GraspGen 原始候选数量；为 `0` 表示模型未产生候选，
  可尝试降低 `grasp_threshold` 或检查目标几何是否适合当前夹爪。
- `collision_filter`：碰撞过滤后/前的候选数量，例如 `0/80` 表示所有候选碰撞。
- `tabletop_plane_found` / `tabletop_best_inlier_ratio`：是否找到桌面平面及最佳内点比例。
- `tabletop_filter`：桌面 clearance 过滤后/前的候选数量。
- `source_gripper_tabletop_sweep_enabled`：是否实际执行了 source-gripper 候选 mesh sweep。
- `tabletop_auto_tuned` / `tabletop_auto_tune_reason`：低矮目标 adaptive retry
  是否生效，以及 retry 成功或跳过原因。
- `final_grasp_count`：最终返回的抓取数量。

### 3. 订阅抓取结果

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 topic echo /grasp_planner/grasps
```

该话题发布最近一次服务调用的抓取结果，便于检查 frame_id、pose matrix 和置信度。

## 调试 grasp_verifier_node

`grasp_verifier_node` 面向抓取保持状态验证。典型调用时机是夹爪闭合后和抬升/保持之后；
它不移动机器人，只读取最新传感器状态并返回 `SUCCESS`、`FAILED` 或 `UNCERTAIN`。

启动节点：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_verifier_node
```

SO101 默认假设夹爪关节 `6` 的 `0.0` 为闭合、`1.0` 为打开。如果使用其他机器人或话题，可覆盖参数：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_verifier_node --ros-args \
  -p gripper_joint:=6 \
  -p joint_state_topic:=/joint_states \
  -p joint_current_topic:=/so101_follower/joint_currents \
  -p wrist_depth_topic:=/camera/wrist/aligned_depth_to_color/image_raw \
  -p gripper_closed_position:=0.0 \
  -p gripper_contact_min_opening:=0.08 \
  -p current_contact_threshold_a:=0.08
```

`gripper_joint` 默认值为 SO101 的 `6`。其他机器人必须显式覆盖该参数；如果主动设为空，
节点不会再自动猜测夹爪关节，响应 `message` 会标出夹爪证据已禁用，返回结果通常只能依赖
腕部深度证据并趋向 `STATUS_UNCERTAIN`。

调用服务：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 service call /grasp_verifier/verify_grasp ibrobot_msgs/srv/VerifyGrasp "{task_id: 'pick_001', text_prompt: 'banana', expected_target_width_m: 0.035, post_grasp_wait_s: 0.2}"
```

响应字段：

- `success` / `status`：`success=true` 只对应 `STATUS_SUCCESS`；`STATUS_UNCERTAIN` 表示证据不足，不等同失败。
- `confidence`：当前状态判断的置信度。
- `message`：融合判断摘要。
- `evidence`：稳定的 `key: value` 诊断行，包括夹爪开度、电流、腕部深度有效比例和遮挡状态。

**触发方式与边界**：`verify_grasp` 是被动 ROS 2 服务，仍需调用方主动触发。
`manipulation_execution/pick_executor_node` 会按 `robot.grasp_execution.verification` 策略在 close 后
自动调用；监督式测试和 Hermes 通过同一个 `PickObject` action 进入该路径。
每次调用独立采样当前传感器状态，不追踪抓取历史。

当前版本只做后验状态融合，不做外部 RGBD 目标跟踪。返回 `UNCERTAIN` 时，required 策略会保守停止；
其他上层流程也应重观察或进入保守放置/重试策略，不能把它当作成功。

评分权重和阈值可通过 ROS 参数调节，默认行为保持保守融合：

- `score_gripper_contact_success` / `score_gripper_contact_failure`：夹爪残余开度对成功/失败的权重。
- `score_gripper_residual_success` / `score_gripper_residual_failure`：残余开度处于中间区间时的双向弱证据。
- `score_current_contact_success` / `score_current_contact_failure`：夹爪电流高于/低于接触阈值的权重。
- `score_wrist_occlusion_success`：腕部深度遮挡的弱正向权重；设为 `0.0` 可改为纯诊断。
- `score_success_threshold` / `score_failure_threshold` / `score_margin_threshold`：最终状态判定阈值。

## grasp_planner_node 参数说明

模型和输入：

- `device`：推理设备，默认 `cuda`。
- `gripper_config`：GraspGen 模型配置，默认 `graspgen_robotiq_2f_140.yml`。
- `model_dir`：GraspGen 模型根目录；空字符串表示使用默认模型目录。
- `depth_topic`：对齐深度图输入。
- `camera_info_topic`：与深度/RGB 对应的相机内参。
- `detect_service`：目标检测服务，默认 `/perception/grounding_detect`。
- `segment_service`：可选分割服务。为空时使用 `GroundingDetect` 内联返回的 mask；配置 endpoint 时始终
  把检测 bbox 传给 `SegmentDetections`。
- `depth_scale`：整数深度转米比例，默认 `1000.0`；浮点深度会自动按米处理。

抓取生成：

- `grasp_threshold`：GraspGen discriminator 阈值，默认 `0.5`。
- `num_grasps`：每轮生成的候选数量，默认 `800`。
- `topk_num_grasps`：每轮保留的 top-K 候选，默认 `50`。
- `enable_object_cloud_completion`：启用目标点云补全，默认 `true`。当前在目标 mask
  内对局部 depth hole 做邻域均值补点，并可配合 prismatic side extrude 生成连接桌面的
  空心表面壳；不会向物体内部填充实心体积点。
- `object_cloud_completion_mode`：补全模式，默认 `mask_depth_inpaint`；关闭补全时等价于 `none`。
- `object_cloud_completion_max_points`：单次最多新增目标点数量，默认 `5000`。
- `object_cloud_completion_kernel_size`：补点邻域窗口，默认 `5` 像素。
- `object_cloud_completion_min_neighbors`：补点所需的最少有效邻居数，默认 `6`。
- `enable_object_cloud_prismatic_extrude`：把目标外轮廓沿桌面法线补成空心侧墙，默认
  `true`。这层会进入补全后的中间点云；实际送给 GraspGen 的输入还会经过 outlier
  removal 和下采样，并写入 `object_cloud_graspgen_input.ply`。
- `object_cloud_prismatic_extrude_max_points`：侧墙补点上限，默认 `8000`。
- `object_cloud_prismatic_extrude_layers`：外轮廓到桌面之间的采样层数，默认 `8`。
- `enable_scene_cloud_table_holes`：启用目标 footprint 附近的 scene table dense patch，默认
  `true`（在线节点显式 opt-in；底层 wrapper 默认关闭以保持旧调用方行为）。这些生成点会进入
  scene cloud，并参与 collision/tabletop filter 和 debug 可视化。
- `scene_cloud_table_holes_max_points`：scene table dense patch 补点上限，默认 `8000`。

输入同步：

- `input_buffer_size`：深度/CameraInfo 缓冲帧数，默认 `30`。
- `sync_max_age_sec`：mask 时间戳与深度/内参允许的最大时间差，默认 `0.20` 秒。

碰撞过滤：

- `enable_collision_filter`：启用 GraspGen 碰撞过滤，默认 `true`。
- `collision_threshold`：碰撞距离阈值，单位米，默认 `0.005`。
- `collision_gripper`：碰撞检测和可视化用 gripper 名称；空字符串表示沿用模型 gripper。

SO101 adapter 等目标执行器与 GraspGen 源夹爪不一致时，不要把源夹爪 collision filter
作为 hard gate；启动节点时显式设置 `enable_collision_filter:=false`，并在执行侧使用目标夹爪
自己的 tabletop/height/IK guard。

桌面过滤：

- `enable_tabletop_filter`：启用桌面平面过滤，默认 `true`。
- `enable_source_gripper_tabletop_sweep`：桌面拟合成功后，是否逐候选检查 GraspGen/source gripper
  mesh 的 final-to-pregrasp clearance，默认 `false`。仅在需要分析 source-gripper clearance 或将其
  作为过滤依据时开启；桌面平面、object-top 和执行桌面输出不受影响。
- `require_tabletop_filter`：找不到可接受桌面平面时返回空结果，默认 `true`。
- `tabletop_clearance`：夹爪 mesh 到桌面的最小距离，默认 `0.003` 米。
- `tabletop_pregrasp_distance`：沿 GraspGen/source gripper approach 轴后退检查距离，默认 `0.08` 米。
- `tabletop_pregrasp_steps`：pre-grasp 到 final grasp 中间检查步数，默认 `5`。
- `tabletop_ransac_threshold`：桌面 RANSAC 内点距离阈值，默认 `0.006` 米。
- `tabletop_min_inlier_ratio`：接受桌面平面的最小内点比例，默认 `0.15`。
- `tabletop_filter_mode`：桌面过滤模式，默认 `strict`。`adaptive` 会对草莓等低矮目标
  按目标高度自适应降低 clearance 和 pre-grasp 检查距离；`soft` 在低矮目标严格过滤为
  空时，允许保留不低于 hard floor 的风险最低候选，并在诊断中标记 `tabletop_relaxed`。
  桌面过滤始终对 final pose 到 pre-grasp path 的所有采样点取最小 clearance，并以该
  最小值作为硬过滤条件。
- `adaptive_tabletop_low_profile_height`：低矮目标高度阈值，默认 `0.035` 米。
- `adaptive_tabletop_clearance_min` / `adaptive_tabletop_clearance_max`：自适应 clearance
  下限/上限，默认 `0.001` / `0.002` 米。
- `adaptive_tabletop_clearance_height_ratio`：低矮目标 clearance 与目标高度的比例，默认 `0.12`。
- `adaptive_tabletop_pregrasp_min`：自适应 pre-grasp sweep 距离下限，默认 `0.02` 米。
- `adaptive_tabletop_pregrasp_height_ratio`：pre-grasp sweep 距离与目标高度的比例，默认 `2.0`。
- `adaptive_tabletop_hard_floor`：`soft` 模式低矮目标 fallback 允许的最低桌面距离，默认 `-0.002` 米。
- `adaptive_tabletop_auto_tune`：`adaptive` 模式低矮目标被桌面过滤清空时自动 retry，默认 `true`。
- `adaptive_tabletop_retry_clearances`：auto-tune 依次尝试的 clearance 列表，逗号分隔字符串，
  默认 `0.002,0.001` 米；只会尝试低于当前 adaptive clearance 的正值。
- `adaptive_tabletop_retry_pregrasp_height_ratio`：auto-tune retry 的 pre-grasp sweep 距离与目标高度比例，
  默认 `1.5`。
- `adaptive_tabletop_retry_hard_floor`：auto-tune 允许 retry 的最低首轮候选 clearance，默认 `-0.003` 米；
  低于该值视为明显穿桌，不自动放宽。

抓取草莓等贴桌低矮目标时，推荐先用 `adaptive` 模式，不要直接关闭桌面过滤：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p tabletop_filter_mode:=adaptive \
  -p adaptive_tabletop_low_profile_height:=0.035 \
  -p adaptive_tabletop_clearance_min:=0.001 \
  -p adaptive_tabletop_clearance_max:=0.002 \
  -p adaptive_tabletop_pregrasp_min:=0.02
```

`adaptive_tabletop_auto_tune` 默认开启。若首轮 adaptive 因低矮贴桌目标被过滤为空，节点会在
`tabletop_best_candidate_clearance_m` 未明显低于桌面的前提下，自动尝试更小 clearance 和更短
pre-grasp path，并在诊断中记录 `tabletop_auto_tuned`、`tabletop_auto_tune_attempts` 和
`tabletop_auto_tune_reason`。

如果 Robotiq 源夹爪 tabletop gate 疑似误杀 SO101 adapter 后可执行的候选，可使用
`diagnostic` 模式做真机前诊断。该模式仍拟合桌面并记录每个候选的 tabletop clearance
统计，但不使用 Robotiq mesh 一票否决候选：

```bash
-p tabletop_filter_mode:=diagnostic
```

如果执行侧已使用 SO101 自身 mesh 做 fail-closed tabletop hard gate，可进一步跳过仅供诊断的
Robotiq 候选扫描，同时保留 table plane 和 object-top：

```bash
-p enable_tabletop_filter:=true -p enable_source_gripper_tabletop_sweep:=false
```

如果仍希望保留一个近桌安全下限，可临时使用 `soft`。此时必须检查 `grasp_result.json`
中 `tabletop_relaxed`、`tabletop_best_candidate_clearance_m`、`tabletop_auto_tune_reason`
和预览图，确认夹爪没有明显穿桌：

```bash
-p tabletop_filter_mode:=soft -p adaptive_tabletop_hard_floor:=-0.002
```

在线调试输出：

- `save_debug_outputs`：默认是否为请求写完整调试产物，默认 `false`；单次请求可用
  `debug_output_mode` 覆盖。
- `debug_output_dir`：输出根目录，默认 `outputs/grasp_pipeline`。
- `debug_max_save_grasps`：保存 gripper mesh/line 的 top grasp 数量，默认 `10`。
- `debug_render_preview`：是否后台生成 PNG/HTML 预览，默认 `true`。
- `debug_render_show_scene`：PNG 中显示整场景点云而非仅目标点云，默认 `false`。
- `debug_render_labels`：是否生成带 index/confidence/depth 标签的 PNG，默认 `true`。
- `debug_render_max_points`：PNG 预览最大采样点数，默认 `60000`。
- `debug_render_width` / `debug_render_height`：PNG 尺寸，默认 `1280` / `720`。
- `debug_render_interactive`：是否生成自包含交互式 HTML，默认 `true`。
- `debug_interactive_show_scene`：HTML 是否包含 scene 点云 trace，默认 `true`。
- `debug_interactive_max_points`：HTML 中 object + scene 最大嵌入点数，默认 `120000`；
  设为 `0` 表示嵌入全部点，文件会很大。

## 在线调试产物

当节点参数 `save_debug_outputs:=true`，或单次请求设置 `debug_output_mode` 为
`diagnostic` / `full` 时，请求会在
`<debug_output_dir>/<timestamp>_<prompt>_<time-ns>/` 下写入调试产物，并在响应的
`debug_output_dir` 字段返回该目录。

`debug_output_mode` 行为：

- `default` 或空字符串：沿用节点 `save_debug_outputs` 参数；参数为 `true` 时等价于
  `full`，为 `false` 时等价于 `none`。
- `none`：不写任何调试文件。
- `diagnostic`：只写 `grasp_result.json`，不生成点云、gripper mesh/line 或预览图。
- `full`：写 `grasp_result.json`、点云 PLY、gripper mesh/line 和 PNG/HTML 预览；如果
  请求在检测或同步阶段提前失败，则只能写诊断 JSON。

完整输出包含：

- `grasp_result.json`：prompt、阈值、相机内参、诊断字段、每个 grasp 的 pose、
  confidence 和点云元数据；即使最终抓取数量为 `0` 也会记录诊断。
- `object_cloud.ply`：补全后的目标点云壳，保留兼容旧调试工具。
- `object_cloud_raw.ply`：启用目标点云补全时额外写出，表示补全前目标点云。
- `object_cloud_completed.ply`：启用目标点云补全时额外写出，表示补全后、outlier
  removal 和下采样前的目标点云壳。
- `object_cloud_graspgen_input.ply`：实际送入 GraspGen 的空心表面壳；已完成 outlier
  removal 和下采样，不包含物体内部实心采样点。
- `scene_cloud.ply`：非目标场景点云，用于碰撞上下文；启用 `enable_scene_cloud_table_holes`
  时，目标 footprint 附近的 dense local table patch 也会写在这里，并进入后续 collision/tabletop
  filter。
- 颜色约定：绿色为 raw object，橙色 `(255, 170, 0)` 为 mask-depth inpaint，青色
  `(0, 200, 255)` 为目标 prismatic side wall，紫色 `(168, 85, 247)` 为桌面补点。
- `grasp_cloud.ply`：目标 + 场景点云。
- `grasp_grippers.ply`：top grasp 的夹爪 collision mesh。
- `grasp_lines.ply`：top grasp 的夹爪控制点线框。
- `grasp_preview_labeled.png`：headless 2D 投影预览，带 grasp index、confidence 和 depth 标签。
- `grasp_preview.html`：自包含交互式 3D 视图，包含 object/scene 点云、
  confidence 着色夹爪线框和 grasp pose hover 信息。
- `grasp_preview_meta.json`：预览渲染状态、采样点数和错误信息。

PNG/HTML 渲染在后台 daemon thread 中执行；service response 不等待图像导出。
因此 `grasp_result.json` 可能先出现，预览文件稍后出现。

## 调试 test_graspgen.py

`test_graspgen.py` 不依赖 ROS service。它读取已有 fixture 中的 `result.json`、mask、深度和
CameraInfo，直接运行 GraspGen，适合单独验证 GraspGen
模型、过滤器和可视化。

基础命令：

```bash
fixture_dir="outputs/grounded_sam2/REPLACE_WITH_EXISTING_FIXTURE"
test -d "$fixture_dir" && source .shrc_local && \
  python3 src/manipulation_service/test_graspgen.py --data-dir "$fixture_dir"
```

显示交互式 Open3D 视图和分数标签：

```bash
fixture_dir="outputs/grounded_sam2/REPLACE_WITH_EXISTING_FIXTURE"
test -d "$fixture_dir" && source .shrc_local && \
  python3 src/manipulation_service/test_graspgen.py \
  --data-dir "$fixture_dir" \
  --show \
  --show-scores
```

只验证 GraspGen 候选生成，临时关闭桌面过滤：

```bash
fixture_dir="outputs/grounded_sam2/REPLACE_WITH_EXISTING_FIXTURE"
test -d "$fixture_dir" && source .shrc_local && \
  python3 src/manipulation_service/test_graspgen.py \
  --data-dir "$fixture_dir" \
  --no-enable-tabletop-filter \
  --no-require-tabletop-filter
```

常用参数：

- `--data-dir`：已有 RGB-D/mask fixture 目录；历史 `outputs/grounded_sam2/...` 数据仍可直接 replay。
- `--show`：打开 Open3D 交互式 3D 视图；无桌面环境时不要加。
- `--show-scores`：在可视化中显示 confidence 标签。
- `--show-scene`：显示完整 scene 点云；默认只显示目标点云，更清晰。
- `--collision-gripper`：碰撞过滤和可视化使用的 gripper 名称；默认沿用模型 gripper。
- `--max-save-grasps`：写入 PLY 的 top grasp 数量，默认 `10`。
- `--num-grasps`：每轮生成候选数，默认 `800`。
- `--topk-num-grasps`：每轮保留 top-K，默认 `50`。
- `--min-grasps`：累计达到该数量后停止，默认 `5`。
- `--max-tries`：最多推理轮数，默认 `4`。
- `--grasp-threshold`：GraspGen discriminator 阈值，默认 `0.5`。
- `--collision-threshold`：碰撞距离阈值，单位米，默认 `0.005`。
- `--depth-scale`：整数深度转米比例，默认 `1000.0`。
- `--enable-tabletop-filter` / `--no-enable-tabletop-filter`：打开/关闭桌面过滤。
- `--require-tabletop-filter` / `--no-require-tabletop-filter`：找不到桌面平面时是否返回空结果。
- `--tabletop-clearance`：夹爪到桌面的最小距离，默认 `0.003` 米。
- `--tabletop-pregrasp-distance`：pre-grasp 后退检查距离，默认 `0.08` 米。
- `--tabletop-pregrasp-steps`：pre-grasp sweep 检查步数，默认 `5`。
- `--tabletop-ransac-threshold`：桌面平面 RANSAC 阈值，默认 `0.006` 米。
- `--tabletop-min-inlier-ratio`：接受桌面平面的最小内点比例，默认 `0.15`。
- `--tabletop-filter-mode`：`strict` / `adaptive` / `soft` / `diagnostic`，用于验证低矮目标的自适应桌面过滤；
  `diagnostic` 只记录 Robotiq tabletop clearance，不过滤候选。
- `--adaptive-tabletop-*`：与在线节点同名的自适应桌面过滤参数。
- `--adaptive-tabletop-auto-tune` / `--no-adaptive-tabletop-auto-tune`：打开/关闭 adaptive retry。
- `--adaptive-tabletop-retry-clearances`：离线 auto-tune retry clearance 列表，例如 `0.002,0.001`。
- `--enable-scene-cloud-table-holes` / `--no-enable-scene-cloud-table-holes`：离线显式打开/关闭
  目标 footprint 附近的 scene dense table patch；默认关闭以复现 wrapper 旧行为。
- `--scene-cloud-table-holes-max-points`：离线 scene dense table patch 最大补点数。
- `--fx --fy --cx --cy`：手动指定相机内参；仅旧数据目录缺少 CameraInfo 时需要。

输出目录为 `<data-dir>/graspgen_output/`，包含：

- `grasp_result.json`：GraspGen 诊断字段、每个 grasp 的 pose、confidence 和 collision 状态；
  即使最终抓取数量为 `0` 也会写入。
- `object_cloud.ply` / `scene_cloud.ply` / `grasp_cloud.ply`：目标、场景和合并点云。
- `grasp_grippers.ply`：top grasp 的夹爪 mesh。
- `grasp_lines.ply`：top grasp 的夹爪线框。
- `grasp_preview.png`：离线预览截图。

注意：离线预览中的编号仍是 GraspGen 服务返回顺序，不包含 SO101 正式执行层的
接触点质心重排、动态宽度补偿或 IK/workspace 过滤结果。`PickObject.MODE_PLAN_ONLY` 是 Gateway 内部
诊断字段，当前 v1 catalog 未对外公开；若要提供完整候选筛选的无运动入口，必须先扩展版本化 catalog 和
`SkillCommand` 参数契约，不能直接调用 delegated action。

## 常见调试路径

### 只调 perception，不调 GraspGen

分别调用配置的 `GroundingDetect` 和可选 `SegmentDetections` endpoint，确认 bbox、mask 和 model
runtime info 正确后，再进入本包调 GraspGen。

### 只调 GraspGen，不启动 ROS service

使用已有 RGB-D/mask fixture 目录运行 `test_graspgen.py`。这样可以排除
ROS topic 同步、service timeout 和在线 perception 的影响。

### 调完整在线链路

1. 启动相机。
2. 由 robot-config 启动抓取所需的 generic model service。
3. 启动 `grasp_planner_node`，建议先开启 `save_debug_outputs`。
4. 调用 `/grasp_planner/plan_grasp`。
5. 查看输出目录中的 `grasp_preview_labeled.png` 和 `grasp_preview.html`。

## 排障

- `GroundingDetect service not available`：检测 model service 未启动，或 `detect_service` 参数不对。
- `segment_service_unavailable`：配置了独立分割 endpoint，但对应 `SegmentDetections` model service 未就绪。
- `No synchronized depth/CameraInfo`：mask 时间戳与深度/内参差距超过
  `sync_max_age_sec`，检查相机话题和时间戳。
- `No grasps generated`：先查看 `diagnostic_details` 或 `grasp_result.json` 中的
  `failure_stage` 和 `failure_reason`，再按阶段处理。
- `failure_stage: point_cloud`：查看 `mask_pixel_count`、`valid_depth_in_mask_count`
  和 `valid_depth_ratio_in_mask`。mask 为空说明 perception 没有有效分割；mask 内
  有效深度为 `0` 时检查 RGB/depth 对齐、RealSense 深度空洞、深度编码和 `depth_scale`。
- `failure_stage: graspgen_inference`：GraspGen 没有原始候选，先降低
  `grasp_threshold`，再检查目标局部几何是否适合当前夹爪。
- `failure_stage: collision_filter`：所有候选被场景碰撞过滤，临时关闭
  `enable_collision_filter` 可确认模型是否本身有候选；同时检查 `scene_cloud.ply`
  是否包含过多目标附近噪点。
- `failure_stage: tabletop_filter`：查看 `tabletop_best_inlier_ratio`、`tabletop_failure_reason`、
  `tabletop_filter`、`tabletop_auto_tuned`、`tabletop_auto_tune_reason`、
  `scene_cloud.ply` 是否包含桌面，必要时调整 `tabletop_ransac_threshold`、
  `tabletop_min_inlier_ratio`、`tabletop_clearance` 或 `tabletop_pregrasp_distance`。
- HTML/PNG 未立刻出现：预览在后台线程生成，等待几秒后查看
  `grasp_preview_meta.json`。
- `ModuleNotFoundError`：未执行 `source .shrc_local`，或 GraspGen 依赖未安装。

## 架构边界

- 机器人和相机拓扑应保留在 `robot_config` YAML 中。
- 本包不拥有 perception 模型推理、机器人控制、MoveIt 轨迹执行、策略推理或数据集转换。
- 完整抓取编排由 `manipulation_execution` 消费本包的 `PlanGrasp` 和 `VerifyGrasp` 服务完成。
- GraspGen 作为可选重量级运行依赖处理；若后续增加 server backend，ROS service
  合约应保持稳定。

# robot_config

ros2_control 和外设的统一机器人配置系统。

## 概述

本软件包为机器人硬件提供统一的配置系统，桥接以下组件：

- **ros2_control**：用于关节/电机控制接口
- **外设**：用于相机和其他设备（通过现有 ROS2 驱动）
- **tensormsg**：用于 ML 策略 I/O 契约

目标是建立机器人硬件配置的单一数据源，消除不同配置系统之间的重复。

声源转向由 `robot.embodied.idle_behaviors.sound_orientation` 管理，schema 默认关闭，仅支持移动底盘的
`nav_turn`。`mode: keyword` 使用配置中的完整 ASR 短语触发一次转向；`mode: periodic` 忽略 ASR 文本，
由 `sound_following` Skill 在 `inactive` / `active` 会话之间切换，并按 `periodic_interval_sec` 消费新的
`SpeechDirection.segment_id`。`enabled` 只决定是否启动节点，`default_active` 才决定 periodic 会话初始状态；
运动授权仍只能由 `authorize_motion` launch 参数提供。`lekiwi_nav_grasp` 的 hybrid stage 默认启用
periodic（会话仍为 inactive，需显式激活）；其他 stage 与其余机器人 profile 保持关闭，可经
`lekiwi_nav_grasp_sound_real` overlay 显式开启。periodic 模式只消费 `SpeechDirection`，校验上不要求
`voice_asr.enabled`；keyword 模式仍要求 ASR 与 speech direction 同时可用。

`VoiceASRNode` 和 `speech_direction_node` 都订阅 `audio_io` 发布的共享音频话题，不直接竞争打开 ReSpeaker
设备。完整状态机、Gateway binding、watchdog 和 reset 契约见 `docs/idle_sound_orientation_design_zh.md`。

通用机器人 profile 不固化具体设备实例的相机序列号。多设备部署应通过部署侧 instance
override 注入序列号，避免把某一台实物设备绑定到所有同型号 profile。

## 特性

- **单一 YAML 配置**：在一个文件中定义 ros2_control、相机和 ML 契约
- **使用现有 ROS2 相机驱动**：
  - `usb_cam` 用于 USB 相机（基于 OpenCV）
  - `realsense2_camera` 用于 RealSense D400 系列
- **TF 发布**：自动发布相机坐标系变换
- **标定支持**：标准 ROS2 camera_info_manager 集成
- **tensormsg 集成**：契约通过名称引用外设
- **RealSense contract normalization**：将驱动原生 topic 收口到统一 `/camera/{name}/...` 接口；高带宽
  RGB-D 可由驱动直接 remap，CameraInfo relay 仅负责规范化 `frame_id`
- **通用模型服务编排**：通过顶层 `perception_services.services` 列表启动任意强类型 model-service plugin

### ros2_control 启动门禁

真机控制器由单个 `robot_config/controller_spawner` 进程逐个加载和配置，再通过一次 `switch_controller` 调用统一
激活当前控制模式要求的 controller，并在同一次切换中停用配置为 inactive 的 controller，避免部分激活窗口。该入口
修复 ROS 2 Humble 官方 spawner 未向 `load_controller` 传递 service call timeout 的问题，并在调用超时后重新读取
lifecycle 状态，接受服务端已经完成的操作。组 spawner 成功退出即作为控制器启动屏障，通过后才启动 MoveIt、teleop
或 task executor。发现、service call 和 switch 的等待上限都来自 robot YAML 的
`robot.controller_startup_timeout.hardware`，不再维护第二套 hardware lifecycle readiness 判定。

## 架构

```
robot_config YAML（单一数据源）
        │
        ├───► ros2_control（关节/电机）
        │       └───► so101_hardware 插件
        │
        ├───► 相机驱动（现有 ROS2 包）
        │       ├───► usb_cam（USB 相机）
        │       └───► realsense2_camera（RealSense D400）
        │               ├───► direct topic remap（高带宽 RGB-D）
        │               └───► topic_relay（CameraInfo/frame_id 规范化）
        │
        └───► tensormsg 契约（ML I/O）
                └───► PolicyBridge / EpisodeRecorder
```

### 通用模型服务 SSOT

模型服务使用顶层 `perception_services`，不属于具体消费者。每个 enabled entry 必须显式选择 manifest 中的命名
deployment，不能直接填写 `backend` 或 `device`：

```yaml
robot:
  perception_services:
    services:
      - id: depth_front
        enabled: true
        required: true
        bundle_path: models/depth/front
        deployment: cpu
        adapter_class: depth_service.plugin:DepthServicePlugin
        service_type: depth_msgs/srv/EstimateDepth
        endpoint: /depth/front
        node_name: depth_front_service
        runtime_options: {}
```

配置 loader 验证 bundle/manifest/deployment、plugin 和 service type 语法，并拒绝重复 ID、node name 和 endpoint。
Launch builder 对每个 enabled entry 启动一个相互独立的 `perception_service/model_service_node`，不维护模型家族
白名单或 family-to-executable map。Manifest/deployment fingerprint 是结构化运行身份；artifact 内容校验属于
模型打包流程，不通过服务启动时扫描大文件完成。通用层只用 dummy/echo plugin 做 CI 参考，生产 adapter 和具体
service entry 由消费功能自行注册。

语义建图在 `semantic_mapping.perception.semantic_roles` 中仅把 `sam2_masks`、`ram_plus_tags`、
`siglip2_image`、`siglip2_text` 和可选 `gdino_confirmation` 绑定到上述 service ID。语义层验证每个 role 的
精确 service type、required/optional policy、manifest semantic identity，以及 SigLIP image/text embedding
metadata 兼容性；通用 parser 仍不维护模型 family 词表。有效 role bindings、service declarations、deployment
fingerprint 和 semantic identity fingerprint 共同生成与路径和列表顺序无关的正 63-bit configuration generation。

## Sim camera pose override（仅限仿真标定，opt-in）

仿真启动时，URDF / MJCF 中相机位姿默认来自 robot YAML 与
`launch_builders/sim_backend/camera_presets.py:PRESETS`。为了让开发者在仿真里
调出的视角可以跨重启复用，launch adapter 层会通过
`launch_builders/sim_backend/camera_overrides.py` 读取一条不进版本控制的用户级旁路：

| 项 | 说明 |
|---|---|
| 文件路径 | `~/.ros/ibrobot/sim_camera_overrides/<camera_name>.yaml` |
| 写入方 | `dataset_tools.camera_alignment`（stub）和 `sim_models.sim_camera_adjuster`（真实位姿） |
| 字段 | `parent_frame`, `pose.{x,y,z,roll,pitch,yaw}`, `fovy_deg` |
| 直接生效平台 | `gazebo` |
| MuJoCo 路径 | 不直接读取同一套角度，而是由 `mujoco_adapter.py` 做显式坐标系转换 |
| 文件缺失或 stub | 回退到 `PRESETS`，保持历史默认行为 |

这条 override 只服务于开发者标定，不应该被运行时业务逻辑当作新的 SSOT。
如果仿真相机姿态和 YAML 里看到的不一致，先检查这个目录里是否残留了历史 override。

## 配置示例

### 机器人配置 overlay

当一个机器人配置只扩展现有机器人时，可在 `robot` 下声明 `base_config`，避免复制完整 YAML：

```yaml
robot:
  name: so101_with_extension
  base_config: so101_single_arm
  teleoperation:
    devices:
      __append__:
        - name: extension_device
          type: custom
```

`base_config` 只允许引用同目录的机器人 YAML；相对路径以当前 overlay 文件为基准，无扩展名时补
`.yaml`。这个边界保证继承配置中的相对资源路径仍具有唯一语义。mapping 递归合并，scalar 和 list 默认
替换；list 只有使用仅含 `__append__` 的 mapping 才追加。loader 拒绝循环引用、mapping/list
容器类型不匹配，以及含额外字段的 `__append__`。规范化结果以 `_config_path` 标记最外层配置，并按基线到 overlay 的顺序
在 `_config_sources` 保留完整来源链，供诊断和审计使用。

```yaml
robot:
  name: so101_single_arm
  type: so101
  robot_type: so_101

  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  peripherals:
    - type: camera
      name: top
      driver: opencv  # 使用 usb_cam 包
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame
      # 可选 ISP 参数（usb_cam 全部 V4L2 控件，详见 usb_cam_node.cpp:65-85）：
      # brightness: 128            # 0-255
      # contrast: 128              # 0-255
      # saturation: 128            # 0-255
      # sharpness: 128             # 0-255
      # gain: 0                    # 0-255
      # auto_white_balance: false  # 与 white_balance 互斥
      # white_balance: 4600        # 2000-10000 K
      # autoexposure: false        # 与 exposure 互斥
      # exposure: 312              # V4L2 absolute exposure
      # autofocus: false
      # focus: 0                   # 0-1023

    - type: camera
      name: wrist
      driver: realsense  # 使用 realsense2_camera 包
      serial_number: "12345678"
      width: 640
      height: 480
      fps: 30
      depth_width: 640
      depth_height: 480
      frame_id: camera_wrist_frame

  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top  # 引用上面的相机
        image:
          resize: [480, 640]
      - key: observation.current
        topic: /so101_follower/joint_currents
        type: ibrobot_msgs/msg/JointCurrent
        selector:
          names: ["current.1", "current.2", "current.3", "current.4", "current.5", "current.6"]
```

## Observation video transport

Observation transport is part of the contract and is absent by default. An omitted
`transport` field keeps the existing DDS image behavior. Explicit RTP is only valid
for `sensor_msgs/msg/Image` observations using `rgb8` or `bgr8`, and requires a
distributed inference pipeline. RTP configuration is fail-closed: it does not fall
back to DDS when a stream or codec is unavailable.

```yaml
observations:
  - key: observation.images.top
    topic: /camera/top/image_raw
    type: sensor_msgs/msg/Image
    peripheral: top
    image: {resize: [480, 640], encoding: rgb8}
    transport:
      mode: rtp
      stream_id: top
      endpoint: {host: 192.168.10.20, port: 5004}
      codec: h264
      encoder_backend: nvidia  # 本机 RTX 使用 NVENC；也可用 auto/software/ascend
      decoder_backend: ascend
      h264: {profile: main, bitrate_bps: 4000000, gop_frames: 15}
      media:
        width: 640
        height: 480
        frame_rate_hz: 30
        pixel_format: nv12
        color_space: bt709
        color_range: limited
      buffer: {sender_queue_frames: 2, receiver_queue_packets: 256, decoded_frame_capacity: 32, retention_ms: 1000}
      readiness: {keyframe_timeout_ms: 3000, timestamp_mapping_max_age_ms: 1000, max_inter_camera_skew_ms: 50}
      recording: {integrity_mode: strict}
      security: none
```

The contract fingerprint includes stream identity, endpoint, codec/media reconstruction
semantics, buffering/readiness limits, and image metadata. Consequently endpoint
changes require matching edge and cloud deployments; there are no excluded endpoint
overrides in this milestone. RTP/UDP has no authentication, confidentiality, or
integrity protection and is limited to a trusted robot network. Use an explicit
`mode: dds` contract for rollback or recording-compatible deployments.

Development examples are available in `config/robots/dev_rtp_single_camera.yaml`
and `config/robots/dev_rtp_multi_camera.yaml`; production profiles remain DDS by
default.

`transport.recording.integrity_mode` controls episode-level RTP recording integrity:

- `strict` is the default. Any RTP packet gap, timestamp mapping failure, or session
  generation change rejects the episode and removes every RTP stream recorded for it.
- `tolerant` preserves the episode and records gaps in each `.h264.json` NDJSON sidecar.
- All RTP observations in one contract must use the same mode. Mixed strict/tolerant
  declarations are rejected during contract validation.

`nvidia` 当前仅支持 encoder，典型组合是 edge 使用 `encoder_backend: nvidia`，310B cloud 使用
`decoder_backend: ascend`。`auto` 的 encoder 探测顺序为 `ascend`、`nvidia`、`software`；显式选择
不可用 backend 时直接拒绝启动。

## Grasp execution target gripper

`robot.grasp_execution.planner_node` 中的 ROS 参数会传给 `grasp_planner_node`；`host_runtime` 是
bringup 消费的进程启动配置，不会注入 ROS 参数。当 GraspGen source gripper 与目标执行器不一致，
且执行侧已有目标夹爪 mesh tabletop hard gate 时，可设置
`enable_source_gripper_tabletop_sweep: false`，跳过源夹爪逐候选扫描；`enable_tabletop_filter` 仍应保持
`true`，以继续输出 table plane 和 object-top。

Ascend 板端可分别在 `perception_node` 和 `planner_node` 下限制 CPU 数值库线程：

```yaml
host_runtime:
  omp_threads: 4
  blas_threads: 1
```

`embodied_bringup` 会据此设置进程级 `OMP_NUM_THREADS` 和 `OPENBLAS_NUM_THREADS`，并让 GNU OpenMP
worker 使用被动等待，避免 NPU 请求完成后继续抢占 CPU。该配置只影响对应节点进程，不影响 MoveIt、
相机驱动或其他机器人配置。

## 已持物容器放置 SSOT

支持容器放置验证的机器人在顶层 `robot.placement_execution` 声明唯一运行时配置。该配置维护 placement
action、固定 1–5 号关节 `place_joint_positions`、移动时长、腕部 RGB topic、检测分割 endpoint、夹爪反馈超时，
以及二维掩码包含验证的阈值和重采样次数；启用时缺少强制字段会在 launch 前失败。固定放置关节目标由
`placement_execution.motion` 管理：到达 3 号电机 raw 1500 后开爪，随后只将 3 号电机移动到 raw 1700
（`verify_joint_position: -0.533825`）做验证，
最后恢复到 raw 1500。候选放置位规划、
动态 IK/FK、深度/TF、持物门禁和恢复日志不属于该能力。

公开技能由 `skill_catalog/config/skills/place_in_container` 和对应 profile 暴露，要求显式传入 `target_name` 和
`container_name`，并通过 `placement_pipeline` 委托到 `/manipulation/execute_place`。执行器依次移动到固定
`place_container`、打开夹爪、移动 3 号关节到验证位、进行视觉验证，最后返回释放位；`container_name` 只作为
释放后容器检测 query，不改变固定运动，也不触发候选规划。成功要求夹爪反馈确认打开，并连续确认目标物品的
二维分割区域位于指定容器区域内。夹爪 6 号关节只在释放阶段通过 `open_gripper` 单独控制。

真机抓取配置可在 robot YAML 的 `robot.grasp_execution.target_gripper` 下声明目标夹爪几何。
SO101 单动爪使用 `fixed_finger_contact_ee` 作为固定指侧参考点，`closing_axis_ee` 表示从固定指
指向目标宽度中心的方向。`manipulation_execution/pick_executor_node` 会根据 GraspGen 候选的
`target_width_m` 计算有效接触中心：

```text
dynamic_fixed_finger_margin_m = min(
    fixed_finger_margin_max_m,
    fixed_finger_margin_m
        + max(0, fixed_finger_margin_width_ref_m - target_width_m)
        * fixed_finger_margin_width_gain,
)

effective_center = fixed_finger_contact_ee
                 + closing_axis_ee * 0.5 * (target_width_m + width_clearance_m)
                 + closing_axis_ee * dynamic_fixed_finger_margin_m
```

`fixed_finger_margin_m` 是额外远离固定指的基础安全距离，用于降低固定指先碰物体边缘或上表面的风险。
当前 SO101 RealSense 抓取配置默认基础值为 `0.010 m`；目标窄于 `fixed_finger_margin_width_ref_m=0.035 m`
时按 `fixed_finger_margin_width_gain=0.25` 增加安全余量，并由 `fixed_finger_margin_max_m=0.016 m` 封顶。

`target_gripper.fixed_finger_base_side` 是独立的候选硬约束。启用后，执行器在 `base` 坐标系 XY 平面
计算“目标宽度中心到固定指”与“目标宽度中心到 `reference_point_base`”的夹角余弦；低于
`min_alignment_cos` 的候选会在 IK/FK 准备前拒绝。SO101 默认参考机器人 base 原点且阈值为 `0.0`，
因此固定指必须位于物体朝机器人一侧，移动指从外侧闭合。无法获得可靠目标宽度区间时也会拒绝，
避免在固定指方向未知时继续执行。

候选 IK/FK 准备阶段会把预测接触点残差投影到实际闭合轴，并按
`target_gripper.fixed_finger_robust_gap` 的参数计算预测余量：

```text
effective_gap = fixed_finger_gap_m + contact_error_along_closing_axis_m
required_gap = fixed_finger_target_gap_m - max_target_gap_deficit_m
gap_deficit = max(0, required_gap - effective_gap)
pass = gap_deficit <= measurement_tolerance_m
```

朝固定指方向的误差为负，会缩小 `effective_gap`。准备阶段不会据此执行硬拒绝，而是把未经容差放宽的
`robust_gap_headroom` 交给 `prepared_candidate_scoring` 做软排序。启用
`target_gripper.fixed_finger_robust_gap.enabled` 时，执行器在下降完成后、闭爪前用低位实测接触残差再次执行
同一计算；测量不可用或门禁失败会先撤回 pregrasp，再以 `FIXED_FINGER_ROBUST_GAP_REJECTED` 切换候选。
当前两个 SO101 RealSense 抓取 profile 都将该开关设为 `false`，因此下降后直接闭爪。

`candidate_target_offset_base_m` 是候选目标在 `base_frame` 中的三轴平移补偿。它统一作用于 grasp、approach、
grasp 和规划接触点，不改变相机测得的物体宽度端点。SO101 hand-eye profile 使用旧 310P marker test
验证过的 `[0.0, 0.0, -0.008]`；该值属于机器人/手眼执行几何，因此由 robot YAML 管理，action 客户端不提供
覆盖参数。

`execution_scoring.topdown_weight` 控制垂直向下抓取在源候选排序中的软加分。SO101 hand-eye
profile 使用 `0.50`；`candidate_selection.topdown_weight` 保持相同值作为兼容回退。该权重只改变候选顺序，
不跳过 workspace、桌面碰撞、IK/FK 或姿态门禁。

`target_gripper.ik_orientation_guard` 约束 position-only IK 的实际 FK 朝向。SO101 的 joint5 对 TCP 位置
几乎不产生梯度，因此超出执行门限时会保持在 seed 附近；执行器将超限 joint5 seed 翻转 `±π`，让固定指和
活动指换侧，再比较 GraspGen 目标和实际 FK 的接近轴、180° 对称闭合轴直线及固定指内侧关系。当前配置使用：

```yaml
ik_orientation_guard:
  enabled: true
  approach_axis_ee: [0.0, 0.0, 1.0]
  closing_axis_180_symmetric: false
  joint5_home_max_delta_rad: 1.5707963267948966
  joint5_limit_epsilon_rad: 0.001
  joint5_stage_continuity: true
  joint5_stage_max_delta_rad: 1.5707963267948966
  max_approach_error_deg: 40.0
  max_closing_error_deg: 30.0
  moveit_orientation_search:
    enabled: false
    approach_tolerance_deg: 15.0
    free_rotation_tolerance_deg: 180.0
    constraint_weight: 1.0
    max_attempts: 3
```

HOME joint5 由同一 robot YAML 的 `ros2_control.reset_positions["5"]` 提供。初始候选只检查
`abs(candidate_joint5 - home_joint5) <= joint5_home_max_delta_rad + joint5_limit_epsilon_rad`，不把观察位
joint5 当作抓取安全原点；候选选定以后，approach、pregrasp 和 grasp 才使用阶段连续性门阻止真正的
半圈翻腕。Hermes 与监督式客户端都向同一个 `/manipulation/execute_pick` 发送 `PickObject` goal。

MoveIt LMA 仍使用 `position_only_ik: true`。当前 SO101 profile 不启用
`moveit_orientation_search`。OrientationConstraint
不会为 5-DOF 机械臂创造额外自由度，反而可能把本应由最终 FK 门禁解释的姿态误差变成 `NO_IK_SOLUTION`。
最终 IK/FK 结果仍必须通过硬门禁；310P 和 PC profile 均使用 40°/30°。

可靠开口参数由 `prepared_candidate_scoring` 声明，因为准备阶段是从这个块里读取它们的：

```yaml
prepared_candidate_scoring:
  reliable_max_opening_m: 0.072
  moving_finger_min_clearance_m: 0.003
```

这两个值进入 `fixed_finger_envelope_score`，按 `moving_gap = reliable_max_opening_m - far_extent` 折算成
`moving_score`，再与 `fixed_score` 加权成候选软分数 —— **它们是排序偏好，不是硬拒绝**。把它们写到
`target_gripper` 下面会让准备阶段读不到，静默回落到硬编码默认值。

真正的固定指硬拒绝是 `target_gripper.fixed_finger_robust_gap`，在降到抓取位姿之后、闭合之前用实测
接触残差评估一次；不通过则撤回 pregrasp 并以 `FIXED_FINGER_ROBUST_GAP_REJECTED` 继续 candidate
fallback。准备阶段目前没有对应的前置硬门，规划位姿固定指间隙已为负的候选仍会被执行一次才被拒。

闭合轴的 180° 对称只表示两指闭合直线相同，不表示固定指身份可忽略。执行器必须再用实际 FK 位姿执行
`fixed_finger_base_side` 硬检查；固定指仍在外侧或目标宽度区间缺失时拒绝当前候选并继续 candidate fallback。

候选准备加速由同一 `grasp_execution.ik` 配置控制：

```yaml
ik:
  worker_count: 4
  worker_namespace_prefix: /ik_worker
  auto_start_workers: true
```

`embodied_pipeline.launch.py` 会自动启动对应数量的隔离 MoveIt worker。正式执行器为整批候选
固定一份共同 `/joint_states` seed，并按候选排名固定 round-robin 分配到 worker，再按原候选顺序
合并结果；最终补偿与运动不进入 worker。候选进入 worker 前的 SO101 mesh/tabletop 检查
也由该执行器的唯一公共实现完成，使用凸包缓存和批量向量化路径。

SO101 真机的 MoveIt 完成语义也由同一 robot YAML 的 `moveit` 域声明：

```yaml
moveit:
  joint_state_topic: /arm_joint_state_broadcaster/joint_states
  motion_status_hold_s: 0.0
  motion_feedback_timeout_s: 0.3
  motion_feedback_tolerance_rad: 0.12
  motion_require_tf_sync: true
  # LeKiWiSystemHardware does not publish ibrobot_msgs/JointCurrent.
  motion_hardware_feedback_topic: ""
```

配置硬件心跳话题时，网关只在 MoveIt 终态之后同时看到新的硬件读取心跳、收敛关节样本和覆盖该样本时间戳的末端 TF 时
返回成功；LeKiWi 将该可选话题留空，使用新的收敛关节样本与 TF 屏障，不再依赖固定 `0.3 s` sleep。由该屏障覆盖的 `contact_realign.settle_sec` 和
`pose_diagnostics.settle_sec` 可设为 `0.0`；相机与夹爪稳定等待不在此屏障覆盖范围内。仿真启动会
清空硬件心跳话题，只保留关节与 TF 屏障。

LeKiWi 抓取配置还使用顶层 `motion_mode` 作为常驻控制器的命令授权 SSOT：

```yaml
motion_mode:
  enabled: true
  navigation_enabled_on_startup: false
  navigation_enabled_topic: motion_mode/navigation_enabled
  navigation_mode_ack_topic: motion_mode/base_navigation_enabled
  set_navigation_enabled_service: motion_mode/set_navigation_enabled
  manipulation_controllers: [arm_trajectory_controller, gripper_trajectory_controller]
  navigation_controllers: [base_velocity_controller]
  manipulation_control_mode: moveit_planning
  navigation_control_mode: base_navigation
  transition_timeout_s: 2.0
```

默认抓取模式激活机械臂/夹爪控制器并把底盘控制器保持 inactive。任务切换只调用上述服务；
MoveIt Gateway 按顺序停止当前运动域、切换 controller manager 中的 active/inactive 控制器集合，并等待
底盘 bridge 的零速确认。硬件节点不重启，且 1～6 轴与 7～9 轴始终互斥。

`robot.grasp_execution.prepared_candidate_scoring` 控制 IK/FK 后软排序。SO101 使用候选目标宽度区间和
候选规划姿态计算固定指到目标前缘的间隙，并以动态 margin 为期望值计算包络分数。该分数与
接触点 XY/Z 质量、目标体积质心距离和 GraspGen 置信度加权。`robust_gap_headroom_weight` 和
`robust_gap_headroom_scale_m` 进一步把 IK/FK 预测接触残差对应的安全间隙余量归一化后纳入排序；
软排序本身只改变执行顺序。启用 `fixed_finger_robust_gap` 时，独立硬门禁使用下降后的低位实测残差，
不直接使用准备阶段的预测余量。

## 导航端点 SSOT

`robot.navigation.command_server.action_name` 是 `ExecuteNavigation` action 名的**唯一**配置来源
（默认 `/navigation/execute`）。`robot_config.loader` 在加载阶段把它投影到
`robot_execution_endpoints.navigation_action_name`，并参与 canonical execution-context digest。

旧字段 `embodied.execution.navigation_action_name` 已退役。`load_robot_config_dict()` 检测到该键时直接
报错，要求改用 `robot.navigation.command_server.action_name`。`skill_library` 的 nav_* primitive
dispatch、`navigation_command_server` 的 Action server 注册以及 `nav_cmd` CLI 都从该 SSOT 读取，
不允许在节点参数或代码里重新声明该 action name。

### 公开 API

`robot_config.loader` 暴露以下与导航端点相关的公开 API：

- `navigation_endpoint_projection(robot_config_dict)`：从完整 robot YAML 解析出导航端点身份
  （`action_name`、`costmap_readiness_timeout`、`cancel_cleanup_timeout_sec`、`active` 布尔）。
  只在 `robot.navigation.command_server` 存在且 `action_name` 非空时返回非空 projection。
- `robot_context_schema_version(robot_config_dict)`：根据解析后的 stage 与 endpoint projection 决定
  `context_schema_version`。`hybrid` stage 返回 `3`，独立 navigation stage 返回 `2`，其他 stage 返回 `1`。
  返回值写入 `SkillRobotContext.context_schema_version`，并进入
  canonical execution-context digest preimage；切换版本会强制新 registry generation。
- `robot_execution_endpoints(robot_config_dict)`：返回 V1 10-role endpoint map，加上 V2/V3 时的
  `navigation_action_name`。V1 profile 上 `navigation_action_name` 必须为空，V2/V3 profile 上必须非空。
- `validate_navigation_endpoint_contract(robot_config_dict)`：在 loader 阶段校验导航端点契约的
  一致性。拒绝下列情况：V1 profile 上声明非空 `navigation.command_server.action_name`、
  V2 profile 上 `action_name` 为空、`action_name` 与 `embodied.execution.navigation_action_name`
  同时存在但取值不一致、或 `costmap_readiness_timeout` 非正。任一不满足时 `load_robot_config_dict()`
  在返回配置前直接报错。

### `nav_stage` 参数与 stage 解析

`nav_stage` 是 `robot.launch.py` 的 launch argument。标准导航配置声明 `mapping` / `navigation`；
移动抓取与导航统一配置 `lekiwi_nav_grasp` 还声明 `grasp` 和 `hybrid`。缺省时由
`robot.default_nav_stage` 决定。stage 在 canonical loader 中解析，builder 和具身运行时只会看到
解析后的单一配置：

- `grasp`：兼容入口，仅选择腕部 RealSense 与 MoveIt 机械臂/夹爪控制器；
  `base_velocity_controller` 保持 inactive，不启动 MID-360、FAST-LIO 或 Nav2。
- `mapping`：启动 SLAM Toolbox（LiDAR）或 RTAB-Map mapping（RealSense），不启动 Nav2 / AMCL /
  Collision Monitor。统一配置只选择 MID-360 与底盘控制器，并关闭抓取/放置运行时；`/cmd_vel`
  直接由 `cmd_vel_bridge` 消费。
- `navigation`：兼容入口，启动 AMCL、Nav2 完整栈与 `navigation_command_server`，但不启动 Hermes；
  `/cmd_vel` 经 Collision Monitor 收口为 `/cmd_vel_safe` 再交由 bridge 消费。
- `hybrid`：统一配置的默认入口，同时保留腕部 RealSense、MID-360、抓取/放置依赖、AMCL、Nav2、
  `navigation_command_server` 和单一 Hermes runtime。统一 skill catalog 同时暴露操作与导航技能，
  skill executor 在每个技能执行前通过 `motion_mode/set_navigation_enabled` 切换控制权。

统一配置通过 active/inactive controller 集合与 `motion_mode` bridge gate 双重隔离 1～6 轴和 7～9 轴。
`hybrid` 支持一次启动后交替执行抓取和导航，但不允许两类运动并发。

`grasp` / `mapping` 使用 V1 context，独立 `navigation` 使用 V2 context，`hybrid` 使用 V3 context。
V3 同时携带 V1 操作 primitive 与 V2 导航 primitive，并在 digest preimage 中加入
`supported_control_modes`，防止与旧 snapshot 混用。

### `base_navigation` 控制模式

`base_navigation` 是 V2/V3 context 授权导航 primitive 的控制模式。它在 `control_modes` 中声明
底盘速度控制器集合（典型为 `joint_state_broadcaster` + `base_velocity_controller`）。V2 runtime
要求 `active_control_mode == 'base_navigation'`；V3 hybrid runtime 则在每个 nav_* 技能下发前通过
`motion_mode` 切换到该模式。

```yaml
robot:
  default_control_mode: "base_navigation"

  control_modes:
    base_navigation:
      description: "底盘导航控制模式（V2/V3 context）"
      controllers:
        - joint_state_broadcaster
        - base_velocity_controller
      inference:
        enabled: false
```

`base_navigation` 与 `moveit_planning`、`teleop`、`model_inference` 互斥（同一时刻只能激活一种
控制模式）。V1/V2 profile 仍要求启动模式与 profile 一致；V3 hybrid profile 允许 catalog 中的技能
分别声明 `moveit_planning` 或 `base_navigation`，由 skill executor 在安全校验后、下发 primitive 或
delegated executor 前调用 `motion_mode` 服务。切换失败时技能 fail closed，不下发运动。

### context_schema_version 与 digest preimage

`context_schema_version` 由解析后的 stage 与 `navigation_endpoint_projection` 决定，并写入
`SkillRobotContext.context_schema_version`。`robot_config.loader.robot_config_digest` 把
`context_schema_version`、`execution_endpoints`（含 `navigation_action_name`）、命名位姿/目标、
关节限位、工作空间、控制模式、timeout policy 与相对运动语义一起纳入 digest preimage。
V3 额外包含 `supported_control_modes`。切换 `context_schema_version` 会改变 digest，强制
`skill_catalog` 重新编译并生成新 registry generation。V1、V2 与 V3 snapshot 在 registry 层
永远不可互换校验。

## 控制模式配置

robot_config 包支持双控制模式，以满足不同 AI 模型的需求：

### 可用控制模式

#### 1. teleop 模式（人工遥操作）

**适用于：** 人工遥操作设备（leader arm、游戏手柄、VR设备）

**特点：**
- 实时直接控制
- 零延迟直通（< 5ms）
- 支持多种输入设备
- 内置安全过滤（关节限位）

**配置：**
```yaml
robot:
  default_control_mode: "teleop"

  control_modes:
    teleop:
      description: "人工遥操作模式（直接控制）"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: false

  teleoperation:
    enabled: true
    # 单设备使用 active_device；双臂等多输入设备使用 active_devices。
    active_device: "so101_leader"
    # active_devices: ["left_leader", "right_leader"]
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"
        target:
          arm_joint_names: ["1", "2", "3", "4", "5"]
          gripper_joint_names: ["6"]
          arm_command_topic: "/arm_position_controller/commands"
          gripper_command_topic: "/gripper_position_controller/commands"
```

多设备遥操作时，`active_devices` 按名称选择 `devices` 中的多个输入设备。每个
设备可通过 `target` 指定要控制的关节组和控制器命令话题；未指定时回退到机器人级
`joints.arm` / `joints.gripper` 以及默认单臂控制器话题。

通用 `target.publish_groups` 可声明任意有序关节组；`target.actuator` 可直接引用
`auxiliary_actuators` 的关节顺序与命令话题，避免重复配置。launch 会拒绝多个 active devices
占用同一 command topic，但允许 topic 不重叠的 arm 与 hand 输入并存。非 ros2_control
执行器配置在 `auxiliary_actuators`，driver package/executable 和关节数不受 Aero 7 维限制；其关节放在 `joints.hand`，不得加入
`joints.all`。外部 source/actuator 可用 `active_control_modes` 声明所属控制模式；仿真会跳过
`mock: false` 的真实外设，但仍可启动显式 `mock: true` 的离线组件。Aero Hand 配置将两个外设
限定在 `teleop` 真机路径，且不会在该路径静默降级为 mock。启动前设置外部手套 SDK 路径：

```bash
export MHANDPRO_SDK_LIB=/absolute/path/libVDMocapSDK_mHandPro.so
export AERO_HAND_RIGHT_PORT=/dev/serial/by-id/<aero-hand-id>
ros2 run robot_teleop calibrate_glove --side right --lib-path "$MHANDPRO_SDK_LIB" \
  --raw-output ~/.calibrate/aero_hand_right_sdk_capture.json
```

默认完整标定写入 `feature_schema: aero_compact_v3`，使用 20 个骨骼节点和 5 个厂商虚拟指尖。
用户标定保存四个原始拇指特征及 MCP/IP 组合坐标的 neutral/active 端点；robot_config 保存
MCP/IP 肌腱权重、Aero 输出比例、deadband 和最大步长。经真机轴向验证，掌内 pitch/yaw 分别驱动 CMC 外展/屈曲轴，
MCP/IP 屈曲只驱动组合腱轴。
`hand_sources.mhandpro` 独占厂商进程并发布统一 `HumanHandState`，`hand_retarget` 再通过
目标插件生成 Aero 7 维或其他机械手的任意维度输出。`hand_retarget` 必须声明 `side`，可用
`source_name` 绑定发布者；消费端同时校验 `human_hand_geometry_v1` schema，防止跨手或跨语义接线。
厂商姿态偏置不能跨 SDK 进程复用。真实遥操
启动后，手套输出保持锁定。`startup_p_pose: interactive` 由统一 launch 在启动硬件前提示一次，source
节点随后在同一个 SDK 进程自动完成 P-pose 及质量检查；`startup_p_pose: manual`（默认）则通过：
`ros2 run robot_teleop calibrate_glove --side right --runtime-service
/hand_sources/mhandpro/calibrate_p_pose`。映射 JSON 与 raw capture 仍可长期复用；后续
retarget 调参使用 `ros2 run robot_teleop analyze_glove_capture
--input ~/.calibrate/aero_hand_right_sdk_capture.json --side right --update-calibration
~/.calibrate/aero_hand_right_calibrate.json` 离线完成，不重复采集手势。

真实 launch 会统一检查 `ros2_control`、active teleop devices、`auxiliary_actuators` 和
`hand_sources` 的串口或 `exclusive_resources`，任何重复资源都会在节点启动前拒绝。一个
mHandPro source 用 `sides: [right]`、`[left]` 或 `[left, right]` 选择单手或双手；双手默认
`failure_policy: require_all`，任一侧断连都会锁住两侧输出。`allow_available` 允许仅连接或保留任一
可用侧，不会为恢复缺失侧而中断健康侧。source 保持健康话题
`/hand_sources/mhandpro/<side>/health`，并在线程中按有界指数退避自动重连，避免阻塞 ROS timer；需要
P-pose 的配置在重连后重新进入门禁。真实 Aero auxiliary actuator 还必须能从当前控制模式的
safety SSOT 解析全部关节限位；launch 将有序上下限传给硬件节点，由硬件边界再次限幅。缺少任一
关节限位时拒绝启动。
`aero_hand_teleop.yaml` 用一个 `hand_profiles` SSOT 同时声明 `right`、`left`、`dual`，统一 launch 的
`hand_profile` 参数只选择 profile，不复制 SDK、频率、安全或重定向参数；双手 profile 仍只启动一个
共享 source worker。厂商原始 `MHandProFrame` 默认关闭；只有诊断或原始录制配置显式设置
`hand_sources.mhandpro.publish_raw_frame: true` 时才发布，并应同时把对应 `/frame` topic 加入录制清单。

SO-101 集成配置 `so101_arm_aero_hand.yaml` 使用 `base_config: so101_single_arm` 加 hand-specific
overlay；不得再复制整份 SO-101 YAML。loader 会先解析基线，再应用显式 overlay，`teleoperation.devices`
等有序列表只有通过 `__append__` 才会追加。生产路径固定为
`hand_sources.mhandpro -> HumanHandState -> hand_retarget -> auxiliary_actuators`；旧的
`mhandpro_glove` 设备类型已移除，作为 active teleop device 会在 launch validation 阶段明确拒绝。
完整的首次标定、日常启动步骤和真机安全注意事项见
[`robot_teleop` 真机快速使用流程](../robot_teleop/README.md#真机快速使用流程)。

无硬件测试应在测试配置中显式设置 Aero actuator 和 `hand_sources.mhandpro` 的 `mock: true`，并将 retarget device
`calib_file` 指向 `$(find robot_teleop)/config/mhandpro_mock_right_calibration.json`。不要把测试
fixture 用作真实设备标定。

**启动命令：**
```bash
# 遥操作模式（episodic 录制）
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=teleop \
  record:=true \
  record_mode:=episodic \
  use_sim:=false

# 遥操作模式（episodic + Rerun）
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=teleop \
  record:=true \
  record_mode:=episodic \
  record_visualizer:=rerun \
  use_sim:=false

# 另一个终端启动录制客户端
ros2 run dataset_tools record_cli

# 录制完成后转换为 LeRobot 数据集
ros2 run dataset_tools bag_to_lerobot \
  --bags-dir ~/rosbag/episodes/so101_single_arm \
  --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
  --out /path/to/output_dataset
```

#### 2. model_inference 模式（高频位置控制）

**适用于：** 端到端模仿学习模型（ACT、pi0、Diffusion Policy）

**特点：**
- 高频控制（50-100Hz）
- 低延迟（1-3ms）
- 直接基于话题的位置命令
- 反应迅速、运动流畅

**配置：**
```yaml
robot:
  default_control_mode: "model_inference"

  control_modes:
    model_inference:
      description: "高频端到端控制模式（ACT/pi0）"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: true
        pipelines:
          policy:
            model_path: models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515
            deployment: cpu
            execution_mode: monolithic # 或 distributed
            request_timeout: 5.0
            default_task: ""
            runtime_options: {}
      executor:
        type: topic
        mode: model_inference
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 50
        control_frequency: 20.0
      dispatch:
        scheduler: continuous   # continuous | wait_for_feedback，与 executor.type 存在配对约束
        chunking: full_chunk    # 缺省 full_chunk
        blending: none          # none | temporal_ensemble，缺省 none；唯一融合开关
```

### 动作分发策略 SSOT

`executor`、可选的 `dispatch` 和 `inference` 是 `robot.control_modes.<mode>` 下的同级块。
`robot_config.dispatch_strategies.resolve_dispatch_strategies` 统一解析策略，两个 launch
分支和直接节点入口均校验合法组合：legacy 支持 `topic` + `continuous` 或
`benchmark` + `wait_for_feedback`；scheduled 仅支持 `topic` + `continuous`。

| YAML 字段（相对于控制模式） | 默认值 | 节点参数 / 作用 |
|----------------------------|--------|-----------------|
| `executor.type` | `topic` | `executor_type`，输出通道 |
| `dispatch.scheduler` | `continuous` | `scheduler_mode`，逐 tick 调度；另支持 `wait_for_feedback` |
| `dispatch.chunking` | `full_chunk` | `chunking_strategy`，支持 `full_chunk` / `auto_horizon` |
| `dispatch.blending` | `none` | `blending_strategy`，另支持 `temporal_ensemble` |
| `executor.watermark_threshold` | `20` | 剩余动作数的补货阈值，0 表示耗尽后补货；不改变 blending 选择 |
| `executor.queue_size` | `100` | queue 容量，不限制 smoother 存储 |
| `executor.control_frequency` | `100.0` | 控制 tick 频率（Hz），不是推理频率 |
| `executor.chunk_size` | `100` | smoother 权重表大小，不控制模型实际返回步数 |
| `executor.temporal_ensemble_coeff` | `0.01` | 指数融合系数 |
| `executor.smoothing_device` | `''` | smoother 设备，空值使用输入 tensor 设备（NumPy 为 CPU） |

入口与策略组合矩阵：

| 入口 | executor.type / dispatch.scheduler | dispatch.chunking | dispatch.blending / 存储 |
|------|------------------------------------|-------------------|-------------------------|
| legacy continuous | `topic` / `continuous` | `full_chunk` 或 `auto_horizon` | `none` queue 或已有 manager 透传；`temporal_ensemble` smoother |
| legacy benchmark | `benchmark` / `wait_for_feedback` | 仅 `full_chunk`（`auto_horizon` 组合报错） | 同上，消费等待匹配反馈提交 |
| scheduled | 仅 `topic` / `continuous` | `full_chunk` 或 `auto_horizon` | `none` queue；`temporal_ensemble` 独立 smoother |

- 策略字段缺失、null 或空字符串采用默认：executor 为 `topic`，scheduler 为
  `continuous`，chunking 为 `full_chunk`，blending 为 `none`。
  false、0、list、dict、未知名及大小写变体拒绝，不再作为缺省值。
- `executor.temporal_smoothing_enabled` 已删除；配置加载（包括继承的 base_config）和
  launch 均拒绝该键，不论值是什么或是否与 blending 一致。迁移时删除旧键，将原来的
  `true` 改为 `dispatch.blending: temporal_ensemble`，`false` 改为 `dispatch.blending: none`。
  `temporal_ensemble_coeff`、`chunk_size`、`smoothing_device` 等调参字段保持不变。
- 历史 `executor.type: action` 仅由 launch 边界映射为 `topic`。直接节点参数
  `executor_type` 不接受此别名。ROS 参数类型约束仍适用，YAML null 默认语义不适用于 ROS CLI override。
- `inference.scheduler.enable` 选择产品入口，不等同于 `dispatch.scheduler`。
  scheduled 显式选择 benchmark 或 wait_for_feedback 会报错，不再静默忽略。
- 两节点的 `~/toggle_smoothing`（Empty）先校验有效组合，拒绝时保留配置及计划并记日志，
  不伪造响应错误字段；start/stop/get_status 为 Trigger。legacy reset 为 Empty，scheduled
  restart_session 为 Trigger。合法 toggle 不代表跨 store 动作迁移，详见 action_dispatch 矩阵。
- 非法/非有限 chunk 的受控拒绝是有意行为修正：legacy continuous 保留旧计划，benchmark
  inference-failed，scheduled safe-stop/close。queue 超容仍分别为 legacy 保留最新动作、
  scheduled 失败关闭；smoother 不受 queue 容量约束。
- AutoHorizon 已开放 `dispatch.chunking: auto_horizon`（仅 `topic` executor；
  `benchmark` 组合报错），端到端两段配置示例见 action_dispatch README
  「AutoHorizon 集成边界」；attention 估计与 runtime options 归 inference_service
  （见其 README）。RTC 配置名尚未开放，远端缓存确认和相对动作重锚定也未实现。
  runtime options 会参与 profile 兼容性指纹，开关变化需要重新标定。ensemble 没有精确
  单源坐标，本地 revision/接纳身份不能作为 RTC 远端确认。

耗尽后补货与异步提前补货的完整解释统一放在
[推理补货与融合配置示例](../action_dispatch/README.md#推理补货与融合配置示例)
（[English](../action_dispatch/README.en.md#inference-replenishment-and-blending-examples)）。
运行时存储、失败处理和 toggle 限制见
[调度策略分层与状态归属](../action_dispatch/README.md#调度策略分层与状态归属)及
[运行时切换](../action_dispatch/README.md#运行时切换)。

### 推理调度控制面

调度只有一个功能开关：`control_modes.<mode>.inference.scheduler.enable`，缺失时按 `false` 处理。
仿真暂不支持启用该开关：`runtime_target=simulation`（含 Gazebo、MuJoCo、mock）运行推理时必须设置
`scheduler.enable=false`，否则 launch 在生成控制与仿真节点前报错。仿真场景不管理调度 session/binding。
`inference.enabled` 仍只表示该控制模式是否启用推理，不是调度模式开关。

当 `scheduler.enable` 缺失或为 `false` 时，launch graph 与原路径相同：

```text
executor.inference_pipeline
  -> pipeline_policy_node /inference/<pipeline_id>/dispatch
  -> pipeline_policy_node /inference/<pipeline_id>/reset
  -> action_dispatcher_node
```

该分支不创建 Global Scheduler、ScheduledActionDispatcher、product session、scheduled endpoint 或 serving
status。显式保留 `scheduler` 块并设置 `enable: false` 时，完整调度配置可以原样保留为 dormant 配置，
不会生成 runtime policy，也不会改变 legacy launch graph、节点参数、接口、线程数或 backend 执行方式，
因此启停只需修改这一处开关。若整个 `scheduler` 块缺失，scheduled 字段仍按 unknown field 拒绝，以保留
旧配置的严格拼写检查。
仓库当前发布的机器人 YAML 都未启用该开关，因此默认生产启动仍是这条 legacy 路径；下面的 `true` 片段是配置
契约示例，不是默认切流。

开启调度时，每个 pipeline 必须显式声明 pipeline-scoped endpoint、兼容组、真实硬件资源和公开容量。
`profile_path` 可选；只有实际参与 priority-0 deadline 准入的候选才需要有效的离线 profile：

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      scheduler:
        enable: true
        global_endpoints:
          readiness: /inference/scheduler/ready
          open_session: /inference/session/open
          dispatch: /inference/dispatch
          close_session: /inference/session/close
      pipelines:
        policy:
          model_path: models/policy
          deployment: ascend_310p3
          execution_mode: monolithic
          transport:
            open_session: /inference/policy/session/open
            dispatch: /inference/policy/scheduled_dispatch
            close_session: /inference/policy/session/close
            serving_status: /inference/policy/serving_status
          required: true
          compatibility_group: so101_action
          hardware_resource_id: ascend:0
          hardware_profile_fingerprint: <calibration-environment-sha256>
          profile_path: /absolute/path/to/measured-profile.yaml  # optional; required for priority-0 admission
          public_capacity:
            session_control: {max_in_flight: 1}
            action_generation: {max_in_flight: 1}
    executor:
      type: topic
      mode: model_inference
      inference_pipeline: policy
      inference_fallback_chain: []
      inference_priority: 0
      inference_retry:
        max_not_started_attempts: 3  # 首次请求之后最多自动重试 3 次
        initial_backoff_ms: 50
        max_backoff_ms: 500
```

Scheduler 开启时只接受 schema v3 whole-graph monolithic deployment，生产路径为 Open/Dispatch/Close。
对于 `scheduling.stage_policy: independent`，可在 `scheduling.stages.<producer>` 下设置
`max_snapshot_age_ms: 5000`（默认值，正整数）。producer 是 manifest `execution` 中的第一个角色；
terminal 或 sequential stage 不允许设置该字段。时效从观测采样开始计算，包含 producer 排队与执行时间，
阈值进入 runtime policy fingerprint 并经 launch 下发。刷新不能使过期的同一观测重新有效。
分布式 pipeline 保持 legacy protocol v2，不能与 `scheduler.enable=true` 组合。`inference_priority` 使用 `0` 表示
最高优先级，数值越大优先级越低；通用 wire 范围是非负 int32，具体 backend 范围和映射由 backend 校验。
默认 `global_policy: fifo` 且 `priority_zero_deadline_admission.enable: false` 时，priority-0 直接下发 target，
不建立资源等待队列、不使用 profile，也不接受非空 fallback chain；启用 scheduler 后该无效组合会在配置加载时报错。
FIFO 开启 profile 准入时，同一 `hardware_resource_id` 上已准入的 priority-0 按 reservation FIFO 串行下发，
实际轮到执行时重新检查 deadline，支持 fallback，但仅支持 sequential pipeline。
`global_policy: edf` 按绝对 deadline 排序尚未开始的 priority-0，同 deadline 按 FIFO，支持 fallback；
EDF 必须关闭 profile 准入，不抢占执行中的请求，不预测或保证完成时间。
deadline 驱动（EDF 或 FIFO profile 准入）的 priority-0 只服务 sequential pipeline：
independent pipeline 把一个请求拆到两个 worker 重叠执行，per-request deadline 排序无法评估也无法兑现。
`inference_priority=0` 时 target 或 fallback 链选择 independent pipeline 会在配置加载时报错，
Global 也会按 serving status 能力在运行时跳过 independent 候选。
`hardware_profile_fingerprint` 独立标识离线标定环境，不能使用
资源 ID 代替。
profile entry 使用 `global_proxy` scope。action-generation entry 必须声明 pipeline serving status 发布的
`input_contract_fingerprint` 和标定覆盖的 `prompt_bytes_max`；session-control entry 使用空 fingerprint 和
`prompt_bytes_max: 0`。profile identity 使用 `profile_compatibility_fingerprint`，不再绑定 endpoint 名称、
required 状态或 compatibility group。其他 priority 只下发 target，
不做 fallback 或 deadline 准入。缺失或无效 profile 不影响 readiness、EDF 和非零优先级请求；仅在 FIFO
profile 准入实际遍历到该候选时将其判为不可准入并继续 fallback，全部不可准入时返回 `no_feasible_deadline`。
Global Dispatch ingress 共四个有界 context，lower-priority 最多占两个；因此低优先级请求不能耗尽 priority-0
保留容量。Open/Close 和 pipeline-local endpoint 仍各使用两个有界 context。
公开 Open 只创建逻辑 session，不使用 `executor.inference_pipeline` 或 fallback 做初始模型绑定；pipeline
Open/reset 在某个 Dispatch 首次选中该候选时惰性执行。

本地 compiled artifact 需要 manifest content SHA-256。Global readiness 要求真实后端至少报告一个通用
priority level 和匹配的资源身份；当默认 `inference_priority` 大于 0 时，还会要求默认
`inference_pipeline` 在线并支持该优先级，不要求其他 pipeline 支持多优先级。
`public_capacity.action_generation.max_in_flight` 可以大于 1，但 pipeline 启动时
会验证它是正整数且不超过 backend 的 `max_in_flight_per_instance`；pipeline Dispatch ingress 至少保留两个
context 处理重复请求，并随更高执行容量扩展。没有可用 execution slot 时立即返回错误，不排队。

**启动的控制器：**
- `arm_position_controller` (JointGroupPositionController)
- `gripper_position_controller` (ForwardCommandController)

机器人 recording 配置中的 `lerobot_norm_mode` 决定 LeRobot 动作/观测与 `ros2_control`
弧度命令之间的转换方式。`range_m100_100` 使用机械臂 `[-100,100]`、
夹爪 `[0,100]`；`degrees` 对机械臂关节使用 centered degrees，但
`joints.gripper` 中的夹爪关节仍保持 `[0,100]` 开合语义。
真机模式从 `ros2_control.calib_file` 读取舵机校准范围；`use_sim:=true`
仿真模式不依赖该校准文件，而是从生成后的 URDF 关节 `limit` 读取弧度范围。

Observation 的 `align` 配置同时服务离线数据和在线推理，但两个时间阈值含义不同：

- `tol_ms`：`strategy: asof` 的时间对齐容差，离线录制/转换和在线采样都会使用；
  值小于等于 `0` 时退化为 `hold`。`hold` 和 `drop` 不使用该值。
- `max_age_ms`：在线推理允许复用的最大样本年龄，基于节点本地接收时钟计算，不能通过
  回拨推理请求时间戳绕过。缺失或只有未来时间戳的样本始终会
  被拒绝；大于 `0` 时还会拒绝超龄样本。以上情况都返回 `observation_not_ready`，而不是补零运行模型。

```yaml
contract:
  observations:
    - key: observation.images.top
      align:
        strategy: hold
        tol_ms: 1500
        max_age_ms: 500
```

LeRobot 转换 metadata 中的标定来源字段保持稳定契约。这里的
`calibration_source` / `calibration_sources` 是数据集 metadata 输出字段，
不是 `ros2_control` YAML 输入 schema：

- `calibration_source`：兼容旧消费者的单字符串字段，始终取第一个解析到的标定文件路径。
- `calibration_sources`：完整的多标定文件路径列表，多标定源场景应优先读取该字段。

单臂旧式 `ros2_control.calib_file` 会映射到 `arm` 命名空间：

```yaml
robot:
  ros2_control:
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
```

双臂或更多来源通过 `ros2_control.xacro_args.calib_file_<namespace>` 声明。后缀即 LeRobot 转换使用的关节命名空间，也会作为 xacro/URDF 参数名的一部分。`<namespace>` 允许字母、数字和下划线；建议优先使用 `left`、`right`、`front` 这类与本体/机械臂语义相关的名称，便于和 `joint_names`、数据集 metadata 及策略特征对齐：

```yaml
robot:
  ros2_control:
    xacro_args:
      calib_file_front: $(env HOME)/.calibrate/front_calibrate.json
      calib_file_left: $(env HOME)/.calibrate/left_calibrate.json
      calib_file_right: $(env HOME)/.calibrate/right_calibrate.json
      calib_file_1: $(env HOME)/.calibrate/extra_calibrate.json
```

从磁盘加载 SO-101 数字关节标定时，`robot_config` 会按每个来源的 key 后缀转换为内部命名空间键，例如 `"1"` 到 `"6"` 映射为 `joint1_arm` 到 `joint6_arm`、`jointN_left` / `jointN_right` 或 `jointN_1`。`calib_file_<namespace>` 支持任意数量的唯一命名空间；如果合并后的标定键冲突，加载会直接报错。`calib_file_1` 的 namespace 就是 `1`，不会额外推断为 `left` 或 `right`。

LeRobot 转换 metadata 中的标定来源字段保持稳定契约：

- `calibration_source`：兼容旧消费者的单字符串字段，始终取第一个解析到的标定文件路径。
- `calibration_sources`：完整的多标定文件路径列表，多标定源场景应优先读取该字段。

当机器人通过 `ros2_control.calib_file` 配置单个标定文件时，这两个字段都指向同一来源；当机器人通过 `ros2_control.xacro_args.calib_file_1`、`calib_file_2` 等按编号配置多个标定文件时，不需要额外合并标定文件，`calibration_source` 仍保持首个路径，完整有序列表写入 `calibration_sources`。

**命令接口：**
```bash
# 机械臂位置命令
ros2 topic pub /arm_position_controller/commands std_msgs/msg/Float64MultiArray "data: [1.0, 2.0, 3.0, 4.0, 5.0]"
```

### Tracing（ros2_tracing / LTTng）

`robot.launch.py` 支持两个 tracing 相关 launch 参数：

- `enable_tracing:=true`：在启动时开启 LTTng tracing session
- `trace_session_name:=...`：覆盖默认 session 名 `ib_robot_trace`

按包内架构，`robot.launch.py` 只保留编排职责；LTTng session 的创建、命名冲突处理、
以及 shutdown 时的 stop/destroy 生命周期都由
`robot_config/launch_builders/tracing.py` 统一管理。

`enable_tracing:=true` 仅为本次 launch 的业务子进程设置 `IB_TRACE_ENABLED=1`，
包括带独立环境的推理进程和 controller readiness 后启动的进程。移除提前对父进程
`os.environ` 的赋值，使用退出时恢复的 launch scope，避免污染同进程后续 launch。
不改变 control mode、节点参数或模型加载策略。
保留既有失败契约：LTTng 会话检查、创建、事件启用或 start 失败仍中止 launch；
失败清理只针对本次成功 create 的会话，不销毁同名已有会话。不新增 strict 配置。

启动不再自动导出 topology sidecar，也不依赖 topology 模块。普通 trace 分析无需机器人
YAML 或 topology 元数据。需要配置声明图时，可独立使用已有 `load_robot_section`
展开 sibling `base_config`，不经过包含模型可用性检查的完整业务 loader：

```bash
source .shrc_local
ros2 run robot_config ibrobot-trace-topology \
    src/robot_config/config/robots/so101_arm_aero_hand.yaml \
    --control-mode model_inference > /tmp/ibrobot-topology.json
```

该命令只生成可选元数据，不启动机器人、不加载模型、不放宽业务 loader 的校验。
输出标注 `metadata.provenance=declared`、`runtime_verified=false`，不是实际
started nodes 清单；不应用 launch CLI 覆盖（除显式 `--control-mode`）、nav stage 或
运行时派生配置。独立导出失败不会拒绝机器人启动，也不会停止或销毁已成功录制的 LTTng 会话。
`ibrobot_tracing` Core 不直接读取机器人 YAML。

### Voice ASR（语音识别）

`robot_config` 通过 `robot.voice_asr` 作为语音识别节点的机器人级单一配置来源，并由
`robot_config/launch_builders/voice_asr.py` 注入到 `voice_asr_service` 节点。

常用启动覆盖参数：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  voice_asr_auto_start:=true \
  voice_asr_realtime_pre_roll_seconds:=0.5
```

| Launch 参数 | 作用 |
| --- | --- |
| `voice_asr_auto_start` | 临时覆盖 `robot.voice_asr.enabled`，设为 `true` 时强制启动 ASR 节点 |
| `voice_asr_realtime_pre_roll_seconds` | 临时覆盖实时识别 pre-roll 时长 |

`voice_asr_service` 的包级默认值与 `robot_config` 中的 `VoiceASRConfig` 默认值保持同步；
具体机器人仍应以 `config/robots/<robot>.yaml` 中的 `robot.voice_asr` 为准。实时麦克风设备由
`robot.audio_io` 及其引用的 microphone peripheral 统一配置。

### Voice TTS（语音合成）

`robot_config` 通过 `robot.voice_tts` 启停 `voice_tts_service`。启用时必须显式提供 ZipVoice
`bundle_path` 和 manifest 中的 named `deployment`；不接受独立的 `backend` 或 `device` 选择，
也不会在加载失败后切换后端。完整系统从 `robot.launch.py` 启动，包级 launch 仅用于调试。

TTS 对外提供 `/voice_tts/synthesize` typed service。请求和响应携带音频字节而不是服务端文件路径，
并通过文本、prompt、分段数和响应字节上限约束单个 DDS response。真实模型未就绪时服务返回
`MODEL_NOT_READY`；部署身份和 readiness 由响应中的 `ModelRuntimeInfo` 报告。
同一配置还启动 `/voice_tts/play` 服务，接收播放端本机 WAV 绝对路径，通过 `audio_common`
共享播放 Topic 同步发布并返回明确的
成功状态、错误码和消息；该播放节点不依赖模型 session。
launch builder 只解析配置和创建节点，不提前打开模型 bundle。节点启动时校验 bundle 并加载 session，因而
`exit_on_init_failure=false` 能在模型存储暂不可用时保留服务并返回 `MODEL_NOT_READY`；
TTS 由通用 `inference_service/model_service_node` 承载，节点启动时加载 named deployment，节点退出时等待当前
合成结束并释放模型资源。
该配置不会启用请求级初始化重试；修复 bundle、依赖或设备后必须重启 TTS 节点才能恢复。
相对 `bundle_path` 以 `.shrc_local` 设置的绝对 `WORKSPACE` 为根目录解析，例如默认值对应
`$WORKSPACE/models/zipvoice`。
当前经 310P1 真机核查的 `ascend_310p` deployment 支持固定 bundle prompt、中文/数字/常用标点和 24 kHz
WAV；它尚不支持请求级 prompt，调用时返回 `UNSUPPORTED_PROMPT`。该限制属于 deployment capability，
不是 `robot_config` 的隐式后端选择。

`voice_tts` 同时维护三个语义不同的超时字段：`playback_timeout_sec` 是 `/voice_tts/play`
同步播放超时；`synthesis_timeout_sec` 是 Hermes `post_llm_call` TTS hook 等待
`/voice_tts/synthesize` 返回合成结果的超时（由 `hermes-robot-configure` 写入 wrapper 环境变量）；
`tts_timeout_sec` 是 `embodied_agent` 视觉游戏播报节点的内部 RPC 超时。三者互不替代。

### 真机手眼配置

SO101 抓取使用同级独立配置 `config/robots/lekiwi_handeye_realsense_grasp.yaml`。用户应直接在
这份 YAML 中填写从动臂串口、相机序列号和 leader 配置；`scripts/handeye_calibrator.py` 质量检查
通过后会就地更新 `peripherals[name=wrist].transform`。多台物理机器人应分别复制独立 YAML，避免
不同实例的端口和标定值互相覆盖。`config_path` 仍可用于加载 workspace 外部的完整 robot YAML；
overlay 只在同一目录内解析 sibling `base_config`，不跨目录拼接资源路径语义不同的配置。

### 具身 AI 流水线（Embodied AI Pipeline）

`robot_config` 通过 `robot.embodied` 字段统一管理具身 AI 链路配置，但不直接启动
具身业务节点。完整运行时由 `embodied_bringup` 消费同一份 YAML 并编排下游节点，
以保持依赖方向为 `embodied_bringup -> robot_config`。

**前提条件**：具身流水线当前只在 `moveit_planning` 控制模式下可用。

#### 启动参数

| Launch 参数 | 作用 |
| --- | --- |
| `with_embodied` | 在 `robot_config` 基础 launch 中仅保留兼容覆盖；完整具身链路请使用 `embodied_bringup` |

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  use_sim:=true \
  moveit_display:=false
```

#### YAML 配置结构

```yaml
robot:
  name: example_robot
  # The selected Gateway mode must be a key in control_modes.
  control_modes:
    moveit_planning: {}
  skill_required_control_mode: moveit_planning

  embodied:
    enabled: false              # 默认关闭；通过 embodied_bringup launch 临时开启
    entry_mode: hermes          # hermes（默认）| agent（自然语言 Agent 孵化入口）
    debug_tracing: true

    timeouts:
      task_budget_sec: 180.0         # 任务端到端总预算
      scene_freshness_sec: 0.5       # 图像/深度新鲜度门槛
      model_idle_timeout_sec: 120.0  # 大模型输出空闲超时
      visual_game_timeout_sec: 130.0  # 视觉游戏 accepted 到 terminal 的总 deadline
      visual_game_result_retention_sec: 300.0  # terminal 查询记录最长保留时间
      rpc_timeout_sec: 5.0           # action/server/service 统一 RPC 超时
      gripper_settle_sec: 1.5        # 夹爪稳定等待时间

    planner:
      mode: vlm_api             # rule / vlm_api / hybrid
      scene_sources:
        primary_camera_topic: /camera/front_camera/color/image_raw
        primary_camera_info_topic: /camera/front_camera/color/camera_info
        primary_aligned_depth_topic: /camera/front_camera/aligned_depth_to_color/image_raw
        primary_pointcloud_topic: /camera/front_camera/depth/color/points
        ee_pose_topic: /robot_status/ee_pose
        joint_state_topic: /joint_states
        require_depth: true
        require_pointcloud: false
      vlm_api:
        provider: openai_compatible
        base_url: http://localhost:8000/v1
        model: Qwen3.5-9B
        api_key_env: ""

    visual_game_result_capacity: 128  # ledger 满时拒绝新请求，不提前驱逐 retention 内结果
    visual_game_event_topic: /embodied/visual_game_events
    visual_games:
      sorting_hat:
        enabled: false        # 趣味视觉游戏默认关闭
        handler: sorting_hat_v1
        summary: 根据主相机中的人物形象判断其霍格沃茨学院

    execution:
      relative_motion_reference_frame: base
      relative_motion_step_m: 0.03
      relative_motion_direction_mapping:
        forward:  [1, 0, 0]
        backward: [-1, 0, 0]
        left:     [0, 1, 0]
        right:    [0, -1, 0]
        up:       [0, 0, 1]
        down:     [0, 0, -1]

    safety:
      workspace:
        x: [-0.05, 0.45]
        y: [-0.35, 0.35]
        z: [0.05, 0.55]

    named_poses:
      home:           {position: {x: 0.15, y: 0.0, z: 0.30}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
      observe_table:  {position: {x: 0.20, y: 0.0, z: 0.35}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}

    # Skill manifests / implementation bodies / capability schemas / primitive
    # sequences are owned by `skill_catalog` (config/skills/<name>/manifest.yaml
    # + implementations/<implementation>.yaml). Robot YAML only selects the
    # source, profile, and runtime Gateway wiring:
    skill_catalog_source_mode: development          # installed | development | production
    skill_catalog_source_root: src/skill_catalog   # required in development/production
    skill_catalog_profile: so101_single_arm         # required when embodied.enabled is true

    named_targets:
      demo_object:
        observe_pose:  {position: {x: 0.25, y: 0.0, z: 0.26}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
        pregrasp_pose: {position: {x: 0.25, y: 0.0, z: 0.16}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
        grasp_pose:    {position: {x: 0.25, y: 0.0, z: 0.10}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}

    # entry_mode: agent 时的孵化 Agent 运行时（由 embodied_bringup 注入
    # ibrobot_agent_node 参数；字段约束见下方「Agent 孵化入口」一节）
    agent:
      enabled: true
      incubation: true            # 孵化期强制标记，必须为 true
      execution_enabled: false    # 运动默认关闭；开启需非空 test_allowlist
      channel_id: agent_cli
      principal_id: local_operator
      request_topic: /agent/request
      response_topic: /agent/response
      event_topic: /agent/event
      control_topic: /agent/control
      test_allowlist: [wave_hello, nod_yes]   # 执行白名单；execution_enabled=true 时必须非空
      ledger_path: /tmp/ibrobot-agent/requests.sqlite3        # 请求 ledger（SQLite WAL）
      conversation_path: /tmp/ibrobot-agent/conversation.sqlite3
      deployment_lock_path: /tmp/ibrobot-agent/agent.lock     # 部署锁，防双实例
      max_session_turns: 12       # 会话记忆滚动窗口
      clarification_ttl_sec: 300.0  # 一次性澄清上下文有效期
      event_queue_size: 128
      planner:
        mode: vlm                 # rule（仅仿真执行）| vlm
        provider: kimicode        # kimicode | openai_compatible
        base_url: https://api.kimi.com/coding/v1
        api_key_env: KIMICODE_API_KEY   # 密钥只允许环境变量名；字面 api_key 会被校验拒绝
        model: kimi-for-coding
        temperature: 1.0          # kimi-for-coding 强制 temperature=1
```

#### Agent 孵化入口（entry_mode: agent）

`embodied.entry_mode: agent` 由 `validate_agent_entry_config()`（`robot_config.loader`）
在 raw-dict launch 门禁与 typed `validate_config()` 两层统一校验，字段约束：

| 字段 | 约束 |
| --- | --- |
| `embodied.entry_mode` | `hermes` 或 `agent`；其他值报错 |
| `agent.enabled` | `entry_mode=agent` 时必须为 `true` |
| `agent.incubation` | 必须为 `true`（孵化期强制标记） |
| `agent.execution_enabled` | 布尔；为 `true` 时 `test_allowlist` 必须非空 |
| `agent.test_allowlist` | 非空字符串列表；执行白名单 |
| `agent.request_topic` / `agent.event_topic` | 必须是以 `/` 开头的绝对 ROS topic 名 |
| `agent.ledger_path` / `conversation_path` / `deployment_lock_path` | 非空路径 |
| `agent.max_session_turns` / `event_queue_size` | 正整数 |
| `agent.clarification_ttl_sec` | 正数 |
| `agent.planner.mode` | `rule` 或 `vlm` |
| `agent.planner.provider` | vlm 模式下 `kimicode` 或 `openai_compatible` |
| `agent.planner.base_url` / `model` | vlm 模式下必填非空 |
| `agent.planner.api_key_env` | kimicode 必填；密钥只允许环境变量名，**字面 `api_key` 字段会被直接拒绝** |
| `agent.planner.temperature` | `kimi-for-coding` 模型强制 `1` |

仓库自带 SO-101 孵化 profile：`so101_agent_manual`（真机手动无运动）、
`so101_agent_manual_hardware`（真机手动执行）、`so101_single_arm_agent_test`
（rule Planner 无运动测试）、`so101_single_arm_agent_gazebo`（Gazebo 执行）、
`so101_single_arm_agent_hardware`（真机执行）、`so101_single_arm_agent_hardware_stop`
（真机停止验证）。运行时行为与安全边界见 `ibrobot_agent` README。

#### Capability Gateway 接线契约

`robot_config` 不再承载技能执行 SSOT。`robot.embodied.skill_templates` 已移除；`load_robot_config_dict()`
检测到该键时直接报错，要求改用 `embodied.skill_catalog_profile`。技能的 `manifest`、
`capability` schema、`description` 与 `primitive_sequence` 全部由 `skill_catalog`
（`config/skills/<name>/manifest.yaml` + `implementations/<implementation>.yaml`）持有并校验，
公开 catalog 字段、命名位姿名称、timeout policy 和 digest 由 `skill_catalog` 编译器输出。

`load_robot_config_dict()` 是 launch、CLI 和 catalog 共用的规范化加载入口；它在返回配置前仅执行
Gateway 接线一致性校验，不重复 manifest/capability 校验。配置 `embodied.skill_catalog_profile`
时 `skill_required_control_mode` 必须是 `control_modes` 的非空成员；`embodied.enabled: true`
时 `embodied.skill_catalog_profile` 必须非空。`skill_catalog_source_mode` 只接受
`installed`、`development`、`production`；后两者要求 `skill_catalog_source_root` 指向有效目录。

`robot_config.loader.robot_config_digest` 是传入 `skill_catalog` 编译器的 canonical execution-context
digest；它刻意排除 `skill_catalog_source_mode` / `skill_catalog_source_root` / `skill_catalog_profile`、
解析后的 config 路径以及无关机器人配置，仅覆盖命名位姿/目标、关节限位、工作空间、控制模式、timeout policy、
相对运动语义和 execution endpoints。仅切换 source/profile/path 不改变执行身份；切换上述执行语义字段才会
触发新的 registry generation。

共享配置解析的选择顺序为：显式 `config_path`、显式 `config_name`、`ROBOT_CONFIG`、`ROBOT_NAME`、
默认 `so101_single_arm`。按名称查找时先查已安装 `robot_config` 的 `config/robots/`，再查源码树的
`config/robots/`；显式路径必须存在。primitive sequence、关节/笛卡尔坐标和目标绑定仍是 `skill_catalog`
私有实现数据；运行时 ROS service/action endpoint 则由 `robot_config` 配置并进入 canonical execution context。

#### visual_games 一致性

`embodied.visual_games` 声明非运动视觉能力（如分院帽）的 handler、公开描述、播报策略与开关。
视觉游戏只通过 Agent 的 `robot-skill start-game` 控制面触发；`trigger_mode`、`trigger_aliases` 和
`embodied.entry` 已移除，loader 遇到这些旧字段会明确拒绝配置，避免形成不可见的失效入口。
camera/VLM 由 `embodied.perception` 管理，游戏 deadline/retention 由 `embodied.timeouts` 管理。启用
`announce: true` 的视觉游戏会在 TTS service 可用时由 announcer 调用该 service；未配置或不可用时跳过播报，游戏控制面不受影响。
静默游戏不依赖 TTS。需要播报的终态统一由 announcer 消费事件后调用该 service，调用方不得重复播报。
`validate_config()` 强制一致性：
任一游戏 `enabled=true` 而 perception 条件不满足时返回错误，配置阶段即拦截。
`start_visual_game_service` 与 `get_visual_game_result_service` 声明 Agent 的异步 start/query 控制面，
`visual_game_event_topic` 声明供 TTS/UI/日志订阅的 accepted/terminal 事件边界；
`summary` 是公开描述，`handler` 必须引用共享 visual-game registry 中存在的实现契约。loader 同时校验
enabled、summary 和 handler，并拒绝不受支持的游戏。游戏 capability digest 覆盖启用的
游戏契约、timeout/retention、ledger capacity 和服务名，与运动 capability digest 相互独立。已知游戏缺省
handler/summary 由统一 registry 补齐。视觉游戏只读取上述规范字段。

触发入口固定为 Agent：使用 `robot-skill start-game GAME --request-id ID` 发起，并通过
`robot-skill game-result --request-id ID` 查询终态。修改游戏 YAML 后需重启 pipeline。

#### skill 加载与热更新边界

这里的“热加载”指进程不重启时更新运行时定义。运动 Skill catalog 支持受控的显式热加载：
`robot-skill reload-catalog --request-id ID --force` 请求 Gateway 从当前配置的 catalog source 重新编译，
并原子激活完整的新 snapshot。它不是文件系统自动监听，不会重新加载 ROS interface、robot config、节点参数
或 Python 实现；这些变化仍需重新构建并重启对应 pipeline。Hermes 的 `/reload-skills` 只更新 Agent 指令文件，
也不更新运动 catalog。

视觉游戏当前不支持热加载。`visual_games_json`、timeout policy、game capability digest 和 handler
registry 都在相关节点启动时冻结。修改机器人 YAML 后需重启 pipeline；修改 Python handler 后需重新构建并
重启。`robot-skill reload-catalog` 不会影响视觉游戏。

更多具身节点说明，详见各子包 README：
- [`embodied_agent`](../embodied_agent/README.md)
- [`perception_service`](../perception_service/README.md)
- [`skill_library`](../skill_library/README.md)
- [`safety_guard`](../safety_guard/README.md)

#### 2. moveit_planning 模式（轨迹规划控制）

**适用于：** 基于规划的模型（VoxPoser、VLM、目标条件化）

**特点：**
- 轨迹插值和时间参数化
- 通过 MoveIt 动作接口执行
- 碰撞检测和避障
- 更平滑的轨迹

**配置：**
```yaml
robot:
  default_control_mode: "moveit_planning"

  control_modes:
    moveit_planning:
      description: "MoveIt 轨迹规划模式"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller
```

**启动的控制器：**
- `arm_trajectory_controller` (JointTrajectoryController)
- `gripper_trajectory_controller` (JointTrajectoryController)

**命令接口：**
```bash
# 通过 MoveIt 动作执行轨迹
ros2 action send_goal /arm_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{...}"
```

### 完整控制模式配置示例

```yaml
robot:
  name: so101_single_arm
  type: so101
  default_control_mode: "model_inference"  # 或 "teleop" 或 "moveit_planning"

  joints:
    arm: ["1", "2", "3", "4", "5"]
    gripper: ["6"]
    all: ["1", "2", "3", "4", "5", "6"]

  # 控制模式配置
  control_modes:
    teleop:
      description: "人工遥操作模式"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller

    model_inference:
      description: "高频端到端控制模式"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller

    moveit_planning:
      description: "MoveIt轨迹规划模式"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller

  # 遥操作配置
  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"

  # 硬件配置
  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  # 外设（相机、传感器）
  peripherals:
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30

  # ML 契约
  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top
        image:
          resize: [480, 640]
    actions:
      - key: action
        topic: /arm_position_controller/commands  # 根据模式变化
        ros_type: std_msgs/msg/Float64MultiArray
        names: ["1", "2", "3", "4", "5", "6"]
```

### 模式切换工作原理

1. **配置阶段：**
   - `robot.launch.py` 从 YAML 读取 `default_control_mode`
   - 可通过 `control_mode:=xxx` 命令行参数覆盖
   - 验证模式是否存在于 `control_modes` 部分

2. **控制器生成：**
   - 仅生成所选模式中列出的控制器
   - 确保无控制器冲突（同一关节不能被多个控制器控制）

3. **动作分发集成：**
   - `action_dispatch` 节点从 `robot_config` 读取当前模式
   - 推理分发按策略选择 TopicExecutor 或 BenchmarkExecutor；MoveIt 规划使用独立执行路径
   - 为上游推理服务提供统一 API

### 控制模式故障排除

#### 模式未切换

**问题：** 命令行覆盖未生效

**解决方案：** 确保 `control_mode` 参数拼写正确：
```bash
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning use_sim:=true
```

#### 控制器冲突

**问题：** 同一关节被多个控制器控制

**解决方案：** 检查配置，确保每种模式使用互斥的控制器：
```yaml
control_modes:
  model_inference:
    controllers:
      - arm_position_controller      # 位置控制器
  moveit_planning:
    controllers:
      - arm_trajectory_controller    # 轨迹控制器（不同类型）
```

#### 执行器类型不匹配

**问题：** 推理服务发送位置命令但启动了轨迹控制器

**解决方案：** 确保控制模式与模型类型匹配：
- ACT/pi0 模型 → `model_inference` 模式
- VoxPoser/VLM 模型 → `moveit_planning` 模式
- 人工遥操作 → `teleop` 模式

## 使用方法

### 启动机器人

```bash
# 启动真实硬件
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm

# 启动仿真
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# 契约级 mock 仿真
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true sim_platform:=mock control_mode:=model_inference

# Ascend310P PI0.5 native Torch end-to-end validation with hardware_mock
ros2 launch robot_config robot.launch.py robot_config:=so101_pi05_ascend_310p_mock use_sim:=true control_mode:=model_inference

# MoveIt 规划模式（带 RViz）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true

# MoveIt 模式无 RViz（headless）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true moveit_display:=false

# 分布式推理 — 单机调试（YAML 中 policy pipeline 必须为 distributed）
# 终端 1：Edge
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# 终端 2：Cloud
ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cpu

# 分布式推理 — 跨机器部署（端侧仅启动 Edge，Cloud 在算力机器上单独启动）
# 端侧：
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# 算力机器（需设置相同 ROS_DOMAIN_ID）：
# ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cuda
```

相对 `model_path` 只相对于绝对路径环境变量 `WORKSPACE` 解析。Pipeline ID 决定默认 Action、
reset、health 和分布式 topic；多个 pipeline 时 `executor.inference_pipeline` 必须明确选择一个。

### 验证配置

```bash
# 直接使用 Python 验证机器人配置文件
python3 src/robot_config/robot_config/scripts/validate_config.py \
    src/robot_config/config/robots/so101_single_arm.yaml
```

## 相机驱动

`peripherals[]` 条目支持 `disabled: true` 标志：设置后该外设在 `load_robot_config()`
与 perception launch builder 中被整体跳过（不加载配置、不生成相机节点）。用于
Agent 孵化等无视觉 profile 显式关闭继承自 base config 的相机。注意统一的禁用开关
是 `disabled`（极性为 true=禁用），不要使用 `enabled: false`（该键不被识别）。

### USB 相机（通过 `usb_cam`）

```yaml
- type: camera
  name: usb_cam
  driver: opencv
  index: 0
  width: 640
  height: 480
  fps: 30
  pixel_format: mjpeg2rgb
  frame_id: camera_frame
  camera_info_url: file://$(env HOME)/.ros/camera_info/top.yaml
```

**安装：**
```bash
sudo apt install ros-humble-usb-cam
```

### RealSense 相机（通过 `realsense2_camera`）

```yaml
- type: camera
  name: realsense
  driver: realsense
  serial_number: "12345678"
  width: 640
  height: 480
  fps: 30
  depth_width: 640
  depth_height: 480
  depth_fps: 30
  enable_depth: true
  enable_color: true
  align_depth: true
  # 高带宽 image/pointcloud 由驱动直接改名，避免通过第二个 DDS relay 复制。
  # CameraInfo 仍使用轻量 relay，把 frame_id 规范为 optical_frame_id。
  direct_topic_remap: true
  # 仅在确有 PointCloud2 消费者时开启；RGB-D 抓取可直接从对齐深度构造点云。
  enable_pointcloud: false
```

`direct_topic_remap` 默认是 `false`，保留旧的全 relay 行为。对单机高带宽 RealSense pipeline，建议设为
`true`；下游仍使用 `/camera/{name}/image_raw` 和
`/camera/{name}/aligned_depth_to_color/{image_raw,camera_info}`，不需要了解驱动原生 topic 前缀。

**安装：**
```bash
# Ubuntu
sudo apt install ros-humble-librealsense2*
# openEuler
sudo dnf install ros-humble-librealsense2*
```

## 相机标定

相机内参可以存储在标准 ROS2 位置：

```yaml
- type: camera
  name: top
  driver: opencv
  index: 0
  width: 640
  height: 480
  camera_info_url: file://$(env HOME)/.ros/camera_info/top_camera.yaml
```

可以使用标准 ROS2 相机标定工具创建标定文件：

```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.024 \
  image:=/camera/top/image_raw
```

### 色彩 / ISP 校准 — Override 机制

YAML 是相机参数的 **单一真理来源（SSOT）**，但日常调机时手动改 YAML
不便。为此 robot_config 在加载相机参数时按以下顺序合并：

1. `peripherals/[name]` YAML 默认值；
2. `~/.ros/ibrobot/camera_isp_overrides/{camera_name}.json` 中的 override
   （由 `camera_isp_calibrator` 写入，键为 V4L2 参数名）；

**override 中的值覆盖 YAML**。这样校准结果可以跨次启动复用而不污染
SSOT；删除 JSON 即可回退到 YAML 默认。

加载逻辑见 `robot_config/launch_builders/camera_isp_overrides.py`，
合并点见 `robot_config/launch_builders/perception.py`（opencv 分支）。
配套校准工具：`ros2 run dataset_tools camera_isp_calibrator`。

## tensormsg 集成

robot_config 通过允许观察通过名称引用外设来与 tensormsg 契约集成：

```yaml
# 在 robot_config 中
peripherals:
  - type: camera
    name: top
    width: 640
    height: 480

# 在 contract 部分
contract:
  observations:
    - key: observation.images.top
      topic: /camera/top
      peripheral: top  # 自动填充外设的 width、height、fps
```

当契约加载时，将自动包含外设定义中的相机元数据。

## 故障排除

### 相机无法打开

检查 USB 权限：
```bash
ls -l /dev/video*
sudo chmod 666 /dev/video0
```

或将用户添加到 `video` 组：
```bash
sudo usermod -a -G video $USER
```

### RealSense 相机未找到

安装 librealsense2：
```bash
# Ubuntu
sudo apt install librealsense2-utils librealsense2-dev
sudo apt install ros-humble-librealsense2*

# openEuler
sudo dnf install librealsense2-utils librealsense2-devel
sudo dnf install ros-humble-librealsense2*
```

检查相机是否连接：
```bash
realsense-viewer
```

### 标定文件未找到

确保路径正确并以 `file://` 开头：
```yaml
camera_info_url: file:///home/user/.ros/camera_info/top.yaml
```

### 控制器加载失败

如果遇到 "Controller already loaded" 错误，运行清理脚本：
```bash
./scripts/cleanup_ros.sh
```

## 仿真依赖（MuJoCo）

`use_sim:=true` 模式依赖 `ros-humble-mujoco-ros2-control`，通过 rosdep 声明为运行时依赖：

```bash
# 安装所有依赖（含 MuJoCo 仿真包）
rosdep install --from-paths src --ignore-src -y
```

该命令会自动安装 `ros-humble-mujoco-ros2-control` 及其传递依赖
（`mujoco_ros2_control_msgs`、`mujoco_ros2_control_plugins`、`mujoco_vendor`）。
不再需要手动初始化 git submodule。

### 仿真推理控制面板

在 MuJoCo 仿真推理调试时，推荐使用仓库根目录下的
`scripts/model_infer_panel.py`，提供 Start/Stop 推理、随机场景、AutoTest
全量评估等控制能力：

```bash
python3 scripts/model_infer_panel.py
# 浏览器访问 http://<robot-ip>:8766
```

相机图像可视化请使用已有的 Rerun 链路：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=model_inference \
  use_sim:=true \
  record_visualizer:=rerun
```

## 参考资料

- [usb_cam GitHub](https://github.com/ros-drivers/usb_cam) - ROS2 USB 相机驱动
- [realsense-ros GitHub](https://github.com/realsenseai/realsense-ros) - Intel RealSense ROS2 封装
- [action_dispatch README](../action_dispatch/README.md) - 详细执行器文档
- [docs/architecture.md](../../docs/architecture.md) - 系统架构概览

## 许可证

Apache-2.0

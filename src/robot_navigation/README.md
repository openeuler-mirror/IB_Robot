# robot_navigation

机器人导航包，集成了语音识别、Nav2 导航、全向轮底盘桥接和定位融合功能。

## 导航 Profile

本包同时提供 RealSense/RTAB-Map 链路和 MID-360 LiDAR 链路，以下描述按 profile 与 stage 区分：

| Profile | `map -> odom` | `odom -> base_link` | 主要速度链 |
|---|---|---|---|
| RealSense mapping | RTAB-Map | cmd_vel bridge | `/cmd_vel -> bridge` |
| RealSense navigation | RTAB-Map localization | EKF | `/cmd_vel -> bridge` |
| MID-360 mapping | SLAM Toolbox | FAST-LIO odom bridge | `/cmd_vel -> bridge` |
| MID-360 navigation | AMCL | FAST-LIO odom bridge | `/cmd_vel -> Collision Monitor -> /cmd_vel_safe -> bridge` |

本文中 RTAB-Map、EKF、DWB 和“不启动 AMCL”的章节仅适用于 RealSense navigation profile；RealSense
mapping 由 cmd_vel bridge 发布 `odom -> base_link`。LiDAR profile 以 `lekiwi_lidar.yaml` 的 resolved stage 配置为准。

## 功能特性

- **语音控制**: 基于 `voice_asr_service` (sherpa-onnx 本地 ASR) + `voice_control` 关键词匹配的语音导航
- **导航控制**: Nav2 导航 Goal 客户端，支持语音触发导航，到达后自动触发机械臂推理
- **全向局部控制**: 提供 Nav2 MPPI `Omni` 候选 profile，实机门禁通过前默认保留 DWB
- **底盘桥接**: `cmd_vel_bridge_node` 通过 IK/FK 将标准 `/cmd_vel` 桥接到 ros2_control 全向轮速度指令 (rad/s)，并发布里程计
- **定位融合**: EKF (robot_localization) 融合底盘里程计速度，RTAB-Map 视觉 SLAM 提供全局定位修正
- **建图保存**: `save_rtabmap_map` 包装 Nav2 map_saver，将 `/rtabmap/map` 默认保存为 `rtabmap.yaml/rtabmap.pgm`
- **任务联动**: 语音 → 导航 → 到达 → 触发 action_dispatcher 评估，形成完整任务链

## 系统架构

```text
  用户语音 ──► voice_asr_node (sherpa-onnx) ──► /voice_command
                                                     │
                                                     ▼
                                              voice_control (关键词匹配)
                                                     │
                                                     ▼
                                         /voice_asr/keyword_matched (JSON)
                                                     │
                                                     ▼
                                            nav2_goal_client
                                                     │
                                                     ▼
                                         Nav2 NavigateToPose Action
                                                     │
                                                     ▼
                                        Nav2 规划 + 发布 /cmd_vel
                                                     │
                                                     ▼
                                        到达目标 ──► /action_dispatcher/start_evaluate

  /cmd_vel ─────► cmd_vel_bridge_node (IK) ──► /base_velocity_controller/commands (rad/s)
                    │  (FK)
                    └──► /odom (nav_msgs/Odometry)
                              │
                              ▼
                    EKF (robot_localization) ──► TF: odom → base_link (30Hz)
                                                   │
  RTAB-Map (视觉 SLAM) ──► TF: map → odom (~1Hz) ◄──┘
```

### TF 树结构

```text
map ──(RTAB-Map)──► odom ──(EKF)──► base_link ──► ... ──► sensor frames
```

| TF 变换 | 发布者 | 频率 | 说明 |
|---------|--------|------|------|
| `map → odom` | RTAB-Map | ~1Hz | 视觉 SLAM 全局定位修正 |
| `odom → base_link` | EKF | 30Hz | 融合底盘里程计速度，平滑输出 |

上图只描述 RealSense navigation。该 profile 的 `cmd_vel_bridge_node` 仅发布 `/odom` 话题（供 EKF 订阅），
不发布 TF（`publish_tf: false`）；RealSense mapping 不启动 EKF，由 bridge 发布 `odom -> base_link`。
Nav2 使用静态地图与 RTAB-Map 提供的 `map → odom` 定位结果，不并行启动 AMCL，避免多个节点竞争发布同一全局定位 TF。

## 节点列表

| 节点 | 功能 | 入口点 |
|------|------|--------|
| `voice_control` | 语音关键词匹配 + 导航桥接 (sherpa-onnx 本地 ASR) | `robot_navigation.voice_control` |
| `nav2_goal_client` | Nav2 导航 Goal 客户端 + 评估触发 | `robot_navigation.nav2_goal_client` |
| `cmd_vel_bridge_node` | cmd_vel → ros2_control 桥接 + 里程计发布 | `robot_navigation.cmd_vel_bridge_node` |
| `fast_lio_odom_bridge` | 规范化 FAST-LIO odometry 与 TF | `robot_navigation.fast_lio_odom_bridge` |
| `navigation_lifecycle_coordinator` | 在 warm-up 后启动 Nav2 lifecycle | `robot_navigation.navigation_lifecycle_coordinator` |
| `navigation_command_server` | 提供原子导航 Action 与取消服务 | `robot_navigation.navigation_command_server` |
| `nav_cmd` | 原子导航命令 CLI | `robot_navigation.nav_cmd` |
| `save_rtabmap_map` | 保存 RTAB-Map OccupancyGrid 地图 | `robot_navigation.save_rtabmap_map` |
| `save_lidar_map` | 保存 LiDAR OccupancyGrid 地图 | `robot_navigation.save_lidar_map` |

## 使用入口

先按目的选择入口；完整机器人链路统一由 `robot_config` 启动，`robot_navigation` 只保留导航子系统、节点和 PC 端 RViz 预设。

| 你要做什么 | 看哪一节 | 推荐命令入口 |
|---|---|---|
| 按现场顺序完成标定、采集、建图和导航 | [现场直接运行流程](#现场直接运行流程) | `robot_config/robot.launch.py` |
| 复刻建图与导航环境 | [环境复刻](#环境复刻) | `robot_config:=lekiwi_realsense_mapping` / `robot_config:=lekiwi_realsense_navigation` |
| 跑本包测试 | [测试验证](#测试验证) | `colcon test --packages-select robot_navigation` |
| 验证完整 Gazebo/Nav2 仿真 | [测试验证](#测试验证) | `NAV_TEST_PROFILE=full colcon test --packages-select robot_config ...` |
| 实机建图 | [建图流程](#建图流程) | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_realsense_mapping` |
| 实机导航 | [导航流程](#导航流程) | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_realsense_navigation control_mode:=teleop` |
| PC 端观察 | [PC 端 RViz 观察](#pc-端-rviz-观察) | `lekiwi_realsense_*_rviz.launch.py` / `lekiwi_lidar_*_rviz.launch.py` |
| 单独调试导航节点 | [底层调试入口](#底层调试入口) | `ros2 run robot_navigation ...` |

`robot_config` 的优势：YAML 单一数据源，自动启动控制器、相机、TF、定位、Nav2 和导航节点，并通过 `control_mode` 切换运行模式。

LeKiwi RealSense 导航的 body-frame 速度与加速度包络由 `lekiwi_realsense_navigation.yaml` 的
`robot.navigation.motion_envelope` 唯一维护。`nav2_params.yaml` 中的 MPPI 和
velocity smoother 参数是该包络的镜像，FAST-LIO、SLAM 与其他导航消费者不得
另建权威值。静态测试会同时检查镜像漂移和整个速度盒的轮空间可行性。

## 现场直接运行流程

本节集中给出标定、语义数据采集、RealSense 建图/导航和 LiDAR 建图/导航的可执行命令。后续章节保留
架构原理、单点验证、底层调试和故障排除说明。

### 1. 环境与构建

开发板和 PC 侧必须网络互通，并使用相同的 `ROS_DOMAIN_ID`。每个新终端按顺序执行：

```bash
source .shrc_local
export ROS_DOMAIN_ID=<domain-id>
source install/setup.sh
```

代码修改后从仓库根目录使用统一构建脚本：

```bash
./scripts/build.sh -- --packages-up-to \
  ibrobot_msgs robot_calibration semantic_mapping robot_navigation robot_config
```

两种导航模式使用独立 profile、地图和 RViz 预设：

| 模式 | 工作流 | 开发板 profile | PC 侧 RViz |
|---|---|---|---|
| RealSense | 建图 | `robot_config:=lekiwi_realsense_mapping` | `lekiwi_realsense_mapping_rviz.launch.py` |
| RealSense | 导航 | `robot_config:=lekiwi_realsense_navigation control_mode:=teleop` | `lekiwi_realsense_navigation_rviz.launch.py` |
| LiDAR | 建图 | `robot_config:=lekiwi_lidar nav_stage:=mapping` | `lekiwi_lidar_mapping_rviz.launch.py` |
| LiDAR | 导航 | `robot_config:=lekiwi_lidar nav_stage:=navigation` | `lekiwi_lidar_navigation_rviz.launch.py` |

RealSense 模式使用 D435 和 RTAB-Map；LiDAR 模式使用 MID-360、FAST-LIO 和 SLAM Toolbox/AMCL。两种模式
不能混用地图文件或 RViz 预设。

### 2. D435i/MID-360 标定

详细采集要求和产物契约见 [`robot_calibration/README.md`](../robot_calibration/README.md)。标定期间不要同时启动
导航主链。开发板执行采集：

```bash
ros2 run robot_calibration calib_capture
```

需要移动机器人时，在开发板另一个终端运行键盘遥控；每次录制前停稳机器人：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

原始包默认写入 `~/.ros/ibrobot/calib/raw/<capture-id>.raw.tar`。将原始包复制到 PC 后处理：

```bash
ros2 run robot_calibration calib_process \
  --input <path-to>/<capture-id>.raw.tar

xdg-open ~/.ros/ibrobot/calib/process/<capture-id>/test-overlay.png

scp ~/.ros/ibrobot/calib/candidates/<capture-id>.candidate.tar \
  <development-board-host>:~/.ros/ibrobot/calib/candidates/
```

开发板启动候选实时验证，并保持该进程运行：

```bash
ros2 run robot_calibration calib_validate \
  --input ~/.ros/ibrobot/calib/candidates/<capture-id>.candidate.tar
```

PC 侧在同一 ROS domain 中观察实时叠加：

```bash
ros2 run robot_calibration calib_view --mode validate
```

确认投影正确后，在开发板结束验证进程并批准候选：

```bash
ros2 run robot_calibration calib_approve \
  --input ~/.ros/ibrobot/calib/candidates/<capture-id>.candidate.tar
```

批准结果写入 `~/.ros/ibrobot/calib/current/`。`calib_validate` 只验证候选，不会自动批准。

### 3. 3D 语义数据采集

该流程同时启动 MID-360、FAST-LIO、SLAM Toolbox、D435i 和连续 MCAP 录制。详细说明见
[`semantic_mapping/README.md`](../semantic_mapping/README.md)。

开发板终端 A 启动采集主链，并在保存结束前保持运行：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=lekiwi_semantic_capture
```

开发板终端 B 移动底盘完成采集轨迹：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

PC 侧观察低带宽图像、点云和地图预览：

```bash
ros2 launch semantic_mapping lekiwi_semantic_capture_rviz.launch.py
```

采集完成后，在开发板另一个终端执行保存：

```bash
ros2 run semantic_mapping save_semantic_map
```

保存命令会停止 recorder、保存地图、校验 MCAP/TF/标定快照并生成 `.tar.gz`。保存成功后再由操作员结束主 launch。

### 4. RealSense 建图

开发板终端 A 启动 D435、RTAB-Map mapping 和底盘桥接：

```bash
ros2 launch robot_config robot.launch.py \
  use_sim:=false \
  robot_config:=lekiwi_realsense_mapping
```

开发板终端 B 启动键盘遥控：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

PC 侧启动 RealSense 建图 RViz：

```bash
ros2 launch robot_navigation lekiwi_realsense_mapping_rviz.launch.py
```

建图主链保持运行时，在开发板保存地图：

```bash
ros2 run robot_navigation save_rtabmap_map
```

占据栅格默认输出到 `~/.ros/ibrobot/maps/rtabmap.yaml` 和 `rtabmap.pgm`；RTAB-Map 运行时数据库写入
`~/.ros/ibrobot/maps/rtabmap.db`。

### 5. RealSense 导航

开发板启动 RealSense 定位和 Nav2。显式使用 `control_mode:=teleop`，避免启用该 profile 默认的模型推理模式：

```bash
ros2 launch robot_config robot.launch.py \
  use_sim:=false \
  robot_config:=lekiwi_realsense_navigation \
  control_mode:=teleop
```

PC 侧启动 RealSense 导航 RViz：

```bash
ros2 launch robot_navigation lekiwi_realsense_navigation_rviz.launch.py
```

在 RViz 中使用 `2D Goal Pose` 或 `Nav2 Goal` 下发目标。该模式加载 `rtabmap.yaml` 和 `rtabmap.db`。

### 6. LiDAR 建图

开发板终端 A 启动 MID-360、FAST-LIO、SLAM Toolbox 和底盘桥接：

```bash
ros2 launch robot_config robot.launch.py \
  use_sim:=false \
  robot_config:=lekiwi_lidar \
  nav_stage:=mapping
```

开发板终端 B 启动键盘遥控：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

PC 侧启动 LiDAR 建图 RViz：

```bash
ros2 launch robot_navigation lekiwi_lidar_mapping_rviz.launch.py
```

建图主链保持运行时，在开发板保存地图：

```bash
ros2 run robot_navigation save_lidar_map
```

默认从 `/map` 原子保存到 `~/.ros/ibrobot/maps/map.yaml` 和 `map.pgm`。

### 7. LiDAR 导航

开发板启动 MID-360、FAST-LIO、AMCL、Nav2、Collision Monitor 和原子命令服务：

```bash
ros2 launch robot_config robot.launch.py \
  use_sim:=false \
  robot_config:=lekiwi_lidar \
  nav_stage:=navigation
```

PC 侧启动 LiDAR 导航 RViz：

```bash
ros2 launch robot_navigation lekiwi_lidar_navigation_rviz.launch.py
```

等待启动 warm-up 完成后，在开发板检查 Action 可发现性：

```bash
ros2 run robot_navigation nav_cmd --action-name /navigation/execute status
```

`status` 不是定位收敛判据。发送运动命令前还应确认地图已加载、Nav2 lifecycle active，并且
`map -> base_link` 定位稳定。平移和绝对坐标单位为米，CLI 转向角单位为度：

```bash
ros2 run robot_navigation nav_cmd --action-name /navigation/execute forward 0.5
ros2 run robot_navigation nav_cmd --action-name /navigation/execute backward 0.5
ros2 run robot_navigation nav_cmd --action-name /navigation/execute leftward 0.5
ros2 run robot_navigation nav_cmd --action-name /navigation/execute rightward 0.5
ros2 run robot_navigation nav_cmd --action-name /navigation/execute turn-left 10
ros2 run robot_navigation nav_cmd --action-name /navigation/execute turn-right 10
ros2 run robot_navigation nav_cmd --action-name /navigation/execute absolute <x> <y> <yaw_deg>
ros2 run robot_navigation nav_cmd --action-name /navigation/execute cancel
```

导航 stage 的速度链是 `/cmd_vel -> Collision Monitor -> /cmd_vel_safe -> cmd_vel_bridge`。导航期间不要另启
`teleop_twist_keyboard` 或其他直接发布 `/cmd_vel`、`/cmd_vel_safe` 的节点；当前链路没有速度所有权仲裁器。

## 环境复刻

### 1. 运行环境选择

LeKiwi 建图和导航支持两类运行方式：

| 模式 | ROS 节点运行位置 | Ubuntu PC 角色 | 适用场景 |
|---|---|---|---|
| openEuler 开发板主运行 | 资源较充足的开发板 | SSH + RViz 远程观察 | 实机复刻、板端验证 |
| Ubuntu 单机运行 | Ubuntu PC / 笔记本 | 运行所有节点 + RViz | 本机调试、算法验证、没有开发板的联调 |

跨机器部署时，开发板和 PC 必须网络互通，并使用相同的 `ROS_DOMAIN_ID`。openEuler 方案下 Ubuntu 仍然是必需的观察端；Ubuntu 方案下 Ubuntu 是运行端。

典型组合：

| 组合 | ROS 节点跑在 | Ubuntu 角色 | 网络连接 | 设备接入 | 小车能否自由跑 |
|---|---|---|---|---|---|
| A | openEuler @ 开发板 | SSH + RViz | 网线接 Ubuntu PC | RealSense/底盘接开发板 | 否，小车在脚边 |
| B | openEuler @ 开发板 | SSH + RViz | 网线接 Ubuntu 笔记本 | RealSense/底盘接开发板 | 是，笔记本跟着小车 |
| C | openEuler @ 开发板 | SSH + RViz | WiFi | RealSense/底盘接开发板 | 是，PC 不动 |
| D | Ubuntu @ 笔记本 | 运行 + RViz | 无跨机通信 | RealSense/底盘接笔记本 | 是，笔记本跟着小车 |
| E | Ubuntu @ PC | 运行 + RViz | 无跨机通信 | RealSense/底盘接 PC | 否，小车在脚边 |

推荐先用组合 A 验证主链路：网线稳定，小车在脚边，小范围完成建图、保存地图、加载地图和导航。需要小车真正自由跑时，再切到 B/C/D。

### 2. 终端环境

每个新终端都需要先加载 ROS 和工作区环境。具体加载方式参考 IB Robot 仓主目录下的 README，并确保开发板和 PC 使用相同的 `ROS_DOMAIN_ID`，例如 `<your_id>`。

### 3. 编译

编译方式参考 IB Robot 仓主目录下的 README。本文件只说明导航相关入口、测试和排查方法，不重复维护完整工作区编译命令。

### 4. 硬件与网络注意事项

- 使用 RealSense 深度相机时，必须接 USB 3.0 蓝口，不建议使用长 USB 延长线；深度流数据量大，USB 2.0 或长线容易导致掉帧、重枚举或无图像。
- 底盘控制链默认走 `/dev/ttyACM0`，插拔顺序变化时可能变成 `/dev/ttyACM1`。
- 推荐计算侧和执行侧分开供电：开发板 + USB 外设一套电源，底盘/机械臂电机一套电源，避免电机瞬时电流导致板子重启或 RealSense 掉线。
- 完整导航链路会同时运行 RealSense、RTAB-Map、ros2_control、Nav2 和相关桥接节点，对 CPU 与内存资源要求较高；建议使用具备较高算力和较大内存资源的开发板，资源水平至少接近中高端边缘计算板，才能更稳定地跑完整链路。
- WiFi 下如果 SSH/DDS 卡顿，优先确认两端 `ROS_DOMAIN_ID` 一致，再检查 DDS 实现、网卡绑定、单播 peers、路由器频段和 RealSense 带宽配置。

### 5. robot_config 入口

| 入口 YAML | 用途 | 启动命令 |
|---|---|---|
| `lekiwi_realsense_mapping.yaml` | base-only 建图：底盘 + RealSense + RTAB-Map mapping + cmd_vel_bridge | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_realsense_mapping` |
| `lekiwi_realsense_navigation.yaml` | 完整导航：加载静态地图 + Nav2 + RTAB-Map localization + EKF + cmd_vel_bridge | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_realsense_navigation control_mode:=teleop` |

RealSense 建图和导航使用独立配置文件。`lekiwi_realsense_mapping` 只负责建图主链；`lekiwi_realsense_navigation` 消费保存好的地图做定位与导航。

`lekiwi_realsense_navigation.yaml` 中导航相关配置形态如下：

```yaml
navigation:
  enabled: true
  motion_envelope:
    velocity:
      vx: {min: -0.13, max: 0.13}
      vy: {min: -0.11, max: 0.11}
      wz: {min: -0.44, max: 0.44}
    acceleration:
      vx: {min: -0.65, max: 0.65}
      vy: {min: -0.55, max: 0.55}
      wz: {min: -2.2, max: 2.2}
  nav2_bringup:
    enabled: true
    map_file: "$(env HOME)/.ros/ibrobot/maps/rtabmap.yaml"
    # MPPI promotion 前保持 DWB 为实机默认值。
    params_file: "$(find robot_navigation)/config/nav2_params_dwb.yaml"
  ekf_rtabmap:
    enabled: true
    rtabmap:
      rtabmap_args: "--Mem/InitWMWithAllNodes true --Mem/IncrementalMemory true --Mem/PermanentMemory false --Mem/STMSize 8 --Reg/Force3DoF true"
  cmd_vel_bridge:
    enabled: true
    publish_tf: false         # EKF 发布 TF，bridge 只发布 /odom 话题
    max_radps: 4.602          # 最大轮速 (rad/s)
    cmd_vel_topic: /cmd_vel
    joint_states_topic: /joint_states
    odom_topic: /odom
  robot_navigation:
    enabled: true
    enable_voice_control: true
    destinations:
      point_a: {x: 0.0, y: 0.2, theta: 1.5708}  # rad (90 deg)
      point_b: {x: 0.2, y: 0.0, theta: 0.0}
  rviz:
    enabled: true
```

MPPI 未通过完整门禁时，`lekiwi_realsense_navigation.yaml` 保持选择
`$(find robot_navigation)/config/nav2_params_dwb.yaml`。验证 MPPI 候选时显式选择
`nav2_params.yaml`，全部门禁通过后才可将它设为实机默认值。仿真对应
`nav2_sim_params.yaml` 与 `nav2_sim_params_dwb.yaml`。切换前必须终止当前导航目标并
重启 Nav2；切换不会改变 `NavigateToPose`、`FollowPath`、costmap 或 `/cmd_vel` 接口。

DWB/MPPI 对照矩阵和 promotion gate 定义在
`config/nav2/controller_regression.yaml`。采集的 baseline/candidate JSON 均包含 `runs`
数组后，可执行：

```bash
ros2 run robot_navigation evaluate_controller_regression \
  --matrix $(ros2 pkg prefix robot_navigation)/share/robot_navigation/config/nav2/controller_regression.yaml \
  --baseline /path/to/dwb.json \
  --candidate /path/to/mppi.json
```

只有成功率不下降、无新增碰撞或持续振荡、无命令越界且 controller computation
P95 严格低于 50 ms 时才返回成功。仅有 `/cmd_vel` 到达间隔不能替代 computation trace。

## 建图流程

建图入口使用 `lekiwi_realsense_mapping.yaml`，链路包含 base-only ros2_control、RealSense、TF、RTAB-Map mapping 和 `/cmd_vel` 底盘桥接。它不会自动启动键盘遥控，需要另一个终端发布 `/cmd_vel`。

终端 A 启动建图主链：

```bash
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_realsense_mapping
```

终端 B 启动键盘遥控：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

`teleop_twist_keyboard` 默认发布到 `/cmd_vel`，会被 `lekiwi_realsense_mapping.yaml` 中的 `navigation.cmd_vel_bridge.cmd_vel_topic: /cmd_vel` 消费。常用键位：`i` 前进、`,` 后退、`j` 左转、`l` 右转、`k` 停止；`w/x/e/c/q/z` 是速度倍率调整键。

终端 C 保存地图：

```bash
ros2 run robot_navigation save_rtabmap_map -f ~/.ros/ibrobot/maps/rtabmap
```

默认等价于：

```bash
ros2 run nav2_map_server map_saver_cli -t /rtabmap/map -f ~/.ros/ibrobot/maps/rtabmap
```

输出约定：

```text
~/.ros/ibrobot/maps/rtabmap.yaml     # Nav2 map_server 加载
~/.ros/ibrobot/maps/rtabmap.pgm      # 占据栅格图像
~/.ros/ibrobot/maps/rtabmap.db       # RTAB-Map localization 复用
```

后续导航默认从 `lekiwi_realsense_navigation.yaml` 的 `navigation.nav2_bringup.map_file: "$(env HOME)/.ros/ibrobot/maps/rtabmap.yaml"` 加载地图。

## 导航流程

启动导航主链：

```bash
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_realsense_navigation control_mode:=teleop
```

RViz 启动后，使用 `2D Goal Pose` 或 `Nav2 Goal` 在地图上点击目标。Nav2 会规划全局路径并通过配置的速度链驱动底盘；
MID-360 navigation 使用 `/cmd_vel -> Collision Monitor -> /cmd_vel_safe -> bridge`，局部规划器根据代价地图避障。

如果需要导航评估模式并联动推理链：

```bash
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_realsense_navigation control_mode:=navi
```

## PC 端 RViz 观察

LiDAR 和 RealSense/RTAB-Map 使用明确分开的 RViz 配置，不能混用：

LiDAR 使用一个简短的动态避障开关：

```yaml
dyn_avoid_enabled: false  # DWB + LaserScan + /cmd_vel
dyn_avoid_enabled: true   # MPPI + PointCloud2 + Collision Monitor + /cmd_vel_safe
```

该开关由 `lekiwi_lidar` 的 navigation stage 设置，用户不需要同时维护
controller、costmap、Collision Monitor 和底盘速度 topic。

```bash
# LiDAR 建图
ros2 launch robot_navigation lekiwi_lidar_mapping_rviz.launch.py

# LiDAR 导航
ros2 launch robot_navigation lekiwi_lidar_navigation_rviz.launch.py

# RealSense/RTAB-Map 建图
ros2 launch robot_navigation lekiwi_realsense_mapping_rviz.launch.py

# RealSense/RTAB-Map 导航
ros2 launch robot_navigation lekiwi_realsense_navigation_rviz.launch.py
```

开发板上已经启动 `lekiwi_realsense_mapping` 或 `lekiwi_realsense_navigation` 后，PC 端不需要在板端本地打开 RViz。只要网络互通且 `ROS_DOMAIN_ID` 一致，PC 可以直接观察远端 ROS 图。

PC 端同样需要先加载 ROS 和工作区环境，具体方式参考 IB Robot 仓主目录下的 README。如果 PC 本地也有同一份工作区，并希望使用工作区里的 launch/config，再 source 该 overlay。两端 `ROS_DOMAIN_ID` 必须一致，例如 `<your_id>`。

先确认能看到远端 ROS 图：

```bash
ros2 topic list
ros2 topic echo /tf
```

建图观察：

```bash
ros2 launch robot_navigation lekiwi_lidar_mapping_rviz.launch.py
```

该预设打开 `TF`、`RobotModel`、`/map`、`/scan` 和 `/cloud_registered_body`。

导航观察：

```bash
ros2 launch robot_navigation lekiwi_lidar_navigation_rviz.launch.py
```

该预设打开 `TF`、`RobotModel`、`/map`、`/plan`、`/cloud_registered_body` 和 `/cmd_vel_safe`。

自定义 RViz 配置：

```bash
ros2 launch robot_navigation lekiwi_lidar_mapping_rviz.launch.py rviz_config:=/path/to/custom.rviz
```

## 单点验证

如果一键 `robot_config` 入口因相机型号、内核版本、DDS 或 TF 问题失败，先按下面顺序做单点验证。

### 1. RealSense standalone 出流

```bash
ros2 launch realsense2_camera rs_launch.py
```

RealSense 的 profile、分辨率、帧率、同步和深度对齐参数可以根据具体设备、算力和场景按需修改与调优。单点验证时重点确认彩色图、深度图和 camera_info 等基础话题能够稳定发布，并且 TF 与时间戳能被下游 RTAB-Map 正常消费。

### 2. 静态 TF

```bash
ros2 run tf2_ros static_transform_publisher \
  0 0 0.3 0 0 0 base_link camera_link
```

这只用于单点验证 RTAB-Map。正式建图/导航时，`robot_config` 会按 `lekiwi_*.yaml` 中 `peripherals.realsense.transform` 自动发布。

### 3. RTAB-Map 消费 RGBD

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/depth/image_rect_raw \
  camera_info_topic:=/camera/camera/color/camera_info \
  frame_id:=camera_link \
  approx_sync:=true \
  queue_size:=20 \
  rtabmap_viz:=false \
  rviz:=false \
  database_path:=/tmp/rtabmap_true_$(date +%Y%m%d_%H%M%S).db \
  rtabmap_args:="--Mem/InitWMWithAllNodes true --Mem/IncrementalMemory true --Mem/PermanentMemory false --Mem/STMSize 8"
```

判断 RTAB-Map 真的吃到数据：日志出现 `Odom: quality=...`，`rtabmap` 主节点 `Rate=1.00s`。如果一直 `Did not receive data since 5 seconds!`，先检查 topic 名和 TF。`approx_sync=false` 可作为板端 fallback，但默认配置仍保留 `true`。

### 4. 测试卫生

每轮前检查真实进程、ROS graph 和 graph cache，避免残留进程污染：

```bash
ps -ef | grep -E 'realsense2_camera_node|rtabmap|rgbd_odometry|static_transform_publisher' | grep -v grep
ros2 node list
ros2 topic list
ros2 daemon stop
ros2 daemon start
```

旧 `/tmp/rtabmap.db` 可能引入脏库错误。每轮单点验证用新的 `database_path:=/tmp/rtabmap_..._$(date ...).db`。测板端频率时先关闭 `rqt_image_view` 等 PC 侧 viewer，避免外部订阅者影响测量。

## 底层调试入口

本节用于排查 `robot_config` 完整链路之外的导航子系统问题，例如单独确认 Nav2 子系统、RViz 预设或某个 `robot_navigation` 节点是否可运行。正常建图和导航仍优先使用前文的 `robot_config` 入口。

```bash
# Nav2 子系统入口：map_server + Nav2 navigation_launch.py
# 通常由 robot_config 根据 lekiwi_realsense_navigation.yaml 间接包含
ros2 launch robot_navigation nav2_bringup.launch.py

# 指定地图
ros2 launch robot_navigation nav2_bringup.launch.py map:=/path/to/rtabmap.yaml

# PC 端打开建图观察 RViz 预设
ros2 launch robot_navigation lekiwi_lidar_mapping_rviz.launch.py

# PC 端打开导航观察 RViz 预设
ros2 launch robot_navigation lekiwi_lidar_navigation_rviz.launch.py

# 单独运行节点
ros2 run robot_navigation voice_control
ros2 run robot_navigation nav2_goal_client
ros2 run robot_navigation cmd_vel_bridge_node
ros2 run robot_navigation save_rtabmap_map

# Legacy/debug-only EKF + RTAB-Map entry. For full LeKiwi bringup, use robot_config.
ros2 launch robot_navigation ekf_rtabmap_launch.py
```

直接启动 `nav2_bringup.launch.py` 时，调用方需要确保以下节点已经由 `robot_config` 或其他入口启动：

1. `robot_state_publisher` 和 `/tf_static`
2. ros2_control、controller spawner 与 `/joint_states`
3. `/odom`、`map -> odom` 或等价定位链路
4. 需要时单独启动 `nav2_goal_client`、语音节点和 RViz

## 测试验证

构建统一使用仓库脚本；`colcon test` 用于单点验证某个包或模块的功能，便于把 `robot_navigation` 的 standalone
测试和 `robot_config` 维护的完整仿真测试分开执行。

### 1. 通用编译

```bash
./scripts/build.sh -- --packages-up-to robot_config robot_navigation
```

### 2. robot_navigation standalone 测试

这些测试不启动完整机器人、Gazebo 或 Nav2 bringup，适合 Ubuntu 和 openEuler 都执行。

```bash
colcon test --packages-select robot_navigation --base-paths src --event-handlers console_direct+
```

预期结果：`robot_navigation` 单元测试与软件闭环 E2E 通过。

### 3. robot_config minimal 测试

openEuler、无 Gazebo 或无桌面环境跑 minimal profile。该 profile 会显式跳过 Gazebo/Nav2 完整仿真，只验证测试发现和 skip 策略。

```bash
NAV_TEST_PROFILE=minimal colcon test --packages-select robot_config --base-paths src --event-handlers console_direct+ --pytest-args -k test_navigation_simulation
```

预期结果：`robot_config` navigation simulation 测试 `3 skipped`，skip 原因为 minimal profile 禁用 Gazebo 仿真。

### 4. robot_config full 仿真测试

Ubuntu 且已安装 Gazebo/ros_gz 时运行 full profile。该测试由 `robot_config` 维护，启动 Gazebo、ros2_control、控制器、Nav2、`robot_navigation` 节点和测试辅助节点。

```bash
NAV_TEST_PROFILE=full colcon test --packages-select robot_config --base-paths src --event-handlers console_direct+ --pytest-args -k test_navigation_simulation
```

预期结果：3 个 Gazebo/Nav2 导航 E2E 通过。测试进程会打印临时日志目录，例如 `/tmp/robot_config_nav_sim_xxx`，其中包含 `robot_launch.log`、`odom_bridge.log`、`gt_odom_node.log` 和 `cmd_vel_relay.log`。

### 5. Gazebo GUI 可视化测试

GUI 不是必跑项，只用于 Ubuntu 桌面环境人工观察 Gazebo 窗口。pytest 仍会自动发送导航目标，Gazebo 窗口只负责可视化。

```bash
SIM_GUI=1 NAV_TEST_PROFILE=full colcon test --packages-select robot_config --base-paths src --event-handlers console_direct+ --pytest-args -k test_navigation_simulation
```

## ROS 接口

### voice_control

| 类型 | 话题/服务 | 类型 | 方向 | 说明 |
|------|-----------|------|------|------|
| 订阅 | `/voice_command` | `std_msgs/String` | 输入 | voice_asr_node 输出的识别文本 |
| 订阅 | `/voice_asr/keywords` | `std_msgs/String` | 输入 | 动态更新关键词 (JSON) |
| 发布 | `/voice_asr/keyword_matched` | `std_msgs/String` | 输出 | 匹配的关键词 (JSON) |
| 发布 | `/voice_asr/nav_stop` | `std_msgs/String` | 输出 | 停止导航命令 |
| 服务客户端 | `/action_dispatcher/stop_evaluate` | `std_srvs/Trigger` | 调用 | 匹配 "停止" 时调用 |
| 服务客户端 | `/voice_asr_node/set_hotwords` | `ibrobot_msgs/srv/SetHotwords` | 调用 | 启动时注册热词 |

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `topic_text` | `/voice_command` | voice_asr_node 输出话题 |
| `topic_keyword_matched` | `/voice_asr/keyword_matched` | 匹配结果输出话题 |
| `topic_nav_stop` | `/voice_asr/nav_stop` | 导航停止话题 |
| `keywords_json` | `{}` | 关键词 JSON（从 launch 传入） |
| `keywords_file` | `""` | 关键词 JSON 文件路径 |
| `destinations_json` | `{}` | 目的地名称 → 坐标映射 |

### nav2_goal_client

| 类型 | 话题/服务 | 类型 | 方向 | 说明 |
|------|-----------|------|------|------|
| 订阅 | `/voice_asr/keyword_matched` | `std_msgs/String` | 输入 | 语音命令 |
| 订阅 | `/voice_asr/nav_stop` | `std_msgs/String` | 输入 | 语音停止导航命令 |
| Action | `navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | 调用 | Nav2 导航目标 |
| 服务客户端 | `/action_dispatcher/start_evaluate` | `std_srvs/Trigger` | 调用 | 到达后触发评估 |

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `x` | `0.0` | 默认目标 X 坐标 |
| `y` | `0.0` | 默认目标 Y 坐标 |
| `theta` | `0.0` | 默认目标朝向 (弧度) |
| `timeout_sec` | `60.0` | 导航超时 (秒) |
| `global_frame` | `map` | 目标坐标系 |
| `enable_feedback` | `true` | 是否打印 Nav2 反馈 |
| `trigger_evaluation` | `false` | 到达后是否触发 action_dispatcher 评估 |
| `subscribe_voice` | `true` | 是否订阅语音命令 |
| `topic_keyword_matched` | `/voice_asr/keyword_matched` | 语音命令话题 |
| `topic_nav_stop` | `/voice_asr/nav_stop` | 语音停止导航话题 |

### 导航原子命令接口

LiDAR navigation 阶段提供独立于上层任务框架的单线执行接口：

`ExecuteNavigation` 只负责固定目标和一次性的相对导航命令，不是动态目标 follower，也不提供路径替换协议。
需要路径级控制时，使用 Nav2 原生 `/compute_path_to_pose`（planner ID `GridBased`）和 `/follow_path`
（controller ID `FollowPath`）；这些原生接口不包含本 Action 的单任务和速度归零确认语义。

| 类型 | 名称 | 类型 | 说明 |
|------|------|------|------|
| Action | `/navigation/execute` | `ibrobot_msgs/action/ExecuteNavigation` | 下发绝对或相对目标，接收阶段、进度和最终结果 |
| Service | `/navigation/cancel_current` | `std_srvs/srv/Trigger` | 中断当前受管目标并等待最终速度指令归零 |

绝对目标使用 `map` frame。相对目标支持前进、后退、左右横移和左右旋转，并在请求时根据
`map -> base_link` 固定目标位姿，再交给 Nav2 通过当前动态避障链规划和执行。接口同时只执行
一个目标；需要改变目标时，调用方先等待 cancel service 成功，再发送新 Action goal。

该接口提供两种调用方式：

- `nav_cmd`：面向人工操作、现场验收和快速诊断的 CLI 包装。
- `ExecuteNavigation` Action：面向其他 ROS 2 模块的程序化 API。业务模块应复用持久化的
  `ActionClient`，避免每次启动 ROS CLI 进程产生发现延迟。

#### CLI 用法

CLI 的平移参数单位为米，转向参数单位为度。`--action-name` 现在是必填参数，默认值为
`/navigation/execute`，必须与 `robot_config.navigation.command_server.action_name` 保持一致：

```bash
ros2 run robot_navigation nav_cmd status --action-name /navigation/execute
ros2 run robot_navigation nav_cmd forward 0.10 --action-name /navigation/execute
ros2 run robot_navigation nav_cmd backward 0.10 --action-name /navigation/execute
ros2 run robot_navigation nav_cmd leftward 0.10 --action-name /navigation/execute
ros2 run robot_navigation nav_cmd rightward 0.10 --action-name /navigation/execute
ros2 run robot_navigation nav_cmd turn-left 10 --action-name /navigation/execute
ros2 run robot_navigation nav_cmd turn-right 10 --action-name /navigation/execute
ros2 run robot_navigation nav_cmd absolute 1.0 0.5 90 --action-name /navigation/execute
ros2 run robot_navigation nav_cmd cancel --action-name /navigation/execute
```

`status` 同时检查 `/navigation/execute` 和 Nav2 `/navigate_to_pose` 是否可用；`cancel` 会等待
当前导航任务结束并确认安全速度输出归零。

##### Action name SSOT

`/navigation/execute` 的 action name 来自 `robot_config.navigation.command_server.action_name`
配置（SSOT）。`robot_config.loader` 会在加载阶段校验该字段，并将其作为
`robot_execution_endpoints` 的 `navigation_action_name` 投影进 canonical execution-context
digest。下游节点（`navigation_command_server`、`nav_cmd`、`skill_library` 的
`ExecuteNavigation` dispatch）必须从该 SSOT 读取，不允许在节点参数或代码中重新声明该 action name。
旧字段 `embodied.execution.navigation_action_name` 已退役，loader 检测到该键时直接报错。

##### Costmap readiness 检查

`navigation_command_server` 在向 Nav2 发送 `NavigateToPose` goal 前，会先检查
`/global_costmap/costmap` 数据是否非空（避免在空 costmap 上规划）。检查以
`costmap_readiness_timeout` 参数控制，默认 `60.0` 秒：

- 在该超时窗口内一直探测 `/global_costmap/costmap`，直到拿到非空 occupancy grid 才向 Nav2
  发送 goal；
- 超时仍未拿到非空 costmap 时，`ExecuteNavigation` 直接以 `NAV2_UNAVAILABLE` 终态返回，
  不会向 Nav2 发送 goal，也不会取得 root lease；
- 该检查只覆盖 costmap 数据可用性，不保证定位收敛或 planner/controller 已就绪，调用方仍需
  自行确认 `map -> base_link` 稳定与 Nav2 lifecycle active。

##### Cancel / cleanup 语义

`/navigation/cancel_current` 与 `ExecuteNavigation` goal 取消遵循同一清理路径：
cancel 请求会等待 Nav2 取消当前 goal、并等待底盘速度指令稳定到零。若速度未能在
`cancel_cleanup_timeout_sec` 内收敛到零（典型原因：Collision Monitor 仍在发布非零速度、
底盘 bridge 仍在执行残余指令、或下游 controller 拒绝取消），action 进入
`STOP_TIMEOUT` 终态：

- `STOP_TIMEOUT` 是 fail-closed 终态：`ExecuteNavigation.Result.success=false`、
  `error_code=STOP_TIMEOUT`，调用方**不得**释放 root lease，必须显式 finalization 才能恢复；
- `INTERNAL_ERROR` 终态同样不释放 root lease，由 coordinator ledger terminal record
  保留 lease 直到操作员介入或后续 `FinalizeWorkflowExecution` 以匹配终态幂等收敛；
- 正常 `SUCCEEDED`/`CANCELLED` 终态才会释放 root lease 与 runtime bundle retention。

#### ROS 2 Action API

其他模块可以直接使用 `ibrobot_msgs/action/ExecuteNavigation`，无需启动 CLI 子进程：

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from ibrobot_msgs.action import ExecuteNavigation


class NavigationCaller(Node):
    def __init__(self):
        super().__init__("navigation_caller")
        self.client = ActionClient(self, ExecuteNavigation, "/navigation/execute")

    def send_forward(self, distance_m):
        goal = ExecuteNavigation.Goal()
        goal.command_type = ExecuteNavigation.Goal.FORWARD
        goal.value = distance_m
        return self.client.send_goal_async(goal, feedback_callback=self.on_feedback)

    def on_feedback(self, feedback_message):
        feedback = feedback_message.feedback
        self.get_logger().info(
            f"state={feedback.state}, distance_remaining={feedback.distance_remaining:.3f} m"
        )


rclpy.init()
node = NavigationCaller()
node.client.wait_for_server()
goal_future = node.send_forward(0.5)
rclpy.spin_until_future_complete(node, goal_future)
goal_handle = goal_future.result()
if goal_handle is None or not goal_handle.accepted:
    raise RuntimeError("navigation goal was rejected")

result_future = goal_handle.get_result_async()
rclpy.spin_until_future_complete(node, result_future)
result = result_future.result().result
if not result.success:
    raise RuntimeError(f"navigation failed: {result.error_code} {result.message}")

node.destroy_node()
rclpy.shutdown()
```

API 中的 `value` 对平移命令使用米，对转向命令使用弧度；例如左转 10 度应设置为
`math.radians(10.0)`。绝对目标使用 `ABSOLUTE_POSE`，`target_pose.header.frame_id` 必须为
`map`，并填写 `target_pose.pose`。调用方可以通过 `goal_handle.cancel_goal_async()` 取消当前
Action，也可以调用 `/navigation/cancel_current`，后者会在速度归零确认后返回。

`ExecuteNavigation.Result` 提供 `success`、`error_code`、`message` 和
`resolved_target_pose`；执行过程中的 `Feedback` 提供 `state`、`distance_remaining`、
`estimated_time_remaining` 和 `number_of_recoveries`。接口同时只管理一个导航任务，业务模块在
发送新目标前应先完成当前目标或等待取消服务成功。

MID-360 navigation 使用 AMCL 发布 `map -> odom`，FAST-LIO odom bridge 发布 `odom -> base_link`，动态避障
速度链为 `/cmd_vel -> Collision Monitor -> /cmd_vel_safe -> cmd_vel_bridge`。本文后续 RTAB-Map/EKF 和
不启动 AMCL 的章节只适用于 RealSense legacy profile。navigation stage 运行时不要额外启动未经仲裁的
`teleop_twist_keyboard` 或其他直接发布速度的节点；本 PR 不提供完整速度仲裁器。

### cmd_vel_bridge_node

| 类型 | 话题 | 类型 | 方向 | QoS | 说明 |
|------|------|------|------|-----|------|
| 订阅 | `/cmd_vel` | `geometry_msgs/Twist` | 输入 | Reliable | 速度指令 (vx, vy, vtheta) |
| 订阅 | `/joint_states` | `sensor_msgs/JointState` | 输入 | Best Effort | 轮子反馈 (joints "7", "8", "9") |
| 条件订阅 | `/motion_mode/navigation_enabled` | `std_msgs/Bool` | 输入 | Transient Local | 导航命令授权；抓取模式持续输出零速 |
| 发布 | `/base_velocity_controller/commands` | `std_msgs/Float64MultiArray` | 输出 | Reliable | 原始轮速 [left, back, right] rad/s |
| 发布 | `/odom` | `nav_msgs/Odometry` | 输出 | Reliable | 里程计 |
| 条件发布 | `/motion_mode/base_navigation_enabled` | `std_msgs/Bool` | 输出 | Reliable | 已清除旧命令并输出零速的模式确认 |
| 条件发布 | TF: `odom → base_link` | TransformStamped | 输出 | - | 仅当 `publish_tf: true` 时发布 |

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `wheel_radius` | `0.05` | 轮子半径 (m) |
| `base_radius` | `0.125` | 底盘半径 (m) |
| `max_radps` | `4.602` | 最大轮速 (rad/s)，对应 base_vel_max_raw=3000 steps/s |
| `odom_frame` | `odom` | 里程计坐标系 |
| `base_frame` | `base_link` | 机器人基坐标系 |
| `publish_tf` | `true` | 是否发布 TF（启用 EKF 时应设为 `false`） |
| `control_frequency` | `50.0` | 控制频率 (Hz) |
| `cmd_timeout` | `0.5` | 无指令超时后归零输出 (s) |
| `cmd_vel_topic` | `/cmd_vel` | 订阅的速度指令话题 |
| `joint_states_topic` | `/joint_states` | 订阅的关节状态话题（轮子反馈） |
| `odom_topic` | `/odom` | 发布的里程计话题 |
| `motion_mode_enabled` | `false` | 是否启用抓取/导航常驻互锁 |
| `navigation_enabled_on_startup` | `true` | 互锁启用时的初始底盘授权状态 |
| `navigation_enabled_topic` | `/motion_mode/navigation_enabled` | Gateway 发布的模式状态 |
| `navigation_mode_ack_topic` | `/motion_mode/base_navigation_enabled` | 底盘模式确认话题 |

**运动学**: 三个全向轮安装角度分别为 150°、-90°、30°（对应 joint 7/left, 8/back, 9/right）。IK 通过 3×3 矩阵 `M @ [vx, vy, vtheta]` 计算轮速，输出单位为 rad/s（ros2_control 速度指令接口）。FK 通过 M 的伪逆将轮子反馈还原为机体速度。超过 `max_radps` 时按比例缩放所有轮速。

互锁启用后，抓取模式会忽略新的 `/cmd_vel` 并在每个控制周期发布三轮零速。任何模式切换都会先清除缓存速度，
因此进入导航后必须收到一条新的 `/cmd_vel` 才会移动，旧的 Nav2 指令不会在切换后恢复执行。

## 关键词配置

`config/keywords.json` 定义语音关键词（正则表达式）和动作映射：

```json
{
  "keywords": {
    "去.*a点|到.*a点|a点": {
      "type": "destination",
      "info": {"destination": "point_a"}
    },
    "捡.*蓝色方块|拿.*蓝色方块|蓝色方块": {
      "type": "action",
      "info": {"task_description": "Pick up the blue square"}
    },
    "停止|停下": {
      "type": "stop",
      "info": {"task_description": "Stop current action"}
    }
  }
}
```

`destination` 类型的 `destination` 字段通过 `destinations_json` 参数（来自 `robot_config`）解析为实际坐标 (x, y, theta)。

| 类型 | 作用 | info 字段 |
|------|------|-----------|
| `destination` | 触发导航 | `destination` — 在 destinations_json 中查找坐标 |
| `action` | 设置任务描述 | `task_description` — 到达后传给 action_dispatcher |
| `stop` | 停止导航和推理 | `task_description` — "Stop current action" |

## 定位融合 (EKF)

`config/ekf.yaml` 配置 `robot_localization` EKF 节点，30Hz，2D 模式：

| 输入源 | 话题 | 融合内容 | 作用 |
|--------|------|---------|------|
| 底盘里程计 | `/odom` | X/Y 速度 + 偏航角速度 | 高频实时速度（来自 cmd_vel_bridge FK） |

**设计说明**:
- EKF 仅融合底盘里程计速度（`/odom`），输出平滑的 `odom → base_link` TF
- **不**融合 RTAB-Map 视觉里程计（`/rtabmap/odom`），因为两者 child_frame 不同会导致冲突
- RTAB-Map 通过 `map → odom` TF 做全局定位修正，EKF 不参与该环节
- `publish_tf: true`：EKF 接管 `odom → base_link` TF 发布，`cmd_vel_bridge` 的 `publish_tf` 需设为 `false`

## RTAB-Map 配置

RTAB-Map 以定位模式运行（`localization: true`），通过视觉 SLAM 发布 `map → odom` TF。

关键参数：
- `frame_id: camera_link` — RTAB-Map 使用的相机坐标系，由 `robot_config` 的 RealSense adapter 和 TF 配置提供
- `approx_sync: true` — RGB 和 Depth 近似同步
- `--Reg/Force3DoF true` — 强制 2D 模式，防止相机倾斜导致的 pitch/yaw 偏差（对 2D 全向轮机器人至关重要）
- `--Mem/InitWMWithAllNodes true` — 初始化时加载所有节点
- `--Mem/STMSize 8` — 短期记忆大小
- 更新频率约 ~1Hz（`map → odom` TF），受视觉特征提取计算开销限制（`Vis/MaxFeatures: 1000`）

## Nav2 配置概要

`config/nav2_params.yaml` 配置完整 Nav2 栈：

| 组件 | 关键参数 |
|------|---------|
| **Localization** | 不启动 AMCL；RTAB-Map 使用保存地图发布 `map → odom`，EKF 发布 `odom → base_link` |
| **Controller** | DWB 局部规划器，20Hz，max vel 0.26 m/s，max theta 1.0 rad/s，goal tolerance xy: 0.05 / yaw: 0.1 |
| **Local costmap** | 3×3m 滚动窗口，0.05m 分辨率，robot radius 0.22m，`/scan` 话题，`transform_tolerance: 3.0` |
| **Global costmap** | Static + Obstacle + Inflation 层，`transform_tolerance: 3.0` |
| **Planner** | NavfnPlanner，tolerance: 0.5 |
| **Behaviors** | spin, backup, drive_on_heading, assisted_teleop, wait |
| **Velocity smoother** | max [0.26, 0.26, 1.0]，odom_topic: `/odometry/filtered` |

### 重要配置说明

- **`use_sim_time: False`**: 所有 Nav2 节点均使用系统时间。实车必须为 `False`，否则 TF 查找失败 ("Transform data too old")
- **`transform_tolerance: 3.0`**: RTAB-Map 更新频率约 1Hz，需较宽松的 TF 容差避免 "Transform timeout"
- **DWB 参数调优**: `sim_time: 1.0`（轨迹预测时长），`PathAlign.scale: 12.0`，`GoalAlign.scale: 8.0`（降低权重减少短距离导航振荡）

## 完整工作流程

```text
 1. 用户说："去a点"
 2. voice_asr_node (sherpa-onnx) 本地语音识别，输出 "去a点" 到 /voice_command
 3. voice_control 订阅 /voice_command，用正则匹配关键词，从 destinations_json 解析 point_a 的坐标
 4. 发布 JSON 到 /voice_asr/keyword_matched
 5. nav2_goal_client 收到消息，发送 NavigateToPose Action 给 Nav2
 6. Nav2 进行全局/局部规划，通过 controller_server 发布 /cmd_vel
 7. cmd_vel_bridge_node 将 /cmd_vel 通过 IK 转换为全向轮角速度 (rad/s)
 8. 发布到 /base_velocity_controller/commands，经由 ros2_control → lekiwi_hardware 驱动电机
 9. cmd_vel_bridge_node 同时通过 FK 计算里程计，发布 /odom
10. EKF 融合 /odom 速度数据，发布 odom → base_link TF
11. RTAB-Map 通过视觉 SLAM 发布 map → odom TF 进行全局修正
12. 到达目标后，nav2_goal_client 检查是否有缓存的 task_description
13. 如果有，调用 /action_dispatcher/start_evaluate 触发机械臂推理
14. 如果用户说"停止"，voice_control 调用 /action_dispatcher/stop_evaluate 并取消导航
```

## 启动文件

| 文件 | 启动内容 |
|------|---------|
| `nav2_bringup.launch.py` | Nav2 子系统：map_server + Nav2 navigation_launch.py |
| `ekf_rtabmap_launch.py` | Legacy/debug-only RTAB-Map + EKF 入口；正式 LeKiwi 实机链路由 `robot_config` 启动 |
| `lekiwi_lidar_mapping_rviz.launch.py` | PC 端 LiDAR 建图观察 RViz 预设 |
| `lekiwi_lidar_navigation_rviz.launch.py` | PC 端 LiDAR 导航观察 RViz 预设 |
| `lekiwi_realsense_mapping_rviz.launch.py` | RealSense/RTAB-Map 建图 RViz 预设 |
| `lekiwi_realsense_navigation_rviz.launch.py` | RealSense/RTAB-Map 导航 RViz 预设 |

## 目录结构

```
robot_navigation/
├── config/
│   ├── keywords.json              # 关键词正则 → 动作映射
│   ├── ekf.yaml                   # EKF 传感器融合配置
│   ├── config.rviz                # RViz2 可视化配置
│   └── nav2_params.yaml           # Nav2 完整参数栈
├── launch/
│   ├── nav2_bringup.launch.py     # Nav2 子系统入口
│   ├── ekf_rtabmap_launch.py      # Legacy/debug-only EKF + RTAB-Map
│   ├── lekiwi_lidar_mapping_rviz.launch.py
│   ├── lekiwi_lidar_navigation_rviz.launch.py
│   ├── lekiwi_realsense_mapping_rviz.launch.py
│   └── lekiwi_realsense_navigation_rviz.launch.py
├── robot_navigation/
│   ├── voice_control.py           # 语音关键词匹配 + 导航桥接
│   ├── nav2_goal_client.py        # Nav2 Action 客户端 + 评估触发
│   └── cmd_vel_bridge_node.py     # cmd_vel 桥接 + IK/FK + 里程计
├── test/
│   ├── test_voice_control.py      # 语音控制 pytest 测试
│   ├── test_cmd_vel_bridge.py     # cmd_vel 桥接 pytest 测试（IK/FK/里程计）
│   ├── test_nav2_goal_client.py   # Nav2 Goal 客户端 pytest 测试
│   └── e2e/
│       ├── mock_servers.py        # Mock Trigger/Action 服务
│       └── test_pipeline_software.py   # 软件闭环 E2E（无物理仿真）
└── README.md
```

完整机器人启动链路下的导航仿真测试由 `robot_config` 维护。

## 依赖

### ROS2 包

- `nav2_bringup` — Nav2 导航栈
- `robot_localization` — EKF 传感器融合
- `robot_state_publisher` — URDF 坐标变换
- `joint_state_publisher` — 关节状态发布
- `rtabmap_launch` — RTAB-Map 视觉 SLAM
- `rviz2` — 可视化
- `robot_config` — 配置加载（destinations、contract）
- `action_dispatch` — 动作分发与评估（通过服务调用联动）

### Python 包

- `sherpa-onnx` — 本地语音识别（由 voice_asr_service 包提供）
- `ibrobot_msgs` — 自定义消息/服务（SetHotwords 等）

## 故障排除

### 导航时机器人不走直线（偏移/扭转）

导航偏移通常由以下原因叠加导致，按优先级排查：

1. **`use_sim_time` 配置错误**: 实车必须设为 `False`，否则 TF 查找失败
   ```bash
   ros2 param get /controller_server use_sim_time  # 应返回 "false"
   ```

2. **RTAB-Map pitch 偏差**: 相机安装倾斜会导致 yaw 偏差，需启用 `--Reg/Force3DoF true`
   ```bash
   ros2 run tf2_ros tf2_echo map odom  # 检查是否有非零 pitch/roll
   ```

3. **全局定位 TF 冲突**: 多个节点同时广播 `map → odom` 会导致 TF 跳变。正式 LeKiwi 实机链路不启动 AMCL，由 RTAB-Map 负责 `map → odom`
   ```bash
   ros2 run tf2_ros tf2_echo map odom
   ros2 topic echo /rtabmap/info
   ```

4. **EKF 订阅了不存在的话题**: EKF 无法更新会导致 TF 停滞
   ```bash
   ros2 topic hz /odom  # 确认有数据
   ```

5. **TF 发布冲突**: `cmd_vel_bridge` 和 EKF 同时发布 `odom → base_link` TF
   ```bash
   ros2 param get /cmd_vel_bridge publish_tf  # 启用 EKF 时应为 "false"
   ```

### 麦克风无法使用

```bash
sudo usermod -a -G audio $USER
# 重新登录后生效
```

### 语音识别无输出

```bash
# 检查 voice_asr_node 是否在运行
ros2 node list | grep voice

# 检查 /voice_command 是否有数据
ros2 topic echo /voice_command

# 检查 voice_control 是否收到文本
ros2 topic echo /rosout --filter "msg.name == 'voice_control'"
```

### 底盘无响应

```bash
# 检查 cmd_vel_bridge 是否正常发布
ros2 topic echo /base_velocity_controller/commands

# 检查里程计输出
ros2 topic echo /odom

# 检查 /cmd_vel 是否有输入
ros2 topic echo /cmd_vel
```

### 导航不触发评估

```bash
# 检查 nav2_goal_client 参数
ros2 param get /nav2_goal_client trigger_evaluation

# 检查 action_dispatcher 服务是否可用
ros2 service list | grep action_dispatcher
```

### EKF 融合异常

```bash
# 检查输入源是否正常
ros2 topic hz /odom

# 检查 EKF 输出
ros2 topic echo /odometry/filtered

# 检查 TF 树是否完整
ros2 run tf2_tools view_frames
```

### 语音命令未被 nav2_goal_client 接收

```bash
# /voice_asr/keyword_matched 话题需持续发布（非 --once），QoS 类型可能不匹配
# 测试时使用 -r 2 持续发布
ros2 topic pub -r 2 /voice_asr/keyword_matched std_msgs/msg/String "{data: '{\"type\": \"destination\", \"info\": {\"destination\": \"point_a\"}}'}"
```

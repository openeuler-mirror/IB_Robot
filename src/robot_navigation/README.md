# robot_navigation

机器人导航包，集成了语音识别、Nav2 导航、全向轮底盘桥接和定位融合功能。

## 功能特性

- **语音识别**: 基于 FunASR 的实时语音识别和正则关键词匹配
- **导航控制**: Nav2 导航 Goal 客户端，支持语音触发导航，到达后自动触发机械臂推理
- **底盘桥接**: `cmd_vel_bridge_node` 通过 IK/FK 将标准 `/cmd_vel` 桥接到 ros2_control 全向轮速度指令 (rad/s)，并发布里程计
- **定位融合**: EKF (robot_localization) 融合底盘里程计速度，RTAB-Map 视觉 SLAM 提供全局定位修正
- **任务联动**: 语音 → 导航 → 到达 → 触发 action_dispatcher 评估，形成完整任务链

## 系统架构

```text
  用户语音 ──► funasr_client_node ──► /voice_asr/keyword_matched
                    │                         │
                    │                         ▼
                    │               nav2_goal_client
                    │                         │
                    │                         ▼
                    │               Nav2 NavigateToPose Action
                    │                         │
                    │                         ▼
                    │              Nav2 规划 + 发布 /cmd_vel
                    │                         │
                    │                         ▼
                    │              到达目标 ──► /action_dispatcher/start_evaluate
                    │
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

注意: `cmd_vel_bridge_node` 仅发布 `/odom` 话题（供 EKF 订阅），不发布 TF（`publish_tf: false`）。
AMCL 的 `tf_broadcast` 设为 `false`，避免与 RTAB-Map 的 `map → odom` TF 冲突。

## 节点列表

| 节点 | 功能 | 入口点 |
|------|------|--------|
| `funasr_client_node` | FunASR 语音识别 + 关键词匹配 | `robot_navigation.funasr_client_node` |
| `nav2_goal_client` | Nav2 导航 Goal 客户端 + 评估触发 | `robot_navigation.nav2_goal_client` |
| `cmd_vel_bridge_node` | cmd_vel → ros2_control 桥接 + 里程计发布 | `robot_navigation.cmd_vel_bridge_node` |

## 快速开始

### 1. 编译

```bash
cd ~/workspace/IB_Robot
colcon build --packages-select robot_navigation robot_config
source install/setup.bash
```

### 2. 启动 FunASR 服务器（语音控制需要）

```bash
# 一键安装（首次）
bash scripts/install.sh ${deploy_path}

# 启动服务（默认端口 10086）
bash scripts/start_service.sh ${deploy_path}
```

### 3. 通过 robot_config 启动（推荐）

`robot_config` 集成了导航功能，通过 YAML 配置文件统一管理所有参数。以 `lekiwi_navi` 机器人为例：

```bash
# 设置 ROS_DOMAIN_ID（避免与其他 ROS2 系统冲突）
export ROS_DOMAIN_ID=<your_id>

# 遥操
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=teleop

# 导航评估模式（带推理）
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=navi

```

**配置示例** (`lekiwi_navi.yaml`):

```yaml
navigation:
  enabled: true
  nav2_bringup:
    enabled: true
    map_file: "~/workspace/map/rtabmap.yaml"
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
  funasr:
    host: "127.0.0.1"
    port: "10095"
  rviz:
    enabled: true
```

**优势**:
- YAML 单一数据源，统一配置管理
- 自动启动相关组件（控制器、相机、TF、定位等）
- 支持 `control_mode` 切换不同运行模式

### 4. 直接启动（备选方案）

如果不使用 `robot_config`，可以直接启动 `robot_navigation` 的 launch 文件：

```bash
# 完整启动 (Nav2 + robot_state_publisher + nav2_goal_client + RViz)
ros2 launch robot_navigation nav2_bringup.launch.py

# 指定地图
ros2 launch robot_navigation nav2_bringup.launch.py map:=/path/to/rtabmap.yaml

# 仅语音识别
ros2 launch robot_navigation funasr_client.launch.py

# EKF + RTAB-Map + RealSense
ros2 launch robot_navigation ekf_rtabmap_launch.py
```

### 5. 单独运行节点

```bash
ros2 run robot_navigation funasr_client_node
ros2 run robot_navigation nav2_goal_client
ros2 run robot_navigation cmd_vel_bridge_node
```

## ROS 接口

### funasr_client_node

| 类型 | 话题/服务 | 类型 | 方向 | 说明 |
|------|-----------|------|------|------|
| 订阅 | `/voice_asr/keywords` | `std_msgs/String` | 输入 | 动态更新关键词 (JSON) |
| 发布 | `/voice_asr/text` | `std_msgs/String` | 输出 | ASR 原始识别文本 |
| 发布 | `/voice_asr/status` | `std_msgs/String` | 输出 | 连接状态 (connecting/connected/error/disconnected) |
| 发布 | `/voice_asr/keyword_matched` | `std_msgs/String` | 输出 | 匹配的关键词 (JSON) |
| 发布 | `/voice_asr/nav_stop` | `std_msgs/String` | 输出 | 停止导航命令 |
| 服务 | `~/start` | `std_srvs/Trigger` | 调用 | 启动语音识别 |
| 服务 | `~/stop` | `std_srvs/Trigger` | 调用 | 停止语音识别 |
| 服务客户端 | `/action_dispatcher/stop_evaluate` | `std_srvs/Trigger` | 调用 | 匹配 "停止" 时调用 |

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `127.0.0.1` | FunASR 服务器地址 |
| `port` | `10095` | FunASR 服务器端口 |
| `use_itn` | `true` | 逆文本规范化 |
| `mode` | `2pass` | 识别模式 |
| `chunk_size` | `[5, 10, 5]` | 音频分块大小 |
| `chunk_interval` | `10` | 分块间隔 |
| `hotword_msg` | `""` | 热词 |
| `keywords_file` | `""` | 关键词 JSON 文件路径 |
| `keywords_json` | `{}` | 关键词 JSON（从 launch 传入） |
| `destinations_json` | `{}` | 目的地名称 → 坐标映射 |
| `auto_start` | `true` | 是否自动启动 |

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

### cmd_vel_bridge_node

| 类型 | 话题 | 类型 | 方向 | QoS | 说明 |
|------|------|------|------|-----|------|
| 订阅 | `/cmd_vel` | `geometry_msgs/Twist` | 输入 | Reliable | 速度指令 (vx, vy, vtheta) |
| 订阅 | `/joint_states` | `sensor_msgs/JointState` | 输入 | Best Effort | 轮子反馈 (joints "7", "8", "9") |
| 发布 | `/base_velocity_controller/commands` | `std_msgs/Float64MultiArray` | 输出 | Reliable | 原始轮速 [left, back, right] rad/s |
| 发布 | `/odom` | `nav_msgs/Odometry` | 输出 | Reliable | 里程计 |
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

**运动学**: 三个全向轮安装角度分别为 150°、-90°、30°（对应 joint 7/left, 8/back, 9/right）。IK 通过 3×3 矩阵 `M @ [vx, vy, vtheta]` 计算轮速，输出单位为 rad/s（ros2_control 速度指令接口）。FK 通过 M 的伪逆将轮子反馈还原为机体速度。超过 `max_radps` 时按比例缩放所有轮速。

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
- `frame_id: realsense_link` — 相机坐标系
- `approx_sync: true` — RGB 和 Depth 近似同步
- `--Reg/Force3DoF true` — 强制 2D 模式，防止相机倾斜导致的 pitch/yaw 偏差（对 2D 全向轮机器人至关重要）
- `--Mem/InitWMWithAllNodes true` — 初始化时加载所有节点
- `--Mem/STMSize 8` — 短期记忆大小
- 更新频率约 ~1Hz（`map → odom` TF），受视觉特征提取计算开销限制（`Vis/MaxFeatures: 1000`）

## Nav2 配置概要

`config/nav2_params.yaml` 配置完整 Nav2 栈：

| 组件 | 关键参数 |
|------|---------|
| **AMCL** | OmniMotionModel（全向轮），500-2000 粒子，`tf_broadcast: false`（避免与 RTAB-Map 冲突），`odom_frame_id: "odom"` |
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
2. funasr_client_node 通过 WebSocket 将音频流发送给 FunASR 服务器
3. FunASR 返回 2pass-offline 最终识别文本 "去a点"
4. funasr_client_node 用正则匹配关键词，从 destinations_json 解析 point_a 的坐标
5. 发布 JSON 到 /voice_asr/keyword_matched
6. nav2_goal_client 收到消息，发送 NavigateToPose Action 给 Nav2
7. Nav2 进行全局/局部规划，通过 controller_server 发布 /cmd_vel
8. cmd_vel_bridge_node 将 /cmd_vel 通过 IK 转换为全向轮角速度 (rad/s)
9. 发布到 /base_velocity_controller/commands，经由 ros2_control → lekiwi_hardware 驱动电机
10. cmd_vel_bridge_node 同时通过 FK 计算里程计，发布 /odom
11. EKF 融合 /odom 速度数据，发布 odom → base_link TF
12. RTAB-Map 通过视觉 SLAM 发布 map → odom TF 进行全局修正
13. 到达目标后，nav2_goal_client 检查是否有缓存的 task_description
14. 如果有，调用 /action_dispatcher/start_evaluate 触发机械臂推理
15. 如果用户说"停止"，funasr_client_node 调用 /action_dispatcher/stop_evaluate 并取消导航
```

## FunASR 语音服务部署

`scripts/` 目录提供 FunASR 服务器的一键安装和启动脚本。

### 系统依赖

- `libopenblas-dev`
- `libssl-dev`
- `portaudio19-dev` (Ubuntu) / `portaudio-devel` (CentOS)

### 安装

```bash
bash scripts/install.sh ${deploy_path}
```

自动完成：
- pip 安装 `funasr`、`humanfriendly`
- 下载 onnxruntime 和 ffmpeg（自动识别 x86_64 / aarch64）
- 克隆 FunASR 仓库并编译 WebSocket 服务端

### 启动服务

```bash
bash scripts/start_service.sh ${deploy_path}
```

启动 FunASR 2pass 语音识别 WebSocket 服务，默认端口 `10086`（客户端连接端口通过 `port` 参数配置，默认 `10095`）。

## 启动文件

| 文件 | 启动内容 |
|------|---------|
| `nav2_bringup.launch.py` | Nav2 栈 + robot_state_publisher + nav2_goal_client + RViz2 |
| `funasr_client.launch.py` | 仅 funasr_client_node（读取 keywords.json 和 robot_config destinations） |
| `ekf_rtabmap_launch.py` | RTAB-Map + EKF 融合（RealSense 由 peripherals 启动） |

## 目录结构

```
robot_navigation/
├── config/
│   ├── keywords.json              # 关键词正则 → 动作映射
│   ├── ekf.yaml                   # EKF 传感器融合配置
│   ├── config.rviz                # RViz2 可视化配置
│   └── nav2_params.yaml           # Nav2 完整参数栈
├── launch/
│   ├── nav2_bringup.launch.py     # Nav2 + 状态发布 + GoalClient + RViz
│   ├── funasr_client.launch.py    # 语音识别独立启动
│   └── ekf_rtabmap_launch.py      # EKF + RTAB-Map
├── robot_navigation/
│   ├── funasr_client_node.py      # 语音识别 + 关键词匹配 (~460 行)
│   ├── nav2_goal_client.py        # Nav2 Action 客户端 + 评估触发 (~330 行)
│   └── cmd_vel_bridge_node.py     # cmd_vel 桥接 + IK/FK + 里程计 (~350 行)
├── scripts/
│   ├── install.sh                 # FunASR 服务端一键安装
│   └── start_service.sh           # FunASR 2pass WebSocket 服务启动
└── README.md
```

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

- `pyaudio` — 麦克风音频采集
- `websockets` — FunASR WebSocket 通信

### 系统依赖

- FunASR 2pass 服务端（需单独部署）

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

3. **AMCL 与 RTAB-Map TF 冲突**: 两者同时广播 `map → odom` 会导致 TF 跳变
   ```bash
   ros2 param get /amcl tf_broadcast  # 应返回 "false"
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

### FunASR 连接失败

```bash
# 检查 FunASR 服务器是否在运行
netstat -an | grep 10086

# 检查客户端连接端口配置
ros2 param get /funasr_client_node port
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

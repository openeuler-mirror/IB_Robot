# Dataset Tools

ROS 2 数据集采集与转换工具，用于 LeRobot v3 数据集格式。

## 概述

本包提供以下功能：

- **Episode 录制**: 通过 Action Server 控制的分段录制
- **Bag 转 LeRobot**: 将 ROS 2 bag 转换为 LeRobot v3 数据集格式
- **相机标定辅助**: `camera_alignment` 支持真机 OpenCV 相机与仿真 ROS 2 image topic；仿真模式通过 `rclpy` 订阅图像，并可与 `sim_models` 中的相机调节/ArUco 标定板工具联动

仿真相关能力是可选运行路径：常规录制与 bag 转换仍只依赖 `robot_config` 契约和 ROS 2 bag；只有使用 `camera_alignment --use_sim` 处理仿真相机 topic 时，才会进入 `dataset_tools → sim_models → robot_config` 的跨包交互。

## 架构设计

### 仿真相机标定边界

`camera_alignment --use_sim` 是仿真相机标定辅助路径，不参与 episode 录制或 bag 转换主流程。该路径的职责边界如下：

1. `dataset_tools.camera_alignment` 负责交互式对齐 UI、参考图/参考 marker 数据保存，以及订阅仿真相机 ROS 2 image topic；
2. `sim_models` 负责仿真侧相机调节辅助进程、Gazebo ArUco 标定板 spawn/despawn，以及将调节结果保存为 `~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml`；
3. `robot_config` 在后续 `robot.launch.py use_sim:=true` 启动时读取 robot YAML 和可选 override，生成对应仿真相机位姿。

因此，`sim_models` 只属于 `camera_alignment --use_sim` 的可选运行时依赖；真机相机对齐、录制服务和 bag 转换不依赖该包。

### 单一真理来源 (Single Source of Truth)

所有数据集工具使用 `robot_config` 包下的配置文件作为唯一配置来源，例如：

```
src/robot_config/config/robots/so101_single_arm.yaml
├── contract.observations    ← 观测定义（相机、状态等）
├── contract.actions         ← 动作定义（arm、gripper）
├── contract.rate_hz         ← 采样率
└── control_modes            ← 运行时控制模式配置
```

这确保了：
- 训练数据导出与在线推理配置一致
- 无需维护重复的 contract 文件
- 配置变更自动传播到所有组件

SO101 配置中的 `observation.current` 从 `/so101_follower/joint_currents` 的 `ibrobot_msgs/msg/JointCurrent.current` 字段解码，selector 使用 `current.*`，单位为安培。历史数据集缺少 `observation.current` 时，frame_detector 会跳过 critical frame 检测，并让 freeze frame 检测退化为仅使用速度判断。

LeRobot 单位转换的标定来源同样来自 `robot_config`。单臂旧配置使用
`ros2_control.calib_file`；双臂或更多来源使用
`ros2_control.xacro_args.calib_file_<namespace>`，例如 `calib_file_left`、
`calib_file_right`、`calib_file_front` 或 `calib_file_1`。后缀即 LeRobot joint namespace，
允许字母、数字和下划线；建议优先使用 `left`、`right`、`front` 这类语义名称，
便于和 `joint_names`、数据集 metadata 及策略特征对齐。`calib_file` 不能与
namespace 后缀标定来源混用。

Episode 录制会在 dataset metadata 中保存 LeRobot conversion snapshot。
`bag_to_lerobot` 转换旧 dataset 时按以下顺序恢复转换表：已有 dataset metadata
中的 calibration snapshot、从 `robot_config` 解析出的 named calibration sources、
最后才是 legacy `calibration_file` pathsep 字符串。`policy_eval` 的静态
calibration 检查也复用同一解析规则，因此重复 namespace 或混用 legacy/new schema
会在评估报告中暴露为配置问题。

## 工具

### 1. record_cli - 交互式录制客户端

用于控制 episode 录制的命令行工具。

**启动录制服务**（Ubuntu 录制服务器）：
```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    use_sim:=false
```

**如需启用 Rerun 可视化**：
```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    record_visualizer:=rerun \
    use_sim:=false
```

**启动录制客户端**（同机或另一台设置了相同 `ROS_DOMAIN_ID` 的机器）：
```bash
ros2 run dataset_tools record_cli
```

#### Managed Teleop 逐 Episode 准入与录制完整性

录制服务显式配置 `runtime_set_mode_service` 且 `control_mode=teleop` 时，
`record_episode/get_info` 会公布 `teleop_admission` 配置，默认启动的 CLI 自动发现。
CLI 显式传入的 runtime endpoint 优先。仅为可主动 rearm 的 managed leader 配置此功能；
legacy teleop、phone/VR deadman 和 `model_inference` 不自动启用。

每次提交 Prompt 都依次执行：请求 `idle`、等待新发布的 `ACTIVE / idle / stop_latched=false` 状态、
调用 rearm 并等待最终 enable 结果、等待新发布的 `ACTIVE / stream / stop_latched=false` 状态，
最后才发送录制 goal。状态必须在该请求完成后收到，时间戳不能早于等待起点或晚于当前 ROS 时间。
每个 Prompt 最多尝试三次；失败回到 Prompt，不发送 goal。模式切换/rearm 超时后，
在该请求完成前，即使提交新 Prompt 也不会再发送切换请求；晚到的响应不会自动重试。

| 参数 | 使用方 | 默认值 |
|---|---|---|
| `runtime_set_mode_service` | recorder 公布、CLI 使用 | 空，关闭 managed admission |
| `teleop_rearm_service` | recorder 公布、CLI 使用 | `/robot_teleop_node/rearm` |
| `teleop_stop_service` | recorder 公布、CLI 使用 | 空；managed leader 由 launch 绑定 public stop endpoint |
| `runtime_status_topic` | recorder 公布、CLI 使用 | `/runtime_status` |
| `admission_timeout_sec` | recorder 公布、CLI 使用 | `5.0`，每次服务调用/状态等待上限 |
| `admission_attempts` | recorder 公布、CLI 使用 | `3`，限制在 1 到 3 |
| `shutdown_timeout_sec` | CLI | `15.0`，退出时等待自身 goal 或准入清理的总上限 |
| `require_action_stream` | recorder | `false`，连续动作流校验开关 |
| `action_stream_start_timeout_sec` | recorder | `2.0`，首条动作宽限期 |
| `action_stream_gap_timeout_sec` | recorder | `1.0`，逐动作 topic 最大接收间隔 |

managed leader 的 launch 将 `admission_timeout_sec` 默认设为有效 `rearm_timeout_s + 1.0`，
即默认 6 秒，为服务完成后的响应交付保留余量；显式覆盖也必须大于 rearm 期限。
跨主机部署需要同步 ROS 时间对应的系统时钟，否则源时间戳校验会拒绝准入。

`require_action_stream=true` 时，所有 `contract.actions[].publish_topic` 都必须持续发布，
空 actions 契约会拒绝启动。关节位置保持不变但消息持续发布是有效录制；零动作、任一动作 topic
中断、或中断后恢复但间隔超限，都会令整个当前 episode 失败并删除当前目录，保留此前 episode。
即使在首条动作宽限期内立即结束，也不能保存零动作 episode。稀疏推理模式应保持此开关关闭。
间隔按 recorder 回调接收时间检查；超过截止时间的磁盘或 executor 停顿也会保守地废弃 episode。

Enter 停止只结束当前 episode，遥操作会话保留。CLI 退出（Ctrl+C、`q`、EOF）在取消自己的 goal 后，
立即停止由本 CLI rearm 的 managed teleop 会话，不等待 bag 落盘：先确认 HOLD，再恢复到准入前观测到的
`ACTIVE/idle/unlatched`；不是本 CLI 启用的会话不受影响。rearm 请求一旦发出即视为持有
（服务请求无法撤回），显式拒绝会解除持有，超时未决则保持持有。
Ctrl+C 会保留 ROS context 和 executor，处理尚未返回的 goal acceptance，然后有界等待 cancel/result
及 writer 清理。HOLD 清理与录制收尾共用退出期限，但优先发起 HOLD，慢速落盘不会耗尽其请求机会。
录制结果超时只表示 bag 完成状态未确认，recorder 仍可能在后台收尾；该超时与 HOLD 确认分别报告。
准入期间退出则先等待远端 mode/rearm 请求完成：晚到的 rearm 成功会被停止，
显式失败则直接恢复 idle。
ROS service 无法撤销已在远端执行的请求；若清理超时，CLI 会明确报告 runtime 状态未知、需要操作员停止，
不能把这种情况视作已确认安全停止。

本机回归使用真实 DDS/action executor、mock runtime 服务和 SQLite bag writer，不连接机器人。
部署环境仍需验收：停锁后新 Prompt 的 idle/reset/rearm 流程、逐次 episode 准入、录制中断流整段丢弃、
Enter 保留有效录制、goal acceptance 前后 Ctrl+C 的清理，以及退出 CLI 后遥操作会话确认停止。板端 idle/reset 流程尚需单独实测。

episodic 录制目录现在按 dataset 组织：

```text
<bag_base_dir>/
└── <dataset_name>/
    ├── dataset.yaml
    └── episodes/
        ├── episode_000001/
        │   ├── metadata.yaml
        │   └── *.mcap
        └── episode_000002/
            ├── metadata.yaml
            └── *.mcap
```

- `bag_base_dir` 来自 `robot_config.recording.bag_base_dir`
- `dataset_name` 默认取 `recording.dataset_name`，未配置时回退到机器人名
- `dataset.yaml` 保存 dataset 级元信息；可选通过 `recording.default_task`、`recording.task_family` 预填任务语义
- episode 级 prompt 仍写入各自 bag 的 `metadata.yaml`

**使用方式**：
```
========================================================
Dataset Collection CLI
Enter prompt text to start recording. (Press Enter to reuse: 'get')
Type 'q' or 'quit' to exit.
========================================================
Prompt > get        # 输入任务描述开始录制
[INFO] 🔴 RECORDING STARTED.
Controls while recording:
  Enter       stop and review
  d + Enter   stop, discard, then return to Prompt
  r + Enter   stop, discard, then retry same prompt

# 按 Enter 后，当前 episode 会先完成落盘，再进入确认界面：
========================================
Episode stopped and finalized.
Dataset: <bag_base_dir>/<dataset_name>
Episode: <bag_base_dir>/<dataset_name>/episodes/episode_000001
Messages written: 1894
Prompt: get

[s] save / [d] discard / [r] discard and retry / [q] quit and keep
Choice [s] >       # 直接回车保存；d 删除；r 删除并用同一 prompt 重录；q 保存并退出
✅ KEPT episode_000001
📁 <bag_base_dir>/<dataset_name>/episodes/episode_000001
Prompt > q          # 退出
```

录制中的快捷键：

| 输入 | 行为 |
|---|---|
| Enter | 停止当前 episode 并进入确认界面 |
| `d` + Enter | 停止、删除当前 episode，然后回到 Prompt |
| `r` + Enter | 停止、删除当前 episode，然后用同一 prompt 立即重录 |

确认界面中的快捷键：

| 输入 | 行为 |
|---|---|
| Enter / `s` | 保存当前 episode |
| `d` | 删除当前 episode |
| `r` | 删除当前 episode，并用同一 prompt 重录 |
| `q` | 保存当前 episode 并退出 |

`record_cli` 会在启动时显示 dataset 根目录，并在每个 episode 完成后显示具体 episode 目录。录制结果位于：

```text
<bag_base_dir>/<dataset_name>/episodes/episode_XXXXXX/
```

当前交互假设一个 `episode_recorder` server 同时由一个操作者使用。

`record_cli` 默认按 `control_mode:=teleop` 工作，不触发推理侧 reset。录制模型推理过程时，将客户端控制模式设为 `model_inference`：

```bash
ros2 run dataset_tools record_cli --ros-args -p control_mode:=model_inference
```

此时每个 episode 开始前会重置推理/分发器状态。`robot.launch.py` 在 episodic 模式下会打印与当前 scheduler
分支匹配的完整 `record_cli` 命令；客户端不通过 ROS service 是否存在来猜测：

- **调度启用路径**：显式传入
  ROS 参数 `restart_session_service:=/action_dispatcher/restart_session`。record_cli 调用它执行 safe-stop + Close 旧
  session + Open 新 UUID；失败时不会回退到 direct policy reset，避免绕过 Close 屏障。
- **legacy/关闭路径**：保持 `restart_session_service` 为空。record_cli 优先调用
  `/action_dispatcher/reset` 清理动作队列，并由 legacy dispatcher best-effort 触发 pipeline `/reset`；无
  dispatcher 时回退到 direct policy reset。

Scheduler-enabled 录制命令：

```bash
ros2 run dataset_tools record_cli \
  --ros-args -p control_mode:=model_inference \
  -p restart_session_service:=/action_dispatcher/restart_session
```

可通过 `reset_before_episode`、`dispatcher_reset_service`、`policy_reset_service` 和 `reset_timeout_sec` ROS 参数覆写
legacy 行为；scheduled restart endpoint 统一使用 `restart_session_service` ROS 参数，默认空值。

录制完成后，推荐直接把整个 dataset 根目录转换成 LeRobot v3 数据集：

```bash
ros2 run dataset_tools bag_to_lerobot \
    --bags-dir ~/rosbag/episodes/so101_single_arm \
    --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
    --out /path/to/output_dataset
```

### 2. bag_to_lerobot - Bag 转 LeRobot 数据集

将 ROS 2 episodic dataset 根目录转换为 LeRobot v3 数据集格式。

#### RTP 编码视频录制

跨设备 RTP 模式不把 H.264 数据送入 DDS。cloud `recording_node` 在 RTP receiver
完成 access unit 重组后、解码前直接录制：

```text
episode_000001/
├── metadata.yaml
├── *.mcap                              # DDS action/state
├── observation.images.top.h264         # H.264 Annex-B
└── observation.images.top.h264.json    # NDJSON sidecar
```

sidecar 每行是一个 JSON 对象，包含 `frame_index`、`capture_timestamp_ns`、
`rtp_timestamp`、`keyframe`、`lost_packets`、`session_generation` 和 `dropped`。
`dropped: "timestamp_unmapped"` 的条目只有 metadata，没有对应 `.h264` payload。
转换器因此先过滤所有 `dropped != null` 条目，再将剩余条目按位置与解码帧配对；
非 dropped 条目数与解码帧数不同会拒绝转换。

RTP 视频仍要求同一 episode 中存在 DDS action/state rosbag。转换器自动检测
`.h264` 文件，跳过 rosbag 中同 observation key 的图像 topic，并将两种来源按
`capture_timestamp_ns`/ROS header stamp 对齐到统一 tick grid。输出 MP4 仍走标准
LeRobot 重编码路径。

启动 cloud 录制端：

```bash
ros2 launch inference_service cloud_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/unused/when/recording \
    deployment:=unused \
    recording:=true
```

edge 仍以 `record:=true record_mode:=episodic` 启动机器人，但 RTP 配置不会在 edge
重复启动 `episode_recorder`；episode action server 由 cloud `recording_node` 提供。

跨设备录制前必须使用 NTP/PTP 同步 edge 与 cloud 时钟，并让 action/state observation
使用 `stamp_src: header`。转换日志中的 `source=Annex-B` 和
`max_alignment_error=... ms` 可用于诊断：持续增大的误差通常表示时钟漂移；
`timestamp_unmapped` 表示 RTP timestamp anchor 缺失或过期；配对数量不一致表示裸流
或 sidecar 不完整。strict 模式会在录制阶段删除此类 episode，tolerant 模式则在
`meta/info.json` 的 `integrity.frame_gaps` 中保留 episode、observation 和 frame 定位信息。

**基本用法**：
```bash
ros2 run dataset_tools bag_to_lerobot \
    --bags-dir ~/rosbag/episodes/so101_single_arm \
    --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
    --out /path/to/output_dataset
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--bags-dir` | dataset 根目录或 episodes 目录，自动发现多个 episode bag | 必需 |
| `--robot-config` | robot_config.yaml 路径 | 必需 |
| `--out` | 输出数据集目录 | 必需 |
| `--repo-id` | 数据集 repo_id | `rosbag_v30` |
| `--no-videos` | 存储 PNG 图像而非视频 | `false` |
| `--timestamp` | 时间戳来源 (contract/bag/header) | `contract` |
| `--image-threads` | 图像写入线程数 | `4` |
| `--chunk-size` | 每个 chunk 的帧数 | `1000` |

**输出结构**：
```
output_dataset/
├── videos/
│   ├── observation.images.front/
│   │   └── chunk-000/file-000.mp4
│   ├── observation.images.top/
│   └── observation.images.wrist/
├── data/
│   └── chunk-000/file-000.parquet
└── meta/
    ├── info.json
    ├── tasks.parquet
    ├── stats.json
    └── episodes/
```

### 3. episode_recorder - 录制服务节点

由 launch 文件自动启动的录制服务，提供 `record_episode` Action Server。

通常不需要直接运行，由 `robot.launch.py` 根据 `record_mode:=episodic` 参数自动加载。
录制结果会写到 `<bag_base_dir>/<dataset_name>/episodes/episode_XXXXXX/`，并在 dataset 根目录生成 `dataset.yaml`。

### 4. policy_eval - rosbag policy replay evaluation

`policy_eval` 用于离线比较不同推理后端在同一批 rosbag 观测上的 action chunk 输出。MVP 只支持 ROS bag 输入，不执行动作，不启动 `action_dispatch`、ros2_control、控制器或硬件驱动。

推荐拓扑是先启动最小化推理节点，再运行 frame-gated replay client：

| 命令 | 所属包 | 职责 |
|------|--------|------|
| `ros2 launch inference_service eval_inference.launch.py ...` | `inference_service` | 启动一个最小 `pipeline_policy_node`，加载命名 deployment，并提供 pipeline-scoped `DispatchInfer` Action Server。它不读取 rosbag、不发布观测、不记录结果。 |
| `ros2 run dataset_tools policy_eval capture ...` | `dataset_tools` | 读取 rosbag，按 contract 逐帧发布 observation topics，调用 `DispatchInfer`，并把 action chunk/latency/diagnostics 写入 prediction JSON。 |

```bash
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:=src/robot_config/config/robots/so101_single_arm.yaml \
    model_path:=/path/to/policy_bundle \
    deployment:=cpu \
    pipeline_id:=policy

ros2 run dataset_tools policy_eval capture \
    --bag-dir ~/rosbag/episodes/so101_single_arm/episode_000001 \
    --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
    --policy-path /path/to/policy_bundle \
    --pipeline-id policy \
    --backend-name cpu \
    --out /tmp/cpu_predictions.json \
    --frame-limit 100
```

对另一个后端重复 capture 后比较：

```bash
ros2 run dataset_tools policy_eval compare \
    --reference /tmp/cpu_predictions.json \
    --candidate /tmp/rknn_predictions.json \
    --out /tmp/cpu_vs_rknn_metrics.json
```

`compare` 默认会在指标 JSON 旁生成 PNG 折线图，便于人工复核 backend 差异：

- `*_error_lines.png`：每帧 MAE、max error，以及每个 action 维度的误差曲线。
- `*_raw_action_overview.png`：所有 action 维度的 reference/candidate 原始 action 曲线总览。
- `*_action_dim_N_raw.png`：每个 action 维度单独一张 reference/candidate 原始 action 曲线。

可以用 `--plot-dir /tmp/policy_eval_plots` 指定目录，用 `--no-plots` 关闭绘图。原始 action 曲线默认绘制 action chunk 的第 0 个 step，可用 `--plot-action-step` 调整。

关键语义：

- replay client 会从 rosbag 读取 contract observation topics，逐帧 publish 原始 ROS 消息，然后用同一 timestamp 调用 `/inference/<pipeline_id>/dispatch`。
- `--policy-path` 推荐与 `eval_inference.launch.py model_path:=...` 使用同一个 bundle；该参数只读取 `config.json.input_features` 以过滤未被模型使用的 contract observations。
- 历史 replay 只接受 `--timestamp-policy header` 或 `contract`，且所有选中 observation 必须在 contract 中声明 `stamp_src: header`。ROS 消息重新发布后无法可靠重建 bag/receive timestamp。
- 多 deployment 比较采用串行运行方式；每次启动一个 deployment，并将 prediction JSON 作为比较输入。
- 默认 `policy_state_mode=continuous`，保留同一次 run 内的 policy runtime state；`--policy-state-mode per_frame_reset` 只用于显式诊断，会在每帧前调用 `/inference/<pipeline_id>/reset`。
- `per_frame_reset` 完成后，replay client 会把该帧发布消息的 header stamp 和请求时间重映射到当前 ROS 时间，避免历史 bag 时间戳被 reset cutoff 判定为 reset 前旧消息。prediction JSON 中仍保留原始 bag frame timestamp，并以 `replay_timestamp_mode=live_rebased` 标记该行为。
- replay publisher 使用 best-effort 相机 QoS 时，首次发布可能发生在 DDS endpoint 完全匹配之前。capture 会在 `required policy observations are not ready` 时重复发布同一帧并重试，默认最多等待 5 秒；可用 `--observation-ready-timeout-sec` 调整。其他 backend 或模型错误不会自动重试。
- 离线逐帧回放的 publisher 固定使用 `reliable + keep_last(depth=1)`，确保选定的大图像消息不会因 best-effort 丢包而随机截断评估，也不会在逐帧 gate 中积压旧图像；pipeline 的 best-effort subscription 与 reliable publisher 兼容。该覆盖仅用于 `policy_eval`，不修改生产相机或 contract QoS。
- `per_frame_reset` 为每个原始 bag frame 只生成一个 live timestamp；该帧的所有重发和推理请求都复用同一 timestamp。这样即使大图像 callback 晚一轮到达，也不会因为重试不断推进 goal timestamp 而被永久判定为 stale。
- prediction JSON 会记录 contract fingerprint、timestamp policy、frame stride、backend 名称、calibration 静态检查结果、每帧 action chunk、latency 和 replay stream diagnostics。
- prediction JSON 同时记录 `planned_frame_count`、`successful_frame_count` 和 `complete`。`compare` 默认拒绝任一不完整 run 或成功帧集合不同的结果，防止不同平台各自对随机收到的帧子集计算出不可比指标；`--allow-incompatible` 仅用于显式诊断子集。
- `compare` 同时输出全量 cosine similarity、逐帧 chunk cosine 的均值/最小值、chunk 第 0 步 cosine 的均值/最小值，以及逐 action 维度 cosine。零范数向量的 cosine 记为 `null`，并通过 `undefined_*_cosine_count` 单独计数。
- prediction action 是 postprocessor 后的物理动作，因此普通 `cosine_similarity` 属于物理动作空间。若 reference bundle 的 postprocessor 使用 `MEAN_STD` 且可读取 `action.mean/std`，`compare` 还会输出 `normalized_*_cosine_similarity`，用于与 RKNN/ONNX 图输出及历史 normalized-space 验证结果对齐。
- `--compare-labels` 只在 rosbag 中存在 contract action topics 时启用，用于额外比较录制 action label；backend 正确性比较仍以 reference backend 输出为主。

Non-goals：LeRobot parquet/video dataset replay、并行多 backend 节点、temporal ensemble/action dispatch 评估、sim-in-the-loop 评估和 VLA prompt 评估。

### 5. camera_alignment - 基于 ArUco 的相机对齐工具

用于在数据采集或复现前对齐摄像头视角。`--camera-source` 同时支持真机视频设备和仿真 ROS 2 image topic：

| 输入形态 | 路径 | 行为 |
|---|---|---|
| `/dev/videoN` / 整数 / 本地视频文件 | 真机路径（OpenCV） | 保持原有工作流，支持显式请求分辨率、帧率和采集格式 |
| `/camera/<name>/image_raw` | 仿真路径（ROS 2 topic） | 通过 `rclpy` 订阅图像，并接入 sim calibration 辅助能力 |

**真机用法**：
```bash
ros2 run dataset_tools camera_alignment \
    --camera-source /dev/video0 \
    --width 640 \
    --height 480 \
    --fps 60 \
    --format MJPG \
    --reference-path /tmp/camera_reference_multi.json \
    --reference-image-path /tmp/reference_img.png
```

**仿真用法**：
```bash
ros2 run dataset_tools camera_alignment \
    --camera-source /camera/top/image_raw \
    --markerless \
    --reference-image-path ref_img/reftop.png
```

工具支持：

- 保存当前 ArUco 角点作为参考基准
- 实时显示与参考画面的平均像素误差
- 进入“虚影对齐”界面辅助恢复视角
- 显式请求 OpenCV 采集分辨率、帧率和采集格式，并提示实际生效值

仿真模式下额外行为：

- 启动时尝试通过 `sim_models.aruco_spawner` 向 Gazebo 动态插入 ArUco A4 标定板，退出时自动清理。
- 当输入是 `/camera/<name>/image_raw` 时，会自动切到 `/camera_align/<name>/image_raw` 代理话题，便于与 `sim_camera_adjuster` 联动。
- 按 `p` 保存当前帧到 `capture_YYYYMMDD_HHMMSS.png`。
- 按 `s` 时，`camera_alignment` 会写入 `~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml` 的 stub 文件；真正的位姿值由 `sim_models.sim_camera_adjuster` 保存并在下次仿真启动时被 `robot_config` 读取。

兼容性边界：

- 真机路径的 `OpenCVFrameSource` 工作流保持兼容；ROS 2 依赖只会在仿真 topic 路径下触发。
- markerless 模式只跳过 ArUco 误差计算，不影响真机模式的原有 reference JSON / image 行为。

标定板打印件：真机模式需要 ArUco marker 作为对齐参考，仓库提供机械臂底座 ArUco
marker 打印件（`docs/calibration_targets/机械臂底座marker0and1.pdf`、
`docs/calibration_targets/机械臂底座marker2and3.pdf`），使用 `DICT_4X4_50` 字典族。

详细说明见：

- `docs/tools/camera_alignment.md`

### 6. camera_isp_calibrator - 基于参考图的相机色彩对齐工具

让一台 USB 摄像头（usb_cam 节点）的画面在曝光、白平衡、增益、对比度等
方面尽可能接近一张参考图片，并把结果保存为 override JSON，下次启动
`robot.launch.py` 时自动复用，**不修改 YAML SSOT**。

**运行模式**：
1. ROS bridge 模式：不传 `--camera_index` 时，沿用原有行为，
   订阅 usb_cam 节点（节点名形如 `/top_camera`）并通过 ROS 参数服务兜底写入；
2. OpenCV 直连模式：传入 `--camera_index` 时，直接从本机摄像头索引
   或 `/dev/video*` 读取画面，不需要启动 `robot.launch.py` 或 `usb_cam` 节点。

两种模式都需要准备一张参考图片或视频（视频会取首帧）。直连模式会优先通过
`v4l2-ctl` 写入设备 ISP 参数；建议仍传 `--camera top` / `--camera wrist` 等名称，
这样保存的 override 文件能继续被 `robot.launch.py` 自动复用。

**基本用法**：
```bash
# 方式 A：ROS bridge 模式，先启动机器人 / 摄像头
source .shrc_local
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop

# 另一个终端运行校准工具
source .shrc_local
ros2 run dataset_tools camera_isp_calibrator \
    --camera top \
    --reference /path/to/reference.png

# 方式 B：OpenCV 直连模式，不需要启动 robot.launch.py
source .shrc_local
ros2 run dataset_tools camera_isp_calibrator \
    --camera top \
    --camera_index /dev/video0 \
    --reference /path/to/reference.png

# 也可以使用整数索引；未传 --camera 时保存名会自动派生为 video0
ros2 run dataset_tools camera_isp_calibrator \
    --camera_index 0 \
    --reference /path/to/reference.png
```

**界面交互**（单窗口 cv2 GUI，傻瓜操作）：

- `a`：自动模式（Lab + Planckian 投影），4 次迭代收敛后自动停下
- `c`：**统一 K/C/Sat 搜索**（实验性，详见 §5.1）。有 ROI pair 时走 m 模式 cost，没有就走 AUTO ref-cluster cost；找不到改进会自动回退到 seed。
- 拖动滑条：手动微调 exposure / wb_kelvin / gain / brightness / contrast / saturation / sharpness（松手 0.4s 后才下发，避免抖动）
- `s`：保存到 `~/.ros/ibrobot/camera_isp_overrides/{camera}.json`
- `r`：恢复启动时快照的初始值
- `p`：保存当前 live 帧 PNG 到工作目录
- `?` / `h`：显示 keybinding 提示
- `q`：退出（有未保存改动会先警告一次，再按 `q` 才退出）

**算法**：

| 模式     | 触发    | 说明 |
|----------|---------|------|
| Auto     | 按 `a`  | sRGB→Lab 计算 P50 亮度匹配曝光；CIE xy 色度通过 McCamy 公式投影到 Planckian locus，按 delta-form 调节 white_balance kelvin。每帧迭代再读、最多 4 次。|
| 手动滑条 | 拖动    | 直接下发 V4L2 参数到 usb_cam 节点（已强制 `auto_white_balance=false` / `autoexposure=false`）。|

**保存生效**：保存后下次 `robot.launch.py` 启动时，`perception.py` 会自动
读取 override 并覆盖 YAML 默认值；删除 JSON 即可回退。

#### 6.1 统一 K/C/Sat 色彩搜索（实验性，独立模块）

模块 `dataset_tools/camera_isp/color_search.py` 实现了统一 K/C/Sat 搜索路径，
**与既有 4 阶段流水线（曝光/增益/亮度/锐度）并行存在，不修改任何曝光相关代码**。
旧 `solver` / `hw_pipeline` 全部保留作为初值估计器与失败回退。

公共接口（pure-numpy + scipy；无 ROS / cv2 依赖）：

```python
from dataset_tools.camera_isp.color_search import (
    KCS, SettleConfig, SearchConfig, ClusterConfig,
    kmeans_signature_lab,     # Lab 单边聚类签名
    nn_match_signatures,      # 匈牙利 ΔE2000 指派
    delta_e2000,              # CIEDE2000 (vectorised)
    quantile_distance_L,      # L* 分位数 L1
    cost_24card,              # 24 色卡 cost 工厂
    cost_ref_cluster,         # AUTO ref cost 工厂
    cost_manual_roi,          # m / ROI cost 工厂（带正则）
    frame_capture,            # settle + drop + trimmed-mean
    search_KCS,               # 主搜索 driver（直接 3D 网格 + 可选精修）
    OfflineTables,            # JSON 配置加载
)
```

设计原则（开放给后续迭代）：

- **三模式同构**：`search_KCS` 接收任意 `cost_fn`，driver 不感知模式。
- **失败安全**：未找到改进时回退到 seed 并把硬件值写回 seed。
- **可注入边界**：`HwWriter` / `FrameGrabber` 协议 + `sleeper` 钩子让单元测试无需真实相机。
- **离线表外置**：`camera_isp_offline_tables.json`（per device 可覆盖）承载 K/C/Sat 曲线、settle、search 参数；不再硬编码。

测试：`test/test_camera_isp_color_search.py`（16 个用例，覆盖 ΔE2000、聚类、匈牙利、settle、driver fallback、device caps 裁剪）。

cc24 色彩校准模式需要标准 24 色 ColorChecker 色卡作为参考。仓库提供打印件：
`docs/calibration_targets/标准24色色卡.pdf`，直接打印即可使用。

### 7. lerobot_action_gap_repair - 数据集 action 间隙修复工具

用于分析 LeRobot 数据集中 action 各维度的值分布与非活跃间隙，并可将短间隙用相邻活跃值填充（桥接）。

典型应用场景：遥操作录制时，控制信号可能出现短暂掉零（gripper 或关节值瞬间归零），这些间隙会影响训练质量。该工具先分析间隙分布，再将符合条件的短间隙填充为正常值。

**两种工作模式**：

1. **分析模式**（仅传 `--src-root`）：扫描数据集，输出各维度的值分布和间隙统计。
2. **修复模式**（附加 `--gap-threshold`）：在分析基础上，将短于阈值的非活跃间隙填充为相邻活跃值，输出到新数据集目录。

**基本用法**：

```bash
# 分析模式：查看各 action 维度的值分布和间隙
lerobot_action_gap_repair --src-root /path/to/dataset

# 修复模式：填充长度 ≤ 1 的非活跃间隙
lerobot_action_gap_repair \
    --src-root /path/to/dataset \
    --dst-root /path/to/dataset_bridged \
    --gap-threshold 1 \
    --process-indices 6 8 \
    --target-values 0.1 30.0
```

**参数说明**：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--src-root` | 是 | — | 源 LeRobot 数据集目录 |
| `--dst-root` | 修复模式必填 | — | 修复后数据集输出目录 |
| `--analyze-indices` | 否 | `6 7 8` | 分析模式检查的 action 维度索引 |
| `--process-indices` | 修复模式必填 | — | 需要修复的 action 维度索引 |
| `--target-values` | 修复模式必填 | — | 各 `process-indices` 对应的目标填充值 |
| `--gap-threshold` | 修复模式必填 | — | 间隙长度 ≤ 该值时才修复 |
| `--inactive-value` | 否 | `0.0` | 被视为"非活跃"的值 |
| `--repo-id` | 否 | 数据集目录名 | 用于 stats 重算的 repo_id |
| `--skip-recompute-stats` | 否 | `false` | 跳过 `meta/stats.json` 重算 |

**输出**：

- 分析模式：终端输出各维度的值计数和间隙计数
- 修复模式：在 `--dst-root` 生成完整的新数据集（先拷贝再修改），修复后自动重算 `meta/stats.json`

**特性**：

- 支持跨 parquet 文件边界的 episode 级间隙追踪
- 修复前先校验参数合法性，避免创建不完整的输出目录
- `--skip-recompute-stats` 可在无 lerobot 环境时使用

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│   src/robot_config/config/robots/so101_single_arm.yaml     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ contract (单一真理来源)                               │   │
│  │ - observations (front, top, wrist, state)           │   │
│  │ - actions (arm, gripper)                            │   │
│  │ - rate_hz: 20                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
   │  录制服务    │     │  数据转换    │     │    推理服务      │
   │ episode_    │     │ bag_to_     │     │ pipeline_policy │
   │ recorder    │     │ lerobot     │     │ node            │
   └─────────────┘     └─────────────┘     └─────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ROS 2 Bag          LeRobot Dataset      Model Inference
```

## 配置示例

`src/robot_config/config/robots/so101_single_arm.yaml` 中的 contract 配置：

```yaml
robot:
  name: so101_single_arm
  
  contract:
    rate_hz: 20
    max_duration_s: 90.0
    
    observations:
      - key: observation.images.front
        topic: /camera/front/image_raw
        type: sensor_msgs/msg/Image
        image:
          resize: [480, 640]
          
      - key: observation.images.top
        topic: /camera/top/image_raw
        type: sensor_msgs/msg/Image
        image:
          resize: [480, 640]
          
      - key: observation.images.wrist
        topic: /camera/wrist/image_raw
        type: sensor_msgs/msg/Image
        image:
          resize: [480, 640]
          
      - key: observation.state
        topic: /joint_states
        type: sensor_msgs/msg/JointState
        selector:
          names: [position.1, position.2, position.3, position.4, position.5, position.6]
    
    actions:
      # Arm joints (1-5)
      - key: action
        selector:
          names: [action.0, action.1, action.2, action.3, action.4]
        publish:
          topic: /arm_position_controller/commands
          type: std_msgs/msg/Float64MultiArray
          
      # Gripper joint (6) - same key for consolidation
      - key: action
        selector:
          names: [action.5]
        publish:
          topic: /gripper_position_controller/commands
          type: std_msgs/msg/Float64MultiArray
```

## 注意事项

1. **Action 合并**: 多个 action spec 使用相同的 `key: action` 会被自动合并为一个 6-DOF action
2. **观测过滤**: 推理服务会根据模型的 `config.json` 自动过滤需要的观测
3. **录制模式**: 
   - `record_mode:=continuous` - 持续录制到一个文件
   - `record_mode:=episodic` - 分段录制，需要 `record_cli` 控制

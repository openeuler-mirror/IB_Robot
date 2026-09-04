# 声源方向转向完整真机验证指南

本文用于在 `310p-wm` 上验证当前工作区未暂存代码实现的完整声源方向转向功能。

验证目标是打通以下真实链路：

```text
ReSpeaker 麦克风
  -> VoiceASRNode
  -> /voice_command

ReSpeaker 麦克风
  -> speech_direction_node
  -> /voice/speech_direction

/voice_command + /voice/speech_direction
  -> sound_orientation_node
  -> /embodied/get_skill_gateway_status
  -> /embodied/execute_skill
  -> skill_executor_node
  -> safety_guard
  -> nav_turn
  -> navigation_command_server
  -> Nav2
  -> LeKiwi 底盘
```

本文只描述验证方案和执行步骤，**不代表已经执行验证**。

## 1. 验证范围

### 1.1 本次必须验证

- 本地工作区代码同步到 `310p-wm:/IB_Robot-lwh`。
- 远端受影响 ROS 包使用同步后的源码重新构建。
- `lekiwi_nav_grasp` 的 `navigation` stage 能够启动真实 LeKiwi 硬件。
- `joint_state_broadcaster` 和 `base_velocity_controller` 正常激活。
- MID-360、FAST-LIO、AMCL、Nav2 和 `navigation_command_server` 正常启动。
- 真实 Voice ASR 能从 ReSpeaker 采集并发布最终文本。
- 真实 speech direction 能从 ReSpeaker 采集并发布 `SpeechDirection`。
- `sound_orientation_node` 能识别精确触发词 `转向我`。
- 真实 Gateway 能返回新鲜状态和 `nav_turn.ready=true`。
- 正负声源方向分别映射为真实底盘左转和右转。
- 角度转换满足 `azimuth_rad -> degree` 契约。
- 一次触发最多派发一个 `nav_turn`。
- 普通文本、deadband、过期方向和冷却期间不会误转向。
- 真实底盘动作完成后能够获得确定的 terminal result。

### 1.2 本次不作为通过条件

- 视觉抓取、语义建图和 MoveIt 机械臂动作。
- Hermes/LLM 对自然语言的规划能力。
- TTS 语音播报。
- ASR 和 speech direction 的模型精度评测基准。
- 取消后真实底盘的未知终态恢复演练。

这些功能与本次声源转向主链路不同。本文使用 `navigation` stage，保留真实底盘和导航链路，同时不启动抓取、语义感知和 MoveIt，避免无关模型占用 310P 资源并影响结果。

## 2. 安全规则

本次测试包含真实底盘运动。执行前必须由现场操作员确认：

- LeKiwi 位于空旷、平整区域，底盘周围没有人员、线缆和障碍物。
- 底盘前后左右均留有足够的旋转空间。
- 物理急停、断电或其他硬件停止手段可立即使用。
- 操作员在机器人旁边，不通过 SSH 远程观察代替现场看护。
- SO-101 机械臂处于安全姿态，不会因底盘旋转碰撞周边物体。
- `/dev/ttyACM0` 没有被其他程序占用。
- ReSpeaker 没有被其他录音程序占用。
- 第一轮真实运动只允许使用约 `15°` 的小角度。
- 测试过程中不直接发布 `/cmd_vel`，不直接调用 `/navigation/execute`，不直接操作 controller。
- 不使用 `ros2 action send_goal` 绕过 Skill Gateway 发送导航动作。
- 如果出现运动方向错误、动作状态未知、底盘反馈异常或串口错误，立即使用物理急停并停止后续测试。

`authorize_motion:=true` 只能在无运动验证全部通过、现场安全检查完成后使用。

## 3. 测试环境约定

### 3.1 设备和目录

```text
SSH alias:       310p-wm
Remote workspace: /IB_Robot-lwh
Robot YAML:      lekiwi_nav_grasp.yaml
Robot stage:     navigation
ROS Domain:      73
```

`310p-wm` 当前属于 openEuler Embedded 源码工作区运行方式，使用远端 `/IB_Robot-lwh/.shrc_local`。不要将本流程与 OpenHarmony `/data/roboframe` 发布包流程混用。

所有远端 ROS/Python 命令都必须在同一个 shell 中加载环境：

```sh
cd /IB_Robot-lwh
unset CONDA_DEFAULT_ENV CONDA_PREFIX
. .shrc_local
export ROS_DOMAIN_ID=73
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
```

如果远端实际使用的 DDS 实现不是 Fast DDS，应以远端现有部署为准；所有参与进程必须使用相同的 DDS 配置和 `ROS_DOMAIN_ID`。

### 3.2 日志和临时文件

统一写入：

```text
/data/local/tmp/ibrobot-sound-orientation/full/
```

建议目录：

```text
/data/local/tmp/ibrobot-sound-orientation/full/
├── before-sync/
├── launch-no-motion.log
├── launch-motion.log
├── model-check.log
├── asr.log
├── speech-direction.log
├── sound-orientation.log
├── gateway-status.txt
├── odom-before-left.txt
├── odom-after-left.txt
├── odom-before-right.txt
├── odom-after-right.txt
└── result.md
```

## 4. 同步本地源码

这一步必须先于远端构建和验证。

### 4.1 检查远端工作区

在本地执行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 310p-wm \
  'uname -a; test -d /IB_Robot-lwh && echo WORKSPACE_OK'
```

检查远端源码是否存在未保存修改：

```bash
ssh 310p-wm \
  'git -c safe.directory=/IB_Robot-lwh -C /IB_Robot-lwh status --short'
```

如果远端有未提交或未跟踪内容，不能直接覆盖。先备份：

```bash
ssh 310p-wm '
  mkdir -p /data/local/tmp/ibrobot-sound-orientation/full/before-sync &&
  git -c safe.directory=/IB_Robot-lwh -C /IB_Robot-lwh status --short \
    > /data/local/tmp/ibrobot-sound-orientation/full/before-sync/git-status.txt &&
  git -c safe.directory=/IB_Robot-lwh -C /IB_Robot-lwh diff \
    > /data/local/tmp/ibrobot-sound-orientation/full/before-sync/worktree.patch &&
  git -c safe.directory=/IB_Robot-lwh -C /IB_Robot-lwh diff --cached \
    > /data/local/tmp/ibrobot-sound-orientation/full/before-sync/index.patch
'
```

### 4.2 同步前预览

本地工作区包含多组未提交修改，不能使用无排除项的全量同步。预览时排除构建产物、模型和 LeRobot 子模块：

```bash
rsync -anvi \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='install/' \
  --exclude='log/' \
  --exclude='venv/' \
  --exclude='models/' \
  --exclude='outputs/' \
  --exclude='libs/lerobot/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  ./ 310p-wm:/IB_Robot-lwh/
```

重点确认不会覆盖远端专用的：

- 地图文件；
- 标定文件；
- 硬件实例配置；
- 设备启动脚本；
- 远端未纳入本地 Git 的配置。

### 4.3 正式同步

确认 dry-run 无误后执行：

```bash
rsync -avi \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='install/' \
  --exclude='log/' \
  --exclude='venv/' \
  --exclude='models/' \
  --exclude='outputs/' \
  --exclude='libs/lerobot/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  ./ 310p-wm:/IB_Robot-lwh/
```

本次不要添加 `--delete`。远端的模型目录不通过本次同步覆盖，保留设备上已经补充好的模型。

同步后确认声源转向源码存在：

```bash
ssh 310p-wm \
  'test -f /IB_Robot-lwh/src/embodied_agent/embodied_agent/sound_orientation_node.py &&
   test -f /IB_Robot-lwh/src/embodied_agent/embodied_agent/sound_orientation_policy.py &&
   echo SOUND_ORIENTATION_SOURCE_OK'
```

## 5. 构建远端源码

先停止旧 ROS 进程。不要在旧进程仍运行时替换消息或 Python 包：

```bash
ssh 310p-wm 'ps | grep -E "ros2_control|nav2|fast_lio|livox|skill_executor|sound_orientation" | grep -v grep || true'
```

如果有旧测试进程，先使用其启动终端的正常退出方式停止；不要直接对不属于本次测试的系统进程执行批量 kill。

使用项目构建脚本构建受影响包：

```bash
ssh 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  ./scripts/build.sh -- \
    --packages-select \
    ibrobot_msgs \
    embodied_common \
    skill_catalog \
    safety_guard \
    skill_library \
    robot_navigation \
    robot_config \
    embodied_agent \
    embodied_bringup \
    voice_asr_service
'
```

如果项目构建脚本在远端要求全量构建，改为：

```bash
ssh 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  ./scripts/build.sh
'
```

构建失败时停止，不得继续使用旧的或不完整的 `install` overlay。

构建后确认运行时来自 `/IB_Robot-lwh`：

```bash
ssh 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  python3 -c "import embodied_agent, robot_config; print(embodied_agent.__file__); print(robot_config.__file__)" &&
  ros2 pkg executables embodied_agent &&
  ros2 interface show ibrobot_msgs/msg/SpeechDirection &&
  ros2 interface show ibrobot_msgs/action/SkillCommand
'
```

## 6. 测试配置 overlay

不要修改生产文件：

```text
/IB_Robot-lwh/src/robot_config/config/robots/lekiwi_nav_grasp.yaml
```

因为 `base_config` 只允许引用同目录 sibling YAML，测试 overlay 放在：

```text
/IB_Robot-lwh/src/robot_config/config/robots/lekiwi_nav_grasp_sound_full.yaml
```

测试 overlay 的作用是：

- 继承 `lekiwi_nav_grasp` 的真实硬件、导航和麦克风配置；
- 使用 `navigation` stage，关闭抓取、语义感知和机械臂动作链路；
- 打开真实 Voice ASR；
- 打开真实 speech direction；
- 打开 `sound_orientation`；
- 限制首轮真实转向角度；
- 首轮关闭与本功能无关的 TTS，避免音频播放干扰采集；
- 不修改任何生产源码和生产 YAML。

建议内容如下：

```yaml
robot:
  base_config: lekiwi_nav_grasp
  # navigation stage 使用仓库已有的纯导航 catalog profile。临时测试
  # identity 必须与 profile.robot_name 一致，否则 Gateway fail-fast。
  name: lekiwi_lidar

  voice_asr:
    enabled: true
    auto_download_model: false
    active_mode: continuous
    language: zh
    model_path: models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
    tokens_path: ""
    provider: cpu
    model_type: streaming
    publish_partial: true
    output_topic: /voice_command
    sample_rate: 16000
    chunk_size: 512
    device_index: -1
    device_name: ReSpeaker
    exit_on_init_failure: true

  speech_direction:
    enabled: true
    profile: ascend_310p
    microphone: respeaker
    parameters:
      input_source: device
      mount_yaw_deg: 90.0

  voice_tts:
    enabled: false

  nav_stages:
    navigation:
      # 本功能不依赖相机。排除 front camera，同时清空其标定引用，
      # 避免无关 RealSense 故障影响音频/导航验收。
      peripheral_names: [respeaker, mid360]
      sensor_calibration:
        artifacts:
          base_to_front_camera: ""
      embodied:
        enabled: true
        skill_catalog_source_mode: development
        skill_catalog_source_root: src/skill_catalog
        skill_catalog_profile: lekiwi_lidar
        idle_behaviors:
          sound_orientation:
            enabled: true
            trigger_phrases: ["转向我"]
            direction_topic: /voice/speech_direction
            command_topic: /voice_command
            skill_name: nav_turn
            direction_frame: base_link
            deadband_deg: 15.0
            max_direction_age_sec: 1.3
            direction_wait_sec: 0.5
            cooldown_sec: 2.0
            max_turn_deg: 45.0
            turn_timeout_sec: 30.0
            action_acceptance_timeout_sec: 2.0
            status_retry_sec: 0.5
            reset_status_max_age_sec: 2.0
```

说明：

- `max_turn_deg: 45.0` 只用于测试 overlay，限制异常 DOA 或配置错误造成的大角度转向。
- `turn_timeout_sec: 30.0` 给真实 310P/Nav2 足够的动作收敛时间，避免此前 `10s` watchdog 过早取消小角度动作。
- `voice_asr.device_name: ReSpeaker` 依赖远端 ALSA/Python 音频设备枚举。如果该名称在 ASR 日志中无法匹配，不能直接猜测 index，应根据 ASR 启动日志中的设备列表选择正确设备。
- `name: lekiwi_lidar` 和 `skill_catalog_profile: lekiwi_lidar` 只属于测试 overlay，用于让纯导航 catalog 与 navigation stage 的 `base_navigation` 上下文一致。
- `peripheral_names: [respeaker, mid360]` 关闭与本功能无关的 RealSense 启动，但保留定位和音频所需硬件。
- `voice_tts` 与本功能无关，首轮关闭。TTS 模型可以在核心链路通过后单独验证。
- 不要在本次验证中切回 `hybrid` 全量 catalog。那会引入 MoveIt、抓取、语义感知和额外模型，不属于声源转向的完整闭环。

该文件可以先在本地通过 `apply_patch` 创建，再随源码同步；也可以只在远端测试目录创建。推荐纳入远端测试目录，不提交到生产配置目录。

## 7. 模型和硬件预检

### 7.1 模型路径和 manifest

在远端执行模型检查，不下载、不修改模型：

```bash
ssh 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  python3 scripts/verify_speech_direction_assets.py \
    --manifest-dir /IB_Robot-lwh/models/voice_asr
' | tee /tmp/ibrobot-sound-orientation-model-check.log
```

至少确认：

- Voice ASR streaming bundle 存在 `tokens.txt`、encoder、decoder、joiner 文件；
- speech direction 的 Ascend Silero VAD artifact 存在；
- speech direction 的 FullSubNet FB/SB OM 和 cumulative manifest 存在且校验通过；
- manifest 中的 deployment 名称为 `ascend_310p_silero` 和 `ascend_310p_fullsubnet`；
- 模型 bundle 路径与测试 overlay 相同。

检查 ASR bundle 结构：

```bash
ssh 310p-wm \
  'ls -l /IB_Robot-lwh/models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 &&
   test -f /IB_Robot-lwh/models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/tokens.txt &&
   echo ASR_ASSETS_OK'
```

本次 `navigation` stage 不启动 grasp/semantic/TTS model services，因此这些模型是否存在不影响本功能结论。完整版本在本文中指真实 ASR、真实 speech direction、真实 Gateway、真实导航和真实底盘，不是启动机器人配置中的所有模型。

### 7.2 真实设备

```bash
ssh 310p-wm '
  ls -l /dev/ttyACM0 /dev/snd/pcmC0D0c &&
  arecord -l
'
```

应看到：

```text
ReSpeaker 4 Mic Array
```

执行 3 秒录音 smoke test：

```bash
ssh 310p-wm '
  arecord -D hw:0,0 -f S16_LE -r 16000 -c 6 -d 3 \
    /data/local/tmp/ibrobot-sound-orientation/full/mic-smoke.wav
'
```

通过条件：

- `arecord` 成功打开设备；
- 16 kHz、6 通道录音成功；
- 文件大小合理；
- 录音结束后设备可以再次打开。

### 7.3 先确认没有旧进程

```bash
ssh 310p-wm \
  'ps | grep -E "ros2_control|nav2|fast_lio|livox|voice_asr|speech_direction|skill_executor|sound_orientation" | grep -v grep || true'
```

同一个 ROS Domain 中不允许存在旧的 ASR、speech direction、Gateway 或导航 Action Server。

## 8. 第一轮：完整音频链路但禁止运动

这一轮使用真实模型和真实麦克风，但设置：

```text
authorize_motion=false
```

预期可以验证 ASR、DOA、触发路由和 Gateway 拒绝行为，但底盘不能运动。

### 8.1 启动命令

```bash
ssh -t 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  export ROS_DOMAIN_ID=73 &&
  export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 &&
  ros2 launch embodied_bringup embodied_pipeline.launch.py \
    config_path:=/IB_Robot-lwh/src/robot_config/config/robots/lekiwi_nav_grasp_sound_full.yaml \
    nav_stage:=navigation \
    control_mode:=base_navigation \
    use_sim:=false \
    with_embodied:=true \
    with_perception:=false \
    with_moveit:=false \
    moveit_display:=false \
    authorize_motion:=false
'
```

`with_perception:=false` 表示关闭 Embodied Perception 请求运行时，不等于关闭 Voice ASR 和 speech direction。`navigation` stage 已将通用 perception model services、grasp execution 和 semantic mapping 关闭；真实 ASR 和 speech direction 由测试 overlay 显式打开。

### 8.2 启动日志门禁

必须看到：

```text
LeKiwiSystemHardware: Activated
Controllers are active
Voice ASR model loaded
VoiceASRNode initialized
speech_direction_node 已启动: input_source=device, ... degraded=False
safety_guard ready
skill_executor ready
Navigation lifecycle startup completed
sound orientation ready: triggers=['转向我'], skill=nav_turn
```

出现以下任一情况，停止本轮并修复，不进入真实运动：

- ASR 模型加载失败；
- speech direction 进入 degraded；
- `arecord` 打开失败或持续 EOF；
- 两个音频节点抢占同一设备；
- `skill_executor_node` 启动失败；
- `control_plane_ready` 不为 `true`；
- `nav_turn` capability 不存在；
- 控制器没有 active；
- FAST-LIO、定位或 Nav2 没有 ready；
- `/dev/ttyACM0` 出现读写错误。

### 8.3 Gateway 状态检查

从第二个终端执行：

```bash
ssh 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  export ROS_DOMAIN_ID=73 &&
  export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 &&
  ros2 service call /embodied/get_skill_gateway_status \
    ibrobot_msgs/srv/GetSkillGatewayStatus \
    "{schema_version: 1}" \
    > /data/local/tmp/ibrobot-sound-orientation/full/gateway-status-no-motion.txt
  cat /data/local/tmp/ibrobot-sound-orientation/full/gateway-status-no-motion.txt
'
```

本轮预期：

```text
control_plane_ready: true
control_plane_state: READY
motion_authorized: false
active_control_mode: base_navigation
busy: false
nav_turn.ready: false
reason: MOTION_NOT_AUTHORIZED
```

如果 `nav_turn` 不是 `MOTION_NOT_AUTHORIZED` 而是 catalog、registry 或 control mode 错误，应先修复配置，不得进入授权运动测试。

### 8.4 验证真实 ASR 和真实 DOA

让操作员站在麦克风阵列正前方、左前方和右前方分别说短句。每个位置停留足够时间，避免移动声源影响段级估计。

记录：

- `/voice_status` 是否经历 `LISTENING -> RECOGNIZING -> LISTENING`；
- `/voice_partial` 是否出现中间识别结果；
- `/voice_command` 是否只发布最终结果；
- `/voice/speech_direction` 是否发布 `frame_id=base_link`、有限 `azimuth_rad` 和递增 `seq_id`；
- `/diagnostics` 中 `speech_direction` 是否为非 degraded。

观察命令只读，不得用于发送运动：

```bash
ssh 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  export ROS_DOMAIN_ID=73 &&
  export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 &&
  timeout 30 ros2 topic echo /voice_command std_msgs/msg/String &&
  timeout 30 ros2 topic echo /voice/speech_direction ibrobot_msgs/msg/SpeechDirection
'
```

对于左侧声源，期望 `azimuth_rad > 0`；对于右侧声源，期望 `azimuth_rad < 0`。实际误差以设备安装方向和 DOA 算法回归基线为准。先在未授权阶段寻找能稳定输出约 `20°~30°` 的左右站位并在地面做标记；授权后复用同一站位。不要在 deadband 边界附近测试，也不要让输出超过测试 overlay 的 `45°` 上限。

### 8.5 验证未授权拒绝

在真实麦克风前说：

```text
转向我
```

并分别从左侧和右侧说话。

预期：

- ASR 发布最终文本 `转向我`；
- speech direction 发布新鲜方向；
- `sound_orientation_node` 进行 Gateway 预检查；
- Gateway 因 `motion_authorized=false` 拒绝运动；
- `/navigation/execute` 不产生对应 Goal；
- 底盘不移动；
- sound orientation 不进入 `FAULT_UNKNOWN`。

普通文本，例如：

```text
今天天气不错
```

不得产生 `nav_turn` 请求。

## 9. 第二轮：真实 ASR/DOA 到真实底盘

只有第 8 节所有门禁通过后，才执行本轮。

### 9.1 重启并打开授权

先正常停止上一轮 launch，确认其子进程全部退出，再重新启动：

```bash
ssh -t 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  export ROS_DOMAIN_ID=73 &&
  export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 &&
  ros2 launch embodied_bringup embodied_pipeline.launch.py \
    config_path:=/IB_Robot-lwh/src/robot_config/config/robots/lekiwi_nav_grasp_sound_full.yaml \
    nav_stage:=navigation \
    control_mode:=base_navigation \
    use_sim:=false \
    with_embodied:=true \
    with_perception:=false \
    with_moveit:=false \
    moveit_display:=false \
    authorize_motion:=true
'
```

重新检查 Gateway，必须满足：

```text
control_plane_ready: true
control_plane_state: READY
motion_authorized: true
active_control_mode: base_navigation
busy: false
nav_turn.ready: true
```

如果任一条件不满足，停止，不要说触发词。

### 9.2 左转测试

1. 将机器人放在初始朝向可清楚识别的位置。
2. 记录执行前 `/odometry/filtered` 的 yaw 和位置。
3. 操作员站到未授权阶段标记的左侧位置，使 DOA 稳定输出约 `20°~30°`。
4. 说一次：

   ```text
   转向我
   ```

5. 等待动作完成和唯一 terminal result。
6. 记录执行后 odometry yaw、Gateway status 和日志。

预期：

```text
azimuth_rad > 0
-> direction=left
-> degree=abs(azimuth_rad) * 180 / pi
-> degree <= 45
-> nav_turn
-> ExecuteNavigation.TURN_LEFT
-> LeKiwi 左转
-> SkillCommand 获得确定 terminal result
```

初次验收建议目标角度约 `15°~30°`。真实 DOA 输出如果超过测试 overlay 的 `45°` 限制，应被拒绝，不得执行大角度运动。

### 9.3 右转测试

左转得到确定 terminal result 且 cooldown 完成后：

1. 记录当前 odometry yaw。
2. 操作员移动到未授权阶段标记的右侧位置，使 DOA 稳定输出约 `-20°~-30°`。
3. 再说一次：

   ```text
   转向我
   ```

预期：

```text
azimuth_rad < 0
-> direction=right
-> degree=abs(azimuth_rad) * 180 / pi
-> nav_turn
-> ExecuteNavigation.TURN_RIGHT
-> LeKiwi 右转
-> SkillCommand 获得确定 terminal result
```

### 9.4 运动期间观察

只读观察以下接口：

```bash
ssh 310p-wm '
  cd /IB_Robot-lwh &&
  unset CONDA_DEFAULT_ENV CONDA_PREFIX &&
  . .shrc_local &&
  export ROS_DOMAIN_ID=73 &&
  export FASTDDS_BUILTIN_TRANSPORTS=UDPv4 &&
  ros2 action info /embodied/execute_skill &&
  ros2 action info /navigation/execute &&
  ros2 control list_controllers
'
```

运动期间应看到：

- `/embodied/execute_skill` 的 server 为 `skill_executor_node`；
- `/navigation/execute` 的 server 为 `navigation_command_server`；
- `base_velocity_controller` 保持 active；
- Gateway `busy=true`；
- sound orientation 不再接受第二个方向事件；
- 动作结束后 Gateway 回到 `busy=false`。

不要使用 `/cmd_vel`、`/cmd_vel_safe` 或 `/navigation/execute` 手工补发、停止或重试动作。

## 10. 行为回归矩阵

完成一次左转和右转后，再验证以下不运动场景。每个场景只执行一次，不自动重试。

| 场景 | 操作 | 预期 |
|---|---|---|
| 普通文本 | 说“今天天气不错” | 不产生 `nav_turn` |
| 精确触发 | 说“转向我”且声源在左侧 | 一个 left Goal |
| 精确触发 | 说“转向我”且声源在右侧 | 一个 right Goal |
| 多意图文本 | 说“转向我然后拿起物体” | 固定节点不拆分、不单独转向 |
| 正面声源 | 正前方说“转向我” | deadband 内不运动 |
| 转向期间重复说 | 动作未完成前重复触发 | 不产生第二个 Goal |
| cooldown 重复说 | 动作完成后 cooldown 内触发 | 不产生 Goal |
| 过大方向 | 让 DOA 输出超过 `45°` | 被测试 overlay 拒绝 |
| 无新鲜方向 | 无法获得有效 DOA 时说触发词 | 等待超时，不运动 |
| 未授权 | `authorize_motion=false` 时触发 | Gateway 拒绝，不运动 |
| Gateway busy | 已有合法 Gateway motion task 时触发 | 丢弃，不排队、不重试 |
| 重复事件 | 同一 `seq_id/stamp` 再消费 | 不重复运动 |

`Gateway busy` 场景只能在已有合法、高层 Gateway 任务已经运行时验证。不得直接调用底层导航 Action 制造 busy 状态。

## 11. 完整结果判定

### 11.1 通过

同时满足：

- 真实 ASR 能稳定发布最终 `转向我`。
- 真实 speech direction 能发布非 degraded、带新鲜时间戳的 `SpeechDirection`。
- 左侧声源得到正角度，右侧声源得到负角度。
- 未授权状态不运动。
- 左触发只派发一个 `nav_turn(left)`。
- 右触发只派发一个 `nav_turn(right)`。
- `degree` 与 DOA 弧度转换一致。
- 真实 Gateway、Safety Guard、Skill Executor 和 navigation command server 均参与执行。
- 真实 LeKiwi 底盘按照方向完成小角度转向。
- 每次动作都有唯一、确定的 terminal result。
- 普通文本、deadband、cooldown、过期方向和重复事件不会误触发。
- 测试结束时底盘速度为零，Gateway `busy=false`。

### 11.2 不通过

以下任一情况均不通过：

- 使用 Mock 文本或 Mock `SpeechDirection` 才能完成所谓“完整测试”。
- ASR 或 DOA 节点启动后处于 degraded。
- 真实麦克风并发打开失败。
- `sound_orientation_node` 连接不到 Gateway。
- Gateway registry 或 capability identity 不一致。
- `nav_turn` 被错误映射到机械臂动作。
- 真实底盘转向方向与声源方向相反。
- 触发一次产生多个 Goal。
- Action acceptance 或 terminal result 未知。
- 底盘停止状态不能确认。
- 出现 `SyncRead PacketTx FAILED` 等硬件通信错误且无法排除。
- 为了继续测试而自动重试、复用旧 task ID 或跳过 Gateway。

## 12. 故障处理

### 12.1 ASR 或 DOA 设备占用

如果单独节点都正常，但同时运行失败：

1. 立即保持 `authorize_motion=false`。
2. 停止音频节点和机器人 launch。
3. 确认没有残留 `arecord`、ASR 或 speech direction 进程。
4. 保存启动日志和设备列表。
5. 不通过修改 `sound_orientation_node` 绕过设备冲突。

该情况说明当前双采集链路还未满足生产开启条件。

### 12.2 Action 超时或终态未知

如果日志出现：

```text
SKILL_ACTION_RESULT_TIMEOUT
SKILL_CANCEL_TIMEOUT
FAULT_UNKNOWN
```

必须：

1. 立即停止后续语音触发。
2. 现场确认底盘物理上已停止。
3. 通过只读 Gateway 状态确认 `busy=false`。
4. 保存 `sound_orientation`、`skill_executor`、`navigation_command_server` 和底盘硬件日志。
5. 只有物理停止和 Gateway 状态都确认后，才考虑调用：

   ```text
   /sound_orientation_node/reset_fault
   ```

6. 不得因为“看起来已经停了”就开始下一次运动。

### 12.3 串口或底盘反馈错误

出现以下日志时，本轮真实运动立即停止：

```text
SyncRead PacketTx FAILED
Failed to read hardware state
controller communication error
```

先排查 USB、串口、固件和硬件占用，再决定是否重新测试。不能把硬件反馈错误归因于 ASR 或声源方向功能并继续运动。

### 12.4 Nav2 或定位未 ready

如果 Nav2、AMCL、FAST-LIO 或 costmap 未 ready：

- 可以继续保留 `authorize_motion=false` 做 ASR/DOA 和 Gateway 拒绝验证；
- 不得打开 `authorize_motion=true`；
- 不得用 `/navigation/execute` 直接绕过 readiness。

## 13. 结果记录模板

```markdown
# 声源方向转向真机验证记录

- 日期：
- 操作员：
- 设备：310p-wm
- 工作区：/IB_Robot-lwh
- Git commit：
- ROS_DOMAIN_ID：73
- 测试 overlay：lekiwi_nav_grasp_sound_full.yaml
- authorize_motion=false 预检：PASS / FAIL
- authorize_motion=true 左转：PASS / FAIL / NOT_RUN
- authorize_motion=true 右转：PASS / FAIL / NOT_RUN
- ASR 与 DOA 并发采集：PASS / FAIL
- Gateway control_plane_ready：
- Gateway registry_epoch：
- Gateway registry_generation：
- Gateway registry_digest：
- 左侧 azimuth_rad：
- 左侧 degree：
- 左侧 terminal result：
- 右侧 azimuth_rad：
- 右侧 degree：
- 右侧 terminal result：
- 是否出现硬件通信错误：
- 是否出现终态未知：
- 测试结束时 busy：
- 测试结束时底盘速度：
- 日志目录：
- 结论：
```

## 14. 清理

确认底盘速度为零、Gateway `busy=false`、没有 `FAULT_UNKNOWN` 未处理状态后，正常停止 launch。

检查残留进程：

```bash
ssh 310p-wm \
  'ps | grep -E "ros2_control|nav2|fast_lio|livox|voice_asr|speech_direction|skill_executor|sound_orientation" | grep -v grep || true'
```

测试完成后可以删除：

```text
/IB_Robot-lwh/src/robot_config/config/robots/lekiwi_nav_grasp_sound_full.yaml
/data/local/tmp/ibrobot-sound-orientation/full/
```

不要删除：

- `/IB_Robot-lwh/models/`；
- 远端原有地图和标定文件；
- 远端原有源码修改；
- 远端正式 install 或发布目录。

## 15. 建议执行顺序

```text
1. 备份 310p-wm 远端源码状态
2. 将本地仓库代码同步到 /IB_Robot-lwh
3. 重新构建受影响包
4. 创建临时 full YAML overlay
5. 校验 ASR、speech direction 和其他必要模型
6. 检查 /dev/ttyACM0 和 ReSpeaker
7. authorize_motion=false 启动完整音频链路
8. 验证 ASR、DOA、触发词和未授权拒绝
9. 停止并重新启动 authorize_motion=true
10. 现场确认安全后执行一次左转约15~30度
11. 等待确定 terminal result
12. cooldown 后执行一次右转约15~30度
13. 验证普通文本、deadband、重复和冷却行为
14. 确认 busy=false、底盘速度为零
15. 保存日志并清理临时 overlay
```

本方案的“完整”定义是：**真实语音输入、真实声源方向模型、真实声源转向节点、真实 Skill Gateway、真实导航栈和真实 LeKiwi 底盘全部参与；只有与该功能无关的视觉、抓取和 TTS 子系统可以通过测试 overlay 关闭。**

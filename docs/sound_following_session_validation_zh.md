# 声源跟随会话验证指南

本文定义 LeKiwi 周期声源跟随（periodic sound following）的真机验证流程：会话开关、
真实底盘转向、段级去重、关闭后零派发与普通技能共存。执行者应具备 ROS 2 基础并
遵守本文的安全要求。

## 1. 适用环境

```text
设备 SSH：310p-wm
工作区：/IB_Robot-lwh
机器人配置（生产默认）：lekiwi_nav_grasp.yaml（hybrid stage 默认携带周期声源跟随）
测试覆盖配置（导航专用 overlay）：lekiwi_nav_grasp_sound_real.yaml
ROS_DOMAIN_ID：55
RMW：rmw_cyclonedds_cpp
CycloneDDS 配置：/etc/cyclonedds_310p.xml
```

`lekiwi_nav_grasp` 的 hybrid stage 自本特性合入起默认启用周期声源跟随：节点随启动
运行、`sound_following` Skill 在 catalog 中可用，但会话初始为 `inactive`，必须显式
激活才会转向。其余 stage 与未声明该能力的机器人不受影响。

所有 ROS 命令必须在同一 shell 中加载环境：

```sh
cd /IB_Robot-lwh
source .shrc_local

export ROS_DOMAIN_ID=55
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///etc/cyclonedds_310p.xml
export LD_LIBRARY_PATH="/opt/ros/humble/lib64:/opt/ros/humble/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset FASTDDS_BUILTIN_TRANSPORTS
```

可确认 Python 来源：

```sh
python3 -c 'import sys; print(sys.executable); print(sys.prefix)'
```

应指向 `/IB_Robot-lwh/venv`。

## 2. 功能契约

### 2.1 配置

生产配置（`lekiwi_nav_grasp.yaml` 的 hybrid stage）包含：

```yaml
embodied:
  idle_behaviors:
    sound_orientation:
      enabled: true
      mode: periodic
      default_active: false
      periodic_interval_sec: 10.0
      max_direction_age_sec: 12.0
```

`default_active: false` 是安全默认值。机器人启动后不得因为环境声音自动转动。

### 2.2 控制 Skill

```text
sound_following（delegated executor，内部调用 /sound_orientation_node/set_following）
  motion_direction=forward  → 激活会话
  motion_direction=backward → 停用会话
```

用户和大模型不得直接发布 `/cmd_vel`、调用 controller 或绕过 Gateway 调用导航 action。

### 2.3 会话状态

```text
INACTIVE
ACTIVE
SHUTTING_DOWN
```

- `INACTIVE`：可以接收 DOA，但不能派发 `nav_turn`。
- `ACTIVE`：每 10 秒从未消费且新鲜的方向段中选择最新一段。
- `SHUTTING_DOWN`：停止新派发，等待正在执行的 `nav_turn` 获得明确终态。

## 3. 构建前检查

### 3.1 确认没有旧 pipeline

```sh
ps -eo pid,ppid,pgid,stat,args | grep -E \
  "embodied_pipeline|ros2_control_node|audio_capture_node|voice_asr_node|speech_direction_node|sound_orientation_node|safety_guard_node|skill_executor_node|navigation_command_server|cmd_vel_bridge|fastlio|livox|planner_server|controller_server" \
  | grep -v grep || echo ALL_CLEAN
```

如果有旧实例，优先在其启动终端按 `Ctrl+C`。禁止在旧实例运行时启动第二套 pipeline。

### 3.2 硬件占用

```sh
fuser /dev/ttyACM0 /dev/snd/pcmC0D0c 2>&1 || true
```

启动前应无输出。

### 3.3 构建相关包

```sh
./scripts/build.sh -- --packages-select \
  ibrobot_msgs \
  embodied_common \
  safety_guard \
  skill_catalog \
  skill_library \
  robot_config \
  embodied_agent \
  embodied_bringup
```

## 4. 阶段 A：无运动开关验证

### 4.1 启动

生产默认路径（base 配置，hybrid stage 默认生效；MoveIt 随 hybrid 自动启动）：

```sh
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=lekiwi_nav_grasp \
  use_sim:=false \
  control_mode:=base_navigation \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=false \
  2>&1 | tee /data/local/tmp/sound-following-safe.log
```

导航专用 overlay 路径（无 MoveIt/相机，聚焦声源链路）：

```sh
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  config_path:=/IB_Robot-lwh/src/robot_config/config/robots/lekiwi_nav_grasp_sound_real.yaml \
  nav_stage:=hybrid \
  control_mode:=base_navigation \
  voice_asr_auto_start:=false \
  with_moveit:=false \
  with_embodied:=true \
  entry_mode:=hermes \
  authorize_motion:=false \
  2>&1 | tee /data/local/tmp/sound-following-safe.log
```

说明：`voice_asr_auto_start` 在 launch 中声明默认 `false`，不传参时 ASR 保持关闭，
`voice_asr.enabled` 的 YAML 值只在显式传 `voice_asr_auto_start:=true` 时生效。
`with_moveit` 未传且 stage 为 hybrid + `base_navigation` 时自动置 true。

### 4.2 默认状态

启动日志必须包含：

```text
sound orientation ready: ... mode=periodic, session=inactive
```

### 4.3 内部服务幂等测试

此步骤只验证状态机，不验证自然语言/Gateway。

```sh
SERVICE=/sound_orientation_node/set_following
TYPE=ibrobot_msgs/srv/SetSoundFollowing

ros2 service call "$SERVICE" "$TYPE" "{schema_version: 1, enable: true}"
ros2 service call "$SERVICE" "$TYPE" "{schema_version: 1, enable: true}"
ros2 service call "$SERVICE" "$TYPE" "{schema_version: 1, enable: false}"
ros2 service call "$SERVICE" "$TYPE" "{schema_version: 1, enable: false}"
```

预期顺序：

```text
SOUND_FOLLOWING_ACTIVATED
ALREADY_ACTIVE
SOUND_FOLLOWING_DEACTIVATED
ALREADY_INACTIVE
```

### 4.4 Gateway Skill 测试

`robot-skill` 裸命令默认读取 so101_single_arm 配置，必须显式绑定本次配置；
若 PATH 上是 hermes-robot 生成的绑定包装（报错 `configuration is bound by
hermes-robot`），去掉 `--config-path` 直接使用裸命令，或使用下面的 `rs` 形式：

```sh
rs() { python3 -m robot_skill_cli.cli --config-path \
  /IB_Robot-lwh/src/robot_config/config/robots/lekiwi_nav_grasp.yaml "$@"; }
```

技能与契约检查：

```sh
rs list-skills | grep sound_following
rs describe sound_following
```

开启（未授权时 Gateway 返回 `MOTION_NOT_AUTHORIZED`，属安全门禁，不代表状态机失败）：

```sh
rs run-workflow \
  --text "开启声源跟随" \
  --workflow-json '[{"schema_version":2,"skill_name":"sound_following","motion_direction":"forward"}]'
```

关闭：

```sh
rs run-workflow \
  --text "关闭声源跟随" \
  --workflow-json '[{"schema_version":2,"skill_name":"sound_following","motion_direction":"backward"}]'
```

## 5. 阶段 B：DOA 链路检查

### 5.1 接口检查

```sh
ros2 topic info /audio/capture_stamped -v
ros2 topic info /voice/speech_direction -v
```

要求：

```text
/audio/capture_stamped Publisher count: 1
/voice/speech_direction Publisher count: 1
/voice/speech_direction Subscription count: 1
```

启动日志要求：

```text
speech_direction_node 已启动 ... degraded=False
```

如果出现 `audio contract` 错误，检查 Silero `ascend_310p` deployment：

```json
{
  "sample_rate_hz": 16000,
  "channels": 1,
  "channel_semantics": "silero_mono",
  "sample_dtype": "float32",
  "frame_size": 512,
  "chunk_size": 576,
  "execution_mode": "streaming"
}
```

### 5.2 无人现场的模拟方向输入

本步骤只模拟传感器输出，不绕过 Gateway 发送运动命令。

关闭状态下发布。使用当前 ROS 时钟发布，不能使用零时间戳，否则会被新鲜度检查拒绝：

```sh
python3 - <<'PY'
import time

import rclpy
from ibrobot_msgs.msg import SpeechDirection

rclpy.init()
node = rclpy.create_node("sound_direction_test_source")
publisher = node.create_publisher(SpeechDirection, "/voice/speech_direction", 10)
time.sleep(1.0)
message = SpeechDirection()
message.header.stamp = node.get_clock().now().to_msg()
message.header.frame_id = "base_link"
message.azimuth_rad = 0.35
message.seq_id = 101
message.segment_id = 101
message.direction_type = "seg_end"
publisher.publish(message)
rclpy.spin_once(node, timeout_sec=0.2)
node.destroy_node()
rclpy.shutdown()
PY
```

在 `INACTIVE` 状态下不得出现新的 `skill=nav_turn event=requested`。审计日志的 JSON
字段顺序为 `event` 在前、`skill` 在后，计数命令使用：

```sh
grep -c '"event":"requested".*"skill":"nav_turn"' <日志路径> || true
```

## 6. 阶段 C：授权真实底盘验证

### 6.1 安全要求

- 机器人周围无障碍物、人员和线缆。
- 现场人员可使用物理急停或断电。
- 初次验证方向建议不超过 30°；注意配置的 `max_turn_deg` 上限（默认 180°）。
- 禁止直接发布 `/cmd_vel` 或直接调用 Nav2 action。

### 6.2 启动

先停止阶段 A pipeline，然后以 `authorize_motion:=true` 重复 4.1 的启动命令
（base 路径或 overlay 路径均可），日志改为
`/data/local/tmp/sound-following-motion.log`。

### 6.3 开启并转向

```sh
rs run-workflow \
  --text "开启声源跟随" \
  --workflow-json '[{"schema_version":2,"skill_name":"sound_following","motion_direction":"forward"}]'
```

发布一条新方向（换新的 `segment_id`），或在机器人侧面发声。等待最多 15 秒，日志应出现：

```text
skill=nav_turn
event=requested
event=accepted
event=primitive_started
event=terminal
```

每个 `segment_id` 最多触发一次转向；同一 `segment_id` 重复发布（含刷新时间戳）用于
验证段级去重。

### 6.4 关闭后不得转向

```sh
rs run-workflow \
  --text "关闭声源跟随" \
  --workflow-json '[{"schema_version":2,"skill_name":"sound_following","motion_direction":"backward"}]'
```

记录关闭前 `nav_turn requested` 数量，再发布一个新 `segment_id` 或再次发声，等待 15 秒：

```sh
BEFORE=$(grep -c '"event":"requested".*"skill":"nav_turn"' \
  /data/local/tmp/sound-following-motion.log || true)
# ... 发布新方向或发声 ...
AFTER=$(grep -c '"event":"requested".*"skill":"nav_turn"' \
  /data/local/tmp/sound-following-motion.log || true)

test "$BEFORE" = "$AFTER" \
  && echo "PASS: disabled session did not dispatch nav_turn" \
  || echo "FAIL: nav_turn dispatched after disable"
```

## 7. 普通 Skill 共存测试

关闭声源跟随后，执行一个普通 Skill。base hybrid stage 提供完整移动操作技能集，例如：

```sh
rs run-workflow \
  --text "打开夹爪" \
  --workflow-json '[{"schema_version":1,"skill_name":"open_gripper_skill"}]'
```

导航专用 overlay（`lekiwi_nav_grasp_sound_real`）只含导航技能，改用：

```sh
rs run-workflow \
  --text "底盘左转15度" \
  --workflow-json '[{"schema_version":2,"skill_name":"nav_turn","direction":"left","degree":15.0}]'
```

预期：

- 普通 Skill 正常执行；
- 环境声音不再产生 `nav_turn`；
- Gateway 不处于遗留 busy 状态。

## 8. 通过标准

必须同时满足：

```text
[ ] 启动默认 inactive
[ ] enable（sound_following forward）成功
[ ] 重复 enable 幂等成功
[ ] active 状态下新 segment 触发 nav_turn
[ ] 同一 segment 只触发一次
[ ] disable（sound_following backward）成功
[ ] 重复 disable 幂等成功
[ ] disable 后新 segment 不触发 nav_turn
[ ] 关闭后普通 Skill 正常执行
[ ] pipeline 清理后无残留节点/设备占用
```

## 9. 常见故障分流

### 9.1 `skill catalog startup failed`

检查：

```text
manifest schema_version 与 implementation schema_version 一致
delegated executor 已加入 SUPPORTED_SKILL_EXECUTORS
profile 中 Skill 名称与 manifest 名称一致
runtime enabled 与 catalog enabled 一致（sound_following 与 imitate_human_motion 双向）
```

### 9.2 `robot-skill: configuration is bound by hermes-robot`

PATH 上的 `robot-skill` 是 hermes-robot 生成的绑定包装：直接使用裸命令（其绑定配置
正确时），或使用本文 4.4 的 `rs` 形式显式指定配置。

### 9.3 `speech_direction degraded=True`

先修复模型部署/audio contract。不要在 DOA 降级状态下进行底盘验证。

### 9.4 有 DOA，没有 `nav_turn`

检查：

```text
session=active
Gateway READY
motion_authorized=true
方向时间戳未超过 max_direction_age_sec
segment_id 尚未消费
```

### 9.5 只转一次

检查上一动作是否获得明确终态，以及会话是否卡在 `SHUTTING_DOWN`、`COOLDOWN` 或 `FAULT_UNKNOWN`。

### 9.6 多套孤儿节点

如果出现多个同名节点、设备负载异常或 WiFi 不稳定，停止所有旧 launch，再启动唯一实例。不要在同一个串口或声卡上并行启动多套 pipeline。

## 10. 清理

在启动终端按 `Ctrl+C`，然后确认：

```sh
ps -eo pid,ppid,pgid,stat,args | grep -E \
  "embodied_pipeline|ros2_control_node|audio_capture_node|voice_asr_node|speech_direction_node|sound_orientation_node|safety_guard_node|skill_executor_node|navigation_command_server|cmd_vel_bridge|fastlio|livox|planner_server|controller_server" \
  | grep -v grep || echo ALL_CLEAN

fuser /dev/ttyACM0 /dev/snd/pcmC0D0c 2>&1 || true
```

## 11. 测试报告模板

```markdown
## 环境

- Device: 310p-wm
- Commit/branch:
- ROS_DOMAIN_ID: 55
- Config:
- authorize_motion:

## 开关结果

- Default state:
- Enable result:
- Repeated enable result:
- Disable result:
- Repeated disable result:

## 声源转向

- DOA segment_id:
- azimuth_rad:
- nav_turn task_id:
- terminal success/error:
- Physical motion observed:

## 关闭后验证

- nav_turn count before:
- nav_turn count after:
- Ordinary Skill result:

## Remaining issues

-
```

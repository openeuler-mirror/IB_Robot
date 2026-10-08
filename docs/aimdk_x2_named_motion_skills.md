# 灵犀 X2 命名动作技能（aimdk_x2_skills）

本部署把 X2 运行时自带的 5 个预设社交手势暴露为 embodied 技能，可被 Hermes / planner / `robot-skill`
按中英文别名选中，并且只经 Capability Gateway 执行。

| 技能 | 运行时动作名 | 中文别名（节选） |
| --- | --- | --- |
| `wave_hand` | `wave` | 挥手、打招呼、再见 |
| `handshake` | `handshake` | 握手 |
| `raise_hand` | `raise_hand` | 举手 |
| `blow_kiss` | `blow_kiss` | 飞吻 |
| `clap_hands` | `clap` | 鼓掌、拍手 |

不暴露：姿态（坐下、蹲下、躺下、起立）、上下楼梯、语音。姿态会改变支撑状态，楼梯没有完成判定，语音需要
自由文本而技能请求没有该字段。

## 架构

```text
Hermes / planner / robot-skill
  -> /embodied/execute_skill
     Gateway：motion_authorized、root lease、运行时状态（ACTIVE、未锁存 stop、模式 idle）、
              required_capabilities（motion.named）、safety_guard（拒绝一切请求参数）
  -> skill_executor_node 的 runtime_named_motion 执行器
     预检：绑定名必须在 RuntimeStatus.capabilities_json 的 motion.named.names 中
  -> /motion/execute_named（ibrobot_msgs/action/ExecuteNamedMotion，target 为空、不打断）
  -> aimdk_runtime（唯一 aimdk_msgs 边界；准入：生命周期、仲裁、模式、平台处于稳定站立）
  -> 厂商 MC 层
```

- 没有新增 primitive，SO-101 / LeKiwi 的执行路径不变。
- 动作名只有一个真源：技能目录的 `binding.motion`，并在执行前对照运行时实时上报的列表检查。
- 端点只来自运行时公开接口描述：`embodied.runtime_named_motion` 是逻辑接口绑定，统一 launch 在运行时就绪后
  解析并注入。

配置细节见 `src/robot_config/README.md`「运行时自持运动的机器人」；目录规则见 `src/skill_catalog/README.md`
「Runtime named-motion skills」；执行器语义见 `src/skill_library/README.md` §8.7。

## 安全须知（先读）

- **开发机可能就在机器人网络上。** X2 的 SDK 网段是 `10.0.1.0/24`（运动控制单元 `10.0.1.40`，开发单元
  `10.0.1.41`）。`use_sim:=true` 的 vendor mock 与真机使用**同名**厂商接口，ROS 2 服务请求会被所有同名
  服务端接收：如果没有隔离，发给 mock 的模式切换、预设动作、急停卸力请求，真机也会执行。
- 因此 mock 运行时必须同时满足：`ROS_LOCALHOST_ONLY=1`、一个**不是**机器人使用的 `ROS_DOMAIN_ID`；
  最好让机器人断电或断开网线，并用 `ping 10.0.1.40` 确认不通。
- pytest 套件由仓库根 `conftest.py` 自动隔离（不要设置 `DISABLE_ROS_ISOLATION`）；手动 `ros2 launch` 与
  `robot-skill` **没有**自动隔离。
- 运动默认未授权：`authorize_motion:=false` 时所有技能返回 `MOTION_NOT_AUTHORIZED`，运行时收不到任何请求。
- 命名动作开始后**不可取消**（见下文）。真机上唯一的中断手段是操作员调用 `/runtime/stop`。

## 构建

需要 ROS 2 Humble 与和机器人固件匹配的 AimDK overlay（`aimdk_msgs` 不在仓库内，见
`src/robots/aimdk/aimdk_robot/README.md`）：

```bash
source .shrc_local
source ~/aimdk/install/local_setup.bash
./scripts/build.sh -- --packages-up-to aimdk_robot skill_library embodied_bringup safety_guard robot_skill_cli robot_config
```

## 在 vendor mock 上运行

终端 A（启动；运动未授权）：

```bash
source .shrc_local
source ~/aimdk/install/local_setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1      # 隔离，见安全须知
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=aimdk_x2_skills control_mode:=named_motion \
  use_sim:=true authorize_motion:=false
```

日志依次出现 `Runtime ready: aimdk_robot (..., ACTIVE)` 与
`Runtime interfaces bound; consumer snapshot: /tmp/ibrobot_interfaces_XXXX/robot.yaml`，随后启动
`skill_executor_node`、`safety_guard_node`、`agent_plan_node`。

终端 B（同一 DDS 环境；`--config-path` 用上面日志里的绑定后配置）：

```bash
source .shrc_local
source ~/aimdk/install/local_setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
SNAP=/tmp/ibrobot_interfaces_XXXX/robot.yaml
robot-skill --config-path "$SNAP" status            # 未授权时 5 个技能 ready=false
robot-skill --config-path "$SNAP" list-skills
robot-skill --config-path "$SNAP" validate handshake
robot-skill --config-path "$SNAP" execute handshake --task-id demo-1
robot-skill --config-path "$SNAP" cancel --task-id demo-1
```

`robot-skill` 不在 PATH 时可用 `python3 -m robot_skill_cli.cli` 代替。要在 mock 上验证执行成功，用
`authorize_motion:=true` 重新启动终端 A（**仅限已隔离的 mock**）。

## 结果语义

| 情况 | 结果 |
| --- | --- |
| 未授权运动 | `MOTION_NOT_AUTHORIZED`（Gateway 拒绝，不派发） |
| 已有技能在执行 | `SKILL_BUSY` |
| 运行时 stop 已锁存 | `CONTROL_MODE_MISMATCH`：“runtime stop is latched; explicit idle rearm is required”；技能链路不会清除锁存 |
| 运行时处于 `stream` 等其他模式 | `CONTROL_MODE_MISMATCH`：另一个指令源处于活动状态；不会抢切模式 |
| 名字不在运行时上报列表 | `NAMED_MOTION_UNKNOWN`（不派发） |
| 平台拒绝 | `NAMED_MOTION_REJECTED`，透传厂商原因 |
| 平台不在稳定站立 | `NAMED_MOTION_PLATFORM_NOT_READY`，如“platform is in JOINT_DEFAULT; preset motions require STAND_DEFAULT” |
| 运行时超时 | `NAMED_MOTION_TIMEOUT` |
| 成功 | `success=true`，如“preset motion handshake completed (task N)” |

**取消**：Gateway 接受取消请求，但执行器不会把取消转发给运行时（运行时拒绝取消命名动作）。动作照常完成，
结果按真实情况上报（成功时 message 以 `cancel not supported;` 开头）；动作结束前其他技能都返回
`SKILL_BUSY`。技能 deadline 先到时同样保留占用，直到运行时报告终态。

## 真机验收（操作员在场，机器人上吊架）

以下步骤**全部由操作员执行并记录**，不要交给 Agent 自动运行；运行环境改为真机网络与机器人的 domain
（不设 `ROS_LOCALHOST_ONLY`），`use_sim:=false`，时钟必须与机器人同步（见 aimdk_robot README）。

1. 只读：`status`、`list-skills`；确认 `RuntimeStatus.capabilities_json` 中 `motion.named.names` 包含
   5 个动作名。
2. **先验证急停**：不经技能路径，操作员直接让运行时播放一个预设动作，过程中调用
   `/runtime/stop`（`HOLD`），记录动画是否中止及时延。X2 profile 的 `hold_action` 为空，HOLD 可能不会
   打断动画；若不能中止，停在这一步，与运行时作者确认 stop 策略后再继续。
3. `authorize_motion:=true` 启动，执行单个技能（建议 `raise_hand`），记录时长与结果。
4. 让平台离开稳定站立后执行，确认得到 `NAMED_MOTION_PLATFORM_NOT_READY` 而不是动作。
5. 用实测值回填各技能的 `duration_sec_estimate` 与 `timeout_sec`（须大于运行时 `preset_timeout_s` 30 秒）。

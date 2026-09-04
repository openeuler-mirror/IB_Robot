# Hermes 控制 SO-101 真机手动验证指南

本文用于在 Ubuntu/openEuler 主机上，使用当前工作区和 `so101_single_arm.yaml`，通过 Hermes
自然语言控制 SO-101 真机，并验证运行中的 Workflow 是否可以被“别动”中断。

本文是**真机流程**，不使用 Gazebo、MuJoCo 或 Mock。执行任何运动前，操作员必须确认机械臂
周围环境安全、急停可用，并确认当前姿态适合控制器激活和后续运动。

## 1. 验证范围

本次验证链路如下：

```text
Hermes 自然语言
  -> hermes-robot
  -> robot-skill run-workflow
  -> Agent Plan
  -> Skill Gateway / Safety Guard
  -> MoveIt Gateway
  -> MoveIt
  -> ros2_control
  -> SO-101 真机
```

验证内容包括：

- 真机硬件是否能够激活；
- `moveit_planning` 控制器是否全部 active；
- Embodied Runtime、Skill Gateway、Agent Plan 和 MoveIt Gateway 是否 ready；
- Hermes 是否能够通过自然语言完成 `nod_yes`、`wave_hello` 或 `recover_safe_pose`；
- Workflow 执行期间输入“别动”时，是否能够取消当前任务，而不是把消息排队成新的普通请求；
- 取消后是否不再执行 Workflow 中尚未开始的步骤。

本次验证不包含视觉抓取、模型推理、语音识别或 TTS 功能。`nod_yes`、`wave_hello` 和
`recover_safe_pose` 不需要视觉输入，因此启动时关闭 Embodied Perception，避免视觉感知运行时
参与本次动作链路。注意：基础 `robot.launch.py` 仍可能根据机器人 YAML 创建配置中的相机节点；
相机缺失时可能出现相机节点告警或退出，但只要控制器、MoveIt Gateway、Safety Guard、Skill
Executor 和 Agent Plan 正常，非视觉动作仍可继续验证。

## 2. 重要安全规则

- 只能通过 Hermes 和 `robot-skill` 的 Capability Gateway 执行运动。
- 不要直接调用 MoveIt action、ros2_control action、controller topic 或 primitive 命令发送运动。
- `ros2 control list_controllers` 只用于只读检查，不能用于发送运动。
- `authorize_motion:=true` 只能在现场安全检查完成后由操作员显式设置。
- 本文所有真机命令都使用 `use_sim:=false`，不要改成 `true`。
- 如果 Gateway 未 ready、控制器未 active、串口异常或停止状态未知，不要发送新的运动任务。
- 看到“取消请求已发送”不等于机械臂已停止，必须等待唯一的取消 terminal result。
- 如果返回 `SKILL_CANCEL_TIMEOUT`，不得继续执行 `recover_safe_pose`，先进行现场安全处置。
- 只有确认机器人处于安全位置后，才能关闭 ROS 或执行清理脚本。

## 3. 工作区和环境

本文使用环境变量表示工作区。设备上的工作区如果是 `/IB_Robot-refactor`，直接使用下面命令；
如果实际路径不同，只修改 `IBROBOT_WS` 的值。

```bash
export IBROBOT_WS=/IB_Robot-refactor
cd "$IBROBOT_WS"
```

所有终端都必须使用相同的 ROS Domain。本文统一使用 `52`：

```bash
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52
export ROS2CLI_DISABLE_DAEMON=1
```

每次在新终端执行 ROS 2、`robot-skill` 或 Hermes 命令时，都要重新执行上述环境初始化。
不要在不同终端混用其他工作区的 `install` 环境。

## 4. 启动前检查

### 4.1 现场检查

启动前由操作员确认：

1. SO-101 已正确连接并完成与当前设备匹配的校准；
2. 机械臂处于不会在激活时碰撞的姿态；
3. 机械臂工作范围内没有人员、线缆和障碍物；
4. 急停或断电手段在手边且可以立即使用；
5. 没有其他程序占用 SO-101 串口；
6. 本次验证使用的是预期 Git 分支和当前源码构建产物。

### 4.2 主机只读检查

```bash
cd "$IBROBOT_WS"
git branch --show-current
git status --short
ls -l /dev/ttyACM0
```

确认当前用户可以访问串口：

```bash
id
test -r /dev/ttyACM0 && test -w /dev/ttyACM0
```

如果串口权限不足，先处理用户的 `dialout` 权限并重新登录，不要通过临时修改设备权限
绕过安全配置：

```bash
sudo usermod -aG dialout "$USER"
```

重新登录后再执行 `id` 和串口检查。

## 5. 构建当前工作区

代码或消息接口有更新时，先停止旧的 ROS 进程，再构建当前工作区：

```bash
cd "$IBROBOT_WS"
./scripts/cleanup_ros.sh
source .shrc_local
./scripts/build.sh -- \
  --packages-select \
  ibrobot_msgs \
  embodied_common \
  embodied_agent \
  skill_library \
  robot_config \
  embodied_bringup \
  robot_skill_cli
```

如果本次改动涉及其他包，按构建依赖补充对应包，或者执行完整构建：

```bash
source .shrc_local && ./scripts/build.sh
```

构建失败时停止，不要继续使用旧的或不完整的 `install` overlay 启动真机。

## 6. 终端 1：启动真机 Embodied Pipeline

`embodied_pipeline.launch.py` 是本次流程的唯一启动入口。它会先启动基础机器人系统，等待
控制器 active 后，再启动 Embodied Runtime、Agent Plan、Skill Executor 和 Safety Guard。

在终端 1 执行：

```bash
export IBROBOT_WS=/IB_Robot-refactor
cd "$IBROBOT_WS"
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52
export ROS2CLI_DISABLE_DAEMON=1

ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  use_sim:=false \
  control_mode:=moveit_planning \
  authorize_motion:=true \
  with_embodied:=true \
  with_perception:=false \
  moveit_display:=false
```

参数含义：

| 参数 | 值 | 作用 |
|---|---|---|
| `robot_config` | `so101_single_arm` | 使用 SO-101 单臂 SSOT 配置 |
| `use_sim` | `false` | 使用真实串口硬件，不使用仿真 |
| `control_mode` | `moveit_planning` | 使用 MoveIt 轨迹控制，避免加载 ACT 推理模型 |
| `authorize_motion` | `true` | 显式开启运动授权 |
| `with_embodied` | `true` | 启动 Agent Plan、Skill Gateway、Safety Guard 等运行时 |
| `with_perception` | `false` | 本次只验证非视觉动作，关闭 Embodied Perception 运行时 |
| `moveit_display` | `false` | 不启动 RViz，减少真机验证资源占用 |

启动日志中必须看到以下关键信息：

```text
SO101SystemHardware: Activated! Control loop running.
Controllers are active: joint_state_broadcaster, arm_trajectory_controller, gripper_trajectory_controller
safety_guard ready
skill_executor ready
agent_plan_node started
MoveIt Gateway fully initialized
```

以下情况意味着启动未通过，立即停止后续验证：

- `Failed to connect to motors`；
- `Permission denied`；
- `Failed to set the initial state`；
- 控制器 spawner 退出码不是 0；
- `agent_plan_node`、`skill_executor_node` 或 `safety_guard_node` 启动失败；
- `MoveIt Gateway` 没有完成初始化。

保持终端 1 前台运行，不要关闭。

## 7. 终端 2：检查控制器和 Gateway

### 7.1 检查控制器

```bash
export IBROBOT_WS=/IB_Robot-refactor
cd "$IBROBOT_WS"
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52
export ROS2CLI_DISABLE_DAEMON=1

ros2 control list_controllers
```

以下三个控制器必须全部为 `active`：

```text
joint_state_broadcaster       active
arm_trajectory_controller     active
gripper_trajectory_controller active
```

任一控制器不是 `active` 时，不要启动 Hermes，也不要发送运动请求。

### 7.2 检查 Gateway

```bash
export ROBOT_CONFIG_PATH="$IBROBOT_WS/src/robot_config/config/robots/so101_single_arm.yaml"

robot-skill --config-path "$ROBOT_CONFIG_PATH" status
```

必须确认返回内容至少包含：

```text
control_plane_ready: true
control_plane_state: READY
motion_authorized: true
active_control_mode: moveit_planning
busy: false
```

继续检查能力目录：

```bash
robot-skill --config-path "$ROBOT_CONFIG_PATH" list-skills
robot-skill --config-path "$ROBOT_CONFIG_PATH" describe nod_yes
robot-skill --config-path "$ROBOT_CONFIG_PATH" describe wave_hello
robot-skill --config-path "$ROBOT_CONFIG_PATH" describe recover_safe_pose
```

至少确认 `nod_yes`、`wave_hello` 和 `recover_safe_pose` 存在，且 `ready: true`。

如果出现以下错误，停止并修复运行环境：

```text
SERVER_UNAVAILABLE
AGENT_PLAN_UNAVAILABLE
control_plane_ready: false
motion_authorized: false
CAPABILITY_NOT_READY
```

## 8. 终端 3：启动 Hermes CLI

在终端 3 执行：

```bash
export IBROBOT_WS=/IB_Robot-refactor
cd "$IBROBOT_WS"
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52

IBROBOT_HERMES_LAUNCH_TRACE=1 \
hermes-robot \
  --config-path "$IBROBOT_WS/src/robot_config/config/robots/so101_single_arm.yaml" \
  --mode motion \
  -- --cli
```

`hermes-robot` 启动时会检查：

- `hermes` 和 `robot-skill` 是否可发现；
- Hermes 版本和受控 `ibrobot-control` Skill 是否可用；
- Gateway control plane 是否 ready；
- Agent Plan 服务和 action 是否可发现。

启动追踪中应依次看到：

```text
hermes-robot: stage: resolving executables
hermes-robot: stage: checking Hermes version
hermes-robot: stage: validating ROS wire contracts
hermes-robot: stage: checking robot runtime
hermes-robot: stage: resolving Hermes profile
hermes-robot: stage: registering IB-Robot skill
hermes-robot: stage: checking Hermes skill discovery
hermes-robot: stage: starting Hermes Gateway
```

然后应进入：

```text
Welcome to Hermes Agent! Type your message or /help for commands.
Activated skills: ibrobot-control
```

如果卡在 `checking robot runtime`，返回终端 2 重新执行 `robot-skill status`，优先检查
ROS Domain、Gateway 节点和 Agent Plan 接口。不要切换到裸 ROS 命令执行动作。

## 9. Hermes 自然语言动作验证

### 9.1 单步点头

在 Hermes CLI 输入：

```text
让真实机械臂点头
```

Hermes 应将请求转换为只有一个步骤的 Workflow：

```text
1. nod_yes
```

执行过程必须包含：

1. 查询当前 status 和 Skill Catalog；
2. 生成 typed workflow；
3. 展示完整步骤、参数、plan digest、registry identity 和 fresh task ID；
4. 展示并 flush 后立即完成内部 confirm；
5. 执行 `execute-plan`；
6. 等待唯一 terminal result。

通过标准：

```text
success: true
error_code: 空
completed_step_count: 1
```

同时现场确认机械臂确实完成点头。

### 9.2 多步 Workflow

单步动作成功后，在同一个 Hermes 会话输入：

```text
让真实机械臂点头，然后回到安全位置
```

冻结计划必须严格为：

```text
1. nod_yes
2. recover_safe_pose
```

不得增加、删除或重排步骤。必须等待最终结果，并现场确认机器人回到安全位置。

### 9.3 其他非视觉动作

可使用以下请求验证同一自然语言入口：

```text
让真实机械臂挥手
```

或：

```text
请让真实机械臂回到安全位置，只使用 recover_safe_pose
```

每次都必须等待唯一 terminal result，不能只根据“goal accepted”或 Hermes 的中间反馈判断
动作成功。

## 10. 验证“别动”中断

### 10.1 推荐验证方式

使用包含两个步骤的请求，便于确认取消后第二步没有继续提交：

```text
让真实机械臂挥手，然后回到安全位置
```

在机械臂仍在运动、Hermes 尚未输出 `workflow_terminal` 时，立即输入：

```text
别动
```

正确链路应为：

```text
别动
  -> Hermes 抢占当前 robot-skill 工具
  -> run-workflow 收到 SIGINT/SIGTERM
  -> controller.request_stop()
  -> cancel_agent_plan
  -> Agent Plan 取消当前 child Skill
  -> 返回 SKILL_CANCELLED
```

通过标准：

- 返回唯一的取消 terminal result；
- `success: false`；
- `error_code: SKILL_CANCELLED`；
- 机械臂停止当前动作；
- `recover_safe_pose` 没有被提交或执行；
- 没有自动重试、自动恢复或新的运动请求。

仅出现以下信息都不能作为通过依据：

```text
Redirected current turn: '别动'
CancelGoal accepted
消息已发送
工具进程退出
```

### 10.2 结果判定

如果 Hermes 输入“别动”后看到：

```text
Redirected current turn: '别动'
```

但原来的 `robot-skill run-workflow` 继续执行到成功，说明“别动”被 Hermes 当成普通新
turn 排队，未抢占正在运行的工具进程。

这时不能把本次测试标记为通过。保存完整 Hermes 输出，并执行下一节的 CLI 直接中断测试，
区分仓库侧取消链路和 Hermes 工具抢占问题。

## 11. 不经过 Hermes 的 CLI 中断对照测试

该测试只用于定位问题，不替代 Hermes 自然语言验收。

在终端 4 执行：

```bash
export IBROBOT_WS=/IB_Robot-refactor
cd "$IBROBOT_WS"
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52
export ROS2CLI_DISABLE_DAEMON=1
export ROBOT_CONFIG_PATH="$IBROBOT_WS/src/robot_config/config/robots/so101_single_arm.yaml"

robot-skill --config-path "$ROBOT_CONFIG_PATH" \
  run-workflow \
  --text "让真实机械臂挥手，然后回到安全位置" \
  --workflow-json \
  '[{"schema_version":1,"skill_name":"wave_hello"},{"schema_version":1,"skill_name":"recover_safe_pose"}]'
```

命令已经开始执行后，在该命令所在终端按：

```text
Ctrl-C
```

预期结果：

```text
workflow_terminal
success: false
error_code: SKILL_CANCELLED
```

该对照测试的结论：

- 直接 `Ctrl-C` 能返回 `SKILL_CANCELLED`，Hermes 输入“别动”不能：仓库侧 CLI/Gateway
  取消链路正常，问题在 Hermes 工具抢占；
- 直接 `Ctrl-C` 也不能返回 `SKILL_CANCELLED`：继续检查当前 install overlay、Agent Plan
  节点、ROS action cancel 和真机运行日志；
- 返回 `SKILL_CANCEL_TIMEOUT`：停止发送任何动作，现场确认机械臂状态，不能继续验证。

## 12. 失败与停止状态处理

### 12.1 启动失败

以下错误通常表示真机或运行时未准备好：

```text
Permission denied
Failed to connect to motors
controller spawner exited with code 1
SERVER_UNAVAILABLE
AGENT_PLAN_UNAVAILABLE
```

处理顺序：

1. 停止当前 launch；
2. 检查 `/dev/ttyACM0` 权限和串口占用；
3. 确认 `ROS_DOMAIN_ID` 一致；
4. 重新启动真机流水线；
5. 确认三个控制器全部 active；
6. 重新执行 Gateway status；
7. 只有 status 通过后才重启 Hermes。

### 12.2 摄像头和 TTS 非阻塞告警

使用本文的非视觉动作验证时，可能看到类似日志：

```text
No RealSense devices were found
Unable to open camera calibration file
usb_cam_node_exe: process has died
Unable to read JSON file .../zipvoice/inference_manifest.json
```

这些问题属于摄像头或 TTS 配置，可能不影响 `nod_yes`、`wave_hello` 和
`recover_safe_pose`。但以下组件必须正常，否则不能执行动作：

- SO-101 硬件接口；
- 三个控制器；
- MoveIt Gateway；
- Safety Guard；
- Skill Executor；
- Agent Plan；
- Gateway status。

需要验证视觉抓取或语音播报时，必须另外修复相机、标定文件和 TTS 模型，不能把本节的
非视觉验证结果扩大为完整系统通过。

### 12.3 取消超时或未知状态

如果返回：

```text
SKILL_CANCEL_TIMEOUT
```

或者出现 transport failure、结果缺失、取消身份不匹配等情况：

- 不要发送 `recover_safe_pose`；
- 不要输入“继续”；
- 不要更换 task ID 重试；
- 不要关闭控制器来代替取消；
- 先通过现场急停或人工方式确认机械臂安全；
- 保存 Hermes、CLI 和 ROS 日志。

只有以下条件同时满足时，才可以把任务标记为确定取消：

```text
GoalStatus == 5
success == false
error_code == SKILL_CANCELLED
plan/task/registry identity 与本次任务一致
```

## 13. 验证结束和安全回位

如果上一任务明确成功或明确返回 `SKILL_CANCELLED`，在 Hermes 中输入：

```text
请让真实机械臂回到安全位置，只使用 recover_safe_pose
```

确认：

```text
success: true
error_code: 空
completed_step_count: 1
```

并现场确认机械臂处于安全位置。

只有确认安全回位成功后，才可以在启动 ROS 的终端按 `Ctrl-C` 关闭流水线。如有残留进程，
再执行：

```bash
cd "$IBROBOT_WS"
./scripts/cleanup_ros.sh
```

不要通过直接 kill `controller_manager` 或单独 kill 某个控制器代替正常关闭流程。

## 14. 验收记录

建议保存以下信息：

```text
Git branch / commit:
Workspace:
Robot config: src/robot_config/config/robots/so101_single_arm.yaml
ROS_DOMAIN_ID: 52
use_sim: false
control_mode: moveit_planning
authorize_motion: true
with_embodied: true
with_perception: false
moveit_display: false
```

同时记录：

- 三个控制器的 active 状态；
- Gateway status 中的 registry epoch、generation 和 digest；
- `nod_yes` 的 plan digest、task ID 和 terminal result；
- 多步 Workflow 的步骤顺序和 terminal result；
- “别动”输入时间；
- 是否出现 `cancel_agent_plan`；
- 是否返回 `SKILL_CANCELLED`；
- 取消后是否执行了后续步骤；
- Hermes 的完整输出，尤其是 `Redirected current turn` 前后文；
- 最终 `recover_safe_pose` 的成功结果；
- `cleanup_ros.sh` 的执行时间和结果。

本次功能只有在以下两项都通过时才算完成：

1. Hermes 能通过自然语言控制真机并得到唯一成功 terminal result；
2. Hermes 执行 Workflow 期间输入“别动”能够抢占当前工具，得到唯一
   `SKILL_CANCELLED` terminal result，且不会执行后续步骤。

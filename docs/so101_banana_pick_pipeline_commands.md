# LeKiwi 腕部 RealSense 香蕉抓取操作指南

本文用于 LeKiwi 移动操作臂真机抓取（LeKiwi 全向底盘 + SO-101 follower 臂，由 `LeKiwiSystemHardware`
插件统一驱动）：腕部 RealSense 采集 RGB-D，经 Grounding-DINO/SAM2、GraspGen 和 MoveIt 完成目标检测、
抓取规划、执行与验证。抓取执行期间底盘保持静止，仅臂与夹爪运动。

支持两个运行平台：

| 平台 | Robot config | 推理后端 | 命令执行位置 |
|---|---|---|---|
| 310P | `lekiwi_handeye_realsense_grasp` | Ascend NPU | 仓库根目录 |
| PC | `lekiwi_handeye_realsense_grasp_pc` | NVIDIA CUDA | 仓库根目录 |

> **安全要求**：本文第 3.1、4.3、5.2 节会驱动真机。执行前必须清空工作区、确认急停可用，并由操作员对本次运动
> 明确授权。失败、超时或机器人状态未知时禁止自动重试。

## 执行顺序

首次部署先完成第 0 章；除硬件、标定或模型发生变化外，完成后日常抓取可跳过。每次抓取按以下顺序执行：

```text
选择平台并加载环境
  -> 检查设备、标定和模型
  -> 清理残留 ROS 节点
  -> 启动统一 pipeline
  -> 检查控制器、服务和相机
  -> 可选：观测位/规划冒烟测试
  -> 通过 Hermes 或公开技能 CLI 执行抓取
  -> 核对 terminal result 和验证证据
```

日常操作以第 1～6 章为主。第 4 章适用于 Hermes 自然语言控制，第 5 章适用于操作员直接执行已明确选择的
`pick_object` 技能；两者都经过 Capability Gateway。排障和调参见第 7、8 章。

## 0. 首次准备（完成后日常可跳过）

### 0.1 安装、下载和构建

除非另有说明，本文所有命令均在 IB_Robot 仓库根目录执行；所有项目内路径均相对于仓库根目录。

在目标运行平台执行：

```bash
./scripts/setup.sh
source .shrc_local
colcon build --symlink-install --merge-install --packages-up-to \
  lekiwi_description lekiwi_hardware embodied_bringup robot_skill_cli \
  perception_service manipulation_execution
```

PC 额外下载 Grounded-SAM2 Torch bundle：

```bash
./scripts/download_perception_models.sh --gdino-only
```

该脚本不下载 GraspGen checkpoint，也不生成 310P 的 OM bundle。`models/` 是被 Git 忽略的部署目录，fresh clone
中不存在属于正常情况。310P 与 PC 必须分别按第 0.3 节下发或生成与 robot config 一致的 bundle，不能从另一
平台的本地目录推断模型已经就绪。

### 0.2 填写硬件与标定配置

按平台修改对应文件：

- 310P：`src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml`
- PC：`src/robot_config/config/robots/lekiwi_handeye_realsense_grasp_pc.yaml`

启动前必须确认：

- `ros2_control.port` 指向实际机械臂串口；
- 腕部 RealSense 使用 `rs-enumerate-devices` 或相机节点日志中的 librealsense serial 显式绑定；当前 310P
  所接 D435i 为 `349522071345`，不要使用 USB sysfs descriptor serial；
- 手眼标定值来自当前机械臂和相机组合；
- 两份配置的硬件字段保持一致，只让推理后端存在平台差异。

310P 必须使用飞书《310P RealSense 相机使能与部署指导》中定义的 Color-first
`realsense2_camera 4.57.7`，让 RGB Camera 先于 Stereo Module 启动。YAML 参数本身不能改变 sensor
启动顺序；系统仍加载原版 Depth-first wrapper 时，不得继续抓取。当前 310P 配置保持 `640x360@30`，并使用
`initial_reset=false`、`enable_sync=false`、关闭 pointcloud/IMU。抓取链路逐项恢复 alignment，因为 planner
必须消费 `/camera/wrist/aligned_depth_to_color/{image_raw,camera_info}`。

### 0.3 准备模型

两端使用相同的逻辑模型，但 named deployment 和物理 bundle 不同：

| 平台 | 用途 | Bundle | Named deployment | 准备方式 |
|---|---|---|---|---|
| PC | 检测与分割 | `models/grounded_sam2_swint_ogc/` | `torch_cuda` | `download_perception_models.sh --gdino-only` |
| PC | GraspGen | `models/graspgen/` | `torch_cuda` | 从统一 HF bundle 下载 |
| 310P | Grounding-DINO | `models/grounding_dino_swint_seq8_1280x720/` | `ascend_310p` | 下发已 promotion 的 compiled bundle |
| 310P | SAM2 prompt | `models/sam2.1_hiera_tiny_prompt_ascend/` | `ascend_310p` | 下发已 promotion 的 compiled bundle |
| 310P | GraspGen | `models/graspgen/` | `ascend_310p` | 下发已 promotion 的 compiled bundle |

310P 不在板端临时编译模型。将发布流程产出的三个完整 bundle 同步到 `models/` 后检查：

```bash
test -f models/grounding_dino_swint_seq8_1280x720/inference_manifest.json
test -f models/sam2.1_hiera_tiny_prompt_ascend/inference_manifest.json
test -f models/graspgen/inference_manifest.json
```

Ascend 感知 bundle 的 promotion 命令和候选目录约束见
[`src/perception_service/README.md`](../src/perception_service/README.md#91-ascend-310p-抓取感知)；GraspGen 的
ONNX、OM 与 bundle 打包流程见
[`src/model_utils/model_utils/README.md`](../src/model_utils/model_utils/README.md#graspgen-ascend-packaging)。模型转换
中间产物只能放在 `models/_work/`，不能写入上述发布 bundle。

PC 首次部署时先确认 Grounded-SAM2 bundle。GraspGen checkpoint 不由感知模型下载脚本提供；从已审核模型源
取得以下三个文件后，用 packager 复制到标准 bundle 并生成 manifest：

```bash
source .shrc_local
test -f models/grounded_sam2_swint_ogc/inference_manifest.json

GRASPGEN_SOURCE_ROOT=models/_work/graspgen_robotiq_2f_140/source
test -f "$GRASPGEN_SOURCE_ROOT/checkpoints/graspgen_robotiq_2f_140.yml"
test -f "$GRASPGEN_SOURCE_ROOT/checkpoints/graspgen_robotiq_2f_140_gen.pth"
test -f "$GRASPGEN_SOURCE_ROOT/checkpoints/graspgen_robotiq_2f_140_dis.pth"
ros2 run perception_service package_graspgen_torch_bundle \
  --source-root "$GRASPGEN_SOURCE_ROOT" \
  --bundle-root models/graspgen
test -f models/graspgen/inference_manifest.json
```

`models/grasp/checkpoints/` 仅是旧部署的兼容输入布局，不是 fresh clone 应自带的仓库内容。任一检查失败时，先从
已审核发布物补齐模型或重新执行对应 packager，不要继续启动真机抓取，也不要把 ignored 模型文件提交到 Git。

## 1. 每次启动前

### 1.1 在每个终端进入对应工作区并加载环境

310P：

```bash
source .shrc_local
export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1
source install/setup.bash
```

PC：

```bash
source .shrc_local
export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1
source install/setup.bash
```

`pipeline`、检查命令和客户端的 `ROS_DOMAIN_ID`、`ROS_LOCALHOST_ONLY` 必须一致。

### 1.2 静态预检

310P：

```bash
test -e /dev/ttyACM0
python3 scripts/check_handeye_preconditions.py \
  --robot-config src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml \
  --camera-name wrist \
  --check-files
```

PC：

```bash
test -e /dev/ttyACM0
python3 scripts/check_handeye_preconditions.py \
  --robot-config src/robot_config/config/robots/lekiwi_handeye_realsense_grasp_pc.yaml \
  --camera-name wrist \
  --check-files
```

310P 还应确认 `/dev/video0` 以及安装空间：

```bash
test -e /dev/video0
ros2 pkg prefix robot_config
ros2 pkg prefix perception_service
ros2 pkg prefix manipulation_service
ros2 pkg prefix manipulation_execution
ros2 pkg prefix embodied_bringup
ros2 pkg prefix robot_moveit
ros2 pkg prefix so101_hardware
ros2 pkg prefix lekiwi_hardware
ros2 pkg prefix lekiwi_description
```

### 1.3 清理残留节点

上次启动被中断，或出现机器人不响应、重复 MoveIt action server 时执行：

```bash
./scripts/cleanup_ros.sh
```

然后确认现场满足以下条件：

- 人员、工具和无关物体已退出机械臂工作区；
- 香蕉位于相机视野和机械臂工作空间内，周围有足够夹爪间隙；
- 机械臂、夹爪和相机状态正常；
- 急停可立即触达；
- 本次任务使用新的 task ID。

## 2. 启动并检查 pipeline

### 2.1 终端 A：启动统一 pipeline

310P：

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=lekiwi_handeye_realsense_grasp \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

PC：

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=lekiwi_handeye_realsense_grasp_pc \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

只做无运动诊断时，将 `authorize_motion` 改为 `false`，并且不要执行第 3.1、4.3、5.2 节。

统一 launch 会启动 LeKiwi 控制器（含 SO-101 follower 臂与底盘）、RealSense、MoveIt、感知服务、抓取 planner/verifier/executor、IK worker、
Capability Gateway 和安全节点。无需再单独启动这些节点。

等待以下关键日志：

```text
Controllers are active
MoveIt Gateway fully initialized
TaskExecutor ready
GraspPlannerNode ready
PickExecutor ready: action=/manipulation/execute_pick ... ik_workers=4
```

### 2.2 终端 B：运行时检查

先按第 1.1 节重新设置同一平台和 ROS 环境，再执行：

```bash
ros2 control list_controllers
ros2 service list | grep -E 'grounding_detect|segment_detections|plan_grasp|compute_ik|compute_fk|move_to_configuration|verify_grasp'
ros2 action info /move_action
ros2 topic info /camera/wrist/image_raw
ros2 topic info /camera/wrist/aligned_depth_to_color/image_raw
ros2 topic info /camera/wrist/aligned_depth_to_color/camera_info
```

必须满足：

- `joint_state_broadcaster`、`arm_trajectory_controller`、`gripper_trajectory_controller` 均为 `active`；
- `/move_action` 只有一个 `/move_group` action server；
- 检测、规划、IK/FK、运动和验证服务均可用；
- RGB、对齐深度和 CameraInfo 均有发布者。

检查失败时不要发送抓取任务，转到第 7 章排障。

可选启动 RViz：

```bash
rviz2
```

查看 `/camera/wrist/image_raw` 时，将 `Image` display 的 Reliability 设为 `Best Effort`。

## 3. 可选的分阶段验证

pipeline 已稳定运行且配置未变化时，可以跳过本章，直接选择第 4 章 Hermes 流程或第 5 章单技能 CLI 流程。

### 3.1 移动到观测姿态

此命令会产生机械臂运动：

310P：

```bash
robot-skill --config-name lekiwi_handeye_realsense_grasp execute inspect_scene \
  --task-id inspect-scene-001 \
  --timeout-sec 30
```

PC：

```bash
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc execute inspect_scene \
  --task-id inspect-scene-001 \
  --timeout-sec 30
```

观测位姿和速度来自当前 robot config，客户端不应覆盖。

### 3.2 感知与规划冒烟测试

首次启动、重启 pipeline、更换模型/标定/硬件，或出现 readiness 错误时建议执行。

确认感知服务：

```bash
ros2 service list | grep -E '/perception/grasp/(grounding_detect|segment_detections)'
```

只请求抓取规划，不执行运动：

```bash
ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp \
  "{text_prompt: 'banana', confidence_threshold: 0.1, grasp_threshold: 0.5, debug_output_mode: 'diagnostic'}"
```

`PickObject.MODE_PLAN_ONLY` 是 Gateway 内部诊断模式，尚未作为外部 catalog 参数公开。Hermes 和生产调用方
不得伪造 `dispatch_binding`、`dispatch_nonce` 或 `expected_executor` 直连 delegated action。

## 4. 通过 Hermes 执行抓取

Hermes 是正式对外执行入口，复用第 2 章启动的同一套 pipeline，不得再启动 planner、executor 或 Gateway。

### 4.1 启动受控 Hermes CLI

PC 在本机新终端执行；310P 必须 SSH 到运行 pipeline 的同一块板卡后执行。先按第 1.1 节设置平台与 ROS
环境，再按所在平台启动。

PC：

```bash
hermes-robot --config-name lekiwi_handeye_realsense_grasp_pc -- --cli
```

310P：

```bash
hermes-robot --config-name lekiwi_handeye_realsense_grasp -- --cli
```

`hermes-robot` 会检查 Hermes、`ibrobot-control`、绑定配置、Gateway 状态和 Agent plan 接口。预检失败时停止，
不要改用裸 `hermes --cli` 绕过。会话已绑定 robot config，Hermes 不得再次传入 `--config-name` 或
`--config-path`。

### 4.2 规划和校验

在 Hermes CLI 输入：

```text
请使用 ibrobot-control 抓取 banana。
严格按 status -> list-skills -> plan-workflow -> describe pick_object -> validate-plan 执行只读阶段。
计划必须只有一个 pick_object step，target_name 必须是 banana。
请展示完整 step 和参数、plan digest、registry identity 以及新的 task ID。
展示并 flush exact plan 后，立即使用同一 tuple 调用一次 confirm-plan 和 execute-plan，不等待二次确认。
如果计划不正确，我会输入“别动”；收到停止指令后必须取消当前 Agent plan 并等待权威 terminal result。
```

以下任一情况都必须停止：Gateway 未就绪、运动未授权、校验失败、计划增减步骤、目标参数不一致。

### 4.3 展示后立即执行

Hermes 展示并 flush exact plan、digest、registry identity 和新 task ID 后，必须立即使用展示过的 plan token、
digest 和 task ID 调用一次 `confirm-plan`，再用返回的 confirmation token 调用一次 `execute-plan`。
`confirm-plan` 是 Gateway 对 exact plan/task tuple 的内部技术绑定，不是用户二次确认门禁。默认使用 Gateway
task budget，不传 `--timeout-sec`；只有操作员明确要求更短预算时，才给两个命令传入相同 timeout。

如果计划错误或需要停止，在同一 Hermes 会话输入“别动”。当前会话拥有运行中的 `execute-plan` 进程时，由该
进程发起一次取消并等待 terminal result；只有当前会话不再拥有执行进程时，才使用 `cancel-plan`，并且必须同时
携带展示过的 task ID、plan ID/digest、registry epoch/generation/digest 和 expected step count。不得只凭 task ID
构造取消请求，也不得同时使用进程信号和外部 `cancel-plan`。失败、超时或状态未知时不得自动 replan 或重试；
只有唯一 terminal result 能证明任务完成或已经取消。

310P 的 Hermes 会话和 pipeline 必须位于同一块 310P，不能从 PC 本地另起会话跨机绕过 Gateway。

## 5. 通过公开技能直接执行抓取

当操作员已经明确选择单个 `pick_object` 技能时，可以像执行 `place_in_container` 一样直接使用
`robot-skill`。该入口仍由 Capability Gateway 完成契约校验、准入和动作分发，不使用
supervised-direct `pick_action_client`，也不直连 `/manipulation/execute_pick`。

### 5.1 只读检查

在新终端加载项目和 ROS 环境。PC 依次执行：

```bash
source .shrc_local
export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1
source install/setup.bash
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc status
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc list-skills
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc describe pick_object
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc validate pick_object \
  --target-name banana
```

310P 在板端依次执行：

```bash
source .shrc_local
export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1
source install/setup.bash
robot-skill --config-name lekiwi_handeye_realsense_grasp status
robot-skill --config-name lekiwi_handeye_realsense_grasp list-skills
robot-skill --config-name lekiwi_handeye_realsense_grasp describe pick_object
robot-skill --config-name lekiwi_handeye_realsense_grasp validate pick_object \
  --target-name banana
```

确认 Gateway ready、catalog 中存在 `pick_object`，且 `validate` 返回允许执行。任一命令失败都立即停止，
不得修改参数后自动重试。

### 5.2 确认并执行

操作员核对目标名称、现场安全条件满足，并对本次机械臂运动明确确认后，使用全新的 task ID 执行。

PC：

```bash
source .shrc_local
export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1
source install/setup.bash
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc execute pick_object \
  --task-id pick-marker-pc-001 \
  --target-name marker
```

310P：

```bash
source .shrc_local
export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1
source install/setup.bash
robot-skill --config-name lekiwi_handeye_realsense_grasp execute pick_object \
  --task-id pick-marker-310p-001 \
  --target-name marker
```

> 必须先 `source .shrc_local` 与 `source install/setup.bash`。`skill_catalog` 位于
> `build/skill_catalog/`，只有 source 了 `install/setup.bash` 才会进 `sys.path`；在裸
> `(venv)` 里直接跑 `robot-skill` 会报 `ModuleNotFoundError: No module named
> 'skill_catalog'`。客户端的 `ROS_DOMAIN_ID`、`ROS_LOCALHOST_ONLY` 必须与 pipeline 终端一致。

`--target-name` 必须与实际目标一致；上例抓取 `marker`，抓取香蕉时改为 `banana`，并同步使用易识别的
task ID。

默认使用 Gateway 当前 task budget。只有需要更短预算时才添加 `--timeout-sec SEC`；不要超过 Gateway budget。
每次执行必须使用新的 task ID。失败、超时或未收到 terminal result 时不得自动重试。

需要取消当前任务时，在另一终端使用相同 task ID。PC：

```bash
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc cancel --task-id pick-marker-pc-001
```

310P：

```bash
robot-skill --config-name lekiwi_handeye_realsense_grasp cancel --task-id pick-marker-310p-001
```

取消请求成功不等于机械臂已经停止，必须等待该任务进入 terminal 状态。

## 6. 判断结果与检查证据

### 6.1 成功标准

必须同时看到持续 feedback 和唯一 terminal result：

```text
{"event":"feedback", ...}
{"event":"result","data":{"success":true,...},...}
PIPELINE_TIMING stage=graspgen_request ...
```

只有 terminal result 能证明任务已经收敛。超时、连接中断或没有 terminal result 时，任务状态视为未知，
不得自动重试。

### 6.2 自动抓后验证

正式 executor 会自动完成三个验证阶段：

1. `close`：闭爪后、抬升前确认夹持；
2. `transport`：抓取验证成功后移动到固定 `place_container` 关节位置。

`STATUS_FAILED(0)` 或 `STATUS_UNCERTAIN(2)` 在 `verification: required` 策略下都会使任务失败。
executor 会按 robot config 执行保守恢复，不会把恢复完成视为抓取成功。

需要独立读取当前夹持状态时，可调用：

```bash
ros2 service call /grasp_verifier/verify_grasp ibrobot_msgs/srv/VerifyGrasp \
  "{task_id: 'pick_001', text_prompt: 'banana', expected_target_width_m: 0.035, post_grasp_wait_s: 0.2}"
```

结果含义：

- `STATUS_SUCCESS(1)`：融合证据支持抓取成功；
- `STATUS_FAILED(0)`：融合证据支持抓取失败；
- `STATUS_UNCERTAIN(2)`：证据不足，应重新观察并人工决策。

重点检查 `gripper_position`、`gripper_current_abs_a` 和 `wrist_visibility`。close-to-transport 阶段不会重新分割目标，
因为夹爪靠近目标后会明显遮挡腕部相机。

### 6.3 调试证据

当 `planner.debug_output_mode` 为 `diagnostic` 或 `full` 时，重点查看：

| 文件 | 用途 |
|---|---|
| `prepared_candidate_ranking.json` | 候选排序、固定指间隙和各评分项 |
| `pick_pose_diagnostics.json` | 实际位姿误差和接触点 residual |
| `pick_frame_diagnostics.json` | capture-time TF 时间戳和回退模式 |
| `grasp_verification.json` | close 验证证据 |

需要点云和源候选时，临时将 `planner.debug_output_mode` 改为 `full` 并重启 pipeline。

抓取成功后的放置流程见 [so101_place_pipeline_commands.md](so101_place_pipeline_commands.md)。放置必须通过
Capability Gateway 的 `place_in_container` 技能执行。

## 7. 快速排障

| 现象 | 检查与处理 |
|---|---|
| `GroundingDetect service not available` | 执行 `ros2 service list \| grep /perception/grasp`，检查感知 model host 日志 |
| `GraspGen service is not available` | 执行 `ros2 service list \| grep /grasp_planner/plan_grasp`，检查模型 bundle 和 planner 日志 |
| `No synchronized depth/CameraInfo` | 确认三路 `/camera/wrist/...` topic 均有发布者，清理重复相机或 relay 节点 |
| `GraspGen returned zero candidates` | 使用 `debug_output_mode: full` 检查 mask/depth，再谨慎调整 planner 阈值 |
| `Motion timed out after 60.0s` | 检查控制器和 `/joint_states`，确认状态后重启 pipeline |
| `STATUS_ABORTED` 但控制器已完成 | 检查 `/move_action`；若存在多个 `/move_group`，清理残留节点后重启 |
| 抓取位置固定偏差 | 运行手眼预检，核对 wrist transform、平台配置和 IK/FK residual |
| 固定指先碰目标 | 检查 `FK_FIXED_FINGER_BASE_SIDE_*`、候选排序和 fixed-finger 包络，不要只增大 margin |

关键日志的定位含义：

- `PIPELINE_TIMING stage=graspgen_request`：完整 `PlanGrasp` 请求耗时；
- `stage=candidate_geometry_ranking`：几何门禁和初步排序耗时；
- `stage=candidate_ik_fk`：并行 IK/FK 候选准备耗时；
- `IK worker verification passed`：主 MoveIt 与 worker 结果一致；
- `pick candidate preparation failed ... code=...`：候选在执行前被安全或几何门禁拒绝；
- `contact realign phase=...`：接触点补偿 residual；
- `pick candidate failed ... retryable=...`：物理执行失败，只有 `retryable=true` 才允许 executor 换候选；
- `grasp verification phase=...`：抓后验证证据；
- `pipeline_timings_json`：返回 Gateway 的候选数、尝试数、验证状态和分阶段计时。

## 8. 调参原则

所有影响候选、安全门禁、IK/FK、速度、恢复和验证的参数，都只在对应 robot YAML 的
`robot.grasp_execution` 中修改。修改后重启正式 executor。客户端不得维护第二套运行参数。

优先检查以下配置块：

```yaml
planner:
  grasp_threshold: 0.20
  debug_output_mode: diagnostic  # PC 当前值；310P 当前为 none

candidate_selection:
  max_candidates: 80

ik:
  worker_count: 4

contact_compensation:
  enabled: true
  xy_tolerance_m: 0.003
  max_iterations: 6
  max_correction_m: 0.030
  max_z_error_m: 0.020
```

调参顺序：

1. 先用诊断输出区分检测、深度、标定、几何门禁和 IK/FK 问题；
2. 再修改对应 SSOT 参数并重启 pipeline；
3. 用新 task ID 做一次受监督验证；
4. 记录 terminal result 和证据，禁止根据单次结果写入全局偏移。

补充约束：

- 接触补偿只调整 base-X/Y，不补偿 Z；超出修正量或 Z 误差上限的候选会被拒绝；
- `max_candidates` 是送入 IK/FK 的预算，廉价几何门禁会先作用于全部候选；
- `worker_count` 只并行候选准备，最终运动仍由主 MoveIt 串行执行；
- 判断物体是否进入夹口时优先看 `volume_xyz`，不要只看可见表面的 `surface_xyz`；
- `fixed_finger_contact_ee.z` 是夹爪坐标系内的接触深度，不是 base-Z 下压量；
- 无运动排障使用第 3.2 节服务，不要绕过 Gateway 直连 delegated action。

更完整的候选队列与评分说明见
[so101_grasp_dynamic_ik_queue_optimization.md](so101_grasp_dynamic_ik_queue_optimization.md)。

## 接口速查

| 类型 | 名称 |
|---|---|
| 腕部 RGB | `/camera/wrist/image_raw` |
| 对齐深度 | `/camera/wrist/aligned_depth_to_color/image_raw` |
| CameraInfo | `/camera/wrist/aligned_depth_to_color/camera_info` |
| 检测服务 | `/perception/grasp/grounding_detect` |
| 抓取规划服务 | `/grasp_planner/plan_grasp` |
| 抓取验证服务 | `/grasp_verifier/verify_grasp` |
| 内部执行 action | `/manipulation/execute_pick` |
| 外部抓取技能 | `pick_object` |
| 外部放置技能 | `place_in_container` |

310P 的感知分割使用独立 `/perception/grasp/segment_detections` 服务；PC 的 CUDA deployment 在
`grounding_detect` 内联完成分割。两种平台都只发布 RGB、对齐深度和 CameraInfo，GraspGen 直接从对齐深度
构造目标点云，不需要额外的 ROS `PointCloud2` relay。

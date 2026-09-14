# manipulation_execution

`manipulation_execution` 是抓取、放置和 HRI 模拟执行的闭环执行层。它把一次 `PickObject` action 请求编排为
GraspGen 规划、SO101 目标夹爪几何筛选、IK/FK 接触点补偿、安全 primitive 执行和抓后验证；
`ImitateHumanMotion` 则提供 delegated 的人体感知生命周期：在任务窗口内以独立订阅者方式接收配置的 RGB 图像，使用低频 YOLOX 刷新任务内 bbox 缓存，再让 PEAR 对持续更新的最新 RGB 帧独立推理，并校验、采样打印结果。该 executor 只消费视频，不产生任何运动输出。

### HRI RGB 输入与人体感知链

HRI executor 复用统一 pipeline 和 `qos_profile_sensor_data` 订阅配置，不启动或管理 RealSense 设备。
输入 topic 和两个模型服务端点都是 ROS 参数，由 launch 从 `embodied.imitate_human_motion` 注入：

| 参数 | LeKiwi 取值 | 未配置时的回退 |
| --- | --- | --- |
| `rgb_topic` | `/camera/wrist/image_raw` | `embodied.perception.scene_sources.wrist_camera_topic` |
| `yolox_detect_service` | `/perception/hri/yolox_detect` | 同名默认值 |
| `pear_parameters_service` | `/perception/hri/pear_parameters` | 同名默认值 |
| `person_confidence_threshold` | `0.30` | 同名默认值；必须是 `[0, 1]` 内有限数 |
| `yolox_refresh_interval_sec` | `0.25` | `0.25`；必须是正的有限数 |

`rgb_topic` 必须写 bringup remap **之后**的名字。`/camera/wrist_camera/color/image_raw` 是 RealSense
驱动自己的名字，remap 之后在运行图上并不存在。

感知链把 `person_confidence_threshold` 同时传给 YOLOX，并在响应上防御性地过滤非 `person`、
非有限置信度和低于阈值的框；有候选时取置信度最高的一个。有合格 person 时，executor 缓存该框并让 PEAR
使用最新整帧；YOLOX 成功但没有合格 person 时，executor 构造图像中心方框，
边长为 `min(width, height) / 1.25`，再用恰好一个框调用 PEAR。任务首次完成 YOLOX 成功响应前不调用
PEAR；之后的 YOLOX 刷新失败继续使用任务内最近一次成功结果。服务不可用、RPC 超时、模型失败和无效图像
不会创建初始 fallback。像素处理仍全部在 perception adapter 内，本节点不含 `cv2`，也不使用 PEAR 输出驱动关节。

任务 deadline 是相机计数和视觉推理的硬边界：边界后的帧不计数、不入队，worker 不提交过期
请求，边界后返回的响应也不计入任务。每次调用分别跟踪 RPC deadline 和任务 deadline；只有
`rpc_timeout_sec` 先到才增加 `vision_failed`，任务预算先到只取消或忽略该工作。reset/recovery
可以继续使用既有 cleanup budget，但不能延长 RGB 统计窗口。

result 的 message 继续附带 `rgb_input` JSON；feedback 不附带该 JSON。JSON 还包含 `yolox_calls` 与
`pear_calls`，用于观察低频检测和持续 PEAR 推理。`vision_ok` 是 detected 与 fallback
两条路径的有效 PEAR 响应总数，`vision_detected` 和 `vision_fallback` 分别记录来源，并始终满足
`vision_ok == vision_detected + vision_fallback`；`vision_failed` 只记录真实服务、模型、RPC、图像或
输出校验失败。`vision_sample` 增加 `source=yolox` 或 `source=center_fallback`，便于区分真实检测框和
中心 fallback。

`PickObject` 是 delegated action：goal 必须携带 `dispatch_binding`（`DispatchBinding`，含同一 root 的
共享 `task_budget` 和 exact registry identity）以及 `expected_executor`（`DelegatedExecutorIdentity`）；
result 必须返回 `actual_executor`。server 在 goal acceptance 时比对自身实际 identity 与
`expected_executor`，不匹配时直接 reject，且不得仅凭 endpoint 名判断。

`PlaceObject` 使用相同 delegated 契约。`place_executor_node` 只接受 Gateway
构造的 binding，并通过 `/embodied/execute_primitive` 复用受保护的
`move_to_joint_positions` 和 `open_gripper`；不存在 direct place action client。
放置先到固定 `place_container`（3 号电机 raw 1500）开爪，再只将 3 号电机移动到 raw 1700
（`verify_joint_position: -0.533825`）做视觉确认，
最后返回 raw 1500。请求中的
`container_name` 只作为运行时容器检测 query，不参与运动规划；`target_name` 和 `container_name` 原样作为
Grounding-DINO 文本输入，颜色等描述词不得丢失。容器与目标存在多个候选时，执行器按置信度选择最高候选
进行二维包含判定，并保留全部候选证据；目标检测始终使用配置的静态夹爪排除 mask。每次请求在不可逆动作前创建版本化证据目录，保存开爪后的新鲜 JointState、RGB、
容器/目标 mask 和终态；使用 `placement_replay` 可在无 ROS、
无真机条件下重放，旧三维容器规划记录没有当前 manifest 时会被标记为 `legacy_incompatible`。

本包也是抓取行为的唯一实现源。Hermes、生产 catalog 和批量自动化都必须通过
`robot-skill execute pick_object` 进入 Capability Gateway。真机 bring-up 另提供
已安装的 `pick_action_client` 这一显式 supervised-direct 客户端：它先读取当前 Gateway
registry，再以 `supervised_direct=true` 发送一次受监督 goal；这不是 Hermes 入口，也不能伪造 delegated
nonce。`/manipulation/execute_pick` 仍是 Gateway 准入后的 delegated action，Hermes 调用方不得自行构造
binding 或 executor identity。

旧的长脚本实现及 legacy-only replay benchmark 已删除。当前监督式客户端只负责 live registry preflight、
goal 发送和反馈收敛，抓取行为仍只在正式 phases 中实现；`test_pick_single_source.py` 会确保旧脚本不被
重新引入，且客户端保持为唯一的 supervised-direct 入口。

## 架构位置

```text
Hermes -> ibrobot-control -> robot-skill
  -> Agent plan / Gateway -> /embodied/execute_skill (pick_object)
  -> skill_library Gateway (admit + canonicalize dispatch_binding, stamp dispatch_nonce)
  -> /manipulation/execute_pick (PickObject, delegated)
  -> pick_executor_node
       -> /grasp_planner/plan_grasp
       -> /compute_ik + /compute_fk
       -> /embodied/execute_primitive
       -> /grasp_verifier/verify_grasp
```

Hermes 只看到一个原子技能。内部运动仍逐步经过 `skill_library` 和 `safety_guard`，本包不直接发布
控制器命令，也不直接调用 MoveIt 原始位姿接口。

抓取 executor 将同一份 `dispatch_binding` 传给每个 `PrimitiveCommand`。`skill_library` 依据
`dispatch_nonce` 和完整 binding 让 pipeline primitive 借用父 skill 的 root lease；调用方不得绕过
Gateway 自行构造 delegated `PickObject` 请求。

`supervised_direct=true` 仅供 `pick_action_client` 真机 bring-up 使用。此模式要求空
`dispatch_nonce`、完整当前 registry identity 和 `task_budget`，并且它派生的每个 primitive 仍由
`skill_library` 做 direct-root admission；Hermes 和 `robot-skill` 始终使用
`supervised_direct=false` 及 Gateway delegated nonce。该客户端把 `--task-id` 作为人工可读前缀，每次运行
都会追加唯一后缀；实际 ID 会显示在 `PICK_ACTION_SEND` 中。只有排查同一请求身份时才使用
`--exact-task-id`，此时重复运行会被 Gateway ledger 拒绝。

`pick_executor_node.py` 只管理 ROS action、service client、TF 和 joint-state 生命周期。抓取行为按职责拆分到
`phases/flow.py`、`planning.py`、`preparation.py` 和 `execution.py`；共享状态模型与纯转换函数分别位于
`pick_executor_models.py` 和 `pick_executor_helpers.py`。`pipeline_worker.py` 只提供与机器人无关的动态 worker
队列；SO101 joint5 分支、归一化、闭合轴修正和 IK 重试集中在 `so101_kinematics_guard.py`，
不混入通用 pipeline 工具。阶段 mixin 保留原有方法入口，便于隔离测试且不改变 action 行为。

## 公开接口

| 接口 | 类型 | 说明 |
| --- | --- | --- |
| `/embodied/execute_skill` | `ibrobot_msgs/action/SkillCommand` | 对外技能入口；放置使用 `place_in_container`、`target_name` 和 `container_name` |
| `/manipulation/execute_pick` | `ibrobot_msgs/action/PickObject` | Gateway 内部 delegated 抓取执行，不是外部入口 |
| `/manipulation/execute_place` | `ibrobot_msgs/action/PlaceObject` | Gateway 内部固定位置释放与视觉确认，不是外部入口 |
| `/hri/imitate_human_motion` | `ibrobot_msgs/action/ImitateHumanMotion` | Gateway 内部 HRI 模拟执行，不是外部入口 |

goal 携带 `dispatch_binding`（task_id/root_task_id、共享 `task_budget`、exact registry identity）、
`target_query`、`timeout_sec`、`expected_executor` 和仅供 supervised bring-up 的 `supervised_direct`。
`target_query` 是 Grounded-SAM2/GraspGen 的文本查询
（例如 `banana`），不是 `named_targets` 中的静态配置键。result 携带 `actual_executor`、`attempts`、
`verification_status`、`completed_phases` 和 `debug_output_dir`。

`PickObject.mode` 支持 `MODE_EXECUTE`、`MODE_PLAN_ONLY` 和 `MODE_OBSERVE_ONLY`。三个模式共享同一
观测、规划和配置入口；plan-only 会完成正式候选筛选与 IK/FK 准备但不执行抓取。`release_after_success`
可让 executor 在验证成功并运输到 place container 后开爪。该字段属于内部 delegated 契约；当前 v1
catalog 的 `pick_object` 只授权 `MODE_EXECUTE`，且不允许调用方请求自动释放。

目标字段 `target_query` 是 Grounded-SAM2/GraspGen 的文本查询，例如 `banana`，不是
`named_targets` 中的静态配置键。

放置反馈阶段包括 `preflight`、`move_to_place`、`release`、`move_to_verify`、`verify_place` 和
`return_to_place`；抓取反馈阶段包括 `preflight`、`observe`、`planning`、`selecting`、`approach`、`descend`、
`close`、`verify_close`、`transport` 和 `release`。

只有最终抓取验证成功时 action 才返回 `success=true`。动作执行完成但验证失败或不确定分别返回
`GRASP_VERIFICATION_FAILED` 或 `GRASP_UNCERTAIN`。

## 配置

HRI 执行由 `robot.embodied.imitate_human_motion` 启用，`rgb_topic` 与两个模型服务端点见上文表格。关节顺序来自 `robot.joints.arm`，
prepare 起点来自 `robot.ros2_control.reset_positions`，限位来自
`robot.teleoperation.safety.joint_limits`；这些值由 `embodied_bringup` 注入，执行器不维护第二份机器人配置。

所有抓取运行参数来自 robot YAML 的 `robot.grasp_execution`。关键配置包括：

- GraspGen、验证器、IK/FK 服务名。
- 相机 topic 和 base/EE frame。
- source gripper 到目标夹爪的几何适配。
- 基于检测目标体积质心的 contact-distance 排序，以及 confidence/top-down 权重。
- approach、速度、候选数量和物理执行重试次数；IK/FK 准备失败不消耗物理重试次数。
- `ik.worker_count` 个隔离 MoveIt worker 从同一个动态队列准备候选；所有 worker 使用同一份
  `/joint_states` seed，空闲 worker 立即领取下一个候选，结果仍按原候选顺序合并，并在每次抓取前校验
  主 MoveIt 与全部 worker 的 IK 解一致。
- `candidate_selection.selection_attempts` 控制整批候选无安全 IK 解或规划暂时失败时重新获取新帧并规划；
  `TARGET_NOT_VISIBLE`（目标未检测到、置信度不足或没有生成候选）和
  `TARGET_OUTSIDE_WORKSPACE`（目标可见但所有候选均超出工作范围）会立即报错，不继续规划；不可恢复的
  配置、安全门禁和 worker 一致性错误同样不会重试。
- `max_execution_attempts` 控制完整物理抓取次数。close 验证失败后，执行器先按安全恢复路径
  回撤、开爪并返回 `observe_pose`，然后从新观测帧重新规划，不复用目标可能已移动后的旧候选。
  执行中若出现可恢复的 IK/FK、joint5 分支、接触补偿、工作空间、固定指最终检查、桌面碰撞、TF 或
  IK/FK RPC 错误（包括 `IK_ORIENTATION_REJECTED`），同样先返回 `observe_pose`、等待观测稳定，再重新规划并
  消耗下一次执行尝试；即使重试额度已经耗尽，也会先恢复到观测位再返回错误。候选准备阶段的同名 IK/FK
  失败仍由 `candidate_selection.selection_attempts` 处理，不重复消耗物理执行重试次数。
- SO101 tabletop 前置检查由正式 pipeline 的公共实现完成；相机平面变换到 base 后先将法向规范为朝向
  base +Z 的安全半空间，再使用缓存的 mesh 凸包顶点和批量 NumPy 变换处理全部候选。固定姿态平移 sweep
  只计算两个端点，临界阈值附近再回退到单候选精确检查。
- SO101 实际夹爪 STL 的 approach-to-grasp 桌面间隙，以及 joint5 分支修正后的 FK 接近轴/闭合轴硬校验。
- 候选 FK 残差重排、approach 位接触点 realign 和进入 descend 前的最终 IK/FK 接触点补偿阈值。
- IK/FK 准备完成后按候选规划姿态的固定爪前缘包络、目标体积质心距离、FK 接触点 XY/Z 质量和
  置信度做软重排；固定爪包络偏好不会单独硬拒绝候选。
- 非对称单动夹爪先按规划姿态检查固定爪是否位于朝机器人底座的一侧，再按最终 IK 解的 FK 姿态
  复检；最终姿态镜像或缺少目标宽度范围时，在产生运动前拒绝该候选并继续准备其他候选。
- 固定指 robust gap 使用最终 IK/FK 预测接触残差在 approach/descend 前完成硬门禁；下降成功后进入
  commit-to-grasp 状态，`close_gripper` 是下降成功后的第一条动作。低位实测改在闭爪后以 best-effort
  方式记录；诊断异常不影响闭爪和后续验证，也不再回到安全过渡位或切换候选。
- commanded/actual 位姿和接触点 residual 诊断；有 planner 输出目录时写入 `pick_pose_diagnostics.json`。
- 抓取候选使用其 depth capture timestamp 查询 TF，不使用推理完成后的 latest TF；capture/latest/hand-eye
  变换写入 `pick_frame_diagnostics.json`。
- 最终软排序明细写入 `prepared_candidate_ranking.json`，包含固定爪实际/目标间隙、移动爪余量、
  质心距离和综合分数。
- 抓后验证策略。
- `PickObject.Result.pipeline_timings_json` 返回 `phase_preflight`、`phase_observe`、`phase_planning`、
  `phase_selecting`、`phase_open`、`phase_approach`、`phase_descend`、`phase_close`、
  `phase_verify_close`、`phase_transport` 和 `phase_release` 等正式反馈阶段墙钟；`subphase_contact_realign` 是包含在 approach 阶段内的
  嵌套聚合耗时，`subphase_recovery` 是失败安全恢复的嵌套聚合耗时，二者不能与 phase 字段直接相加。

`pick_object` skill 的 catalog manifest（`skill_catalog/config/skills/pick_object/`）负责把技能入口
委托给本执行器，implementation 声明 `kind: delegated_executor`、`executor: grasp_pipeline`、
`required_args: [target_name]`，Gateway 据此构造带 `expected_executor` 的 `PickObject` goal：

```yaml
# skill_catalog/config/skills/pick_object/implementations/<robot>.yaml
kind: delegated_executor
executor: grasp_pipeline
required_args:
- target_name
timeout_sec: 240.0
```

`grasp_pipeline` 是模型驱动 delegated executor。`robot.grasp_execution.model_bundle_path` 与
`model_deployment` 必须指向可由 `inference_manifest` 严格校验的 bundle；Gateway 与本节点分别加载同一
manifest，并在 `expected_executor` / `actual_executor` 中比较 deployment name、fingerprint 和 bundle digest。
模型 identity 缺失、manifest 不可用或两端不一致时 fail closed，不接受抓取 goal。

## 安全边界

- `ImitateHumanMotion` 只接受 Gateway delegated binding 和匹配的 executor identity；goal timeout 必须为正且
  不超过共享 task budget 的剩余时间。
- Mock prepare/play/reset 都通过 `/embodied/execute_primitive`；若 primitive 取消或终态无法确认，runtime
  返回 `CANCEL_CLEANUP_TIMEOUT`、保持 pose state 为 unknown，并且不继续发送 reset，避免重叠运动。
- 正常、已确认失败或已确认取消后使用 `move_to_named_pose(home)` 恢复；下一次任务仍从 prepare 开始。

- goal acceptance 先比对 `expected_executor` 与本节点实际 `DelegatedExecutorIdentity`，不匹配直接
  reject；再校验 `dispatch_binding`：`schema_version=1`、非空 `task_id`/`root_task_id`、完整期望
  registry identity（epoch/generation/digest）、`task_budget.schema_version=1`、`dispatch_nonce` 非空。
  随后校验 budget：`started_at >= 0`、`deadline > now`、`timeout_sec > 0` 且 `timeout_sec <= deadline - now`。
  任一不满足或已有 active goal 时 reject，不产生运动。
- 执行预算消费共享绝对 `task_budget`：`_execute_pick` 用
  `min(goal.timeout_sec, deadline - now)` 作为实际 deadline；若预算在执行前已过期，立即返回
  `TASK_TIMEOUT`（`shared task budget expired before pick execution`）并 abort，不进入任何抓取阶段。
  预算在执行中过期由各阶段 deadline 自然截断。
- 候选选定后，approach、内部安全下降过渡点和最终下降都沿用同一 joint5
  分支，并通过 `move_to_configuration` primitive 执行精确 IK 解；安全层检查完整关节顺序和限位。
- 抓取前检查所有必需服务，缺失时不产生运动。
- 最终 IK 解的 FK 固定爪朝向复检失败时，不执行该候选的 approach/descend/close。
- 固定指间隙不足的候选只在批量候选准备阶段拒绝。在 approach 完成在线接触补偿和最终安全检查后，
  执行器进入 `descend`，依次经过内部安全过渡点、最终下降和 `close_gripper`，不再二次 realign、复核或
  切换候选。内部过渡点仍沿用 `pregrasp` 配置和诊断标签，但不再作为反馈状态。
  闭爪后的 TF/pose/robust-gap 检查同样是 best-effort，不能中断 commit-to-grasp。
- position-only IK 的 joint5 超出执行门限时，将 seed 翻转 `±π` 以交换固定指/活动指所在侧；最终 FK 接近轴、
  180° 对称闭合轴直线或固定指内侧检查失败时拒绝该候选，不再把 joint5 强制归零或停在 `±2.0` 边界。
- close 验证失败时闭爪撤回后再打开；抓取验证成功后直接移动到配置的 `place_container` 关节位置
  打开并返回观察位。验证失败且仍有物理尝试预算时，从该观测位重新检测和规划；如果新观测返回
  `TARGET_NOT_VISIBLE` 或 `TARGET_OUTSIDE_WORKSPACE`，立即终止而不再规划。监督式客户端通过同一个 action
  获得完全相同的恢复行为。
- 桌面平面或 SO101 mesh 不可用时，目标夹爪 tabletop filter fail closed。
- 同一时间只接受一个抓取 goal，并将上游取消请求传给当前 primitive。
- 隔离 worker pool 同时承担候选准备和执行期接触补偿、分支锁定所需的 IK/FK；单个 worker RPC 超时后会
  移除悬挂请求、隔离该 worker 并切换到其他已验证 worker。主 MoveIt 只负责真实规划和运动，避免不可取消的
  `/compute_ik` 请求阻塞后续恢复运动。
- `MoveToConfiguration` 当前是同步 ROS service；取消会停止本地等待，但服务端运动不具备 action 级硬取消。
  因此该 primitive 返回失败或超时后，`PickObject` 会 fail closed 并终止整个 goal，不会自动切换到下一候选。

## 测试

监督式调用必须走 Capability Gateway。先做只读状态和安全校验，再使用全新 task ID 执行：

```bash
source .shrc_local && source install/setup.bash && \
robot-skill --config-name lekiwi_handeye_realsense_grasp status

source .shrc_local && source install/setup.bash && \
robot-skill --config-name lekiwi_handeye_realsense_grasp validate pick_object \
  --target-name banana --timeout-sec 240

source .shrc_local && source install/setup.bash && \
robot-skill --config-name lekiwi_handeye_realsense_grasp execute pick_object \
  --task-id pick-banana-001 --target-name banana --timeout-sec 240
```

每次执行必须使用新的 task ID。失败、timeout 或停止状态未知时不得自动重试；需取消时使用
`robot-skill cancel --task-id <ID>` 并等待 terminal。当前公开 catalog 没有暴露 plan-only、observe-only
或 post-success release；如需这些能力，应新增版本化 catalog 参数和 `SkillCommand` 契约，而不是直接调用
delegated `PickObject`。

```bash
source .shrc_local && source install/setup.bash && \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q --import-mode=importlib src/manipulation_execution/test
```

通用 worker 调度由 `test_pipeline_worker.py` 覆盖；SO101 joint5 运动学约束由
`test_so101_kinematics_guard.py` 覆盖。phases 测试继续验证配置解析、错误码和执行编排。

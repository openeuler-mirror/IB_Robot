# task_dispatch — 任务级执行框架

## 概述

`task_dispatch` 是 `action_dispatch` 的任务级对应组件。

| 维度 | action_dispatch | task_dispatch |
|------|----------------|---------------|
| 控制粒度 | 关节级流式（100Hz） | 任务级序列（事件驱动） |
| 适用场景 | ACT/VLA  imitation learning | MoveIt 规划器（VoxPoser、visual_grasp 等） |
| 执行方式 | topic 流式推送 | action 顺序执行 |

```
┌──────────────────────┐     ┌──────────────────────┐
│   action_dispatch    │     │   task_dispatch       │
│                      │     │                       │
│  ACT/VLA → 100Hz     │     │  Planner → TaskPlan   │
│  stream → Position   │     │  steps → MoveIt       │
│  Controller          │     │  Gateway → Trajectory  │
│                      │     │  Controller            │
└──────────────────────┘     └──────────────────────┘
```

## 架构

```
visual_grasp_planner ─┐
                      ├─→ ExecuteTaskPlan action ─→ TaskExecutorNode
VoxPoser planner ─────┘                               │
                                          ┌───────────┼────────────┐
                                          │           │            │
                                    MOVE_TO_POSE   GRIPPER       WAIT
                                          │           │            │
                                    MoveToPose   FollowJoint    sleep()
                                    service      Trajectory
                                    (同步调用)    action
                                          │           │
                                    moveit_gateway  gripper_trajectory
                                          │        _controller
                                    IK + plan + exec   │
                                          │           │
                                          └─────┬─────┘
                                                │
                                          ros2_control
                                           hardware
```

**TaskExecutorNode** 提供以下执行能力：

| 步骤类型 | 常量值 | 说明 | 底层调用 |
|---------|--------|------|---------|
| `MOVE_TO_POSE` | `0` | 移动机械臂到目标位姿 | `/moveit_gateway/move_to_pose` service |
| `GRIPPER` | `1` | 控制夹爪开合 | `/{gripper_controller}/follow_joint_trajectory` action |
| `WAIT` | `2` | 等待指定时长 | 可中断的 `time.sleep()` |

## ROS 接口

### Action Server

| 接口 | 类型 | 说明 |
|------|------|------|
| `~/execute_task_plan` | `ibrobot_msgs/action/ExecuteTaskPlan` | 主入口，规划器通过此接口发送任务计划 |

### 依赖的上游接口

| 接口 | 类型 | 说明 |
|------|------|------|
| `/moveit_gateway/move_to_pose` | `ibrobot_msgs/srv/MoveToPose` (Service) | 机械臂运动，由 moveit_gateway 提供 |
| `/{gripper_controller}/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` (Action) | 夹爪控制，由 ros2_control 提供 |
| `/joint_states` | `sensor_msgs/msg/JointState` (Topic) | 关节状态反馈 |

`moveit_planning.executor` 可配置 `skip_redundant_gripper_open`、`gripper_open_position`、
`gripper_position_tolerance` 和 `joint_state_max_age_s`。启用后，仅当最近收到的关节反馈足够新、包含夹爪关节，
且当前位置已在打开目标容差内时，执行器才跳过打开轨迹；关闭命令以及陈旧或缺失反馈仍按原路径执行。

### 消息定义

**TaskStep.msg** — 任务步骤：

```
uint8 MOVE_TO_POSE = 0    # 移动到目标位姿
uint8 GRIPPER      = 1    # 夹爪控制
uint8 WAIT         = 2    # 等待

uint8   type                # 步骤类型
string  label               # 可读标签（用于日志和 feedback）
geometry_msgs/Pose target_pose   # MOVE_TO_POSE: 目标位姿
float64 velocity_scaling         # MOVE_TO_POSE: 速度比例 (0.0~1.0, 0.0=默认)
float64 gripper_position         # GRIPPER: 夹爪位置 (0.0=闭合, 1.0=全开)
float64 wait_duration_s          # WAIT: 等待时长（秒）
```

**ExecuteTaskPlan.action**：

```
# Goal
ibrobot_msgs/TaskStep[] steps    # 任务步骤序列
string task_id                    # 任务唯一标识
string task_description           # 任务描述
---
# Result
bool   success
string message
int32  steps_completed           # 已完成的步骤数
float64 total_duration_s         # 总耗时
---
# Feedback
int32   current_step             # 当前步骤索引 (0-indexed)
int32   total_steps
string  current_label
string  status                   # "executing" | "completed" | "failed"
float32 progress                 # 0.0 ~ 1.0
```

## 快速开始

### 1. 编译

```bash
colcon build --packages-select ibrobot_msgs robot_config so101_motion task_dispatch
source install/setup.bash
```

### 2. 启动

Task executor 在 `control_mode` 包含 "moveit" 时自动启动，无需额外配置：

```bash
export ROS_DOMAIN_ID=<你的ID>

# 仿真模式
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=true \
    control_mode:=moveit_planning

# 真机模式
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false \
    control_mode:=moveit_planning
```

启动后检查节点是否就绪：

```bash
ros2 node list | grep task_executor
# 输出: /task_executor

ros2 action list | grep execute_task_plan
# 输出: /task_executor/execute_task_plan
```

### 3. 命令行验证

**测试夹爪开合**（不需要 MoveIt 运动规划，纯 gripper + wait）：

```bash
ros2 action send_goal /task_executor/execute_task_plan ibrobot_msgs/action/ExecuteTaskPlan "{steps: [{type: 2, label: 'wait_1s', wait_duration_s: 1.0}, {type: 1, label: 'open_gripper', gripper_position: 1.0}, {type: 2, label: 'wait_0.5s', wait_duration_s: 0.5}, {type: 1, label: 'close_gripper', gripper_position: 0.0}], task_id: 'test_001', task_description: 'gripper cycle test'}" --feedback
```

以上裸 Action 仅用于隔离的底层开发调试；正常 Hermes/Agent 或真机动作必须经过
Capability Gateway、Safety Guard 和控制模式准入。

预期输出：
```
Result:
    success: true
    message: All steps completed successfully
    steps_completed: 4
    total_duration_s: 3.6s
Goal finished with status: SUCCEEDED
```

**测试完整抓取序列**（含 MOVE_TO_POSE，需要 moveit_gateway 就绪）：

```bash
ros2 action send_goal /task_executor/execute_task_plan ibrobot_msgs/action/ExecuteTaskPlan "{steps: [{type: 1, label: 'open_gripper', gripper_position: 1.0}, {type: 0, label: 'move_above', target_pose: {position: {x: 0.2, y: 0.0, z: 0.3}, orientation: {w: 1.0}}, velocity_scaling: 0.3}, {type: 0, label: 'move_to_grasp', target_pose: {position: {x: 0.2, y: 0.0, z: 0.1}, orientation: {w: 1.0}}, velocity_scaling: 0.2}, {type: 1, label: 'close_gripper', gripper_position: 0.0}, {type: 2, label: 'hold', wait_duration_s: 1.0}, {type: 0, label: 'lift', target_pose: {position: {x: 0.2, y: 0.0, z: 0.4}, orientation: {w: 1.0}}, velocity_scaling: 0.3}], task_id: 'pick_001', task_description: 'pick and place'}" --feedback
```

## Python API 使用

### 基础用法：发送任务计划

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose
from ibrobot_msgs.action import ExecuteTaskPlan
from ibrobot_msgs.msg import TaskStep


class MyPlanner(Node):
    def __init__(self):
        super().__init__('my_planner')
        self._client = ActionClient(
            self, ExecuteTaskPlan, '/task_executor/execute_task_plan'
        )

    def send_pick_task(self):
        self._client.wait_for_server()

        goal = ExecuteTaskPlan.Goal()
        goal.task_id = 'pick_001'
        goal.task_description = 'pick up object'

        # Step 1: 打开夹爪
        step1 = TaskStep()
        step1.type = TaskStep.GRIPPER
        step1.label = 'open'
        step1.gripper_position = 1.0

        # Step 2: 移动到目标位姿上方
        step2 = TaskStep()
        step2.type = TaskStep.MOVE_TO_POSE
        step2.label = 'approach'
        step2.target_pose = Pose()
        step2.target_pose.position.x = 0.2
        step2.target_pose.position.y = 0.0
        step2.target_pose.position.z = 0.3
        step2.target_pose.orientation.w = 1.0
        step2.velocity_scaling = 0.3

        # Step 3: 闭合夹爪
        step3 = TaskStep()
        step3.type = TaskStep.GRIPPER
        step3.label = 'grasp'
        step3.gripper_position = 0.0

        # Step 4: 等待稳定
        step4 = TaskStep()
        step4.type = TaskStep.WAIT
        step4.label = 'settle'
        step4.wait_duration_s = 0.5

        goal.steps = [step1, step2, step3, step4]

        future = self._client.send_goal_async(
            goal, feedback_callback=self._feedback_cb
        )
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f'[{fb.current_step + 1}/{fb.total_steps}] '
            f'{fb.current_label} — {fb.status} ({fb.progress:.0%})'
        )

    def _result_cb(self, future):
        result = future.result().result
        self.get_logger().info(
            f'完成: success={result.success}, '
            f'steps={result.steps_completed}, '
            f'time={result.total_duration_s:.1f}s'
        )
```

## 配置

所有参数从 robot_config YAML（SSOT）自动加载，无需手动配置。
`robot_config_path` 使用 `robot_config` 的公共 loader 解析，包含 `base_config` 的配置继承和
严格的容器类型校验；task_dispatch 不维护独立的 YAML 合并规则。

**夹爪关节名** 从 `joints.gripper` 解析：

```yaml
joints:
  gripper:
    - "6"    # 夹爪关节名（SO-101 使用数字 ID）
```

**夹爪控制器** 从 `control_modes` 中匹配含 `gripper` + `trajectory` 的控制器：

```yaml
control_modes:
  moveit_planning:
    controllers:
      - joint_state_broadcaster
      - arm_trajectory_controller
      - gripper_trajectory_controller  # → /gripper_trajectory_controller/follow_joint_trajectory
```

**自动启动条件**（`launch_builders/task_execution.py`）：

满足以下任一条件时，task_executor 会自动加入 launch：
- `control_mode` 包含 `moveit`
- `control_mode` 包含 `visual_grasp`
- `control_mode` 包含 `voxposer`
- `executor.task_dispatch: true`

## 为现有规划器集成

### visual_grasp_planner

移除原有的 `executor.py`，改为：
1. 计算抓取位姿（YOLO + SAM + PCA + 标定）
2. 构建 `TaskStep[]` 序列
3. 发送 `ExecuteTaskPlan` action goal

### VoxPoser

1. LLM 生成 waypoints
2. 转换为 `TaskStep[]` 序列
3. 发送 `ExecuteTaskPlan` action goal

### 自定义规划器

任何新规划器遵循统一模式：
1. **感知/推理** → 产出目标位姿
2. **构建** `TaskStep[]` 序列
3. **发送** `ExecuteTaskPlan` action goal
4. **监听** feedback 获取执行进度

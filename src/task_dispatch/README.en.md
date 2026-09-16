# task_dispatch — Task-Level Execution Framework

## Overview

`task_dispatch` is the task-level counterpart to `action_dispatch`.

| Dimension | action_dispatch | task_dispatch |
|-----------|----------------|---------------|
| Granularity | Joint-level streaming (100Hz) | Task-level sequencing (event-driven) |
| Use case | ACT/VLA imitation learning | MoveIt planners (VoxPoser, visual_grasp, etc.) |
| Execution | Topic streaming | Sequential action execution |

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

## Architecture

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
                                    (sync call)   action
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

**TaskExecutorNode** provides the following execution capabilities:

| Step Type | Constant | Description | Underlying Call |
|-----------|----------|-------------|-----------------|
| `MOVE_TO_POSE` | `0` | Move arm to target pose | `/moveit_gateway/move_to_pose` service |
| `GRIPPER` | `1` | Control gripper open/close | `/{gripper_controller}/follow_joint_trajectory` action |
| `WAIT` | `2` | Wait for specified duration | Cancellable `time.sleep()` |

## ROS Interfaces

### Action Server

| Interface | Type | Description |
|-----------|------|-------------|
| `~/execute_task_plan` | `ibrobot_msgs/action/ExecuteTaskPlan` | Main entry point for planners |

### Upstream Dependencies

| Interface | Type | Description |
|-----------|------|-------------|
| `/moveit_gateway/move_to_pose` | `ibrobot_msgs/srv/MoveToPose` (Service) | Arm motion, provided by moveit_gateway |
| `/{gripper_controller}/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` (Action) | Gripper control, provided by ros2_control |
| `/joint_states` | `sensor_msgs/msg/JointState` (Topic) | Joint state feedback |

`moveit_planning.executor` may define `skip_redundant_gripper_open`, `gripper_open_position`,
`gripper_position_tolerance`, and `joint_state_max_age_s`. When enabled, an open trajectory is skipped only when
fresh joint feedback contains the gripper joint and is already within tolerance. Close commands and stale or missing
feedback always use the normal trajectory path.

### Message Definitions

**TaskStep.msg** — Task step:

```
uint8 MOVE_TO_POSE = 0    # Move to target pose
uint8 GRIPPER      = 1    # Gripper control
uint8 WAIT         = 2    # Wait

uint8   type                # Step type
string  label               # Human-readable label (for logging and feedback)
geometry_msgs/Pose target_pose   # MOVE_TO_POSE: target pose
float64 velocity_scaling         # MOVE_TO_POSE: velocity ratio (0.0~1.0, 0.0=default)
float64 gripper_position         # GRIPPER: position (0.0=closed, 1.0=fully open)
float64 wait_duration_s          # WAIT: duration in seconds
```

**ExecuteTaskPlan.action**:

```
# Goal
ibrobot_msgs/TaskStep[] steps    # Task step sequence
string task_id                    # Unique task identifier
string task_description           # Task description
---
# Result
bool   success
string message
int32  steps_completed           # Number of completed steps
float64 total_duration_s         # Total duration
---
# Feedback
int32   current_step             # Current step index (0-indexed)
int32   total_steps
string  current_label
string  status                   # "executing" | "completed" | "failed"
float32 progress                 # 0.0 ~ 1.0
```

## Quick Start

### 1. Build

```bash
colcon build --packages-select ibrobot_msgs robot_config so101_motion task_dispatch
source install/setup.bash
```

### 2. Launch

Task executor is automatically launched when `control_mode` contains "moveit":

```bash
export ROS_DOMAIN_ID=<your_id>

# Simulation mode
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=true \
    control_mode:=moveit_planning

# Real hardware
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false \
    control_mode:=moveit_planning
```

Verify the node is running:

```bash
ros2 node list | grep task_executor
# Output: /task_executor

ros2 action list | grep execute_task_plan
# Output: /task_executor/execute_task_plan
```

### 3. CLI Verification

**Test gripper open/close** (no MoveIt motion planning required, gripper + wait only):

```bash
ros2 action send_goal /task_executor/execute_task_plan ibrobot_msgs/action/ExecuteTaskPlan "{steps: [{type: 2, label: 'wait_1s', wait_duration_s: 1.0}, {type: 1, label: 'open_gripper', gripper_position: 1.0}, {type: 2, label: 'wait_0.5s', wait_duration_s: 0.5}, {type: 1, label: 'close_gripper', gripper_position: 0.0}], task_id: 'test_001', task_description: 'gripper cycle test'}" --feedback
```

Expected output:
```
Result:
    success: true
    message: All steps completed successfully
    steps_completed: 4
    total_duration_s: 3.6s
Goal finished with status: SUCCEEDED
```

**Test full pick sequence** (includes MOVE_TO_POSE, requires moveit_gateway to be ready):

```bash
ros2 action send_goal /task_executor/execute_task_plan ibrobot_msgs/action/ExecuteTaskPlan "{steps: [{type: 1, label: 'open_gripper', gripper_position: 1.0}, {type: 0, label: 'move_above', target_pose: {position: {x: 0.2, y: 0.0, z: 0.3}, orientation: {w: 1.0}}, velocity_scaling: 0.3}, {type: 0, label: 'move_to_grasp', target_pose: {position: {x: 0.2, y: 0.0, z: 0.1}, orientation: {w: 1.0}}, velocity_scaling: 0.2}, {type: 1, label: 'close_gripper', gripper_position: 0.0}, {type: 2, label: 'hold', wait_duration_s: 1.0}, {type: 0, label: 'lift', target_pose: {position: {x: 0.2, y: 0.0, z: 0.4}, orientation: {w: 1.0}}, velocity_scaling: 0.3}], task_id: 'pick_001', task_description: 'pick and place'}" --feedback
```

## Python API Usage

### Basic: Sending a Task Plan

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

        # Step 1: Open gripper
        step1 = TaskStep()
        step1.type = TaskStep.GRIPPER
        step1.label = 'open'
        step1.gripper_position = 1.0

        # Step 2: Move above target
        step2 = TaskStep()
        step2.type = TaskStep.MOVE_TO_POSE
        step2.label = 'approach'
        step2.target_pose = Pose()
        step2.target_pose.position.x = 0.2
        step2.target_pose.position.y = 0.0
        step2.target_pose.position.z = 0.3
        step2.target_pose.orientation.w = 1.0
        step2.velocity_scaling = 0.3

        # Step 3: Close gripper
        step3 = TaskStep()
        step3.type = TaskStep.GRIPPER
        step3.label = 'grasp'
        step3.gripper_position = 0.0

        # Step 4: Wait for settling
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
            f'Done: success={result.success}, '
            f'steps={result.steps_completed}, '
            f'time={result.total_duration_s:.1f}s'
        )
```

## Configuration

All parameters are loaded from robot_config YAML (SSOT) automatically.

**Gripper joint name** resolved from `joints.gripper`:

```yaml
joints:
  gripper:
    - "6"    # Gripper joint name (SO-101 uses numeric IDs)
```

**Gripper controller** matched by `gripper` + `trajectory` keyword in `control_modes`:

```yaml
control_modes:
  moveit_planning:
    controllers:
      - joint_state_broadcaster
      - arm_trajectory_controller
      - gripper_trajectory_controller  # → /gripper_trajectory_controller/follow_joint_trajectory
```

**Auto-launch conditions** (`launch_builders/task_execution.py`):

Task executor is automatically added to launch when any of:
- `control_mode` contains `moveit`
- `control_mode` contains `visual_grasp`
- `control_mode` contains `voxposer`
- `executor.task_dispatch: true`

## Integrating Existing Planners

### visual_grasp_planner

Remove the original `executor.py` and instead:
1. Compute grasp poses (YOLO + SAM + PCA + calibration)
2. Build a `TaskStep[]` sequence
3. Send as `ExecuteTaskPlan` action goal

### VoxPoser

1. LLM generates waypoints
2. Convert waypoints to `TaskStep[]` sequence
3. Send as `ExecuteTaskPlan` action goal

### Custom Planners

Any new planner follows the same pattern:
1. **Perception/Reasoning** → produces target poses
2. **Build** `TaskStep[]` sequence
3. **Send** `ExecuteTaskPlan` action goal
4. **Monitor** feedback for execution progress

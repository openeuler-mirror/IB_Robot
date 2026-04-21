# MoveIt Gateway Node 文档

## 概述

`moveit_gateway` 是 IB-Robot 项目中的核心运动控制节点，充当高层控制接口与底层 MoveIt2 运动规划框架之间的网关。该节点专门针对 SO101 5自由度机械臂设计，解决了 5DOF 机械臂在逆运动学（IK）求解中的特殊约束问题。

## 推荐启动场景

### Ubuntu 启动 MoveIt Planning 控制（仿真）

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true
```

如需在板端或无图形界面的环境中运行，可关闭 RViz：

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true \
    moveit_display:=false
```

### 端侧开发板启动 MoveIt Planning 控制（真机）

端侧开发板可运行 openEuler Embedded 或 OpenHarmony，真机场景下与 Ubuntu 的控制接口保持一致，只需关闭仿真模式：

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning
```

两种场景都通过 `/cmd_pose` 发送位姿命令，通过 `/robot_status/ee_pose` 观察末端位姿反馈。

## 主要功能

### 1. 位姿控制 (Pose Control)
- **订阅话题**: `/cmd_pose` (geometry_msgs/Pose)
- **功能**: 接收目标位姿，通过 IK 求解关节角度，控制机械臂运动
- **特点**: 支持多种约束策略和分层容差机制

### 2. 末端执行器状态发布
- **发布话题**: `/robot_status/ee_pose` (geometry_msgs/PoseStamped)
- **功能**: 实时发布末端执行器在 base 坐标系中的位姿

### 3. 关节状态同步
- **订阅话题**: `/joint_states` (sensor_msgs/JointState)
- **功能**: 同步当前关节状态，作为 IK 求解的起始状态

## 5DOF 机械臂的特殊处理

### 问题背景

5自由度机械臂只有 5 个关节，无法满足完整的 6DOF 位姿约束（3个位置 + 3个姿态）。直接使用标准 IK 求解器会导致：
- `NO_IK_SOLUTION` 错误
- IK 求解成功率低

### 解决方案

#### 1. 仅位置 IK (Position-Only IK)

在 `kinematics.yaml` 中配置：
```yaml
position_only_ik: True
```

**为什么需要设置为 True**：

5DOF 机械臂只有 5 个关节自由度，无法同时满足完整的 6DOF 位姿约束（3个位置 + 3个姿态）。当 IK 求解器尝试同时满足位置和姿态约束时，往往会出现：
- `NO_IK_SOLUTION` 错误：目标姿态在数学上不可达
- 求解成功率低：大部分目标位姿无解

**配置后的行为**：

- KDL 使用 `ChainIkSolverPos` 求解器（而非 `ChainIkSolverPos_NR`）
- **只优化位置**：求解器只尝试使末端执行器到达目标位置
- **忽略姿态约束**：优化过程中不考虑姿态目标
- **姿态仍可验证**：传入的 `orientation` 不会影响求解，但可通过 `constraints` 验证最终姿态是否可接受

**位姿如何影响求解**：

虽然 `position_only_ik: True` 忽略姿态优化，但传入的目标姿态仍有影响：
- **作为初始猜测**：不同的姿态目标可能引导求解器收敛到不同的关节空间解
- **约束验证**：如果同时传入 `constraints`，最终解仍需满足姿态容差要求
- **数值稳定性**：合理的目标姿态有助于 Newton-Raphson 迭代收敛

#### 2. 姿态约束优化

虽然启用了 `position_only_ik`，我们仍然提供姿态参考，通过以下方法：

**a) Z轴约束 (constrain_to_z_axis_only)**
- 保持末端执行器 Z 轴方向不变
- 放松绕 Z 轴的旋转（roll）
- 这符合 5DOF 机械臂的能力：2DOF 用于 Z 轴方向（pitch + yaw），剩余 3DOF 用于位置

**b) 姿态投影 (project_orientation_to_shoulder_xz_plane)**
- 将姿态投影到 shoulder 坐标系的 XZ 平面
- 适应 5DOF 机械臂的运动学约束

#### 3. 分层容差策略

从严格到宽松依次尝试：

| 策略 | 容差 (x, y, z) | 说明 |
|------|----------------|------|
| Strict | (0.1, 0.1, 0.05) | X/Y: ±5.7°, Z: ±2.8° |
| Medium | (0.3, 0.3, 0.1) | X/Y: ±17°, Z: ±5.7° |
| Relaxed | (0.5, 0.5, 0.15) | X/Y: ±28°, Z: ±8.6° |
| Z-axis only | (1.0, 1.0, 0.2) | X/Y: ±57°, 几乎不约束 |
| No constraints | None | 完全无姿态约束 |

#### 4. 多策略 Fallback

```
1. Gripper Z-axis constraint → 尝试所有容差
   - 只约束 gripper 的 Z 轴方向
2. Shoulder XZ plane projection → 尝试所有容差
   - 将姿态投影到 shoulder 坐标系 XZ 平面
   - 几何上更精确适配 5DOF 约束
3. Current orientation (保持当前姿态) → 尝试所有容差
4. Default orientation (无旋转) → 尝试所有容差
```

## 坐标系变换

### Shoulder 坐标系计算

为了准确判断工作空间可达性，节点将目标位置从 base 坐标系转换到 shoulder 坐标系：

```
P_shoulder = R × (P_base - T)
```

其中：
- `P_base`: 目标点在 base 坐标系中的位置
- `T`: shoulder 原点在 base 坐标系中的平移偏移
- `R`: base 到 shoulder 的旋转矩阵
- `P_shoulder`: 目标点在 shoulder 坐标系中的位置

### 日志输出

```
[INFO] Target Pose: x=-0.046, y=-0.000, z=0.423
[INFO]   Target in shoulder frame: x=-0.034, y=0.012, z=-0.323
[INFO]   Distance from base origin: 0.426 m
[INFO]   Distance from shoulder origin: 0.325 m
```

## 控制流程图

```
┌───────────────────────────────────────────────────────────┐
│                     moveit_gateway 节点                    │
└───────────────────────────────────────────────────────────┘
                              │
                              ▼
                       ┌───────────────┐
                       │   /cmd_pose    │
                       │     (Pose)     │
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │ cmd_pose_     │
                       │ callback      │
                       └───────┬───────┘
                               │
                               ▼
    ┌─────────────────────────────────────────┐
    │          坐标系变换与距离计算             │
    │  1. 获取 TF 变换                         │
    │  2. 计算 p_relative = P_base - T        │
    │  3. 应用旋转变换 P_shoulder = R × p_relative │
    │  4. 计算距离                             │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │              多策略 IK 求解              │
    │                                         │
    │    策略循环:                             │
    │    1. Z轴约束                            │
    │    2. Shoulder XZ平面投影                │
    │    3. 当前姿态                           │
    │    4. 默认姿态                           │
    │                                         │
    │    容差循环:                             │
    │    1. Strict (0.1)                       │
    │    2. Medium (0.3)                      │
    │    3. Relaxed (0.5)                      │
    │    4. Z-only (1.0)                       │
    │    5. None                               │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
    ┌─────────────────────────────────────────┐
    │            solve_and_move()              │
    │  1. 创建 Constraints (如果有容差)        │
    │  2. 调用 compute_ik_async()              │
    │  3. 等待结果                             │
    │  4. move_to_joint()                      │
    └───────────────────┬─────────────────────┘
                        │
                        ▼
               ┌───────────────────┐
               │    IK 成功？      │
               └───┬───────────┬───┘
                   │  YES      │  NO
                   ▼           ▼
         ┌───────────────┐  ┌───────────────┐
         │   移动成功    │  │ 尝试下一个    │
         │               │  │ 策略/容差     │
         └───────────────┘  └───────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────┐
│              发布 /robot_status/ee_pose                   │
│              (10Hz 定时器)                                │
└───────────────────────────────────────────────────────────┘
```

## 约束创建流程

```
┌─────────────────────────────────────────────────────────────┐
│                    create_orientation_constraint             │
├─────────────────────────────────────────────────────────────┤
│  输入:                                                       │
│    - target_quat: 目标四元数 (x, y, z, w)                   │
│    - link_name: "gripper"                                   │
│    - frame_id: "base"                                       │
│    - tolerances: (x_tol, y_tol, z_tol)                      │
│                                                              │
│  输出:                                                       │
│    - OrientationConstraint 消息                             │
│                                                              │
│  字段设置:                                                   │
│    - orientation.x/y/z/w: 目标姿态                          │
│    - absolute_x_axis_tolerance: X轴容差 (放松 roll)         │
│    - absolute_y_axis_tolerance: Y轴容差 (放松 roll)         │
│    - absolute_z_axis_tolerance: Z轴容差 (保持方向)          │
│    - weight: 1.0 (约束权重)                                 │
└─────────────────────────────────────────────────────────────┘
```

## 四元数数学工具

数学函数使用 `scipy.spatial.transform.Rotation` 和 NumPy 实现。

### quaternion_multiply(q1, q2)
四元数乘法：q = q1 × q2
- 使用 scipy: `R.from_quat() * R.from_quat()`

### quaternion_conjugate(q)
四元数共轭：q* = [-x, -y, -z, w]
- 使用 scipy: `R.from_quat().inv()`

### quaternion_to_rotation_matrix(q)
四元数转 3×3 旋转矩阵，返回 `np.ndarray`
- 使用 scipy: `R.from_quat().as_matrix()`

### rotation_matrix_to_quaternion(R)
3×3 旋转矩阵转四元数
- 使用 scipy: `R.from_matrix().as_quat()`

### constrain_to_z_axis_only(quat)
只约束末端执行器 Z 轴方向，放松绕 Z 轴旋转
- 使用 scipy 进行四元数/矩阵转换
- 使用 NumPy 向量化：`np.linalg.norm()`, `np.dot()`, `np.cross()`, `np.column_stack()`

### project_orientation_to_shoulder_xz_plane(quat)
将姿态投影到 shoulder 坐标系 XZ 平面
- 使用 scipy 进行四元数/矩阵转换
- 使用 NumPy 向量化，同上

## 配置文件

### kinematics.yaml
```yaml
arm:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.01
  kinematics_solver_timeout: 2.0
  kinematics_solver_attempts: 50
  position_only_ik: True  # 5DOF 机械臂关键配置
```

### moveit_controllers.yaml
```yaml
moveit_controller_manager: moveit_simple_controller_manager/MoveItSimpleControllerManager

moveit_simple_controller_manager:
  controller_names:
    - arm_trajectory_controller
    - gripper_trajectory_controller

  arm_trajectory_controller:
    action_ns: follow_joint_trajectory
    type: FollowJointTrajectory
    default: true
    joints: ["1", "2", "3", "4", "5"]

  gripper_trajectory_controller:
    action_ns: follow_joint_trajectory
    type: FollowJointTrajectory
    default: true
    joints: ["6"]
```

## ROS2 话题接口

### 订阅话题

| 话题名 | 消息类型 | 频率 | 说明 |
|--------|----------|------|------|
| /cmd_pose | geometry_msgs/Pose | 按需 | 目标位姿命令 |
| /joint_states | sensor_msgs/JointState | 100Hz | 关节状态反馈 |

### 发布话题

| 话题名 | 消息类型 | 频率 | 说明 |
|--------|----------|------|------|
| /robot_status/ee_pose | geometry_msgs/PoseStamped | 10Hz | 末端执行器位姿 |

## 使用示例

### 发送位姿命令

```bash
ros2 topic pub /cmd_pose geometry_msgs/Pose "{
  position: {x: 0.15, y: 0.0, z: 0.25},
  orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}
}" --once
```

### 查看末端位姿

```bash
ros2 topic echo /robot_status/ee_pose
```

## 多线程模型

节点使用 `MultiThreadedExecutor` 和 `ReentrantCallbackGroup` 实现线程安全：

- **回调组**: 所有订阅者使用同一个可重入回调组
- **执行器**: 多线程执行器处理并发回调
- **线程安全**: IK 调用使用 async 模式，避免内部 spin_once 调用

## 日志级别

- **INFO**: 目标位姿、距离信息、IK 成功
- **WARNING**: IK 失败、TF 查询失败、无效关节状态
- **DEBUG**: 关节状态更新、策略尝试详情

## 故障排查

### IK 持续失败

1. 检查位置是否超出工作空间（shoulder 距离应 < 0.35m）
2. 确认 `position_only_ik: True` 已设置
3. 查看日志中的距离信息

### TF 查询失败

1. 确认 robot_state_publisher 正在运行
2. 检查 URDF 中是否定义了 shoulder link
3. 使用 `ros2 run tf2_tools view_frames` 查看 TF 树

### 关节状态无效

- 确认 joint_states 包含所需的 5 个关节
- 检查关节名称是否为 ['1', '2', '3', '4', '5']

## IK 求解器技术细节

### Goal vs Constraints

`compute_ik_async` 调用时同时传入：
- `quat_xyzw`：目标姿态（主要优化目标）
- `constraints`：姿态容差约束（验证条件）

### KDL 求解器行为

**Newton-Raphson 迭代**：
- 目标：最小化误差函数 `E = E_position + E_orientation`
- 优先收敛到 `pose_stamped.pose` 的精确值
- `constraints` 主要用于验证最终解，而非扩大搜索空间

**5DOF 限制**：
- 当目标姿态数学上不可达（6DOF 约束 vs 5DOF 自由度）时
- KDL 可能无法收敛，返回 `NO_IK_SOLUTION`

### Position-Only IK 配置

**配置 `position_only_ik: True`**：
```yaml
# kinematics.yaml
position_only_ik: True
```

启用后的行为：
- KDL 使用 `ChainIkSolverPos` 而非 `ChainIkSolverPos_NR`
- **只优化位置，忽略姿态**
- constraints 中的姿态约束仍然会被验证
- 解算器在位置精确匹配的前提下，自由调整姿态

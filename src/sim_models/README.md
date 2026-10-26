# sim_models

IB-Robot 仿真相关的场景资产、场景编译器和开发者标定辅助工具。

## 当前职责

| 在范围内 | 不在范围内 |
|---|---|
| Gazebo / MuJoCo 场景模板与道具资产 | 机器人本体 SSOT 配置 |
| 场景编译与运行时装配 | 业务控制逻辑 |
| 开发者用仿真标定辅助工具 | 真机相机标定 |

## 相机标定相关入口

### `ros2 run sim_models sim_camera_adjuster`

交互式 Gazebo 代理相机调参工具，支持：

- 实时调节 `xyz / rpy / fovy`
- 读取已有 override 作为初值
- 保存到 `~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml`
- MuJoCo 目标下自动拉起一个 headless Gazebo 做位姿采集

常见调用方式：

```bash
ros2 run sim_models sim_camera_adjuster \
    --camera /camera/top/image_raw \
    --robot-config so101_single_arm
```

### `sim_models.aruco_spawner`

Python API，用于在仿真里动态插入和清理 ArUco A4 标定板：

```python
from sim_models.aruco_spawner import (
    spawn_aruco_gazebo,
    despawn_aruco_gazebo,
    verify_mujoco_model,
)
```

`dataset_tools.camera_alignment` 在仿真路径下会调用这组接口。

### `pick_banana_task_node`

MuJoCo `pick_banana` 场景任务由 `robot.launch.py` 启动，使用 legacy dispatcher reset。
仿真和 benchmark 始终使用直通推理；`robot.runtime_scheduler.enable=true` 在这些运行目标下保持休眠。
场景节点不管理 scheduler session 或 pipeline binding。后续由业务层获取推理结果并提交给仿真执行，
业务层同时负责场景重置时停止输出、清理旧推理状态和丢弃过期结果。

#### 场景任务服务（`std_srvs/srv/Trigger`）

| 服务 | 作用 |
|---|---|
| `/sim/pick_banana/random_set` | 推进共享放置索引并放置香蕉（banana 到 `placements[k]`，plate 到 layout 位姿） |
| `/sim/pick_banana/reset` | 重置到 layout 默认摆放 |
| `/sim/pick_banana/autotest/start` | 非阻塞启动 AutoTest 批测循环 |

放置索引 *k* 由 `scene_compiler.randomization_placements` 这个两平台共享的生成器定义：
MuJoCo 走一次原子 `ResetWorld` 状态覆盖（手臂 `arm_rest_qpos` + banana/plate 自由关节），
Gazebo 走 `SetEntityPose`，两平台同索引同位姿。

#### `randomization.banana` 配置与校验

| 字段 | 说明 |
|---|---|
| `mode` | `grid`（默认，与历史关键帧网格逐位一致）或 `uniform`（`seed`+`count` 确定性采样） |
| `x_range` / `y_range` | `[min, max]`，有限且 min ≤ max |
| `z_spawn` | 出生高度（香蕉落到桌面），必须有限 |
| `grid_cols` / `grid_rows` / `yaw_steps` | grid 模式的分割数，≥ 1 的整数 |
| `seed` / `count` | uniform 模式必填；必须是整数（拒绝 float/bool），`count >= 1` |

手臂字段 `arm_joint_names` / `arm_rest_qpos` 必须等长、关节名唯一且非空、rest 值有限。
所有校验在节点构造期执行（`randomization_placements` / `arm_rest_pairs` 抛
`ValueError` 并指明字段），坏 layout 在任何运动之前 fail-closed。

#### `autotest_indices` 参数

- INTEGER_ARRAY，默认空 = 跑全部放置索引；例如 `ros2 param set /pick_banana_task_node autotest_indices "[0, 3, 7]"`。
- 每次 `/sim/pick_banana/autotest/start` 时重新读取，两次批测之间可以运行时修改。
- 越界索引会让 start 返回明确的失败信息，且不影响后续操作。
- 恢复“全部索引”：`ros2 param set ... "[]"` **不可用**——空列表被推断为
  BOOL_ARRAY，节点会拒绝（`Wrong parameter type ... got 'Type.BOOL_ARRAY'`），
  且 CLI 退出码仍是 0。需显式发送类型化的空 INTEGER_ARRAY（`type: 7`）：

  ```bash
  ros2 service call /pick_banana_task_node/set_parameters rcl_interfaces/srv/SetParameters \
    "{parameters: [{name: autotest_indices, value: {type: 7, integer_array_value: []}}]}"
  ```

#### 结果文件

每轮 AutoTest 写 `~/ib_robot_eval/pick_banana/<时间戳>/results.jsonl`（可用
`eval_output_dir` 参数改根目录），每行一个 episode：

| 字段 | 含义 |
|---|---|
| `idx` | episode 序号（0 起，与基线语义一致） |
| `placement_index` | 本集实际下发的放置索引 *k*（允许重复/乱序） |
| `banana_spawn` | 本集下发的香蕉位姿 `[x, y, z, yaw]` |
| `success` / `reason` | 判定结果（`placed_on_plate` / `timeout` / `flew_out` / `no_data` / `stopped`） |
| `banana_final_pos` | 评估结束时的观测位姿 |
| `duration_s` | 本集评估耗时 |

#### MuJoCo 观测与 readiness

- 观测话题：`/sim/pick_banana/free_joint_states`（`mujoco_ros2_control_msgs/FreeJointStateArray`），
  由 runtime provider 为含 dynamic 物体的场景配置的 `FreeJointStatePublisherPlugin` 发布（50 Hz，world 系）。
- readiness 条件：`reset_world` 服务可用 **且** banana/plate 都有新鲜样本
  （接收时刻距今 ≤ `POSE_STALE_S = 1.0 s`）。仿真暂停、插件断流或话题停发超过
  该阈值时，readiness 拒绝一切任务操作（服务返回 `backend not ready`），恢复后自动放行。

## 与其他包的关系

```text
dataset_tools.camera_alignment
    ├─► sim_models.aruco_spawner
    └─► ~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml
                                 │
                                 ▼
robot_config.launch_builders.sim_backend.camera_presets
```

`~/.ros/ibrobot/sim_camera_overrides/` 是开发者标定侧的 opt-in 旁路，不是新的配置单一数据源。

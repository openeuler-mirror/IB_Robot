# hardware_mock

> 契约驱动 (contract-driven) 的硬件 Mock 包，让 IB-Robot 的端到端推理流水线
> **在没有真实相机、没有真实机械臂、没有 Gazebo** 的情况下也能完整跑通。

---

## 1. Background

IB-Robot 的标准流程是：

```
相机 / 关节传感器 ──► 推理节点 (LeRobot ACT / RKNN) ──► 动作分发 ──► ros2_control ──► 机械臂
```

在以下场景里，这条链路非常难调：

- **开发机上没有 SO-101 机械臂**，但需要验证 `inference_service`、`action_dispatch`、
  `dataset_tools` 是否正确订阅/发布；
- **CI / 回归测试**里启动 Gazebo + 控制器太重，只想验证消息契约 (topic / type / QoS) 没有漂移；
- **新同事上手**时希望 5 分钟内看到一个"会响应动作消息"的机器人，而不必先配相机驱动、串口、固件。

`hardware_mock` 进行 mock 时能做到：

- 唯一信源：[robot_config](../robot_config/) 下的同一份 `robot.yaml`；
- 自动从 `robot.contract.observations` 推导要 **发布** 的所有话题；
- 自动从 `robot.contract.actions` 推导要 **订阅** 的所有话题；
- 收到 action 立刻反映到内部 `JointModel` 并立即回吐 `joint_states`，形成闭环；
- 启动时硬校验帧率 vs `align.tol_ms`、关节索引、消息类型白名单，**契约不一致就拒绝启动**，
  避免"看起来在跑、其实早就漂了"的隐性 bug。

> 设计原则：**不复刻硬件物理特性**（不做动力学、不做相机畸变），只复刻"契约层"行为。
> 真要做物理仿真请使用专业物理仿真如 Gazebo 等 (`use_sim:=true`)。

---

## 2. 快速使用 (TL;DR)

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_mock:=true \
    control_mode:=model_inference
```

启动后你会看到类似输出：

```
[contract_mock]: ============================================================
[contract_mock]: hardware_mock contract_mock active
[contract_mock]:   joints: ['1','2','3','4','5','6']
[contract_mock]:   PUB  [      image] /camera/top/image_raw (sensor_msgs/msg/Image) @ 30.0 Hz
[contract_mock]:   PUB  [      image] /camera/wrist/image_raw (sensor_msgs/msg/Image) @ 30.0 Hz
[contract_mock]:   PUB  [joint_state] /joint_states (sensor_msgs/msg/JointState) @ 50.0 Hz
[contract_mock]:   SUB  [     action] /action/arm (std_msgs/msg/Float64MultiArray) -> joints ['1','2','3','4','5']
[contract_mock]:   SUB  [     action] /action/gripper (std_msgs/msg/Float64MultiArray) -> joints ['6']
[contract_mock]: ============================================================
```

简单验证：

```bash
ros2 topic hz /camera/top/image_raw          # 应该接近 fps
ros2 topic echo /joint_states --once          # 应该看到 6 个关节
# 手动发一个 action，joint_states 立刻变化
ros2 topic pub --once /action/arm std_msgs/msg/Float64MultiArray \
    "{data: [0.1, 0.0, 0.0, 0.0, 0.0]}"
```

---

## 3. 完整使用 (Full Usage)

### 3.1 与 `robot.launch.py` 的集成

`use_mock` 是顶层开关，**与 `use_sim` 互斥**。启用后会：

| 子系统 | 行为 |
| --- | --- |
| `ros2_control` 节点 | 跳过 (不启动 controller_manager / spawner) |
| 相机驱动 / 雷达 / virtual_relay / TF | 跳过 |
| `voice_asr_service` | 跳过 |
| `navigation` | 跳过 |
| `inference_service` + `action_dispatch` | **正常启动** |
| `hardware_mock.contract_mock` | **追加启动** |

约束：

- 必须 `control_mode:=model_inference`（其他模式下 mock 没有意义，启动会报错）；
- `auto_start_controllers` 会被强制关闭。

### 3.2 单独调试 mock 节点

只想看 mock 在发什么，不需要推理：

```bash
ros2 launch hardware_mock hardware_mock.launch.py \
    robot_config:=so101_single_arm
# 或者直接指向任意 YAML
ros2 launch hardware_mock hardware_mock.launch.py \
    config_path:=/abs/path/to/your_robot.yaml
```

### 3.3 YAML 可调参数

所有 mock 行为都在 `robot.hardware_mock:` 这一节微调（**完全可选**，不写就用默认值）：

```yaml
hardware_mock:
  joint_state_rate_hz: 50          # /joint_states 发布频率，默认 50
  skip_rate_check: false           # 关掉启动期帧率硬校验 (不推荐)
  image_sources:
    top:                           # key = peripheral.name
      kind: checkerboard           # checkerboard | solid | gradient
      tile: 40                     # checkerboard 专用
    wrist:
      kind: solid
      color: '#3399ff'             # solid 专用，#RRGGBB
```

### 3.4 启动期硬校验规则

`build_plan()` 会在启动时直接抛 `ValueError` 的几种情况：

1. **消息类型不在白名单**
   - observations 只支持 `sensor_msgs/msg/Image`、`sensor_msgs/msg/JointState`
   - actions 只支持 `std_msgs/msg/Float64MultiArray`
2. **帧率 vs 同步容忍度不匹配**
   - 规则：`rate_hz >= 2000 / align.tol_ms`
   - 例：`tol_ms: 100` → 至少 20 Hz。低于这个值会让推理拿到陈旧帧，与其在运行时"看起来很安静"，
     不如启动就拒绝。可用 `skip_rate_check: true` 临时绕过。
3. **action selector 越界**
   - 形如 `action.<i>` 必须满足 `0 <= i < len(joints.all)`。
4. **observations 引用了不存在的 peripheral**。

### 3.5 行为细节 (action → state 闭环)

```
inference ──► /action/arm (Float64MultiArray) ──► contract_mock._on_action
                                                       │
                                                       ▼
                                                JointModel.set_by_index
                                                       │
                                                       ▼
                                          立即回吐 /joint_states (不等下一周期)
```

这样下游可以保证：每次发动作之后立刻能 echo 到对应的关节变化，不会出现"看起来发了
但 50 Hz 周期里没收到回执"的假阳性。

### 3.6 测试

```bash
PYTHONPATH=src/hardware_mock python3 -m pytest src/hardware_mock/test -q
```

覆盖：契约编译、关节索引模型、图像源工厂、所有架构性硬校验路径。

---

## 4. 架构图

### 4.1 模块分层

```
┌──────────────────────────────────────────────────────────────────────┐
│                       robot_config/robots/*.yaml                     │
│  joints / peripherals / contract.observations / contract.actions     │
└──────────────────────────────────────────────────────────────────────┘
                                  │ load_robot_config_dict
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       hardware_mock (this package)                   │
│                                                                      │
│   contract_plan.build_plan()  ── 编译 & 硬校验                       │
│        │                                                             │
│        ├── type_registry      消息类型白名单                         │
│        ├── image_sources      checkerboard / solid / gradient        │
│        └── joint_model        有序关节状态存储                       │
│        ▼                                                             │
│   MockPlan ──► contract_mock_node.ContractMockNode                   │
│                    │                                                 │
│                    ├─ Publisher × N    (Image / JointState)          │
│                    ├─ Subscriber × M   (Float64MultiArray)           │
│                    └─ Timer × K        (周期发布)                    │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 运行时数据流

```
                ┌──────────────────────────┐
                │   contract_mock (node)   │
                │                          │
   ┌────timer──►│  ImageGenerator ──► PUB │──► /camera/<name>/image_raw
   │            │                          │
   ┌────timer──►│  JointModel.positions   │──► /joint_states
   │            │        ▲                 │
   │            │        │ set_by_index    │
   │            │        │                 │
   │            │  on_action(msg) ◄── SUB │◄── /action/<key>
   │            │        │                 │              ▲
   │            │        └── 立即再发一次  │              │
   │            │            joint_states  │              │
   │            └──────────────────────────┘              │
   │                                                      │
   │       ┌──────────────────────────────────────────────┴──────────┐
   │       │ inference_service  ──►  action_dispatch  ──►  /action/* │
   │       │       ▲                                                  │
   │       └───────┴── 订阅 /camera/* + /joint_states                 │
   │                                                                  │
   └──────────────────────────────────────────────────────────────────┘
```

### 4.3 启动时序 (`use_mock:=true`)

```
robot.launch.py
   │
   ├─ 解析 use_mock=true ──► 校验互斥 (use_sim)
   │                       └► 强制 control_mode=model_inference
   │                       └► auto_start_controllers=false
   │
   ├─ [跳过] ros2_control / cameras / lidar / nav / voice_asr / TF
   │
   ├─ [启动] inference_service  ──► 订阅 contract.observations
   ├─ [启动] action_dispatch    ──► 发布 contract.actions
   └─ [启动] hardware_mock.contract_mock
                │
                ├─ load_robot_config_dict(robot_config_path)
                ├─ build_plan(robot)   ── 启动期硬校验
                ├─ create_publisher × observations
                ├─ create_subscription × actions
                └─ create_timer × (images + joint_state)
```

---

## 5. 不支持 / 不打算支持

| 项目 | 原因 |
| --- | --- |
| TF 树 | 真实 TF 由 `robot_description` 在非 mock 模式下负责，避免在 mock 里复制一份漂移 |
| 物理动力学 | 用 Gazebo (`use_sim:=true`) |
| 真实图像 / rosbag 回放 | 这是 `dataset_tools` 的职责 |
| `CompressedImage` / `Twist` / `JointTrajectory` 等 | 当前推理流水线契约不使用，需要时再加 |

---

## 6. 文件清单

```
src/hardware_mock/
├── README.md                          # 本文档
├── package.xml
├── setup.py / setup.cfg
├── resource/hardware_mock
├── hardware_mock/
│   ├── __init__.py
│   ├── type_registry.py               # 消息类型白名单 + 解析
│   ├── image_sources.py               # 合成图像生成
│   ├── joint_model.py                 # 内部关节状态
│   ├── contract_plan.py               # YAML → MockPlan 编译 + 硬校验
│   └── contract_mock_node.py          # rclpy 节点入口 (executable: contract_mock)
├── launch/
│   └── hardware_mock.launch.py        # 独立调试用 launch
└── test/
    ├── test_joint_model.py
    ├── test_image_sources.py
    └── test_contract_plan.py
```


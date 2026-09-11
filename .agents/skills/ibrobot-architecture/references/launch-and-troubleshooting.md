# Launch System and Troubleshooting

## When to Read

- 需要选择正确的 launch builder 来组合启动节点（control/perception/simulation/execution/navigation/voice/recording/tracing 等）
- 需要查阅 `robot.launch.py` 支持的 launch arguments 及默认值
- 第一次启动系统、添加新机器人、调试契约合成时
- 遇到配置校验失败、控制器不匹配、推理 pipeline 启动失败等问题需要排查

## Launch System

`src/robot_config/launch/robot.launch.py` 通过 `OpaqueFunction` 编排，直接调用 `launch_builders/` 的构建函数；builder 之间无插件注册机制，为固定 facade。仿真平台经 `sim_backend.get_sim_backend(platform)` 分发到 `GazeboAdapter` / `MujocoAdapter` / `MockAdapter`。

### Launch Builders (30)

| Group | Builders |
|-------|----------|
| Control & description | `control.py` (ros2_control nodes, controller spawners), `description.py` (URDF xacro + camera injection), `hand_sources.py` (hand data sources + profiles) |
| Perception | `perception.py` (camera/lidar drivers, virtual relays), `perception_models.py` (typed model-service plugins from `perception_services.services`), `camera_isp_overrides.py` (ISP calibration) |
| Simulation | `simulation.py` (dispatch), `sim_backend/` (gazebo/mujoco/mock adapters), `sim_peripheral_bridge.py` (ros_gz bridge), `hardware_mock.py` (contract-driven mock) |
| Inference & execution | `execution.py` (named inference pipelines + dispatch routing), `task_execution.py` (`task_executor_node`) |
| Navigation | `navigation.py`, `nav2.py`, `localization.py` (EKF + RTAB-Map), `navigation_command.py` (typed command server), `cmd_vel.py` (chassis bridge), `static_tf.py`, `fast_lio.py` |
| Voice & audio | `audio_io.py`, `voice_asr.py`, `voice_tts.py`, `speech_direction.py`, `voice_nav.py` (ASR → Nav2 goal) |
| Output & observability | `recording.py` (continuous rosbag vs episodic recorder + rerun viewer), `benchmark.py`, `semantic_mapping.py`, `tracing.py` (LTTng session), `moveit.py` (move_group + RViz) |

### Key Launch Arguments (25)

| Argument | Purpose | Default |
|----------|---------|---------|
| `robot_config` | Configuration name (without .yaml) | `so101_single_arm` |
| `config_path` | Full path override | `""` |
| `use_sim` | Simulation flag（空则由 runtime target 推断） | `""` |
| `sim_platform` | Override `simulation.platform` (gazebo/mujoco/mock) | `""` |
| `runtime_target` | hardware/simulation/benchmark override | `""` |
| `auto_start_controllers` | Spawn controllers at startup | `true` |
| `control_mode` | teleop / model_inference / moveit_planning（空 → `default_control_mode`） | `""` |
| `nav_stage` | mapping/navigation/grasp/hybrid | `""` |
| `hand_profile` | Hand profile selection | `""` |
| `with_inference` | Force enable/disable inference（空自动检测） | `""` |
| `inference_pipeline` | Named pipeline ID for overrides | `""` |
| `inference_execution_mode` | monolithic / distributed | `""` |
| `with_moveit` | Auto-detect if 'moveit' in mode name | `""` |
| `with_navigation` | 空 → `robot.navigation.enabled` | `""` |
| `navigation_mode` | full / odom_only / imu_only | `""` |
| `moveit_display` | RViz | `true` |
| `record` | Rosbag recording | `false` |
| `record_mode` | continuous / episodic | `continuous` |
| `voice_asr_auto_start` | Force ASR enabled + continuous | `false` |
| `voice_asr_realtime_pre_roll_seconds` | ASR pre-roll | `""` |
| `with_embodied` | 空 → `robot.embodied.enabled` | `""` |
| `with_perception` | 空 → `robot.embodied.perception.enabled` | `""` |
| `record_visualizer` | none / rerun | `none` |
| `enable_tracing` | ros2_tracing/LTTng | `false` |
| `trace_session_name` | Trace session name | (builder default) |

## Common Patterns

### Launching the System

```bash
# Standard launch
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# Mock backend (no hardware, no Gazebo): runtime target simulation + mock platform
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm \
  use_sim:=true sim_platform:=mock

# Override control mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm \
  control_mode:=moveit_planning

# Mobile base with navigation stage
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_nav_grasp \
  nav_stage:=grasp with_navigation:=true

# With episodic recording
ros2 launch robot_config robot.launch.py control_mode:=teleop \
  record:=true record_mode:=episodic
```

### Adding a New Robot

1. Create YAML: `config/robots/my_robot.yaml`（单顶层键 `robot:`）
2. 需要变体时用 overlay：`base_config: <base-name>`（同目录），列表追加用 `__append__`
3. Define `name`, `joints`, `control_modes`, `peripherals`, `contract`（按需加 `voice_*` / `embodied` / `navigation`）
4. Launch: `ros2 launch robot_config robot.launch.py robot_config:=my_robot`

### Debugging Contracts

契约在 launch 时**内存合成**（`RobotConfig.to_contract()` / `build_contract_from_robot_config_dict()`），不落盘：

```python
# Offline inspection
from robot_config.loader import load_robot_config
rc = load_robot_config("so101_single_arm")
contract = rc.to_contract()
print(contract.rate_hz, [o.key for o in contract.observations])
```

- Launch 日志中配置校验失败会 fail-fast 并给出具体原因（observation source 不在 `peripherals`、inference 引用的模型/pipeline 不存在、executor 类型非法等）。
- `loader.robot_config_digest()` 可用于对比两份配置是否漂移。

## Troubleshooting

### Issue: Config validation fails at launch

**Cause**: SSOT YAML 违反校验规则（`loader.validate_config` 在 launch 内执行）

**Check**:
1. Observation 的 `source` 是否在 `peripherals` 中定义
2. `control_modes.<mode>.inference` 引用的模型/pipeline 是否存在（`inference_config.parse_inference_config`）
3. Executor 类型是否为 `topic` / `benchmark`，controller 名是否出现在 `ros2_control`
4. Overlay 配置的 `base_config` 是否为同目录文件（跨目录、循环会被拒绝）

### Issue: Wrong controllers running

**Cause**: Control mode mismatch（每个 mode 的控制器集合来自 YAML `control_modes.<mode>.controllers`）

**Solution**:
```bash
# For MoveIt / skills
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning

# For policy inference
ros2 launch robot_config robot.launch.py control_mode:=model_inference
```

参考映射：position controllers（topic executor）用于 teleop / model_inference；trajectory controllers（action executor，经 `moveit_gateway` / `task_dispatch`）用于 moveit_planning；`base_controller` / `base_velocity_controller` 用于 base_navigation。

### Issue: Inference pipeline fails to start

**Common Errors**:
1. Manifest 加载失败 - bundle 必须是 schema v3（`inference_manifest.load_inference_manifest` 严格校验，含 artifact digest）
2. Backend conformance 失败 - deployment 的 runtime profile 与后端能力不匹配（如 ascend 要求 `target.runtime=acl`、artifacts 格式 `om`；见 `backends/registry.py` 各 validator）
3. Scheduler 会话失败 - session 化路径需先 `/inference/scheduler/ready` 就绪；重复请求会被 `IdempotencyLedger` 拒绝
4. Distributed 模式无视频 - 检查 RTP 通道协商（`VideoStreamNegotiator`）与 keyframe 等待状态（`WAITING_FOR_KEYFRAME`）

---
name: ibrobot-architecture
description: "Provides deep knowledge of IB-Robot's architecture. Use when user needs to 'understand architecture', 'explain design', 'check SSOT', 'modify robot_config', 'check contract', 'architecture', '架构', '设计说明', '配置加载', '数据流', '契约设计'. Triggers for 'how does it work?', '架构设计', '系统原理', or when modifying core robot parameters and single source of truth files."
---

# IB-Robot Architecture Skill

This skill provides comprehensive knowledge of IB-Robot's layered architecture, design principles, and core components.

**Reference Documentation**: https://deepwiki.com/wuxiaoqiang12/IB_Robot

## Three Architectural Pillars

### 1. Single Source of Truth (robot_config YAML)

The `robot_config` YAML file serves as the **single authoritative source** for all robot specifications:

| Traditional Approach | IB-Robot Approach |
|---------------------|-------------------|
| Separate configs for ros2_control, cameras, ML contracts | One YAML defines everything |
| Manual synchronization between systems | Auto-propagation to all subsystems |
| Configuration drift over time | Guaranteed consistency |

**Key Files**:
- `src/robot_config/config/robots/so101_single_arm.yaml` - Robot configuration (SSOT)
- `src/robot_config/robot_config/loader.py` - Config loader and validator
- `src/robot_config/robot_config/config.py` - `RobotConfig` dataclass

### 2. Contract-Driven Design

A **Contract** defines the observation-action interface between robot and policy:

```python
@dataclass
class Contract:
    name: str
    rate_hz: int
    max_duration_s: float
    observations: list[ObservationSpec]  # Sensors → ML tensors
    actions: list[ActionSpec]            # ML tensors → Actuators
```

**Contract Consumers** (identical processing):
1. `episode_recorder` - Records data during teleoperation
2. `bag_to_lerobot` - Converts rosbag to LeRobot dataset
3. `lerobot_policy_node` - Live inference

**Key Files**:
- `src/robot_config/robot_config/contract_utils.py` - Contract data structures
- `src/robot_config/robot_config/contract_builder.py` - Contract synthesis
- `src/robot_config/robot_config/config.py` - `to_contract()` method

### 3. Control Mode Architecture

Three control modes converge on the same `ros2_control` hardware interface:

| Mode | Controllers | Interface | Frequency | Use Case |
|------|-------------|-----------|-----------|----------|
| `teleop` | `JointGroupPositionController` | Topic | 50 Hz | Human teleoperation |
| `model_inference` | `JointGroupPositionController` | Topic | 100 Hz | AI policy control |
| `moveit_planning` | `JointTrajectoryController` | Action | Variable | Motion planning |

**Key Files**:
- `src/robot_config/launch/robot.launch.py` - Mode selection logic
- `src/robot_config/robot_config/launch_builders/` - Modular launch builders

## Package Architecture

```
src/
├── robot_config/        # Configuration center (SSOT)
│   ├── config/robots/   # YAML configurations
│   ├── launch/          # robot.launch.py orchestrator
│   └── robot_config/    # Python modules
│       ├── loader.py           # Config loading
│       ├── contract_builder.py # Contract synthesis
│       └── launch_builders/    # Modular node generators
├── tensormsg/           # ROS↔Tensor protocol conversion
├── inference_service/   # Policy inference (monolithic/distributed)
├── action_dispatch/     # Action execution with temporal smoothing
├── dataset_tools/       # Episode recording & dataset conversion
├── robot_teleop/        # Teleoperation interfaces
├── robot_moveit/        # Motion planning integration
├── robot_description/   # URDF, SRDF, meshes
└── so101_hardware/      # ros2_control hardware plugin
```

### Package Responsibilities

| Package | Primary Responsibility |
|---------|----------------------|
| `robot_config` | Configuration management, launch orchestration |
| `ibrobot_msgs` | Interface definitions (Actions, Messages) |
| `tensormsg` | ROS↔Tensor protocol conversion |
| `inference_service` | Policy inference (monolithic/distributed) |
| `action_dispatch` | Action execution with temporal smoothing |
| `dataset_tools` | Episode recording and dataset conversion |
| `robot_teleop` | Teleoperation interfaces |
| `robot_moveit` | Motion planning integration |
| `so101_hardware` | Hardware drivers (ros2_control plugin) |

### README as Local Architecture Contract

Each package-level `README.md` is treated as the package's local architecture contract. It should describe the package's responsibilities, public entry points, launch/configuration usage, data flow, dependency boundaries, and known constraints.

When code changes alter any of the following, the package README must be checked and updated if needed:

1. Package responsibilities or prohibited responsibilities
2. Public APIs, CLIs, launch arguments, topics, services, or actions
3. Configuration keys, defaults, or SSOT sources
4. Data flow, tensor/ROS message contracts, or control mode behavior
5. Cross-package dependencies or layer boundaries
6. Operational limitations, required hardware, or setup steps

Architecture reviews should flag README drift as an architecture issue when code behavior and documentation diverge. A stale README is not a minor documentation style problem; it invalidates the package contract and makes future architecture reviews unreliable.

## Data Flow Architecture

### Observation Flow (Sensors → Inference)

```
Camera/JointState → ROS Topic → decode_value() → StreamBuffer → sample() → Preprocessor → Model
```

**Key Code Paths**:
1. `_obs_cb(msg, spec)` in `lerobot_policy_node.py:381-397`
2. `decode_value()` in `contract_utils.py`
3. `_sample_obs_frame()` in `lerobot_policy_node.py:399-420`

### Action Flow (Inference → Hardware)

```
Model → VariantsList → TemporalSmoother → Queue → TopicExecutor → Controller Topic → Hardware
```

**Key Code Paths**:
1. `_result_cb()` in `action_dispatcher_node.py:232-278`
2. `TemporalSmoother.update()` in `temporal_smoother.py`
3. `_control_loop()` in `action_dispatcher_node.py:172-201`

## Inference Execution Modes

### Monolithic Mode (Default)

All inference components in single process with zero-copy tensor passing:

```
lerobot_policy_node process:
  ├─ TensorPreprocessor (CPU)
  ├─ PureInferenceEngine (GPU)
  └─ TensorPostprocessor (CPU)
```

### Distributed Mode (Cloud-Edge)

Edge handles preprocessing/postprocessing, cloud handles GPU inference:

```
Edge Node                    Cloud Node
┌─────────────────┐         ┌─────────────────┐
│ Preprocessor    │ ──────► │ PureInference   │
│ (CPU)           │         │ Engine (GPU)    │
│                 │ ◄────── │                 │
│ Postprocessor   │         └─────────────────┘
│ (CPU)           │
└─────────────────┘
```

**Configuration**:
```yaml
inference:
  mode: distributed  # or monolithic
  edge_node: true    # for edge node
```

## Temporal Smoothing

Cross-frame action chunk smoothing for seamless motion:

```
weight[k] = exp(-temporal_ensemble_coeff * k)
```

- Default `temporal_ensemble_coeff = 0.01` (from ACT paper)
- Precomputed weights for fast blending
- Aligns new chunk with `actions_executed` from previous chunk

**File**: `src/action_dispatch/action_dispatch/temporal_smoother.py`

## Launch System

### Modular Launch Builders

| Builder | Responsibility |
|---------|---------------|
| `control.py` | ros2_control setup, controller spawning |
| `perception.py` | Camera drivers, TF tree |
| `simulation.py` | Gazebo launch |
| `execution.py` | Inference and dispatch nodes |
| `teleop.py` | Teleoperation nodes |
| `recording.py` | Episode recording |

### Key Launch Arguments

| Argument | Purpose | Default |
|----------|---------|---------|
| `robot_config` | Configuration name | `test_cam` |
| `use_sim` | Enable Gazebo simulation | `false` |
| `control_mode` | Override control mode | (from YAML) |
| `with_inference` | Force enable/disable inference | (auto-detect) |
| `record` | Enable episode recording | `false` |

## Common Patterns

### Launching the System

```bash
# Standard launch
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# Override control mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning

# With recording
ros2 launch robot_config robot.launch.py control_mode:=teleop record:=true
```

### Adding a New Robot

1. Create YAML: `config/robots/my_robot.yaml`
2. Define `name`, `joints`, `control_modes`, `models`, `peripherals`
3. Launch: `ros2 launch robot_config robot.launch.py robot_config:=my_robot`

### Debugging Contracts

```bash
# View synthesized contract
cat /tmp/robot_config/contracts/so101_single_arm_teleop.yaml

# Check launch logs
# [robot_config] ✓ Contract synthesis SUCCESS
# [robot_config]   Observations: 2
# [robot_config]   Actions: 6 joints
```

## Troubleshooting

### Issue: ModuleNotFoundError: lerobot

**Cause**: PYTHONPATH not injected properly

**Check**:
1. Look for `[robot_config] PYTHONPATH injection:` in launch logs
2. Verify `AMENT_PREFIX_PATH` includes workspace install directory

### Issue: Wrong controllers running

**Cause**: Control mode mismatch

**Solution**:
```bash
# For MoveIt
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning

# For ACT inference
ros2 launch robot_config robot.launch.py control_mode:=model_inference
```

### Issue: Contract synthesis fails

**Common Errors**:
1. `KeyError: 'robot'` - Use `robot_config['name']` not `robot_config['robot']['name']`
2. `Observation source not found` - Check `peripherals` matches `observations` sources
3. `Model not found` - Add model to `models:` section in YAML

## Key Files Reference

| File | Purpose |
|------|---------|
| `robot_config/config/robots/so101_single_arm.yaml` | Robot configuration (SSOT) |
| `robot_config/launch/robot.launch.py` | Main orchestrator |
| `robot_config/robot_config/loader.py` | Config loading |
| `robot_config/robot_config/contract_builder.py` | Contract synthesis |
| `robot_config/robot_config/contract_utils.py` | Contract data structures |
| `robot_config/robot_config/launch_builders/execution.py` | Inference nodes |
| `inference_service/lerobot_policy_node.py` | Policy inference |
| `action_dispatch/action_dispatcher_node.py` | Action dispatch |
| `action_dispatch/temporal_smoother.py` | Temporal smoothing |
| `dataset_tools/episode_recorder.py` | Episode recording |
| `dataset_tools/bag_to_lerobot.py` | Dataset conversion |

## DeepWiki References

- [IB-Robot Overview](https://deepwiki.com/wuxiaoqiang12/IB_Robot/1-ib-robot-overview)
- [Core Concepts](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3-core-concepts)
- [Single Source of Truth Pattern](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.1-single-source-of-truth-pattern)
- [Contract System](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.2-contract-system)
- [Control Mode Architecture](https://deepwiki.com/wuxiaoqiang12/IB_Robot/3.3-control-mode-architecture)
- [System Architecture](https://deepwiki.com/wuxiaoqiang12/IB_Robot/4-system-architecture)
- [Configuration System](https://deepwiki.com/wuxiaoqiang12/IB_Robot/5-configuration-system-(robot_config))
- [Inference Pipeline](https://deepwiki.com/wuxiaoqiang12/IB_Robot/7-inference-pipeline)
- [Action Dispatch](https://deepwiki.com/wuxiaoqiang12/IB_Robot/8-action-dispatch)
- [Data Pipeline](https://deepwiki.com/wuxiaoqiang12/IB_Robot/9-data-pipeline)

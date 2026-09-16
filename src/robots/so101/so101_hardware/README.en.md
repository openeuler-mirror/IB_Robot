# SO-101 Robotic Arm Hardware Package

Hardware driver package for the SO-101 robotic arm, providing high-performance C++ ros2_control interfaces and Python utilities.

## Overview

This package provides a complete hardware driver solution for the SO-101 robotic arm, supporting two main modes:
- **C++ ros2_control Plugin**: Direct communication via FTServo SDK, low latency, high performance. Ideal for production and control.
- **Python Utilities**: Tools for calibration, data collection (Leader Arm Publisher), and diagnostics.

## Key Features

- **Direct Communication**: Uses FTServo SDK for native communication with Feetech servos.
- **Mixed Package Build**: Uses `ament_cmake_python` to support both C++ plugins and Python scripts in a single package.
- **Startup Position Protection**: Supports `reset_positions` to prevent the arm from jumping to zero on startup (critical for mobile platforms).
- **Lifecycle Management**: Implements standard `on_init`, `on_configure`, `on_activate`, and `on_deactivate` states.
- **Current Feedback**: Converts Feetech `Present_Current` to amperes with STS3215 `1 LSB = 6.5mA` and publishes `ibrobot_msgs/msg/JointCurrent` on `/so101_follower/joint_currents` or `/so101_leader/joint_currents` for `observation.current` dataset export.
- **Safety**: Automatically disables motor torque (Torque Off) on node shutdown. When `on_activate` fails, a fail-closed rollback first disables torque for every motor, then attempts to relock any EPROM left unlocked, and only then closes the serial port — reporting the two failure classes independently. This rollback protects the entire activation, including the final initial sync read: any sync-read transmit failure or per-motor reply failure aborts activation through the same rollback path.

## Architecture

```
ros2_control (Controller Manager)
      ↓
SO101SystemHardware (C++ Plugin)  ←──┐
      ↓                              │
FTServo SDK (C++)                    │
      ↓                              │
Feetech Servos (Hardware)            │
      ↑                              │
Python Utilities (Scripts) ──────────┘
```

## Dependencies

### Git Submodule
FTServo_Linux SDK is included as a git submodule:
```bash
git submodule update --init --recursive
```

### System Dependencies
- ROS 2 Humble
- nlohmann_json library
- hardware_interface, pluginlib, rclcpp_lifecycle
- pyserial (Python driver)

## Building

```bash
cd ~/Research/lerobot_ros2/src/ros2/ros2_ws
source /opt/ros/humble/setup.zsh
# Ensure successful mixed build by setting PYTHONPATH
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH colcon build --packages-select so101_hardware
source install/setup.zsh
```

## Tool Documentation

- [arm_calibration_transfer](docs/tools/arm_calibration_transfer.md): Legacy calibration data migration and new follower calibration file generation
- [arm_calibration_checker](docs/tools/arm_calibration_checker.md): Real-robot verification process for calibration results

## Usage

### 1. Calibrating the Arm (Python)
Calibration must be performed before first use to generate JSON files in `~/.calibrate/`.
```bash
# Calibrate Follower arm
ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

### 2. C++ ros2_control Plugin Configuration
Specify the hardware interface in your URDF:
```xml
<hardware>
  <plugin>so101_hardware/SO101SystemHardware</plugin>
  <param name="port">/dev/ttyACM0</param>
  <param name="calib_file">$(env HOME)/.calibrate/so101_follower_calibrate.json</param>
  <!-- Optional: Safe startup positions (JSON format, radians) -->
  <param name="reset_positions">{"1": 0.0, "2": 0.0}</param>
</hardware>
```

### 3. Leader Arm Publisher
Used for recording demonstration data or teleoperation:
```bash
ros2 run so101_hardware leader_arm_pub --port /dev/ttyACM0 --publish_rate 50.0
```

## Implementation Details

### Startup Positions (Reset Positions)
The `reset_positions` parameter allows you to specify initial joint positions.
- **Configured**: The arm moves smoothly to the specified pose upon activation.
- **Not Configured (Default)**: The arm preserves its current motor position without movement.

### Coordinate Conversion
The plugin handles conversion between steps and radians automatically:
- **Read**: `radians = ((steps - range_min) / range - 0.5) * 2.0 * PI`
- **Write**: `steps = (radians / (2.0 * PI) + 0.5) * range + range_min`

### Activation Rollback
When `on_activate` fails while configuring the servos, a fail-closed rollback runs via `detail::rollback_activation`:

1. **Disable torque first**: `EnableTorque(id, 0)` (with retries) is issued for every `motor_ids_`. This is the safe
   default and is independent of EPROM state. Torque-off runs before relock because some Feetech servos only accept
   the EPROM-lock command while torque is disabled.
2. **Then relock EPROM**: `LockEprom(id)` (with retries) is attempted only for servos still unlocked at the start of
   the rollback (the `unlocked_motors` set), **before the serial port is closed**. In the normal flow each servo is
   removed from the unlocked set once its configuration completes, so only servos left unlocked by a mid-stream
   failure reach the relock step.
3. **Finally close the port**: `sms_sts_.end()`.

The two outcomes are reported independently, neither masking the other:
- Torque-disable failure → `Failed to disable torque for one or more motors during activation abort`;
- EPROM relock failure → `Failed to relock EPROM for N motor(s) during activation abort; persistent parameters may be
  unprotected`, with the specific servo IDs listed in `relock_failures`.

Even when torque disable fails, relock is still attempted (both are best-effort, fail-closed), so callers can decide
whether persistent parameters need a manual reset.

### Initial Sync Read
After torque is enabled, `on_activate` performs one sync-read round to seed `hw_commands_/hw_positions_/
hw_velocities_/hw_currents_` from real feedback. This step is **also protected by the activation rollback**
(fail-closed) and is implemented in `detail::perform_initial_sync_feedback`:

1. **Transmit / bus reply (`syncReadPacketTx`)**: returns the number of bytes received into the SDK buffer; `<= 0`
   means a transmit failure or no reply (timeout). `syncReadBegin` returns `void` (it only allocates the SDK receive
   buffer and stores the timeout), so it is not the fail-closed gate — the Tx return value is.
2. **Per-motor reply (`syncReadPacketRx`)**: returns the memory byte count on success, `0` on failure. **Every**
   motor must return a full, CRC-valid packet before state is seeded and the rollback guard is dismissed (SUCCESS).

If either gate fails, `on_activate` immediately calls `abort_activation()` (torque off / EPROM relock / port close);
the rollback guard is never dismissed with incompletely-initialised `hw_commands_/positions/velocities/currents`.
Failure logs:
- Transmit / bus failure → `Initial sync read transmit failed; aborting activation`;
- A motor Rx failure → `Initial sync read for motor ID <id> failed; aborting activation`, where `<id>` is the
  first motor whose reply packet failed (independent of the EPROM `relock_failures`).

The helper takes the sync-read operations (Tx/Rx) as callbacks, so gtest can cover the Tx-failure, per-motor
Rx-failure, and all-success paths without real hardware.

## Comparison: C++ Plugin vs Python Tools

| Feature | C++ Plugin (Production) | Python Tools (Dev/Calib) |
|---------|-------------------------|--------------------------|
| Latency | Very Low (Direct) | Higher (Python overhead) |
| Performance | High (Real-time) | Medium |
| Mode | ros2_control Interface | Topic Bridge / Scripts |
| Use Case | RL / Trajectory Execution | Calibration / Recording / Diagnostics |

## Constants Configuration
Shared constants can be accessed in Python via:
```python
from so101_hardware.calibration.constants import MOTOR_IDS, JOINT_NAMES, DEFAULT_SERIAL_PORT
```

## Troubleshooting

- **Serial Permissions**: Run `sudo chmod 666 /dev/ttyACM0` or add user to the `dialout` group.
- **Missing Calibration**: If you see `Calibration file not found`, run the `calibrate_arm` tool first.
- **Empty Submodule**: Ensure you have run `git submodule update`.

## License
TODO: License declaration

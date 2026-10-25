# lekiwi_robot

The LeKiwi robot runtime: an SO-101 arm on a three-wheel omni base, sharing one
Feetech bus. It is the only package that knows LeKiwi's chassis; applications
reach it through the robot runtime contract (`robot_runtime.contract`) and
`/runtime/get_status`.

## Members and ownership

| Package | Owns |
| --- | --- |
| `lekiwi_robot` (this package) | Runtime profiles, `runtime.launch.py`, the base node, the FAST-LIO odom bridge |
| `lekiwi_sdk` | Pure chassis kinematics and wheel odometry, `OmniBase`, `MobileManipulator` |
| `lekiwi_hardware` | Thin ros2_control adapter: arm positions (IDs 1-6) and wheel velocities (IDs 7-9) on one bus |
| `lekiwi_description` | Assembled URDF and the MID-360 mount (`config/lekiwi_mid360_mount.yaml`) |
| `lekiwi_calibration` | MID-360 / front camera extrinsic calibration workflow |
| `feetech_sdk` | Feetech servo access, FTServo pinned at `06fd3356` (shared with SO-101) |
| `so101_description`, `so101_sdk`, `so101_motion` | The arm. The LeKiwi arm *is* an SO-101; nothing is forked, and the independence gate lists these three by name |

Arm limits come from the calibration file; the URDF only carries the mechanical
envelope. Wheels have no position limits, only `base.max_wheel_radps`.

## The chassis lives in the profile

The profile's `base:` section is the only source of chassis geometry:
`wheel_joints`, `wheel_radius`, `base_radius`, `max_wheel_radps`,
`control_frequency`, `odom_frame`, `base_frame`, `publish_tf`, and
`navigation_enabled_on_startup` (defaults to `false`). A missing key fails
startup with its name (`lekiwi_robot/base_profile.py`); there are no built-in
LeKiwi constants to fall back on.

`capabilities.base.cmd_vel` (`max_vx`, `max_vy`, `max_wz`, `staleness_s`) is
published in `RuntimeStatus`. The declared limits must be realizable: the
runtime computes the envelope from the SDK kinematics and rejects a profile
whose limits exceed it, naming the axis. For the shipped geometry that is
0.2657 m/s in x, 0.2301 m/s in y and 1.84 rad/s in yaw; the profiles publish
0.26 / 0.23 / 1.5.

## Base node (`lekiwi_robot/base_node.py`)

- `/cmd_vel` (`Twist`) -> `base_velocity_controller/commands` (wheel rad/s)
  through `lekiwi_sdk_py.body_to_wheel_velocities`; commands beyond
  `max_wheel_radps` are scaled down preserving direction. Commands are zeroed
  after `staleness_s` without a fresh one.
- `/joint_states` wheel positions -> `/odom`, plus `odom -> base_link` TF.
  Positions are unwrapped with `lekiwi_sdk_py.wheel_position_deltas`. When the
  runtime also runs FAST-LIO with its TF enabled, FAST-LIO owns that TF and the
  base node publishes `/odom` only, so the transform has one publisher.

### Mode and navigation gate

Two independent conditions admit a base command, and the runtime stop latch
overrides both:

| Layer | Set by | Prevents |
| --- | --- | --- |
| `base_navigation` mode (`allows_base: true`) | `/runtime/set_mode` | Arm and base controllers claiming the same joints |
| Navigation gate (`base.navigation_gate`) | `/motion_mode/set_navigation_enabled` (`SetBool`), acknowledged on `/motion_mode/base_navigation_enabled` | A planner driving the base before an operator armed it |

The gate starts closed and fails closed until the first `RuntimeStatus`
arrives. Enabling or disabling it zeroes the target and publishes a zero wheel
command before the acknowledgment; nothing is queued across a closed gate.
Both names are absolute. The former application handshake used the same names
relative (they coincide only in the root namespace) plus an enable topic,
which the runtime does not serve.

## Arm motion

`so101_motion` plans for the SO-101 arm alone. The runtime renders
`motion.model_xacro` (the SO-101 arm URDF) with this profile's hardware
arguments and hands it to move_group, which consumes the arm-only
`arm_joint_state_broadcaster`. Base-only profiles declare no `motion` section
and start no motion services.

## Supported profiles

Only `lekiwi_lidar` (base only: modes `idle`, `base_navigation`) and
`lekiwi_nav_grasp` (arm + base: adds `stream`, `trajectory`) are shipped. The
other LeKiwi configurations in `robot_config` still use the legacy
provider-less bring-up.

## Running standalone

```bash
ros2 launch lekiwi_robot runtime.launch.py profile:=lekiwi_lidar [simulated:=true]
```

`simulated:=true` keeps the production controller and hardware-adapter path
and replaces only the serial bus with the SDK's simulated transport.

## Gazebo simulation

The Gazebo platform (`sim_platform:=gazebo`) stands in for the physical
sensors: the runtime renders `gpu_lidar` and IMU sensors into the model at
the MID-360 mount (`lekiwi_mid360_mount.yaml`), bridges them through
`ros_gz`, and serves the advertised endpoints (`/livox/lidar`,
`/livox/imu`) on simulation time. Use a fresh `IGN_PARTITION` **and** the
same value in `GZ_PARTITION` (plus `IBROBOT_GAZEBO_HEADLESS=1` without a
display) so runs on a shared host stay isolated: this Fortress install
(`ign gazebo`, gz-transport11) reads only `IGN_PARTITION` and ignores
`GZ_PARTITION`, while newer gz reads `GZ_PARTITION` — setting both covers
either stack.

The launch entries the verification runs use:

```bash
# provider alone (sensors, admission, wheels): task 2.5
ros2 launch lekiwi_robot runtime.launch.py profile:=lekiwi_lidar \
  sim_platform:=gazebo

# composed through the robot_config entry: task 4.2
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_lidar \
  use_sim:=true sim_platform:=gazebo

# navigation (FAST-LIO + Nav2/AMCL): task 5.2 stages the lekiwi_lidar
# profile and the navigation overlay side by side, then launches
ros2 launch robot_config robot.launch.py config_path:=<staged>/lekiwi_navi_sim.yaml \
  use_sim:=true sim_platform:=gazebo nav_stage:=navigation
```

TF ownership in simulation (one publisher per transform):

| Transform | Owner |
| --- | --- |
| `odom → base_link` | wheel odometry (`base_node`), on simulation time; the FAST-LIO bridge publishes `/odometry/filtered` with `publish_tf: false` |
| `map → odom` | AMCL (the navigation stage); never a static transform |
| `base_link → livox_frame` | static TF from the Gazebo embodiment (the rendered model has no sensor link); camera frames stay on `robot_state_publisher` |
| FAST-LIO's own TF | isolated on `/fast_lio/tf_raw` |

The simulated IMU publishes SI units (m/s², rad/s); at rest it reads gravity
(`|acc| ≈ 9.81` m/s² along +z of the sensor frame). Gazebo's pre-settle
samples — the zero window before the physics starts and the landing impulse
when the entity drops onto the ground — are held back by
`robot_runtime/imu_settle_filter` until gravity has been stable for a full
run, so FAST-LIO's gravity initialization only ever sees valid samples.

`lekiwi_sensor_calib` (the MID-360 / camera extrinsic calibration workflow)
needs the physical sensors and is not supported on Gazebo; the simulated
sensors always sit exactly at the mount.

### Lidar-odometry regression gate (D14)

`robot_navigation/test/test_lidar_odometry_gate.py` is the committed FAST-LIO
accuracy gate for this runtime. It lives in `robot_navigation`, not here:
it launches the generic application entry, and this package may not declare
test dependencies on generic packages (the independence gate below audits
the whole dependency closure, test dependencies included). It runs the
production navigation stage (staged from the gate's own
`test/config/lekiwi_lio_sim.yaml` overlay) in the shipped
`nav_room` scene and, with no Nav2 goal active, drives a scripted closed
route on **`/cmd_vel_nav`** — the production chain's entry point — so the
velocity smoother and the collision monitor stay in the loop
(`controller → /cmd_vel_nav → velocity smoother → /cmd_vel_smoothed →
collision monitor → /cmd_vel`). The gate checks that `/cmd_vel` is published
only by the collision monitor and by Nav2's behavior server (its recovery
behaviors write the channel directly, bypassing the smoother and the
monitor — a property of the shipped chain; with no goal active the
behavior server stays silent, and the gate's message-level pairing proves
the monitor is the only publisher that actually drives the base). Every
non-zero command must be relayed unchanged.

The route (west half of the room, 7.0 m of translation, two in-place turns,
two lateral segments, closed; commanded at vx 0.10 / vy 0.08 / wz 0.35 —
below the smoother limits):

| # | Command | From → to (x, y, yaw) |
| --- | --- | --- |
| 1 | forward 0.8 m | (−1.5, −1.5, 0°) → (−0.7, −1.5, 0°) |
| 2 | rotate +90° | → (−0.7, −1.5, 90°) |
| 3 | forward 2.7 m | → (−0.7, 1.2, 90°) |
| 4 | lateral +y 0.8 m (body) | → (−1.5, 1.2, 90°) |
| 5 | rotate −90° | → (−1.5, 1.2, 0°) |
| 6 | lateral −y 2.7 m (body) | → (−1.5, −1.5, 0°) |

Segments close on simulator ground-truth progress (each stops commanding at
the velocity smoother's stopping distance), and every metric is measured in
simulation time. FAST-LIO's `/odometry/filtered`, aligned to ground truth at
the route start, is scored against the committed thresholds
`min(cap, max(2 × worst of five measured runs, floor))`; the spec caps are
ATE RMSE ≤ 0.10 m, max error ≤ 0.20 m, final error ≤ 0.15 m, final yaw error
≤ 3°. Wheel odometry (`/odom`) and AMCL (`/amcl_pose`) are printed and
loosely bounded, never part of the FAST-LIO verdict. Health during the
route: `/odometry/filtered` must carry at least 95 % of the `/livox/lidar`
frames FAST-LIO consumed over the same simulation-time span (the whole
drive, final settle included), with no adjacent stamp gap above 0.35 s —
Gazebo's `gpu_lidar` occasionally renders an empty scan, which FAST-LIO
skips; the fraction floor absorbs those, the gap bound catches a real
stall; no `fastlio_mapping` death or respawn,
no odometry sample stamped ahead of the latest `/clock` received before it (the odom bridge's own rejection log must stay empty), and the TF ownership table above holds
on the live graph. Plots and raw samples go to the test's log directory.

```bash
NAV_TEST_PROFILE=full colcon test --packages-select robot_navigation \
  --pytest-args -k lidar_odometry_gate
```

The gate skips with a reason listing every missing capability on platforms
without ground truth, profile support, lidar simulation or a scene template
(today: everything except Gazebo, e.g. MuJoCo).

Conformance and the independence gate:

```bash
scripts/verify_runtime_independence.sh runtime lekiwi_robot --profile lekiwi_lidar
scripts/verify_runtime_independence.sh runtime lekiwi_robot --profile lekiwi_nav_grasp
```

The base velocity, odometry, staleness and navigation-gate tests are in scope
and must execute, not skip. Use a fresh `ROS_DOMAIN_ID` (at most 232): a stale
runtime left on a shared domain answers for the one under test.

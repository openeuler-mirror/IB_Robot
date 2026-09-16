# so101_robot

The SO-101 robot runtime: the independently deployable unit through which the
SO-101 joins IB-Robot (`robot-runtime-packaging` spec, design D1).

## What it is

A metapackage whose dependency closure is everything needed to run the robot:

| Member | Role |
|---|---|
| `so101_description` | URDF / meshes (`simulated` xacro arg selects the SDK fake bus) |
| `so101_sdk` → `feetech_sdk` | ROS-free hardware SDK |
| `so101_hardware` | thin ros2_control adapter over the SDK |
| `so101_motion` | MoveIt assets + motion server (`/motion/*` services) |
| `robot_runtime` | contract library + runtime facade + conformance suite |
| `ibrobot_msgs` | contract messages |

Plus stock ros2_control controllers. No generic IB-Robot package is in the closure.

## Running standalone

```bash
ros2 launch so101_robot runtime.launch.py profile:=so101_single_arm            # hardware
ros2 launch so101_robot runtime.launch.py profile:=so101_single_arm simulated:=true   # SDK simulated transport
```

`profile` is a path or the name of a file in `profiles/`. The profile is the
single source of runtime parameters: port, calibration, controllers config,
modes → controller sets, capabilities with parameters (incl. stop latency
bounds), command channels, trajectory endpoints, motion frames.

The runtime reaches `ACTIVE` when the initial mode's controllers are active
and publishes `/runtime_status`; `/runtime/set_mode`, `/runtime/stop`,
`/runtime/get_status` and `/motion/*` are then available.

## Conformance

```bash
CONFORMANCE_LAUNCH="ros2 launch so101_robot runtime.launch.py profile:=so101_single_arm simulated:=true" \
CONFORMANCE_PROFILE=$(ros2 pkg prefix so101_robot)/share/so101_robot/profiles/so101_single_arm.yaml \
python3 -m pytest src/robot_runtime/test/test_conformance.py -q
```

This certifies the exact launch graph above (design D8), not a stand-in.

## Scope: single arm only

`so101_robot` ships exactly one supported deployment: the single SO-101 arm
profile `so101_single_arm`. Dual-arm composition (a second hardware instance,
extra joint groups, bimanual teleoperation, coordinated planning or inter-arm
collision avoidance) is **not** supported, tested or advertised by this
package; selections this runtime cannot serve fail explicitly rather than
degrading to single-arm assumptions.

## Public surface (so101-runtime-migration)

Everything application-side consumes the generated public interface
description (`RuntimeStatus.interface_description_json`, schema v1,
`docs/robot_interface_schema.md`):

- **Cameras** — profile `peripherals:` declares the robot's sensors. USB
  cameras run on `usb_cam` (spelling `driver: opencv`), RealSense on
  `realsense2_camera`; color, enabled depth/aligned-depth and CameraInfo are
  published under their declared profiles, pointcloud is opt-in. In simulated
  transport, synthetic publishers serve the same topics with typed payloads
  and the camera monitor measures real readiness from received samples.
- **Model metadata** — the description's `model` projection carries named
  joint groups, admissible physical limits, semantic home positions, motion
  frames and the `joint_conversions` table with its fingerprint. Inference,
  recording and offline conversion consume this projection; private
  calibration files never leave the runtime. Semantic home is served by the
  explicit HOME action, never by startup motion.
- **Teleoperation** — operator input devices live in
  `config/teleop_inputs.yaml` (deployment data, separate from follower
  hardware). Leader, gamepad, phone and VR inputs map onto the public
  `motion.arm.*` interfaces (pose/linear/angular/joints/lease topics,
  start/stop services, HOME action). The runtime-owned Placo servo is the only
  writer of physical controller commands; managed producers admit exclusively
  (teleop `stream` vs policy `policy_stream` are distinct runtime modes and
  cannot be active together), input loss or stop latches require explicit
  re-arm, and every input path triggers the bounded robot-side hold/stop.

The profile's `teleoperation.command_stale_s` is the managed input, lease and
joint-feedback deadline (including HOME feedback). It defaults to 0.5 s for the
SO-101 board and must remain in `(0, 1]`. The leader input inherits this deadline;
operator configurations may request a tighter input bound. Expiry still triggers
HOLD on the first control tick, with no additional consecutive-failure delay.
Phone and VR retain their stricter device-local deadman deadlines.
`runtime_status_stale_s` (default 2.5 s) is a separate liveness budget: the
facade publishes RuntimeStatus as a 1 Hz heartbeat, so its freshness threshold
is derived from that period (~2.5x) and is intentionally not capped by the
1 s command-stream contract.

Teleop admission waits for the controller switch without blocking joint feedback
or stop callbacks. After the switch, it seeds the new session from the latest
accepted joint state. If that state has expired, it waits up to one feedback
window for a new sample without publishing any commands. Actual feedback loss
still refuses admission and requests HOLD. Stops, emergency stops and shutdown
preempt a pending start. An outstanding
mode request remains fenced until its response arrives, so a late response cannot
resume motion or overlap a new start.

For a HOLD during episodic recording, inspect the first `runtime HOLD requested`
or `Disabling managed teleop` warning. The former reports input, lease, feedback
and runtime-status ages; leader-input warnings also identify rejected samples.
`runtime HOLD confirmed` only acknowledges the stop. Before each managed leader
episode, `record_cli` explicitly clears to idle, rearms and checks fresh stream
status. It does not rearm in the background during recording. An episode with
missing or interrupted arm/gripper commands is aborted and discarded, even if
images and joint feedback were saved.

## Application configuration

`robot_config/config/robots/so101_single_arm.yaml` selects this runtime
(`runtime.provider: so101_robot`, `simulation.platform: sdk`) and binds its
contract to the public interfaces above. The retained
`so101_single_arm_legacy.yaml` is the provider-less in-config bring-up used
by unmigrated overlays (e.g. `so101_arm_aero_hand.yaml`) until their own
migration; it is not the new-path configuration.

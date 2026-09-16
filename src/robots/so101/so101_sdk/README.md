# so101_sdk

ROS-free SO-101 arm SDK built on `feetech_sdk`.

## Responsibility

- Follower arm lifecycle: connect (calibration load/validate) → activate (motor configuration + initial feedback sync, rollback on failure) → read/write/hold/stop → deactivate
- Calibration ownership: the SDK's `Calibration::load` is the format authority (SO-101 keyed format + LeKiwi id-field format, both validated with fail-fast on missing file/joint/field)
- Named-joint SI state (radians, rad/s, amperes — callers never depend on array ordering)
- Read-only leader arm (no command surface, never torque-enables follower control)
- Multi-instance independence on distinct buses
- Shared-bus composition via `attach_shared_bus` (scoped operations: arm reads do not fail when a wheel motor drops off)

## Prohibited

- ROS dependencies (links only `feetech_sdk` + `nlohmann_json`)
- MoveIt/planning logic (lives in the robot suite)
- Interactive calibration tooling (the proven Python tool stays in `so101_hardware`)

## Python Bindings

`so101_sdk_py` (pybind11): exposes `Arm`, `ArmConfig`, `LeaderArm`, `SimControl` for Python runtime nodes and tools. `read()` returns plain dicts `{joint: {"position":…, "velocity":…, "effort":…}}` or `None` on failure.

Bindings are a required build artifact (`python3-dev`, `pybind11-dev`), not an
optional feature silently omitted from a successful build. Python objects own
and close their bus. `LeaderArm.disconnect()` explicitly ends read-only input;
`Arm.deactivate()` attempts torque release before closing and reports failure.

### Leader Provisioning Contract

Use the separate, operator-invoked `so101_hardware` `calibrate_arm --arm leader`
workflow to provision firmware and produce the calibration JSON before ordinary
acquisition. Neither `LeaderArm.connect()` nor `read()` writes calibration or
changes torque. The caller must set `LeaderConfig.calibration_version = 1`
(default zero is rejected), specify a complete calibration file, and select
`joint_order` plus `gripper_joint`. The version identifies the existing keyed
JSON schema, so provisioned legacy files do not need to be rewritten.

`connect()` validates integer offsets/ranges, nonzero travel, drive mode
(boolean or integer 0/1), and matching firmware offset/range/position-mode
registers. Missing, malformed, incomplete, mismatched or unreadable calibration
fails with `health().detail`; no automatic provisioning is attempted. The arm
positions use the 4096-tick centered-radian convention. Only the gripper
`position` is a dimensionless opening ratio in `[0, 1]`, with drive-mode inversion.
Every read returns all named joints or no sample; it never replays cached data.

### Activation And Stops

Default follower activation acknowledges a measured-position goal before torque
enable, then seeds all command targets from feedback. `reset_positions` is a
legacy explicit startup-motion override, not semantic HOME. New runtime profiles
must omit it and keep HOME in model metadata for an explicitly admitted operation.
`motor_ids` maps existing nonnumeric URDF joint names to numeric calibration keys;
numeric names remain supported without a mapping. Duplicate or invalid IDs fail.

`Arm.hold()` retains all last commanded targets (including unchanged joints after
a partial write). Runtime HOLD is stronger: the ros2_control adapter's controller
stop hook replaces pending goals with measured positions. TORQUE_OFF keeps the
bus available for inactive feedback but rejects SDK writes until activation.
Reactivation discards stale goals. A failed release is reported, never marked safe.

Synchronized broadcast writes can report serial transport failures but do not
receive per-motor acknowledgements. Physical goal/torque confirmation and stop
latency remain separate hardware acceptance tests.

## Testing

Focused CTest/gtest and Python binding tests run headlessly against simulated
transport. This is protocol/lifecycle coverage, not physics or hardware evidence.

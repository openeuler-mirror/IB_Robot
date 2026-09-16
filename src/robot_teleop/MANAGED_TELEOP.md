# Managed Single-Arm Teleoperation

The provider-backed SO101 path loads operator deployment from
`so101_robot/config/teleop_inputs.yaml`. The application selects one
`active_device` and explicit `target.interfaces` logical IDs. Policy-only
startup does not construct an input source or open its serial device.

The optional `so101_hardware/leader_arm_pub` source owns SDK acquisition.
Provision with the separate `so101_hardware calibrate_arm` workflow first.
Its input configuration must explicitly accept `calibration_version: 1`.
Ordinary connect only validates and reads firmware calibration; it never writes
calibration, configures follower goals, or enables torque. Invalid or incomplete
reads emit no sample. The public source JointState is marked
`header.frame_id: leader_radians_gripper_ratio_v1`: arm positions are radians;
the explicitly named gripper is an opening ratio. It is not a follower state.
Set `acquisition.enabled: false` to run the source on another host. Input and
runtime clocks must be synchronized for stamped freshness validation.

The generic mapper resolves arm order, physical limits, gripper closed/open
conversion, Cartesian frames and endpoints from one bound public description.
Only the SO101 runtime constructs Placo with its effective rendered model.
The `/motion/arm/joints` intent receives stamped SI targets; no migrated input
publishes directly to physical position controllers. Cartesian pose intent is
clutch-relative displacement and a base-frame rotation delta, not absolute pose.

## Admission and Recovery

Startup is idle and does not follow or HOME. For leader following, explicitly
call the mapper's `~/rearm` Trigger service after fresh input is available.
The service awaits the final runtime admission response, not merely a queued
request. For leader input it first clears the cache and requires a valid sample
produced after the rearm request, then checks freshness again on admission.
Receipt age and stamped source age must both fit the input freshness budget.
Concurrent rearm calls are rejected. A separate reentrant service callback group
and awaited rclpy Futures leave source/status subscriptions, E-stop, control
timers and service replies runnable even with a SingleThreadedExecutor.
Timeout or STOP fences late start replies; a late success requests another stop,
and an older stop acknowledgement cannot acknowledge that newer request.
Phone/gamepad callers retain the queued boolean backend `enable()` API and their
operator deadman/release requirements; `enable_with_result()` is the final-result
path used by the mapper service.
Gamepad A, phone deadman, or a VR trigger release/press requests admission.
HOME is an explicit admitted action (mapper `~/home`, gamepad X, phone HOME,
or VR secondary button). It requires fresh input throughout execution.

One source may be selected. Runtime start refuses a second admitted session,
policy_stream, trajectory, missing feedback, or a stop latch. Joint, pose and
velocity intent modes cannot interleave during one admitted arm session; a
gamepad mode switch stops the old session and requires rearm.

Input loss, deadman release, stale runtime/robot feedback or ownership loss
discards cached targets and requests robot-side `StopRuntime(HOLD)`. The
robot-side watchdog remains active even if the mapper process disappears.
A stop service reports success only after HOLD confirmation. The configured
input timeout is checked every servo control period; subsequent execution-stop
latencies belong to the runtime's published `runtime.stop` bounds.

Reconnection does not clear a runtime stop. Explicitly request
`SetRuntimeMode(idle)` to clear the latch, then rearm with a fresh reference.
Policy/teleop handover must stop the old producer, confirm idle/clear, and only
then admit policy_stream or teleop stream. The guarantee covers framework-managed
producers, not arbitrary external DDS writers bypassing the public input path.

Providerless configurations retain the legacy device and solver launch paths.
Tests are device-free; simulated/hardware HOLD latency remains a separate
runtime conformance measurement, not a claim derived from unit tests.

## Timing Parameters and Local Measurement

TeleopNode accepts these ROS parameters (the launch builder propagates device
overrides before teleoperation-level settings):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `latency_warn_s` | `0.0` | Zero selects `1/control_frequency`; positive values set the wall-time warning budget. |
| `diagnostics_period_s` | `1.0` | Minimum monotonic elapsed interval between loop diagnostics/warnings. |
| `rearm_timeout_s` | `5.0` | Total bounded source-wait plus runtime-admission wait. Uses steady time, independent of ROS simulation time. |
| `runtime_status_stale_s` | `2.5` | RuntimeStatus heartbeat freshness (servo node and motion server). RuntimeStatus is a 1 Hz liveness heartbeat, so the budget is ~2.5x the publish period and must stay well above one heartbeat gap; it is deliberately NOT bounded by the <= 1 s command-stream (`command_stale_s`) contract. |

Frequency, diagnostic interval and rearm timeout must be finite and positive;
the latency setting must be finite and non-negative. Loop timing uses
`perf_counter` and measures successful read/filter/publish callback wall duration,
not CPU utilization or timer scheduling delay. Statistics retain the EMA and
lifetime maximum; diagnostics are emitted on completed publish cycles.

A local host benchmark of 20,000 six-joint all-clipped calls, with logging output
disabled, measured **66.69 us wall / 66.64 us CPU before** and
**3.80 us wall / 3.80 us CPU after**. A separate 2,000-call cProfile pass attributed
0.210 of 0.274 seconds to NumPy scalar `isclose` and 0.037 seconds to scalar
`clip`. Logging was already rate-limited (first three and every hundredth clip).
Scalar arithmetic removes the measured NumPy overhead without vectorization;
clip counts retain the same asymmetric tolerance. Non-finite targets reject the
whole command, including joints without configured limits; malformed bounds fail
at initialization. Managed control requests stop when the filter rejects a command.

Reproduce locally with `python3 src/robot_teleop/test/benchmark_safety_filter.py`
after loading `.shrc_local` in a clean shell. The original measurement used a
list comprehension to drive calls; the script uses a loop to avoid result-list
allocation. These are host microbenchmark numbers, **not board speedup claims**
and not an explanation of the reported 10-18 ms end-to-end loop wall duration.
`test_managed_rearm_executor.py` uses real ROS service/subscription delivery with
synthetic input and fake admission services only; it never connects to a robot.

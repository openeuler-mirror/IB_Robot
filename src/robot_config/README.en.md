# robot_config

Unified robot configuration system for ros2_control and peripherals.

## Overview

This package provides a unified configuration system for robot hardware that bridges:

- **ros2_control**: For joint/motor control interfaces
- **Peripherals**: For cameras and other devices (via existing ROS2 drivers)
- **tensormsg**: For ML policy I/O contracts

The goal is to have a single source of truth for robot hardware configuration, eliminating duplication between different configuration systems.

## Features

- **Single YAML configuration**: Define ros2_control, cameras, and ML contracts in one file
- **Uses existing ROS2 camera drivers**:
  - `usb_cam` for USB cameras (OpenCV-based)
  - `realsense2_camera` for RealSense D400 series
- **TF publishing**: Automatic camera frame transform publishing
- **Calibration support**: Standard ROS2 camera_info_manager integration
- **tensormsg integration**: Contracts reference peripherals by name

## Architecture

```
robot_config YAML (single source of truth)
        │
        ├───► ros2_control (joints/motors)
        │       └───► so101_hardware plugin
        │
        ├───► Camera drivers (existing ROS2 packages)
        │       ├───► usb_cam (USB cameras)
        │       └───► realsense2_camera (RealSense D400)
        │
        └───► tensormsg contracts (ML I/O)
                └───► PolicyBridge / EpisodeRecorder
```

## Configuration Example

```yaml
robot:
  name: so101_single_arm
  type: so101
  robot_type: so_101

  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  peripherals:
    - type: camera
      name: top
      driver: opencv  # Uses usb_cam package
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame

    - type: camera
      name: wrist
      driver: realsense  # Uses realsense2_camera package
      serial_number: "12345678"
      width: 640
      height: 480
      fps: 30
      depth_width: 640
      depth_height: 480
      frame_id: camera_wrist_frame

  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top  # References camera above
        image:
          resize: [480, 640]
```

## Robot Configuration Overlays

When a robot configuration only extends an existing robot, declare `base_config` under `robot` instead
of copying the complete YAML:

```yaml
robot:
  name: so101_with_extension
  base_config: so101_single_arm
  teleoperation:
    devices:
      __append__:
        - name: extension_device
          type: custom
```

`base_config` may reference only a sibling robot YAML. Relative references resolve from the overlay file
and gain a `.yaml` suffix when omitted; the sibling boundary keeps relative resource paths unambiguous.
Mappings merge recursively, while scalars and lists replace by default. A list is appended only by a mapping
whose sole key is `__append__`. The loader rejects cycles, mapping/list container type mismatches, and
ambiguous append mappings. Normalized output keeps the outer file in `_config_path` and the complete base-to-overlay
provenance chain in `_config_sources` for diagnostics and audits.

## Capability Gateway Public Contract

Versioned packages under `skill_catalog/config/skills/` are the public Capability Gateway SSOT. Robot YAML selects an
exact catalog source/profile and retains only robot execution context such as poses, limits, timeouts, and endpoints.
Capability metadata is explicit and is not derived from a primitive sequence. Every enabled catalog package must declare
`schema_version: 1`, `summary`, `domain`, `moves_robot`, `required_control_mode`, `parameters`, and
`recovery_policy`.

| Field | Validation |
| --- | --- |
| `summary` / `domain` | Non-empty strings |
| `moves_robot` | Boolean |
| `required_control_mode` | `teleop`, `model_inference`, or `moveit_planning`, exactly equal to global `skill_required_control_mode` |
| `parameters` | Strict object schema: `type: object`, `additionalProperties: false`; only `target_name`, `container_name`, `place_name`, `motion_direction`, and `motion_distance`; unique `required` entries that name declared properties |
| `recovery_policy` | `never_retry`, `ask_user`, or `recover_safe_pose` |

String parameter definitions permit only `type` and a non-empty `enum`; a `motion_direction` enum is limited to
`forward`, `backward`, `left`, `right`, `up`, and `down`. `motion_distance` must be a `number` with
`exclusiveMinimum: 0` and a `meters` or `degrees` `unit`. Any other schema key or public request property is rejected.

`load_robot_config_dict()` is the canonical normalized loader for robot execution context. `skill_catalog` compiles the
selected profile and validates capability, implementation, delegated-executor, and Gateway invariants. Inline
`embodied.skill_templates` is not a runtime source of skill identity. Model-driven delegated executors additionally
declare `model_bundle_path` and `model_deployment`; all runtime participants load the same strict
`inference_manifest.json`, and startup/catalog compilation fails if that identity cannot be verified.

Shared config selection is ordered as explicit `config_path`, explicit `config_name`, `ROBOT_CONFIG`, `ROBOT_NAME`,
then `so101_single_arm`. A name resolves first from the installed `robot_config/config/robots/` directory and then
from the source `config/robots/` directory; an explicit path must exist. The public catalog exposes only capability
fields, named-pose names, timeout policy, and a digest. Primitive sequences, joint/cartesian coordinates, target
bindings, and ROS service/action/topic names remain private implementation data.

## Observation Video Transport

Observation transport is part of the contract. Omitting `transport` preserves the existing DDS image path. Explicit
RTP is valid only for `sensor_msgs/msg/Image` observations using `rgb8` or `bgr8` in a distributed pipeline, and it
fails closed rather than falling back to DDS.

```yaml
contract:
  observations:
    - key: observation.images.top
      topic: /camera/top/image_raw
      type: sensor_msgs/msg/Image
      image: {resize: [480, 640], encoding: rgb8}
      transport:
        mode: rtp
        stream_id: top
        endpoint: {host: 192.168.10.20, port: 5004}
        codec: h264
        encoder_backend: nvidia
        decoder_backend: ascend
        h264: {profile: main, bitrate_bps: 4000000, gop_frames: 15}
        media: {width: 640, height: 480, frame_rate_hz: 30, pixel_format: nv12,
                color_space: bt709, color_range: limited}
        buffer: {sender_queue_frames: 2, receiver_queue_packets: 256,
                 decoded_frame_capacity: 32, retention_ms: 1000}
        readiness: {keyframe_timeout_ms: 3000, timestamp_mapping_max_age_ms: 1000,
                    max_inter_camera_skew_ms: 50}
        security: none
```

Stream IDs and endpoint port pairs must be unique. The contract fingerprint includes endpoint, codec/media
reconstruction semantics, buffering/readiness limits, and image metadata, so edge and cloud must deploy matching
configuration. `software`, `ascend`, and `auto` backend policies are resolved independently on each host; an
unavailable explicit backend rejects startup. RTP/UDP has no authentication, confidentiality, or integrity and is
restricted to a trusted robot network. Use an explicit matching `mode: dds` contract for rollback or DDS-based
recording. Development examples are `config/robots/dev_rtp_single_camera.yaml` and
`config/robots/dev_rtp_multi_camera.yaml`; production profiles remain DDS by default.

`nvidia` is currently encoder-only. A typical split uses `encoder_backend: nvidia` on an RTX edge host and
`decoder_backend: ascend` on the 310B cloud. Encoder `auto` probes `ascend`, `nvidia`, then `software`; an
unavailable explicit backend rejects startup.

## Control Mode Configuration

The robot_config package supports dual control modes for different AI model requirements:

### Available Control Modes

#### 1. teleop Mode (Human Teleoperation)

**Use for:** Human teleoperation devices (leader arm, gamepad, VR device)

**Characteristics:**
- Real-time direct control
- Zero-latency passthrough (< 5ms)
- Multiple input device support
- Built-in safety filters (joint limits)

**Configuration:**
```yaml
robot:
  default_control_mode: "teleop"

  control_modes:
    teleop:
      description: "Human teleoperation mode (direct control)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: false

  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"
```

`active_devices` may still combine independent leader/joint inputs. The current
`target.arm_command_topic` and `target.gripper_command_topic` together define the
controller-ownership boundary: every active device must exclusively own both topics
that it actually publishes. Launch validation rejects sharing either topic. SO-101
VR takes its gripper topic from `vr_config.so101_gripper_topic`, while its Placo arm
output currently uses `/arm_position_controller/commands`. Multi-arm configurations
must therefore separate both arm and gripper topics, not only the arm topic.
The current
SO-101 Placo path is one execution resource per target arm, so a launch may select
at most one Phone, SO-101 VR, or Xbox Cartesian input.

Phone devices use only the built-in WebPhone transport; when specified,
`phone_config.backend` must be `webphone`. HTTPS/WSS
ports, TLS files, and `command_stale_s` live under `phone_config.web`. The launch
builder validates ports, timeout, and certificate/key pairing, and preserves the
device-level `control_frequency` when constructing the teleop node.
WebPhone also requires `command_stale_s + 1 / control_frequency <= 0.22s`.
This bounds stale detection and stop-request dispatch, not the arm's physical
stopping time. The stale timeout defaults to 0.18 seconds; a 50 Hz device may
explicitly use 0.2 seconds.
WebPhone uses the Placo clutch-relative pose contract. The browser supplies either
WebXR AR poses or optical-flow virtual poses when WebXR is unavailable. Neither
tracking path differentiates pose into velocity. Phone therefore requires `teleoperation.cartesian.solver:
placo_servo` and has no velocity-mode switch. A matching legacy `input_mode: pose`
remains readable, while `velocity` fails before launch. Because Phone commands are
relative to the clutch baseline, every `end_effector_bounds` axis must satisfy
`min < 0 < max`.
The Phone Placo path reads `position_only` from
`teleoperation.cartesian.placo_servo.position_only`. The launch builder passes the
resolved setting to the Phone Placo path. Teleop Home has one target source:
`ros2_control.reset_positions`, validated and injected in selected-arm joint order.
Phone and SO-101 VR call the same `/so101_placo_servo_node/return_home` action. Its
terminal result requires fresh `/joint_states` samples to remain within the maximum
joint-error tolerance; it does not use a Cartesian named pose, status topic, or fixed
settle delay. The
launch builder also injects `command_stale_s` as a Phone-only Placo command lease;
the YAML default keeps that lease disabled for VR/Xbox. Phone refreshes the lease
only after a valid command is obtained for the current cycle or
while a controlled Home is active. Empty input and SDK/conversion failures cannot use
process liveness as a substitute for a valid command. Home requires finite
`reset_positions` for every selected arm joint and values inside the Placo joint limits.
Phone launch rejects MoveIt Servo because the Phone contract is pose-only and Home
is owned by Placo. After any Home terminal result the
operator must release and re-press deadman; the gripper holds its last target while Home
runs. The launch builder also injects `safety.estop_topic` into both Placo and the
standalone VR node, preserving `E-stop > Home/start/pose/twist` without relying on
TeleopNode to relay the stop. Phone pose mode must also
enable at least one tracking source: WebXR AR or optical flow.

WebPhone has no account authentication and is supported only on a trusted internal
network. Origin and single-client checks do not establish operator identity. Do
not use public forwarding, reverse/cloud tunnels, guest Wi-Fi, or untrusted VPNs;
restrict HTTP/WSS access to the robot control subnet and stop the service when idle.

For multiple inputs, `active_devices` may select devices whose final command topics are disjoint.
Explicit `target.publish_groups` define ordered command groups; `target.actuator` reuses an auxiliary
actuator's joint order and command topic without duplication. Launch rejects duplicate topic ownership.
Non-`ros2_control` hardware belongs under `auxiliary_actuators`; its configured driver and channel count
are target-independent. Aero Hand joints stay in `joints.hand`, outside `joints.all` and the unified
`/joint_states` stream. External components may declare `active_control_modes`; simulation skips real
`mock: false` devices while retaining explicit mocks. The Aero Hand configuration limits both devices
to real `teleop` launches and does not silently fall back to mock mode there. Set the external SDK path
and calibrate before launch:

```bash
export MHANDPRO_SDK_LIB=/absolute/path/libVDMocapSDK_mHandPro.so
export AERO_HAND_RIGHT_PORT=/dev/serial/by-id/<aero-hand-id>
ros2 run robot_teleop calibrate_glove --side right --lib-path "$MHANDPRO_SDK_LIB" \
  --raw-output ~/.calibrate/aero_hand_right_sdk_capture.json
```

The default full calibration writes `feature_schema: aero_compact_v3` and uses 20 skeleton nodes plus
five vendor virtual fingertips. User calibration stores neutral/active endpoints for four raw thumb
features plus the MCP/IP projection, while robot_config owns the MCP/IP tendon weights, Aero output
coverage, deadbands, and maximum steps. Hardware validation maps palm-local pitch/yaw to CMC
abduction/flexion, while
MCP/IP flexion drives only the combined tendon axis. Vendor pose offsets do not survive an
SDK process restart. `hand_sources.mhandpro` owns acquisition and publishes reusable `HumanHandState`;
the `hand_retarget` device converts that state to Aero or another target hand. Each retarget device
declares its side and may bind `source_name`; consumers also require the `human_hand_geometry_v1`
schema so cross-hand or cross-schema wiring fails closed. Real output therefore
remains locked after teleop startup until P-pose alignment and automatic frame quality checks pass.
With `startup_p_pose: interactive`, unified launch prompts once before hardware starts and the source
then calibrates in the same SDK process. The default `startup_p_pose: manual` instead uses
`ros2 run robot_teleop calibrate_glove --side right --runtime-service
/hand_sources/mhandpro/calibrate_p_pose` in that same worker. After retaining
the raw capture, use
`ros2 run robot_teleop analyze_glove_capture --input
~/.calibrate/aero_hand_right_sdk_capture.json --side right --update-calibration
~/.calibrate/aero_hand_right_calibrate.json` for offline retarget iteration instead of asking the
operator to repeat gestures.
Real launch validates serial and `exclusive_resources` ownership across `ros2_control`, active teleop
devices, `auxiliary_actuators`, and `hand_sources` before starting nodes. One mHandPro source selects
one or both gloves with `sides: [right]`, `[left]`, or `[left, right]`; dual-hand control defaults to
`failure_policy: require_all`, so either disconnect locks both outputs. `allow_available` accepts or
retains any connected side without interrupting it to recover a missing peer. Per-side health is
published on `/hand_sources/mhandpro/<side>/health`. Reconnect runs outside the ROS timer with bounded
exponential backoff; configurations requiring P-pose return to that gate after reconnect.
Each real Aero auxiliary actuator must also resolve every hardware joint from the active safety SSOT.
Launch passes ordered limits to the hardware node for a second clamp and rejects incomplete limits.
`aero_hand_teleop.yaml` declares `right`, `left`, and `dual` in one `hand_profiles` SSOT. Unified launch's
`hand_profile` argument selects a profile without duplicating SDK, timing, safety, or retarget settings;
the dual profile still launches one shared source worker. Raw vendor `MHandProFrame` publication is off
by default; diagnostic or raw-recording profiles must explicitly set `publish_raw_frame: true` and add
the corresponding `/frame` topic to their recording list.
The SO-101 integration profile `so101_arm_aero_hand.yaml` uses `base_config: so101_single_arm` and
contains only the hand-specific overlay. The loader resolves the base first; ordered lists such as
`teleoperation.devices` are appended only through the explicit `__append__` form. The production path is
`hand_sources.mhandpro -> HumanHandState -> hand_retarget -> auxiliary_actuators`; the legacy
`mhandpro_glove` device type has been removed and is rejected when selected as an active teleop device.
See the [`robot_teleop` real-hardware quick start](../robot_teleop/README.en.md#real-hardware-quick-start)
for first-time calibration, normal startup, and safety precautions.

Hardware-free tests must explicitly set both the Aero actuator and `hand_sources.mhandpro` to `mock: true` and use
`$(find robot_teleop)/config/mhandpro_mock_right_calibration.json`. The fixture is not a real-device
calibration.

**Launch command:**
```bash
# Teleop mode (with auto recording)
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop record:=true
```

#### 2. model_inference Mode (High-Frequency Position Control)

**Use for:** End-to-end imitation learning models (ACT, pi0, Diffusion Policy)

**Characteristics:**
- High-frequency control (50-100Hz)
- Low latency (1-3ms)
- Direct topic-based position commands
- Reactive, fluid movements

**Configuration:**
```yaml
robot:
  default_control_mode: "model_inference"

  control_modes:
    model_inference:
      description: "High-frequency end-to-end control mode (ACT/pi0)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: true
        pipelines:
          policy:
            model_path: models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515
            deployment: cpu
            execution_mode: monolithic # Or distributed
            request_timeout: 5.0
            default_task: ""
            runtime_options: {}
      executor:
        type: topic
        mode: model_inference
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 50
        control_frequency: 20.0
```

Scheduled inference is opt-in through `control_modes.<mode>.inference.scheduler.enable`, which defaults to `false`.
Independent stages support `scheduling.stages.<producer>.max_snapshot_age_ms` (positive integer, default `5000`).
The producer is the first manifest execution role; terminal and sequential stages reject this option. The age limit
includes producer queueing and execution from observation capture, participates in the runtime policy fingerprint,
and is forwarded by launch. Reprocessing an expired observation does not renew its validity.
Simulation inference (`runtime_target=simulation`, including Gazebo, MuJoCo and mock) requires
`scheduler.enable=false`. Launch rejects the enabled combination before generating control or simulation nodes.
Simulation scenes do not manage scheduler sessions or pipeline bindings.
With an explicit `scheduler` block and `enable: false`, the complete scheduled configuration may remain dormant while
the generated launch graph and node parameters stay identical to the legacy path. This makes `enable` a one-line
rollback switch. If the entire `scheduler` block is absent, scheduled fields are still rejected as unknown fields.
With the default FIFO policy and profile admission disabled, priority-0 dispatches the target directly without a resource
queue. Nonempty default fallback chains are rejected during configuration loading when the scheduler is enabled.
Fallback requires EDF or FIFO with profile admission enabled; deadline-driven priority-0 (EDF or FIFO admission)
supports sequential pipelines only — selecting an independent-stage pipeline as the priority-0 target or in its fallback
chain fails configuration loading, and Global skips independent candidates by serving-status capability at runtime.
On the scheduled path, `profile_path` is optional and is consulted only when a priority-0 request actually considers
that pipeline under FIFO with profile deadline admission enabled. `global_policy: edf` instead orders not-yet-started
priority-0 requests on each hardware resource by absolute deadline, preserves FIFO for ties, and does not preempt active
work. EDF expires queued work without using profiles to promise finish feasibility, so profile deadline admission must
remain disabled with EDF. Readiness requires at least one generic backend priority level. For a configured default
priority greater than zero, it additionally verifies that `executor.inference_pipeline` is online and supports that
priority; other pipelines do not need multi-priority support.

**Launched controllers:**
- `arm_position_controller` (JointGroupPositionController)
- `gripper_position_controller` (ForwardCommandController)

The robot recording-level `lerobot_norm_mode` controls conversion between LeRobot
action/observation units and `ros2_control` radians. `range_m100_100` uses
arm `[-100,100]` and gripper `[0,100]`; `degrees` uses centered degrees for
arm joints while joints listed in `joints.gripper` keep `[0,100]` open/close
semantics.

Observation `align` settings serve both offline data processing and live inference, but their time limits have
different meanings:

- `tol_ms`: timestamp alignment tolerance for `strategy: asof`, used by both offline resampling and live sampling;
  values less than or equal to `0` fall back to `hold`. `hold` and `drop` do not use it.
- `max_age_ms`: maximum sample age accepted by live inference, measured from the node's local receipt clock so it
  cannot be bypassed by rewinding a request timestamp. Missing samples and histories containing only future-dated
  samples are always rejected; values greater than `0` also reject stale samples. These cases return
  `observation_not_ready` instead of running the model with zero padding.

```yaml
contract:
  observations:
    - key: observation.images.top
      align:
        strategy: hold
        tol_ms: 1500
        max_age_ms: 500
```

**Command interface:**
```bash
# Arm position commands
ros2 topic pub /arm_position_controller/commands std_msgs/msg/Float64MultiArray "data: [1.0, 2.0, 3.0, 4.0, 5.0]"

# Gripper position commands
ros2 topic pub /gripper_position_controller/commands std_msgs/msg/Float64MultiArray "data: [0.5]"
```

#### 2. moveit_planning Mode (Trajectory Planning)

**Use for:** Planning-based models (VoxPoser, VLM, goal-conditioned policies)

**Characteristics:**
- MoveIt integration (OMPL/Pilz planners)
- Time-parameterized trajectories
- Action-based execution with monitoring
- Collision avoidance support

**Configuration:**
```yaml
robot:
  default_control_mode: "moveit_planning"

  control_modes:
    moveit_planning:
      description: "MoveIt trajectory planning mode"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller
```

**Launched controllers:**
- `arm_trajectory_controller` (JointTrajectoryController)
- `gripper_trajectory_controller` (JointTrajectoryController)

**Command interface:**
```bash
# List available actions
ros2 action list

# Execute trajectory via action
ros2 action send_goal /arm_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{...}"
```

### Overriding Control Mode at Runtime

Control mode can be overridden via command line:

```bash
# Use default mode from config file
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm

# Override to teleop mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop

# Override to model_inference mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference

# Override to moveit_planning mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning
```

### Mode Selection Decision Guide

```
What type of model are you using?
│
├─ Human teleoperation (leader arm, gamepad, VR)
│  └─ Use teleop mode
│     ├─ Real-time direct control
│     ├─ Zero-latency passthrough (< 5ms)
│     └─ Built-in safety filters
│
├─ End-to-end imitation learning (ACT, pi0, Diffusion)
│  └─ Use model_inference mode
│     ├─ Model outputs high-frequency position streams (50-100Hz)
│     ├─ Needs minimal latency (< 5ms)
│     └─ No trajectory planning required
│
└─ Planning-based (VoxPoser, VLM, goal-conditioned)
   └─ Use moveit_planning mode
      ├─ Model outputs sparse waypoints or goals
      ├─ Needs collision avoidance
      ├─ Requires MoveIt integration
      └─ Time-parameterized trajectories important
```

### Complete Configuration Example

```yaml
robot:
  name: so101_single_arm
  type: so101
  robot_type: so_101

  # Control mode management
  default_control_mode: "model_inference"  # Can be overridden via command line

  control_modes:
    teleop:
      description: "Human teleoperation mode (direct control)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: false

    model_inference:
      description: "High-frequency end-to-end control mode (ACT/pi0)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: true
        pipelines:
          policy:
            model_path: models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515
            deployment: cpu
            execution_mode: monolithic # Or distributed
            request_timeout: 5.0
            default_task: ""
            runtime_options: {}
      executor:
        type: topic
        mode: model_inference
        inference_pipeline: policy

    moveit_planning:
      description: "MoveIt trajectory planning mode (VoxPoser/VLM)"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller

  # Unified joint configuration (DRY principle)
  joints:
    arm: ["1", "2", "3", "4", "5"]
    gripper: ["6"]
    all: ["1", "2", "3", "4", "5", "6"]

  # Hardware configuration
  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  # Teleoperation configuration
  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"

  # Peripherals (cameras, sensors)
  peripherals:
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30

  # ML contract
  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top
        image:
          resize: [480, 640]
    actions:
      - key: action
        topic: /arm_position_controller/commands  # Changes based on mode
        ros_type: std_msgs/msg/Float64MultiArray
        names: ["1", "2", "3", "4", "5", "6"]
```

### How Mode Switching Works

1. **Configuration Phase:**
   - `robot.launch.py` reads `default_control_mode` from YAML
   - Can be overridden via `control_mode:=xxx` command line argument
   - Validates mode exists in `control_modes` section

2. **Controller Spawning:**
   - Only controllers listed in the selected mode are spawned
   - Ensures no controller conflicts (same joint can't be controlled by multiple controllers)

3. **Action Dispatch Integration:**
   - `action_dispatch` node reads current mode from `robot_config`
   - Instantiates appropriate executor (TopicExecutor or ActionExecutor)
   - Provides unified API for upstream inference services

### Troubleshooting Control Modes

#### Mode not switching

**Problem:** Command line override not taking effect

**Solution:** Ensure `control_mode` parameter is correctly spelled:
```bash
# Correct
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning

# Incorrect (typo)
ros2 launch robot_config robot.launch.py control_mode:=moveit_planing
```

#### Controller not starting

**Problem:** Controllers fail to activate

**Solution:** Check controller configuration in `so101_hardware/config/so101_controllers.yaml`:
```bash
# Verify controller exists
ros2 control list_controllers

# Check controller configuration
cat src/so101_hardware/config/so101_controllers.yaml | grep -A 10 "arm_trajectory_controller"
```

#### Action server not available

**Problem:** `FollowJointTrajectory` action not found in moveit_planning mode

**Solution:** Ensure trajectory controllers are active:
```bash
# List active controllers
ros2 control list_controllers | grep trajectory

# Should see:
# arm_trajectory_controller[joint_trajectory_controller/JointTrajectoryController] active
# gripper_trajectory_controller[joint_trajectory_controller/JointTrajectoryController] active
```

For more details, see:
- [action_dispatch README](../action_dispatch/README.md) - Detailed executor documentation
- [docs/architecture.md](../../docs/architecture.md) - System architecture overview

### Voice TTS

`robot.voice_tts` is the robot-level single source of truth for enabling and configuring the optional
`voice_tts_service` package. Enabling it requires an explicit ZipVoice `bundle_path` and a named manifest
`deployment`; configuration never selects a separate backend or silently falls back after a load failure. The
full system starts TTS through `robot.launch.py`, while the package launch remains a standalone debugging entry.

The public typed service is `/voice_tts/synthesize`. Requests and responses carry audio bytes rather than
server-local paths and bound text, prompt, segment count, and response size. The launch builder resolves
configuration and creates the node but does not open the model bundle. The shared host validates the bundle and
loads the session at startup, while `exit_on_init_failure=false` can keep the endpoint alive and report
`MODEL_NOT_READY` when model storage is temporarily unavailable.

TTS is hosted by the shared `inference_service/model_service_node`. The named deployment is loaded at node startup;
later requests reuse it, and node shutdown waits for active synthesis and then
releases model resources without stopping the ROS endpoints. Relative `bundle_path` values resolve from the
absolute `WORKSPACE` set by `.shrc_local`.
This mode does not retry initialization on requests; restart the TTS node after repairing the bundle, dependency,
or device.

The verified `ascend_310p` deployment uses the fixed bundle prompt, accepts Chinese, numbers, and common
punctuation, and returns 24 kHz mono WAV. Request-scoped prompts are currently rejected with
`UNSUPPORTED_PROMPT`; this is a deployment capability rather than an implicit backend choice.


## Usage

### Launching the Robot

```bash
# Launch with real hardware
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm

# Launch with simulation
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# Distributed inference, single-host debug. The YAML policy pipeline must be distributed.
# Terminal 1: Edge
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# Terminal 2: Cloud
ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cpu

# Distributed inference — cross-machine (Device only launches Edge; Cloud runs separately on GPU server)
# Device side:
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# GPU server (set same ROS_DOMAIN_ID):
# ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cuda
```

Relative `model_path` values resolve only against an absolute `WORKSPACE` environment variable. The pipeline ID
derives default action, reset, health, and distributed topics. With multiple pipelines,
`executor.inference_pipeline` must select one explicitly.

### Tracing And Topology

`enable_tracing:=true` starts LTTng through `robot_config.launch_builders.tracing`;
`trace_session_name` defaults to `ib_robot_trace`. Default-name collisions get a
unique suffix; explicit name collisions fail without overwriting an existing session.
The builder stops/destroys its session on shutdown. `IB_TRACE_ENABLED=1` is scoped
to this launch's child processes, including explicit inference environments and
processes deferred until controller readiness. There is no early assignment to
the parent's `os.environ`; launch restores the environment on leaving the scope
so later launches do not inherit the flag. Control modes, node parameters and
model-loading behavior are unchanged.

The existing LTTng startup failure contract is retained: session inspection,
creation, event enablement or start failures abort launch. Failure cleanup only
owns sessions successfully created by this invocation. There is no new strict option.

Startup neither exports a topology sidecar nor imports the topology module.
Ordinary trace analysis requires no robot YAML or topology metadata. For an optional
configuration declaration, use the existing `load_robot_section` path, including
sibling `base_config` overlays, without the full business loader's model checks:

```bash
source .shrc_local
ros2 run robot_config ibrobot-trace-topology \
    src/robot_config/config/robots/so101_arm_aero_hand.yaml \
    --control-mode model_inference > /tmp/ibrobot-topology.json
```

This command neither starts the robot nor loads models or weakens business-loader
validation. Its metadata is marked `provenance=declared`, `runtime_verified=false`:
it is not an inventory of started nodes. It does not apply launch CLI overrides
(except the explicit `--control-mode`), nav stages or runtime-derived configuration.
Independent export failure cannot reject robot startup or stop/destroy a successful
LTTng recording. The tracing Core consumes configuration mappings or topology
manifests, not raw robot YAML.

### Validating Configuration

```bash
# Validate a robot config file with Python directly
python3 src/ros2/ros2_ws/src/robot_config/robot_config/scripts/validate_config.py \
    src/ros2/ros2_ws/src/robot_config/config/robots/so101_single_arm.yaml
```

## Camera Drivers

### USB Cameras (via `usb_cam`)

```yaml
- type: camera
  name: usb_cam
  driver: opencv  # Uses usb_cam package
  index: 0  # USB device index (/dev/video0)
  width: 640
  height: 480
  fps: 30
  pixel_format: bgr8  # bgr8, rgb8, mono8, yuyv, etc.
```

**安装 (Install):**
```bash
# Ubuntu
sudo apt install ros-humble-usb-cam
# openEuler
sudo dnf install ros-humble-usb-cam
```

### RealSense D400 Series (via `realsense2_camera`)

```yaml
- type: camera
  name: rs_cam
  driver: realsense  # Uses realsense2_camera package
  serial_number: "12345678"  # Device serial (optional)
  width: 640
  height: 480
  fps: 30
  depth_width: 640
  depth_height: 480
  depth_fps: 30
  enable_pointcloud: false
  align_depth: false
```

**安装 (Install):**
```bash
# Ubuntu
sudo apt install ros-humble-librealsense2*
# openEuler
sudo dnf install ros-humble-librealsense2*
```

## Camera Calibration

Camera intrinsics can be stored in the standard ROS2 location:

```yaml
- type: camera
  name: top
  driver: opencv
  index: 0
  width: 640
  height: 480
  camera_info_url: file://$(env HOME)/.ros/camera_info/top_camera.yaml
```

Calibration files can be created using the standard ROS2 camera calibration tools:

```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.024 \
  image:=/camera/top/image_raw
```

## tensormsg Integration

The robot_config integrates with tensormsg contracts by allowing observations to reference peripherals by name:

```yaml
# In robot_config
peripherals:
  - type: camera
    name: top
    width: 640
    height: 480

# In contract section
contract:
  observations:
    - key: observation.images.top
      topic: /camera/top
      peripheral: top  # Auto-fills width, height, fps from peripheral
```

When the contract is loaded, it will automatically include the camera metadata from the peripheral definition.


The old `robot_interface` package used LeRobot's Robot class directly. This package replaces it with:

1. **No LeRobot dependency in ROS2 layer**: Uses ros2_control directly
2. **ros2_control native**: Standard ROS2 hardware interface
3. **Existing ROS2 camera drivers**: Uses `usb_cam` and `realsense2_camera` packages
4. **Single YAML configuration**: All hardware defined in one place

The two example configurations from `robot_interface` have been manually migrated to:
- `config/robots/so101_single_arm.yaml` (from `single_arm_banana.yaml`)
- `config/robots/so101_dual_arm.yaml` (from `dual_arms_pencil.yaml`)

## Troubleshooting

### Camera not opening

Check USB permissions:
```bash
ls -l /dev/video*
sudo chmod 666 /dev/video0
```

Or add user to `video` group:
```bash
sudo usermod -a -G video $USER
```

### RealSense camera not found

Install librealsense2:
```bash
# Ubuntu
sudo apt install librealsense2-utils librealsense2-dev
sudo apt install ros-humble-librealsense2*

# openEuler
sudo dnf install librealsense2-utils librealsense2-devel
sudo dnf install ros-humble-librealsense2*
```

Check camera is connected:
```bash
realsense-viewer
```

### Calibration file not found

Make sure the path is correct and starts with `file://`:
```yaml
camera_info_url: file:///home/user/.ros/camera_info/top.yaml
```

## References

- [usb_cam GitHub](https://github.com/ros-drivers/usb_cam) - USB camera driver for ROS2
- [realsense-ros GitHub](https://github.com/realsenseai/realsense-ros) - Intel RealSense ROS2 wrapper

## License

Apache-2.0

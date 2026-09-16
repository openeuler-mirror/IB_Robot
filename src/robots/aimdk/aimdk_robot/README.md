# aimdk_robot — AgiBot X2 runtime

The X2's independently deployable runtime unit: it serves the **public runtime
contract** (`/runtime_status`, `/runtime/set_mode`, `/runtime/get_status`,
`/runtime/stop`, `/joint_states`, the declared streaming command channels and
`/cmd_vel`) by bridging to the **vendor motion-control (MC) tier** of the AimDK
SDK. Generic IB-Robot packages bind X2 exactly as they bind SO-101 or LeKiwi and
never see a vendor topic, service or mode name.

Planning and rationale: `openspec/changes/aimdk-runtime-migration/`
(`design.md` decisions D1–D14 are referenced from the profile comments).

## Responsibilities

- Translate contract commands into vendor MC messages and vendor state into
  `RuntimeStatus`, the public interface description and `/joint_states`.
- Own the vendor input-source (arbitration) claim and the vendor mode machine.
- Expose the platform's own capabilities — speech, screen expression, lights,
  named motions, postures, battery, diagnostics, localization, audio, sensors —
  on neutral, project-owned endpoints.
- Declare the vendor's already-published sensors as public interfaces.

## What the wrapper exposes

Generic consumers see only the left column; the right column never leaves this
package.

| Public surface | Type | Platform mechanism |
|---|---|---|
| `/runtime/set_mode`, `/runtime/get_status`, `/runtime/stop`, `/runtime_status` | contract | `SetMcAction`, `/aima/mc/common/state`, `/aima/sm/system_state` |
| `/joint_states`, `/aimdk/body_joint_states` | `sensor_msgs/JointState` | `/aima/hal/joint/{arm,head,waist,leg}/state`, hand state |
| `/aimdk/arm/commands`, `/aimdk/gripper/commands`, `/aimdk/head/commands` | `Float64MultiArray` | `/mc/upper_body_command` (`arm_pos` / `hand_pos` / `head_pos`) |
| `/cmd_vel` | `geometry_msgs/Twist` | `/aima/mc/locomotion/velocity` + input-source arbitration |
| `/motion/execute_named` | `ibrobot_msgs/action/ExecuteNamedMotion` | `SetMcPresetMotion` (wave, handshake, clap…) and posture MC actions (sit, squat, lie, stand up, stairs) |
| `/speech/speak` | `ibrobot_msgs/srv/SpeakText` | `PlayTts` (neutral 0–100 priority stepped onto the platform's TTS layers, background…warning; the safety layer is unreachable) |
| `/expression/play` | `ibrobot_msgs/srv/PlayExpression` | `PlayEmoji` (semantic name → vendor emotion id) |
| `/led/set_pattern` | `ibrobot_msgs/srv/SetLedPattern` | `SetPmuLed` (reports platform preemption instead of faking success) |
| `/localization/relocalize`, `/localization/pose` | `StartRelocalization`, `PoseWithCovarianceStamped` | `/integrated_command`, `/slam/lidar_odom` |
| `/power_state` | `ibrobot_msgs/msg/PowerState` | `/aima/hal/pmu/state` |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | `/aima/hds/{diag,alert}_code_list` |
| `/audio/capture_stamped`, `/audio/info`, `/audio/play` | project audio contract | `/aima/hal/audio/{capture,playback}` |
| camera (raw + `/compressed`) / IMU (chest, torso, RGB-D, stereo, lidar) / touch interfaces | declared, not re-driven | `/aima/hal/sensor/**`, `/aima/hal/imu/**` |

### Vendor QoS is part of the endpoint contract

Each vendor endpoint carries the reliability and durability from the SDK's own
interface tables, in the profile under `vendor.qos`. This is not cosmetic: a
BEST_EFFORT publisher **cannot** serve the platform's RELIABLE subscription, and
ROS 2 reports it once and then silently discards every message —

```
New subscription discovered on topic '/aima/hal/audio/playback',
requesting incompatible QoS. No messages will be sent to it.
Last incompatible policy: RELIABILITY
```

— which is exactly how audio playback was found to be dead on the robot
(2026-09-22). Subscribing TRANSIENT_LOCAL matters for the same reason in the
other direction: the platform's state topics are TRANSIENT_LOCAL, so a
TRANSIENT_LOCAL subscription receives the current value on subscribe instead of
waiting for the next sample. When a firmware changes a QoS, correct the profile
rather than the code.

### Audio playback holds the platform's audio focus

Raw stream playback is focus-gated. `hal_audio` **does not** request focus on
behalf of whoever publishes to `/aima/hal/audio/playback` — only the file
playback service manages focus internally — so audio published without it is
accepted on the wire and never reaches the speaker. There is no error anywhere:
the only symptom is silence.

The bridge therefore takes focus itself, for exactly as long as it is
streaming: the first chunk on `/audio/play` triggers `RequestAudioFocus` (and
is buffered, not dropped, so an utterance does not start mid-word), and
`AbandonAudioFocus` follows once the stream has been quiet for
`vendor.audio.focus_release_idle_s`. Holding focus while idle would keep
preempting the platform's own announcements. Three details are contractual:

- The service status is `SUCCESS` whether or not focus was granted — it reports
  only that the request was processed. `focus_response.focus_gain` is the answer.
- `AbandonAudioFocus` must repeat `pkg_name`, `priority` and `priority_weight`
  exactly as the matching request carried them, or the holder never matches and
  the focus is never released.
- Focus loss is announced on `/aima/hal/audio/focus_response`, not polled. A
  higher (or equal) priority source preempts; the bridge stops publishing
  rather than fighting whatever the platform decided should be heard.

To test playback on a robot, start with the vendor's own path, which proves the
speaker and needs no focus handling, then the contract topic:

```bash
# 1. The platform's own file playback (file lives on the interaction unit
#    PC3, 10.0.1.42; .pcm/.wav, S16LE, world-readable parent directories).
ros2 run py_examples play_audio --ros-args -p audio_path:=/var/tmp/audio/test.wav

# 2. Speech through the contract, which goes via the vendor TTS service.
ros2 service call /speech/speak ibrobot_msgs/srv/SpeakText \
  "{text: '接入验证正常', language: 'zh-CN', priority: 60, interrupt: true}"

# 3. Raw playback through the contract. `ros2 topic pub` cannot carry a real
#    waveform, so generate one: 16 kHz mono S16LE, in ~100 ms chunks.
python3 - <<'PY'
import math, struct, time
import rclpy
from audio_common_msgs.msg import AudioDataStamped

rclpy.init()
node = rclpy.create_node("audio_play_probe")
pub = node.create_publisher(AudioDataStamped, "/audio/play", 10)
rate, chunk = 16000, 1600
frames = [int(12000 * math.sin(2 * math.pi * 440 * n / rate)) for n in range(rate * 2)]
pcm = struct.pack(f"<{len(frames)}h", *frames)
time.sleep(1.0)  # let discovery settle before the first chunk
for offset in range(0, len(pcm), chunk * 2):
    message = AudioDataStamped()
    message.header.stamp = node.get_clock().now().to_msg()
    message.audio.data = list(pcm[offset : offset + chunk * 2])
    pub.publish(message)
    time.sleep(chunk / rate)
node.destroy_node()
rclpy.shutdown()
PY
```

A two-second 440 Hz tone should come out of the speaker. If it does not, read
the runtime's log: `audio focus refused` means another source holds the speaker
(raise `vendor.audio.focus_priority`, band 1..10), and `audio focus lost` means
something preempted mid-stream. Silence with neither message means the audio
never arrived — check `ros2 topic info -v /audio/play` for a type or QoS
mismatch on the publisher's side.

### Stop uses modes the firmware registers, and HOLD switches none

`SetMcAction` matches on `action_desc` against the platform's **configuration**
(the `action` enum field is unused since v0.8.2), and answers code 3, "动作未在
配置中登记", for a name that exists in the `McAction` IDL but not in that
configuration. The enum is the wider set: it still carries names the firmware
dropped. The registered motion modes are exactly the seven the documented mode
table lists and the vendor's `set_mc_action` example offers:

| Mode | Meaning | Availability |
|---|---|---|
| `PASSIVE_DEFAULT` | zero torque, free joints | always permitted, even under safety protection |
| `DAMPING_DEFAULT` | damped joints, safe takeover | always permitted, even under safety protection |
| `JOINT_DEFAULT` | position-control stand, joints locked | permitted, but posture-limited (refused while seated) |
| `STAND_DEFAULT` | force-control stand, active balance | refused under safety protection |
| `LOCOMOTION_DEFAULT` | walk/run (unified with stand since v0.8.0) | refused under safety protection |
| `HEAD_ONLY` | head joints only | flagship only |
| `UPPERBODY_REMOTE_SPLIT` | head + arms + hands | — |

This runtime refuses any other name in the profile at load time rather than
letting the platform refuse it at stop time — `JOINT_FREEZE`,
`SOFT_EMERGENCY_STOP` and `ZERO_TORQUE_DEFAULT` are all IDL-only and were all
refused by the firmware when this profile still named them.

**HOLD switches no mode.** No registered mode holds the robot from every
posture: leaving `STAND_DEFAULT` trades active balance for a position-held
stand, `JOINT_DEFAULT` is refused from a seated posture, and `STAND_DEFAULT`
*from* a seated posture is a stand-up — motion, which is the opposite of a
stop. So HOLD closes command admission and publishes an explicit safe command;
the platform holds whatever mode it is in, and its own 1 s input timeout zeroes
the stream. `TORQUE_OFF` is the policy that does ask for a mode: `soft_estop`
(the default) answers with `DAMPING_DEFAULT` and reports the downgrade, and
`zero_torque_when_seated` answers with `PASSIVE_DEFAULT` only when the reported
body pose is sit, squat or lying, refusing otherwise (design D4).

### Preset motions: the pair is the key, and the stamp is the play time

Two vendor details that neither IDL file reveals:

- **`header.stamp` on `SetMcPresetMotion` is the play time**, not a message
  timestamp: "stamp 用于指定播放时刻（UTC），为 0 时立即播放". Filling it from
  this host's clock schedules the motion at a UTC instant, which is only
  correct when this host's clock matches the robot's — on a separate compute
  pack it asks to play at a time already past, or not yet arrived. This runtime
  sends 0, the documented "play now", which needs no agreement about clocks.
- **`area` is not a body region.** Since v0.8.0 "area 原有的分区概念已经弱化,
  仅和 motion 联合使用映射具体动作": the `(motion, area)` pair is a key into a
  table of animations, and only the tabulated pairs exist. Neither enumeration
  is a guide — `McPresetMotion` lists values that appear in no pair (3015), and
  the table lists pairs that appear in no enumeration (3017/11 applause). The
  documented pairs are in `projection.PRESET_MOTION_AREAS`, and the profile
  contract test holds the shipped profiles to them.

Preset motions also run **only from stable stand** — every documented pair is
annotated "稳定站立模式下执行". The runtime checks the platform's *reported*
action, not its own mode, because a robot on a gantry may never reach stable
stand; a mismatch is refused naming both actions rather than dispatched into
silence. And an accepted dispatch that carries `task_id 0` is a refusal, not a
result: `GetMcPresetMotionState` distinguishes only "executing" from
"completed", so a task that was never created reads as completed the instant it
is asked about.

### Neutral priorities are mapped, never passed through

The contract's priorities are a neutral 0..100. None of the platform's are, so
each is stepped onto the vendor's own scale rather than forwarded:

| Contract | Platform scale | Why it cannot be passed through |
|---|---|---|
| `/speech/speak` | `TtsPriorityLevel` enum (L1…L10) | Scheduling layers, not a range. The life-safety layer is deliberately unreachable. |
| `/expression/play` | small int; **platform faults use 8-10** | The platform shows over-temperature (8) and fall-protection / disabled-arm (10) indications on the face. Anything above 10 hides them — "导致用户无法通过屏幕感知这些故障". Only neutral ≥95 maps above the fault layer. |
| `/led/set_pattern` | ratcheting threshold | Each accepted request *raises* a threshold that later, lower requests must clear, so a raw neutral 100 locks the caller out of its own next call. `preempt` is the documented escape. |

A raw 0..100 would have put ordinary expression requests above the platform's
fault indications — the default priority in a plain `ros2 service call` is
enough to do it.

### Audio capture is a 6-channel interleaved array, not mono

`/audio/capture_stamped` republishes the platform's raw capture unchanged:
4 microphones + 2 echo-reference channels, interleaved per int16 sample. The
channel count is declared in the profile and published on `/audio/info`; a
consumer that assumes mono reads it as noise. The platform's *processed* VAD
stream is the mono 16 kHz one, and it is not bridged (task 7.2).

### The platform's state message is read whole

`/aima/mc/common/state` carries eight fields at 10 Hz. Using all of them costs
nothing beyond a subscription the runtime already has, and each replaces a
slower or weaker source:

| Field | What it gives the bridge |
|---|---|
| `fsm_state` | The balance controller's own state. Wired into the fault registry as `balance`: the platform entering SAFE degrades the runtime instead of leaving it ACTIVE while mode switches are being refused. |
| `input_source` | The arbitration holder, ten times a second. A preemption is seen at once rather than up to a poll period later — and every command published in that window would have been discarded. |
| `speed_status` | The speed envelope the platform is enforcing *now*, which it narrows with battery, load and terrain. `/cmd_vel` is checked against it; the profile stays the ceiling, so the platform may narrow it and never widen it. |
| `motion_status` | Upper-body player state — a faster second source behind the preset-motion poll. |
| `runtime_model` | Per-hand and waist status, a continuous second source behind the one-shot `GetHandType`. |
| `servo_status` | Current head/waist/squat offsets. Note its *limits* live in the IDL comments and cover head, waist and squat only — there are no arm limits here. |

The head travel from those comments (yaw ±0.38, pitch ±0.35) is declared in the
profile and enforced on the head channel: an out-of-travel target is refused,
not clamped, for the same reason a gripper command is.

### Dexterous hand touch rides on the hand state message

`HandStateArray` carries a 36-cell palm, a 36-cell back and five 16-cell
fingertips per hand. They are republished on `/aimdk/hand_touch` as a
`Float64MultiArray` whose layout dimensions name each pad and its size, because
the pads have different cell counts and a fixed-width matrix would pad or
truncate them.

Only a hand whose reported type is in the dexterous family contributes. The
touch fields exist on every hand state message, a gripper's included, where
they are all zeros — that is the absence of a sensor, not a measurement of no
contact, and publishing it would invent a tactile array the robot does not
have. The gripper profile therefore declares no tactile endpoint.

### Declared sensors are what the firmware actually serves

The declared interface set is a promise to consumers, so it lists what was
observed on a real X2 (firmware ≥ v1.1.0, verified 2026-09-22), not everything
the SDK documents. Four differences are deliberate:

| Endpoint | Why it is not declared |
|---|---|
| `lidar_chest_front/lidar_pointcloud` | The SDK documents it (PointCloud2, 10 Hz) but the verified firmware publishes `lidar_raw_data` (`aimdk_msgs/msg/LidarRawData`) instead — a type the public SDK's message package does not define, so no SDK consumer can subscribe to it at all. `perception.lidar` therefore stays undeclared; the lidar's own IMU is a plain `Imu` topic and is declared under `perception.imu`. |
| `/aima/hal/sensor/gnss` | Added in firmware v1.1.0, absent on a v1.1.0+ unit — the GNSS module is an installed option, not part of the platform. |
| `/integrated_command`, `/relocalization_pose` | Relocalization belongs to the vendor's optional SLAM component; both are absent on the verified unit even though `/slam/lidar_odom` is present, so `localization.map` stays undeclared while `localization.pose` does not. |
| `rgb_head_front_center/*`, `*/rgb_image/h265` | The front interaction camera was withdrawn from the open interface set in v0.8.1 ("resource usage too high, replaceable by the other cameras") although the topics may still appear; the h265 streams are undocumented, so their type cannot be promised. |

A deployment whose robot does serve these declares them in its own profile —
each is profile data, not code. Check with `ros2 topic info -v <endpoint>`
before declaring, because the declaration is what the conformance suite and
every consumer will hold the runtime to.

Note that the robot also publishes `/diagnostics` itself; `diagnostic_msgs`
is an aggregation topic by design, so a consumer sees the platform's own
entries alongside the vendor codes this runtime translates.

Not every vendor endpoint is bridged. Volume/mute, file and video playback,
expression groups, LED/emoji state read-back, diagnostic history, goal
navigation, LinkCraft resources and microphone switching are inventoried with
their reasons in the OpenSpec change (tasks 7.x).

## Prohibited

- **No low-level joint tier by default.** `/aima/hal/joint/*/command` has, in the
  vendor's words, "no timeout or fail-safe protection", and the hand variant
  requires stopping the vendor MC application (`aima em stop-app mc`). It is
  recorded in the profile as `vendor.expert_joint_tier` with `enabled: false`;
  turning it on forfeits the platform's balance control and is a deliberate
  deployment act (design D14).
- **No controller manager, no ros2_control hardware component, no controller
  spawners.** The vendor owns real-time control; see design D1.
- **No sensor drivers.** The vendor already publishes cameras, LiDAR and IMUs;
  this runtime declares them, it does not re-drive or relay them (design D7).
- **No whole-body RL.** `aimdk/extra/mc-rl` is the vendor's own tree and is not
  touched; `body.whole_stream` stays declared and unimplemented.
- **Never run SDK applications on the motion-control unit (PC1).** The vendor
  documents this as strictly prohibited.

## Vendor dependency (not vendored)

`aimdk_msgs` is declared as an ordinary dependency and must be provided by a
developer-supplied AimDK overlay. The SDK is **not** copied into this repository
and the extracted tree is git-ignored:

- it is ~490 MB, and
- `aimdk/src/aimdk_msgs/package.xml` currently declares
  `<license>TODO: License declaration</license>` — i.e. no license. Vendoring
  unlicensed third-party material is not permitted by the openEuler AI-assisted
  contribution policy, so vendoring stays blocked until AgiBot declares one.

Build and source the overlay before building or running this package:

```bash
# The vendor's own documented procedure (docs: common/aimdk_build)
source /opt/ros/humble/setup.bash
cd ~/aimdk && colcon build && source install/local_setup.bash
```

On an x86 developer host the messages are generated from
`aimdk_msgs/interface/**/*.{msg,srv}`; on the on-robot aarch64 unit the vendor's
own CMake logic may select its prebuilt package instead. Nothing on the IB-Robot
side special-cases the architecture. Without the overlay the package fails to
start with a message naming what is missing, and the runtime independence gate
skips with the same reason.

## Deployment placement

Raw image streams are ~80–90 MB/s each and the vendor states they must not be
subscribed across compute units. Consumers of the camera interfaces run on the
development compute unit (PC2, Orin NX); the profile records this under
`vendor.compute_units`. Node startup must also stay within the vendor's limit of
roughly two nodes per second while the robot is standing or walking — bulk
launches can destabilize motion control, so the launch entry staggers its nodes.

## Profiles

| Profile | Robot | End effector | Capabilities that differ |
|---|---|---|---|
| `profiles/x2_ultra.yaml` | X2 Ultra | OmniPicker gripper | `gripper.1d` (count 2) |
| `profiles/x2_ultra_omnihand.yaml` | X2 Ultra | OmniHand dexterous hands | `hand.multi_joint`, `hand.gesture`; no `gripper.1d` |

The profile is the single source of runtime parameters: public joint projection,
modes and their vendor MC actions, command channels, declared capabilities, stop
policy, arbitration identity, expression/motion/posture tables and the vendor
endpoint map. Hand type is a profile property, not runtime autodetection — the
bridge verifies it against the vendor's `GetHandType` at startup and refuses to
become ACTIVE on mismatch.

### What the profiles deliberately do not declare

- **No public model.** No URDF is rendered and no `model` block is exported, so
  consumers needing LeRobot normalization metadata fail closed on X2 (design D5,
  decided 2026-09-16). Policy inference and dataset recording on X2 are not part
  of this stage.
- **No trajectory / FK / IK / move-to capabilities** — the vendor MC tier
  exposes no joint-trajectory endpoint.
- **No `base.odom`** — `/slam/lidar_odom` is a map-frame localization that jumps
  on relocalization, not continuous odometry. It is published as
  `localization.pose` with its true frames instead (design D6/D13).
- **No teleoperation** — the public teleoperation projection binds exactly one
  arm group and one gripper; X2 has two of each.
- **No image geometry** — declared explicitly as `null`, because the vendor
  documents that resolution and encoding vary across hardware and software
  revisions and must be read from `CameraInfo`.
- **No `nav.goal`** unless the deployment enables the optional vendor
  navigation component.

## Not exposed here

- **Embodied primitives.** `look_at` / `set_posture` / `play_preset_motion` /
  `speak` as skill primitives would change the primitive contract, the skill
  resolver and the skill executor — code SO-101 and LeKiwi execute. That
  extension is specified in the OpenSpec change and delivered separately; the X2
  deployment ships with `embodied.enabled: false`. The capabilities themselves
  are available now through `/motion/execute_named`, `/speech/speak` and friends.
- **Voice ASR/TTS services.** The runtime publishes the audio contract, but
  enabling `voice_asr` still requires `robot_config` to accept a
  runtime-provided capture source.

## Status

The runtime is implemented and tested against `aimdk_vendor_mock`, which speaks
the real `aimdk_msgs` IDL.

**First hardware contact: 2026-09-22**, on an X2 Ultra on a gantry with an
external aarch64 compute pack. The read-only surface passed in full, and
several profile values in this package now come from that session rather than
from the documentation: the hand-type `NONE` handling, the vendor QoS table,
the declared sensor endpoints (lidar, GNSS and relocalization are *not*
declared because that firmware does not serve them), the registered motion
modes, and the clock-skew limit. Treat those as measured, not assumed.

What is still unverified on hardware: everything that moves the robot. Mode
switching reached the platform, but streaming, stop latencies, locomotion,
posture skills and the interaction surface remain unchecked, user-executed
gates in the OpenSpec change (§9, and the open items in §8b). The stop latency
bounds in the profile are still placeholders — the measured values from the
first session are recorded in the change, not yet promoted into the profile.

## Tests

```bash
source .shrc_local
source ~/aimdk/install/local_setup.bash        # vendor overlay
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1   # keep the mock off shared domains
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/robots/aimdk/aimdk_robot/test -q
```

Without the overlay the integration tests skip with a named reason; the
projection and profile tests still run.

## Running against the mock

```bash
ros2 launch aimdk_robot runtime.launch.py profile:=x2_ultra simulated:=true
```

## Running against the robot (external compute pack)

The vendor's "Host PC / Computing Pack + Cross-Device Networking" mode
(quick_start/prerequisites.html) is the one this runtime is written for: it
runs on a machine you own, joined to the robot's ROS 2 network over the rear
Ethernet port. An aarch64 board such as an Orange Pi AIpro (Ascend 310B) works
as that compute pack — the vendor ships `aimdk_msgs` prebuilt for aarch64, and
the runtime's dependency closure is only `aimdk_robot` + `ibrobot_msgs` +
`robot_runtime`, so nothing else from this repository needs to be on the board.

Vendor requirements for the compute pack: Ubuntu 22.04, ROS 2 Humble, an SDK
copy whose version matches the robot firmware (copy it from the development
computing unit or download it from `x2-aimdk.agibot.com`).

### 1. Network

- Wire the board to the robot's rear SDK Ethernet port; set the board to a
  static `10.0.1.2/24`. The X2 Ultra's development computing unit answers at
  `10.0.1.41` (X2 EDU: motion-control unit at `10.0.1.40`, which must never be
  used as a build/run host).
- DDS must be able to cross the wire: **`ROS_LOCALHOST_ONLY` must be unset or
  `0`**, and `ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` must match the robot's.
  The repository's test gates set `ROS_LOCALHOST_ONLY=1` on purpose; do not
  carry that into the board's shell.
- Prove the link with the vendor's own tools before touching this runtime:

  ```bash
  ros2 topic list | grep '^/aima/'          # robot topics visible
  ros2 run py_examples get_mc_action        # answers with the current MC action
  ```

  No `/aima/*` topics means power, cable, domain or RMW — not this package.

- **Synchronise the board's clock with the robot's.** This is a hard
  requirement, not a nicety. The platform accepts a command only if its stamp
  falls inside a 0.2 s window (`vendor.command.stamp_window_s`) and discards
  every other one **without an error on either side**. A compute pack whose
  clock was never synchronised is typically tens of seconds out, so the
  symptom is a robot that ignores 50 Hz of perfectly valid commands while every
  log stays clean — including the vendor's own `upper_body_control` example.

  ```bash
  sudo timedatectl set-ntp true                    # or point chrony/ptp at the robot
  # confirm against the platform's own stamps, not against a public NTP server:
  ros2 topic echo --once /aima/mc/common/state --field header.stamp
  date +%s.%N
  ```

  The runtime measures this for you: it compares its clock against the
  platform's state stamps and refuses to reach ACTIVE while the difference
  exceeds `vendor.command.clock_skew_limit_s` (0.1 s), reporting
  `clock disagrees with the platform by ...` instead of leaving you with a
  robot that silently does nothing.

  The measurement is the **minimum** difference over a 30 s window, not the
  latest one. A single `local_receive - remote_send` difference is the clock
  offset plus however long that callback waited behind other work, and on a
  node taking ~2500 joint-feedback callbacks a second that wait is not small.
  Delay only ever adds, so the minimum is the honest estimate and a busy moment
  — an interaction service call, a burst of traffic — cannot masquerade as
  drift. If you see a clock fault, it is about the clock.

### 2. Build on the board

```bash
# vendor overlay (once per SDK version)
cd ~/aimdk && colcon build && source install/local_setup.bash
# this runtime: only its closure. `./scripts/build.sh --aimdk` is the same
# build inside the project venv; the plain colcon form below needs no venv.
python3 -m pip install "jsonschema>=4" pyyaml     # Ubuntu's apt jsonschema is 3.x
cd ~/IB_Robot && rosdep install --from-paths src/ibrobot_msgs src/robot_runtime src/robots/aimdk -i -y
colcon build --merge-install --symlink-install --packages-up-to aimdk_robot
source install/setup.bash
```

### 3. Start, and confirm the platform accepted the runtime

Start the robot per the vendor's guide (suspended on the gantry for the first
runs), bring it to `STAND_DEFAULT` with the vendor example, then:

```bash
ros2 launch aimdk_robot runtime.launch.py profile:=x2_ultra      # or x2_ultra_omnihand
ros2 service call /runtime/get_status ibrobot_msgs/srv/GetRuntimeStatus
```

`lifecycle: ACTIVE` means the vendor services answered, `GetHandType` matched
the profile's declared hand, the input source was registered and enabled, and
this host's clock agrees with the platform's. Anything else names its cause in
`faults`: a hand-type mismatch or a clock disagreement is `FAULTED`, a failed
registration or a real preemption is `DEGRADED`, a quiet joint feedback source
is `DEGRADED` naming the group. A non-ACTIVE status always carries its reason —
lifecycle and faults are published as one observation, never separately.

Then check the read-only surface before commanding anything:

```bash
ros2 topic hz /joint_states --qos-reliability best_effort   # the declared rate, public joint order
ros2 topic echo /aimdk/body_joint_states --once   # waist/legs present
ros2 topic echo /power_state --once
ros2 topic echo /runtime_status --once            # capabilities + interface description
```

Note the explicit QoS. Debugging tools default to a RELIABLE subscription,
which is incompatible with the platform's (and this runtime's) BEST_EFFORT
sensor publishers and receives nothing at all — `ros2 topic echo` takes
`--qos-reliability best_effort`, and `ros2 topic hz` has no such flag in
Humble, so measure rates on a topic this runtime publishes rather than on a
vendor one. Subscriptions created long after a publisher occasionally fail to
associate with the vendor's `waist`/`leg`/`hand` state topics; the runtime
subscribes at startup and is unaffected, so `/aimdk/body_joint_states` is the
reliable way to read those groups.

### Arbitration is a takeover rule, not a reservation

The platform's input-source arbitration (`MC_control.html` 仲裁流程) decides
which of several sources drives the robot. Its rule is that a registered,
enabled source becomes the holder **by sending a non-zero command** when there
is no holder, the holder has timed out, or it outranks the holder. Two
consequences shape this runtime's admission:

- An **empty holder** means nothing has sent a valid command yet — the SDK
  states this directly. It is not a lost claim. A runtime that refuses to
  publish until it is the holder can never become the holder, and deadlocks
  itself into a permanent `DEGRADED` that no restart clears.
- A holder this runtime **outranks** is not a lost claim either: streaming is
  precisely how the takeover happens.

So admission requires registration (ADD **and** ENABLE — the platform discards
commands from a source that was added but never enabled), not holdership. Only
a *named* holder whose priority is at or above this runtime's closes output,
and that fault names both the holder and the two priorities.

The claim is re-asserted before each discrete dispatch, which is what the
vendor's own preset-motion client does: a registration made at startup is not
permanent (a stop releases it), and the platform discards requests from a
source that is not currently enabled. A named motion that cannot register is
refused rather than sent into a discard.

Priority stays at 30. The SDK's preset client registers at 40, but that is
exactly the platform's `pnc` planner priority and takeover requires *strictly*
higher, so 40 would not outrank it while moving this runtime out of the
documented 20-39 secondary-development band. It would also change nothing for
preset motions, which the SDK states have no priority protection at all — what
decides those is registration.

Upper-body commands are not arbitrated at all on the platform: the vendor's own
`upper_body_control` example registers no source and still moves the arms. This
runtime is deliberately stricter and closes *every* command channel when it is
outranked — if the remote controller has taken the robot, this runtime should
not be moving its arms either. That is a policy choice, not a platform rule.

### 4. Motion, in this order

Each step is one item of the real-machine gate recorded in the OpenSpec change
(`aimdk-runtime-migration`, section 9). Keep the robot on the gantry until
locomotion.

1. **Modes** — `idle → stream → idle → head → idle → locomotion → idle` through
   `/runtime/set_mode`; each answer must arrive only after `/aima/mc/common/state`
   reports the vendor action (`UPPERBODY_REMOTE_SPLIT`, `HEAD_ONLY`,
   `STAND_DEFAULT`). Ask for a transition the platform refuses (e.g. `stream`
   while the vendor is in `PASSIVE_DEFAULT`) and confirm the vendor code is
   surfaced (`INVALID_POSTURE`, `NO_TRANSITION_PATH`, …), not swallowed.
2. **Upper-body stream** — in `stream`, publish the *measured* arm pose on
   `/aimdk/arm/commands` (no motion expected), then a single joint ±0.1 rad at
   50 Hz. Watch `rejected_counts` in `/runtime_status`: a rising
   `arm_stream` count with no motion means the vendor dropped the command
   (stamp window or arbitration). Stop publishing: the arm must hold, not
   drift, and not return to zero.
3. **End effector** — `/aimdk/gripper/commands` `[0.2, 0.2]` then `[0.8, 0.8]`
   (claw), or the hand channels on the OmniHand profile. Only the hands move.
4. **Stop** — `/runtime/stop HOLD` while streaming: the latch engages, `stream`
   is refused, `idle` clears. HOLD switches **no** motion mode (see below), so
   the robot stays in whatever mode it was in and simply stops receiving
   commands. Record `cancel_latency_s` / `idle_latency_s` from the response.
   `TORQUE_OFF` while standing must answer with `DAMPING_DEFAULT` (the default
   policy's platform-safe equivalent) — do this on the gantry.
5. **Locomotion** — off the gantry, in `locomotion`, enable the gate
   (`/motion_mode/set_navigation_enabled true`), publish `/cmd_vel` at 0.25 m/s
   for one second, then stop publishing: the runtime's staleness zero must
   arrive before the vendor's own 1000 ms timeout. Disable the gate while
   driving: the next command out is zero.
6. **Interaction** — `/speech/speak` at priorities 0 and 100 (both audible,
   never the platform's safety layer), `/expression/play` once and loop,
   `/led/set_pattern`, and `/motion/execute_named wave`: the action must stay
   open until the arm has finished, not return at dispatch.
7. **Sensors** — `ros2 topic hz` on each declared endpoint from the
   description, on the compute unit the profile assigns it to.

### 5. Same suite, real transport

Once 1–4 pass by hand, the contract suite runs unchanged against the robot;
it is the same set of checks the simulated gate applied:

```bash
export CONFORMANCE_LAUNCH="ros2 launch aimdk_robot runtime.launch.py profile:=x2_ultra"
export CONFORMANCE_PROFILE=$(ros2 pkg prefix aimdk_robot)/share/aimdk_robot/profiles/x2_ultra.yaml
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/robot_runtime/test/test_conformance.py -q -rs
```

Be aware of what it commands: the streaming check moves **every arm joint by
+0.3 rad at once** and the hold check expects them to stay there, so clear the
arm workspace and keep the robot on the gantry for this run. The stop checks
use `HOLD` only.

### 6. Write the numbers back

The profile's `runtime.stop` bounds (`cancel_bound_s`, `idle_bound_s`,
`torque_off_bound_s`) are placeholders until measured on the robot; replace
them with the latencies observed in step 4, with margin, and tick the
section-9 items in the OpenSpec tasks.

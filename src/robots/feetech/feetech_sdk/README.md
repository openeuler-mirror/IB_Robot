# feetech_sdk

ROS-free Feetech STS/SMS motor-family SDK.

## Responsibility

- Deterministic serial bus lifecycle (open/close, explicit port-open errors)
- Feetech protocol I/O with explicit failure reporting (a failed sync read never presents cached values as fresh)
- Single-source unit conversion (ticks ↔ radians/rad-s/amperes; raw servo units never escape this package)
- Motor configuration with atomic rollback (torque-off + EPROM relock on any step failure)
- Emergency torque release with per-motor retry
- Simulated transport (protocol-level fake bus) for fully headless testing
- Bus health observability (consecutive failure count, last-success timestamp)

Sync-read decoding is bounded to complete frames in the current receive buffer;
an incomplete response cannot reuse bytes left by an earlier read. Failures
distinguish an incomplete request write, no response before the receive deadline,
missing/incomplete motor feedback (motor ID and received/expected byte counts),
and a complete response with a bad checksum. No failed read returns samples.

The receive timeout follows LeRobot's Feetech `patch_setPacketTimeout`:
`50 ms + (response_bytes + 3) * 10,000 / baudrate` ms, rounded up to whole
milliseconds. Feedback uses 15 data bytes plus 6 framing bytes per motor, so
six motors at 1 Mbps get 52 ms. A healthy response returns as soon as all bytes
arrive. The timeout scales with the requested group size and baudrate; callers
can obtain it through `Bus::sync_read_timeout`. An explicitly supplied I/O
deadline can still shorten it.

## Prohibited

- Any ROS dependency (CI-enforced by `test/check_package_hygiene.cmake` — no rclcpp/rosidl/message packages, no network fetches)
- Robot-level semantics (arm joint names, gripper conventions, calibration file formats — those live in `so101_sdk`/`lekiwi_sdk`)
- Planning or control-loop logic

## SDK Skeleton Convention (for new robot SDKs)

New robot SDK packages should mirror this file layout:

```
<robot>_sdk/
├── include/<robot>/     # public headers (ROS-free, SI units, named joints)
├── src/                 # implementation
├── test/                # gtest + hygiene checks, all registered in colcon
├── python/bindings.cpp  # optional pybind11 for backend nodes
No vendor source is committed in-tree: the FTServo SDK is fetched at
configure time (pinned FetchContent) from the ib_robot mirror.
├── CMakeLists.txt       # ament_cmake, POSITION_INDEPENDENT_CODE ON
└── package.xml          # zero ROS runtime dependencies
```

## Vendored Sources

FTServo Linux SDK — MIT License, fetched via pinned FetchContent
(`https://gitcode.com/ib_robot/FTServo_Linux.git`, revision
`06fd3356dbd7bccd886b5a70d7ae0fccc6c76d38`); same mirror the pre-migration
`so101_hardware` used, now revision-pinned instead of tracking `main`.

## Testing

```bash
colcon build --packages-select feetech_sdk --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select feetech_sdk
```

52 gtest cases + package hygiene check. All run headless against the simulated transport.

# Public Robot Interface Schema v1

The robot runtime generates a public description from its effective profile and
instance overrides. IB-Robot consumes this document, not driver configuration.
JSON on ROS and exported YAML snapshots have one shape, validated by
`src/robot_runtime/robot_runtime/schemas/interface_description.schema.json`.
No second hand-maintained robot manifest is required.

An exported description is a read-only snapshot, not a hardware configuration
command. Change capture settings in the robot profile/instance override, restart
the runtime, then bind consumers to the new digest. This change does not add a
live camera-mode negotiation service.

## Document

| Field | Meaning |
| --- | --- |
| `schema_version` | Public protocol version, currently exactly `1`; unsupported versions fail closed |
| `robot` | Instance `id`, robot `type`, runtime name and software version |
| `execution` | `physical` or `simulated`; synthetic observations are never represented as hardware feedback |
| `interfaces` | Stable logical IDs mapped to ROS topics, services and actions |
| `digest` | SHA-256 of canonical JSON excluding `digest` and `states`; identifies the effective interface contract, not a signature |
| `states` | Independently observed readiness and wire properties; not part of the immutable digest |

Each interface declares `kind`, runtime-relative `direction`, absolute `endpoint`,
fully qualified `message_type`, and a registered `capability`. Topic interfaces
also declare QoS. Direction is `publish`/`subscribe` for topics and `serve` for
services/actions. A consumer observation binds a published topic, and a consumer
action publisher binds a subscribed topic. Service/action clients use the same
descriptor, but are not tensor-to-topic action executors.

Camera streams have separate IDs, for example `camera.front.color`,
`camera.front.color_info`, `camera.front.depth`, `camera.front.depth_info`,
`camera.front.aligned_depth`, and `camera.front.points`. Only enabled streams
are exported. Image interfaces link to a declared `CameraInfo` interface.
Calibration matrices remain authoritative in CameraInfo, and extrinsics in TF.

Lidar interfaces declare their actual wire type. In particular, Livox
`xfer_format: 1` is `livox_ros_driver2/msg/CustomMsg`, not PointCloud2.
`lidar.mid360.imu` and an enabled `.scan` converter are separate typed interfaces.
The description also covers joint state and command channels, trajectory actions,
the runtime services, motion services and mobile-base interfaces.

Private IPs, serial numbers, device paths, calibration file paths, driver-specific
parameters and network configuration are not copied into the public projection.
Logical IDs remain stable when an instance changes those private settings.

## Configured Versus Observed

An image interface has:

- `configured_profile`: source width, height, FPS and wire encoding for this run.
  Unknown wire encoding is `null`; an unknown whole profile may also be `null`.
- `supported_profiles`: explicitly supplied device-supported tuples, or `null`
  when not known. The configured tuple is not evidence of all supported modes.
- `states.<id>.observed_profile`: dimensions and encoding read from received
  images, plus FPS measured from receipt intervals. Unknown values stay `null`.

The runtime reports `unknown` before enough samples, `ready` after valid samples,
`mismatch` for invalid payloads or incompatible dimensions/encoding/frame, and
`stale` after two seconds without new samples. Transient-local static interfaces
do not become stale solely because they publish once. `last_seen` is Unix time;
live remote deployments need synchronized clocks. An old YAML snapshot is useful
for offline conversion, but is not proof that a device is currently online.

The simulated sensor publisher uses the same message types, image sizes, encoding,
frames, QoS and requested rates. Unsupported synthetic image formats or unknown
configured image profiles fail explicitly. Simulation validates integration, not
real-device supported modes or hardware timing guarantees.

## Discovery And Binding

`RuntimeStatus.interface_description_json` and `/runtime/get_status` expose the
document. The existing capability names and `capabilities_json` remain available;
typed interface consumers should use the public descriptor instead of guessing
types from topic names or from the older capability `topics` arrays.

`runtime_interfaces --profile <profile.yaml> --output <description.yaml>` exports
the configured public projection offline. `--peripherals <fragment.yaml>` applies
instance overrides, and `--simulated true|false` selects execution metadata.
`--instance-id` supplies the public deployment identity without changing the profile.
`runtime_interfaces --validate <description.yaml>` validates an existing snapshot.

For live acquisition, `wait_for_runtime` supports `--description-output`,
`--require-interfaces` and `--instance-id`. It validates identity, schema, digest,
capability requirements and live samples. `--interface-requirements` supplies
per-interface consumer constraints, so it continues waiting for sufficient measured
FPS rather than weakening a requirement. Required command subscriptions,
services and actions must also exist with the expected ROS types; topic QoS must
be compatible. The validated description is written atomically.

Set `runtime.instance_id` in a deployment or pass `instance_id:=...` to a robot
launch. Without it, the runtime name is the default singleton identity, not a
globally unique physical-device identifier. Multiple runtime instances must use
isolated ROS graphs/endpoints; an ID alone does not namespace ROS topic names.

IB-Robot observation/action bindings use `interface:` IDs and optional `requires`.
For example, `requires: {width: 640, height: 480, min_fps: 15}` constrains the source;
`image: {resize: [224, 224]}` remains model preprocessing. Source encoding and
source transport dimensions are not inferred from that resize. See
`src/robot_config/INTERFACE_BINDING.md` for YAML and binding errors.

Live robot and embodied launch paths start the provider once, obtain and validate
the descriptor, bind the consumer contract, atomically write a complete effective
consumer YAML, then construct inference/recording/embodied consumers. Failed
binding starts no such consumers. The snapshot remains after shutdown for offline
recording conversion. Existing explicit topic contracts retain their old behavior.

## Simulation Example

`docs/examples/so101_interface_consumer.yaml` contains no camera driver metadata.
It selects the front camera and joint-state interfaces from the SO-101 runtime.
From the workspace root after loading `.shrc_local`:

```bash
ros2 launch robot_config robot.launch.py \
  config_path:="$WORKSPACE/docs/examples/so101_interface_consumer.yaml" \
  use_sim:=true with_inference:=false moveit_display:=false
```

This is a simulated-transport example; it does not command a physical robot.
For offline use, set `runtime.interface_description` to a previously exported
YAML/JSON snapshot or embed the same document. Offline binding performs no ROS I/O
and preserves uncertainty when only configured, rather than observed, values exist.

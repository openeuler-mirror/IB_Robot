# Public Runtime Interface Binding

Contracts may select logical IDs from the robot runtime's public ROS description.
This is opt-in per observation or action publisher. Existing topic/peripheral
contracts retain their legacy resolution and launch behavior.

```yaml
robot:
  name: camera_consumer
  runtime:
    provider: example_robot
    profile: front_camera
    instance_id: unit-1
    target: simulation
    # Offline only: relative to this robot YAML, or an embedded description mapping.
    # interface_description: front_camera_description.yaml
  default_control_mode: teleop
  contract:
    observations:
      - key: observation.images.front
        interface: camera.front.color
        requires: {width: 640, height: 480, encoding: bgr8, min_fps: 20}
        image: {resize: [224, 224], encoding: rgb8}
        align: {strategy: asof, stamp: header, tol_ms: 25}
    actions:
      - key: action
        publish:
          interface: base.cmd_vel
          requires: {message_type: geometry_msgs/msg/Twist, capability: base.cmd_vel}
```

`example_robot` above stands for an installed provider whose profile exposes those
IDs. No camera driver, peripheral inventory, or model bundle is needed to bind the
consumer contract. The provider remains responsible for its driver configuration.

## Offline

`load_robot_config_dict(path)`, `load_robot_config(path).to_contract()`, and
`build_contract_from_robot_config_dict(config)` share the binding step.
`load_contract_config(contract, robot_config=config)` also accepts the complete
runtime context. The public `load_robot_section(path)` API also binds logical IDs
without requiring model bundles. A logical contract without a description fails
closed, including through the legacy standalone contract loader. Standalone
contract exports contain resolved endpoints and source metadata, not logical
binding declarations; retain the full robot YAML for rebinding.

`runtime.interface_description` accepts a YAML/JSON path or an embedded mapping.
The schema is owned and validated by `robot_runtime.interface_description`; the
binder performs no ROS I/O. `bind_robot_interfaces(config, descriptor)` returns a
deep copy without changing either argument.

Identity checks compare `runtime.provider` with `robot.runtime_name`, optional
`runtime.instance_id` with `robot.id`, optional `runtime.runtime_version` with
`robot.runtime_version`, and an explicitly configured robot `type` with the
description's `robot.type`. Hardware/simulation targets must match execution.
The consumer `name` is not a runtime instance ID.

Observations require provider-published topics; action publishers require
provider-subscribed topics. Explicit topic/type values must match exactly. Explicit
QoS uses DDS requested/offered reliability and durability compatibility; history
must be `keep_last` and queue depth is a positive local setting.

Requirements support `width`, `height`, `encoding`, `min_fps`, `message_type`, and
`capability`. Ready observed profiles take precedence. Unknown measured FPS never
falls back to configured FPS. Offline binding otherwise uses configured profiles
with explicit `_interface_source.uncertainty` entries; this is not proof of live
readiness. Unknown encoding cannot satisfy an encoding requirement.

Each bound stream retains `_interface_source`, including the descriptor digest,
state, source profile and its origin, configured/observed profiles, frame, camera
info topic, provider QoS, and uncertainty. `image.resize` and `image.encoding` remain
model preprocessing settings. RTP dimensions/FPS default from the source profile,
not model resize; explicit transport settings are preserved. Nothing is projected
back into generic `peripherals`.

## Live Launch

`robot.launch.py` detects logical IDs and explicitly defers their binding during
the structural load. It starts the provider once and runs `wait_for_runtime` with
`--description-output`, `--require-interfaces`, optional `--instance-id`, and the
existing capability `--required` arguments. The waiter must exit successfully only
after the requested published interfaces are ready and the snapshot is written.

On success the launch callback validates and binds the live description, writes a
complete `{robot: effective_config}` atomically into one launch-owned temporary
YAML, then constructs consumers. Recording and inference receive that same
`_config_path`; continuous recording uses public descriptor topics rather than
guessed peripheral paths. The continuation cannot launch the provider or simulation
backend a second time. A failed waiter, invalid description, or unsatisfied binding
shuts down without constructing consumers. This path requires a provider and is
not used for benchmark targets. An offline snapshot does not bypass live readiness.

Successful snapshots remain available after launch shutdown because recording
metadata references their paths for offline conversion. Failed launches without a
materialized snapshot remove their temporary directory. These paths are under the
system temporary directory and remain subject to its retention policy.

## Integration Boundaries

The waiter accepts ROS launch arguments and waits for multiple valid image samples
before declaring an observed stream ready. Required command subscriptions,
services and actions are verified against the ROS graph. Configured FPS never
substitutes for measured FPS in a live requirement check.

The embodied launch uses the same provider continuation for logical bindings, so
its visual and motion consumers also wait for the shared effective snapshot.
Existing motion authorization is unchanged.

`SpecView` carries publisher QoS into `TopicExecutor`. Supported topic executions
are Float64MultiArray, JointTrajectory and a three-component Twist (body-frame
vx/vy/wz). Twist values must already be SI body velocities, with three selector
names for scheduled safe-stop and `safety_behavior: zeros`; holding a nonzero
velocity is rejected. Unsupported message types fail before publishers start.
Runtime service/action descriptors do not imply arbitrary policy executor support.
The legacy dispatcher propagates named interface-binding errors.

Focused tests are in `test/test_interface_binding.py` and require the real public
description module and schema. No replacement schema or ROS message stubs are used.

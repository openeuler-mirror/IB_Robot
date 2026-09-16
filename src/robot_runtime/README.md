# robot_runtime

Robot runtime contract layer: the canonical capability vocabulary, the runtime
facade that serves the contract over a ros2_control execution stack, the mock
runtime reference implementation, launch-time capability reconciliation, the
runtime profile loader, and the parameterized conformance suite every runtime
must pass.

## Responsibility

- **Public interface schema** (`schemas/interface_description.schema.json`):
  versioned effective ROS endpoints, identity/digest, camera source profiles,
  QoS and independently observed states. See `docs/robot_interface_schema.md`.
  `runtime_interfaces` exports/validates YAML; `/runtime/get_status` exposes
  the same document in `RuntimeStatus.interface_description_json`.

- **Contract surface** (`contract.py`): canonical, runtime-neutral topic and
  service names (`/runtime_status`, `/runtime/set_mode`, `/runtime/stop`,
  `/motion/compute_fk`, `/motion/move_to_pose`, ...), lifecycle and outcome
  vocabularies. Generic consumers depend on these names only.
- **Capability vocabulary** (`capabilities.py`): frozen, extension-only
  registry (`joint.*`, `gripper.*`, `base.*`, `motion.*`, `runtime.stop`,
  reserved X2 names) plus the launch-time reconciliation check.
- **Runtime profile** (`profile.py`): the single source of runtime parameters
  (identity, modes → controller sets, declared capabilities with parameters,
  stop latency bounds, command channels, trajectory endpoints, hardware
  components). Missing required keys fail fast naming the key and the file.
- **Mode model** (`modes.py`): named controller activation sets with
  validated transitions and per-channel rejected-command counters. It does
  not arbitrate commands: "one command source per joint group" is enforced
  by the execution stack (controller_manager activation), never re-implemented here.
- **Runtime facade** (`facade_node.py`): non-real-time node that maps
  `SetRuntimeMode` onto `controller_manager/switch_controller`, publishes
  `RuntimeStatus` (1 Hz + on change), and implements `StopRuntime` with the
  contract's ordered guarantees — cancel trajectory goals → idle activation
  set → (TORQUE_OFF) hardware component deactivation — measuring each step.
- **Mock runtime** (`mock_runtime_node.py`): full-contract in-memory
  implementation with deterministic planar-chain kinematics and a base
  profile. Baseline for the conformance suite and the provider used by the
  core-only independence gate.
- **Reconciliation** (`wait_for_runtime.py`): replaces `wait_for_controllers`
  as the upper-layer start condition; fails fast naming missing capabilities.
- **Launch support** (`launch_support.py`): profile path resolution
  (`$(find)` / `$(env)`), xacro rendering, controller_manager parameter
  files, spawners derived from the mode table, and the shared stack
  composition used by `<robot>_robot` launch entries.
- **Conformance suite** (`test/test_conformance.py`): the executable contract.

## Prohibited

- Carrying joint commands or implementing trajectory interpolation,
  arbitration, or watchdogs (ros2_control controllers own execution).
- Robot-specific logic of any kind (that lives in `<robot>_robot` runtimes).
- Depending on a motion-planning framework (runtimes implement the motion
  services; this package only names them).

## Running the conformance suite

Headless against the mock runtime (spun in-process):

```bash
python3 -m pytest src/robot_runtime/test/test_conformance.py -q
```

Against a real runtime through its own launch entry (design D8 — the suite
never substitutes a stand-in stack):

```bash
CONFORMANCE_LAUNCH="ros2 launch so101_robot runtime.launch.py profile:=so101_single_arm simulated:=true" \
CONFORMANCE_PROFILE=<path> \
python3 -m pytest src/robot_runtime/test/test_conformance.py -q
```

Tests are scoped by the capabilities the target declares in `RuntimeStatus`;
every assertion message names the requirement and the observed deviation.

## Independence gates

`scripts/verify_runtime_independence.sh` (repo root) automates both
directions of the robot-runtime-packaging acceptance: `runtime <robot>_robot`
builds the closure in isolation and certifies it, `core` builds the generic
set with every robot package excluded and composes against the mock runtime.

## Stop guarantees (facade)

| Policy | Guarantees, in order | Reached via |
|---|---|---|
| `HOLD` | cancel in-flight trajectories and reject streaming; idle activation set with joints holding; lifecycle `STOPPED`, latched | `<action>/_action/cancel_goal` (cancel-all) → `switch_controller` (STRICT) |
| `TORQUE_OFF` | `HOLD` guarantees, then joint torque released | + `set_hardware_component_state(inactive)` → adapter `on_deactivate` → SDK safe-stop |

The latch clears only on `SetRuntimeMode("idle")`; after `TORQUE_OFF` the
facade reactivates the hardware components before clearing.

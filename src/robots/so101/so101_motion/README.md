# so101_motion

SO-101 motion package, member of the `so101_robot` runtime (also reused by
`lekiwi_robot`, whose arm is an SO-101). MoveIt assets plus the motion server
that implements the runtime-neutral motion services.

## Responsibility

- **Motion assets** (`config/so101/`): SRDF, kinematics plugin config, joint
  limits, OMPL planning pipeline, MoveIt controller mapping, RViz config.
- **Motion server** (`scripts/motion_server.py`): serves the
  `robot-motion-services` contract over MoveIt —
  `/motion/move_to_pose` (`MoveToPose`), `/motion/move_to_joint`
  (`MoveToConfiguration`), `/motion/compute_fk` (`ComputeFk`),
  `/motion/compute_ik` (`ComputeIk`, one endpoint per IK worker namespace,
  declared in the runtime profile's `motion.ik.endpoints`). Owns the SO-101
  5-DOF orientation strategies (gripper Z-axis constraint, shoulder-plane
  projection, current-orientation fallback) so callers never supply
  robot-specific strategy. Gated by `RuntimeStatus`: executing moves are
  `REJECTED` while the runtime is `STOPPED` or the mode does not permit
  trajectory execution, and a runtime stop pre-empts an in-flight move
  (`CANCELLED`).
- **Launch** (`launch/motion.launch.py`): `move_group` + motion server
  (+ RViz, + isolated IK workers via `ik_workers.launch.py`). Included by the
  runtime launch; all arguments derive from the runtime profile.
- **Placo servo** (`scripts/so101_placo_servo_node.py`): SO-101 Cartesian
  teleoperation backend (QP differential IK).

## Prohibited

- Reading `robot_config` or any generic IB-Robot package.
- Controller switching or navigation gating (the runtime facade and the
  base node own modes; the former gateway's motion-mode block was removed).
- Exposing MoveIt message types to generic consumers.

## Testing

`test/test_motion_server_lifecycle.py` (motion ownership, cancellation,
post-motion feedback barrier, runtime gate and outcomes) runs headless with
fakes. The full service contract is certified by the runtime conformance
suite launched through `so101_robot/launch/runtime.launch.py`.

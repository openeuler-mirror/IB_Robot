# so101_suite

SO-101 robot suite: robot-specific grasp geometry and wrist guards consumed by the generic manipulation pipeline through configuration-declared providers.

## Responsibility

- **Gripper mesh geometry** (`gripper_geometry.py`): SO-101 STL loading, convex hull, jaw motion, tabletop clearance, batch collision metrics
- **Joint-5 wrist guards** (`wrist_guard.py`): branch continuity, closing-axis correction, within-limit checks, retry logic

## Prohibited

- Importing from `manipulation_execution` (avoids a circular dependency; the suite is self-contained)
- Generic grasp math (lives in `manipulation_execution.geometry`)

## Design

Per design D8: the generic manipulation pipeline resolves robot-specific geometry/guards through configuration-declared providers. Robots without an `so101_suite` provider have the corresponding grasp features explicitly disabled (tabletop filtering, joint-5 guards) rather than crashing.

## Testing

Direct unit tests in `test/`:
- `test_gripper_geometry.py` — synthetic-STL mesh tests (headless, no repo meshes needed), quaternion normalization, batch/scalar parity, plus real-mesh parity when `so101_description` meshes are present
- `test_wrist_guard.py` — joint-5 branch continuity, closing-axis correction, retry, HOME limits
- `test_provider_contract.py` — verifies the suite satisfies the generic pipeline's provider contracts (`manipulation_execution.providers`)

The generic pipeline (`manipulation_execution`) declares no dependency on this package and resolves it only through configuration-declared providers at runtime; the generic core stays buildable and launchable without any robot suite.

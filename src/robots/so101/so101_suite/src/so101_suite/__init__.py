"""SO-101 robot suite: robot-specific grasp geometry and wrist guards.

Contains the SO-101-specific parts extracted from the former
``manipulation_execution.so101_geometry`` and
``manipulation_execution.so101_kinematics_guard`` modules (design D8):

- Gripper mesh geometry (STL loading, convex hull, jaw motion)
- Joint-5 wrist roll guards (branch continuity, closing-axis correction)

The generic manipulation pipeline in ``manipulation_execution`` consumes
these through configuration-declared providers. Robots without an
``so101_suite`` provider have the corresponding grasp features explicitly
disabled.
"""

from so101_suite.gripper_geometry import (
    gripper_geometry_metrics_batch,
    gripper_mesh_min_z,
    gripper_mesh_vertices,
    tabletop_clearance,
)
from so101_suite.wrist_guard import (
    Joint5RetryResult,
    apply_joint5_retry,
    canonicalize_joint5,
    joint5_branch_continuity_check,
    joint5_branch_delta,
    joint5_branch_filter_check,
    joint5_closing_axis_correction,
    joint5_within_abs_limit,
)

__all__ = [
    "Joint5RetryResult",
    "apply_joint5_retry",
    "canonicalize_joint5",
    "gripper_geometry_metrics_batch",
    "gripper_mesh_min_z",
    "gripper_mesh_vertices",
    "joint5_branch_continuity_check",
    "joint5_branch_delta",
    "joint5_branch_filter_check",
    "joint5_closing_axis_correction",
    "joint5_within_abs_limit",
    "tabletop_clearance",
]

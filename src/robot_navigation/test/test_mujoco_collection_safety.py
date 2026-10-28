"""Collection safety without the MuJoCo environment (robot_navigation).

A machine without the simulator (openEuler: no ``mujoco_ros2_control`` rpms,
no Python ``mujoco`` in the venv) still collects this package — including the
lidar-odometry gate that moved here from lekiwi_robot (PR !535 rewrite).
That fails invisibly when the gate module imported a MuJoCo-only package at
module level: the whole file errors during collection and every test in it —
including simulator-independent ones — disappears.

The gate module is imported in a fresh interpreter subprocess with those
imports blocked before anything else runs; the subprocess mechanics live in
``robot_runtime.testing.collection_safety`` (shared with lekiwi_robot's
collection-safety test).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_runtime.testing.collection_safety import import_module_with_blocked_imports

#: The MuJoCo-only imports the gate must keep inside gated code paths.
_BLOCKED_MODULES = ("mujoco", "mujoco_ros2_control_msgs", "mujoco_ros2_control")

_GATE_MODULES = ("test_lidar_odometry_gate",)

_TEST_DIR = Path(__file__).resolve().parent


@pytest.mark.parametrize("module", _GATE_MODULES)
def test_the_gate_module_collects_without_the_mujoco_environment(module):
    completed = import_module_with_blocked_imports(module, blocked=_BLOCKED_MODULES, cwd=_TEST_DIR)
    assert completed.returncode == 0, (
        f"{module} does not import without the MuJoCo environment "
        f"(a module-level MuJoCo import breaks collection on openEuler):\n"
        f"{completed.stderr.strip()[-2000:]}"
    )

# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collection safety without the MuJoCo environment (review rounds 1-2).

A machine without the simulator (openEuler: no ``mujoco_ros2_control`` rpms,
no Python ``mujoco`` in the venv) must still COLLECT every other test of this
package. That fails invisibly when a test module imports a MuJoCo-only
package at module level: the whole file errors during collection and every
test in it — including simulator-independent ones — disappears.

Each gate module is imported in a FRESH interpreter subprocess with those
imports blocked BEFORE anything else runs. The subprocess mechanics live in
``robot_runtime.testing.collection_safety`` (shared with robot_navigation's
copy of this guarantee for the moved lidar-odometry gate); this test pins the
guarantee for this package's modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_runtime.testing.collection_safety import import_module_with_blocked_imports

#: The MuJoCo-only imports the gate modules must keep inside gated code paths.
_BLOCKED_MODULES = ("mujoco", "mujoco_ros2_control_msgs", "mujoco_ros2_control")

#: Every module of this package that touches the MuJoCo platform. The
#: lidar-odometry gate moved to robot_navigation/test and is covered by that
#: package's own collection-safety test.
_GATE_MODULES = (
    "test_mujoco_platform",
    "test_mujoco_stop_gate",
    "test_chassis_tracking_gate",
)

_TEST_DIR = Path(__file__).resolve().parent


@pytest.mark.parametrize("module", _GATE_MODULES)
def test_the_gate_module_collects_without_the_mujoco_environment(module):
    completed = import_module_with_blocked_imports(module, blocked=_BLOCKED_MODULES, cwd=_TEST_DIR)
    assert completed.returncode == 0, (
        f"{module} does not import without the MuJoCo environment "
        f"(a module-level MuJoCo import breaks collection on openEuler):\n"
        f"{completed.stderr.strip()[-2000:]}"
    )

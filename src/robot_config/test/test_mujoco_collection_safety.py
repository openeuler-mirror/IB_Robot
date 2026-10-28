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

"""Collection safety without the MuJoCo environment (robot_config).

A machine without the simulator (openEuler: no ``mujoco_ros2_control`` rpms,
no Python ``mujoco`` in the venv) still collects this package — including
the navigation end-to-end test that runs on both physics platforms (the
MuJoCo leg since lekiwi-mujoco-runtime-platform D15). That fails invisibly
when the e2e module imported a MuJoCo-only package at module level: the
whole file errors during collection and every test in it — including the
simulator-independent staging and decision-layer cases — disappears
(review round 1 asked for the same guard robot_navigation and lekiwi_robot
already carry).

The e2e module is imported in a fresh interpreter subprocess with those
imports blocked before anything else runs; the subprocess mechanics live in
``robot_runtime.testing.collection_safety`` (the shared helper).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_runtime.testing.collection_safety import import_module_with_blocked_imports

#: The MuJoCo-only imports the e2e module must keep inside gated code paths.
_BLOCKED_MODULES = ("mujoco", "mujoco_ros2_control_msgs", "mujoco_ros2_control")

#: The e2e module imports its helpers as the source-path ``test`` package
#: (``from test.navigation_sim_support import ...`` — setup.py deliberately
#: does not install that package), so the subprocess runs from the package
#: root and imports the module THROUGH the package, exactly like the live
#: pytest collection does.
_E2E_MODULES = ("test.test_navigation_simulation",)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", _E2E_MODULES)
def test_the_e2e_module_collects_without_the_mujoco_environment(module):
    completed = import_module_with_blocked_imports(module, blocked=_BLOCKED_MODULES, cwd=_PACKAGE_ROOT)
    assert completed.returncode == 0, (
        f"{module} does not import without the MuJoCo environment "
        f"(a module-level MuJoCo import breaks collection on openEuler):\n"
        f"{completed.stderr.strip()[-2000:]}"
    )

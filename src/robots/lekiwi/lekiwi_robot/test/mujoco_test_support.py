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

"""The shared MuJoCo environment gate for lekiwi_robot's simulation tests.

One rule everywhere (lekiwi-mujoco-runtime-platform D10, review round 1): a
missing MuJoCo environment skips the test with a reason naming what is
missing — unless ``IBROBOT_REQUIRE_SIM_PLATFORMS`` lists ``mujoco``, in which
case the skip becomes a failure, so an acceptance run can never silently lose
the platform. The probe and the require-or-skip rule are the shared
``robot_runtime.testing.sim_environment`` implementations (one copy for the
chassis gate, the lidar-odometry gate and robot_config's navigation e2e);
this module keeps only the mujoco-only convenience wrapper. Imports of
MuJoCo-only packages (the Python ``mujoco`` module, ``mujoco_ros2_control``,
``mujoco_ros2_control_msgs``) belong INSIDE the gated code paths: a
module-level import breaks collection of the whole file on a machine without
the simulator (openEuler), hiding every other test in the package
(``test_mujoco_collection_safety.py`` pins that).
"""

from __future__ import annotations

from robot_runtime.testing.sim_environment import mujoco_environment_missing, skip_or_fail


def require_mujoco() -> None:
    """Skip by default; fail when ``IBROBOT_REQUIRE_SIM_PLATFORMS`` demands mujoco.

    Review round 1: this wrapper used to branch on
    ``required_sim_platforms()`` itself — a second copy of the require-or-skip
    rule that could drift from the shared one. It now delegates to
    ``skip_or_fail`` (one rule everywhere, the chassis gate, the
    lidar-odometry gate and robot_config's navigation e2e included).
    """
    reason = mujoco_environment_missing()
    if reason is None:
        return
    skip_or_fail("mujoco", f"the MuJoCo environment is unavailable: {reason}", context="lekiwi_robot simulation tests")

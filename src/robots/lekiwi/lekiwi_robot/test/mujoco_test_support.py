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
the platform. Imports of MuJoCo-only packages (the Python ``mujoco`` module,
``mujoco_ros2_control_msgs``) belong INSIDE the gated code paths: a
module-level import breaks collection of the whole file on a machine without
the simulator (openEuler), hiding every other test in the package
(``test_mujoco_collection_safety.py`` pins that).
"""

from __future__ import annotations

import os

import pytest

REQUIRE_ENV = "IBROBOT_REQUIRE_SIM_PLATFORMS"


def mujoco_environment_missing() -> str | None:
    """``None`` when the MuJoCo simulation environment is usable, else the reason."""
    try:
        from ament_index_python import get_resources
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        try:
            get_package_share_directory("mujoco_ros2_control")
        except PackageNotFoundError:
            return "the mujoco_ros2_control package is not installed"
        if "mujoco_3d_lidar" not in get_resources("mujoco_plugins"):
            return "the mujoco_3d_lidar ament resource is not installed"
    except ImportError:
        return "ament_index_python is unavailable"
    try:
        import mujoco as python_mujoco

        from robot_runtime.mujoco import simulator_versions

        if simulator_versions().mujoco != python_mujoco.__version__:
            return "Python mujoco does not match the simulator's libmujoco"
    except ImportError:
        return "the Python mujoco package is unavailable"
    return None


def require_mujoco() -> None:
    """Skip by default; fail when ``IBROBOT_REQUIRE_SIM_PLATFORMS`` demands mujoco."""
    reason = mujoco_environment_missing()
    if reason is None:
        return
    required = {entry.strip().lower() for entry in os.environ.get(REQUIRE_ENV, "").split(",") if entry.strip()}
    if "mujoco" in required:
        pytest.fail(f"platform 'mujoco' was required but its environment is unavailable: {reason}")
    pytest.skip(f"the MuJoCo environment is unavailable: {reason}")

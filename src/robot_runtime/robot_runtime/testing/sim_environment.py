"""Simulator-environment availability and the required-platform rule (D10/D15).

Code capability and simulator ENVIRONMENT are two separate checks (the
lidar-odometry gate's D14.2/D10 split): a platform's code path may exist
while the host cannot run its simulator. Tests that need a real simulator
skip with a reason naming what is missing — unless
``IBROBOT_REQUIRE_SIM_PLATFORMS`` (comma-separated platform names) lists the
platform, in which case any skip of it fails instead, so an acceptance run
can never silently lose a platform.

This is test-support API, not runtime API (see ``robot_runtime/testing``):
importing this module starts no process and creates no ROS entity; the
probes run short-lived subprocesses and ament lookups only when called. The
module lives here — not in a robot package — so any test in a package that
depends on robot_runtime (robot_config's navigation end-to-end test,
lekiwi_robot's gates) can share ONE rule without a test_depend cycle
(lekiwi-mujoco-runtime-platform D15).
"""

from __future__ import annotations

import os
import subprocess

import pytest

#: Acceptance runs list the platforms whose skips must instead fail (D10).
REQUIRED_SIM_PLATFORMS_ENV = "IBROBOT_REQUIRE_SIM_PLATFORMS"


def gazebo_available() -> bool:
    """Whether a Gazebo CLI can start (either stack: Fortress ``ign`` or gz)."""
    for cmd in (["ign", "gazebo", "--version"], ["gz", "sim", "--version"]):
        try:
            if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def mujoco_environment_missing() -> str | None:
    """Why the MuJoCo simulator environment is unusable, or None when fine.

    The ament resources the C++ node needs (the control package and the
    ``mujoco_3d_lidar`` engine extension) and a Python ``mujoco`` matching
    the simulator's libmujoco (the compiler must bake the same version).
    """
    try:
        from ament_index_python import get_resources
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        try:
            get_package_share_directory("mujoco_ros2_control")
        except PackageNotFoundError:
            return "the mujoco_ros2_control package is not installed (apt install ros-humble-mujoco-ros2-control)"
        resources = get_resources("mujoco_plugins")
        if "mujoco_3d_lidar" not in resources:
            return "the mujoco_3d_lidar ament resource is not installed (apt install ros-humble-mujoco-3d-lidar)"
    except ImportError:
        return "ament_index_python is unavailable"
    try:
        import mujoco as python_mujoco  # noqa: PLC0415

        from robot_runtime.mujoco import simulator_versions

        versions = simulator_versions()
        if versions.mujoco is None:
            return "the mujoco_vendor libmujoco library is not installed"
        if python_mujoco.__version__ != versions.mujoco:
            return (
                f"Python mujoco {python_mujoco.__version__} != simulator libmujoco {versions.mujoco}; "
                "the compiler and the simulator must match"
            )
    except ImportError:
        return "the Python mujoco package is unavailable"
    return None


def environment_missing(sim_platform: str) -> str | None:
    """Why the platform's simulator cannot run here, or None when it can."""
    if sim_platform == "gazebo":
        return None if gazebo_available() else "ign/gz gazebo is unavailable"
    if sim_platform == "mujoco":
        return mujoco_environment_missing()
    return f"no environment check for platform {sim_platform!r}"


def required_sim_platforms() -> set[str]:
    """The platforms ``IBROBOT_REQUIRE_SIM_PLATFORMS`` names (lowercased)."""
    listed = os.environ.get(REQUIRED_SIM_PLATFORMS_ENV, "").strip()
    if not listed:
        return set()
    return {entry.strip().lower() for entry in listed.split(",") if entry.strip()}


def skip_or_fail(sim_platform: str, reason: str, context: str) -> None:
    """Skip — unless acceptance runs required this platform: then fail.

    ``context`` names the calling test in the failure message (e.g.
    ``"navigation simulation"``, ``"lidar-odometry gate"``).
    """
    if sim_platform in required_sim_platforms():
        pytest.fail(f"{context}: platform {sim_platform!r} was required by {REQUIRED_SIM_PLATFORMS_ENV}: {reason}")
    pytest.skip(reason)

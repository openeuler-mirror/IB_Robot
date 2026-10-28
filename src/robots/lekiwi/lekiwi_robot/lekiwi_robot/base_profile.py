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

"""The runtime profile's ``base:`` section is the only source of chassis geometry.

Pure functions (no ROS context) so the launch entry and tests share them:

- :func:`base_node_parameters` turns the profile into base node parameters and
  fails naming any missing key. There are deliberately no LeKiwi constants to
  fall back on -- a profile that forgot its wheel radius must not silently
  drive with someone else's.
- :func:`validate_base_limits` rejects published ``base.cmd_vel`` limits the
  wheels cannot realize. The envelope comes from the SDK's kinematics, the same
  implementation that projects commands, so the check cannot disagree with it.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import lekiwi_sdk_py as lk

from robot_runtime.launch_support import RuntimeProfileError, render_robot_description

# Keys the base node needs; every one must be declared by the profile.
REQUIRED_BASE_KEYS = (
    "wheel_joints",
    "wheel_radius",
    "base_radius",
    "max_wheel_radps",
    "control_frequency",
    "odom_frame",
    "base_frame",
    "publish_tf",
)
# Published velocity limits, by axis, and the body velocity that probes each.
LIMIT_AXES = {"max_vx": (1.0, 0.0, 0.0), "max_vy": (0.0, 1.0, 0.0), "max_wz": (0.0, 0.0, 1.0)}
# Declared limits are usually rounded from the envelope; allow that rounding
# but nothing a real axis could notice.
_LIMIT_TOLERANCE = 1e-6


def _require(section: dict[str, Any], key: str, where: str, profile_path: str) -> Any:
    if key not in section or section[key] is None:
        raise RuntimeProfileError(f"runtime profile missing required key '{where}{key}': {profile_path}")
    return section[key]


def geometry_from_base(base: dict[str, Any]) -> lk.BaseGeometry:
    geometry = lk.BaseGeometry()
    geometry.wheel_radius = float(base["wheel_radius"])
    geometry.base_radius = float(base["base_radius"])
    geometry.max_wheel_radps = float(base["max_wheel_radps"])
    return geometry


def realizable_limits(geometry: lk.BaseGeometry) -> dict[str, float]:
    """Largest speed per axis at which no wheel exceeds ``max_wheel_radps``."""
    unbounded = lk.BaseGeometry()
    unbounded.wheel_radius = geometry.wheel_radius
    unbounded.base_radius = geometry.base_radius
    unbounded.mount_angles = geometry.mount_angles
    unbounded.max_wheel_radps = math.inf
    limits = {}
    for axis, probe in LIMIT_AXES.items():
        peak = max(abs(w) for w in lk.body_to_wheel_velocities(*probe, unbounded))
        limits[axis] = geometry.max_wheel_radps / peak if peak > 0.0 else math.inf
    return limits


def validate_base_limits(profile: dict[str, Any], profile_path: str) -> None:
    base = profile.get("base") or {}
    cmd_vel = (profile.get("capabilities") or {}).get("base.cmd_vel")
    if not base or cmd_vel is None:
        return
    for key in ("wheel_radius", "base_radius", "max_wheel_radps"):
        _require(base, key, "base.", profile_path)
    bounds = realizable_limits(geometry_from_base(base))
    for axis, bound in bounds.items():
        declared = _require(cmd_vel, axis, "capabilities.base.cmd_vel.", profile_path)
        value = float(declared)
        if not math.isfinite(value) or value <= 0.0:
            # NaN/Inf/negative limits are not a clamp (min/max turns NaN into
            # +limit); the profile must declare a positive finite envelope.
            raise RuntimeProfileError(
                f"capabilities.base.cmd_vel.{axis} must be a finite positive limit, got {declared!r}: {profile_path}"
            )
        if value > bound + _LIMIT_TOLERANCE:
            raise RuntimeProfileError(
                f"capabilities.base.cmd_vel.{axis}={value:g} exceeds what the wheels can realize "
                f"({bound:.4f} at base.max_wheel_radps={float(base['max_wheel_radps']):g}): {profile_path}"
            )


def lidar_odometry_owns_tf(fast_lio_config: dict[str, Any], simulated: bool) -> bool:
    """True when the lidar odometry chain will publish ``odom -> base_link``.

    ``odom -> base_link`` must have one publisher. When the runtime launches
    FAST-LIO with its TF enabled, that is the authority (it is what the stage
    selected), and wheel odometry keeps publishing ``/odom`` without the TF.
    Mirrors ``robot_runtime.peripherals.fast_lio_nodes``, which launches nothing
    in simulation.
    """
    return (
        bool(fast_lio_config.get("enabled", False)) and not simulated and bool(fast_lio_config.get("publish_tf", True))
    )


def base_node_parameters(profile: dict[str, Any], profile_path: str) -> dict[str, Any]:
    base = _require(profile, "base", "", profile_path)
    for key in REQUIRED_BASE_KEYS:
        _require(base, key, "base.", profile_path)
    if not base["wheel_joints"]:
        raise RuntimeProfileError(f"runtime profile key 'base.wheel_joints' is empty: {profile_path}")
    validate_base_limits(profile, profile_path)
    cmd_vel = _require(profile.get("capabilities") or {}, "base.cmd_vel", "capabilities.", profile_path)
    # D3.2: the published envelope reaches the base node as parameters — the
    # one authoritative clamp ahead of the IK and the wheel saturation. The
    # axes are exactly LIMIT_AXES, so a missing key fails here naming it.
    envelope = {axis: float(_require(cmd_vel, axis, "capabilities.base.cmd_vel.", profile_path)) for axis in LIMIT_AXES}
    return {
        "wheel_joints": [str(j) for j in base["wheel_joints"]],
        "wheel_radius": float(base["wheel_radius"]),
        "base_radius": float(base["base_radius"]),
        "max_wheel_radps": float(base["max_wheel_radps"]),
        "control_frequency": float(base["control_frequency"]),
        "staleness_s": float(_require(cmd_vel, "staleness_s", "capabilities.base.cmd_vel.", profile_path)),
        **envelope,
        "joint_states_topic": str(_require(profile, "joint_state_topic", "", profile_path)),
        "odom_frame": str(base["odom_frame"]),
        "base_frame": str(base["base_frame"]),
        "publish_tf": bool(base["publish_tf"]),
        # The one geometry-free key keeps a default, and it is the safe one:
        # the navigation gate starts closed unless the profile says otherwise.
        "navigation_enabled_on_startup": bool(base.get("navigation_enabled_on_startup", False)),
        "base_modes": [
            name for name, spec in profile["modes"].items() if name != "initial" and (spec or {}).get("allows_base")
        ],
    }


def arm_motion_model(profile: dict[str, Any], profile_path: str | Path, simulated: bool) -> str:
    """Effective URDF for arm motion planning: the SO-101 arm alone.

    so101_motion's SRDF describes the robot ``so101``, and MoveIt is fed the
    arm-only joint states, so its model must be the arm without the chassis.
    It is rendered from ``motion.model_xacro`` with this profile's hardware
    arguments (port, calibration, simulated), exactly as the SO-101 runtime
    renders its own description, so calibrated limits stay authoritative.
    """
    motion = profile.get("motion") or {}
    xacro = _require(motion, "model_xacro", "motion.", str(profile_path))
    arm = copy.deepcopy(profile)
    arm["description"] = {
        "xacro": xacro,
        "xacro_args": dict((profile.get("description") or {}).get("xacro_args") or {}),
    }
    return render_robot_description(arm, Path(profile_path), simulated)

"""Runtime profile: the single source of runtime parameters (robot-runtime-packaging spec).

A runtime profile is a YAML file owned by a ``<robot>_robot`` package. The
facade, the runtime launch entry, and the conformance suite all read it.
Missing required keys fail fast naming the key and the profile path; no
robot-instance value (ports, calibration paths) is embedded in code.

Facade-relevant shape::

    runtime:
      name: so101_robot
      version: 0.1.0
    simulated: false
    controller_manager: controller_manager
    modes:                                  # see robot_runtime.modes.ModeModelConfig
      initial: idle
      idle: {controllers: [], transitions: [stream, trajectory]}
      ...
    capabilities:                           # names from robot_runtime.capabilities
      joint.state: {joint_count: 6, rate_hz: 50.0}
      joint.trajectory: {joint_count: 6}
      runtime.stop: {cancel_bound_s: 0.5, idle_bound_s: 1.0, torque_off_bound_s: 2.0}
    joints: ["1", "2", "3", "4", "5", "6"]
    command_channels:                       # observed for rejected-command counting
      - {channel: arm_stream, topic: /arm_position_controller/commands}
    trajectory_actions:                     # cancelled on stop
      - /arm_trajectory_controller/follow_joint_trajectory
    hardware_components: [so101_system]     # deactivated on TORQUE_OFF
    joint_state_topic: /joint_states
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_runtime.capabilities import validate_capability_set
from robot_runtime.contract import IDLE_MODE, STOP_POLICIES

_REQUIRED_TOP_LEVEL = ("runtime", "modes", "capabilities", "joints")
_REQUIRED_RUNTIME = ("name", "version")
_REQUIRED_STOP_BOUNDS = ("cancel_bound_s", "idle_bound_s", "torque_off_bound_s")


class ProfileError(ValueError):
    """Raised when a runtime profile is missing or invalid."""


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate a runtime profile. Raises ``ProfileError`` naming the key and path."""
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ProfileError(f"runtime profile not found: {profile_path}")
    with profile_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ProfileError(f"runtime profile must be a mapping: {profile_path}")
    validate_profile(data, profile_path)
    return data


def validate_profile(data: dict[str, Any], source: str | Path = "<profile>") -> None:
    """Validate the facade-relevant sections. Raises ``ProfileError``."""
    for key in _REQUIRED_TOP_LEVEL:
        if key not in data:
            raise ProfileError(f"runtime profile missing required key {key!r}: {source}")
    runtime = data["runtime"]
    if not isinstance(runtime, dict):
        raise ProfileError(f"runtime profile 'runtime' must be a mapping: {source}")
    for key in _REQUIRED_RUNTIME:
        if not str(runtime.get(key, "")).strip():
            raise ProfileError(f"runtime profile missing required key 'runtime.{key}': {source}")

    modes = data["modes"]
    if not isinstance(modes, dict) or IDLE_MODE not in modes:
        raise ProfileError(f"runtime profile 'modes' must declare {IDLE_MODE!r}: {source}")

    capabilities = data["capabilities"]
    if isinstance(capabilities, list):
        capabilities = {str(name): {} for name in capabilities}
        data["capabilities"] = capabilities
    if not isinstance(capabilities, dict) or not capabilities:
        raise ProfileError(f"runtime profile 'capabilities' must be a non-empty mapping or list: {source}")
    invalid = validate_capability_set(list(capabilities))
    if invalid:
        raise ProfileError(f"runtime profile declares unknown capabilities {invalid}: {source}")
    for name, params in capabilities.items():
        if params is None:
            capabilities[name] = {}
        elif not isinstance(params, dict):
            raise ProfileError(f"runtime profile capability {name!r} parameters must be a mapping: {source}")
    if "runtime.stop" in capabilities:
        bounds = capabilities["runtime.stop"]
        for key in _REQUIRED_STOP_BOUNDS:
            value = bounds.get(key)
            if not isinstance(value, int | float) or value <= 0:
                raise ProfileError(
                    f"runtime profile 'capabilities.runtime.stop.{key}' must be a positive number: {source}"
                )

    joints = data["joints"]
    if not isinstance(joints, list) or not joints or not all(str(j).strip() for j in joints):
        raise ProfileError(f"runtime profile 'joints' must be a non-empty list of names: {source}")

    ros2_control = data.get("ros2_control") or {}
    for key, default in (
        ("simulated", False),
        ("controller_manager", "controller_manager"),
        ("command_channels", []),
        ("trajectory_actions", []),
        ("hardware_components", list(ros2_control.get("hardware_components", []))),
        ("joint_state_topic", "/joint_states"),
    ):
        data.setdefault(key, default)
    for entry in data["command_channels"]:
        if not isinstance(entry, dict) or not entry.get("channel") or not entry.get("topic"):
            raise ProfileError(f"runtime profile 'command_channels' entries need 'channel' and 'topic': {source}")
        channel_type = str(entry.get("type", "float64_array")).lower()
        if channel_type not in ("float64_array", "twist"):
            raise ProfileError(f"runtime profile unsupported command channel type {channel_type!r}: {source}")
        entry["type"] = channel_type

    default_policy = str(data.get("stop_default_policy", STOP_POLICIES[0]))
    if default_policy not in STOP_POLICIES:
        raise ProfileError(f"runtime profile 'stop_default_policy' must be one of {STOP_POLICIES}: {source}")
    data["stop_default_policy"] = default_policy
    if "fast_lio" in data and not isinstance(data["fast_lio"], dict):
        raise ProfileError(f"runtime profile 'fast_lio' must be a mapping: {source}")


def stop_bounds(profile: dict[str, Any]) -> dict[str, float]:
    """Stop latency bounds from the profile (empty when runtime.stop is not declared)."""
    bounds = profile.get("capabilities", {}).get("runtime.stop") or {}
    return {key: float(bounds[key]) for key in _REQUIRED_STOP_BOUNDS if key in bounds}

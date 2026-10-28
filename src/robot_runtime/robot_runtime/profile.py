"""Runtime profile: the single source of runtime parameters (robot-runtime-packaging spec).

A runtime profile is a YAML file owned by a ``<robot>_robot`` package. The
facade, the runtime launch entry, and the conformance suite all read it.
Missing required keys fail fast naming the key and the profile path; no
robot-instance value (ports, calibration paths) is embedded in code.

Facade-relevant shape::

    runtime:
      name: so101_robot
      version: 0.1.0
    use_sim: false                            # default execution (physical)
    sim_platforms: [sdk, gazebo]              # supported simulation platforms
    sim_platform: sdk                         # default platform (in sim_platforms)
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

import math
from pathlib import Path
from typing import Any

import yaml

from robot_runtime.capabilities import validate_capability_set
from robot_runtime.contract import IDLE_MODE, STOP_POLICIES

#: Simulation platforms a runtime profile may declare (runtime-simulation-platform D1).
#: ``sdk`` is the vendor SDK virtual bus behind the production hardware adapter
#: (no physics); physical execution is not a simulation platform.
SIM_PLATFORMS = ("sdk", "gazebo", "mujoco")

#: Platforms that replace the robot with a physics simulator
#: (so101-mujoco-runtime-platform D1). Every ``== "gazebo"`` test that really
#: means "a physics simulator provides the clock and the world-side rules"
#: must test membership here instead; tests about Gazebo specifics (entity
#: creation, gz bridges) keep the literal platform.
PHYSICS_SIM_PLATFORMS = ("gazebo", "mujoco")

#: Physics platforms whose sensor injection stands in for a lidar
#: (lekiwi-gazebo-runtime-platform D14.2): the rendering path consults this
#: next to the driver check (``LIDAR_SIM_DRIVERS``) before injecting a lidar
#: sensor, and the lidar-odometry gate reads it as the "the runtime simulates
#: lidar here" capability — a platform outside it renders no lidar, so the
#: constant cannot drift from production behaviour.
LIDAR_SIM_PLATFORMS = ("gazebo", "mujoco")


def lidar_driver_simulated(driver: str, sim_platform: str | None) -> bool:
    """Whether a physics platform's lidar injection stands in for ``driver`` (D5).

    The driver sets are platform-specific (Gazebo simulates every driver in
    ``robot_runtime.gazebo.LIDAR_SIM_DRIVERS``, MuJoCo only the MID-360 —
    ``robot_runtime.mujoco.MUJOCO_LIDAR_SIM_DRIVERS``); a driver the selected
    platform does not stand in for is neither simulated nor advertised there.
    Imported lazily so the platform modules keep importing this one.
    """
    if sim_platform == "gazebo":
        from robot_runtime.gazebo import LIDAR_SIM_DRIVERS

        return driver in LIDAR_SIM_DRIVERS
    if sim_platform == "mujoco":
        from robot_runtime.mujoco import MUJOCO_LIDAR_SIM_DRIVERS

        return driver in MUJOCO_LIDAR_SIM_DRIVERS
    return False


def resolve_sensor_pose(periph: dict, platform: str = "gazebo") -> dict:
    """A non-camera sensor's simulator attachment transform (D5/D13).

    The shared resolution for every physics platform (D5.2): the peripheral's
    ``simulation.<platform>.transform`` (the simulator attachment pose, D13)
    over the plain ``transform`` (the public static TF). The resolved
    transform's ``parent_frame`` is the model frame the sensor attaches to.
    """
    section = (periph.get("simulation") or {}).get(platform) or {}
    declared = section.get("transform") if isinstance(section, dict) else None
    if isinstance(declared, dict) and declared:
        return dict(declared)
    return dict(periph.get("transform") or {})


_REQUIRED_TOP_LEVEL = ("runtime", "modes", "capabilities", "joints")
_REQUIRED_RUNTIME = ("name", "version")
_REQUIRED_STOP_BOUNDS = ("cancel_bound_s", "idle_bound_s", "torque_off_bound_s")


class ProfileError(ValueError):
    """Raised when a runtime profile is missing or invalid."""


def _validate_sim_selection(data: dict[str, Any], source: str | Path) -> None:
    """Normalize and validate ``use_sim`` / ``sim_platforms`` / ``sim_platform`` (D1).

    A profile declares the simulation platforms it supports and a default
    platform. The legacy boolean ``simulated`` key stays an alias of ``use_sim``:
    both present and disagreeing fail naming both values. Without
    ``sim_platforms`` the profile supports ``[sdk]`` (today's behaviour);
    ``sim_platform`` defaults to ``sdk`` and must be declared.
    """
    sim_platforms = data.get("sim_platforms")
    if sim_platforms is None:
        sim_platforms = ["sdk"]
    elif not isinstance(sim_platforms, list) or not sim_platforms:
        raise ProfileError(
            f"runtime profile 'sim_platforms' must be a non-empty list from {list(SIM_PLATFORMS)}: {source}"
        )
    else:
        sim_platforms = [str(entry).strip() for entry in sim_platforms]
        if any(entry not in SIM_PLATFORMS for entry in sim_platforms) or len(set(sim_platforms)) != len(sim_platforms):
            raise ProfileError(
                f"runtime profile 'sim_platforms' must be unique values from {list(SIM_PLATFORMS)}: {source}"
            )
    data["sim_platforms"] = sim_platforms

    has_simulated = "simulated" in data
    if has_simulated and not isinstance(data["simulated"], bool):
        raise ProfileError(f"runtime profile 'simulated' must be a boolean (alias of 'use_sim'): {source}")
    has_use_sim = "use_sim" in data
    if has_use_sim and not isinstance(data["use_sim"], bool):
        raise ProfileError(f"runtime profile 'use_sim' must be a boolean: {source}")
    if has_simulated and has_use_sim and data["simulated"] != data["use_sim"]:
        raise ProfileError(
            f"runtime profile selects use_sim: {data['use_sim']} but 'simulated: {data['simulated']}' disagrees; "
            f"pass at most one selection: {source}"
        )
    if not has_use_sim:
        data["use_sim"] = bool(data.get("simulated", False))
    # The legacy boolean stays consistent with the effective selection so
    # consumers reading ``profile['simulated']`` cannot disagree with it.
    data["simulated"] = data["use_sim"]

    sim_platform = str(data.get("sim_platform", "") or "").strip()
    if not sim_platform:
        sim_platform = "sdk"
    if sim_platform not in SIM_PLATFORMS:
        raise ProfileError(f"runtime profile 'sim_platform' must be one of {list(SIM_PLATFORMS)}: {source}")
    if sim_platform not in sim_platforms:
        raise ProfileError(
            f"runtime profile sim_platform {sim_platform!r} is not in the supported platforms {sim_platforms}: {source}"
        )
    data["sim_platform"] = sim_platform


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

    conformance = data.get("conformance")
    if conformance is not None:
        if not isinstance(conformance, dict):
            raise ProfileError(f"runtime profile 'conformance' must be a mapping: {source}")
        start_positions = conformance.get("start_positions")
        if start_positions is not None:
            if not isinstance(start_positions, dict) or not start_positions:
                raise ProfileError(
                    f"runtime profile 'conformance.start_positions' must be a non-empty mapping of joint names to radians: {source}"
                )
            # The conformance reset drives the arm joints only (D9: a subset
            # of arm joints), with the same arm_joints-or-joints rule the
            # suite's harness uses — a gripper joint would be silently ignored.
            arm_joints = data.get("arm_joints", joints)
            non_arm = sorted(str(joint) for joint in start_positions if str(joint) not in {str(j) for j in arm_joints})
            if non_arm:
                raise ProfileError(
                    f"runtime profile 'conformance.start_positions' names non-arm joints {non_arm}: {source}"
                )
            for joint, value in start_positions.items():
                # bool is an int subclass; true/false must not become 1.0/0.0 rad targets
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                    raise ProfileError(
                        f"runtime profile 'conformance.start_positions.{joint}' must be a finite number: {source}"
                    )

    _validate_sim_selection(data, source)

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

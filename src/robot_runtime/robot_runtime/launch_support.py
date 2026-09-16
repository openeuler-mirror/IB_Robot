"""Launch-time helpers shared by ``<robot>_robot`` runtime launch files.

Runtime launch entries may depend on ``robot_runtime`` and ``ibrobot_msgs``
only (robot-runtime-packaging spec), so the pieces of ros2_control bring-up
that are not robot-specific live here rather than in a generic IB-Robot
launch package: profile path resolution, xacro rendering, the
controller_manager parameter file, and controller spawners derived from the
profile's mode table.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_runtime.contract import IDLE_MODE
from robot_runtime.interface_description import build_description
from robot_runtime.path_subst import resolve_path
from robot_runtime.peripherals import (
    load_peripherals_file,
    merge_peripherals,
    peripheral_nodes,
    synthetic_perception_nodes,
)


def render_xacro(xacro_path: str, args: dict[str, Any]) -> str:
    """Run xacro and return the URDF string. Raises on failure with xacro's stderr."""
    command = ["xacro", xacro_path] + [f"{key}:={value}" for key, value in args.items()]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed for {xacro_path}:\n{result.stderr}")
    return result.stdout


def xacro_value(value: Any) -> str:
    """Render a Python value as a xacro argument (bools lowercase, dicts as JSON)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict | list):
        import json

        return json.dumps(value)
    return str(value)


def write_controller_manager_params(robot_description: str, extra_params: dict[str, Any] | None = None) -> str:
    """Write robot_description under the ``controller_manager`` node name.

    ros2_control_node creates a node called ``controller_manager`` while launch
    writes dict parameters under the executable name, so a parameter file with
    the correct key is the reliable way to hand over the description without
    a global ``__node`` remapping.
    """
    params = {"robot_description": robot_description}
    if extra_params:
        params.update(extra_params)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="runtime_cm_") as handle:
        yaml.safe_dump({"controller_manager": {"ros__parameters": params}}, handle, default_flow_style=False)
        return handle.name


def profile_controllers(profile: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return (broadcasters, initially_active, initially_inactive) controllers from the profile.

    Broadcasters are always active. The initial mode's controllers start
    active; every other controller named by any mode starts inactive so the
    facade can switch activation sets without spawning.
    """
    modes = dict(profile.get("modes") or {})
    initial = str(modes.pop("initial", IDLE_MODE))
    broadcasters = [str(c) for c in (profile.get("ros2_control") or {}).get("broadcasters", [])]
    active: list[str] = []
    inactive: list[str] = []
    for name, spec in modes.items():
        for controller in (spec or {}).get("controllers", []):
            controller = str(controller)
            target = active if str(name) == initial else inactive
            if controller not in target and controller not in active:
                target.append(controller)
    inactive = [c for c in inactive if c not in active]
    return broadcasters, active, inactive


def spawner_nodes(
    broadcasters: list[str],
    active: list[str],
    inactive: list[str],
    controller_manager: str = "controller_manager",
    timeout_s: float = 30.0,
) -> list[Node]:
    """Stock controller_manager spawners: broadcasters + active together, inactive with ``--inactive``."""
    nodes: list[Node] = []
    common = ["--controller-manager", controller_manager, "--controller-manager-timeout", str(int(timeout_s))]
    if broadcasters or active:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                name="runtime_spawner_active",
                arguments=[*broadcasters, *active, *common],
                output="screen",
            )
        )
    if inactive:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                name="runtime_spawner_inactive",
                arguments=[*inactive, "--inactive", *common],
                output="screen",
            )
        )
    return nodes


def profile_share_path(package: str, profile_name: str) -> Path:
    """Path of ``<package>/profiles/<profile_name>.yaml`` in the installed share directory."""
    return Path(get_package_share_directory(package)) / "profiles" / f"{profile_name}.yaml"


# ---------------------------------------------------------------------------
# Whole-stack composition shared by <robot>_robot runtime launch files
# ---------------------------------------------------------------------------


class RuntimeProfileError(ValueError):
    """A runtime profile lacks a key the launch entry needs (named in the message)."""


def resolve_profile_argument(package: str, raw: str) -> Path:
    """``profile:=`` is a file path or the name of a profile shipped in ``<package>/profiles``."""
    raw = (raw or "").strip()
    if not raw:
        raise RuntimeProfileError(
            f"{package} runtime.launch.py requires profile:=<path-or-name> (robot-runtime-packaging: no default robot)"
        )
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    shipped = profile_share_path(package, raw)
    if shipped.is_file():
        return shipped
    raise RuntimeProfileError(
        f"runtime profile not found: {raw!r} (searched as a path and as {package}/profiles/{raw}.yaml)"
    )


def resolve_simulated(profile: dict[str, Any], override: str) -> bool:
    override = (override or "").strip().lower()
    if override == "":
        return bool(profile.get("simulated", False))
    return override in ("1", "true", "yes")


def render_robot_description(profile: dict[str, Any], profile_path: Path, simulated: bool) -> str:
    """Render the profile's xacro with hardware parameters; fails fast on missing keys or calibration."""
    description = profile.get("description") or {}
    if not description.get("xacro"):
        raise RuntimeProfileError(f"runtime profile missing required key 'description.xacro': {profile_path}")
    hardware = profile.get("hardware") or {}
    xacro_args = {str(k): xacro_value(v) for k, v in (description.get("xacro_args") or {}).items()}
    calib = resolve_path(hardware["calib_file"]) if hardware.get("calib_file") else ""
    xacro_args.update(
        {
            "simulated": xacro_value(simulated),
            "port": str(hardware.get("port", "")),
            "calib_file": calib,
            "reset_positions": xacro_value(hardware.get("reset_positions") or {}),
        }
    )
    for key, value in (hardware.get("xacro_args") or {}).items():
        xacro_args[str(key)] = xacro_value(value)
    if not simulated and calib and not Path(calib).is_file():
        raise RuntimeProfileError(
            f"calibration file not found: {calib} (profile {profile_path}); "
            "run the calibration tool declared by the deployment before launching the runtime"
        )
    return render_xacro(resolve_path(description["xacro"]), xacro_args)


def runtime_stack_actions(
    profile: dict[str, Any],
    profile_path: Path,
    simulated: bool,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
) -> list:
    """robot_state_publisher + ros2_control_node + controller spawners + runtime facade.

    ``peripherals_file`` (deployment overrides; profile ``peripherals:`` are
    the robot's defaults) additionally composes the sensor driver nodes, or
    synthetic perception publishers in simulated transport (D12), and feeds
    the facade so RuntimeStatus declares the perception surface.
    """
    if robot_description is None:
        robot_description = render_robot_description(profile, profile_path, simulated)
    ros2_control = profile.get("ros2_control") or {}
    if not ros2_control.get("controllers_config"):
        raise RuntimeProfileError(
            f"runtime profile missing required key 'ros2_control.controllers_config': {profile_path}"
        )
    controllers_config = resolve_path(ros2_control["controllers_config"])
    cm_params = write_controller_manager_params(robot_description)
    broadcasters, active, inactive = profile_controllers(profile)
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[cm_params, controllers_config],
            remappings=[("~/robot_description", "/robot_description")],
            output="screen",
        ),
        *spawner_nodes(broadcasters, active, inactive, controller_manager=str(profile["controller_manager"])),
        Node(
            package="robot_runtime",
            executable="runtime_facade",
            name="runtime_facade",
            output="screen",
            parameters=[
                {
                    "profile": str(profile_path),
                    "peripherals": peripherals_file,
                    "simulated": simulated,
                    "initial_mode": str((profile.get("modes") or {}).get("initial", IDLE_MODE)),
                    **(
                        {"interface_description_json": json.dumps(interface_description)}
                        if interface_description is not None
                        else {}
                    ),
                    **({"instance_id": instance_id} if instance_id else {}),
                }
            ],
        ),
    ]


def peripheral_stack_actions(
    profile: dict[str, Any], peripherals_file: str, simulated: bool, *, description: dict | None = None
) -> list:
    """Sensor peripheral composition for a runtime launch (D12).

    Physical drivers + static TF on real hardware; synthetic perception
    publishers on the same topics in simulated transport.
    """
    fragment = load_peripherals_file(peripherals_file) if peripherals_file else {}
    peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    if simulated:
        effective = {**profile, "fast_lio": {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}}
        return synthetic_perception_nodes(
            peripherals, description or build_description(effective, peripherals, simulated=True)
        )
    return peripheral_nodes(peripherals, use_sim=False)


def motion_launch_arguments(
    profile: dict[str, Any], profile_path: Path, simulated: bool, display: bool
) -> dict[str, str]:
    """Launch arguments for a MoveIt-backed motion package (so101_motion/motion.launch.py)."""
    motion = profile.get("motion") or {}
    missing = [k for k in ("arm_group_name", "base_link", "ee_link", "shoulder_link") if not motion.get(k)]
    if missing:
        raise RuntimeProfileError(f"runtime profile missing required keys motion.{missing}: {profile_path}")
    trajectory_modes = [
        name
        for name, spec in (profile.get("modes") or {}).items()
        if name != "initial" and (spec or {}).get("allows_trajectory")
    ]
    arm_joints = [str(j) for j in profile.get("arm_joints", profile["joints"])]
    return {
        "use_sim_time": "false",
        "display": xacro_value(display),
        "joint_names": " ".join(arm_joints),
        "joint_state_topic": str(motion.get("joint_state_topic", profile["joint_state_topic"])),
        "arm_group_name": str(motion["arm_group_name"]),
        "base_link": str(motion["base_link"]),
        "ee_link": str(motion["ee_link"]),
        "shoulder_link": str(motion["shoulder_link"]),
        "motion_feedback_timeout_s": str(motion.get("motion_feedback_timeout_s", 0.3)),
        "motion_feedback_tolerance_rad": str(motion.get("motion_feedback_tolerance_rad", 0.12)),
        "motion_require_tf_sync": xacro_value(bool(motion.get("motion_require_tf_sync", True))),
        "motion_hardware_feedback_topic": "" if simulated else str(motion.get("motion_hardware_feedback_topic", "")),
        "trajectory_modes": " ".join(trajectory_modes) or "trajectory",
        "ik_worker_count": str(int(motion.get("ik_worker_count", 0))),
    }

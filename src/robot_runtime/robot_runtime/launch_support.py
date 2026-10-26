"""Launch-time helpers shared by ``<robot>_robot`` runtime launch files.

Runtime launch entries may depend on ``robot_runtime`` and ``ibrobot_msgs``
only (robot-runtime-packaging spec), so the pieces of ros2_control bring-up
that are not robot-specific live here rather than in a generic IB-Robot
launch package: profile path resolution, xacro rendering, the
controller_manager parameter file, and controller spawners derived from the
profile's mode table.

The Gazebo simulation platform (openspec ``so101-gazebo-runtime-transport``
D5/D7) adds platform-aware description rendering and ``gazebo_embodiment_actions``:
the runtime provider owns its embodiment (description, entity creation,
controllers, sensors and bridges, facade) while the world is owned either by
generic launch or, standalone, by a minimal empty world started here.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from launch.actions import (
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from robot_runtime.contract import IDLE_MODE
from robot_runtime.gazebo import (
    bridge_config_node,
    camera_bridge_entries,
    camera_sensor_urdf,
)
from robot_runtime.interface_description import build_description
from robot_runtime.path_subst import resolve_path
from robot_runtime.peripherals import (
    audio_nodes,
    load_peripherals_file,
    merge_peripherals,
    peripheral_nodes,
    synthetic_perception_nodes,
)
from robot_runtime.profile import PHYSICS_SIM_PLATFORMS, SIM_PLATFORMS

logger = logging.getLogger(__name__)

#: The provider-alone minimal empty world (physics + user commands + scene
#: broadcaster + the sensors system so runtime-owned cameras render, a light
#: and a ground plane). Shipped by robot_runtime; the stock gz empty.sdf has
#: no Sensors system plugin, so runtime cameras would never publish in it.
_EMPTY_WORLD_NAME = "empty"


def _empty_world_path() -> str:
    return str(Path(get_package_share_directory("robot_runtime")) / "worlds" / "empty.sdf")


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


def write_node_params(node_name: str, params: dict[str, Any]) -> str:
    """Write ``params`` under ``node_name`` as a ros__parameters YAML file.

    ros2_control_node creates its nodes with the executable-derived names, so
    a parameter file keyed by the node name is the reliable way to hand over
    parameters without a global ``__node`` remapping.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="runtime_cm_") as handle:
        yaml.safe_dump({node_name: {"ros__parameters": params}}, handle, default_flow_style=False)
        return handle.name


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
    return write_node_params("controller_manager", params)


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
    use_sim_time: bool = False,
) -> list[Node]:
    """Stock controller_manager spawners: broadcasters + active together, inactive with ``--inactive``."""
    nodes: list[Node] = []
    common = ["--controller-manager", controller_manager, "--controller-manager-timeout", str(int(timeout_s))]
    parameters = [{"use_sim_time": use_sim_time}] if use_sim_time else []
    if broadcasters or active:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                name="runtime_spawner_active",
                arguments=[*broadcasters, *active, *common],
                parameters=parameters,
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
                parameters=parameters,
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


def _parse_bool_argument(value: str, name: str) -> bool:
    text = (value or "").strip().lower()
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    raise RuntimeProfileError(f"cannot parse {name}:={value!r} as a boolean")


def resolve_sim_selection(
    profile: dict[str, Any],
    use_sim_arg: str,
    sim_platform_arg: str,
    simulated_arg: str,
    *,
    source: str = "",
) -> tuple[bool, str | None]:
    """Resolve the effective simulation selection (runtime-simulation-platform D1).

    Uses generic launch's vocabulary: ``use_sim:=`` selects simulated or
    physical execution, ``sim_platform:=`` selects the simulation platform.
    ``sim_platform:=X`` alone implies simulation on X; ``use_sim:=true``
    alone runs the profile's default platform; ``use_sim:=false`` with a
    platform fails naming both. ``simulated:=`` stays a deprecated alias of
    ``use_sim:=`` (a warning names ``use_sim``); disagreeing with ``use_sim:=``
    fails naming both. A platform the profile does not declare fails naming
    the platform, the declared set and the profile.
    """
    use_sim_explicit: bool | None = None
    simulated_override = (simulated_arg or "").strip().lower()
    if simulated_override:
        use_sim_explicit = _parse_bool_argument(simulated_override, "simulated")
        logger.warning(
            "simulated:=%s is deprecated; use use_sim:=%s (same effect)",
            simulated_override,
            str(use_sim_explicit).lower(),
        )
    use_sim_override = (use_sim_arg or "").strip().lower()
    if use_sim_override:
        use_sim_value = _parse_bool_argument(use_sim_override, "use_sim")
        if use_sim_explicit is not None and use_sim_explicit != use_sim_value:
            raise RuntimeProfileError(
                f"conflicting selection: use_sim:={use_sim_value} and simulated:={str(use_sim_explicit).lower()}; "
                "pass at most one"
            )
        use_sim_explicit = use_sim_value

    sim_platform = (sim_platform_arg or "").strip().lower()
    if sim_platform and sim_platform not in SIM_PLATFORMS:
        raise RuntimeProfileError(f"unknown sim_platform {sim_platform!r}: expected one of {list(SIM_PLATFORMS)}")

    # A platform alone implies simulation; otherwise the profile's selection
    # applies unless an argument overrode it.
    if use_sim_explicit is None:
        # the profile-level `simulated` key stays the legacy alias of `use_sim`
        use_sim = bool(sim_platform) or bool(profile.get("use_sim", profile.get("simulated", False)))
    else:
        use_sim = use_sim_explicit
    if not use_sim:
        if sim_platform:
            raise RuntimeProfileError(
                f"physical execution (use_sim:=false) cannot select sim_platform:={sim_platform!r}; pass at most one"
            )
        return False, None

    platform = sim_platform or str(profile.get("sim_platform") or "sdk")
    supported = [str(entry) for entry in (profile.get("sim_platforms") or ["sdk"])]
    if platform not in supported:
        raise RuntimeProfileError(
            f"sim_platform {platform!r} is not supported by this runtime profile (supported: {supported}, "
            f"default: {profile.get('sim_platform', 'sdk')!r}{f'; profile: {source}' if source else ''})"
        )
    return True, platform


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


_PACKAGE_URI_PATTERN = None


def _resolve_package_uris(robot_description: str) -> str:
    """Rewrite ``package://<pkg>/...`` mesh URIs to absolute ``file://`` paths.

    A description spawned on the Gazebo platform must be resolvable by a
    simulator server the provider did not configure (D7b): no resource-path
    environment hand-off is needed when the URDF carries absolute file URIs.
    """
    import re

    global _PACKAGE_URI_PATTERN
    if _PACKAGE_URI_PATTERN is None:
        _PACKAGE_URI_PATTERN = re.compile(r"package://(\w+)/")

    def replace(match: re.Match[str]) -> str:
        package = match.group(1)
        try:
            share = Path(get_package_share_directory(package))
        except Exception as exc:  # PackageNotFoundError at render time is fatal
            raise RuntimeProfileError(
                f"description references package://{package}/ but {package} is not in the ament index: {exc}"
            ) from exc
        return share.as_uri() + "/"

    return _PACKAGE_URI_PATTERN.sub(replace, robot_description)


def render_gazebo_description(
    profile: dict[str, Any],
    profile_path: Path,
    *,
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> str:
    """Render the profile's xacro for the Gazebo platform (D5, D7b).

    ``use_sim`` selects the ros2_control simulation plugin (gz
    ``GazeboSimSystem``), the SDK flag stays off, the gz plugin reads the
    profile's controllers parameter file and the effort state interface is
    exposed for the simulated joint-current publisher (D6b). Camera sensors
    are injected for the effective peripherals, meshes resolve to absolute
    ``file://`` URIs and the SDK/hardware descriptions are untouched.
    """
    description = profile.get("description") or {}
    if not description.get("xacro"):
        raise RuntimeProfileError(f"runtime profile missing required key 'description.xacro': {profile_path}")
    ros2_control = profile.get("ros2_control") or {}
    controllers_config = ros2_control.get("controllers_config") or ""
    xacro_args = {str(k): xacro_value(v) for k, v in (description.get("xacro_args") or {}).items()}
    xacro_args.update(
        {
            "use_sim": "true",
            "simulated": "false",  # the SDK virtual transport flag is off under Gazebo
            "sim_plugin": "gz_ros2_control/GazeboSimSystem",
            "sim_effort_state": "true",
            **(
                {"gz_ros2_control_parameters_file": str(Path(resolve_path(controllers_config)).resolve())}
                if controllers_config
                else {}
            ),
        }
    )
    for key, value in (profile.get("hardware") or {}).get("xacro_args", {}).items():
        xacro_args[str(key)] = xacro_value(value)
    robot_description = render_xacro(resolve_path(description["xacro"]), xacro_args)
    cameras_xml = camera_sensor_urdf(
        profile.get("peripherals") or [],
        platform="gazebo",
        spawn_pose=spawn_pose,
        frame_contract=True,
    )
    if cameras_xml:
        robot_description = robot_description.replace("</robot>", cameras_xml + "\n</robot>", 1)
    return _resolve_package_uris(robot_description)


def _on_success(start_actions, success_message: str, failure_reason: str):
    """Run launch actions only when the target process exits successfully."""

    def _handler(event, _context):
        if event.returncode == 0:
            return start_actions
        return [EmitEvent(event=Shutdown(reason=failure_reason))]

    return _handler


def simulated_joint_current_actions(profile: dict[str, Any], sim_platform: str = "gazebo") -> list:
    """The effort→JointCurrent republisher for a physics platform (D6b).

    Started only when the profile declares a ``joint.current`` interface whose
    runtime endpoint and joint names parameterize the robot-agnostic node; the
    effort→current scale comes from ``simulation.<sim_platform>.joint_current.scale``
    (default 1 per joint, matching the hardware adapter's effort-as-current
    semantics).
    """
    interface = (profile.get("interfaces") or {}).get("joint.current")
    if not interface or interface.get("message_type") != "ibrobot_msgs/msg/JointCurrent":
        return []
    joint_names = [str(name) for name in interface.get("joint_names") or []]
    if not joint_names:
        return []
    scale_section = (profile.get("simulation") or {}).get(sim_platform) or {}
    scale = scale_section.get("joint_current", {}).get("scale")
    if not isinstance(scale, list) or len(scale) != len(joint_names):
        scale = [1.0] * len(joint_names)
    return [
        Node(
            package="robot_runtime",
            executable="joint_current_publisher",
            name="simulated_joint_current",
            output="screen",
            parameters=[
                {
                    "joint_names": joint_names,
                    "endpoint": str(interface.get("endpoint", "/joint_currents")),
                    "scale": [float(value) for value in scale],
                    "joint_states_topic": str(profile.get("joint_state_topic", "/joint_states")),
                    # physics platforms run every provider node on sim time
                    "use_sim_time": True,
                }
            ],
        )
    ]


def gazebo_embodiment_actions(
    profile: dict[str, Any],
    profile_path: Path,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
    gazebo_world: str = "",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list:
    """The provider's Gazebo embodiment (D2/D5): no ``ros2_control_node``.

    robot_state_publisher on sim time; a minimal empty world plus ``/clock``
    bridge when no world is supplied (standalone verification); ``create``
    waits for the world clock, then spawns the entity at the given pose
    (including yaw); controller spawners target the in-simulator controller
    manager after ``create`` exits; the facade, the camera bridge and the
    simulated joint-current publisher complete the embodiment. The world,
    when supplied by generic launch, is not started or bridged here.
    """
    if robot_description is None:
        robot_description = render_gazebo_description(profile, profile_path, spawn_pose=spawn_pose)
    ros2_control = profile.get("ros2_control") or {}
    if not ros2_control.get("controllers_config"):
        raise RuntimeProfileError(
            f"runtime profile missing required key 'ros2_control.controllers_config': {profile_path}"
        )
    broadcasters, active, inactive = profile_controllers(profile)
    controller_manager = str(profile["controller_manager"])
    model_name = str(profile["runtime"]["name"])
    peripherals = merge_peripherals(
        profile.get("peripherals"),
        (load_peripherals_file(peripherals_file) if peripherals_file else {}).get("peripherals"),
    )

    actions: list[Any] = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}, {"use_sim_time": True}],
        )
    ]

    world_name = (gazebo_world or "").strip()
    if not world_name:
        # Standalone: bring a minimal empty world so the provider is verifiable
        # in isolation; the world's clock is bridged here because nobody else
        # owns it. IBROBOT_GAZEBO_HEADLESS=1 mirrors the generic headless mode.
        world_file = _empty_world_path()
        headless = "-s --headless-rendering " if os.environ.get("IBROBOT_GAZEBO_HEADLESS", "0") == "1" else ""
        gz_args = f"{headless}-r {world_file}"
        actions.extend(
            [
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])
                    ),
                    launch_arguments={"gz_args": gz_args}.items(),
                ),
                Node(
                    package="ros_gz_bridge",
                    executable="parameter_bridge",
                    arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
                    output="screen",
                ),
            ]
        )
        world_name = _EMPTY_WORLD_NAME

    clock_waiter = Node(
        package="robot_runtime",
        executable="wait_for_clock",
        name="wait_for_gazebo_clock",
        output="screen",
        arguments=["--topic", "/clock", "--timeout", "60.0"],
    )
    x, y, z, yaw = (float(value) for value in spawn_pose)
    create_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world",
            world_name,
            "-name",
            model_name,
            "-allow_renaming=true",
            "-topic",
            "/robot_description",
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            str(z),
            "-Y",
            str(yaw),
        ],
        output="screen",
    )
    spawners = spawner_nodes(broadcasters, active, inactive, controller_manager=controller_manager, use_sim_time=True)
    actions.extend(
        [
            clock_waiter,
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=clock_waiter,
                    on_exit=_on_success(
                        [create_entity], "creating robot entity", "Gazebo clock probe failed before robot spawn"
                    ),
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=create_entity,
                    on_exit=_on_success(
                        spawners, "entity created; spawning controllers", "Gazebo entity creation failed"
                    ),
                )
            ),
            Node(
                package="robot_runtime",
                executable="runtime_facade",
                name="runtime_facade",
                output="screen",
                parameters=[
                    {
                        "profile": str(profile_path),
                        "peripherals": peripherals_file,
                        "use_sim": True,
                        "sim_platform": "gazebo",
                        "initial_mode": str((profile.get("modes") or {}).get("initial", IDLE_MODE)),
                        "use_sim_time": True,
                        **(
                            {"interface_description_json": json.dumps(interface_description)}
                            if interface_description is not None
                            else {}
                        ),
                        **({"instance_id": instance_id} if instance_id else {}),
                    }
                ],
            ),
            *simulated_joint_current_actions(profile),
        ]
    )
    camera_entries = [
        entry
        for periph in peripherals
        if periph.get("type") == "camera"
        for entry in camera_bridge_entries(periph, full_streams=True)
    ]
    if camera_entries:
        actions.append(
            bridge_config_node(camera_entries, Path(tempfile.gettempdir()) / "ibrobot_runtime_camera_bridge.yaml")
        )
    return actions


def render_mujoco_description(
    profile: dict[str, Any],
    profile_path: Path,
    peripherals: list | None = None,
    *,
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    scene: str = "",
) -> str:
    """Render the MuJoCo-platform robot description (URDF) for RSP + the plugin (D2/D3/D5).

    Steps: the simulator version check (``check_simulator_versions`` — a
    CameraPlugin-capable ``mujoco_ros2_control`` and a Python ``mujoco``
    matching the simulator's libmujoco are hard preconditions, D2); the
    runtime-private robot MJCF (template + meshes from ``simulation.mujoco``,
    cameras injected, base at the spawn pose including yaw); the composed
    robot+scene model (``compose_mujoco_model``, the domain-keyed ``.mjb`` —
    a missing ``mujoco`` Python package is fatal on the provider path); the
    xacro URDF with ``sim_plugin=mujoco_ros2_control/MujocoSystemInterface``,
    ``simulated=false``, ``mujoco_model=<composed path>``,
    ``auto_register_cameras=false`` (the deprecated sensor auto-registration
    stays off; cameras run through the CameraPlugin parameters, D3),
    ``sim_effort_state=true`` and ``mujoco_headless`` from
    ``IBROBOT_MUJOCO_HEADLESS``; the camera TF (mount + optical frames). The
    URDF carries no camera ``<sensor>`` blocks.
    """
    from robot_runtime.mujoco import (
        camera_tf_urdf,
        check_simulator_versions,
        render_robot_mjcf,
    )
    from sim_models.scene_compiler import compose_mujoco_model

    description = profile.get("description") or {}
    if not description.get("xacro"):
        raise RuntimeProfileError(f"runtime profile missing required key 'description.xacro': {profile_path}")
    check_simulator_versions()
    effective_peripherals = profile.get("peripherals") if peripherals is None else peripherals
    robot_mjcf = render_robot_mjcf(profile, spawn_pose, effective_peripherals)
    sdf_meshes = ((profile.get("simulation") or {}).get("mujoco") or {}).get("sdf_meshes") or []
    # D2/D3 revision (5b.2): an empty scene composes with the sim_models
    # ``empty`` scene (ground plane + light) — a robot-only model would render
    # on a black void. ``scene=None`` (an explicit robot-only request from a
    # direct API caller) keeps the robot-only composition.
    scene_name = None if scene is None else ((scene or "").strip() or "empty")
    composed = compose_mujoco_model(str(robot_mjcf), scene_name, sdf_meshes)
    if composed.suffix != ".mjb":
        raise RuntimeProfileError(
            f"the MuJoCo embodiment needs the mujoco Python package (composition fell back to {composed})"
        )

    xacro_args = {str(k): xacro_value(v) for k, v in (description.get("xacro_args") or {}).items()}
    xacro_args.update(
        {
            "use_sim": "true",
            "simulated": "false",  # the SDK virtual transport flag is off under MuJoCo
            "sim_plugin": "mujoco_ros2_control/MujocoSystemInterface",
            "sim_effort_state": "true",
            "mujoco_model": str(composed),
        }
    )
    if os.environ.get("IBROBOT_MUJOCO_HEADLESS", "0") == "1":
        xacro_args["mujoco_headless"] = "true"
    for key, value in (profile.get("hardware") or {}).get("xacro_args", {}).items():
        xacro_args[str(key)] = xacro_value(value)
    robot_description = render_xacro(resolve_path(description["xacro"]), xacro_args)
    cameras_xml = camera_tf_urdf(effective_peripherals, spawn_pose=spawn_pose)
    if cameras_xml:
        robot_description = robot_description.replace("</robot>", cameras_xml + "\n</robot>", 1)
    return _resolve_package_uris(robot_description)


# The node name mujoco_ros2_control's hardware interface hard-codes for the
# node that carries the simulator (and CameraPlugin) parameters. Global
# --params-file arguments reach it because it is created with default node
# options and automatically_declare_parameters_from_overrides.
MUJOCO_SIMULATION_NODE = "mujoco_ros2_control_node"


def mujoco_embodiment_actions(
    profile: dict[str, Any],
    profile_path: Path,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
    mujoco_scene: str = "",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list:
    """The provider's MuJoCo embodiment (D2): one ``mujoco_ros2_control`` process.

    That process is both the simulator and the controller manager and takes
    the place of the provider's own ``ros2_control_node``: robot_state_publisher,
    spawners, the facade, the camera depth fan-out relays and the simulated
    joint-current publisher all run on sim time against it. No world process
    is started here — with no scene the robot model runs alone (the old
    adapter's no-scene behaviour), with a scene the composed model already
    contains it (D3). The spawners start immediately; their
    ``--controller-manager-timeout`` waits for the in-process controller
    manager. ``IBROBOT_MUJOCO_HEADLESS=1`` mirrors the Gazebo headless mode:
    with no display the CameraPlugin renders through EGL (D4), so no
    ``xvfb-run`` wrapper is needed. The cameras (and, for scenes with
    dynamic objects, the FreeJointStatePublisherPlugin the scene task
    observes, issue #195) are configured through a per-launch params file
    keyed under ``MUJOCO_SIMULATION_NODE`` carrying the merged
    ``mujoco_plugins`` tree; MuJoCo plugins load from the executable
    directory and ament resources only, so no ``MUJOCO_PLUGIN_PATH`` is set.
    """
    from robot_runtime.mujoco import (
        camera_plugin_parameters,
        free_joint_state_plugin_parameters,
        merge_mujoco_plugin_parameters,
        mujoco_camera_relays,
    )

    if robot_description is None:
        robot_description = render_mujoco_description(profile, profile_path, spawn_pose=spawn_pose, scene=mujoco_scene)
    ros2_control = profile.get("ros2_control") or {}
    if not ros2_control.get("controllers_config"):
        raise RuntimeProfileError(
            f"runtime profile missing required key 'ros2_control.controllers_config': {profile_path}"
        )
    controllers_config = resolve_path(ros2_control["controllers_config"])
    broadcasters, active, inactive = profile_controllers(profile)
    peripherals = merge_peripherals(
        profile.get("peripherals"),
        (load_peripherals_file(peripherals_file) if peripherals_file else {}).get("peripherals"),
    )
    cm_params = write_controller_manager_params(robot_description)
    mujoco_node_parameters = [cm_params, controllers_config, {"use_sim_time": True}]
    # One params file carries every simulator plugin: the cameras and, when
    # the composed scene has dynamic objects, the free-joint state publisher
    # the scene task observes (issue #195).
    plugin_parameters = merge_mujoco_plugin_parameters(
        camera_plugin_parameters(profile, peripherals),
        free_joint_state_plugin_parameters(mujoco_scene),
    )
    if plugin_parameters:
        mujoco_node_parameters.append(write_node_params(MUJOCO_SIMULATION_NODE, plugin_parameters))
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}, {"use_sim_time": True}],
        ),
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            output="screen",
            parameters=mujoco_node_parameters,
        ),
        *spawner_nodes(
            broadcasters,
            active,
            inactive,
            controller_manager=str(profile["controller_manager"]),
            timeout_s=60.0,
            use_sim_time=True,
        ),
        Node(
            package="robot_runtime",
            executable="runtime_facade",
            name="runtime_facade",
            output="screen",
            parameters=[
                {
                    "profile": str(profile_path),
                    "peripherals": peripherals_file,
                    "use_sim": True,
                    "sim_platform": "mujoco",
                    "initial_mode": str((profile.get("modes") or {}).get("initial", IDLE_MODE)),
                    "use_sim_time": True,
                    **(
                        {"interface_description_json": json.dumps(interface_description)}
                        if interface_description is not None
                        else {}
                    ),
                    **({"instance_id": instance_id} if instance_id else {}),
                }
            ],
        ),
        *simulated_joint_current_actions(profile, sim_platform="mujoco"),
        *mujoco_camera_relays(peripherals),
    ]


def runtime_stack_actions(
    profile: dict[str, Any],
    profile_path: Path,
    simulated: bool,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
    sim_platform: str = "",
    gazebo_world: str = "",
    mujoco_scene: str = "",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list:
    """robot_state_publisher + ros2_control_node + controller spawners + runtime facade.

    ``peripherals_file`` (deployment overrides; profile ``peripherals:`` are
    the robot's defaults) additionally composes the sensor driver nodes, or
    synthetic perception publishers in simulated execution (D12), and feeds
    the facade so RuntimeStatus declares the perception surface.
    ``sim_platform`` (runtime-simulation-platform D1/D7) is forwarded to the
    facade so its public description carries the selected platform; the
    ``gazebo`` platform dispatches to the embodiment variant (D5): no
    ``ros2_control_node``, entity creation in the named world (empty ⇒
    standalone minimal world), spawners against the in-simulator controller
    manager and the facade on sim time; the ``mujoco`` platform dispatches to
    its embodiment variant (D2): one ``mujoco_ros2_control`` process that is
    both the simulator and the controller manager, with the composed
    robot+scene model; hardware and sdk are unchanged.
    """
    if simulated and sim_platform == "gazebo":
        return gazebo_embodiment_actions(
            profile,
            profile_path,
            peripherals_file,
            instance_id,
            robot_description=robot_description,
            interface_description=interface_description,
            gazebo_world=gazebo_world,
            spawn_pose=spawn_pose,
        )
    if simulated and sim_platform == "mujoco":
        return mujoco_embodiment_actions(
            profile,
            profile_path,
            peripherals_file,
            instance_id,
            robot_description=robot_description,
            interface_description=interface_description,
            mujoco_scene=mujoco_scene,
            spawn_pose=spawn_pose,
        )
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
                    "use_sim": simulated,
                    "initial_mode": str((profile.get("modes") or {}).get("initial", IDLE_MODE)),
                    **({"sim_platform": sim_platform} if sim_platform else {}),
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
    profile: dict, peripherals_file: str, simulated: bool, *, description: dict | None = None, sim_platform: str = ""
) -> list:
    """Sensor peripheral composition for a runtime launch (D12).

    Physical drivers + static TF on real hardware; synthetic perception
    publishers on the same topics on the SDK simulation platform. Under a
    physics platform the camera streams come from the simulator's sensors and
    bridges (embodiment, D6), so only the host audio devices remain here.
    Audio devices are host devices, so their ALSA owners run in EVERY
    execution (the legacy robot_config audio_io entry also launched them
    under simulation).
    """
    fragment = load_peripherals_file(peripherals_file) if peripherals_file else {}
    peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    if simulated and sim_platform in PHYSICS_SIM_PLATFORMS:
        return audio_nodes(peripherals)
    if simulated:
        effective = {**profile, "fast_lio": {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}}
        return [
            *synthetic_perception_nodes(
                peripherals, description or build_description(effective, peripherals, use_sim=True, sim_platform="sdk")
            ),
            *audio_nodes(peripherals),
        ]
    return peripheral_nodes(peripherals, use_sim=False)


def motion_launch_arguments(
    profile: dict[str, Any], profile_path: Path, simulated: bool, sim_platform: str, display: bool
) -> dict[str, str]:
    """Launch arguments for a MoveIt-backed motion package (so101_motion/motion.launch.py).

    Every motion node (move_group, motion server, IK workers and the optional
    RViz) runs on the platform's clock: simulation time on a physics platform,
    wall time otherwise (D7).
    """
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
        "use_sim_time": "true" if simulated and sim_platform in PHYSICS_SIM_PLATFORMS else "false",
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

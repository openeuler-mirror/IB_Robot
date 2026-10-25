"""Gazebo simulation backend adapter.

Migrated from robot_config.launch_builders.simulation (generate_gazebo_nodes).
Logic is unchanged from T1; this file is a pure structural reorganization.

Camera bridge topic naming contains a known path mismatch — the bridge
publishes on the raw Gazebo sensor topic, not on /camera/{name}/image_raw.
This will be fixed in T3.
"""

import copy
import os
import re
from pathlib import Path

from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool, resolve_ros_path

from .base_adapter import SimBackendAdapter

logger = get_colored_logger("robot_config.sim_backend.gazebo")

_FIND_PATTERN = re.compile(r"\$\(find\s+(\w+)\)")


def _description_packages(robot_config: dict) -> list[str]:
    """Packages whose share dir Gazebo needs for ``package://`` mesh URIs.

    Derived from ``ros2_control.urdf_path`` plus any ``$(find pkg)`` the
    top-level xacro includes (an assembled robot may pull its arm model from
    another description package). Order is preserved, duplicates dropped.
    """
    urdf_path = str((robot_config.get("ros2_control") or {}).get("urdf_path", "") or "")
    packages = _FIND_PATTERN.findall(urdf_path)
    resolved = resolve_ros_path(urdf_path)
    try:
        if resolved and Path(resolved).is_file():
            packages.extend(_FIND_PATTERN.findall(Path(resolved).read_text(encoding="utf-8")))
    except OSError as e:
        logger.info(f"WARNING: Could not scan {resolved} for description includes: {e}")
    return list(dict.fromkeys(packages))


def _start_actions_on_success(start_actions, success_message: str, failure_reason: str):
    """Run launch actions only when the target process exits successfully."""

    def _handler(event, _context):
        if event.returncode == 0:
            logger.info(success_message)
            return start_actions

        logger.error(f"{failure_reason} (returncode={event.returncode})")
        return [EmitEvent(event=Shutdown(reason=failure_reason))]

    return _handler


def _gazebo_gui_env_actions(robot_config: dict) -> list:
    """Optional GUI / EGL workaround env actions (shared by both Gazebo modes)."""
    actions = []
    gz_gui = robot_config.get("gazebo_gui") or {}
    qt_platform = gz_gui.get("qt_platform") or robot_config.get("gazebo_qt_platform")
    if qt_platform:
        actions.append(SetEnvironmentVariable("QT_QPA_PLATFORM", str(qt_platform)))
        logger.info(f"Gazebo GUI: QT_QPA_PLATFORM={qt_platform}")
    libgl_sw = gz_gui.get("libgl_always_software", robot_config.get("gazebo_libgl_always_software"))
    if parse_bool(libgl_sw, default=False):
        actions.append(SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"))
        logger.info(
            "Gazebo GUI: LIBGL_ALWAYS_SOFTWARE=1 "
            "(software GL; set gazebo_gui.libgl_always_software: false if GPU view works)"
        )
    return actions


def _gz_server_include(gz_args: str):
    """The ros_gz_sim server launch with the given gz_args."""
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("ros_gz_sim"),
                    "launch",
                    "gz_sim.launch.py",
                ]
            )
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )


def _headless_prefix() -> str:
    """IBROBOT_GAZEBO_HEADLESS=1 switches to server-only headless mode (no GUI,
    no rendering window). Used by sim_camera_adjuster when the primary
    simulator is MuJoCo and a parallel Gazebo stack is spawned purely for
    camera pose calibration. Default (env unset) preserves the normal Gazebo
    UX exactly."""
    if os.environ.get("IBROBOT_GAZEBO_HEADLESS", "0") == "1":
        logger.info("IBROBOT_GAZEBO_HEADLESS=1 -> Gazebo server-only headless mode")
        return "-s --headless-rendering "
    return ""


class GazeboAdapter(SimBackendAdapter):
    """Gazebo Ignition simulation backend.

    Handles:
    - GZ_SIM_RESOURCE_PATH / IGN_GAZEBO_RESOURCE_PATH environment variables
    - Gazebo server and GUI launch via ros_gz_sim
    - Robot entity creation (spawn from /robot_description topic)
    - Clock bridge (/clock)
    - Joint state bridge (/world/<world_name>/model/.../joint_state)
    - Camera image and camera_info bridges per peripheral
    """

    provides_clock = True
    needs_ros2_control = True

    def _resolve_world_and_spawn(self, robot_config: dict) -> tuple[str, dict]:
        """Resolve the world file and the robot spawn pose (shared by both modes).

        The world is a sim_models scene: ``simulation.scene`` when set,
        otherwise the ``empty`` scene (the generic fallback). A configured
        scene that fails to load falls back to ``empty`` with a warning; an
        unresolvable ``empty`` scene fails the launch.
        Only a configured scene sets a spawn pose: ``simulation.robot_spawn``
        with the scene layout as fallback (the legacy precedence); x/y/z are
        written back as ``initial_pose_{x,y,z}`` for the provider-less create
        arguments. The pose may additionally carry ``yaw`` (consumed by the
        world-only mode's provider hand-off; the provider-less create keeps
        x/y/z only).
        """
        from sim_models.scene_compiler import get_gazebo_world_path, get_scene_layout

        spawn: dict = {}
        world_path = ""
        scene_name = robot_config.get("simulation", {}).get("scene")
        if scene_name:
            try:
                world_path = str(get_gazebo_world_path(scene_name))
                logger.info(f"sim_models scene '{scene_name}' → {world_path}")
                layout = get_scene_layout(scene_name)
                spawn = copy.deepcopy(
                    robot_config.get("simulation", {}).get("robot_spawn", {}) or layout.get("robot_spawn", {})
                )
                for axis in ("x", "y", "z"):
                    key = f"initial_pose_{axis}"
                    if axis in spawn:
                        robot_config[key] = spawn[axis]
                        logger.info(f"scene robot_spawn.{axis} = {spawn[axis]}")
            except Exception as e:
                logger.warning(f"scene '{scene_name}' load failed: {e}; falling back to the empty scene world")
                spawn = {}
                world_path = ""
        if not world_path:
            # No scene configured (or the configured scene failed to load):
            # the generic fallback world is the sim_models ``empty`` scene.
            world_path = str(get_gazebo_world_path("empty"))
            logger.info(f"sim_models scene 'empty' → {world_path}")
        return world_path, spawn

    def start_world_only(self, robot_config: dict) -> tuple[list, str, tuple[float, float, float, float]]:
        """World-only mode (runtime-simulation-transport D4): generic launch
        owns the world for a runtime provider on the Gazebo transport.

        Emits the GUI env workarounds, the scene-resolved Gazebo server and
        the ``/clock`` bridge, and returns ``(actions, world_name,
        spawn_pose)`` with the spawn pose (x, y, z, yaw) resolved with the
        scene/``robot_spawn`` precedence. No ``create`` (the provider spawns
        its own embodiment), no joint-state bridge and no camera bridges:
        those are embodiment, owned by the provider. Scene assets resolve to
        absolute mesh URIs inside the generated world, so no
        description-derived resource paths are set (D7b: the provider's
        description carries absolute ``file://`` mesh URIs itself).
        """
        actions: list = []
        actions.extend(_gazebo_gui_env_actions(robot_config))
        world_path, spawn = self._resolve_world_and_spawn(robot_config)

        gazebo_world_name = robot_config.get("gazebo_world_name", "demo")
        logger.info(f"Gazebo world name (world-only hand-off): {gazebo_world_name}")

        headless_prefix = _headless_prefix()
        gz_args = f"{headless_prefix}-r {world_path}"
        actions.append(_gz_server_include(gz_args))
        actions.append(
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
                output="screen",
            )
        )
        spawn_pose = (
            float(spawn.get("x", 0.0)),
            float(spawn.get("y", 0.0)),
            float(spawn.get("z", 0.0)),
            float(spawn.get("yaw", 0.0)),
        )
        self._gazebo_world_name = gazebo_world_name
        return actions, gazebo_world_name, spawn_pose

    def start_backend(self, robot_config: dict) -> tuple:
        """Launch Gazebo and core infrastructure.

        Sets environment variables, spawns the robot entity, and starts
        clock + joint state bridges. Camera bridges are handled separately
        by spawn_peripheral_bridges().

        Returns:
            (actions, create_entity_node) for ``OnProcessExit``-triggered spawners.
        """
        actions = []

        # ---- Environment variable setup ----
        # Gazebo resolves ``package://<pkg>/meshes/...`` mesh URIs through
        # GZ_SIM_RESOURCE_PATH. The description package is whatever the robot
        # YAML's urdf_path points at (e.g. so101_description); the generic
        # launch must not hardcode a robot.
        gazebo_resource_paths = []
        gazebo_model_paths = []

        for pkg_name in _description_packages(robot_config):
            try:
                pkg_share = FindPackageShare(pkg_name).find(pkg_name)
            except Exception as e:
                logger.info(f"WARNING: Could not find description package '{pkg_name}': {e}")
                continue
            install_share = os.path.dirname(pkg_share)
            if install_share not in gazebo_resource_paths:
                gazebo_resource_paths.append(install_share)
                gazebo_model_paths.append(install_share)
            logger.info(f"Added {pkg_name} share path for Gazebo: {install_share}")
            mesh_root = Path(pkg_share) / "meshes"
            if mesh_root.exists():
                mesh_files = list(mesh_root.rglob("*.stl")) + list(mesh_root.rglob("*.dae"))
                logger.info(f"Verified {len(mesh_files)} mesh files under {mesh_root}")
            else:
                logger.info(f"WARNING: Mesh directory not found at {mesh_root}")

        if gazebo_resource_paths:
            combined_resource = ":".join(gazebo_resource_paths)
            actions.append(SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", combined_resource))
            actions.append(SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", combined_resource))

        if gazebo_model_paths:
            combined_model = ":".join(gazebo_model_paths)
            actions.append(SetEnvironmentVariable("GAZEBO_MODEL_PATH", combined_model))
            actions.append(SetEnvironmentVariable("GZ_SIM_MODEL_PATH", combined_model))

        # ---- Optional GUI / EGL workarounds ----
        actions.extend(_gazebo_gui_env_actions(robot_config))

        # ---- World file resolution (scene override + robot spawn) ----
        world_path, _spawn = self._resolve_world_and_spawn(robot_config)

        # Must match <world name="..."> inside the loaded .world/.sdf (the sim_models scene worlds use "demo").
        # ros_gz_sim create: if -world is omitted, it picks worlds_msg.data(0) from /gazebo/worlds
        # (order not guaranteed) — entity can end up in the wrong world. Always pass -world explicitly.
        gazebo_world_name = robot_config.get("gazebo_world_name", "demo")
        logger.info(f"Gazebo world name (create + bridges): {gazebo_world_name}")

        # ---- Entity creation (spawn robot from /robot_description) ----
        # gflags names use underscores (see `ros2 run ros_gz_sim create --help`):
        #   -allow_renaming   NOT -allow-renaming (latter is not a valid flag).
        create_args = [
            "-world",
            gazebo_world_name,
            "-name",
            robot_config.get("name", "so101"),
            "-allow_renaming=true",
            "-topic",
            "/robot_description",
            "-x",
            str(robot_config.get("initial_pose_x", 0.0)),
            "-y",
            str(robot_config.get("initial_pose_y", 0.0)),
            "-z",
            str(robot_config.get("initial_pose_z", 0.0)),
        ]
        create_entity = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=create_args,
            output="screen",
        )

        # ---- Clock bridge ----
        clock_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
            output="screen",
        )

        # ---- Joint state bridge ----
        joint_state_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/world/{gazebo_world_name}/model/{robot_config.get('name', 'so101')}"
                f"/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model"
            ],
            output="screen",
        )

        # ---- Gazebo server + GUI (must be listed before spawn; see below) ----
        # The world path always exists: _resolve_world_and_spawn generates it
        # or raises (a missing empty scene fails the launch, no bare server).
        gz_args = f"{_headless_prefix()}-r {world_path}"

        clock_ready_waiter = Node(
            package="robot_config",
            executable="wait_for_clock",
            name="wait_for_gazebo_clock",
            arguments=[
                "--topic",
                "/clock",
                "--timeout",
                "60.0",
            ],
            output="screen",
        )
        logger.info("Gazebo spawn: waiting for /clock before creating robot entity")
        logger.info(f"ros_gz_sim create argv: {create_args}")

        actions.extend(
            [
                _gz_server_include(gz_args),
                clock_bridge,
                joint_state_bridge,
                clock_ready_waiter,
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=clock_ready_waiter,
                        on_exit=_start_actions_on_success(
                            [create_entity],
                            success_message="Gazebo clock detected; creating robot entity.",
                            failure_reason="Gazebo clock probe failed before robot spawn.",
                        ),
                    )
                ),
            ]
        )

        # Store model name for use by spawn_peripheral_bridges()
        self._model_name = robot_config.get("name", "so101")
        self._gazebo_world_name = gazebo_world_name

        return actions, create_entity

    def load_scene(self, scene_file_path: str) -> list:
        """Stub: scene loading for Gazebo (T5 implementation)."""
        logger.info("GazeboAdapter.load_scene: not implemented yet (T5)")
        return []

    def ensure_controller_manager(self, robot_config: dict) -> list:
        """Gazebo provides controller_manager via gz_ros2_control plugin."""
        return []

    def spawn_peripheral_bridges(self, peripherals: list) -> list:
        """Create ros_gz_bridge nodes for peripheral topics.

        Delegates to sim_peripheral_bridge.generate_peripheral_sim_bridges()
        which maps each peripheral type to the correct bridge nodes and
        remaps Gazebo sensor topics to the ROS naming contract:
          /camera/{name}/image_raw
          /camera/{name}/camera_info

        Requires start_backend() to have been called first (sets _model_name).

        Args:
            peripherals: List of peripheral dicts from robot_config["peripherals"].

        Returns:
            List of ros_gz_bridge Node actions.
        """
        from robot_config.launch_builders.sim_peripheral_bridge import (
            generate_peripheral_sim_bridges,
        )

        model_name = getattr(self, "_model_name", "so101")
        world_name = getattr(self, "_gazebo_world_name", "demo")
        logger.info(f"Gazebo: spawning peripheral bridges (world: {world_name}, model: {model_name})")
        return generate_peripheral_sim_bridges(peripherals, model_name, world_name)

    def update_object_pose(self, object_name: str, pose) -> None:
        """[Reserved for v1] Runtime object pose update via Gazebo service."""
        pass

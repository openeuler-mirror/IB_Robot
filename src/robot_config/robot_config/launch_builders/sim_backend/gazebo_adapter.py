"""Gazebo simulation backend adapter.

Migrated from robot_config.launch_builders.simulation (generate_gazebo_nodes).
Logic is unchanged from T1; this file is a pure structural reorganization.

Camera bridge topic naming contains a known path mismatch — the bridge
publishes on the raw Gazebo sensor topic, not on /camera/{name}/image_raw.
This will be fixed in T3.
"""

import os
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


def _start_actions_on_success(start_actions, success_message: str, failure_reason: str):
    """Run launch actions only when the target process exits successfully."""

    def _handler(event, _context):
        if event.returncode == 0:
            logger.info(success_message)
            return start_actions

        logger.error(f"{failure_reason} (returncode={event.returncode})")
        return [EmitEvent(event=Shutdown(reason=failure_reason))]

    return _handler


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
        gazebo_resource_paths = []
        gazebo_model_paths = []

        try:
            install_share = os.path.dirname(FindPackageShare("robot_description").find("robot_description"))
            gazebo_resource_paths.append(install_share)
            gazebo_model_paths.append(install_share)
            logger.info(f"Added install share path for Gazebo: {install_share}")

            # Add lekiwi_description for chassis mesh resolution
            try:
                lekiwi_share = os.path.dirname(FindPackageShare("lekiwi_description").find("lekiwi_description"))
                gazebo_resource_paths.append(lekiwi_share)
                gazebo_model_paths.append(lekiwi_share)
                logger.info(f"Added lekiwi_description share path for Gazebo: {lekiwi_share}")
            except Exception:
                pass  # lekiwi_description not installed, skip silently

            robot_desc_share = FindPackageShare("robot_description").find("robot_description")
            mesh_path = os.path.join(robot_desc_share, "meshes", "lerobot", "so101")
            if Path(mesh_path).exists():
                mesh_files = list(Path(mesh_path).glob("*.stl"))
                logger.info(f"Verified {len(mesh_files)} mesh files at {mesh_path}")
            else:
                logger.info(f"WARNING: Mesh directory not found at {mesh_path}")
        except Exception as e:
            logger.info(f"WARNING: Could not find robot_description package: {e}")

        if gazebo_resource_paths:
            combined_resource = ":".join(gazebo_resource_paths)
            actions.append(SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", combined_resource))
            actions.append(SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", combined_resource))

        if gazebo_model_paths:
            combined_model = ":".join(gazebo_model_paths)
            actions.append(SetEnvironmentVariable("GAZEBO_MODEL_PATH", combined_model))
            actions.append(SetEnvironmentVariable("GZ_SIM_MODEL_PATH", combined_model))

        # ---- Optional GUI / EGL workarounds ----
        # Symptom: gray/empty 3D view while physics and ROS topics work; log may show
        #   libEGL warning: egl: failed to create dri2 screen
        # Enable via robot YAML ``gazebo_gui`` (see so101_single_arm.yaml).
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

        # ---- World file resolution ----
        world_file = robot_config.get("world_file", "")
        if not world_file:
            world_file = "$(find robot_config)/config/worlds/simulation.world"
        elif not world_file.startswith("$("):
            world_file = f"$(find robot_config)/config/worlds/{world_file}"

        world_path = resolve_ros_path(world_file)

        # Scene override: if simulation.scene is set, replace the default world
        # with the sim_models template (resolves mesh paths to absolute URIs).
        # Also reads robot_spawn from layout.yaml to override initial_pose_z,
        # so the scene (not the robot YAML) owns the table-relative spawn position.
        _scene_name = robot_config.get("simulation", {}).get("scene")
        if _scene_name:
            try:
                from sim_models.scene_compiler import get_gazebo_world_path, get_scene_layout

                world_path = str(get_gazebo_world_path(_scene_name))
                logger.info(f"sim_models scene '{_scene_name}' → {world_path}")
                layout = get_scene_layout(_scene_name)
                spawn = layout.get("robot_spawn", {})
                for axis in ("x", "y", "z"):
                    key = f"initial_pose_{axis}"
                    if axis in spawn:
                        robot_config[key] = spawn[axis]
                        logger.info(f"scene robot_spawn.{axis} = {spawn[axis]}")
            except ImportError:
                logger.warning("sim_models not installed; ignoring simulation.scene")
            except Exception as e:
                logger.warning(f"scene '{_scene_name}' load failed: {e}; falling back to default world")

        if not Path(world_path).exists():
            logger.info(f"WARNING: World file not found at {world_path}")

        # Must match <world name="..."> inside the loaded .world/.sdf (simulation.world uses "demo").
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
        gz_args = ""
        if Path(world_path).exists():
            gz_args = f"-r {world_path}"
        else:
            logger.info("Using empty Gazebo world (world file not found)")

        gazebo_launch = IncludeLaunchDescription(
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
                gazebo_launch,
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

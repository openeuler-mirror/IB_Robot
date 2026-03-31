"""MuJoCo simulation backend adapter.

Orchestrates the ros2_control_node which loads MujocoSystemInterface as a
hardware plugin inside the controller_manager.

Architecture:
    - Single process: ros2_control_node (controller_manager + MujocoSystemInterface plugin)
    - No separate bridge nodes: camera topics remapped directly via Node remappings
    - URDF hardware plugin: mujoco_ros2_control/MujocoSystemInterface (set by description.py)
    - gz_create_entity=None → robot.launch.py starts deferred_sim_spawners directly

Camera convention:
    When use_default_transform=True, camera poses are loaded from
    camera_presets.py which stores MuJoCo-native values directly
    (camera forward = -Z).  No cross-platform conversion is needed.
"""

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from launch_ros.actions import Node
from robot_config.logger_utils import get_colored_logger

from robot_config.utils import resolve_ros_path
from .base_adapter import SimBackendAdapter

logger = get_colored_logger("robot_config.sim_backend.mujoco")


class MujocoAdapter(SimBackendAdapter):
    """MuJoCo simulation backend using mujoco_ros2_control."""

    def start_backend(self, robot_config: dict) -> tuple:
        """Launch the mujoco_ros2_control node (simulator + controller_manager).

        Returns:
            (actions, None) — None because there is no gz_create_entity;
            robot.launch.py will call actions.extend(deferred_sim_spawners) directly.
        """
        from robot_config.launch_builders.description import generate_robot_description
        from sim_models.scene_compiler import get_mujoco_scene_path, get_scene_layout

        # 1. Robot MuJoCo XML: template + base_pos substitution + YAML-driven cameras
        #    Must run before generate_robot_description so mujoco_model_path can be
        #    injected into the URDF <hardware> block (new upstream API).
        scene_name = robot_config.get("simulation", {}).get("scene")
        robot_spawn = {}
        if scene_name:
            try:
                layout = get_scene_layout(scene_name)
                robot_spawn = layout.get("robot_spawn", {})
            except Exception as e:
                logger.warning(f"could not load layout for '{scene_name}': {e}")

        peripherals = robot_config.get("peripherals", [])
        robot_xml_path = self._generate_robot_mujoco_xml(robot_spawn, peripherals)

        # 2. Scene XML (no scene → use robot XML directly)
        if scene_name:
            try:
                mujoco_model_path = str(get_mujoco_scene_path(scene_name, robot_xml_path))
                logger.info(f"MuJoCo scene: {mujoco_model_path}")
            except Exception as e:
                logger.warning(
                    f"scene '{scene_name}' failed: {e}; using robot-only XML"
                )
                mujoco_model_path = robot_xml_path
        else:
            mujoco_model_path = robot_xml_path

        # 3. Generate URDF with MujocoSystemInterface plugin, model path injected via xacro
        result = generate_robot_description(robot_config, True,
                                            mujoco_model_path=mujoco_model_path)
        if result is None:
            raise RuntimeError("[mujoco_adapter] generate_robot_description failed")
        _, robot_desc_params = result

        # 4. MUJOCO_PLUGIN_PATH for mesh decoders (libstl_decoder.so etc.)
        mujoco_plugin_path = self._find_mujoco_plugin_path()

        # 5. Controllers config (so101_hardware/config/so101_controllers.yaml)
        ros2_ctrl = robot_config.get("ros2_control", {})
        controllers_cfg = resolve_ros_path(ros2_ctrl.get("controllers_config", ""))

        # 6. Camera topic remappings (YAML-driven, MuJoCo default→contract)
        remappings = self._build_camera_remappings(peripherals)

        # 7. Build node parameters (mujoco_model_path now in URDF, not a node param)
        params = [
            robot_desc_params,
            {'use_sim_time': True},
        ]
        if controllers_cfg and Path(controllers_cfg).exists():
            params.append(controllers_cfg)
        else:
            logger.warning(f"controllers_config not found at '{controllers_cfg}'")

        additional_env = {}
        if mujoco_plugin_path:
            additional_env['MUJOCO_PLUGIN_PATH'] = mujoco_plugin_path
            logger.info(f"MUJOCO_PLUGIN_PATH={mujoco_plugin_path}")

        mujoco_node = Node(
            package='mujoco_ros2_control',
            executable='ros2_control_node',
            output='screen',
            parameters=params,
            remappings=remappings,
            additional_env=additional_env,
        )
        logger.info(f"Starting ros2_control_node, model={mujoco_model_path}")
        return [mujoco_node], None  # None → spawners start immediately (30s timeout)

    def load_scene(self, scene_file_path: str) -> list:
        """Scene loading is handled inside start_backend(); no extra actions needed."""
        return []

    def ensure_controller_manager(self, robot_config: dict) -> list:
        """mujoco_ros2_control embeds the controller_manager; nothing extra needed."""
        return []

    def spawn_peripheral_bridges(self, peripherals: list) -> list:
        """No bridge nodes needed; topic renaming is done via Node remappings in start_backend()."""
        return []

    def update_object_pose(self, object_name: str, pose) -> None:
        """Reserved for T7 (episode parametrization)."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_mujoco_plugin_path(self) -> str:
        """Locate the MuJoCo plugin directory (libstl_decoder.so, libobj_decoder.so).

        Returns the directory path string, or "" if not found.
        """
        try:
            import mujoco
            plugin_dir = os.path.join(os.path.dirname(mujoco.__file__), "plugin")
            if os.path.isdir(plugin_dir):
                return plugin_dir
            logger.warning(f"mujoco plugin dir not found at {plugin_dir}")
        except ImportError:
            logger.warning(
                "mujoco Python package not found; MUJOCO_PLUGIN_PATH will not be set"
            )
        return ""

    def _generate_robot_mujoco_xml(self, robot_spawn: dict, peripherals: list) -> str:
        """Generate /tmp/so101_mujoco.xml from the template with YAML-driven cameras.

        Steps:
        1. String-replace {{MESHES_DIR}} and {{ROBOT_BASE_POS}} in the template.
        2. Parse XML and inject <camera> elements for each opencv-driver peripheral.
        3. Write to /tmp/so101_mujoco.xml and return the path.

        Camera convention:
            YAML stores Gazebo/URDF pose (camera looks in +Z).
            MuJoCo cameras look in -Z → mj_roll = yaml_roll + pi.
            pitch and yaw are used unchanged.
        """
        from ament_index_python.packages import get_package_share_directory

        pkg = get_package_share_directory('robot_description')
        meshes_dir = os.path.join(pkg, 'meshes', 'lerobot', 'so101')
        template_path = os.path.join(pkg, 'mujoco', 'so101.xml.template')

        if not os.path.exists(template_path):
            raise FileNotFoundError(
                f"[mujoco_adapter] MuJoCo template not found: {template_path}"
            )

        base_x = robot_spawn.get('x', 0.0)
        base_y = robot_spawn.get('y', 0.0)
        base_z = robot_spawn.get('z', 0.0)

        # Step 1: string substitution (placeholders are inside attribute values)
        with open(template_path) as f:
            content = f.read()
        content = content.replace('{{MESHES_DIR}}', meshes_dir)
        content = content.replace('{{ROBOT_BASE_POS}}', f'{base_x} {base_y} {base_z}')

        # Step 2: parse XML and inject YAML-driven cameras
        root = ET.fromstring(content)
        worldbody = root.find('worldbody')
        if worldbody is None:
            raise RuntimeError("[mujoco_adapter] <worldbody> not found in so101.xml.template")

        from .camera_presets import get_preset

        for periph in peripherals:
            if periph.get('type') != 'camera':
                continue
            if periph.get('driver') != 'opencv':
                continue  # realsense / other real-hardware cameras: skip

            name = periph['name']          # YAML name, e.g. "top"
            cam_name = f'{name}_camera'    # MJCF name convention, e.g. "top_camera"
                                           # must match _build_camera_remappings

            # --- preset lookup ---
            t = periph.get('transform', {})
            cam_fovy = periph.get('fovy', 60)
            if periph.get('use_default_transform', False):
                preset = get_preset("mujoco", name)
                if preset:
                    t = preset
                    cam_fovy = preset.get('fovy', cam_fovy)

            parent_frame = t.get('parent_frame', 'world')
            pos = f"{t.get('x', 0.0)} {t.get('y', 0.0)} {t.get('z', 0.0)}"

            # Camera orientation: values are already in MuJoCo convention
            # (camera forward = -Z) when loaded from presets.
            # No cross-platform conversion needed.
            mj_roll = t.get('roll', 0.0)
            mj_pitch = t.get('pitch', 0.0)
            mj_yaw = t.get('yaw', 0.0)
            euler = f'{mj_roll:.6f} {mj_pitch:.6f} {mj_yaw:.6f}'

            fovy = str(cam_fovy)
            resolution = f"{periph.get('width', 640)} {periph.get('height', 480)}"

            cam_elem = ET.Element('camera')
            cam_elem.set('name', cam_name)
            cam_elem.set('pos', pos)
            cam_elem.set('euler', euler)
            cam_elem.set('fovy', fovy)
            cam_elem.set('resolution', resolution)

            if parent_frame in ('world', 'worldbody'):
                worldbody.append(cam_elem)
                logger.info(f"Injected camera '{cam_name}' into worldbody")
            else:
                body = worldbody.find(f'.//body[@name="{parent_frame}"]')
                if body is not None:
                    body.append(cam_elem)
                    logger.info(
                        f"Injected camera '{cam_name}' into body '{parent_frame}'"
                    )
                else:
                    logger.warning(
                        f"body '{parent_frame}' not found "
                        f"for camera '{cam_name}'; skipping"
                    )

        # Step 3: write to /tmp/
        out_path = '/tmp/so101_mujoco.xml'
        ET.indent(root, space='  ')
        ET.ElementTree(root).write(out_path, encoding='unicode', xml_declaration=True)
        logger.info(f"Robot MJCF written to {out_path}")
        return out_path

    def _build_camera_remappings(self, peripherals: list) -> list:
        """Build Node remappings from raw MuJoCo topics to the ROS contract.

        MuJoCo publishes:  /{name}_camera/color, /{name}_camera/camera_info
        Contract requires: /camera/{name}/image_raw, /camera/{name}/camera_info

        Only cameras with driver="opencv" are included (must match
        _generate_robot_mujoco_xml injection filter).

        YAML name="top" → camera name="top_camera" (f"{name}_camera" convention).
        """
        remappings = []
        for periph in peripherals:
            if periph.get('type') != 'camera':
                continue
            if periph.get('driver') != 'opencv':
                continue
            name = periph['name']
            mj_cam = f'{name}_camera'
            remappings.extend([
                (f'/{mj_cam}/color',       f'/camera/{name}/image_raw'),
                (f'/{mj_cam}/camera_info', f'/camera/{name}/camera_info'),
            ])
        return remappings

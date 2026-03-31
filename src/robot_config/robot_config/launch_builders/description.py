"""Robot description layer: URDF building from YAML configuration.

Handles xacro processing and dynamic camera URDF injection.
Separated from control.py so both Gazebo and MuJoCo adapters can reuse
the same URDF generation without controller logic.

Public API:
    generate_robot_description(robot_config, use_sim, mujoco_model_path=None) -> (full_urdf, params) | None
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import xacro as _xacro_lib
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path, parse_bool

logger = get_colored_logger("robot_config.description")


def _build_cameras_urdf_from_yaml(peripherals: list, platform: str = "gazebo") -> str:
    """从 YAML peripherals 动态生成相机 link/joint/gazebo XML。

    命名约定（与 sim_peripheral_bridge.py 的 bridge 路径对齐）：
    - URDF link 名     = YAML frame_id（如 camera_top_frame）
      → robot_state_publisher TF 帧名与图像消息 header.frame_id 一致
    - Gazebo sensor 名 = {name}_camera（如 top_camera）
      → 与 sim_peripheral_bridge.py sensor_name 约定一致
    - Gazebo topic     = {name}_camera/image
      → Ignition Gazebo 发布于 /{name}_camera/image，由 bridge 桥接

    固定相机（parent_frame=world/base）和随臂相机（parent_frame=gripper）
    生成逻辑完全相同，区别仅在于 parent link 的值。
    MuJoCo 使用的 <gazebo reference> 块会被忽略（不影响 MJCF 解析）。

    当 use_default_transform=True 时，从 camera_presets 读取平台专属默认位姿。

    Args:
        peripherals: YAML peripherals 列表。
        platform: 当前仿真平台 ("gazebo" 或 "mujoco")。
    """
    from robot_config.launch_builders.sim_backend.camera_presets import get_preset

    parts = []
    for periph in peripherals:
        if periph.get("type") != "camera":
            continue
        name        = periph["name"]
        frame_id    = periph.get("frame_id", f"camera_{name}_frame")

        # --- preset 查找 ---
        t           = periph.get("transform", {})
        cam_fovy    = periph.get("fovy", 60)
        if periph.get("use_default_transform", False):
            preset = get_preset(platform, name)
            if preset:
                t = preset
                cam_fovy = preset.get("fovy", cam_fovy)

        parent      = t.get("parent_frame", "world")
        x           = t.get("x",     0.0)
        y           = t.get("y",     0.0)
        z           = t.get("z",     0.0)
        roll        = t.get("roll",  0.0)
        pitch       = t.get("pitch", 0.0)
        yaw         = t.get("yaw",   0.0)
        width       = periph.get("width",  640)
        height      = periph.get("height", 480)
        fps         = periph.get("fps",     30)
        sensor_name = f"{name}_camera"

        # fovy (vertical, degrees) → horizontal_fov (radians)
        fovy_rad = math.radians(cam_fovy)
        hfov_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * width / height)

        parts.append(f"""
    <!-- {name} camera (generated from YAML peripherals) -->
    <link name="{frame_id}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.01"/>
            <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
        </inertial>
    </link>
    <joint name="{name}_camera_joint" type="fixed">
        <parent link="{parent}"/>
        <child  link="{frame_id}"/>
        <origin xyz="{x} {y} {z}" rpy="{roll} {pitch} {yaw}"/>
    </joint>
    <gazebo reference="{frame_id}">
        <sensor type="camera" name="{sensor_name}">
            <update_rate>{fps}</update_rate>
            <camera>
                <horizontal_fov>{hfov_rad:.4f}</horizontal_fov>
                <image>
                    <width>{width}</width>
                    <height>{height}</height>
                    <format>R8G8B8</format>
                </image>
                <clip><near>0.01</near><far>100</far></clip>
            </camera>
            <always_on>true</always_on>
            <visualize>true</visualize>
            <topic>{sensor_name}/image</topic>
        </sensor>
    </gazebo>""")
    return "\n".join(parts)


def _inject_mujoco_camera_sensors(full_urdf: str, peripherals: list) -> str:
    """Inject MuJoCo <sensor> blocks into the <ros2_control> element of the URDF.

    The upstream mujoco_ros2_control plugin reads frame_name/image_topic from
    the URDF <sensor> params rather than using hardcoded defaults.  This makes
    YAML ``optical_frame_id`` the single source of truth for camera frame IDs.

    Only called when platform == "mujoco".  Gazebo and real-hardware paths are
    completely untouched — this function is never invoked for them.

    Args:
        full_urdf:   Complete URDF XML string (post camera-injection).
        peripherals: List of peripheral dicts from YAML config.

    Returns:
        Modified URDF string with sensor blocks injected, or the original
        string unchanged if no opencv cameras are present or parsing fails.
    """
    opencv_cams = [p for p in peripherals
                   if p.get("type") == "camera" and p.get("driver") == "opencv"]
    if not opencv_cams:
        return full_urdf
    try:
        root = ET.fromstring(full_urdf)
        ros2_ctrl = root.find(".//ros2_control")
        if ros2_ctrl is None:
            return full_urdf
        for periph in opencv_cams:
            name = periph["name"]
            cam_name = f"{name}_camera"   # must match mujoco_adapter.py convention
            optical_frame = periph.get("optical_frame_id",
                                       f"camera_{name}_optical_frame")
            sensor = ET.SubElement(ros2_ctrl, "sensor")
            sensor.set("name", cam_name)
            for param_key, param_val in [
                ("frame_name",  optical_frame),
                ("image_topic", f"{cam_name}/color"),
                ("info_topic",  f"{cam_name}/camera_info"),
                ("depth_topic", f"{cam_name}/depth"),
            ]:
                p = ET.SubElement(sensor, "param", name=param_key)
                p.text = param_val
        logger.info(
            f"Injected {len(opencv_cams)} MuJoCo camera sensor(s) into ros2_control block"
        )
        return ET.tostring(root, encoding="unicode")
    except ET.ParseError as e:
        logger.warning(f"could not inject MuJoCo camera sensors: {e}")
        return full_urdf


def generate_robot_description(robot_config: dict, use_sim, mujoco_model_path: str = None):
    """Build the full robot URDF from YAML config (xacro + camera injection).

    Processes the base xacro URDF and dynamically injects camera links/joints
    from YAML ``peripherals`` entries. This is the single source of truth for
    robot_description used by both robot_state_publisher and simulation adapters.

    Args:
        robot_config: Robot configuration dict (from YAML).
        use_sim:      Simulation mode flag (string or bool).

    Returns:
        ``(full_urdf_str, robot_description_params)`` on success, where
        ``robot_description_params`` is a dict ready to pass to
        ``robot_state_publisher`` parameters (includes ``use_sim_time`` when
        use_sim is True).

        ``None`` on any failure (missing path, xacro error). Caller should
        treat None as a non-recoverable config error and abort.
    """
    is_sim = parse_bool(use_sim, default=False)
    ros2_control_config = robot_config.get("ros2_control", {})

    # Resolve and validate URDF path
    urdf_path = ros2_control_config.get("urdf_path")
    if not urdf_path:
        logger.warning("No urdf_path specified")
        return None
    urdf_path = resolve_ros_path(urdf_path)
    logger.info(f"URDF path: {urdf_path}")
    if not Path(urdf_path).exists():
        logger.warning(f"URDF file not found at {urdf_path}")
        return None

    # Build xacro parameter mappings from YAML
    port = ros2_control_config.get("port", "/dev/ttyACM0")
    calib_file = resolve_ros_path(ros2_control_config.get("calib_file", ""))
    reset_positions_json = json.dumps(ros2_control_config.get("reset_positions", {}))

    xacro_mappings = {
        'use_sim': 'true' if is_sim else 'false',
        'port': port,
        'calib_file': calib_file,
        # Pass raw JSON — no extra quotes. xacro receives this as a plain
        # string arg and injects it into <param name="reset_positions"> as
        # text content (not an XML attribute), so no XML-escaping is needed.
        # Wrapping in f"'{json}'" caused the hardware plugin to receive a
        # literal single-quoted string '{"1":0.0}' and fail to parse it.
        'reset_positions': reset_positions_json,
    }
    if is_sim:
        sim_platform = robot_config.get("simulation", {}).get("platform", "gazebo")
        if sim_platform == "mujoco":
            xacro_mappings["sim_plugin"] = "mujoco_ros2_control/MujocoSystemInterface"
            if mujoco_model_path:
                xacro_mappings["mujoco_model"] = mujoco_model_path
        # Gazebo: xacro default="gz_ros2_control/GazeboSimSystem" — no explicit mapping needed
        _gz_ctrl_yaml = resolve_ros_path("$(find so101_hardware)/config/so101_controllers.yaml")
        if Path(_gz_ctrl_yaml).exists():
            xacro_mappings["gz_ros2_control_parameters_file"] = str(Path(_gz_ctrl_yaml).resolve())

    try:
        doc = _xacro_lib.process_file(urdf_path, mappings=xacro_mappings)
        base_urdf = doc.toxml()
    except Exception as e:
        logger.error(f"xacro processing failed: {e}")
        return None

    # Dynamically inject camera URDF blocks from YAML peripherals
    peripherals = robot_config.get("peripherals", [])
    sim_platform = robot_config.get("simulation", {}).get("platform", "gazebo") if is_sim else "gazebo"
    cameras_xml = _build_cameras_urdf_from_yaml(peripherals, platform=sim_platform)
    if cameras_xml:
        full_urdf = base_urdf.replace("</robot>", cameras_xml + "\n</robot>", 1)
        logger.info(
            f"Injected {sum(1 for p in peripherals if p.get('type') == 'camera')} camera(s) into URDF from YAML"
        )
    else:
        full_urdf = base_urdf

    # MuJoCo camera sensor injection is disabled: apt mujoco_ros2_control 0.0.2
    # crashes when initializing sensors for body-attached cameras (e.g. wrist_camera
    # on the gripper body).  The fix exists in the upstream repo (3 commits beyond
    # the 0.0.2 tag) but is not yet in the apt package.  Camera topic remappings
    # in mujoco_adapter.py still work; frame_name defaults to the MJCF camera name.
    # Re-enable once apt is updated past 0.0.2.
    # if is_sim and robot_config.get("simulation", {}).get("platform") == "mujoco":
    #     full_urdf = _inject_mujoco_camera_sensors(full_urdf, peripherals)

    robot_description_params = {"robot_description": full_urdf}
    if is_sim:
        robot_description_params["use_sim_time"] = True

    return full_urdf, robot_description_params

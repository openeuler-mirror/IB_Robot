import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Package paths
    pkg_robot_navigation = get_package_share_directory("robot_navigation")
    pkg_rtabmap = get_package_share_directory("rtabmap_launch")
    pkg_realsense = get_package_share_directory("realsense2_camera")

    # 1. RealSense Camera
    realsense_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_realsense, "launch", "rs_launch.py")),
        launch_arguments={
            "align_depth.enable": "true",
            "enable_sync": "true",
            #'depth_module.depth_profile': '640x480x30',
            #'rgb_camera.color_profile': '640x480x30',
        }.items(),
    )

    # 2. RTAB-Map
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_rtabmap, "launch", "rtabmap.launch.py")),
        launch_arguments={
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "database_path": "~/workspace/map/rtabmap.db",
            "use_sim_time": "false",
            "localization": "true",
            "frame_id": "camera_link",
            "rtabmap_args": "--Mem/InitWMWithAllNodes true --Mem/IncrementalMemory true --Mem/PermanentMemory false --Mem/STMSize 8",
            "approx_sync": "true",
            "queue_size": "20",
        }.items(),
    )

    # 4. EKF Node
    ekf_config = os.path.join(pkg_robot_navigation, "config", "ekf.yaml")
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config],
    )

    # 5. Static TF: base_link -> camera_link (30cm height)
    base_to_camera_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera_tf",
        arguments=["0", "0", "0.3", "0", "0", "0", "base_link", "camera_link"],
    )

    return LaunchDescription(
        [
            realsense_camera_launch,
            base_to_camera_tf,
            rtabmap_launch,
            ekf_node,
        ]
    )

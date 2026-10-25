from glob import glob

from setuptools import find_packages, setup

package_name = "robot_navigation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Launch files
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        # Config files
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.json")),
        ("share/" + package_name + "/config/nav2", glob("config/nav2/*.yaml")),
        # Shared test resources used by robot_config navigation simulation tests
        ("share/" + package_name + "/config/test", glob("config/test/*.world")),
        # Simulation map resources
        ("share/" + package_name + "/config/maps", glob("config/maps/*.yaml") + glob("config/maps/*.pgm")),
        # RViz config
        ("share/" + package_name + "/config", glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yanhan",
    maintainer_email="yanhan31@huawei.com",
    description="Robot navigation package with Nav2 client, voice control, and chassis driver",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            # Nav2 Goal Client
            "nav2_goal_client = robot_navigation.nav2_goal_client:main",
            # Launch-time navigation admission for runtime-provider stages
            "navigation_admission = robot_navigation.navigation_admission:main",
            # Voice Control (bridges voice_asr_service to nav2_goal_client)
            "voice_control = robot_navigation.voice_control:main",
            # Save RTAB-Map occupancy grid through nav2_map_server
            "save_rtabmap_map = robot_navigation.save_rtabmap_map:main",
            "amcl_global_localization = robot_navigation.amcl_global_localization:main",
            "save_lidar_map = robot_navigation.save_lidar_map:main",
            # Normalize official FAST-LIO odometry for navigation consumers.
            "fast_lio_odom_bridge = robot_navigation.fast_lio_odom_bridge:main",
            "navigation_command_server = robot_navigation.navigation_command_server:main",
            "navigation_lifecycle_coordinator = robot_navigation.navigation_lifecycle_coordinator:main",
            "nav_cmd = robot_navigation.nav_cmd:main",
            "evaluate_controller_regression = robot_navigation.controller_regression:main",
        ],
        # Cross-package launch-builder registration (openspec change
        # ``migrate-business-launch-builders``): robot_config discovers this
        # group when building the launch registry, so the navigation stack's
        # launch wiring registers from its owning package without any
        # robot_config change. Installing this package does not enable the
        # entry; participation still comes from the with_navigation switch /
        # robot.navigation.enabled.
        "ibrobot.launch_builders": [
            "navigation = robot_navigation.launch_builders.navigation:NAVIGATION_BUILDER",
        ],
    },
)

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
            # CmdVel Bridge (replaces chassis_driver for ros2_control path)
            "cmd_vel_bridge_node = robot_navigation.cmd_vel_bridge_node:main",
            # Voice Control (bridges voice_asr_service to nav2_goal_client)
            "voice_control = robot_navigation.voice_control:main",
        ],
    },
)

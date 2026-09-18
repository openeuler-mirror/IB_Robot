from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "robot_config"


def _package_data_files(base_dir: str):
    entries = []
    for path in sorted(Path(base_dir).rglob("*")):
        if not path.is_file():
            continue
        install_dir = f"share/{package_name}/{path.parent.as_posix()}"
        entries.append((install_dir, [str(path)]))
    return entries


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        (
            "share/" + package_name + "/test/e2e",
            glob("test/e2e/gt_odom_node.py") + glob("test/e2e/cmd_vel_relay.py"),
        ),
        ("share/" + package_name + "/test/config", glob("test/config/*.yaml")),
    ]
    + _package_data_files("config"),
    install_requires=["setuptools", "pyyaml", "inference_manifest"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Unified robot configuration system for ros2_control and peripherals",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "ibrobot-trace-topology = robot_config.tracing_topology:main",
            "wait_for_clock = robot_config.wait_for_clock:main",
            "wait_for_controllers = robot_config.wait_for_controllers:main",
            "controller_spawner = robot_config.controller_spawner:main",
            "topic_relay = robot_config.topic_relay:main",
            "semantic_mapping_recorder = robot_config.semantic_mapping_recorder:main",
        ],
    },
)

from glob import glob

from setuptools import find_packages, setup

package_name = "object_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Developers",
    maintainer_email="dev@openeuler.org",
    description="Single-target RGB-D tracking and offline evaluation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "target_tracker_node = object_tracker.target_tracker_node:main",
            "dynamic_target_follower_node = object_tracker.dynamic_target_follower_node:main",
            "dynamic_target_follower = object_tracker.dynamic_target_follower_node:main",
            "mock_slam_nav_interfaces = object_tracker.mock_slam_nav_interfaces:main",
            "object_tracking_evaluate = object_tracker.offline_evaluator:main",
        ],
    },
)

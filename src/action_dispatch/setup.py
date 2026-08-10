from setuptools import find_packages, setup

package_name = "action_dispatch"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="LeRobot-ROS2 Team",
    maintainer_email="dev@example.com",
    description="Pull-based action dispatch package for LeRobot-ROS2 integration",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "action_dispatcher_node = action_dispatch.action_dispatcher_node:main",
            "scheduled_action_dispatcher_node = action_dispatch.scheduled_action_dispatcher_node:main",
        ],
    },
)

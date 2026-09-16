from glob import glob

from setuptools import find_packages, setup

package_name = "robot_runtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["schemas/*.json"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "pyyaml", "jsonschema>=4,<5"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Robot runtime contract layer: capability vocabulary, runtime facade, mock runtime, conformance suite",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "runtime_facade = robot_runtime.facade_node:main",
            "mock_runtime = robot_runtime.mock_runtime_node:main",
            "wait_for_runtime = robot_runtime.wait_for_runtime:main",
            "topic_relay = robot_runtime.topic_relay:main",
            "synthetic_perception = robot_runtime.synthetic_perception:main",
            "runtime_interfaces = robot_runtime.interface_description:main",
        ],
    },
)

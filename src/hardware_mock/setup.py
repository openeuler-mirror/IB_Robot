from glob import glob

from setuptools import find_packages, setup

package_name = "hardware_mock"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "pyyaml", "numpy"],
    zip_safe=True,
    maintainer="IB-Robot Maintainers",
    maintainer_email="dev@ib-robot.local",
    description=(
        "Contract-driven mock that publishes observation topics and subscribes "
        "to action topics declared in robot_config YAML."
    ),
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "contract_mock = hardware_mock.contract_mock_node:main",
        ],
    },
)

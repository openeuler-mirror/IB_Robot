from setuptools import find_packages, setup

package_name = "aero_hand_hardware"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools", "aero-open-sdk==0.1.0.dev1"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Team",
    maintainer_email="maintainer@example.com",
    description="ROS 2 driver for the seven-joint Aero Hand",
    license="Apache-2.0",
    entry_points={"console_scripts": ["aero_hand_node = aero_hand_hardware.aero_hand_node:main"]},
)

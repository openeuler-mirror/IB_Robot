from glob import glob

from setuptools import find_packages, setup

package_name = "aimdk_robot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/profiles", glob("profiles/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="AgiBot X2 (AimDK) robot runtime: public runtime contract bridged onto the vendor MC tier",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "aimdk_runtime = aimdk_robot.runtime_node:main",
            "aimdk_vendor_mock = aimdk_robot.vendor_mock_node:main",
        ],
    },
)

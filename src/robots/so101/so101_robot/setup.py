from glob import glob

from setuptools import find_packages, setup

package_name = "so101_robot"

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
    description="SO-101 robot runtime: standalone launch of the complete SO-101 stack from a runtime profile",
    license="Apache-2.0",
)

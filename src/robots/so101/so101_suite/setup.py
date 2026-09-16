from setuptools import setup

package_name = "so101_suite"

setup(
    name=package_name,
    version="0.1.0",
    packages=["so101_suite"],
    package_dir={"so101_suite": "src/so101_suite"},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="SO-101 robot suite: robot-specific grasp geometry and wrist guards",
    license="Apache-2.0",
)

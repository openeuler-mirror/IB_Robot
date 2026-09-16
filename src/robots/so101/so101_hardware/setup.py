from setuptools import setup

package_name = "so101_hardware"

setup(
    name=package_name,
    version="2.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "rclpy", "sensor_msgs", "ibrobot_msgs", "pyserial", "feetech-servo-sdk"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="SO-101 robot hardware with C++ ros2_control and Python tools",
    license="Apache-2.0",
)

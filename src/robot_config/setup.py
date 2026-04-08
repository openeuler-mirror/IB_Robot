from setuptools import setup
from setuptools import find_packages
from glob import glob

package_name = 'robot_config'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/robots', glob('config/robots/*.yaml')),
        ('share/' + package_name + '/config/worlds', glob('config/worlds/*.world')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='xqw',
    maintainer_email='wuxiaoqiang.rtos@huawei.com',
    description='Unified robot configuration system for ros2_control and peripherals',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'wait_for_clock = robot_config.wait_for_clock:main',
            'wait_for_controllers = robot_config.wait_for_controllers:main',
        ],
    },
)

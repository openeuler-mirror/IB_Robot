#!/usr/bin/env python3

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "robot_teleop"
vendor_root = os.path.join("..", "..", "third_party", "vendor", "mhandpro", "3.0.20")
vendor_metadata = [path for path in glob(os.path.join(vendor_root, "*")) if os.path.isfile(path)]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml") + glob("config/*.json")),
        (os.path.join("share", package_name, "web"), glob("web/*")),
        (
            os.path.join("share", package_name, "vendor", "mhandpro", "3.0.20"),
            vendor_metadata,
        ),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Team",
    maintainer_email="maintainer@example.com",
    description="Minimal serial-to-controller bridge for zero-latency teleoperation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "teleop_node = robot_teleop.teleop_node:main",
            "calibrate_glove = robot_teleop.calibrate_glove:main",
            "analyze_glove_capture = robot_teleop.analyze_glove_capture:main",
            "mhandpro_source_node = robot_teleop.mhandpro_source_node:main",
            "vr_teleop = robot_teleop.vr_teleop:main",
        ],
    },
)

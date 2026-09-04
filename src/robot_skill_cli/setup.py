from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "robot_skill_cli"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/skills/ibrobot-control", ["resource/ibrobot-control/SKILL.md"]),
        (
            "share/" + package_name + "/hermes",
            [
                "resource/hermes/README.md",
                "resource/hermes/POLICY.md",
                "resource/hermes/SOUL.md",
                "resource/hermes/sync_hermes.sh",
            ],
        ),
        (
            "share/" + package_name + "/hermes/hooks",
            [path for path in glob("resource/hermes/hooks/*") if Path(path).is_file()],
        ),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Controlled CLI surface for LLM/Agent ROS access in IB-Robot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "robot-skill = robot_skill_cli.cli:main",
            "robot-skill-closed-loop = robot_skill_cli.closed_loop_runner:main",
            "hermes-robot = robot_skill_cli.hermes_launcher:main",
            "hermes-robot-configure = robot_skill_cli.hermes_configure:main",
            "hermes-robot-speak = robot_skill_cli.hermes_tts_hook:main",
            "hermes-robot-lifecycle-speech = robot_skill_cli.hermes_lifecycle_speech:main",
            "ibrobot-perceive = robot_skill_cli.perceive_cli:main",
        ]
    },
)

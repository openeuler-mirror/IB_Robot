from setuptools import find_packages, setup

package_name = "manipulation_execution"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Contributors",
    maintainer_email="roboguru.92@gmail.com",
    description="Closed-loop manipulation execution for agent-facing robot skills",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pick_action_client = manipulation_execution.pick_action_client:main",
            "pick_executor_node = manipulation_execution.pick_executor_node:main",
            "place_executor_node = manipulation_execution.placement_executor_node:main",
            "imitate_human_motion_executor_node = manipulation_execution.imitate_human_motion_executor_node:main",
            "placement_replay = manipulation_execution.placement_replay:main",
        ],
    },
)

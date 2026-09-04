from setuptools import find_packages, setup

package_name = "embodied_agent"

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
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Embodied minimum-closure task entry, planning, and execution nodes",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "task_entry_node = embodied_agent.task_entry_node:main",
            "task_executor_node = embodied_agent.task_executor_node:main",
            "agent_plan_node = embodied_agent.agent_plan_node:main",
            "sound_orientation_node = embodied_agent.sound_orientation_node:main",
            "visual_game_gateway_node = embodied_agent.visual_game_gateway_node:main",
            "visual_game_announcer_node = embodied_agent.visual_game_announcer_node:main",
        ],
    },
)

from setuptools import find_packages, setup

package_name = "ibrobot_agent"

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
    description="Lightweight natural-language Agent for controlled IB-Robot workflows",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ibrobot_agent_node = ibrobot_agent.node:main",
        ],
    },
)

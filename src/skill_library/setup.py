from setuptools import find_packages, setup

package_name = "skill_library"

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
    description="Embodied minimum-closure skill and primitive execution servers",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "skill_executor_node = skill_library.skill_executor_node:main",
        ],
    },
)

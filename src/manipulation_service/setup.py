from setuptools import find_packages, setup

package_name = "manipulation_service"

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
    description="Manipulation service: GraspGen planning for IB-Robot",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "grasp_planner_node = manipulation_service.grasp_planner_node:main",
            "grasp_verifier_node = manipulation_service.grasp_verifier_node:main",
        ],
    },
)

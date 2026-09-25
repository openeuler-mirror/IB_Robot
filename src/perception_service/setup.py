from setuptools import find_packages, setup

package_name = "perception_service"

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
    description="Continuous multimodal scene understanding service for embodied interaction",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "perception_service_node = perception_service.perception_service_node:main",
            "package_perception_bundles = perception_service.package_perception_bundles:main",
            "package_ascend_perception_bundles = perception_service.package_ascend_perception_bundles:main",
            "package_graspgen_ascend_bundle = perception_service.package_graspgen_ascend_bundle:main",
            "package_graspgen_torch_bundle = perception_service.package_graspgen_torch_bundle:main",
        ],
    },
)

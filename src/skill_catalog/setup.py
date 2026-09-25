import os

from setuptools import find_packages, setup

package_name = "skill_catalog"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ]
    + [
        (
            os.path.join("share", package_name, root),
            [os.path.join(root, name) for name in files],
        )
        for root, _dirs, files in os.walk("config")
    ],
    install_requires=["setuptools", "pyyaml", "jsonschema"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Lightweight skill package compiler and immutable catalog registry for IB-Robot",
    license="Apache-2.0",
    entry_points={"console_scripts": ["skill-catalog-materialize = skill_catalog.release:main"]},
)

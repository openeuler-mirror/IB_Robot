import os
from glob import glob

from setuptools import find_packages, setup

package_name = "embodied_common"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(package_name + "/*.yaml")),
    ],
    install_requires=["setuptools", "pyyaml", "inference_manifest"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Shared neutral helpers for embodied pipeline packages",
    license="Apache-2.0",
)

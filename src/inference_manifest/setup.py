from setuptools import find_packages, setup

package_name = "inference_manifest"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    package_data={package_name: ["inference_manifest.schema.json"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["pydantic>=2,<3", "jsonschema>=4,<5"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Dependency-neutral inference bundle manifest APIs",
    license="Apache-2.0",
    python_requires=">=3.10",
)

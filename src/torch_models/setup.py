from setuptools import find_packages, setup

package_name = "torch_models"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    package_data={f"{package_name}.pi05_ascend_310p": ["README.md"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE", "NOTICE"]),
    ],
    install_requires=["setuptools", "torch", "typing_extensions"],
    zip_safe=True,
    maintainer="yidahao",
    maintainer_email="haoyida@huawei.com",
    description="Self-developed PyTorch models without ROS runtime dependencies",
    license="Apache-2.0",
    python_requires=">=3.10",
)

from glob import glob

from setuptools import find_packages, setup

package_name = "ibrobot_tracing"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="IB-Robot Team",
    maintainer_email="dev@example.com",
    description="Offline tracing, performance analysis, and Python instrumentation for IB-Robot",
    license="Apache-2.0",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ibrobot-trace = ibrobot_tracing.cli:main",
        ],
    },
)

from setuptools import find_packages, setup

package_name = "benchmark_libero"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="ib-robot",
    maintainer_email="ib-robot@example.com",
    description="LIBERO child adapter package for IB-Robot benchmark evaluation",
    license="Apache-2.0",
    entry_points={
        "ibrobot.benchmark_adapters": [
            "libero = benchmark_libero.plugin:create_plugin",
        ],
    },
)

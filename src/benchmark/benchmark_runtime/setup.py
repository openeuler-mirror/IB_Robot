from setuptools import find_packages, setup

package_name = "benchmark_runtime"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pyyaml", "observation_transport"],
    zip_safe=True,
    maintainer="ib-robot",
    maintainer_email="ib-robot@example.com",
    description="Generic ROS runtime for IB-Robot benchmark evaluation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "benchmark_environment_node = benchmark_runtime.environment_node:main",
            "benchmark_evaluator_node = benchmark_runtime.evaluator_node:main",
        ],
        # Concrete adapters register in their own packages; the generic
        # runtime remains provider-agnostic.
    },
)

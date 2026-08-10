from setuptools import find_packages, setup

package_name = "observation_transport"

setup(
    name=package_name,
    version="0.3.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "av>=15,<16", "tensormsg"],
    zip_safe=True,
    maintainer="ib-robot",
    maintainer_email="ib-robot@example.com",
    description="Provider-neutral direct-frame video transport for IB-Robot observations",
    license="Apache-2.0",
)

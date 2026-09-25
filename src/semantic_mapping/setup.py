from glob import glob

from setuptools import find_packages, setup

package_name = "semantic_mapping"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        (
            "share/" + package_name + "/scripts",
            ["scripts/publish_rgbd_fixture.py", "scripts/verify_rgbd_fixture.py"],
        ),
    ],
    install_requires=["setuptools", "pyyaml"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Developers",
    maintainer_email="dev@openeuler.org",
    description="Persistent RGB-D 3D semantic mapping with open-vocabulary perception",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "semantic_mapping_node = semantic_mapping.semantic_mapping_node:main",
            "offline_mapping_node = semantic_mapping.offline_mapping_node:main",
            "semantic_map_migrate = semantic_mapping.migrate_database:main",
            "semantic_map_render_labels = semantic_mapping.final_label_visualization:main",
            "semantic_map_manual_labels = semantic_mapping.manual_labels:main",
            "semantic_map_standoff = semantic_mapping.standoff_query:main",
            "save_semantic_map = semantic_mapping.dataset_capture:main",
        ],
    },
)

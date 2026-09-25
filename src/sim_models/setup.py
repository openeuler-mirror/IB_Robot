from glob import glob

from setuptools import find_packages, setup

package_name = "sim_models"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ament resource index marker
        ("share/ament_index/resource_index/packages", ["resource/sim_models"]),
        # package.xml
        ("share/sim_models", ["package.xml"]),
        # empty scene (static world files, no meshes)
        ("share/sim_models/scenes/empty", glob("scenes/empty/*")),
        # pick_banana scene: layout + templates (NOT meshes — separate entry below)
        (
            "share/sim_models/scenes/pick_banana",
            glob("scenes/pick_banana/*.yaml") + glob("scenes/pick_banana/*.template"),
        ),
        # pick_banana mesh assets, grouped by role:
        #   *.obj — MuJoCo visual meshes (banana_g.obj with PBR material via
        #           material.mtl; other *.obj files are alternative mesh variants)
        #   *.stl — CoACD convex-decomposition collision hulls (banana_col_*.stl,
        #           plate_col_*.stl) plus banana_g.stl Gazebo visual mesh
        #           (STL avoids OGRE2 GL3Plus buffer double-lock under software
        #           rendering) and plate_collision.stl
        #   *.png — PBR texture maps (basecolor, normal, metallic/roughness)
        #   *.mtl — OBJ material definitions referencing the PNG textures
        (
            "share/sim_models/scenes/pick_banana/meshes",
            glob("scenes/pick_banana/meshes/*.obj")
            + glob("scenes/pick_banana/meshes/*.stl")
            + glob("scenes/pick_banana/meshes/*.png")
            + glob("scenes/pick_banana/meshes/*.mtl"),
        ),
        # aruco_a4 calibration prop
        ("share/sim_models/props/calib/aruco_a4/meshes", glob("props/calib/aruco_a4/meshes/*")),
        ("share/sim_models/props/calib/aruco_a4/gazebo", glob("props/calib/aruco_a4/gazebo/*")),
        ("share/sim_models/props/calib/aruco_a4/mujoco", glob("props/calib/aruco_a4/mujoco/*")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB Robot Contributors",
    maintainer_email="ib-robot@openeuler.org",
    description="Scene assets and scene compiler for IB-Robot simulation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_camera_adjuster = sim_models.sim_camera_adjuster:main",
            "pick_banana_task_node = sim_models.tasks.pick_banana:main",
        ],
    },
)

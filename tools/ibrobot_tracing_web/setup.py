import os
import warnings
from glob import glob
from os.path import relpath
from pathlib import Path

from setuptools import find_packages, setup

package_name = "ibrobot_tracing_web"
package_root = Path(__file__).resolve().parent
workspace_root = Path(__file__).resolve().parents[2]
dist_override = os.environ.get("IBROBOT_TRACING_WEB_DIST")
web_root = (
    Path(dist_override).expanduser().resolve()
    if dist_override
    else workspace_root / "web" / "ibrobot_tracing_ui" / "dist"
)
if not web_root.is_dir():
    warnings.warn(f"Vue assets not found at {web_root}; building API-only package", stacklevel=1)
web_files = [
    (str(Path("share") / package_name / "web" / path.relative_to(web_root).parent), [relpath(path, package_root)])
    for path in web_root.rglob("*")
    if path.is_file()
]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ]
    + web_files,
    # Keep version constraints aligned with requirements/tracing-web.txt; the
    # package.xml entries describe ROS dependencies rather than pip constraints.
    install_requires=["fastapi>=0.100,<1", "pydantic>=2,<3", "setuptools", "uvicorn>=0.23,<1"],
    # ament_python packages must declare the pytest extra or colcon silently
    # falls back to unittest, collects 0 tests and reports a false OK.
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    maintainer="IB-Robot Team",
    maintainer_email="dev@example.com",
    description="LAN-capable FastAPI backend for IB-Robot offline trace analysis",
    license="Apache-2.0",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "ibrobot-tracing-web = ibrobot_tracing_web.cli:main",
        ],
    },
)

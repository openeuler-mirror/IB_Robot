"""Package-local pytest configuration for inference_service.

This package ships its own ``pytest.ini`` (dual test directories and the
launch_testing plugin opt-out), which makes this directory the pytest
rootdir and truncates the upward conftest.py search at this directory. The
workspace-root ``conftest.py`` therefore never loads, and with it the
per-process ROS domain allocation would be silently skipped under the
standard ``colcon test`` entry - exactly the isolation gap the root conftest
exists to close. ``confcutdir`` cannot help here: it is a command-line-only
option and cannot be set from an ini file.

Instead of duplicating the isolation logic, this conftest imports the
root module and re-exports its hooks, so both entry points share one
implementation and the reservation lifecycle stays single-sourced. The
markers (``model_bundle`` / ``repo_asset``) and their skip logic come along
for the same reason: this package's suites use them.
"""

from __future__ import annotations

import importlib.util
import pathlib

_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ROOT_CONFTEST = _WORKSPACE_ROOT / "conftest.py"


def _load_root_conftest():
    spec = importlib.util.spec_from_file_location("ibrobot_root_conftest", _ROOT_CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_root = _load_root_conftest()

# Re-export the shared hooks so pytest picks them up from this package too.
DOMAIN_ID_ENV = _root.DOMAIN_ID_ENV
pytest_configure = _root.pytest_configure
pytest_unconfigure = _root.pytest_unconfigure
pytest_collection_modifyitems = _root.pytest_collection_modifyitems
model_bundle_available = _root.model_bundle_available
repo_asset_available = _root.repo_asset_available

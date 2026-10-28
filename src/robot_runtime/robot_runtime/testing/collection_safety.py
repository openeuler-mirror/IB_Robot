"""Import-safety of test modules without optional simulator packages.

A machine without a simulator (openEuler: no ``mujoco_ros2_control`` rpms,
no Python ``mujoco`` in the venv) must still COLLECT every other test of a
package. That fails invisibly when a test module imports a simulator-only
package at module level: the whole file errors during collection and every
test in it — including simulator-independent ones — disappears.

Each module is imported in a FRESH interpreter subprocess with those imports
blocked BEFORE anything else runs (a ``sys.modules`` entry of ``None`` makes
``import`` raise ImportError). The subprocess isolates the blocking from the
calling process's import cache: in-process blocking could be bypassed by an
already-cached module or leak into subsequent tests.

Test-support API, not runtime API (see ``robot_runtime.testing``): importing
this module starts no process; each helper call spawns one short-lived child.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def import_module_with_blocked_imports(
    module: str,
    *,
    blocked: Iterable[str],
    cwd: Path | str,
    timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a fresh interpreter with ``blocked`` imports poisoned.

    ``sys.modules[<name>] = None`` is installed before anything else, so any
    later ``import <name>`` — at module level or lazily — raises ImportError.
    The return code is 0 exactly when the module imports without the blocked
    packages; the caller asserts on it with the captured stderr.
    """
    script = "\n".join(
        [
            "import sys",
            *(f"sys.modules[{name!r}] = None  # import {name} now raises ImportError" for name in blocked),
            "import importlib",
            "importlib.import_module(sys.argv[1])",
        ]
    )
    return subprocess.run(
        [sys.executable, "-c", script, module],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

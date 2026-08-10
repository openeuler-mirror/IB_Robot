#!/usr/bin/env python3
"""Setup-owned trusted init-state loader for the pinned LIBERO provider.

Used by the off-screen smoke in verify_env.sh. Replicates the production
adapter's trusted loading semantics (src/benchmark/adapters/libero/
benchmark_libero/init_state_loader.py) without depending on a built ROS
package, so it works in a fresh setup before colcon build.

Security: weights_only=False is ONLY acceptable because the init-state
files are pinned repository assets inside libs/libero at commit
8f1084e3132a39270c3a13ebe37270a43ece2a01. The path is resolved through
the pinned LIBERO API (get_libero_path + task metadata). This must NOT
be used for arbitrary external checkpoints.
"""

from __future__ import annotations

import os
from typing import Any


class SetupInitStateLoadError(RuntimeError):
    """Raised when the trusted LIBERO init-state file cannot be loaded."""


def resolve_init_states_path(task: Any, get_libero_path_fn: Any) -> str:
    """Resolve the trusted init-state file path from pinned task metadata."""
    if task is None:
        raise SetupInitStateLoadError("task metadata is None; cannot resolve init-state path")
    problem_folder = getattr(task, "problem_folder", None)
    init_states_file = getattr(task, "init_states_file", None)
    if not problem_folder or not init_states_file:
        raise SetupInitStateLoadError(
            f"task metadata missing problem_folder/init_states_file; "
            f"got problem_folder={problem_folder!r}, init_states_file={init_states_file!r}"
        )
    base = get_libero_path_fn("init_states")
    if not base:
        raise SetupInitStateLoadError("get_libero_path('init_states') returned empty")
    return os.path.join(str(base), str(problem_folder), str(init_states_file))


def load_trusted_init_states(init_states_path: str) -> Any:
    """Load a trusted LIBERO init-state file with weights_only=False.

    SECURITY: weights_only=False is acceptable ONLY because this file is a
    pinned repository asset under libs/libero at a verified commit. The path
    is resolved through the pinned LIBERO API. Must NOT be used for arbitrary
    external checkpoints.
    """
    import torch  # noqa: PLC0415 -- heavy import deferred

    if not os.path.isfile(init_states_path):
        raise SetupInitStateLoadError(f"trusted init-state file does not exist: {init_states_path}")
    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:
        # torch < 2.6 does not accept weights_only kwarg; fall back to
        # the legacy default (which was weights_only=False).
        return torch.load(init_states_path)

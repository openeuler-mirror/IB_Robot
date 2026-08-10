"""Explicit trusted init-state loader for the pinned LIBERO provider (Trusted initial-state loading).

LIBERO runtime Trusted initial-state loading: the pinned ``libs/libero`` ``benchmark.get_task_init_states(i)``
calls ``torch.load(init_states_path)`` without ``weights_only=False``. Under
torch 2.6+ the default changed to ``weights_only=True``, which rejects the
legacy numpy pickle used by LIBERO's init-state files.

This module resolves the trusted fixed init-state asset from the pinned
LIBERO task metadata + ``get_libero_path("init_states")`` and loads it
locally with an explicit ``torch.load(path, weights_only=False)``.

Security comment: the init-state files are pinned repository assets inside
``libs/libero`` at commit ``8f1084e3132a39270c3a13ebe37270a43ece2a01``. The
adapter only loads files resolved through the pinned LIBERO API and never
accepts arbitrary user paths. ``weights_only=False`` is therefore acceptable
here; it must NOT be used for arbitrary external checkpoints.
"""

from __future__ import annotations

import os
from typing import Any


class InitStateLoadError(RuntimeError):
    """Raised when the trusted LIBERO init-state file cannot be loaded."""


def resolve_init_states_path(task: Any, get_libero_path_fn: Any) -> str:
    """Resolve the trusted init-state file path from pinned LIBERO task metadata.

    ``task`` is a ``libero.libero.benchmark.Task`` NamedTuple carrying
    ``problem_folder`` and ``init_states_file``. ``get_libero_path_fn`` is the
    pinned ``libero.libero.get_libero_path`` callable. The resolved path is
    ``<get_libero_path("init_states")>/<task.problem_folder>/<task.init_states_file>``.
    """
    if task is None:
        raise InitStateLoadError("task metadata is None; cannot resolve init-state path")
    problem_folder = getattr(task, "problem_folder", None)
    init_states_file = getattr(task, "init_states_file", None)
    if not problem_folder or not init_states_file:
        raise InitStateLoadError(
            f"task metadata is missing problem_folder/init_states_file; "
            f"got problem_folder={problem_folder!r}, init_states_file={init_states_file!r}"
        )
    base = get_libero_path_fn("init_states")
    if not base:
        raise InitStateLoadError("get_libero_path('init_states') returned an empty path")
    return os.path.join(str(base), str(problem_folder), str(init_states_file))


def load_trusted_init_states(init_states_path: str) -> Any:
    """Load a trusted LIBERO init-state file with explicit ``weights_only=False``.

    The file is a pinned repository asset under ``libs/libero`` at the
    verified commit. ``weights_only=False`` is required because the pinned
    LIBERO provider pickles init states with the legacy numpy format. This
    function must NOT be used for arbitrary external checkpoints.
    """
    import torch  # noqa: PLC0415 -- heavy import deferred to adapter boundary

    if not os.path.isfile(init_states_path):
        raise InitStateLoadError(f"trusted init-state file does not exist: {init_states_path}")
    # SECURITY: weights_only=False is acceptable ONLY because this file is a
    # pinned repository asset under libs/libero at a verified commit. The
    # adapter resolves the path through the pinned LIBERO API and never
    # accepts arbitrary user-supplied paths.
    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:
        # torch < 2.6 does not accept weights_only kwarg; fall back to the
        # legacy default (which was weights_only=False).
        return torch.load(init_states_path)

"""LIBERO provider identity probe (LIBERO runtime).

Fails closed unless the imported ``libero`` provider is the pinned
repository provider at ``libs/libero`` (commit
``8f1084e3132a39270c3a13ebe37270a43ece2a01``, package version ``0.1.0``).

The probe is only called from the LIBERO adapter ``configure()`` (heavy
import happens there, not during plugin discovery). It verifies:

- the imported ``libero`` module resolves under this repository's
  ``libs/libero`` tree or its documented editable/install equivalent;
- the package distribution version is ``0.1.0`` where available;
- the required APIs exist: ``benchmark.get_benchmark_dict``,
  ``get_libero_path``, ``OffScreenRenderEnv``, suite task/init-state access;
- Pinned provider verification: the actual ``libs/libero`` submodule HEAD matches the pinned commit
  SHA ``8f1084e3132a39270c3a13ebe37270a43ece2a01`` (full SHA, not just a
  prefix). Mismatch fails closed with both expected and actual full SHA in
  the error message. The submodule is read-only: the probe never modifies
  it;
- an installed ``hf-libero`` or another ``libero`` provider cannot win
  imports nondeterministically.

The probe must NOT modify ``libs/libero`` to add identity metadata.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProviderProbeError(RuntimeError):
    """Raised when the LIBERO provider identity cannot be verified."""


EXPECTED_LIBERO_PACKAGE_VERSION = "0.1.0"
# Pinned provider verification: the probe now verifies the actual ``libs/libero`` HEAD against the
# full expected SHA, not just a prefix. The prefix constant is kept for
# backwards-compatible identity snapshots.
EXPECTED_LIBERO_COMMIT_SHA = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
EXPECTED_LIBERO_COMMIT_SHA_PREFIX = "8f1084e3"

# The repository root contains the ``libs/libero`` submodule. The probe walks
# up from this file to find the expected on-disk path of the pinned provider.
_THIS_FILE = Path(__file__).resolve()
# benchmark_libero/version_probe.py -> src/benchmark/adapters/libero/benchmark_libero/
_ADAPTER_PKG_DIR = _THIS_FILE.parent.parent
# Walk up to the repository root that contains ``libs/libero``.
_REPO_ROOT = _ADAPTER_PKG_DIR
while _REPO_ROOT.parent != _REPO_ROOT and not (_REPO_ROOT / "libs" / "libero").exists():
    _REPO_ROOT = _REPO_ROOT.parent


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Snapshot of the verified LIBERO provider identity."""

    module_path: str
    distribution_version: str | None
    on_disk_repo_libero_path: str | None
    apis_present: tuple[str, ...]
    expected_commit_prefix: str
    expected_commit_sha: str
    actual_commit_sha: str | None


def _repo_libero_path() -> str | None:
    candidate = _REPO_ROOT / "libs" / "libero"
    if candidate.exists():
        return str(candidate)
    return None


def _check_module_path(module: Any) -> str:
    """Return the resolved on-disk module path.

    Handles both regular packages (``module.__file__`` is set to
    ``pkg/__init__.py``) and namespace packages (``module.__file__`` is
    ``None``; ``module.__path__`` is a list of directories). For namespace
    packages, returns the first path entry so the caller can verify it
    lives under the repository's ``libs/libero`` tree.
    """
    path = getattr(module, "__file__", None)
    if path and isinstance(path, str):
        return path
    # Namespace package: __file__ is None; __path__ is a list of directories.
    module_path = getattr(module, "__path__", None)
    if module_path is None:
        raise ProviderProbeError("libero provider did not expose __file__ or __path__; cannot verify on-disk path")
    try:
        first = next(iter(module_path))
    except StopIteration as exc:
        raise ProviderProbeError("libero provider __path__ is empty; cannot verify on-disk path") from exc
    if not isinstance(first, str) or not first:
        raise ProviderProbeError(f"libero provider __path__ entry is not a non-empty string: {first!r}")
    return first


def _verify_path_in_repo(path: str, repo_libero_path: str | None) -> None:
    """The libero module file must live under the repo's ``libs/libero``
    tree OR a documented editable install that resolves to the same tree.
    """
    if repo_libero_path is None:
        # No ``libs/libero`` in the repository: the probe cannot prove the
        # provider is the pinned one. Fail closed.
        raise ProviderProbeError(
            "could not locate the repository's libs/libero tree; cannot prove "
            "the imported libero provider is the pinned one"
        )

    real_path = os.path.realpath(path)
    real_repo = os.path.realpath(repo_libero_path)
    if not real_path.startswith(real_repo + os.sep):
        raise ProviderProbeError(
            f"imported libero module path '{real_path}' does not live under the "
            f"repository's libs/libero tree '{real_repo}'; an installed hf-libero "
            "or another provider may have stolen the import. Refusing to proceed."
        )


def _verify_distribution_version() -> str | None:
    """Return the imported ``libero`` distribution version if declared.

    Some editable installs do not declare a distribution; that is acceptable
    as long as the on-disk path check passes. When the distribution is
    present, the version must equal ``0.1.0``.
    """
    try:
        version = importlib.metadata.version("libero")
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"failed to read libero distribution version: {exc}") from exc
    if version != EXPECTED_LIBERO_PACKAGE_VERSION:
        raise ProviderProbeError(
            f"libero distribution version is '{version}', expected "
            f"'{EXPECTED_LIBERO_PACKAGE_VERSION}'; the pinned provider may "
            "have been replaced by an installed hf-libero or another version"
        )
    return version


def _verify_required_apis(module: Any) -> tuple[str, ...]:
    """Verify the required LIBERO APIs are present on the imported module."""
    required = (
        ("libero.libero.benchmark", "get_benchmark_dict"),
        ("libero.libero", "get_libero_path"),
        ("libero.libero.envs", "OffScreenRenderEnv"),
    )
    for sub_path, attr in required:
        try:
            submodule = importlib.import_module(sub_path)
        except Exception as exc:  # noqa: BLE001
            raise ProviderProbeError(f"failed to import required LIBERO submodule '{sub_path}': {exc}") from exc
        if not hasattr(submodule, attr):
            raise ProviderProbeError(
                f"LIBERO submodule '{sub_path}' is missing required attribute '{attr}'; provider API mismatch"
            )
    return tuple(attr for _, attr in required)


def _read_submodule_head(repo_libero_path: str) -> str:
    """Read the actual ``libs/libero`` submodule HEAD commit SHA.

    Uses ``git -C <path> rev-parse HEAD``. The probe never modifies the
    submodule. Raises :class:`ProviderProbeError` if git is unavailable or
    the SHA cannot be proven (e.g., detached-unresolved state, missing
    .git directory, command timeout).
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_libero_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProviderProbeError("git is not available; cannot verify libs/libero submodule HEAD") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProviderProbeError("git rev-parse HEAD on libs/libero timed out; cannot verify submodule HEAD") from exc
    except Exception as exc:  # noqa: BLE001
        raise ProviderProbeError(f"unexpected error running git rev-parse HEAD on libs/libero: {exc}") from exc
    if result.returncode != 0:
        raise ProviderProbeError(
            f"git rev-parse HEAD on libs/libero failed (exit {result.returncode}): "
            f"{result.stderr.strip() or 'no stderr'}; cannot verify submodule HEAD"
        )
    sha = result.stdout.strip()
    if not sha or len(sha) != 40:
        raise ProviderProbeError(
            f"git rev-parse HEAD on libs/libero returned an unexpected value: "
            f"{result.stdout!r}; cannot verify submodule HEAD"
        )
    return sha


def _verify_submodule_commit(repo_libero_path: str) -> str:
    """Verify the actual ``libs/libero`` HEAD matches the pinned commit.

    Pinned provider verification: the probe verifies the full 40-character SHA, not just a prefix.
    Mismatch fails closed with both expected and actual full SHA in the
    error message. The submodule is read-only.
    """
    actual = _read_submodule_head(repo_libero_path)
    if actual != EXPECTED_LIBERO_COMMIT_SHA:
        raise ProviderProbeError(
            f"libs/libero submodule HEAD mismatch: expected full SHA "
            f"'{EXPECTED_LIBERO_COMMIT_SHA}' but got '{actual}'. The pinned "
            "provider must be checked out at the exact commit recorded in the "
            "LIBERO runtime frozen identity. Refusing to proceed."
        )
    return actual


def probe_libero_provider() -> ProviderIdentity:
    """Verify the imported LIBERO provider is the pinned repository provider.

    Imports the ``libero`` package lazily here (not at module import time),
    so plugin discovery does not create a MuJoCo context. Returns a frozen
    :class:`ProviderIdentity` snapshot with the verified path, version,
    APIs and the actual submodule HEAD SHA.

    Raises :class:`ProviderProbeError` on any mismatch.
    """
    # Pre-flight: refuse to run if a stale libero is already in
    # ``sys.modules`` from an earlier import attempt. The probe must be the
    # first import of libero in the environment.
    if "libero" in sys.modules:
        # If the existing module's path does not satisfy the repo tree check,
        # fail closed now to avoid a silent provider switch.
        existing = sys.modules["libero"]
        path = _check_module_path(existing)
        repo_path = _repo_libero_path()
        _verify_path_in_repo(path, repo_path)
    else:
        try:
            module = importlib.import_module("libero")
        except Exception as exc:  # noqa: BLE001
            raise ProviderProbeError(
                f"failed to import 'libero' provider: {exc}. The pinned "
                "libs/libero provider must be installed (editable or pip "
                "install -e) before the LIBERO adapter can run."
            ) from exc
        path = _check_module_path(module)
        repo_path = _repo_libero_path()
        _verify_path_in_repo(path, repo_path)

    version = _verify_distribution_version()
    apis = _verify_required_apis(importlib.import_module("libero"))

    # Pinned provider verification: verify the actual libs/libero HEAD matches the pinned commit.
    actual_commit_sha: str | None = None
    if repo_path is not None:
        actual_commit_sha = _verify_submodule_commit(repo_path)

    return ProviderIdentity(
        module_path=path,
        distribution_version=version,
        on_disk_repo_libero_path=repo_path,
        apis_present=apis,
        expected_commit_prefix=EXPECTED_LIBERO_COMMIT_SHA_PREFIX,
        expected_commit_sha=EXPECTED_LIBERO_COMMIT_SHA,
        actual_commit_sha=actual_commit_sha,
    )

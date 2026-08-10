#!/usr/bin/env python3
"""Unified provider identity collector for LIBERO benchmark.

Used by both setup_python_venv.sh (install-time) and verify_env.sh
(verify-time) to avoid contract drift.

Exit codes:
  0 = all checks passed, PROVIDER_IDENTITY_OK printed
  1 = any check failed

Checks:
  1. import libero succeeds
  2. libero.__path__ non-empty, all paths under pinned checkout
  3. import libero.libero succeeds
  4. libero.libero.__file__ non-empty, under pinned checkout
  5. OffScreenRenderEnv importable and is a class
  6. direct_url.json exists, editable location = libs/libero
  7. parent gitlink == submodule HEAD
  8. submodule working tree clean
  9. git commands all return 0
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_git(args: list[str], cwd: str) -> tuple[int, str, str]:
    """Run a git command and return (rc, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git", "-C", cwd] + args,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def verify_provider_identity() -> int:
    errors: list[str] = []

    workspace = os.environ.get("WORKSPACE", ".")
    repo_libero = Path(workspace) / "libs" / "libero"
    real_repo = os.path.realpath(str(repo_libero))

    print(f"repo_libero_path={repo_libero}")
    print(f"real_repo={real_repo}")

    # 1. Import libero
    try:
        mod = importlib.import_module("libero")
    except Exception as exc:
        print(f"libero_import_failed={exc}", file=sys.stderr)
        return 1

    # 2. libero.__file__ may be None (namespace package)
    libero_file = getattr(mod, "__file__", None)
    print(f"libero_file={libero_file}")

    libero_paths = list(getattr(mod, "__path__", []))
    print(f"libero_path_count={len(libero_paths)}")
    for i, p in enumerate(libero_paths):
        print(f"libero_path[{i}]={p}")
        resolved = os.path.realpath(p)
        if not resolved.startswith(real_repo + os.sep) and resolved != real_repo:
            errors.append(f"libero_path[{i}] {resolved} not under {real_repo}")

    if not libero_paths:
        errors.append("libero.__path__ is empty")

    # 3. Import libero.libero
    try:
        libero_libero = importlib.import_module("libero.libero")
    except Exception as exc:
        print(f"libero_libero_import_failed={exc}", file=sys.stderr)
        return 1

    # 4. libero.libero.__file__ must be non-empty and under pinned checkout
    libero_libero_file = getattr(libero_libero, "__file__", None)
    print(f"libero_libero_file={libero_libero_file}")
    if not libero_libero_file:
        errors.append("libero.libero.__file__ is None or empty")
    else:
        resolved_file = os.path.realpath(libero_libero_file)
        if not resolved_file.startswith(real_repo + os.sep):
            errors.append(f"libero.libero.__file__ {resolved_file} not under {real_repo}")

    # 5. OffScreenRenderEnv importable and is a class
    try:
        from libero.libero.envs import OffScreenRenderEnv

        is_class = isinstance(OffScreenRenderEnv, type)
        print(f"OffScreenRenderEnv_is_class={is_class}")
        if not is_class:
            errors.append("OffScreenRenderEnv is not a class")
    except Exception as exc:
        print(f"OffScreenRenderEnv_import_failed={exc}", file=sys.stderr)
        return 1

    # 6. direct_url.json must exist, editable=True, and location must match
    try:
        dist = importlib.metadata.distribution("libero")
        print(f"libero_distribution_version={dist.version}")
        direct_url_json = dist.read_text("direct_url.json")
        if not direct_url_json:
            errors.append("direct_url.json missing or empty — cannot verify editable location")
            print("libero_editable_location=MISSING")
        else:
            du = json.loads(direct_url_json)
            url = du.get("url", "")
            editable_loc = url[7:] if url.startswith("file://") else url
            print(f"libero_editable_location={editable_loc}")
            if not editable_loc:
                errors.append("direct_url.json has no url field")
            else:
                resolved_loc = os.path.realpath(editable_loc)
                if resolved_loc != real_repo:
                    errors.append(f"editable location {resolved_loc} != {real_repo}")
            # Editable-install check: Check dir_info.editable is True (boolean, not string)
            dir_info = du.get("dir_info")
            if not isinstance(dir_info, dict):
                errors.append("direct_url.json has no dir_info or it is not a mapping")
                print("libero_editable_flag=MISSING")
            else:
                editable_flag = dir_info.get("editable")
                print(f"libero_editable_flag={editable_flag}")
                if editable_flag is not True:
                    errors.append(f"dir_info.editable is {editable_flag!r}, expected True (boolean)")
    except importlib.metadata.PackageNotFoundError:
        errors.append("libero distribution not found")
    except Exception as exc:
        print(f"libero_distribution_error={exc}")
        errors.append(f"distribution check failed: {exc}")

    # 7+8+9. Git checks — each must return 0
    rc_gitlink, out_gitlink, err_gitlink = _run_git(["ls-tree", "HEAD", "libs/libero"], workspace)
    if rc_gitlink != 0:
        errors.append(f"git ls-tree failed (rc={rc_gitlink}): {err_gitlink}")
        print("parent_gitlink=GIT_ERROR")
    else:
        parts = out_gitlink.split()
        gitlink_sha = parts[2] if len(parts) >= 3 else ""
        print(f"parent_gitlink={gitlink_sha}")

    rc_head, out_head, err_head = _run_git(["rev-parse", "HEAD"], str(repo_libero))
    if rc_head != 0:
        errors.append(f"git rev-parse HEAD failed (rc={rc_head}): {err_head}")
        print("submodule_head=GIT_ERROR")
    else:
        head_sha = out_head
        print(f"submodule_head={head_sha}")
        # Compare only if both succeeded
        if rc_gitlink == 0 and gitlink_sha and head_sha and gitlink_sha != head_sha:
            errors.append(f"gitlink {gitlink_sha} != HEAD {head_sha}")

    rc_dirty, out_dirty, err_dirty = _run_git(["status", "--porcelain"], str(repo_libero))
    if rc_dirty != 0:
        errors.append(f"git status failed (rc={rc_dirty}): {err_dirty}")
        print("submodule_dirty=GIT_ERROR")
    else:
        is_dirty = bool(out_dirty)
        print(f"submodule_dirty={is_dirty}")
        if is_dirty:
            errors.append("submodule working tree is dirty")

    # Report
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("PROVIDER_IDENTITY_OK")
    return 0


def main() -> int:
    return verify_provider_identity()


if __name__ == "__main__":
    sys.exit(main())

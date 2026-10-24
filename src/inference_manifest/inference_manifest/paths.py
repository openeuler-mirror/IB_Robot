"""Safe path handling for policy bundle contents."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from inference_manifest.errors import ManifestPathError

_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def normalize_bundle_path(path: str) -> str:
    """Validate and return one canonical POSIX bundle-relative path."""

    if not path:
        raise ManifestPathError("bundle-relative path must not be empty")
    if "\x00" in path:
        raise ManifestPathError(f"bundle-relative path contains a NUL byte: {path!r}")
    if "\\" in path:
        raise ManifestPathError(f"bundle-relative path must use POSIX separators, not backslashes: {path!r}")
    if path.startswith("/"):
        raise ManifestPathError(f"absolute path is not allowed in a bundle manifest: {path!r}")
    if _DRIVE_PREFIX.match(path):
        raise ManifestPathError(f"drive-qualified path is not allowed in a bundle manifest: {path!r}")

    parts = path.split("/")
    if any(part == "" for part in parts):
        raise ManifestPathError(f"empty path segments are not allowed: {path!r}")
    if any(part == "." for part in parts):
        raise ManifestPathError(f"dot path segments are not allowed: {path!r}")
    if any(part == ".." for part in parts):
        raise ManifestPathError(f"parent traversal is not allowed: {path!r}")
    return "/".join(parts)


def normalize_unique_paths(paths: Iterable[str], description: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = normalize_bundle_path(raw_path)
        if path in seen:
            raise ManifestPathError(f"{description} contains duplicate normalized path {path!r}")
        seen.add(path)
        normalized.append(path)
    return tuple(normalized)


def resolve_bundle_path(bundle_root: Path, relative_path: str) -> Path:
    """Resolve an existing path and prove that it remains under the bundle root."""

    normalized = normalize_bundle_path(relative_path)
    try:
        root = _resolve_existing_path(bundle_root)
    except OSError as exc:
        raise ManifestPathError(f"Unable to resolve bundle root {bundle_root}: {exc}") from exc
    if not root.is_dir():
        raise ManifestPathError(f"Bundle root is not a directory: {bundle_root}")

    candidate = root.joinpath(*normalized.split("/"))
    try:
        resolved = _resolve_existing_path(candidate)
    except FileNotFoundError as exc:
        if candidate.is_symlink():
            raise ManifestPathError(f"broken symlink in bundle path {normalized!r}") from exc
        raise ManifestPathError(f"bundle path does not exist: {normalized!r}") from exc
    except OSError as exc:
        raise ManifestPathError(f"Unable to resolve bundle path {normalized!r}: {exc}") from exc

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestPathError(f"bundle path escapes the bundle root through a symlink: {normalized!r}") from exc
    return resolved


def _resolve_existing_path(path: Path) -> Path:
    try:
        return path.resolve(strict=True)
    except RuntimeError as exc:
        # Python 3.10/3.11 use RuntimeError instead of OSError(ELOOP).
        raise ManifestPathError(f"Unable to resolve bundle path {str(path)!r}: {exc}") from exc


def resolve_bundle_file(bundle_root: Path, relative_path: str) -> Path:
    resolved = resolve_bundle_path(bundle_root, relative_path)
    if not resolved.is_file():
        raise ManifestPathError(f"bundle path is not a regular file: {relative_path!r}")
    return resolved


def depends_on_file_replacement(path: Path, targets: set[Path], *, excluded_target: Path | None = None) -> bool:
    """Check a file's symlink chain against entries that will be replaced.

    Targets must be file entries with resolved parents. Writers must reject
    directory targets first: resolving parents here assumes they stay intact.
    Hardlinks to another entry survive atomic replacement and are not aliases
    for this purpose. Callers must exclude concurrent path mutations.
    An excluded target is skipped without ending traversal of its symlink chain.
    """
    _resolve_existing_path(path)
    while True:
        entry = _resolve_existing_path(path.parent) / path.name
        if entry != excluded_target and entry in targets:
            return True
        if not entry.is_symlink():
            return False
        target = entry.readlink()
        path = target if target.is_absolute() else entry.parent / target

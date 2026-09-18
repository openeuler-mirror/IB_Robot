"""Restricted trace-source discovery with opaque public identifiers."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir


class CatalogError(RuntimeError):
    pass


class SourceNotFoundError(CatalogError):
    pass


class UnsafeSourceError(CatalogError):
    pass


class SourceChangedError(CatalogError):
    pass


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    name: str
    kind: str
    fingerprint: str
    modified_at: datetime
    size_bytes: int
    file_count: int
    path: Path
    root: Path


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    generation: int
    refreshed_at: datetime
    sources: tuple[SourceRecord, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ScanResult:
    fingerprint: str
    modified_ns: int
    size_bytes: int
    file_count: int
    has_ctf_metadata: bool


_LOG_SUFFIXES = {".log", ".txt", ".trace"}


def _stat_key(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _open_entry(parent_fd: int, name: str, *, directory: bool = False) -> Iterator[int]:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise UnsafeSourceError("Symbolic links are not allowed")
        if not (stat.S_ISDIR(metadata.st_mode) or (not directory and stat.S_ISREG(metadata.st_mode))):
            raise UnsafeSourceError("Only directories and regular trace files are allowed")
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory or stat.S_ISDIR(metadata.st_mode):
            flags |= os.O_DIRECTORY
        fd = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        raise SourceChangedError("Trace source changed or no longer exists") from exc
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise UnsafeSourceError("Symbolic links or replaced directories are not allowed") from exc
        raise
    try:
        opened = os.fstat(fd)
        if not (stat.S_ISDIR(opened.st_mode) or stat.S_ISREG(opened.st_mode)):
            raise UnsafeSourceError("Only directories and regular trace files are allowed")
        if _stat_key(opened)[:3] != _stat_key(metadata)[:3]:
            raise SourceChangedError("Trace source changed while it was being opened")
        yield fd
        # Ancestors may gain unrelated files; only their identity must stay fixed.
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise SourceChangedError("Trace source changed or no longer exists") from exc
        if _stat_key(current)[:3] != _stat_key(opened)[:3]:
            raise SourceChangedError("Trace source changed while it was being read")
    finally:
        os.close(fd)


@contextmanager
def _open_path(path: Path, *, directory: bool = False) -> Iterator[int]:
    if not path.is_absolute() or ".." in path.parts:
        raise UnsafeSourceError("Trace sources must have absolute paths without parent traversal")
    with ExitStack() as stack:
        fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, fd)
        for index, part in enumerate(path.parts[1:], start=1):
            fd = stack.enter_context(_open_entry(fd, part, directory=directory or index < len(path.parts) - 1))
        yield fd


class SourceCatalog:
    def __init__(
        self,
        roots: tuple[Path, ...],
        *,
        max_sources: int = 256,
        max_scan_entries: int = 100_000,
        max_source_bytes: int = 1_073_741_824,
    ):
        self._roots = tuple(self._prepare_root(root) for root in roots)
        self._max_sources = max_sources
        self._max_scan_entries = max_scan_entries
        self._max_source_bytes = max_source_bytes
        self._lock = threading.RLock()
        self._refresh_lock = threading.Lock()
        self._generation = 0
        self._refreshed_at = datetime.now(timezone.utc)
        self._sources: dict[str, SourceRecord] = {}
        self._warnings: tuple[str, ...] = ()
        self._last_refresh = float("-inf")

    @staticmethod
    def _prepare_root(root: Path) -> Path:
        path = Path(os.path.abspath(root))
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current /= part
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                break
            if stat.S_ISLNK(mode):
                raise UnsafeSourceError(f"Trace root contains a symbolic link: {root}")
        if path.exists() and not path.is_dir():
            raise UnsafeSourceError(f"Trace root is not a directory: {root}")
        return path

    @staticmethod
    def _source_id(root: Path, relative_name: str) -> str:
        digest = hashlib.sha256()
        digest.update(os.fsencode(root))
        digest.update(b"\0")
        digest.update(relative_name.encode("utf-8", errors="surrogateescape"))
        return digest.hexdigest()[:32]

    def _scan(
        self, fd: int, *, destination: Path | None = None, cancelled: Callable[[], bool] | None = None
    ) -> _ScanResult:
        digest = hashlib.sha256()
        size_bytes = 0
        file_count = 0
        entry_count = 0
        modified_ns = 0
        has_ctf_metadata = False

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise CancelledError("Trace snapshot was cancelled")

        def visit(item_fd: int, relative: Path, output: Path | None) -> None:
            nonlocal entry_count, file_count, has_ctf_metadata, modified_ns, size_bytes
            check_cancelled()
            metadata = os.fstat(item_fd)
            digest.update(relative.as_posix().encode("utf-8", errors="surrogateescape"))
            digest.update(f"\0{_stat_key(metadata)}\0".encode())
            modified_ns = max(modified_ns, metadata.st_mtime_ns)
            if stat.S_ISDIR(metadata.st_mode):
                os.lseek(item_fd, 0, os.SEEK_SET)
                with os.scandir(item_fd) as iterator:
                    remaining = self._max_scan_entries - entry_count
                    entries = list(islice(iterator, remaining + 1))
                if len(entries) > remaining:
                    raise UnsafeSourceError("Trace source exceeds the configured scan-entry limit")
                # Reserve siblings before recursion consumes the remaining budget.
                entry_count += len(entries)
                if output is not None:
                    output.mkdir(mode=0o700)
                for entry in sorted(entries, key=lambda entry: entry.name):
                    with _open_entry(item_fd, entry.name) as child_fd:
                        visit(child_fd, relative / entry.name, output / entry.name if output is not None else None)
            elif stat.S_ISREG(metadata.st_mode):
                file_count += 1
                size_bytes += metadata.st_size
                if size_bytes > self._max_source_bytes:
                    raise UnsafeSourceError("Trace source exceeds the configured byte limit")
                has_ctf_metadata = has_ctf_metadata or relative.name == "metadata"
                if output is not None:
                    os.lseek(item_fd, 0, os.SEEK_SET)
                    with output.open("xb") as stream:
                        remaining_bytes = metadata.st_size
                        while remaining_bytes:
                            check_cancelled()
                            chunk = os.read(item_fd, min(remaining_bytes, 1024 * 1024))
                            if not chunk:
                                raise SourceChangedError("Trace source changed while it was being copied")
                            stream.write(chunk)
                            remaining_bytes -= len(chunk)
                        check_cancelled()
                        # Probe for growth without copying beyond the verified byte budget.
                        if os.read(item_fd, 1):
                            raise SourceChangedError("Trace source changed while it was being copied")
            else:
                raise UnsafeSourceError("Special files are not allowed in trace sources")
            if _stat_key(os.fstat(item_fd)) != _stat_key(metadata):
                raise SourceChangedError("Trace source changed while it was being scanned")

        try:
            visit(fd, Path(), destination)
        except RecursionError as exc:
            raise UnsafeSourceError("Trace source directory nesting exceeds the scan limit") from exc
        return _ScanResult(digest.hexdigest(), modified_ns, size_bytes, file_count, has_ctf_metadata)

    def _record(self, root: Path, path: Path, kind: str, scan: _ScanResult) -> SourceRecord:
        relative = path.relative_to(root)
        name = path.name if relative == Path(".") else relative.as_posix()
        return SourceRecord(
            source_id=self._source_id(root, relative.as_posix()),
            name=name,
            kind=kind,
            fingerprint=scan.fingerprint,
            modified_at=datetime.fromtimestamp(scan.modified_ns / 1_000_000_000, tz=timezone.utc),
            size_bytes=scan.size_bytes,
            file_count=scan.file_count,
            path=path,
            root=root,
        )

    def _candidate(self, root: Path, path: Path) -> SourceRecord | None:
        with _open_path(path) as fd:
            if stat.S_ISREG(os.fstat(fd).st_mode):
                if path.suffix.lower() not in _LOG_SUFFIXES:
                    return None
                return self._record(root, path, "log", self._scan(fd))
            scan = self._scan(fd)
            return self._record(root, path, "ctf", scan) if scan.has_ctf_metadata else None

    def _refresh(self) -> CatalogSnapshot:
        discovered: dict[str, SourceRecord] = {}
        warnings: list[str] = []
        for root_index, root in enumerate(self._roots, start=1):
            if not root.exists():
                warnings.append(f"Trace root {root_index} does not exist")
                continue
            try:
                with _open_path(root, directory=True) as fd, os.scandir(fd) as iterator:
                    entries = list(islice(iterator, self._max_scan_entries + 1))
            except (OSError, CatalogError) as exc:
                warnings.append(f"Could not scan trace root {root_index}: {type(exc).__name__}")
                continue
            if len(entries) > self._max_scan_entries:
                entries.pop()
                warnings.append(f"Trace root {root_index} exceeds the configured scan-entry limit")
            paths = sorted((root / entry.name for entry in entries), key=lambda path: path.name)
            for path in paths:
                if len(discovered) >= self._max_sources:
                    warnings.append(f"Catalog source limit ({self._max_sources}) reached")
                    break
                try:
                    record = self._candidate(root, path)
                except (OSError, CatalogError) as exc:
                    warnings.append(f"Skipped {path.name!r}: {exc}")
                    continue
                if record is not None:
                    discovered[record.source_id] = record
        refreshed_at = datetime.now(timezone.utc)
        with self._lock:
            self._generation += 1
            self._refreshed_at = refreshed_at
            self._sources = discovered
            self._warnings = tuple(warnings)
            return self.snapshot()

    def refresh(self) -> CatalogSnapshot:
        with self._refresh_lock:
            return self._refresh()

    def refresh_coalesced(self, *, min_interval_s: float = 2.0) -> CatalogSnapshot:
        if not self._refresh_lock.acquire(blocking=False):
            return self.snapshot()
        try:
            now = time.monotonic()
            if now - self._last_refresh < min_interval_s:
                return self.snapshot()
            result = self._refresh()
            self._last_refresh = time.monotonic()
            return result
        finally:
            self._refresh_lock.release()

    def snapshot(self) -> CatalogSnapshot:
        with self._lock:
            return CatalogSnapshot(
                generation=self._generation,
                refreshed_at=self._refreshed_at,
                sources=tuple(sorted(self._sources.values(), key=lambda source: (source.name, source.source_id))),
                warnings=self._warnings,
            )

    def get(self, source_id: str) -> SourceRecord:
        with self._lock:
            source = self._sources.get(source_id)
        if source is None:
            raise SourceNotFoundError("Trace source was not found")
        return source

    @contextmanager
    def _open_source(self, source: SourceRecord) -> Iterator[int]:
        if source.root not in self._roots:
            raise UnsafeSourceError("Trace source is outside its configured root")
        try:
            source.path.relative_to(source.root)
        except ValueError as exc:
            raise UnsafeSourceError("Trace source is outside its configured root") from exc
        with _open_path(source.path, directory=source.kind == "ctf") as fd:
            if source.kind == "log" and not stat.S_ISREG(os.fstat(fd).st_mode):
                raise UnsafeSourceError("Only regular trace files are allowed")
            yield fd

    @staticmethod
    def _validate_scan(source: SourceRecord, current: _ScanResult) -> None:
        if source.kind == "ctf" and not current.has_ctf_metadata:
            raise SourceChangedError("CTF metadata is no longer present")
        if current.fingerprint != source.fingerprint:
            raise SourceChangedError("Trace source changed; refresh the source catalog and submit a new job")

    def validate(self, source: SourceRecord) -> None:
        with self._open_source(source) as fd:
            self._validate_scan(source, self._scan(fd))

    @contextmanager
    def source_snapshot(self, source: SourceRecord, *, cancelled: Callable[[], bool] | None = None) -> Iterator[Path]:
        """Isolate a reader from mutable source paths, not from a hostile same-UID process."""
        if shutil.disk_usage(gettempdir()).free < source.size_bytes:
            raise UnsafeSourceError("Insufficient space for the trace snapshot")
        with TemporaryDirectory(prefix="ibrobot-trace-") as temporary:
            snapshot = Path(temporary) / source.path.name
            with self._open_source(source) as fd:
                self._validate_scan(source, self._scan(fd, cancelled=cancelled))
                self._validate_scan(source, self._scan(fd, destination=snapshot, cancelled=cancelled))
                # Recheck earlier siblings too: later copies may race edits to them.
                self._validate_scan(source, self._scan(fd, cancelled=cancelled))
            # Close/validate all source descriptors before handing off. The result
            # belongs to source.fingerprint even when the source changes later.
            yield snapshot
            if cancelled is not None and cancelled():
                raise CancelledError("Trace analysis was cancelled")

import os
import stat
from concurrent.futures import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ibrobot_tracing_web.catalog import SourceCatalog, SourceChangedError, SourceNotFoundError, UnsafeSourceError


@pytest.fixture
def snapshot_directories(monkeypatch, tmp_path):
    created = []
    fd_count = len(os.listdir("/proc/self/fd"))

    def temporary_directory(**kwargs):
        temporary = TemporaryDirectory(dir=tmp_path, **kwargs)
        created.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr("ibrobot_tracing_web.catalog.TemporaryDirectory", temporary_directory, raising=False)
    yield created
    assert all(not path.exists() for path in created)
    assert len(os.listdir("/proc/self/fd")) == fd_count


def test_catalog_uses_opaque_ids_and_detects_source_changes(tmp_path):
    trace = tmp_path / "robot.log"
    trace.write_text("first\n", encoding="utf-8")
    catalog = SourceCatalog((tmp_path,))

    snapshot = catalog.refresh()

    assert len(snapshot.sources) == 1
    source = snapshot.sources[0]
    assert source.name == "robot.log"
    assert source.kind == "log"
    assert str(tmp_path) not in source.source_id
    assert len(source.source_id) == 32
    assert catalog.get(source.source_id) == source
    with pytest.raises(SourceNotFoundError):
        catalog.get("../../robot.log")

    trace.write_text("changed contents\n", encoding="utf-8")
    with pytest.raises(SourceChangedError, match="refresh"):
        catalog.validate(source)
    refreshed = catalog.refresh().sources[0]
    assert refreshed.source_id == source.source_id
    assert refreshed.fingerprint != source.fingerprint


def test_catalog_discovers_ctf_and_rejects_symlinks(tmp_path):
    trace = tmp_path / "session"
    metadata = trace / "ust" / "uid" / "1000" / "64-bit"
    metadata.mkdir(parents=True)
    (metadata / "metadata").write_text("ctf metadata", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    (trace / "linked").symlink_to(outside)
    linked_log = tmp_path / "linked.log"
    linked_log.symlink_to(outside)
    catalog = SourceCatalog((tmp_path,))

    snapshot = catalog.refresh()

    assert snapshot.sources == ()
    assert sum("Symbolic links" in warning for warning in snapshot.warnings) == 2

    (trace / "linked").unlink()
    linked_log.unlink()
    snapshot = catalog.refresh()
    assert [(source.name, source.kind) for source in snapshot.sources] == [("session", "ctf")]


def test_catalog_rejects_a_symlinked_root(tmp_path):
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(UnsafeSourceError, match="symbolic link"):
        SourceCatalog((linked_root,))


def test_catalog_reports_missing_root_without_exposing_it(tmp_path):
    root = tmp_path / "missing"
    snapshot = SourceCatalog((root,)).refresh()

    assert snapshot.sources == ()
    assert snapshot.warnings == ("Trace root 1 does not exist",)
    assert str(root) not in snapshot.warnings[0]


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links are not supported")
def test_source_becoming_a_symlink_is_rejected(tmp_path):
    trace = tmp_path / "robot.log"
    trace.write_text("trace", encoding="utf-8")
    catalog = SourceCatalog((tmp_path,))
    source = catalog.refresh().sources[0]
    replacement = tmp_path / "replacement"
    replacement.write_text("trace", encoding="utf-8")
    trace.unlink()
    trace.symlink_to(replacement)

    with pytest.raises(UnsafeSourceError, match="Symbolic links"):
        catalog.validate(source)


def test_catalog_bounds_root_entries_before_sorting(tmp_path):
    for name in ("one.log", "two.log", "three.log"):
        (tmp_path / name).write_text("trace", encoding="utf-8")

    snapshot = SourceCatalog((tmp_path,), max_scan_entries=2).refresh()

    assert len(snapshot.sources) == 2
    assert "scan-entry limit" in snapshot.warnings[0]


@pytest.mark.parametrize("operation", ["refresh", "validate", "source_snapshot"])
@pytest.mark.parametrize("trailing_directory", [False, True])
def test_catalog_enforces_recursive_entry_budget(tmp_path, operation, trailing_directory):
    trace = tmp_path / "session"
    nested = trace / "a"
    nested.mkdir(parents=True)
    for name in ("metadata", "stream0", "stream1"):
        (nested / name).write_text("ctf", encoding="utf-8")
    (tmp_path / "z-valid.log").write_text("trace", encoding="utf-8")
    catalog = SourceCatalog((tmp_path,), max_scan_entries=4)
    snapshot = catalog.refresh()
    assert [source.name for source in snapshot.sources] == ["session", "z-valid.log"]
    assert snapshot.warnings == ()
    source = snapshot.sources[0]
    catalog.validate(source)

    (trace / "y").write_text("excess", encoding="utf-8")
    if trailing_directory:
        (trace / "z").mkdir()

    if operation == "validate":
        with pytest.raises(UnsafeSourceError, match="scan-entry limit"):
            catalog.validate(source)
    elif operation == "source_snapshot":
        with pytest.raises(UnsafeSourceError, match="scan-entry limit"), catalog.source_snapshot(source):
            pytest.fail("An over-budget source must not reach the reader")
    else:
        snapshot = catalog.refresh()
        assert [source.name for source in snapshot.sources] == ["z-valid.log"]
        assert snapshot.warnings == ("Skipped 'session': Trace source exceeds the configured scan-entry limit",)


def test_catalog_allows_empty_directory_at_exact_entry_budget(tmp_path):
    trace = tmp_path / "session"
    nested = trace / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "metadata").write_text("ctf", encoding="utf-8")
    (trace / "z").mkdir()
    catalog = SourceCatalog((tmp_path,), max_scan_entries=4)

    snapshot = catalog.refresh()

    assert [source.name for source in snapshot.sources] == ["session"]
    assert snapshot.warnings == ()
    assert snapshot.sources[0].file_count == 1
    catalog.validate(snapshot.sources[0])
    with catalog.source_snapshot(snapshot.sources[0]) as copied:
        assert (copied / "z").is_dir()


def test_catalog_rejects_sources_over_the_byte_limit(tmp_path):
    (tmp_path / "large.log").write_bytes(b"12345")

    snapshot = SourceCatalog((tmp_path,), max_source_bytes=4).refresh()

    assert snapshot.sources == ()
    assert "byte limit" in snapshot.warnings[0]


@pytest.mark.parametrize("kind", ["log", "ctf"])
def test_source_snapshot_is_private_bounded_and_cleaned(tmp_path, snapshot_directories, kind):
    root = tmp_path / "traces"
    root.mkdir()
    trace = root / ("robot.log" if kind == "log" else "session.ctf")
    if kind == "ctf":
        (trace / "ust").mkdir(parents=True)
        content = trace / "ust" / "metadata"
    else:
        content = trace
    content.write_bytes(b"trace")
    catalog = SourceCatalog((root,), max_source_bytes=5, max_scan_entries=2)
    source = catalog.refresh().sources[0]

    with catalog.source_snapshot(source) as snapshot:
        assert snapshot != trace
        assert stat.S_IMODE(snapshot_directories[0].stat().st_mode) == 0o700
        copied = snapshot / "ust" / "metadata" if kind == "ctf" else snapshot
        assert copied.read_bytes() == b"trace"
        assert stat.S_ISREG(copied.lstat().st_mode)
        assert copied.stat().st_ino != content.stat().st_ino
        assert catalog.get(source.source_id) == source

    assert not snapshot.exists()
    assert all(not path.exists() for path in snapshot_directories)


@pytest.mark.parametrize("kind", ["log", "ctf"])
@pytest.mark.parametrize("replacement", ["parent", "directory", "source", "regular", "fifo"])
def test_snapshot_rejects_replacement_between_stat_and_open(
    tmp_path, monkeypatch, snapshot_directories, kind, replacement
):
    root = tmp_path / "traces"
    root.mkdir()
    trace = root / ("robot.log" if kind == "log" else "session.ctf")
    if kind == "ctf":
        (trace / "ust").mkdir(parents=True)
        content = trace / "ust" / "metadata"
    else:
        content = trace
    content.write_bytes(b"safe")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / content.name).write_bytes(b"secret")
    catalog = SourceCatalog((root,))
    source = catalog.refresh().sources[0]
    original_open = os.open
    replaced = False

    def replace_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        target = root if replacement == "parent" else content.parent if replacement == "directory" else content
        if not replaced and path == target.name and dir_fd is not None:
            replaced = True
            target.rename(target.with_name(target.name + "-old"))
            if replacement == "fifo":
                os.mkfifo(target)
                assert flags & os.O_NONBLOCK
            elif replacement == "regular":
                target.write_bytes(b"secret")
            else:
                target.symlink_to(outside if replacement in {"parent", "directory"} else outside / content.name)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises((UnsafeSourceError, SourceChangedError)), catalog.source_snapshot(source):
        pytest.fail("A replaced source must not reach the reader")
    assert replaced
    assert all(not path.exists() for path in snapshot_directories)


@pytest.mark.parametrize("mutation", ["grow", "rewrite", "truncate", "cancel"])
def test_snapshot_rejects_changes_during_copy_and_cleans_up(tmp_path, monkeypatch, snapshot_directories, mutation):
    root = tmp_path / "traces"
    root.mkdir()
    trace = root / "robot.log"
    trace.write_bytes(b"1234")
    os.utime(trace, ns=(1_000_000_000, 1_000_000_000))
    catalog = SourceCatalog((root,), max_source_bytes=4)
    source = catalog.refresh().sources[0]
    original_read = os.read
    reads = []
    received = []
    cancelled = False

    def mutate_during_read(fd, count):
        nonlocal cancelled
        reads.append(count)
        if len(reads) == 1:
            if mutation == "grow":
                trace.write_bytes(b"1234" + b"x" * 100)
            elif mutation == "rewrite":
                trace.write_bytes(b"abcd")
            elif mutation == "truncate":
                trace.write_bytes(b"1")
            else:
                cancelled = True
        chunk = original_read(fd, count)
        received.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "read", mutate_during_read)
    expected = CancelledError if mutation == "cancel" else SourceChangedError
    with pytest.raises(expected), catalog.source_snapshot(source, cancelled=lambda: cancelled):
        pytest.fail("A changing source must not reach the reader")
    assert reads
    assert sum(received) <= 5
    assert snapshot_directories
    assert all(not path.exists() for path in snapshot_directories)


def test_snapshot_rechecks_previously_copied_ctf_files(tmp_path, monkeypatch, snapshot_directories):
    trace = tmp_path / "traces" / "session"
    trace.mkdir(parents=True)
    metadata = trace / "metadata"
    metadata.write_bytes(b"ctf")
    os.utime(metadata, ns=(1_000_000_000, 1_000_000_000))
    stream = trace / "stream"
    stream.write_bytes(b"data")
    catalog = SourceCatalog((trace.parent,))
    source = catalog.refresh().sources[0]
    original_read = os.read

    def mutate_previous_file(fd, count):
        if os.fstat(fd).st_ino == stream.stat().st_ino:
            metadata.write_bytes(b"new")
        return original_read(fd, count)

    monkeypatch.setattr(os, "read", mutate_previous_file)
    with pytest.raises(SourceChangedError), catalog.source_snapshot(source):
        pytest.fail("Changes to previously copied files must be detected")
    assert all(not path.exists() for path in snapshot_directories)


@pytest.mark.parametrize("failure", ["exception", "timeout"])
def test_snapshot_cleans_up_when_reader_fails(tmp_path, snapshot_directories, failure):
    trace = tmp_path / "robot.log"
    trace.write_bytes(b"trace")
    catalog = SourceCatalog((tmp_path,))
    source = catalog.refresh().sources[0]
    expected = {"change": SourceChangedError, "exception": RuntimeError, "timeout": TimeoutError}[failure]

    with pytest.raises(expected), catalog.source_snapshot(source) as snapshot:
        assert snapshot.read_bytes() == b"trace"
        if failure == "change":
            trace.write_bytes(b"changed")
            assert snapshot.read_bytes() == b"trace"
        else:
            raise expected("reader failed")
    assert snapshot_directories
    assert all(not path.exists() for path in snapshot_directories)


def test_snapshot_is_detached_after_copy(tmp_path, snapshot_directories):
    trace = tmp_path / "robot.log"
    trace.write_bytes(b"accepted version")
    catalog = SourceCatalog((tmp_path,))
    source = catalog.refresh().sources[0]
    with catalog.source_snapshot(source) as snapshot:
        trace.unlink()
        trace.symlink_to("/path/outside/trace/root")
        assert snapshot.read_bytes() == b"accepted version"
    assert all(not path.exists() for path in snapshot_directories)


def test_ctf_snapshot_enforces_total_byte_budget(tmp_path, snapshot_directories):
    trace = tmp_path / "session"
    trace.mkdir()
    (trace / "metadata").write_bytes(b"ctf")
    (trace / "stream").write_bytes(b"1")
    catalog = SourceCatalog((tmp_path,), max_source_bytes=4)
    source = catalog.refresh().sources[0]
    with catalog.source_snapshot(source) as copied:
        assert sum(path.stat().st_size for path in copied.iterdir()) == 4

    (trace / "stream").write_bytes(b"12")
    with pytest.raises(UnsafeSourceError, match="byte limit"), catalog.source_snapshot(source):
        pytest.fail("CTF files must share one source byte budget")

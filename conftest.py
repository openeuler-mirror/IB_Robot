"""Workspace-wide pytest configuration.

Allocates a unique ``ROS_DOMAIN_ID`` per pytest process so that tests which
create real ROS nodes cannot see each other's traffic.

``colcon test`` runs one ``pytest`` process per package, and runs packages in
parallel. Without isolation every one of them joins DDS domain 0, discovers the
others' publishers and services, and the suites become order-dependent and
flaky. ``ament_cmake_ros`` solves this for CMake packages with
``run_test_isolated.py``, which wraps the test command in
``domain_coordinator.domain_id()``. There is no equivalent wrapper for
``ament_python`` packages, so the same ``domain_coordinator`` reservation is
applied here, at the workspace root, where every ``pytest`` invocation that
loads this conftest passes through.

This conftest only loads when pytest's upward conftest search reaches the
workspace root. A package-local ``pytest.ini`` makes that package directory
the rootdir and truncates the search at that directory, so a package with
its own ini must load this isolation logic explicitly from a package-local
``conftest.py`` (``confcutdir`` is a command-line-only option and cannot be
set from an ini), or its node tests silently lose domain isolation under
the standard ``colcon test`` entry.

``domain_coordinator.domain_id()`` reserves an ID by binding TCP port
``32768 + id``. The reservation is held for as long as the socket is open, so
the context must stay open for the whole session; it is closed in
``pytest_unconfigure``.

Escape hatches match ``run_test_isolated.py``:

* ``ROS_DOMAIN_ID`` already set - respected, no new ID is allocated.
* ``DISABLE_ROS_ISOLATION`` set - no ID is allocated and no variable is set.
  Tests that require isolation then fail closed rather than silently sharing
  domain 0.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from collections import Counter

import pytest

_ISOLATION_STACK: contextlib.ExitStack | None = None

#: Tests assert against this instead of reading ``ROS_DOMAIN_ID`` directly, so
#: that "an ID was deliberately allocated for this test process" is
#: distinguishable from "the developer happened to export ``ROS_DOMAIN_ID``".
DOMAIN_ID_ENV = "IBROBOT_TEST_ROS_DOMAIN_ID"


def pytest_configure(config):
    """Reserve a ROS domain for this pytest process before collection starts."""
    global _ISOLATION_STACK

    config.addinivalue_line(
        "markers",
        "model_bundle(name): skip unless the named bundle exists under models/",
    )
    config.addinivalue_line(
        "markers",
        "repo_asset(path): skip unless the workspace-relative path exists",
    )
    # robot-skill hands its ROS transport to a running resident per-user daemon
    # (and starts one when IBROBOT_ROBOT_SKILL_DAEMON=spawn), which outlives the
    # command. Under test that daemon would outlive the
    # pytest process and the reserved domain, so tests that run the real CLI use
    # the in-process bridge; the daemon is covered by its own unit tests.
    # Forced, not defaulted: a developer's exported value must not leak a daemon.
    os.environ["IBROBOT_ROBOT_SKILL_DAEMON"] = "0"

    if "DISABLE_ROS_ISOLATION" in os.environ:
        return

    stack = contextlib.ExitStack()
    existing = os.environ.get("ROS_DOMAIN_ID")
    if existing:
        # Respect an explicitly chosen domain, the way run_test_isolated.py does.
        # Nothing is reserved, so the caller owns the collision risk.
        domain_id = existing
    else:
        try:
            from domain_coordinator import domain_id as _reserve_domain_id
        except ImportError:
            # No ROS environment available. Leave the variables unset so that
            # tests needing a real ROS graph fail with a clear message instead
            # of quietly running on the default domain.
            stack.close()
            return
        domain_id = str(stack.enter_context(_reserve_domain_id()))
        os.environ["ROS_DOMAIN_ID"] = domain_id

    os.environ[DOMAIN_ID_ENV] = domain_id
    # Keep DDS discovery on the loopback interface: test nodes must not reach a
    # robot, a simulator, or a colleague's machine on the same network. This is
    # an override, not a default: the ROS environment exports
    # ROS_LOCALHOST_ONLY=0, and a unique domain alone does not stop multicast
    # discovery from leaving the host.
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    _ISOLATION_STACK = stack


def pytest_unconfigure(config):
    """Release the reserved domain so another process can take it."""
    global _ISOLATION_STACK

    if _ISOLATION_STACK is not None:
        _ISOLATION_STACK.close()
        _ISOLATION_STACK = None


# ---------------------------------------------------------------------------
# Locally provisioned asset availability
# ---------------------------------------------------------------------------
#
# Several suites drive real robot configs whose perception services reference
# model bundles under models/. Those bundles are gitignored and fetched by
# scripts/download_models.py, so on a fresh clone they are simply absent and the
# tests failed with "bundle_path does not exist". A missing download is not a
# defect, and reporting it as a failure trains people to ignore red.
#
# Mark such a test with:
#     @pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
# and it is skipped when the bundle is not present, reported normally when it is.
#
# What stays enforced: robot configs may only reference bundles that
# scripts/download_models.py knows how to fetch. That check needs no assets and
# lives in src/robot_config/test/test_model_bundle_availability.py, so a typo or
# an unfetchable bundle is still a failure, not a skip.
#
# Recorded captures under outputs/ are gitignored for the same reason and get the
# same treatment through @pytest.mark.repo_asset("<workspace-relative path>").

_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent
_MODELS_ROOT = _WORKSPACE_ROOT / "models"


def model_bundle_available(name: str) -> bool:
    """Return whether a downloaded model bundle is present in this workspace."""
    return (_MODELS_ROOT / name).is_dir()


def repo_asset_available(relative_path: str) -> bool:
    """Return whether a gitignored, locally produced asset is present."""
    return (_WORKSPACE_ROOT / relative_path).exists()


def pytest_collection_modifyitems(config, items):
    """Skip tests whose required local asset is absent from this workspace."""
    for item in items:
        for marker in item.iter_markers(name="model_bundle"):
            for name in marker.args:
                if not model_bundle_available(name):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=(
                                f"model bundle {name!r} is not downloaded; "
                                f"run scripts/download_models.py to enable this test"
                            )
                        )
                    )
                    break
        for marker in item.iter_markers(name="repo_asset"):
            for relative_path in marker.args:
                if not repo_asset_available(relative_path):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=(
                                f"{relative_path} is absent; it is a gitignored capture "
                                f"produced by a recorded run, not part of the checkout"
                            )
                        )
                    )
                    break


@pytest.fixture
def track_digest_reads(monkeypatch):
    """Count digest/comparison bytes via Path.open(...).read() on selected paths.

    Excludes copy2/OS I/O, other read APIs and unlisted aliases; this is not a
    whole-publication I/O budget. An optional callback mutates after a read.
    """
    original_open = pathlib.Path.open

    def track(paths, *, after_read=None):
        counts = Counter()

        class CountedReader:
            def __init__(self, path, stream):
                self.path = path
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def read(self, size=-1):
                data = self.stream.read(size)
                counts[self.path] += len(data)
                if after_read is not None:
                    after_read(self.path, data)
                return data

        def open_file(path, mode="r", *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            return CountedReader(path, stream) if mode == "rb" and path in paths else stream

        monkeypatch.setattr(pathlib.Path, "open", open_file)
        return counts

    return track

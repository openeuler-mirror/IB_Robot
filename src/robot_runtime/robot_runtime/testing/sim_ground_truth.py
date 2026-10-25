"""Simulation ground truth as a shared test-support module (openspec D14.1).

Tests that compare runtime estimates with the simulator read the model's
world pose through this one platform-neutral helper: it runs the simulator's
own topic echo (``ign topic -e -t /world/<w>/pose/info``) as a child of the
test process and parses the protobuf-text stream into
:class:`GroundTruthSample` values stamped in simulation time. Nothing is
advertised on the ROS graph (D4), so no runtime can observe it.

This is test-support API, not runtime API: importing this module starts no
process, imports no simulator or ``rclpy`` module and creates no ROS entity.
Only :meth:`GroundTruthStream.start` spawns a child; :meth:`GroundTruthStream.close`
terminates it and joins the reader thread.

The reader is chunked (up to 64 KiB per read, not line iteration): the pose
stream is ~230 KB/s and a per-line append onto a growing multi-MB buffer
cannot keep up under GIL contention — the pipe backpressures, ``ign topic -e``
blocks on write and the ground truth freezes at a stale pose (the failure the
navigation e2e hit before this module existed). The chunks are read with
``read1`` on the raw binary pipe — a single underlying read that returns
whatever has arrived — so a ``read(n)`` cannot sit waiting for a full ``n``
bytes and deliver the stream in ~0.3 s batches; a stale-batched ground truth
terminates route segments whole degrees late (measured while bringing up the
D14 gate).

Interpolation never extrapolates: a stamp outside the recorded span, or one
falling across a gap longer than :data:`MAX_GAP_SEC`, returns ``None`` (D14.1).
"""

from __future__ import annotations

import bisect
import codecs
import contextlib
import math
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

#: Two samples further apart than this are never interpolated across (D14.1).
MAX_GAP_SEC = 0.05

_READ_CHUNK = 65536
_BUFFER_LIMIT = 2_000_000
_BUFFER_KEEP = 500_000
_SAMPLE_LIMIT = 60_000

# Protobuf text omits fields set to their default, so every component of the
# position/orientation blocks is optional and absent means zero (w defaults
# to 1: the identity quaternion).
_STAMP_RE = re.compile(r"stamp \{\s*sec: (\d+)\s*nsec: (\d+)")
_POSITION_RE = re.compile(r"position \{\s*(?:x: ([-\d.e+]+))?\s*(?:y: ([-\d.e+]+))?\s*(?:z: ([-\d.e+]+))?\s*\}")
_ORIENTATION_RE = re.compile(
    r"orientation \{\s*(?:x: ([-\d.e+]+))?\s*(?:y: ([-\d.e+]+))?\s*(?:z: ([-\d.e+]+))?\s*(?:w: ([-\d.e+]+))?\s*\}"
)


class UnsupportedGroundTruth(RuntimeError):
    """The platform has no ground-truth implementation (D14.1)."""


def mirror_partition_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """The child environment with the gz partition mirrored across both
    variable names (a copy; ``None`` inherits unchanged).

    Fortress (gz-transport11 — the ``ign`` command line on this install)
    reads only ``IGN_PARTITION`` and silently ignores ``GZ_PARTITION``;
    newer gz reads ``GZ_PARTITION``. A caller that isolates its run sets
    one of the two; whichever is present is mirrored into the other so the
    child joins the intended partition on both stacks. Both set (possibly
    differently) or neither: left untouched — the caller's explicit choice.
    """
    if env is None:
        return None
    child = dict(env)
    ignition = child.get("IGN_PARTITION")
    gz = child.get("GZ_PARTITION")
    if ignition and not gz:
        child["GZ_PARTITION"] = ignition
    elif gz and not ignition:
        child["IGN_PARTITION"] = gz
    return child


def terminate_partition_processes(partition: str) -> None:
    """Terminate every process started under one gz partition value.

    Test-support companion of :func:`mirror_partition_env`: the partition is
    an environment variable (``pkill -f`` cannot see it) inherited by every
    launch child, and ``ros2 launch`` grandchildren escape the launch
    process group — a leaked bridge or simulator then collides with the next
    run. Scans ``/proc`` for the ``GZ_PARTITION=<partition>`` marker (the
    tests set both variables to the same value) and kills exactly that tree,
    SIGTERM first, SIGKILL on survivors.
    """
    marker = f"GZ_PARTITION={partition}".encode()
    for name in ("SIGTERM", "SIGKILL"):
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            matched = []
            for proc in Path("/proc").iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    if marker in (proc / "environ").read_bytes().split(b"\0"):
                        matched.append(int(proc.name))
                except OSError:
                    continue
            if not matched:
                return
            for pid in matched:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, getattr(signal, name))
            time.sleep(0.5)


class GroundTruthSample:
    """One model pose, stamped in simulation time by the pose message."""

    __slots__ = ("stamp_sec", "x", "y", "z", "yaw")

    def __init__(self, stamp_sec: float, x: float, y: float, z: float, yaw: float):
        self.stamp_sec = float(stamp_sec)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = float(yaw)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GroundTruthSample(stamp_sec={self.stamp_sec:.3f}, x={self.x:.3f}, y={self.y:.3f}, "
            f"z={self.z:.3f}, yaw={self.yaw:.3f})"
        )

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.stamp_sec, self.x, self.y, self.z, self.yaw)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GroundTruthSample):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Planar yaw of a quaternion (the only angle a planar gate can use)."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi] (the shortest signed difference between two yaws)."""
    return math.atan2(math.sin(angle), math.cos(angle))


def parse_world_pose_messages(buffer: str, model: str, *, final: bool = False) -> tuple[list[GroundTruthSample], str]:
    """Parse complete pose messages from buffered ``ign topic -e`` text.

    The stream is ``gz.msgs.Pose_V`` in protobuf text: one ``header`` carrying
    the simulation-time stamp, then ``pose {`` entries each with ``name``, an
    optional ``id`` and ``position``/``orientation`` blocks. One sample is
    produced per message that carries ``model``; the fields are taken between
    the model's name and the next ``pose {`` so other models in the same
    message cannot leak into the match.

    A message is only known complete once the next one's ``header {`` arrives,
    so the trailing chunk stays in the returned tail — unless ``final`` is set
    (the stream reached EOF: nothing more will arrive, so the trailing chunk
    is the last message and is parsed too).
    """
    samples: list[GroundTruthSample] = []
    chunks = buffer.split("header {")
    complete = chunks[:-1] if not final else chunks
    tail = "" if final else chunks[-1]
    for chunk in complete:
        name_at = chunk.find(f'name: "{model}"')
        if name_at < 0:
            continue
        next_pose = chunk.find("pose {", name_at)
        block = chunk[name_at : next_pose if next_pose > 0 else len(chunk)]
        stamp = _STAMP_RE.search(chunk)
        position = _POSITION_RE.search(block)
        orientation = _ORIENTATION_RE.search(block)
        if not (stamp and position and orientation):
            continue

        def _component(value: str | None, default: float) -> float:
            return float(value) if value is not None else default

        samples.append(
            GroundTruthSample(
                int(stamp.group(1)) + int(stamp.group(2)) / 1e9,
                _component(position.group(1), 0.0),
                _component(position.group(2), 0.0),
                _component(position.group(3), 0.0),
                yaw_from_quaternion(
                    _component(orientation.group(1), 0.0),
                    _component(orientation.group(2), 0.0),
                    _component(orientation.group(3), 0.0),
                    _component(orientation.group(4), 1.0),
                ),
            )
        )
    return samples, tail


class GroundTruthStream:
    """The live world-pose stream of one model (Gazebo implementation).

    ``start()`` spawns ``ign topic -e`` on the world's pose-info topic and a
    reader thread; ``close()`` terminates the child and joins the thread.
    Samples carry the messages' own simulation-time stamps, so no ROS-side
    clock pairing is needed.
    """

    def __init__(self, world: str, model: str, env: dict[str, str] | None = None):
        self._world = str(world)
        self._model = str(model)
        self._env = env
        self._samples: list[GroundTruthSample] = []
        self._stamps: list[float] | None = None  # rebuilt when samples change
        self._last_sample_wall = 0.0  # wall time of the last parsed sample
        self._buffer = ""
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None

    @property
    def world(self) -> str:
        return self._world

    @property
    def model(self) -> str:
        return self._model

    def start(self) -> GroundTruthStream:
        """Spawn the pose echo child and its reader (no-op on restart)."""
        if self._proc is not None:
            return self
        # binary stdout: BufferedReader.read1 returns whatever has arrived
        # in one underlying read (a text-mode read(n) would block until n
        # bytes and batch the stream); the incremental decoder keeps
        # multi-byte characters split across reads intact. The env mirrors
        # the gz partition across IGN_/GZ_PARTITION (Fortress reads only
        # IGN_PARTITION), so the echo joins the caller's run either way.
        self._proc = subprocess.Popen(
            ["ign", "topic", "-e", "-t", f"/world/{self._world}/pose/info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=mirror_partition_env(self._env),
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            chunk = self._proc.stdout.read1(_READ_CHUNK)
            if not chunk:
                # EOF (the child was closed): the withheld trailing message
                # is the last one — parse it too so nothing recorded is lost
                with self._lock:
                    samples, tail = parse_world_pose_messages(self._buffer, self._model, final=True)
                    self._buffer = tail
                    if samples:
                        self._samples.extend(samples)
                        self._stamps = None
                        self._last_sample_wall = time.time()
                return
            with self._lock:
                self._buffer += decoder.decode(chunk)
                samples, tail = parse_world_pose_messages(self._buffer, self._model)
                self._buffer = tail
                if samples:
                    self._samples.extend(samples)
                    # the cache is dropped on every change, not just length
                    # changes: a trim that lands on a previously seen length
                    # must never serve stale stamps
                    self._stamps = None
                    self._last_sample_wall = time.time()
                if len(self._samples) > 2 * _SAMPLE_LIMIT:
                    del self._samples[:-_SAMPLE_LIMIT]
                    self._stamps = None
                if len(self._buffer) > _BUFFER_LIMIT:
                    self._buffer = self._buffer[-_BUFFER_KEEP:]

    def wait_for(self, n: int, timeout: float) -> bool:
        """Wait until ``n`` samples are recorded; False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self._samples) >= n:
                    return True
            time.sleep(0.05)
        with self._lock:
            return len(self._samples) >= n

    def _stamp_index(self) -> list[float]:
        """The cached stamp list of the current samples (lock held by caller)."""
        if self._stamps is None:
            self._stamps = [sample.stamp_sec for sample in self._samples]
        return self._stamps

    def latest_sample_age(self) -> float | None:
        """Seconds since the last parsed sample, or None if none arrived.

        A watchdog can use this to fail a run whose echo child died: a
        live stream refreshes it at the pose-info rate (S2)."""
        with self._lock:
            if not self._samples:
                return None
            return time.time() - self._last_sample_wall

    def samples(self) -> list[GroundTruthSample]:
        """Snapshot of the recorded samples, oldest first."""
        with self._lock:
            return list(self._samples)

    def latest(self) -> GroundTruthSample | None:
        with self._lock:
            return self._samples[-1] if self._samples else None

    def latest_time(self) -> float | None:
        sample = self.latest()
        return sample.stamp_sec if sample is not None else None

    def position(self) -> tuple[float, float] | None:
        sample = self.latest()
        return (sample.x, sample.y) if sample is not None else None

    def window(self, begin: float, end: float) -> list[GroundTruthSample]:
        """Samples with ``begin <= stamp <= end`` (bisected, oldest first)."""
        with self._lock:
            stamps = self._stamp_index()
            return [
                self._samples[i] for i in range(bisect.bisect_left(stamps, begin), bisect.bisect_right(stamps, end))
            ]

    def interpolate(self, stamp_sec: float) -> GroundTruthSample | None:
        """The pose at ``stamp_sec``, or None outside the span or across a gap.

        Linear in position, shortest-arc in yaw; never extrapolates (D14.1).
        """
        with self._lock:
            samples = self._samples
            if not samples or stamp_sec < samples[0].stamp_sec or stamp_sec > samples[-1].stamp_sec:
                return None
            stamps = self._stamp_index()
            index = bisect.bisect_left(stamps, stamp_sec)
            if index < len(samples) and stamps[index] == stamp_sec:
                return samples[index]
            previous, following = samples[index - 1], samples[index]
            if following.stamp_sec - previous.stamp_sec > MAX_GAP_SEC:
                return None
            t = (stamp_sec - previous.stamp_sec) / (following.stamp_sec - previous.stamp_sec)
            return GroundTruthSample(
                stamp_sec,
                previous.x + t * (following.x - previous.x),
                previous.y + t * (following.y - previous.y),
                previous.z + t * (following.z - previous.z),
                previous.yaw + t * wrap_angle(following.yaw - previous.yaw),
            )

    def close(self) -> None:
        """Terminate the echo child and join the reader thread."""
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            if self._proc.stdout is not None:
                self._proc.stdout.close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def open_ground_truth(
    sim_platform: str, *, world: str, model: str, env: dict[str, str] | None = None
) -> GroundTruthStream:
    """The ground-truth stream of ``model`` in ``world`` on ``sim_platform``.

    Gazebo is the only implemented platform; anything else raises
    :class:`UnsupportedGroundTruth`. The stream is returned unstarted: only
    :meth:`GroundTruthStream.start` spawns a process, so a capability probe
    can call this without launching anything (D14.2).
    """
    if sim_platform != "gazebo":
        raise UnsupportedGroundTruth(
            f"no ground-truth implementation for platform {sim_platform!r} (supported: 'gazebo')"
        )
    return GroundTruthStream(world=world, model=model, env=env)

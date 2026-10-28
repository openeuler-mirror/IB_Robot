"""Simulation ground truth as a shared test-support module (openspec D14.1).

Tests that compare runtime estimates with the simulator read the model's
world pose through this one platform-neutral helper. On Gazebo it runs the
simulator's own topic echo (``ign topic -e -t /world/<w>/pose/info``) as a
child of the test process; on MuJoCo it subscribes, on a private ROS context,
to the floating-base state ``mujoco_ros2_control`` publishes for the robot's
free joint. Nothing is advertised by the runtime and no runtime node
consumes it: the MuJoCo stream lives on a simulator-private topic that the
interface description never lists (lekiwi-mujoco-runtime-platform D4 — a
documented relaxation of this module's original "nothing ground-truth on the
ROS graph": ground truth MAY exist on a simulator-private topic, never
consumed or advertised by the runtime).

This is test-support API, not runtime API: importing this module starts no
process, imports no simulator or ``rclpy`` module and creates no ROS entity.
Only :meth:`GroundTruthStream.start` spawns a child or a context; each
stream's :meth:`close` terminates it and joins the reader thread.

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


#: Neutral per-run marker every test launch inherits, on every platform
#: (review round 1): the gz partition value only exists on Gazebo runs, so
#: ``terminate_partition_processes`` never swept a MuJoCo launch's escaped
#: descendants — the sweep marker was missing exactly where the simulators
#: differ most. Tests that launch a simulator set this to a unique per-run
#: value in the child environment and sweep it in teardown.
RUN_MARKER_ENV = "IBROBOT_TEST_RUN"


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


def terminate_marked_processes(markers: dict[str, str]) -> None:
    """Terminate every process whose environment carries any marker pair.

    Generic form of the gz partition sweep (test-support companion of
    :func:`mirror_partition_env`): a launch marker is an environment
    variable (``pkill -f`` cannot see it) inherited by every launch child,
    and ``ros2 launch`` grandchildren escape the launch process group — a
    leaked bridge, simulator or runtime node then collides with the next
    run. Scans ``/proc`` for processes whose ``environ`` carries any
    ``{variable: value}`` pair from ``markers`` and kills exactly that set,
    SIGTERM first, SIGKILL on survivors.

    ``markers`` should list every per-run marker the launcher set — e.g. the
    neutral ``RUN_MARKER_ENV`` plus ``GZ_PARTITION`` (same run value), so
    one call sweeps both a Gazebo run's gz bus escapees and a MuJoCo run's
    descendants.
    """
    wanted = {f"{variable}={value}".encode() for variable, value in markers.items()}
    for name in ("SIGTERM", "SIGKILL"):
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            matched = []
            for proc in Path("/proc").iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    environ = (proc / "environ").read_bytes().split(b"\0")
                except OSError:
                    continue
                if any(marker in environ for marker in wanted):
                    matched.append(int(proc.name))
            if not matched:
                return
            for pid in matched:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, getattr(signal, name))
            time.sleep(0.5)


def terminate_partition_processes(partition: str) -> None:
    """Terminate every process started under one gz partition value.

    Test-support companion of :func:`mirror_partition_env`: the partition is
    an environment variable (``pkill -f`` cannot see it) inherited by every
    launch child, and ``ros2 launch`` grandchildren escape the launch
    process group — a leaked bridge or simulator then collides with the next
    run. Scans ``/proc`` for the ``GZ_PARTITION=<partition>`` marker (the
    tests set both variables to the same value) and kills exactly that tree,
    SIGTERM first, SIGKILL on survivors. Runs that also set the neutral
    ``RUN_MARKER_ENV`` should sweep it via :func:`terminate_marked_processes`
    instead — Gazebo's partition sweep alone never matched a MuJoCo run
    (review round 1).
    """
    terminate_marked_processes({"GZ_PARTITION": partition})


def _read_ppid(pid: int) -> int:
    """The PPid of a /proc entry, or 0 when unreadable/malformed."""
    try:
        status = (Path("/proc") / str(pid) / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for line in status.splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return 0


def ancestor_pids(start_pid: int | None = None, *, read_ppid=None) -> set[int]:
    """The calling process's pid plus every ancestor, INCLUDING PID 1.

    The scan's own invocation chain — pytest, its colcon parent, the
    wrapper shells, init — legitimately carries the ROS domain (a wrapper
    under an exported ROS_DOMAIN_ID is the scanning process's ANCESTOR,
    not a leftover; review round 1 flagged colcon itself). Review round 2:
    the walk adds each pid before stopping, PID 1 included — it is an
    ancestor like any other. ``read_ppid`` is injectable so the walk can be
    unit-tested with a mocked chain.
    """
    if read_ppid is None:
        read_ppid = _read_ppid
    excluded: set[int] = set()
    pid = os.getpid() if start_pid is None else start_pid
    while pid and pid not in excluded:
        excluded.add(pid)
        if pid == 1:
            break
        pid = read_ppid(pid)
    return excluded


def leftover_domain_processes(domain: str, *, exclude_pids: frozenset[int] | set[int] = frozenset()) -> list[str]:
    """Processes still carrying this ROS domain in their environment.

    Passive diagnostics for the gate's two-legs-one-domain layout (review:
    run-2 startup stall): both legs of a parametrized gate share one pytest
    process and therefore one ``ROS_DOMAIN_ID``, so a leftover from the
    previous leg — or a foreign runtime squattering the domain — is
    invisible to checks that only run between whole test invocations. The
    caller prints the result after its marker sweep: whatever still carries
    the domain then is either a leak the sweep cannot see (a process that
    dropped its inherited environment) or a foreign process. Never asserted.
    """
    if not domain:
        return []
    excluded = ancestor_pids() | set(exclude_pids)
    wanted = f"ROS_DOMAIN_ID={domain}".encode()
    leftovers = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) in excluded:
            continue
        try:
            environ = (proc / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if wanted in environ:
            try:
                cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
            except OSError:
                cmd = "<unreadable>"
            leftovers.append(f"pid {proc.name}: {cmd}")
    return leftovers


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

    Gazebo echoes the world's pose-info topic; MuJoCo subscribes to the
    floating-base state the simulator publishes for the robot's free joint
    (D9). Anything else raises :class:`UnsupportedGroundTruth`. The stream is
    returned unstarted: only :meth:`GroundTruthStream.start` spawns a process
    or a context, so a capability probe can call this without launching
    anything (D14.2).
    """
    if sim_platform == "gazebo":
        return GroundTruthStream(world=world, model=model, env=env)
    if sim_platform == "mujoco":
        return MujocoGroundTruthStream(world=world, model=model, env=env)
    raise UnsupportedGroundTruth(
        f"no ground-truth implementation for platform {sim_platform!r} (supported: 'gazebo', 'mujoco')"
    )


#: The free-joint body each tracked model publishes as its floating-base
#: ``child_frame_id`` on MuJoCo (D9): the gates pass their simulator-neutral
#: model name and the stream checks the incoming messages against the body.
_MUJOCO_GROUND_TRUTH_BODIES = {"lekiwi_robot": "base_link"}


def mujoco_ground_truth_body(model: str) -> str:
    """The MJCF free-joint body a model name publishes on MuJoCo (D9)."""
    return _MUJOCO_GROUND_TRUTH_BODIES.get(model, model)


class MujocoGroundTruthStream(GroundTruthStream):
    """The floating-base state stream of one model (MuJoCo implementation).

    ``mujoco_ros2_control`` publishes the free joint's world pose as
    ``nav_msgs/Odometry`` on a simulator-namespaced topic — never advertised
    by the runtime, never consumed by a runtime node (D4/D9). The stream
    subscribes on a private ROS context (the domain taken from ``env``), so
    it can observe a launch running on a domain of the test's choosing
    without touching the test's own context. Only the pose is used (the
    message's world-frame twist is not a body-frame report).
    """

    def __init__(self, world: str, model: str, env: dict[str, str] | None = None):
        super().__init__(world=world, model=mujoco_ground_truth_body(model), env=env)
        self._context = None
        self._node = None
        self._executor = None
        self._thread = None

    def start(self) -> MujocoGroundTruthStream:
        """Create the private context, node and executor thread (no-op on restart)."""
        if self._node is not None:
            return self
        import rclpy  # noqa: PLC0415  (test-support module: no ROS import at module load)
        from nav_msgs.msg import Odometry  # noqa: PLC0415

        from robot_runtime.mujoco import FLOATING_BASE_STATE_TOPIC

        # The rmw layer reads ROS_DOMAIN_ID at context creation: give the
        # private context the LAUNCH's domain (the env dict the caller
        # captured), restoring the process environment immediately after.
        domain = str((self._env or {}).get("ROS_DOMAIN_ID") or os.environ.get("ROS_DOMAIN_ID", "0"))
        previous = os.environ.get("ROS_DOMAIN_ID")
        os.environ["ROS_DOMAIN_ID"] = domain
        try:
            self._context = rclpy.Context()
            rclpy.init(args=[], context=self._context)
        finally:
            if previous is None:
                os.environ.pop("ROS_DOMAIN_ID", None)
            else:
                os.environ["ROS_DOMAIN_ID"] = previous
        self._node = rclpy.create_node(
            "mujoco_ground_truth",
            context=self._context,
            parameter_overrides=[rclpy.parameter.Parameter("use_sim_time", value=True)],
        )
        self._node.create_subscription(
            Odometry,
            FLOATING_BASE_STATE_TOPIC,
            self._on_odom,
            100,
        )
        self._executor = rclpy.executors.SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()
        return self

    def _on_odom(self, message) -> None:
        if str(message.child_frame_id or "") != self._model:
            return
        pose = message.pose.pose
        sample = GroundTruthSample(
            message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
            pose.position.x,
            pose.position.y,
            pose.position.z,
            yaw_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
        )
        with self._lock:
            # A reset_world restores the initial state AND the clock, so the
            # next sample's stamp goes backwards. window()/interpolate()
            # bisect a strictly ordered span, so the stream starts a NEW
            # epoch instead of appending (the pre-reset history is dropped;
            # cross-epoch interpolation is refused by construction — review
            # round 1). An equal stamp starts the epoch too: the bisected
            # span must stay strictly increasing.
            if self._samples and sample.stamp_sec <= self._samples[-1].stamp_sec:
                self._samples = []
            self._samples.append(sample)
            # the same amortised cap as the Gazebo reader (the ROS stream
            # has no reader thread to trim it)
            if len(self._samples) > 2 * _SAMPLE_LIMIT:
                del self._samples[:-_SAMPLE_LIMIT]
            self._stamps = None
            self._last_sample_wall = time.time()

    def close(self) -> None:
        """Shut the executor and the private context down and join the thread."""
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._context is not None:
            try:
                self._context.shutdown()
                self._context.try_destroy()
            except Exception:  # noqa: BLE001  (a torn-down context is success here)
                pass
            self._context = None

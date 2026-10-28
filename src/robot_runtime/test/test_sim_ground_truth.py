"""The shared ground-truth module (openspec lekiwi-gazebo-runtime-platform D14.1).

Recorded ``ign topic -e -t /world/<w>/pose/info`` text drives the parser:
several models per message (no leaking between them), a message split across
reads, yaw from the quaternion, message stamps in simulation time. The
interpolation rules (inside span, gap > 50 ms, outside span, shortest-arc
yaw) and the unsupported-platform error are checked directly. Importing the
module must pull in neither ``rclpy`` nor a simulator module (D14.1).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
import time

import pytest

from robot_runtime.testing.sim_ground_truth import (
    MAX_GAP_SEC,
    RUN_MARKER_ENV,
    GroundTruthSample,
    UnsupportedGroundTruth,
    ancestor_pids,
    leftover_domain_processes,
    mirror_partition_env,
    open_ground_truth,
    parse_world_pose_messages,
    terminate_marked_processes,
    terminate_partition_processes,
    yaw_from_quaternion,
)

# Recorded from a live Fortress echo of /world/empty/pose/info (protobuf
# text omits zero fields; one header per message, one pose entry per model).
# Real messages arrive ~17 ms apart, which the interpolation tests mirror:
# the >50 ms gap rule must not silently turn every sample pair into a hole.
RECORDED = textwrap.dedent(
    """\
    header {
      stamp {
        sec: 3
        nsec: 989000000
      }
    }
    pose {
      name: "ground_plane"
      id: 4
      position {
      }
      orientation {
        w: 1
      }
    }
    pose {
      name: "lekiwi_robot"
      id: 6
      position {
        x: -1.5
        y: -0.25
        z: 0.05
      }
      orientation {
        z: 0.3826834323650898
        w: 0.9238795325112867
      }
    }
    pose {
      name: "sun"
      id: 8
      position {
        z: 10
      }
      orientation {
        w: 1
      }
    }

    header {
      stamp {
        sec: 4
        nsec: 6000000
      }
    }
    pose {
      name: "lekiwi_robot"
      id: 6
      position {
        x: -1.4
      }
      orientation {
        y: -0.1
        z: 0.7
        w: 0.6
      }
    }

    """
)

# The yaw of "lekiwi_robot" at sec 3: quaternion (0, 0, sin(11.25 deg), cos(11.25 deg))
FIRST_YAW = yaw_from_quaternion(0.0, 0.0, 0.3826834323650898, 0.9238795325112867)


def test_recorded_text_yields_one_sample_per_message_of_the_model():
    # a message is only known complete once the next one starts: without
    # final, the second message waits in the tail
    samples, tail = parse_world_pose_messages(RECORDED, "lekiwi_robot")
    assert [sample.stamp_sec for sample in samples] == [3.989]
    assert "sec: 4" in tail
    # at EOF the trailing message is the last one and is parsed too
    samples, tail = parse_world_pose_messages(RECORDED, "lekiwi_robot", final=True)
    assert [sample.stamp_sec for sample in samples] == [3.989, 4.006]
    assert tail == ""
    first = samples[0]
    assert (first.x, first.y, first.z) == (-1.5, -0.25, 0.05)
    assert first.yaw == pytest.approx(FIRST_YAW)
    # the second message carries the default-zero spelling: absent y/z stay 0
    # and the yaw follows its full quaternion
    second = samples[1]
    assert (second.x, second.y, second.z) == (-1.4, 0.0, 0.0)
    assert second.yaw == pytest.approx(yaw_from_quaternion(0.0, -0.1, 0.7, 0.6))


def test_a_message_split_across_reads_is_reassembled():
    cut = RECORDED.index('name: "lekiwi_robot"')
    samples, tail = parse_world_pose_messages(RECORDED[:cut], "lekiwi_robot")
    assert samples == []  # the model's entry is still incomplete
    samples, tail = parse_world_pose_messages(tail + RECORDED[cut:], "lekiwi_robot", final=True)
    assert [sample.stamp_sec for sample in samples] == [3.989, 4.006]
    # a split inside the "header {" keyword itself is preserved the same way
    split = RECORDED.index("header {", 10) + len("head")
    samples, tail = parse_world_pose_messages(RECORDED[:split], "lekiwi_robot")
    assert samples == []
    samples, tail = parse_world_pose_messages(tail + RECORDED[split:], "lekiwi_robot", final=True)
    assert [sample.stamp_sec for sample in samples] == [3.989, 4.006]


def test_other_models_do_not_leak_into_the_sample():
    # ground_plane precedes and sun follows the model in the same message:
    # the parsed pose is exactly the model's own block
    samples, _ = parse_world_pose_messages(RECORDED, "lekiwi_robot", final=True)
    assert (samples[0].x, samples[0].y, samples[0].z) == (-1.5, -0.25, 0.05)
    # a model that is the last entry of a message (no following "pose {")
    first_message = RECORDED[: RECORDED.index("\nheader {", 10)].rstrip() + "\n"
    samples, _ = parse_world_pose_messages(first_message, "sun", final=True)
    assert [(sample.x, sample.y, sample.z) for sample in samples] == [(0.0, 0.0, 10.0)]
    # a model absent from the stream yields nothing
    samples, _ = parse_world_pose_messages(RECORDED, "unknown_model", final=True)
    assert samples == []


def test_message_stamps_carry_simulation_time():
    samples, _ = parse_world_pose_messages(RECORDED, "lekiwi_robot", final=True)
    assert samples[0].stamp_sec == pytest.approx(3.989)
    assert samples[1].stamp_sec == pytest.approx(4.006)
    # nsec is a fraction, not a milliseconds value
    text = RECORDED.replace("nsec: 6000000", "nsec: 987654321")
    samples, _ = parse_world_pose_messages(text, "lekiwi_robot", final=True)
    assert samples[1].stamp_sec == pytest.approx(4.987654321)


def test_yaw_from_quaternion_covers_the_full_circle():
    zero = yaw_from_quaternion(0.0, 0.0, 0.0, 1.0)
    assert zero == pytest.approx(0.0)
    quarter = yaw_from_quaternion(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert quarter == pytest.approx(math.pi / 2)
    back = yaw_from_quaternion(0.0, 0.0, -math.sin(math.pi / 4), math.cos(math.pi / 4))
    assert back == pytest.approx(-math.pi / 2)
    # 180 degrees: either quaternion spelling lands on pi (atan2 branch)
    assert abs(yaw_from_quaternion(0.0, 0.0, 1.0, 0.0)) == pytest.approx(math.pi)


def _stream_with(samples):
    stream = open_ground_truth("gazebo", world="demo", model="lekiwi_robot")
    stream._samples = list(samples)
    return stream


def test_interpolate_inside_the_span_is_linear_in_position():
    # 20 ms apart, the real pose-info cadence: pairs interpolate
    stream = _stream_with(
        [
            GroundTruthSample(1.0, 0.0, 0.0, 0.0, 0.0),
            GroundTruthSample(1.02, 1.0, 0.5, 0.1, 0.2),
        ]
    )
    pose = stream.interpolate(1.01)
    assert pose is not None
    assert (pose.x, pose.y, pose.z) == (0.5, 0.25, 0.05)
    assert pose.yaw == pytest.approx(0.1)
    # an exact sample stamp returns that sample verbatim
    assert stream.interpolate(1.02) == stream.samples()[1]


def test_interpolate_takes_the_shortest_arc_in_yaw():
    stream = _stream_with(
        [
            GroundTruthSample(1.0, 0.0, 0.0, 0.0, 3.0),
            GroundTruthSample(1.02, 0.0, 0.0, 0.0, -3.0),
        ]
    )
    pose = stream.interpolate(1.01)
    assert pose is not None
    # 3.0 rad -> -3.0 rad: the shortest arc runs through pi, not through 0
    assert pose.yaw == pytest.approx(math.pi)
    # a wrapped continuation interpolates forward through the branch cut
    stream = _stream_with(
        [
            GroundTruthSample(1.0, 0.0, 0.0, 0.0, 3.1),
            GroundTruthSample(1.02, 0.0, 0.0, 0.0, -3.1),
        ]
    )
    pose = stream.interpolate(1.01)
    assert pose is not None
    assert pose.yaw == pytest.approx(math.pi)


def test_interpolate_returns_none_outside_the_span_or_across_a_gap():
    stream = _stream_with(
        [
            GroundTruthSample(1.0, 0.0, 0.0, 0.0, 0.0),
            GroundTruthSample(1.4, 1.0, 0.0, 0.0, 0.0),
            GroundTruthSample(1.42, 1.0, 0.0, 0.0, 0.0),
        ]
    )
    # the 0.4 s hole between the first two samples is never interpolated across
    assert stream.interpolate(1.2) is None
    # the 20 ms pair interpolates
    assert stream.interpolate(1.41) is not None
    # the gap boundary, clear of float rounding at exactly MAX_GAP_SEC:
    # just under the limit the pair interpolates, just above it refuses
    stream = _stream_with(
        [
            GroundTruthSample(1.0, 0.0, 0.0, 0.0, 0.0),
            GroundTruthSample(1.0 + MAX_GAP_SEC + 0.01, 1.0, 0.0, 0.0, 0.0),
        ]
    )
    assert stream.interpolate(1.0 + MAX_GAP_SEC / 2) is None
    stream = _stream_with(
        [
            GroundTruthSample(1.0, 0.0, 0.0, 0.0, 0.0),
            GroundTruthSample(1.0 + MAX_GAP_SEC - 0.01, 1.0, 0.0, 0.0, 0.0),
        ]
    )
    assert stream.interpolate(1.0 + MAX_GAP_SEC / 2 - 0.005) is not None
    # outside the recorded span: never extrapolated
    assert stream.interpolate(0.5) is None
    assert stream.interpolate(2.0) is None
    # an empty stream has nothing to say
    assert _stream_with([]).interpolate(1.0) is None


def test_unsupported_platform_is_named_and_both_platforms_open_without_starting():
    with pytest.raises(UnsupportedGroundTruth, match="teleport"):
        open_ground_truth("teleport", world="demo", model="lekiwi_robot")
    # both implemented platforms open unstarted: no process, no context, no node
    gazebo = open_ground_truth("gazebo", world="demo", model="lekiwi_robot")
    assert gazebo.world == "demo" and gazebo.model == "lekiwi_robot"
    assert gazebo._proc is None and gazebo._thread is None
    gazebo.close()
    mujoco = open_ground_truth("mujoco", world="demo", model="lekiwi_robot")
    assert mujoco.model == "base_link", "the gate's model name maps to the free-joint body (D9)"
    assert mujoco._context is None and mujoco._node is None and mujoco._thread is None
    # close() on a never-started stream terminates nothing
    mujoco.close()


def test_mujoco_stream_reads_a_fake_floating_base_publisher():
    """D9: a private-domain fake publisher feeds the stream; close() joins.

    Uses the same mechanics the gates do: the stream's env dict names the
    domain of the observed launch (here: this test's own private domain), the
    publisher stamps simulation time and the free-joint body as the
    child_frame_id, and only matching messages become samples.
    """
    import rclpy
    from nav_msgs.msg import Odometry

    from robot_runtime.testing.sim_ground_truth import MujocoGroundTruthStream

    domain = 215
    previous = os.environ.get("ROS_DOMAIN_ID")
    os.environ["ROS_DOMAIN_ID"] = str(domain)
    try:
        rclpy.init()
        publisher_node = rclpy.create_node("fake_floating_base")
        publisher = publisher_node.create_publisher(Odometry, "/simulator/floating_base_state", 10)

        stream = MujocoGroundTruthStream(world="demo", model="lekiwi_robot", env={"ROS_DOMAIN_ID": str(domain)})
        stream.start()

        def publish(stamp, x, yaw):
            message = Odometry()
            message.header.stamp.sec = int(stamp)
            message.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
            message.child_frame_id = "base_link"
            message.pose.pose.position.x = x
            message.pose.pose.orientation.z = math.sin(yaw / 2.0)
            message.pose.pose.orientation.w = math.cos(yaw / 2.0)
            publisher.publish(message)

        # a foreign child frame must not leak into the samples
        foreign = Odometry()
        foreign.child_frame_id = "some_other_body"
        foreign.pose.pose.position.x = 99.0
        publisher.publish(foreign)
        # strictly increasing simulation-time stamps, as the real
        # floating-base publisher emits them (a backwards stamp would start a
        # new epoch, see test_mujoco_stream_starts_a_new_epoch_on_a_backwards_stamp)
        for index in range(20):
            publish(1.0 + index * 0.04, 1.0, 0.0)
            publish(1.02 + index * 0.04, 2.0, 0.2)
            time.sleep(0.01)
        assert stream.wait_for(10, 10.0), f"no samples arrived (got {len(stream.samples())})"
        samples = stream.samples()
        assert all(s.x in (1.0, 2.0) for s in samples), "the foreign body leaked in"
        # the recorded stamps are the messages' own simulation-time stamps
        stamps = [s.stamp_sec for s in samples]
        assert stamps == sorted(set(stamps))
        # interpolation between the first RECORDED pose pair (discovery may
        # have dropped the earliest published messages)
        middle = stream.interpolate((stamps[0] + stamps[1]) / 2)
        assert middle is not None and abs(middle.x - 1.5) < 1e-6 and abs(middle.yaw - 0.1) < 1e-6
        assert stream.position() is not None
        stream.close()
        publisher_node.destroy_node()
        rclpy.shutdown()
    finally:
        if previous is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = previous


def _floating_base_message(stamp: float, x: float, yaw: float):
    from nav_msgs.msg import Odometry

    message = Odometry()
    message.header.stamp.sec = int(stamp)
    message.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
    message.child_frame_id = "base_link"
    message.pose.pose.position.x = x
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return message


def test_mujoco_stream_starts_a_new_epoch_on_a_backwards_stamp():
    """Review round 1: a reset_world jumps the stamps back; the bisected
    window/interpolate assume a strictly ordered span, so the pre-reset
    history is dropped and cross-epoch interpolation is refused."""
    from robot_runtime.testing.sim_ground_truth import MujocoGroundTruthStream

    stream = MujocoGroundTruthStream(world="demo", model="lekiwi_robot", env={"ROS_DOMAIN_ID": "0"})
    stream._on_odom(_floating_base_message(10.0, 1.0, 0.0))
    stream._on_odom(_floating_base_message(10.04, 2.0, 0.2))
    assert stream.interpolate(10.02).x == pytest.approx(1.5)
    # the reset: the clock and the pose jump back
    stream._on_odom(_floating_base_message(0.5, -1.5, 0.0))
    assert [s.stamp_sec for s in stream.samples()] == [0.5]
    # the old epoch's span is gone: no window, no interpolation into it
    assert stream.window(9.0, 11.0) == []
    assert stream.interpolate(10.02) is None
    assert stream.interpolate(0.75) is None  # outside the new (single-sample) span
    # the new epoch accumulates and interpolates normally
    stream._on_odom(_floating_base_message(0.54, -1.0, 0.0))
    middle = stream.interpolate(0.52)
    assert middle is not None and middle.x == pytest.approx(-1.25)
    assert [s.stamp_sec for s in stream.window(0.5, 0.54)] == [0.5, 0.54]
    # an EQUAL stamp also starts a new epoch (the span stays strictly increasing)
    stream._on_odom(_floating_base_message(0.54, -1.0, 0.0))
    assert [s.stamp_sec for s in stream.samples()] == [0.54]


def test_mujoco_stream_caps_its_history_like_the_gazebo_reader():
    from robot_runtime.testing import sim_ground_truth

    stream = sim_ground_truth.MujocoGroundTruthStream(world="demo", model="lekiwi_robot", env={})
    limit = sim_ground_truth._SAMPLE_LIMIT
    for index in range(2 * limit + 10):
        stream._on_odom(_floating_base_message(index * 0.02, float(index), 0.0))
    # the amortised Gazebo cap: the trim fires past 2x and keeps the newest
    # limit samples (here: 60000 trimmed + the 9 appends after the trim)
    assert len(stream.samples()) == limit + 9
    # the newest samples survived
    stamps = [s.stamp_sec for s in stream.samples()]
    assert stamps[-1] == pytest.approx((2 * limit + 9) * 0.02)


def test_the_child_env_mirrors_the_gz_partition_across_both_names():
    """Fortress reads only IGN_PARTITION; newer gz reads GZ_PARTITION — a
    caller setting one gets the other mirrored so the echo child joins the
    intended partition on both stacks."""
    gz_only = {"GZ_PARTITION": "lekiwi_run_a"}
    mirrored = mirror_partition_env(gz_only)
    assert mirrored == {"GZ_PARTITION": "lekiwi_run_a", "IGN_PARTITION": "lekiwi_run_a"}
    assert gz_only == {"GZ_PARTITION": "lekiwi_run_a"}, "the caller's env is not mutated"

    ignition_only = {"IGN_PARTITION": "lekiwi_run_b", "PATH": "/bin"}
    mirrored = mirror_partition_env(ignition_only)
    assert mirrored == {"IGN_PARTITION": "lekiwi_run_b", "GZ_PARTITION": "lekiwi_run_b", "PATH": "/bin"}


def test_the_marker_sweep_kills_only_marked_processes():
    """Review round 1: the neutral IBROBOT_TEST_RUN marker sweep must kill a
    process carrying the marker (a launch child that escaped the process
    group) and must leave every unmarked process alone — this is what makes a
    MuJoCo teardown sweep possible where the Gazebo-only partition marker
    never matched."""
    run = f"lekiwi_sweep_test_{os.getpid()}"
    marked_env = {**os.environ, RUN_MARKER_ENV: run}
    sleeper = "/bin/sleep"
    marked = subprocess.Popen([sleeper, "30"], env=marked_env)
    unmarked = subprocess.Popen([sleeper, "30"], env=os.environ)
    try:
        assert marked.poll() is None and unmarked.poll() is None
        terminate_marked_processes({RUN_MARKER_ENV: run})
        assert marked.poll() is not None, "the marked process survived the SIGTERM/SIGKILL sweep"
        assert unmarked.poll() is None, "the sweep killed a process without the marker"
    finally:
        for process in (marked, unmarked):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            else:
                process.wait(timeout=5)


def test_leftover_domain_processes_list_only_processes_on_that_domain():
    """The between-legs diagnostic: a process whose environment carries the
    domain is listed (pid + command), one on another domain or none is not."""
    domain = f"98_{os.getpid()}"  # never a real ROS domain: diagnostics only
    on_domain_env = {**os.environ, "ROS_DOMAIN_ID": domain}
    on_domain = subprocess.Popen(["/bin/sleep", "30"], env=on_domain_env)
    off_domain = subprocess.Popen(["/bin/sleep", "30"], env=os.environ)
    try:
        listed = leftover_domain_processes(domain)
        assert len(listed) == 1, f"expected exactly the on-domain process, got {listed}"
        assert f"pid {on_domain.pid}" in listed[0], listed
        assert "sleep 30" in listed[0], listed
        assert leftover_domain_processes("") == [], "an empty domain scans nothing"
    finally:
        for process in (on_domain, off_domain):
            process.kill()
            process.wait(timeout=5)


def test_the_scan_excludes_the_scanning_process_own_ancestor_chain():
    """A wrapper that carries the domain (colcon under an exported
    ROS_DOMAIN_ID) is the scanning process's PARENT, not a leftover — the
    first version flagged colcon itself (review round 1). A bash parent
    carrying the domain must not be listed; only foreign processes are."""
    import shlex

    domain = f"97_{os.getpid()}"  # never a real ROS domain: diagnostics only
    script = (
        "from robot_runtime.testing.sim_ground_truth import leftover_domain_processes\n"
        f"found = leftover_domain_processes({domain!r})\n"
        "print('LEFTOVERS:' + repr(found))\n"
    )
    completed = subprocess.run(
        ["/bin/bash", "-c", f"python3 -c {shlex.quote(script)}; echo bash-parent-still-here"],
        env={**os.environ, "ROS_DOMAIN_ID": domain},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "bash-parent-still-here" in completed.stdout
    marker = next(line for line in completed.stdout.splitlines() if line.startswith("LEFTOVERS:"))
    assert marker == "LEFTOVERS:[]", (
        f"the scanning process's own bash parent (carrying the domain) must be excluded, got {marker}"
    )


def test_the_ancestor_walk_includes_pid_one_and_stops_there():
    """Review round 2: the walk adds each pid before stopping, PID 1
    included, and terminates on unreadable parents and cycles."""
    # a mocked chain: 100 -> 99 -> 1
    chain = {100: 99, 99: 1}
    assert ancestor_pids(100, read_ppid=chain.get) == {100, 99, 1}
    # an unreadable parent stops the walk without looping
    assert ancestor_pids(100, read_ppid=lambda _pid: 0) == {100}
    # a pathological cycle terminates
    assert ancestor_pids(100, read_ppid=lambda _pid: 100) == {100}
    # start_pid=1 short-circuits
    assert ancestor_pids(1, read_ppid=lambda _pid: 0) == {1}


def test_the_partition_sweep_still_matches_the_gz_partition_marker():
    """The Gazebo sweep keeps its contract: GZ_PARTITION matches, and a
    process carrying only the neutral marker is not its business."""
    run = f"lekiwi_gz_sweep_test_{os.getpid()}"
    marked_env = {**os.environ, "GZ_PARTITION": run}
    gz_marked = subprocess.Popen(["/bin/sleep", "30"], env=marked_env)
    neutral_only_env = {**os.environ, RUN_MARKER_ENV: f"unrelated_{os.getpid()}"}
    neutral_only = subprocess.Popen(["/bin/sleep", "30"], env=neutral_only_env)
    try:
        terminate_partition_processes(run)
        assert gz_marked.poll() is not None, "the GZ_PARTITION-marked process survived the sweep"
        assert neutral_only.poll() is None, "the partition sweep killed a process it must not match"
    finally:
        for process in (gz_marked, neutral_only):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            else:
                process.wait(timeout=5)

    # both set (possibly differently) or neither: the caller's explicit choice
    both = {"GZ_PARTITION": "a", "IGN_PARTITION": "b"}
    assert mirror_partition_env(both) == both
    plain = {"PATH": "/bin"}
    assert mirror_partition_env(plain) == {"PATH": "/bin"}
    # inheriting the process environment stays inheriting
    assert mirror_partition_env(None) is None


def test_window_and_latest_read_the_recorded_samples():
    stream = _stream_with(
        [
            GroundTruthSample(1.0, 0.0, 0.0, 0.0, 0.0),
            GroundTruthSample(1.4, 1.0, 2.0, 0.0, 0.0),
            GroundTruthSample(1.5, 1.0, 2.0, 0.0, 0.0),
        ]
    )
    assert [sample.stamp_sec for sample in stream.window(1.1, 1.5)] == [1.4, 1.5]
    assert stream.window(2.0, 3.0) == []
    assert stream.latest().stamp_sec == 1.5
    assert stream.latest_time() == 1.5
    assert stream.position() == (1.0, 2.0)
    assert stream.samples()[-1].stamp_sec == 1.5


def _fake_ign_environment(tmp_path, script_body: str) -> dict[str, str]:
    """An env whose PATH resolves ``ign`` to a fake emitting ``script_body``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ign").write_text(f"#!/bin/sh\n{script_body}\n", encoding="utf-8")
    (bin_dir / "ign").chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}


# One complete Pose_V echo message carrying the model at (x, y) with yaw 0.
# The printf emits the protobuf text exactly as `ign topic -e` does (zero
# fields omitted, one blank line between messages).


def _fake_message(sec: int, x: float) -> str:
    body = (
        (
            "header {\\n  stamp {\\n    sec: SEC\\n    nsec: 0\\n  }\\n}\\n"
            'pose {\\n  name: "lekiwi_robot"\\n  id: 6\\n  position {\\n    x: X\\n  }\\n'
            "orientation {\\n    w: 1\\n  }\\n}\\n\\n"
        )
        .replace("SEC", str(sec))
        .replace("X", str(x))
    )
    # single-quoted for the shell; printf itself turns the \n into newlines
    quoted = body.replace("'", "'\\''")
    return f"printf '{quoted}'\n"


def test_the_live_reader_consumes_a_scripted_pose_stream(tmp_path):
    """The reader thread parses a real child stream incrementally: message 1
    arrives, a pause follows (read1 must return early, not batch), message 2
    arrives — both samples are visible BEFORE the child exits."""
    env = _fake_ign_environment(
        tmp_path,
        _fake_message(3, -1.5) + "sleep 0.4\n" + _fake_message(4, -1.4),
    )
    stream = open_ground_truth("gazebo", world="demo", model="lekiwi_robot", env=env).start()
    try:
        assert stream.wait_for(2, 10.0), f"expected 2 samples, got {len(stream.samples())}"
        stamps = [sample.stamp_sec for sample in stream.samples()]
        assert stamps == [3.0, 4.0]
        assert stream.latest().x == -1.4
        # the first message was parsed long before the child exited: with a
        # batching reader the first sample only appears after the second
        assert stream.latest_sample_age() is not None and stream.latest_sample_age() < 10.0
    finally:
        stream.close()


def test_the_eof_flush_parses_the_withheld_final_message(tmp_path):
    """The last complete message is only known complete at EOF: the reader
    must flush it then (D14.1), so a one-message stream still yields its
    sample after the child exits."""
    env = _fake_ign_environment(tmp_path, _fake_message(7, -1.25))
    stream = open_ground_truth("gazebo", world="demo", model="lekiwi_robot", env=env).start()
    try:
        # without the EOF flush this wait times out: the single message is
        # the trailing chunk and is withheld until the stream ends
        assert stream.wait_for(1, 10.0), f"the EOF flush lost the message, got {len(stream.samples())}"
        sample = stream.latest()
        assert (sample.stamp_sec, sample.x) == (7.0, -1.25)
    finally:
        stream.close()


def test_importing_the_module_pulls_in_no_ros_or_simulator_module():
    """D14.1: importing starts nothing and imports no rclpy/simulator module."""
    code = (
        "import sys\n"
        "import robot_runtime.testing.sim_ground_truth\n"
        "forbidden = ['rclpy']\n"
        "roots = ('gz', 'ign', 'ros_gz', 'ros_gz_sim', 'ros_gz_bridge', 'launch')\n"
        "leaked = [name for name in sys.modules if name in forbidden or name.split('.')[0] in roots]\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=os.environ)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"the module leaked imports: {result.stdout}"

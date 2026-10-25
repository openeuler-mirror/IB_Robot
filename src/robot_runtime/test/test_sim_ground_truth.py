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

import pytest

from robot_runtime.testing.sim_ground_truth import (
    MAX_GAP_SEC,
    GroundTruthSample,
    UnsupportedGroundTruth,
    mirror_partition_env,
    open_ground_truth,
    parse_world_pose_messages,
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


def test_unsupported_platform_is_named_and_gazebo_spawns_nothing():
    with pytest.raises(UnsupportedGroundTruth, match="mujoco"):
        open_ground_truth("mujoco", world="demo", model="lekiwi_robot")
    # the gazebo stream is returned unstarted: opening spawns no process
    stream = open_ground_truth("gazebo", world="demo", model="lekiwi_robot")
    assert stream.world == "demo" and stream.model == "lekiwi_robot"
    assert stream._proc is None and stream._thread is None
    # close() on a never-started stream terminates nothing
    stream.close()


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

# Copyright 2026 IB_Robot Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The published cmd_vel envelope clamp (lekiwi-mujoco-runtime-platform D3.2).

``_clamp_to_envelope`` is the one authoritative clamp in the command path —
ahead of the IK and the SDK wheel saturation — so it is tested directly (the
``object.__new__`` pattern of ``test_base_gating.py``: the clamp is pure
state, no ROS context). The parameters reach the node from
``base_profile.base_node_parameters``; those and the missing-key failure are
pinned here too.
"""

import copy
import math
import threading
import time
from pathlib import Path

import lekiwi_sdk_py as lk
import pytest
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from lekiwi_robot.base_node import LeKiwiBaseNode
from lekiwi_robot.base_profile import base_node_parameters

_LIDAR_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_lidar.yaml"


def make_clamper(**overrides) -> LeKiwiBaseNode:
    node = object.__new__(LeKiwiBaseNode)
    node._max_vx = 0.26
    node._max_vy = 0.23
    node._max_wz = 1.5
    for key, value in overrides.items():
        setattr(node, f"_{key}", value)
    return node


@pytest.mark.parametrize(
    ("vx", "vy", "wz", "expected"),
    [
        (0.1, 0.05, 0.5, (0.1, 0.05, 0.5)),  # inside: unchanged
        (0.26, 0.23, 1.5, (0.26, 0.23, 1.5)),  # exactly at the limits
        (0.5, 0.0, 0.0, (0.26, 0.0, 0.0)),  # vx over
        (-0.5, 0.0, 0.0, (-0.26, 0.0, 0.0)),  # vx under (per axis, sign kept)
        (0.0, 0.5, 0.0, (0.0, 0.23, 0.0)),  # vy over
        (0.0, 0.0, 3.0, (0.0, 0.0, 1.5)),  # wz over
        (0.0, 0.0, -3.0, (0.0, 0.0, -1.5)),  # wz under
        (1.0, -1.0, 4.0, (0.26, -0.23, 1.5)),  # all axes clamped independently
    ],
)
def test_the_envelope_clamps_each_axis_independently(vx, vy, wz, expected):
    assert make_clamper()._clamp_to_envelope(vx, vy, wz) == expected


def test_the_envelope_clamps_the_target_not_the_reported_odometry():
    """The clamp sits in the command path only: an over-limit target is
    clamped before the IK, it never raises, and the odometry integration of
    the realised wheel motion stays untouched (the feedback path has no
    clamp)."""
    node = make_clamper()
    clamped = node._clamp_to_envelope(2.0, -2.0, 10.0)
    assert clamped == (0.26, -0.23, 1.5)


def test_shipped_profile_carries_the_envelope_into_the_node_parameters():
    from robot_runtime.profile import load_profile

    profile = load_profile(_LIDAR_PROFILE)
    params = base_node_parameters(profile, "lekiwi_lidar.yaml")
    cmd_vel = profile["capabilities"]["base.cmd_vel"]
    assert params["max_vx"] == cmd_vel["max_vx"]
    assert params["max_vy"] == cmd_vel["max_vy"]
    assert params["max_wz"] == cmd_vel["max_wz"]


def test_a_missing_envelope_axis_fails_naming_the_key(tmp_path):
    """base_profile requires every axis: no silent unbounded fallback."""
    from robot_runtime.launch_support import RuntimeProfileError
    from robot_runtime.profile import load_profile

    profile = load_profile(_LIDAR_PROFILE)
    for axis in ("max_vx", "max_vy", "max_wz"):
        stripped = copy.deepcopy(profile)
        del stripped["capabilities"]["base.cmd_vel"][axis]
        with pytest.raises(RuntimeProfileError, match=f"capabilities.base.cmd_vel.{axis}"):
            base_node_parameters(stripped, str(tmp_path / "profile.yaml"))


@pytest.mark.parametrize("bad", [-0.26, 0.0, float("inf"), float("nan")])
def test_a_non_finite_or_non_positive_envelope_axis_fails_at_profile_load(tmp_path, bad):
    """min/max turns NaN into +limit and an inverted bound clamps the wrong
    way, so the profile loader rejects anything but finite positive limits."""
    from robot_runtime.launch_support import RuntimeProfileError
    from robot_runtime.profile import load_profile

    profile = load_profile(_LIDAR_PROFILE)
    profile["capabilities"]["base.cmd_vel"]["max_vx"] = bad
    with pytest.raises(RuntimeProfileError, match="finite positive limit"):
        base_node_parameters(profile, str(tmp_path / "profile.yaml"))


# ---------------------------------------------------------------------------
# The command path behaviour: _on_cmd_vel -> _control_loop -> wheel publish
# (review round 1). A NaN/Inf component would survive the envelope clamp as
# +limit, so it is rejected at the subscription boundary and the target is
# zeroed instead. The harness wires the real methods onto the state they
# touch (the object.__new__ pattern of test_base_gating.py).
# ---------------------------------------------------------------------------


class _StubLogger:
    """The rclpy logger signature: one preformatted message, keyword-only
    options (review round 2) — a printf-style second positional would raise
    TypeError on the real logger, so the stub must not accept one either."""

    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message: str, **kwargs):
        self.warnings.append(message)


class _WheelRecorder:
    def __init__(self):
        self.commands: list[list[float]] = []

    def publish(self, msg: Float64MultiArray):
        self.commands.append(list(msg.data))


def _driven_node(**limit_overrides) -> tuple[LeKiwiBaseNode, _WheelRecorder, _StubLogger]:
    node = object.__new__(LeKiwiBaseNode)
    node._lock = threading.Lock()
    node._stop_latched = False
    node._nav_enabled = True
    node._status_seen = True
    node._active_mode = "base_navigation"
    node._base_modes = {"base_navigation"}
    node._target = (0.0, 0.0, 0.0)
    node._last_cmd = 0.0
    node._staleness = 5.0
    node._last_published_zero = False
    node._max_vx = limit_overrides.get("max_vx", 0.26)
    node._max_vy = limit_overrides.get("max_vy", 0.23)
    node._max_wz = limit_overrides.get("max_wz", 1.5)
    geometry = lk.BaseGeometry()
    geometry.wheel_radius = 0.05
    geometry.base_radius = 0.125
    geometry.max_wheel_radps = 4.602
    node._geometry = geometry
    recorder = _WheelRecorder()
    logger = _StubLogger()
    node._wheel_pub = recorder
    node.get_logger = lambda: logger
    return node, recorder, logger


def _twist(vx=0.0, vy=0.0, wz=0.0) -> Twist:
    message = Twist()
    message.linear.x, message.linear.y, message.angular.z = vx, vy, wz
    return message


def _last_wheels(recorder: _WheelRecorder) -> list[float]:
    assert recorder.commands, "the control loop published no wheel command"
    return recorder.commands[-1]


def test_a_moving_command_reaches_the_wheels():
    node, recorder, logger = _driven_node()
    node._on_cmd_vel(_twist(vx=0.1))
    node._control_loop()
    assert max(abs(value) for value in _last_wheels(recorder)) > 0.0
    assert logger.warnings == []


def test_a_legal_zero_command_stays_zero():
    node, recorder, logger = _driven_node()
    node._on_cmd_vel(_twist())
    node._control_loop()
    assert _last_wheels(recorder) == [0.0, 0.0, 0.0]
    assert logger.warnings == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_command_is_rejected_and_zeroes_the_target(value):
    node, recorder, logger = _driven_node()
    node._on_cmd_vel(_twist(vx=0.1))
    node._control_loop()
    assert max(abs(value) for value in _last_wheels(recorder)) > 0.0
    node._on_cmd_vel(_twist(vx=value))
    node._control_loop()
    assert _last_wheels(recorder) == [0.0, 0.0, 0.0]
    assert node._target == (0.0, 0.0, 0.0)
    assert any("non-finite" in warning for warning in logger.warnings)


def test_a_non_finite_component_alone_poisons_the_whole_command():
    node, recorder, _logger = _driven_node()
    node._on_cmd_vel(_twist(vx=0.1, wz=math.nan))
    node._control_loop()
    assert _last_wheels(recorder) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# The node startup validation: finite positive limits only (a real node with
# parameter overrides; the root conftest isolates the ROS domain).
# ---------------------------------------------------------------------------

_NODE_PARAMS = [
    "wheel_joints:=['7', '8', '9']",
    "wheel_radius:=0.05",
    "base_radius:=0.125",
    "max_wheel_radps:=4.602",
    "control_frequency:=50.0",
    "staleness_s:=0.5",
    "max_vx:=0.26",
    "max_vy:=0.23",
    "max_wz:=1.5",
    "odom_frame:=odom",
    "base_frame:=base_link",
    "publish_tf:=false",
]


@pytest.mark.parametrize("bad", ["-0.26", "0.0", ".inf", ".nan"])
def test_node_startup_rejects_a_non_finite_or_non_positive_limit(bad):
    """RuntimeError names the parameter; nothing is left constructed."""
    replacement = f"max_vx:={bad}"
    params = [replacement if p.startswith("max_vx:") else p for p in _NODE_PARAMS]
    rclpy.init(args=["--ros-args"] + sum([["-p", p] for p in params], []))
    try:
        with pytest.raises(RuntimeError, match="'max_vx' must be a finite positive limit"):
            LeKiwiBaseNode()
    finally:
        rclpy.shutdown()


def test_node_startup_accepts_the_shipped_envelope():
    node = None
    rclpy.init(args=["--ros-args"] + sum([["-p", p] for p in _NODE_PARAMS], []))
    try:
        node = LeKiwiBaseNode()
        assert (node._max_vx, node._max_vy, node._max_wz) == (0.26, 0.23, 1.5)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_the_real_node_survives_a_nan_command_and_publishes_zero_wheels():
    """Review round 2, node level with the REAL rclpy logger: a NaN command
    while moving takes the logging path (printf-style args would raise
    TypeError there), the node stays alive, and a zero wheel command actually
    reaches the ROS graph."""
    from rclpy.executors import SingleThreadedExecutor

    rclpy.init(args=["--ros-args"] + sum([["-p", p] for p in _NODE_PARAMS], []))
    node = None
    collector = None
    executor = None
    try:
        node = LeKiwiBaseNode()
        collector = rclpy.create_node("wheel_command_collector")
        received: list[list[float]] = []
        collector.create_subscription(
            Float64MultiArray, "/base_velocity_controller/commands", lambda msg: received.append(list(msg.data)), 10
        )
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(collector)
        # admitted state without a runtime status round trip
        node._nav_enabled = True
        node._status_seen = True
        node._active_mode = "base_navigation"

        def drain(timeout: float = 5.0) -> list[float]:
            received.clear()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not received:
                executor.spin_once(timeout_sec=0.05)
            assert received, "no wheel command reached the ROS graph"
            return received[-1]

        node._on_cmd_vel(_twist(vx=0.1))
        node._control_loop()
        assert max(abs(value) for value in drain()) > 0.0
        # the NaN command: the real logger's warning path runs here
        node._on_cmd_vel(_twist(vx=math.nan))
        node._control_loop()
        assert drain() == [0.0, 0.0, 0.0]
        # the node survived and still commands motion afterwards
        node._on_cmd_vel(_twist(vx=0.05))
        node._control_loop()
        assert max(abs(value) for value in drain()) > 0.0
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if collector is not None:
            collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

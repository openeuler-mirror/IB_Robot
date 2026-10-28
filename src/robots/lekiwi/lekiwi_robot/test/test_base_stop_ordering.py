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

"""The stop boundary must be the last thing on the wire.

When navigation is switched off the node publishes zero wheel speeds, then
acknowledges, then returns success. Nothing may appear on the wheel topic after
that acknowledgment -- an operator who has been told the base is stopped must
not then watch it move.

This is a threading property, not a ROS one, so it is tested without a context:
the two callbacks are driven directly from two threads with the publishers
replaced by recorders. The recorder makes the first wheel publish slow, which
is what opens the interleaving window wide enough to be deterministic rather
than timing-dependent.
"""

import threading
import time
import types

from lekiwi_robot.base_node import LeKiwiBaseNode


class Recorder:
    """Captures the ordered sequence of wheel and ack publications."""

    def __init__(self, release: threading.Event, delay_s: float = 0.05):
        self.events: list[tuple[str, object]] = []
        self._release = release
        self._delay_s = delay_s
        self._delayed = False

    def wheels(self, radps) -> None:
        value = tuple(float(v) for v in radps)
        if not self._delayed and any(v != 0.0 for v in value):
            # The publish is in flight. Let the disable service start now, and
            # stay in flight long enough for it to finish if it is able to.
            self._delayed = True
            self._release.set()
            time.sleep(self._delay_s)
        self.events.append(("wheels", value))

    def ack(self, enabled: bool) -> None:
        self.events.append(("ack", bool(enabled)))


def make_node(recorder: Recorder) -> LeKiwiBaseNode:
    node = object.__new__(LeKiwiBaseNode)
    node._lock = threading.Lock()
    node._stop_latched = False
    node._nav_enabled = True
    node._status_seen = True
    node._active_mode = "base_navigation"
    node._base_modes = {"base_navigation"}
    node._staleness = 10.0
    node._last_published_zero = False
    node._target = (0.2, 0.0, 0.0)
    node._last_cmd = time.monotonic()
    # the published cmd_vel envelope (D3.2): the target stays inside it here
    node._max_vx = 0.26
    node._max_vy = 0.23
    node._max_wz = 1.5
    node._geometry = _geometry()
    node._publish_wheels = recorder.wheels
    node._publish_ack = lambda: recorder.ack(node._nav_enabled)
    return node


def _geometry():
    import lekiwi_sdk_py as lk

    geometry = lk.BaseGeometry()
    geometry.wheel_radius = 0.05
    geometry.base_radius = 0.125
    geometry.max_wheel_radps = 10.0
    return geometry


def test_no_wheel_command_is_published_after_the_disable_ack():
    release = threading.Event()
    recorder = Recorder(release)
    node = make_node(recorder)

    def disable():
        release.wait(timeout=2.0)
        request = types.SimpleNamespace(data=False)
        response = types.SimpleNamespace(success=False, message="")
        node._on_set_navigation(request, response)

    disabler = threading.Thread(target=disable)
    disabler.start()
    node._control_loop()
    disabler.join(timeout=5.0)
    assert not disabler.is_alive()

    kinds = [kind for kind, _ in recorder.events]
    assert "ack" in kinds, f"the disable service never acknowledged: {recorder.events}"
    after_ack = recorder.events[kinds.index("ack") + 1 :]
    moving = [value for kind, value in after_ack if kind == "wheels" and any(v != 0.0 for v in value)]
    assert not moving, f"wheel command published after the disable ack: {recorder.events}"


def test_disable_publishes_a_zero_boundary_before_acknowledging():
    release = threading.Event()
    recorder = Recorder(release)
    node = make_node(recorder)

    request = types.SimpleNamespace(data=False)
    response = types.SimpleNamespace(success=False, message="")
    node._on_set_navigation(request, response)

    assert response.success is True
    assert recorder.events[0] == ("wheels", (0.0, 0.0, 0.0))
    assert recorder.events[1] == ("ack", False)

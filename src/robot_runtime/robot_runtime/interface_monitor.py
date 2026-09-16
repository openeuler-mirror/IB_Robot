"""Observe published interfaces without treating configuration as device feedback."""

from __future__ import annotations

import threading
import time
from collections import deque
from copy import deepcopy

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message


class InterfaceMonitor:
    def __init__(self, node, description: dict):
        self._lock = threading.Lock()
        self._states = {}
        self._times = {}
        self._last_receipt = {}
        self._subscriptions = []
        self._latched = set()
        for name, interface in description["interfaces"].items():
            if interface["kind"] != "topic" or interface["direction"] != "publish":
                continue
            if interface["capability"] == "runtime.status":
                continue  # Do not recursively monitor RuntimeStatus itself.
            self._states[name] = {
                "state": "unknown",
                "observed_profile": None,
                "observed_frame_id": None,
                "last_seen": None,
                "detail": "No sample received",
            }
            self._times[name] = deque(maxlen=32)
            msg_type = get_message(interface["message_type"])
            latched = interface["qos"]["durability"] == "transient_local"
            if latched:
                self._latched.add(name)
            # Mirror the declared QoS instead of assuming sensor-style
            # BEST_EFFORT: a late-joining BEST_EFFORT subscriber does not
            # reliably receive the historical sample of a RELIABLE
            # TRANSIENT_LOCAL publisher (e.g. /robot_description), which left
            # those interfaces stuck at "No sample received". A subscriber
            # requesting at most the publisher's offered reliability always
            # matches, so mirroring the declaration is safe for every kind.
            reliability = (
                ReliabilityPolicy.RELIABLE
                if interface["qos"].get("reliability") == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            )
            qos = QoSProfile(
                depth=5,
                reliability=reliability,
                durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE,
            )
            self._subscriptions.append(
                node.create_subscription(
                    msg_type,
                    interface["endpoint"],
                    lambda msg, key=name, spec=interface: self.observe(key, spec, msg),
                    qos,
                )
            )

    def observe(self, name, interface, msg):
        now = time.monotonic()
        with self._lock:
            times = self._times[name]
            times.append(now)
            self._last_receipt[name] = now
            state = self._states[name]
            state.update(state="ready", last_seen=time.time(), detail="Sample received; rate is receipt-time measured")
            header = getattr(msg, "header", None)
            frame_id = header.frame_id if header else None
            state["observed_frame_id"] = frame_id
            errors = []
            if interface.get("frame_id") and frame_id != interface["frame_id"]:
                errors.append(f"frame_id {frame_id!r} != {interface['frame_id']!r}")
            if interface["message_type"] == "sensor_msgs/msg/Image":
                if (
                    msg.width <= 0
                    or msg.height <= 0
                    or not msg.encoding
                    or msg.step <= 0
                    or len(msg.data) != msg.step * msg.height
                ):
                    state["observed_profile"] = None
                    errors.append("invalid Image dimensions, encoding or payload length")
                else:
                    fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
                    observed = {"width": msg.width, "height": msg.height, "encoding": msg.encoding, "fps": fps}
                    state["observed_profile"] = observed
                    for field in ("width", "height", "encoding"):
                        configured = (interface["configured_profile"] or {}).get(field)
                        if configured is not None and observed[field] != configured:
                            errors.append(f"{field} {observed[field]!r} != configured {configured!r}")
                    if len(times) < 3:
                        state.update(state="unknown", detail="Waiting for samples to measure the publication rate")
            if errors:
                times.clear()  # Invalid samples must not inflate the measured valid-stream rate.
                state.update(state="mismatch", detail="; ".join(errors))

    def states(self) -> dict:
        with self._lock:
            states = deepcopy(self._states)
            now = time.monotonic()
            for name, state in states.items():
                last = self._last_receipt.get(name)
                if last is not None and name not in self._latched and now - last > 2.0:
                    state.update(state="stale", detail="No sample received for more than 2 seconds")
            return states

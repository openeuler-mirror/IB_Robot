"""Contract → runtime spec compilation for hardware_mock.

Reads the ``robot`` dict produced by :mod:`robot_config.loader` and emits
the concrete pub/sub specifications the mock node executes. All architectural
validation (type whitelist, joint mapping, publish rate vs align tolerance)
happens here so failures surface at launch time, not mid-episode.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hardware_mock.image_sources import ImageSourceSpec, resolve_spec
from hardware_mock.type_registry import (
    ensure_publish_supported,
    ensure_subscribe_supported,
)

# ----- runtime spec dataclasses ----------------------------------------------


@dataclass
class ObservationSpec:
    key: str
    topic: str
    msg_type: str
    rate_hz: float
    qos: dict[str, Any]
    kind: str  # 'image' | 'joint_state'
    # Image-only fields
    image: ImageSourceSpec | None = None
    frame_id: str = ""
    # JointState-only fields
    joint_names: list[str] = field(default_factory=list)


@dataclass
class ActionSpec:
    key: str
    topic: str
    msg_type: str
    qos: dict[str, Any]
    # Mapping action vector index -> joint index in JointModel.joint_ids
    index_to_joint_index: list[int] = field(default_factory=list)


@dataclass
class MockPlan:
    """Everything the mock node needs after parsing."""

    joint_ids: list[str]
    initial_positions: dict[str, float]
    observations: list[ObservationSpec]
    actions: list[ActionSpec]
    joint_state_rate_hz: float


# ----- helpers ---------------------------------------------------------------


_ACTION_INDEX_RE = re.compile(r"^action\.(\d+)$")


def _get_peripheral(robot: dict[str, Any], name: str) -> dict[str, Any]:
    for p in robot.get("peripherals", []) or []:
        if p.get("name") == name:
            return p
    raise ValueError(
        f"contract.observations references peripheral '{name}' but it is not declared in robot.peripherals"
    )


def _resolve_image_dims(periph: dict[str, Any], obs: dict[str, Any]) -> tuple[int, int]:
    """Pick (width, height). Prefer contract image.resize=[H,W], fall back to peripheral."""
    img_cfg = obs.get("image") or {}
    resize = img_cfg.get("resize")
    if isinstance(resize, list | tuple) and len(resize) == 2:
        h, w = int(resize[0]), int(resize[1])
        return w, h
    return int(periph.get("width", 640)), int(periph.get("height", 480))


def _validate_image_rate(key: str, rate_hz: float, align: dict[str, Any] | None, skip: bool) -> None:
    if skip or not align:
        return
    tol_ms = align.get("tol_ms")
    if tol_ms is None:
        return
    min_rate = 2000.0 / float(tol_ms)
    if rate_hz + 1e-6 < min_rate:
        raise ValueError(
            f"observation '{key}' publish rate {rate_hz:.2f} Hz is below the safe "
            f"minimum {min_rate:.2f} Hz (= 2 / (align.tol_ms={tol_ms}ms / 1000)). "
            "Increase peripheral fps, relax align.tol_ms, or set "
            "robot.hardware_mock.skip_rate_check: true to bypass."
        )


def _validate_joint_rate(key: str, rate_hz: float, align: dict[str, Any] | None, skip: bool) -> None:
    _validate_image_rate(key, rate_hz, align, skip)  # same rule


def _build_action_index_map(selector_names: Sequence[str], joint_ids: Sequence[str], action_key: str) -> list[int]:
    """Map each 'action.<i>' entry to its joint index in joint_ids.

    Rule: ``action.<i>`` corresponds to ``joint_ids[i]``. This is the
    same convention the policy adapters use; if the contract is laid out
    differently the launch must fail loud.
    """
    mapping: list[int] = []
    n_joints = len(joint_ids)
    for entry in selector_names:
        m = _ACTION_INDEX_RE.match(str(entry))
        if not m:
            raise ValueError(
                f"action '{action_key}' selector entry '{entry}' does not match "
                "'action.<int>'. hardware_mock only supports indexed action selectors."
            )
        idx = int(m.group(1))
        if not 0 <= idx < n_joints:
            raise ValueError(
                f"action '{action_key}' selector '{entry}' maps to joint index {idx} "
                f"which is outside joints.all (size={n_joints})"
            )
        mapping.append(idx)
    if not mapping:
        raise ValueError(f"action '{action_key}' has empty selector.names")
    return mapping


# ----- main entry point ------------------------------------------------------


def build_plan(robot: dict[str, Any]) -> MockPlan:
    """Compile a :class:`MockPlan` from the full ``robot`` config dict.

    Raises:
        ValueError: For any architectural inconsistency. Never returns a
            partially-valid plan.
    """
    joints_cfg = robot.get("joints") or {}
    joint_ids = list(joints_cfg.get("all") or [])
    if not joint_ids:
        raise ValueError("hardware_mock requires robot.joints.all to be a non-empty list")

    reset = (robot.get("ros2_control") or {}).get("reset_positions") or {}
    initial = {jid: float(reset.get(jid, 0.0)) for jid in joint_ids}

    mock_cfg = robot.get("hardware_mock") or {}
    skip_rate_check = bool(mock_cfg.get("skip_rate_check", False))
    joint_state_rate_hz = float(mock_cfg.get("joint_state_rate_hz", 50.0))
    if joint_state_rate_hz <= 0:
        raise ValueError("hardware_mock.joint_state_rate_hz must be > 0")
    image_overrides = mock_cfg.get("image_sources") or {}

    contract = robot.get("contract") or {}
    observations_raw = contract.get("observations") or []
    actions_raw = contract.get("actions") or []
    if not observations_raw:
        raise ValueError("contract.observations is empty; nothing for hardware_mock to publish")
    if not actions_raw:
        raise ValueError("contract.actions is empty; nothing for hardware_mock to subscribe")

    # --- observations ---------------------------------------------------
    obs_specs: list[ObservationSpec] = []
    for obs in observations_raw:
        key = obs.get("key")
        topic = obs.get("topic")
        msg_type = obs.get("type")
        if not (key and topic and msg_type):
            raise ValueError(f"observation missing key/topic/type: {obs}")
        ensure_publish_supported(msg_type)
        qos = obs.get("qos") or {}
        align = obs.get("align") or {}

        if msg_type == "sensor_msgs/msg/Image":
            periph_name = obs.get("peripheral")
            if not periph_name:
                raise ValueError(f"observation '{key}' is an Image but has no 'peripheral' reference")
            periph = _get_peripheral(robot, periph_name)
            width, height = _resolve_image_dims(periph, obs)
            rate_hz = float(periph.get("fps", 30))
            if rate_hz <= 0:
                raise ValueError(f"peripheral '{periph_name}' fps must be > 0")
            _validate_image_rate(key, rate_hz, align, skip_rate_check)
            img_spec = resolve_spec(periph_name, width, height, image_overrides)
            obs_specs.append(
                ObservationSpec(
                    key=key,
                    topic=topic,
                    msg_type=msg_type,
                    rate_hz=rate_hz,
                    qos=qos,
                    kind="image",
                    image=img_spec,
                    frame_id=str(periph.get("optical_frame_id") or periph.get("frame_id") or periph_name),
                )
            )
        elif msg_type == "sensor_msgs/msg/JointState":
            _validate_joint_rate(key, joint_state_rate_hz, align, skip_rate_check)
            obs_specs.append(
                ObservationSpec(
                    key=key,
                    topic=topic,
                    msg_type=msg_type,
                    rate_hz=joint_state_rate_hz,
                    qos=qos,
                    kind="joint_state",
                    joint_names=list(joint_ids),
                )
            )
        else:  # pragma: no cover - guarded by ensure_publish_supported
            raise ValueError(f"unsupported observation type '{msg_type}'")

    # --- actions --------------------------------------------------------
    act_specs: list[ActionSpec] = []
    for act in actions_raw:
        key = act.get("key") or "action"
        publish = act.get("publish") or {}
        topic = publish.get("topic")
        msg_type = publish.get("type")
        if not (topic and msg_type):
            raise ValueError(f"action '{key}' missing publish.topic or publish.type")
        ensure_subscribe_supported(msg_type)
        qos = publish.get("qos") or {}
        selector = act.get("selector") or {}
        names = selector.get("names") or []
        idx_map = _build_action_index_map(names, joint_ids, key)
        act_specs.append(
            ActionSpec(
                key=key,
                topic=topic,
                msg_type=msg_type,
                qos=qos,
                index_to_joint_index=idx_map,
            )
        )

    return MockPlan(
        joint_ids=joint_ids,
        initial_positions=initial,
        observations=obs_specs,
        actions=act_specs,
        joint_state_rate_hz=joint_state_rate_hz,
    )

"""Unit tests for RuntimeState: lifecycle, faults, stop latch, status assembly, change notification."""

import json

from builtin_interfaces.msg import Time

from robot_runtime.contract import LIFECYCLE_ACTIVE, LIFECYCLE_CONNECTING, LIFECYCLE_STOPPED, STOP_TORQUE_OFF
from robot_runtime.modes import ModeModel, ModeModelConfig
from robot_runtime.state import RuntimeState


def _state(events: list[str] | None = None) -> RuntimeState:
    modes = ModeModel(
        ModeModelConfig.from_profile(
            {"modes": {"initial": "idle", "idle": {"transitions": ["stream"]}, "stream": {"transitions": ["idle"]}}}
        )
    )
    return RuntimeState(
        "unit_runtime",
        "9.9.9",
        {"joint.state": {"joint_count": 1}, "runtime.stop": {"cancel_bound_s": 0.1}},
        modes,
        on_change=(lambda: events.append("change")) if events is not None else None,
    )


def test_initial_state_and_message_fields():
    state = _state()
    msg = state.to_msg(Time())
    assert msg.runtime_name == "unit_runtime" and msg.runtime_version == "9.9.9"
    assert msg.lifecycle == LIFECYCLE_CONNECTING
    assert msg.capabilities == ["joint.state", "runtime.stop"]
    assert json.loads(msg.capabilities_json)["runtime.stop"] == {"cancel_bound_s": 0.1}
    assert msg.active_mode == "idle" and msg.declared_modes == ["idle", "stream"]
    assert not msg.stop_latched and msg.stop_policy == "" and msg.faults == []


def test_change_notification_only_on_actual_change():
    events: list[str] = []
    state = _state(events)
    state.set_lifecycle(LIFECYCLE_ACTIVE)
    state.set_lifecycle(LIFECYCLE_ACTIVE)
    state.add_fault("x")
    state.add_fault("x")
    state.clear_faults()
    state.clear_faults()
    assert events == ["change", "change", "change"]


def test_stop_latch_engage_and_clear():
    state = _state()
    state.set_lifecycle(LIFECYCLE_ACTIVE)
    state.engage_stop(STOP_TORQUE_OFF)
    msg = state.to_msg(Time())
    assert msg.lifecycle == LIFECYCLE_STOPPED and msg.stop_latched and msg.stop_policy == STOP_TORQUE_OFF
    state.clear_stop()
    msg = state.to_msg(Time())
    assert msg.lifecycle == LIFECYCLE_ACTIVE and not msg.stop_latched and msg.stop_policy == ""


def test_stop_epoch_increments_even_when_already_latched():
    state = _state()
    assert state.stop_epoch == 0
    state.engage_stop(STOP_TORQUE_OFF)
    assert state.stop_epoch == 1
    state.engage_stop(STOP_TORQUE_OFF)
    assert state.stop_epoch == 2
    state.clear_stop()
    assert state.stop_epoch == 2
    state.engage_stop(STOP_TORQUE_OFF)
    assert state.stop_epoch == 3


def test_clear_stop_requires_current_epoch():
    state = _state()
    epoch = state.stop_epoch
    state.engage_stop(STOP_TORQUE_OFF)
    assert not state.clear_stop_if_unchanged(epoch)
    assert state.stop_latched and state.lifecycle == LIFECYCLE_STOPPED
    assert state.stop_policy == STOP_TORQUE_OFF
    assert state.clear_stop_if_unchanged(state.stop_epoch)
    assert not state.stop_latched and state.lifecycle == LIFECYCLE_ACTIVE
    assert state.stop_policy == ""
    assert state.clear_stop_if_unchanged(state.stop_epoch)


def test_rejections_serialized_as_parallel_arrays():
    state = _state()
    state.modes.note_rejected("gripper_stream")
    state.modes.note_rejected("arm_stream")
    state.modes.note_rejected("arm_stream")
    msg = state.to_msg(Time())
    assert msg.rejected_channels == ["arm_stream", "gripper_stream"]
    assert list(msg.rejected_counts) == [2, 1]

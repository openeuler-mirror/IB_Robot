"""JointState selector extraction must use joint-name semantics, not indices."""

from types import MethodType, SimpleNamespace

from dataset_tools.rerun_viewer import RerunViewer


class _Logger:
    def warning(self, _message):
        pass


def _extract(names, message):
    node = SimpleNamespace(_joint_state_name_warned=False, get_logger=lambda: _Logger())
    node._extract_joint_state = MethodType(RerunViewer._extract_joint_state, node)
    return node._extract_joint_state(message, names)


def _joint_state():
    # Hardware publishes joints in bus order, not sorted: index 1 is joint "4".
    message = SimpleNamespace(name=["2", "4", "1", "3", "5", "6"])
    message.position = [0.2, 0.4, 0.1, 0.3, 0.5, 0.6]
    message.velocity = [2.0, 4.0, 1.0, 3.0, 5.0, 6.0]
    return message


def test_position_selector_resolves_by_joint_name():
    assert _extract(["position.1", "position.6"], _joint_state()) == [0.1, 0.6]


def test_velocity_selector_resolves_by_joint_name():
    assert _extract(["velocity.4"], _joint_state()) == [4.0]


def test_unsorted_message_order_does_not_confuse_the_lookup():
    # The historical bug treated the suffix as a 1-based index, reading joint
    # "4" (index 1) when the selector named joint "1".
    assert _extract(["position.1"], _joint_state()) != [0.4]


def test_bare_joint_name_reads_position():
    assert _extract(["1", "6"], _joint_state()) == [0.1, 0.6]


def test_unknown_joint_resolves_to_nan_not_zero():
    import math

    values = _extract(["position.9"], _joint_state())
    assert values is not None and math.isnan(values[0])

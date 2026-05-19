"""Tests for hardware_mock.joint_model.JointModel."""

import pytest

from hardware_mock.joint_model import JointModel


def test_initial_positions_default_zero_when_missing():
    jm = JointModel(["1", "2", "3"], initial={"2": 1.5})
    assert jm.joint_ids == ["1", "2", "3"]
    assert jm.positions() == [0.0, 1.5, 0.0]


def test_set_by_index_partial_update_preserves_order():
    jm = JointModel(["a", "b", "c", "d"])
    jm.set_by_index([1, 3], [0.7, -0.2])
    assert jm.positions() == [0.0, 0.7, 0.0, -0.2]


def test_set_by_index_length_mismatch_raises():
    jm = JointModel(["a", "b"])
    with pytest.raises(ValueError):
        jm.set_by_index([0], [1.0, 2.0])


def test_set_by_index_out_of_range_raises():
    jm = JointModel(["a", "b"])
    with pytest.raises(IndexError):
        jm.set_by_index([5], [1.0])


def test_duplicate_joint_ids_rejected():
    with pytest.raises(ValueError):
        JointModel(["a", "a"])


def test_empty_joint_ids_rejected():
    with pytest.raises(ValueError):
        JointModel([])

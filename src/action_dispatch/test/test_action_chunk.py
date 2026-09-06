import pytest

from action_dispatch.action_chunk import (
    validate_execution_horizon,
)


def test_execution_horizon_zero_means_full_chunk():
    assert validate_execution_horizon(0, 5) is None
    assert validate_execution_horizon(3, 5) == 3


@pytest.mark.parametrize(
    "value",
    [True, -1, 6, "2"],
)
def test_execution_horizon_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        validate_execution_horizon(value, 5)


def test_execution_horizon_rejects_boolean_zero():
    with pytest.raises(ValueError):
        validate_execution_horizon(False, 5)

import math

import numpy as np
import pytest

from robot_teleop.safety_filter import SafetyFilter


def test_scalar_clipping_and_count_match_numpy_reference():
    bounds = {"1": {"min": -1.0, "max": 2.0}, "6": {"min": -0.5, "max": 1.5}}
    safety = SafetyFilter(bounds)
    count = {name: 0 for name in bounds}
    for value in [-100, -1.000001, -0.500001, 0, 1.500001, 2.00001, 100]:
        result = safety.apply_limits(dict.fromkeys(bounds, value))
        for name, limits in bounds.items():
            expected = float(np.clip(value, limits["min"], limits["max"]))
            assert result[name] == expected
            count[name] += not np.isclose(value, expected, atol=1e-6)
    assert safety.get_clip_statistics() == count


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("name", ["limited", "unlimited"])
def test_nonfinite_commands_fail_closed(value, name):
    safety = SafetyFilter({"limited": {"min": -1.0, "max": 1.0}})
    assert safety.apply_limits({"valid": 0.0, name: value}) == {}


@pytest.mark.parametrize(
    "bounds",
    [{"min": math.nan}, {"max": math.nan}, {"min": 2, "max": 1}, {"min": math.inf}, {"max": -math.inf}],
)
def test_invalid_bounds_rejected(bounds):
    with pytest.raises(ValueError, match="Invalid joint limits"):
        SafetyFilter({"1": bounds})


def test_unbounded_finite_values_pass_and_statistics_reset():
    safety = SafetyFilter({"1": {"max": 1}})
    assert safety.apply_limits({"1": -10.0, "other": 2.0}) == {"1": -10.0, "other": 2.0}
    safety.apply_limits({"1": 2.0})
    counts = safety.get_clip_statistics()
    counts.clear()
    assert safety.get_clip_statistics() == {"1": 1}
    safety.reset_statistics()
    assert safety.get_clip_statistics() == {}

import pytest

from ibrobot_tracing.statistics import percentile


def test_percentile_uses_nearest_rank_for_tail_metrics() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.0
    assert percentile(values, 0.95) == 4.0
    assert percentile(values, 0.99) == 4.0


def test_percentile_rejects_empty_values_and_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="between zero and one"):
        percentile([1.0], 1.1)

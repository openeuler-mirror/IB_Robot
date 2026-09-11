"""Cross-layer strategy name consistency.

The action_dispatch strategy modules import their strategy vocabulary
directly from the canonical robot_config SSOT, so name drift is impossible
by construction. These tests pin the factory side: every SSOT chunking name
has a planner implementation and the two blending names map to the two plan
stores the dispatchers use.
"""

from __future__ import annotations

from action_dispatch.action_blending import SUPPORTED_BLENDING
from action_dispatch.chunk_planning import SUPPORTED_CHUNKING


def test_every_chunking_name_has_a_factory_implementation():
    from action_dispatch.chunk_planning import create_chunk_planner

    for name in SUPPORTED_CHUNKING:
        planner = create_chunk_planner(name)
        assert planner.chunking_strategy == name


def test_blending_names_cover_plan_stores():
    """The two blending names map to the two plan stores the dispatchers use."""
    from action_dispatch.active_plan import ActivePlan
    from action_dispatch.temporal_smoother import TemporalSmootherManager

    for smoother in (None, TemporalSmootherManager()):
        assert ActivePlan(capacity=2, watermark=0, smoother=smoother).take_action().source == "empty"
    assert {"none", "temporal_ensemble"} == set(SUPPORTED_BLENDING)

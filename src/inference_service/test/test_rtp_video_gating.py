"""RTP video freshness gating tests for PipelinePolicyNode.

Ported from the legacy ``test_pure_inference_engine.py`` when that file was
removed with the second execution architecture.  These tests pin the
sender-side freshness gate introduced for distributed RTP video streams:
keys in ``rtp_video_keys`` must be gated on what the sender actually put on
the wire (``_video_stream_manager.latest_sent_capture_ns``), never on the
local subscription buffer.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from inference_service.pipeline_policy_node import ObservationNotReadyError, PipelinePolicyNode


def _rtp_video_node(last_sent: dict[str, int], *, buffer_entry: tuple[int, int, object] | None):
    """Fake node whose local buffer and sender-side record can disagree.

    ``buffer_entry`` seeds the subscription buffer (what the device received)
    while ``last_sent`` feeds the manager query (what left on the wire), so
    tests can pin the freshness gate to the wire-side view.
    """
    spec = SimpleNamespace(
        key="observation.images.top",
        topic="/camera/top/image_raw",
        resample_policy="hold",
        asof_tol_ms=0,
    )
    state = SimpleNamespace(
        spec=spec,
        max_age_ns=500_000_000,
        step_ns=50_000_000,
        history_window_ns=1_000_000_000,
        history=[buffer_entry] if buffer_entry is not None else [],
    )
    return SimpleNamespace(
        _subs={spec.key: state},
        _state_specs=[],
        _frequency=20.0,
        _n_obs_steps=1,
        _observation_lock=threading.Lock(),
        _sample_observation=PipelinePolicyNode._sample_observation,
        _sample_observation_history=PipelinePolicyNode._sample_observation_history,
        _video_stream_manager=SimpleNamespace(latest_sent_capture_ns=lambda key: last_sent.get(key, 0)),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000)),
    )


def test_rtp_video_gate_fails_closed_when_nothing_left_the_device():
    # The local buffer holds a frame that is perfectly fresh for the buffer
    # gate, but the sender never put anything on the wire: the compute side
    # has no frame at all, so the decision must fail closed.
    node = _rtp_video_node({}, buffer_entry=(1_000_000_000, 1_000_000_000, object()))

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000, rtp_video_keys={"observation.images.top"})

    issue = error.value.details["observations"][0]
    assert issue["key"] == "observation.images.top"
    assert issue["reason"] == "video_not_sent"


def test_rtp_video_gate_reports_stale_wire_record_despite_fresh_buffer():
    # Fresh buffer entry, but the last frame that actually left the device is
    # older than max_age (network outage / encode backlog): the gate must
    # read the wire-side age, not the buffer-side age.
    node = _rtp_video_node(
        {"observation.images.top": 100_000_000},
        buffer_entry=(1_000_000_000, 1_000_000_000, object()),
    )

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000, rtp_video_keys={"observation.images.top"})

    issue = error.value.details["observations"][0]
    assert issue["reason"] == "video_send_stale"
    assert issue["age_ms"] == pytest.approx(900.0)


def test_rtp_video_gate_passes_on_recent_wire_send_and_skips_local_value():
    # A recent wire send passes even though the local buffer is empty; the
    # returned observations never carry the local video value because the
    # policy consumes decoded frames on the compute side.
    node = _rtp_video_node({"observation.images.top": 900_000_000}, buffer_entry=None)

    observations = PipelinePolicyNode._sample_observations(
        node, 1_000_000_000, rtp_video_keys={"observation.images.top"}
    )

    assert observations == {}


def test_rtp_video_gate_without_manager_fails_closed():
    # A distributed video key with no manager attached (rollover raced the
    # decision) must fail closed rather than read as forever-fresh.
    node = _rtp_video_node({}, buffer_entry=(1_000_000_000, 1_000_000_000, object()))
    del node._video_stream_manager

    with pytest.raises(ObservationNotReadyError) as error:
        PipelinePolicyNode._sample_observations(node, 1_000_000_000, rtp_video_keys={"observation.images.top"})

    assert error.value.details["observations"][0]["reason"] == "video_not_sent"

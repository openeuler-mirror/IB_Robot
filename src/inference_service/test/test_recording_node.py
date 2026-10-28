"""Tests for the recording node's cloud handshake.

Edge RTP streams stay dormant until they observe a ROLE_CLOUD heartbeat:
`EdgeVideoStreamManager.descriptors()` returns an empty tuple while no session
is bound, so the sender silently publishes nothing and the receiver writes a
zero-byte .h264. Recording has no inference backend, so the recorder itself has
to supply that heartbeat.
"""

import pytest

from inference_service.distributed import (
    EdgeSession,
    FeatureSummary,
    PipelineIdentity,
    PolicySummary,
)
from inference_service.recording_node import RecordingCloudHandshake


def _identity(pipeline_id: str = "policy") -> PipelineIdentity:
    return PipelineIdentity(
        pipeline_id=pipeline_id,
        manifest_schema_version=2,
        bundle_uuid="123e4567-e89b-42d3-a456-426614174000",
        bundle_revision=1,
        bundle_digest="a" * 64,
        deployment_name="recording",
        deployment_uuid="123e4567-e89b-42d3-a456-426614174001",
        deployment_revision=1,
        deployment_fingerprint="b" * 64,
        policy=PolicySummary(
            policy_type="act",
            inputs=(
                FeatureSummary("observation.state", "STATE", (9,)),
                FeatureSummary("observation.images.front", "VISUAL", (3, 480, 640)),
            ),
            outputs=(FeatureSummary("action", "ACTION", (9,)),),
            action_dimension=9,
        ),
    )


def test_recording_handshake_activates_edge_session():
    edge = EdgeSession(_identity())
    edge.start()
    handshake = RecordingCloudHandshake("policy")

    cloud_status = handshake.observe_edge(edge.local_status())
    edge.observe_cloud(cloud_status)

    assert cloud_status.ready
    assert cloud_status.runtime_state == "recording"
    assert edge.ready
    # The session identity is what the edge binds its video streams to; without
    # it descriptors() stays empty and no video is ever sent.
    assert edge.session == (cloud_status.session_id, cloud_status.session_generation)


def test_recording_handshake_rejects_another_pipeline():
    edge = EdgeSession(_identity("other"))
    edge.start()
    handshake = RecordingCloudHandshake("policy")

    with pytest.raises(ValueError, match="recording pipeline mismatch"):
        handshake.observe_edge(edge.local_status())


def test_recording_handshake_reports_no_status_before_any_edge_contact():
    assert RecordingCloudHandshake("policy").status() is None

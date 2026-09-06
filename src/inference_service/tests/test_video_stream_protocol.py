from __future__ import annotations

from dataclasses import replace

import pytest

from inference_service.distributed import (
    PROTOCOL_VERSION,
    CloudSession,
    DistributedCloudService,
    DistributedProtocolError,
    DistributedRequest,
    FeatureSummary,
    Operation,
    PeerRole,
    PipelineIdentity,
    PipelineStatus,
    PolicySummary,
    StreamNegotiationError,
    StreamNegotiationRequirements,
    StreamReference,
    VideoStreamDescriptor,
    VideoStreamNegotiator,
    VideoStreamRequirement,
    VideoStreamRuntimeStatus,
    VideoTransportCapabilities,
    negotiate_video_streams,
)
from inference_service.distributed.ros_protocol import (
    video_descriptor_from_message,
    video_descriptor_to_message,
    video_status_from_message,
    video_status_to_message,
)


def test_protocol_version_is_bumped_for_video_stream_contract():
    # v6 adds the result-level execution_horizon to the distributed wire
    # contract; the video stream transport shares the same protocol version.
    assert PROTOCOL_VERSION == 6


def test_video_descriptor_and_status_ros_messages_round_trip():
    descriptor = _descriptor()
    status = VideoStreamRuntimeStatus(
        protocol_version=PROTOCOL_VERSION,
        pipeline_id="policy",
        session_id="session",
        session_generation=1,
        observation_key="observation.images.top",
        stream_id="top",
        lifecycle_state="ready",
        ready=True,
        selected_backend="software",
        timestamp_mapping_valid=True,
        mapping_rtp_timestamp=90_000,
        mapping_capture_timestamp_ns=1_000_000_000,
        keyframe_ready=True,
        encoded_frames=10,
        decoded_frames=9,
        sent_packets=20,
        received_packets=19,
        dropped_frames=1,
        dropped_packets=2,
        lost_packets=1,
        sender_queue_depth=1,
        receiver_queue_depth=2,
        decoded_buffer_depth=3,
        reconnect_count=1,
    )

    assert video_descriptor_from_message(video_descriptor_to_message(descriptor)) == descriptor
    assert video_status_from_message(video_status_to_message(status)) == status


def test_matching_rtp_negotiation_checks_both_fingerprints():
    negotiated = negotiate_video_streams(_requirements(), _capabilities(), (_descriptor(),))

    assert negotiated["observation.images.top"].contract_fingerprint == "contract"
    assert negotiated["observation.images.top"].deployment_fingerprint == "deployment"


@pytest.mark.parametrize(
    ("field_name", "value", "code"),
    [
        ("protocol_version", 2, "protocol_version_mismatch"),
        ("session_generation", 2, "descriptor_mismatch"),
        ("contract_fingerprint", "wrong", "contract_fingerprint_mismatch"),
        ("deployment_fingerprint", "wrong", "deployment_fingerprint_mismatch"),
        ("stream_id", "wrong", "descriptor_mismatch"),
    ],
)
def test_rtp_negotiation_rejects_descriptor_mismatch(field_name, value, code):
    with pytest.raises(StreamNegotiationError) as error:
        negotiate_video_streams(_requirements(), _capabilities(), (replace(_descriptor(), **{field_name: value}),))

    assert error.value.code == code


def test_explicit_decoder_backend_is_fail_closed_without_transport_fallback():
    requirements = replace(
        _requirements(),
        streams=(VideoStreamRequirement("observation.images.top", "top", decoder_backend="ascend"),),
    )

    with pytest.raises(StreamNegotiationError) as error:
        negotiate_video_streams(requirements, _capabilities(), (_descriptor(),))

    assert error.value.code == "unsupported_backend"


def test_dds_compatibility_negotiates_v3_without_streams_and_rejects_rtp_descriptors():
    requirements = replace(_requirements(), transport_mode="dds", streams=())

    assert negotiate_video_streams(requirements, _capabilities(), ()) == {}
    with pytest.raises(StreamNegotiationError) as error:
        negotiate_video_streams(requirements, _capabilities(), (_descriptor(),))
    assert error.value.code == "descriptor_mismatch"

    negotiator = VideoStreamNegotiator(requirements, _capabilities())
    assert negotiator.ready is True
    negotiator.validate_request(())


def test_old_peer_version_is_rejected_even_for_dds_compatibility():
    requirements = replace(_requirements(), transport_mode="dds", streams=())
    capabilities = replace(_capabilities(), protocol_version=2)

    with pytest.raises(StreamNegotiationError) as error:
        negotiate_video_streams(requirements, capabilities, ())

    assert error.value.code == "protocol_version_mismatch"


def test_late_descriptor_discovery_gates_requests_until_complete():
    requirements = replace(
        _requirements(),
        streams=(
            VideoStreamRequirement("observation.images.top", "top"),
            VideoStreamRequirement("observation.images.wrist", "wrist"),
        ),
    )
    negotiator = VideoStreamNegotiator(requirements, _capabilities())

    assert not negotiator.observe_descriptor(_descriptor())
    with pytest.raises(StreamNegotiationError) as missing:
        negotiator.validate_request((StreamReference("observation.images.top", "top"),))
    assert missing.value.code == "missing_stream"
    assert negotiator.observe_descriptor(
        replace(_descriptor(), observation_key="observation.images.wrist", stream_id="wrist", endpoint_port=5006)
    )
    negotiator.validate_request(
        (
            StreamReference("observation.images.top", "top"),
            StreamReference("observation.images.wrist", "wrist"),
        )
    )


def test_unexpected_late_descriptor_is_rejected_immediately():
    negotiator = VideoStreamNegotiator(_requirements(), _capabilities())

    with pytest.raises(StreamNegotiationError) as error:
        negotiator.observe_descriptor(
            replace(_descriptor(), observation_key="observation.images.wrist", stream_id="wrist")
        )

    assert error.value.code == "missing_stream"


def test_negotiator_reset_invalidates_descriptors_and_old_generation():
    negotiator = VideoStreamNegotiator(_requirements(), _capabilities())
    assert negotiator.observe_descriptor(_descriptor())

    negotiator.reset("new-session", 2)

    assert negotiator.ready is False
    with pytest.raises(StreamNegotiationError) as error:
        negotiator.observe_descriptor(_descriptor())
    assert error.value.code == "descriptor_mismatch"


def test_duplicate_semantic_key_and_request_reference_mismatch_fail_closed():
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            _requirements(),
            streams=(
                VideoStreamRequirement("observation.images.top", "top"),
                VideoStreamRequirement("observation.images.top", "wrist"),
            ),
        )

    negotiator = VideoStreamNegotiator(_requirements(), _capabilities())
    negotiator.observe_descriptor(_descriptor())
    with pytest.raises(StreamNegotiationError) as mismatch:
        negotiator.validate_request((StreamReference("observation.images.top", "wrong"),))
    assert mismatch.value.code == "descriptor_mismatch"


def test_stale_descriptor_does_not_poison_active_session_negotiation():
    negotiator = VideoStreamNegotiator(_requirements(), _capabilities())
    with pytest.raises(StreamNegotiationError) as stale:
        negotiator.observe_descriptor(replace(_descriptor(), session_id="stale-session"))
    assert stale.value.code == "descriptor_mismatch"

    assert negotiator.observe_descriptor(_descriptor())
    assert negotiator.ready


def test_cloud_session_admission_returns_structured_stream_negotiation_error():
    negotiator = VideoStreamNegotiator(_requirements(), _capabilities())
    identity = _identity()
    cloud = CloudSession(identity, request_stream_validator=negotiator.validate_request)
    edge_status = PipelineStatus(
        role=PeerRole.EDGE,
        identity=identity,
        sequence=1,
        runtime_state="handshaking",
    )
    assert cloud.observe_edge(edge_status, backend_ready=True) is None
    status = cloud.status(
        backend_ready=True,
        backend_state="ready",
        reset_supported=True,
        cancellation_supported=True,
    )
    request = DistributedRequest(
        operation=Operation.INFER,
        pipeline_id="policy",
        request_id="request",
        session_id=status.session_id,
        session_generation=status.session_generation,
        deployment_fingerprint="deployment",
        observation_timestamp_ns=1,
        stream_references=(StreamReference("observation.images.top", "top"),),
    )

    with pytest.raises(DistributedProtocolError) as error:
        cloud.validate_request(request)

    assert error.value.code == "missing_stream"
    assert error.value.error.stage == "handshake"


def test_cloud_service_session_rollover_invalidates_old_stream_descriptors():
    negotiator = VideoStreamNegotiator(_requirements(), _capabilities())
    assert negotiator.observe_descriptor(_descriptor())
    identity = _identity()
    service = DistributedCloudService(identity, _Runtime(), stream_negotiator=negotiator)
    edge_status = PipelineStatus(
        role=PeerRole.EDGE,
        identity=identity,
        sequence=1,
        runtime_state="handshaking",
    )

    cloud_status = service.observe_edge(edge_status)

    assert cloud_status.session_id
    assert negotiator.requirements.session_id == cloud_status.session_id
    assert negotiator.requirements.session_generation == cloud_status.session_generation
    assert negotiator.ready is False
    with pytest.raises(StreamNegotiationError) as error:
        negotiator.validate_request((StreamReference("observation.images.top", "top"),))
    assert error.value.code == "missing_stream"


def _requirements() -> StreamNegotiationRequirements:
    return StreamNegotiationRequirements(
        pipeline_id="policy",
        session_id="session",
        session_generation=1,
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        streams=(VideoStreamRequirement("observation.images.top", "top"),),
        transport_mode="rtp",
    )


def _capabilities() -> VideoTransportCapabilities:
    return VideoTransportCapabilities(decoder_backends=("software",))


def _descriptor() -> VideoStreamDescriptor:
    return VideoStreamDescriptor(
        protocol_version=PROTOCOL_VERSION,
        pipeline_id="policy",
        session_id="session",
        session_generation=1,
        observation_key="observation.images.top",
        stream_id="top",
        endpoint_host="127.0.0.1",
        endpoint_port=5004,
        ssrc=123,
        payload_type=96,
        codec="h264",
        codec_profile="main",
        width=640,
        height=480,
        frame_rate_hz=20.0,
        rtp_clock_rate=90_000,
        pixel_format="nv12",
        color_space="bt709",
        color_range="limited",
        encoder_backend="software",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
    )


def _identity() -> PipelineIdentity:
    policy = PolicySummary(
        policy_type="act",
        inputs=(FeatureSummary("observation.state", "state", (6,)),),
        outputs=(FeatureSummary("action", "action", (6,)),),
        action_dimension=6,
    )
    return PipelineIdentity(
        pipeline_id="policy",
        manifest_schema_version=1,
        bundle_uuid="bundle",
        bundle_revision=1,
        bundle_digest="digest",
        deployment_name="cpu",
        deployment_uuid="deployment-uuid",
        deployment_revision=1,
        deployment_fingerprint="deployment",
        policy=policy,
    )


class _Runtime:
    capabilities = type(
        "Capabilities",
        (),
        {"resettable": True, "stateful": True, "supports_cancellation": True},
    )()

    @staticmethod
    def health():
        backend_health = type("BackendHealth", (), {"ready": True})()
        state = type("State", (), {"value": "ready"})()
        return type("Health", (), {"ready": True, "state": state, "backend_health": backend_health})()

    @staticmethod
    def reset(deadline=None):
        return None

    @staticmethod
    def close():
        return None

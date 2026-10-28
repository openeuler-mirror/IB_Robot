from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from inference_service.compute_video_streams import ComputeVideoStreamManager
from inference_service.distributed import PROTOCOL_VERSION, VideoStreamDescriptor, VideoStreamRuntimeStatus
from inference_service.observation_sync import ObservationSynchronizationError
from inference_service.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    CodecMetrics,
    VideoCodecRegistry,
    VideoDecoder,
    VideoFrame,
)
from inference_service.video_rtp import StreamLifecycleState, StreamMetrics, StreamStatus
from robot_config.contract_utils import SpecView
from robot_config.observation_transport import (
    H264Spec,
    ObservationTransportSpec,
    RtpEndpointSpec,
    VideoBufferSpec,
    VideoMediaSpec,
    VideoReadinessSpec,
)


class _Decoder(VideoDecoder):
    @property
    def state(self):
        return CodecLifecycleState.RUNNING

    @property
    def metrics(self):
        return CodecMetrics()

    def decode(self, packet):
        return []

    def reset(self):
        pass

    def close(self, timeout_s=1.0):
        pass


class _Receiver:
    def __init__(self, **options):
        self.options = options
        self.started = False
        self.closed = False

    @property
    def status(self):
        return StreamStatus(
            self.options["stream_id"],
            StreamLifecycleState.READY if self.started else StreamLifecycleState.CONFIGURED,
            self.started,
            self.options["selected_backend"],
            StreamMetrics(),
        )

    def start(self):
        self.started = True

    def close(self, timeout_s=1.0):
        self.closed = True


def _spec(key="observation.images.top", stream_id="top", port=5004):
    return SpecView(
        key=key,
        topic=f"/camera/{stream_id}",
        ros_type="sensor_msgs/msg/Image",
        is_action=False,
        names=[],
        image_resize=(2, 4),
        image_encoding="rgb8",
        image_channels=3,
        resample_policy="hold",
        asof_tol_ms=0,
        max_age_ms=1000,
        stamp_src="header",
        clamp=None,
        safety_behavior=None,
        transport=ObservationTransportSpec(
            mode="rtp",
            stream_id=stream_id,
            endpoint=RtpEndpointSpec("127.0.0.1", port),
            h264=H264Spec(),
            encoder_backend="software",
            decoder_backend="software",
            media=VideoMediaSpec(4, 2, 30.0),
            buffer=VideoBufferSpec(),
            readiness=VideoReadinessSpec(max_inter_camera_skew_ms=10),
        ),
    )


def _descriptor(spec, session_id="session", generation=1, ssrc=123):
    transport = spec.transport
    return VideoStreamDescriptor(
        protocol_version=PROTOCOL_VERSION,
        pipeline_id="policy",
        session_id=session_id,
        session_generation=generation,
        observation_key=spec.key,
        stream_id=transport.stream_id,
        endpoint_host=transport.endpoint.host,
        endpoint_port=transport.endpoint.port,
        ssrc=ssrc,
        payload_type=96,
        codec="h264",
        codec_profile="main",
        width=4,
        height=2,
        frame_rate_hz=30.0,
        rtp_clock_rate=90_000,
        pixel_format="nv12",
        color_space="bt709",
        color_range="limited",
        encoder_backend="software",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
    )


def _manager(specs, *, n_obs_steps=1):
    registry = VideoCodecRegistry()
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        decoder_factory=lambda **_options: _Decoder(),
    )
    receivers = []

    def receiver_factory(**options):
        receiver = _Receiver(**options)
        receivers.append(receiver)
        return receiver

    manager = ComputeVideoStreamManager(
        pipeline_id="policy",
        session_id="session",
        session_generation=1,
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=specs,
        rate_hz=30.0,
        n_obs_steps=n_obs_steps,
        codec_registry=registry,
        receiver_factory=receiver_factory,
    )
    return manager, receivers


def _status(descriptor, capture_timestamp_ns, *, encoded_frames=1):
    """Build a sender-originated status for the compute side to observe.

    ``encoded_frames`` defaults to 1 because the compute manager only adopts an
    advertised timestamp mapping once the sender reports having encoded a frame.
    Pass 0 to model a sender that has announced a mapping but not yet produced
    one, which is what a receiver sees before the first frame.
    """
    return VideoStreamRuntimeStatus(
        protocol_version=PROTOCOL_VERSION,
        pipeline_id=descriptor.pipeline_id,
        session_id=descriptor.session_id,
        session_generation=descriptor.session_generation,
        observation_key=descriptor.observation_key,
        stream_id=descriptor.stream_id,
        lifecycle_state="ready",
        ready=True,
        selected_backend="software",
        # The compute side only accepts sender-originated status: see
        # ComputeVideoStreamManager.observe_status, which drops anything else.
        status_origin="sender",
        timestamp_mapping_valid=True,
        encoded_frames=encoded_frames,
        mapping_rtp_timestamp=90_000,
        mapping_capture_timestamp_ns=capture_timestamp_ns,
        keyframe_ready=True,
    )


def test_compute_stream_manager_reconstructs_canonical_tensor_after_mapping():
    spec = _spec()
    manager, receivers = _manager((spec,))
    descriptor = _descriptor(spec)
    assert manager.observe_descriptor(descriptor)
    assert receivers[0].started
    assert manager.observe_descriptor(descriptor)
    assert len(receivers) == 1
    capture_timestamp_ns = 1_000_000_000
    assert manager.observe_status(_status(descriptor, capture_timestamp_ns), receive_time_ns=capture_timestamp_ns)
    frame = VideoFrame(
        np.full((2, 4, 3), [32, 64, 128], dtype=np.uint8),
        capture_timestamp_ns,
        capture_timestamp_ns,
        4,
        2,
        "rgb24",
    )
    receivers[0].options["frame_buffer"].push(
        capture_timestamp_ns,
        frame,
        receive_time_ns=capture_timestamp_ns,
    )

    inputs = manager.assemble_inputs(capture_timestamp_ns, now_ns=capture_timestamp_ns)

    assert inputs[spec.key].shape == (3, 2, 4)
    assert inputs[spec.key].dtype == np.float32
    np.testing.assert_allclose(inputs[spec.key][:, 0, 0], np.array([32, 64, 128]) / 255.0)


def test_compute_stream_manager_enforces_multi_camera_skew_and_session_reset():
    top = _spec()
    wrist = _spec("observation.images.wrist", "wrist", 5006)
    manager, receivers = _manager((top, wrist))
    top_descriptor = _descriptor(top)
    wrist_descriptor = _descriptor(wrist, ssrc=456)
    assert not manager.observe_descriptor(top_descriptor)
    assert manager.observe_descriptor(wrist_descriptor)
    for receiver, descriptor, timestamp in zip(
        receivers,
        (top_descriptor, wrist_descriptor),
        (1_000_000_000, 1_020_000_000),
        strict=True,
    ):
        manager.observe_status(_status(descriptor, timestamp), receive_time_ns=1_020_000_000)
        receiver.options["frame_buffer"].push(
            timestamp,
            VideoFrame(np.zeros((2, 4, 3), dtype=np.uint8), timestamp, timestamp, 4, 2, "rgb24"),
            receive_time_ns=timestamp,
        )

    with pytest.raises(ObservationSynchronizationError) as error:
        manager.assemble_inputs(1_020_000_000, now_ns=1_020_000_000)
    assert error.value.details["streams"][0]["reason"] == "skewed"

    manager.reset_session("new-session", 2)
    assert all(receiver.closed for receiver in receivers)
    assert not manager.negotiator.ready
    assert manager.observe_descriptor(replace(_descriptor(top, "new-session", 2), ssrc=789)) is False


def test_compute_stream_manager_starts_one_receiver_set_for_concurrent_descriptors():
    top = _spec()
    wrist = _spec("observation.images.wrist", "wrist", 5006)
    manager, receivers = _manager((top, wrist))
    assert not manager.observe_descriptor(_descriptor(top))

    descriptor = _descriptor(wrist, ssrc=456)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(manager.observe_descriptor, (descriptor, descriptor)))

    assert results == (True, True)
    assert len(receivers) == 2
    assert all(receiver.started and not receiver.closed for receiver in receivers)


def test_compute_stream_manager_assigns_unique_ascend_decoder_channels():
    top_spec = _spec()
    wrist_spec = _spec("observation.images.wrist", "wrist", 5006)
    top = replace(top_spec, transport=replace(top_spec.transport, decoder_backend="ascend"))
    wrist = replace(wrist_spec, transport=replace(wrist_spec.transport, decoder_backend="ascend"))
    decoder_options = []
    registry = VideoCodecRegistry()
    registry.register(
        "ascend",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        decoder_factory=lambda **options: decoder_options.append(options) or _Decoder(),
    )
    manager = ComputeVideoStreamManager(
        pipeline_id="policy",
        session_id="session",
        session_generation=1,
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(wrist, top),
        rate_hz=30.0,
        codec_registry=registry,
        receiver_factory=_Receiver,
    )

    assert not manager.observe_descriptor(replace(_descriptor(top), encoder_backend="software"))
    assert manager.observe_descriptor(replace(_descriptor(wrist, ssrc=456), encoder_backend="software"))

    assert sorted(options["channel_id"] for options in decoder_options) == [1, 2]


def test_recording_only_manager_skips_decoder_creation_and_wires_recorder():
    spec = _spec()
    decoder_options = []
    registry = VideoCodecRegistry()
    registry.register(
        "software",
        priority=0,
        probe=lambda _kind: CodecCapabilities(pixel_formats=("rgb24",)),
        decoder_factory=lambda **options: decoder_options.append(options) or _Decoder(),
    )

    class Coordinator:
        def __init__(self):
            self.recorders = {}

        def register_recorder(self, key, recorder):
            self.recorders[key] = recorder

    coordinator = Coordinator()
    receivers = []

    def receiver_factory(**options):
        receivers.append(_Receiver(**options))
        return receivers[-1]

    manager = ComputeVideoStreamManager(
        pipeline_id="recording",
        session_id="session",
        session_generation=1,
        contract_fingerprint="contract",
        deployment_fingerprint="recording",
        observation_specs=(spec,),
        rate_hz=30.0,
        codec_registry=registry,
        receiver_factory=receiver_factory,
        recording_coordinator=coordinator,
        decode=False,
        validate_deployment_fingerprint=False,
    )
    descriptor = replace(_descriptor(spec), pipeline_id="recording")
    assert manager.observe_descriptor(descriptor)

    assert decoder_options == []
    assert receivers[0].options["decode"] is False
    assert receivers[0].options["recorder"] is coordinator.recorders[spec.key]


def test_compute_stream_manager_pads_multi_step_history_from_first_keyframe():
    spec = _spec()
    manager, receivers = _manager((spec,), n_obs_steps=3)
    descriptor = _descriptor(spec)
    assert manager.observe_descriptor(descriptor)
    capture_timestamp_ns = 1_000_000_000
    manager.observe_status(_status(descriptor, capture_timestamp_ns), receive_time_ns=capture_timestamp_ns)
    frame = VideoFrame(
        np.full((2, 4, 3), 64, dtype=np.uint8),
        capture_timestamp_ns,
        capture_timestamp_ns,
        4,
        2,
        "rgb24",
    )
    receivers[0].options["frame_buffer"].push(
        capture_timestamp_ns,
        frame,
        receive_time_ns=capture_timestamp_ns,
    )

    inputs = manager.assemble_inputs(capture_timestamp_ns, now_ns=capture_timestamp_ns)

    assert inputs[spec.key].shape == (1, 3, 3, 2, 4)
    np.testing.assert_array_equal(inputs[spec.key][0, 0], inputs[spec.key][0, 2])


def test_compute_stream_diagnostic_snapshots_are_sorted_and_require_mapping_for_readiness():
    top = _spec()
    wrist = _spec("observation.images.wrist", "wrist", 5006)
    manager, receivers = _manager((wrist, top))

    configured = manager.diagnostic_snapshots()
    assert [snapshot.observation_key for snapshot in configured] == [top.key, wrist.key]
    assert all(snapshot.lifecycle_state == "configured" and not snapshot.ready for snapshot in configured)
    assert all(snapshot.selected_encoder_backend == "pending" for snapshot in configured)
    assert all(snapshot.selected_decoder_backend == "software" for snapshot in configured)
    assert all(snapshot.security == "none/trusted-network-only" for snapshot in configured)

    top_descriptor = _descriptor(top)
    wrist_descriptor = _descriptor(wrist, ssrc=456)
    assert not manager.observe_descriptor(top_descriptor)
    assert manager.observe_descriptor(wrist_descriptor)
    assert all(receiver.started for receiver in receivers)
    assert not any(snapshot.ready for snapshot in manager.diagnostic_snapshots())

    manager.observe_status(_status(top_descriptor, 1_000_000_000), receive_time_ns=1_000_000_000)
    readiness = {snapshot.observation_key: snapshot.ready for snapshot in manager.diagnostic_snapshots()}
    assert readiness == {top.key: True, wrist.key: False}


def test_compute_stream_status_reports_receiver_metrics():
    spec = _spec()
    manager, receivers = _manager((spec,))
    descriptor = _descriptor(spec)
    assert manager.observe_descriptor(descriptor)
    capture_timestamp_ns = 1_000_000_000
    manager.observe_status(
        _status(descriptor, capture_timestamp_ns, encoded_frames=0), receive_time_ns=capture_timestamp_ns
    )
    receivers[0].options["frame_buffer"].push(
        capture_timestamp_ns,
        VideoFrame(np.zeros((2, 4, 3), dtype=np.uint8), capture_timestamp_ns, capture_timestamp_ns, 4, 2, "rgb24"),
        receive_time_ns=capture_timestamp_ns,
    )

    status = manager.statuses()[0]

    assert status.pipeline_id == "policy"
    assert status.selected_backend == "software"
    assert not status.timestamp_mapping_valid
    assert status.decoded_buffer_depth == 1

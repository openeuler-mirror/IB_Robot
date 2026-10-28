from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.compute_video_streams import ComputeVideoStreamManager
from inference_service.device_video_streams import DeviceVideoStreamManager
from inference_service.distributed import (
    DistributedCloudService,
    EdgeSession,
    FeatureSummary,
    Operation,
    PipelineIdentity,
    PolicySummary,
)
from inference_service.video_rtp import H264RtpReceiver, H264RtpSender
from robot_config.contract_utils import SpecView
from robot_config.observation_transport import (
    H264Spec,
    ObservationTransportSpec,
    RtpEndpointSpec,
    VideoBufferSpec,
    VideoMediaSpec,
    VideoReadinessSpec,
)


class _MemoryDatagramSender:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._datagrams: list[bytes] = []

    def sendto(self, data: bytes, _endpoint: tuple[str, int]) -> int:
        with self._lock:
            self._datagrams.append(data)
        return len(data)

    def drain(self) -> list[bytes]:
        with self._lock:
            datagrams = self._datagrams
            self._datagrams = []
        return datagrams

    def close(self) -> None:
        pass


class _Runtime:
    capabilities = SimpleNamespace(resettable=True, stateful=True, supports_cancellation=False)

    def __init__(self) -> None:
        self.inputs = None
        self.reset_calls = 0

    @staticmethod
    def health():
        return SimpleNamespace(
            state=SimpleNamespace(value="ready"),
            backend_health=SimpleNamespace(ready=True),
        )

    def infer(self, _request_id, inputs, *, prompt=None, deadline=None):
        self.inputs = inputs
        return SimpleNamespace(
            action=np.zeros((1, 6), dtype=np.float32),
            actual_chunk_size=1,
            backend_latency_ms=1.0,
        )

    def reset(self, deadline=None):
        self.reset_calls += 1

    def cancel(self, request_id, deadline=None):
        raise AssertionError("cancellation is not expected")

    def close(self):
        pass


def test_software_video_stream_round_trip_merges_dds_state_and_returns_action():
    spec = _video_spec()
    identity = _identity()
    runtime = _Runtime()
    datagrams = _MemoryDatagramSender()
    receivers = []

    def sender_factory(**options):
        return H264RtpSender(**options, datagram_sender=datagrams, initial_sequence=10)

    def receiver_factory(**options):
        options.pop("endpoint")
        receiver = H264RtpReceiver(**options)
        receivers.append(receiver)
        return receiver

    compute_streams = ComputeVideoStreamManager(
        pipeline_id="policy",
        session_id="pending",
        session_generation=1,
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(spec,),
        rate_hz=20.0,
        receiver_factory=receiver_factory,
    )
    cloud = DistributedCloudService(identity, runtime, stream_manager=compute_streams)
    edge_session = EdgeSession(identity)
    edge_session.start()
    cloud_status = cloud.observe_edge(edge_session.local_status())
    edge_session.observe_cloud(cloud_status)
    session_id, generation = edge_session.session

    device_streams = DeviceVideoStreamManager(
        pipeline_id="policy",
        contract_fingerprint="contract",
        deployment_fingerprint="deployment",
        observation_specs=(spec,),
        sender_factory=sender_factory,
    )
    device_streams.bind_session(session_id, generation)
    descriptor = device_streams.descriptors()[0]
    assert compute_streams.observe_descriptor(descriptor)

    capture_timestamp_ns = time.time_ns()
    assert device_streams.submit_ros_image(
        spec.key,
        _image_message(),
        capture_timestamp_ns=capture_timestamp_ns,
        receive_timestamp_ns=capture_timestamp_ns,
    )
    _wait_until(lambda: device_streams.statuses()[0].timestamp_mapping_valid)
    sender_status = device_streams.statuses()[0]
    assert compute_streams.observe_status(sender_status, receive_time_ns=time.time_ns())
    for datagram in datagrams.drain():
        receivers[0].process_datagram(datagram, receive_time_ns=time.time_ns())
    assert receivers[0].status.ready

    state = np.arange(6, dtype=np.float32)[None, :]
    request = edge_session.prepare_request(
        Operation.INFER,
        "mixed-infer",
        inputs={"observation.state": state},
        observation_timestamp_ns=capture_timestamp_ns,
        stream_references=device_streams.stream_references,
    )
    result = cloud.handle(request)

    assert result.success
    assert result.actual_chunk_size == 1
    assert np.array_equal(result.action, np.zeros((1, 6), dtype=np.float32))
    assert np.array_equal(runtime.inputs["observation.state"], state)
    image = runtime.inputs[spec.key]
    assert image.shape == (3, 48, 64)
    assert image.dtype == np.float32
    assert image.flags.c_contiguous
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0

    # Losing the configured RTP frame must fail closed even if a caller supplies
    # a same-key DDS tensor that could otherwise mask the interruption.
    receivers[0].frame_buffer.reset()
    with pytest.raises(ValueError, match="collide with tensor input semantics"):
        edge_session.prepare_request(
            Operation.INFER,
            "dds-fallback",
            inputs={"observation.state": state, spec.key: np.ones_like(image)},
            observation_timestamp_ns=capture_timestamp_ns,
            stream_references=device_streams.stream_references,
        )
    interrupted = edge_session.prepare_request(
        Operation.INFER,
        "interrupted-infer",
        inputs={"observation.state": state},
        observation_timestamp_ns=capture_timestamp_ns,
        stream_references=device_streams.stream_references,
    )
    interrupted_result = cloud.handle(interrupted)

    assert not interrupted_result.success
    assert interrupted_result.error.code == "observation_not_ready"
    assert interrupted_result.error.stage == "backend"
    assert interrupted_result.error.details["streams"][0]["reason"] == "missing"
    assert runtime.inputs[spec.key] is image

    device_streams.close()
    cloud.close()


def _wait_until(predicate, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def _image_message():
    y, x = np.indices((48, 64))
    image = np.stack((x * 4, y * 5, (x + y) * 2), axis=-1).astype(np.uint8)
    return SimpleNamespace(width=64, height=48, step=64 * 3, encoding="rgb8", data=image.tobytes())


def _video_spec():
    return SpecView(
        key="observation.images.top",
        topic="/camera/top",
        ros_type="sensor_msgs/msg/Image",
        is_action=False,
        names=[],
        image_resize=(48, 64),
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
            stream_id="top",
            endpoint=RtpEndpointSpec("127.0.0.1", 5004),
            h264=H264Spec(bitrate_bps=300_000, gop_frames=5),
            encoder_backend="software",
            decoder_backend="software",
            media=VideoMediaSpec(64, 48, 20.0),
            buffer=VideoBufferSpec(),
            readiness=VideoReadinessSpec(),
        ),
    )


def _identity():
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
        policy=PolicySummary(
            policy_type="act",
            inputs=(
                FeatureSummary("observation.state", "state", (6,)),
                FeatureSummary("observation.images.top", "visual", (3, 48, 64)),
            ),
            outputs=(FeatureSummary("action", "action", (6,)),),
            action_dimension=6,
        ),
    )

from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from inference_service.observation_sync import (
    ObservationSynchronizationError,
    RtpTimestampMapper,
    StreamSelection,
    select_synchronized_streams,
)
from inference_service.software_video_codec import SoftwareH264Decoder, SoftwareH264Encoder
from inference_service.video_codec import VideoFrame
from inference_service.video_rtp import H264RtpReceiver, H264RtpSender, StreamLifecycleState
from robot_config.contract_utils import StreamBuffer, iter_specs
from robot_config.generators.contract import build_contract_from_robot_config_dict
from tensormsg.converter import TensorMsgConverter, decoded_frame_to_chw_float

pytest.importorskip("av")

_CAPTURE_NS = 1_000_000_000
_FRAME_PERIOD_NS = 50_000_000
_SSRC = 0x1122_3344


class Image:
    __module__ = "sensor_msgs.msg._image"

    def __init__(self, frame: np.ndarray) -> None:
        self.height, self.width, _ = frame.shape
        self.encoding = "rgb8"
        self.step = self.width * 3
        self.is_bigendian = False
        self.data = frame.tobytes()


class _MemoryDatagrams:
    def __init__(self) -> None:
        self.datagrams: list[bytes] = []

    def sendto(self, payload: bytes, _endpoint: tuple[str, int]) -> int:
        self.datagrams.append(payload)
        return len(payload)

    def close(self) -> None:
        pass


def test_same_frame_dds_and_software_h264_rtp_paths_preserve_semantics_action_and_efficiency():
    source = _source_frame()
    message = Image(source)

    dds_started = time.perf_counter()
    dds_tensor = TensorMsgConverter.decode(message, _image_spec())
    dds_latency_s = time.perf_counter() - dds_started

    encoder = _encoder(gop_frames=5)
    decoder = SoftwareH264Decoder()
    datagrams = _MemoryDatagrams()
    sender = H264RtpSender(
        stream_id="top",
        endpoint=("127.0.0.1", 5004),
        ssrc=_SSRC,
        queue_capacity=2,
        datagram_sender=datagrams,
        initial_sequence=10,
    )
    buffer = StreamBuffer("hold", _FRAME_PERIOD_NS, max_age_ns=1_000_000_000, retention_ns=1_000_000_000)
    mapper = RtpTimestampMapper(1_000_000_000, observation_key="observation.images.top", stream_id="top")
    receiver = H264RtpReceiver(
        stream_id="top",
        observation_key="observation.images.top",
        ssrc=_SSRC,
        decoder=decoder,
        frame_buffer=buffer,
        timestamp_mapper=mapper,
        session_generation=1,
        packet_queue_capacity=16,
    )
    receiver.start()
    mapper.update(90_000, _CAPTURE_NS, _CAPTURE_NS, session_generation=1)

    rtp_started = time.perf_counter()
    encoded = encoder.encode(VideoFrame(source, _CAPTURE_NS, _CAPTURE_NS, 64, 48, "rgb24"))
    for access_unit in encoded:
        sender.enqueue(access_unit)
        sender.send_pending()
    decoded = []
    for index, datagram in enumerate(datagrams.datagrams):
        decoded.extend(receiver.process_datagram(datagram, receive_time_ns=_CAPTURE_NS + index * 1000))
    decoded_encoding = {"rgb24": "rgb8", "bgr24": "bgr8"}[decoded[-1].pixel_format]
    rtp_tensor = decoded_frame_to_chw_float(decoded[-1].data, encoding=decoded_encoding)
    rtp_latency_s = time.perf_counter() - rtp_started

    assert dds_tensor.shape == rtp_tensor.shape == (3, 48, 64)
    assert dds_tensor.dtype == rtp_tensor.dtype == np.float32
    assert np.argmax(dds_tensor.mean(axis=(1, 2))) == np.argmax(rtp_tensor.mean(axis=(1, 2))) == 1
    assert np.mean(np.abs(dds_tensor - rtp_tensor)) < 0.03
    np.testing.assert_allclose(_fake_model_action(rtp_tensor), _fake_model_action(dds_tensor), atol=0.03)
    assert sum(len(packet.payload) for packet in encoded) < len(message.data)
    assert dds_latency_s < 1.0
    assert rtp_latency_s < 2.0

    encoder.close()
    sender.close()
    receiver.close()


def test_three_camera_fixture_enforces_skew_and_bounds_each_retained_history():
    fixture = Path(__file__).with_name("fixtures") / "three_camera_transport.yaml"
    contract = build_contract_from_robot_config_dict(yaml.safe_load(fixture.read_text(encoding="utf-8")))
    specs = tuple(iter_specs(contract))

    assert len(specs) == 3
    assert {spec.transport.stream_id for spec in specs} == {"top", "left_wrist", "right_wrist"}
    assert len({spec.transport.endpoint.port for spec in specs}) == 3
    assert all(spec.transport.buffer.sender_queue_frames == 2 for spec in specs)
    assert all(spec.transport.buffer.receiver_queue_packets == 16 for spec in specs)

    streams = {}
    for index, spec in enumerate(specs):
        buffer = StreamBuffer("hold", _FRAME_PERIOD_NS, retention_ns=200_000_000)
        for frame_index in range(30):
            timestamp = _CAPTURE_NS + frame_index * _FRAME_PERIOD_NS + index * 5_000_000
            buffer.push(timestamp, (index, frame_index), receive_time_ns=timestamp)
        assert len(buffer) <= 5
        streams[spec.key] = StreamSelection(spec.key, spec.transport.stream_id, buffer, True, True)

    target_ns = _CAPTURE_NS + 29 * _FRAME_PERIOD_NS + 10_000_000
    selected = select_synchronized_streams(
        streams,
        target_ns,
        now_ns=target_ns,
        max_inter_camera_skew_ns=15_000_000,
    )
    assert len(selected) == 3

    with pytest.raises(ObservationSynchronizationError) as error:
        select_synchronized_streams(
            streams,
            target_ns,
            now_ns=target_ns,
            max_inter_camera_skew_ns=9_000_000,
        )
    assert {issue["reason"] for issue in error.value.details["streams"]} == {"skewed"}
    assert {issue["skew_ns"] for issue in error.value.details["streams"]} == {10_000_000}


def test_sustained_localhost_udp_stream_recovers_in_new_session_epoch():
    udp_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_receiver.bind(("127.0.0.1", 0))
    endpoint = udp_receiver.getsockname()
    buffer = StreamBuffer("hold", _FRAME_PERIOD_NS, max_age_ns=2_000_000_000, retention_ns=300_000_000)
    mapper = RtpTimestampMapper(2_000_000_000, observation_key="observation.images.top", stream_id="top")
    receiver = H264RtpReceiver(
        stream_id="top",
        observation_key="observation.images.top",
        ssrc=_SSRC,
        decoder=SoftwareH264Decoder(),
        frame_buffer=buffer,
        timestamp_mapper=mapper,
        session_generation=1,
        packet_queue_capacity=32,
        endpoint=endpoint,
        datagram_receiver=udp_receiver,
    )
    sender = H264RtpSender(stream_id="top", endpoint=endpoint, ssrc=_SSRC, queue_capacity=2)
    encoder = _encoder(gop_frames=4)
    receiver.start()
    sender.start()
    mapper.update(90_000, _CAPTURE_NS, time.time_ns(), session_generation=1)

    _send_frames(sender, encoder, start_ns=_CAPTURE_NS, count=16)
    _wait_until(lambda: receiver.status.metrics.decoded_frames >= 12)

    assert receiver.status.state is StreamLifecycleState.READY
    assert receiver.status.metrics.decoded_frames >= 12
    assert len(buffer) <= 7
    old_values = tuple(item[2].capture_timestamp_ns for item in buffer.history)

    receiver.reset(2)
    encoder.reset()
    assert not buffer.history
    assert not mapper.ready
    new_epoch_ns = 5_000_000_000
    mapper.update(90_000, new_epoch_ns, time.time_ns(), session_generation=2)
    _send_frames(sender, encoder, start_ns=new_epoch_ns, count=6)
    _wait_until(lambda: bool(buffer.history))

    assert all(item[2].capture_timestamp_ns >= new_epoch_ns for item in buffer.history)
    assert not set(old_values) & {item[2].capture_timestamp_ns for item in buffer.history}
    encoder.close()
    sender.close()
    receiver.close()


def _source_frame() -> np.ndarray:
    y, x = np.indices((48, 64))
    return np.stack((32 + x, 96 + y, 16 + (x + y) // 2), axis=-1).astype(np.uint8)


def _image_spec():
    fixture = Path(__file__).with_name("fixtures") / "three_camera_transport.yaml"
    contract = build_contract_from_robot_config_dict(yaml.safe_load(fixture.read_text(encoding="utf-8")))
    return next(iter(iter_specs(contract)))


def _fake_model_action(tensor: np.ndarray) -> np.ndarray:
    channel_means = tensor.mean(axis=(1, 2), dtype=np.float64)
    return np.asarray([*channel_means, channel_means[0] - channel_means[2]], dtype=np.float32)


def _encoder(*, gop_frames: int) -> SoftwareH264Encoder:
    return SoftwareH264Encoder(
        width=64,
        height=48,
        frame_rate_hz=20.0,
        bitrate_bps=300_000,
        gop_frames=gop_frames,
    )


def _send_frames(sender: H264RtpSender, encoder: SoftwareH264Encoder, *, start_ns: int, count: int) -> None:
    for index in range(count):
        timestamp_ns = start_ns + index * _FRAME_PERIOD_NS
        frame = np.full((48, 64, 3), (index * 7) % 200, dtype=np.uint8)
        for access_unit in encoder.encode(VideoFrame(frame, timestamp_ns, timestamp_ns, 64, 48, "rgb24")):
            sender.enqueue(access_unit)
        _wait_until(lambda: sender.status.metrics.queued_frames == 0)


def _wait_until(predicate, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")

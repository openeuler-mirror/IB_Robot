from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.ascend_ffmpeg_video_codec import (
    AscendFfmpegH264Decoder,
    AscendFfmpegH264Encoder,
    _FixedFrameReader,
    build_ascend_child_environment,
    probe_ascend_codec_diagnostic,
    register_ascend_backend,
    resolve_ascend_ffmpeg,
)
from inference_service.video_codec import (
    CodecLifecycleState,
    EncodedPacket,
    VideoCodecError,
    VideoCodecRegistry,
    VideoFrame,
)
from inference_service.video_rtp import RtpPacket


class _QueuePipe:
    def __init__(self) -> None:
        self._chunks: deque[bytes] = deque()
        self._closed = False
        self._condition = threading.Condition()
        self.read_sizes: list[int] = []

    def put(self, payload: bytes) -> None:
        with self._condition:
            self._chunks.append(payload)
            self._condition.notify_all()

    def read(self, _size: int = -1) -> bytes:
        with self._condition:
            self.read_sizes.append(_size)
            while not self._chunks and not self._closed:
                self._condition.wait()
            return self._chunks.popleft() if self._chunks else b""

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class _InputPipe(io.BytesIO):
    def __init__(self, on_write=None) -> None:
        super().__init__()
        self._on_write = on_write

    def write(self, payload: bytes) -> int:
        result = super().write(payload)
        if self._on_write is not None:
            self._on_write(payload)
        return result


class _FakeProcess:
    def __init__(self, *, stdin=None, stdout=None, stderr=None) -> None:
        self.stdin = stdin or _InputPipe()
        self.stdout = stdout
        self.stderr = stderr or io.BytesIO()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        if self.stdout is not None:
            self.stdout.close()
        return 0

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9


class _FakeSocket:
    def __init__(self) -> None:
        self.datagrams: deque[bytes] = deque()
        self.bound = None
        self.timeout = None
        self.closed = False

    def bind(self, address):
        self.bound = address

    def getsockname(self):
        return ("127.0.0.1", 24000)

    def settimeout(self, timeout):
        self.timeout = timeout

    def recvfrom(self, _size):
        if self.closed:
            raise OSError("closed")
        if not self.datagrams:
            raise TimeoutError("empty")
        return self.datagrams.popleft(), ("127.0.0.1", 10000)

    def sendto(self, data, _endpoint):
        self.datagrams.append(data)
        return len(data)

    def close(self):
        self.closed = True


class _DrainProbeSocket(_FakeSocket):
    """Fail if drain waits again after receiving a complete access unit."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)
        super().settimeout(timeout)

    def recvfrom(self, size):
        if self.datagrams:
            return super().recvfrom(size)
        if self.timeout == 0:
            raise BlockingIOError
        raise AssertionError("drain waited after a complete access unit")


class _DrainTimeoutProbeSocket(_FakeSocket):
    """Record settimeout values to verify drain wait/skip decisions."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float | None] = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)
        super().settimeout(timeout)


def test_module_import_has_no_pyav_or_ascend_dependency():
    script = (
        "import sys; "
        "import inference_service.ascend_ffmpeg_video_codec; "
        "assert 'av' not in sys.modules; "
        "assert not any(name.startswith('acl') for name in sys.modules)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)

    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_ffmpeg_resolution_order_and_child_only_library_path(tmp_path: Path):
    explicit = _executable(tmp_path / "explicit" / "ffmpeg")
    prefix = tmp_path / "private"
    prefixed = _executable(prefix / "bin" / "ffmpeg")
    (prefix / "lib").mkdir()
    environment = {
        "IBROBOT_ASCEND_FFMPEG": str(explicit),
        "IBROBOT_ASCEND_FFMPEG_PREFIX": str(prefix),
        "LD_LIBRARY_PATH": "/inherited",
    }

    assert resolve_ascend_ffmpeg(environment) == str(explicit)
    explicit.unlink()
    assert resolve_ascend_ffmpeg(environment) == str(prefixed)
    child = build_ascend_child_environment(str(prefixed), environment)

    assert child["LD_LIBRARY_PATH"] == f"{prefix / 'lib'}:/inherited"
    assert environment["LD_LIBRARY_PATH"] == "/inherited"


def test_child_environment_can_isolate_ascend_toolkit_variables(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "private" / "bin" / "ffmpeg")
    environment = {
        "IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV": "1",
        "ASCEND_HOME_PATH": "/toolkit",
        "ASCEND_OPP_PATH": "/opp",
        "TOOLCHAIN_HOME": "/toolchain",
    }

    child = build_ascend_child_environment(str(ffmpeg), environment)

    assert "ASCEND_HOME_PATH" not in child
    assert "ASCEND_OPP_PATH" not in child
    assert "TOOLCHAIN_HOME" not in child


def test_standard_ascend_wrapper_isolated_by_default(tmp_path: Path):
    del tmp_path
    wrapper = Path("/usr/bin/ffmpeg-ascend")
    environment = {
        "ASCEND_TOOLKIT_HOME": "/cann-8.3",
        "ASCEND_OPP_PATH": "/cann-8.3/opp",
        "TOOLCHAIN_HOME": "/toolchain",
        "LD_LIBRARY_PATH": "/inherited",
        "PYTHONPATH": "/python",
    }

    child = build_ascend_child_environment(str(wrapper), environment)

    assert "ASCEND_TOOLKIT_HOME" not in child
    assert "ASCEND_OPP_PATH" not in child
    assert "TOOLCHAIN_HOME" not in child
    assert child["LD_LIBRARY_PATH"] == "/inherited"
    assert child["PYTHONPATH"] == "/python"
    assert environment["ASCEND_TOOLKIT_HOME"] == "/cann-8.3"


def test_standard_ascend_wrapper_can_opt_out_of_default_isolation(tmp_path: Path):
    del tmp_path
    wrapper = Path("/usr/bin/ffmpeg-ascend")
    environment = {
        "IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV": "0",
        "ASCEND_TOOLKIT_HOME": "/cann-8.3",
        "TOOLCHAIN_HOME": "/toolchain",
    }

    child = build_ascend_child_environment(str(wrapper), environment)

    assert child["ASCEND_TOOLKIT_HOME"] == "/cann-8.3"
    assert child["TOOLCHAIN_HOME"] == "/toolchain"


def test_private_ffmpeg_keeps_existing_environment_without_isolation(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "private" / "bin" / "ffmpeg")
    environment = {
        "ASCEND_TOOLKIT_HOME": "/cann-8.3",
        "TOOLCHAIN_HOME": "/toolchain",
    }

    child = build_ascend_child_environment(str(ffmpeg), environment)

    assert child["ASCEND_TOOLKIT_HOME"] == "/cann-8.3"
    assert child["TOOLCHAIN_HOME"] == "/toolchain"


def test_rpm_payload_ffmpeg_is_isolated_by_default_and_keeps_device_env(tmp_path: Path):
    """The versioned payload binary behind the RPM wrapper gets the same
    default isolation as the wrapper itself: probing it unisolated was shown
    to hang on the first frame on the real board."""
    del tmp_path
    payload = Path("/usr/local/ffmpeg-ascend-611/bin/ffmpeg")
    environment = {
        "ASCEND_TOOLKIT_HOME": "/cann-8.3",
        "ASCEND_OPP_PATH": "/cann-8.3/opp",
        "TOOLCHAIN_HOME": "/toolchain",
        "ASCEND_RT_VISIBLE_DEVICES": "1",
        "ASCEND_DEVICE_ID": "1",
    }

    child = build_ascend_child_environment(str(payload), environment)

    assert "ASCEND_TOOLKIT_HOME" not in child
    assert "ASCEND_OPP_PATH" not in child
    assert "TOOLCHAIN_HOME" not in child
    # Device visibility and selection policy are runtime resource
    # configuration, not installation paths; isolation must never strip
    # them or multi-NPU deployments lose their device pinning.
    assert child["ASCEND_RT_VISIBLE_DEVICES"] == "1"
    assert child["ASCEND_DEVICE_ID"] == "1"

    environment["IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV"] = "0"
    child = build_ascend_child_environment(str(payload), environment)
    assert child["ASCEND_TOOLKIT_HOME"] == "/cann-8.3"


def test_wrapper_isolation_preserves_ascend_device_variables(tmp_path: Path):
    del tmp_path
    wrapper = Path("/usr/bin/ffmpeg-ascend")
    environment = {
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "ASCEND_DEVICE_ID": "0",
        "ASCEND_TOOLKIT_HOME": "/cann-8.3",
    }

    child = build_ascend_child_environment(str(wrapper), environment)

    assert child["ASCEND_RT_VISIBLE_DEVICES"] == "0,1"
    assert child["ASCEND_DEVICE_ID"] == "0"
    assert "ASCEND_TOOLKIT_HOME" not in child


def test_standard_ascend_wrapper_honors_explicit_prefix(tmp_path: Path):
    wrapper = Path("/usr/bin/ffmpeg-ascend")
    prefix = tmp_path / "explicit-ffmpeg"
    (prefix / "lib").mkdir(parents=True)
    environment = {
        "IBROBOT_ASCEND_FFMPEG_PREFIX": str(prefix),
        "ASCEND_TOOLKIT_HOME": "/cann-8.3",
        "LD_LIBRARY_PATH": "/inherited",
    }

    child = build_ascend_child_environment(str(wrapper), environment)

    assert "ASCEND_TOOLKIT_HOME" not in child
    assert child["LD_LIBRARY_PATH"] == f"{prefix / 'lib'}:/inherited"


@pytest.mark.parametrize(
    ("kind", "direction", "listing"),
    [
        ("encoder", "-encoders", " V..... h264_ascend Ascend H264"),
        ("decoder", "-decoders", " V..... h264_ascend Ascend H264"),
    ],
)
def test_probe_accepts_any_ffmpeg_version_with_h264_ascend(tmp_path: Path, kind, direction, listing):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = "ffmpeg version 4.4.4-private\n" if "-version" in command else listing
        return SimpleNamespace(returncode=0, stdout=output)

    diagnostic = probe_ascend_codec_diagnostic(kind, environ={"IBROBOT_ASCEND_FFMPEG": str(ffmpeg)}, run=run)

    assert diagnostic.available is True
    assert diagnostic.code == "available"
    assert calls[1][0][-1] == direction
    assert all(call[1]["timeout"] == 2.0 for call in calls)


def test_probe_accepts_ffmpeg_611(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    outputs = deque(["ffmpeg version 6.1.1\n", " V..... h264_ascend Ascend H264"])

    diagnostic = probe_ascend_codec_diagnostic(
        "decoder",
        environ={"IBROBOT_ASCEND_FFMPEG": str(ffmpeg)},
        run=lambda _command, **_kwargs: SimpleNamespace(returncode=0, stdout=outputs.popleft()),
    )

    assert diagnostic.available is True


def test_probe_returns_structured_direction_failure(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    outputs = deque(["ffmpeg version 5.0\n", " V..... libx264"])

    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=outputs.popleft())

    missing_codec = probe_ascend_codec_diagnostic("encoder", environ={"IBROBOT_ASCEND_FFMPEG": str(ffmpeg)}, run=run)

    assert (missing_codec.available, missing_codec.code) == (False, "codec_direction_missing")


def test_backend_registration_is_optional_and_named_ascend(monkeypatch):
    registry = VideoCodecRegistry()
    monkeypatch.setattr(
        "inference_service.ascend_ffmpeg_video_codec.probe_ascend_codec_diagnostic",
        lambda kind: SimpleNamespace(available=True, kind=kind),
    )
    register_ascend_backend(registry)

    resolved = registry.resolve("ascend", "encoder")

    assert resolved.name == "ascend"
    assert resolved.capabilities.hardware_accelerated is True


def test_encoder_command_environment_rtp_and_timestamp_pairing(tmp_path: Path):
    ffmpeg, prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _FakeSocket()
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 1234, 99, b"\x65encoded").to_bytes())
    spawned = []

    def process_factory(command, **kwargs):
        spawned.append((command, kwargs))
        return _FakeProcess()

    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        bitrate_bps=100_000,
        gop_frames=10,
        device_id=2,
        channel_id=3,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: fake_socket,
    )
    frame = VideoFrame(np.zeros((2, 4, 3), dtype=np.uint8), 987_654, 987_700, 4, 2, "rgb24")

    packets = encoder.encode(frame)

    command, kwargs = spawned[0]
    assert _option(command, "-c:v") == "h264_ascend"
    assert _option(command, "-device_id") == "2"
    assert _option(command, "-channel_id") == "3"
    assert _option(command, "-pix_fmt") == "rgb24"
    assert _option(command, "-vf") == "scale=out_color_matrix=bt709:out_range=tv,format=nv12"
    assert _option(command, "-bf") == "0"
    assert command[-1].startswith("rtp://127.0.0.1:24000?")
    assert kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(prefix / "lib"))
    assert packets[0].capture_timestamp_ns == 987_654
    assert packets[0].rtp_timestamp == 1234
    assert packets[0].keyframe is True
    assert packets[0].payload == b"\x00\x00\x00\x01\x65encoded"
    assert encoder.metrics.input_frames == encoder.metrics.output_frames == 1
    encoder.close()
    assert fake_socket.closed is True


def test_encoder_writes_raw_rgb_and_reset_recreates_process_and_socket(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket()]
    for item in sockets:
        item.datagrams.append(RtpPacket(96, True, 1, 90, 1, b"\x41x").to_bytes())
    processes = []

    def process_factory(_command, **_kwargs):
        process = _FakeProcess()
        processes.append(process)
        return process

    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=2,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: sockets[len(processes)],
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)
    encoder.encode(VideoFrame(image, 20, 20, 4, 2, "rgb24"))

    assert processes[0].stdin.getvalue() == image.tobytes()
    encoder.reset()
    assert len(processes) == 2
    assert sockets[0].closed is True
    assert encoder.state is CodecLifecycleState.RUNNING
    encoder.close()


def test_decoder_command_fixed_rgb24_output_and_fifo_timestamp_pairing(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    output_pipe = _QueuePipe()
    process = _FakeProcess(stdout=output_pipe)
    spawned = []
    sockets = [_FakeSocket(), _FakeSocket()]

    def process_factory(command, **kwargs):
        spawned.append((command, kwargs))
        return process

    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        device_id=4,
        channel_id=5,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: sockets.pop(0),
    )
    rgb = np.full((2, 4, 3), 80, dtype=np.uint8).tobytes()
    output_pipe.put(rgb)

    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))

    command = spawned[0][0]
    assert command[-1] == "pipe:1"
    assert _option(command, "-device_id") == "4"
    assert _option(command, "-channel_id") == "5"
    assert _option(command, "-fflags") == "nobuffer"
    assert _option(command, "-analyzeduration") == "0"
    assert _option(command, "-probesize") == "32"
    assert _option(command, "-vsync") == "0"
    assert _option(command, "-pix_fmt") == "rgb24"
    assert _option(command, "-vf") == "scale=in_color_matrix=bt709:in_range=tv"
    assert spawned[0][1]["stdout"] is subprocess.PIPE
    assert spawned[0][1]["stdin"] is subprocess.PIPE
    assert _option(command, "-i") == "pipe:0"
    assert frames[0].capture_timestamp_ns == 123_456
    assert frames[0].pixel_format == "rgb24"
    assert frames[0].data.shape == (2, 4, 3)
    assert frames[0].keyframe is True
    assert decoder.metrics.output_frames == 1
    decoder.close()


def test_fixed_frame_reader_drops_oldest_frames_when_backlog_is_full():
    pipe = _QueuePipe()
    frame_bytes = 4
    reader = _FixedFrameReader(pipe, frame_bytes, max_frames=2)
    pipe.put(b"aaaa")
    pipe.put(b"bbbb")
    pipe.put(b"cccc")

    deadline = time.monotonic() + 1.0
    while reader.depth < 2 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert reader.depth == 2
    assert reader.drain() == [b"bbbb", b"cccc"]
    assert reader.take_dropped_count() == 1
    pipe.close()
    reader.join(1.0)


def test_decoder_preserves_delayed_output_and_matching_metadata(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket()]
    output_pipe = _QueuePipe()
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(stdout=output_pipe),
        socket_factory=lambda *_args: sockets.pop(0),
    )
    rgb = np.full((2, 4, 3), 80, dtype=np.uint8).tobytes()

    old_timestamp = time.time_ns() - 1_000_000_000
    current_timestamp = time.time_ns()
    assert decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65old", 999, old_timestamp)) == []
    output_pipe.put(rgb)
    output_pipe.put(rgb)
    time.sleep(0.01)
    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65new", 999, current_timestamp))

    assert [frame.capture_timestamp_ns for frame in frames] == [old_timestamp, current_timestamp]
    assert decoder.metrics.dropped_stale_decoder_frames == 0
    assert decoder.metrics.metadata_fifo_depth == 0
    decoder.close()


def test_decoder_waits_for_asynchronous_ffmpeg_output(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    output_pipe = _QueuePipe()
    process = _FakeProcess(stdout=output_pipe)
    sockets = [_FakeSocket(), _FakeSocket()]
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        io_timeout_s=0.5,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: process,
        socket_factory=lambda *_args: sockets.pop(0),
    )
    rgb = np.full((2, 4, 3), 80, dtype=np.uint8).tobytes()

    producer = threading.Thread(target=lambda: (time.sleep(0.01), output_pipe.put(rgb)))
    producer.start()
    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))
    producer.join()

    assert len(frames) == 1
    assert frames[0].capture_timestamp_ns == 123_456
    decoder.close()


def test_decoder_output_wait_is_bounded_by_frame_rate(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket()]
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        io_timeout_s=0.5,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(stdout=_QueuePipe()),
        socket_factory=lambda *_args: sockets.pop(0),
    )

    started = time.monotonic()
    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))
    elapsed = time.monotonic() - started

    assert frames == []
    assert elapsed < 0.1
    decoder.close()


def test_decoder_reader_reassembles_output_datagrams(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket()]
    output_pipe = _QueuePipe()
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(stdout=output_pipe),
        socket_factory=lambda *_args: sockets.pop(0),
    )
    rgb = np.full((2, 4, 3), 80, dtype=np.uint8).tobytes()
    output_pipe.put(rgb[:5])
    output_pipe.put(rgb[5:])

    frames = decoder.decode(EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True))

    assert len(frames) == 1
    decoder.close()


def test_decoder_reset_recreates_process_and_close_is_idempotent(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    processes = []

    def process_factory(_command, **_kwargs):
        process = _FakeProcess(stdout=_QueuePipe())
        processes.append(process)
        return process

    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
    )

    decoder.reset()
    decoder.close(timeout_s=0.1)
    decoder.close(timeout_s=0.1)

    assert len(processes) == 2
    assert decoder.state is CodecLifecycleState.CLOSED
    with pytest.raises(VideoCodecError, match="closed"):
        decoder.reset()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"width": 3}, "even dimensions"),
        ({"frame_rate_hz": 29.97}, "integral frame rate"),
        ({"color_range": "full"}, "limited color range"),
        ({"device_id": -1}, "cannot be negative"),
    ],
)
def test_encoder_rejects_unsupported_configuration(tmp_path: Path, options, message):
    defaults = {
        "width": 4,
        "height": 2,
        "frame_rate_hz": 20,
        "bitrate_bps": 1000,
        "gop_frames": 2,
        "ffmpeg_path": str(_executable(tmp_path / "ffmpeg")),
    }
    defaults.update(options)

    with pytest.raises(ValueError, match=message):
        AscendFfmpegH264Encoder(**defaults)


def test_process_failure_sets_failed_state_metrics_and_stderr_tail(tmp_path: Path):
    ffmpeg = _executable(tmp_path / "ffmpeg")
    fake_socket = _FakeSocket()
    process = _FakeProcess(stderr=io.BytesIO(b"ascend channel failed\n"))
    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        bitrate_bps=1000,
        gop_frames=2,
        ffmpeg_path=str(ffmpeg),
        process_factory=lambda *_args, **_kwargs: process,
        socket_factory=lambda *_args: fake_socket,
    )
    process.returncode = 1

    with pytest.raises(VideoCodecError) as error:
        encoder.encode(VideoFrame(np.zeros((2, 4, 3), dtype=np.uint8), 1, 1, 4, 2, "rgb24"))

    assert error.value.code == "process_exited"
    assert encoder.state is CodecLifecycleState.FAILED
    assert encoder.metrics.errors == 1
    encoder.close()


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _private_ffmpeg(tmp_path: Path):
    prefix = tmp_path / "private"
    ffmpeg = _executable(prefix / "bin" / "ffmpeg")
    (prefix / "lib").mkdir()
    return ffmpeg, prefix, {"IBROBOT_ASCEND_FFMPEG_PREFIX": str(prefix), "LD_LIBRARY_PATH": "/system"}


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_encoder_pipeline_delay_first_frame_empty_second_frame_yields_delayed_output(tmp_path: Path):
    """Ascend DVPP has a 1-frame pipeline delay: frame N output arrives only
    after frame N+1 is submitted.  encode() must return [] for the first
    frame instead of blocking until timeout."""
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _FakeSocket()
    spawned = []

    def process_factory(_command, **_kwargs):
        process = _FakeProcess()
        spawned.append(process)
        return process

    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=2,
        drain_timeout_s=0.01,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: fake_socket,
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    # Frame 1: no datagram available yet → should return [] (pipeline priming)
    packets1 = encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24"))
    assert packets1 == [], "first frame should return empty due to pipeline delay"
    assert encoder.metrics.input_frames == 1
    assert encoder.metrics.output_frames == 0

    # Frame 2: datagram for frame 1 arrives → should return delayed packet
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 100, 1, b"\x65frame1").to_bytes())
    packets2 = encoder.encode(VideoFrame(image, 2_000, 2_000, 4, 2, "rgb24"))
    assert len(packets2) == 1, "second frame should yield the delayed first-frame output"
    assert packets2[0].capture_timestamp_ns == 1_000
    assert packets2[0].keyframe is True
    assert encoder.metrics.input_frames == 2
    assert encoder.metrics.output_frames == 1

    encoder.close()


def test_encoder_steady_state_drains_one_packet_per_call(tmp_path: Path):
    """In steady state each encode() call drains the previous frame's
    delayed output, yielding one packet per call after the first."""
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _FakeSocket()
    spawned = []

    def process_factory(_command, **_kwargs):
        process = _FakeProcess()
        spawned.append(process)
        return process

    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=2,
        drain_timeout_s=0.01,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: fake_socket,
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)
    total_output = 0

    for i in range(5):
        # Pre-load the delayed output for the previous frame
        if i > 0:
            fake_socket.datagrams.append(RtpPacket(96, True, 7, 100 + i, 1, f"\x65frame{i - 1}".encode()).to_bytes())
        packets = encoder.encode(VideoFrame(image, i * 1_000, i * 1_000, 4, 2, "rgb24"))
        if i == 0:
            assert packets == [], "first frame should prime the pipeline"
        else:
            assert len(packets) == 1, f"frame {i} should yield delayed output"
            assert packets[0].capture_timestamp_ns == (i - 1) * 1_000
        total_output += len(packets)

    assert total_output == 4, "5 frames, 1 priming delay → 4 output packets"
    assert encoder.metrics.input_frames == 5
    assert encoder.metrics.output_frames == 4
    encoder.close()


def test_encoder_stops_drain_wait_after_complete_access_unit(tmp_path: Path):
    """A complete AU must not consume the remaining drain timeout."""
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _DrainProbeSocket()
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 100, 1, b"\x65frame").to_bytes())
    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=2,
        drain_timeout_s=0.08,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(),
        socket_factory=lambda *_args: fake_socket,
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    packets = encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24"))

    assert len(packets) == 1
    assert 0.0 in fake_socket.timeouts
    encoder.close()


def test_encoder_skips_drain_wait_after_fruitless_drain(tmp_path: Path):
    """After one fruitless drain, the next drain must return immediately: a
    fixed drain_timeout_s would cap the encode cadence at 1/timeout (20fps
    for the default 50ms), below the 30fps stream target."""
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _DrainTimeoutProbeSocket()
    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=2,
        drain_timeout_s=0.08,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(),
        socket_factory=lambda *_args: fake_socket,
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    # Frame 1: pipeline priming; the drain is allowed to wait its window.
    assert encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24")) == []
    assert any(t and t > 0 for t in fake_socket.timeouts), "first drain should wait"

    # Frame 2: nothing queued; the drain must return without waiting.
    marker = len(fake_socket.timeouts)
    assert encoder.encode(VideoFrame(image, 2_000, 2_000, 4, 2, "rgb24")) == []
    assert fake_socket.timeouts[marker:] and all(t == 0.0 for t in fake_socket.timeouts[marker:]), (
        "drain after a fruitless drain must not wait"
    )

    # Frame 3: queued output is still collected despite the skipped wait,
    # and finding output re-enables waiting for the next call.
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 9_000, 1, b"\x65frame1").to_bytes())
    packets = encoder.encode(VideoFrame(image, 3_000, 3_000, 4, 2, "rgb24"))
    assert [packet.capture_timestamp_ns for packet in packets] == [1_000]

    marker = len(fake_socket.timeouts)
    assert encoder.encode(VideoFrame(image, 4_000, 4_000, 4, 2, "rgb24")) == []
    assert any(t and t > 0 for t in fake_socket.timeouts[marker:]), "wait must be re-enabled after output flows"

    encoder.close()


def test_encoder_retires_timestamp_of_access_unit_lost_to_packet_loss(tmp_path: Path):
    """A datagram gap kills one access unit; the timestamp FIFO must skip the
    matching input frame instead of shifting every later frame by one."""
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _FakeSocket()
    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=10,
        drain_timeout_s=0.01,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(),
        socket_factory=lambda *_args: fake_socket,
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    # Frame 1 primes the pipeline and produces no output.
    assert encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24")) == []

    # Frame 2 drains frame 1's access unit (RTP timestamp step is 9000 at 10fps).
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 9_000, 1, b"\x65frame1").to_bytes())
    packets2 = encoder.encode(VideoFrame(image, 2_000, 2_000, 4, 2, "rgb24"))
    assert [packet.capture_timestamp_ns for packet in packets2] == [1_000]

    # Frame 3's access unit is lost (sequence jumps from 7 to 9).  The next
    # surviving unit carries an RTP timestamp two frames ahead, so frame 2's
    # timestamp must be retired rather than paired with frame 3's payload.
    fake_socket.datagrams.append(RtpPacket(96, True, 9, 27_000, 1, b"\x65frame3").to_bytes())
    packets3 = encoder.encode(VideoFrame(image, 3_000, 3_000, 4, 2, "rgb24"))
    assert [packet.capture_timestamp_ns for packet in packets3] == [3_000]

    encoder.close()


def test_encoder_retires_timestamps_for_multiple_lost_access_units(tmp_path: Path):
    """Consecutive lost access units retire one timestamp per missing frame."""
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    fake_socket = _FakeSocket()
    encoder = AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=10,
        drain_timeout_s=0.01,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(),
        socket_factory=lambda *_args: fake_socket,
    )
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    assert encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24")) == []

    fake_socket.datagrams.append(RtpPacket(96, True, 7, 9_000, 1, b"\x65frame1").to_bytes())
    assert encoder.encode(VideoFrame(image, 2_000, 2_000, 4, 2, "rgb24"))[0].capture_timestamp_ns == 1_000

    # Frames 3 and 4 are submitted but their datagrams never arrive; the
    # datagram for frame 5 shows up with an RTP timestamp four frames ahead
    # of frame 1's unit, so two queued timestamps must be retired.
    assert encoder.encode(VideoFrame(image, 3_000, 3_000, 4, 2, "rgb24")) == []
    assert encoder.encode(VideoFrame(image, 4_000, 4_000, 4, 2, "rgb24")) == []
    fake_socket.datagrams.append(RtpPacket(96, True, 11, 45_000, 1, b"\x65frame5").to_bytes())
    packets = encoder.encode(VideoFrame(image, 5_000, 5_000, 4, 2, "rgb24"))
    assert [packet.capture_timestamp_ns for packet in packets] == [5_000]

    encoder.close()


def test_decoder_skips_output_wait_after_fruitless_wait(tmp_path: Path):
    """After one full output wait times out, subsequent empty calls return
    immediately instead of burning the wait window on every decode()."""
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    sockets = [_FakeSocket(), _FakeSocket()]
    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=20,
        io_timeout_s=0.5,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(stdout=_QueuePipe()),
        socket_factory=lambda *_args: sockets.pop(0),
    )
    packet = EncodedPacket(b"\x00\x00\x00\x01\x65x", 999, 123_456, keyframe=True)

    started = time.monotonic()
    assert decoder.decode(packet) == []
    waited = time.monotonic() - started
    assert waited >= 0.02, "first empty decode must wait the bounded output window"

    started = time.monotonic()
    assert decoder.decode(packet) == []
    skipped = time.monotonic() - started
    assert skipped < 0.01, "second empty decode must not wait again"

    decoder.close()


def test_decoder_allows_normal_hardware_pipeline_depth(tmp_path: Path):
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    processes = []
    sockets = [_FakeSocket(), _FakeSocket()]

    def process_factory(_command, **_kwargs):
        process = _FakeProcess(stdout=_QueuePipe())
        processes.append(process)
        return process

    decoder = AscendFfmpegH264Decoder(
        width=4,
        height=2,
        frame_rate_hz=30,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=process_factory,
        socket_factory=lambda *_args: sockets.pop(0),
    )
    start_ns = time.time_ns()

    for index in range(12):
        packet = EncodedPacket(b"\x00\x00\x00\x01\x41x", index, start_ns + index * 33_333_333)
        assert decoder.decode(packet) == []
    keyframe = EncodedPacket(b"\x00\x00\x00\x01\x65x", 12, start_ns + 12 * 33_333_333, keyframe=True)
    assert decoder.decode(keyframe) == []

    assert len(processes) == 1
    assert decoder.metrics.metadata_fifo_depth == 13
    decoder.close()


def _encoder_with_fake_socket(tmp_path: Path, fake_socket: _FakeSocket) -> AscendFfmpegH264Encoder:
    ffmpeg, _prefix, environment = _private_ffmpeg(tmp_path)
    return AscendFfmpegH264Encoder(
        width=4,
        height=2,
        frame_rate_hz=10,
        bitrate_bps=1000,
        gop_frames=10,
        drain_timeout_s=0.01,
        ffmpeg_path=str(ffmpeg),
        environ=environment,
        process_factory=lambda *_args, **_kwargs: _FakeProcess(),
        socket_factory=lambda *_args: fake_socket,
    )


def test_encoder_fails_closed_when_packet_loss_precedes_first_output(tmp_path: Path):
    """When output is lost before the first surviving access unit, the number
    of missing outputs is unknowable; pairing the FIFO against the encoded
    output cannot be proven and must fail closed instead of silently pairing
    the second frame's payload with the first frame's timestamp."""
    fake_socket = _FakeSocket()
    encoder = _encoder_with_fake_socket(tmp_path, fake_socket)
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    # Frame 1 primes the pipeline and produces no output yet.
    assert encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24")) == []

    # The first access unit is damaged (a fragment is missing), so the first
    # complete output arrives only after packet loss was observed.
    fake_socket.datagrams.append(RtpPacket(96, False, 7, 9_000, 1, b"\x41partial").to_bytes())
    fake_socket.datagrams.append(RtpPacket(96, True, 9, 27_000, 1, b"\x65survivor").to_bytes())

    with pytest.raises(VideoCodecError, match="packet loss before the first encoded output"):
        encoder.encode(VideoFrame(image, 2_000, 2_000, 4, 2, "rgb24"))
    assert encoder.state is CodecLifecycleState.FAILED

    encoder.close()


def test_encoder_retire_survives_rtp_timestamp_wraparound(tmp_path: Path):
    """RTP timestamps wrap every ~13h15m at the 90kHz clock; the retire delta
    must be computed modulo 2**32 or the FIFO never retires again and every
    later frame stays misaligned by one."""
    fake_socket = _FakeSocket()
    encoder = _encoder_with_fake_socket(tmp_path, fake_socket)
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    assert encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24")) == []

    # Access unit 1 lands just below the uint32 boundary.
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 0xFFFF_FF00, 1, b"\x65frame1").to_bytes())
    assert encoder.encode(VideoFrame(image, 2_000, 2_000, 4, 2, "rgb24"))[0].capture_timestamp_ns == 1_000

    # Access unit 3 is two frames ahead across the wrap; one queued
    # timestamp must be retired modulo 2**32.
    assert encoder.encode(VideoFrame(image, 3_000, 3_000, 4, 2, "rgb24")) == []
    wrapped = (0xFFFF_FF00 + 18_000) & 0xFFFF_FFFF
    fake_socket.datagrams.append(RtpPacket(96, True, 9, wrapped, 1, b"\x65frame3").to_bytes())
    packets = encoder.encode(VideoFrame(image, 4_000, 4_000, 4, 2, "rgb24"))
    assert [packet.capture_timestamp_ns for packet in packets] == [3_000]

    encoder.close()


def test_encoder_fails_closed_on_backwards_rtp_timestamp(tmp_path: Path):
    """An RTP timestamp that moves backwards by half the modulo space or more
    indicates corrupt or out-of-order input and must fail closed."""
    fake_socket = _FakeSocket()
    encoder = _encoder_with_fake_socket(tmp_path, fake_socket)
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    assert encoder.encode(VideoFrame(image, 1_000, 1_000, 4, 2, "rgb24")) == []
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 18_000, 1, b"\x65frame1").to_bytes())
    assert encoder.encode(VideoFrame(image, 2_000, 2_000, 4, 2, "rgb24"))[0].capture_timestamp_ns == 1_000

    fake_socket.datagrams.append(RtpPacket(96, True, 9, 9_000, 1, b"\x65backwards").to_bytes())
    with pytest.raises(VideoCodecError, match="moved backwards"):
        encoder.encode(VideoFrame(image, 3_000, 3_000, 4, 2, "rgb24"))
    assert encoder.state is CodecLifecycleState.FAILED

    encoder.close()


def test_encoder_discard_pending_output_prevents_cross_session_pairing(tmp_path: Path):
    """Access units of a retired session still in flight through the DVPP
    pipeline must not surface as the first output of the new session; session
    rollover discards them and re-pairs the next output with the next input."""
    fake_socket = _FakeSocket()
    encoder = _encoder_with_fake_socket(tmp_path, fake_socket)
    image = np.full((2, 4, 3), 30, dtype=np.uint8)

    # Frame 111 of the retired session is submitted and its access unit is
    # in flight through the hardware pipeline when the session rolls over.
    assert encoder.encode(VideoFrame(image, 111, 111, 4, 2, "rgb24")) == []
    fake_socket.datagrams.append(RtpPacket(96, True, 7, 9_000, 1, b"\x65retired").to_bytes())

    encoder.discard_pending_output()

    # The new session's first frame must pair with its own timestamp, never
    # with the retired session's frame 111.
    assert encoder.encode(VideoFrame(image, 222, 222, 4, 2, "rgb24")) == []
    fake_socket.datagrams.append(RtpPacket(96, True, 9, 27_000, 1, b"\x65fresh").to_bytes())
    packets = encoder.encode(VideoFrame(image, 333, 333, 4, 2, "rgb24"))
    assert [packet.capture_timestamp_ns for packet in packets] == [222]

    encoder.close()

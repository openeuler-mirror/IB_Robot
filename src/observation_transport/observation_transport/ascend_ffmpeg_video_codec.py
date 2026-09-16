"""Optional Ascend H.264 codec backed by a private FFmpeg process with h264_ascend support."""

from __future__ import annotations

import fcntl
import os
import queue
import re
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import numpy as np

from observation_transport.rtp_sender import H264Depacketizer, RtpPacket
from observation_transport.video_codec import (
    CodecCapabilities,
    CodecLifecycleState,
    CodecMetrics,
    EncodedPacket,
    VideoCodecError,
    VideoCodecRegistry,
    VideoDecoder,
    VideoEncoder,
    VideoFrame,
)

_BACKEND = "ascend"
_PRIVATE_FFMPEG_PATHS = (
    "/usr/bin/ffmpeg-ascend",
    "/usr/local/bin/ffmpeg-ascend",
    "/usr/local/ffmpeg-ascend-611/bin/ffmpeg",
    "/home/HwHiAiUser/ffmpeg-ascend-cann83/install/bin/ffmpeg",
    "/usr/local/Ascend/ffmpeg/bin/ffmpeg",
    "/usr/local/Ascend/ascend-toolkit/latest/tools/ffmpeg/bin/ffmpeg",
    "/opt/ascend/ffmpeg/bin/ffmpeg",
)
_STDERR_TAIL_BYTES = 8192
_ASCEND_WRAPPER_NAMES = {"ffmpeg-ascend"}
# Binaries that live in an RPM-installed ffmpeg-ascend runtime tree. The RPM
# wrapper and its payload binary belong to the same installation unit, so both
# get the default environment isolation; a payload path probed as "available"
# but launched unisolated hangs on the first frame on the real board.
_RPM_ASCEND_RUNTIME_PATHS = {
    Path("/usr/bin/ffmpeg-ascend"),
    Path("/usr/local/bin/ffmpeg-ascend"),
}
_RPM_ASCEND_PAYLOAD_PATTERN = re.compile(r"^ffmpeg-ascend-[\d.]+/bin/ffmpeg$")
# CANN installation-path variables that the RPM wrapper is known to leak into
# the child environment. Device-visibility and selection policy variables
# (ASCEND_RT_VISIBLE_DEVICES, ASCEND_DEVICE_ID, ...) are runtime resource
# configuration, not installation paths, and must be preserved.
_ASCEND_INSTALL_PATH_ENV = {
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_HOME_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_NNRT_HOME",
    "ASCEND_NNAE_HOME",
    "TOOLCHAIN_HOME",
}


@dataclass(frozen=True, slots=True)
class AscendCodecProbeDiagnostic:
    """Machine-readable result of one FFmpeg/backend direction probe."""

    kind: Literal["encoder", "decoder"]
    available: bool
    code: str
    reason: str
    ffmpeg_path: str | None = None
    version: str = ""
    codec: str = "h264_ascend"


def resolve_ascend_ffmpeg(environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve only the explicitly configured or known private FFmpeg binary."""
    environment = os.environ if environ is None else environ
    candidates: list[str] = []
    explicit = environment.get("IBROBOT_ASCEND_FFMPEG", "").strip()
    if explicit:
        candidates.append(explicit)
    prefix = environment.get("IBROBOT_ASCEND_FFMPEG_PREFIX", "").strip()
    if prefix:
        candidates.append(str(Path(prefix) / "bin" / "ffmpeg"))
    candidates.extend(_PRIVATE_FFMPEG_PATHS)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _is_rpm_ascend_runtime(ffmpeg_path: str | Path) -> bool:
    """Whether *ffmpeg_path* belongs to an RPM-installed ffmpeg-ascend tree.

    Covers the distro wrapper links and the versioned payload binaries they
    wrap (``/usr/local/ffmpeg-ascend-611/bin/ffmpeg`` and friends). Only these
    get the default environment isolation: a wrapper script re-derives its own
    installation paths, and inherited CANN install-path variables from an
    unrelated toolkit are known to break it.
    """
    path = Path(ffmpeg_path).expanduser()
    if path in _RPM_ASCEND_RUNTIME_PATHS:
        return True
    try:
        relative = path.resolve().relative_to(Path("/usr/local"))
    except (ValueError, OSError):
        return False
    return _RPM_ASCEND_PAYLOAD_PATTERN.match(str(relative)) is not None


def build_ascend_child_environment(
    ffmpeg_path: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build process-local library search paths without mutating this process."""
    child = dict(os.environ if environ is None else environ)
    wrapper_path = Path(ffmpeg_path).expanduser()
    is_standard_wrapper = wrapper_path.name in _ASCEND_WRAPPER_NAMES and wrapper_path.parent in {
        Path("/usr/bin"),
        Path("/usr/local/bin"),
    }
    # The wrapper and the payload binary it dispatches to belong to the same
    # RPM installation, so both default to the isolated environment; probing
    # the payload without isolation was shown to hang on the first frame.
    is_rpm_runtime = is_standard_wrapper or _is_rpm_ascend_runtime(wrapper_path)
    isolate = child.get("IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV", "").strip()
    if isolate == "1" or (is_rpm_runtime and isolate != "0"):
        for name in _ASCEND_INSTALL_PATH_ENV:
            child.pop(name, None)
    configured_prefix = child.get("IBROBOT_ASCEND_FFMPEG_PREFIX", "").strip()
    if is_standard_wrapper and not configured_prefix:
        return child
    prefix = Path(configured_prefix).expanduser() if configured_prefix else Path(ffmpeg_path).resolve().parent.parent
    library_candidates = (
        prefix / "lib",
        prefix / "lib64",
        Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64"),
        Path("/usr/local/Ascend/driver/lib64"),
    )
    private_dirs = [str(path) for path in library_candidates if path.is_dir()]
    inherited = child.get("LD_LIBRARY_PATH", "")
    if inherited:
        private_dirs.append(inherited)
    if private_dirs:
        child["LD_LIBRARY_PATH"] = os.pathsep.join(private_dirs)
    return child


def probe_ascend_codec_diagnostic(
    kind: Literal["encoder", "decoder"],
    *,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., Any] = subprocess.run,
    timeout_s: float = 2.0,
) -> AscendCodecProbeDiagnostic:
    if kind not in {"encoder", "decoder"}:
        raise ValueError(f"unsupported codec kind {kind!r}")
    ffmpeg = resolve_ascend_ffmpeg(environ)
    if ffmpeg is None:
        return AscendCodecProbeDiagnostic(kind, False, "ffmpeg_not_found", "private Ascend FFmpeg was not found")
    child_env = build_ascend_child_environment(ffmpeg, environ)
    try:
        version_result = run(
            [ffmpeg, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
            env=child_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AscendCodecProbeDiagnostic(kind, False, "version_probe_failed", str(exc), ffmpeg)
    version_output = version_result.stdout or ""
    version = version_output.splitlines()[0] if version_output else ""
    if version_result.returncode != 0:
        return AscendCodecProbeDiagnostic(
            kind, False, "version_probe_failed", version or "FFmpeg exited nonzero", ffmpeg
        )
    direction_flag = "-encoders" if kind == "encoder" else "-decoders"
    try:
        codec_result = run(
            [ffmpeg, "-hide_banner", direction_flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
            env=child_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AscendCodecProbeDiagnostic(kind, False, "direction_probe_failed", str(exc), ffmpeg, version)
    output = codec_result.stdout or ""
    if codec_result.returncode != 0:
        return AscendCodecProbeDiagnostic(
            kind, False, "direction_probe_failed", output.strip() or "FFmpeg exited nonzero", ffmpeg, version
        )
    if not re.search(r"(?m)^\s*[A-Z.]{6}\s+h264_ascend(?:\s|$)", output):
        return AscendCodecProbeDiagnostic(
            kind,
            False,
            "codec_direction_missing",
            f"h264_ascend is not listed by {direction_flag}",
            ffmpeg,
            version,
        )
    return AscendCodecProbeDiagnostic(kind, True, "available", "h264_ascend is available", ffmpeg, version)


def probe_ascend_codec(kind: str) -> CodecCapabilities | None:
    if kind not in {"encoder", "decoder"}:
        raise ValueError(f"unsupported codec kind {kind!r}")
    diagnostic = probe_ascend_codec_diagnostic(kind)
    if not diagnostic.available:
        raise VideoCodecError(
            diagnostic.code,
            diagnostic.reason,
            backend=_BACKEND,
            details={
                "kind": diagnostic.kind,
                "ffmpeg_path": diagnostic.ffmpeg_path or "",
                "version": diagnostic.version,
                "codec": diagnostic.codec,
            },
        )
    return CodecCapabilities(pixel_formats=("rgb24", "bgr24"), hardware_accelerated=True)


def register_ascend_backend(registry: VideoCodecRegistry, *, priority: int = 100) -> None:
    registry.register(
        _BACKEND,
        priority=priority,
        probe=probe_ascend_codec,
        encoder_factory=AscendFfmpegH264Encoder,
        decoder_factory=AscendFfmpegH264Decoder,
    )


class _StderrTail:
    def __init__(self, pipe: Any, limit: int = _STDERR_TAIL_BYTES) -> None:
        self._pipe = pipe
        self._limit = limit
        self._data = bytearray()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._drain, name="ascend-ffmpeg-stderr", daemon=True)
        self._thread.start()

    @property
    def text(self) -> str:
        with self._lock:
            return bytes(self._data).decode("utf-8", errors="replace")

    def join(self, timeout_s: float) -> None:
        self._thread.join(max(0.0, timeout_s))

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(1024)
                if not chunk:
                    return
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                with self._lock:
                    self._data.extend(chunk)
                    del self._data[: max(0, len(self._data) - self._limit)]
        except (OSError, ValueError):
            return


class _StdinWriter:
    """Persistent FFmpeg stdin writer enforcing per-write timeouts.

    Spawning a fresh thread per frame (the previous design) cost ~60 thread
    create/join cycles per second for dual 30fps streams and churned the GIL
    on the 3-core edge board.  One daemon thread now serves writes strictly
    in order; each :meth:`write` still blocks its caller until completion or
    timeout, preserving the backpressure that bounds the caller's queue.
    """

    _STOP = object()
    _TIMEOUT = object()

    def __init__(self, stdin: Any, timeout_s: float) -> None:
        self._stdin = stdin
        self._timeout_s = timeout_s
        self._requests: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="ascend-ffmpeg-stdin", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            payload = self._requests.get()
            if payload is self._STOP:
                return
            try:
                written = self._stdin.write(payload)
                if written is not None and written != len(payload):
                    raise OSError(f"short FFmpeg stdin write: {written}/{len(payload)}")
                flush = getattr(self._stdin, "flush", None)
                if flush is not None:
                    flush()
                self._results.put(None)
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._results.put(exc)
                return

    def write(self, payload: bytes) -> object:
        """Block until *payload* is written; None, an exception, or _TIMEOUT."""
        self._requests.put(payload)
        try:
            return self._results.get(timeout=self._timeout_s)
        except queue.Empty:
            return self._TIMEOUT

    def stop(self, timeout_s: float) -> None:
        # Closing stdin first unblocks a write stuck in the pipe so the join
        # below can actually complete; posting _STOP alone never interrupts
        # an in-flight write, which left the writer thread and the FFmpeg
        # process alive after a timeout on the real board.
        with suppress(OSError, ValueError, AttributeError):
            self._stdin.close()
        self._requests.put(self._STOP)
        self._thread.join(max(0.0, timeout_s))


class _FixedFrameReader:
    def __init__(self, pipe: Any, frame_bytes: int, max_frames: int = 3) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self._pipe = pipe
        self._frame_bytes = frame_bytes
        self._max_frames = max_frames
        self._frames: deque[bytes] = deque()
        self._dropped_frames = 0
        self._buffer = bytearray()
        self._error: Exception | None = None
        self._eof = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._read, name="ascend-ffmpeg-stdout", daemon=True)
        self._thread.start()

    def get(self, timeout_s: float) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._frames and self._error is None and not self._eof:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._frames:
                return self._frames.popleft()
            if self._error is not None:
                raise self._error
            return None

    def drain(self) -> list[bytes]:
        with self._condition:
            if self._error is not None:
                raise self._error
            frames = list(self._frames)
            self._frames.clear()
            return frames

    @property
    def depth(self) -> int:
        with self._condition:
            return len(self._frames)

    def take_dropped_count(self) -> int:
        with self._condition:
            dropped = self._dropped_frames
            self._dropped_frames = 0
            return dropped

    def join(self, timeout_s: float) -> None:
        self._thread.join(max(0.0, timeout_s))

    def _read(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(min(64 * 1024, max(4096, self._frame_bytes - len(self._buffer))))
                if not chunk:
                    with self._condition:
                        self._eof = True
                        self._condition.notify_all()
                    return
                with self._condition:
                    self._buffer.extend(chunk)
                    while len(self._buffer) >= self._frame_bytes:
                        if len(self._frames) >= self._max_frames:
                            self._frames.popleft()
                            self._dropped_frames += 1
                        self._frames.append(bytes(self._buffer[: self._frame_bytes]))
                        del self._buffer[: self._frame_bytes]
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()


class _DatagramPipe:
    def __init__(self, udp_socket: Any) -> None:
        self._socket = udp_socket

    def read(self, _size: int) -> bytes:
        while True:
            try:
                return self._socket.recvfrom(65535)[0]
            except TimeoutError:
                continue
            except OSError:
                return b""

    def close(self) -> None:
        self._socket.close()


class _AscendProcessCodec:
    def __init__(
        self, *, ffmpeg_path: str | None, process_factory: Callable[..., Any], environ: Mapping[str, str] | None
    ) -> None:
        self._ffmpeg = ffmpeg_path or resolve_ascend_ffmpeg(environ)
        if self._ffmpeg is None:
            raise VideoCodecError("ffmpeg_not_found", "private Ascend FFmpeg was not found", backend=_BACKEND)
        self._process_factory = process_factory
        self._environ = environ
        self._process: Any = None
        self._stderr: _StderrTail | None = None
        self._stdin_writer: _StdinWriter | None = None
        self._state = CodecLifecycleState.CREATED
        self._metrics = CodecMetrics()

    @property
    def state(self) -> CodecLifecycleState:
        return self._state

    @property
    def metrics(self) -> CodecMetrics:
        return self._metrics

    def _spawn(self, command: list[str], **streams: Any) -> Any:
        stdin = streams.pop("stdin", subprocess.PIPE)
        try:
            process = self._process_factory(
                command,
                stdin=stdin,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=build_ascend_child_environment(self._ffmpeg, self._environ),
                **streams,
            )
        except OSError as exc:
            self._state = CodecLifecycleState.FAILED
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
            raise VideoCodecError("process_start_failed", str(exc), backend=_BACKEND) from exc
        if (stdin is subprocess.PIPE and process.stdin is None) or process.stderr is None:
            raise VideoCodecError("process_start_failed", "FFmpeg pipes were not created", backend=_BACKEND)
        self._process = process
        self._stderr = _StderrTail(process.stderr)
        if stdin is subprocess.PIPE:
            self._stdin_writer = _StdinWriter(process.stdin, self._io_timeout_s)
        self._state = CodecLifecycleState.RUNNING
        return process

    def _write(self, payload: bytes) -> None:
        self._require_running()
        if self._process.poll() is not None:
            self._fail("process_exited", "FFmpeg exited before accepting input")
        writer = self._stdin_writer
        if writer is None:
            self._fail("process_exited", "FFmpeg stdin is not writable")
        result = writer.write(payload)
        if result is _StdinWriter._TIMEOUT:
            # The write is stuck and the writer thread stays blocked in the
            # pipe; close stdin now so the blocked write fails, the writer
            # thread exits, and reset()/close() can reap the process instead
            # of leaving both alive after the timeout.
            with suppress(OSError, ValueError, AttributeError):
                self._process.stdin.close()
            self._fail("process_write_timeout", "timed out writing to FFmpeg stdin")
        if isinstance(result, BaseException):
            self._fail("process_write_failed", str(result), cause=result)

    def _fail(self, code: str, message: str, *, recoverable: bool = True, cause: Exception | None = None) -> None:
        self._state = CodecLifecycleState.FAILED
        self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)
        error = VideoCodecError(
            code,
            message,
            backend=_BACKEND,
            recoverable=recoverable,
            details={"stderr_tail": self._stderr.text if self._stderr is not None else ""},
        )
        if cause is None:
            raise error
        raise error from cause

    def _require_running(self) -> None:
        if self._state is not CodecLifecycleState.RUNNING:
            raise VideoCodecError("invalid_state", f"codec is {self._state.value}", backend=_BACKEND)

    def _require_not_closed(self) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            raise VideoCodecError("invalid_state", "codec is closed", backend=_BACKEND)

    def _stop_process(self, timeout_s: float) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        process = self._process
        if process is None:
            return
        deadline = time.monotonic() + timeout_s
        if self._stdin_writer is not None:
            # Stop the writer first so it does not race with stdin.close().
            self._stdin_writer.stop(max(0.0, deadline - time.monotonic()))
            self._stdin_writer = None
        with suppress(AttributeError, OSError, ValueError):
            process.stdin.close()
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
        if self._stderr is not None:
            self._stderr.join(max(0.0, deadline - time.monotonic()))
        self._process = None


class AscendFfmpegH264Encoder(_AscendProcessCodec, VideoEncoder):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        frame_rate_hz: float,
        bitrate_bps: int,
        gop_frames: int,
        input_pixel_format: str = "rgb24",
        profile: str = "main",
        color_space: str = "bt709",
        color_range: str = "limited",
        device_id: int = 0,
        channel_id: int = 0,
        io_timeout_s: float = 1.0,
        drain_timeout_s: float = 0.05,
        ffmpeg_path: str | None = None,
        process_factory: Callable[..., Any] = subprocess.Popen,
        socket_factory: Callable[..., Any] = socket.socket,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        _validate_common(width, height, frame_rate_hz, color_space, color_range, device_id, channel_id, io_timeout_s)
        if bitrate_bps <= 0 or gop_frames <= 0:
            raise ValueError("bitrate and GOP must be positive")
        if input_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("Ascend encoder input_pixel_format must be rgb24 or bgr24")
        if profile not in {"baseline", "main", "high"}:
            raise ValueError("Ascend encoder profile must be baseline, main, or high")
        if drain_timeout_s <= 0:
            raise ValueError("drain_timeout_s must be positive")
        super().__init__(ffmpeg_path=ffmpeg_path, process_factory=process_factory, environ=environ)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(frame_rate_hz)
        self._bitrate = int(bitrate_bps)
        self._gop = int(gop_frames)
        self._input_pixel_format = input_pixel_format
        self._profile = profile
        self._device_id = int(device_id)
        self._channel_id = int(channel_id)
        self._io_timeout_s = float(io_timeout_s)
        self._drain_timeout_s = float(drain_timeout_s)
        self._skip_next_drain_wait = False
        self._socket_factory = socket_factory
        self._socket: Any = None
        self._depacketizer = H264Depacketizer()
        self._timestamps: deque[int] = deque()
        self._last_rtp_timestamp: int | None = None
        self._lost_packets = 0
        self._start()

    def _start(self) -> None:
        udp_socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.bind(("127.0.0.1", 0))
        udp_socket.settimeout(self._io_timeout_s)
        host, port = udp_socket.getsockname()[:2]
        query = urlencode({"pkt_size": 1200})
        command = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self._input_pixel_format,
            "-video_size",
            f"{self._width}x{self._height}",
            "-framerate",
            str(self._fps),
            "-i",
            "pipe:0",
            # FFmpeg's NEON-optimized swscale converts rgb24/bgr24 to the
            # limited-range BT.709 NV12 that h264_ascend requires. Doing the
            # conversion here keeps the ~15-20ms/frame numpy cost out of the
            # GIL-constrained Python process, which matters on the 3-core
            # edge board.
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv,format=nv12",
            "-an",
            "-c:v",
            "h264_ascend",
            "-device_id",
            str(self._device_id),
            "-channel_id",
            str(self._channel_id),
            "-profile",
            str({"baseline": 0, "main": 1, "high": 2}[self._profile]),
            "-rc_mode",
            "0",
            "-bf",
            "0",
            "-gop",
            str(self._gop),
            "-frame_rate",
            str(self._fps),
            "-max_bit_rate",
            str(max(2, round(self._bitrate / 1000))),
            "-f",
            "rtp",
            f"rtp://{host}:{port}?{query}",
        ]
        try:
            self._socket = udp_socket
            self._spawn(command, stdout=subprocess.DEVNULL)
            self._grow_stdin_pipe_buffer()
        except Exception:
            udp_socket.close()
            self._socket = None
            raise

    def _grow_stdin_pipe_buffer(self) -> None:
        """Raise the FFmpeg stdin pipe buffer to several whole frames.

        The default 64KB pipe holds only a fraction of one 640x480 rgb24
        frame, so every write blocks until FFmpeg drains it and ties the
        caller's cadence to the encoder's per-frame latency. A multi-frame
        buffer lets writes complete immediately while FFmpeg consumes at
        its own pace. Non-fatal when the kernel caps the request.
        """
        set_pipe_size = getattr(fcntl, "F_SETPIPE_SZ", None)
        if set_pipe_size is None or self._process is None or self._process.stdin is None:
            return
        with suppress(OSError, ValueError):
            fcntl.fcntl(self._process.stdin.fileno(), set_pipe_size, 4 * 1024 * 1024)

    def encode(self, frame: VideoFrame) -> list[EncodedPacket]:
        self._require_running()
        if (frame.width, frame.height) != (self._width, self._height):
            raise VideoCodecError("invalid_frame", "frame dimensions do not match encoder", backend=_BACKEND)
        if frame.pixel_format != self._input_pixel_format:
            raise VideoCodecError("invalid_pixel_format", f"expected {self._input_pixel_format}", backend=_BACKEND)
        if frame.color_space.lower() != "bt709" or frame.color_range.lower() != "limited":
            raise VideoCodecError("unsupported_color", "Ascend H.264 requires limited-range BT.709", backend=_BACKEND)
        array = np.asarray(frame.data)
        if array.shape != (self._height, self._width, 3) or array.dtype != np.uint8:
            raise VideoCodecError("invalid_frame", "expected a uint8 HWC color frame", backend=_BACKEND)
        try:
            # The raw rgb24/bgr24 frame goes straight to FFmpeg; its swscale
            # filter graph performs the BT.709 limited-range NV12 conversion
            # (see _start) off the Python GIL.
            self._write(array.tobytes())
            self._timestamps.append(frame.capture_timestamp_ns)
            packets = self._drain_access_units(self._drain_timeout_s)
        except VideoCodecError:
            raise
        except Exception as exc:
            self._fail("encode_failed", str(exc), cause=exc)
        self._metrics = replace(
            self._metrics,
            input_frames=self._metrics.input_frames + 1,
            output_frames=self._metrics.output_frames + len(packets),
            output_packets=self._metrics.output_packets + len(packets),
        )
        return packets

    def _drain_access_units(self, timeout_s: float) -> list[EncodedPacket]:
        """Collect encoded access units within *timeout_s*.

        The Ascend DVPP hardware encoder has an internal pipeline delay:
        output for frame N typically arrives only after frame N+1 has been
        submitted. Wait up to *timeout_s* for the first complete access unit,
        then collect only datagrams that are already queued. When no output is
        ready (e.g. the very first frame), an empty list is returned and the
        delayed output surfaces on the next call.

        Waiting the full window on every fruitless call would cap the encode
        cadence at 1/timeout_s (20fps for the default 50ms), below the 30fps
        target on the CPU-starved edge board.  After one empty drain,
        subsequent calls return immediately until output actually flows
        again, mirroring the decoder's ``_skip_next_output_wait``.
        """
        packets: list[EncodedPacket] = []
        wait_s = 0.0 if self._skip_next_drain_wait else timeout_s
        self._skip_next_drain_wait = False
        deadline = time.monotonic() + wait_s
        while True:
            remaining = deadline - time.monotonic()
            # Once one access unit arrived, or when no wait is allowed, only
            # collect datagrams that are already queued.  A skipped wait still
            # polls the socket buffer once so queued output is not starved.
            self._socket.settimeout(0.0 if packets or remaining <= 0 else remaining)
            try:
                datagram = self._socket.recvfrom(65535)[0]
                access_unit, lost = self._depacketizer.push(RtpPacket.from_bytes(datagram))
                self._lost_packets += lost
            except (TimeoutError, BlockingIOError):
                break
            except (OSError, ValueError) as exc:
                self._fail("invalid_rtp", str(exc), cause=exc)
            if access_unit is None:
                # A datagram gap damages an access unit that is then discarded
                # by the depacketizer and never surfaces as output.  Its input
                # timestamp is retired by _retire_lost_timestamps once the next
                # surviving access unit reports the frame gap.
                continue
            if not self._timestamps:
                self._fail("timestamp_underflow", "encoded output has no input timestamp")
            if self._last_rtp_timestamp is None and self._lost_packets:
                # Output was lost before the first surviving access unit, so
                # the number of missing outputs is unknowable and pairing the
                # FIFO against the encoded output cannot be proven.  Fail
                # closed so the caller resets the encoder instead of pairing
                # the second frame's payload with the first frame's timestamp.
                self._fail(
                    "timestamp_misaligned",
                    "packet loss before the first encoded output broke the timestamp FIFO",
                )
            self._retire_lost_timestamps(access_unit.timestamp)
            capture_timestamp_ns = self._timestamps.popleft()
            packets.append(
                EncodedPacket(
                    access_unit.payload,
                    access_unit.timestamp,
                    capture_timestamp_ns,
                    keyframe=access_unit.keyframe,
                )
            )
        self._skip_next_drain_wait = not packets
        return packets

    def _retire_lost_timestamps(self, rtp_timestamp: int) -> None:
        """Drop input timestamps whose encoded output was lost to packet loss.

        FFmpeg stamps output access units with frame-rate spaced RTP
        timestamps, so the gap between consecutive surviving units tells how
        many input frames went missing in between (a damaged access unit
        produces no output at all).  Retiring those timestamps here keeps the
        FIFO aligned with the encoded output instead of shifting every later
        frame by one.

        The delta is computed modulo 2**32: RTP timestamps wrap roughly every
        13h15m at the 90kHz clock, and a plain subtraction at the wrap
        boundary yields a huge negative number that never retires anything
        and permanently misaligns the FIFO. Deltas at or above half the
        module space indicate out-of-order or corrupt input and fail closed.
        """
        if self._last_rtp_timestamp is None:
            self._last_rtp_timestamp = rtp_timestamp
            return
        delta = (rtp_timestamp - self._last_rtp_timestamp) & 0xFFFFFFFF
        if delta >= 0x80000000:
            self._fail(
                "timestamp_misaligned",
                f"encoded output RTP timestamp moved backwards ({delta:#x} delta)",
            )
        self._last_rtp_timestamp = rtp_timestamp
        frame_gap = round(delta * self._fps / 90000)
        for _ in range(max(0, frame_gap - 1)):
            if self._timestamps:
                self._timestamps.popleft()

    def discard_pending_output(self) -> None:
        """Drop access units still in flight through the DVPP pipeline.

        The hardware encoder has a pipeline delay, so access units of the
        retired session can surface as the first output after a session
        rollover. Draining whatever is already queued and clearing the
        timestamp FIFO re-pairs the next output with the next input without
        respawning the FFmpeg process.
        """
        self._require_running()
        with suppress(VideoCodecError, OSError, ValueError):
            self._drain_access_units(0.0)
        self._timestamps.clear()
        self._last_rtp_timestamp = None
        self._lost_packets = 0
        self._skip_next_drain_wait = False
        self._depacketizer.reset()

    def reset(self) -> None:
        self._require_not_closed()
        with suppress(Exception):
            self._drain_access_units(self._drain_timeout_s)
        self._stop_process(self._io_timeout_s)
        if self._socket is not None:
            self._socket.close()
        self._timestamps.clear()
        self._last_rtp_timestamp = None
        self._lost_packets = 0
        self._skip_next_drain_wait = False
        self._depacketizer.reset()
        self._state = CodecLifecycleState.CREATED
        self._start()

    def close(self, timeout_s: float = 1.0) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            return
        self._stop_process(timeout_s)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._timestamps.clear()
        self._state = CodecLifecycleState.CLOSED


class AscendFfmpegH264Decoder(_AscendProcessCodec, VideoDecoder):
    def __init__(
        self,
        *,
        width: int,
        height: int,
        frame_rate_hz: float,
        output_pixel_format: str = "rgb24",
        color_space: str = "bt709",
        color_range: str = "limited",
        device_id: int = 0,
        channel_id: int = 0,
        io_timeout_s: float = 1.0,
        ffmpeg_path: str | None = None,
        process_factory: Callable[..., Any] = subprocess.Popen,
        socket_factory: Callable[..., Any] = socket.socket,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        _validate_common(width, height, frame_rate_hz, color_space, color_range, device_id, channel_id, io_timeout_s)
        if output_pixel_format not in {"rgb24", "bgr24"}:
            raise ValueError("Ascend decoder output_pixel_format must be rgb24 or bgr24")
        super().__init__(ffmpeg_path=ffmpeg_path, process_factory=process_factory, environ=environ)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(frame_rate_hz)
        self._output_pixel_format = output_pixel_format
        self._device_id = int(device_id)
        self._channel_id = int(channel_id)
        self._io_timeout_s = float(io_timeout_s)
        self._output_wait_s = min(self._io_timeout_s, 0.5 / frame_rate_hz)
        self._skip_next_output_wait = False
        self._socket_factory = socket_factory
        self._socket: Any = None
        self._output_pipe: Any = None
        self._input_endpoint: tuple[str, int] | None = None
        self._frame_metadata: deque[tuple[int, bool]] = deque()
        self._raw_output_frames = 0
        self._metrics_started_ns = time.monotonic_ns()
        self._reader: _FixedFrameReader | None = None
        self._start()

    def _start(self) -> None:
        # Feed the annex-B access units through stdin instead of a loopback
        # UDP socket: with probesize-limited stream inputs FFmpeg cannot
        # negotiate an rgb24 filtergraph before packets arrive, and the UDP
        # datagram boundary is another loss surface. A pipe is also the shape
        # the encoder side already uses.
        command = [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
            "-f",
            "h264",
            "-c:v",
            "h264_ascend",
            "-device_id",
            str(self._device_id),
            "-channel_id",
            str(self._channel_id),
            "-i",
            "pipe:0",
            "-an",
            # The decoder emits limited-range BT.709 NV12; swscale converts to
            # full-range rgb24 inside ffmpeg so the Python side never pays the
            # per-frame numpy color conversion on the GIL-constrained host.
            "-vf",
            "scale=in_color_matrix=bt709:in_range=tv",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self._output_pixel_format,
            "-vsync",
            "0",
            "pipe:1",
        ]
        try:
            process = self._spawn(command, stdout=subprocess.PIPE)
            if process.stdout is None:
                raise VideoCodecError("process_start_failed", "FFmpeg stdout pipe was not created", backend=_BACKEND)
            self._output_pipe = process.stdout
        except Exception:
            if self._output_pipe is not None:
                self._output_pipe.close()
            self._output_pipe = None
            raise
        self._reader = _FixedFrameReader(self._output_pipe, self._width * self._height * 3, max_frames=3)

    def decode(self, packet: EncodedPacket) -> list[VideoFrame]:
        self._require_running()
        self._frame_metadata.append((packet.capture_timestamp_ns, packet.keyframe))
        try:
            self._write(packet.payload)
            assert self._reader is not None
            self._retire_reader_drops()
            frames = self._reader.drain()
            if not frames:
                # The DVPP decoder pipelines output, so most calls find nothing
                # queued and the frame surfaces with a later access unit
                # anyway.  Waiting the full window on every empty call only
                # added latency, so after one fruitless wait subsequent calls
                # return immediately until output actually flows again.
                wait_s = 0.0 if self._skip_next_output_wait else self._output_wait_s
                first_frame = self._reader.get(wait_s)
                if first_frame is not None:
                    frames = [first_frame, *self._reader.drain()]
                    self._skip_next_output_wait = False
                else:
                    self._skip_next_output_wait = True
        except VideoCodecError:
            raise
        except Exception as exc:
            self._fail("decode_failed", str(exc), cause=exc)
        reader_drops = self._retire_reader_drops()
        self._raw_output_frames += len(frames) + reader_drops
        decoded = []
        for nv12 in frames:
            metadata = self._take_metadata()
            if metadata is None:
                break
            capture_timestamp_ns, keyframe = metadata
            decoded.append(self._convert_frame(nv12, capture_timestamp_ns, keyframe))
        self._update_decoder_metrics(
            input_frames=1,
            output_frames=len(decoded),
            output_packets=1,
        )
        return decoded

    def _update_decoder_metrics(
        self,
        *,
        input_frames: int = 0,
        output_frames: int = 0,
        output_packets: int = 0,
        output_age_ns: int | None = None,
        dropped_frames: int = 0,
    ) -> None:
        elapsed_s = max((time.monotonic_ns() - self._metrics_started_ns) / 1e9, 1e-9)
        total_input = self._metrics.input_frames + input_frames
        total_output = self._metrics.output_frames + output_frames
        self._metrics = replace(
            self._metrics,
            input_frames=total_input,
            output_frames=total_output,
            output_packets=self._metrics.output_packets + output_packets,
            decoder_backlog_depth=len(self._frame_metadata),
            decoder_output_age_ns=(self._metrics.decoder_output_age_ns if output_age_ns is None else output_age_ns),
            dropped_stale_decoder_frames=self._metrics.dropped_stale_decoder_frames + dropped_frames,
            metadata_fifo_depth=len(self._frame_metadata),
            input_frame_rate_hz=total_input / elapsed_s,
            output_frame_rate_hz=self._raw_output_frames / elapsed_s,
        )

    def _take_metadata(self) -> tuple[int, bool] | None:
        if not self._frame_metadata:
            return None
        return self._frame_metadata.popleft()

    def _retire_reader_drops(self) -> int:
        if self._reader is None:
            return 0
        dropped = self._reader.take_dropped_count()
        for _ in range(min(dropped, len(self._frame_metadata))):
            self._frame_metadata.popleft()
        if dropped:
            self._metrics = replace(
                self._metrics,
                dropped_stale_decoder_frames=self._metrics.dropped_stale_decoder_frames + dropped,
                decoder_backlog_depth=len(self._frame_metadata),
                metadata_fifo_depth=len(self._frame_metadata),
            )
        return dropped

    def _convert_frame(self, raw: bytes, capture_timestamp_ns: int, keyframe: bool) -> VideoFrame:
        if capture_timestamp_ns < 0:
            self._fail("timestamp_underflow", "decoded output has no input timestamp")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(self._height, self._width, 3).copy()
        return VideoFrame(
            image,
            capture_timestamp_ns,
            capture_timestamp_ns,
            self._width,
            self._height,
            self._output_pixel_format,
            color_space="bt709",
            color_range="full",
            keyframe=keyframe,
        )

    def reset(self) -> None:
        self._require_not_closed()
        # An IDR-boundary reset runs on the receiver processing thread, so it
        # must not spend multiple io timeouts waiting for a graceful exit.
        reset_timeout_s = min(self._io_timeout_s, 0.25)
        self._stop_process(reset_timeout_s)
        if self._output_pipe is not None:
            self._output_pipe.close()
            self._output_pipe = None
        if self._reader is not None:
            self._reader.join(reset_timeout_s)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._input_endpoint = None
        self._frame_metadata.clear()
        self._skip_next_output_wait = False
        self._state = CodecLifecycleState.CREATED
        self._start()

    def close(self, timeout_s: float = 1.0) -> None:
        if self._state is CodecLifecycleState.CLOSED:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        self._stop_process(timeout_s)
        if self._output_pipe is not None:
            self._output_pipe.close()
            self._output_pipe = None
        if self._reader is not None:
            self._reader.join(max(0.0, deadline - time.monotonic()))
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._input_endpoint = None
        self._frame_metadata.clear()
        self._state = CodecLifecycleState.CLOSED


def _validate_common(
    width: int,
    height: int,
    frame_rate_hz: float,
    color_space: str,
    color_range: str,
    device_id: int,
    channel_id: int,
    io_timeout_s: float,
) -> None:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("Ascend H.264 requires positive even dimensions")
    if frame_rate_hz <= 0 or not float(frame_rate_hz).is_integer():
        raise ValueError("Ascend H.264 requires a positive integral frame rate")
    if color_space.lower() != "bt709":
        raise ValueError("Ascend H.264 supports BT.709 color space only")
    if color_range.lower() != "limited":
        raise ValueError("Ascend H.264 supports limited color range only")
    if device_id < 0 or channel_id < 0:
        raise ValueError("device_id and channel_id cannot be negative")
    # Ascend DVPP VENC channels are a per-device hardware resource with a
    # 0..127 range (the encoder uses 1..N for dense allocation).  device_id 0
    # is the only scope the manager currently passes; multi-device support
    # would require a resource allocation contract, not a code change here.
    if channel_id > 127:
        raise ValueError(f"Ascend DVPP channel_id must be 0..127; got {channel_id}")
    if io_timeout_s <= 0:
        raise ValueError("io_timeout_s must be positive")

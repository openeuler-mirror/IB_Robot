"""Inference-side RTP receiver built on the shared observation transport wire SSOT."""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference_service.h264_stream_recorder import H264StreamRecorder

from observation_transport.rtp_sender import (
    DatagramReceiver,
    DatagramSender,
    H264AccessUnit,
    H264Depacketizer,
    H264RtpSender,
    RtpPacket,
    StreamLifecycleState,
    StreamMetrics,
    StreamStatus,
    VideoRtpError,
    packetize_h264,
    split_annex_b,
)
from observation_transport.video_codec import EncodedPacket, VideoCodecError, VideoDecoder, VideoFrame

from inference_service.observation_sync import ObservationSynchronizationError, RtpTimestampMapper
from robot_config.contract_utils import StreamBuffer

_RTP_HEADER_SIZE = 12


class H264RtpReceiver:
    """Validate, reconstruct, decode, map, and buffer one H.264 RTP stream."""

    def __init__(
        self,
        *,
        stream_id: str,
        observation_key: str,
        ssrc: int,
        decoder: VideoDecoder,
        frame_buffer: StreamBuffer,
        timestamp_mapper: RtpTimestampMapper,
        session_generation: int,
        packet_queue_capacity: int,
        payload_type: int = 96,
        selected_backend: str = "software",
        endpoint: tuple[str, int] | None = None,
        datagram_receiver: DatagramReceiver | None = None,
        max_datagram_size: int = 65535,
        recorder: H264StreamRecorder | None = None,
        decode: bool = True,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if not stream_id or not observation_key or session_generation < 1 or packet_queue_capacity <= 0:
            raise ValueError("RTP receiver requires stream identity, session, and positive queue capacity")
        self.stream_id = stream_id
        self.observation_key = observation_key
        self.ssrc = ssrc
        self.payload_type = payload_type
        self.decoder = decoder
        self.frame_buffer = frame_buffer
        self.timestamp_mapper = timestamp_mapper
        self.session_generation = session_generation
        self.selected_backend = selected_backend
        self._recorder = recorder
        self._decode = decode
        self._clock = time.time_ns if clock is None else clock
        self._frame_count = 0
        self._recording_generation: int | None = None
        if max_datagram_size <= _RTP_HEADER_SIZE:
            raise ValueError("max_datagram_size must exceed the RTP header size")
        if datagram_receiver is not None and endpoint is None:
            raise ValueError("an injected datagram receiver requires an endpoint")
        if endpoint is not None and (not endpoint[0] or not 1 <= endpoint[1] <= 65535):
            raise ValueError("RTP receiver endpoint must have a host and port in 1..65535")
        self.endpoint = endpoint
        self._socket = datagram_receiver
        self._max_datagram_size = max_datagram_size
        self._capacity = packet_queue_capacity
        self._queue: deque[tuple[bytes, int]] = deque()
        self._condition = threading.Condition(threading.RLock())
        self._lock = self._condition
        self._processing_lock = threading.RLock()
        self._depacketizer = H264Depacketizer()
        self._state = StreamLifecycleState.CONFIGURED
        self._metrics = StreamMetrics()
        self._last_error = ""
        self._have_sps = False
        self._have_pps = False
        self._keyframe_ready = False
        self._stopping = False
        self._receive_thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None

    @property
    def status(self) -> StreamStatus:
        with self._lock:
            return StreamStatus(
                self.stream_id,
                self._state,
                self._state is StreamLifecycleState.READY,
                self.selected_backend,
                self._metrics,
                self._last_error,
            )

    def start(self) -> None:
        with self._lock:
            if self._state is not StreamLifecycleState.CONFIGURED:
                raise VideoRtpError("invalid_state", f"receiver is {self._state.value}", stream_id=self.stream_id)
            self._state = StreamLifecycleState.STARTING
            if self.endpoint is not None:
                if self._socket is None:
                    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                    udp_socket.bind(self.endpoint)
                    self._socket = udp_socket
                self._socket.settimeout(0.1)
                self._receive_thread = threading.Thread(
                    target=self._receive_loop,
                    name=f"rtp-recv-{self.stream_id}",
                    daemon=True,
                )
                self._process_thread = threading.Thread(
                    target=self._process_loop,
                    name=f"rtp-decode-{self.stream_id}",
                    daemon=True,
                )
                self._receive_thread.start()
                self._process_thread.start()
            self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME

    def enqueue_datagram(self, datagram: bytes, *, receive_time_ns: int | None = None) -> None:
        with self._lock:
            if self._state in {StreamLifecycleState.FAILED, StreamLifecycleState.STOPPED}:
                raise VideoRtpError("invalid_state", f"receiver is {self._state.value}", stream_id=self.stream_id)
            dropped = 0
            if len(self._queue) >= self._capacity:
                self._queue.popleft()
                dropped = 1
            self._queue.append((datagram, time.time_ns() if receive_time_ns is None else receive_time_ns))
            self._metrics = replace(
                self._metrics,
                queued_packets=len(self._queue),
                dropped_packets=self._metrics.dropped_packets + dropped,
                receiver_queue_overflow_drops=self._metrics.receiver_queue_overflow_drops + dropped,
            )
            if dropped:
                self._last_error = "receiver_queue_overflow"
            self._condition.notify()

    def process_pending(self) -> bool:
        with self._processing_lock:
            pending = self._pop_pending()
            if pending is None:
                return False
            datagram, receive_time_ns = pending
            self._process_datagram_locked(datagram, receive_time_ns=receive_time_ns)
            return True

    def _pop_pending(self) -> tuple[bytes, int] | None:
        with self._lock:
            if not self._queue:
                return None
            pending = self._queue.popleft()
            self._metrics = replace(self._metrics, queued_packets=len(self._queue))
            return pending

    def process_datagram(self, datagram: bytes, *, receive_time_ns: int) -> list[VideoFrame]:
        with self._processing_lock:
            return self._process_datagram_locked(datagram, receive_time_ns=receive_time_ns)

    def _process_datagram_locked(self, datagram: bytes, *, receive_time_ns: int) -> list[VideoFrame]:
        receive_monotonic_ns = time.monotonic_ns()
        try:
            packet = RtpPacket.from_bytes(datagram)
        except ValueError as exc:
            self._degrade("invalid_rtp_packet", str(exc), dropped_packets=1)
            return []
        if packet.ssrc != self.ssrc or packet.payload_type != self.payload_type:
            self._degrade(
                "stream_identity_mismatch", "RTP SSRC or payload type does not match descriptor", dropped_packets=1
            )
            return []
        with self._lock:
            jitter_ns = self._updated_jitter_ns(packet.timestamp, receive_monotonic_ns)
            self._metrics = replace(
                self._metrics,
                received_packets=self._metrics.received_packets + 1,
                receive_monotonic_ns=receive_monotonic_ns,
                jitter_ns=jitter_ns,
            )
        access_unit, lost_packets = self._depacketizer.push(packet)
        if self._depacketizer.last_reordered:
            dropped_capture_timestamp_ns = self._map_dropped_timestamp(packet.timestamp, receive_time_ns)
            with self._lock:
                self._metrics = replace(
                    self._metrics,
                    reordered_packets=self._metrics.reordered_packets + 1,
                    last_dropped_capture_timestamp_ns=dropped_capture_timestamp_ns,
                )
                self._last_error = "rtp_reordered_packet"
            return []
        if lost_packets:
            dropped_capture_timestamp_ns = self._map_dropped_timestamp(packet.timestamp, receive_time_ns)
            with self._lock:
                self._metrics = replace(
                    self._metrics,
                    last_dropped_capture_timestamp_ns=dropped_capture_timestamp_ns,
                )
            self._have_sps = False
            self._have_pps = False
            self._keyframe_ready = False
            self._degrade(
                "packet_loss",
                f"lost {lost_packets} RTP packets",
                lost_packets=lost_packets,
                sequence_gap_events=1,
            )
        if access_unit is None:
            if lost_packets:
                self._record_access_unit(
                    b"",
                    capture_timestamp_ns=None,
                    rtp_timestamp=packet.timestamp,
                    keyframe=False,
                    lost_packets=lost_packets,
                    dropped="rtp_sequence_gap",
                )
            return []
        self._have_sps = self._have_sps or access_unit.has_sps
        self._have_pps = self._have_pps or access_unit.has_pps
        if not self._keyframe_ready:
            if not (access_unit.keyframe and self._have_sps and self._have_pps):
                with self._lock:
                    if self._state is not StreamLifecycleState.DEGRADED:
                        self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME
                return []
            # A producer-side IDR recovery resets the encoder context.  A
            # stateful hardware decoder must be reset at the same boundary;
            # otherwise it can accept the new SPS/PPS and keep returning empty
            # output, which falsely drives another recovery request forever.
            # Do not reset a fresh decoder before its first keyframe.
            if self._metrics.decoded_frames > 0:
                try:
                    self.decoder.reset()
                except VideoCodecError as exc:
                    self._keyframe_ready = False
                    self._have_sps = False
                    self._have_pps = False
                    self._degrade("decoder_reset_failed", str(exc), decode_errors=1)
                    return []
            self._keyframe_ready = True
            with self._lock:
                self._metrics = replace(
                    self._metrics,
                    recovery_keyframes=self._metrics.recovery_keyframes + 1,
                )
        try:
            capture_timestamp_ns = self.timestamp_mapper.map(
                access_unit.timestamp,
                now_ns=receive_time_ns,
                session_generation=self.session_generation,
            )
        except ObservationSynchronizationError as exc:
            self._record_access_unit(
                access_unit.payload,
                capture_timestamp_ns=None,
                rtp_timestamp=access_unit.timestamp,
                keyframe=access_unit.keyframe,
                lost_packets=lost_packets,
                dropped="timestamp_unmapped",
            )
            self._keyframe_ready = False
            self._degrade("timestamp_mapping_unavailable", str(exc))
            return []
        self._record_access_unit(
            access_unit.payload,
            capture_timestamp_ns=capture_timestamp_ns,
            rtp_timestamp=access_unit.timestamp,
            keyframe=access_unit.keyframe,
            lost_packets=lost_packets,
        )
        if not self._decode:
            with self._lock:
                self._state = StreamLifecycleState.READY
                self._last_error = ""
            return []
        try:
            decode_start_monotonic_ns = time.monotonic_ns()
            frames = self.decoder.decode(
                EncodedPacket(
                    access_unit.payload,
                    access_unit.timestamp,
                    capture_timestamp_ns,
                    keyframe=access_unit.keyframe,
                )
            )
            decode_end_monotonic_ns = time.monotonic_ns()
        except VideoCodecError as exc:
            self._keyframe_ready = False
            self._have_sps = False
            self._have_pps = False
            self._degrade("decode_failed", str(exc), decode_errors=1)
            return []
        received_frames = []
        if frames:
            decoded_receive_time_ns = self._clock()
            for frame in frames:
                received_frame = replace(
                    frame,
                    receive_timestamp_ns=decoded_receive_time_ns,
                )
                self.frame_buffer.push(
                    received_frame.capture_timestamp_ns,
                    received_frame,
                    receive_time_ns=decoded_receive_time_ns,
                )
                received_frames.append(received_frame)
        with self._lock:
            if frames:
                self._state = StreamLifecycleState.READY
                self._last_error = ""
            elif self._state not in {StreamLifecycleState.READY, StreamLifecycleState.DEGRADED}:
                # Pipelined hardware decoders may accept an access unit without
                # producing its frame synchronously. Once a frame has made the
                # stream ready, an empty drain is not a new keyframe boundary.
                self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME
            decoder_metrics = self.decoder.metrics
            last_capture_timestamp_ns = (
                received_frames[-1].capture_timestamp_ns if received_frames else self._metrics.last_capture_timestamp_ns
            )
            self._metrics = replace(
                self._metrics,
                decoded_frames=self._metrics.decoded_frames + len(frames),
                last_capture_timestamp_ns=last_capture_timestamp_ns,
                decode_start_monotonic_ns=decode_start_monotonic_ns,
                decode_end_monotonic_ns=decode_end_monotonic_ns,
                decoder_backlog_depth=decoder_metrics.decoder_backlog_depth,
                decoder_output_age_ns=decoder_metrics.decoder_output_age_ns,
                dropped_stale_decoder_frames=decoder_metrics.dropped_stale_decoder_frames,
                metadata_fifo_depth=decoder_metrics.metadata_fifo_depth,
                decoder_input_frame_rate_hz=decoder_metrics.input_frame_rate_hz,
                decoder_output_frame_rate_hz=decoder_metrics.output_frame_rate_hz,
            )
        return received_frames

    def reset(self, session_generation: int) -> None:
        if session_generation < 1:
            raise ValueError("session_generation must be positive")
        with self._processing_lock, self._lock:
            self._queue.clear()
            self._depacketizer.reset()
            self.decoder.reset()
            self.frame_buffer.reset()
            self.timestamp_mapper.reset(session_generation)
            self.session_generation = session_generation
            self._frame_count = 0
            self._have_sps = False
            self._have_pps = False
            self._keyframe_ready = False
            self._last_error = ""
            self._state = StreamLifecycleState.WAITING_FOR_KEYFRAME
            self._metrics = replace(
                self._metrics,
                queued_packets=0,
                reconnect_count=self._metrics.reconnect_count + 1,
            )

    def close(self, timeout_s: float = 1.0) -> None:
        if timeout_s < 0:
            raise ValueError("timeout_s cannot be negative")
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._socket is not None:
            self._socket.close()
        deadline = time.monotonic() + timeout_s
        for thread in (self._receive_thread, self._process_thread):
            if thread is None:
                continue
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise VideoRtpError("shutdown_timeout", "receiver did not stop in time", stream_id=self.stream_id)
        with self._processing_lock:
            self.decoder.close(timeout_s)
        with self._lock:
            self._queue.clear()
            self._state = StreamLifecycleState.STOPPED
            self._metrics = replace(self._metrics, queued_packets=0)

    def _map_dropped_timestamp(self, rtp_timestamp: int, receive_time_ns: int) -> int:
        try:
            return self.timestamp_mapper.map(
                rtp_timestamp,
                now_ns=receive_time_ns,
                session_generation=self.session_generation,
            )
        except ObservationSynchronizationError:
            return 0

    def _record_access_unit(
        self,
        payload: bytes,
        *,
        capture_timestamp_ns: int | None,
        rtp_timestamp: int,
        keyframe: bool,
        lost_packets: int,
        dropped: str | None = None,
    ) -> None:
        """Record one reconstructed access unit without disrupting reception on I/O errors."""
        if self._recorder is None:
            return
        recording_generation = self._recorder.recording_generation()
        if recording_generation is None:
            self._recording_generation = None
            return
        if recording_generation != self._recording_generation:
            self._frame_count = 0
            self._recording_generation = recording_generation
        frame_index = self._frame_count
        self._frame_count += 1
        try:
            self._recorder.write_access_unit(
                payload=payload,
                capture_timestamp_ns=capture_timestamp_ns,
                rtp_timestamp=rtp_timestamp,
                frame_index=frame_index,
                keyframe=keyframe,
                lost_packets=lost_packets,
                session_generation=self.session_generation,
                dropped=dropped,
            )
        except (OSError, ValueError) as exc:
            with self._lock:
                self._last_error = f"recording_failed: {exc}"

    def _receive_loop(self) -> None:
        assert self._socket is not None
        while True:
            with self._lock:
                if self._stopping:
                    return
            try:
                datagram, _source = self._socket.recvfrom(self._max_datagram_size)
            except TimeoutError:
                continue
            except OSError as exc:
                with self._lock:
                    if self._stopping:
                        return
                self.fail(exc)
                return
            self.enqueue_datagram(datagram)

    def _process_loop(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
            try:
                self.process_pending()
            except Exception as exc:
                self.fail(exc)
                return

    def fail(self, error: Exception) -> None:
        with self._lock:
            self._state = StreamLifecycleState.FAILED
            self._last_error = str(error)
            self._metrics = replace(self._metrics, errors=self._metrics.errors + 1)

    def _degrade(
        self,
        code: str,
        message: str,
        *,
        dropped_packets: int = 0,
        lost_packets: int = 0,
        decode_errors: int = 0,
        sequence_gap_events: int = 0,
    ) -> None:
        with self._lock:
            self._state = StreamLifecycleState.DEGRADED
            self._last_error = f"{code}: {message}"
            self._metrics = replace(
                self._metrics,
                dropped_packets=self._metrics.dropped_packets + dropped_packets,
                lost_packets=self._metrics.lost_packets + lost_packets,
                decode_errors=self._metrics.decode_errors + decode_errors,
                sequence_gap_events=self._metrics.sequence_gap_events + sequence_gap_events,
                errors=self._metrics.errors + 1,
            )

    def _updated_jitter_ns(self, rtp_timestamp: int, receive_monotonic_ns: int) -> int:
        transit_ns = receive_monotonic_ns - round(rtp_timestamp * 1_000_000_000 / 90_000)
        previous = getattr(self, "_last_transit_ns", None)
        self._last_transit_ns = transit_ns
        if previous is None:
            return self._metrics.jitter_ns
        delta = abs(transit_ns - previous)
        return round(self._metrics.jitter_ns + (delta - self._metrics.jitter_ns) / 16)


__all__ = [
    "DatagramReceiver",
    "DatagramSender",
    "H264AccessUnit",
    "H264Depacketizer",
    "H264RtpReceiver",
    "H264RtpSender",
    "RtpPacket",
    "StreamLifecycleState",
    "StreamMetrics",
    "StreamStatus",
    "VideoRtpError",
    "packetize_h264",
    "split_annex_b",
]

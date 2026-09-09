"""H.264 RTP stream recorder for dataset collection.

Records H.264 access units from RTP receivers to Annex-B format files with
NDJSON sidecar metadata for integrity tracking and converter timestamp alignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal

from robot_config.observation_transport import NON_FAULT_DROP_REASONS


@dataclass(frozen=True, slots=True)
class AccessUnitMetadata:
    """Per-frame metadata written to NDJSON sidecar."""

    frame_index: int
    capture_timestamp_ns: int | None
    rtp_timestamp: int
    keyframe: bool
    lost_packets: int
    session_generation: int
    dropped: str | None = None  # Reason if frame was dropped (e.g., "timestamp_unmapped")


class H264StreamRecorder:
    """Records H.264 access units to Annex-B stream with NDJSON sidecar.

    Thread-safe: All public methods can be called from RTP receiver threads.
    Episode lifecycle: start_episode() → write_access_unit() (N times) → stop_episode()
    """

    _ANNEX_B_START_CODE = b"\x00\x00\x00\x01"
    # Annex-B permits a three-byte start code as well; only the four-byte form is
    # written, but both count as "already prefixed" when inspecting a payload.
    _ANNEX_B_SHORT_START_CODE = b"\x00\x00\x01"

    def __init__(self, *, integrity_mode: Literal["strict", "tolerant"] = "strict") -> None:
        """Initialize recorder with integrity policy.

        Args:
            integrity_mode: "strict" discards episodes with any frame drops;
                          "tolerant" preserves episodes with gap metadata.
        """
        if integrity_mode not in {"strict", "tolerant"}:
            raise ValueError(f"integrity_mode must be 'strict' or 'tolerant', got '{integrity_mode}'")

        self._integrity_mode = integrity_mode
        self._lock = Lock()

        # Episode state (protected by lock)
        self._episode_dir: Path | None = None
        self._obs_key: str | None = None
        self._stream_file: Path | None = None
        self._sidecar_file: Path | None = None
        self._last_stream_file: Path | None = None
        self._last_sidecar_file: Path | None = None
        self._stream_handle = None
        self._sidecar_handle = None
        self._frame_count: int = 0
        self._invalid: bool = False
        self._initial_generation: int | None = None
        self._waiting_for_keyframe: bool = True
        self._lost_packets = 0
        self._timestamp_mapping_failures = 0
        self._recording_generation = 0

    def start_episode(self, episode_dir: Path, obs_key: str) -> None:
        """Start recording an episode.

        Args:
            episode_dir: Episode directory path (e.g., episodes/episode_000042/)
            obs_key: Observation key with dots replaced by underscores in filename
                    (e.g., "observation.images.top" → "observation.images.top.h264")

        Raises:
            ValueError: If already recording
            OSError: If files cannot be created
        """
        with self._lock:
            if self._episode_dir is not None:
                raise ValueError("Cannot start new episode while previous episode is active")

            self._episode_dir = Path(episode_dir)
            self._obs_key = obs_key
            self._frame_count = 0
            self._invalid = False
            self._initial_generation = None
            self._waiting_for_keyframe = True
            self._recording_generation += 1
            self._lost_packets = 0
            self._timestamp_mapping_failures = 0

            # Create files: observation.images.top.h264 and observation.images.top.h264.json
            safe_key = obs_key.replace("/", "_")  # sanitize if topic-like
            self._stream_file = self._episode_dir / f"{safe_key}.h264"
            self._sidecar_file = self._episode_dir / f"{safe_key}.h264.json"
            self._last_stream_file = self._stream_file
            self._last_sidecar_file = self._sidecar_file

            self._episode_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._stream_handle = self._stream_file.open("wb")
                self._sidecar_handle = self._sidecar_file.open("w", encoding="utf-8")
            except OSError:
                if self._stream_handle is not None:
                    self._stream_handle.close()
                for path in (self._stream_file, self._sidecar_file):
                    if path.exists():
                        path.unlink()
                self._episode_dir = None
                self._stream_handle = None
                self._sidecar_handle = None
                raise

    def write_access_unit(
        self,
        *,
        payload: bytes,
        capture_timestamp_ns: int | None,
        rtp_timestamp: int,
        frame_index: int,
        keyframe: bool,
        lost_packets: int,
        session_generation: int,
        dropped: str | None = None,
    ) -> None:
        """Write an access unit to the stream and metadata to sidecar.

        Args:
            payload: H.264 NAL unit(s) for this access unit
            capture_timestamp_ns: RTP-mapped capture timestamp (None if mapping failed)
            rtp_timestamp: RTP timestamp in 90kHz ticks
            frame_index: Sequential frame index from receiver
            keyframe: True if this is an IDR frame
            lost_packets: Count of RTP packets lost since previous frame
            session_generation: Receiver session generation
            dropped: Reason string if frame was dropped (e.g., "timestamp_unmapped")

        Thread-safe: Can be called from RTP receiver processing thread.
        """
        with self._lock:
            if self._episode_dir is None:
                raise ValueError("No active episode; call start_episode() first")

            # Check session generation consistency
            if self._initial_generation is None:
                self._initial_generation = session_generation
            elif session_generation != self._initial_generation:
                lost_packets = max(lost_packets, 1)  # Treat generation change as gap

            if self._waiting_for_keyframe:
                # An episode that opens mid-stream begins on P-frames that
                # predict from access units it never recorded. Writing them
                # would inflate the sidecar with entries no decoder can turn
                # into frames, so hold every payload back until the first clean
                # keyframe arrives.
                if keyframe and dropped is None and payload:
                    self._waiting_for_keyframe = False
                else:
                    payload = b""
                    dropped = dropped or "pre_keyframe"

            # Detect integrity violations. Waiting for the opening keyframe is
            # normal stream entry rather than a transport fault.
            has_gap = lost_packets > 0 or (dropped is not None and dropped not in NON_FAULT_DROP_REASONS)
            self._lost_packets += lost_packets
            if dropped == "timestamp_unmapped":
                self._timestamp_mapping_failures += 1
            if has_gap and not self._invalid:
                self._invalid = True
                if self._integrity_mode == "strict":
                    # Strict mode: mark invalid immediately but continue writing
                    # (files will be deleted in stop_episode)
                    pass

            # Write Annex-B access unit (even if dropped, for frame count consistency)
            if not dropped and payload:
                # The depacketizer already emits Annex-B, so prefixing unconditionally
                # would open every access unit with a zero-length NAL. Decoders skip
                # those, but the stream is malformed and each frame wastes four bytes.
                # Both Annex-B start-code forms count as already prefixed, so the guard
                # matches what the format permits rather than what today's depacketizer
                # happens to emit.
                if not payload.startswith((self._ANNEX_B_START_CODE, self._ANNEX_B_SHORT_START_CODE)):
                    self._stream_handle.write(self._ANNEX_B_START_CODE)
                self._stream_handle.write(payload)
                self._stream_handle.flush()

            # Write sidecar metadata entry (NDJSON: one JSON object per line)
            metadata = AccessUnitMetadata(
                frame_index=frame_index,
                capture_timestamp_ns=capture_timestamp_ns,
                rtp_timestamp=rtp_timestamp,
                keyframe=keyframe,
                lost_packets=lost_packets,
                session_generation=session_generation,
                dropped=dropped,
            )
            # Use dataclass __dict__ to get clean JSON (None fields included)
            json.dump(
                {
                    "frame_index": metadata.frame_index,
                    "capture_timestamp_ns": metadata.capture_timestamp_ns,
                    "rtp_timestamp": metadata.rtp_timestamp,
                    "keyframe": metadata.keyframe,
                    "lost_packets": metadata.lost_packets,
                    "session_generation": metadata.session_generation,
                    "dropped": metadata.dropped,
                },
                self._sidecar_handle,
            )
            self._sidecar_handle.write("\n")
            self._sidecar_handle.flush()

            self._frame_count += 1

    def stop_episode(self) -> bool:
        """Stop recording and close files.

        Returns:
            True if recorded files were preserved; False if discarded.

        Strict mode: Deletes files and returns False if episode was marked invalid.
        Tolerant mode: Always preserves files and returns True; integrity gaps are
        recorded in the sidecar for downstream filtering.

        Use :meth:`is_invalid` before stopping to inspect integrity status
        independently of whether files were kept.
        """
        with self._lock:
            if self._episode_dir is None:
                return True  # No active episode, nothing to do

            try:
                if self._stream_handle is not None:
                    self._stream_handle.close()
                if self._sidecar_handle is not None:
                    self._sidecar_handle.close()

                # Strict mode: delete files if invalid
                if self._integrity_mode == "strict" and self._invalid:
                    if self._stream_file and self._stream_file.exists():
                        self._stream_file.unlink()
                    if self._sidecar_file and self._sidecar_file.exists():
                        self._sidecar_file.unlink()
                    return False  # Files discarded

                # Tolerant mode or strict mode with no gaps: files preserved
                return True

            finally:
                # Reset state
                self._episode_dir = None
                self._obs_key = None
                self._stream_file = None
                self._sidecar_file = None
                self._stream_handle = None
                self._sidecar_handle = None
                self._frame_count = 0
                self._invalid = False
                self._initial_generation = None
                self._lost_packets = 0
                self._timestamp_mapping_failures = 0

    def discard_files(self) -> None:
        """Delete this recorder's output for the most recently stopped episode."""
        with self._lock:
            for path in (self._last_stream_file, self._last_sidecar_file):
                if path is not None and path.exists():
                    path.unlink()

    def is_recording(self) -> bool:
        """Check if currently recording an episode."""
        with self._lock:
            return self._episode_dir is not None

    def recording_generation(self) -> int | None:
        """Return the active episode generation, or None when recording is inactive."""
        with self._lock:
            return self._recording_generation if self._episode_dir is not None else None

    def is_invalid(self) -> bool:
        """Check if current episode has been marked invalid."""
        with self._lock:
            return self._invalid

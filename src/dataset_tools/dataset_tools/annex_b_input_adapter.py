"""Annex-B H.264 + NDJSON sidecar input adapter for bag_to_lerobot converter.

Reads video streams recorded by cross-device RTP observation transport:
  - observation.images.top.h264        (Annex-B bitstream)
  - observation.images.top.h264.json   (NDJSON sidecar with timestamps + integrity metadata)

The sidecar contains one entry per access unit (including dropped frames that have
no payload in the bitstream). Adapter pairs decoded frames with non-dropped sidecar
entries by position after filtering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av

from dataset_tools.bag_to_lerobot import IntegrityReport, VideoFrameEntry, VideoInputAdapter
from robot_config.observation_transport import NON_FAULT_DROP_REASONS


class AnnexBInputAdapter(VideoInputAdapter):
    """Read H.264 Annex-B streams with NDJSON sidecar metadata.

    On-disk layout produced by RTP recording:
        observation.images.top.h264        # Annex-B bitstream (start codes + NALs)
        observation.images.top.h264.json   # NDJSON: one JSON object per line

    Sidecar schema (per design D1):
        {
            "frame_index": 0,
            "capture_timestamp_ns": 1000000000,  # null when dropped
            "rtp_timestamp": 90000,
            "keyframe": true,
            "lost_packets": 0,
            "session_generation": 1,
            "dropped": null  # or "timestamp_unmapped"
        }

    Pairing logic:
        1. Decode .h264 → N frames
        2. Parse sidecar → M entries (M ≥ N due to dropped frames in sidecar)
        3. Filter dropped entries (dropped != null) → M' non-dropped entries
        4. Verify M' == N, then pair: decoded_frame[i] ↔ non_dropped_sidecar[i]

    Integrity reporting:
        IntegrityReport aggregates all gaps:
        - lost_packets > 0  (RTP packet loss, frame still decoded)
        - dropped != null   (frame not in bitstream, only in sidecar)
    """

    def __init__(self, h264_path: Path) -> None:
        """Open H.264 file and parse sidecar.

        Parameters
        ----------
        h264_path : Path
            Path to `.h264` file; sidecar is inferred as `{stem}.h264.json`.

        Raises
        ------
        FileNotFoundError
            When sidecar does not exist.
        ValueError
            When sidecar format is invalid or pairing fails.
        """
        self._h264_path = h264_path
        self._sidecar_path = h264_path.with_suffix(h264_path.suffix + ".json")
        if not self._sidecar_path.exists():
            raise FileNotFoundError(f"Sidecar not found: {self._sidecar_path}")

        # Parse sidecar (NDJSON: one JSON object per line)
        self._sidecar_entries = self._load_sidecar(self._sidecar_path)

        # Separate dropped vs non-dropped entries
        self._dropped_entries = [e for e in self._sidecar_entries if e.get("dropped") is not None]
        self._non_dropped_entries = [e for e in self._sidecar_entries if e.get("dropped") is None]

        # Observation key derived from filename (e.g., "observation.images.top.h264" → "observation.images.top")
        self._obs_key = h264_path.stem

        self._frames: list[VideoFrameEntry] | None = None

    @staticmethod
    def _load_sidecar(path: Path) -> list[dict[str, Any]]:
        """Load NDJSON sidecar (one JSON object per line)."""
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{lineno}: {exc}") from exc
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid sidecar entry at {path}:{lineno}: expected a JSON object")
                entries.append(entry)

        if not entries:
            raise ValueError(f"Sidecar is empty: {path}")

        required_fields = {
            "frame_index",
            "capture_timestamp_ns",
            "rtp_timestamp",
            "keyframe",
            "lost_packets",
            "session_generation",
        }
        previous_frame_index = -1
        for index, entry in enumerate(entries, start=1):
            missing = required_fields - entry.keys()
            if missing:
                raise ValueError(f"Invalid sidecar entry at {path}:{index}: missing {sorted(missing)}")

            frame_index = entry["frame_index"]
            if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index <= previous_frame_index:
                raise ValueError(
                    f"Invalid sidecar entry at {path}:{index}: frame_index must be a strictly increasing integer"
                )
            previous_frame_index = frame_index

            dropped = entry.get("dropped")
            if dropped is None:
                timestamp_ns = entry["capture_timestamp_ns"]
                if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool):
                    raise ValueError(
                        f"Invalid sidecar entry at {path}:{index}: non-dropped frame requires integer "
                        "capture_timestamp_ns"
                    )
            elif not isinstance(dropped, str) or not dropped:
                raise ValueError(f"Invalid sidecar entry at {path}:{index}: dropped must be null or a reason string")

            lost_packets = entry["lost_packets"]
            if not isinstance(lost_packets, int) or isinstance(lost_packets, bool) or lost_packets < 0:
                raise ValueError(
                    f"Invalid sidecar entry at {path}:{index}: lost_packets must be a non-negative integer"
                )
        return entries

    def list_observations(self) -> list[str]:
        """Return the single observation key served by this adapter."""
        return [self._obs_key]

    def read_frames(self, obs_key: str) -> list[VideoFrameEntry]:
        """Decode .h264 and pair with non-dropped sidecar entries.

        Parameters
        ----------
        obs_key : str
            Observation key to read (must match this adapter's key).

        Returns
        -------
        list[VideoFrameEntry]
            Frames in sidecar order (capture timestamp, not decode order).

        Raises
        ------
        ValueError
            When obs_key does not match or pairing fails.
        """
        if obs_key != self._obs_key:
            raise ValueError(f"Adapter serves {self._obs_key}, not {obs_key}")

        if self._frames is None:
            decoded_images = self._decode_h264()
            if len(self._non_dropped_entries) != len(decoded_images):
                raise ValueError(
                    f"Pairing mismatch for {obs_key}: {len(self._non_dropped_entries)} non-dropped "
                    f"sidecar entries != {len(decoded_images)} decoded frames"
                )

            self._frames = [
                VideoFrameEntry(
                    timestamp_ns=entry["capture_timestamp_ns"],
                    image=image,
                    keyframe=bool(entry["keyframe"]),
                    lost_packets=entry["lost_packets"],
                )
                for image, entry in zip(decoded_images, self._non_dropped_entries, strict=True)
            ]
        return self._frames

    def _decode_h264(self) -> list[Any]:
        """Decode the Annex-B stream to RGB arrays and close the input immediately."""
        try:
            with av.open(str(self._h264_path), "r", format="h264") as container:
                return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        except (av.error.FFmpegError, OSError) as exc:
            raise ValueError(f"Failed to decode Annex-B stream {self._h264_path}: {exc}") from exc

    def integrity_report(self, obs_key: str) -> IntegrityReport:
        """Build integrity report from sidecar metadata.

        Aggregates all gaps:
        - lost_packets > 0  (RTP packet loss)
        - dropped != null   (timestamp mapping failure, frame not in bitstream)

        ``pre_keyframe`` is excluded: it marks the access units held back before the
        first clean keyframe, which is how a recording enters a running stream rather
        than a transport fault. ``h264_stream_recorder`` makes the same exclusion when
        it computes ``has_gap``; the two sides have to agree on a given ``dropped``
        value or a normal episode is reported as unclean.

        Parameters
        ----------
        obs_key : str
            Observation key (must match this adapter's key).

        Returns
        -------
        IntegrityReport
            Summary with ``clean`` flag and ``frame_gaps`` list.
        """
        if obs_key != self._obs_key:
            raise ValueError(f"Adapter serves {self._obs_key}, not {obs_key}")

        gaps = []

        for entry in self._sidecar_entries:
            dropped = entry.get("dropped")
            if dropped is not None and dropped not in NON_FAULT_DROP_REASONS:
                gaps.append(
                    {
                        "frame_index": entry["frame_index"],
                        "reason": dropped,
                    }
                )
            elif entry["lost_packets"] > 0:
                gaps.append(
                    {
                        "frame_index": entry["frame_index"],
                        "lost_packets": entry["lost_packets"],
                        "reason": "rtp_sequence_gap",
                    }
                )

        clean = len(gaps) == 0
        return IntegrityReport(clean=clean, frame_gaps=gaps if not clean else None)

    def close(self) -> None:
        """Release cached decoded frames."""
        self._frames = None

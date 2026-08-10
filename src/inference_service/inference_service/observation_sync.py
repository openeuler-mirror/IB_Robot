"""Timestamp mapping and synchronized selection for streamed observations."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass

from robot_config.contract_utils import StreamBuffer

RTP_CLOCK_RATE = 90_000
_RTP_MODULUS = 1 << 32
_RTP_HALF_MODULUS = 1 << 31


@dataclass(frozen=True, slots=True)
class SynchronizationIssue:
    reason: str
    observation_key: str
    stream_id: str
    details: Mapping[str, object]


class ObservationSynchronizationError(RuntimeError):
    code = "observation_not_ready"
    recoverable = True
    stage = "observation_sync"

    def __init__(self, issues: list[SynchronizationIssue]) -> None:
        self.issues = tuple(issues)
        self.details = {
            "streams": [
                {
                    "reason": issue.reason,
                    "observation_key": issue.observation_key,
                    "stream_id": issue.stream_id,
                    **dict(issue.details),
                }
                for issue in issues
            ]
        }
        summary = ", ".join(
            f"{issue.stream_id} ({issue.observation_key}): {issue.reason}"
            + (
                f" ({issue.details['selected_timestamp_ns'] / 1_000_000:.3f}ms, "
                f"skew={issue.details['skew_ns'] / 1_000_000:.3f}ms)"
                if issue.reason == "skewed"
                else ""
            )
            for issue in issues
        )
        super().__init__(f"streamed observations are not ready: {summary}")


class TargetFrameDroppedError(RuntimeError):
    """Definitive failure for a requested frame known to be irrecoverably dropped."""

    code = "target_frame_dropped"
    recoverable = False
    stage = "observation_sync"

    def __init__(self, issues: list[SynchronizationIssue]) -> None:
        self.issues = tuple(issues)
        self.details = {
            "streams": [
                {
                    "reason": issue.reason,
                    "observation_key": issue.observation_key,
                    "stream_id": issue.stream_id,
                    **dict(issue.details),
                }
                for issue in issues
            ]
        }
        summary = ", ".join(f"{issue.stream_id} ({issue.observation_key}): {issue.reason}" for issue in issues)
        super().__init__(f"requested streamed observation was dropped: {summary}")


@dataclass(frozen=True, slots=True)
class StreamSelection:
    observation_key: str
    stream_id: str
    buffer: StreamBuffer
    timestamp_mapping_ready: bool = True
    keyframe_ready: bool = True
    pad_before_first: bool = False
    future_tolerance_ns: int = 0
    session_generation: int = 0
    last_dropped_capture_timestamp_ns: int = 0
    last_dropped_admission_id: str = ""
    dropped_capture_history: tuple[tuple[int, str, str], ...] = ()
    last_drop_reason: str = ""


@dataclass(frozen=True, slots=True)
class SelectedStreamValue:
    observation_key: str
    stream_id: str
    capture_timestamp_ns: int
    receive_timestamp_ns: int
    value: object


class RtpTimestampMapper:
    """Map 32-bit 90 kHz RTP timestamps into a session capture-time clock."""

    def __init__(self, max_mapping_age_ns: int, *, observation_key: str, stream_id: str) -> None:
        if max_mapping_age_ns <= 0:
            raise ValueError("max_mapping_age_ns must be positive")
        if not observation_key or not stream_id:
            raise ValueError("timestamp mapper requires observation_key and stream_id")
        self.max_mapping_age_ns = int(max_mapping_age_ns)
        self.observation_key = observation_key
        self.stream_id = stream_id
        self._lock = threading.RLock()
        self.reset()

    def reset(self, session_generation: int = 0) -> None:
        if session_generation < 0:
            raise ValueError("session_generation cannot be negative")
        with self._lock:
            self.session_generation = int(session_generation)
            self._rtp_timestamp: int | None = None
            self._capture_timestamp_ns: int | None = None
            self._mapping_receive_time_ns: int | None = None

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._rtp_timestamp is not None

    def update(
        self,
        rtp_timestamp: int,
        capture_timestamp_ns: int,
        receive_time_ns: int,
        *,
        session_generation: int,
    ) -> None:
        if session_generation < 1:
            raise ValueError("timestamp mappings require a positive session generation")
        if not 0 <= rtp_timestamp < _RTP_MODULUS:
            raise ValueError("rtp_timestamp must fit in uint32")
        if capture_timestamp_ns < 0 or receive_time_ns < 0:
            raise ValueError("capture and receive timestamps cannot be negative")
        with self._lock:
            if self.session_generation != session_generation:
                self.reset(session_generation)
            self._rtp_timestamp = int(rtp_timestamp)
            self._capture_timestamp_ns = int(capture_timestamp_ns)
            self._mapping_receive_time_ns = int(receive_time_ns)

    def map(self, rtp_timestamp: int, *, now_ns: int, session_generation: int) -> int:
        if not 0 <= rtp_timestamp < _RTP_MODULUS:
            raise ValueError("rtp_timestamp must fit in uint32")
        with self._lock:
            if session_generation != self.session_generation or self._rtp_timestamp is None:
                raise ObservationSynchronizationError([self._mapping_issue("unmapped", session_generation)])
            assert self._capture_timestamp_ns is not None
            assert self._mapping_receive_time_ns is not None
            mapping_age_ns = max(0, int(now_ns) - self._mapping_receive_time_ns)
            if mapping_age_ns > self.max_mapping_age_ns:
                raise ObservationSynchronizationError(
                    [
                        self._mapping_issue(
                            "stale",
                            session_generation,
                            constraint="timestamp_mapping",
                            mapping_age_ns=mapping_age_ns,
                            max_mapping_age_ns=self.max_mapping_age_ns,
                        )
                    ]
                )
            delta = (int(rtp_timestamp) - self._rtp_timestamp + _RTP_HALF_MODULUS) % _RTP_MODULUS
            delta -= _RTP_HALF_MODULUS
            return self._capture_timestamp_ns + round(delta * 1_000_000_000 / RTP_CLOCK_RATE)

    def _mapping_issue(self, reason: str, session_generation: int, **details: object) -> SynchronizationIssue:
        return SynchronizationIssue(
            reason=reason,
            observation_key=self.observation_key,
            stream_id=self.stream_id,
            details={"session_generation": session_generation, **details},
        )


def select_synchronized_streams(
    streams: Mapping[str, StreamSelection],
    target_timestamp_ns: int,
    *,
    now_ns: int,
    max_inter_camera_skew_ns: int,
) -> dict[str, SelectedStreamValue]:
    """Select all required streams against one target or fail as one operation."""
    if max_inter_camera_skew_ns < 0:
        raise ValueError("max_inter_camera_skew_ns cannot be negative")
    selected, issues = _select_stream_entries(streams, target_timestamp_ns, now_ns=now_ns)
    if issues:
        raise ObservationSynchronizationError(issues)
    timestamps = [item.capture_timestamp_ns for item in selected.values()]
    if not timestamps or max(timestamps) - min(timestamps) <= max_inter_camera_skew_ns:
        return selected

    aligned = _find_latest_aligned_selection(
        streams,
        target_timestamp_ns,
        now_ns=now_ns,
        max_inter_camera_skew_ns=max_inter_camera_skew_ns,
    )
    if aligned is not None:
        return aligned

    minimum_timestamp_ns = min(timestamps)
    maximum_timestamp_ns = max(timestamps)
    skew_ns = maximum_timestamp_ns - minimum_timestamp_ns
    issues = [
        _stream_issue(
            streams[key],
            "skewed",
            selected_timestamp_ns=item.capture_timestamp_ns,
            minimum_timestamp_ns=minimum_timestamp_ns,
            maximum_timestamp_ns=maximum_timestamp_ns,
            skew_ns=skew_ns,
            max_inter_camera_skew_ns=max_inter_camera_skew_ns,
            target_timestamp_ns=target_timestamp_ns,
            session_generation=streams[key].session_generation,
        )
        for key, item in selected.items()
    ]
    raise ObservationSynchronizationError(issues)


def _find_latest_aligned_selection(
    streams: Mapping[str, StreamSelection],
    target_timestamp_ns: int,
    *,
    now_ns: int,
    max_inter_camera_skew_ns: int,
) -> dict[str, SelectedStreamValue] | None:
    """Find the newest real timestamp combination within the skew bound.

    Do not use the normal ``hold`` lookup for this search: hold can return an
    older sample for one stream and a newer sample for another, which recreates
    the very cross-camera skew this fallback is meant to avoid.
    """
    histories = {
        key: tuple(entry for entry in stream.buffer.entries() if entry[0] <= target_timestamp_ns)
        for key, stream in streams.items()
    }
    if any(not entries for entries in histories.values()):
        return None
    best: tuple[int, int, dict[str, SelectedStreamValue]] | None = None
    for entries in histories.values():
        for candidate_timestamp_ns, _receive_timestamp_ns, _value in entries:
            chosen: dict[str, SelectedStreamValue] = {}
            for observation_key, stream in streams.items():
                entry = _nearest_entry(stream.buffer.entries(), candidate_timestamp_ns)
                if entry is None:
                    break
                capture_timestamp_ns, receive_timestamp_ns, value = entry
                chosen[observation_key] = SelectedStreamValue(
                    observation_key=observation_key,
                    stream_id=stream.stream_id,
                    capture_timestamp_ns=capture_timestamp_ns,
                    receive_timestamp_ns=receive_timestamp_ns,
                    value=value,
                )
            if len(chosen) != len(streams):
                continue
            timestamps = [item.capture_timestamp_ns for item in chosen.values()]
            skew_ns = max(timestamps) - min(timestamps)
            if skew_ns > max_inter_camera_skew_ns:
                continue
            newest_ns = min(timestamps)
            candidate = (newest_ns, -skew_ns, chosen)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return None if best is None else best[2]


def _nearest_entry(
    entries: tuple[tuple[int, int, object], ...], target_timestamp_ns: int
) -> tuple[int, int, object] | None:
    return min(
        (entry for entry in entries if abs(entry[0] - target_timestamp_ns) <= 50_000_000),
        key=lambda entry: (abs(entry[0] - target_timestamp_ns), -entry[0]),
        default=None,
    )


def _select_stream_entries(
    streams: Mapping[str, StreamSelection],
    target_timestamp_ns: int,
    *,
    now_ns: int,
) -> tuple[dict[str, SelectedStreamValue], list[SynchronizationIssue]]:
    selected: dict[str, SelectedStreamValue] = {}
    issues: list[SynchronizationIssue] = []
    dropped_issues: list[SynchronizationIssue] = []
    for observation_key, stream in streams.items():
        if observation_key != stream.observation_key:
            raise ValueError(f"stream mapping key {observation_key!r} does not match its observation key")
        dropped = next(
            (item for item in reversed(stream.dropped_capture_history) if item[0] == target_timestamp_ns),
            None,
        )
        if dropped is None and stream.last_dropped_capture_timestamp_ns == target_timestamp_ns:
            dropped = (
                target_timestamp_ns,
                stream.last_dropped_admission_id,
                stream.last_drop_reason,
            )
        if dropped is not None:
            dropped_issues.append(
                _stream_issue(
                    stream,
                    "target_frame_dropped",
                    target_timestamp_ns=target_timestamp_ns,
                    session_generation=stream.session_generation,
                    admission_id=dropped[1],
                    drop_reason=dropped[2],
                )
            )
            continue
        if not stream.timestamp_mapping_ready:
            issues.append(
                _stream_issue(
                    stream,
                    "unmapped",
                    target_timestamp_ns=target_timestamp_ns,
                    session_generation=stream.session_generation,
                )
            )
            continue
        if not stream.keyframe_ready:
            issues.append(
                _stream_issue(
                    stream,
                    "pre_keyframe",
                    target_timestamp_ns=target_timestamp_ns,
                    session_generation=stream.session_generation,
                )
            )
            continue
        item, issue = stream.buffer.select_entry(target_timestamp_ns, now_ns=now_ns)
        if issue is not None:
            future = None
            if issue.get("reason") == "newer_than_request" and stream.pad_before_first:
                future = stream.buffer.first_entry()
            elif stream.future_tolerance_ns > 0:
                candidate = stream.buffer.first_entry_after(target_timestamp_ns)
                candidate_live_age_ns = max(0, now_ns - candidate[1]) if candidate is not None else 0
                if (
                    candidate is not None
                    and candidate[0] - target_timestamp_ns <= stream.future_tolerance_ns
                    and (stream.buffer.max_age_ns <= 0 or candidate_live_age_ns <= stream.buffer.max_age_ns)
                ):
                    future = candidate
            if future is not None:
                item = future
                issue = None
        if issue is not None:
            reason = str(issue["reason"])
            if reason == "newer_than_request":
                reason = "missing"
            issues.append(
                _stream_issue(
                    stream,
                    reason,
                    target_timestamp_ns=target_timestamp_ns,
                    session_generation=stream.session_generation,
                    **{key: value for key, value in issue.items() if key != "reason"},
                )
            )
            continue
        assert item is not None
        capture_timestamp_ns, receive_timestamp_ns, value = item
        selected[observation_key] = SelectedStreamValue(
            observation_key=observation_key,
            stream_id=stream.stream_id,
            capture_timestamp_ns=capture_timestamp_ns,
            receive_timestamp_ns=receive_timestamp_ns,
            value=value,
        )
    if dropped_issues:
        raise TargetFrameDroppedError(dropped_issues)
    return selected, issues


def _stream_issue(stream: StreamSelection, reason: str, **details: object) -> SynchronizationIssue:
    return SynchronizationIssue(
        reason=reason,
        observation_key=stream.observation_key,
        stream_id=stream.stream_id,
        details=details,
    )

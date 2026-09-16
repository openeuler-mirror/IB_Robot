"""Shared helpers for reading SO-101 Feetech motor currents."""

from __future__ import annotations

from collections.abc import Iterable

from so101_hardware.calibration.constants import CURRENT_RAW_TO_AMPERE

# Feetech STS3215 Present_Current uses bit 15 as sign bit (sign-magnitude).
# The LeRobot SDK does NOT decode this (Present_Current is absent from
# STS_SMS_SERIES_ENCODINGS_TABLE), so we handle it here.
_CURRENT_SIGN_BIT = 15


def _decode_present_current(raw: int) -> int:
    if raw & (1 << _CURRENT_SIGN_BIT):
        return -(raw & ((1 << _CURRENT_SIGN_BIT) - 1))
    return raw


def read_motor_currents(bus, joint_names: Iterable[str], logger, warning_prefix: str) -> dict[str, float]:
    """Read Present_Current from a Feetech bus and return amperes by joint name."""
    names = list(joint_names)
    try:
        raw_currents = bus.sync_read("Present_Current", normalize=False)
    except Exception as exc:
        logger.warn(f"{warning_prefix}: {exc}", throttle_duration_sec=5.0)
        return {name: 0.0 for name in names}

    if not raw_currents:
        return {name: 0.0 for name in names}

    return {name: float(_decode_present_current(raw_currents.get(name, 0))) * CURRENT_RAW_TO_AMPERE for name in names}

"""Tests for so101_hardware calibration module."""

import pytest


def test_calibration_constants_agree_with_each_other():
    """The per-motor tables must stay the same length and agree on the full-turn motor.

    This replaces a set of asserts that restated each constant's own literal
    (MOTOR_COUNT == 6, DEFAULT_SERIAL_PORT == "/dev/ttyACM0", ...). Those could
    only fail when someone deliberately changed the value, which is exactly when
    the change was intended. The relations below are what nothing else guards:
    adding a motor to one table and not the others is a real, silent break.
    """
    from so101_hardware.calibration.constants import (
        DEFAULT_MOTOR_CONFIGS,
        FULL_TURN_MOTOR_ID,
        JOINT_NAMES,
        MOTOR_COUNT,
        MOTOR_IDS,
    )

    assert len(MOTOR_IDS) == MOTOR_COUNT
    assert len(JOINT_NAMES) == MOTOR_COUNT
    assert len(DEFAULT_MOTOR_CONFIGS) == MOTOR_COUNT
    assert {int(key) for key in DEFAULT_MOTOR_CONFIGS} == set(MOTOR_IDS)

    # The full-turn motor is the one configured for the 0..100 range.
    assert FULL_TURN_MOTOR_ID in MOTOR_IDS
    assert DEFAULT_MOTOR_CONFIGS[str(FULL_TURN_MOTOR_ID)]["mode"] == "RANGE_0_100"


def test_read_motor_currents_converts_raw_values():
    """Test Feetech Present_Current conversion helper."""
    from so101_hardware.motor_current import read_motor_currents

    class Bus:
        def sync_read(self, register, normalize=False):
            assert register == "Present_Current"
            assert normalize is False
            return {"1": 12, "2": 3}

    class Logger:
        def warn(self, msg, **kwargs):
            raise AssertionError(msg)

    currents = read_motor_currents(Bus(), ["1", "2", "3"], Logger(), "failed")

    assert currents["1"] == pytest.approx(0.078)
    assert currents["2"] == pytest.approx(0.0195)
    assert currents["3"] == 0.0


def test_read_motor_currents_returns_zero_on_read_failure():
    """Test current helper fallback when the bus read fails."""
    from so101_hardware.motor_current import read_motor_currents

    class Bus:
        def sync_read(self, register, normalize=False):
            raise RuntimeError("offline")

    class Logger:
        def __init__(self):
            self.messages = []

        def warn(self, msg, **kwargs):
            self.messages.append((msg, kwargs))

    logger = Logger()
    currents = read_motor_currents(Bus(), ["1", "2"], logger, "failed")

    assert currents == {"1": 0.0, "2": 0.0}
    assert logger.messages[0][0] == "failed: offline"
    assert logger.messages[0][1] == {"throttle_duration_sec": 5.0}

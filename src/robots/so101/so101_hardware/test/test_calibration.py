"""Tests for so101_hardware calibration module."""

import pytest


def test_calibration_constants():
    """Test that calibration constants are defined."""
    from so101_hardware.calibration.constants import (
        CURRENT_RAW_TO_AMPERE,
        DEFAULT_CONTROL_RATE,
        DEFAULT_LEADER_PUBLISH_RATE,
        DEFAULT_SERIAL_PORT,
        JOINT_NAMES,
        MOTOR_COUNT,
        MOTOR_IDS,
    )

    assert MOTOR_COUNT == 6
    assert len(MOTOR_IDS) == 6
    assert len(JOINT_NAMES) == 6
    assert DEFAULT_SERIAL_PORT == "/dev/ttyACM0"
    assert DEFAULT_LEADER_PUBLISH_RATE == 50.0
    assert DEFAULT_CONTROL_RATE == 100.0
    assert pytest.approx(0.0065) == CURRENT_RAW_TO_AMPERE


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

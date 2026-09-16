"""Persisted camera calibration compatibility with the dataset_tools writer."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robot_runtime import peripherals

INT_RANGES = {
    "brightness": (-255, 255),
    "contrast": (0, 255),
    "saturation": (0, 255),
    "sharpness": (0, 255),
    "gain": (0, 255),
    "white_balance": (2000, 10000),
    "exposure": (0, 20000),
    "focus": (0, 1023),
}
BOOL_KEYS = ("auto_white_balance", "autoexposure", "autofocus")


@pytest.fixture
def override_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_HOME", str(tmp_path))
    path = tmp_path / "ibrobot" / "camera_isp_overrides" / "top.json"
    path.parent.mkdir(parents=True)
    return path


@pytest.mark.parametrize("ros_home", [None, "", "custom"])
def test_calibrator_round_trip(tmp_path, monkeypatch, caplog, ros_home):
    from dataset_tools import camera_isp_calibrator as calibrator

    source_root = Path(__file__).resolve().parents[2]
    assert Path(calibrator.__file__).resolve() == source_root / "dataset_tools/dataset_tools/camera_isp_calibrator.py"
    assert Path(peripherals.__file__).resolve() == source_root / "robot_runtime/robot_runtime/peripherals.py"
    monkeypatch.setenv("HOME", str(tmp_path))
    if ros_home is None:
        monkeypatch.delenv("ROS_HOME", raising=False)
    else:
        monkeypatch.setenv("ROS_HOME", str(tmp_path / ros_home) if ros_home else "")
    base = tmp_path / ("custom" if ros_home else ".ros")
    path = base / "ibrobot" / "camera_isp_overrides" / "top.json"
    assert calibrator._override_path("top") == path
    assert set(peripherals._ISP_KEYS) == set(calibrator._ALL_KEYS) == set(INT_RANGES) | set(BOOL_KEYS)
    assert len(peripherals._ISP_KEYS) == len(calibrator._ALL_KEYS) == 11

    values = {key: lo for key, (lo, _) in INT_RANGES.items()}
    values.update(dict.fromkeys(BOOL_KEYS, False))
    window = object.__new__(calibrator.CalibratorWindow)
    window._bridge = SimpleNamespace(camera_name="top")
    window._applied = values.copy()
    window._device_caps = {}
    window._notify = Mock()
    window._save_override()
    assert not window._dirty_save
    saved = json.loads(path.read_text())
    assert saved["_camera"] == "top"
    assert "_saved_at" in saved
    with caplog.at_level(logging.INFO, logger=peripherals.logger.name):
        assert peripherals.load_isp_override("top") == values
    assert not caplog.records


def test_missing_override_logs_info(override_path, caplog):
    with caplog.at_level(logging.INFO, logger=peripherals.logger.name):
        assert peripherals.load_isp_override("top") == {}
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert "top" in caplog.text
    assert str(override_path) in caplog.text


@pytest.mark.parametrize("key,bounds", INT_RANGES.items())
def test_numeric_bounds_and_conversion(override_path, key, bounds):
    lo, hi = bounds
    for value, expected in [(lo, lo), (hi, hi), (lo + 0.9, int(lo + 0.9))]:
        override_path.write_text(json.dumps({key: value}))
        result = peripherals.load_isp_override("top")
        assert result == {key: expected}
        assert type(result[key]) is int


@pytest.mark.parametrize("key,bounds", INT_RANGES.items())
def test_invalid_numbers_drop_only_bad_key(override_path, caplog, key, bounds):
    lo, hi = bounds
    for value in (lo - 1, hi + 1, True, False, "auto", None, [], {}, float("nan"), float("inf")):
        caplog.clear()
        override_path.write_text(json.dumps({key: value, "autofocus": False}))
        assert peripherals.load_isp_override("top") == {"autofocus": False}
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert key in caplog.text
        assert "invalid" in caplog.text


@pytest.mark.parametrize("key", BOOL_KEYS)
def test_boolean_types(override_path, caplog, key):
    for value in (True, False):
        override_path.write_text(json.dumps({key: value}))
        assert peripherals.load_isp_override("top") == {key: value}
    for value in (0, 1, 1.0, "false", None, [], {}):
        caplog.clear()
        override_path.write_text(json.dumps({key: value, "brightness": -32}))
        assert peripherals.load_isp_override("top") == {"brightness": -32}
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert key in caplog.text


def test_metadata_ignored_and_unknown_keys_warn(override_path, caplog):
    override_path.write_text(json.dumps({"brightness": 32, "_camera": "top", "_saved_at": "today", "typo": 1}))
    assert peripherals.load_isp_override("top") == {"brightness": 32}
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "typo" in caplog.text
    assert "_camera" not in caplog.text
    assert "_saved_at" not in caplog.text


@pytest.mark.parametrize("content", ["{", "[]", "null"])
def test_corrupt_override_is_ignored(override_path, caplog, content):
    override_path.write_text(content)
    assert peripherals.load_isp_override("top") == {}
    assert caplog.records[0].levelno == logging.WARNING


def test_unreadable_override_is_ignored(override_path, monkeypatch, caplog):
    monkeypatch.setattr("builtins.open", Mock(side_effect=PermissionError("denied")))
    assert peripherals.load_isp_override("top") == {}
    assert caplog.records[0].levelno == logging.WARNING


def test_camera_applies_valid_override_and_preserves_yaml_fallback(override_path, monkeypatch):
    override_path.write_text(json.dumps({"brightness": -32, "exposure": "auto", "io_method": "invalid"}))
    node = Mock()
    monkeypatch.setattr(peripherals, "Node", node)
    peripherals.peripheral_nodes(
        [{"type": "camera", "name": "top", "driver": "opencv", "brightness": 0, "exposure": 312, "io_method": "mmap"}],
        use_sim=False,
    )
    params = node.call_args.kwargs["parameters"][0]
    assert params["brightness"] == -32
    assert params["exposure"] == 312
    assert params["io_method"] == "mmap"

import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_STUB_MODULE_NAMES = (
    "robot_teleop",
    "robot_teleop.base_teleop",
    "robot_teleop.devices",
    "robot_teleop.devices.leader_arm",
)
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _STUB_MODULE_NAMES}

base_module = types.ModuleType("robot_teleop")
base_teleop_module = types.ModuleType("robot_teleop.base_teleop")


class BaseTeleopDevice:
    def __init__(self, config, node=None):
        self._config = config
        self._node = node
        self._is_connected = False


base_teleop_module.BaseTeleopDevice = BaseTeleopDevice
sys.modules["robot_teleop"] = base_module
sys.modules["robot_teleop.base_teleop"] = base_teleop_module
sys.modules["robot_teleop.devices"] = types.ModuleType("robot_teleop.devices")

leader_arm_path = Path(__file__).resolve().parents[1] / "robot_teleop" / "devices" / "leader_arm.py"
spec = importlib.util.spec_from_file_location("robot_teleop.devices.leader_arm", leader_arm_path)
assert spec is not None
assert spec.loader is not None
leader_arm_module = importlib.util.module_from_spec(spec)
sys.modules["robot_teleop.devices.leader_arm"] = leader_arm_module
spec.loader.exec_module(leader_arm_module)
LeaderArmDevice = leader_arm_module.LeaderArmDevice

for module_name, original_module in _ORIGINAL_MODULES.items():
    if original_module is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = original_module


class FakeMotorsBus:
    def __init__(self, positions):
        self._positions = positions

    def sync_read(self, _register, normalize=False):
        assert normalize is False
        return self._positions


def _connected_device(config, positions, calibration=None):
    device = LeaderArmDevice(config)
    device._is_connected = True
    device.motors_bus = FakeMotorsBus(positions)
    device.calibration = calibration
    return device


def test_leader_arm_uses_direct_gripper_joint_names_only():
    device = LeaderArmDevice({"target": {"gripper_joint_names": ["joint6_left"]}})

    assert device.gripper_joints == {"6"}


def test_gripper_normalization_failure_skips_target_without_radians(caplog):
    device = _connected_device(
        {"joint_mapping": {"6": "joint6_left"}, "gripper_joint_names": ["joint6_left"]},
        {"6": 4095},
        calibration={},
    )

    targets = device.get_joint_targets()

    assert "joint6_left" not in targets
    assert "skipping publish" in caplog.text


def test_gripper_target_normalizes_with_drive_mode():
    device = _connected_device(
        {"joint_mapping": {"6": "joint6_left"}, "gripper_joint_names": ["joint6_left"]},
        {"6": 25},
        calibration={"6": SimpleNamespace(range_min=0, range_max=100, drive_mode=1)},
    )

    assert device.get_joint_targets()["joint6_left"] == 0.75


def test_arm_joint_still_uses_radians_path():
    device = _connected_device({}, {"1": 2049})

    assert device.get_joint_targets()["1"] == device.rad_per_step


# --- follower gripper stroke -----------------------------------------------
# The leader normalizes the gripper to 0~1 and the follower runs a radian
# position controller. The device always emits the RAW ratio; the single
# ratio -> radian conversion lives in TeleopNode, whose endpoints come from
# the runtime public description or (provider-less deployments) from
# get_gripper_stroke below — never a device-side second mapping.

# Representative follower calibration; the real values are read at runtime.
_FOLLOWER_RANGE_MIN = 1622
_FOLLOWER_RANGE_MAX = 3091
_TICKS_PER_RAD = 4096.0 / (2 * math.pi)
_EXPECTED_RAD_MIN = (_FOLLOWER_RANGE_MIN - 2048.0) / _TICKS_PER_RAD  # ~ -0.6535
_EXPECTED_RAD_MAX = (_FOLLOWER_RANGE_MAX - 2048.0) / _TICKS_PER_RAD  # ~ +1.5999


def _follower_calib(tmp_path, entry=None):
    payload = {
        "6": entry if entry is not None else {"range_min": _FOLLOWER_RANGE_MIN, "range_max": _FOLLOWER_RANGE_MAX}
    }
    path = tmp_path / "so101_follower_calibrate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _gripper_device(tmp_path, raw, *, follower_calib_file=None, follower_named=False):
    """Build a connected device. Defaults to the shipped config shape.

    ibrobot_runtime/leader_teleop_params.yaml uses gripper_joint_names: ["6"]
    with no joint_mapping, and the follower calibration is keyed by joint id.
    follower_named=True exercises the other shape _is_gripper_joint accepts.
    """
    if follower_named:
        config = {"joint_mapping": {"6": "joint6_left"}, "gripper_joint_names": ["joint6_left"]}
    else:
        config = {"gripper_joint_names": ["6"]}
    if follower_calib_file is not None:
        config["follower_calib_file"] = follower_calib_file
    device = _connected_device(
        config,
        {"6": raw},
        calibration={"6": SimpleNamespace(range_min=0, range_max=100, drive_mode=0)},
    )
    device._load_follower_gripper_stroke()
    return device


def test_gripper_target_stays_the_raw_ratio_with_follower_calibration(tmp_path):
    # raw 75 over a 0~100 leader range is 0.75 open. The device must NOT map
    # it to radians itself: TeleopNode owns the single ratio -> radian mapping
    # (a device-side second mapping double-converts and breaks the gripper).
    device = _gripper_device(tmp_path, 75, follower_calib_file=_follower_calib(tmp_path))

    assert device.get_joint_targets()["6"] == pytest.approx(0.75)


def test_fully_closed_leader_emits_ratio_zero_not_fake_radians(tmp_path):
    """Fully closing the leader emits 0.0 (the ratio), not a stale 0 rad command.

    The historical bug published the unmapped percentage as radians; the fix
    routes every ratio through the shared mapping, so the device output must
    stay in [0, 1] regardless of the follower calibration state.
    """
    device = _gripper_device(tmp_path, 0, follower_calib_file=_follower_calib(tmp_path))

    assert device.get_joint_targets()["6"] == pytest.approx(0.0)


def test_get_gripper_stroke_reports_the_follower_calibration(tmp_path):
    """Provider-less fallback: the stroke is the endpoint source without a runtime."""
    device = _gripper_device(tmp_path, 0, follower_calib_file=_follower_calib(tmp_path))

    stroke = device.get_gripper_stroke()

    assert stroke == (_EXPECTED_RAD_MIN, _EXPECTED_RAD_MAX)


def test_get_gripper_stroke_none_without_follower_calibration(tmp_path):
    device = _gripper_device(tmp_path, 0)

    assert device.get_gripper_stroke() is None


def test_gripper_limits_track_the_follower_calibration(tmp_path):
    device = _gripper_device(tmp_path, 0, follower_calib_file=_follower_calib(tmp_path))

    limits = device.get_gripper_limits()

    assert limits["6"]["min"] == pytest.approx(_EXPECTED_RAD_MIN)
    assert limits["6"]["max"] == pytest.approx(_EXPECTED_RAD_MAX)


def test_gripper_limits_empty_without_follower_calibration(tmp_path):
    device = _gripper_device(tmp_path, 0)

    assert device.get_gripper_limits() == {}


def test_missing_follower_calibration_keeps_the_legacy_percentage(tmp_path):
    """Backward compatibility: no follower calibration must not change behaviour."""
    device = _gripper_device(tmp_path, 75)

    assert device.get_joint_targets()["6"] == pytest.approx(0.75)


def test_follower_named_gripper_still_finds_the_calibration(tmp_path):
    """gripper_joint_names may hold the follower name; the calibration is keyed by id.

    Without the reverse lookup this configuration silently falls back to the
    0~1 command, i.e. straight back to the half-close bug.
    """
    device = _gripper_device(tmp_path, 0, follower_calib_file=_follower_calib(tmp_path), follower_named=True)

    assert device.get_joint_targets()["joint6_left"] == pytest.approx(0.0)
    assert device.get_gripper_limits()["joint6_left"]["min"] == pytest.approx(_EXPECTED_RAD_MIN)
    assert device.get_gripper_stroke() == (_EXPECTED_RAD_MIN, _EXPECTED_RAD_MAX)


def test_malformed_follower_calibration_keeps_the_legacy_percentage(tmp_path):
    device = _gripper_device(tmp_path, 75, follower_calib_file=_follower_calib(tmp_path, entry={"range_min": 1622}))

    assert device.get_gripper_limits() == {}
    assert device.get_joint_targets()["6"] == pytest.approx(0.75)

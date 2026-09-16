"""Tests for the static hand-eye precondition checker."""

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_handeye_preconditions.py"
_SPEC = importlib.util.spec_from_file_location("check_handeye_preconditions", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
precheck = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = precheck
_SPEC.loader.exec_module(precheck)


def _robot_config(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "robots" / name


def test_gripper_mounted_realsense_config_passes_eye_in_hand_precheck():
    result = precheck.run_precheck(_robot_config("lekiwi_handeye_realsense_grasp.yaml"), camera_name="wrist")

    assert result.passed
    assert [camera.name for camera in result.cameras] == ["wrist"]
    assert [camera.name for camera in result.mounted_cameras] == ["wrist"]


def test_base_mounted_camera_fails_eye_in_hand_precheck(tmp_path):
    config_path = tmp_path / "base_mounted.yaml"
    config_path.write_text(
        """
robot:
  name: base_mounted
  type: so101
  robot_type: so_101
  moveit:
    base_link: base
    ee_link: gripper
  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
  peripherals:
    - type: camera
      name: front
      driver: realsense
      frame_id: camera_front_link
      optical_frame_id: camera_front_optical_frame
      transform:
        parent_frame: base
""",
        encoding="utf-8",
    )
    result = precheck.run_precheck(config_path)

    assert not result.passed
    assert result.mounted_cameras == []
    assert any("no camera peripheral is mounted under 'gripper'" in item for item in result.errors)


def test_standard_so101_config_has_wrist_camera_for_eye_in_hand_precheck():
    result = precheck.run_precheck(_robot_config("so101_single_arm_legacy.yaml"), camera_name="wrist")

    assert result.passed
    assert [camera.name for camera in result.mounted_cameras] == ["wrist"]

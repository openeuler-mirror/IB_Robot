"""Gripper range authority: calibration on hardware, URDF convention in simulation."""

import json

import pytest

from robot_runtime.model_metadata import build_model_metadata

URDF = """<?xml version="1.0"?>
<robot name="t">
  <link name="base"/>
  <link name="shoulder"/>
  <link name="gripper"/>
  <joint name="1" type="revolute">
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-1.0" upper="1.0"/>
  </joint>
  <joint name="6" type="revolute">
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="0.0" upper="1.0"/>
  </joint>
</robot>
"""

# Joint 1 ticks sit inside the URDF window; joint 6 (gripper) exceeds the
# normalized 0..1 convention on both ends, as any real calibration does.
CALIBRATION = {
    "1": {"range_min": 1000, "range_max": 3000},
    "6": {"range_min": 1560, "range_max": 3080},
}


def _profile(calib_file=None):
    profile = {
        "joints": ["1", "6"],
        "arm_joints": ["1"],
        "gripper_joints": ["6"],
        "motion": {"base_link": "base", "ee_link": "gripper", "shoulder_link": "shoulder"},
        "home_positions": {},
        "hardware": {},
    }
    if calib_file is not None:
        profile["hardware"]["calib_file"] = str(calib_file)
    return profile


def _write_calibration(tmp_path):
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(CALIBRATION))
    return path


def test_physical_gripper_limits_come_from_calibration_not_urdf(tmp_path):
    model = build_model_metadata(
        _profile(_write_calibration(tmp_path)),
        simulated=False,
        robot_description=URDF,
    )
    # Arm joint: conversion range intersected with the (mechanical) URDF window.
    assert model["joint_limits"]["1"]["min"] == pytest.approx(-1.0)
    assert model["joint_limits"]["1"]["max"] == pytest.approx(1.0)
    # Gripper: the 0..1 URDF convention must not clamp the calibrated range.
    assert model["joint_limits"]["6"]["min"] == pytest.approx((1560 - 2048.0) / 651.8986469)
    assert model["joint_limits"]["6"]["max"] == pytest.approx((3080 - 2048.0) / 651.8986469)


def test_simulated_gripper_keeps_the_urdf_convention():
    model = build_model_metadata(_profile(), simulated=True, robot_description=URDF)
    assert model["authority"] == "urdf"
    assert model["joint_limits"]["6"]["min"] == pytest.approx(0.0)
    assert model["joint_limits"]["6"]["max"] == pytest.approx(1.0)
    assert model["joint_limits"]["1"]["min"] == pytest.approx(-1.0)
    assert model["joint_limits"]["1"]["max"] == pytest.approx(1.0)


def test_physical_gripper_missing_from_urdf_is_rejected(tmp_path):
    # Existence stays validated even though the calibration is the range
    # authority: a profile gripper absent from the rendered URDF must fail.
    urdf_without_6 = URDF.replace('<joint name="6" type="revolute">', '<joint name="7" type="revolute">')
    with pytest.raises(KeyError, match="missing lower/upper limits"):
        build_model_metadata(
            _profile(_write_calibration(tmp_path)),
            simulated=False,
            robot_description=urdf_without_6,
        )


def test_conversion_lookup_error_hints_at_field_selectors():
    # A caller passing message field selectors ("position.<joint>") instead of
    # joint names is the recurring confusion; the error must name it.
    from robot_runtime.model_metadata import build_joint_conversion_table_from_model

    model = build_model_metadata(_profile(), simulated=True, robot_description=URDF)
    with pytest.raises(ValueError, match="field selectors"):
        build_joint_conversion_table_from_model(model, ["position.1", "position.6"], "degrees")

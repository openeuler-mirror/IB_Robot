import subprocess
from pathlib import Path

import pytest
import yaml

from robot_calibration.viewer import (
    display_environment,
    overlay_decoder_command,
    preview_decoder_command,
    rviz_command,
    start_viewer,
)


def test_rviz_command_selects_capture_and_validate_configs():
    share = Path("/opt/share/robot_calibration")

    assert rviz_command("capture", share) == ["rviz2", "-d", str(share / "rviz/calib_capture.rviz")]
    assert rviz_command("validate", share) == ["rviz2", "-d", str(share / "rviz/calib_validate.rviz")]


def test_capture_preview_decoder_uses_compressed_transport():
    assert preview_decoder_command() == [
        "ros2",
        "run",
        "robot_calibration",
        "calib_preview_decode",
    ]


def test_overlay_decoder_uses_compressed_transport():
    assert overlay_decoder_command() == [
        "ros2",
        "run",
        "robot_calibration",
        "calib_preview_decode",
        "--input-topic",
        "/calib/overlay/compressed",
        "--output-topic",
        "/calib/overlay",
        "--node-name",
        "robot_calibration_overlay_decoder",
    ]


def test_validate_viewer_starts_preview_decoder(monkeypatch, tmp_path):
    calls = []

    class FakeProcess:
        def __init__(self, command):
            self.command = command

        def poll(self):
            return None

    monkeypatch.setattr("robot_calibration.viewer.display_environment", lambda: {"DISPLAY": ":0"})
    monkeypatch.setattr("robot_calibration.viewer.package_share", lambda: tmp_path)
    monkeypatch.setattr(
        "robot_calibration.viewer.subprocess.Popen",
        lambda command, **kwargs: calls.append((command, kwargs)) or FakeProcess(command),
    )

    session = start_viewer("validate")

    assert session is not None
    assert [call[0] for call in calls] == [
        preview_decoder_command(),
        overlay_decoder_command(),
        rviz_command("validate", tmp_path),
    ]


def test_viewer_can_redirect_child_output_to_capture_log(monkeypatch, tmp_path):
    """Every child gets its stdout and stderr sent to the capture log.

    Previously asserted by grepping viewer.py for the literal signature and the
    string "stdout=log", which passed whether or not the redirect worked and
    failed on any rename. Assert the Popen kwargs instead.
    """
    calls = []

    class FakeProcess:
        def __init__(self, command):
            self.command = command

        def poll(self):
            return None

    monkeypatch.setattr("robot_calibration.viewer.display_environment", lambda: {"DISPLAY": ":0"})
    monkeypatch.setattr("robot_calibration.viewer.package_share", lambda: tmp_path)
    monkeypatch.setattr(
        "robot_calibration.viewer.subprocess.Popen",
        lambda command, **kwargs: calls.append((command, kwargs)) or FakeProcess(command),
    )

    log_path = tmp_path / "viewer.log"
    assert start_viewer("validate", log_path=log_path) is not None

    assert calls, "no child process was started"
    for _command, kwargs in calls:
        assert kwargs["stdout"] is not None
        assert kwargs["stdout"].name == str(log_path)
        assert kwargs["stderr"] == subprocess.STDOUT
    assert log_path.exists()


def test_viewer_does_not_override_ros_transport_configuration():
    """The viewer inherits the caller's ROS transport rather than picking its own.

    Previously asserted by grepping viewer.py for these names, which silently
    stopped checking anything the moment the assignment moved into a helper.
    Assert the environment actually handed to the child instead.
    """
    caller_environment = {
        "DISPLAY": ":0",
        "ROS_DOMAIN_ID": "42",
        "RMW_IMPLEMENTATION": "rmw_cyclonedds_cpp",
        "CYCLONEDDS_URI": "file:///etc/cyclonedds.xml",
    }

    child_environment = display_environment(caller_environment)

    for name, value in caller_environment.items():
        assert child_environment[name] == value, f"viewer changed {name}"
    assert "domain_bridge" not in " ".join(rviz_command("validate", Path("/tmp")))


def test_display_environment_preserves_existing_display(tmp_path):
    assert display_environment({"DISPLAY": ":7"}, tmp_path) == {"DISPLAY": ":7"}


def test_display_environment_discovers_local_x11_desktop(tmp_path):
    (tmp_path / "X1").touch()

    assert display_environment({}, tmp_path)["DISPLAY"] == ":1"


def test_display_environment_rejects_host_without_desktop(tmp_path):
    with pytest.raises(RuntimeError, match="图形桌面"):
        display_environment({}, tmp_path)


def test_calibration_image_displays_use_sensor_data_qos():
    rviz_root = Path(__file__).parents[1] / "rviz"

    for name in ("calib_capture.rviz", "calib_validate.rviz"):
        document = yaml.safe_load((rviz_root / name).read_text(encoding="utf-8"))
        image_displays = [
            display
            for display in document["Visualization Manager"]["Displays"]
            if display["Class"] == "rviz_default_plugins/Image"
        ]
        assert image_displays
        assert all(display["Topic"]["Reliability Policy"] == "Best Effort" for display in image_displays)
        assert all(display["Value"] is True for display in image_displays)


def test_validation_rviz_uses_only_low_bandwidth_preview_topics():
    rviz_path = Path(__file__).parents[1] / "rviz/calib_validate.rviz"
    text = rviz_path.read_text(encoding="utf-8")

    assert "/camera/front/image_raw" not in text
    assert "/cloud_registered_body" not in text


def test_capture_rviz_uses_only_low_bandwidth_preview_topics():
    rviz_path = Path(__file__).parents[1] / "rviz/calib_capture.rviz"
    document = yaml.safe_load(rviz_path.read_text(encoding="utf-8"))
    displays = document["Visualization Manager"]["Displays"]

    topics = {display["Topic"]["Value"] for display in displays}
    assert topics == {"/calib/preview/cloud", "/calib/preview/image"}
    assert "/camera/front/image_raw" not in rviz_path.read_text(encoding="utf-8")
    assert "/cloud_registered_body" not in rviz_path.read_text(encoding="utf-8")


def test_calibration_pointcloud_displays_use_sensor_data_qos():
    rviz_root = Path(__file__).parents[1] / "rviz"

    for name in ("calib_capture.rviz", "calib_validate.rviz"):
        document = yaml.safe_load((rviz_root / name).read_text(encoding="utf-8"))
        cloud_displays = [
            display
            for display in document["Visualization Manager"]["Displays"]
            if display["Class"] == "rviz_default_plugins/PointCloud2"
        ]
        assert cloud_displays
        assert all(display["Topic"]["Reliability Policy"] == "Best Effort" for display in cloud_displays)
        assert all(display["Color Transformer"] == "Intensity" for display in cloud_displays)
        assert all(display["Use rainbow"] is True for display in cloud_displays)

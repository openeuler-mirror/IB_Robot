import json
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from robot_calibration import workflow
from robot_calibration.workflow import capture_initialization_message, logical_sensor_name, resolve_capture_input


def test_user_workflow_uses_logical_sensor_names_without_serial_arguments():
    assert logical_sensor_name("camera_front") == "front"
    assert logical_sensor_name("front") == "front"
    assert logical_sensor_name("wrist_camera") == "wrist"


def test_resolve_capture_input_accepts_sealed_directory_and_archive(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "manifest.json").write_text(json.dumps({"capture_id": "calib-1", "sealed": True}), encoding="utf-8")
    archive = tmp_path / "calib-1.raw.tar"
    archive.write_bytes(b"tar")

    assert resolve_capture_input(capture) == ("directory", capture)
    assert resolve_capture_input(archive) == ("archive", archive)


def test_solve_cli_help_exposes_one_input_workflow():
    completed = subprocess.run(
        ["python3", "-c", "from robot_calibration.workflow import solve_main; solve_main(['--help'])"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--input" in completed.stdout
    assert "--scene-01" not in completed.stdout


def test_capture_cli_help_does_not_require_capture_identifier():
    completed = subprocess.run(
        ["python3", "-c", "from robot_calibration.workflow import capture_main; capture_main(['--help'])"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--duration" in completed.stdout
    assert "--capture-id" not in completed.stdout


def test_capture_starts_sensors_then_base_control():
    assert workflow.sensor_calibration_launch_command() == [
        "ros2",
        "launch",
        "robot_config",
        "sensor_calibration.launch.py",
        "robot_config:=lekiwi_sensor_calib",
    ]


def test_resolve_capture_input_rejects_unknown_path(tmp_path):
    with pytest.raises(ValueError, match="capture input"):
        resolve_capture_input(tmp_path / "missing")


def test_logged_process_redirects_stdout_and_stderr(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(workflow.subprocess, "Popen", fake_popen)
    log_path = tmp_path / "capture.log"

    workflow._start_logged_process(["sensor-command"], log_path, start_new_session=True)

    assert captured["command"] == ["sensor-command"]
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdout"].name == str(log_path)


def test_stop_owned_process_cleans_process_group_after_leader_exits(monkeypatch):
    signals = []
    group_alive = True

    class FinishedProcess:
        pid = 1234

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    def fake_killpg(pgid, sig):
        nonlocal group_alive
        signals.append((pgid, sig))
        if sig == workflow.signal.SIGINT:
            group_alive = False
        if sig == 0 and not group_alive:
            raise ProcessLookupError

    monkeypatch.setattr(workflow.os, "killpg", fake_killpg)

    workflow._stop_owned_process(FinishedProcess())

    assert signals == [
        (1234, workflow.signal.SIGINT),
        (1234, 0),
    ]


def test_retain_failed_recording_renames_staging_directory(tmp_path):
    recording = tmp_path / ".calib-1.recording"
    recording.mkdir()
    (recording / "scene-01").mkdir()

    failed = workflow._retain_failed_recording(recording)

    assert failed == tmp_path / ".calib-1.failed"
    assert not recording.exists()
    assert (failed / "scene-01").is_dir()


def test_realsense_serial_falls_back_to_unavailable(monkeypatch):
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, stdout="", stderr="missing"),
    )

    assert workflow._discover_realsense_serial() == "unavailable"


def test_realsense_serial_reads_installed_utility(monkeypatch):
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="Device info:\n  Serial Number: 123456789\n"
        ),
    )

    assert workflow._discover_realsense_serial() == "123456789"


def test_default_paths_do_not_depend_on_current_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    assert workflow.default_calib_root() == tmp_path / ".ros/ibrobot/calib"
    assert workflow.default_raw_dir() == tmp_path / ".ros/ibrobot/calib/raw"
    assert workflow.default_process_dir("calib-1") == tmp_path / ".ros/ibrobot/calib/process/calib-1"
    assert workflow.default_candidate_dir() == tmp_path / ".ros/ibrobot/calib/candidates"


def test_default_fast_calib_workspace_is_repository_relative(monkeypatch):
    monkeypatch.delenv("FAST_CALIB_WORKSPACE", raising=False)

    assert workflow._default_workspace() == workflow._repo_root()


def test_default_mount_is_the_active_robot_profile_source():
    assert workflow._default_mount() == (
        Path(__file__).parents[2] / "robot_config/config/hardware/lekiwi_mid360_mount.yaml"
    )


def test_solve_archive_contains_three_candidate_artifacts_from_mount_and_camera_info(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "manifest.json").write_text(
        json.dumps({"capture_id": "calib-1", "devices": {"camera": "C1", "lidar": "L1"}}),
        encoding="utf-8",
    )
    mount = tmp_path / "lekiwi_mid360_mount.yaml"
    mount.write_text(
        "schema_version: '1.0'\nstatus: provisional\nparent_frame: base_link\n"
        "lidar_frame: livox_frame\nbody_frame: body\n"
        "translation_m: [0.11, -0.22, 0.33]\nrpy_deg: [0.0, 0.0, 90.0]\n",
        encoding="utf-8",
    )
    camera_info = {
        "schema_version": "1.0",
        "frame_id": "camera_front_optical_frame",
        "width": 848,
        "height": 480,
        "distortion_model": "plumb_bob",
        "D": [0.1, -0.2, 0.003, 0.004, 0.0],
        "K": [606.0, 0.0, 321.0, 0.0, 605.0, 254.0, 0.0, 0.0, 1.0],
        "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "P": [606.0, 0.0, 321.0, 0.0, 0.0, 605.0, 254.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    }

    def fake_export(_capture, exported):
        for index, scene in enumerate(workflow.REQUIRED_SCENES):
            scene_path = exported / scene
            scene_path.mkdir(parents=True)
            value = camera_info if scene == "scene-04-test" else {**camera_info, "width": 640 + index}
            (scene_path / "camera_info.yaml").write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    def fake_detector(_workspace, _templates, _exported, observations):
        for scene in workflow.REQUIRED_SCENES:
            (observations / scene).mkdir(parents=True)
            (observations / scene / "observation.yaml").write_text("observation\n", encoding="utf-8")
            (observations / f"{scene}.yaml").write_text("parameters\n", encoding="utf-8")

    def fake_solve(**kwargs):
        kwargs["output"].write_text("extrinsic\n", encoding="utf-8")
        kwargs["report"].write_text("{}\n", encoding="utf-8")
        return {}, {
            "training_joint_rmse_m": 0.01,
            "test_rmse_m": 0.02,
            "correspondence_margin_m": 0.1,
        }

    used_mounts = []

    def fake_candidate(**kwargs):
        used_mounts.append(kwargs["mount"])
        kwargs["output"].write_text(
            yaml.safe_dump(
                {
                    "schema_version": "1.0",
                    "calibration_version": "front-1",
                    "status": "candidate",
                    "device": {"name": "front_camera", "serial": "C1"},
                    "transform": {
                        "parent_frame": "base_link",
                        "child_frame": "camera_front_optical_frame",
                        "translation": [0.0, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(workflow, "export_capture", fake_export)
    monkeypatch.setattr(workflow, "run_detector", fake_detector)
    monkeypatch.setattr(workflow, "solve_joint_calibration", fake_solve)
    monkeypatch.setattr(workflow, "create_candidate_artifact", fake_candidate)
    monkeypatch.setattr(workflow, "_default_mount", lambda: mount)

    def fake_overlay(_result, _exported, output):
        output.write_bytes(b"png")
        return 7

    monkeypatch.setattr("robot_calibration.overlay.render_test_overlay", fake_overlay)
    (tmp_path / "process").mkdir()

    archive = workflow.solve_user_workflow(
        capture,
        tmp_path / "process/output",
        workspace=tmp_path / "workspace",
        candidate_dir=tmp_path / "candidates",
    )

    with tarfile.open(archive) as source:
        names = set(source.getnames())
        mount_value = yaml.safe_load(source.extractfile("base_to_mid360.yaml"))
        intrinsics_value = yaml.safe_load(source.extractfile("front_camera_intrinsics.yaml"))
    expected_artifacts = {
        "base_to_front_camera.candidate.yaml",
        "base_to_mid360.yaml",
        "front_camera_intrinsics.yaml",
    }
    calibration_artifacts = {name for name in names if name.startswith("base_to_") or name.startswith("front_camera_")}
    assert calibration_artifacts == expected_artifacts
    assert mount_value["transform"]["translation"] == [0.11, -0.22, 0.33]
    assert intrinsics_value["camera_info"]["k"] == camera_info["K"]
    assert used_mounts == [tmp_path / "process/output/base_to_mid360.yaml"]


def test_combined_workflow_parser_uses_calib_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    capture = workflow._parser().parse_args(["capture"])
    solve = workflow._parser().parse_args(["solve", "--input", "calib-1.raw.tar"])

    assert capture.output == tmp_path / ".ros/ibrobot/calib/raw"
    assert solve.output is None


def test_combined_solve_defaults_to_capture_specific_process_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    captured = {}

    def fake_solve(input_path, output, *, workspace):
        captured["input"] = input_path
        captured["output"] = output
        return tmp_path / "candidate.tar"

    monkeypatch.setattr(workflow, "solve_user_workflow", fake_solve)

    result = workflow.main(["solve", "--input", "calib-1.raw.tar"])

    assert result == 0
    assert captured["output"] == tmp_path / ".ros/ibrobot/calib/process/calib-1"


def test_capture_initialization_message_points_to_detailed_log(tmp_path):
    log_path = tmp_path / "capture.log"

    assert capture_initialization_message(log_path) == f"初始化中... 详细日志请见 {log_path.absolute()}"


def test_default_record_command_records_fixed_required_topics(tmp_path):
    command = workflow._record_command(tmp_path / "scene-01")

    assert "--no-discovery" not in command
    assert all(topic in command for topic in workflow.REQUIRED_TOPICS)


def test_capture_preview_is_not_part_of_raw_bag_contract(tmp_path):
    command = workflow._record_command(tmp_path / "scene-01")

    assert not any(topic.startswith("/calib/preview/") for topic in command)
    assert set(command[-len(workflow.REQUIRED_TOPICS) :]) == set(workflow.REQUIRED_TOPICS)


def test_capture_record_command_uses_fixed_required_topics(tmp_path):
    command = workflow._record_command(tmp_path / "scene-01")

    assert "--no-discovery" not in command
    assert command[-len(workflow.REQUIRED_TOPICS) :] == list(workflow.REQUIRED_TOPICS)


def test_recording_window_allows_rosbag_to_start_before_duration(monkeypatch):
    sleeps = []

    workflow._wait_for_recording_window(10.0, sleep_fn=sleeps.append)

    assert sleeps == [2.0, 10.0]


def test_sensor_readiness_reports_missing_topics_if_sensor_launch_exits(tmp_path, monkeypatch):
    class SensorProcess:
        def poll(self):
            return 1

    class Node:
        def create_subscription(self, *_args):
            return object()

        def destroy_node(self):
            return None

    class Rclpy:
        def ok(self):
            return False

        def init(self):
            return None

        def create_node(self, _name):
            return Node()

        def shutdown(self):
            return None

    monkeypatch.setattr(workflow, "rclpy", Rclpy())
    monkeypatch.setattr(workflow, "get_message", lambda _message_type: object)

    with pytest.raises(RuntimeError) as exc_info:
        workflow._wait_for_required_topics(tmp_path / "capture.log", sensor_process=SensorProcess())

    message = str(exc_info.value)
    assert "sensor calibration launch exited before readiness" in message
    assert all(topic in message for topic in workflow.REQUIRED_TOPICS)


def test_sensor_readiness_uses_one_rclpy_node_for_all_topics(tmp_path, monkeypatch):
    callbacks = {}
    events = []

    class Node:
        def create_subscription(self, _message_class, topic, callback, _qos):
            callbacks[topic] = callback
            return object()

        def destroy_node(self):
            events.append("destroy")

    class Rclpy:
        initialized = False

        def ok(self):
            return self.initialized

        def init(self):
            self.initialized = True
            events.append("init")

        def create_node(self, name):
            events.append(("node", name))
            return Node()

        def spin_once(self, _node, timeout_sec):
            assert timeout_sec == 1.0
            for callback in tuple(callbacks.values()):
                callback(object())

        def shutdown(self):
            self.initialized = False
            events.append("shutdown")

    monkeypatch.setattr(workflow, "rclpy", Rclpy())
    monkeypatch.setattr(workflow, "get_message", lambda _message_type: object)

    workflow._wait_for_required_topics(tmp_path / "capture.log")

    assert set(callbacks) == set(workflow.REQUIRED_TOPICS)
    assert events == ["init", ("node", "calib_capture_readiness"), "destroy", "shutdown"]


def test_capture_waits_for_sensor_messages_before_first_prompt(tmp_path, monkeypatch):
    events = []

    class Recorder:
        def poll(self):
            return None

        def send_signal(self, _signal):
            return None

        def wait(self):
            return 0

    def start_recorder(command, _log_path, **_kwargs):
        scene_dir = Path(command[command.index("--output") + 1])
        scene_dir.mkdir()
        (scene_dir / "metadata.yaml").write_text("metadata", encoding="utf-8")
        return Recorder()

    def wait_for_topics(_log_path, **_kwargs):
        events.append("ready")

    def prompt(_message):
        events.append("prompt")

    monkeypatch.setattr(workflow, "_start_sensor_calibration", lambda _log_path: None)
    monkeypatch.setattr(workflow, "_start_capture_preview", lambda _log_path: None)
    monkeypatch.setattr(workflow, "start_viewer", lambda *_args: None)
    monkeypatch.setattr(workflow, "_wait_for_required_topics", wait_for_topics)
    monkeypatch.setattr(workflow, "_start_logged_process", start_recorder)
    monkeypatch.setattr(workflow, "validate_fast_calib_bag", lambda _scene_dir: None)
    monkeypatch.setattr(workflow, "_capture_summary", lambda *_args: [])
    monkeypatch.setattr(workflow, "_discover_realsense_serial", lambda: "camera")
    monkeypatch.setattr(workflow, "finalize_capture", lambda *_args, **_kwargs: tmp_path / "capture")
    monkeypatch.setattr(workflow.shutil, "rmtree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow.ArtifactStore, "export_capture", lambda *_args: "digest")

    workflow.capture_user_workflow(tmp_path, duration_s=1, input_fn=prompt, sleep_fn=lambda _value: None)

    assert events == ["ready", "prompt", "prompt", "prompt", "prompt"]


def test_capture_prompts_once_per_scene_and_reports_saving(tmp_path, monkeypatch, capsys):
    prompts = []

    class Recorder:
        def poll(self):
            return None

        def send_signal(self, _signal):
            return None

        def wait(self):
            return 0

    def start_recorder(command, _log_path, **_kwargs):
        scene_dir = Path(command[command.index("--output") + 1])
        scene_dir.mkdir()
        (scene_dir / "metadata.yaml").write_text("metadata", encoding="utf-8")
        return Recorder()

    monkeypatch.setattr(workflow, "_start_sensor_calibration", lambda _log_path: None)
    monkeypatch.setattr(workflow, "_start_capture_preview", lambda _log_path: None)
    monkeypatch.setattr(workflow, "start_viewer", lambda *_args: None)
    monkeypatch.setattr(workflow, "_wait_for_required_topics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow, "_start_logged_process", start_recorder)
    monkeypatch.setattr(workflow, "validate_fast_calib_bag", lambda _scene_dir: None)
    monkeypatch.setattr(workflow, "_capture_summary", lambda *_args: [])
    monkeypatch.setattr(workflow, "_discover_realsense_serial", lambda: "camera")
    monkeypatch.setattr(workflow, "finalize_capture", lambda *_args, **_kwargs: tmp_path / "capture")
    monkeypatch.setattr(workflow.shutil, "rmtree", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(workflow.ArtifactStore, "export_capture", lambda *_args: "digest")

    workflow.capture_user_workflow(tmp_path, duration_s=1, input_fn=prompts.append, sleep_fn=lambda _value: None)

    assert len(prompts) == 4
    assert all("按 Enter 开始录制" in prompt for prompt in prompts)
    output = capsys.readouterr().out
    assert "录制完成，请移动到另一个位置，准备开始 scene-02" in output
    assert "录制完成，请移动到另一个位置，准备开始 scene-04-test" in output
    assert "录制完成" in output
    assert output.count("录制完成，请移动到另一个位置") == 3
    assert "正在保存并打包数据，请耐心等待" in output

from pathlib import Path

from robot_config.audio_contract import find_microphone_params, is_audio_io_enabled
from robot_config.config import PeripheralConfig, RobotConfig, Ros2ControlConfig
from robot_config.launch_builders import audio_io, speech_direction, voice_asr
from robot_config.loader import load_audio_io_config, validate_config


def _audio_config(enabled=True):
    return {
        "audio_io": {
            "enabled": enabled,
            "microphone": "respeaker",
            "capture_topic": "/shared/capture",
            "capture_stamped_topic": "/shared/capture_stamped",
            "audio_info_topic": "/shared/info",
            "playback_topic": "/shared/play",
            "playback_device": "plughw:1,0",
            "playback_sample_rate": 24000,
            "playback_channels": 1,
            "playback_sample_format": "S16LE",
        },
        "voice_asr": {"enabled": True},
        "voice_tts": {"enabled": True},
        "peripherals": [
            {
                "name": "respeaker",
                "type": "microphone",
                "params": {
                    "device": "hw:2,0",
                    "channels": 6,
                    "sample_rate": 16000,
                    "sample_format": "S16LE",
                },
            }
        ],
    }


def test_audio_contract_helpers_support_yaml_mappings_and_dataclasses():
    config = _audio_config()
    assert is_audio_io_enabled(config["audio_io"])
    assert find_microphone_params(config["peripherals"], "respeaker")["channels"] == 6

    loaded = load_audio_io_config(config["audio_io"])
    peripheral = PeripheralConfig(type="microphone", name="respeaker", driver="alsa", params={"channels": 4})
    assert is_audio_io_enabled(loaded)
    assert find_microphone_params([peripheral], "respeaker")["channels"] == 4


def test_audio_contract_helper_requires_explicit_enablement():
    assert not is_audio_io_enabled({"enabled": False})
    assert is_audio_io_enabled({"enabled": True})


def test_audio_io_builder_skips_disabled_configuration():
    assert audio_io.generate_audio_io_actions(_audio_config(enabled=False)) == []


def test_audio_io_loader_keeps_capture_and_playback_contracts_separate():
    config = load_audio_io_config(
        {"enabled": True, "microphone": "respeaker", "playback_sample_rate": 22050, "playback_channels": 2}
    )

    assert config.enabled is True
    assert config.microphone == "respeaker"
    assert config.playback_sample_rate == 22050
    assert config.playback_channels == 2


def test_audio_io_validation_requires_a_real_microphone_peripheral():
    config = RobotConfig(
        name="test_robot",
        type="so101",
        robot_type="so_101",
        ros2_control=Ros2ControlConfig(hardware_plugin="so101_hardware/SO101SystemHardware", params={}),
    )
    config.audio_io.enabled = True
    config.audio_io.microphone = "missing"
    config.voice_asr.enabled = True

    errors = validate_config(config)

    assert "audio_io.microphone must reference exactly one peripheral with type=microphone: 'missing'" in errors


def test_audio_io_validation_accepts_a_real_microphone_peripheral():
    config = RobotConfig(
        name="test_robot",
        type="lekiwi",
        robot_type="lekiwi",
        ros2_control=Ros2ControlConfig(hardware_plugin="sts_hardware_interface/STSHardwareInterface", params={}),
        peripherals=[
            PeripheralConfig(
                type="microphone",
                name="respeaker",
                driver="alsa",
                params={"device": "hw:0,0", "channels": 2, "sample_rate": 16000, "sample_format": "S16LE"},
            )
        ],
    )
    config.audio_io.enabled = True
    config.audio_io.microphone = "respeaker"
    config.voice_asr.enabled = True

    errors = validate_config(config)

    assert not any(error.startswith("audio_io.") for error in errors)


def test_audio_io_builder_routes_independent_capture_and_playback_formats():
    actions = audio_io.generate_audio_io_actions(_audio_config())

    assert len(actions) == 2
    assert [action.node_package for action in actions] == ["audio_capture", "audio_play"]
    capture = actions[0]
    playback = actions[1]
    assert capture.node_executable == "audio_capture_node"
    assert playback.node_executable == "audio_play_node"

    def parameters(action):
        return {key[0].text: value for key, value in action._Node__parameters[0].items()}

    assert parameters(capture)["channels"] == 6
    assert parameters(capture)["sample_rate"] == 16000
    assert parameters(playback)["channels"] == 1
    assert parameters(playback)["sample_rate"] == 24000

    def remappings(action):
        def text(value):
            if isinstance(value, list | tuple):
                value = value[0]
            return getattr(value, "text", str(value))

        return {(text(source), text(target)) for source, target in action._Node__remappings}

    assert ("audio", "/shared/capture") in remappings(capture)
    assert ("audio_stamped", "/shared/capture_stamped") in remappings(capture)
    assert ("audio", "/shared/play") in remappings(playback)


def test_speech_direction_uses_shared_stamped_topic(monkeypatch, tmp_path):
    launch_file = tmp_path / "launch" / "speech_direction.launch.py"
    launch_file.parent.mkdir()
    launch_file.touch()
    monkeypatch.setattr(speech_direction, "_voice_asr_service_share", lambda: Path(tmp_path))
    config = _audio_config()
    config["speech_direction"] = {
        "enabled": True,
        "profile": "ubuntu",
        "microphone": "respeaker",
        "parameters": {},
    }

    actions = speech_direction.generate_speech_direction_actions(config)

    arguments = dict(actions[0]._IncludeLaunchDescription__launch_arguments)
    assert arguments["speech_direction_audio_topic"] == "/shared/capture_stamped"
    assert arguments["microphone_channels"] == "6"


def test_voice_asr_uses_shared_stamped_topic_and_microphone_channels(monkeypatch, tmp_path):
    monkeypatch.setattr(
        voice_asr,
        "_load_voice_asr_service",
        lambda: {
            "bundle_path": "models/voice_asr/demo",
            "deployment": "torch_cpu",
            "language": "zh",
            "max_recording_duration": 10.0,
            "vad_sensitivity": 0.6,
            "vad_bundle_path": "models/silero-vad",
            "vad_deployment": "torch_cpu",
            "realtime_pre_roll_seconds": 0.5,
            "publish_partial": True,
            "output_topic": "/voice_command",
            "sample_rate": 16000,
            "chunk_size": 512,
            "buffer_seconds": 5.0,
            "exit_on_init_failure": False,
        },
    )
    monkeypatch.setattr(voice_asr, "validate_voice_asr_model_config", lambda **_kwargs: [])
    config = _audio_config()
    config["voice_asr"] = {
        "enabled": True,
        "bundle_path": str(tmp_path),
        "deployment": "torch_cpu",
        "active_mode": "manual",
    }

    node = voice_asr.generate_voice_asr_nodes(config)[0]
    parameters = {}
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            raw_value = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
            if raw_value.endswith("\n..."):
                raw_value = raw_value[:-4].rstrip()
        parameters[key] = raw_value

    assert parameters["audio_topic"] == "/shared/capture_stamped"
    assert parameters["audio_channels"] == 6

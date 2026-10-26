"""Audio interface bindings for the voice consumers (runtime-owned-audio-io).

Covers the three bindable fields (voice_asr.audio_input streaming,
speech_direction.audio_input, voice_tts.audio_output): declaration surface,
type/direction checks, the topic + format metadata written into the binding
result, absent-metadata fields staying absent, and a lekiwi-style ALSA
description without the vendor-only interfaces (audio.speech /
audio.enhanced_capture) binding cleanly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from robot_config.interface_binding import (  # noqa: E402
    InterfaceBindingError,
    bind_robot_interfaces,
    required_interface_ids,
)

_AUDIOStamped = "audio_common_msgs/msg/AudioDataStamped"


@pytest.fixture
def description_api():
    return importlib.import_module("robot_runtime.interface_description")


def _audio_description(description_api, *, with_format=True, vendor_speech=False):
    """A minimal ALSA-runtime description: audio.capture / audio.play only."""
    capture_metadata = (
        {"sample_rate_hz": 16000, "channels": 6, "sample_format": "S16LE", "channel_indices": [1, 2, 3, 4]}
        if with_format
        else {}
    )
    play_metadata = {"sample_rate_hz": 24000, "channels": 1, "sample_format": "S16LE"} if with_format else {}
    qos = {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10}
    doc = {
        "schema_version": 1,
        "robot": {"id": "unit-1", "type": "lekiwi", "runtime_name": "lekiwi_robot", "runtime_version": "0.1.0"},
        "execution": "simulated",
        "interfaces": {
            "audio.capture": {
                "capability": "interaction.audio_capture",
                "kind": "topic",
                "direction": "publish",
                "endpoint": "/audio/capture_stamped",
                "message_type": _AUDIOStamped,
                "qos": qos,
                **capture_metadata,
            },
            "audio.play": {
                "capability": "interaction.audio_playback",
                "kind": "topic",
                "direction": "subscribe",
                "endpoint": "/audio/play",
                "message_type": _AUDIOStamped,
                "qos": qos,
                **play_metadata,
            },
        },
        "states": {},
    }
    if vendor_speech:
        doc["interfaces"]["audio.speech"] = {
            "capability": "interaction.audio_capture",
            "kind": "topic",
            "direction": "publish",
            "endpoint": "/audio/speech_chunk",
            "message_type": "ibrobot_msgs/msg/SpeechAudioChunk",
            "qos": qos,
        }
    doc["digest"] = description_api.description_digest(doc)
    description_api.validate_description(doc)
    return doc


def _config(**sections):
    config = {
        "name": "consumer",
        "type": "lekiwi",
        "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp", "target": "simulation"},
    }
    config.update(sections)
    return config


def test_all_three_audio_bindings_resolve_topic_and_format(description_api):
    config = _config(
        voice_asr={
            "enabled": True,
            "input_mode": "shared_audio",
            "audio_input": {"interface": "audio.capture", "type": _AUDIOStamped},
        },
        speech_direction={"enabled": True, "audio_input": {"interface": "audio.capture", "type": _AUDIOStamped}},
        voice_tts={"enabled": True, "audio_output": {"interface": "audio.play", "type": _AUDIOStamped}},
    )
    assert required_interface_ids(config) == ["audio.capture", "audio.play"]
    bound = bind_robot_interfaces(config, _audio_description(description_api))
    for section, field, topic in (
        ("voice_asr", "audio_input", "/audio/capture_stamped"),
        ("speech_direction", "audio_input", "/audio/capture_stamped"),
        ("voice_tts", "audio_output", "/audio/play"),
    ):
        item = bound[section][field]
        assert item["topic"] == topic
        assert item["_interface_source"]["id"] == ("audio.capture" if field == "audio_input" else "audio.play")
    capture = bound["voice_asr"]["audio_input"]
    assert capture["sample_rate_hz"] == 16000
    assert capture["channels"] == 6
    assert capture["sample_format"] == "S16LE"
    assert capture["channel_indices"] == [1, 2, 3, 4]
    playback = bound["voice_tts"]["audio_output"]
    assert playback["sample_rate_hz"] == 24000
    assert playback["channels"] == 1
    assert "channel_indices" not in playback


def test_absent_format_metadata_stays_absent(description_api):
    config = _config(
        voice_tts={"enabled": True, "audio_output": {"interface": "audio.play", "type": _AUDIOStamped}},
    )
    bound = bind_robot_interfaces(config, _audio_description(description_api, with_format=False))
    output = bound["voice_tts"]["audio_output"]
    assert output["topic"] == "/audio/play"
    assert "sample_rate_hz" not in output and "channels" not in output and "sample_format" not in output


def test_streaming_voice_asr_rejects_the_external_message_type(description_api):
    config = _config(
        voice_asr={
            "enabled": True,
            "input_mode": "shared_audio",
            "audio_input": {"interface": "audio.capture", "type": "ibrobot_msgs/msg/SpeechAudioChunk"},
        },
    )
    with pytest.raises(InterfaceBindingError, match="type_mismatch.*AudioDataStamped required"):
        bind_robot_interfaces(config, _audio_description(description_api, vendor_speech=True))


def test_external_voice_asr_still_binds_the_speech_chunks_interface(description_api):
    config = _config(
        voice_asr={
            "enabled": True,
            "input_mode": "external_vad",
            "audio_input": {"interface": "audio.speech", "type": "ibrobot_msgs/msg/SpeechAudioChunk"},
        },
    )
    bound = bind_robot_interfaces(config, _audio_description(description_api, vendor_speech=True))
    assert bound["voice_asr"]["audio_input"]["topic"] == "/audio/speech_chunk"


def test_playback_binding_rejects_a_publish_direction_interface(description_api):
    config = _config(
        voice_tts={"enabled": True, "audio_output": {"interface": "audio.capture", "type": _AUDIOStamped}},
    )
    with pytest.raises(InterfaceBindingError, match="direction_mismatch.*expected provider 'subscribe'"):
        bind_robot_interfaces(config, _audio_description(description_api))


def test_undeclared_audio_fields_are_not_required(description_api):
    config = _config(voice_tts={"enabled": False})
    assert required_interface_ids(config) == []
    bound = bind_robot_interfaces(config, _audio_description(description_api))
    assert "audio_input" not in (bound.get("voice_asr") or {})
    assert "audio_output" not in (bound.get("voice_tts") or {})


def test_lekiwi_style_description_without_vendor_interfaces_binds(description_api):
    """The spec scenario: no audio.speech / audio.enhanced_capture, still fine."""
    description = _audio_description(description_api)
    assert not {"audio.speech", "audio.enhanced_capture"} & set(description["interfaces"])
    config = _config(
        speech_direction={"enabled": True, "audio_input": {"interface": "audio.capture", "type": _AUDIOStamped}},
        voice_tts={"enabled": True, "audio_output": {"interface": "audio.play", "type": _AUDIOStamped}},
        voice_asr={
            "enabled": True,
            "input_mode": "shared_audio",
            "audio_input": {"interface": "audio.capture", "type": _AUDIOStamped},
        },
    )
    bound = bind_robot_interfaces(config, description)
    assert bound["speech_direction"]["audio_input"]["topic"] == "/audio/capture_stamped"
    assert bound["voice_tts"]["audio_output"]["topic"] == "/audio/play"
    assert bound["voice_asr"]["audio_input"]["channels"] == 6


# --- CLI-enabled consumers on the shipped provider deployments -------------------
#
# `voice_asr_auto_start:=true` enables a deployment's voice_asr section without
# further config edits, so every provider deployment that ships the section
# must also declare its audio_input binding: the runtime-ready callback
# materializes the bound consumer configuration and rejects an enabled
# consumer without one (validate_runtime_owned_audio_config). The binding is
# inert while the section stays disabled (`_bindings` skips disabled
# consumers), so declaring it costs nothing.


@pytest.mark.model_bundle(
    "yolox_x_640",
    "pear_parameter_network",
    "grounding_dino_swint_seq8_1280x720",
    "sam2.1_hiera_tiny_prompt_ascend",
    "sam2.1_hiera_tiny",
    "ram_plus_swin_large_14m",
    "siglip2_so400m_patch14_384",
)
def test_lekiwi_nav_grasp_asr_cli_override_validates_and_binds(description_api):
    """The real lekiwi_nav_grasp accepts `voice_asr_auto_start:=true` end to end."""
    from robot_config.config_path import resolve_robot_config_path
    from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

    config = load_robot_config_dict(
        resolve_robot_config_path(config_name="lekiwi_nav_grasp"), defer_interface_binding=True
    )
    # What the voice_asr_auto_start:=true config-override hook writes.
    config["voice_asr"]["enabled"] = True
    config["voice_asr"]["active_mode"] = "continuous"
    validate_robot_config_dict(config, deferred_interfaces=True)

    # The declared binding resolves the capture topic and the array format
    # against a runtime that projects audio.capture.
    asr = dict(config["voice_asr"])
    bound = bind_robot_interfaces(_config(voice_asr=asr), _audio_description(description_api))
    capture = bound["voice_asr"]["audio_input"]
    assert capture["topic"] == "/audio/capture_stamped"
    assert capture["channels"] == 6
    assert capture["sample_rate_hz"] == 16000


def test_aimdk_x2_asr_cli_override_validates():
    """The X2 base deployment ships the same audio.capture declaration."""
    from robot_config.config_path import resolve_robot_config_path
    from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

    config = load_robot_config_dict(resolve_robot_config_path(config_name="aimdk_x2"), defer_interface_binding=True)
    config["voice_asr"]["enabled"] = True
    config["voice_asr"]["active_mode"] = "continuous"
    validate_robot_config_dict(config, deferred_interfaces=True)
    assert config["voice_asr"]["audio_input"]["interface"] == "audio.capture"
    assert config["voice_asr"]["audio_input"]["type"] == _AUDIOStamped

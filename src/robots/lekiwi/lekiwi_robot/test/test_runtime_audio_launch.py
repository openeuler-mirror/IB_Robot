"""LeKiwi runtime launch audio rendering (openspec change ``runtime-owned-audio-io``).

Renders the shipped ``runtime.launch.py`` against the real ``lekiwi_nav_grasp``
profile (which declares the respeaker microphone) plus the deployment speaker
fragment — the same peripheral data the migrated ``lekiwi_nav_grasp``
deployment config hands the runtime — and pins the generated ALSA nodes item
by item to the nodes robot_config's legacy ``audio_io`` entry used to launch
for the same device parameters, in both physical and simulated execution.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import launch
import yaml

_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "lekiwi_nav_grasp.yaml"
_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py"

# The speaker the migrated lekiwi_nav_grasp deployment declares: an
# enabled-true override of the profile-declared (default-disabled) device,
# parameters carried over verbatim from its former audio_io.playback_* keys.
_SPEAKER_FRAGMENT = {
    "peripherals": [
        {
            "type": "speaker",
            "name": "speaker",
            "enabled": True,
            "params": {"device": "", "channels": 1, "sample_rate": 24000, "sample_format": "S16LE"},
        }
    ]
}

_EXPECTED_CAPTURE_PARAMS = {
    "src": "alsasrc",
    "dst": "appsink",
    "format": "wave",
    "device": "hw:0,0",
    "channels": 6,
    "sample_rate": 16000,
    "sample_format": "S16LE",
    "depth": 16,
}

_EXPECTED_PLAYBACK_PARAMS = {
    "dst": "alsasink",
    "device": "",
    "format": "wave",
    "channels": 1,
    "depth": 16,
    "sample_rate": 24000,
    "sample_format": "S16LE",
}


def _runtime_launch_module():
    spec = importlib.util.spec_from_file_location("lekiwi_robot_runtime_launch_under_test", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(simulated: str, peripherals_file: str, profile: Path = _PROFILE):
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(profile),
            "simulated": simulated,
            "use_sim": "",
            "sim_platform": "",
            "gazebo_world": "",
            "spawn_x": "0.0",
            "spawn_y": "0.0",
            "spawn_z": "0.05",
            "spawn_yaw": "0.0",
            "display": "false",
            "instance_id": "",
            "peripherals": peripherals_file,
        }
    )
    return _runtime_launch_module()._launch_setup(context)


def _profile_without_lidar(tmp_path):
    """The shipped profile minus the MID-360 lidar for physical-mode renders.

    ``livox_ros_driver2`` is a third-party driver this workspace does not
    build; its node composition reads the vendor config file, so physical
    renders in CI use a copy without the lidar. Audio composition is
    untouched by that removal.
    """
    profile = yaml.safe_load(_PROFILE.read_text(encoding="utf-8"))
    profile["peripherals"] = [p for p in profile.get("peripherals", []) if p.get("type") != "lidar"]
    path = tmp_path / "lekiwi_no_lidar.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    return path


def _nodes(actions):
    return [action for action in actions if type(action).__name__ == "Node"]


def _node(nodes, name):
    matches = [node for node in nodes if node._Node__node_name == name]
    assert len(matches) == 1, f"expected exactly one node named {name!r}"
    return matches[0]


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _params(node):
    decoded = yaml.safe_load
    return {
        _text(key): (decoded(_text(value)) if isinstance(value, tuple | list) else value)
        for key, value in node._Node__parameters[0].items()
    }


def test_runtime_launch_runs_the_alsa_owners_in_both_modes(tmp_path):
    fragment = tmp_path / "peripherals.yaml"
    fragment.write_text(yaml.safe_dump(_SPEAKER_FRAGMENT), encoding="utf-8")
    physical_profile = _profile_without_lidar(tmp_path)
    for simulated, profile in (("false", physical_profile), ("true", _PROFILE)):
        actions = _render(simulated, str(fragment), profile)
        nodes = _nodes(actions)

        capture = _node(nodes, "audio_capture")
        assert (capture.node_package, capture.node_executable) == ("audio_capture", "audio_capture_node")
        assert _params(capture) == _EXPECTED_CAPTURE_PARAMS
        remappings = [(_text(source), _text(target)) for source, target in capture._Node__remappings]
        assert remappings == [
            ("audio", "/audio/capture"),
            ("audio_stamped", "/audio/capture_stamped"),
            ("audio_info", "/audio/info"),
        ]

        playback = _node(nodes, "audio_play")
        assert (playback.node_package, playback.node_executable) == ("audio_play", "audio_play_node")
        assert _params(playback) == _EXPECTED_PLAYBACK_PARAMS
        remappings = [(_text(source), _text(target)) for source, target in playback._Node__remappings]
        assert remappings == [("audio", "/audio/play")]


def test_physical_selection_passes_no_platform_to_the_facade(tmp_path):
    """Physical execution must reach the facade with no platform (PR !515 P1).

    The launch used to default the platform to "sdk" on the physical path,
    which the facade's selection resolver rejects; the regression goes through
    the real entry: rendered facade parameters must resolve without error.
    """
    from robot_runtime.facade_node import resolve_facade_selection
    from robot_runtime.profile import load_profile

    actions = _render("false", "", _profile_without_lidar(tmp_path))
    facade = _node(_nodes(actions), "runtime_facade")
    params = _params(facade)
    assert params["use_sim"] is False
    assert "sim_platform" not in params, "physical execution must carry no platform"

    profile = load_profile(_profile_without_lidar(tmp_path))
    assert resolve_facade_selection(profile, bool(params["use_sim"]), str(params.get("sim_platform", ""))) == (
        False,
        None,
    )


def test_runtime_launch_without_a_speaker_launches_only_the_capture_owner(tmp_path):
    fragment = tmp_path / "peripherals.yaml"
    fragment.write_text(yaml.safe_dump({"peripherals": []}), encoding="utf-8")
    actions = _render("false", str(fragment), _profile_without_lidar(tmp_path))
    names = {node._Node__node_name for node in _nodes(actions)}
    assert "audio_capture" in names, "the profile's respeaker microphone still owns the capture node"
    assert "audio_play" not in names

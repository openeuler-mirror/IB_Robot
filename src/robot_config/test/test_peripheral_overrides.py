"""runtime-peripheral-overrides: configuration overrides on the provider path.

Openspec change ``runtime-deployment-peripherals`` revision 2 (tasks 3.2-3.4):
a provider configuration's ``peripherals`` entries may set the enabled state
and device settings of runtime-declared peripherals — only ``driver`` and
undeclared peripherals stay runtime-owned — and the entries reach the runtime
verbatim through the existing ``peripherals:=`` fragment (profile < robot
configuration). Also covers the 3.3 audit: robot_config readers of
``robot_config["peripherals"]`` must not treat an override entry as a camera
definition on the live-binding provider path.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.runtime import _write_peripherals_fragment, materialize_interface_config
from robot_config.launch_registry.registry import LaunchBuildContext
from robot_config.loader import load_robot_config_dict

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"


def _provider_config(**runtime):
    return {
        "name": "consumer",
        "runtime": {"provider": "so101_robot", "profile": "so101_single_arm", **runtime},
        "simulation": {"platform": "sdk"},
    }


def _fragment_text(path):
    return Path(path).read_text(encoding="utf-8")


def _legacy_fragment_text(config):
    """The pre-change serialisation: the byte-identity oracle."""
    fragment = {}
    peripherals = config.get("peripherals") or []
    fast_lio = (config.get("navigation") or {}).get("fast_lio") or {}
    if peripherals:
        fragment["peripherals"] = peripherals
    if fast_lio.get("enabled"):
        fragment["fast_lio"] = fast_lio
    if not fragment:
        return ""
    return yaml.safe_dump(fragment, default_flow_style=False, sort_keys=False)


def _raw_provider_chain(path: Path, seen: frozenset[str] = frozenset()) -> bool:
    """Whether a config (or its base_config chain) selects a runtime provider."""
    if path.name in seen or not path.is_file():
        return False
    data = yaml.safe_load(path.read_text()) or {}
    robot = data.get("robot") or {}
    runtime = robot.get("runtime") or {}
    if isinstance(runtime, dict) and runtime.get("provider"):
        return True
    base = robot.get("base_config")
    if isinstance(base, str) and base.strip():
        base_path = path.parent / (base if base.endswith(".yaml") else f"{base}.yaml")
        return _raw_provider_chain(base_path, seen | {path.name})
    return False


# ---------------------------------------------------------------------------
# 3.4: the fragment hand-off carries override entries verbatim
# ---------------------------------------------------------------------------


def test_device_fields_reach_the_fragment_verbatim():
    entry = {
        "type": "camera",
        "name": "top",
        "width": 1280,
        "height": 720,
        "fps": 15,
        "pixel_format": "yuyv",
    }
    config = _provider_config(require_model=True)
    config["peripherals"] = [entry, {"type": "camera", "name": "front", "disabled": True}]
    fragment = _write_peripherals_fragment(config)
    assert yaml.safe_load(_fragment_text(fragment))["peripherals"] == config["peripherals"]


def test_the_runtime_merge_applies_the_configuration_layer():
    # profile < robot configuration: the fragment entry deep-merges over the
    # profile's device settings and the runtime owns the driver.
    from robot_runtime.interface_description import merge_peripherals

    profile = [{"type": "camera", "name": "top", "driver": "opencv", "width": 640, "height": 480, "fps": 30}]
    merged = merge_peripherals(profile, [{"type": "camera", "name": "top", "width": 1280, "fps": 15}])
    assert merged == [{"type": "camera", "name": "top", "driver": "opencv", "width": 1280, "height": 480, "fps": 15}]


def test_live_binding_releases_no_consumers_when_the_waiter_fails(monkeypatch, tmp_path):
    """D9: through generic launch on a live-binding configuration, consumers
    are constructed only in the waiter's OnProcessExit on success — an
    undeclared override name fails the provider process, the waiter exits
    non-zero, and no bound consumer is constructed or started."""
    from launch.actions import EmitEvent

    import robot_config.launch_builders.runtime as runtime_module

    provider_launch = tmp_path / "runtime.launch.py"
    provider_launch.write_text("# provider is not executed by this construction test\n", encoding="utf-8")
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(provider_launch))

    config = _provider_config(require_model=True)
    config["peripherals"] = [{"type": "camera", "name": "tpo", "width": 1280}]
    constructed = []

    def construct_consumers(effective):
        constructed.append(effective)
        return [object()]

    actions = runtime_module.generate_bound_runtime_actions(
        config, construct_consumers, use_sim=True, sim_platform="sdk"
    )
    handlers = [
        getattr(action.event_handler, "_OnActionEventBase__on_event", None)
        for action in actions
        if type(action).__name__ == "RegisterEventHandler"
    ]
    on_exit = next(handler for handler in handlers if handler is not None)
    deferred = on_exit(type("ProcessExited", (), {"returncode": 1})(), type("Context", (), {"is_shutdown": False})())
    assert deferred and all(isinstance(action, EmitEvent) for action in deferred), (
        "a failed waiter must shut the launch down, not start consumers"
    )
    assert constructed == [], "no bound consumer may be constructed on failure"


# ---------------------------------------------------------------------------
# 3.3: the audit — robot_config readers must not treat an override entry as a
# camera definition on the live-binding provider path
# ---------------------------------------------------------------------------


def _descriptor_with_top_camera(width, height, fps):
    import time

    from robot_runtime.interface_description import build_description, description_digest

    profile = {
        "runtime": {"name": "so101_robot", "version": "0.1"},
        "modes": {"idle": {"controllers": []}},
        "capabilities": {"joint.state": {}},
        "joints": ["j1"],
        "peripherals": [
            {"type": "camera", "name": "top", "driver": "opencv", "width": width, "height": height, "fps": fps}
        ],
    }
    description = build_description(profile, use_sim=True)
    # mark the camera interface ready so a require_ready binding accepts it
    description["states"]["camera.top.color"] = {
        "state": "ready",
        "observed_profile": {"width": width, "height": height, "fps": fps, "encoding": "rgb8"},
        "observed_frame_id": "camera_top_frame",
        "last_seen": time.time(),
        "detail": "",
    }
    description["digest"] = description_digest(description)
    return description


def _binding_config():
    config = _provider_config()
    config["contract"] = {
        "rate_hz": 20,
        "observations": [
            {
                "key": "observation.images.top",
                "interface": "camera.top.color",
                "requires": {"width": 1280, "height": 720, "min_fps": 15},
                "image": {"resize": [720, 1280]},
            }
        ],
    }
    return config


def test_bound_consumers_take_camera_settings_from_the_description(tmp_path):
    """The observation binding reads the bound description, not
    ``robot_config["peripherals"]``: a device-field override entry does not
    change the effective contract output (3.3 audit, generators/contract +
    observation_transport + contract_builder)."""
    descriptor = _descriptor_with_top_camera(1280, 720, 15)
    base = _binding_config()
    with_entry = copy.deepcopy(base)
    # a device-field entry that does NOT match the description's profile:
    # the binding must ignore it entirely
    with_entry["peripherals"] = [{"type": "camera", "name": "top", "width": 640, "height": 480, "fps": 5}]

    effective_without = materialize_interface_config(base, descriptor, tmp_path / "without.yaml")
    effective_with = materialize_interface_config(with_entry, descriptor, tmp_path / "with.yaml")

    assert effective_with["contract"]["observations"] == effective_without["contract"]["observations"]
    source = effective_with["contract"]["observations"][0]["_interface_source"]
    assert (source["profile"]["width"], source["profile"]["height"], source["profile"]["fps"]) == (1280, 720, 15), (
        "the bound profile comes from the description, never from the configuration entry"
    )


def test_perception_builder_ignores_override_entries_under_a_provider():
    """The generic perception builder owns physical drivers only on the
    provider-less path; under a runtime provider a device-field entry composes
    no camera driver node (3.3 audit, launch_builders/perception)."""
    from robot_config.launch_builders.perception import build_perception

    config = _provider_config(require_model=True)
    config["peripherals"] = [{"type": "camera", "name": "top", "driver": "ignored", "width": 1280, "index": 3}]
    context = LaunchBuildContext(robot_config=config, use_sim=False, sim_platform="sdk", node_use_sim_time=False)
    result = build_perception(context)
    assert result.actions == [], "the runtime owns physical perception under a provider"


def test_virtual_camera_relays_cannot_compose_from_override_entries():
    """Virtual camera relays match ``driver: virtual`` entries; the loader
    rejects driver on require_model provider entries, so an override entry can
    never become a relay (3.3 audit, launch_builders/perception)."""
    from robot_config.launch_builders.perception import generate_virtual_camera_relays

    config = _provider_config(require_model=True)
    config["peripherals"] = [{"type": "camera", "name": "wrist", "source_topic": "/camera/top/image_raw"}]
    assert generate_virtual_camera_relays(config) == []


_APPROVED_ARTIFACT = {
    "schema_version": "1.0",
    "status": "approved",
    "device": {"name": "front_camera", "serial": "camera-1"},
    "transform": {
        "parent_frame": "base_link",
        "child_frame": "camera_front_optical_frame",
        "translation": [0.12, -0.34, 0.56],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
    },
}


def _write_config(tmp_path, name, robot):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"robot": robot}), encoding="utf-8")
    return path


def test_approved_calibration_skips_require_model_override_entries(tmp_path, monkeypatch):
    """3.3 audit (loader camera calibration): on the require_model provider
    path calibration files are runtime-owned; an override entry carrying
    ``optical_frame_id`` is not a camera definition to provision a transform
    into — the load neither crashes nor mutates the entry."""
    artifact = tmp_path / "home" / ".ros" / "ibrobot" / "calib" / "current" / "base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(yaml.safe_dump(_APPROVED_ARTIFACT), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    entry = {"type": "camera", "name": "front", "optical_frame_id": "camera_front_optical_frame", "width": 1280}
    config = _write_config(
        tmp_path,
        "require_model_override.yaml",
        {
            "name": "require_model_override",
            "runtime": {"provider": "so101_robot", "profile": "so101_single_arm", "require_model": True},
            "simulation": {"platform": "sdk"},
            "sensor_calibration": {
                "artifacts": {"base_to_front_camera": "~/.ros/ibrobot/calib/current/base_to_front_camera.yaml"}
            },
            "peripherals": [entry],
        },
    )
    loaded = load_robot_config_dict(config, defer_interface_binding=True)
    assert loaded["peripherals"] == [entry], "the override entry must pass through verbatim"


def test_approved_calibration_still_applies_without_require_model(tmp_path, monkeypatch):
    """Non-require_model provider configurations keep the artifact flow: their
    public entries are the camera definitions (LeKiwi's front camera)."""
    artifact = tmp_path / "home" / ".ros" / "ibrobot" / "calib" / "current" / "base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(yaml.safe_dump(_APPROVED_ARTIFACT), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    config = _write_config(
        tmp_path,
        "provider_definition.yaml",
        {
            "name": "provider_definition",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "sensor_calibration": {
                "artifacts": {"base_to_front_camera": "~/.ros/ibrobot/calib/current/base_to_front_camera.yaml"}
            },
            "peripherals": [
                {
                    "type": "camera",
                    "name": "front",
                    "optical_frame_id": "camera_front_optical_frame",
                    "transform": {"parent_frame": "base_link", "x": 0.0, "y": 0.0, "z": 0.0},
                }
            ],
        },
    )
    loaded = load_robot_config_dict(config, defer_interface_binding=True)
    transform = next(item for item in loaded["peripherals"] if item["name"] == "front")["transform"]
    assert (transform["x"], transform["y"], transform["z"]) == (0.12, -0.34, 0.56)


def test_mid360_mount_updates_peripheral_entries_on_the_provider_path(tmp_path):
    """lekiwi-gazebo-runtime-platform D13: on the provider path the mount
    creates or updates the partial (lidar, mid360) entry — the public
    body -> livox_frame identity transform plus the base_link-relative
    simulator attachment pose — next to the navigation provisioning (static
    TF, fast_lio frames). Device settings the entry carries survive."""
    from robot_config.loader import resolve_ros_path

    mount_file = Path(resolve_ros_path("$(find lekiwi_description)/config/lekiwi_mid360_mount.yaml"))
    assert mount_file.is_file()
    entry = {"type": "lidar", "name": "mid360", "frame_id": "livox_frame", "publish_freq": 10.0}
    config = _write_config(
        tmp_path,
        "provider_mount.yaml",
        {
            "name": "provider_mount",
            "runtime": {"provider": "so101_robot", "profile": "so101_single_arm", "require_model": True},
            "simulation": {"platform": "sdk"},
            "mid360_mount_file": str(mount_file),
            "peripherals": [entry],
        },
    )
    loaded = load_robot_config_dict(config, defer_interface_binding=True)
    assert len(loaded["peripherals"]) == 1, "the existing entry is updated, not duplicated"
    updated = loaded["peripherals"][0]
    # device settings survive
    assert updated["frame_id"] == "livox_frame" and updated["publish_freq"] == 10.0
    # both D13 pose keys are set on the provider path
    assert updated["transform"] == {
        "parent_frame": "body",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    assert updated["simulation"]["gazebo"]["transform"]["parent_frame"] == "base_link"
    static_tfs = loaded["navigation"]["static_tfs"]
    assert any(item.get("child_frame") == "body" for item in static_tfs), "navigation provisioning still applies"
    fast_lio = loaded["navigation"]["fast_lio"]
    # translation_m [-0.08, 0, 0.20] with rpy_deg [0, 0, 90]: the inverse
    # body→base provision is -Rz(-90°)·t = [0, -0.08, -0.20]
    assert fast_lio["body_to_base_translation"] == pytest.approx([0.0, -0.08, -0.20])


def test_mid360_mount_still_mounts_without_require_model(tmp_path):
    """LeKiwi-style provider configurations keep the mount: the public lidar
    entry is the device definition."""
    config = load_robot_config_dict(ROBOTS / "lekiwi_nav_grasp.yaml", defer_interface_binding=True)
    lidar = next(item for item in config["peripherals"] if item.get("type") == "lidar")
    assert lidar["transform"] == {
        "parent_frame": "body",
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }


# ---------------------------------------------------------------------------
# 3.4: byte identity, spellings, snapshot
# ---------------------------------------------------------------------------


def test_shipped_provider_fragments_are_byte_identical():
    """Every shipped provider configuration (including overlays inheriting the
    provider through base_config) produces the exact pre-change fragment
    bytes — with the sensor mount and the approved-calibration artifact
    declared (lekiwi_nav_grasp) and the loaded_config entry included.

    Two documented exceptions: the three LeKiwi configurations migrated for
    the all-provider driver ban (review M1) deliberately dropped their
    redundant ``driver`` keys, so their fragments differ by design; their
    effective peripheral sets are pinned by
    ``test_lekiwi_migration_effective_sets`` instead.

    unitree_g1.yaml cannot load in any environment since the
    runtime-owned-audio-io change (its audio_io section predates the rule and
    fails validation on c8be174e already); it produces no fragment today
    either, so it is out of the byte-identity comparison by necessity.
    """
    known_preexisting_load_failures = {"unitree_g1.yaml"}
    migrated_effective_set_configs = {
        "lekiwi_nav_grasp.yaml",
        "lekiwi_hybrid_switch_test.yaml",
        "lekiwi_nav_grasp_sound_real.yaml",
    }
    load_failures = []
    compared = 0
    for path in sorted(ROBOTS.glob("*.yaml")):
        try:
            config = load_robot_config_dict(path, defer_interface_binding=True)
        except Exception:
            if _raw_provider_chain(path):
                load_failures.append(path.name)
            continue
        if not (config.get("runtime") or {}).get("provider"):
            continue
        if path.name in migrated_effective_set_configs:
            continue
        assert "_disabled_peripherals" not in config["runtime"] or config["runtime"]["_disabled_peripherals"] == [], (
            path.name
        )
        fragment = _write_peripherals_fragment(config)
        expected = _legacy_fragment_text(config)
        actual = _fragment_text(fragment) if fragment else ""
        assert actual == expected, path.name
        compared += 1
    assert load_failures == sorted(known_preexisting_load_failures)
    assert compared >= 7, f"expected the shipped provider set, compared {compared}"
    # the loaded_config entry: a config dict without the loader's private key
    # (a caller that skips the loader) serialises exactly its peripherals
    loaded_config_entry = _provider_config()
    loaded_config_entry["peripherals"] = [{"type": "camera", "name": "top", "disabled": True}]
    fragment = _write_peripherals_fragment(loaded_config_entry)
    assert _fragment_text(fragment) == _legacy_fragment_text(loaded_config_entry)


def test_lekiwi_migration_effective_sets():
    """Review M1: the three LeKiwi configurations migrated off redundant
    ``driver`` keys keep the exact pre-migration effective peripheral sets
    through the runtime merge — the speaker appears only where it did before
    (lekiwi_nav_grasp), and the profile now declares it disabled by default
    so configurations whose stages exclude it (sound_real) and overlays that
    disable it (hybrid_switch_test) are unchanged."""
    from robot_runtime.interface_description import merge_peripherals
    from robot_runtime.profile import load_profile

    profile = load_profile(ROBOTS.parents[2] / "robots/lekiwi/lekiwi_robot/profiles/lekiwi_nav_grasp.yaml")
    profile_peripherals = profile.get("peripherals")

    def effective(name):
        config = load_robot_config_dict(ROBOTS / name, defer_interface_binding=True)
        application = list(config.get("peripherals") or []) + list(
            (config.get("runtime") or {}).get("_disabled_peripherals") or []
        )
        return merge_peripherals(profile_peripherals, application)

    nav_grasp = effective("lekiwi_nav_grasp.yaml")
    assert [(entry["type"], entry["name"]) for entry in nav_grasp] == [
        ("microphone", "respeaker"),
        ("speaker", "speaker"),
        ("camera", "wrist"),
        ("camera", "front"),
        ("lidar", "mid360"),
    ]
    speaker = next(entry for entry in nav_grasp if entry["name"] == "speaker")
    assert speaker["driver"] == "alsa"  # the profile's declaration
    assert speaker.get("enabled", True) is True  # this deployment enables it
    assert speaker["params"] == {
        "device": "",
        "channels": 1,
        "sample_rate": 24000,
        "sample_format": "S16LE",
    }

    hybrid = effective("lekiwi_hybrid_switch_test.yaml")
    assert [(entry["type"], entry["name"]) for entry in hybrid] == [
        ("camera", "wrist"),
        ("camera", "front"),
        ("lidar", "mid360"),
    ], "the switch test keeps audio off: respeaker and speaker stay disabled"

    sound_real = effective("lekiwi_nav_grasp_sound_real.yaml")
    assert [(entry["type"], entry["name"]) for entry in sound_real] == [
        ("microphone", "respeaker"),
        ("camera", "wrist"),
        ("camera", "front"),
        ("lidar", "mid360"),
    ], "sound_real never had a speaker: the profile default keeps it off"

    # the migrated entries carry no driver and still validate
    from robot_config.loader import validate_runtime_provider_config

    for name in ("lekiwi_nav_grasp.yaml", "lekiwi_hybrid_switch_test.yaml", "lekiwi_nav_grasp_sound_real.yaml"):
        config = load_robot_config_dict(ROBOTS / name, defer_interface_binding=True)
        assert all("driver" not in entry for entry in config["peripherals"]), name
        assert not [error for error in validate_runtime_provider_config(config) if "driver" in error], name


def test_driver_is_rejected_for_every_provider_configuration(tmp_path):
    """Review M1: the driver ban covers providers without require_model too —
    the runtime owns the driver on every provider path."""
    config = _write_config(
        tmp_path,
        "lekiwi_driver.yaml",
        {
            "name": "lekiwi_driver",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "peripherals": [{"type": "camera", "name": "front", "driver": "realsense"}],
        },
    )
    with pytest.raises(ValueError, match="may not set driver: driver is runtime-owned"):
        load_robot_config_dict(config, defer_interface_binding=True)


def test_undeclared_device_is_rejected_by_the_runtime_merge(tmp_path):
    """Review M1: a provider configuration without require_model naming a
    peripheral the runtime does not declare loads (the entry carries no
    driver), and the runtime's completeness rule rejects it before any bound
    consumer is constructed."""
    from robot_runtime.interface_description import InterfaceDescriptionError, merge_peripherals
    from robot_runtime.profile import load_profile

    config = _write_config(
        tmp_path,
        "lekiwi_undeclared.yaml",
        {
            "name": "lekiwi_undeclared",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "peripherals": [{"type": "camera", "name": "spare_cam", "width": 1280}],
        },
    )
    loaded = load_robot_config_dict(config, defer_interface_binding=True)
    profile = load_profile(ROBOTS.parents[2] / "robots/lekiwi/lekiwi_robot/profiles/lekiwi_nav_grasp.yaml")
    with pytest.raises(InterfaceDescriptionError, match=r"\('camera', 'spare_cam'\) is not declared"):
        merge_peripherals(profile.get("peripherals"), loaded["peripherals"])


def test_migration_keeps_the_robot_config_consumers_unchanged(tmp_path):
    """Review M1(b): the migrated LeKiwi configurations still satisfy the
    robot_config-side consumers — the speech-direction microphone and
    semantic-mapping camera validators accept driver-less override entries on
    the provider path, and the mid360 mount still applies."""
    sound_real = load_robot_config_dict(ROBOTS / "lekiwi_nav_grasp_sound_real.yaml", defer_interface_binding=True)
    microphone = next(entry for entry in sound_real["peripherals"] if entry.get("type") == "microphone")
    assert "driver" not in microphone  # the override carries no driver
    from robot_config.loader import validate_semantic_mapping_config, validate_speech_direction_config

    assert not validate_speech_direction_config(sound_real), "speech_direction validates without a config driver"

    nav_grasp = load_robot_config_dict(ROBOTS / "lekiwi_nav_grasp.yaml", defer_interface_binding=True)
    front = next(entry for entry in nav_grasp["peripherals"] if entry.get("name") == "front")
    assert "driver" not in front and front.get("align_depth") is True
    assert not validate_semantic_mapping_config(nav_grasp), "semantic_mapping validates without a config driver"

    lidar = next(entry for entry in nav_grasp["peripherals"] if entry.get("type") == "lidar")
    assert lidar["transform"]["parent_frame"] == "body", "the mid360 mount still applies"


def test_no_shipped_non_require_model_provider_config_disables_peripherals():
    """D2: no shipped provider configuration without require_model uses a
    disable spelling, so the recording never changes a shipped fragment."""
    offenders = []
    for path in sorted(ROBOTS.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        robot = data.get("robot") or {}
        runtime = robot.get("runtime") or {}
        if not isinstance(runtime, dict) or not runtime.get("provider") or runtime.get("require_model"):
            continue
        for entry in robot.get("peripherals") or []:
            if isinstance(entry, dict) and (entry.get("disabled") or entry.get("enabled") is False):
                offenders.append(f"{path.name}:{entry.get('name')}")
    assert not offenders, offenders


def test_loader_records_disabled_entries_for_providers_without_require_model(tmp_path):
    config = _write_config(
        tmp_path,
        "lekiwi_like.yaml",
        {
            "name": "lekiwi_like",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "peripherals": [
                {"type": "camera", "name": "front"},
                {"type": "camera", "name": "top", "disabled": True},
                {"type": "camera", "name": "wrist", "enabled": True},
            ],
        },
    )
    loaded = load_robot_config_dict(config, defer_interface_binding=True)
    assert [(entry["name"], entry.get("disabled")) for entry in loaded["peripherals"]] == [
        ("front", None),
        ("wrist", None),
    ]
    assert loaded["runtime"]["_disabled_peripherals"] == [{"type": "camera", "name": "top", "disabled": True}]
    # a provider-less configuration records nothing
    providerless = _write_config(
        tmp_path,
        "legacy.yaml",
        {"name": "legacy", "peripherals": [{"type": "camera", "name": "top", "disabled": True}]},
    )
    loaded = load_robot_config_dict(providerless, defer_interface_binding=True)
    assert loaded["peripherals"] == [] and "_disabled_peripherals" not in (loaded.get("runtime") or {})


def test_user_supplied_disabled_peripherals_key_is_rejected(tmp_path):
    config = _write_config(
        tmp_path,
        "rogue.yaml",
        {
            "name": "rogue",
            "runtime": {
                "provider": "so101_robot",
                "profile": "so101_single_arm",
                "require_model": True,
                "_disabled_peripherals": [{"type": "camera", "name": "top", "disabled": True}],
            },
            "simulation": {"platform": "sdk"},
        },
    )
    with pytest.raises(ValueError, match="runtime._disabled_peripherals is framework-generated"):
        load_robot_config_dict(config, defer_interface_binding=True)


def test_lekiwi_disable_spellings_are_equivalent():
    """Both spellings disable the runtime camera on a provider without
    require_model: `disabled: true` through the recorded entry, `enabled:
    false` through the public list; the runtime merge removes the profile's
    device in both cases (the spec's both-spellings scenario)."""
    from robot_runtime.interface_description import merge_peripherals

    profile = [{"type": "camera", "name": "front", "driver": "camera_ros", "width": 640, "height": 480, "fps": 30}]

    def effective(disabled_spelling):
        config = _provider_config()
        entry = (
            {"type": "camera", "name": "front", "disabled": True}
            if disabled_spelling
            else {"type": "camera", "name": "front", "enabled": False}
        )
        # simulate the loader: the disabled spelling is filtered and recorded,
        # the enabled spelling stays in the public list
        if disabled_spelling:
            config["runtime"]["_disabled_peripherals"] = [entry]
        else:
            config["peripherals"] = [entry]
        fragment = _write_peripherals_fragment(config)
        composed = yaml.safe_load(_fragment_text(fragment))["peripherals"]
        return merge_peripherals(profile, composed)

    assert effective(True) == effective(False) == []


def test_so101_rtp_distributed_toggle_is_unchanged():
    loaded = load_robot_config_dict(ROBOTS / "so101_rtp_distributed.yaml", defer_interface_binding=True)
    assert {"name": "front", "type": "camera", "disabled": True} in loaded["peripherals"]
    fragment = _write_peripherals_fragment(loaded)
    data = yaml.safe_load(_fragment_text(fragment))
    assert data["peripherals"] == loaded["peripherals"]


def test_snapshot_drops_the_private_key_and_reloads(tmp_path):
    entry = {"type": "camera", "name": "top", "width": 1280, "height": 720, "fps": 15}
    config = _provider_config()
    config["peripherals"] = [entry]
    config["runtime"]["_disabled_peripherals"] = [{"type": "camera", "name": "wrist", "disabled": True}]
    before = copy.deepcopy(config)
    destination = tmp_path / "robot.yaml"
    effective = materialize_interface_config(config, _descriptor_with_top_camera(1280, 720, 15), destination)
    document = yaml.safe_load(destination.read_text(encoding="utf-8"))["robot"]
    assert "_disabled_peripherals" not in document["runtime"]
    assert document["peripherals"] == [entry]
    # the input configuration is not mutated
    assert config == before
    # the returned effective config keeps the application layer for consumers
    assert effective["peripherals"] == [entry]
    # the snapshot reloads and validates
    reloaded = load_robot_config_dict(destination, defer_interface_binding=True)
    assert reloaded["peripherals"] == [entry]
    assert "_disabled_peripherals" not in reloaded["runtime"]


# ---------------------------------------------------------------------------
# 4.6 (carried): every shipped configuration loads through the new merge
# ---------------------------------------------------------------------------

_PROVIDER_PROFILE_DIRS = {
    "so101_robot": "so101",
    "lekiwi_robot": "lekiwi",
    "unitree_g1_robot": "unitree",
    "aimdk_robot": "aimdk",
}


def test_every_shipped_provider_configuration_merges_with_its_profile():
    """No shipped configuration regresses under the stricter merge: each
    provider configuration's effective fragment (application peripherals plus
    the recorded disabled entries) merges with its provider profile."""
    from robot_runtime.interface_description import merge_peripherals
    from robot_runtime.profile import load_profile

    source_root = ROBOTS.parents[2] / "robots"
    checked = 0
    for path in sorted(ROBOTS.glob("*.yaml")):
        try:
            config = load_robot_config_dict(path, defer_interface_binding=True)
        except Exception:
            continue  # the pre-existing load failures are pinned by the byte-identity test
        runtime = config.get("runtime") or {}
        provider = runtime.get("provider")
        if not provider:
            continue
        application = list(config.get("peripherals") or []) + list(runtime.get("_disabled_peripherals") or [])
        if provider == "mock_runtime":
            merged = merge_peripherals([], application)
        else:
            profile_path = (
                source_root
                / _PROVIDER_PROFILE_DIRS[provider]
                / provider
                / "profiles"
                / (str(runtime["profile"]) + ".yaml")
            )
            assert profile_path.is_file(), f"{path.name}: profile {profile_path} not found"
            profile = load_profile(profile_path)
            merged = merge_peripherals(profile.get("peripherals"), application)
        assert isinstance(merged, list)
        checked += 1
    assert checked >= 10, f"expected the shipped provider set, checked {checked}"


# ---------------------------------------------------------------------------
# Review round 2, Major 1: the semantic-mapping consumer's camera interfaces
# gate the runtime readiness waiter on every provider path
# ---------------------------------------------------------------------------


def test_semantic_mapping_interface_ids_follow_the_description_naming():
    from robot_config.interface_binding import semantic_mapping_interface_ids

    base = {"semantic_mapping": {"enabled": True, "camera": {"peripheral": "front"}}}
    assert semantic_mapping_interface_ids(base) == [
        "camera.front.color",
        "camera.front.aligned_depth",
        "camera.front.color_info",
    ]
    assert semantic_mapping_interface_ids({"semantic_mapping": {"enabled": False}}) == []
    assert semantic_mapping_interface_ids({}) == []
    # the query-only node subscribes to no camera
    query_only = {"semantic_mapping": {"enabled": True, "query_only": True, "camera": {"peripheral": "front"}}}
    assert semantic_mapping_interface_ids(query_only) == []
    # no camera section configured: nothing to gate on
    assert semantic_mapping_interface_ids({"semantic_mapping": {"enabled": True}}) == []


def _waiter_flag_values(robot_config, flag, **kwargs):
    from launch.actions import IncludeLaunchDescription

    from robot_config.launch_builders.runtime import generate_runtime_provider_actions

    actions, waiter = generate_runtime_provider_actions(robot_config, use_sim=kwargs.pop("use_sim", True), **kwargs)
    assert next(action for action in actions if isinstance(action, IncludeLaunchDescription))
    arguments = waiter._Node__arguments
    if flag not in arguments:
        return []
    start = arguments.index(flag) + 1
    end = next((i for i in range(start, len(arguments)) if arguments[i].startswith("--")), len(arguments))
    return arguments[start:end]


def _waiter_interface_ids(robot_config, **kwargs):
    return _waiter_flag_values(robot_config, "--declared-interfaces", **kwargs)


def _lekiwi_semantic_config(query_only=False):
    return {
        "name": "lekiwi_semantic",
        "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
        "simulation": {"platform": "sdk"},
        "semantic_mapping": {
            "enabled": True,
            "query_only": query_only,
            "camera": {"peripheral": "front"},
        },
    }


def test_semantic_mapping_interfaces_gate_the_waiter_without_live_binding():
    """A non-require_model provider configuration (LeKiwi) keeps its
    non-binding startup flow AND gains the interface gate: the waiter's
    --require-interfaces carries the semantic-mapping camera IDs."""
    from launch import LaunchContext

    from robot_config.launch_registry.builtin_registry import build_builtin_registry
    from robot_config.launch_registry.preparation import prepare_launch

    config = _lekiwi_semantic_config(query_only=False)
    assert _waiter_interface_ids(config) == [
        "camera.front.color",
        "camera.front.aligned_depth",
        "camera.front.color_info",
    ]
    assert _waiter_flag_values(config, "--require-interfaces") == [], "the readiness gate itself stays contract-driven"
    context = LaunchContext()
    context.launch_configurations.update({"use_sim": "true", "sim_platform": ""})
    prepared = prepare_launch(context, build_builtin_registry(), loaded_config=config)
    assert prepared.live_binding_required is False, "the LeKiwi startup flow must not become live-binding"


def test_query_only_semantic_mapping_adds_no_gate():
    assert _waiter_interface_ids(_lekiwi_semantic_config(query_only=True)) == []
    assert _waiter_interface_ids(_provider_config(require_model=True)) == []


def test_semantic_mapping_declaration_stays_separate_from_contract_requirements():
    """On the binding path the contract keeps driving --require-interfaces /
    --interface-requirements exactly as before this change (upstream
    behaviour), while the semantic-mapping IDs ride --declared-interfaces."""
    config = _provider_config(require_model=True)
    config["semantic_mapping"] = {"enabled": True, "camera": {"peripheral": "front"}}
    config["contract"] = {
        "observations": [
            {"key": "observation.images.front", "interface": "camera.front.color", "requires": {"min_fps": 15}}
        ]
    }
    assert _waiter_flag_values(config, "--require-interfaces", description_output="/tmp/description.yaml") == [
        "camera.front.color"
    ], "the readiness gate stays exactly the contract's interface set"
    assert _waiter_interface_ids(config, description_output="/tmp/description.yaml") == [
        "camera.front.color",
        "camera.front.aligned_depth",
        "camera.front.color_info",
    ]
    from launch.actions import IncludeLaunchDescription

    from robot_config.launch_builders.runtime import generate_runtime_provider_actions

    actions, waiter = generate_runtime_provider_actions(
        config, use_sim=True, description_output="/tmp/description.yaml"
    )
    assert next(action for action in actions if isinstance(action, IncludeLaunchDescription))
    arguments = waiter._Node__arguments
    assert "--interface-requirements" in arguments, "the binding path keeps its contract requirements"


def _front_description(peripherals):
    """The description a lekiwi-like runtime would serve for these effective
    peripherals. No readiness states are crafted on purpose: the semantic
    gate is a declaration check, so a not-yet-ready stream must pass it."""
    from robot_runtime.interface_description import build_description

    profile = {
        "runtime": {"name": "lekiwi_robot", "version": "0.1"},
        "modes": {"idle": {"controllers": []}},
        "capabilities": {"joint.state": {}},
        "joints": ["j1"],
        "peripherals": peripherals,
    }
    return build_description(profile, use_sim=True)


def test_semantic_mapping_negative_cases_fail_at_the_waiter_with_named_errors():
    """The three review cases: an opencv camera, a realsense whose final
    streams are colour-only, and a disabled camera — none DECLARES the
    aligned-depth interface the online semantic-mapping consumer needs, and
    the readiness waiter names the undeclared interface before any
    runtime_ready consumer starts. A declared but not-yet-ready stream
    passes: the shared barrier must not block on a slow driver."""
    import json
    from types import SimpleNamespace

    from robot_config.interface_binding import semantic_mapping_interface_ids
    from robot_runtime.wait_for_runtime import validate_runtime_snapshot

    config = _lekiwi_semantic_config(query_only=False)
    required = semantic_mapping_interface_ids(config)
    assert "camera.front.aligned_depth" in required and "camera.front.color" in required

    cases = {
        "opencv camera carries no aligned depth": [
            {"type": "camera", "name": "front", "driver": "opencv", "width": 640, "height": 480, "fps": 30}
        ],
        "realsense with colour-only streams": [
            {
                "type": "camera",
                "name": "front",
                "driver": "realsense",
                "streams": ["color"],
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        ],
        "disabled camera": [],
    }

    def snapshot_for(peripherals):
        return SimpleNamespace(
            runtime_name="lekiwi_robot",
            runtime_version="0.1",
            lifecycle="ACTIVE",
            capabilities=["runtime.status"],
            interface_description_json=json.dumps(_front_description(peripherals)),
        )

    for _case, peripherals in cases.items():
        with pytest.raises(ValueError, match="runtime does not declare interface 'camera.front."):
            validate_runtime_snapshot(
                snapshot_for(peripherals), runtime_name="lekiwi_robot", declared_interfaces=required
            )

    # declared but not ready (no crafted states): the gate passes — only the
    # missing aligned-depth declaration fails below
    opencv = snapshot_for(
        [{"type": "camera", "name": "front", "driver": "opencv", "width": 640, "height": 480, "fps": 30}]
    )
    validate_runtime_snapshot(
        opencv, runtime_name="lekiwi_robot", declared_interfaces=["camera.front.color", "camera.front.color_info"]
    )
    with pytest.raises(ValueError, match="does not declare interface 'camera.front.aligned_depth'"):
        validate_runtime_snapshot(
            opencv,
            runtime_name="lekiwi_robot",
            declared_interfaces=["camera.front.color", "camera.front.aligned_depth", "camera.front.color_info"],
        )


def test_the_shipped_lekiwi_rgbd_camera_satisfies_the_semantic_gate():
    """lekiwi_nav_grasp's effective front camera (realsense, aligned depth)
    publishes every interface the gate requires — the shipped configuration
    keeps passing."""
    from robot_runtime.interface_description import merge_peripherals
    from robot_runtime.profile import load_profile

    profile = load_profile(ROBOTS.parents[2] / "robots/lekiwi/lekiwi_robot/profiles/lekiwi_nav_grasp.yaml")
    config = load_robot_config_dict(ROBOTS / "lekiwi_nav_grasp.yaml", defer_interface_binding=True)
    merged = merge_peripherals(profile.get("peripherals"), config.get("peripherals"))
    description = _front_description(merged)
    for interface_id in ("camera.front.color", "camera.front.aligned_depth", "camera.front.color_info"):
        assert interface_id in description["interfaces"], interface_id


# ---------------------------------------------------------------------------
# Review round 2, Minor 1: disabled entries and non-mapping entries cannot
# bypass the provider entry validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "camera", "name": "front", "driver": "realsense"},
        {"type": "camera", "name": "front", "driver": "realsense", "enabled": False},
        {"type": "camera", "name": "front", "driver": "realsense", "disabled": True},
    ],
)
def test_driver_is_rejected_in_every_toggle_state(tmp_path, entry):
    config = _write_config(
        tmp_path,
        "driver_toggle.yaml",
        {
            "name": "driver_toggle",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "peripherals": [entry],
        },
    )
    with pytest.raises(ValueError, match="may not set driver: driver is runtime-owned"):
        load_robot_config_dict(config, defer_interface_binding=True)


def test_a_disabled_declared_entry_is_recorded_not_rejected(tmp_path):
    config = _write_config(
        tmp_path,
        "disabled_declared.yaml",
        {
            "name": "disabled_declared",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "peripherals": [
                {"type": "camera", "name": "front", "width": 1280},
                {"type": "camera", "name": "wrist", "disabled": True},
            ],
        },
    )
    loaded = load_robot_config_dict(config, defer_interface_binding=True)
    assert [entry["name"] for entry in loaded["peripherals"]] == ["front"]
    assert loaded["runtime"]["_disabled_peripherals"] == [{"type": "camera", "name": "wrist", "disabled": True}]


def test_a_disabled_undeclared_entry_is_rejected_by_the_runtime_merge(tmp_path):
    """The loader cannot know the runtime's declarations; the completeness
    rule rejects the undeclared disabled entry when the runtime merges."""
    from robot_runtime.interface_description import InterfaceDescriptionError, merge_peripherals
    from robot_runtime.profile import load_profile

    config = _write_config(
        tmp_path,
        "disabled_undeclared.yaml",
        {
            "name": "disabled_undeclared",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "peripherals": [{"type": "camera", "name": "spare_cam", "disabled": True}],
        },
    )
    loaded = load_robot_config_dict(config, defer_interface_binding=True)
    # the fragment's application layer: the public list plus the recorded
    # disabled entries
    application = list(loaded["peripherals"]) + list(loaded["runtime"]["_disabled_peripherals"])
    profile = load_profile(ROBOTS.parents[2] / "robots/lekiwi/lekiwi_robot/profiles/lekiwi_nav_grasp.yaml")
    with pytest.raises(InterfaceDescriptionError, match=r"\('camera', 'spare_cam'\) is not declared"):
        merge_peripherals(profile.get("peripherals"), application)


def test_a_non_mapping_provider_entry_is_an_error_not_a_drop(tmp_path):
    config = _write_config(
        tmp_path,
        "non_mapping.yaml",
        {
            "name": "non_mapping",
            "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp"},
            "simulation": {"platform": "sdk"},
            "peripherals": [{"type": "camera", "name": "front"}, "just a string"],
        },
    )
    with pytest.raises(ValueError, match="peripherals entries require type and name"):
        load_robot_config_dict(config, defer_interface_binding=True)
    # provider-less configurations keep the historical silent drop
    providerless = _write_config(
        tmp_path,
        "providerless.yaml",
        {"name": "providerless", "peripherals": [{"type": "camera", "name": "front"}, "just a string"]},
    )
    loaded = load_robot_config_dict(providerless, defer_interface_binding=True)
    assert [entry["name"] for entry in loaded["peripherals"]] == ["front"]

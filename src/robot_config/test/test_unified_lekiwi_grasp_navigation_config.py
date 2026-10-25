from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.perception import generate_camera_nodes
from robot_config.loader import load_robot_config_dict, robot_context_schema_version, validate_motion_mode_config
from robot_config.perception_runtime_config import parse_perception_runtime_config

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "src/robot_config/config/robots/lekiwi_nav_grasp.yaml"
PROFILE_PATH = ROOT / "src/robots/lekiwi/lekiwi_robot/profiles/lekiwi_nav_grasp.yaml"


@pytest.fixture(autouse=True)
def _front_camera_calibration(tmp_path, monkeypatch):
    """Install an approved front-camera artifact into an isolated HOME.

    The hybrid stage enables semantic mapping, which fails closed when the
    front camera has no mounting transform. Real calibration values are
    deployment-specific; the placeholder keeps the contract loadable while
    ``test_unified_profile_applies_only_the_front_camera_artifact`` still
    verifies the projection with its own explicit artifact.
    """
    artifact = tmp_path / ".ros/ibrobot/calib/current/base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status": "approved",
                "device": {"name": "front_camera", "serial": "043322073551"},
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
    monkeypatch.setenv("HOME", str(tmp_path))


def test_unified_profile_preserves_arm_base_hardware_and_motion_ownership():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["robot"]

    assert config["joints"]["all"] == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # The ros2_control chain and the controller-switch handshake are
    # runtime-owned (openspec lekiwi-application-migration tasks 1.2/3.1):
    # no application restatement is allowed to survive the migration.
    assert "ros2_control" not in config
    assert "motion_mode" not in config
    assert "cmd_vel_bridge" not in config["navigation"]

    peripherals = {item["name"]: item for item in config["peripherals"]}
    # runtime-owned-audio-io: the speaker peripheral replaces the former
    # audio_io playback keys; the runtime launches it as the ALSA owner.
    # runtime-peripheral-overrides: the entry is an override of the
    # profile-declared speaker (the profile owns the driver and defaults to
    # disabled); this deployment opts in with its parameters.
    assert set(peripherals) == {"respeaker", "speaker", "wrist", "front", "mid360"}
    assert peripherals["speaker"] == {
        "type": "speaker",
        "name": "speaker",
        "enabled": True,
        "params": {"device": "", "channels": 1, "sample_rate": 24000, "sample_format": "S16LE"},
    }
    assert peripherals["respeaker"] == {
        "type": "microphone",
        "name": "respeaker",
        "params": {
            "device": "hw:0,0",
            "channels": 6,
            "sample_rate": 16000,
            "sample_format": "S16LE",
            "channel_indices": [1, 2, 3, 4],
        },
    }
    assert peripherals["wrist"]["transform"]["parent_frame"] == "gripper"
    assert peripherals["front"]["transform"]["parent_frame"] == "base_link"
    assert peripherals["front"]["driver_camera_name"] == "front"
    assert peripherals["front"]["align_depth"] is True
    # the drivers are the runtime profile's declarations, not config keys
    from robot_runtime.profile import load_profile

    profile_peripherals = {entry["name"]: entry for entry in load_profile(PROFILE_PATH).get("peripherals") or []}
    assert profile_peripherals["mid360"]["driver"] == "livox_mid360"
    assert profile_peripherals["wrist"]["driver"] == "realsense"
    assert profile_peripherals["front"]["driver"] == "realsense"
    assert profile_peripherals["speaker"]["driver"] == "alsa"
    assert profile_peripherals["speaker"]["enabled"] is False, "the profile default is off"
    assert config["semantic_mapping"]["camera"] == {
        "peripheral": "front",
        "mounting": "fixed",
        "parent_frame": "base_link",
        "rgb_topic": "/camera/front/image_raw",
        "depth_topic": "/camera/front/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/camera/front/camera_info",
    }
    assert config["speech_direction"]["enabled"] is True
    assert config["speech_direction"]["profile"] == "ascend_310p"
    assert config["speech_direction"]["microphone"] == "respeaker"


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_unified_profile_resolves_grasp_mapping_and_navigation_stages():
    hybrid = load_robot_config_dict(CONFIG_PATH)
    grasp = load_robot_config_dict(CONFIG_PATH, nav_stage="grasp")
    mapping = load_robot_config_dict(CONFIG_PATH, nav_stage="mapping")
    navigation = load_robot_config_dict(CONFIG_PATH, nav_stage="navigation")

    assert hybrid["nav_stage"] == "hybrid"
    assert hybrid["default_control_mode"] == "base_navigation"
    assert hybrid["skill_required_control_mode"] == "moveit_planning"
    assert hybrid["embodied"]["skill_catalog_profile"] == "lekiwi_handeye_realsense_grasp_lidar_sound_following"
    assert hybrid["navigation"]["enabled"] is True
    assert hybrid["navigation"]["command_server"]["enabled"] is True
    assert hybrid["mid360_mount_file"].endswith("lekiwi_mid360_mount.yaml")
    assert [item["name"] for item in hybrid["peripherals"]] == ["respeaker", "wrist", "front", "mid360"]
    assert "peripheral_names" not in hybrid
    assert hybrid["grasp_execution"]["ik"]["worker_count"] == 1
    assert hybrid["grasp_execution"]["ik"]["verification_position_tolerance_m"] == 0.001
    assert hybrid["grasp_execution"]["ik"]["verification_orientation_tolerance_deg"] == 1.0
    assert hybrid["control_modes"]["base_navigation"]["runtime_mode"] == "base_navigation"
    assert "controllers" not in hybrid["control_modes"]["base_navigation"]
    assert hybrid["control_modes"]["moveit_planning"]["runtime_mode"] == "trajectory"
    assert "inactive_controllers" not in hybrid["control_modes"]["moveit_planning"]
    # The pre-migration motion_mode handshake is gone; the hybrid stage
    # declares the boot-time admission intent instead (design A3/A5).
    assert "motion_mode" not in hybrid
    assert hybrid["navigation"]["open_gate_on_startup"] is True

    assert grasp["nav_stage"] == "grasp"
    assert grasp["navigation"]["enabled"] is False
    assert [item["name"] for item in grasp["peripherals"]] == ["respeaker", "wrist"]

    assert mapping["nav_stage"] == "mapping"
    assert mapping["default_control_mode"] == "base_navigation"
    assert mapping["navigation"]["fast_lio"]["enabled"] is True
    assert mapping["navigation"]["slam_toolbox"]["enabled"] is True
    assert mapping["navigation"]["nav2_bringup"]["enabled"] is False
    assert "cmd_vel_bridge" not in mapping["navigation"]
    assert mapping["navigation"]["open_gate_on_startup"] is True
    assert mapping["embodied"]["skill_catalog_profile"] == ""
    assert mapping["perception_services"]["services"] == []
    assert mapping["grasp_execution"]["enabled"] is False
    assert mapping["placement_execution"]["enabled"] is False
    assert mapping["contract"]["observations"] == []
    assert mapping["contract"]["actions"] == []
    assert [item["name"] for item in mapping["peripherals"]] == ["respeaker", "front", "mid360"]
    assert "peripheral_names" not in mapping
    assert mapping["control_modes"]["base_navigation"]["runtime_mode"] == "base_navigation"
    assert "controllers" not in mapping["control_modes"]["base_navigation"]

    assert navigation["nav_stage"] == "navigation"
    assert navigation["default_control_mode"] == "base_navigation"
    assert navigation["skill_required_control_mode"] == "base_navigation"
    assert navigation["embodied"]["skill_catalog_profile"] == ""
    assert navigation["perception_services"]["services"] == []
    assert navigation["grasp_execution"]["enabled"] is False
    assert navigation["placement_execution"]["enabled"] is False
    assert navigation["contract"]["observations"] == []
    assert navigation["contract"]["actions"] == []
    assert [item["name"] for item in navigation["peripherals"]] == ["respeaker", "front", "mid360"]
    assert "peripheral_names" not in navigation
    assert navigation["navigation"]["fast_lio"]["enabled"] is True
    assert navigation["navigation"]["slam_toolbox"]["enabled"] is False
    assert navigation["navigation"]["nav2_bringup"]["use_amcl"] is True
    assert "cmd_vel_bridge" not in navigation["navigation"]
    assert navigation["navigation"]["command_server"]["enabled"] is True
    assert navigation["navigation"]["command_server"]["stop_velocity_topic"] == "/cmd_vel"
    assert navigation["navigation"]["open_gate_on_startup"] is True
    assert "motion_mode" not in navigation


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_grasp_stage_uses_matching_v1_manipulation_catalog():
    grasp = load_robot_config_dict(CONFIG_PATH, nav_stage="grasp")

    assert grasp["name"] == "lekiwi_handeye_realsense_grasp"
    assert grasp["embodied"]["skill_catalog_profile"] == "lekiwi_handeye_realsense_grasp"
    assert robot_context_schema_version(grasp) == 1


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_hybrid_keeps_static_semantic_query_and_disables_online_services():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["robot"]
    assert raw["semantic_mapping"]["enabled"] is True
    raw_services = {service["id"]: service for service in raw["perception_services"]["services"]}
    assert all(
        raw_services[service_id]["enabled"]
        for service_id in (
            "semantic_sam2_masks",
            "semantic_ram_plus_tags",
            "semantic_siglip2_image",
            "semantic_siglip2_text",
        )
    )
    assert raw["nav_stages"]["hybrid"]["semantic_mapping"]["query_only"] is True

    hybrid = load_robot_config_dict(CONFIG_PATH)

    assert hybrid["semantic_mapping"]["enabled"] is True
    assert hybrid["semantic_mapping"]["query_only"] is True
    slam = hybrid["semantic_mapping"]["slam"]
    assert slam["global_frame"] == "map"
    assert slam["cloud_map_topic"] == "/cloud_registered_body"
    assert slam["geometry_map_id"] == "fastlio-lekiwi-newbag"
    assert slam["geometry_map_hash"] == "newbag-offline-hash-001"
    assert slam["localization_session_id"] == "newbag-offline-session-001"
    assert slam["calibration_id"] == "newbag-offline-calib-001"
    assert slam["urdf_hash"] == "newbag-offline-urdf-001"
    assert hybrid["semantic_mapping"]["persistence"]["database_path"] == "~/maps/lab-083830/semantic_map.sqlite3"
    roles = hybrid["semantic_mapping"]["perception"]["semantic_roles"]
    assert roles == {
        "sam2_masks": "semantic_sam2_masks",
        "ram_plus_tags": "semantic_ram_plus_tags",
        "siglip2_image": "semantic_siglip2_image",
        "siglip2_text": "semantic_siglip2_text",
    }
    services = {service["id"]: service for service in hybrid["perception_services"]["services"]}
    assert all(services[role]["enabled"] for role in roles.values())
    assert services["semantic_sam2_masks"]["deployment"] == "ascend_310p"
    assert services["semantic_sam2_masks"]["bundle_path"].endswith("sam2.1_hiera_tiny")
    assert services["semantic_siglip2_text"]["required"] is False
    # Endpoints stay disjoint from the grasp services in the same stage.
    endpoints = [service["endpoint"] for service in services.values()]
    assert len(endpoints) == len(set(endpoints))
    # The query-only stage excludes the semantic-role services from the
    # enabled set, so only the grasp pair survives to node selection (the
    # node generator itself lives in inference_service now — task A.5 — and
    # the enabled-service names are this package's SSOT fact).
    runtime = parse_perception_runtime_config(hybrid)
    assert {service.node_name for service in runtime.enabled_services} == {
        "grasp_grounding_detect",
        "grasp_segment_detections",
    }


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_lean_stages_keep_semantic_mapping_disabled():
    for stage in ("grasp", "mapping", "navigation"):
        config = load_robot_config_dict(CONFIG_PATH, nav_stage=stage)
        assert config["semantic_mapping"]["enabled"] is False, stage
        enabled = [
            service["id"] for service in config["perception_services"]["services"] if service.get("enabled", False)
        ]
        assert "semantic_sam2_masks" not in enabled, stage
        assert "semantic_siglip2_image" not in enabled, stage


def test_unified_profile_launches_wrist_and_front_realsense_drivers():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["robot"]

    # The configuration entries are overrides of runtime-declared cameras (no
    # driver keys since the all-provider driver ban); the realsense drivers
    # are the lekiwi_nav_grasp profile's declarations, so the merged set —
    # what the runtime launches — carries them.
    from robot_runtime.interface_description import merge_peripherals
    from robot_runtime.profile import load_profile

    profile = load_profile(PROFILE_PATH)
    merged = merge_peripherals(profile.get("peripherals"), config.get("peripherals"))
    by_name = {entry["name"]: entry for entry in merged if entry.get("type") == "camera"}
    assert {name for name, entry in by_name.items() if entry.get("driver") == "realsense"} == {"wrist", "front"}
    # ... and the driver-composed nodes the runtime generates from the merged
    # set name exactly those two cameras
    drivers = [
        node
        for node in generate_camera_nodes({"peripherals": merged}, use_sim=False)
        if vars(node).get("_Node__package") == "realsense2_camera"
    ]
    assert {vars(node).get("_Node__node_name") for node in drivers} == {"wrist_camera", "front"}


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_hybrid_startup_gate_is_rejected_as_runtime_owned():
    """The hybrid startup-match rule went with navigation_enabled_on_startup.

    Under a provider the loader rejects the field outright (design A5): the
    stage declares navigation.open_gate_on_startup and a launch-time admission
    step opens the gate through the contract service after the runtime is
    ready.
    """
    hybrid = load_robot_config_dict(CONFIG_PATH)
    assert "motion_mode" not in hybrid

    restated = deepcopy(hybrid)
    restated["motion_mode"] = {"navigation_enabled_on_startup": True}
    errors = validate_motion_mode_config(restated)
    assert any(
        "motion_mode.navigation_enabled_on_startup is runtime-owned" in error
        and "navigation.open_gate_on_startup" in error
        for error in errors
    )


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_unified_profile_applies_only_the_front_camera_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / ".ros/ibrobot/calib/current/base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status": "approved",
                "device": {"name": "front_camera", "serial": "043322073551"},
                "transform": {
                    "parent_frame": "base_link",
                    "child_frame": "camera_front_optical_frame",
                    "translation": [0.12, -0.03, 0.42],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    hybrid = load_robot_config_dict(CONFIG_PATH)
    front = next(item for item in hybrid["peripherals"] if item["name"] == "front")
    wrist = next(item for item in hybrid["peripherals"] if item["name"] == "wrist")
    assert [front["transform"][key] for key in ("x", "y", "z")] == [0.12, -0.03, 0.42]
    assert wrist["transform"]["parent_frame"] == "gripper"

    grasp = load_robot_config_dict(CONFIG_PATH, nav_stage="grasp")
    assert [item["name"] for item in grasp["peripherals"]] == ["respeaker", "wrist"]

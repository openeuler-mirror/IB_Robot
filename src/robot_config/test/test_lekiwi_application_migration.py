"""Configuration migration tests (openspec lekiwi-application-migration 1.6).

Pins the application-half migration on the three migrated configurations:

- provider/profile identity and the require_model choice per configuration;
- runtime-owned data (ros2_control, controller lists, the motion_mode
  handshake, bridge geometry) is absent and, when restated, rejected by the
  loader naming the field (design A2/A5);
- logical bindings survive: wheel-joint groups match the runtime profile,
  stages declare boot-time gate intent, application policy stays put;
- overlays inherit the migrated parent coherently with no silently ignored
  legacy keys;
- the six provider-less configurations 1.5 records as unsupported stay
  loadable fixtures without a bridge block.
"""

from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.runtime import _write_peripherals_fragment
from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

ROOT = Path(__file__).resolve().parents[3]
ROBOTS = ROOT / "src/robot_config/config/robots"
PROFILES = ROOT / "src/robots/lekiwi/lekiwi_robot/profiles"

MIGRATED = ("lekiwi_lidar", "lekiwi_nav_grasp", "lekiwi_sensor_calib")
UNSUPPORTED = (
    "lekiwi_handeye_realsense_grasp",
    "lekiwi_handeye_realsense_grasp_pc",
    "lekiwi_realsense_mapping",
    "lekiwi_realsense_navigation",
    "lekiwi_rtp_distributed",
    "lekiwi_semantic_capture",
)


def _load(name, **kwargs):
    return load_robot_config_dict(ROBOTS / f"{name}.yaml", defer_interface_binding=True, **kwargs)


def _raw(name):
    return yaml.safe_load((ROBOTS / f"{name}.yaml").read_text(encoding="utf-8"))["robot"]


# --- provider identity -------------------------------------------------------


@pytest.mark.parametrize("name", MIGRATED)
def test_migrated_configurations_select_the_lekiwi_runtime(name):
    # nav_grasp's default (hybrid) stage references model bundles; the mapping
    # stage carries the same runtime selection without that dependency.
    config = _load(name, nav_stage="mapping") if name == "lekiwi_nav_grasp" else _load(name)
    assert config["runtime"]["provider"] == "lekiwi_robot"
    assert config["runtime"]["profile"] in {"lekiwi_lidar", "lekiwi_nav_grasp"}


def test_no_migrated_configuration_uses_require_model():
    """require_model does not fit the stage-based LeKiwi configurations.

    nav_grasp stages select sensor subsets through peripheral_names and
    sensor_calib adds a deployment camera -- both need the D12 peripherals
    fragment that require_model forbids. lekiwi_lidar additionally stays
    loadable by offline tools (recording, capture, catalog) that have no live
    runtime to bind against; require_model would force deferred live binding
    on every plain load. The loader's provider-conditional rejections and the
    profile's ownership of every hardware fact carry the A2 guarantee instead.
    """
    for name in MIGRATED:
        assert "require_model" not in _raw(name)["runtime"], name


@pytest.mark.parametrize("name", MIGRATED)
def test_migrated_configurations_restate_no_runtime_owned_data(name):
    config = _load(name, nav_stage="mapping") if name == "lekiwi_nav_grasp" else _load(name)
    assert "ros2_control" not in config
    assert "motion_mode" not in config
    assert "cmd_vel_bridge" not in config["navigation"]
    for mode_name, mode in config.get("control_modes", {}).items():
        assert "controllers" not in mode, mode_name
        assert "inactive_controllers" not in mode, mode_name


# --- runtime-owned fields are rejected, not merged (A2/A5) -------------------


def _rejections(config):
    try:
        validate_robot_config_dict(config)
    except ValueError as exc:
        return str(exc)
    return ""


def _probe(name, mutate):
    config = _load(name, nav_stage="mapping") if name == "lekiwi_nav_grasp" else _load(name)
    mutate(config)
    return _rejections(config)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        (
            "motion_mode.controller_switch_service",
            lambda c: c.update({"motion_mode": {"controller_switch_service": "controller_manager/switch_controller"}}),
        ),
        (
            "motion_mode.manipulation_controllers",
            lambda c: c.update({"motion_mode": {"manipulation_controllers": ["arm_trajectory_controller"]}}),
        ),
        (
            "motion_mode.navigation_controllers",
            lambda c: c.update({"motion_mode": {"navigation_controllers": ["base_velocity_controller"]}}),
        ),
        (
            "motion_mode.navigation_enabled_on_startup",
            lambda c: c.update({"motion_mode": {"navigation_enabled_on_startup": True}}),
        ),
        (
            "motion_mode.set_navigation_enabled_service",
            lambda c: c.update(
                {"motion_mode": {"set_navigation_enabled_service": "/motion_mode/set_navigation_enabled"}}
            ),
        ),
        (
            "navigation.cmd_vel_bridge",
            lambda c: c["navigation"].update({"cmd_vel_bridge": {"enabled": True, "wheel_radius": 0.05}}),
        ),
    ],
)
@pytest.mark.parametrize("name", MIGRATED)
def test_runtime_owned_fields_fail_loading_naming_the_field(name, field, mutate):
    report = _probe(name, mutate)
    assert field in report, report
    assert "runtime-owned" in report


def test_provider_less_motion_mode_still_validates_for_legacy_configs():
    # The relative-name handshake rule stays for provider-less configurations
    # (the unsupported handeye pair still declares it); only provider-bound
    # configurations are rejected from restating it at all.
    motion_mode = _raw("lekiwi_handeye_realsense_grasp")["motion_mode"]
    assert motion_mode["controller_switch_service"] == "controller_manager/switch_controller"
    assert not motion_mode["controller_switch_service"].startswith("/")


# --- logical bindings --------------------------------------------------------


@pytest.mark.parametrize("name", MIGRATED)
def test_wheel_joint_groups_match_the_runtime_profile(name):
    config = _load(name, nav_stage="mapping") if name == "lekiwi_nav_grasp" else _load(name)
    profile_name = config["runtime"]["profile"]
    profile = yaml.safe_load((PROFILES / f"{profile_name}.yaml").read_text(encoding="utf-8"))
    wheel_joints = [str(joint) for joint in profile["base"]["wheel_joints"]]

    assert config["joints"]["base"] == wheel_joints


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_control_modes_select_runtime_modes():
    lidar = _load("lekiwi_lidar")
    assert {mode: spec["runtime_mode"] for mode, spec in lidar["control_modes"].items()} == {
        "teleop": "base_navigation",
        "base_navigation": "base_navigation",
    }
    grasp = _load("lekiwi_nav_grasp", nav_stage="grasp")
    assert {mode: spec["runtime_mode"] for mode, spec in grasp["control_modes"].items()} == {
        "teleop": "stream",
        "moveit_planning": "trajectory",
        "base_navigation": "base_navigation",
    }


@pytest.mark.parametrize(
    ("name", "stage"),
    [
        ("lekiwi_lidar", "mapping"),
        ("lekiwi_lidar", "navigation"),
        ("lekiwi_nav_grasp", "mapping"),
        ("lekiwi_nav_grasp", "navigation"),
        ("lekiwi_sensor_calib", None),
    ],
)
def test_base_driving_stages_declare_boot_gate_intent(name, stage):
    config = _load(name) if stage is None else _load(name, nav_stage=stage)
    assert config["navigation"]["open_gate_on_startup"] is True


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("yolox_x_640")
def test_hybrid_stage_declares_boot_gate_intent():
    assert _load("lekiwi_nav_grasp", nav_stage="hybrid")["navigation"]["open_gate_on_startup"] is True


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_grasp_stage_declares_no_boot_gate_intent():
    grasp = _load("lekiwi_nav_grasp", nav_stage="grasp")
    assert grasp["navigation"]["enabled"] is False
    assert "open_gate_on_startup" not in grasp["navigation"]


def test_deployment_fragment_carries_sensor_inventory_and_fast_lio_policy():
    calib = _load("lekiwi_sensor_calib")
    fragment_path = _write_peripherals_fragment(calib)
    try:
        fragment = yaml.safe_load(Path(fragment_path).read_text(encoding="utf-8"))
    finally:
        Path(fragment_path).unlink(missing_ok=True)

    # the front camera override plus the mount-created partial MID-360 entry
    # (lekiwi-gazebo-runtime-platform D13) both reach the runtime fragment
    assert [item["name"] for item in fragment["peripherals"]] == ["front", "mid360"]
    assert fragment["fast_lio"]["enabled"] is True


# --- application policy survives ----------------------------------------------


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("yolox_x_640")
def test_nav_grasp_keeps_stage_policy_and_deployment_inventory():
    hybrid = _load("lekiwi_nav_grasp", nav_stage="hybrid")
    # The speaker joined every stage list when the runtime took ownership of
    # audio devices (upstream runtime-owned-audio-io); the stage subsets are
    # otherwise unchanged by the migration.
    assert [item["name"] for item in hybrid["peripherals"]] == ["respeaker", "speaker", "wrist", "front", "mid360"]
    assert hybrid["semantic_mapping"]["query_only"] is True
    assert hybrid["grasp_execution"]["enabled"] is True
    assert hybrid["skill_required_control_mode"] == "moveit_planning"
    assert hybrid["default_control_mode"] == "base_navigation"
    # The provider launch hands the stage sensor subset to the runtime.
    assert "peripheral_names" not in hybrid

    grasp = _load("lekiwi_nav_grasp", nav_stage="grasp")
    assert [item["name"] for item in grasp["peripherals"]] == ["respeaker", "speaker", "wrist"]


def test_navigation_stages_keep_stop_velocity_on_the_admitted_channel():
    for name in ("lekiwi_lidar", "lekiwi_nav_grasp"):
        navigation = _load(name, nav_stage="navigation")["navigation"]
        assert navigation["command_server"]["stop_velocity_topic"] == "/cmd_vel", name


# --- overlays resolve coherently (1.3) ----------------------------------------


@pytest.mark.parametrize(
    ("name", "stage"),
    [
        ("lekiwi_hybrid_switch_test", "hybrid"),
        ("lekiwi_nav_grasp_sound_real", "navigation"),
        ("lekiwi_nav_grasp_sound_real", "hybrid"),
    ],
)
def test_overlays_inherit_the_migrated_parent(name, stage):
    config = _load(name, nav_stage=stage)
    assert config["runtime"]["provider"] == "lekiwi_robot"
    assert config["runtime"]["profile"] == "lekiwi_nav_grasp"
    # No legacy key survived the merge only to be ignored.
    assert "motion_mode" not in config
    assert "cmd_vel_bridge" not in config["navigation"]
    assert "ros2_control" not in config
    for mode in config["control_modes"].values():
        assert "controllers" not in mode
    # The parent's boot gate intent flows through the overlay.
    assert config["navigation"]["open_gate_on_startup"] is True


def test_hybrid_switch_test_overlay_overrides_stage_policy():
    config = _load("lekiwi_hybrid_switch_test", nav_stage="hybrid")
    assert config["perception_services"]["enabled"] is False
    assert config["grasp_execution"]["enabled"] is False
    assert config["embodied"]["skill_catalog_profile"] == "lekiwi_hybrid_switch_test"
    # The overlay narrows the stage to the lidar; respeaker/speaker stay in
    # the list only as disabled overrides so the runtime keeps them off
    # (runtime-owned-audio-io: a base_config overlay replaces the whole
    # peripherals list).
    peripherals = {item["name"]: item for item in config["peripherals"]}
    assert list(peripherals) == ["respeaker", "speaker", "mid360"]
    assert peripherals["respeaker"]["enabled"] is False
    assert peripherals["speaker"]["enabled"] is False


# --- the unsupported six stay loadable fixtures (A5) ---------------------------


@pytest.mark.parametrize(
    "name",
    ["lekiwi_realsense_mapping", "lekiwi_realsense_navigation", "lekiwi_rtp_distributed", "lekiwi_semantic_capture"],
)
def test_unsupported_configurations_remain_loadable_without_a_bridge(name):
    config = _load(name)
    assert "runtime" not in config
    assert "cmd_vel_bridge" not in config["navigation"]
    assert (ROBOTS / f"{name}.yaml").read_text(encoding="utf-8").startswith("#")


@pytest.mark.parametrize("name", ["lekiwi_handeye_realsense_grasp", "lekiwi_handeye_realsense_grasp_pc"])
def test_bundle_gated_unsupported_configurations_drop_the_bridge_block(name):
    # These two only fail loading on missing model bundles (the known
    # mainline condition); the bridge block removal is checked on the raw
    # document so the fixture shape is still pinned.
    raw = _raw(name)
    assert "runtime" not in raw
    assert "cmd_vel_bridge" not in raw["navigation"]


# --- task executor derivation (lekiwi-application-migration task 3.1) ----------
# The hybrid-derivation tests moved to task_dispatch's builder suite with
# generate_task_executor_node (upstream ``migrate-business-launch-builders``):
# task_dispatch owns the builder and already depends on robot_config, so the
# tests that pin its behavior follow the module to avoid a test-depend cycle.


# --- regression guard: no config may restate bridge geometry --------------------


@pytest.mark.parametrize(
    "name",
    sorted(path.stem for path in ROBOTS.glob("lekiwi*.yaml")),
)
def test_no_lekiwi_configuration_restates_bridge_geometry(name):
    navigation = _raw(name).get("navigation", {})
    assert "cmd_vel_bridge" not in navigation, name

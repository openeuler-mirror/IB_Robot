from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict

CONFIG = Path(__file__).parent.parent / "config" / "robots" / "lekiwi_handeye_realsense_grasp.yaml"
PC_CONFIG = Path(__file__).parent.parent / "config" / "robots" / "lekiwi_handeye_realsense_grasp_pc.yaml"


def _write(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutate(data["robot"]["placement_execution"])
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720", "grounded_sam2_swint_ogc")
def test_repository_placement_contract_loads():
    config = load_robot_config_dict(CONFIG)
    placement = config["placement_execution"]
    assert placement["executor"] == "placement_pipeline"
    assert placement["required_args"] == ["target_name", "container_name"]
    assert "container_query" not in placement["verification"]
    assert placement["verification"]["required"] is True
    assert placement["verification"]["required_confirmations"] == 2
    assert placement["verification"]["target_exclusion"] == {
        "enabled": True,
        "mask_path": "$(find robot_config)/config/placement_masks/so101_wrist_gripper_sam_mask.png",
        "min_detection_overlap": 0.35,
        "polygons": [],
    }
    assert placement["detect_service"] == "/perception/grasp/grounding_detect"
    assert placement["segment_service"] == "/perception/grasp/segment_detections"
    assert placement["rgb_topic"] == "/camera/wrist/image_raw"
    assert placement["debug_output_root"] == "$(env WORKSPACE)/outputs/placement_pipeline"
    assert placement["motion"] == {
        "place_pose": "place_container",
        "place_joint_names": ["1", "2", "3", "4", "5"],
        "place_joint_positions": {
            "1": -0.047553,
            "2": -0.073631,
            "3": -0.840621,
            "4": 1.497165,
            "5": -1.570790,
        },
        "place_duration_sec": 5.0,
        "post_release": {
            "verify_joint_name": "3",
            "verify_joint_position": -0.533825,
            "verify_duration_sec": 2.0,
            "return_duration_sec": 2.0,
        },
    }
    assert "ik_service" not in placement
    assert "camera" not in placement
    assert config["embodied"]["skill_catalog_profile"] == "lekiwi_handeye_realsense_grasp"


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720", "grounded_sam2_swint_ogc")
def test_pc_placement_contract_uses_grounding_masks_without_a_separate_segmenter():
    placement = load_robot_config_dict(PC_CONFIG)["placement_execution"]

    assert placement["detect_service"] == "/perception/grasp/grounding_detect"
    assert placement["segment_service"] == ""
    assert placement["rgb_topic"] == "/camera/wrist/image_raw"
    assert placement["debug_output_root"] == "$(env WORKSPACE)/outputs/placement_pipeline"
    assert placement["motion"]["place_joint_positions"] == {
        "1": -0.047553,
        "2": -0.073631,
        "3": -0.840621,
        "4": 1.497165,
        "5": -1.570790,
    }
    assert placement["motion"]["post_release"] == {
        "verify_joint_name": "3",
        "verify_joint_position": -0.533825,
        "verify_duration_sec": 2.0,
        "return_duration_sec": 2.0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["verification"].update(required=False), "verification.required"),
        (lambda cfg: cfg.update(debug_output_root=""), "debug_output_root"),
        (
            lambda cfg: cfg["sensor"].update(gripper_feedback_timeout_sec=0.0),
            "gripper_feedback_timeout_sec",
        ),
        (lambda cfg: cfg["verification"].update(min_inside_mask_fraction=1.1), "min_inside_mask_fraction"),
        (lambda cfg: cfg["verification"].update(container_inset_ratio=0.5), "container_inset_ratio"),
        (lambda cfg: cfg["verification"].update(required_confirmations=5), "required_confirmations"),
        (lambda cfg: cfg["motion"].update(place_duration_sec=0.0), "place_duration_sec"),
        (
            lambda cfg: cfg["motion"]["post_release"].update(verify_joint_name="6"),
            "verify_joint_name must be listed in place_joint_names",
        ),
        (
            lambda cfg: cfg["motion"]["post_release"].update(verify_joint_position=float("nan")),
            "verify_joint_position must be finite",
        ),
        (
            lambda cfg: cfg["motion"]["post_release"].update(return_duration_sec=0.0),
            "return_duration_sec must be finite and positive",
        ),
        (
            lambda cfg: cfg["motion"].update(place_joint_names=["1", "2", "3", "4", "4"]),
            "must not contain duplicates",
        ),
        (
            lambda cfg: cfg["motion"]["place_joint_positions"].pop("5"),
            "keys must exactly match place_joint_names",
        ),
        (
            lambda cfg: cfg["motion"]["place_joint_positions"].update({"3": float("nan")}),
            r"place_joint_positions\.3 must be finite",
        ),
        (lambda cfg: cfg.update(plan_service="/plan_kinematic_path"), "unsupported key"),
    ],
)
def test_invalid_placement_contract_fails_before_launch(tmp_path, mutation, message):
    with pytest.raises(ValueError, match=message):
        load_robot_config_dict(_write(tmp_path, mutation))

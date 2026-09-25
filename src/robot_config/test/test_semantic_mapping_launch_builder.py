"""Tests for the robot_config semantic mapping launch builder."""

from pathlib import Path

import pytest

from robot_config.launch_builders.semantic_mapping import generate_semantic_mapping_nodes

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "src/robot_config/config/robots/lekiwi_nav_grasp.yaml"

_NODE_FIELDS = {
    "package": "_Node__package",
    "executable": "_Node__node_executable",
    "name": "_Node__node_name",
    "parameters": "_Node__parameters",
}


def _node_parameters(node) -> dict:
    parameters = vars(node)[_NODE_FIELDS["parameters"]][0]
    normalized = {}
    for key, value in parameters.items():
        normalized_key = "".join(getattr(part, "text", str(part)) for part in key)
        if isinstance(value, tuple):
            normalized_value = "".join(getattr(part, "text", str(part)) for part in value).removesuffix("\n...\n")
            normalized_value = normalized_value.strip("'\"\n")
        else:
            normalized_value = value
        normalized[normalized_key] = normalized_value
    return normalized


def _require_calibration(monkeypatch, tmp_path, translation=None):
    import yaml

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
                    "translation": translation or [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))


def test_disabled_config_starts_no_mapping_node():
    from robot_config.loader import load_robot_config_dict

    config = load_robot_config_dict(CONFIG_PATH, nav_stage="navigation")

    assert generate_semantic_mapping_nodes(config) == []


# Drives a robot config whose perception services reference a gitignored model
# bundle; without scripts/download_models.py the config cannot load. A missing
# download is not a defect, so skip rather than fail. See root conftest.py.
@pytest.mark.model_bundle("grounding_dino_swint_seq8_1280x720")
def test_hybrid_config_starts_query_only_mapping_node(monkeypatch, tmp_path):
    from robot_config.loader import load_robot_config_dict

    _require_calibration(monkeypatch, tmp_path)
    config = load_robot_config_dict(CONFIG_PATH)

    nodes = generate_semantic_mapping_nodes(config)

    assert len(nodes) == 1
    node = nodes[0]
    assert vars(node)[_NODE_FIELDS["package"]] == "semantic_mapping"
    assert vars(node)[_NODE_FIELDS["executable"]] == "semantic_mapping_node"
    assert vars(node)[_NODE_FIELDS["name"]] == "semantic_mapping"
    parameters = _node_parameters(node)
    assert parameters["mapping_backend"] == "service"
    assert parameters["online_processing_enabled"] is False
    assert parameters["cloud_map_topic"] == "/cloud_registered_body"
    assert parameters["global_frame"] == "map"
    assert parameters["geometry_map_hash"] == "newbag-offline-hash-001"
    assert parameters["database_path"] == "~/maps/lab-083830/semantic_map.sqlite3"
    assert parameters["database_path"].endswith("semantic_map.sqlite3")

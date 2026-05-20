import json
from unittest.mock import patch

import pytest

from model_utils.export_onnx_atc import _act_input_shape, convert_onnx_to_om, write_om_manifest


def test_write_om_manifest_records_policy_artifact(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    om_path = policy_dir / "model.om"
    om_path.write_bytes(b"om")

    manifest_path = write_om_manifest(str(policy_dir), {"type": "act"}, str(om_path))

    assert manifest_path == policy_dir / "config.om.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "policy_type": "act",
        "backend": "ascend_om",
        "artifacts": {"policy": "model.om"},
        "execution": ["policy"],
    }


def test_write_om_manifest_rejects_non_act_policy(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        write_om_manifest(str(tmp_path), {"type": "pi05"}, str(tmp_path / "model.om"))


def test_act_input_shape_only_includes_runtime_inputs():
    config = {
        "input_features": {
            "observation.state": {"shape": [6]},
            "observation.state.current": {"shape": [6]},
            "observation.images.front": {"shape": [3, 240, 320]},
        }
    }

    assert _act_input_shape(config) == "observation.state:1,6;observation.images.front:1,3,240,320"


def test_convert_onnx_to_om_uses_config_input_shape():
    with patch("model_utils.export_onnx_atc.subprocess.run") as run:
        run.return_value.returncode = 0

        assert convert_onnx_to_om(
            {"input_features": {"observation.state": {"shape": [6]}}},
            "/tmp/model.onnx",
            "/tmp/model.om",
            "Ascend310B1",
        )

    run.assert_called_once_with(
        [
            "atc",
            "--framework=5",
            "--soc_version=Ascend310B1",
            "--model=/tmp/model.onnx",
            "--output=/tmp/model",
            "--input_shape=observation.state:1,6",
        ],
        check=False,
    )

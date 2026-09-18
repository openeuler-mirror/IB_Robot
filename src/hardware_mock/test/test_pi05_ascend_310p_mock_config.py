from pathlib import Path

import pytest

from hardware_mock.contract_plan import build_plan
from robot_config.inference_config import parse_inference_config
from robot_config.loader import load_robot_config_dict

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "robot_config" / "config" / "robots" / "so101_pi05_ascend_310p_mock.yaml"
)


def test_pi05_ascend_310p_mock_contract_plan() -> None:
    robot = load_robot_config_dict(CONFIG_PATH)

    assert robot["simulation"]["platform"] == "mock"
    plan = build_plan(robot)
    assert plan.joint_state_rate_hz == 20.0
    assert {observation.topic for observation in plan.observations} == {
        "/camera/top/image_raw",
        "/camera/wrist/image_raw",
        "/joint_states",
    }
    assert [action.topic for action in plan.actions] == ["/joint_commands"]


def test_pi05_ascend_310p_mock_pipeline(monkeypatch) -> None:
    workspace = CONFIG_PATH.parents[4]
    bundle = workspace / "models" / "pi05" / "pi05-doublecam-fp32" / "019200-torch-npu"
    if not bundle.is_dir():
        pytest.skip(f"PI0.5 Torch-NPU bundle is not installed at {bundle}")
    monkeypatch.setenv("WORKSPACE", str(workspace))
    config = load_robot_config_dict(CONFIG_PATH)
    inference = parse_inference_config(config, "model_inference")
    pipeline = inference.pipelines["policy"]

    assert pipeline.deployment == "torch-npu"
    assert pipeline.execution_mode == "monolithic"
    assert pipeline.runtime_options == {"model_dtype": "fp16"}
    assert pipeline.request_timeout == 1800.0
    assert pipeline.validated_manifest.manifest.model.architecture_class == "pi05-ascend-310p"

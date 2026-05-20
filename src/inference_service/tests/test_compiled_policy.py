from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from inference_service.core.compiled_policy import (
    ACTCompiledAdapter,
    CompiledPolicyWrapper,
    OMRuntimeSession,
    PI05CompiledAdapter,
    PI05OMRuntimeSession,
    PI05RuntimeInputs,
    SD3403RuntimeSession,
    create_compiled_model_adapter,
    create_runtime_session,
    load_compiled_manifest,
    resolve_om_model_path,
    resolve_pi05_om_paths,
)


class FakeRuntimeSession:
    def __init__(self, output=None):
        self.output = output if output is not None else np.zeros((1, 2, 6), dtype=np.float32)
        self.loaded = None
        self.inputs = None

    def load(self, policy_path, config, device):
        self.loaded = (policy_path, config, device)

    def execute(self, inputs):
        self.inputs = inputs
        return [self.output]

    def release(self):
        pass


def _act_config(**updates):
    config = {
        "type": "act",
        "chunk_size": 2,
        "input_features": {
            "observation.state": {"shape": [3]},
            "observation.images.side": {"shape": [3, 4, 5]},
            "observation.images.gripper": {"shape": [3, 4, 5]},
        },
        "output_features": {"action": {"shape": [6]}},
    }
    config.update(updates)
    return config


def _write_policy(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _write_manifest(tmp_path, manifest):
    (tmp_path / "config.om.json").write_text(json.dumps(manifest), encoding="utf-8")


def _pi05_config(**updates):
    config = {
        "type": "pi05",
        "chunk_size": 50,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.language.tokens": {"shape": [48]},
            "observation.language.attention_mask": {"shape": [48]},
        },
        "output_features": {"action": {"shape": [6]}},
    }
    config.update(updates)
    return config


def test_adapter_selection_from_config_type():
    adapter = create_compiled_model_adapter(_act_config(), "rknn")

    assert isinstance(adapter, ACTCompiledAdapter)
    assert adapter.policy_type == "act"
    assert adapter.uses_action_chunking is True

    pi05_adapter = create_compiled_model_adapter(_pi05_config(), "ascend_om")
    assert isinstance(pi05_adapter, PI05CompiledAdapter)
    assert pi05_adapter.policy_type == "pi05"
    assert pi05_adapter.uses_action_chunking is True


def test_adapter_rejects_missing_and_unsupported_type():
    with pytest.raises(ValueError, match="missing required type"):
        create_compiled_model_adapter({"input_features": {}}, "rknn")

    with pytest.raises(ValueError, match="does not support policy type"):
        create_compiled_model_adapter({"type": "diffusion"}, "rknn")

    with pytest.raises(ValueError, match="does not support PI05"):
        create_compiled_model_adapter(_pi05_config(), "rknn")


def test_act_input_mapping_uses_declared_order_and_camera_names():
    adapter = ACTCompiledAdapter.from_config(_act_config(), "rknn")

    inputs = adapter.prepare_inputs(
        {
            "observation.state": torch.full((3,), 1.0),
            "observation.images.side": torch.full((3, 4, 5), 2.0),
            "observation.images.gripper": torch.full((3, 4, 5), 3.0),
        }
    )

    assert [arr.shape for arr in inputs] == [(1, 3), (1, 3, 4, 5), (1, 3, 4, 5)]
    assert float(inputs[0][0, 0]) == 1.0
    assert float(inputs[1][0, 0, 0, 0]) == 2.0
    assert float(inputs[2][0, 0, 0, 0]) == 3.0


def test_act_input_mapping_rejects_missing_tensor():
    adapter = ACTCompiledAdapter.from_config(_act_config(), "rknn")

    with pytest.raises(KeyError, match="observation.images.gripper"):
        adapter.prepare_inputs(
            {
                "observation.state": torch.ones(3),
                "observation.images.side": torch.ones(3, 4, 5),
            }
        )


def test_act_decodes_om_action_chunk():
    adapter = ACTCompiledAdapter.from_config(_act_config(), "ascend_om")
    action = adapter.decode_outputs([np.arange(12, dtype=np.float32)], torch.device("cpu"))

    assert action.shape == (2, 6)


def test_act_decodes_sd3403_crop_and_updates_chunk_size():
    adapter = ACTCompiledAdapter.from_config(_act_config(chunk_size=1), "ascend_om_3403")
    action = adapter.decode_outputs([np.arange(16, dtype=np.float32)], torch.device("cpu"))

    assert action.shape == (2, 6)
    assert adapter.get_chunk_size() == 2


def test_pi05_adapter_prepares_runtime_inputs_and_slices_padding():
    adapter = PI05CompiledAdapter.from_config(_pi05_config(), "ascend_om")

    inputs = adapter.prepare_inputs(
        {
            "observation.images.front": torch.full((1, 3, 224, 224), 1.0),
            "observation.language.tokens": torch.arange(48).reshape(1, 48),
            "observation.language.attention_mask": torch.ones(1, 48, dtype=torch.bool),
            "_noise": torch.zeros(1, 50, 32),
        }
    )

    assert isinstance(inputs, PI05RuntimeInputs)
    assert inputs.images[0].shape == (1, 3, 224, 224)
    assert inputs.tokens.dtype == np.int64
    assert inputs.masks.dtype == np.bool_
    assert inputs.noise.shape == (1, 50, 32)

    action = adapter.decode_outputs(torch.zeros(1, 50, 32), torch.device("cpu"))

    assert action.shape == (50, 6)
    assert adapter.get_chunk_size() == 50


def test_compiled_wrapper_reports_metadata_and_runtime_device(tmp_path):
    _write_policy(tmp_path, _act_config(input_features={"observation.state": {"shape": [3]}}))
    runtime = FakeRuntimeSession(output=np.arange(12, dtype=np.float32))
    wrapper = CompiledPolicyWrapper("rknn", runtime_session=runtime)
    device = torch.device("cpu")

    wrapper.load(str(tmp_path), device)
    action = wrapper.infer({"observation.state": torch.ones(3)})

    assert runtime.loaded[0] == str(tmp_path)
    assert runtime.loaded[1]["type"] == "act"
    assert runtime.loaded[2] == device
    assert runtime.inputs[0].shape == (1, 3)
    assert action.shape == (2, 6)
    assert wrapper.policy_type == "act"
    assert wrapper.backend_type == "rknn"
    assert wrapper.uses_action_chunking is True


def test_compiled_wrapper_requires_config_json(tmp_path):
    wrapper = CompiledPolicyWrapper("rknn", runtime_session=FakeRuntimeSession())

    with pytest.raises(FileNotFoundError, match="config.json"):
        wrapper.load(str(tmp_path), torch.device("cpu"))


def test_runtime_dependencies_import_lazily():
    before = set(sys.modules)

    OMRuntimeSession()
    SD3403RuntimeSession()
    PI05OMRuntimeSession()

    imported = set(sys.modules) - before
    assert "acl" not in imported
    assert "rknnlite.api" not in imported


def test_ascend_runtime_session_selects_pi05_from_config():
    assert isinstance(create_runtime_session("ascend_om", _act_config()), OMRuntimeSession)
    assert isinstance(create_runtime_session("ascend_om", _pi05_config()), PI05OMRuntimeSession)


def test_manifest_resolves_single_om_policy_role(tmp_path):
    model = tmp_path / "om" / "act.om"
    model.parent.mkdir()
    model.write_bytes(b"om")
    _write_policy(tmp_path, _act_config())
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om",
            "artifact_dir": "om",
            "artifacts": {"policy": "act.om"},
            "execution": ["policy"],
        },
    )

    manifest = load_compiled_manifest(str(tmp_path), "ascend_om", "act")

    assert resolve_om_model_path(str(tmp_path), _act_config(), manifest) == model.resolve()


def test_manifest_resolves_pi05_roles_and_execution(tmp_path):
    vlm = tmp_path / "om" / "vlm.om"
    ae = tmp_path / "om" / "action_expert.om"
    vlm.parent.mkdir()
    vlm.write_bytes(b"vlm")
    ae.write_bytes(b"ae")
    _write_policy(tmp_path, _pi05_config())
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "pi05",
            "backend": "ascend_om",
            "artifact_dir": "om",
            "artifacts": {
                "vlm": "vlm.om",
                "action_expert": "action_expert.om",
            },
            "execution": ["vlm", "action_expert"],
        },
    )

    manifest = load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")

    assert resolve_pi05_om_paths(str(tmp_path), _pi05_config(), manifest) == (vlm.resolve(), ae.resolve())


def test_manifest_rejects_wrong_backend_policy_and_execution(tmp_path):
    model = tmp_path / "model.om"
    model.write_bytes(b"om")
    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "pi05",
            "backend": "rknn",
            "artifacts": {"policy": "model.om"},
        },
    )
    with pytest.raises(ValueError, match="does not match requested backend"):
        load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")

    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "act",
            "backend": "ascend_om",
            "artifacts": {"policy": "model.om"},
        },
    )
    with pytest.raises(ValueError, match="does not match config type"):
        load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")

    _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "policy_type": "pi05",
            "backend": "ascend_om",
            "artifacts": {"vlm": "model.om", "action_expert": "model.om"},
            "execution": ["action_expert", "vlm"],
        },
    )
    manifest = load_compiled_manifest(str(tmp_path), "ascend_om", "pi05")
    with pytest.raises(ValueError, match="execution must be"):
        resolve_pi05_om_paths(str(tmp_path), _pi05_config(), manifest)


def test_manifest_required_for_om_resolution(tmp_path):
    _write_policy(tmp_path, _act_config())

    with pytest.raises(FileNotFoundError, match="config.om.json"):
        resolve_om_model_path(str(tmp_path), _act_config())


def test_pi05_runtime_builds_prefix_mask_and_forwards():
    class FakeModel:
        prefix_seq_len = 52

        def __init__(self):
            self.forward_args = None

        def forward(self, images, tokens, masks, prefix_mask, noise=None):
            self.forward_args = (images, tokens, masks, prefix_mask, noise)
            return torch.zeros(1, 50, 32)

    session = PI05OMRuntimeSession()
    session._model = FakeModel()
    inputs = PI05RuntimeInputs(
        images=[np.ones((1, 3, 224, 224), dtype=np.float32)],
        tokens=np.ones((1, 48), dtype=np.int64),
        masks=np.ones((1, 48), dtype=np.bool_),
        noise=np.zeros((1, 50, 32), dtype=np.float32),
    )

    output = session.execute(inputs)

    _, _, _, prefix_mask, noise = session._model.forward_args
    assert output.shape == (1, 50, 32)
    assert prefix_mask.shape == (1, 1, 52, 52)
    assert noise is inputs.noise


def test_sd3403_runtime_uses_worker_public_array_api():
    class FakeWorker:
        def __init__(self):
            self.inputs = None
            self.closed = False

        def execute_arrays(self, inputs):
            self.inputs = inputs
            return np.arange(16, dtype=np.float32)

        def close(self):
            self.closed = True

    session = SD3403RuntimeSession()
    session._worker = FakeWorker()
    inputs = [np.ones((1, 3), dtype=np.float32)]

    outputs = session.execute(inputs)

    assert session._worker.inputs is inputs
    assert outputs[0].shape == (16,)

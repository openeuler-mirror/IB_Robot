from __future__ import annotations

import numpy as np
import pytest

from inference_manifest import ArtifactBindings, TensorBinding
from inference_manifest.metadata import PolicyFeature, PolicyMetadata
from inference_service.codecs import (
    ACTPolicyCodec,
    BindingError,
    BindingPolicyCodec,
    CodecRequest,
    MissingSemanticTensorError,
    PI05PolicyCodec,
    SmolVLAPolicyCodec,
    create_policy_codec,
)


def _binding(
    semantic: str,
    *,
    runtime_name: str | None,
    index: int | None,
    dtype: str,
    shape: tuple[int, ...],
    layout: str | None = None,
) -> TensorBinding:
    return TensorBinding(
        semantic=semantic,
        runtime_name=runtime_name,
        index=index,
        dtype=dtype,
        shape=shape,
        layout=layout,
    )


def _artifact_bindings(
    inputs: tuple[TensorBinding, ...],
    outputs: tuple[TensorBinding, ...] | None = None,
) -> ArtifactBindings:
    return ArtifactBindings(
        inputs=inputs,
        outputs=outputs
        or (
            _binding(
                "action",
                runtime_name="runtime_actions",
                index=0,
                dtype="float32",
                shape=(1, 2, 3),
            ),
        ),
    )


class _TensorLike:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def detach(self) -> _TensorLike:
        return self

    def cpu(self) -> _TensorLike:
        return self

    def numpy(self) -> np.ndarray:
        return self._value


def _policy_metadata(policy_type: str, *, action_dim: int = 3) -> PolicyMetadata:
    return PolicyMetadata(
        policy_type=policy_type,
        input_features={
            "observation.state": PolicyFeature(type="STATE", shape=(3,)),
            "observation.images.top": PolicyFeature(type="VISUAL", shape=(3, 2, 4)),
            "observation.images.wrist": PolicyFeature(type="VISUAL", shape=(3, 2, 4)),
        },
        output_features={"action": PolicyFeature(type="ACTION", shape=(action_dim,))},
        required_files=("config.json", "policy_preprocessor.json", "policy_postprocessor.json"),
    )


@pytest.mark.parametrize("invalid", [None, "state", "cache_abi", "terminal_input", "stateful", "reverse_link"])
def test_pi05_staged_reuse_rejects_unverified_dependencies(invalid):
    from inference_manifest import CompiledDeployment

    def bindings(inputs, outputs):
        return {
            direction: [dict(semantic=name, index=i, dtype="float32", shape=[1]) for i, name in enumerate(names)]
            for direction, names in (("inputs", inputs), ("outputs", outputs))
        }

    prefix = ["internal.past_kv", "internal.prefix_pad_masks"]
    value = {
        "execution_contract": {
            "state_scope": "request",
            "execution_structure": "iterative",
            "orchestration_visibility": "executor",
            "cancellation_granularity": "checkpoint",
        },
        "runtime_profile": {"backend": "ascend", "target": {"runtime": "acl"}, "profile": {"device_id": 0}},
        "execution": ["vlm", "action_expert"],
        "artifacts": {role: {"path": f"{role}.om", "format": "om"} for role in ("vlm", "action_expert")},
        "bindings": {
            "vlm": bindings(["observation.images.top", "observation.language.tokens"], prefix),
            "action_expert": bindings(prefix + ["time", "noise"], ["action"]),
        },
    }
    if invalid == "state":
        value["bindings"]["vlm"]["inputs"][0]["semantic"] = "observation.state"
    elif invalid == "cache_abi":
        value["bindings"]["action_expert"]["inputs"][0]["shape"] = [2]
    elif invalid == "terminal_input":
        value["bindings"]["action_expert"]["inputs"][-1]["semantic"] = "observation.state"
    elif invalid == "stateful":
        value["execution_contract"]["state_scope"] = "stream"
        value["execution_contract"].update(state_bank_mode="runtime_exclusive", max_open_streams=1)
    elif invalid == "reverse_link":
        value["device_links"] = [
            dict(
                semantic="internal.past_kv",
                producer="action_expert",
                consumer="vlm",
                transport="device_pointer",
                owner="producer",
            )
        ]
    # Family validation also guards callers that construct a deployment directly.
    deployment = CompiledDeployment.model_validate({**value, "device_links": []})
    if invalid == "reverse_link":
        from inference_manifest import DeviceLink

        deployment = deployment.model_copy(
            update={"device_links": tuple(DeviceLink(**link) for link in value["device_links"])}
        )
    if invalid is None:
        PI05PolicyCodec.validate_staged_deployment(deployment)
    else:
        with pytest.raises(ValueError, match="PI0.5"):
            PI05PolicyCodec.validate_staged_deployment(deployment)


def test_codec_maps_variable_camera_semantics_runtime_names_and_explicit_indices():
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.images.gripper_left",
                runtime_name="camera_gripper",
                index=2,
                dtype="float32",
                shape=(1, 3, 2, 4),
                layout="NCHW",
            ),
            _binding(
                "observation.state",
                runtime_name="joint_state",
                index=0,
                dtype="float32",
                shape=(1, 3),
            ),
            _binding(
                "observation.images.overhead_custom",
                runtime_name="camera_overhead",
                index=1,
                dtype="float32",
                shape=(1, 3, 2, 4),
                layout="NCHW",
            ),
        )
    )
    request = CodecRequest(
        {
            "observation.images.overhead_custom": np.full((1, 3, 2, 4), 2.0),
            "observation.state": _TensorLike(np.array([[1, 2, 3]], dtype=np.float64)),
            "observation.images.gripper_left": np.full((1, 3, 2, 4), 3.0),
        }
    )

    encoded = BindingPolicyCodec().encode_inputs(request, bindings)

    assert list(encoded.by_runtime_name) == ["camera_gripper", "joint_state", "camera_overhead"]
    assert [float(value.flat[0]) for value in encoded.ordered_values] == [1.0, 2.0, 3.0]
    assert encoded.by_runtime_name["joint_state"].dtype == np.dtype("float32")


@pytest.mark.parametrize(
    "indices, message",
    [
        ((0, 0), "duplicate runtime indices"),
        ((0, 2), "contiguous and start at zero"),
    ],
)
def test_runtime_indices_must_be_unique_and_contiguous(indices, message):
    with pytest.raises(ValueError, match=message):
        ArtifactBindings(
            inputs=(
                _binding(
                    "observation.state",
                    runtime_name="state",
                    index=indices[0],
                    dtype="float32",
                    shape=(1, 3),
                ),
                _binding(
                    "observation.language.tokens",
                    runtime_name="tokens",
                    index=indices[1],
                    dtype="int64",
                    shape=(1, 4),
                ),
            ),
            outputs=(
                _binding(
                    "action",
                    runtime_name="actions",
                    index=0,
                    dtype="float32",
                    shape=(1, 2, 3),
                ),
            ),
        )


def test_runtime_indices_must_be_declared_for_all_or_no_bindings():
    with pytest.raises(ValueError, match="indices for every binding or omit them"):
        ArtifactBindings(
            inputs=(
                _binding(
                    "observation.state",
                    runtime_name="state",
                    index=0,
                    dtype="float32",
                    shape=(1, 3),
                ),
                _binding(
                    "observation.language.tokens",
                    runtime_name="tokens",
                    index=None,
                    dtype="int64",
                    shape=(1, 4),
                ),
            ),
            outputs=(
                _binding(
                    "action",
                    runtime_name="actions",
                    index=0,
                    dtype="float32",
                    shape=(1, 2, 3),
                ),
            ),
        )


def test_nhwc_conversion_applies_only_to_declared_image_binding():
    canonical = np.arange(1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.images.front",
                runtime_name="front",
                index=0,
                dtype="float32",
                shape=(1, 2, 4, 3),
                layout="NHWC",
            ),
        )
    )

    converted = (
        BindingPolicyCodec()
        .encode_inputs(CodecRequest({"observation.images.front": canonical}), bindings)
        .ordered_values[0]
    )

    assert converted.shape == (1, 2, 4, 3)
    np.testing.assert_array_equal(converted, np.transpose(canonical, (0, 2, 3, 1)))
    assert converted.flags.c_contiguous


def test_nchw_image_and_non_image_rank_four_preserve_axis_order():
    image = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
    attention = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.images.wrist",
                runtime_name="wrist",
                index=0,
                dtype="float32",
                shape=(1, 3, 2, 4),
                layout="NCHW",
            ),
            _binding(
                "observation.language.attention_bias",
                runtime_name="attention_bias",
                index=1,
                dtype="float32",
                shape=(1, 3, 2, 4),
                layout="NCHW",
            ),
        )
    )

    image_output, attention_output = (
        BindingPolicyCodec()
        .encode_inputs(
            CodecRequest(
                {
                    "observation.images.wrist": image,
                    "observation.language.attention_bias": attention,
                }
            ),
            bindings,
        )
        .ordered_values
    )

    np.testing.assert_array_equal(image_output, image)
    np.testing.assert_array_equal(attention_output, attention)


def test_mixed_dtypes_and_dynamic_shapes_are_converted_and_validated():
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.state",
                runtime_name="state",
                index=0,
                dtype="float16",
                shape=(-1, 3),
            ),
            _binding(
                "observation.language.tokens",
                runtime_name="tokens",
                index=1,
                dtype="int64",
                shape=(1, -1),
            ),
            _binding(
                "observation.language.attention_mask",
                runtime_name="mask",
                index=2,
                dtype="bool",
                shape=(1, -1),
            ),
        )
    )

    values = (
        BindingPolicyCodec()
        .encode_inputs(
            CodecRequest(
                {
                    "observation.state": [[1, 2, 3], [4, 5, 6]],
                    "observation.language.tokens": np.array([[1, 2, 3, 4]], dtype=np.int32),
                    "observation.language.attention_mask": [[1, 1, 0, 0]],
                }
            ),
            bindings,
        )
        .ordered_values
    )

    assert [value.dtype for value in values] == [np.dtype("float16"), np.dtype("int64"), np.dtype("bool")]
    assert [value.shape for value in values] == [(2, 3), (1, 4), (1, 4)]

    with pytest.raises(BindingError, match=r"axis 1: expected 3, got 4"):
        BindingPolicyCodec().encode_inputs(
            CodecRequest(
                {
                    "observation.state": np.zeros((2, 4)),
                    "observation.language.tokens": np.zeros((1, 4)),
                    "observation.language.attention_mask": np.zeros((1, 4)),
                }
            ),
            bindings,
        )


def test_missing_semantic_tensor_names_available_inputs():
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.images.side_camera",
                runtime_name="side",
                index=0,
                dtype="float32",
                shape=(1, 3, 2, 4),
                layout="NCHW",
            ),
        )
    )

    with pytest.raises(MissingSemanticTensorError, match="observation.images.side_camera"):
        BindingPolicyCodec().encode_inputs(CodecRequest({"observation.state": np.zeros((1, 3))}), bindings)


def test_output_decoding_selects_action_by_semantic_not_position():
    outputs = (
        _binding(
            "internal.attention",
            runtime_name="attention_output",
            index=0,
            dtype="float16",
            shape=(1, 2, 2),
        ),
        _binding(
            "action",
            runtime_name="policy_output",
            index=1,
            dtype="float32",
            shape=(1, 2, 3),
        ),
    )
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.state",
                runtime_name="state",
                index=0,
                dtype="float32",
                shape=(1, 3),
            ),
        ),
        outputs,
    )
    action = np.arange(6, dtype=np.float64).reshape(1, 2, 3)

    named_result = BindingPolicyCodec().decode_outputs(
        {
            "policy_output": action,
            "attention_output": np.ones((1, 2, 2), dtype=np.float32),
        },
        bindings,
    )
    indexed_result = BindingPolicyCodec().decode_outputs([np.ones((1, 2, 2), dtype=np.float32), action], bindings)

    assert named_result.action.dtype == np.dtype("float32")
    np.testing.assert_array_equal(named_result.action, indexed_result.action)
    np.testing.assert_array_equal(named_result.semantic_tensors["action"], action.astype(np.float32))


def test_direct_output_is_rejected_when_multiple_bindings_make_it_ambiguous():
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.state",
                runtime_name="state",
                index=0,
                dtype="float32",
                shape=(1, 3),
            ),
        ),
        (
            _binding(
                "internal.aux",
                runtime_name="aux",
                index=0,
                dtype="float32",
                shape=(1, 2),
            ),
            _binding(
                "action",
                runtime_name="action",
                index=1,
                dtype="float32",
                shape=(1, 2, 3),
            ),
        ),
    )

    with pytest.raises(BindingError, match="ambiguous"):
        BindingPolicyCodec().decode_outputs(np.zeros((1, 2, 3)), bindings)


def test_act_codec_prepares_state_and_images_by_semantic_binding_and_decodes_flat_action():
    metadata = _policy_metadata("act")
    bindings = ArtifactBindings(
        inputs=(
            _binding(
                "observation.images.wrist",
                runtime_name="camera_1",
                index=1,
                dtype="float32",
                shape=(1, 3, 4, 8),
                layout="NCHW",
            ),
            _binding(
                "observation.state",
                runtime_name="state",
                index=0,
                dtype="float32",
                shape=(1, 3),
            ),
        ),
        outputs=(
            _binding(
                "action",
                runtime_name="runtime_action",
                index=0,
                dtype="float32",
                shape=(1, 2, 3),
            ),
        ),
    )
    codec = ACTPolicyCodec(metadata)
    request = CodecRequest(
        {
            "observation.state": np.array([1, 2, 3], dtype=np.float64),
            "observation.images.wrist": np.arange(24, dtype=np.float32).reshape(3, 2, 4),
        }
    )

    encoded = codec.encode_inputs(request, bindings)
    decoded = codec.decode_outputs(
        {"runtime_action": np.arange(6, dtype=np.float32)},
        bindings,
    )

    assert encoded.ordered_values[0].shape == (1, 3)
    assert encoded.ordered_values[1].shape == (1, 3, 4, 8)
    assert decoded.action.shape == (1, 2, 3)
    assert decoded.actual_chunk_size == 2


def test_act_codec_prepares_singular_image_semantic():
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.image",
                runtime_name="camera",
                index=0,
                dtype="float32",
                shape=(1, 3, 4, 8),
                layout="NCHW",
            ),
        )
    )

    encoded = ACTPolicyCodec(_policy_metadata("act")).encode_inputs(
        CodecRequest({"observation.image": np.ones((3, 2, 4), dtype=np.float32)}),
        bindings,
    )

    assert encoded.ordered_values[0].shape == (1, 3, 4, 8)


@pytest.mark.parametrize(
    ("policy_type", "codec_type"),
    [("pi05", PI05PolicyCodec), ("smolvla", SmolVLAPolicyCodec)],
)
def test_vla_codecs_prepare_state_images_language_noise_time_and_crop_action(policy_type, codec_type):
    metadata = _policy_metadata(policy_type, action_dim=3)
    bindings = ArtifactBindings(
        inputs=(
            _binding(
                "observation.language.attention_mask",
                runtime_name="mask",
                index=4,
                dtype="bool",
                shape=(1, 4),
            ),
            _binding(
                "observation.images.wrist",
                runtime_name="wrist",
                index=2,
                dtype="float32",
                shape=(1, 3, 4, 4),
                layout="NCHW",
            ),
            _binding(
                "observation.state",
                runtime_name="state",
                index=0,
                dtype="float32",
                shape=(1, 5),
            ),
            _binding(
                "observation.language.tokens",
                runtime_name="tokens",
                index=3,
                dtype="int64",
                shape=(1, 4),
            ),
            _binding(
                "observation.images.top",
                runtime_name="top",
                index=1,
                dtype="float32",
                shape=(1, 3, 4, 4),
                layout="NCHW",
            ),
            _binding(
                "noise",
                runtime_name="noise",
                index=5,
                dtype="float16",
                shape=(1, 2, 5),
            ),
            _binding(
                "time",
                runtime_name="time",
                index=6,
                dtype="float32",
                shape=(1,),
            ),
        ),
        outputs=(
            _binding(
                "action",
                runtime_name="velocity",
                index=0,
                dtype="float32",
                shape=(1, 2, 5),
            ),
        ),
    )
    codec = codec_type(metadata)
    request = CodecRequest(
        {
            "observation.state": np.array([[1, 2, 3]], dtype=np.float32),
            "observation.images.top": np.ones((1, 3, 2, 4), dtype=np.float32),
            "observation.images.wrist": np.full((1, 3, 2, 4), 2.0, dtype=np.float32),
            "lang_tokens": np.array([[1, 2, 3, 4]], dtype=np.int32),
            "lang_masks": np.array([[1, 1, 0, 0]], dtype=np.int32),
            "_noise": np.ones((1, 2, 5), dtype=np.float32),
            "_time": np.array([0.5], dtype=np.float64),
        }
    )

    encoded = codec.encode_inputs(request, bindings)
    decoded = codec.decode_outputs(
        {"velocity": np.arange(10, dtype=np.float32).reshape(1, 2, 5)},
        bindings,
    )

    ordered = encoded.ordered_values
    assert [value.shape for value in ordered] == [
        (1, 5),
        (1, 3, 4, 4),
        (1, 3, 4, 4),
        (1, 4),
        (1, 4),
        (1, 2, 5),
        (1,),
    ]
    np.testing.assert_array_equal(ordered[0], np.array([[1, 2, 3, 0, 0]], dtype=np.float32))
    assert ordered[3].dtype == np.dtype("int64")
    assert ordered[4].dtype == np.dtype("bool")
    assert ordered[5].dtype == np.dtype("float16")
    assert decoded.action.shape == (1, 2, 3)
    np.testing.assert_array_equal(decoded.action, np.arange(10, dtype=np.float32).reshape(1, 2, 5)[..., :3])
    assert decoded.actual_chunk_size == 2


def test_smolvla_codec_matches_left_top_padding_and_nhwc_runtime_layout():
    bindings = _artifact_bindings(
        (
            _binding(
                "observation.images.top",
                runtime_name="pixel_values",
                index=0,
                dtype="float32",
                shape=(1, 4, 4, 3),
                layout="NHWC",
            ),
        )
    )
    image = np.ones((1, 3, 2, 4), dtype=np.float32)

    encoded = SmolVLAPolicyCodec(_policy_metadata("smolvla")).encode_inputs(
        CodecRequest({"observation.images.top": image}),
        bindings,
    )

    converted = encoded.ordered_values[0]
    assert converted.shape == (1, 4, 4, 3)
    np.testing.assert_array_equal(converted[:, :2], np.zeros((1, 2, 4, 3), dtype=np.float32))
    np.testing.assert_array_equal(converted[:, 2:], np.ones((1, 2, 4, 3), dtype=np.float32))


def test_policy_codec_registry_selects_only_from_policy_metadata():
    assert isinstance(create_policy_codec(_policy_metadata("act")), ACTPolicyCodec)
    assert isinstance(create_policy_codec(_policy_metadata("pi05")), PI05PolicyCodec)
    assert isinstance(create_policy_codec(_policy_metadata("smolvla")), SmolVLAPolicyCodec)

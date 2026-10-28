from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from inference_manifest import ArtifactBindings, DeviceLink, TensorBinding
from inference_service.codecs import (
    BindingPolicyCodec,
    CodecRequest,
    ExecutionFrame,
    ExecutionPlanError,
    build_execution_plan,
)
from inference_service.codecs.bindings import validate_artifact_bindings


def _tensor(
    semantic: str,
    runtime_name: str,
    index: int,
    shape: tuple[int, ...],
    *,
    dtype: str = "float32",
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


def _host_plan_bindings() -> dict[str, ArtifactBindings]:
    return {
        "vision": ArtifactBindings(
            inputs=(_tensor("observation.images.front", "image", 0, (1, 3, 2, 4), layout="NCHW"),),
            outputs=(_tensor("internal.vision_features", "features", 0, (1, -1, 8)),),
        ),
        "prefill": ArtifactBindings(
            inputs=(
                _tensor("internal.vision_features", "features", 0, (1, -1, 8)),
                _tensor("observation.language.tokens", "tokens", 1, (1, -1), dtype="int64"),
            ),
            outputs=(_tensor("internal.hidden", "hidden", 0, (1, -1, 8)),),
        ),
        "action": ArtifactBindings(
            inputs=(
                _tensor("internal.vision_features", "features", 0, (1, -1, 8)),
                _tensor("internal.hidden", "hidden", 1, (1, -1, 8)),
            ),
            outputs=(_tensor("action", "actions", 0, (1, 2, 3)),),
        ),
    }


def test_artifact_bindings_accept_rank_four_non_image_without_layout():
    bindings = ArtifactBindings(
        inputs=(_tensor("internal.past_key.0", "past_key_0", 0, (1, 4, 1, 2)),),
        outputs=(_tensor("action", "action", 0, (1, 2, 3)),),
    )

    validate_artifact_bindings(bindings)


def test_host_internal_links_preserve_role_order_and_last_consumer_lifetime():
    plan = build_execution_plan(("vision", "prefill", "action"), _host_plan_bindings())

    assert plan.role_names == ("vision", "prefill", "action")
    assert [(link.semantic, link.producer, link.consumers) for link in plan.host_links] == [
        ("internal.vision_features", "vision", ("prefill", "action")),
        ("internal.hidden", "prefill", ("action",)),
    ]
    assert plan.host_links[0].owner == "execution_frame"
    assert plan.host_links[0].lifetime == "through_last_consumer"

    frame = ExecutionFrame(plan)
    assert frame.begin_role("vision") == {}
    source = np.ones((1, 4, 8), dtype=np.float32)
    frame.finish_role("vision", {"internal.vision_features": source})
    source.fill(9.0)
    assert frame.live_host_semantics == ("internal.vision_features",)

    prefill_inputs = frame.begin_role("prefill")
    np.testing.assert_array_equal(prefill_inputs["internal.vision_features"], np.ones((1, 4, 8)))
    frame.finish_role("prefill", {"internal.hidden": np.full((1, 2, 8), 2.0, dtype=np.float32)})
    assert frame.live_host_semantics == ("internal.hidden", "internal.vision_features")

    action_inputs = frame.begin_role("action")
    assert set(action_inputs) == {"internal.vision_features", "internal.hidden"}
    frame.finish_role("action")
    assert frame.live_host_semantics == ()


def test_execution_frame_rejects_role_order_and_missing_producer_output():
    plan = build_execution_plan(("vision", "prefill", "action"), _host_plan_bindings())
    frame = ExecutionFrame(plan)

    with pytest.raises(ExecutionPlanError, match="out of order"):
        frame.begin_role("action")

    frame.begin_role("vision")
    with pytest.raises(ExecutionPlanError, match="did not provide host-visible output"):
        frame.finish_role("vision")
    assert frame.live_host_semantics == ()


def test_execution_frame_repeats_bounded_loop_and_retains_pre_loop_host_value():
    plan = build_execution_plan(("vision", "prefill", "action"), _host_plan_bindings())
    frame = ExecutionFrame(plan)

    frame.begin_role("vision")
    frame.finish_role("vision", {"internal.vision_features": np.ones((1, 4, 8), dtype=np.float32)})
    frame.begin_role("prefill")
    frame.finish_role("prefill", {"internal.hidden": np.ones((1, 2, 8), dtype=np.float32)})
    frame.configure_loop(("action",), 2)

    assert set(frame.begin_role("action")) == {"internal.hidden", "internal.vision_features"}
    frame.finish_role("action")
    assert frame.live_host_semantics == ("internal.hidden", "internal.vision_features")
    assert set(frame.begin_role("action")) == {"internal.hidden", "internal.vision_features"}
    frame.finish_role("action")
    assert frame.live_host_semantics == ()

    with pytest.raises(ExecutionPlanError, match="no remaining roles"):
        frame.begin_role("action")


def test_execution_frame_rejects_invalid_loop_region():
    plan = build_execution_plan(("vision", "prefill", "action"), _host_plan_bindings())
    frame = ExecutionFrame(plan)

    with pytest.raises(ExecutionPlanError, match="positive integer"):
        frame.configure_loop(("vision",), 0)
    with pytest.raises(ExecutionPlanError, match="next contiguous"):
        frame.configure_loop(("prefill", "action"), 2)


def test_device_only_link_is_descriptive_and_not_materialized_in_host_frame():
    bindings = _host_plan_bindings()
    link = DeviceLink(
        semantic="internal.hidden",
        producer="prefill",
        consumer="action",
        transport="device_pointer",
        owner="producer",
        lifetime="inference",
    )

    plan = build_execution_plan(("vision", "prefill", "action"), bindings, (link,))

    assert [
        (item.semantic, item.producer_binding, item.transport, item.owner, item.lifetime) for item in plan.device_links
    ] == [("internal.hidden", "output", "device_pointer", "producer", "inference")]
    assert all(host.semantic != "internal.hidden" for host in plan.host_links)
    assert not hasattr(plan.device_links[0], "pointer")
    assert not hasattr(plan.device_links[0], "address")

    frame = ExecutionFrame(plan)
    frame.begin_role("vision")
    frame.finish_role("vision", {"internal.vision_features": np.ones((1, 4, 8), dtype=np.float32)})
    frame.begin_role("prefill")
    frame.finish_role("prefill")
    action_inputs = frame.begin_role("action")
    assert set(action_inputs) == {"internal.vision_features"}


def test_input_sourced_device_link_never_materializes_in_host_frame():
    cache = _tensor("internal.cache", "cache", 1, (1, 4, 8), dtype="float16")
    bindings = {
        "prefill": ArtifactBindings(
            inputs=(
                _tensor("observation.language.tokens", "tokens", 0, (1, 4), dtype="int64"),
                cache,
            ),
            outputs=(_tensor("internal.hidden", "hidden", 0, (1, 4, 8)),),
        ),
        "decode": ArtifactBindings(
            inputs=(
                _tensor("internal.hidden", "hidden", 0, (1, 4, 8)),
                cache.model_copy(update={"runtime_name": "decode_cache"}),
            ),
            outputs=(_tensor("action", "action", 0, (1, 2, 3)),),
        ),
    }
    link = DeviceLink(
        semantic="internal.cache",
        producer="prefill",
        consumer="decode",
        producer_binding="input",
        transport="device_pointer",
        owner="producer",
    )

    plan = build_execution_plan(("prefill", "decode"), bindings, (link,))

    assert plan.device_links[0].producer_binding == "input"
    assert all(host.semantic != "internal.cache" for host in plan.host_links)
    assert not hasattr(plan.device_links[0], "pointer")
    assert not hasattr(plan.device_links[0], "address")

    frame = ExecutionFrame(plan)
    assert frame.begin_role("prefill") == {}
    frame.finish_role("prefill", {"internal.hidden": np.ones((1, 4, 8), dtype=np.float32)})
    assert set(frame.begin_role("decode")) == {"internal.hidden"}


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"dtype": "float32"}, "dtype differs"),
        ({"shape": (1, 4, 7)}, "shape differs"),
    ],
)
def test_input_sourced_device_link_validates_source_and_target_abi(update, message):
    source = _tensor("internal.cache", "cache", 0, (1, 4, 8), dtype="float16")
    bindings = {
        "prefill": ArtifactBindings(
            inputs=(source,),
            outputs=(_tensor("prefill.status", "status", 0, (1,), dtype="int32"),),
        ),
        "decode": ArtifactBindings(
            inputs=(source.model_copy(update={"runtime_name": "decode_cache", **update}),),
            outputs=(_tensor("action", "action", 0, (1, 2, 3)),),
        ),
    }
    link = DeviceLink(
        semantic="internal.cache",
        producer="prefill",
        consumer="decode",
        producer_binding="input",
        transport="device_pointer",
        owner="producer",
    )

    with pytest.raises(ExecutionPlanError, match=message):
        build_execution_plan(("prefill", "decode"), bindings, (link,))


@pytest.mark.parametrize(
    "mutate, message",
    [
        ("consumer_first", "must execute before"),
        ("dtype", "dtype differs"),
        ("shape", "shape differs"),
        ("multiple_producers", "multiple producers"),
    ],
)
def test_execution_plan_validates_producer_consumer_contracts(mutate, message):
    bindings = _host_plan_bindings()
    execution = ("vision", "prefill", "action")

    if mutate == "consumer_first":
        execution = ("prefill", "vision", "action")
    elif mutate == "dtype":
        prefill = bindings["prefill"]
        bindings["prefill"] = prefill.model_copy(
            update={
                "inputs": (
                    _tensor("internal.vision_features", "features", 0, (1, -1, 8), dtype="float16"),
                    prefill.inputs[1],
                )
            }
        )
    elif mutate == "shape":
        prefill = bindings["prefill"]
        bindings["prefill"] = prefill.model_copy(
            update={
                "inputs": (
                    _tensor("internal.vision_features", "features", 0, (1, -1, 7)),
                    prefill.inputs[1],
                )
            }
        )
    else:
        prefill = bindings["prefill"]
        bindings["prefill"] = prefill.model_copy(
            update={
                "outputs": (
                    _tensor("internal.hidden", "hidden", 0, (1, -1, 8)),
                    _tensor("internal.vision_features", "duplicate", 1, (1, -1, 8)),
                )
            }
        )

    with pytest.raises(ExecutionPlanError, match=message):
        build_execution_plan(execution, bindings)


@dataclass(frozen=True)
class _EquivalentFixture:
    bindings: ArtifactBindings
    request: CodecRequest
    expected_runtime_inputs: tuple[np.ndarray, ...]
    runtime_outputs: dict[str, np.ndarray]
    expected_action: np.ndarray


def _equivalent_fixture(*, image_name: str, state_name: str, layout: str, image_first: bool) -> _EquivalentFixture:
    image = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
    state = np.array([[1, 2, 3]], dtype=np.float32)
    image_binding = _tensor(
        "observation.images.variable_camera",
        image_name,
        0 if image_first else 1,
        (1, 2, 4, 3) if layout == "NHWC" else (1, 3, 2, 4),
        layout=layout,
    )
    state_binding = _tensor("observation.state", state_name, 1 if image_first else 0, (1, 3))
    action = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    bindings = ArtifactBindings(
        inputs=(image_binding, state_binding),
        outputs=(_tensor("action", "selected_action", 0, (1, 2, 3)),),
    )
    runtime_image = np.transpose(image, (0, 2, 3, 1)) if layout == "NHWC" else image
    expected_inputs = (runtime_image, state) if image_first else (state, runtime_image)
    return _EquivalentFixture(
        bindings=bindings,
        request=CodecRequest(
            {
                "observation.state": state,
                "observation.images.variable_camera": image,
            }
        ),
        expected_runtime_inputs=expected_inputs,
        runtime_outputs={"selected_action": action},
        expected_action=action,
    )


@pytest.fixture
def ascend() -> _EquivalentFixture:
    return _equivalent_fixture(image_name="camera_tensor", state_name="state_tensor", layout="NCHW", image_first=False)


@pytest.fixture
def rknn() -> _EquivalentFixture:
    return _equivalent_fixture(image_name="input_0", state_name="input_1", layout="NHWC", image_first=True)


@pytest.fixture
def hmm() -> _EquivalentFixture:
    return _equivalent_fixture(image_name="images", state_name="proprio", layout="NCHW", image_first=True)


@pytest.mark.parametrize("fixture_name", ["ascend", "rknn", "hmm"])
def test_named_runtime_fixtures_have_equivalent_codec_semantics(request, fixture_name):
    fixture: _EquivalentFixture = request.getfixturevalue(fixture_name)
    codec = BindingPolicyCodec()

    encoded = codec.encode_inputs(fixture.request, fixture.bindings)
    decoded = codec.decode_outputs(fixture.runtime_outputs, fixture.bindings)

    for actual, expected in zip(encoded.ordered_values, fixture.expected_runtime_inputs, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(decoded.action, fixture.expected_action)

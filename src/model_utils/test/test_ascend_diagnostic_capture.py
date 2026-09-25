from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inference_service.model_sessions import AscendOmModelSession


def _capture_session(captured):
    session = AscendOmModelSession(
        0,
        # The session now requires the ACL runtime to be injected rather than
        # discovered. This test exercises diagnostic capture only and never
        # loads a model, so a placeholder satisfies the constructor without
        # pulling in a real Ascend runtime - the same stand-in the
        # StatefulAscendOmModelSession tests use.
        runtime_manager=object(),
        diagnostic_capture=lambda name, value: captured.setdefault(name, np.asarray(value).copy()),
    )
    bindings = SimpleNamespace(
        inputs=(
            SimpleNamespace(index=0, semantic="observation.images.top"),
            SimpleNamespace(index=1, semantic="observation.language.tokens"),
            SimpleNamespace(index=2, semantic="observation.language.attention_mask"),
            SimpleNamespace(index=3, semantic="prefix_att_2d_masks_4d"),
        )
    )
    deployment = SimpleNamespace(bindings={"vlm": bindings})
    session._loaded_deployment = lambda: deployment
    return session


def test_ascend_session_diagnostic_capture_records_named_values_without_acl():
    captured = {}
    session = _capture_session(captured)

    session._capture_role_inputs(
        "vlm",
        {
            0: np.array([1.0], dtype=np.float32),
            1: np.array([2], dtype=np.int64),
            2: np.array([True], dtype=np.bool_),
            3: np.array([3.0], dtype=np.float32),
        },
    )
    session._capture_role_outputs(
        "vlm",
        {
            "internal.past_kv": np.array([4.0], dtype=np.float16),
            "internal.prefix_pad_masks": np.array([False], dtype=np.bool_),
        },
    )

    assert set(captured) == {
        "vlm_in_image_0",
        "vlm_in_lang_tokens",
        "vlm_in_lang_masks",
        "vlm_in_prefix_mask_4d",
        "past_kv_tensor",
        "prefix_pad_masks",
    }
    np.testing.assert_array_equal(captured["past_kv_tensor"], np.array([4.0], dtype=np.float16))


def test_ascend_session_diagnostic_capture_constructor_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        AscendOmModelSession(0, diagnostic_capture=True)
